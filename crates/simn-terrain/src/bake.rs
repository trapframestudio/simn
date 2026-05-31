//! Generic map baker: `BakeSpec` → canonical `heightmap.r32` +
//! `terrain.toml` + (on first bake) a scene skeleton.
//!
//! Per-map specs live in `tools/bakes/<map_id>.toml`. Baking is
//! idempotent on the asset side (every run rewrites the terrain
//! data) and preserve-on-the-scene-side (the `.tscn` is written once
//! and never touched again, so authored content survives re-bakes).
//!
//! The sampling pipeline:
//! 1. Target grid: `(W - 1) * spacing` × `(H - 1) * spacing` in UTM.
//! 2. For each vertex, inverse-project UTM → WGS84 (Snyder TM series).
//! 3. Bilinearly sample the source DEM at that lat/lon.
//! 4. Write each sample as little-endian f32 (literal meters).
//!
//! Anything not covered by the sampling pipeline — Blender-authored
//! detail, hand-carved chokepoints, POI flattening — lives in a
//! future layered-asset system (see walkthrough). Today: re-baking
//! wholesale replaces `heightmap.r32`, so treat bakes as a seed
//! operation, not a mid-iteration refresh — the editor-side
//! `Sync to Canonical` button is the round-trip path.

use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};

use crate::features::{
    ensure_worldcover_tile, map_esa_worldcover_class, read_worldcover_tile, FeatureClass,
    WorldCoverTile,
};
use crate::metadata::{TerrainMetadata, CURRENT_FORMAT_VERSION};
use crate::sampler::encode_r32;
use crate::spec::{BakeSource, BakeSpec, FeaturesSource};

/// Per-bake summary.
#[derive(Debug, Clone)]
pub struct BakeReport {
    pub asset_dir: PathBuf,
    pub scene_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub spacing_m: f32,
    pub vert_min_m: f32,
    pub vert_max_m: f32,
    pub observed_y_min: f32,
    pub observed_y_max: f32,
    pub scene_created: bool,
    pub source_label: String,
    /// Human label of the feature source used, if any (e.g.
    /// `"ESA WorldCover 2021 N45W123"`).
    pub features_label: Option<String>,
}

/// Slope threshold above which a cell is reclassified as cliff / rock,
/// overriding whatever vegetation label the land-cover raster assigned.
/// 0.7 corresponds to ~45° from vertical; below that the surface reads
/// as cliff-like regardless of what grew there historically.
const CLIFF_NORMAL_Y_THRESHOLD: f32 = 0.7;

/// Bake one map end-to-end. `asset_dir` receives `heightmap.r32` +
/// `terrain.toml` (rewritten every run); `scene_path` receives the
/// scene skeleton on first bake only.
pub fn bake_map(
    spec: &BakeSpec,
    asset_dir: &Path,
    scene_path: &Path,
    dem_cache_dir: &Path,
) -> Result<BakeReport> {
    // Parse the zone up-front so a bogus spec fails before we download
    // DEM tiles. All temperate-maritime bakes land in zone 10N or 11N;
    // anything else isn't wired.
    let _zone = parse_utm_zone_n(&spec.bounds.utm_zone)?;

    let width = spec.bounds.width();
    let height = spec.bounds.height();

    // Log when alignment snapped the requested extent up — keeps the
    // bake step transparent about why the output dims may exceed the
    // spec's `extent_x` / `extent_z`.
    if !spec.bounds.extent_x_was_aligned() || !spec.bounds.extent_z_was_aligned() {
        tracing::info!(
            map_id = %spec.map_id,
            requested_extent_x = spec.bounds.extent_x,
            requested_extent_z = spec.bounds.extent_z,
            aligned_extent_x = spec.bounds.aligned_extent_x(),
            aligned_extent_z = spec.bounds.aligned_extent_z(),
            region_size_m = spec.bounds.region_size_m(),
            "snapped bake extent up to next region multiple"
        );
    }

    // Load source.
    let source_label = spec.source.label();
    let samples_f = match &spec.source {
        BakeSource::SrtmHgt { tile, path } => {
            let tile_path = match path {
                Some(p) => p.clone(),
                None => ensure_srtm_tile(tile, dem_cache_dir)?,
            };
            sample_from_srtm(&tile_path, spec, width, height)?
        }
        BakeSource::Usgs3dep1m { srtm_fallback } => {
            sample_from_usgs3dep(spec, width, height, srtm_fallback.as_deref(), dem_cache_dir)?
        }
        BakeSource::MapzenTerrarium { zoom } => {
            sample_from_terrarium(spec, width, height, *zoom, dem_cache_dir)?
        }
    };

    // **Smooth out residual tile-boundary seams.** Mapzen Terrarium's
    // RGB encoding has ~0.1 m precision rounding bias; adjacent
    // tiles disagree on shared-edge pixels by a fraction of a meter.
    // Without smoothing, those 1-pixel jumps feed into slope
    // computation → false steep slopes → spurious Cliff overrides
    // at every tile boundary (the wedge "cliffs" the user kept
    // seeing).
    //
    // σ = 1.5 grid cells (3σ ≈ 9 cells = 18 m at 2 m spacing) is
    // much smaller than any real terrain feature but big enough
    // to dissolve sub-meter rounding jumps from multiple cells
    // around. Bumped from 1.0 in PR #56 — σ=1 left visible faint
    // seams in some places; 1.5 cleaned them.
    //
    // **Note:** this does NOT help bad source-data tiles. A whole
    // ~150 m tile of corrupted Mapzen data dominates its interior
    // even after a 18 m blur. Those need cache-delete + re-fetch
    // (or upstream tile validation at load time, future work).
    // See `validate_terrarium_tile` for the diagnostic logging
    // that flags suspicious tiles for manual intervention.
    let samples_f = smooth_heightmap_f32(&samples_f, width as usize, height as usize, 1.5);

    // Observed min/max → padded vert range. v2 stores literal f32
    // meters in `.r32`, so vert_min/vert_max are gameplay metadata
    // (camera bounds, sky shader, lossy PNG export) — not the
    // storage encoding parameters they were in v1.
    let (y_min, y_max) = minmax(&samples_f);
    let vert_min = (y_min - 5.0).floor();
    let vert_max = (y_max + 5.0).ceil();

    // Compute the optional feature layer before we write the metadata,
    // so the sidecar can carry its BLAKE3 digest. OSM overlays layer
    // on top of the base ESA classification; skipped when the spec
    // doesn't declare [features] (nothing to overlay onto).
    let (features_bytes, features_label) = match &spec.features {
        Some(source) => {
            let label = source.label();
            let mut bytes =
                bake_features(spec, source, &samples_f, vert_min, vert_max, dem_cache_dir)?;
            let mut full_label = label;
            // Snapshot the ESA-only state so that, after the OSM
            // polygon overlay runs, we can derive a "this cell got
            // OSM-painted" mask by diffing. Cells where the post-
            // landcover byte differs from the post-ESA byte got
            // their class from OSM and should be preserved through
            // the σ=8 smoothing pass below. Without this, the wide
            // kernel softens our crisp polygon edges right back into
            // ~48 m blobs.
            let bytes_post_esa = bytes.clone();
            if let Some(osm_cfg) = &spec.osm {
                // Landcover polygons override the ESA raster with
                // human-digitized feature edges (treelines, lake
                // shores, built-up boundaries). Run *before* the
                // highway overlay so roads still win where they
                // overlap landcover.
                if osm_cfg.landcover {
                    apply_osm_landcover_overlay(spec, &mut bytes, dem_cache_dir)?;
                    full_label.push_str(" + OSM landcover");
                }
                if osm_cfg.roads {
                    apply_osm_highway_overlay(spec, &mut bytes, dem_cache_dir)?;
                    full_label.push_str(" + OSM highways");
                }
            }

            // Smooth ESA 10 m staircase boundaries into organic
            // curves. Physical kernel radius ≈ 16 m (3σ at σ=8
            // cells × 2 m spacing) is enough to dissolve the 2-3
            // ESA-cell rectangular clusters that show up at
            // heterogeneous terrain in playable bakes. **Express
            // σ in physical meters** so the smoothing means the
            // same thing at any spacing — at 2 m spacing this
            // resolves to 8 cells (legacy behavior), at 100 m
            // backdrop spacing it becomes 0.16 cells (effectively
            // no-op, which is correct: each backdrop cell
            // already averages ~25 ESA cells, the staircase
            // artifact is invisible by construction).
            //
            // The polygon-preserve mask keeps OSM-painted cells at
            // their input class, so OSM polygon edges stay at their
            // crisp human-digitized resolution while ESA-only cells
            // dissolve. OSM line classes (PavedRoad / UnpavedRoad /
            // Trail) are still restored inside
            // `smooth_class_boundaries` regardless of mask.
            const SMOOTH_SIGMA_METERS: f32 = 16.0;
            let smooth_sigma_cells = SMOOTH_SIGMA_METERS / spec.bounds.spacing;
            let polygon_mask: Option<Vec<bool>> =
                if spec.osm.as_ref().map(|cfg| cfg.landcover).unwrap_or(false) {
                    Some(
                        bytes
                            .iter()
                            .zip(bytes_post_esa.iter())
                            .map(|(after, before)| after != before)
                            .collect(),
                    )
                } else {
                    None
                };
            bytes = crate::features::smooth_class_boundaries(
                &bytes,
                width as usize,
                height as usize,
                smooth_sigma_cells,
                polygon_mask.as_deref(),
            );
            full_label.push_str(&format!(
                " + boundary smooth σ={smooth_sigma_cells:.2}cells (≈{SMOOTH_SIGMA_METERS:.0}m)"
            ));
            (Some(bytes), Some(full_label))
        }
        None => (None, None),
    };
    let features_blake3 = features_bytes
        .as_ref()
        .map(|b| blake3::hash(b).to_hex().to_string())
        .unwrap_or_default();

    // From the post-everything `features.r8` byte grid, derive an
    // 8-channel splatmap pair (two RGBA8 images) that the terrain
    // shader will sample for biome blending. Per-class σ tuning
    // gives soft treelines but crisp cliffs / building edges.
    // See `splatmap.rs` for channel layout.
    let splatmap_pair = features_bytes
        .as_ref()
        .map(|b| crate::splatmap::bake_splatmap_pair(b, width as usize, height as usize));

    // Smooth road-density texture for sub-cell-precision road
    // rendering. The categorical line classes in features.r8 are
    // 1-3 cells wide, which renders as visible 2 m stair-step at
    // every road edge / corner / intersection. Per-class Gaussian
    // blur at bake time + filter_linear in the shader gives sub-
    // cell-precision smooth roads. RGBA8: R=Paved, G=Unpaved,
    // B=Trail, A=reserved. See `road_density.rs`.
    let road_density_bytes = features_bytes
        .as_ref()
        .map(|b| crate::road_density::bake_road_density(b, width as usize, height as usize));

    // Write canonical asset pair.
    fs::create_dir_all(asset_dir)?;
    let bytes = encode_r32(&samples_f);
    let blake3_hex = blake3::hash(&bytes).to_hex().to_string();
    // Iteration 5-13 Phase A2: write an all-zeros `nav_mask.r8` on
    // initial bake so freshly baked maps are self-consistent — no
    // "missing nav_mask file" warnings on first `Heightmap::load`,
    // and designers see a real file when they go to paint their
    // first overrides. Format version 1.
    let nav_mask_bytes = vec![0u8; (width as usize) * (height as usize)];
    let nav_mask_blake3 = blake3::hash(&nav_mask_bytes).to_hex().to_string();
    fs::write(asset_dir.join("nav_mask.r8"), &nav_mask_bytes)?;

    let metadata = TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: spec.map_id.clone(),
        width,
        height,
        spacing_m: spec.bounds.spacing,
        vert_min_m: vert_min,
        vert_max_m: vert_max,
        origin_utm_zone: spec.bounds.utm_zone.clone(),
        origin_utm_easting: spec.bounds.origin_east,
        origin_utm_northing: spec.bounds.origin_north,
        blake3: blake3_hex,
        features_blake3,
        region_size_m: spec.bounds.region_size_m(),
        playable_extent_x_m: spec.bounds.extent_x as f32,
        playable_extent_z_m: spec.bounds.extent_z as f32,
        nav_mask_format_version: crate::nav_mask::NAV_MASK_FORMAT_VERSION,
        nav_mask_blake3,
    };
    fs::write(asset_dir.join("terrain.toml"), toml::to_string(&metadata)?)?;
    fs::write(asset_dir.join("heightmap.r32"), bytes)?;
    // Clean up a stale v1 `.r16` if a previous bake left one
    // alongside. v2 is canonical; both files coexisting risks the
    // wrong one being committed.
    let stale_r16 = asset_dir.join("heightmap.r16");
    if stale_r16.exists() {
        let _ = fs::remove_file(&stale_r16);
    }
    if let Some(fb) = &features_bytes {
        fs::write(asset_dir.join("features.r8"), fb)?;
    } else {
        // If this map previously had a features.r8 and the spec has
        // since dropped the [features] section, clean it up so the
        // asset dir matches what the new metadata claims.
        let stale = asset_dir.join("features.r8");
        if stale.exists() {
            let _ = fs::remove_file(&stale);
        }
    }
    // Write splatmap pair (or clean up stale ones if the spec
    // dropped features). Stored as raw RGBA8 bytes — 4 × W × H — so
    // they can be uploaded to a `Texture2D` directly without a PNG
    // decode in the shader-loading hot path.
    let splat_a_path = asset_dir.join("splatmap_a.rgba8");
    let splat_b_path = asset_dir.join("splatmap_b.rgba8");
    if let Some(pair) = &splatmap_pair {
        fs::write(&splat_a_path, &pair.map_a)?;
        fs::write(&splat_b_path, &pair.map_b)?;
    } else {
        for stale in [&splat_a_path, &splat_b_path] {
            if stale.exists() {
                let _ = fs::remove_file(stale);
            }
        }
    }
    let road_density_path = asset_dir.join("road_density.rgba8");
    if let Some(rd) = &road_density_bytes {
        fs::write(&road_density_path, rd)?;
    } else if road_density_path.exists() {
        let _ = fs::remove_file(&road_density_path);
    }

    // Emit scene skeleton once.
    let scene_created = write_scene_once(scene_path, &spec.map_id, vert_max)?;

    Ok(BakeReport {
        asset_dir: asset_dir.to_owned(),
        scene_path: scene_path.to_owned(),
        width,
        height,
        spacing_m: spec.bounds.spacing,
        vert_min_m: vert_min,
        vert_max_m: vert_max,
        observed_y_min: y_min,
        observed_y_max: y_max,
        scene_created,
        source_label,
        features_label,
    })
}

/// Produce the per-cell `features.r8` byte grid from a [`FeaturesSource`].
/// Uses the heightmap samples for slope derivation so cliffs override the
/// land-cover label where the surface actually tilts steeply.
fn bake_features(
    spec: &BakeSpec,
    source: &FeaturesSource,
    height_samples: &[f32],
    _vert_min: f32,
    _vert_max: f32,
    cache_dir: &Path,
) -> Result<Vec<u8>> {
    let width = spec.bounds.width() as usize;
    let height = spec.bounds.height() as usize;
    let spacing = spec.bounds.spacing as f64;
    let zone = parse_utm_zone_n(&spec.bounds.utm_zone)?;

    // Load the land-cover tile.
    let tile = match source {
        FeaturesSource::EsaWorldCover { tile, path } => {
            let tile_path = match path {
                Some(p) => p.clone(),
                None => ensure_worldcover_tile(tile, cache_dir)?,
            };
            read_worldcover_tile(&tile_path, tile)?
        }
    };
    println!("  loaded {}", source.label());

    let mut out = vec![FeatureClass::Unknown as u8; width * height];
    let w = width as i32;
    let h = height as i32;

    for row in 0..h {
        let northing = spec.bounds.origin_north - (row as f64) * spacing;
        for col in 0..w {
            let easting = spec.bounds.origin_east + (col as f64) * spacing;
            let (lat, lon) = utm_zone_n_to_wgs84(easting, northing, zone);
            let esa = sample_worldcover(&tile, lat, lon);
            let mut class = map_esa_worldcover_class(esa);

            // Slope override. Compute the surface normal from four
            // neighboring height samples; a near-horizontal normal
            // means the surface is cliff-like regardless of what
            // grew on it historically. Edge cells fall back to the
            // raster-derived class so no synthetic cliff ring forms
            // at the map boundary.
            if row > 0 && row < h - 1 && col > 0 && col < w - 1 {
                let idx =
                    |r: i32, c: i32| -> f32 { height_samples[(r as usize) * width + (c as usize)] };
                let dy_dx = (idx(row, col + 1) - idx(row, col - 1)) / (2.0 * spacing as f32);
                let dy_dz = (idx(row + 1, col) - idx(row - 1, col)) / (2.0 * spacing as f32);
                let len = (dy_dx * dy_dx + 1.0 + dy_dz * dy_dz).sqrt();
                let n_y = 1.0 / len;
                if n_y < CLIFF_NORMAL_Y_THRESHOLD {
                    class = FeatureClass::Cliff;
                }
            }

            out[(row as usize) * width + (col as usize)] = class as u8;
        }
    }
    Ok(out)
}

/// Nearest-neighbor WorldCover lookup for a lat/lon inside the tile.
/// Wrapper around [`WorldCoverTile::sample_at`] that documents the
/// projection assumption (ESA WorldCover is pre-projected to WGS84).
fn sample_worldcover(tile: &WorldCoverTile, lat: f64, lon: f64) -> u8 {
    tile.sample_at(lat, lon)
}

/// Sample every grid vertex from the USGS 3DEP 1 m mosaic.
///
/// Discovery + download happens once per bake; tiles are held in
/// memory for the duration of the sampling pass (each tile is ~400
/// MB uncompressed, but only 4-8 tiles overlap any single map's bbox).
///
/// When `srtm_fallback` is provided, any grid cell that falls in a
/// 3DEP gap (no tile covers it, or all covering tiles report NODATA)
/// falls back to the SRTM bilinear sample at the same lat/lon. This
/// is what keeps maps that straddle a 3DEP project boundary intact
/// instead of punching holes in the heightmap.
fn sample_from_usgs3dep(
    spec: &BakeSpec,
    width: u32,
    height: u32,
    srtm_fallback: Option<&str>,
    cache_dir: &Path,
) -> Result<Vec<f32>> {
    let zone = parse_utm_zone_n(&spec.bounds.utm_zone)?;
    // Compute the WGS84 bbox the TNM API expects.
    let bbox = crate::osm::spec_wgs84_bbox(spec, zone, |e, n| utm_zone_n_to_wgs84(e, n, zone));
    let tnm_bbox = crate::usgs3dep::LatLonBbox {
        south: bbox.south,
        west: bbox.west,
        north: bbox.north,
        east: bbox.east,
    };
    let urls = crate::usgs3dep::discover_3dep_tiles(tnm_bbox, cache_dir)?;
    if urls.is_empty() {
        return Err(anyhow!(
            "no USGS 3DEP 1 m tiles found for bbox {:?}",
            tnm_bbox
        ));
    }
    println!("  3DEP: {} tile(s) discovered", urls.len());

    let mut tiles = Vec::with_capacity(urls.len());
    for url in &urls {
        let path = crate::usgs3dep::ensure_3dep_tile(url, cache_dir)?;
        let tile = crate::usgs3dep::read_3dep_tile(&path)?;
        println!(
            "    loaded {} ({:.0}-{:.0} E, {:.0}-{:.0} N)",
            path.file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default(),
            tile.east_min,
            tile.east_max(),
            tile.north_min,
            tile.north_max(),
        );
        tiles.push(tile);
    }

    // Optional SRTM fallback for gaps.
    let srtm_grid = if let Some(tile) = srtm_fallback {
        let path = ensure_srtm_tile(tile, cache_dir)?;
        Some((path, tile.to_string()))
    } else {
        None
    };

    // Cascade 2: 3DEP 1/3 arc-second seamless (~10 m, full-CONUS).
    // Pre-load all tiles that overlap the map bbox. For a single map
    // this is usually 1-2 tiles (each covers 1° × 1°).
    let zone_for_corners = zone;
    let corners = [
        (spec.bounds.origin_east, spec.bounds.origin_north),
        (
            spec.bounds.origin_east + spec.bounds.aligned_extent_x(),
            spec.bounds.origin_north,
        ),
        (
            spec.bounds.origin_east,
            spec.bounds.origin_north - spec.bounds.aligned_extent_z(),
        ),
        (
            spec.bounds.origin_east + spec.bounds.aligned_extent_x(),
            spec.bounds.origin_north - spec.bounds.aligned_extent_z(),
        ),
    ];
    let mut seamless_ids: std::collections::BTreeSet<String> = Default::default();
    for (e, n) in corners {
        let (lat, lon) = utm_zone_n_to_wgs84(e, n, zone_for_corners);
        seamless_ids.insert(crate::usgs3dep::tile_id_13_for(lat, lon));
    }
    let mut seamless_tiles: Vec<crate::usgs3dep::Tile13> = Vec::new();
    for id in &seamless_ids {
        match crate::usgs3dep::ensure_3dep_13_tile(id, cache_dir) {
            Ok(path) => match crate::usgs3dep::read_3dep_13_tile(&path, id) {
                Ok(t) => {
                    seamless_tiles.push(t);
                    println!("    loaded 3DEP 1/3 arc-sec tile {id}");
                }
                Err(e) => tracing::warn!(tile = id, error = %e, "failed to read 3DEP 1/3"),
            },
            Err(e) => tracing::warn!(tile = id, error = %e, "failed to fetch 3DEP 1/3"),
        }
    }

    let mut srtm_samples_f: Option<Vec<f32>> = None;
    if let Some((srtm_path, _)) = &srtm_grid {
        // Pre-sample SRTM across the whole grid so the final fallback
        // is O(1) per cell instead of re-opening the tile each miss.
        srtm_samples_f = Some(sample_from_srtm_path(srtm_path, spec, width, height)?);
    }

    let mut out = Vec::with_capacity((width * height) as usize);
    let mut lidar_cells = 0usize;
    let mut seamless_cells = 0usize;
    let mut srtm_cells = 0usize;
    let mut zero_fill_cells = 0usize;

    for row in 0..height {
        let northing = spec.bounds.origin_north - (row as f64) * (spec.bounds.spacing as f64);
        for col in 0..width {
            let easting = spec.bounds.origin_east + (col as f64) * (spec.bounds.spacing as f64);
            let idx = (row * width + col) as usize;
            // Cascade: 1 m LIDAR → 10 m seamless → SRTM fallback → 0.
            let v = if let Some(v) = crate::usgs3dep::sample_mosaic(&tiles, easting, northing) {
                lidar_cells += 1;
                v
            } else {
                let (lat, lon) = utm_zone_n_to_wgs84(easting, northing, zone_for_corners);
                let mut v13 = None;
                for t in &seamless_tiles {
                    if let Some(v) = t.sample(lat, lon) {
                        v13 = Some(v);
                        break;
                    }
                }
                if let Some(v) = v13 {
                    seamless_cells += 1;
                    v
                } else if let Some(s) = srtm_samples_f.as_ref() {
                    srtm_cells += 1;
                    s[idx]
                } else {
                    zero_fill_cells += 1;
                    0.0
                }
            };
            out.push(v);
        }
    }
    let total = (width * height) as usize;
    println!(
        "  3DEP cascade: 1m LIDAR {} ({:.1}%), 10m seamless {} ({:.1}%), SRTM {} ({:.1}%), zero-fill {}",
        lidar_cells,
        100.0 * lidar_cells as f64 / total as f64,
        seamless_cells,
        100.0 * seamless_cells as f64 / total as f64,
        srtm_cells,
        100.0 * srtm_cells as f64 / total as f64,
        zero_fill_cells,
    );
    Ok(out)
}

/// Wrapper around [`sample_from_srtm`] that opens a given tile path
/// directly. Factored out for the 3DEP fallback path.
fn sample_from_srtm_path(
    hgt_path: &Path,
    spec: &BakeSpec,
    width: u32,
    height: u32,
) -> Result<Vec<f32>> {
    sample_from_srtm(hgt_path, spec, width, height)
}

/// Sample every grid vertex from Mapzen Terrarium elevation tiles.
///
/// Computes the set of Terrarium tiles the map's bbox needs at the
/// chosen zoom, fetches them (cached under `<dem_cache>/terrarium/`),
/// and walks the grid sampling each vertex bilinearly from the
/// matching tile. Typical bake: 15-25 tiles, ~1 MB download, < 10 s
/// fetch on cold cache.
fn sample_from_terrarium(
    spec: &BakeSpec,
    width: u32,
    height: u32,
    zoom: u8,
    cache_dir: &Path,
) -> Result<Vec<f32>> {
    let zone = parse_utm_zone_n(&spec.bounds.utm_zone)?;
    let spacing = spec.bounds.spacing as f64;

    // Determine the set of (tile_x, tile_y) ids we need. Walk the
    // map bbox's four corners in lat/lon, then inset the bounds by
    // a tile to include any on-boundary samples.
    let corners = [
        (spec.bounds.origin_east, spec.bounds.origin_north),
        (
            spec.bounds.origin_east + spec.bounds.aligned_extent_x(),
            spec.bounds.origin_north,
        ),
        (
            spec.bounds.origin_east,
            spec.bounds.origin_north - spec.bounds.aligned_extent_z(),
        ),
        (
            spec.bounds.origin_east + spec.bounds.aligned_extent_x(),
            spec.bounds.origin_north - spec.bounds.aligned_extent_z(),
        ),
    ];
    let mut tx_min = u32::MAX;
    let mut tx_max = 0u32;
    let mut ty_min = u32::MAX;
    let mut ty_max = 0u32;
    for (e, n) in corners {
        let (lat, lon) = utm_zone_n_to_wgs84(e, n, zone);
        let (fx, fy) = crate::terrarium::lonlat_to_tile_frac(lat, lon, zoom);
        tx_min = tx_min.min(fx.floor() as u32);
        tx_max = tx_max.max(fx.ceil() as u32);
        ty_min = ty_min.min(fy.floor() as u32);
        ty_max = ty_max.max(fy.ceil() as u32);
    }
    // 1-tile buffer on each side. The cross-tile bilinear in
    // `sample_mosaic` fetches neighbors at +1 pixel; if our bbox
    // grazes the east/south edge of a tile, that +1 lookup needs
    // the next tile loaded. Cheap insurance — each extra tile is
    // ~50 KB.
    tx_min = tx_min.saturating_sub(1);
    ty_min = ty_min.saturating_sub(1);
    tx_max = tx_max.saturating_add(1);
    ty_max = ty_max.saturating_add(1);
    let needed_count = ((tx_max - tx_min + 1) * (ty_max - ty_min + 1)) as usize;
    println!(
        "  Terrarium: z{zoom}, tiles x=[{tx_min}..{tx_max}] y=[{ty_min}..{ty_max}] ({needed_count} tiles)"
    );

    let mut tiles = Vec::with_capacity(needed_count);
    for ty in ty_min..=ty_max {
        for tx in tx_min..=tx_max {
            let path = crate::terrarium::ensure_terrarium_tile(zoom, tx, ty, cache_dir)?;
            let tile = crate::terrarium::read_terrarium_tile(&path, zoom, tx, ty)?;
            tiles.push(tile);
        }
    }

    // **Bad-tile diagnostic.** Two failure modes worth flagging:
    //
    // 1. **Truly absurd mean difference from neighbors** (> 500 m).
    //    Mountainous regions like the Cascade Range routinely have
    //    300-400 m differences between adjacent ~150 m tiles
    //    (river bottom vs ridge), so the threshold has to be high
    //    or the diagnostic is just noise.
    //
    // 2. **Flat tile in non-flat region.** A genuinely corrupt
    //    Mapzen PNG decodes to near-zero elevations everywhere;
    //    its internal range is tiny (< 5 m) while neighbors have
    //    hundreds of meters of variation. This is the real
    //    signature of a bad cached download.
    //
    // Either way, the prescription is the same: `rm` the cached
    // PNG and re-run the bake.
    const MEAN_DIFF_THRESHOLD_M: f32 = 500.0;
    const FLAT_RANGE_THRESHOLD_M: f32 = 5.0;
    const NEIGHBOR_VARIABLE_THRESHOLD_M: f32 = 50.0;

    for tile in &tiles {
        let mean = tile.data.iter().sum::<f32>() / tile.data.len() as f32;
        let internal_min = tile.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let internal_max = tile.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let internal_range = internal_max - internal_min;

        let mut neighbor_means: Vec<f32> = Vec::with_capacity(8);
        let mut neighbor_max_range: f32 = 0.0;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = tile.x as i32 + dx;
                let ny = tile.y as i32 + dy;
                if nx < 0 || ny < 0 {
                    continue;
                }
                if let Some(n) = tiles
                    .iter()
                    .find(|t| t.zoom == zoom && t.x == nx as u32 && t.y == ny as u32)
                {
                    let nm = n.data.iter().sum::<f32>() / n.data.len() as f32;
                    neighbor_means.push(nm);
                    let n_min = n.data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let n_max = n.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    neighbor_max_range = neighbor_max_range.max(n_max - n_min);
                }
            }
        }
        if neighbor_means.is_empty() {
            continue;
        }
        neighbor_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = neighbor_means[neighbor_means.len() / 2];
        let mean_diff = (mean - median).abs();

        let extreme_mean_diff = mean_diff > MEAN_DIFF_THRESHOLD_M;
        let flat_in_variable_area = internal_range < FLAT_RANGE_THRESHOLD_M
            && neighbor_max_range > NEIGHBOR_VARIABLE_THRESHOLD_M;

        if extreme_mean_diff || flat_in_variable_area {
            let cache_file =
                cache_dir.join(format!("terrarium_z{zoom}_x{}_y{}.png", tile.x, tile.y));
            let reason = if flat_in_variable_area {
                format!(
                    "tile is nearly flat (range {:.1} m) but neighbors vary up to {:.0} m — \
                     classic corrupt-download signature",
                    internal_range, neighbor_max_range,
                )
            } else {
                format!(
                    "mean {:.0} m differs from neighbor median {:.0} m by {:.0} m \
                     (very rare even in steep terrain)",
                    mean, median, mean_diff,
                )
            };
            println!(
                "  ⚠ suspicious tile z={} x={} y={}: {}\n   To re-fetch:\n      rm '{}'\n   \
                 then re-run the bake.",
                zoom,
                tile.x,
                tile.y,
                reason,
                cache_file.display(),
            );
        }
    }

    let mut out = Vec::with_capacity((width * height) as usize);
    let mut missing = 0usize;
    for row in 0..height {
        let northing = spec.bounds.origin_north - (row as f64) * spacing;
        for col in 0..width {
            let easting = spec.bounds.origin_east + (col as f64) * spacing;
            let (lat, lon) = utm_zone_n_to_wgs84(easting, northing, zone);
            match crate::terrarium::sample_mosaic(&tiles, lat, lon, zoom) {
                Some(v) => out.push(v),
                None => {
                    missing += 1;
                    out.push(0.0);
                }
            }
        }
    }
    if missing > 0 {
        tracing::warn!(missing, total = (width * height) as usize, "Terrarium gaps");
    }
    Ok(out)
}

/// Fetch OSM highway ways for the map's bbox and rasterize them on
/// top of the base feature grid. Separated from `bake_features` so
/// future overlays (water polygons, buildings) can compose the same
/// way without tangling the ESA sampling pass.
fn apply_osm_highway_overlay(
    spec: &BakeSpec,
    features_grid: &mut [u8],
    cache_dir: &Path,
) -> Result<()> {
    let zone = parse_utm_zone_n(&spec.bounds.utm_zone)?;
    let bbox = crate::osm::spec_wgs84_bbox(spec, zone, |e, n| utm_zone_n_to_wgs84(e, n, zone));
    let resp = crate::osm::fetch_highways(&bbox, cache_dir)?;
    crate::osm::apply_osm_highways_overlay(
        &resp,
        spec,
        zone,
        spec.bounds.width(),
        spec.bounds.height(),
        features_grid,
    );
    Ok(())
}

/// Fetch OSM `natural=*` / `landuse=*` / `water=*` polygons (and
/// multipolygon relations) for the map's bbox and rasterize them
/// over the ESA base classification. Sits between `bake_features`
/// and `apply_osm_highway_overlay` in the pipeline so roads still
/// win over landcover where they overlap.
fn apply_osm_landcover_overlay(
    spec: &BakeSpec,
    features_grid: &mut [u8],
    cache_dir: &Path,
) -> Result<()> {
    let zone = parse_utm_zone_n(&spec.bounds.utm_zone)?;
    let bbox = crate::osm::spec_wgs84_bbox(spec, zone, |e, n| utm_zone_n_to_wgs84(e, n, zone));
    let resp = crate::osm::fetch_osm_landcover(&bbox, cache_dir)?;
    crate::osm::apply_osm_polygon_landcover(
        &resp,
        spec,
        zone,
        spec.bounds.width(),
        spec.bounds.height(),
        features_grid,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Source: SRTM 1-arcsec .hgt
// ---------------------------------------------------------------------------

/// Ensure the SRTM tile file exists on disk; download + gunzip from
/// the public Mapzen/AWS skadi mirror if missing. Returns the path to
/// the decompressed `.hgt` file.
pub fn ensure_srtm_tile(tile: &str, cache_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(cache_dir)?;
    let hgt = cache_dir.join(format!("{tile}.hgt"));
    if hgt.exists() {
        return Ok(hgt);
    }
    if tile.len() < 3 {
        return Err(anyhow!("invalid SRTM tile id {tile:?}"));
    }
    let prefix = &tile[..3]; // "N45" etc.
    let url = format!("https://elevation-tiles-prod.s3.amazonaws.com/skadi/{prefix}/{tile}.hgt.gz");
    let gz_path = cache_dir.join(format!("{tile}.hgt.gz"));
    println!("fetching {tile} from skadi mirror…");
    let curl = Command::new("curl")
        .arg("-fsSL")
        .arg("-o")
        .arg(&gz_path)
        .arg(&url)
        .status()
        .context("running curl (needed for first-time SRTM fetch)")?;
    if !curl.success() {
        return Err(anyhow!("curl failed fetching {url}"));
    }
    let gunzip = Command::new("gunzip")
        .arg(&gz_path)
        .status()
        .context("running gunzip")?;
    if !gunzip.success() {
        return Err(anyhow!("gunzip failed on {}", gz_path.display()));
    }
    Ok(hgt)
}

const SRTM_SIDE: usize = 3601;

/// Sample the SRTM tile at every vertex of the spec's target grid.
fn sample_from_srtm(hgt_path: &Path, spec: &BakeSpec, width: u32, height: u32) -> Result<Vec<f32>> {
    let bytes = fs::read(hgt_path).with_context(|| format!("reading {}", hgt_path.display()))?;
    let expected = SRTM_SIDE * SRTM_SIDE * 2;
    if bytes.len() != expected {
        return Err(anyhow!(
            "SRTM tile {} has wrong byte count {} (expected {})",
            hgt_path.display(),
            bytes.len(),
            expected
        ));
    }
    let srtm: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();

    // Tile id → SW corner lat/lon. E.g. `N45W123` → (45.0, -123.0).
    // Prefer the SRTM variant's tile id; fall back to parsing the
    // path's stem for the 3DEP-fallback call site that passes a tile
    // directly.
    let tile_id = match &spec.source {
        BakeSource::SrtmHgt { tile, .. } => tile.clone(),
        BakeSource::Usgs3dep1m { srtm_fallback } => srtm_fallback
            .clone()
            .ok_or_else(|| anyhow!("sample_from_srtm called without a tile id"))?,
        BakeSource::MapzenTerrarium { .. } => {
            return Err(anyhow!(
                "sample_from_srtm called for a Mapzen Terrarium spec — this shouldn't happen"
            ));
        }
    };
    let (tile_lat_min, tile_lon_min) = parse_srtm_tile_sw(&tile_id)?;

    let zone = parse_utm_zone_n(&spec.bounds.utm_zone)?;

    let mut out = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        let northing = spec.bounds.origin_north - (row as f64) * (spec.bounds.spacing as f64);
        for col in 0..width {
            let easting = spec.bounds.origin_east + (col as f64) * (spec.bounds.spacing as f64);
            let (lat, lon) = utm_zone_n_to_wgs84(easting, northing, zone);
            let h = sample_srtm_at(&srtm, SRTM_SIDE, tile_lat_min, tile_lon_min, lat, lon);
            out.push(h);
        }
    }
    Ok(out)
}

/// Parse `"10N"` → 10 (zone number), rejecting anything outside
/// the PNW-relevant 10N/11N pair or the southern hemisphere.
fn parse_utm_zone_n(zone: &str) -> Result<u32> {
    let Some(digits) = zone.strip_suffix('N') else {
        return Err(anyhow!(
            "UTM zone {zone:?} must end in 'N' (southern hemisphere not supported)"
        ));
    };
    let n: u32 = digits
        .parse()
        .map_err(|_| anyhow!("UTM zone {zone:?} has non-numeric digits"))?;
    if !(10..=11).contains(&n) {
        return Err(anyhow!(
            "UTM zone {zone:?} out of the PNW-supported range (10N, 11N)"
        ));
    }
    Ok(n)
}

/// Parse `"N45W123"` → (45.0, -123.0).
fn parse_srtm_tile_sw(tile: &str) -> Result<(f64, f64)> {
    // Format: N|S <2-digit-lat> W|E <3-digit-lon>
    if tile.len() != 7 {
        return Err(anyhow!("SRTM tile id must be 7 chars (got {tile:?})"));
    }
    let lat_sign = match &tile[0..1] {
        "N" => 1.0,
        "S" => -1.0,
        other => return Err(anyhow!("unexpected lat prefix {other:?}")),
    };
    let lat: f64 = tile[1..3]
        .parse()
        .map_err(|_| anyhow!("bad lat digits in {tile:?}"))?;
    let lon_sign = match &tile[3..4] {
        "E" => 1.0,
        "W" => -1.0,
        other => return Err(anyhow!("unexpected lon prefix {other:?}")),
    };
    let lon: f64 = tile[4..]
        .parse()
        .map_err(|_| anyhow!("bad lon digits in {tile:?}"))?;
    Ok((lat_sign * lat, lon_sign * lon))
}

fn sample_srtm_at(
    data: &[i16],
    side: usize,
    tile_lat_min: f64,
    tile_lon_min: f64,
    lat: f64,
    lon: f64,
) -> f32 {
    // Row: north-up. r=0 is lat = tile_lat_max; r=side-1 is lat = tile_lat_min.
    let tile_lat_max = tile_lat_min + 1.0;
    let row_f = (tile_lat_max - lat) * (side - 1) as f64;
    let col_f = (lon - tile_lon_min) * (side - 1) as f64;
    let max_idx = (side - 1) as f64 - 1e-9;
    let row = row_f.clamp(0.0, max_idx);
    let col = col_f.clamp(0.0, max_idx);
    let r0 = row.floor() as usize;
    let c0 = col.floor() as usize;
    let fr = (row - r0 as f64) as f32;
    let fc = (col - c0 as f64) as f32;
    let at = |r: usize, c: usize| -> f32 {
        let v = data[r * side + c];
        // SRTM voids are i16::MIN. The temperate-maritime has no known
        // voids; treat as sea level if one slipped in.
        if v == i16::MIN {
            0.0
        } else {
            v as f32
        }
    };
    let h00 = at(r0, c0);
    let h01 = at(r0, c0 + 1);
    let h10 = at(r0 + 1, c0);
    let h11 = at(r0 + 1, c0 + 1);
    let top = h00 * (1.0 - fc) + h01 * fc;
    let bot = h10 * (1.0 - fc) + h11 * fc;
    top * (1.0 - fr) + bot * fr
}

// ---------------------------------------------------------------------------
// Inverse UTM 10N → WGS84 (Snyder 1987)
// ---------------------------------------------------------------------------

const A: f64 = 6_378_137.0;
const F: f64 = 1.0 / 298.257_223_563;
const UTM_K0: f64 = 0.9996;
const UTM_FALSE_EASTING: f64 = 500_000.0;
const UTM_FALSE_NORTHING: f64 = 0.0;

/// Central meridian (deg) for UTM zone `n` (northern hemisphere).
/// Zone 1 is centered on -177°; each zone is 6° east.
fn utm_zone_central_lon_deg(zone: u32) -> f64 {
    -183.0 + 6.0 * (zone as f64)
}

fn utm_zone_n_to_wgs84(easting: f64, northing: f64, zone: u32) -> (f64, f64) {
    let lon0 = utm_zone_central_lon_deg(zone) * PI / 180.0;
    let e2 = 2.0 * F - F * F;
    let e_prime_sq = e2 / (1.0 - e2);
    let x = easting - UTM_FALSE_EASTING;
    let y = northing - UTM_FALSE_NORTHING;
    let m = y / UTM_K0;
    let mu = m / (A * (1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2 * e2 * e2 / 256.0));
    let e1 = (1.0 - (1.0 - e2).sqrt()) / (1.0 + (1.0 - e2).sqrt());
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();
    let sin_phi1 = phi1.sin();
    let cos_phi1 = phi1.cos();
    let tan_phi1 = sin_phi1 / cos_phi1;
    let n1 = A / (1.0 - e2 * sin_phi1 * sin_phi1).sqrt();
    let t1 = tan_phi1 * tan_phi1;
    let c1 = e_prime_sq * cos_phi1 * cos_phi1;
    let r1 = A * (1.0 - e2) / (1.0 - e2 * sin_phi1 * sin_phi1).powf(1.5);
    let d = x / (n1 * UTM_K0);
    let lat = phi1
        - (n1 * tan_phi1 / r1)
            * (d * d / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * e_prime_sq) * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1
                    - 252.0 * e_prime_sq
                    - 3.0 * c1 * c1)
                    * d.powi(6)
                    / 720.0);
    let lon = lon0
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * e_prime_sq + 24.0 * t1 * t1)
                * d.powi(5)
                / 120.0)
            / cos_phi1;
    (lat * 180.0 / PI, lon * 180.0 / PI)
}

// ---------------------------------------------------------------------------
// Scene skeleton emission
// ---------------------------------------------------------------------------

/// Emit `<scene_path>` with a default environment + sun +
/// `TerrainNode` + `PlayerSpawn`. Never overwrites — returns `false`
/// if the scene already exists. Once the file is there, authored
/// content is the author's to modify.
///
/// **Skipped for map ids prefixed with `_`** — those are
/// non-playable infrastructure assets (`_regional` backdrop, future
/// `_skybox` etc). They get a heightmap + terrain.toml but no
/// player-facing scene; the Godot side wires them in through a
/// dedicated node class instead of a stand-alone scene.
pub fn write_scene_once(scene_path: &Path, map_id: &str, vert_max: f32) -> Result<bool> {
    if map_id.starts_with('_') {
        return Ok(false);
    }
    if scene_path.exists() {
        return Ok(false);
    }
    let node_name = {
        let mut s = map_id.to_string();
        if let Some(first) = s.get_mut(0..1) {
            first.make_ascii_uppercase();
        }
        s
    };
    let safe_spawn_y = (vert_max + 50.0).ceil() as i32;
    // Every freshly-baked scene wraps the map in the shared
    // `weather_rig.tscn` — that's the canonical home for the
    // WorldEnvironment, Sun + Moon directional lights, fog tuning,
    // and the SunshineClouds driver. Inlining sky/light per map
    // (the previous template) drifts as we tune shared atmospheric
    // settings. Now there's exactly one place to tune weather and
    // every map picks it up automatically. The `[editable path]`
    // directive marks the rig instance as editable so per-map tweaks
    // to the env / lights / cloud driver land as overrides on that
    // map only — runtime-driven fields (sun rotation, env fog,
    // sky blend) still get clobbered each frame from sim, but
    // static fields persist.
    let template = format!(
        r#"[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://scripts/real_map.gd" id="1_map"]
[ext_resource type="PackedScene" path="res://scenes/weather/weather_rig.tscn" id="2_rig"]

[node name="{name}" type="Node3D"]
script = ExtResource("1_map")
region_id = "{id}"

[node name="WeatherRig" parent="." instance=ExtResource("2_rig")]

[node name="Terrain" type="TerrainNode" parent="."]
map_id = "{id}"
auto_load = true

[node name="PlayerSpawn" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, {spawn_y}, 0)

[editable path="WeatherRig"]
"#,
        name = node_name,
        id = map_id,
        spawn_y = safe_spawn_y,
    );
    if let Some(parent) = scene_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(scene_path, template)?;
    Ok(true)
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Separable 2D Gaussian blur on a row-major f32 grid. Used at
/// bake time to dissolve sub-cell rounding artifacts from the DEM
/// source (Mapzen Terrarium 0.1 m encoding precision) before they
/// feed into slope-derived class overrides. σ in grid cells; 3σ
/// kernel; clamp-to-edge at borders.
fn smooth_heightmap_f32(src: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let n = src.len();
    if sigma <= 0.0 || n == 0 {
        return src.to_vec();
    }
    let radius = (3.0 * sigma).ceil().max(1.0) as usize;
    let kernel: Vec<f32> = {
        let sigma2 = 2.0 * sigma * sigma;
        let size = 2 * radius + 1;
        let mut k = Vec::with_capacity(size);
        let mut sum = 0.0f32;
        for i in 0..size {
            let x = i as f32 - radius as f32;
            let w = (-x * x / sigma2).exp();
            k.push(w);
            sum += w;
        }
        for w in &mut k {
            *w /= sum;
        }
        k
    };
    let mut tmp = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let mut sum = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let sx = (x as isize + k as isize - radius as isize).clamp(0, width as isize - 1)
                    as usize;
                sum += src[row + sx] * kw;
            }
            tmp[row + x] = sum;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let sy = (y as isize + k as isize - radius as isize).clamp(0, height as isize - 1)
                    as usize;
                sum += tmp[sy * width + x] * kw;
            }
            out[y * width + x] = sum;
        }
    }
    out
}

fn minmax(v: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &x in v {
        if x < lo {
            lo = x;
        }
        if x > hi {
            hi = x;
        }
    }
    (lo, hi)
}

/// Load a spec file from disk.
pub fn load_spec(path: &Path) -> Result<BakeSpec> {
    let text =
        fs::read_to_string(path).with_context(|| format!("reading spec {}", path.display()))?;
    let spec: BakeSpec =
        toml::from_str(&text).with_context(|| format!("parsing spec {}", path.display()))?;
    Ok(spec)
}
