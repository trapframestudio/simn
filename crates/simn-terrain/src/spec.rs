//! Schema for `tools/bakes/<map_id>.toml` — the spec file that
//! describes a real map's DEM source + UTM bounds + grid resolution.
//! Consumed by the `terrain_bake` CLI and by any future tooling that
//! needs to re-derive a map from its source data.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// One map's complete bake recipe. Reads from
/// `tools/bakes/<map_id>.toml`; everything needed to reproduce the
/// canonical `heightmap.r32` + `terrain.toml` lives here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BakeSpec {
    /// Stable identifier (directory name, RegionGraph key, scene file
    /// stem). Must match the TOML's filename.
    pub map_id: String,
    pub bounds: BakeBounds,
    pub source: BakeSource,
    /// Optional feature-classification layer. When present, the
    /// baker writes `features.r8` alongside `heightmap.r32`. When
    /// absent, no feature layer is produced and the client falls
    /// back to slope-derived vertex coloring.
    #[serde(default)]
    pub features: Option<FeaturesSource>,
    /// Optional OpenStreetMap overlay. Rasterizes selected OSM
    /// features onto `features.r8` after the base classification;
    /// requires `features` to be set (there's nothing to overlay
    /// without a base grid).
    #[serde(default)]
    pub osm: Option<OsmOverlay>,
}

/// UTM-anchored map bounds. The spec's `extent_x` / `extent_z` are
/// the *requested* minimum extents in meters — the bake snaps them
/// up to the next region-aligned multiple so Terrain3D's region
/// grid (and every consumer that reads `terrain.toml`) tiles cleanly.
/// Final dimensions: `(W - 1) * spacing` × `(H - 1) * spacing`, with
/// `(W - 1) * spacing` ≡ 0 (mod `region_size_m`).
///
/// See `docs/book/src/planning/static-foliage-plan.md` →
/// "Cross-layer conventions" for why this contract exists. Without it,
/// canonical bakes whose extents aren't region multiples land
/// partially inside Terrain3D regions that don't get saved, shifting
/// the visible terrain off canonical center.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BakeBounds {
    /// UTM zone, e.g. `"10N"`. Only zone 10N is supported today.
    pub utm_zone: String,
    /// UTM easting of the map's NW corner (meters).
    pub origin_east: f64,
    /// UTM northing of the map's NW corner (meters).
    pub origin_north: f64,
    /// East-west *requested* map extent in meters. Final extent is
    /// snapped up to the next multiple of `region_size_m`.
    pub extent_x: f64,
    /// North-south *requested* map extent in meters. Final extent is
    /// snapped up to the next multiple of `region_size_m`.
    pub extent_z: f64,
    /// Distance between adjacent grid samples, meters.
    pub spacing: f32,
    /// Region edge length in *vertices*. Must match Terrain3D's
    /// `region_size`; default 1024 = 2048 m at 2 m spacing. Override
    /// only if a map needs a non-default Terrain3D region size.
    #[serde(default = "default_region_size_vertices")]
    pub region_size_vertices: u32,
}

fn default_region_size_vertices() -> u32 {
    1024
}

impl BakeBounds {
    /// Region edge length in world meters: `region_size_vertices * spacing`.
    pub fn region_size_m(&self) -> f32 {
        self.region_size_vertices as f32 * self.spacing
    }

    /// Snap a requested extent up to the next multiple of
    /// `2 * region_size_m`. Used for both axes.
    ///
    /// Why `2 ×`: maps are centered at world `(0, 0)`, so the NW
    /// corner sits at `(-extent / 2, -extent / 2)`. For Terrain3D's
    /// region grid (anchored at integer multiples of `region_size_m`
    /// from world origin) to tile the image cleanly, `extent / 2`
    /// must itself be a region multiple — i.e. `extent` must be a
    /// multiple of `2 * region_size_m`. An odd region count
    /// (cascade_locks at 4500 m → 6144 m = 3 regions) leaves the
    /// image straddling region edges by half a region width on each
    /// side; the eastern strip lands in an unbaked region and the
    /// visible terrain ends up offset west of world `(0, 0)`. Even
    /// region counts always center cleanly.
    fn align_extent(&self, requested_m: f64) -> f64 {
        let region = self.region_size_m() as f64;
        if region <= 0.0 {
            return requested_m;
        }
        let unit = region * 2.0;
        // Number of `2×region` blocks needed to cover `requested_m`.
        let n = (requested_m / unit).ceil().max(1.0);
        n * unit
    }

    /// Aligned east-west extent in meters. ≥ `extent_x`, multiple
    /// of `region_size_m`.
    pub fn aligned_extent_x(&self) -> f64 {
        self.align_extent(self.extent_x)
    }

    /// Aligned north-south extent in meters. ≥ `extent_z`, multiple
    /// of `region_size_m`.
    pub fn aligned_extent_z(&self) -> f64 {
        self.align_extent(self.extent_z)
    }

    /// Final grid width — derived from the aligned extent. The
    /// `+ 1` matches the canonical `(W - 1) * spacing` extent
    /// convention used everywhere downstream.
    pub fn width(&self) -> u32 {
        (self.aligned_extent_x() as f32 / self.spacing) as u32 + 1
    }

    /// Final grid height. See [`Self::width`].
    pub fn height(&self) -> u32 {
        (self.aligned_extent_z() as f32 / self.spacing) as u32 + 1
    }

    /// True if the requested extent already aligns to the region
    /// grid (no padding strip will be added on this axis).
    pub fn extent_x_was_aligned(&self) -> bool {
        (self.aligned_extent_x() - self.extent_x).abs() < 0.001
    }

    /// True if the requested extent already aligns to the region
    /// grid (no padding strip will be added on this axis).
    pub fn extent_z_was_aligned(&self) -> bool {
        (self.aligned_extent_z() - self.extent_z).abs() < 0.001
    }
}

/// DEM source descriptor.
///
/// - [`BakeSource::SrtmHgt`] — NASA SRTM 1-arcsec (~30 m native).
///   Global coverage, simplest pipeline, lowest detail. Appropriate
///   for maps outside CONUS or for rough-pass iteration.
/// - [`BakeSource::Usgs3dep1m`] — USGS 3DEP 1-meter LIDAR. CONUS only.
///   Road cuts, cliff edges, gullies and drainages render with real
///   definition instead of being smoothed by the 30 m upsample. This
///   is the preferred source for any PNW map the player actually
///   spends time in.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum BakeSource {
    /// NASA SRTM 1-arcsec `.hgt` tile (3601×3601, i16 big-endian).
    SrtmHgt {
        /// Tile id like `"N45W123"`. The loader derives the download
        /// URL from this and caches the file under `/tmp/noosphere-dem/`.
        tile: String,
        /// Optional override for where the tile file lives locally.
        /// If set, the loader skips the download path entirely.
        #[serde(default)]
        path: Option<PathBuf>,
    },
    /// USGS 3DEP 1-meter LIDAR DEM. Tiles are auto-discovered for the
    /// map's bounding box via The National Map (TNM) Access API;
    /// mosaicked at bake time.
    Usgs3dep1m {
        /// Optional SRTM tile to fall back on for cells that no 3DEP
        /// tile covers (e.g. across a project boundary, or over
        /// water where 3DEP's bare-earth DEM has NODATA). When
        /// `None`, gaps become zero elevation.
        #[serde(default)]
        srtm_fallback: Option<String>,
    },
    /// Mapzen Terrarium elevation tiles on AWS (PNG-encoded RGB).
    /// Small (~50 KB / tile), seamless, and composited from the best
    /// underlying DEMs per region (USGS NED / 3DEP where covered,
    /// SRTM elsewhere). Zoom picks the effective spacing: 14 ≈ 10 m,
    /// 15 ≈ 5 m, 16 ≈ 2.4 m at lat 45°.
    MapzenTerrarium {
        /// TMS zoom level. 15 is the preferred default for Noosphere
        /// — 2-3× the effective resolution of SRTM without per-map
        /// LIDAR discovery / multi-hundred-MB downloads.
        #[serde(default = "default_terrarium_zoom")]
        zoom: u8,
    },
}

fn default_terrarium_zoom() -> u8 {
    15
}

impl BakeSource {
    /// Short human label for logging.
    pub fn label(&self) -> String {
        match self {
            BakeSource::SrtmHgt { tile, .. } => format!("SRTM 1-arcsec {tile}"),
            BakeSource::Usgs3dep1m { srtm_fallback } => match srtm_fallback {
                Some(t) => format!("USGS 3DEP 1 m (fallback {t})"),
                None => "USGS 3DEP 1 m".into(),
            },
            BakeSource::MapzenTerrarium { zoom } => {
                format!("Mapzen Terrarium (zoom {zoom})")
            }
        }
    }
}

/// Classification-source descriptor. Today only ESA WorldCover 2021
/// is wired — it's delivered as WGS84-aligned 3° × 3° GeoTIFF tiles,
/// so sampling slots into the existing lat/lon pipeline without a
/// separate reprojection step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FeaturesSource {
    /// ESA WorldCover v200 2021 — 10 m global, 11 classes.
    EsaWorldCover {
        /// Tile id like `"N45W123"`. 3° × 3° SW corner — note this
        /// is a different grid than SRTM's 1° × 1° tiles despite
        /// sharing the letter format.
        tile: String,
        /// Optional override for a local `.tif` path. Skips the
        /// download step when set.
        #[serde(default)]
        path: Option<PathBuf>,
    },
}

impl FeaturesSource {
    /// Short human label for logging.
    pub fn label(&self) -> String {
        match self {
            FeaturesSource::EsaWorldCover { tile, .. } => {
                format!("ESA WorldCover 2021 {tile}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(extent_x: f64, extent_z: f64, spacing: f32) -> BakeBounds {
        BakeBounds {
            utm_zone: "10N".into(),
            origin_east: 0.0,
            origin_north: 0.0,
            extent_x,
            extent_z,
            spacing,
            region_size_vertices: 1024,
        }
    }

    #[test]
    fn aligned_already_returns_input() {
        // 8192 m = 4 regions on X (even); 4096 m = 2 regions on Z
        // (even). Both half-extents (4096, 2048) are region multiples.
        let bounds = b(8192.0, 4096.0, 2.0);
        assert!(bounds.extent_x_was_aligned());
        assert!(bounds.extent_z_was_aligned());
        assert_eq!(bounds.aligned_extent_x(), 8192.0);
        assert_eq!(bounds.aligned_extent_z(), 4096.0);
        assert_eq!(bounds.width(), 4097);
        assert_eq!(bounds.height(), 2049);
    }

    #[test]
    fn cascade_locks_extent_snaps_up_to_even_region_count() {
        // Real-world example: cascade_locks spec is 4500 × 4000,
        // spacing 2, region 1024 vertices = 2048 m. Centered, so
        // alignment unit is `2 * region_size = 4096 m`. Aligned:
        // 8192 × 4096, 4 × 2 regions. Half-extent (4096, 2048) is
        // itself a region multiple → image tiles regions cleanly.
        let bounds = b(4500.0, 4000.0, 2.0);
        assert!(!bounds.extent_x_was_aligned());
        assert!(!bounds.extent_z_was_aligned());
        assert_eq!(bounds.aligned_extent_x(), 8192.0);
        assert_eq!(bounds.aligned_extent_z(), 4096.0);
        assert_eq!(bounds.width(), 4097);
        assert_eq!(bounds.height(), 2049);
        // (W - 1) * spacing / 2 must be a clean multiple of the
        // region size — that's what guarantees centered tiling.
        let region_m = bounds.region_size_m();
        let half_x = (bounds.width() - 1) as f32 * bounds.spacing * 0.5;
        let half_z = (bounds.height() - 1) as f32 * bounds.spacing * 0.5;
        assert!(
            half_x % region_m == 0.0,
            "half_x {half_x} not multiple of {region_m}"
        );
        assert!(
            half_z % region_m == 0.0,
            "half_z {half_z} not multiple of {region_m}"
        );
    }

    #[test]
    fn region_size_scales_with_spacing() {
        // 1024 vertices × 4 m spacing = 4096 m per region. Alignment
        // unit = 2 * region = 8192 m.
        let bounds = b(1000.0, 1000.0, 4.0);
        assert_eq!(bounds.region_size_m(), 4096.0);
        // 1000 m requested → snap up to 8192 (2 regions, even).
        assert_eq!(bounds.aligned_extent_x(), 8192.0);
    }
}

/// OSM overlay toggles. Each flag opts into rasterizing one class
/// of OpenStreetMap features onto the base `features.r8` grid.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OsmOverlay {
    /// When true, fetch OSM `highway=*` ways via the Overpass API
    /// and paint them as `PavedRoad` / `UnpavedRoad` / `Trail`
    /// feature classes. Roads override the base ESA class but not
    /// water; unpaved doesn't override paved; trails don't override
    /// either road class.
    #[serde(default)]
    pub roads: bool,

    /// When true, fetch OSM `natural=*` / `landuse=*` / `water=*`
    /// polygons (and multipolygon relations) and rasterize them as
    /// their corresponding [`FeatureClass`][fc]. Polygons override
    /// ESA's 10 m raster — OSM is human-digitized at real feature
    /// edges, so boundaries read as organic curves instead of the
    /// staircase ESA produces. Roads (if also enabled) layer on top
    /// of landcover, matching today's precedence.
    ///
    /// [fc]: crate::features::FeatureClass
    #[serde(default)]
    pub landcover: bool,
}
