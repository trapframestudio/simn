//! OSM highway overlay — fetches OpenStreetMap `highway=*` ways for
//! a map's bounding box via the Overpass API and rasterizes them onto
//! `features.r8` as [`FeatureClass::PavedRoad`], [`FeatureClass::UnpavedRoad`],
//! or [`FeatureClass::Trail`].
//!
//! Wiring: [`apply_osm_highways_overlay`] is called by the baker after
//! the base ESA WorldCover classification + slope-derived cliff pass.
//! Roads/trails have precedence over ESA classes (they're more accurate
//! for narrow features at our 2 m sampling), but never paint over
//! [`FeatureClass::Water`] — bridges exist in OSM but the display
//! artifact of "road across river" is negligible vs. the alternative
//! bug of "road clips through water."
//!
//! Response caching: Overpass is rate-limited and each query takes
//! 5–60 s. We cache per-map by bbox hash under the same `dem_cache_dir`
//! used for SRTM/WorldCover, so re-bakes reuse the JSON response
//! rather than re-hitting the API.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::features::FeatureClass;
use crate::spec::BakeSpec;

/// Resolved class for a highway way. Used to select brush width +
/// target [`FeatureClass`] during rasterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadClass {
    Paved,
    Unpaved,
    Trail,
}

impl RoadClass {
    /// Rendered line width in meters. Intentionally wider than
    /// real-world carriageway (paved ~7 m, dirt track ~3 m) so the
    /// feature reads clearly at our 2 m grid sampling without
    /// becoming a one-cell thread that aliases in and out of view.
    pub fn width_m(self) -> f64 {
        match self {
            RoadClass::Paved => 8.0,
            RoadClass::Unpaved => 5.0,
            RoadClass::Trail => 2.5,
        }
    }

    pub fn feature_class(self) -> FeatureClass {
        match self {
            RoadClass::Paved => FeatureClass::PavedRoad,
            RoadClass::Unpaved => FeatureClass::UnpavedRoad,
            RoadClass::Trail => FeatureClass::Trail,
        }
    }
}

/// Classify an OSM way by its `highway=*` tag + optional `surface=*`
/// tag. Returns `None` for highway values we don't render (abandoned,
/// proposed, raceway, etc.).
pub fn classify_highway(tags: &BTreeMap<String, String>) -> Option<RoadClass> {
    let highway = tags.get("highway")?.as_str();
    let surface = tags.get("surface").map(String::as_str);

    // Trail classes: always non-vehicular regardless of surface.
    match highway {
        "path" | "footway" | "bridleway" | "cycleway" | "steps" | "pedestrian" => {
            return Some(RoadClass::Trail);
        }
        _ => {}
    }

    // Explicitly unpaved surfaces override the highway-class default.
    let unpaved_by_surface = matches!(
        surface,
        Some(
            "dirt"
                | "gravel"
                | "unpaved"
                | "ground"
                | "earth"
                | "grass"
                | "sand"
                | "compacted"
                | "fine_gravel"
                | "pebblestone"
        )
    );
    let paved_by_surface = matches!(
        surface,
        Some("asphalt" | "paved" | "concrete" | "paving_stones" | "cobblestone" | "bricks")
    );

    match highway {
        "motorway" | "motorway_link" | "trunk" | "trunk_link" | "primary" | "primary_link"
        | "secondary" | "secondary_link" | "tertiary" | "tertiary_link" | "residential"
        | "living_street" | "unclassified" | "road" => {
            if unpaved_by_surface {
                Some(RoadClass::Unpaved)
            } else {
                Some(RoadClass::Paved)
            }
        }
        "service" => {
            if paved_by_surface {
                Some(RoadClass::Paved)
            } else {
                Some(RoadClass::Unpaved)
            }
        }
        "track" => Some(RoadClass::Unpaved),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Overpass API
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct OverpassResponse {
    pub elements: Vec<OverpassElement>,
}

#[derive(Debug, Deserialize)]
pub struct OverpassElement {
    #[serde(rename = "type")]
    pub element_type: String,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    /// Way geometry (direct vertices) or relation-member geometry
    /// (set on each member instead of the element). Empty for
    /// relation-typed elements — walk `members` for those.
    #[serde(default)]
    pub geometry: Vec<OverpassPoint>,
    /// Only populated for `element_type == "relation"`. Multipolygon
    /// relations carry one `OverpassMember` per outer/inner ring,
    /// each with its own `geometry` when the Overpass query uses
    /// `out geom`.
    #[serde(default)]
    pub members: Vec<OverpassMember>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct OverpassPoint {
    pub lat: f64,
    pub lon: f64,
}

/// One member of an OSM relation. Our rasterizer consumes only
/// multipolygon relations (outer/inner ring assembly); other roles
/// are ignored. `member_type` is usually `"way"` for multipolygons;
/// `node`/`relation` members are skipped by the consumer.
#[derive(Debug, Deserialize, Clone)]
pub struct OverpassMember {
    #[serde(rename = "type")]
    pub member_type: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub geometry: Vec<OverpassPoint>,
}

/// Fetch OSM `highway=*` ways for a WGS84 bounding box. Results
/// are cached under `<cache_dir>/overpass-highways-<bbox>.json`.
pub fn fetch_highways(bbox: &Bbox, cache_dir: &Path) -> Result<OverpassResponse> {
    fs::create_dir_all(cache_dir)?;
    let cache_file = cache_dir.join(format!(
        "overpass-highways-{:.4}_{:.4}_{:.4}_{:.4}.json",
        bbox.south, bbox.west, bbox.north, bbox.east,
    ));
    if cache_file.exists() {
        let text = fs::read_to_string(&cache_file)
            .with_context(|| format!("reading {}", cache_file.display()))?;
        let resp: OverpassResponse = serde_json::from_str(&text)?;
        return Ok(resp);
    }

    let query = format!(
        "[out:json][timeout:90];\
         (way[\"highway\"]({s},{w},{n},{e}););\
         out geom;",
        s = bbox.south,
        w = bbox.west,
        n = bbox.north,
        e = bbox.east,
    );
    // Overpass load-balances poorly; the public main instance 504s
    // regularly. Try mirrors in order until one succeeds.
    const MIRRORS: &[&str] = &[
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ];
    println!(
        "  querying Overpass for highways in bbox [{}]…",
        bbox.short()
    );
    let mut last_err: Option<String> = None;
    for mirror in MIRRORS {
        let output = Command::new("curl")
            .arg("-fsSL")
            .arg("--max-time")
            .arg("120")
            .arg("-X")
            .arg("POST")
            .arg("--data-urlencode")
            .arg(format!("data={query}"))
            .arg(*mirror)
            .output()
            .context("running curl for Overpass")?;
        if !output.status.success() {
            last_err = Some(format!(
                "{mirror}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
            continue;
        }
        // Partial / HTML error pages sometimes come back with 200;
        // validate the body parses as JSON before caching.
        match serde_json::from_slice::<OverpassResponse>(&output.stdout) {
            Ok(resp) => {
                fs::write(&cache_file, &output.stdout)?;
                return Ok(resp);
            }
            Err(e) => {
                last_err = Some(format!("{mirror}: non-JSON response ({e})"));
                continue;
            }
        }
    }
    Err(anyhow!(
        "all Overpass mirrors failed. last error: {}",
        last_err.unwrap_or_default()
    ))
}

/// WGS84 lat/lon bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub south: f64,
    pub west: f64,
    pub north: f64,
    pub east: f64,
}

impl Bbox {
    pub fn short(&self) -> String {
        format!(
            "{:.3},{:.3},{:.3},{:.3}",
            self.south, self.west, self.north, self.east
        )
    }
}

// ---------------------------------------------------------------------------
// Forward UTM (WGS84 → UTM zone N, northern hemisphere), Snyder 1987
// ---------------------------------------------------------------------------

const A: f64 = 6_378_137.0;
const F: f64 = 1.0 / 298.257_223_563;
const UTM_K0: f64 = 0.9996;
const UTM_FALSE_EASTING: f64 = 500_000.0;

/// Forward transverse-Mercator for UTM zone `n` (northern hemisphere).
/// Central meridian is `-183 + 6n` degrees, matching the inverse in
/// `bake.rs::utm_zone_n_to_wgs84`.
pub fn wgs84_to_utm_zone_n(lat_deg: f64, lon_deg: f64, zone: u32) -> (f64, f64) {
    let lon0 = (-183.0 + 6.0 * (zone as f64)) * PI / 180.0;
    let phi = lat_deg * PI / 180.0;
    let lam = lon_deg * PI / 180.0;
    let e2 = 2.0 * F - F * F;
    let e_prime_sq = e2 / (1.0 - e2);
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    let tan_phi = sin_phi / cos_phi;
    let n = A / (1.0 - e2 * sin_phi * sin_phi).sqrt();
    let t = tan_phi * tan_phi;
    let c = e_prime_sq * cos_phi * cos_phi;
    let a_ = (lam - lon0) * cos_phi;

    let m = A
        * ((1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2.powi(3) / 256.0) * phi
            - (3.0 * e2 / 8.0 + 3.0 * e2 * e2 / 32.0 + 45.0 * e2.powi(3) / 1024.0)
                * (2.0 * phi).sin()
            + (15.0 * e2 * e2 / 256.0 + 45.0 * e2.powi(3) / 1024.0) * (4.0 * phi).sin()
            - (35.0 * e2.powi(3) / 3072.0) * (6.0 * phi).sin());

    let x = UTM_K0
        * n
        * (a_
            + (1.0 - t + c) * a_.powi(3) / 6.0
            + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * e_prime_sq) * a_.powi(5) / 120.0);
    let y = UTM_K0
        * (m + n
            * tan_phi
            * (a_ * a_ / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * c * c) * a_.powi(4) / 24.0
                + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * e_prime_sq) * a_.powi(6) / 720.0));
    (x + UTM_FALSE_EASTING, y)
}

// ---------------------------------------------------------------------------
// Bounding-box helpers
// ---------------------------------------------------------------------------

/// Compute a WGS84 lat/lon bounding box that covers the entire UTM
/// map extent, padded slightly so edge features aren't lost at the
/// boundary. Returns (south, west, north, east) in degrees.
pub fn spec_wgs84_bbox(
    spec: &BakeSpec,
    zone: u32,
    utm_to_wgs84: impl Fn(f64, f64) -> (f64, f64),
) -> Bbox {
    let e0 = spec.bounds.origin_east;
    let n0 = spec.bounds.origin_north;
    let e1 = e0 + spec.bounds.aligned_extent_x();
    let n1 = n0 - spec.bounds.aligned_extent_z();
    // UTM rectangle is not axis-aligned in lat/lon — compute all four corners.
    let corners = [
        utm_to_wgs84(e0, n0),
        utm_to_wgs84(e1, n0),
        utm_to_wgs84(e0, n1),
        utm_to_wgs84(e1, n1),
    ];
    let (mut lat_min, mut lat_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut lon_min, mut lon_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for (lat, lon) in corners {
        lat_min = lat_min.min(lat);
        lat_max = lat_max.max(lat);
        lon_min = lon_min.min(lon);
        lon_max = lon_max.max(lon);
    }
    // Expand by ~50 m on each side so ways exactly on the boundary
    // aren't cropped. 50 m ≈ 0.0005° in latitude.
    let pad = 0.001;
    let _ = zone;
    Bbox {
        south: lat_min - pad,
        west: lon_min - pad,
        north: lat_max + pad,
        east: lon_max + pad,
    }
}

// ---------------------------------------------------------------------------
// Rasterization
// ---------------------------------------------------------------------------

/// Apply OSM highways as a feature overlay on top of an existing
/// feature grid. Mutates `buffer` in place.
pub fn apply_osm_highways_overlay(
    resp: &OverpassResponse,
    spec: &BakeSpec,
    zone: u32,
    width: u32,
    height: u32,
    buffer: &mut [u8],
) {
    let spacing = spec.bounds.spacing as f64;
    let ox = spec.bounds.origin_east;
    let oy = spec.bounds.origin_north;

    let mut count_paved = 0usize;
    let mut count_unpaved = 0usize;
    let mut count_trail = 0usize;

    for elem in &resp.elements {
        if elem.element_type != "way" {
            continue;
        }
        let Some(class) = classify_highway(&elem.tags) else {
            continue;
        };
        if elem.geometry.len() < 2 {
            continue;
        }
        let target_class = class.feature_class();
        let brush_radius_cells = class.width_m() * 0.5 / spacing;
        // A way tagged `bridge=yes` (or `viaduct`, `aqueduct`, etc.)
        // is allowed to paint over water. Without this, bridges that
        // cross rivers get skipped by the "never paint over Water"
        // precedence rule and disappear from the feature grid.
        let is_bridge = matches!(
            elem.tags.get("bridge").map(String::as_str),
            Some(
                "yes"
                    | "viaduct"
                    | "aqueduct"
                    | "boardwalk"
                    | "cantilever"
                    | "covered"
                    | "movable"
                    | "simple_brunnel"
                    | "trestle"
                    | "truss"
            )
        );

        let points: Vec<(f64, f64)> = elem
            .geometry
            .iter()
            .map(|p| {
                let (east, north) = wgs84_to_utm_zone_n(p.lat, p.lon, zone);
                let col = (east - ox) / spacing;
                let row = (oy - north) / spacing;
                (col, row)
            })
            .collect();

        for seg in points.windows(2) {
            rasterize_segment(
                width,
                height,
                buffer,
                seg[0],
                seg[1],
                brush_radius_cells,
                target_class,
                is_bridge,
            );
        }
        match class {
            RoadClass::Paved => count_paved += 1,
            RoadClass::Unpaved => count_unpaved += 1,
            RoadClass::Trail => count_trail += 1,
        }
    }
    println!(
        "  rasterized OSM: {count_paved} paved, {count_unpaved} unpaved, {count_trail} trails"
    );
}

#[allow(clippy::too_many_arguments)]
fn rasterize_segment(
    grid_w: u32,
    grid_h: u32,
    buffer: &mut [u8],
    p1: (f64, f64),
    p2: (f64, f64),
    brush_radius_cells: f64,
    target: FeatureClass,
    is_bridge: bool,
) {
    let w = grid_w as i32;
    let h = grid_h as i32;
    let r = brush_radius_cells.ceil() as i32;
    let x_min = (p1.0.min(p2.0).floor() as i32 - r).max(0);
    let x_max = (p1.0.max(p2.0).ceil() as i32 + r).min(w - 1);
    let y_min = (p1.1.min(p2.1).floor() as i32 - r).max(0);
    let y_max = (p1.1.max(p2.1).ceil() as i32 + r).min(h - 1);
    if x_min > x_max || y_min > y_max {
        return;
    }
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let len_sq = dx * dx + dy * dy;
    let r_sq = brush_radius_cells * brush_radius_cells;
    let target_byte = target as u8;
    let water = FeatureClass::Water as u8;
    let paved = FeatureClass::PavedRoad as u8;
    let unpaved = FeatureClass::UnpavedRoad as u8;

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let px = x as f64;
            let py = y as f64;
            let t = if len_sq > 1e-9 {
                ((px - p1.0) * dx + (py - p1.1) * dy) / len_sq
            } else {
                0.0
            };
            let t = t.clamp(0.0, 1.0);
            let cx = p1.0 + t * dx;
            let cy = p1.1 + t * dy;
            let dsq = (px - cx).powi(2) + (py - cy).powi(2);
            if dsq > r_sq {
                continue;
            }
            let idx = (y * w + x) as usize;
            let existing = buffer[idx];
            // Never paint over water — EXCEPT when the way is tagged
            // as a bridge. That's how Bridge of the Gods, Hood River
            // Bridge, and every other road crossing a river actually
            // renders on the feature grid; without the override,
            // the water-precedence rule skipped them entirely.
            if existing == water && !is_bridge {
                continue;
            }
            // Trails don't override roads; unpaved doesn't override paved.
            let allow = match target_byte {
                t if t == FeatureClass::Trail as u8 => existing != paved && existing != unpaved,
                t if t == unpaved => existing != paved,
                _ => true,
            };
            if allow {
                buffer[idx] = target_byte;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OSM vector-polygon land cover
// ---------------------------------------------------------------------------

/// Map an OSM tag bag → canonical [`FeatureClass`] for a polygon
/// feature. Returns `None` when no tag in the bag is one we care
/// about. Mirror of [`classify_highway`] for way/relation polygons
/// instead of ways with `highway=*`.
///
/// Precedence inside the bag: water wins over vegetation wins over
/// landuse wins over bare/wetland/snow. This matters when a polygon
/// is double-tagged (e.g. a `natural=water` also tagged
/// `landuse=reservoir` — read as water).
pub fn classify_osm_polygon(tags: &BTreeMap<String, String>) -> Option<FeatureClass> {
    let t = |k: &str| tags.get(k).map(String::as_str);

    // Water first — rivers, lakes, reservoirs, riverbanks.
    if matches!(t("natural"), Some("water"))
        || tags.contains_key("water")
        || matches!(t("waterway"), Some("riverbank" | "dock"))
        || matches!(t("landuse"), Some("reservoir" | "basin"))
    {
        return Some(FeatureClass::Water);
    }

    // Glacier / permanent snow — before vegetation so high-altitude
    // icefields don't read as grassland.
    if matches!(t("natural"), Some("glacier" | "snow")) {
        return Some(FeatureClass::Snow);
    }

    // Forest: `natural=wood` (untouched) or `landuse=forest` (managed).
    if matches!(t("natural"), Some("wood")) || matches!(t("landuse"), Some("forest" | "wood")) {
        return Some(FeatureClass::Forest);
    }

    // Shrub / scrub / heath.
    if matches!(t("natural"), Some("scrub" | "heath")) {
        return Some(FeatureClass::Shrubland);
    }

    // Grassland / meadow.
    if matches!(t("natural"), Some("grassland" | "fell"))
        || matches!(t("landuse"), Some("meadow" | "grass" | "village_green"))
        || matches!(t("leisure"), Some("park" | "golf_course" | "pitch"))
    {
        return Some(FeatureClass::Grassland);
    }

    // Cropland.
    if matches!(
        t("landuse"),
        Some("farmland" | "orchard" | "vineyard" | "allotments")
    ) {
        return Some(FeatureClass::Cropland);
    }

    // Built-up: residential, commercial, industrial, retail.
    if matches!(
        t("landuse"),
        Some(
            "residential"
                | "commercial"
                | "industrial"
                | "retail"
                | "construction"
                | "military"
                | "railway"
                | "cemetery"
                | "institutional"
                | "garages"
        )
    ) || matches!(t("aeroway"), Some("aerodrome"))
    {
        return Some(FeatureClass::BuiltUp);
    }

    // Bare ground / rock / scree.
    if matches!(
        t("natural"),
        Some("bare_rock" | "scree" | "shingle" | "sand" | "beach")
    ) || matches!(t("landuse"), Some("quarry" | "landfill"))
    {
        return Some(FeatureClass::Bare);
    }

    // Wetland.
    if matches!(t("natural"), Some("wetland" | "marsh" | "bog")) {
        return Some(FeatureClass::Wetland);
    }

    None
}

/// Fetch OSM polygon land-cover features (`natural=*` / `landuse=*`
/// / `water=*`) for a WGS84 bounding box. Mirrors [`fetch_highways`]
/// — same mirror-fallback loop, same on-disk cache pattern, different
/// cache key + query. Pulls both ways (simple polygons) and
/// `type=multipolygon` relations (for big features with islands /
/// enclaves).
pub fn fetch_osm_landcover(bbox: &Bbox, cache_dir: &Path) -> Result<OverpassResponse> {
    fs::create_dir_all(cache_dir)?;
    let cache_file = cache_dir.join(format!(
        "overpass-landcover-{:.4}_{:.4}_{:.4}_{:.4}.json",
        bbox.south, bbox.west, bbox.north, bbox.east,
    ));
    if cache_file.exists() {
        let text = fs::read_to_string(&cache_file)
            .with_context(|| format!("reading {}", cache_file.display()))?;
        let resp: OverpassResponse = serde_json::from_str(&text)?;
        return Ok(resp);
    }

    // `out geom;` gives us inline geometry on way elements and, for
    // relation elements, on each member — which is what we need to
    // assemble multipolygons without a second pass. We over-fetch
    // (query more tags than we rasterize) rather than enumerate
    // every recognized value in the query itself; classification
    // happens on our side via `classify_osm_polygon`.
    let query = format!(
        "[out:json][timeout:180];\
         (\
           way[natural]({s},{w},{n},{e});\
           way[landuse]({s},{w},{n},{e});\
           way[water]({s},{w},{n},{e});\
           way[\"waterway\"=\"riverbank\"]({s},{w},{n},{e});\
           way[leisure]({s},{w},{n},{e});\
           relation[natural][type=multipolygon]({s},{w},{n},{e});\
           relation[landuse][type=multipolygon]({s},{w},{n},{e});\
           relation[water][type=multipolygon]({s},{w},{n},{e});\
         );\
         out geom;",
        s = bbox.south,
        w = bbox.west,
        n = bbox.north,
        e = bbox.east,
    );
    const MIRRORS: &[&str] = &[
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ];
    println!(
        "  querying Overpass for landcover polygons in bbox [{}]…",
        bbox.short()
    );
    let mut last_err: Option<String> = None;
    for mirror in MIRRORS {
        let output = Command::new("curl")
            .arg("-fsSL")
            .arg("--max-time")
            .arg("180")
            .arg("-X")
            .arg("POST")
            .arg("--data-urlencode")
            .arg(format!("data={query}"))
            .arg(*mirror)
            .output()
            .context("running curl for Overpass")?;
        if !output.status.success() {
            last_err = Some(format!(
                "{mirror}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
            continue;
        }
        match serde_json::from_slice::<OverpassResponse>(&output.stdout) {
            Ok(resp) => {
                fs::write(&cache_file, &output.stdout)?;
                return Ok(resp);
            }
            Err(e) => {
                last_err = Some(format!("{mirror}: non-JSON response ({e})"));
                continue;
            }
        }
    }
    Err(anyhow!(
        "all Overpass mirrors failed. last error: {}",
        last_err.unwrap_or_default()
    ))
}

/// Rasterize a polygon (plus optional inner rings / holes) into the
/// feature grid by even-odd scanline fill. All rings are given in
/// grid-cell floating-point coordinates (`x` = column, `y` = row).
///
/// Even-odd rule handles holes automatically: every scanline is
/// filled in alternating "in/out" segments based on edge crossings,
/// so inner rings subtract from outer fills regardless of winding.
///
/// `overwrite_water` — when true (used for water polygons), paints
/// through any existing class. When false, skips cells that are
/// already `Water` (so landuse=forest next to an existing river
/// doesn't flood into the river).
fn rasterize_polygon(
    grid_w: u32,
    grid_h: u32,
    buffer: &mut [u8],
    outer: &[(f64, f64)],
    inners: &[Vec<(f64, f64)>],
    target_class: FeatureClass,
    overwrite_water: bool,
) {
    if outer.len() < 3 {
        return;
    }
    let w = grid_w as i32;
    let h = grid_h as i32;
    let target_byte = target_class as u8;
    let water_byte = FeatureClass::Water as u8;

    // Collect edges from every ring. An edge is represented as
    // (y_min, y_max, x_at_y_min, slope_x_per_y). Horizontal edges
    // contribute no intersections so they're filtered out.
    let mut edges: Vec<(f64, f64, f64, f64)> = Vec::new();
    let push_ring = |edges: &mut Vec<(f64, f64, f64, f64)>, ring: &[(f64, f64)]| {
        let n = ring.len();
        if n < 3 {
            return;
        }
        for i in 0..n {
            let (x0, y0) = ring[i];
            let (x1, y1) = ring[(i + 1) % n];
            if (y1 - y0).abs() < f64::EPSILON {
                continue;
            }
            let (ya, xa, yb, xb) = if y0 < y1 {
                (y0, x0, y1, x1)
            } else {
                (y1, x1, y0, x0)
            };
            let slope = (xb - xa) / (yb - ya);
            edges.push((ya, yb, xa, slope));
        }
    };
    push_ring(&mut edges, outer);
    for inner in inners {
        push_ring(&mut edges, inner);
    }
    if edges.is_empty() {
        return;
    }

    // Overall polygon y-range, clipped to the grid.
    let y_min_f = edges.iter().map(|e| e.0).fold(f64::INFINITY, f64::min);
    let y_max_f = edges.iter().map(|e| e.1).fold(f64::NEG_INFINITY, f64::max);
    let y0 = (y_min_f.ceil() as i32).max(0);
    let y1 = (y_max_f.floor() as i32).min(h - 1);

    let mut xs: Vec<f64> = Vec::with_capacity(edges.len());
    for y in y0..=y1 {
        let yf = y as f64 + 0.5;
        xs.clear();
        for &(ya, yb, xa, slope) in &edges {
            // Half-open [ya, yb): edge contributes when yf is on ya
            // but not on yb. Avoids double-counting at vertex joins.
            if yf >= ya && yf < yb {
                xs.push(xa + (yf - ya) * slope);
            }
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Fill pairs of x intersections as spans.
        let row_base = (y * w) as usize;
        for pair in xs.chunks_exact(2) {
            let x_start = (pair[0].ceil() as i32).max(0);
            let x_end = (pair[1].floor() as i32).min(w - 1);
            if x_end < x_start {
                continue;
            }
            for x in x_start..=x_end {
                let idx = row_base + x as usize;
                if !overwrite_water && buffer[idx] == water_byte {
                    continue;
                }
                buffer[idx] = target_byte;
            }
        }
    }
}

/// Apply OSM vector-polygon land cover on top of the ESA raster
/// classification. Polygons with a classifiable tag bag
/// (see [`classify_osm_polygon`]) are rasterized by even-odd fill;
/// multipolygon relations assemble `outer` / `inner` rings from
/// member ways.
///
/// Precedence: Water polygons paint through everything in this
/// pass. Non-water polygons skip cells already painted as Water
/// (so a forest touching a river doesn't overwrite the river).
/// The OSM highway overlay runs *after* this pass and still wins
/// over polygons where they overlap roads.
///
/// Each map's `landcover = true` opt-in triggers this pass between
/// the ESA+slope bake and the OSM highway overlay (see
/// `bake.rs::bake_map`).
pub fn apply_osm_polygon_landcover(
    resp: &OverpassResponse,
    spec: &BakeSpec,
    zone: u32,
    width: u32,
    height: u32,
    buffer: &mut [u8],
) {
    let spacing = spec.bounds.spacing as f64;
    let ox = spec.bounds.origin_east;
    let oy = spec.bounds.origin_north;

    let project = |pts: &[OverpassPoint]| -> Vec<(f64, f64)> {
        pts.iter()
            .map(|p| {
                let (east, north) = wgs84_to_utm_zone_n(p.lat, p.lon, zone);
                let col = (east - ox) / spacing;
                let row = (oy - north) / spacing;
                (col, row)
            })
            .collect()
    };

    let mut count_by_class: [usize; 24] = [0; 24];

    for elem in &resp.elements {
        let Some(class) = classify_osm_polygon(&elem.tags) else {
            continue;
        };
        let overwrite_water = matches!(class, FeatureClass::Water);

        if elem.element_type == "way" {
            if elem.geometry.len() < 3 {
                continue;
            }
            let outer = project(&elem.geometry);
            rasterize_polygon(width, height, buffer, &outer, &[], class, overwrite_water);
            count_by_class[class as usize] += 1;
        } else if elem.element_type == "relation" {
            // Assemble outer + inner rings from members. Skip member
            // roles we don't recognize; a malformed multipolygon with
            // no outers is just dropped silently (the classifier
            // already deemed it a candidate, but without geometry we
            // can't paint anything).
            let mut outers: Vec<Vec<(f64, f64)>> = Vec::new();
            let mut inners: Vec<Vec<(f64, f64)>> = Vec::new();
            for m in &elem.members {
                if m.member_type != "way" || m.geometry.len() < 3 {
                    continue;
                }
                let ring = project(&m.geometry);
                match m.role.as_str() {
                    "outer" | "" => outers.push(ring),
                    "inner" => inners.push(ring),
                    _ => continue,
                }
            }
            // Pair each outer with all inners — cheap and correct
            // for the common case (one outer + N holes). Complex
            // multi-outer relations (e.g. an archipelago of islands
            // as separate outers) end up with holes applied to every
            // outer, which would over-subtract. For our bake bbox
            // sizes this case is rare; we accept the approximation.
            for outer in &outers {
                rasterize_polygon(
                    width,
                    height,
                    buffer,
                    outer,
                    &inners,
                    class,
                    overwrite_water,
                );
            }
            if !outers.is_empty() {
                count_by_class[class as usize] += 1;
            }
        }
    }

    let summary = [
        (FeatureClass::Water, "water"),
        (FeatureClass::Forest, "forest"),
        (FeatureClass::Shrubland, "shrub"),
        (FeatureClass::Grassland, "grass"),
        (FeatureClass::Cropland, "crop"),
        (FeatureClass::BuiltUp, "built"),
        (FeatureClass::Bare, "bare"),
        (FeatureClass::Wetland, "wetland"),
        (FeatureClass::Snow, "snow"),
    ];
    let parts: Vec<String> = summary
        .iter()
        .filter_map(|(c, name)| {
            let n = count_by_class[*c as usize];
            if n == 0 {
                None
            } else {
                Some(format!("{n} {name}"))
            }
        })
        .collect();
    println!("  rasterized OSM landcover: {}", parts.join(", "));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(k: &str, v: &str) -> BTreeMap<String, String> {
        let mut t = BTreeMap::new();
        t.insert(k.into(), v.into());
        t
    }

    fn tags(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn classifies_major_roads_as_paved() {
        assert_eq!(
            classify_highway(&tag("highway", "motorway")),
            Some(RoadClass::Paved)
        );
        assert_eq!(
            classify_highway(&tag("highway", "secondary")),
            Some(RoadClass::Paved)
        );
        assert_eq!(
            classify_highway(&tag("highway", "residential")),
            Some(RoadClass::Paved)
        );
    }

    #[test]
    fn classifies_tracks_as_unpaved() {
        assert_eq!(
            classify_highway(&tag("highway", "track")),
            Some(RoadClass::Unpaved)
        );
    }

    #[test]
    fn service_defaults_to_unpaved_but_flips_on_surface() {
        assert_eq!(
            classify_highway(&tag("highway", "service")),
            Some(RoadClass::Unpaved)
        );
        assert_eq!(
            classify_highway(&tags(&[("highway", "service"), ("surface", "asphalt")])),
            Some(RoadClass::Paved)
        );
    }

    #[test]
    fn surface_tag_flips_major_road_to_unpaved() {
        assert_eq!(
            classify_highway(&tags(&[("highway", "tertiary"), ("surface", "dirt")])),
            Some(RoadClass::Unpaved)
        );
    }

    #[test]
    fn path_and_footway_are_trails() {
        assert_eq!(
            classify_highway(&tag("highway", "path")),
            Some(RoadClass::Trail)
        );
        assert_eq!(
            classify_highway(&tag("highway", "footway")),
            Some(RoadClass::Trail)
        );
        assert_eq!(
            classify_highway(&tag("highway", "bridleway")),
            Some(RoadClass::Trail)
        );
    }

    #[test]
    fn unknown_highway_types_skip() {
        assert_eq!(classify_highway(&tag("highway", "proposed")), None);
        assert_eq!(classify_highway(&tag("other", "value")), None);
    }

    #[test]
    fn forward_utm_lands_in_expected_zone_range() {
        // For a point well within UTM 10N (lon -122.41°, which is
        // ~0.6° east of zone 10N's central meridian), forward
        // projection should put easting noticeably east of the
        // false easting 500 000.
        let (e, n) = wgs84_to_utm_zone_n(45.54, -122.41, 10);
        assert!(e > 530_000.0 && e < 560_000.0, "easting out of range: {e}");
        assert!(
            n > 5_035_000.0 && n < 5_060_000.0,
            "northing out of range: {n}"
        );
    }

    // ---- OSM polygon landcover -----------------------------------------

    #[test]
    fn classify_osm_polygon_recognizes_water_tags() {
        assert_eq!(
            classify_osm_polygon(&tag("natural", "water")),
            Some(FeatureClass::Water)
        );
        assert_eq!(
            classify_osm_polygon(&tag("waterway", "riverbank")),
            Some(FeatureClass::Water)
        );
        assert_eq!(
            classify_osm_polygon(&tag("landuse", "reservoir")),
            Some(FeatureClass::Water)
        );
    }

    #[test]
    fn classify_osm_polygon_recognizes_forest_tags() {
        assert_eq!(
            classify_osm_polygon(&tag("natural", "wood")),
            Some(FeatureClass::Forest)
        );
        assert_eq!(
            classify_osm_polygon(&tag("landuse", "forest")),
            Some(FeatureClass::Forest)
        );
    }

    #[test]
    fn classify_osm_polygon_prefers_water_over_landuse() {
        // A water body tagged with both `natural=water` and a
        // landuse hint must classify as Water — protects rivers /
        // reservoirs from being labeled forest or grassland when
        // mappers double-tag.
        let bag = tags(&[("natural", "water"), ("landuse", "reservoir")]);
        assert_eq!(classify_osm_polygon(&bag), Some(FeatureClass::Water));
    }

    #[test]
    fn classify_osm_polygon_skips_untracked_tags() {
        // A polygon with tags we don't map (amenity, highway, random)
        // returns None so the apply loop drops it silently.
        assert_eq!(classify_osm_polygon(&tag("amenity", "parking")), None);
        assert_eq!(classify_osm_polygon(&tag("highway", "footway")), None);
        assert_eq!(classify_osm_polygon(&BTreeMap::new()), None);
    }

    #[test]
    fn rasterize_polygon_fills_axis_aligned_square() {
        // A 4×4 solid square in a 10×10 grid. Verify all 16 interior
        // cells paint Forest; border stays Unknown.
        let w = 10u32;
        let h = 10u32;
        let mut buf = vec![FeatureClass::Unknown as u8; (w * h) as usize];
        let outer: Vec<(f64, f64)> = vec![(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)];
        rasterize_polygon(w, h, &mut buf, &outer, &[], FeatureClass::Forest, false);
        // Inner cells at (4,4)..(6,6) should be Forest.
        for y in 4..=6 {
            for x in 4..=6 {
                assert_eq!(
                    buf[y * w as usize + x],
                    FeatureClass::Forest as u8,
                    "interior cell ({x},{y}) not filled"
                );
            }
        }
        // Outside corners stay unknown.
        assert_eq!(buf[0], FeatureClass::Unknown as u8);
        assert_eq!(buf[(w * h - 1) as usize], FeatureClass::Unknown as u8);
    }

    #[test]
    fn rasterize_polygon_respects_inner_hole() {
        // An outer 20×20 square with a 6×6 inner hole.
        // Even-odd fill should leave the hole interior Unknown.
        let w = 30u32;
        let h = 30u32;
        let mut buf = vec![FeatureClass::Unknown as u8; (w * h) as usize];
        let outer: Vec<(f64, f64)> = vec![(5.0, 5.0), (25.0, 5.0), (25.0, 25.0), (5.0, 25.0)];
        let inner: Vec<(f64, f64)> = vec![(12.0, 12.0), (18.0, 12.0), (18.0, 18.0), (12.0, 18.0)];
        rasterize_polygon(
            w,
            h,
            &mut buf,
            &outer,
            &[inner],
            FeatureClass::Forest,
            false,
        );
        // Middle of hole — Unknown.
        assert_eq!(
            buf[15 * w as usize + 15],
            FeatureClass::Unknown as u8,
            "hole interior was filled"
        );
        // Just inside the outer, outside the hole — Forest.
        assert_eq!(
            buf[8 * w as usize + 8],
            FeatureClass::Forest as u8,
            "outer ring was not filled"
        );
    }

    #[test]
    fn rasterize_polygon_preserves_water_when_not_overwriting() {
        // With `overwrite_water = false`, a landuse polygon must not
        // flood into existing Water cells.
        let w = 10u32;
        let h = 10u32;
        let mut buf = vec![FeatureClass::Water as u8; (w * h) as usize];
        let outer: Vec<(f64, f64)> = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        rasterize_polygon(w, h, &mut buf, &outer, &[], FeatureClass::Forest, false);
        assert!(
            buf.iter().all(|&b| b == FeatureClass::Water as u8),
            "non-water polygon leaked into water cells"
        );
    }

    #[test]
    fn rasterize_polygon_overwrites_when_flag_set() {
        // Water polygons paint through anything, since open water is
        // the most authoritative classification at the site.
        let w = 10u32;
        let h = 10u32;
        let mut buf = vec![FeatureClass::Forest as u8; (w * h) as usize];
        let outer: Vec<(f64, f64)> = vec![(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)];
        rasterize_polygon(w, h, &mut buf, &outer, &[], FeatureClass::Water, true);
        assert_eq!(
            buf[5 * w as usize + 5],
            FeatureClass::Water as u8,
            "water polygon failed to paint through forest"
        );
    }

    #[test]
    fn rasterize_polygon_clips_to_grid_edges() {
        // Polygon extending beyond the grid on two sides must clip
        // cleanly, not panic or wrap. Verify one corner and the
        // opposite off-grid side.
        let w = 8u32;
        let h = 8u32;
        let mut buf = vec![FeatureClass::Unknown as u8; (w * h) as usize];
        let outer: Vec<(f64, f64)> = vec![(-3.0, -3.0), (4.0, -3.0), (4.0, 4.0), (-3.0, 4.0)];
        rasterize_polygon(w, h, &mut buf, &outer, &[], FeatureClass::Forest, false);
        assert_eq!(buf[0], FeatureClass::Forest as u8, "NW corner unfilled");
        // Far SE corner of grid is outside the polygon — stays unknown.
        assert_eq!(
            buf[(h - 1) as usize * w as usize + (w - 1) as usize],
            FeatureClass::Unknown as u8
        );
    }
}
