//! `terrain.toml` schema — describes the canonical heightmap alongside it.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CURRENT_FORMAT_VERSION: u32 = 2;

/// Metadata sidecar for a canonical heightmap.
///
/// Serialized as `terrain.toml` alongside `heightmap.r32` in the map's
/// asset directory. Keep field additions backwards-compatible; bump
/// [`CURRENT_FORMAT_VERSION`] on schema changes.
///
/// **Format history.**
/// - v1 (retired 2026-05-03): `heightmap.r16` u16 LE. `vert_min_m` /
///   `vert_max_m` defined the linear scaling that decoded each u16
///   to meters.
/// - v2 (current): `heightmap.r32` f32 LE, literal meters. `vert_min_m`
///   / `vert_max_m` are gameplay metadata only (camera bounds, sky
///   shader hints, the lossy `.png` inspection path) and no longer
///   participate in storage decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainMetadata {
    /// Schema version. Loader rejects unknown versions.
    pub format_version: u32,

    /// Stable identifier (matches the asset directory name, e.g. `"corbett"`).
    pub map_id: String,

    /// Grid width in samples.
    pub width: u32,

    /// Grid height in samples.
    pub height: u32,

    /// Horizontal distance between adjacent samples, in meters.
    pub spacing_m: f32,

    /// Lower gameplay clamp on elevation (meters). Used as a hint
    /// for camera bounds, sky shader, and the lossy 16-bit PNG
    /// export path. **No longer** defines storage encoding (v2
    /// stores literal f32 meters in `.r32`); was the u16=0 anchor
    /// in v1.
    pub vert_min_m: f32,

    /// Upper gameplay clamp on elevation (meters). See
    /// [`Self::vert_min_m`] for the v1→v2 semantic shift.
    pub vert_max_m: f32,

    /// UTM zone designator, e.g. `"10N"`. Game-world origin lives at
    /// (`origin_utm_easting`, `origin_utm_northing`) in that zone.
    pub origin_utm_zone: String,

    /// UTM easting (meters) corresponding to world-local `(x=0)`.
    pub origin_utm_easting: f64,

    /// UTM northing (meters) corresponding to world-local `(z=0)` on the north edge.
    pub origin_utm_northing: f64,

    /// BLAKE3 hex digest of the `.r32` file contents. Empty string
    /// skips the integrity check (used for ephemeral test fixtures
    /// and the editor-side `Sync to Canonical` path before its
    /// gdext hash helper recomputes the digest).
    #[serde(default)]
    pub blake3: String,

    /// BLAKE3 hex digest of the paired `features.r8` file, if one
    /// was produced at bake time. Empty string (the default when
    /// the field is absent) means the map has no feature layer —
    /// the client falls back to slope-derived vertex coloring.
    #[serde(default)]
    pub features_blake3: String,

    /// Region edge length in world meters that this bake aligns to.
    /// `(W - 1) * spacing_m` is guaranteed to be a multiple of this
    /// value (and likewise `(H - 1) * spacing_m`), so Terrain3D's
    /// region grid tiles the map without partial regions. Default
    /// `2048.0` matches Terrain3D's stock 1024-vertex regions at
    /// 2 m spacing. Older `terrain.toml` files (pre-alignment
    /// contract) get this default — re-bake them to make the
    /// alignment guarantee real.
    ///
    /// See `docs/book/src/planning/static-foliage-plan.md` →
    /// "Cross-layer conventions" for the full contract.
    #[serde(default = "default_region_size_m")]
    pub region_size_m: f32,

    /// **Playable** area extent in world meters — the geographic
    /// region the bake spec originally requested. Always ≤ the
    /// rendered canonical extent (`(W-1)*spacing_m`); the difference
    /// is the padded strip that exists for region alignment but
    /// isn't intended for gameplay. Sim-side and foliage-side code
    /// filters placements to within `[-playable/2, +playable/2]`
    /// (centered convention). When `0.0` (the legacy default),
    /// callers fall back to the full canonical extent.
    #[serde(default)]
    pub playable_extent_x_m: f32,

    /// Playable Z extent. See [`Self::playable_extent_x_m`].
    #[serde(default)]
    pub playable_extent_z_m: f32,

    /// Format version of the paired `nav_mask.r8` file, if one was
    /// produced. `0` (the default when the field is absent) means
    /// the map has no nav-override layer; the sim's nav grid is
    /// built purely from slope + feature class.
    ///
    /// Iteration 5-13 ships `NAV_MASK_FORMAT_VERSION = 1` —
    /// `crate::nav_mask::decode` maps `0 → Default`, `1 → ForceBlocked`,
    /// `2 → ForceWalkable`. See
    /// `docs/book/src/planning/sim-iteration-5-13-plan.md`.
    #[serde(default)]
    pub nav_mask_format_version: u8,

    /// BLAKE3 hex digest of the paired `nav_mask.r8` file, if one
    /// was produced. Empty string (the default when the field is
    /// absent) means the map has no nav-override layer. Validated
    /// at load time the same way `features_blake3` is.
    #[serde(default)]
    pub nav_mask_blake3: String,
}

fn default_region_size_m() -> f32 {
    2048.0
}

impl TerrainMetadata {
    /// Number of f32 samples the paired `.r32` file must contain.
    pub fn sample_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// World-local extent of the map in meters: `[width_m, height_m]`.
    /// Measured from the first sample to the last — i.e. `(W - 1)`
    /// cells wide, not `W`. This matches the convention used by both
    /// the ArrayMesh vertex layout (first vertex at x=0, last at
    /// x=(W-1)*spacing) and Godot's `HeightMapShape3D` collision
    /// extent. Using `W * spacing` instead gives a half-cell offset
    /// that multiplies into meaningful Y error on any real slope —
    /// e.g. 2 m XZ drift becomes ~1 m Y drift on a 50% grade, enough
    /// to bury an NPC up to the waist.
    pub fn extent_m(&self) -> [f32; 2] {
        [
            (self.width - 1) as f32 * self.spacing_m,
            (self.height - 1) as f32 * self.spacing_m,
        ]
    }

    /// Load just the metadata sidecar from a map's asset directory.
    /// Cheaper than [`Heightmap::load`] when only the geographic
    /// frame is needed (e.g. computing a sibling map's offset for
    /// a backdrop without paying to deserialize the full
    /// `heightmap.r32`).
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("terrain.toml");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let meta: Self =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(meta)
    }
}
