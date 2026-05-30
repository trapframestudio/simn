//! [`Heightmap`] — the public type loaded once per map, queried per tick.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

use crate::features::FeatureClass;
use crate::metadata::{TerrainMetadata, CURRENT_FORMAT_VERSION};
use crate::nav_mask::{self, NavOverride, NAV_MASK_FORMAT_VERSION};
use crate::sampler::{bilinear_f32, decode_r32};

/// A loaded, validated heightmap for one map.
///
/// Construct with [`Heightmap::load`]. All queries are world-local:
/// `(0, 0)` is the NW corner, +X goes east, +Z goes south, matching
/// Godot's left-handed coordinate convention.
///
/// Optionally carries a paired `features.r8` classification layer
/// on the same `W × H` grid; [`Heightmap::sample_feature`] looks up
/// the class at a given world position, falling back to
/// [`FeatureClass::Unknown`] when no feature layer was produced at
/// bake time.
///
/// Samples are **literal meters above sea level**, stored as f32
/// (canonical `heightmap.r32`, format_version 2). The legacy u16
/// path was retired 2026-05-03; see [`crate::sampler::legacy_v1`].
pub struct Heightmap {
    metadata: TerrainMetadata,
    samples: Vec<f32>,
    features: Option<Vec<u8>>,
    /// Splatmap A — RGBA8, length `4 * W * H`. R = Forest channel,
    /// G = Grassland, B = Water, A = Cropland (see
    /// `splatmap::SPLATMAP_A_CHANNELS`). `None` when the bake didn't
    /// produce one (legacy maps from before the splatmap pipeline).
    splatmap_a: Option<Vec<u8>>,
    /// Splatmap B — RGBA8, length `4 * W * H`. R = Bare, G = BuiltUp,
    /// B = Cliff, A = Snow.
    splatmap_b: Option<Vec<u8>>,
    /// Road density — RGBA8, length `4 * W * H`. R = PavedRoad
    /// density, G = UnpavedRoad, B = Trail, A = reserved. Smoothed
    /// at bake time via per-class Gaussian; sampled with
    /// `filter_linear` in the shader for sub-cell-precision road
    /// edges. `None` for legacy bakes.
    road_density: Option<Vec<u8>>,
    /// Iteration 5-13 Phase A1: designer-painted nav-override mask.
    /// Length `W * H`, row-major NW-up; one byte per cell decoded via
    /// [`nav_mask::decode`]. `None` when the map has no painted
    /// overrides (`metadata.nav_mask_blake3` empty). See
    /// `docs/book/src/planning/sim-iteration-5-13-plan.md`.
    nav_mask: Option<Vec<u8>>,
}

impl std::fmt::Debug for Heightmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Heightmap")
            .field("metadata", &self.metadata)
            .field(
                "samples",
                &format_args!("[{} f32 meters]", self.samples.len()),
            )
            .finish()
    }
}

impl Heightmap {
    /// Construct directly from parts. Skips disk I/O and the BLAKE3
    /// integrity check; intended for tests and procedural generators.
    /// Returns an error only if `samples.len()` mismatches the metadata.
    pub fn from_raw(metadata: TerrainMetadata, samples: Vec<f32>) -> Result<Self> {
        if samples.len() != metadata.sample_count() {
            return Err(anyhow!(
                "terrain {} sample count mismatch: got {} expected {} ({}x{})",
                metadata.map_id,
                samples.len(),
                metadata.sample_count(),
                metadata.width,
                metadata.height
            ));
        }
        Ok(Self {
            metadata,
            samples,
            features: None,
            splatmap_a: None,
            splatmap_b: None,
            road_density: None,
            nav_mask: None,
        })
    }

    /// Test-only constructor: build a [`Heightmap`] with explicit
    /// metadata, samples, optional feature layer, and optional
    /// nav-override mask. Skips disk I/O and integrity checks.
    /// Used by the sim's `nav.rs` tests + the
    /// `nav_mask_e2e.rs` integration test that paints a corridor
    /// block and asserts A* routes around it.
    #[doc(hidden)]
    pub fn from_raw_with_layers(
        metadata: TerrainMetadata,
        samples: Vec<f32>,
        features: Option<Vec<u8>>,
        nav_mask: Option<Vec<u8>>,
    ) -> Result<Self> {
        let expected = metadata.sample_count();
        if samples.len() != expected {
            return Err(anyhow!(
                "terrain {} sample count mismatch: got {} expected {}",
                metadata.map_id,
                samples.len(),
                expected
            ));
        }
        if let Some(f) = features.as_ref() {
            if f.len() != expected {
                return Err(anyhow!(
                    "terrain {} features size mismatch: got {} expected {}",
                    metadata.map_id,
                    f.len(),
                    expected
                ));
            }
        }
        if let Some(m) = nav_mask.as_ref() {
            if m.len() != expected {
                return Err(anyhow!(
                    "terrain {} nav_mask size mismatch: got {} expected {}",
                    metadata.map_id,
                    m.len(),
                    expected
                ));
            }
        }
        Ok(Self {
            metadata,
            samples,
            features,
            splatmap_a: None,
            splatmap_b: None,
            road_density: None,
            nav_mask,
        })
    }

    /// Load the canonical heightmap + metadata from a directory.
    ///
    /// Expects `dir/terrain.toml` and `dir/heightmap.r32` to exist.
    /// Validates that sample count matches the declared grid size and
    /// (if set) that the BLAKE3 digest of `heightmap.r32` matches
    /// `metadata.blake3`.
    pub fn load(dir: &Path) -> Result<Self> {
        let toml_path = dir.join("terrain.toml");
        let r32_path = dir.join("heightmap.r32");

        let toml_text = fs::read_to_string(&toml_path)
            .with_context(|| format!("reading {}", toml_path.display()))?;
        let metadata: TerrainMetadata = toml::from_str(&toml_text)
            .with_context(|| format!("parsing {}", toml_path.display()))?;

        if metadata.format_version != CURRENT_FORMAT_VERSION {
            return Err(anyhow!(
                "unsupported terrain format version {} (this build expects {}). \
                 Run `cargo run -p simn-terrain --bin migrate_canonical_format` \
                 to migrate v1 (.r16) maps to v2 (.r32).",
                metadata.format_version,
                CURRENT_FORMAT_VERSION
            ));
        }
        if metadata.width == 0 || metadata.height == 0 {
            return Err(anyhow!(
                "terrain {} has zero-sized grid ({}x{})",
                metadata.map_id,
                metadata.width,
                metadata.height
            ));
        }
        if metadata.spacing_m <= 0.0 {
            return Err(anyhow!(
                "terrain {} has non-positive spacing_m {}",
                metadata.map_id,
                metadata.spacing_m
            ));
        }
        if metadata.vert_max_m <= metadata.vert_min_m {
            return Err(anyhow!(
                "terrain {} has non-positive vertical range [{}, {}]",
                metadata.map_id,
                metadata.vert_min_m,
                metadata.vert_max_m
            ));
        }

        let bytes =
            fs::read(&r32_path).with_context(|| format!("reading {}", r32_path.display()))?;

        if !metadata.blake3.is_empty() {
            let actual = blake3::hash(&bytes).to_hex().to_string();
            if actual != metadata.blake3 {
                return Err(anyhow!(
                    "terrain {} hash mismatch: toml={} actual={}",
                    metadata.map_id,
                    metadata.blake3,
                    actual
                ));
            }
        }

        let samples = decode_r32(&bytes).ok_or_else(|| {
            anyhow!(
                "heightmap.r32 for {} has misaligned byte length {} (must be multiple of 4)",
                metadata.map_id,
                bytes.len()
            )
        })?;
        if samples.len() != metadata.sample_count() {
            return Err(anyhow!(
                "terrain {} sample count mismatch: file={} expected={} ({}x{})",
                metadata.map_id,
                samples.len(),
                metadata.sample_count(),
                metadata.width,
                metadata.height
            ));
        }

        // Optional feature layer. Present iff metadata has a non-empty
        // features_blake3; loader validates size + hash the same way
        // it does for the heightmap.
        let features = if metadata.features_blake3.is_empty() {
            None
        } else {
            let r8_path = dir.join("features.r8");
            let feature_bytes =
                fs::read(&r8_path).with_context(|| format!("reading {}", r8_path.display()))?;
            let actual = blake3::hash(&feature_bytes).to_hex().to_string();
            if actual != metadata.features_blake3 {
                return Err(anyhow!(
                    "terrain {} features hash mismatch: toml={} actual={}",
                    metadata.map_id,
                    metadata.features_blake3,
                    actual
                ));
            }
            if feature_bytes.len() != metadata.sample_count() {
                return Err(anyhow!(
                    "terrain {} features size mismatch: file={} expected={} ({}x{})",
                    metadata.map_id,
                    feature_bytes.len(),
                    metadata.sample_count(),
                    metadata.width,
                    metadata.height
                ));
            }
            Some(feature_bytes)
        };

        // Optional splatmap pair — present alongside features when
        // the map was baked with the splatmap pipeline. Loaded
        // best-effort: missing files just leave the field None and
        // the runtime falls back to the categorical features path.
        let load_splatmap = |name: &str| -> Option<Vec<u8>> {
            let path = dir.join(name);
            if !path.exists() {
                return None;
            }
            match fs::read(&path) {
                Ok(b) if b.len() == 4 * metadata.sample_count() => Some(b),
                Ok(_) => {
                    tracing::warn!(
                        map_id = %metadata.map_id,
                        file = name,
                        "splatmap size mismatch, ignoring"
                    );
                    None
                }
                Err(e) => {
                    tracing::warn!(
                        map_id = %metadata.map_id,
                        file = name,
                        error = %e,
                        "splatmap read failed, ignoring"
                    );
                    None
                }
            }
        };
        let splatmap_a = load_splatmap("splatmap_a.rgba8");
        let splatmap_b = load_splatmap("splatmap_b.rgba8");
        let road_density = load_splatmap("road_density.rgba8");

        // Iteration 5-13 Phase A1: optional designer-painted
        // nav-override mask. Present iff `nav_mask_blake3` is set in
        // metadata; loaded strictly (not best-effort) so a hash
        // mismatch surfaces as an error rather than silent fallback.
        // Format-version mismatch is also an error so a future v2
        // schema doesn't get misread as v1.
        let nav_mask = if metadata.nav_mask_blake3.is_empty() {
            None
        } else {
            if metadata.nav_mask_format_version != NAV_MASK_FORMAT_VERSION {
                return Err(anyhow!(
                    "terrain {} nav_mask format version mismatch: toml={} expected={}",
                    metadata.map_id,
                    metadata.nav_mask_format_version,
                    NAV_MASK_FORMAT_VERSION
                ));
            }
            let r8_path = dir.join("nav_mask.r8");
            let mask_bytes =
                fs::read(&r8_path).with_context(|| format!("reading {}", r8_path.display()))?;
            let actual = blake3::hash(&mask_bytes).to_hex().to_string();
            if actual != metadata.nav_mask_blake3 {
                return Err(anyhow!(
                    "terrain {} nav_mask_blake3 mismatch: toml={} actual={}",
                    metadata.map_id,
                    metadata.nav_mask_blake3,
                    actual
                ));
            }
            if mask_bytes.len() != metadata.sample_count() {
                return Err(anyhow!(
                    "terrain {} nav_mask size mismatch: file={} expected={} ({}x{})",
                    metadata.map_id,
                    mask_bytes.len(),
                    metadata.sample_count(),
                    metadata.width,
                    metadata.height
                ));
            }
            Some(mask_bytes)
        };

        tracing::info!(
            map_id = %metadata.map_id,
            width = metadata.width,
            height = metadata.height,
            spacing_m = metadata.spacing_m,
            has_features = features.is_some(),
            has_splatmap = splatmap_a.is_some() && splatmap_b.is_some(),
            has_road_density = road_density.is_some(),
            has_nav_mask = nav_mask.is_some(),
            "loaded heightmap"
        );

        Ok(Self {
            metadata,
            samples,
            features,
            splatmap_a,
            splatmap_b,
            road_density,
            nav_mask,
        })
    }

    /// Metadata sidecar.
    pub fn metadata(&self) -> &TerrainMetadata {
        &self.metadata
    }

    /// Grid width in samples.
    pub fn width(&self) -> u32 {
        self.metadata.width
    }

    /// Grid height in samples.
    pub fn height(&self) -> u32 {
        self.metadata.height
    }

    /// World-local extent in meters: `[width_m, height_m]`.
    pub fn extent_m(&self) -> [f32; 2] {
        self.metadata.extent_m()
    }

    /// World-local spacing between samples in meters.
    pub fn spacing_m(&self) -> f32 {
        self.metadata.spacing_m
    }

    /// Elevation in meters at world-local `(x, z)`. Clamps to edges
    /// when the query falls outside the grid.
    pub fn sample(&self, x: f32, z: f32) -> f32 {
        let u = x / self.metadata.spacing_m;
        let v = z / self.metadata.spacing_m;
        bilinear_f32(&self.samples, self.width(), self.height(), u, v)
    }

    /// Whether a paired `features.r8` layer was loaded alongside
    /// the heightmap.
    pub fn has_features(&self) -> bool {
        self.features.is_some()
    }

    /// Raw `features.r8` byte grid, row-major with NW origin. Each
    /// byte is a [`FeatureClass`] discriminant. Used by the Godot
    /// side to ship the grid to the GPU as a single-channel R8
    /// texture for the terrain shader.
    pub fn features_bytes(&self) -> Option<&[u8]> {
        self.features.as_deref()
    }

    /// Splatmap A bytes — RGBA8 row-major, `4 * W * H`. See
    /// `splatmap::SPLATMAP_A_CHANNELS` for channel layout.
    pub fn splatmap_a_bytes(&self) -> Option<&[u8]> {
        self.splatmap_a.as_deref()
    }

    /// Splatmap B bytes — RGBA8 row-major, `4 * W * H`. See
    /// `splatmap::SPLATMAP_B_CHANNELS` for channel layout.
    pub fn splatmap_b_bytes(&self) -> Option<&[u8]> {
        self.splatmap_b.as_deref()
    }

    /// Road density bytes — RGBA8 row-major, `4 * W * H`. See
    /// [`crate::road_density`] for channel layout (R=Paved, G=Unpaved,
    /// B=Trail, A=reserved).
    pub fn road_density_bytes(&self) -> Option<&[u8]> {
        self.road_density.as_deref()
    }

    /// Iteration 5-13 Phase A1: raw `nav_mask.r8` bytes for the
    /// Godot side to ship to the GPU / round-trip back to Terrain3D
    /// when re-seeding regions from canonical. `None` when no
    /// override layer was authored. Length = `width * height`.
    pub fn nav_mask_bytes(&self) -> Option<&[u8]> {
        self.nav_mask.as_deref()
    }

    /// Iteration 5-13 Phase A1: per-cell painter override at grid
    /// coordinate `(col, row)`. Returns [`NavOverride::Default`]
    /// when no mask is loaded, when `(col, row)` is out of bounds,
    /// or when the byte at that offset decodes to an unknown
    /// value. Cheap O(1) lookup intended for `GridNavQuery`'s
    /// per-cell build loop — no bilinear interpolation (overrides
    /// are categorical, like `FeatureClass`).
    pub fn nav_override_at(&self, col: usize, row: usize) -> NavOverride {
        let Some(mask) = self.nav_mask.as_ref() else {
            return NavOverride::Default;
        };
        let w = self.metadata.width as usize;
        let h = self.metadata.height as usize;
        if col >= w || row >= h {
            return NavOverride::Default;
        }
        nav_mask::decode(mask[row * w + col])
    }

    /// Look up the feature class at world-local `(x, z)`. Nearest-
    /// neighbor (classes are categorical; interpolation would be
    /// meaningless). Returns [`FeatureClass::Unknown`] when no
    /// feature layer is loaded or when the query is out of bounds.
    pub fn sample_feature(&self, x: f32, z: f32) -> FeatureClass {
        let Some(features) = self.features.as_ref() else {
            return FeatureClass::Unknown;
        };
        let w = self.metadata.width as i32;
        let h = self.metadata.height as i32;
        let col = (x / self.metadata.spacing_m).round() as i32;
        let row = (z / self.metadata.spacing_m).round() as i32;
        let col = col.clamp(0, w - 1) as usize;
        let row = row.clamp(0, h - 1) as usize;
        let idx = row * (w as usize) + col;
        FeatureClass::from_u8(features[idx])
    }

    /// Unit-length surface normal at world-local `(x, z)`, pointing up.
    /// Computed by central differences with step = one grid cell.
    pub fn sample_normal(&self, x: f32, z: f32) -> [f32; 3] {
        let h = self.metadata.spacing_m;
        let dy_dx = (self.sample(x + h, z) - self.sample(x - h, z)) / (2.0 * h);
        let dy_dz = (self.sample(x, z + h) - self.sample(x, z - h)) / (2.0 * h);
        let nx = -dy_dx;
        let ny = 1.0;
        let nz = -dy_dz;
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        [nx / len, ny / len, nz / len]
    }

    // ---------------------------------------------------------------
    // Centered-world API. This is the canonical runtime convention:
    // world `(0, 0)` = center of the playable map, X east, Z south.
    // Matches what `terrain3d_loader.gd` imports regions at, what
    // `ground_cover.gd` queries with, and what game scenes assume
    // (player spawns at world `(0, Y, 0)` = terrain center). New code
    // should prefer these over the NW-origin variants above unless
    // it specifically needs to walk the grid in row-major order
    // (where NW-origin coords are natural).
    //
    // See `docs/book/src/planning/static-foliage-plan.md` →
    // "Cross-layer conventions" for the reasoning.
    // ---------------------------------------------------------------

    /// Half the world extent on each axis: `[width_m / 2, height_m / 2]`.
    /// Useful as the offset between centered and NW-origin coords.
    pub fn extent_m_half(&self) -> [f32; 2] {
        let [w, h] = self.metadata.extent_m();
        [w * 0.5, h * 0.5]
    }

    /// Region edge length in world meters this map aligns to. Per
    /// the cross-layer alignment contract, `(W - 1) * spacing_m` is
    /// guaranteed to be a multiple of this. See [`TerrainMetadata`].
    pub fn region_size_m(&self) -> f32 {
        self.metadata.region_size_m
    }

    /// Playable area extent in meters: `[playable_x, playable_z]`.
    /// Falls back to the full canonical extent when the bake didn't
    /// record a playable area (legacy maps). Always ≤ the canonical
    /// extent on each axis.
    pub fn playable_extent_m(&self) -> [f32; 2] {
        let canonical = self.metadata.extent_m();
        let px = if self.metadata.playable_extent_x_m > 0.0 {
            self.metadata.playable_extent_x_m.min(canonical[0])
        } else {
            canonical[0]
        };
        let pz = if self.metadata.playable_extent_z_m > 0.0 {
            self.metadata.playable_extent_z_m.min(canonical[1])
        } else {
            canonical[1]
        };
        [px, pz]
    }

    /// Half the playable extent on each axis. Centered foliage /
    /// sim placements should stay within `[-half, +half]` on both
    /// axes to avoid landing in the region-alignment padding.
    pub fn playable_extent_m_half(&self) -> [f32; 2] {
        let [x, z] = self.playable_extent_m();
        [x * 0.5, z * 0.5]
    }

    /// True if centered world `(x, z)` is within the playable area.
    pub fn is_playable(&self, x: f32, z: f32) -> bool {
        let [hx, hz] = self.playable_extent_m_half();
        x.abs() <= hx && z.abs() <= hz
    }

    /// Elevation at centered world `(x, z)`. Equivalent to
    /// `sample(x + half_w, z + half_h)`.
    pub fn sample_centered(&self, x: f32, z: f32) -> f32 {
        let [hw, hh] = self.extent_m_half();
        self.sample(x + hw, z + hh)
    }

    /// Surface normal at centered world `(x, z)`.
    pub fn sample_normal_centered(&self, x: f32, z: f32) -> [f32; 3] {
        let [hw, hh] = self.extent_m_half();
        self.sample_normal(x + hw, z + hh)
    }

    /// Feature class at centered world `(x, z)`.
    pub fn sample_feature_centered(&self, x: f32, z: f32) -> FeatureClass {
        let [hw, hh] = self.extent_m_half();
        self.sample_feature(x + hw, z + hh)
    }

    /// Convert centered world `(x, z)` to a `(col, row)` pixel index
    /// into the `width × height` grid, or `None` if outside.
    /// Splatmap / road-density consumers use this for byte indexing.
    pub fn world_to_pixel(&self, x: f32, z: f32) -> Option<(usize, usize)> {
        let [hw, hh] = self.extent_m_half();
        let u = (x + hw) / self.metadata.spacing_m;
        let v = (z + hh) / self.metadata.spacing_m;
        let w = self.metadata.width as i32;
        let h = self.metadata.height as i32;
        let col = u as i32;
        let row = v as i32;
        if col < 0 || row < 0 || col >= w || row >= h {
            return None;
        }
        Some((col as usize, row as usize))
    }
}
