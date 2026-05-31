//! simn-terrain — Canonical heightmap loader + sampler for SIMN.
//!
//! Engine-agnostic. The server is authoritative for terrain elevation;
//! this crate is the single source of truth the server consults. The
//! Godot side (in `simn-godot`) builds its render mesh + collider from
//! the same canonical file, and a parity test (arriving in a later
//! phase) ensures both sides agree to < 1 mm at any queried (x, z).
//!
//! # Canonical format
//!
//! Each map lives in its own directory under `godot/assets/terrain/`:
//!
//! ```text
//! godot/assets/terrain/<map_id>/
//! ├── heightmap.r32    // raw 32-bit LE float, row-major, N-up, W*H samples (literal meters)
//! └── terrain.toml     // metadata (grid dims, spacing, vertical clamps, UTM origin, hash)
//! ```
//!
//! The `.r32` is a headerless grid of f32 values, each storing the
//! elevation in literal meters above sea level. Format version 2;
//! the legacy v1 `.r16` u16-quantized format was retired 2026-05-03
//! (see [`sampler::legacy_v1`] for the migration helpers).
//!
//! # Sampling
//!
//! [`Heightmap::sample`] performs bilinear interpolation in world-local
//! space, where `(0, 0)` is the north-west corner of the map and the
//! positive-Z axis runs south (matching Godot's left-handed default).
//! Out-of-bounds queries clamp to the nearest edge sample.
//!
//! When the Godot-side parity test lands, the sampling algorithm here
//! may need to switch from bilinear to triangle interpolation to match
//! `HeightMapShape3D`'s internal triangulation exactly. Until then,
//! bilinear is correct and simpler.

pub mod bake;
pub mod convert;
pub mod features;
pub mod heightmap;
pub mod metadata;
pub mod nav_mask;
pub mod osm;
pub mod road_density;
pub mod sampler;
pub mod spec;
pub mod splatmap;
pub mod terrarium;
pub mod usgs3dep;

pub use bake::{bake_map, load_spec, write_scene_once, BakeReport};
pub use convert::{sync_canonical_to_png, sync_png_to_canonical, SyncReport};
pub use features::FeatureClass;
pub use heightmap::Heightmap;
pub use metadata::TerrainMetadata;
pub use nav_mask::{decode as decode_nav_mask, NavOverride, NAV_MASK_FORMAT_VERSION};
pub use spec::{BakeBounds, BakeSource, BakeSpec, FeaturesSource, OsmOverlay};
