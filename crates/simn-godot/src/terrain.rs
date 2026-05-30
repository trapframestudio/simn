//! [`TerrainNode`] — loads a canonical heightmap and materializes it
//! as Godot visual mesh + physics collision.
//!
//! Lives as a `StaticBody3D` subclass so collision attaches directly.
//! On `ready` (or a later `load_map` call) it:
//!
//! 1. Resolves `res://assets/terrain/<map_id>/` to an OS path via
//!    `ProjectSettings::globalize_path`.
//! 2. Calls `simn_terrain::Heightmap::load` — same sampler the server
//!    consults.
//! 3. Builds a visual `MeshInstance3D` (ArrayMesh) and a
//!    `CollisionShape3D` (HeightMapShape3D), both derived from the
//!    same f32 grid (canonical `heightmap.r32`, format_version 2).
//!
//! The terrain is centered on the node's position, so a 5 km × 5 km
//! map placed at `(0, 0, 0)` extends ±2.5 km on X and Z.

use godot::classes::base_material_3d::{CullMode, Flags};
use godot::classes::image::Format as ImageFormat;
use godot::classes::mesh::{ArrayType, PrimitiveType};
use godot::classes::{
    ArrayMesh, CollisionShape3D, HeightMapShape3D, IStaticBody3D, Image, ImageTexture, Material,
    MeshInstance3D, ProjectSettings, ShaderMaterial, StandardMaterial3D, StaticBody3D, Texture2D,
    Texture2DArray,
};
use godot::prelude::*;
use simn_terrain::{FeatureClass, Heightmap};
use std::path::PathBuf;

/// Fallback path for the terrain ShaderMaterial when no material
/// is assigned on the `TerrainNode.terrain_material` export. This
/// is the canonical Godot-authored material — edit it in the
/// Godot editor to tune any shader uniform without a Rust rebuild.
const TERRAIN_MATERIAL_PATH: &str = "res://resources/materials/terrain.tres";

/// One diffuse texture per [`FeatureClass`] discriminant (0-23).
/// Gaps in the enum (11-19) reuse the Unknown fallback so the
/// array has a contiguous layer range that indexes directly by
/// the byte value in `features.r8`. Sources mix 2K, 4K, and 8K —
/// `build_diffuse_array` resizes each layer to a common edge
/// length before stacking (`Texture2DArray::create_from_images`
/// rejects mixed sizes).
///
/// **PNW palette (PR #109 + follow-ups).** Forest, Grassland, Bare,
/// BuiltUp, Cliff, UnpavedRoad, and Trail use the same Megascans /
/// AmbientCG picks the HTerrain plugin's MULTISPLAT16 texture set
/// settled on (see `godot/scripts/terrain/hterrain_texture_set_builder.gd`
/// for the canonical list and per-class rationale). Both renderers
/// thus see the same palette; switching between them is a shader
/// change, not a content change. Variant arrays below intentionally
/// pull from *other* texture families to break up tiling.
pub(crate) const TERRAIN_CLASS_DIFFUSE: [&str; 24] = [
    // 0  Unknown — neutral ground/rock fallback
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 1  Water — placeholder; real water shader later
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 2  Forest — Megascans `forest_floor` (sfjmafua). Doug-fir
    //     needles + debris, the PNW-region pick from PR #109's
    //     HTerrain swap.
    "res://assets/textures/terrain/megascans/forest_floor/T_sfjmafua_8K_B.png",
    // 3  Shrubland — rocks-scrub aerial (no Megascans match yet)
    "res://assets/textures/terrain/aerial_rocks_02_4k.gltf/textures/aerial_rocks_02_diff_4k.jpg",
    // 4  Grassland — Megascans `wild_grass` (sfknaeoa). Open-meadow
    //     palette to match HTerrain's tuned set.
    "res://assets/textures/terrain/megascans/wild_grass/T_sfknaeoa_8K_B.png",
    // 5  Cropland — cultivated brown soil (PNW orchard floor)
    "res://assets/textures/terrain/brown_mud_02_4k.gltf/textures/brown_mud_02_diff_4k.jpg",
    // 6  BuiltUp — AmbientCG Concrete026 (PR #109 HTerrain pick;
    //     replaces brushed_concrete which read too smooth/clean).
    "res://assets/textures/terrain/Concrete026_4K-PNG/Concrete026_4K-PNG_Color.png",
    // 7  Bare — AmbientCG Ground073 (volcanic scree / talus, PNW pick)
    "res://assets/textures/terrain/Ground073_4K-PNG/Ground073_4K-PNG_Color.png",
    // 8  Snow — real PolyHaven snow_02 (replaces placeholder)
    "res://assets/textures/terrain/snow_02_4k.gltf/textures/snow_02_diff_4k.jpg",
    // 9  Wetland — mud with leaves (marsh)
    "res://assets/textures/terrain/brown_mud_leaves_01_4k.gltf/textures/brown_mud_leaves_01_diff_4k.jpg",
    // 10 Moss — Megascans `nordic_moss` (se4rwei). Dense PNW
    //     old-growth understory moss; matches HTerrain L14.
    "res://assets/textures/terrain/megascans/nordic_moss/T_se4rwei_8K_B.png",
    // 11-19 unused enum slots — reuse Unknown fallback so layer
    //       indexing by raw byte doesn't need a remap table
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 20 Cliff — Megascans `mine_rock_wall` (uebmddyn). Brighter
    //     exposed rock for cliff faces; matches HTerrain L6.
    "res://assets/textures/terrain/megascans/mine_rock_wall/T_uebmddyn_8K_B.png",
    // 21 PavedRoad — AmbientCG Asphalt010 (HTerrain L8 already)
    "res://assets/textures/terrain/Asphalt010_4K-PNG/Asphalt010_4K-PNG_Color.png",
    // 22 UnpavedRoad — Megascans `military_trenches_dirt_fine` (yd0keak).
    //     Fine-grained dirt with embedded stones; matches HTerrain L9.
    "res://assets/textures/terrain/megascans/military_trenches_dirt_fine/T_yd0keak_2k_B.png",
    // 23 Trail — Megascans `mossy_rocky_ground` (vcrkeeb). Trampled
    //     moss + rock for hiker trails; matches HTerrain L10.
    "res://assets/textures/terrain/megascans/mossy_rocky_ground/T_vcrkeeb_8K_B.png",
];

/// Variant diffuse per FeatureClass. Pools intentionally pull from
/// **different texture families** (rock + grass + dirt + leaf litter
/// rather than four near-identical mosses), because at the variant-
/// blend strengths we use, four within-family textures all read as
/// "one tiled photo." Cross-family variants give visible material
/// breakup at the per-fragment level — patches of dirt inside a
/// grass field, rocky strips inside a cropland — without any
/// per-vertex hand-painting. Same 24-layer layout; same fallback
/// strategy for enum gaps.
const TERRAIN_CLASS_DIFFUSE_VARIANT: [&str; 24] = [
    // 0  Unknown — reuse primary
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 1  Water — reuse primary (water gets procedural shader later)
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 2  Forest — Megascans `mossy_grass` (vd3mebls). HTerrain L13
    //     leafy/weed variant — darker mossy tones that blend into
    //     forest_floor + nordic_moss neighbors.
    "res://assets/textures/terrain/megascans/mossy_grass/T_vd3mebls_8K_B.png",
    // 3  Shrubland — rocks with moss
    "res://assets/textures/terrain/aerial_rocks_01_4k.gltf/textures/aerial_rocks_01_diff_4k.jpg",
    // 4  Grassland — Megascans `rocky_steppe` (ulgmbhwn). HTerrain L12
    //     grass-rock transition — breaks up the wild_grass primary
    //     with rocky outcrops where pasture meets exposed bedrock.
    "res://assets/textures/terrain/megascans/rocky_steppe/T_ulgmbhwn_8K_B.png",
    // 5  Cropland — different mud variation
    "res://assets/textures/terrain/brown_mud_03_4k.gltf/textures/brown_mud_03_diff_4k.jpg",
    // 6  BuiltUp — anti-slip concrete (distinct surface)
    "res://assets/textures/terrain/anti_slip_concrete_4k.gltf/textures/anti_slip_concrete_diff_4k.jpg",
    // 7  Bare — different aerial dirt (NEW PolyHaven asset)
    "res://assets/textures/terrain/dirt_aerial_02_4k.gltf/textures/dirt_aerial_02_diff_4k.jpg",
    // 8  Snow — snow_05 (variant of snow_02 primary)
    "res://assets/textures/terrain/snow_05_4k.gltf/textures/snow_05_diff_4k.jpg",
    // 9  Wetland — moss patches at water edge (cross-family)
    "res://assets/textures/terrain/Moss004_4K-PNG/Moss004_4K-PNG_Color.png",
    // 10 Moss — Moss004 variant
    "res://assets/textures/terrain/Moss004_4K-PNG/Moss004_4K-PNG_Color.png",
    // 11-19 unused enum slots — reuse Unknown
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 20 Cliff — `mossy_rock` (HTerrain L11 cliff variant — deep-moss
    //     boulder patches on basalt faces).
    "res://assets/textures/terrain/mossy_rock_4k.gltf/textures/mossy_rock_diff_4k.jpg",
    // 21 PavedRoad — different asphalt (wear variation)
    "res://assets/textures/terrain/Asphalt020L_4K-PNG/Asphalt020L_4K-PNG_Color.png",
    // 22 UnpavedRoad — brick-gravel variant
    "res://assets/textures/terrain/brick_gravel_4k.gltf/textures/brick_gravel_diff_4k.jpg",
    // 23 Trail — brown_mud_02 variant
    "res://assets/textures/terrain/brown_mud_02_4k.gltf/textures/brown_mud_02_diff_4k.jpg",
];

/// Second-variant diffuse per FeatureClass — a THIRD family
/// member, blended at a different noise frequency than
/// TERRAIN_CLASS_DIFFUSE_VARIANT so each class shows *two*
/// independent variation patterns layered together. The close-up
/// "all-the-same-texture repeating" look only goes away once the
/// per-fragment texture choice varies across several different
/// images, not just two.
const TERRAIN_CLASS_DIFFUSE_VARIANT2: [&str; 24] = [
    // 0  Unknown — another ground-rock
    "res://assets/textures/terrain/aerial_rocks_01_4k.gltf/textures/aerial_rocks_01_diff_4k.jpg",
    // 1  Water — placeholder (same as primary)
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 2  Forest — mossy-rock for damp canopy patches
    "res://assets/textures/terrain/mossy_rock_4k.gltf/textures/mossy_rock_diff_4k.jpg",
    // 3  Shrubland — third rocks variant
    "res://assets/textures/terrain/aerial_rocks_04_4k.gltf/textures/aerial_rocks_04_diff_4k.jpg",
    // 4  Grassland — DIRT patches (cross-family — bare patches in grass)
    "res://assets/textures/terrain/brown_mud_4k.gltf/textures/brown_mud_diff_4k.jpg",
    // 5  Cropland — rocks in field (cross-family)
    "res://assets/textures/terrain/brown_mud_rocks_01_4k.gltf/textures/brown_mud_rocks_01_diff_4k.jpg",
    // 6  BuiltUp — clean pebbles / gravel patches (cross-family)
    "res://assets/textures/terrain/clean_pebbles_4k.gltf/textures/clean_pebbles_diff_4k.jpg",
    // 7  Bare — rocky outcrop
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 8  Snow — snow_02 (paired with primary for natural fresh-vs-trampled blend)
    "res://assets/textures/terrain/snow_02_4k.gltf/textures/snow_02_diff_4k.jpg",
    // 9  Wetland — brown_mud_03 (third mud variant)
    "res://assets/textures/terrain/brown_mud_03_4k.gltf/textures/brown_mud_03_diff_4k.jpg",
    // 10 Moss — moss-on-rock (cross-family — the rock substrate showing through)
    "res://assets/textures/terrain/mossy_rock_4k.gltf/textures/mossy_rock_diff_4k.jpg",
    // 11-19 unused enum slots
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 20 Cliff — AmbientCG `Rock028` (HTerrain L15 light cliff variant —
    //     paler grey rock that breaks up the mine_rock_wall + mossy_rock
    //     dominant pair).
    "res://assets/textures/terrain/Rock028_4K-PNG/Rock028_4K-PNG_Color.png",
    // 21 PavedRoad — Asphalt015 (different wear pattern)
    "res://assets/textures/terrain/Asphalt015_4K-PNG/Asphalt015_4K-PNG_Color.png",
    // 22 UnpavedRoad — clean pebbles (third unpaved variant)
    "res://assets/textures/terrain/clean_pebbles_4k.gltf/textures/clean_pebbles_diff_4k.jpg",
    // 23 Trail — brown_mud_03 (third trail variant)
    "res://assets/textures/terrain/brown_mud_03_4k.gltf/textures/brown_mud_03_diff_4k.jpg",
];

/// Third-variant diffuse per FeatureClass — a FOURTH family member
/// stacked on top of primary + variant + variant2. Blended at a
/// further-decoupled noise frequency so the resulting ground has
/// four textures per class visible in different patches, with the
/// mixing happening smoothly via noise at three independent scales.
/// This is the "pull in as many variants as we have" layer.
const TERRAIN_CLASS_DIFFUSE_VARIANT3: [&str; 24] = [
    // 0  Unknown — aerial_rocks_02
    "res://assets/textures/terrain/aerial_rocks_02_4k.gltf/textures/aerial_rocks_02_diff_4k.jpg",
    // 1  Water — placeholder
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 2  Forest — leafy forest floor (NEW PolyHaven asset)
    "res://assets/textures/terrain/forrest_ground_03_4k.gltf/textures/forrest_ground_03_diff_4k.jpg",
    // 3  Shrubland — sparse grass in scrub (cross-family)
    "res://assets/textures/terrain/aerial_grass_rock_4k.gltf/textures/aerial_grass_rock_diff_4k.jpg",
    // 4  Grassland — grass-with-rocks (cross-family — rock outcrops in grass)
    "res://assets/textures/terrain/aerial_grass_rock_4k.gltf/textures/aerial_grass_rock_diff_4k.jpg",
    // 5  Cropland — grass strips between rows (cross-family)
    "res://assets/textures/terrain/aerial_grass_rock_4k.gltf/textures/aerial_grass_rock_diff_4k.jpg",
    // 6  BuiltUp — dirt yards / vacant lots (cross-family)
    "res://assets/textures/terrain/brown_mud_4k.gltf/textures/brown_mud_diff_4k.jpg",
    // 7  Bare — sparse grass patches (cross-family)
    "res://assets/textures/terrain/aerial_grass_rock_4k.gltf/textures/aerial_grass_rock_diff_4k.jpg",
    // 8  Snow — rocky outcrops poking through snow (high-altitude realism)
    "res://assets/textures/terrain/aerial_rocks_02_4k.gltf/textures/aerial_rocks_02_diff_4k.jpg",
    // 9  Wetland — marsh grass clumps (cross-family)
    "res://assets/textures/terrain/aerial_grass_rock_4k.gltf/textures/aerial_grass_rock_diff_4k.jpg",
    // 10 Moss — mossy grass-rock (cross-family — moss with grass blades)
    "res://assets/textures/terrain/aerial_grass_rock_4k.gltf/textures/aerial_grass_rock_diff_4k.jpg",
    // 11-19 unused enum slots
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_diff_4k.jpg",
    // 20 Cliff — aerial_rocks_01 (fourth cliff variant)
    "res://assets/textures/terrain/aerial_rocks_01_4k.gltf/textures/aerial_rocks_01_diff_4k.jpg",
    // 21 PavedRoad — Asphalt024A (fourth asphalt wear)
    "res://assets/textures/terrain/Asphalt024A_4K-PNG/Asphalt024A_4K-PNG_Color.png",
    // 22 UnpavedRoad — Gravel022 (fourth gravel)
    "res://assets/textures/terrain/Gravel022_4K-PNG/Gravel022_4K-PNG_Color.png",
    // 23 Trail — brown_mud_leaves (leaf-strewn trail)
    "res://assets/textures/terrain/brown_mud_leaves_01_4k.gltf/textures/brown_mud_leaves_01_diff_4k.jpg",
];

/// Tangent-space normal map per FeatureClass — same 24-layer layout
/// as the diffuse arrays. PolyHaven `_nor_gl_4k.jpg` and AmbientCG
/// `_NormalGL.png` are both OpenGL convention (+Y up), which matches
/// what the shader expects (no Y flip needed).
///
/// Only one normal map per class — variants share the primary's
/// surface micro-detail, since the eye picks up *color* variation far
/// more than normal variation, and quadrupling the per-class normal
/// sample cost for marginal payoff isn't worth the GPU budget.
///
/// `compress/normal_map=1` in each `.import` makes Godot store these
/// as RG-only (BC5/BPTC). The shader reconstructs Z from the unit
/// length constraint. See `sample_normal_triplanar()` in the shader.
const TERRAIN_CLASS_NORMAL: [&str; 24] = [
    // 0  Unknown
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    // 1  Water — placeholder (flat normal, no real perturbation)
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    // 2  Forest — Megascans `forest_floor` GL normal
    "res://assets/textures/terrain/megascans/forest_floor/T_sfjmafua_8K_N.png",
    // 3  Shrubland
    "res://assets/textures/terrain/aerial_rocks_02_4k.gltf/textures/aerial_rocks_02_nor_gl_4k.jpg",
    // 4  Grassland — Megascans `wild_grass`
    "res://assets/textures/terrain/megascans/wild_grass/T_sfknaeoa_8K_N.png",
    // 5  Cropland
    "res://assets/textures/terrain/brown_mud_02_4k.gltf/textures/brown_mud_02_nor_gl_4k.jpg",
    // 6  BuiltUp — AmbientCG Concrete026
    "res://assets/textures/terrain/Concrete026_4K-PNG/Concrete026_4K-PNG_NormalGL.png",
    // 7  Bare — AmbientCG Ground073
    "res://assets/textures/terrain/Ground073_4K-PNG/Ground073_4K-PNG_NormalGL.png",
    // 8  Snow — snow_02 normal (real snow)
    "res://assets/textures/terrain/snow_02_4k.gltf/textures/snow_02_nor_gl_4k.jpg",
    // 9  Wetland
    "res://assets/textures/terrain/brown_mud_leaves_01_4k.gltf/textures/brown_mud_leaves_01_nor_gl_4k.jpg",
    // 10 Moss — Megascans `nordic_moss`
    "res://assets/textures/terrain/megascans/nordic_moss/T_se4rwei_8K_N.png",
    // 11-19 unused enum slots
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_nor_gl_4k.jpg",
    // 20 Cliff — Megascans `mine_rock_wall`
    "res://assets/textures/terrain/megascans/mine_rock_wall/T_uebmddyn_8K_N.png",
    // 21 PavedRoad
    "res://assets/textures/terrain/Asphalt010_4K-PNG/Asphalt010_4K-PNG_NormalGL.png",
    // 22 UnpavedRoad — Megascans `military_trenches_dirt_fine`
    "res://assets/textures/terrain/megascans/military_trenches_dirt_fine/T_yd0keak_2k_N.png",
    // 23 Trail — Megascans `mossy_rocky_ground`
    "res://assets/textures/terrain/megascans/mossy_rocky_ground/T_vcrkeeb_8K_N.png",
];

/// Single-channel roughness per FeatureClass. Layer index = class
/// discriminant. Where a bundle ships only an `_arm_4k.jpg` (AO/Rough/
/// Metal packed) but no dedicated `_rough_4k.jpg`, substitute a
/// similar-material rough from another bundle — the visual difference
/// is subtle compared to "no roughness map at all," and it keeps the
/// per-layer texture format uniform.
///
/// PolyHaven roughness textures use the standard convention: dark =
/// glossy, bright = rough. AmbientCG matches. The shader samples `.r`
/// and writes directly to ROUGHNESS so the surfaces respond
/// physically to specular highlights (wet asphalt vs dry, mossy rock
/// vs polished stone).
const TERRAIN_CLASS_ROUGHNESS: [&str; 24] = [
    // 0  Unknown
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    // 1  Water — placeholder; real water shader sets its own roughness
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    // 2  Forest
    "res://assets/textures/terrain/aerial_grass_rock_4k.gltf/textures/aerial_grass_rock_rough_4k.jpg",
    // 3  Shrubland
    "res://assets/textures/terrain/aerial_rocks_02_4k.gltf/textures/aerial_rocks_02_rough_4k.jpg",
    // 4  Grassland
    "res://assets/textures/terrain/Moss001_4K-PNG/Moss001_4K-PNG_Roughness.png",
    // 5  Cropland
    "res://assets/textures/terrain/brown_mud_02_4k.gltf/textures/brown_mud_02_rough_4k.jpg",
    // 6  BuiltUp — AmbientCG Concrete026 Roughness (matches diffuse)
    "res://assets/textures/terrain/Concrete026_4K-PNG/Concrete026_4K-PNG_Roughness.png",
    // 7  Bare — AmbientCG Ground073 Roughness (matches diffuse)
    "res://assets/textures/terrain/Ground073_4K-PNG/Ground073_4K-PNG_Roughness.png",
    // 8  Snow — snow_02 roughness (real snow — fresh snow has subtle specular)
    "res://assets/textures/terrain/snow_02_4k.gltf/textures/snow_02_rough_4k.jpg",
    // 9  Wetland — substitute (no _rough); use brown_mud (similar mud)
    "res://assets/textures/terrain/brown_mud_4k.gltf/textures/brown_mud_rough_4k.jpg",
    // 10 Moss
    "res://assets/textures/terrain/Moss002_4K-PNG/Moss002_4K-PNG_Roughness.png",
    // 11-19 unused enum slots
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    "res://assets/textures/terrain/aerial_ground_rock_4k.gltf/textures/aerial_ground_rock_rough_4k.jpg",
    // 20 Cliff — substitute (no _rough); use coast_land_rocks (similar rocky)
    "res://assets/textures/terrain/coast_land_rocks_01_4k.gltf/textures/coast_land_rocks_01_rough_4k.jpg",
    // 21 PavedRoad
    "res://assets/textures/terrain/Asphalt010_4K-PNG/Asphalt010_4K-PNG_Roughness.png",
    // 22 UnpavedRoad — substitute (no _rough); use brown_mud_rocks (similar gravelly)
    "res://assets/textures/terrain/brown_mud_rocks_01_4k.gltf/textures/brown_mud_rocks_01_rough_4k.jpg",
    // 23 Trail
    "res://assets/textures/terrain/brown_mud_4k.gltf/textures/brown_mud_rough_4k.jpg",
];

#[derive(GodotClass)]
#[class(tool, init, base=StaticBody3D)]
pub struct TerrainNode {
    /// Logical map id, matches a subdirectory under `res://assets/terrain/`.
    #[export]
    map_id: GString,

    /// If true, `load_map` runs automatically in `ready`.
    #[export]
    auto_load: bool,

    /// Shader/material authored in the Godot editor. If unset, falls
    /// back to `res://resources/materials/terrain.tres`. **All shader
    /// tuning (tile scale, variant strength, debug mode, warp
    /// amplitude, etc.) lives on this material's shader parameters.**
    /// Edit the .tres in the Inspector; no Rust rebuild needed.
    ///
    /// Rust only assigns the runtime-variable uniforms at load time:
    /// `features_texture` (this map's classification grid) and
    /// `terrain_extent_m` (this map's bounds). Plus the 4 texture
    /// arrays (primary + 3 variants), which are a coherent pack
    /// shipped with the engine.
    /// Pre-populated to the canonical `terrain.tres` so the
    /// designer can click into the material's shader parameters
    /// straight from the TerrainNode inspector — no per-scene Quick
    /// Load. Godot logs an "Instantiated ShaderMaterial used as
    /// default value" warning on every node construction because
    /// resource defaults are shared across instances; here the
    /// sharing is intentional (single live `terrain.tres` →
    /// Inspector slider edits flow to the running mesh in real
    /// time), so we accept the warning. `load_map` still has a
    /// runtime fallback to `try_load(TERRAIN_MATERIAL_PATH)` for
    /// the case where this gets cleared in a scene.
    #[export]
    #[init(val = try_load::<ShaderMaterial>(TERRAIN_MATERIAL_PATH).ok())]
    terrain_material: Option<Gd<ShaderMaterial>>,

    /// Loaded heightmap retained for scene-side sampling queries
    /// (`sample_height`). `None` before `load_map` succeeds.
    heightmap: Option<Heightmap>,

    base: Base<StaticBody3D>,
}

#[godot_api]
impl IStaticBody3D for TerrainNode {
    fn ready(&mut self) {
        if self.auto_load && !self.map_id.is_empty() {
            let id = self.map_id.clone();
            self.load_map(id);
        }
    }
}

#[godot_api]
impl TerrainNode {
    /// Load the named map, replacing any previously-built terrain.
    #[func]
    pub fn load_map(&mut self, map_id: GString) {
        let map_id_str = map_id.to_string();
        let res_path = format!("res://assets/terrain/{map_id_str}");
        let globalized = ProjectSettings::singleton().globalize_path(&GString::from(&res_path));
        let dir = PathBuf::from(globalized.to_string());

        let heightmap = match Heightmap::load(&dir) {
            Ok(h) => h,
            Err(e) => {
                let msg = format!("{map_id_str}: {e}");
                godot_error!("TerrainNode: {msg}");
                self.base_mut()
                    .emit_signal("terrain_error", &[GString::from(&msg).to_variant()]);
                return;
            }
        };

        self.remove_generated_children();
        // Resolve the shader material: prefer the `terrain_material`
        // export (Godot-authored asset the designer tuned), fall back
        // to the canonical res:// path, fall back further to None
        // which makes `build_terrain_material` pick the vertex-color
        // StandardMaterial3D.
        let template = self
            .terrain_material
            .clone()
            .or_else(|| try_load::<ShaderMaterial>(TERRAIN_MATERIAL_PATH).ok());
        let mesh_inst = build_mesh_instance(&heightmap, template);
        let collision = build_collision_shape(&heightmap);

        let vert_count = heightmap.width() * heightmap.height();
        let tri_count = (heightmap.width() - 1) * (heightmap.height() - 1) * 2;
        let aabb = mesh_inst.get_aabb();
        godot_print!(
            "TerrainNode: built mesh {} verts, {} tris, AABB pos={} size={}",
            vert_count,
            tri_count,
            aabb.position,
            aabb.size
        );

        self.base_mut().add_child(&mesh_inst);
        self.base_mut().add_child(&collision);

        godot_print!(
            "TerrainNode: loaded {} ({}x{} @ {}m)",
            map_id_str,
            heightmap.width(),
            heightmap.height(),
            heightmap.metadata().spacing_m
        );

        // Retain the heightmap for scene-side `sample_height` queries.
        self.heightmap = Some(heightmap);

        self.base_mut()
            .emit_signal("terrain_loaded", &[map_id.to_variant()]);
    }

    /// Ground elevation at a scene-local (x, z), in meters. `x`/`z`
    /// are in the terrain node's local coordinate system (centered
    /// at the node origin, matching the visual mesh + collision
    /// shape). Returns 0.0 before the heightmap has loaded.
    ///
    /// GDScript callers: `terrain.sample_height(x, z)` is the right
    /// way to place props, transition cubes, or spawn markers on the
    /// visible surface — the sim's Position.y for bases + NPCs uses
    /// the same sampling so placements are consistent.
    #[func]
    pub fn sample_height(&self, x: f32, z: f32) -> f32 {
        let Some(hm) = self.heightmap.as_ref() else {
            return 0.0;
        };
        let [w, h] = hm.extent_m();
        hm.sample(x + w * 0.5, z + h * 0.5)
    }

    /// Iteration 5-14 follow-up. Canonical grid dimensions for
    /// GDScript callers that need to walk the heightmap (typically
    /// to push live Terrain3D samples through
    /// `SimHost::attach_region_terrain_from_packed_heights`).
    /// Returns `(0, 0)` if the heightmap hasn't loaded yet.
    #[func]
    pub fn grid_dims(&self) -> Vector2i {
        match self.heightmap.as_ref() {
            Some(hm) => Vector2i {
                x: hm.width() as i32,
                y: hm.height() as i32,
            },
            None => Vector2i { x: 0, y: 0 },
        }
    }

    /// Iteration 5-14 follow-up. Canonical world-space spacing
    /// between heightmap samples in meters. Pairs with `grid_dims`
    /// for the live-Terrain3D push path.
    #[func]
    pub fn spacing_m(&self) -> f32 {
        self.heightmap
            .as_ref()
            .map(|hm| hm.spacing_m())
            .unwrap_or(0.0)
    }

    /// Emitted after a successful load. `map_id` is the loaded id.
    #[signal]
    fn terrain_loaded(map_id: GString);

    /// Emitted when a load fails. `msg` carries the underlying error text.
    #[signal]
    fn terrain_error(msg: GString);

    /// Drop any previously-generated children (mesh, collision). Does
    /// not touch children added in the scene file — we only tear down
    /// nodes we named with the internal prefix below.
    fn remove_generated_children(&mut self) {
        const PREFIX: &str = "__Terrain_";
        let mut to_remove = Vec::new();
        {
            let base = self.base();
            for i in 0..base.get_child_count() {
                if let Some(child) = base.get_child(i) {
                    if child.get_name().to_string().starts_with(PREFIX) {
                        to_remove.push(child);
                    }
                }
            }
        }
        for child in to_remove {
            self.base_mut().remove_child(&child);
            child.free();
        }
    }
}

/// Horizontal reach of the visual skirt ring beyond the heightmap
/// edge, in meters. Hides the abrupt mesh boundary against the
/// horizon without affecting collision or sim extents.
///
/// **Keep this short relative to SKIRT_DROP_M.** Too-gentle a slope
/// on a wide skirt (e.g. 1500 m × 50 m drop = ~2° slope) creates
/// a near-flat huge surface whose triplanar top-projection gets
/// stretched horribly under anisotropic filtering — reads as
/// horizontal banded streaks at map edges (verified 2026-04-24).
/// 250 m × 120 m drop = ~26° slope keeps the edge-hiding job done
/// without the anisotropic blow-up.
const SKIRT_OUT_M: f32 = 250.0;

/// How far below the adjacent heightmap edge the skirt ring sits.
/// A steep drop matters as much as the horizontal reach — see the
/// `SKIRT_OUT_M` doc above.
const SKIRT_DROP_M: f32 = 120.0;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Slope-derived ground color for maps that shipped without a
/// `features.r8` classification layer. Blends three source colors
/// (ground / rock / water) by slope + elevation heuristics.
///
/// Fallback path — preferred code path is
/// [`classify_feature_color`] which uses ESA WorldCover data from
/// `features.r8`. Kept alive so test maps and any map that hasn't
/// been re-baked with `[features]` in its spec still render
/// sensibly instead of falling back to a flat tint.
fn classify_ground_color(t: f32, flatness: f32) -> (f32, f32, f32) {
    let ground = (0.35 + 0.40 * t, 0.55 - 0.15 * t, 0.35 - 0.15 * t);
    let rock = (0.50, 0.45, 0.40);
    let water = (0.20, 0.35, 0.45);

    let cliff_w = 1.0 - smoothstep(0.5, 0.8, flatness);
    let water_w = if flatness > 0.97 {
        smoothstep(0.05, 0.0, t)
    } else {
        0.0
    };
    let ground_w = (1.0 - cliff_w - water_w).max(0.0);

    (
        ground.0 * ground_w + rock.0 * cliff_w + water.0 * water_w,
        ground.1 * ground_w + rock.1 * cliff_w + water.1 * water_w,
        ground.2 * ground_w + rock.2 * cliff_w + water.2 * water_w,
    )
}

/// Color for a given feature class from `features.r8`. Elevation
/// parameter `t` tints within a class (e.g. forest darkens slightly
/// at higher elevation to hint at conifer density changes) without
/// changing the categorical identity. First-pass palette — slice 5
/// will replace vertex-color tinting with real textured materials.
fn classify_feature_color(class: FeatureClass, t: f32) -> (f32, f32, f32) {
    let shade = |base: (f32, f32, f32), lo: f32, hi: f32| -> (f32, f32, f32) {
        let mul = lo + (hi - lo) * t;
        (base.0 * mul, base.1 * mul, base.2 * mul)
    };
    match class {
        FeatureClass::Water => (0.12, 0.28, 0.40),
        FeatureClass::Forest => shade((0.18, 0.35, 0.18), 0.8, 1.1),
        FeatureClass::Shrubland => shade((0.45, 0.45, 0.25), 0.85, 1.1),
        FeatureClass::Grassland => shade((0.55, 0.58, 0.28), 0.85, 1.15),
        FeatureClass::Cropland => shade((0.62, 0.55, 0.25), 0.9, 1.15),
        FeatureClass::BuiltUp => (0.45, 0.42, 0.40),
        FeatureClass::Bare => shade((0.60, 0.52, 0.38), 0.9, 1.1),
        FeatureClass::Snow => (0.90, 0.92, 0.95),
        FeatureClass::Wetland => (0.30, 0.38, 0.28),
        FeatureClass::Moss => (0.42, 0.50, 0.32),
        FeatureClass::Cliff => shade((0.50, 0.45, 0.40), 0.85, 1.1),
        // Paved road: dark neutral asphalt, very slight warmth.
        FeatureClass::PavedRoad => (0.22, 0.22, 0.24),
        // Unpaved / gravel: warm tan/ochre, reads as dirt track.
        FeatureClass::UnpavedRoad => (0.48, 0.38, 0.24),
        // Trail: lighter dusty tan, distinct from unpaved road.
        FeatureClass::Trail => (0.58, 0.48, 0.32),
        FeatureClass::Unknown => (0.40, 0.40, 0.40),
    }
}

fn build_mesh_instance(
    hm: &Heightmap,
    shader_template: Option<Gd<ShaderMaterial>>,
) -> Gd<MeshInstance3D> {
    let w = hm.width() as i32;
    let h = hm.height() as i32;
    let spacing = hm.metadata().spacing_m;
    let half_x = (w - 1) as f32 * spacing * 0.5;
    let half_z = (h - 1) as f32 * spacing * 0.5;

    // Extend the visual grid by one ring on each side. The outer
    // ring is the skirt — pushed outward by SKIRT_OUT_M and dropped
    // SKIRT_DROP_M below its adjacent edge sample. Sim + collision
    // continue to operate on the original W×H grid; the skirt is
    // render-only.
    let we = w + 2;
    let he = h + 2;

    let vertex_count = (we * he) as usize;
    let mut vertices = PackedVector3Array::new();
    let mut normals = PackedVector3Array::new();
    let mut uvs = PackedVector2Array::new();
    let mut colors = PackedColorArray::new();
    vertices.resize(vertex_count);
    normals.resize(vertex_count);
    uvs.resize(vertex_count);
    colors.resize(vertex_count);

    let vert_min = hm.metadata().vert_min_m;
    let vert_max = hm.metadata().vert_max_m;
    let vert_span = (vert_max - vert_min).max(1.0);

    // Track min/max Y of the real grid only (skirt elevations are
    // derived, not sampled, and would skew the diagnostic).
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;

    for ze in 0..he {
        for xe in 0..we {
            // Clamp to real grid indices; the outer ring clamps to
            // the nearest edge sample so the skirt inherits its Y
            // from the adjacent real vertex.
            let x = (xe - 1).clamp(0, w - 1);
            let z = (ze - 1).clamp(0, h - 1);
            let in_skirt = xe == 0 || xe == we - 1 || ze == 0 || ze == he - 1;

            let world_x_real = x as f32 * spacing;
            let world_z_real = z as f32 * spacing;
            let edge_y = hm.sample(world_x_real, world_z_real);

            let sx = if xe == 0 {
                -SKIRT_OUT_M
            } else if xe == we - 1 {
                SKIRT_OUT_M
            } else {
                0.0
            };
            let sz = if ze == 0 {
                -SKIRT_OUT_M
            } else if ze == he - 1 {
                SKIRT_OUT_M
            } else {
                0.0
            };
            let (world_x, world_z, y) = if in_skirt {
                (world_x_real + sx, world_z_real + sz, edge_y - SKIRT_DROP_M)
            } else {
                (world_x_real, world_z_real, edge_y)
            };

            let n = hm.sample_normal(world_x_real, world_z_real);
            let idx = (ze * we + xe) as usize;
            vertices[idx] = Vector3::new(world_x - half_x, y, world_z - half_z);
            normals[idx] = Vector3::new(n[0], n[1], n[2]);
            uvs[idx] = Vector2::new(xe as f32 / (we - 1) as f32, ze as f32 / (he - 1) as f32);
            // Classification-driven tint. Prefers features.r8 from
            // ESA WorldCover when present; falls back to the slope +
            // elevation heuristic for maps that haven't been baked
            // with a [features] source yet. Skirt ring dims 20% so it
            // reads as receding terrain. Debug-tier pending slice 5's
            // real textured shader.
            let t = ((edge_y - vert_min) / vert_span).clamp(0.0, 1.0);
            let flatness = n[1].clamp(0.0, 1.0);
            let (mut r, mut g, mut b) = if hm.has_features() {
                classify_feature_color(hm.sample_feature(world_x_real, world_z_real), t)
            } else {
                classify_ground_color(t, flatness)
            };
            if in_skirt {
                r *= 0.8;
                g *= 0.8;
                b *= 0.8;
            }
            colors[idx] = Color::from_rgba(r, g, b, 1.0);
            if !in_skirt {
                if edge_y < y_min {
                    y_min = edge_y;
                }
                if edge_y > y_max {
                    y_max = edge_y;
                }
            }
        }
    }

    let triangle_count = ((we - 1) * (he - 1) * 2) as usize;
    let mut indices = PackedInt32Array::new();
    indices.resize(triangle_count * 3);
    let mut tri_i = 0usize;
    for ze in 0..(he - 1) {
        for xe in 0..(we - 1) {
            let tl = ze * we + xe;
            let tr = tl + 1;
            let bl = tl + we;
            let br = bl + 1;
            // Triangle 1: tl, bl, tr
            indices[tri_i] = tl;
            indices[tri_i + 1] = bl;
            indices[tri_i + 2] = tr;
            // Triangle 2: tr, bl, br
            indices[tri_i + 3] = tr;
            indices[tri_i + 4] = bl;
            indices[tri_i + 5] = br;
            tri_i += 6;
        }
    }

    let mut arrays = VarArray::new();
    arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
    arrays.set(ArrayType::VERTEX.ord() as usize, &vertices.to_variant());
    arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
    arrays.set(ArrayType::TEX_UV.ord() as usize, &uvs.to_variant());
    arrays.set(ArrayType::COLOR.ord() as usize, &colors.to_variant());
    arrays.set(ArrayType::INDEX.ord() as usize, &indices.to_variant());

    godot_print!(
        "TerrainNode: sampled y range [{:.2}, {:.2}] (metadata says [{:.2}, {:.2}])",
        y_min,
        y_max,
        vert_min,
        vert_max
    );

    let mut mesh = ArrayMesh::new_gd();
    mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);

    // Prefer the Godot-authored ShaderMaterial + features.r8 path
    // when both are available; fall back to vertex-color
    // StandardMaterial3D for maps that shipped without a feature
    // layer (test maps, maps re-baked before slice 2 landed).
    let material: Gd<Material> = build_terrain_material(hm, shader_template);
    mesh.surface_set_material(0, &material);

    let mut mi = MeshInstance3D::new_alloc();
    mi.set_name("__Terrain_Mesh");
    mi.set_mesh(&mesh);
    mi
}

/// Build the material for the terrain mesh. If the map has a
/// `features.r8` AND a Godot-authored ShaderMaterial was provided,
/// wires up the runtime-only uniforms (features texture + terrain
/// extent + texture arrays) on the live template resource and
/// returns it. Falls back to vertex-color StandardMaterial3D in all
/// other cases.
///
/// **Do not duplicate the template.** Gd<ShaderMaterial> is a
/// ref-counted handle to the loaded `terrain.tres`; using it
/// directly means Inspector slider edits on the .tres flow straight
/// to the running mesh. Duplicating was tried earlier (to avoid
/// "per-map writes leaking into the shared asset") but broke the
/// designer workflow completely — sliders just did nothing. If we
/// ever need per-map override of tunables, the right fix is to
/// expose those as explicit @export overrides on TerrainNode, not
/// to isolate the runtime material from its source.
fn build_terrain_material(
    hm: &Heightmap,
    shader_template: Option<Gd<ShaderMaterial>>,
) -> Gd<Material> {
    if let (Some(features), Some(template)) = (hm.features_bytes(), shader_template) {
        if let Some(mat) = try_build_feature_shader_material(hm, features, template) {
            return mat.upcast();
        }
    }
    build_vertex_color_material().upcast()
}

/// Wire up the runtime-variable uniforms on the **live** template
/// ShaderMaterial (not a duplicate — see `build_terrain_material`
/// above for why):
///   - `features_texture` — single-channel R8 image built from the
///     map's `features.r8` byte grid
///   - `terrain_extent_m` — this map's world-space size
///   - 4× `*_array` sampler2DArrays — the coherent terrain texture
///     pack (primary + 3 variants per FeatureClass)
///
/// All the *tunable* uniforms (tile scale, variant strength, debug
/// mode, warp parameters, etc.) are set by the designer in the
/// Inspector and already live on the template — we leave those
/// alone and every map picks up the current values automatically.
///
/// Returns `None` if the features Image allocation refuses; caller
/// falls back to the vertex-color path.
fn try_build_feature_shader_material(
    hm: &Heightmap,
    features: &[u8],
    template: Gd<ShaderMaterial>,
) -> Option<Gd<ShaderMaterial>> {
    let w = hm.width() as i32;
    let h = hm.height() as i32;
    let expected = (w * h) as usize;
    if features.len() != expected {
        godot_warn!(
            "TerrainNode: features byte count mismatch ({} vs {}×{}={}), using vertex-color fallback",
            features.len(),
            w,
            h,
            expected
        );
        return None;
    }
    let mut bytes = PackedByteArray::new();
    bytes.resize(features.len());
    for (i, &b) in features.iter().enumerate() {
        bytes[i] = b;
    }
    let image = Image::create_from_data(w, h, false, ImageFormat::R8, &bytes)?;
    let texture = ImageTexture::create_from_image(&image)?;

    // Use the template ShaderMaterial directly — no duplicate. This
    // is what makes Inspector slider edits (tile scale, warp, debug
    // mode, etc.) flow through to the running mesh. The per-map
    // writes below (features_texture, extent, arrays) overwrite
    // whatever the previous map set, which is fine because only one
    // terrain is live at a time.
    let mut mat = template;
    mat.set_shader_parameter("features_texture", &texture.to_variant());

    // Optional splatmap pair — RGBA8 textures with per-channel blend
    // weights for 8 class groups (see `simn_terrain::splatmap`).
    // Filter linear so neighbor cells interpolate smoothly (the
    // whole point of moving from categorical class indices to
    // splatmaps). When absent (legacy bake without splatmaps) the
    // shader falls back via the `splatmap_present` uniform.
    let mut splatmap_present = false;
    if let (Some(sa_bytes), Some(sb_bytes)) = (hm.splatmap_a_bytes(), hm.splatmap_b_bytes()) {
        let expected_rgba = (w * h * 4) as usize;
        if sa_bytes.len() == expected_rgba && sb_bytes.len() == expected_rgba {
            let mut a_pba = PackedByteArray::new();
            a_pba.resize(sa_bytes.len());
            for (i, &b) in sa_bytes.iter().enumerate() {
                a_pba[i] = b;
            }
            let mut b_pba = PackedByteArray::new();
            b_pba.resize(sb_bytes.len());
            for (i, &b) in sb_bytes.iter().enumerate() {
                b_pba[i] = b;
            }
            if let (Some(a_img), Some(b_img)) = (
                Image::create_from_data(w, h, false, ImageFormat::RGBA8, &a_pba),
                Image::create_from_data(w, h, false, ImageFormat::RGBA8, &b_pba),
            ) {
                if let (Some(a_tex), Some(b_tex)) = (
                    ImageTexture::create_from_image(&a_img),
                    ImageTexture::create_from_image(&b_img),
                ) {
                    mat.set_shader_parameter("splatmap_a", &a_tex.to_variant());
                    mat.set_shader_parameter("splatmap_b", &b_tex.to_variant());
                    splatmap_present = true;
                }
            }
        } else {
            godot_warn!(
                "TerrainNode: splatmap byte size mismatch (got A={}, B={}, expected {} each)",
                sa_bytes.len(),
                sb_bytes.len(),
                expected_rgba
            );
        }
    }
    mat.set_shader_parameter("splatmap_present", &splatmap_present.to_variant());

    // Road density — RGBA8 with smooth per-class road weights
    // (R=Paved, G=Unpaved, B=Trail, A=reserved). Sampled with
    // filter_linear in the shader for sub-cell-precision road
    // edges; replaces the previous shader-time 5×5 Gaussian.
    let mut road_density_present = false;
    if let Some(rd_bytes) = hm.road_density_bytes() {
        let expected_rgba = (w * h * 4) as usize;
        if rd_bytes.len() == expected_rgba {
            let mut rd_pba = PackedByteArray::new();
            rd_pba.resize(rd_bytes.len());
            for (i, &b) in rd_bytes.iter().enumerate() {
                rd_pba[i] = b;
            }
            if let Some(rd_img) = Image::create_from_data(w, h, false, ImageFormat::RGBA8, &rd_pba)
            {
                if let Some(rd_tex) = ImageTexture::create_from_image(&rd_img) {
                    mat.set_shader_parameter("road_density", &rd_tex.to_variant());
                    road_density_present = true;
                }
            }
        } else {
            godot_warn!(
                "TerrainNode: road_density byte size mismatch (got {}, expected {})",
                rd_bytes.len(),
                expected_rgba
            );
        }
    }
    mat.set_shader_parameter("road_density_present", &road_density_present.to_variant());

    // Primary diffuse array — one layer per FeatureClass. Best-effort:
    // if any single texture fails to load (missing asset, misnamed
    // path), log + fall through without the array uniform. The shader
    // then samples a default empty texture and reads black for that
    // class only; the rest of the map still gets its proper tints.
    if let Some(array) = build_diffuse_array(&TERRAIN_CLASS_DIFFUSE, "primary") {
        mat.set_shader_parameter("diffuse_array", &array.to_variant());
    } else {
        godot_warn!(
            "TerrainNode: diffuse_array build failed — shader will render black \
             where texture lookups happen. Check paths in TERRAIN_CLASS_DIFFUSE."
        );
    }

    // Variant diffuse array — a second texture per class from the
    // same family, blended in the shader by low-frequency noise so
    // every class shows natural within-class variation. Best-effort
    // again; if absent the shader falls back to primary-only.
    if let Some(array) = build_diffuse_array(&TERRAIN_CLASS_DIFFUSE_VARIANT, "variant") {
        mat.set_shader_parameter("variant_array", &array.to_variant());
    }

    // Third-variant diffuse array — another family member blended
    // at a decoupled noise frequency so the primary/variant pattern
    // isn't itself a visible repeat. Three textures per class at
    // two noise scales = natural-looking patches across the map.
    if let Some(array) = build_diffuse_array(&TERRAIN_CLASS_DIFFUSE_VARIANT2, "variant2") {
        mat.set_shader_parameter("variant2_array", &array.to_variant());
    }

    // Fourth-variant diffuse array — one more family member to
    // maximize within-class variation. Every class now shows up
    // to four different textures blending smoothly across the
    // terrain at three independent noise scales.
    if let Some(array) = build_diffuse_array(&TERRAIN_CLASS_DIFFUSE_VARIANT3, "variant3") {
        mat.set_shader_parameter("variant3_array", &array.to_variant());
    }

    // Per-class normal + roughness arrays. Both are best-effort: a
    // missing layer signals `*_present` false and the shader falls
    // back to a flat normal / fixed roughness for that class. Used
    // primary-only (no variants) — the cost of triplanar-sampling
    // four normal arrays per fragment doesn't pay off; one normal
    // map per class is plenty for surface micro-detail.
    let mut normal_present = false;
    if let Some(array) = build_diffuse_array(&TERRAIN_CLASS_NORMAL, "normal") {
        mat.set_shader_parameter("normal_array", &array.to_variant());
        normal_present = true;
    }
    mat.set_shader_parameter("normal_present", &normal_present.to_variant());

    let mut roughness_present = false;
    if let Some(array) = build_diffuse_array(&TERRAIN_CLASS_ROUGHNESS, "roughness") {
        mat.set_shader_parameter("roughness_array", &array.to_variant());
        roughness_present = true;
    }
    mat.set_shader_parameter("roughness_present", &roughness_present.to_variant());

    let [ext_x, ext_z] = hm.extent_m();
    mat.set_shader_parameter("terrain_extent_m", &Vector2::new(ext_x, ext_z).to_variant());

    Some(mat)
}

/// Load a list of diffuse texture paths and stack into a
/// `Texture2DArray`. Requires every texture to decode to the same
/// dimensions (Godot's enforcement, not ours). Returns `None` if
/// any layer fails to load or if create_from_images rejects the
/// stack. `label` only affects diagnostic logging.
/// Common edge length we resize every texture-array layer to before
/// stacking. `Texture2DArray::create_from_images` rejects mixed sizes,
/// and our source bundles ship at 2K, 4K, and 8K (Megascans).
/// 2048 matches `HTerrainTextureSetBuilder::PACK_RESOLUTION` so the
/// HTerrain plugin path and the TerrainNode path see the same fidelity.
const TEXTURE_ARRAY_RESOLUTION: i32 = 2048;

pub(crate) fn build_diffuse_array(paths: &[&str], label: &str) -> Option<Gd<Texture2DArray>> {
    use godot::builtin::Array;
    use godot::classes::image::Interpolation;
    let mut images: Array<Gd<Image>> = Array::new();
    for (idx, path) in paths.iter().enumerate() {
        match try_load::<Texture2D>(*path) {
            Ok(tex) => match tex.get_image() {
                Some(mut img) => {
                    // Resize every layer to the common edge length.
                    // Bilinear is the right call on diffuse / normal /
                    // roughness alike — Lanczos-only artifacts on
                    // unpacked normals aren't visible at the 2K target,
                    // and bilinear is what HTerrain's builder uses too.
                    if img.get_width() != TEXTURE_ARRAY_RESOLUTION
                        || img.get_height() != TEXTURE_ARRAY_RESOLUTION
                    {
                        img.resize_ex(TEXTURE_ARRAY_RESOLUTION, TEXTURE_ARRAY_RESOLUTION)
                            .interpolation(Interpolation::BILINEAR)
                            .done();
                    }
                    images.push(&img);
                }
                None => {
                    godot_warn!(
                        "TerrainNode: {label} class {idx} texture {path:?} has no Image payload"
                    );
                    return None;
                }
            },
            Err(e) => {
                godot_warn!(
                    "TerrainNode: {label} class {idx} texture {path:?} failed to load: {e:?}"
                );
                return None;
            }
        }
    }
    let mut arr = Texture2DArray::new_gd();
    if arr.create_from_images(&images) != godot::global::Error::OK {
        godot_warn!("TerrainNode: {label} Texture2DArray::create_from_images rejected the stack");
        return None;
    }
    Some(arr)
}

/// Pre-slice-5 vertex-color material. Used when features.r8 is
/// absent (test maps) or the shader fails to load.
fn build_vertex_color_material() -> Gd<StandardMaterial3D> {
    let mut mat = StandardMaterial3D::new_gd();
    mat.set_albedo(Color::from_rgba(1.0, 1.0, 1.0, 1.0));
    mat.set_roughness(0.9);
    mat.set_flag(Flags::ALBEDO_FROM_VERTEX_COLOR, true);
    // Double-sided: if vertex winding happens to be wrong we still see
    // the surface rather than a silent backface-culled void.
    mat.set_cull_mode(CullMode::DISABLED);
    mat
}

fn build_collision_shape(hm: &Heightmap) -> Gd<CollisionShape3D> {
    let w = hm.width() as i32;
    let h = hm.height() as i32;
    let spacing = hm.metadata().spacing_m;

    let mut shape = HeightMapShape3D::new_gd();
    shape.set_map_width(w);
    shape.set_map_depth(h);

    // HeightMapShape3D expects samples in a PackedFloat32Array, row-major.
    let mut map_data = PackedFloat32Array::new();
    map_data.resize((w * h) as usize);
    for z in 0..h {
        for x in 0..w {
            let world_x = x as f32 * spacing;
            let world_z = z as f32 * spacing;
            map_data[(z * w + x) as usize] = hm.sample(world_x, world_z);
        }
    }
    shape.set_map_data(&map_data);

    let mut cs = CollisionShape3D::new_alloc();
    cs.set_name("__Terrain_Collision");
    // HeightMapShape3D samples 1 unit apart in its local space; scale
    // X and Z by `spacing_m` so the physical extent matches the visual
    // mesh and the sim's world-local coordinates.
    cs.set_scale(Vector3::new(spacing, 1.0, spacing));
    cs.set_shape(&shape);
    cs
}
