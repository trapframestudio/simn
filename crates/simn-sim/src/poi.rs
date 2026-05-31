//! Config-driven behavior for POI (base) and activity-point types.
//!
//! The per-type behavior the sim used to hardcode (base nav footprints,
//! the victory-target flag, and the `is_guard` / `is_rest` / hunt-target
//! classification of activity points) is read from the content pack
//! (`poi/base_types.toml`, `ai/activity_types.toml`) instead.
//!
//! Like [`crate::cover::material_table`], these are process-global tables
//! read from the embedded pack: the lookups (`BaseKind::nav_footprint_xz_m`,
//! `ActivityKind::is_guard`, etc.) are called on the tick path with no
//! content handle, so they aren't per-`ContentSource` overridable. The
//! `BaseKind` / `ActivityKind` tags remain the engine's POI/activity
//! vocabulary; this module makes their *behavior* data-driven.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::components::BaseKind;
use crate::content::ContentSource;
use crate::resources::ActivityKind;

#[derive(Clone, Copy, Debug, Deserialize)]
struct BaseTypeEntry {
    kind: BaseKind,
    #[serde(default)]
    nav_footprint: Option<[f32; 2]>,
    #[serde(default)]
    is_victory_target: bool,
}

#[derive(Deserialize)]
struct BaseTypesFile {
    base_type: Vec<BaseTypeEntry>,
}

static BASE_TYPES: OnceLock<HashMap<BaseKind, BaseTypeEntry>> = OnceLock::new();

fn base_types() -> &'static HashMap<BaseKind, BaseTypeEntry> {
    BASE_TYPES.get_or_init(|| {
        let toml = ContentSource::Embedded
            .read_str("poi/base_types.toml")
            .expect("embedded poi/base_types.toml present");
        let file: BaseTypesFile = toml::from_str(&toml).expect("base_types.toml parse");
        file.base_type.into_iter().map(|e| (e.kind, e)).collect()
    })
}

/// Blocked nav footprint `[x, z]` in metres for a base kind, or `None`
/// for an open/unstructured site. Data-driven via `poi/base_types.toml`.
pub fn base_nav_footprint(kind: BaseKind) -> Option<[f32; 2]> {
    base_types().get(&kind).and_then(|e| e.nav_footprint)
}

/// Whether capturing a base of this kind ends the faction contest.
/// Data-driven via `poi/base_types.toml`.
pub fn base_is_victory_target(kind: BaseKind) -> bool {
    base_types()
        .get(&kind)
        .map(|e| e.is_victory_target)
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ActivityTypeEntry {
    kind: ActivityKind,
    #[serde(default)]
    is_guard: bool,
    #[serde(default)]
    is_rest: bool,
    #[serde(default)]
    hunt_target: bool,
}

#[derive(Deserialize)]
struct ActivityTypesFile {
    activity_type: Vec<ActivityTypeEntry>,
}

static ACTIVITY_TYPES: OnceLock<HashMap<ActivityKind, ActivityTypeEntry>> = OnceLock::new();

fn activity_types() -> &'static HashMap<ActivityKind, ActivityTypeEntry> {
    ACTIVITY_TYPES.get_or_init(|| {
        let toml = ContentSource::Embedded
            .read_str("ai/activity_types.toml")
            .expect("embedded ai/activity_types.toml present");
        let file: ActivityTypesFile = toml::from_str(&toml).expect("activity_types.toml parse");
        file.activity_type
            .into_iter()
            .map(|e| (e.kind, e))
            .collect()
    })
}

/// Whether NPCs hold a guard post at this activity type.
pub fn activity_is_guard(kind: ActivityKind) -> bool {
    activity_types()
        .get(&kind)
        .map(|e| e.is_guard)
        .unwrap_or(false)
}

/// Whether this activity type is a rest / socialize spot.
pub fn activity_is_rest(kind: ActivityKind) -> bool {
    activity_types()
        .get(&kind)
        .map(|e| e.is_rest)
        .unwrap_or(false)
}

/// Whether a curious/greedy NPC investigates this activity type at a
/// hostile base (stashes, lookouts, workbenches by default).
pub fn activity_is_hunt_target(kind: ActivityKind) -> bool {
    activity_types()
        .get(&kind)
        .map(|e| e.hunt_target)
        .unwrap_or(false)
}
