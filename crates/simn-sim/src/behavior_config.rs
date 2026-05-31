//! TOML-driven behavior tuning for NPC movement, planning, combat,
//! aggro, and goal arbitration. Edit `content/behavior.toml` to tweak
//! without recompiling.

use bevy_ecs::prelude::Resource;
use serde::Deserialize;

#[derive(Resource, Clone, Debug, Deserialize)]
pub struct BehaviorConfig {
    pub movement: MovementConfig,
    pub planning: PlanningConfig,
    pub combat: CombatConfig,
    pub aggro: AggroConfig,
    pub arbitration: ArbitrationConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MovementConfig {
    pub walk_speed_mps: f32,
    pub arrive_radius_m: f32,
    pub engage_range_m: f32,
    pub squad_arrive_radius_m: f32,
    pub squad_member_arrive_m: f32,
    pub path_recompute_dist_m: f32,
    pub path_max_age_ticks: u64,
    pub path_budget_per_tick: u32,
    pub long_march_prob: f64,
    pub long_march_radius_m: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PlanningConfig {
    pub planner_interval_ticks: u64,
    pub objective_default_duration_ticks: u64,
    pub patrol_duration_ticks: u64,
    pub guard_tenure_ticks: u64,
    pub rest_duration_ticks: u64,
    pub guard_duration_ticks: u64,
    pub investigate_duration_ticks: u64,
    pub investigate_arrival_dwell_ticks: u64,
    pub wander_duration_ticks: u64,
    pub explore_duration_ticks: u64,
    pub relieve_duration_ticks: u64,
    pub wander_drift_reroll_ticks: u64,
    pub dispersion: DispersionConfig,
    pub stuck_detection: StuckDetectionConfig,
    pub cohesion: CohesionConfig,
    pub dwell: DwellConfig,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DispersionConfig {
    pub min_dist_m: f32,
    pub max_dist_m: f32,
    pub arrive_radius_m: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct StuckDetectionConfig {
    pub progress_m: f32,
    pub timeout_ticks: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CohesionConfig {
    pub break_distance_m: f32,
    pub regroup_duration_ticks: u64,
    pub regroup_cooldown_ticks: u64,
    pub failed_regroup_disable_ticks: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DwellConfig {
    pub guard_ticks: u64,
    pub rest_ticks: u64,
    pub patrol_waypoint_ticks: u64,
    pub default_ticks: u64,
    /// Per-NPC fractional jitter applied to dwell durations so squad
    /// members don't all expire their dwell on the same tick. Range
    /// `[1 - frac, 1 + frac]` × the base duration, deterministic per
    /// `npc.id`. 0.0 disables.
    #[serde(default = "default_dwell_jitter_frac")]
    pub jitter_frac: f32,
    /// How often a guarding NPC nudges its position within the
    /// formation ring during a long Guard dwell. Without this NPCs
    /// stand perfectly still for the entire `guard_ticks` (~20 min
    /// real), which reads as frozen. 0 disables.
    #[serde(default = "default_guard_shift_interval_ticks")]
    pub guard_shift_interval_ticks: u64,
    /// Max radius (m) of the per-shift position nudge applied to
    /// guarding NPCs. Kept small so the NPC stays within its formation
    /// slot.
    #[serde(default = "default_guard_shift_radius_m")]
    pub guard_shift_radius_m: f32,
}

fn default_dwell_jitter_frac() -> f32 {
    0.30
}
fn default_guard_shift_interval_ticks() -> u64 {
    500
}
fn default_guard_shift_radius_m() -> f32 {
    1.8
}

#[derive(Clone, Debug, Deserialize)]
pub struct CombatConfig {
    pub cover_search_radius_m: f32,
    pub peek_interval_ticks: u64,
    pub peek_duration_ticks: u64,
    pub suppression_hit_threshold: usize,
    pub suppression_attacker_threshold: usize,
    pub suppression_window_ticks: u64,
    pub suppression_duration_ticks: u64,
    pub retreat_health_frac: f32,
    pub pointman_push_health_frac: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AggroConfig {
    pub decay_ticks: u64,
    pub pursue_progress_m: f32,
    pub pursue_timeout_ticks: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ArbitrationConfig {
    pub hysteresis_prio_delta: u8,
    pub commitment_ticks: u64,
    pub responder_cap_per_target: u8,
    pub solo_borrow_radius_m: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WorldTimeConfig {
    pub day_length_seconds: f32,
    pub start_hour: f32,
}

impl BehaviorConfig {
    pub fn load() -> Self {
        static CACHE: std::sync::OnceLock<BehaviorConfig> = std::sync::OnceLock::new();
        CACHE
            .get_or_init(|| Self::parse(&crate::ContentSource::Embedded))
            .clone()
    }

    /// Load from an explicit content source; see [`crate::items::ItemRegistry::load_from`].
    pub fn load_from(src: &crate::ContentSource) -> Self {
        match src {
            crate::ContentSource::Embedded => Self::load(),
            other => Self::parse(other),
        }
    }

    fn parse(src: &crate::ContentSource) -> Self {
        let text = src
            .read_str("ai/behavior.toml")
            .unwrap_or_else(|e| panic!("behavior content load failed: {e}"));
        toml::from_str(&text).expect("behavior.toml must parse")
    }
}

impl WorldTimeConfig {
    pub fn load() -> Self {
        static CACHE: std::sync::OnceLock<WorldTimeConfig> = std::sync::OnceLock::new();
        CACHE.get_or_init(Self::parse_from_toml).clone()
    }

    // World-clock tuning is an embedded-only carve-out: it's read once
    // by `WorldTime::new()` deep in `build_world` (no content handle
    // there) and is sim-clock config rather than game content. Routed
    // through the embedded pack so it isn't double-embedded.
    fn parse_from_toml() -> Self {
        let text = crate::ContentSource::Embedded
            .read_str("world/world_time.toml")
            .expect("embedded world_time.toml present");
        toml::from_str(&text).expect("world_time.toml must parse")
    }
}
