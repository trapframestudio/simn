//! ECS resources (world-scoped singletons).

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::components::NpcId;
use crate::delta::WorldDelta;

use crate::region::RegionId;

/// Togglable behavior-logging. When `enabled`, NPC systems bump
/// counters here instead of emitting one tracing line per event;
/// `Sim::tick` flushes a single summary line per `FLUSH_INTERVAL`
/// ticks under target `npc.behavior`. Rare events (deaths,
/// migrations) are also emitted individually — those are the
/// interesting ones and they're not chatty at steady state.
/// Off by default. The in-game F9 toggle and the `watch` example
/// both flip `enabled`.
#[derive(Resource, Default, Debug)]
pub struct BehaviorLog {
    pub enabled: bool,
    pub spawns: u32,
    pub deaths: u32,
    pub migrations: u32,
    pub aggro_acquisitions: u32,
    pub objectives: std::collections::HashMap<&'static str, u32>,
    pub spawns_by_faction: std::collections::HashMap<String, u32>,
    pub deaths_by_cause: std::collections::HashMap<&'static str, u32>,
    pub migrations_by_region: std::collections::HashMap<RegionId, u32>,
    pub last_flush_tick: u64,
}

impl BehaviorLog {
    /// Ticks between summary flushes. 100 ticks = 5s at the sim's
    /// fixed 20Hz, which feels responsive without spamming the
    /// console.
    pub const FLUSH_INTERVAL: u64 = 100;

    pub fn reset_counters(&mut self) {
        self.spawns = 0;
        self.deaths = 0;
        self.migrations = 0;
        self.aggro_acquisitions = 0;
        self.objectives.clear();
        self.spawns_by_faction.clear();
        self.deaths_by_cause.clear();
        self.migrations_by_region.clear();
    }
}

/// Monotonic tick counter plus fixed timestep.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug)]
pub struct SimClock {
    pub tick: u64,
    pub fixed_dt_ms: u32,
}

impl SimClock {
    /// 20Hz default — matches the network broadcast rate so one sim
    /// tick == one broadcast opportunity.
    pub const DEFAULT_DT_MS: u32 = 50;

    pub fn new() -> Self {
        Self {
            tick: 0,
            fixed_dt_ms: Self::DEFAULT_DT_MS,
        }
    }
}

impl Default for SimClock {
    fn default() -> Self {
        Self::new()
    }
}

/// On-disk paths for the snapshot and journal files. Not serialized —
/// rebuilt from the save directory at start time.
#[derive(Resource, Clone, Debug)]
pub struct SavePaths {
    pub snapshot: PathBuf,
    pub journal: PathBuf,
}

impl SavePaths {
    /// Resolve standard paths within a save directory.
    pub fn in_dir(dir: &std::path::Path) -> Self {
        Self {
            snapshot: dir.join("world.save"),
            journal: dir.join("world.journal"),
        }
    }

    /// Resolve paths for a named run within the user-data root. Each
    /// run (solo or coop-host) owns its own subdirectory
    /// `user_data/saves/<run_id>/` so runs are independent on disk.
    /// Joining clients don't use `SavePaths` at all — their mirror
    /// sim has no persistence.
    pub fn in_run_dir(user_data_dir: &std::path::Path, run_id: &str) -> Self {
        Self::in_dir(&user_data_dir.join("saves").join(run_id))
    }
}

/// Which region(s) are "online" — i.e. get the full per-NPC tick
/// (aggro pair scan, FOV/LOS, combat, formation offsets, position
/// interpolation). NPCs in other regions still spawn, die, migrate,
/// and receive squad objectives, but skip the expensive O(n²) work.
/// Set by `Sim::set_active_region` (called from `SimHost` when the
/// player enters a region). Backed by a `HashSet` for O(1) lookups
/// (per `sim-hardening-plan.md` §4 — landed 2026-05-09).
/// Multiple regions can be active if
/// multiple players are in different regions (future multiplayer);
/// for now it's typically one. Backed by a `HashSet` for O(1)
/// `is_active` lookups — every NPC tick currently calls this for
/// online/offline tier branching, so a Vec scan would scale
/// poorly with `O(active_regions × NPCs)`. Iteration order isn't
/// stable, but no consumer iterates this set so determinism is
/// unaffected (per `sim-hardening-plan.md` §4).
#[derive(Resource, Default, Clone, Debug)]
pub struct ActiveRegions {
    pub regions: std::collections::HashSet<RegionId>,
}

impl ActiveRegions {
    pub fn is_active(&self, region: RegionId) -> bool {
        self.regions.contains(&region)
    }
}

/// In-world clock. Independent of [`SimClock`] — `SimClock::tick` is
/// the fixed-tick engine counter, `WorldTime` is what the game's
/// inhabitants would read off a watch.
///
/// Advanced by the `advance_world_time` system each tick, scaled so
/// the configured `day_length_seconds` of real time equals one
/// in-world day. Not journaled; recovered from the latest snapshot
/// on load, so a crash can cost up to a snapshot interval of
/// time-of-day drift. Day/night visuals shouldn't care about that.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct WorldTime {
    /// In-world days elapsed since the save was created.
    pub day: u32,
    /// Seconds into the current in-world day, in `[0, day_length_seconds)`.
    pub seconds_of_day: f32,
    /// Real-time seconds that count as one full in-world day. See
    /// [`Self::DEFAULT_DAY_LENGTH`] (7200s, ~12× real) for the default.
    /// Tests override this via `force_world_time_for_test` when they
    /// need a shorter cycle.
    pub day_length_seconds: f32,
}

impl WorldTime {
    /// Real seconds per in-world day. 7200 real seconds = 2 real
    /// hours per in-world day → 1 in-world hour = 5 real minutes.
    /// Tuned to feel like a survival sandbox ~2.4 hr/day, biased a
    /// hair tighter so evals don't have to sit through a 2.5-hour
    /// cycle. Drop this to iterate on day-night visuals faster;
    /// ~300–600s is good for "see a full cycle every eval".
    pub const DEFAULT_DAY_LENGTH: f32 = 7200.0;

    pub fn new() -> Self {
        let cfg = crate::behavior_config::WorldTimeConfig::load();
        let seconds_of_day = (cfg.start_hour / 24.0) * cfg.day_length_seconds;
        Self {
            day: 0,
            seconds_of_day,
            day_length_seconds: cfg.day_length_seconds,
        }
    }

    /// Fraction of the in-world day elapsed (0.0 at midnight,
    /// 0.5 at noon, approaches 1.0 at the next midnight).
    pub fn day_fraction(&self) -> f32 {
        if self.day_length_seconds <= 0.0 {
            return 0.0;
        }
        (self.seconds_of_day / self.day_length_seconds).clamp(0.0, 1.0)
    }

    /// Sun angle above the horizon, in radians. 0 at sunrise (06:00)
    /// and sunset (18:00), π/2 at noon, negative at night. Bias by
    /// -π/2 so 06:00 sits on the horizon for a standard east-west
    /// arc. Used by the Godot sky controller to rotate the
    /// `DirectionalLight3D` and by any gameplay code that wants to
    /// know "is the sun up right now?".
    pub fn sun_angle_rad(&self) -> f32 {
        // Map day fraction to a full 2π rotation offset so dawn is
        // at 0.25 (06:00), noon at 0.5, dusk at 0.75, midnight at
        // 0 / 1.0. Sin of the offset from noon gives the elevation.
        use std::f32::consts::PI;
        let t = self.day_fraction();
        // Noon = 0.5; offset from noon maps to angle.
        let from_noon = (t - 0.5) * 2.0 * PI;
        // cos peaks at noon (from_noon=0), zero at dawn/dusk
        // (from_noon=±π/2), negative at night.
        from_noon.cos() * (PI / 2.0)
    }

    /// `true` if the sun is above the horizon.
    pub fn is_daytime(&self) -> bool {
        self.sun_angle_rad() > 0.0
    }

    /// Synodic-month fraction in `[0.0, 1.0)`. `0.0` = new moon
    /// (dark), `0.5` = full moon, wraps at 1.0 back to new. Derived
    /// from `day + day_fraction` modulo `LUNAR_CYCLE_DAYS` so the
    /// cycle stays in phase across saves.
    pub fn moon_phase(&self) -> f32 {
        let continuous = self.day as f32 + self.day_fraction();
        (continuous % Self::LUNAR_CYCLE_DAYS) / Self::LUNAR_CYCLE_DAYS
    }

    /// Moon illumination in `[0.0, 1.0]`. `0.0` at new, `1.0` at
    /// full, smooth cosine curve between. Used by the Godot
    /// moonlight to modulate brightness so nights get meaningfully
    /// darker around a new moon.
    pub fn moon_illumination(&self) -> f32 {
        use std::f32::consts::PI;
        let p = self.moon_phase();
        // cos peaks at full (phase 0.5). Map [0,1] → cos from -π..π.
        let from_full = (p - 0.5) * 2.0 * PI;
        let v = (1.0 - from_full.cos()) / 2.0;
        v.clamp(0.0, 1.0)
    }

    /// Moon elevation above the horizon, in radians. Shares the
    /// same daily arc as the sun but lags by the lunar phase
    /// fraction — so at full moon it rises when the sun sets, at
    /// new moon it sits near the sun (dark and invisible), at
    /// first quarter it trails six hours behind.
    pub fn moon_angle_rad(&self) -> f32 {
        use std::f32::consts::PI;
        let t = self.day_fraction();
        // Moon's "noon" is offset by phase: full moon culminates at
        // 00:00 (phase 0.5 pushes moon noon to midnight).
        let lunar_t = (t - self.moon_phase() + 1.0) % 1.0;
        let from_noon = (lunar_t - 0.5) * 2.0 * PI;
        from_noon.cos() * (PI / 2.0)
    }

    /// Synodic month in days (29.53). Matches real Earth's lunar
    /// cycle. A config resource could override this later if we
    /// want fantasy calendars.
    pub const LUNAR_CYCLE_DAYS: f32 = 29.53;
}

impl Default for WorldTime {
    fn default() -> Self {
        Self::new()
    }
}

/// Global weather state. Simple for now — a single enum applied
/// world-wide with a next-transition tick. Per-region weather (and
/// spatial weather fronts) lands later when #20 region state does.
/// The Godot sky controller lerps between visual states so changes
/// read as gradual.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeatherState {
    pub current: Weather,
    pub next: Weather,
    /// Sim tick at which `current` flips to `next` and `next`
    /// re-rolls.
    pub transitions_at_tick: u64,
}

/// Weather palette modeled on temperate maritime climate — lots of
/// gradation at the overcast / drizzle / light-rain end of the
/// spectrum where the real region spends most of its time, plus
/// seasonally-appropriate variants (marine layer, smoke haze).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Weather {
    /// Crisp blue sky. Rarer than you'd think west of the Cascades.
    Clear,
    /// Scattered cumulus, sun between breaks.
    PartlyCloudy,
    /// Flat gray ceiling. The PNW baseline 8 months of the year.
    Overcast,
    /// Shallow coastal/valley stratus that sits under a few hundred
    /// meters and burns off by mid-morning. Atmospheric, not oppressive.
    MarineLayer,
    /// Dense inland fog. Visibility genuinely limited.
    Fog,
    /// The signature "is it even raining?" coastal mist. Always
    /// wet, never heavy.
    Drizzle,
    /// Steady moderate rain.
    LightRain,
    /// Frontal rain — heavy, sustained, often wind-driven.
    HeavyRain,
    /// Sustained high wind without necessarily any rain. Coastal
    /// gales and inland "Silver Falls" events.
    Windstorm,
    /// Thunder-and-lightning cells; less common than rain but does
    /// happen in summer and during strong fall fronts.
    Thunderstorm,
    /// Summer wildfire haze. Stagnant high pressure traps smoke
    /// from upwind fires — orange sun, reduced visibility, no
    /// precipitation.
    SmokeHaze,
}

impl WeatherState {
    /// Start clear; first transition rolls at the end of the first
    /// in-game hour (see `advance_weather`). Snapshot recovery will
    /// restore this exactly.
    pub fn new() -> Self {
        Self {
            current: Weather::Clear,
            next: Weather::Clear,
            transitions_at_tick: 0,
        }
    }
}

impl Default for WeatherState {
    fn default() -> Self {
        Self::new()
    }
}

/// All weather variants in display order.
pub const ALL_WEATHER: &[Weather] = &[
    Weather::Clear,
    Weather::PartlyCloudy,
    Weather::Overcast,
    Weather::MarineLayer,
    Weather::Fog,
    Weather::Drizzle,
    Weather::LightRain,
    Weather::HeavyRain,
    Weather::Windstorm,
    Weather::Thunderstorm,
    Weather::SmokeHaze,
];

/// Per-region faction control. A `primary` faction "holds" the
/// region; `contested_by` lists other factions actively present;
/// `tension` is `0.0` (stable) → `1.0` (open conflict). Read by
/// future encounter-spawn / NPC AI systems.
///
/// Random-seeded today; authored when the real region map lands.
#[derive(Resource, Serialize, Deserialize, Clone, Debug, Default)]
pub struct RegionControl {
    #[serde(serialize_with = "crate::det_serde::sorted_map")]
    pub by_region: HashMap<RegionId, RegionControlState>,
}

/// `primary` and `contested_by` carry registry name strings so saves
/// stay valid across registry edits. Empty `primary` means a region
/// without a clear owner (peripheral / abandoned).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RegionControlState {
    pub primary: Option<String>,
    pub contested_by: Vec<String>,
    pub tension: f32,
}

impl RegionControlState {
    pub fn uncontested(faction: &str) -> Self {
        Self {
            primary: Some(faction.to_string()),
            contested_by: Vec::new(),
            tension: 0.0,
        }
    }
}

/// Monotonically-incrementing source of [`NpcId`]s. Stored as a
/// resource so the next id survives save/load — entity ids reshuffle
/// across loads, but NPC ids must be stable forever (chronicle keys,
/// journal references, eventually persistent quest state).
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct NpcIdCounter(pub u64);

impl NpcIdCounter {
    /// Return the next id and advance the counter. Named `mint` to
    /// avoid clashing with `Iterator::next`.
    pub fn mint(&mut self) -> NpcId {
        self.0 = self.0.wrapping_add(1);
        NpcId(self.0)
    }
}

/// Monotonically-incrementing source of [`crate::components::WoundId`]s.
/// Same shape as [`NpcIdCounter`]: stored as a resource so the next
/// id survives save/load. Wound ids are referenced from journal
/// records (`WoundAdded` / `WoundTreatmentChanged`); reuse would
/// silently corrupt replay.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct WoundIdCounter(pub u64);

impl WoundIdCounter {
    /// Return the next id and advance the counter.
    pub fn mint(&mut self) -> crate::components::WoundId {
        self.0 = self.0.wrapping_add(1);
        crate::components::WoundId(self.0)
    }
}

/// Same shape as [`WoundIdCounter`] for [`crate::components::EffectId`]s.
/// Persisted; never reused. Referenced by `EffectApplied` journal
/// records.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct EffectIdCounter(pub u64);

impl EffectIdCounter {
    pub fn mint(&mut self) -> crate::components::EffectId {
        self.0 = self.0.wrapping_add(1);
        crate::components::EffectId(self.0)
    }
}

/// Monotonic counter for [`crate::components::CraftJob::id`]. Persisted;
/// referenced by `CraftJobQueued` / `CraftJobCancelled` journal
/// records so reuse would alias distinct jobs.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct JobIdCounter(pub u32);

impl JobIdCounter {
    pub fn mint(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(1);
        self.0
    }
}

/// Monotonic counter for [`crate::components::ProjectileId`]. Persisted
/// so snapshot round-trips preserve in-flight bullet identities; the
/// `ProjectileSpawned` / `ProjectileImpacted` delta pair uses this id
/// to correlate trace + impact FX on the client.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct ProjectileIdCounter(pub u64);

impl ProjectileIdCounter {
    pub fn mint(&mut self) -> crate::components::ProjectileId {
        self.0 = self.0.wrapping_add(1);
        crate::components::ProjectileId(self.0)
    }
}

/// Monotonic counter for [`crate::components::ContainerId`]. Persisted;
/// referenced by `WorldContainerSpawned` / `WorldContainerItemAdded`
/// / `WorldContainerItemRemoved` / `WorldContainerDespawned` journal
/// records so reuse would alias distinct containers across reloads.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct ContainerIdCounter(pub u32);

impl ContainerIdCounter {
    pub fn mint(&mut self) -> crate::components::ContainerId {
        self.0 = self.0.wrapping_add(1);
        crate::components::ContainerId(self.0)
    }
}

/// One entry in the [`CorpseIndex`] — a fast spatial cache for the
/// loot arbiter so it doesn't have to scan every `WorldContainer`
/// every tick looking for corpses to assign as Loot targets.
#[derive(Clone, Copy, Debug)]
pub struct CorpseIndexEntry {
    pub pos: [f32; 3],
    pub region: RegionId,
    pub faction: crate::faction::registry::FactionId,
    pub spawned_tick: u64,
}

/// Index of currently-active corpse containers keyed by
/// [`crate::components::ContainerId`]. Inserted by `npc_death_check`
/// (and `npc_age` for natural deaths) when a corpse container spawns;
/// pruned by a lifecycle sweep when stale (>~10 min real) or when the
/// container is despawned. The arbiter reads this each tick when
/// resolving the `Loot` personality drive.
///
/// BTreeMap so iteration is deterministic for determinism-sensitive
/// tests. Sizes are small — corpses despawn after ~10 min of game
/// time, so even active firefights stay under a few dozen entries.
#[derive(Resource, Default)]
pub struct CorpseIndex {
    pub by_container: std::collections::BTreeMap<crate::components::ContainerId, CorpseIndexEntry>,
}

/// Tunable medical / drug / contamination timings. Lives in the world
/// so tests can override without touching constants. Not snapshotted —
/// defaults are recreated per `Sim::new` / `Sim::load`, and tuning
/// changes belong in code, not save files. All durations are in 20Hz
/// ticks; the conventional anchor is 1 in-world hour = 6000 ticks at
/// the default `WorldTime::DEFAULT_DAY_LENGTH = 7200`.
#[derive(Resource, Clone, Copy, Debug)]
pub struct MedConfig {
    /// Ticks a `Bandaged` wound takes to flip to `Healed`.
    pub heal_ticks_bandaged: u64,
    /// Ticks a `Stitched` wound takes to flip to `Healed`. Default is
    /// half of `heal_ticks_bandaged` — the reward for using the kit.
    pub heal_ticks_stitched: u64,
    /// Ticks of untreated wound age before it auto-flips to infected.
    /// Disinfecting before this elapses prevents infection. Default
    /// 12000 (= 2 in-world hours = ~10 real min).
    pub infection_trigger_ticks: u64,
    /// Ticks an antibiotics dose takes to fully clear infection.
    pub antibiotics_clear_ticks: u64,
    /// Ticks a tourniquet may be on before necrosis HP-damage starts.
    pub necrosis_warning_ticks: u64,
    /// Ticks (after necrosis_warning_ticks) before damage escalates.
    pub necrosis_severe_ticks: u64,
    /// Tolerance threshold above which another dose triggers overdose.
    pub overdose_threshold: f32,
    /// Tolerance threshold above which (no active dose + delay) triggers withdrawal.
    pub withdrawal_threshold: f32,
    /// Tolerance value below which active withdrawal lifts.
    pub withdrawal_lift_threshold: f32,
    /// Ticks since last dose required before withdrawal can trigger.
    pub withdrawal_delay_ticks: u64,
    /// Tolerance points decayed per in-world second. Default chosen so
    /// 100 → 0 takes ~50 real minutes (4 in-world hours).
    pub tolerance_decay_per_in_world_sec: f32,
    /// Pain value at or above which stamina regen is halved.
    pub pain_regen_threshold: f32,
    /// Radiation/Toxicity value at or above which slow HP drain triggers.
    pub contamination_hp_threshold: f32,
    /// HP drained per in-world second when rad or tox is above the
    /// threshold (each contributes independently).
    pub contamination_hp_drain_per_sec: f32,
    /// Per-tick passive decay of radiation toward 0.
    pub radiation_decay_per_in_world_sec: f32,
    /// Per-tick passive decay of toxicity toward 0.
    pub toxicity_decay_per_in_world_sec: f32,
}

impl Default for MedConfig {
    fn default() -> Self {
        Self {
            heal_ticks_bandaged: crate::systems::wounds::DEFAULT_HEAL_TICKS_BANDAGED,
            heal_ticks_stitched: crate::systems::wounds::DEFAULT_HEAL_TICKS_BANDAGED / 2,
            infection_trigger_ticks: 12_000, // 2 in-world hours
            antibiotics_clear_ticks: 3_000,  // 10 in-world min
            necrosis_warning_ticks: 6_000,   // 1 in-world hour
            necrosis_severe_ticks: 6_000,    // +1 in-world hour beyond warning
            overdose_threshold: 75.0,
            withdrawal_threshold: 50.0,
            withdrawal_lift_threshold: 25.0,
            withdrawal_delay_ticks: 24_000, // 4 in-world hours
            tolerance_decay_per_in_world_sec: 25.0 / 300.0, // 25 / in-world hour
            pain_regen_threshold: 50.0,
            contamination_hp_threshold: 80.0,
            contamination_hp_drain_per_sec: 0.1,
            radiation_decay_per_in_world_sec: 0.02,
            toxicity_decay_per_in_world_sec: 0.05,
        }
    }
}

/// Phase 2 ballistics tuning. Loaded once from
/// `crates/simn-sim/content/ballistics.toml` at `Sim::new` /
/// `Sim::load`. Engine code reads these directly — never
/// hardcodes. Not snapshotted; the TOML is the single source of
/// truth, and tuning changes belong in code review, not save
/// files.
#[derive(Resource, Clone, Debug)]
pub struct BallisticsConfig {
    /// Downward acceleration applied to projectile Y velocity per
    /// fixed dt (m/s²).
    pub gravity_mps2: f32,
    /// Lower clamp for the retained-energy range-falloff multiplier.
    /// `damage *= clamp(E_impact / round.reference_energy_j, floor, 1.0)`.
    pub retained_energy_floor: f32,
    /// How much damage reduction each armor class above the round's
    /// penetration class applies to the blocked (blunt) branch.
    pub blocked_damage_ratio_per_class_short: f32,
    /// Muzzle origin forward offset from shooter position (m). Keeps
    /// tracers from spawning inside the player capsule.
    pub muzzle_forward_m: f32,
    /// Muzzle origin up offset (m). Shooter eye height.
    pub muzzle_up_m: f32,
    /// Per-body-part soft-damage multipliers. Head is the skill-reward
    /// multiplier; torso is 1.0; limbs de-emphasized. Modders tune in
    /// TOML; engine never hardcodes.
    pub body_part_soft_multipliers: BodyPartMultipliers,
}

#[derive(Clone, Debug)]
pub struct BodyPartMultipliers {
    pub head: f32,
    pub torso: f32,
    pub left_arm: f32,
    pub right_arm: f32,
    pub left_leg: f32,
    pub right_leg: f32,
}

impl BodyPartMultipliers {
    pub fn get(&self, part: crate::components::BodyPart) -> f32 {
        use crate::components::BodyPart;
        match part {
            BodyPart::Head => self.head,
            BodyPart::Torso => self.torso,
            BodyPart::LeftArm => self.left_arm,
            BodyPart::RightArm => self.right_arm,
            BodyPart::LeftLeg => self.left_leg,
            BodyPart::RightLeg => self.right_leg,
        }
    }
}

impl BallisticsConfig {
    /// Load ballistics tuning from the compile-embedded
    /// `content/ballistics.toml`. Called by `Sim::new` / `Sim::load`.
    /// Cached process-wide via `OnceLock` — every `Sim::new` calls
    /// this and the parse cost was paying ~30+ tests' tax in the
    /// test suite.
    pub fn load() -> Self {
        static CACHE: std::sync::OnceLock<BallisticsConfig> = std::sync::OnceLock::new();
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
        #[derive(serde::Deserialize)]
        struct Raw {
            gravity_mps2: f32,
            retained_energy_floor: f32,
            blocked_damage_ratio_per_class_short: f32,
            muzzle_offset: RawMuzzle,
            body_part_soft_multipliers: RawMults,
        }
        #[derive(serde::Deserialize)]
        struct RawMuzzle {
            forward_m: f32,
            up_m: f32,
        }
        #[derive(serde::Deserialize)]
        struct RawMults {
            head: f32,
            torso: f32,
            left_arm: f32,
            right_arm: f32,
            left_leg: f32,
            right_leg: f32,
        }
        let text = src
            .read_str("combat/ballistics.toml")
            .unwrap_or_else(|e| panic!("ballistics content load failed: {e}"));
        let raw: Raw = toml::from_str(&text).expect("ballistics.toml parses");
        Self {
            gravity_mps2: raw.gravity_mps2,
            retained_energy_floor: raw.retained_energy_floor,
            blocked_damage_ratio_per_class_short: raw.blocked_damage_ratio_per_class_short,
            muzzle_forward_m: raw.muzzle_offset.forward_m,
            muzzle_up_m: raw.muzzle_offset.up_m,
            body_part_soft_multipliers: BodyPartMultipliers {
                head: raw.body_part_soft_multipliers.head,
                torso: raw.body_part_soft_multipliers.torso,
                left_arm: raw.body_part_soft_multipliers.left_arm,
                right_arm: raw.body_part_soft_multipliers.right_arm,
                left_leg: raw.body_part_soft_multipliers.left_leg,
                right_leg: raw.body_part_soft_multipliers.right_leg,
            },
        }
    }
}

/// Inventory weight tuning. Global (every player shares the same
/// cap today); once a skill system lands, `weight_cap_kg` will
/// become a per-player derived stat. Not snapshotted — defaults are
/// recreated per `Sim::new` / `Sim::load`.
///
/// The overweight penalty is consumed by [`crate::systems::regen_stamina`]:
/// if `current_weight > weight_cap_kg`, stamina regen is multiplied
/// by `overweight_regen_mult`. No hard cap on pickup (players can
/// always loot one more item) — only the regen hit, same shape as
/// the low-hunger / high-pain penalties.
#[derive(Resource, Clone, Copy, Debug)]
pub struct InventoryConfig {
    pub weight_cap_kg: f32,
    pub overweight_regen_mult: f32,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            weight_cap_kg: 50.0,
            overweight_regen_mult: 0.5,
        }
    }
}

/// Per-region per-faction desired live NPC count. The spawn driver
/// uses these to decide whether to top up populations on each spawn
/// pass. Seeded from [`crate::resources::RegionControl`] at sim init
/// (primary faction → larger target, contesting → smaller).
/// `by_region: region → faction-name → target count`. Faction is
/// keyed by the registry name string (`"coalition"`) so saves stay valid
/// across registry edits.
#[derive(Resource, Serialize, Deserialize, Clone, Debug, Default)]
pub struct PopulationTargets {
    #[serde(serialize_with = "crate::det_serde::sorted_nested_map")]
    pub by_region: HashMap<RegionId, HashMap<String, u32>>,
}

impl PopulationTargets {
    pub fn target(&self, region: RegionId, faction: &str) -> u32 {
        self.by_region
            .get(&region)
            .and_then(|m| m.get(faction))
            .copied()
            .unwrap_or(0)
    }

    pub fn set(&mut self, region: RegionId, faction: &str, count: u32) {
        self.by_region
            .entry(region)
            .or_default()
            .insert(faction.to_string(), count);
    }
}

/// Per-tick snapshot of NPC positions (and their region + alive
/// state, via Health current) keyed by `NpcId`. Built first thing
/// each tick by `index_npc_positions` so downstream systems
/// (`tick_npc_goals`, `npc_combat`) can look up an aggro target's
/// position without taking another query borrow on the NPC table.
///
/// Also includes per-group centroids so cohesion checks (in
/// `tick_npc_goals`) can read them without iterating the NPC table
/// twice.
#[derive(Resource, Default)]
pub struct NpcPositionIndex {
    pub by_id: HashMap<NpcId, NpcPositionEntry>,
    pub group_centroids: HashMap<u64, GroupCentroid>,
}

#[derive(Clone, Copy, Debug)]
pub struct NpcPositionEntry {
    pub pos: [f32; 3],
    pub region: RegionId,
    pub health: f32,
    pub group: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct GroupCentroid {
    pub pos: [f32; 3],
    pub region: RegionId,
    pub member_count: u32,
}

/// Spatial-hash cell size in meters. Chosen so `within-cell + 4
/// directional neighbor cells` covers every pair inside the default
/// `PerceptionConfig::sight_radius_m` (80m). Making the cell larger
/// would include more false-positive pair comparisons; making it
/// smaller wouldn't reduce pair count further at the current sight
/// radius. Tunable later if sight radius changes.
pub const SPATIAL_CELL_SIZE_M: f32 = 100.0;

/// One entry in an [`NpcSpatialHash`] grid — a cache of the NPC's
/// identity plus the per-tick read-only state the aggro pair scan
/// needs. Copied from the NPC query during `rebuild_spatial_hash`
/// so the scan itself doesn't take another borrow on the NPC table.
#[derive(Clone, Copy, Debug)]
pub struct SpatialEntry {
    pub npc_id: NpcId,
    pub entity: bevy_ecs::entity::Entity,
    pub pos: [f32; 3],
    pub yaw: f32,
    pub faction: crate::faction::registry::FactionId,
    pub region: RegionId,
}

/// Per-region spatial grid. Cells are keyed by integer coordinates
/// `(floor(x / cell_size_m), floor(z / cell_size_m))`; each cell
/// holds indices into the flat `entries` vec. Empty cells aren't
/// stored (HashMap). Y is ignored — the game is effectively 2D for
/// aggro purposes.
#[derive(Default, Clone, Debug)]
pub struct SpatialGrid {
    pub cell_size_m: f32,
    pub entries: Vec<SpatialEntry>,
    pub cells: HashMap<(i32, i32), Vec<u32>>,
}

impl SpatialGrid {
    pub fn new(cell_size_m: f32) -> Self {
        Self {
            cell_size_m,
            entries: Vec::new(),
            cells: HashMap::new(),
        }
    }

    /// Append an entry and slot its index into the cell bucket.
    pub fn insert(&mut self, entry: SpatialEntry) {
        let cell = Self::cell_of(entry.pos, self.cell_size_m);
        let idx = u32::try_from(self.entries.len()).unwrap_or(u32::MAX);
        self.entries.push(entry);
        self.cells.entry(cell).or_default().push(idx);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.cells.clear();
    }

    /// Cell coordinate containing `pos`. Floor-div so negative
    /// coordinates land in the right bucket (`-50m / 100m → -1`, not `0`).
    #[inline]
    pub fn cell_of(pos: [f32; 3], cell_size_m: f32) -> (i32, i32) {
        let cx = (pos[0] / cell_size_m).floor() as i32;
        let cz = (pos[2] / cell_size_m).floor() as i32;
        (cx, cz)
    }
}

/// Per-region spatial hash of all NPCs, rebuilt every tick by
/// `rebuild_spatial_hash`. Consumed by `npc_aggro`'s Pass 2 to avoid
/// the original O(Σ n_r²) pair scan — lookups are now
/// `within-cell + 4 directional neighbors` per NPC, roughly linear
/// in total NPC count.
///
/// Transient (not serialized); rebuilt from ECS state each tick.
/// Per-region grids because cross-region NPC pairs can never see
/// each other (aggro already filters by region).
#[derive(Resource, Default)]
pub struct NpcSpatialHash {
    pub by_region: HashMap<RegionId, SpatialGrid>,
}

impl NpcSpatialHash {
    /// Reset every grid's entries + cells (keep allocations). Called
    /// once per tick by `rebuild_spatial_hash`.
    pub fn clear(&mut self) {
        for grid in self.by_region.values_mut() {
            grid.clear();
        }
    }

    /// Get-or-insert the grid for a region, with the default cell
    /// size. Used by the rebuild system when it first encounters an
    /// NPC in a given region during the tick.
    pub fn grid_mut(&mut self, region: RegionId) -> &mut SpatialGrid {
        self.by_region
            .entry(region)
            .or_insert_with(|| SpatialGrid::new(SPATIAL_CELL_SIZE_M))
    }
}

/// Registry of active guard posts: keyed on `(region, quantized
/// base position)` → `{ group_id, since_tick }`. A squad with a
/// `Guard` objective whose `post_key` is `Some(...)` is "on post"
/// and its objective does not expire by time; it only clears when
/// another squad arrives to `Relieve` them, or when the holding
/// squad dies (pruned on the planner's next pass).
///
/// Transient resource — not serialized. Rebuilt from whatever
/// guard objectives exist after load.
#[derive(Resource, Default)]
pub struct GuardPosts {
    pub by_key: HashMap<(RegionId, [i32; 3]), GuardPostInfo>,
}

#[derive(Clone, Copy, Debug)]
pub struct GuardPostInfo {
    pub group_id: u64,
    pub since_tick: u64,
    /// Faction of the squad holding this post. Used by
    /// `build_relieve` to gate relief to same-faction posts only —
    /// a Coalition squad has no business "relieving" a directorate guard, and
    /// cross-faction post takeover should require combat, not a
    /// peaceful handoff.
    pub faction: crate::faction::registry::FactionId,
}

/// Iteration 5-13 Phase D: a single designer-placed interaction
/// area. Scene-authored via `InteractionAreaMarker3D`, enumerated
/// by the Godot bridge on map load, stored here for the sim's
/// squad planner + future per-objective consumers.
///
/// Transient — like [`GuardPosts`] and [`crate::nav::NavQueries`],
/// rebuilt from the scene on every region attach. No snapshot
/// persistence; deleting a marker drops the area on the next
/// attach.
#[derive(Clone, Debug)]
pub struct InteractionArea {
    /// Stable per-area id. Either the designer-provided `area_id`
    /// from the marker or an auto-derived `scene_path:node_name`
    /// fallback. Must be unique within a region.
    pub id: String,
    /// Free-form descriptor: `"rest"`, `"work"`, `"socialize"`,
    /// `"scavenge"`, `"guard_post"`, `"patrol_node"`,
    /// `"campfire"`, `"workbench"`, or any mod-defined string.
    /// Unknown values are not an error — the squad planner
    /// scores them with low utility.
    pub kind: String,
    /// World-space center (Y included for completeness; the
    /// occupancy + reachability logic uses only XZ).
    pub pos: [f32; 3],
    /// XZ half-size in meters. The "arrival" radius for the area
    /// is roughly `max(extents.x, extents.y)` — see Phase D3.
    pub extents: [f32; 2],
    /// Faction the area is restricted to. `None` = any faction.
    /// When `Some`, `Sim::reserve_interaction_area` rejects
    /// reservations from other-faction NPCs.
    pub faction: Option<crate::faction::registry::FactionId>,
    /// Max NPCs that can hold the area simultaneously.
    pub capacity: u32,
    /// Current occupant count. Incremented on
    /// `reserve_interaction_area`, decremented on
    /// `release_interaction_area`. Tracked here on the
    /// resource (not on the marker) so it survives the
    /// authoring-side scene reload.
    pub occupants: u32,
    /// Free-form per-area metadata from the marker's `tags`
    /// dictionary. Downstream consumers (future Work / Socialize
    /// objective kinds, mod scripts) read it.
    pub tags: HashMap<String, String>,
}

/// Iteration 5-13 Phase D: per-region registry of designer-placed
/// interaction areas. Storage is `by_region`; `by_id` indexes into
/// `by_region` for O(1) reserve / release calls.
///
/// Lifecycle mirrors [`crate::nav::NavQueries`]: cleared + rebuilt
/// by `Sim::attach_region_interaction_areas`. Transient.
#[derive(Resource, Default, Clone, Debug)]
pub struct InteractionAreas {
    pub by_region: HashMap<RegionId, Vec<InteractionArea>>,
    /// `area_id → (region_id, index_into_by_region[region_id])`.
    /// Kept in sync with `by_region` on every attach. Index
    /// validity is invariant: never read this map across an
    /// `attach_region_interaction_areas` call without reading
    /// it again afterward.
    pub by_id: HashMap<String, (RegionId, usize)>,
    /// Iteration 5-13 Phase D3. Set of `(npc_id, area_id)` pairs
    /// that have already emitted `InteractionStarted`. Used by
    /// `tick_npc_goals` to dedupe per-tick Started spam, and by
    /// `squad_planner` to fire `InteractionEnded` for every NPC
    /// that was actively interacting at the moment a squad's
    /// `Rest` objective gets replaced. Cleared along with the
    /// `by_region` set on every `attach_region_interaction_areas`
    /// (no cross-attach memory).
    pub started: HashMap<String, Vec<u64>>,
}

impl InteractionAreas {
    /// Iteration 5-13 Phase D3. Internal (sim-private) reserve
    /// that the squad planner calls after it has already
    /// validated faction + capacity in its candidate filter.
    /// Pre-filtered callers MUST pass an id that exists; the
    /// method does the bookkeeping (`occupants += 1`) without
    /// re-running the faction check. External callers should
    /// use `Sim::reserve_interaction_area` which performs the
    /// full validation.
    ///
    /// Returns the same bool semantics as the full check (true
    /// = reserved, false = capacity exhausted / id missing) so
    /// the planner can detect a race where another system
    /// reserved between filter and apply.
    pub fn reserve_internal(&mut self, area_id: &str) -> bool {
        let Some(&(region, idx)) = self.by_id.get(area_id) else {
            return false;
        };
        let Some(areas) = self.by_region.get_mut(&region) else {
            return false;
        };
        let Some(area) = areas.get_mut(idx) else {
            return false;
        };
        if area.occupants >= area.capacity {
            return false;
        }
        area.occupants += 1;
        true
    }

    /// Iteration 5-13 Phase D3. Internal counterpart to
    /// [`Self::reserve_internal`]. Saturating decrement; safe
    /// to call on an unknown / already-zeroed area (returns
    /// `false` instead of panicking).
    pub fn release_internal(&mut self, area_id: &str) -> bool {
        let Some(&(region, idx)) = self.by_id.get(area_id) else {
            return false;
        };
        let Some(areas) = self.by_region.get_mut(&region) else {
            return false;
        };
        let Some(area) = areas.get_mut(idx) else {
            return false;
        };
        if area.occupants > 0 {
            area.occupants -= 1;
        }
        true
    }

    /// Iteration 5-13 Phase D3. Return `true` if `(npc_id, area_id)`
    /// is already marked as having emitted `InteractionStarted`.
    /// `tick_npc_goals` uses this to dedupe per-tick Started spam
    /// — only the first tick of arrival emits.
    pub fn is_started(&self, npc_id: u64, area_id: &str) -> bool {
        self.started
            .get(area_id)
            .is_some_and(|v| v.contains(&npc_id))
    }

    /// Iteration 5-13 Phase D3. Mark `(npc_id, area_id)` as
    /// having emitted `InteractionStarted`. Caller (the goals
    /// system) should also push the event onto the world-event
    /// queue.
    pub fn mark_started(&mut self, npc_id: u64, area_id: &str) {
        let entry = self.started.entry(area_id.to_string()).or_default();
        if !entry.contains(&npc_id) {
            entry.push(npc_id);
        }
    }

    /// Iteration 5-13 Phase D3. Drain the list of NPCs that
    /// emitted Started for `area_id`. Returns them so the
    /// caller can fire matching `InteractionEnded` events.
    /// Used by `squad_planner` on `Rest` objective swap.
    pub fn drain_started_for_area(&mut self, area_id: &str) -> Vec<u64> {
        self.started.remove(area_id).unwrap_or_default()
    }
}

// ── Activity Points (Phase 2 smart terrain) ─────────────────────

/// Typed activity a designer places at a location. NPCs compete for
/// slots based on faction, distance, personality, and capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Deserialize)]
pub enum ActivityKind {
    GuardStatic,
    GuardPerimeter,
    PatrolWaypoint,
    RestSpot,
    Lookout,
    Campfire,
    Workbench,
    Stash,
    SniperNest,
    AmbushPoint,
}

impl ActivityKind {
    pub fn is_guard(&self) -> bool {
        crate::poi::activity_is_guard(*self)
    }

    pub fn is_rest(&self) -> bool {
        crate::poi::activity_is_rest(*self)
    }
}

/// A single designer-placed activity point. Scene-authored via
/// `ActivityPointMarker3D`, registered on region attach. Transient.
#[derive(Clone, Debug)]
pub struct ActivityPoint {
    pub id: u64,
    pub kind: ActivityKind,
    pub pos: [f32; 3],
    pub facing_yaw: f32,
    pub faction: Option<crate::faction::registry::FactionId>,
    pub radius_m: f32,
    pub capacity: u8,
    pub priority: i8,
    pub loop_id: Option<String>,
    pub occupants: Vec<crate::components::NpcId>,
    pub claimed_by_groups: Vec<u64>,
}

impl ActivityPoint {
    pub fn has_capacity(&self) -> bool {
        self.claimed_by_groups.len() < self.capacity as usize
    }

    pub fn is_claimed_by(&self, group_id: u64) -> bool {
        self.claimed_by_groups.contains(&group_id)
    }
}

/// A designer-drawn patrol route. Each waypoint is a world-space
/// position; NPCs walk between them in order (looping or
/// out-and-back).
#[derive(Clone, Debug)]
pub struct PatrolRoute {
    pub id: String,
    pub waypoints: Vec<[f32; 3]>,
    pub faction: Option<crate::faction::registry::FactionId>,
    pub is_loop: bool,
    pub priority: i8,
    pub claimed_by_group: Option<u64>,
}

/// Per-region registry of activity points and patrol routes.
/// Transient — rebuilt from the scene on every region attach.
#[derive(Resource, Default)]
pub struct ActivityPoints {
    pub by_region: HashMap<RegionId, Vec<ActivityPoint>>,
    pub routes_by_region: HashMap<RegionId, Vec<PatrolRoute>>,
    next_id: u64,
}

impl ActivityPoints {
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn clear_region(&mut self, region: RegionId) {
        self.by_region.remove(&region);
        self.routes_by_region.remove(&region);
    }

    pub fn points_in_region(&self, region: RegionId) -> &[ActivityPoint] {
        self.by_region
            .get(&region)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn routes_in_region(&self, region: RegionId) -> &[PatrolRoute] {
        self.routes_by_region
            .get(&region)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn release_group(&mut self, region: RegionId, group_id: u64) {
        if let Some(points) = self.by_region.get_mut(&region) {
            for pt in points.iter_mut() {
                if pt.claimed_by_groups.contains(&group_id) {
                    pt.claimed_by_groups.retain(|&g| g != group_id);
                    pt.occupants.clear();
                }
            }
        }
        if let Some(routes) = self.routes_by_region.get_mut(&region) {
            for route in routes.iter_mut() {
                if route.claimed_by_group == Some(group_id) {
                    route.claimed_by_group = None;
                }
            }
        }
    }
}

// ── Authored Spawn Points (Phase 2G) ────────────────────────────

/// A designer-placed spawn point with faction, rate, and squad-size
/// configuration. Scene-authored via `SpawnPointMarker3D`. Transient.
#[derive(Clone, Debug)]
pub struct AuthoredSpawnPoint {
    pub id: u64,
    pub region: RegionId,
    pub pos: [f32; 3],
    pub faction: crate::faction::registry::FactionId,
    pub spawn_rate_per_min: f32,
    pub max_concurrent: u8,
    pub squad_size: (u8, u8),
    pub spread_radius_m: f32,
    pub loadout_tier: u8,
    pub enabled: bool,
    pub active_squads: Vec<u64>,
    pub last_spawn_tick: u64,
    pub initial_delay_ticks: u64,
}

impl AuthoredSpawnPoint {
    pub fn has_capacity(&self) -> bool {
        self.active_squads.len() < self.max_concurrent as usize
    }

    pub fn ticks_between_spawns(&self, dt_ms: u32) -> u64 {
        if self.spawn_rate_per_min <= 0.0 {
            return u64::MAX; // one-shot
        }
        let ticks_per_min = 60_000.0 / dt_ms as f32;
        (ticks_per_min / self.spawn_rate_per_min).ceil() as u64
    }
}

/// Per-region registry of authored spawn points. Transient.
#[derive(Resource, Default)]
pub struct AuthoredSpawnPoints {
    pub by_region: HashMap<RegionId, Vec<AuthoredSpawnPoint>>,
    next_id: u64,
}

impl AuthoredSpawnPoints {
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn clear_region(&mut self, region: RegionId) {
        self.by_region.remove(&region);
    }

    pub fn remove_squad(&mut self, group_id: u64) {
        for points in self.by_region.values_mut() {
            for pt in points.iter_mut() {
                pt.active_squads.retain(|&g| g != group_id);
            }
        }
    }
}

/// Per-squad objective state. Set/refreshed by `squad_planner`,
/// consumed by `tick_npc_goals` for any NPC with a `Group`.
/// Transient — not serialized; the planner re-seeds on startup.
#[derive(Resource, Default)]
pub struct SquadObjectives {
    pub by_group: HashMap<u64, SquadObjectiveState>,
}

#[derive(Clone, Debug)]
pub struct SquadObjectiveState {
    pub objective: SquadObjective,
    pub set_at_tick: u64,
    pub recently_visited: std::collections::VecDeque<[i32; 3]>,
    /// One-shot "walk out from spawn" target. Seeded on the
    /// squad's first planner-visit and cleared once the centroid
    /// reaches it. While `Some`, the executor steers the squad
    /// toward this point instead of the active objective's
    /// nominal target, and the planner won't re-roll the
    /// objective. Stops NPCs from picking `Guard` / `Rest` /
    /// `Patrol` at the exact base they spawned at — they have to
    /// walk somewhere first.
    pub disperse_target: Option<[f32; 3]>,
    /// Centroid + tick of the most recent "made meaningful
    /// progress" planner observation. Used by stuck-detection:
    /// a squad with a movement-oriented objective whose centroid
    /// hasn't moved `STUCK_PROGRESS_M` in `STUCK_TICKS` is
    /// presumed stuck (unreachable target, packed in with peers,
    /// wedged on geometry) and forcibly expired so it re-rolls.
    /// `None` = no observation yet; the next planner pass
    /// initializes both.
    pub last_progress_pos: Option<[f32; 3]>,
    pub last_progress_tick: u64,
    /// Live drift target for Wander objectives. Without this,
    /// `squad_target` returned the centroid (where members
    /// already were) so Wander squads stood still. Refreshed by
    /// the planner whenever the centroid arrives within
    /// `WANDER_DRIFT_ARRIVE_M` of the current target, or every
    /// `WANDER_DRIFT_REROLL_TICKS` even mid-leg. Cleared on
    /// objective swap. Only consulted when
    /// `objective` is `Wander`.
    pub wander_drift_target: Option<[f32; 3]>,
    /// When this kind was force-expired by stuck-detection.
    /// Used as a one-pick ban so the immediate re-roll doesn't
    /// pick the same kind that just got the squad stuck. Cleared
    /// after one successful pick. `None` when no recent expiry.
    pub last_stuck_kind: Option<crate::resources::SquadObjectiveKindTag>,
    /// Sim tick before which cohesion-break detection is
    /// suppressed for this squad. Set after a Regroup naturally
    /// times out without the members gathering — the squad has
    /// an unreachable outlier, and re-triggering Regroup every
    /// minute just thrashes between Regroup and movement. With
    /// break detection disabled the squad commits to its
    /// movement objective and lets the straggler fend for
    /// themselves. Re-enabled automatically once the tick
    /// passes; if cohesion is still broken at that point, the
    /// cycle starts over.
    pub cohesion_break_disabled_until: u64,
    /// Tick at which the squad first arrived within
    /// `ARRIVED_AT_TARGET_M` of its current objective's
    /// position-target. Set by stuck-detection when the squad
    /// transitions from "in transit" to "at target", cleared
    /// when they leave or the objective changes. Used by the
    /// planner to cap arrival-dwell for objectives that would
    /// otherwise freeze a squad at the target for the entire
    /// dwell window (e.g., Investigate's 4-min timer with NPCs
    /// standing still).
    pub arrived_at_tick: Option<u64>,
    /// Tick at which the squad most recently exited a `Regroup`
    /// objective. Used by `cohesion_pass` to suppress an
    /// immediate re-Regroup right after exiting one (the squad
    /// needs grace to actually commit to the next objective and
    /// traverse some distance before re-evaluating cohesion).
    /// `None` for freshly-seeded squads — they're allowed to
    /// Regroup immediately if their members are spread out.
    pub last_regroup_exit_tick: Option<u64>,
    /// Heading (radians) of the last Wander drift leg. New drift
    /// targets are biased toward this heading ±90° for smoother
    /// movement. `None` on first drift or after objective swap.
    pub last_drift_heading: Option<f32>,
}

/// Lightweight discriminant of `SquadObjective` for cross-pick
/// state (e.g. "this squad just got stuck on Wander, don't
/// re-pick Wander this turn"). Mirrors the variants of
/// `SquadObjective` but carries no payload — used as a small,
/// Copy-able tag in fields that need to remember "what kind".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SquadObjectiveKindTag {
    Patrol,
    Guard,
    Rest,
    Investigate,
    Explore,
    Relieve,
    Wander,
    Regroup,
}

#[derive(Clone, Debug)]
pub enum SquadObjective {
    /// Move sequentially through a list of base positions.
    Patrol {
        route: Vec<[f32; 3]>,
        current_idx: usize,
        expires_at: u64,
    },
    /// Hold position at a specific base. If `post_key` is `Some(..)`
    /// the squad is a registered guard post and their objective
    /// doesn't expire by time — only by being relieved (another
    /// squad's `Relieve` arriving) or by the squad dying. Transient
    /// guard stints (picked when no post is wanted) leave
    /// `post_key` as `None` and still honor `expires_at`.
    Guard {
        base_pos: [f32; 3],
        expires_at: u64,
        post_key: Option<(RegionId, [i32; 3])>,
    },
    /// Chill at a same-faction base (preferably a Safehouse/Outpost):
    /// wander to it, then idle nearby. Distinct from `Guard`
    /// semantically — this is "off-duty downtime" rather than
    /// defensive posture. Aggro still preempts.
    ///
    /// Iteration 5-13 Phase D3: `area_id` is `Some(id)` when the
    /// planner picked a designer-placed `InteractionArea` (kind
    /// `"rest"`) over a generic base position. The planner
    /// reserves the area on assignment and releases on objective
    /// change / death / cohesion override. `None` means the
    /// legacy base-position fallback path.
    Rest {
        base_pos: [f32; 3],
        expires_at: u64,
        area_id: Option<String>,
    },
    /// Move to an open-world point and dwell briefly.
    Investigate { target: [f32; 3], expires_at: u64 },
    /// Walk to an existing guard post and take over. When any
    /// squadmate arrives within the relief radius, the planner
    /// swaps post ownership: the old holder's Guard objective
    /// clears (they re-roll next pass) and this squad becomes the
    /// new `Guard` post-holder.
    Relieve {
        post_key: (RegionId, [i32; 3]),
        dest_pos: [f32; 3],
        expires_at: u64,
    },
    /// Head to a portal that leads to `dest_region`. When the squad
    /// arrives at `portal_pos`, `npc_portal_cross` relocates them
    /// to the destination region's reciprocal portal. Gives squads
    /// an actual "explore the next map" behavior instead of silent
    /// teleportation.
    Explore {
        dest_region: RegionId,
        portal_pos: [f32; 3],
        expires_at: u64,
    },
    /// Long-range wander; planner re-rolls on expiry.
    Wander { expires_at: u64 },
    /// Cohesion override — head toward a fixed rally point until
    /// expiry. The rally is captured at the moment Regroup activates
    /// (typically the squad centroid at that instant) and stays
    /// constant across ticks. Storing the position here instead of
    /// re-reading `index.group_centroids` each tick prevents the
    /// classic "swirl" feedback loop where moving members drag the
    /// centroid with them, which the members then chase — they
    /// spiral inward into a single point.
    Regroup {
        rally_pos: [f32; 3],
        expires_at: u64,
    },
}

impl SquadObjective {
    pub fn expires_at(&self) -> u64 {
        match self {
            SquadObjective::Patrol { expires_at, .. } => *expires_at,
            SquadObjective::Guard { expires_at, .. } => *expires_at,
            SquadObjective::Rest { expires_at, .. } => *expires_at,
            SquadObjective::Investigate { expires_at, .. } => *expires_at,
            SquadObjective::Explore { expires_at, .. } => *expires_at,
            SquadObjective::Relieve { expires_at, .. } => *expires_at,
            SquadObjective::Wander { expires_at } => *expires_at,
            SquadObjective::Regroup { expires_at, .. } => *expires_at,
        }
    }
}

/// Buffer of deltas produced by ECS systems during a tick. The
/// `Sim::tick` orchestrator drains this after running the schedule
/// and appends each entry to the journal — systems don't own the
/// journal directly. Empty between ticks.
#[derive(Resource, Default)]
pub struct PendingDeltas {
    pub events: Vec<WorldDelta>,
}

impl PendingDeltas {
    pub fn push(&mut self, delta: WorldDelta) {
        self.events.push(delta);
    }

    pub fn drain(&mut self) -> Vec<WorldDelta> {
        std::mem::take(&mut self.events)
    }
}

/// Marker resource present when a `Sim` is running in **mirror mode**
/// — a client-side read-only reflection of a host's authoritative sim.
///
/// In mirror mode, mutation methods on `Sim` (move_player, apply_bandage,
/// grant_item, …) buffer their deltas but don't append to any journal
/// (there is no journal). Host-sent deltas arrive via
/// `apply_external_delta` and mutate ECS directly. NPC-mutating systems
/// (`spawn_npcs`, `tick_npc_goals`, `npc_combat`, …) are absent from
/// the mirror schedule — they'd diverge from host state because their
/// RNG seeds mix in `Entity::to_bits()` which isn't stable across
/// instances.
///
/// Authoritative sims (`Solo` / `Host`) don't insert this resource.
#[derive(Resource, Default, Debug)]
pub struct MirrorMode;

/// Kills accrued by attacker NPCs during the current tick. `npc_combat`
/// pushes a credit when a hit brings the target's `vital_min` to
/// zero; `apply_kill_credits` drains the resource right after,
/// mutating each killer's `NpcCharacter::kills` and refreshing rank.
/// Split into two systems so npc_combat doesn't fight bevy's
/// query-aliasing rules trying to mutate one NPC while iterating
/// another.
#[derive(Resource, Default, Debug)]
pub struct PendingKillCredits {
    pub credits: HashMap<NpcId, u32>,
}

impl PendingKillCredits {
    pub fn credit(&mut self, killer: NpcId) {
        *self.credits.entry(killer).or_insert(0) += 1;
    }

    pub fn drain(&mut self) -> HashMap<NpcId, u32> {
        std::mem::take(&mut self.credits)
    }
}

/// Phase 4A v1 — visible projectile spawns deferred from
/// `npc_combat` to the end-of-tick `Sim::npc_fire_projectile`
/// drain. Bevy systems can't easily call back into `Sim` methods
/// (the &mut World aliasing rules fight), so npc_combat writes
/// shot intent here and `Sim::tick` reads it off and spawns the
/// projectile entities + journals `ProjectileSpawned` deltas.
///
/// Each entry corresponds to a fire decision in npc_combat. The
/// dice-based damage application still runs at fire time in
/// npc_combat (4A v1 keeps the damage path intact); the
/// projectile is a cosmetic tracer until 4A v2 migrates damage
/// onto the projectile-hit branch.
#[derive(Resource, Default, Debug)]
pub struct PendingNpcShots {
    pub shots: Vec<NpcShotIntent>,
}

#[derive(Clone, Debug)]
pub struct NpcShotIntent {
    pub shooter_id: NpcId,
    pub shooter_pos: [f32; 3],
    pub shooter_region: RegionId,
    pub target_pos: [f32; 3],
    pub accuracy: u8,
    /// Phase 4B v1 — round to fire. Derived by `npc_combat` via
    /// `default_npc_round_for_faction(shooter.faction)`. Keeps
    /// the `Sim::npc_fire_projectile` helper data-agnostic.
    pub round_id: crate::items::ItemId,
}

impl PendingNpcShots {
    pub fn push(&mut self, intent: NpcShotIntent) {
        self.shots.push(intent);
    }

    pub fn drain(&mut self) -> Vec<NpcShotIntent> {
        std::mem::take(&mut self.shots)
    }
}

/// Per-region terrain heightmaps. Populated by the engine bridge once
/// a region's map asset is available; regions without an entry skip
/// Y-clamping (legacy flat behavior).
///
/// Transient — not serialized. Re-attached on every sim startup.
#[derive(Resource, Default)]
pub struct TerrainMaps {
    by_region: HashMap<RegionId, simn_terrain::Heightmap>,
}

impl TerrainMaps {
    pub fn attach(&mut self, region: RegionId, heightmap: simn_terrain::Heightmap) {
        self.by_region.insert(region, heightmap);
    }

    pub fn detach(&mut self, region: RegionId) {
        self.by_region.remove(&region);
    }

    pub fn has(&self, region: RegionId) -> bool {
        self.by_region.contains_key(&region)
    }

    /// Sample ground elevation at world-local `(x, z)`. Returns `None`
    /// if the region has no attached terrain.
    ///
    /// World coordinates are centered on the region origin (e.g. a
    /// 5 km map spans `[-2500, 2500]` on each axis). The heightmap
    /// internally uses NW-corner origin, so this method offsets by
    /// half-extent before sampling.
    pub fn ground_at(&self, region: RegionId, world_x: f32, world_z: f32) -> Option<f32> {
        self.by_region.get(&region).map(|hm| {
            let [w, h] = hm.extent_m();
            hm.sample(world_x + w * 0.5, world_z + h * 0.5)
        })
    }
}
