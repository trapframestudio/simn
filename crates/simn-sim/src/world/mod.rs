//! `Sim` — the public face of the simulation.
//!
//! Wraps a `bevy_ecs::World`, owns the journal writer and save paths,
//! and exposes the operations the game layer calls: spawn/despawn
//! players, move players, move them between regions, read their
//! state. Each mutating call writes both the ECS state and a
//! [`WorldDelta`] to the journal so the two are always consistent.
//!
//! The tick loop is minimal for this slice: advance the clock, flush
//! the journal, and periodically roll a snapshot. Systems that do
//! real work (NPC simulation, tier hand-off) land here later.

use anyhow::{Context, Result};
use bevy_ecs::schedule::Schedule;
use std::path::Path;

use bevy_ecs::prelude::World;

use crate::chronicle::LifeChronicle;
use crate::components::{
    ActiveEffect, BaseKind, BodyParts, Contamination, DrugKind, Health, NpcId, Pain, Stamina,
    SurvivalStats, Wound, WoundId,
};
use crate::delta::WorldDelta;
use crate::persistence::{
    read_journal, read_snapshot, write_snapshot, write_snapshot_to_vec, JournalWriter,
};
use crate::region::{RegionGraph, RegionId};
use crate::resources::{
    ActiveRegions, NpcIdCounter, NpcPositionIndex, NpcSpatialHash, PendingDeltas,
    PopulationTargets, SavePaths, SimClock, SquadObjectives, WeatherState, WorldTime,
};
use crate::world_seed::{seed_random_world_content, DEFAULT_SEED};

pub(crate) mod containers;
mod debug;
mod drugs;
pub(crate) mod hitbox;
mod inventory;
mod npc_view;
mod persistence;
mod player;
mod population;
mod projectiles;
mod registration;
mod replication;
mod survival;
mod tick;
pub mod weapons;
mod wounds;

// Re-export the free-function persistence surface so the rest of the
// `world` module (and `pub(crate)` consumers like `npc_death_check`)
// can keep calling `find_npc_in(...)`, `apply_delta(...)`, etc. without
// a path prefix. `persistence.rs` owns the bodies.
pub(crate) use persistence::find_npc_in;
use persistence::{apply_delta, serialize_world, spawn_serialized};

pub(crate) use inventory::merge_item_stack;
pub use inventory::{CraftabilityReport, InputStatus};
pub use survival::{food_profile, water_profile, ConsumeProfile};

/// Default: roll a snapshot every 600 ticks (~30s at 20Hz).
pub const SNAPSHOT_INTERVAL_TICKS: u64 = 600;

/// Online-near radius for the `SimView.npcs_by_region` cache: NPCs
/// within this distance of an active-region player are exposed to
/// the renderer; ones outside still simulate (active-region tier
/// filter only excludes offline regions) but aren't surfaced to
/// per-frame `npcs_near` reads. 800 m gives the renderer a wide
/// horizon for immersion (NPCs visible in the distance), staying
/// generous vs `NPC_DRAW_DISTANCE_M = 300` so the renderer's draw
/// gate is the actual cap — this radius is the prefilter that
/// keeps `npcs_by_region` cheap on the worker side.
pub const ONLINE_NEAR_RADIUS_M: f32 = 800.0;

// goal_tag, derive_goal_tag, squad_objective_tag → world/npc_view.rs

/// A read-only view of one player's authoritative state. Returned by
/// [`Sim::player_view`] for the engine-side bridge.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerView {
    pub steam_id: u64,
    pub region: RegionId,
    pub pos: [f32; 3],
    pub yaw: f32,
    /// Aggregate health, mirrored from `min(body_parts.head, body_parts.torso)`.
    /// Kept on the view so existing death-gate consumers don't need to
    /// learn body parts.
    pub health: Health,
    pub stamina: Stamina,
    pub body_parts: BodyParts,
    pub survival: SurvivalStats,
    /// All active wounds. Empty for an uninjured player. Each entry
    /// pairs the stable `WoundId` with the wound's current state.
    pub wounds: Vec<(WoundId, Wound)>,
    /// Derived per-tick pain meter, `[0, 100]`.
    pub pain: Pain,
    /// Radiation + Toxicity meters.
    pub contamination: Contamination,
    /// All in-flight effects (drugs + statuses).
    pub active_effects: Vec<ActiveEffect>,
    /// Per-drug tolerance entries.
    pub drug_tolerance: Vec<(DrugKind, f32)>,
}

/// Outcome of a `Sim::apply_drug` call. Reported for HUD/UI feedback
/// (e.g. "you overdosed; effect ignored, take a breath") and for tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrugOutcome {
    /// The drug applied normally — effect (and crash, if applicable)
    /// scheduled, tolerance bumped.
    Effect,
    /// Tolerance was past `MedConfig::overdose_threshold` AND another
    /// dose of the same drug was active. The active dose is unchanged;
    /// the player gets an `OverdoseDisorientation` effect on top, and
    /// tolerance still bumps. No second active dose is added.
    Overdose,
}

/// Read-only view of one NPC. Returned by [`Sim::npcs_in_region`] /
/// [`Sim::each_npc`] for the engine-side bridge.
#[derive(Debug, Clone, PartialEq)]
pub struct NpcView {
    pub id: NpcId,
    pub faction: crate::faction::registry::FactionId,
    pub region: RegionId,
    pub pos: [f32; 3],
    pub yaw: f32,
    pub health: Health,
    /// Per-part HP pools. `None` only on NPCs loaded from snapshots
    /// written before NPCs gained `BodyParts`; normal runs always have
    /// this component (attached at spawn / on replay).
    pub body_parts: Option<BodyParts>,
    /// Active wounds on the NPC (bleed/infection/treatment state).
    /// Cheap copy — typical N is 0–3. The Godot bridge surfaces the
    /// full list in the `npcs_in_region` view dict so debug labels
    /// and future medic UIs can inspect wounds per NPC without a
    /// second bridge call.
    pub wounds: Vec<(WoundId, Wound)>,
    /// Short tag for the current FSM state: `"idle" | "move" | "rest" | "pursue"`.
    pub goal: &'static str,
    /// 0 when solo.
    pub group_id: u64,
    /// 0 when not aggroed.
    pub aggro_target: u64,
    /// Procedural display name "First Last", rolled from the global
    /// multicultural pool. Empty string for NPCs loaded from snapshots
    /// written before names landed.
    pub name: String,
    /// Cultural / ethnic-origin bucket the name was rolled from.
    /// `None` on legacy NPCs.
    pub nationality: Option<crate::names::NationalityBucket>,
    /// Tactical combat stance. `None` when not in combat.
    pub combat_stance: Option<&'static str>,
    /// Squad combat role. `None` when not in combat.
    pub combat_role: Option<&'static str>,
    /// Universal STALKER-style threat tier. `None` on legacy NPCs.
    pub rank: Option<crate::components::NpcRank>,
    /// Visual pose during a dwell ("standing" | "sitting" | "crouching"),
    /// or `None` when the NPC is not in a dwell state. Renderers use
    /// this to pick an idle animation; sim-side movement is unaffected.
    pub dwell_pose: Option<&'static str>,
    /// Source tag for the active goal — which arbiter lane produced
    /// the current `goal` ("aggro_solo" / "aggro_squad" / "squad_obj"
    /// / "survival" / "blackboard_*" / "personality" / "idle"). Useful
    /// in the debug HUD so we can tell whether an NPC is on-task vs
    /// reacting to a distraction.
    pub goal_source: &'static str,
    /// Numeric priority the arbiter resolved at, including hysteresis
    /// and personality biases. Higher = more committed. Lets the
    /// debug HUD show "why isn't this getting preempted?" at a glance.
    pub goal_priority: u8,
}

/// Read-only view of one base. Returned by [`Sim::bases_in_region`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaseView {
    pub region: RegionId,
    pub pos: [f32; 3],
    pub kind: BaseKind,
    pub faction: crate::faction::registry::FactionId,
    pub health: Health,
}

/// Per-segment tick duration breakdown. Recorded each tick and
/// surfaced via [`Sim::recent_tick_perf`] so the bridge can show
/// which system group is eating the budget without needing
/// `NSPH_VERBOSE=1`.
#[derive(Clone, Copy, Default, Debug)]
pub struct TickSegments {
    pub player: std::time::Duration,
    /// Sub-segment of perception: `index_npc_positions` +
    /// `rebuild_spatial_hash` + `drain_world_events` +
    /// `clear_los_cache` + `sweep_squad_blackboards`. O(N) scans.
    pub npc_index: std::time::Duration,
    /// Finer breakdown of the npc_index segment — each is one system.
    /// Filled from the thread-local profile slots in
    /// [`crate::systems::drain_perception_slots`].
    pub clear_los: std::time::Duration,
    pub sweep_bb: std::time::Duration,
    pub position_index: std::time::Duration,
    pub drain_events: std::time::Duration,
    pub spatial_hash: std::time::Duration,
    /// TEMP — count of WorldEvents drained this tick. Bucketing
    /// hint for the drain_world_events bisect.
    pub event_count: u32,
    /// Sub-segment of perception: `sweep_threats`. Per-NPC threat
    /// scoring over the threat board.
    pub npc_threats: std::time::Duration,
    /// Sub-segment of perception: `npc_aggro` + `apply_threat_priority`.
    /// The dense pair-scan.
    pub npc_aggro: std::time::Duration,
    /// Convenience sum of the three perception sub-segments. Same
    /// shape as before the split so existing dashboards keep working.
    pub npc_perception: std::time::Duration,
    pub npc_planning: std::time::Duration,
    /// Planning sub-systems (recorded via `ProfGuard` on drop).
    pub squad_planner: std::time::Duration,
    pub goal_arbitration: std::time::Duration,
    pub tick_npc_goals: std::time::Duration,
    pub npc_combat: std::time::Duration,
    pub npc_lifecycle: std::time::Duration,
    pub offline_loot: std::time::Duration,
    pub total: std::time::Duration,
}

/// Rolling avg + p99 per segment. Returned by
/// [`Sim::recent_tick_perf`] and bridged to GDScript via
/// `SimHost::tick_perf()`.
#[derive(Clone, Copy, Default, Debug)]
pub struct TickPerfReport {
    pub samples: usize,
    pub avg_total_ms: f32,
    pub p99_total_ms: f32,
    pub avg_player_ms: f32,
    pub p99_player_ms: f32,
    pub avg_npc_index_ms: f32,
    pub p99_npc_index_ms: f32,
    pub avg_clear_los_ms: f32,
    pub p99_clear_los_ms: f32,
    pub avg_sweep_bb_ms: f32,
    pub p99_sweep_bb_ms: f32,
    pub avg_position_index_ms: f32,
    pub p99_position_index_ms: f32,
    pub avg_drain_events_ms: f32,
    pub p99_drain_events_ms: f32,
    pub avg_spatial_hash_ms: f32,
    pub p99_spatial_hash_ms: f32,
    pub avg_event_count: f32,
    pub max_event_count: u32,
    pub avg_npc_threats_ms: f32,
    pub p99_npc_threats_ms: f32,
    pub avg_npc_aggro_ms: f32,
    pub p99_npc_aggro_ms: f32,
    pub avg_npc_perception_ms: f32,
    pub p99_npc_perception_ms: f32,
    pub avg_npc_planning_ms: f32,
    pub p99_npc_planning_ms: f32,
    pub avg_squad_planner_ms: f32,
    pub p99_squad_planner_ms: f32,
    pub avg_goal_arbitration_ms: f32,
    pub p99_goal_arbitration_ms: f32,
    pub avg_tick_npc_goals_ms: f32,
    pub p99_tick_npc_goals_ms: f32,
    pub avg_npc_combat_ms: f32,
    pub p99_npc_combat_ms: f32,
    pub avg_npc_lifecycle_ms: f32,
    pub p99_npc_lifecycle_ms: f32,
    pub avg_offline_loot_ms: f32,
    pub p99_offline_loot_ms: f32,
}

pub struct Sim {
    world: World,
    /// Player systems (clock, world time, weather, survival, wounds,
    /// crafting). Cheap and ordering-independent of NPC pipeline.
    schedule_player: Schedule,
    /// Perception sub-group 1: position index, spatial hash rebuild,
    /// world-event drain, LOS-cache clear, squad-blackboard sweep.
    /// Linear in NPC count.
    schedule_npc_index: Schedule,
    /// Perception sub-group 2: `sweep_threats` (per-NPC threat
    /// board scoring).
    schedule_npc_threats: Schedule,
    /// Perception sub-group 3: `npc_aggro` (dense pair-scan) +
    /// `apply_threat_priority`. The hot path at high NPC density.
    schedule_npc_aggro: Schedule,
    /// NPC planning + combat: squad planner, goal arbitration,
    /// pathfind/movement, combat resolution.
    schedule_npc_planning: Schedule,
    /// NPC lifecycle + replication broadcast: kill credits, death
    /// check, regroup, portal cross, age, spawn, clamp Y,
    /// broadcast.
    schedule_npc_lifecycle: Schedule,
    /// Offline tier (2 Hz heartbeat) + loot restock cadence.
    schedule_offline_loot: Schedule,
    /// `Some` for authoritative sims (solo / host); `None` for mirror
    /// sims (coop clients) which have no disk persistence.
    journal: Option<JournalWriter>,
    /// `Some` paired with `journal` above.
    save_paths: Option<SavePaths>,
    snapshot_interval: u64,
    /// Deltas produced during the most recent `tick()` call. The
    /// `SimHost` wrapper drains this after each tick via
    /// [`Sim::drain_tick_deltas`] and broadcasts over the network
    /// (host role) or ignores (solo / mirror).
    last_tick_deltas: Vec<WorldDelta>,
    /// Two-slot ring of render-facing snapshots. `[0]` is `prev`,
    /// `[1]` is `curr` (most recently published). Built at the end
    /// of every `tick()`. Renderer reads via [`Self::snapshot_pair`]
    /// and lerps between the two for smooth motion at any frame
    /// rate. Plan-doc reference:
    /// `docs/book/src/planning/threaded-sim-plan.md` §4. PR A
    /// builds the data path; PR B wires GDScript consumption;
    /// PR C moves the sim to its own thread, at which point this
    /// ring becomes the cross-thread handoff.
    snapshot_ring: [Option<crate::snapshot::SimSnapshot>; 2],
    /// Background thread that owns the periodic snapshot disk write.
    /// `serialize_world` stays on the worker (needs `&mut World`),
    /// but the atomic-tmp-then-rename disk I/O for snapshot bytes
    /// goes through this writer so a multi-MB snapshot doesn't
    /// stall the tick loop for 200-1000 ms every 30 s of sim time.
    /// `None` on mirror sims (no disk persistence).
    snapshot_writer: Option<crate::persistence::SnapshotWriter>,
    /// Rolling window of recent per-segment tick durations. Records
    /// the last `TICK_PERF_WINDOW` samples. Cost: one `VecDeque`
    /// push per tick (< 1 µs); deliberately unconditional so live
    /// debugging works without a process restart.
    tick_perf_history: std::collections::VecDeque<TickSegments>,
}

/// Maximum number of recent tick durations retained for
/// [`Sim::recent_tick_perf`]. ~10 seconds at 20 Hz.
pub const TICK_PERF_WINDOW: usize = 200;

impl Sim {
    /// Load an existing save if present, otherwise create a fresh sim
    /// with the given default region graph. If a snapshot exists but
    /// fails to load (version mismatch, corruption, schema changes
    /// during dev), the stale files are removed and we start fresh
    /// rather than erroring up to the engine. Logs a warning so the
    /// loss is visible.
    pub fn load_or_new(save_paths: SavePaths, default_graph: RegionGraph) -> Result<Self> {
        Self::load_or_new_with_content(save_paths, default_graph, crate::ContentSource::Embedded)
    }

    /// [`Self::load_or_new`] with an explicit content source threaded
    /// through both the load and fresh-seed paths. The pack must match
    /// the one any existing save was created with.
    pub fn load_or_new_with_content(
        save_paths: SavePaths,
        default_graph: RegionGraph,
        content: crate::ContentSource,
    ) -> Result<Self> {
        if !save_paths.snapshot.exists() {
            return Self::new_with_content(save_paths, default_graph, content);
        }
        match Self::load_with_content(save_paths.clone(), content.clone()) {
            Ok(mut sim) => {
                // The `RegionGraph` is structural metadata defined by
                // code, not gameplay state — it carries region names,
                // map scenes, neighbor topology, and portal positions.
                // When a new map ships (e.g. cascade_locks added to
                // the gorge corridor), older snapshots still carry
                // their stale graph; without this overwrite the
                // worker would cache the old graph at spawn and the
                // bridge's `region_transitions` / `region_map_scene`
                // would return empty for any region added after the
                // save was first created. Region IDs are append-only,
                // so overwriting the graph is safe: existing entities
                // keep their `InRegion(RegionId)` and resolve against
                // the same numeric ids. New regions become available;
                // removed regions (rare) leave dangling refs that the
                // existing `regions().get(id)` already handles via
                // `Option`.
                sim.world.insert_resource(default_graph);
                Ok(sim)
            }
            Err(e) => {
                tracing::warn!("load failed ({e:#}); discarding stale save and reseeding");
                let _ = std::fs::remove_file(&save_paths.snapshot);
                let _ = std::fs::remove_file(&save_paths.journal);
                Self::new_with_content(save_paths, default_graph, content)
            }
        }
    }

    /// Create a fresh sim with the given region graph.
    pub fn new(save_paths: SavePaths, graph: RegionGraph) -> Result<Self> {
        Self::new_with_seed(save_paths, graph, DEFAULT_SEED)
    }

    /// Fresh sim with the default seed and an explicit content source.
    pub fn new_with_content(
        save_paths: SavePaths,
        graph: RegionGraph,
        content: crate::ContentSource,
    ) -> Result<Self> {
        Self::new_with_seed_and_content(save_paths, graph, DEFAULT_SEED, content)
    }

    /// In-memory variant of [`Self::new_with_seed`] for tests: same
    /// world build, full schedule, **no journal and no snapshot
    /// I/O**. The journal append + flush per tick was the dominant
    /// cost in the test suite (~410 ms/tick in debug); this path drops
    /// it. Production code should use [`Self::new`] / [`Self::load_or_new`]
    /// — without persistence the sim has nothing to recover from on
    /// restart.
    pub fn new_in_memory(graph: RegionGraph) -> Self {
        Self::new_in_memory_with_seed(graph, DEFAULT_SEED)
    }

    /// In-memory variant of [`Self::new_with_seed`]; see [`Self::new_in_memory`].
    /// Also clears `PopulationTargets` post-seed so the spawn loop
    /// doesn't fill the test world with thousands of NPCs the test
    /// doesn't care about — that was the dominant tick cost in the
    /// debug test suite (see TEMP comment in `world_seed.rs`). Tests
    /// that need NPCs call `set_population_target_for_test` to opt back
    /// in, the same shape most NPC tests already use.
    pub fn new_in_memory_with_seed(graph: RegionGraph, seed: u64) -> Self {
        Self::new_in_memory_with_content(graph, seed, crate::ContentSource::Embedded)
    }

    /// In-memory sim built from an explicit content source; see
    /// [`Self::new_in_memory_with_seed`] and [`Self::new_with_content`].
    pub fn new_in_memory_with_content(
        graph: RegionGraph,
        seed: u64,
        content: crate::ContentSource,
    ) -> Self {
        let mut world = Self::build_world(graph, seed, &content);
        world.resource_mut::<PopulationTargets>().by_region.clear();
        Self {
            world,
            schedule_player: build_schedule_player(),
            schedule_npc_index: build_schedule_npc_index(),
            schedule_npc_threats: build_schedule_npc_threats(),
            schedule_npc_aggro: build_schedule_npc_aggro(),
            schedule_npc_planning: build_schedule_npc_planning(),
            schedule_npc_lifecycle: build_schedule_npc_lifecycle(),
            schedule_offline_loot: build_schedule_offline_loot(),
            journal: None,
            save_paths: None,
            snapshot_interval: SNAPSHOT_INTERVAL_TICKS,
            last_tick_deltas: Vec::new(),
            snapshot_ring: [None, None],
            // No disk persistence on in-memory sims → no writer.
            snapshot_writer: None,
            tick_perf_history: std::collections::VecDeque::with_capacity(TICK_PERF_WINDOW),
        }
    }

    /// Create a fresh sim with an explicit random seed for the
    /// content generator. Useful for tests and for production
    /// servers that want a unique world per instance.
    pub fn new_with_seed(save_paths: SavePaths, graph: RegionGraph, seed: u64) -> Result<Self> {
        Self::new_with_seed_and_content(save_paths, graph, seed, crate::ContentSource::Embedded)
    }

    /// Fresh sim with an explicit seed AND content source. The bridge
    /// uses this (and [`Self::new_with_content`]) to supply a game's
    /// own content pack instead of the embedded example pack.
    pub fn new_with_seed_and_content(
        save_paths: SavePaths,
        graph: RegionGraph,
        seed: u64,
        content: crate::ContentSource,
    ) -> Result<Self> {
        ensure_parent_dir(&save_paths.snapshot)?;
        ensure_parent_dir(&save_paths.journal)?;

        let mut world = Self::build_world(graph, seed, &content);

        // Write an initial snapshot so the journal has a starting tick
        // to pair against. Snapshots are cheap when the world is
        // empty and this simplifies load logic.
        let body = serialize_world(&mut world);
        write_snapshot(&save_paths.snapshot, 0, &body).context("write initial snapshot")?;
        let journal = JournalWriter::open(&save_paths.journal, 0)?;

        Ok(Self {
            world,
            schedule_player: build_schedule_player(),
            schedule_npc_index: build_schedule_npc_index(),
            schedule_npc_threats: build_schedule_npc_threats(),
            schedule_npc_aggro: build_schedule_npc_aggro(),
            schedule_npc_planning: build_schedule_npc_planning(),
            schedule_npc_lifecycle: build_schedule_npc_lifecycle(),
            schedule_offline_loot: build_schedule_offline_loot(),
            journal: Some(journal),
            save_paths: Some(save_paths),
            snapshot_interval: SNAPSHOT_INTERVAL_TICKS,
            last_tick_deltas: Vec::new(),
            snapshot_ring: [None, None],
            // Background disk writer for periodic snapshots. Spawned
            // here (not lazily on first roll) so the thread is ready
            // long before the 30 s mark, and so we can assert it's
            // alive in `roll_snapshot`'s expect/unwrap-free path.
            snapshot_writer: Some(crate::persistence::SnapshotWriter::spawn()),
            tick_perf_history: std::collections::VecDeque::with_capacity(TICK_PERF_WINDOW),
        })
    }

    /// Build the resource-populated `World` shared by [`Self::new_with_seed`]
    /// and [`Self::new_in_memory_with_seed`]. Identical world; the two
    /// callers differ only in whether they wire up disk persistence.
    fn build_world(graph: RegionGraph, seed: u64, content: &crate::ContentSource) -> World {
        let mut world = World::new();
        world.insert_resource(SimClock::new());
        world.insert_resource(WorldTime::new());
        world.insert_resource(WeatherState::new());
        world.insert_resource(NpcIdCounter::default());
        world.insert_resource(crate::resources::WoundIdCounter::default());
        world.insert_resource(crate::resources::EffectIdCounter::default());
        world.insert_resource(crate::resources::JobIdCounter::default());
        world.insert_resource(crate::resources::ProjectileIdCounter::default());
        world.insert_resource(crate::resources::ContainerIdCounter::default());
        world.insert_resource(crate::resources::CorpseIndex::default());
        world.insert_resource(crate::resources::BallisticsConfig::load_from(content));
        world.insert_resource(crate::resources::MedConfig::default());
        world.insert_resource(crate::systems::NpcLastHealTick::default());
        world.insert_resource(crate::resources::InventoryConfig::default());
        world.insert_resource(crate::behavior_config::BehaviorConfig::load_from(content));
        world.insert_resource(LifeChronicle::default());
        world.insert_resource(PendingDeltas::default());
        world.insert_resource(crate::resources::PendingKillCredits::default());
        world.insert_resource(crate::resources::PendingNpcShots::default());
        world.insert_resource(NpcPositionIndex::default());
        world.insert_resource(NpcSpatialHash::default());
        world.insert_resource(SquadObjectives::default());
        world.insert_resource(crate::resources::GuardPosts::default());
        world.insert_resource(crate::resources::InteractionAreas::default());
        world.insert_resource(crate::resources::ActivityPoints::default());
        world.insert_resource(crate::resources::AuthoredSpawnPoints::default());
        world.insert_resource(crate::cover::CoverVolumes::default());
        world.insert_resource(crate::patrol_zone::PatrolZones::default());
        world.insert_resource(ActiveRegions::default());
        world.insert_resource(crate::offline_tier::OfflineTierClock::default());
        world.insert_resource(crate::pda_log::PdaEventLog::default());
        world.insert_resource(crate::resources::BehaviorLog::default());
        world.insert_resource(crate::perception::PerceptionConfig::default());
        world.insert_resource(crate::perception::LosService::default());
        world.insert_resource(crate::items::ItemRegistry::load_from(content));
        world.insert_resource(crate::items::RecipeRegistry::load_from(content));
        world.insert_resource(crate::items::EquipmentSlotRegistry::load_from(content));
        world.insert_resource(crate::loot_containers::LootContainerRegistry::load_from(
            content,
        ));
        world.insert_resource(crate::loot_pools::LootPoolRegistry::load_from(content));
        world.insert_resource(crate::names::NameRegistry::load_from(content));
        let __items = world.resource::<crate::items::ItemRegistry>().clone();
        world.insert_resource(crate::npc_loadouts::NpcLoadoutRegistry::load_from(
            content, &__items,
        ));
        world.insert_resource(crate::resources::TerrainMaps::default());
        world.insert_resource(crate::nav::NavQueries::default());
        world.insert_resource(crate::los_cache::LosCache::default());
        world.insert_resource(crate::squad_blackboard::SquadBlackboards::default());
        world.insert_resource(crate::world_event_bus::WorldEventQueue::default());
        // Reusable scratch buffers for hot per-tick systems. Cleared
        // and re-filled in place by their owning systems so we don't
        // re-allocate snapshot Vecs / lookup HashMaps every tick.
        world.init_resource::<crate::systems::AggroScratch>();

        // Faction registry is rebuilt from TOML at every startup —
        // it's content config, not save state. The runtime-mutable
        // drift + per-player rep resources start empty (step 7 will
        // wire their persistence into snapshots).
        world.insert_resource(crate::faction::registry::load_from(content));
        world.insert_resource(crate::faction::registry::RelationDeltas::default());
        world.insert_resource(crate::faction::registry::PlayerReputation::default());

        // Seed random faction control + bases before the graph moves
        // into the world (we still need to read it). Also derives
        // initial population targets from the seeded RegionControl.
        seed_random_world_content(&mut world, &graph, seed);
        world.insert_resource(graph);

        world
    }

    /// Load an existing sim: read the snapshot, replay the journal
    /// tail, resume.
    pub fn load(save_paths: SavePaths) -> Result<Self> {
        Self::load_with_content(save_paths, crate::ContentSource::Embedded)
    }

    /// Load an existing sim with an explicit content source. The pack
    /// MUST match the one the save was created with — content is not
    /// stored in the snapshot. See [`Self::new_with_content`].
    pub fn load_with_content(save_paths: SavePaths, content: crate::ContentSource) -> Result<Self> {
        // Borrow once so the registry loads below read identically to
        // `build_world` (`load_from(content)`).
        let content = &content;
        let (snap_tick, body) = read_snapshot(&save_paths.snapshot).context("read snapshot")?;
        let (journal_tick, deltas) = read_journal(&save_paths.journal).context("read journal")?;

        let mut world = World::new();
        let mut clock = body.clock;
        clock.tick = snap_tick;
        world.insert_resource(clock);
        world.insert_resource(body.region_graph);
        world.insert_resource(body.world_time);
        world.insert_resource(body.weather);
        world.insert_resource(body.region_control);
        // Summary cache is `#[serde(skip)]`, so the loaded
        // chronicle arrives with zeroed totals — rebuild from
        // records before inserting as a resource.
        let mut chronicle = body.chronicle;
        chronicle.rebuild_summary_cache();
        world.insert_resource(chronicle);
        world.insert_resource(body.npc_id_counter);
        world.insert_resource(body.wound_id_counter);
        world.insert_resource(body.effect_id_counter);
        world.insert_resource(body.job_id_counter);
        world.insert_resource(body.projectile_id_counter);
        world.insert_resource(body.container_id_counter);
        // CorpseIndex is transient — corpse markers re-populate from
        // the journal's `WorldContainerSpawned` deltas + `NpcDied`
        // events on tick (the index is just a spatial cache for the
        // loot arbiter; the authoritative state lives on
        // `WorldContainer` entities). Insert empty on load and let
        // future deaths re-seed it.
        world.insert_resource(crate::resources::CorpseIndex::default());
        world.insert_resource(crate::resources::BallisticsConfig::load_from(content));
        world.insert_resource(crate::resources::MedConfig::default());
        world.insert_resource(crate::systems::NpcLastHealTick::default());
        world.insert_resource(crate::resources::InventoryConfig::default());
        world.insert_resource(crate::behavior_config::BehaviorConfig::load_from(content));
        world.insert_resource(body.population_targets);
        world.insert_resource(PendingDeltas::default());
        world.insert_resource(crate::resources::PendingKillCredits::default());
        world.insert_resource(crate::resources::PendingNpcShots::default());
        world.insert_resource(NpcPositionIndex::default());
        world.insert_resource(NpcSpatialHash::default());
        world.insert_resource(SquadObjectives::default());
        world.insert_resource(crate::resources::GuardPosts::default());
        // Iteration 5-13 Phase D2: same lifecycle as `build_world` —
        // empty on load; map-load path repopulates from scene markers.
        world.insert_resource(crate::resources::InteractionAreas::default());
        world.insert_resource(ActiveRegions::default());
        // Restore offline-tier clock from snapshot (Phase 1B). The
        // `build_world` path inserts a fresh `default()`; here we
        // overwrite with whatever the snapshot recorded so the slow
        // tier resumes mid-run.
        world.insert_resource(body.offline_tier_clock);
        // PdaEventLog (Phase 1F) is NOT persisted across save/load —
        // toast notifications are transient by design. Still need
        // to insert the resource, otherwise `offline_combat` panics
        // on `ResMut<PdaEventLog>`.
        world.insert_resource(crate::pda_log::PdaEventLog::default());
        world.insert_resource(crate::resources::BehaviorLog::default());
        world.insert_resource(crate::perception::PerceptionConfig::default());
        world.insert_resource(crate::perception::LosService::default());
        world.insert_resource(crate::items::ItemRegistry::load_from(content));
        world.insert_resource(crate::items::RecipeRegistry::load_from(content));
        world.insert_resource(crate::items::EquipmentSlotRegistry::load_from(content));
        world.insert_resource(crate::loot_containers::LootContainerRegistry::load_from(
            content,
        ));
        world.insert_resource(crate::loot_pools::LootPoolRegistry::load_from(content));
        world.insert_resource(crate::names::NameRegistry::load_from(content));
        let __items = world.resource::<crate::items::ItemRegistry>().clone();
        world.insert_resource(crate::npc_loadouts::NpcLoadoutRegistry::load_from(
            content, &__items,
        ));
        world.insert_resource(crate::resources::TerrainMaps::default());
        world.insert_resource(crate::nav::NavQueries::default());
        world.insert_resource(crate::los_cache::LosCache::default());
        world.insert_resource(crate::squad_blackboard::SquadBlackboards::default());
        world.insert_resource(crate::world_event_bus::WorldEventQueue::default());
        // Resources added by the sim overhaul — transient, rebuilt
        // from scene markers on region attach.
        world.insert_resource(crate::resources::ActivityPoints::default());
        world.insert_resource(crate::resources::AuthoredSpawnPoints::default());
        world.insert_resource(crate::cover::CoverVolumes::default());
        world.insert_resource(crate::patrol_zone::PatrolZones::default());
        // Reusable scratch buffers for hot per-tick systems. Cleared
        // and re-filled in place by their owning systems so we don't
        // re-allocate snapshot Vecs / lookup HashMaps every tick.
        world.init_resource::<crate::systems::AggroScratch>();
        // Faction registry rebuilt from canonical TOML on every load
        // (it's content config, not save state). Drift + per-player
        // rep restored from the snapshot — playthrough rep evolution
        // survives save/load.
        world.insert_resource(crate::faction::registry::load_from(content));
        world.insert_resource(body.relation_deltas);
        world.insert_resource(body.player_reputation);
        for se in body.entities {
            spawn_serialized(&mut world, se);
        }

        // Apply journal deltas if they belong to this snapshot.
        if journal_tick == snap_tick {
            for delta in deltas {
                apply_delta(&mut world, &delta);
            }
        } else if journal_tick != 0 {
            tracing::warn!(
                "journal snapshot_tick {} doesn't match snapshot tick {}; discarding journal",
                journal_tick,
                snap_tick
            );
        }

        // Reopen journal (truncates-and-rewrites header if tick mismatched).
        let journal = JournalWriter::open(&save_paths.journal, snap_tick)?;

        Ok(Self {
            world,
            schedule_player: build_schedule_player(),
            schedule_npc_index: build_schedule_npc_index(),
            schedule_npc_threats: build_schedule_npc_threats(),
            schedule_npc_aggro: build_schedule_npc_aggro(),
            schedule_npc_planning: build_schedule_npc_planning(),
            schedule_npc_lifecycle: build_schedule_npc_lifecycle(),
            schedule_offline_loot: build_schedule_offline_loot(),
            journal: Some(journal),
            save_paths: Some(save_paths),
            snapshot_interval: SNAPSHOT_INTERVAL_TICKS,
            last_tick_deltas: Vec::new(),
            snapshot_ring: [None, None],
            // Background disk writer — see `new_with_seed`.
            snapshot_writer: Some(crate::persistence::SnapshotWriter::spawn()),
            tick_perf_history: std::collections::VecDeque::with_capacity(TICK_PERF_WINDOW),
        })
    }
    // tick() and recent_tick_perf() → world/tick.rs

    /// Flush the journal and write a final snapshot. Call on graceful
    /// shutdown. No-op on mirror sims (no disk state to flush).
    pub fn shutdown(&mut self) -> Result<()> {
        let tick = self.current_tick();
        if self.save_paths.is_some() {
            self.roll_snapshot(tick)?;
        }
        if let Some(ref mut journal) = self.journal {
            journal.flush_and_sync()?;
        }
        // Drain + join the background snapshot writer thread so the
        // final snapshot reaches disk before we return. Without
        // this, a process exit immediately after `shutdown` could
        // truncate a pending snapshot mid-rename.
        if let Some(mut writer) = self.snapshot_writer.take() {
            writer.shutdown();
        }
        Ok(())
    }

    /// Record one delta: append to the journal (authoritative only)
    /// and stage for broadcast via [`Self::drain_tick_deltas`]. Used
    /// by every mutation method + the schedule's `PendingDeltas`
    /// drain in [`Self::tick`]. Keeps journal + wire in lockstep so
    /// replay and replication observe the same stream.
    fn record_delta(&mut self, delta: WorldDelta) -> Result<()> {
        if let Some(ref mut journal) = self.journal {
            journal.append(&delta).context("journal append")?;
        }
        self.last_tick_deltas.push(delta);
        Ok(())
    }

    pub fn current_tick(&self) -> u64 {
        self.world.resource::<SimClock>().tick
    }

    /// Borrow the active [`FactionRegistry`]. Loaded from the
    /// canonical TOML at sim startup; not mutable at runtime (re-load
    /// requires restart).
    pub fn faction_registry(&self) -> &crate::faction::registry::FactionRegistry {
        self.world
            .resource::<crate::faction::registry::FactionRegistry>()
    }

    /// Lookup convenience for the registry's per-faction debug color
    /// (RGB, 0..=255 per channel). Used by the GDScript debug overlay
    /// to color minimap dots, marker pills, and dev-mode NPC tints.
    /// `None` when the faction name isn't in the registry.
    pub fn faction_debug_color(&self, name: &str) -> Option<[u8; 3]> {
        let reg = self.faction_registry();
        reg.id_of(name).map(|id| reg.def(id).debug_color)
    }

    /// Borrow the runtime [`RelationDeltas`] (faction-vs-faction
    /// drift accumulator).
    pub fn relation_deltas(&self) -> &crate::faction::registry::RelationDeltas {
        self.world
            .resource::<crate::faction::registry::RelationDeltas>()
    }

    /// Mutable borrow for callers wiring the drift API. Most gameplay
    /// goes through [`Sim::shift_faction_relation`] /
    /// [`Sim::shift_player_rep`] (step 7).
    pub fn relation_deltas_mut(&mut self) -> &mut crate::faction::registry::RelationDeltas {
        self.world
            .resource_mut::<crate::faction::registry::RelationDeltas>()
            .into_inner()
    }

    /// Borrow the per-player reputation table.
    pub fn player_reputation(&self) -> &crate::faction::registry::PlayerReputation {
        self.world
            .resource::<crate::faction::registry::PlayerReputation>()
    }

    pub fn player_reputation_mut(&mut self) -> &mut crate::faction::registry::PlayerReputation {
        self.world
            .resource_mut::<crate::faction::registry::PlayerReputation>()
            .into_inner()
    }

    /// Apply a runtime drift to the faction-vs-faction relation
    /// matrix, journal the shift so it survives save/load, and
    /// stage it for broadcast over the wire. `reason` is a
    /// free-form tag for the chronicle UI ("killed_lineman_crew",
    /// "freed_attuned_prisoner"); not machine-parsed.
    ///
    /// Returns `Ok` on success. Errors when either faction name
    /// isn't in the active registry. Mirror sims don't journal
    /// (host-authoritative); they apply the broadcast delta when it
    /// arrives.
    pub fn shift_faction_relation(
        &mut self,
        a: &str,
        b: &str,
        delta: i16,
        reason: &str,
    ) -> Result<()> {
        let reg = self.faction_registry();
        let Some(id_a) = reg.id_of(a) else {
            anyhow::bail!("unknown faction in shift_faction_relation: {a}");
        };
        let Some(id_b) = reg.id_of(b) else {
            anyhow::bail!("unknown faction in shift_faction_relation: {b}");
        };
        // Apply to the live resource so subsequent reads see it
        // even if journaling is off (mirror sim).
        let registry = reg.clone();
        let mut deltas = self
            .world
            .resource_mut::<crate::faction::registry::RelationDeltas>();
        crate::faction::registry::shift_faction_relation(
            &registry,
            deltas.as_mut(),
            id_a,
            id_b,
            delta,
        );
        // Journal + broadcast.
        self.record_delta(WorldDelta::FactionRelationShift {
            a: a.to_string(),
            b: b.to_string(),
            delta,
            reason: reason.to_string(),
        })
    }

    /// Apply a runtime drift to one player's faction reputation,
    /// journal the shift, stage for broadcast. Player A's drift does
    /// not move player B's standing — `PlayerReputation` is keyed
    /// per-SteamId.
    pub fn shift_player_rep(
        &mut self,
        steam_id: u64,
        faction: &str,
        delta: i16,
        reason: &str,
    ) -> Result<()> {
        let reg = self.faction_registry();
        let Some(id_f) = reg.id_of(faction) else {
            anyhow::bail!("unknown faction in shift_player_rep: {faction}");
        };
        let registry = reg.clone();
        let mut rep = self
            .world
            .resource_mut::<crate::faction::registry::PlayerReputation>();
        crate::faction::registry::shift_player_rep(&registry, rep.as_mut(), steam_id, id_f, delta);
        self.record_delta(WorldDelta::PlayerRepShift {
            steam_id,
            faction: faction.to_string(),
            delta,
            reason: reason.to_string(),
        })
    }

    /// Swap in a custom `LosProvider`. Call from the engine-facing
    /// crate (`simn-godot`) after `Sim::new`/`load` so FOV-filtered
    /// sightings go through a real raycast against the Godot
    /// physics world. Pass `Arc::new(AlwaysVisibleLos)` to reset.
    pub fn install_los_provider(
        &mut self,
        provider: std::sync::Arc<dyn crate::perception::LosProvider>,
    ) {
        self.world
            .resource_mut::<crate::perception::LosService>()
            .provider = provider;
    }

    /// Mutable access to the perception config (FOV, sight radius,
    /// exposure thresholds). Changes take effect next tick.
    pub fn perception_config_mut(&mut self) -> &mut crate::perception::PerceptionConfig {
        self.world
            .resource_mut::<crate::perception::PerceptionConfig>()
            .into_inner()
    }
    // initial_bulk_seed_npcs through regions() → world/population.rs

    // interaction areas + register_authored_base → world/registration.rs

    // upsert_player through advance_time → world/player.rs

    // region_control through population_targets → world/npc_view.rs

    // apply_damage through find_player_entity → world/player.rs

    /// Serialize the current world state to an in-memory snapshot
    /// byte vector. Identical bytes to what `roll_snapshot` would
    /// write to disk (magic + version + tick + body length + body
    /// bytes + blake3 hash).
    ///
    /// Used by:
    /// - The determinism harness (compare two sims tick-by-tick).
    /// - The eventual replication path — host serializes once,
    ///   broadcasts the same bytes to clients + writes to disk.
    pub fn write_snapshot_to_vec(&mut self) -> Result<Vec<u8>> {
        let tick = self.world.resource::<crate::resources::SimClock>().tick;
        let body = serialize_world(&mut self.world);
        write_snapshot_to_vec(tick, &body)
    }

    fn roll_snapshot(&mut self, tick: u64) -> Result<()> {
        let Some(save_paths) = self.save_paths.clone() else {
            return Ok(()); // mirror sim: nothing to persist
        };
        // Serialize in-memory on the worker (mandatory — needs
        // `&mut World`). Encode bytes synchronously. Hand off the
        // disk write to the background `SnapshotWriter` so the
        // atomic-tmp-then-rename I/O (typically 200-1000 ms for a
        // multi-MB snapshot) doesn't stall the tick loop. Journal
        // rotation stays on the worker because the journal-file
        // handle lives here.
        let body = serialize_world(&mut self.world);
        let bytes = crate::persistence::write_snapshot_to_vec(tick, &body)?;
        if let Some(writer) = self.snapshot_writer.as_ref() {
            writer.enqueue(save_paths.snapshot.clone(), bytes, tick)?;
        } else {
            // Fallback (shouldn't happen in production): the writer
            // is only `None` on mirror / in-memory sims, which
            // already short-circuited above via `save_paths`. Write
            // sync as a defensive fallback.
            crate::persistence::write_snapshot_bytes(&save_paths.snapshot, &bytes)?;
        }
        if let Some(ref mut journal) = self.journal {
            journal.rotate(tick)?;
        }
        Ok(())
    }
}
// tally_delta through build_schedule_* → world/tick.rs

pub(crate) use tick::{
    build_schedule_empty, build_schedule_npc_aggro, build_schedule_npc_index,
    build_schedule_npc_index_only_mirror, build_schedule_npc_lifecycle,
    build_schedule_npc_planning, build_schedule_npc_threats, build_schedule_offline_loot,
    build_schedule_player, build_schedule_player_mirror,
};

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create save dir {}", parent.display()))?;
    }
    Ok(())
}
