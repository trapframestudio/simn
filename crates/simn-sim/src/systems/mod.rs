//! Systems that run inside the tick schedule.
//!
//! Pure systems (movement, perception, regen, world-time advance)
//! don't journal — they're deterministic from snapshot state +
//! elapsed ticks. Event-driven mutations (spawn, despawn, region
//! migration, death) push to `PendingDeltas` for `Sim::tick` to
//! drain into the journal.

/// Diagnostic timing guard — drop logs system elapsed if it exceeds
/// the threshold. Add `let _t = SysTimer::new("name");` at the top
/// of any system to time it; any exit path triggers Drop.
/// Suppressed via `SIMN_QUIET=1`.
pub(crate) struct SysTimer {
    name: &'static str,
    start: std::time::Instant,
}

impl SysTimer {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for SysTimer {
    fn drop(&mut self) {
        let e = self.start.elapsed();
        if is_verbose_logging() && e.as_millis() > 2 {
            eprintln!("[sys {}] {:?}", self.name, e);
        }
    }
}

/// Read once at startup: are verbose sim diagnostics enabled?
/// Default off so a vanilla launch is quiet (no per-tick
/// `[sys …]` / `[spawn_npcs …]` / `[tick_npc_goals …]` /
/// `[sim.tick …]` lines spamming the console). Set
/// `SIMN_VERBOSE=1` in the environment to re-enable when
/// profiling or debugging a tick path.
///
/// Replaces the old `SIMN_QUIET` env-var gate, which was
/// inverted (logs ON by default) and re-checked on every
/// per-tick log site. This one is read once into a `OnceLock`
/// so steady-state per-tick cost is one atomic load.
fn verbose_logging_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("SIMN_VERBOSE").is_ok())
}

pub fn is_verbose_logging() -> bool {
    verbose_logging_enabled()
}

/// Always-on per-system profile slots for the perception sub-group.
/// Each `npc_index` system records its elapsed time into a fixed
/// slot; `Sim::tick` reads + resets the whole array after the
/// schedule runs. Thread-local so the worker thread's records don't
/// clash with anything else, and so reads/writes are lock-free.
pub mod prof_slots {
    pub const CLEAR_LOS: usize = 0;
    pub const SWEEP_BB: usize = 1;
    pub const POSITION_INDEX: usize = 2;
    pub const DRAIN_EVENTS: usize = 3;
    pub const SPATIAL_HASH: usize = 4;
    pub const SQUAD_PLANNER: usize = 5;
    pub const GOAL_ARBITRATION: usize = 6;
    pub const TICK_NPC_GOALS: usize = 7;
    pub const NPC_COMBAT: usize = 8;
    pub const NUM: usize = 9;
}

thread_local! {
    static PROF: std::cell::RefCell<[std::time::Duration; prof_slots::NUM]> =
        const { std::cell::RefCell::new([std::time::Duration::ZERO; prof_slots::NUM]) };
}

/// Record the elapsed time for a single system into the
/// thread-local profile array. Slot indices live in
/// [`prof_slots`]. Reads happen via [`drain_perception_slots`]
/// after the relevant schedule segment runs.
pub fn record_perception_slot(slot: usize, d: std::time::Duration) {
    if slot >= prof_slots::NUM {
        return;
    }
    PROF.with(|c| c.borrow_mut()[slot] = d);
}

pub fn drain_perception_slots() -> [std::time::Duration; prof_slots::NUM] {
    PROF.with(|c| {
        let out = *c.borrow();
        *c.borrow_mut() = [std::time::Duration::ZERO; prof_slots::NUM];
        out
    })
}

/// RAII guard that records the elapsed time into the named profile
/// slot on drop. Drops happen on any exit path (including early
/// returns), so one line at the top of a system covers the whole
/// function. Cost: one `Instant::now()` at construction + one at
/// drop (~25 ns each).
pub struct ProfGuard(pub std::time::Instant, pub usize);

impl Drop for ProfGuard {
    fn drop(&mut self) {
        record_perception_slot(self.1, self.0.elapsed());
    }
}

thread_local! {
    static EVENT_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub fn record_event_count(n: usize) {
    EVENT_COUNT.with(|c| c.set(n));
}

pub fn drain_event_count() -> usize {
    EVENT_COUNT.with(|c| {
        let n = c.get();
        c.set(0);
        n
    })
}

pub mod base_capture;
pub mod broadcast_npc_positions;
pub mod clock;
pub mod contamination;
pub mod crafting;
pub mod goal_arbitration;
pub mod inventory;
pub mod kill_credits;
pub mod loot_restock;
pub mod meds;
pub mod npc_age;
pub mod npc_aggro;
pub mod npc_combat;
pub mod npc_death_check;
pub mod npc_goals;
pub mod npc_index;
pub mod npc_join_group;
pub mod npc_migrate;
pub mod npc_portal_cross;
pub mod npc_spatial_hash;
pub mod npc_spawn;
pub mod npc_tactical;
pub mod squad_planner;
pub mod stamina;
pub mod survival;
pub mod terrain;
pub mod threat_board;
pub mod weather;
pub mod world_time;
pub mod wounds;

pub use base_capture::base_capture_check;
pub use broadcast_npc_positions::broadcast_npc_positions;
pub use clock::advance_clock;
pub use contamination::tick_contamination;
pub use crafting::tick_crafting_queue;
pub use goal_arbitration::{biased_priority, goal_arbitration, personality_bias_for_objective};
pub use inventory::tick_perishables;
pub use kill_credits::apply_kill_credits;
pub use loot_restock::tick_loot_restock;
pub use meds::{decay_drug_tolerance, tick_active_effects, tick_pain};
pub use npc_age::age_npcs;
pub use npc_aggro::{npc_aggro, AggroScratch};
pub use npc_combat::{accuracy_hit_multiplier, npc_combat};
pub use npc_death_check::{npc_death_check, prune_corpse_index};
pub use npc_goals::tick_npc_goals;
pub use npc_index::index_npc_positions;
pub use npc_join_group::npc_join_group;
pub use npc_migrate::migrate_npcs;
pub use npc_portal_cross::npc_portal_cross;
pub use npc_spatial_hash::rebuild_spatial_hash;
pub use npc_spawn::spawn_npcs;
pub use npc_tactical::npc_tactical;
pub use squad_planner::{
    cohesion_multiplier_for_leadership, objective_utility, squad_planner, BlackboardSignals,
    ObjKind, SquadPersonality,
};
pub use stamina::regen_stamina;
pub use survival::{apply_survival_effects, drain_survival_stats};
pub use terrain::clamp_npc_terrain_y;
pub use threat_board::{apply_threat_priority, sweep_threats};
pub use weather::advance_weather;
pub use world_time::advance_world_time;
pub use wounds::{
    age_and_heal_wounds, apply_bleed_damage, bleed_rate_multiplier, npc_treat_wounds,
    tick_infection, tick_necrosis, NpcLastHealTick,
};
