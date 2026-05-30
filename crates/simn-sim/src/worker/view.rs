//! `SimView` — denormalized, read-only per-tick view of `Sim`
//! state that the main thread can read without touching the
//! authoritative ECS. Build at end-of-tick on the worker thread
//! (once the worker exists in step 3); publish through an
//! `ArcSwap` (step 5) so any number of main-thread readers can
//! load the latest view lock-free.
//!
//! ## Why a denormalized view rather than round-trip queries
//!
//! HUD code in GDScript calls `player_state`, `weather`,
//! `chronicle_summary` etc. every frame. If every read goes
//! command → worker → reply, the renderer either blocks on the
//! tick boundary (50ms stalls) or grows a per-call request /
//! response state machine. Building the view once at end-of-
//! tick and publishing the whole thing is one walk of the ECS
//! per tick (20 Hz) versus N polled reads per frame (~100 Hz),
//! and the renderer reads are now a single atomic load.
//!
//! ## Scope of step 1
//!
//! This first commit defines `SimView` and a builder for the
//! highest-traffic fields. Step 4 of the threaded-sim rollout
//! extends the view with the remaining `SimHost` reads
//! (inventory grids, crafting queues, containers, region
//! controls, etc.) as it rewires call sites; the goal here is
//! to land the type + the pattern + a builder test, not the
//! full denormalized world.

use std::collections::HashMap;

use crate::chronicle::ChronicleSummary;
use crate::components::{CraftJob, EquippedItem, GridInventory};
use crate::items::{SlotId, ToolTier};
use crate::region::RegionId;
use crate::resources::{RegionControlState, WeatherState, WorldTime};
use crate::world::{PlayerView, Sim};

/// Per-player ancillary state — everything `player_state` reads
/// that isn't already on `PlayerView`. Built alongside the
/// `PlayerView` in [`build_sim_view`] so a single end-of-tick
/// pass produces both. Worker-mode `player_state` reads these
/// fields directly off `SimView::player_extras`.
///
/// Threaded-sim PR C step 4b-v.
#[derive(Clone, Debug, Default)]
pub struct PlayerExtras {
    /// Grid-shaped inventory snapshot. Width × height × slot
    /// stacks. Used by the inventory panel renderer.
    pub inventory: GridInventory,
    /// Total carried weight in kg. Used by encumbrance UI.
    pub weight: f32,
    /// Whether the player is currently within campfire range
    /// (drives crafting station availability + "cooking
    /// available" UI).
    pub near_campfire: bool,
    /// Workbench tier the player is within range of, if any.
    /// `None` when not near a workbench; `Some(tier)` carries
    /// the highest-tier nearby station.
    pub near_workbench: Option<ToolTier>,
    /// Live crafting jobs owned by this player. Order matches
    /// the queue ordering on `Sim::crafting_queue`.
    pub crafting_queue: Vec<CraftJob>,
    /// Currently-equipped items, keyed by slot id. Empty map
    /// when nothing is equipped.
    pub equipment: HashMap<SlotId, EquippedItem>,
}

/// Denormalized read-only snapshot of `Sim` state for
/// main-thread consumers. Cheap to clone field-by-field but
/// expected to be shared via `Arc<ArcSwap<SimView>>` (step 5),
/// so a single rebuild per tick fans out to every reader with
/// zero cost beyond an `Arc` clone.
///
/// Fields are intentionally a subset of `Sim`'s read surface
/// today — see module docs for the rollout plan. Extensions
/// land in step 4 alongside the matching call-site rewire.
#[derive(Clone, Debug)]
pub struct SimView {
    /// Tick this view was built at. Matches `Sim::current_tick`.
    pub tick: u64,
    /// World clock (day + in-game time of day).
    pub world_time: WorldTime,
    /// Global weather state (current band, upcoming, transition tick).
    pub weather: WeatherState,
    /// Aggregate chronicle stats — alive + ever counts, kills,
    /// etc. Cheap to rebuild; see `LifeChronicle::summary`.
    pub chronicle_summary: ChronicleSummary,
    /// Every connected player's full `PlayerView`, keyed by
    /// `steam_id`. HUD reads (`player_state`) look up by id;
    /// builder iterates `PlayerOwned` once and resolves each
    /// player in one pass.
    pub players: HashMap<u64, PlayerView>,
    /// Per-player ancillary state (inventory grid, equipment,
    /// crafting queue, near-station flags). Same keying as
    /// `players`; populated by the same end-of-tick pass.
    /// Threaded-sim PR C step 4b-v.
    pub player_extras: HashMap<u64, PlayerExtras>,
    /// Per-region faction-control snapshot. Drives the debug
    /// overlay's "primary / contested_by / tension" lines.
    /// Keyed by `RegionId`; bridge resolves the region-name →
    /// id via the worker's cached `Arc<RegionGraph>` and looks
    /// up here. Reads were previously round-tripping through
    /// `worker.inspect` once per frame on the debug overlay,
    /// which destroyed FPS while the overlay was up; folding
    /// the state into the view makes it lock-free.
    pub region_control: HashMap<RegionId, RegionControlState>,
    /// Recent PDA log entries (cap 256). Folded into the view so
    /// `SimHost::recent_pda_events_since` reads lock-free instead
    /// of blocking on `worker.inspect`. The PDA toast script polls
    /// this at 4 Hz; pre-cache the entries each tick.
    pub pda_recent: Vec<crate::pda_log::PdaLogEntry>,
    /// `PdaEventLog::high_water()` snapshot — clients seed their
    /// bookmark from this on first open so pre-join events don't
    /// all toast at once.
    pub pda_high_water: u64,
    /// Active-region NPC views. Cached per tick so
    /// `SimHost::npcs_near` can filter on the main thread without
    /// blocking on `worker.inspect`. Keyed by `RegionId`; only
    /// regions in `ActiveRegions` carry entries since offline NPCs
    /// aren't rendered. ~800 NPC views per region in stress tests;
    /// per-tick clone is ~800 KB but eliminates the 20 Hz inspect-
    /// channel block that was costing ~50% of main-thread time.
    pub npcs_by_region: std::collections::HashMap<RegionId, Vec<crate::world::NpcView>>,
}

/// Build a fresh `SimView` from the current `Sim` state. Call
/// at end-of-tick (after `Sim::tick` has run all systems and
/// published the snapshot ring). `&mut Sim` because the
/// underlying ECS queries (`query::<&Position>`, `world.get`)
/// take a mutable handle even though the build is logically
/// read-only.
///
/// **Step 1 scope** — see module docs. Builder is intentionally
/// thin: the goal is to prove the pattern + the test. Future
/// steps extend the fields and the builder in lockstep with
/// the call sites being rewired.
pub fn build_sim_view(sim: &mut Sim) -> SimView {
    let tick = sim.current_tick();
    let world_time = sim.world_time();
    let weather = sim.weather();
    let chronicle_summary = sim.chronicle_summary();
    let mut players: HashMap<u64, PlayerView> = HashMap::new();
    let mut player_extras: HashMap<u64, PlayerExtras> = HashMap::new();
    for steam_id in sim.connected_player_ids() {
        if let Some(view) = sim.player_view(steam_id) {
            players.insert(steam_id, view);
        }
        // Each call below is a quick query against the ECS;
        // they all take `&mut self` because of bevy_ecs query
        // builder semantics, not mutation. Cost per player at
        // typical inventory size is microseconds, dominated by
        // the GridInventory clone.
        let extras = PlayerExtras {
            inventory: sim.inventory_view_grid(steam_id),
            weight: sim.inventory_weight(steam_id),
            near_campfire: sim.near_campfire(steam_id),
            near_workbench: sim.near_workbench(steam_id),
            crafting_queue: sim.crafting_queue(steam_id),
            equipment: sim.equipment_view(steam_id),
        };
        player_extras.insert(steam_id, extras);
    }
    let region_control = sim.region_controls().clone();
    // PDA log snapshot — every tick we clone the recent entries
    // into the view so `recent_pda_events_since` reads lock-free.
    // Capped at 256 entries (matches `PdaEventLog` cap); clone
    // cost is ~16 KB / tick, well worth eliminating the inspect-
    // channel block on the 4 Hz toast polling path.
    let (pda_recent, pda_high_water) = sim.pda_log_view_snapshot();
    // Active-region NPC view list. Cached per tick so the bridge's
    // per-frame `npcs_near` calls read lock-free instead of blocking
    // on `worker.inspect`. Pre-Phase-2-perf the inspect-based path
    // was costing ~25 ms per 50 ms physics frame from
    // `_sync_npc_dummies` running at 20 Hz.
    let npcs_by_region = sim.active_region_npc_views();
    SimView {
        tick,
        world_time,
        weather,
        chronicle_summary,
        players,
        player_extras,
        region_control,
        pda_recent,
        pda_high_water,
        npcs_by_region,
    }
}
