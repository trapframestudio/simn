//! Slice-1 sim/net replication API on `Sim`.
//!
//! Groups every method used by the host → client replication flow:
//!
//! - [`Sim::new_mirror`] — construct a client-side sim with no
//!   persistence and a reduced schedule (NPC behavior systems omitted
//!   because their RNG seeds depend on `Entity::to_bits()` which isn't
//!   stable across sim instances).
//! - [`Sim::serialize_snapshot_body`] — serialize current world state
//!   for on-wire transmission (host side). Reuses the same `SnapshotBody`
//!   shape as disk saves.
//! - [`Sim::apply_external_snapshot`] — rebuild world from a host-sent
//!   snapshot (client side).
//! - [`Sim::apply_external_delta`] — replay one host-broadcast
//!   `WorldDelta` onto the mirror (client side).
//! - [`Sim::drain_tick_deltas`] — drain the per-tick delta buffer for
//!   host broadcast.
//! - [`Sim::set_tick_for_mirror`] — anchor the mirror's clock to the
//!   host's latest tick stamp.
//! - [`Sim::apply_action`] — dispatch a client-originated `ActionKind`
//!   to its matching mutation method (host side).
//!
//! See `docs/book/src/architecture/networking.md` for the wider
//! slice-1 contract.

use anyhow::Result;
use bevy_ecs::prelude::*;

use crate::action::ActionKind;
use crate::chronicle::LifeChronicle;
use crate::delta::WorldDelta;
use crate::items::ItemId;
use crate::persistence::snapshot::SnapshotBody;
use crate::region::RegionGraph;
use crate::resources::{
    ActiveRegions, NpcIdCounter, NpcPositionIndex, NpcSpatialHash, PendingDeltas, SimClock,
    SquadObjectives, WeatherState, WorldTime,
};

use super::persistence::{apply_delta, serialize_world, spawn_serialized};
use super::{Sim, SNAPSHOT_INTERVAL_TICKS};

impl Sim {
    /// Build a mirror sim for a coop client: no disk persistence, no
    /// NPC-mutating systems. The caller is expected to follow up with
    /// [`Self::apply_external_snapshot`] once the host sends its state.
    /// Until then the world has only the default resources and an
    /// empty region graph.
    pub fn new_mirror(graph: RegionGraph) -> Self {
        // NOTE: the mirror (coop client) always uses the EMBEDDED pack.
        // It's replaced wholesale by `apply_external_snapshot` in steady
        // state, and faction keys/relations/gameplay are identical
        // across packs — only cosmetic display/names/chatter would
        // differ on a client. Full client content parity (a Sim-level
        // content source threaded through snapshot-apply) is a
        // multiplayer follow-up.
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
        world.insert_resource(crate::resources::BallisticsConfig::load());
        world.insert_resource(crate::resources::MedConfig::default());
        world.insert_resource(crate::systems::NpcLastHealTick::default());
        world.insert_resource(crate::resources::InventoryConfig::default());
        world.insert_resource(crate::behavior_config::BehaviorConfig::load());
        world.insert_resource(LifeChronicle::default());
        world.insert_resource(PendingDeltas::default());
        world.insert_resource(crate::resources::PendingKillCredits::default());
        world.insert_resource(NpcPositionIndex::default());
        world.insert_resource(NpcSpatialHash::default());
        world.insert_resource(SquadObjectives::default());
        world.insert_resource(crate::resources::GuardPosts::default());
        world.insert_resource(ActiveRegions::default());
        world.insert_resource(crate::resources::BehaviorLog::default());
        world.insert_resource(crate::perception::PerceptionConfig::default());
        world.insert_resource(crate::perception::LosService::default());
        world.insert_resource(crate::items::ItemRegistry::load());
        world.insert_resource(crate::items::RecipeRegistry::load());
        world.insert_resource(crate::items::EquipmentSlotRegistry::load());
        world.insert_resource(crate::names::NameRegistry::load());
        let __items = world.resource::<crate::items::ItemRegistry>().clone();
        world.insert_resource(crate::npc_loadouts::NpcLoadoutRegistry::load(&__items));
        world.insert_resource(crate::resources::RegionControl::default());
        world.insert_resource(crate::resources::PopulationTargets::default());
        world.insert_resource(crate::resources::MirrorMode);
        // Mirror sims need the registry too so any local query reads
        // the same matrix the host does. Drift + per-player rep stay
        // host-authoritative — host broadcasts shifts; mirrors apply.
        world.insert_resource(crate::faction::registry::load_default());
        world.insert_resource(crate::faction::registry::RelationDeltas::default());
        world.insert_resource(crate::faction::registry::PlayerReputation::default());
        world.insert_resource(graph);

        Self {
            world,
            schedule_player: crate::world::build_schedule_player_mirror(),
            schedule_npc_index: crate::world::build_schedule_empty(),
            schedule_npc_threats: crate::world::build_schedule_empty(),
            schedule_npc_aggro: crate::world::build_schedule_empty(),
            schedule_npc_planning: crate::world::build_schedule_empty(),
            schedule_npc_lifecycle: crate::world::build_schedule_npc_index_only_mirror(),
            schedule_offline_loot: crate::world::build_schedule_empty(),
            journal: None,
            save_paths: None,
            snapshot_interval: SNAPSHOT_INTERVAL_TICKS,
            last_tick_deltas: Vec::new(),
            snapshot_ring: [None, None],
            // Mirror sims don't persist; no writer needed.
            snapshot_writer: None,
            tick_perf_history: std::collections::VecDeque::with_capacity(
                crate::world::TICK_PERF_WINDOW,
            ),
        }
    }

    /// Serialize the current world state into a [`SnapshotBody`] for
    /// network transmission. Same shape the authoritative save path
    /// uses for on-disk snapshots, reused here so the `Sim::load` code
    /// path is also the one driving client snapshot handoff.
    pub fn serialize_snapshot_body(&mut self) -> SnapshotBody {
        serialize_world(&mut self.world)
    }

    /// Replace the mirror's world with the authoritative snapshot
    /// sent by the host. Called once on join, and on any resync. The
    /// `SimClock.tick` is anchored to the host's `tick` so subsequent
    /// deltas line up. Safe on authoritative sims too (used by some
    /// tests), but not normally called there.
    pub fn apply_external_snapshot(&mut self, body: SnapshotBody, tick: u64) {
        let mut world = World::new();
        let mut clock = body.clock;
        clock.tick = tick;
        world.insert_resource(clock);
        world.insert_resource(body.region_graph);
        world.insert_resource(body.world_time);
        world.insert_resource(body.weather);
        world.insert_resource(body.region_control);
        world.insert_resource(body.chronicle);
        world.insert_resource(body.npc_id_counter);
        world.insert_resource(body.wound_id_counter);
        world.insert_resource(body.effect_id_counter);
        world.insert_resource(body.job_id_counter);
        world.insert_resource(body.projectile_id_counter);
        world.insert_resource(body.container_id_counter);
        world.insert_resource(body.population_targets);
        world.insert_resource(crate::resources::BallisticsConfig::load());
        world.insert_resource(crate::resources::MedConfig::default());
        world.insert_resource(crate::resources::InventoryConfig::default());
        world.insert_resource(crate::behavior_config::BehaviorConfig::load());
        world.insert_resource(PendingDeltas::default());
        world.insert_resource(crate::resources::PendingKillCredits::default());
        world.insert_resource(NpcPositionIndex::default());
        world.insert_resource(NpcSpatialHash::default());
        world.insert_resource(SquadObjectives::default());
        world.insert_resource(crate::resources::GuardPosts::default());
        world.insert_resource(ActiveRegions::default());
        world.insert_resource(crate::resources::BehaviorLog::default());
        world.insert_resource(crate::perception::PerceptionConfig::default());
        world.insert_resource(crate::perception::LosService::default());
        world.insert_resource(crate::items::ItemRegistry::load());
        world.insert_resource(crate::items::RecipeRegistry::load());
        world.insert_resource(crate::items::EquipmentSlotRegistry::load());
        world.insert_resource(crate::names::NameRegistry::load());
        let __items = world.resource::<crate::items::ItemRegistry>().clone();
        world.insert_resource(crate::npc_loadouts::NpcLoadoutRegistry::load(&__items));
        // Preserve mirror marker if we had one (swap to auth mode would
        // go through new(), not here).
        if self
            .world
            .contains_resource::<crate::resources::MirrorMode>()
        {
            world.insert_resource(crate::resources::MirrorMode);
        }
        // Faction registry rebuilt fresh from the canonical TOML;
        // drift + per-player rep come from the host's snapshot so
        // mirror clients see the same rep evolution.
        world.insert_resource(crate::faction::registry::load_default());
        world.insert_resource(body.relation_deltas);
        world.insert_resource(body.player_reputation);
        for se in body.entities {
            spawn_serialized(&mut world, se);
        }
        self.world = world;
    }

    /// Apply a single delta received from the network (host → client
    /// path). Thin wrapper over the internal `apply_delta` routine
    /// used by journal replay; both paths are now externalized.
    pub fn apply_external_delta(&mut self, delta: &WorldDelta) {
        apply_delta(&mut self.world, delta);
    }

    /// Drain the deltas produced during the most recent `tick()`.
    /// Used by the `SimHost` wrapper to broadcast them to clients
    /// (host role) or to feed tests.
    pub fn drain_tick_deltas(&mut self) -> Vec<WorldDelta> {
        std::mem::take(&mut self.last_tick_deltas)
    }

    /// Build and publish a render-facing snapshot of active-region
    /// NPC poses for the current tick. Rotates the 2-slot ring:
    /// the previous `curr` becomes the new `prev`, and the freshly-
    /// built snapshot becomes the new `curr`. Called at the end of
    /// every [`Sim::tick`].
    ///
    /// Snapshots include only NPCs in any region listed by
    /// [`crate::resources::ActiveRegions`] — offline-region NPCs
    /// are frozen by the tier filter and not rendered, so omitting
    /// them saves allocation and iteration cost. Sorted by
    /// `NpcId` for stable iteration + O(log n) lookup via
    /// [`crate::snapshot::SimSnapshot::find`].
    pub(crate) fn publish_snapshot(&mut self, tick: u64) {
        use crate::components::{InRegion, Npc, Position, Rotation};

        let active = self
            .world
            .resource::<crate::resources::ActiveRegions>()
            .clone();
        let mut npcs: Vec<crate::snapshot::NpcSnapshot> = Vec::new();
        let mut q = self
            .world
            .query::<(&Npc, &InRegion, &Position, &Rotation)>();
        for (npc, region, pos, rot) in q.iter(&self.world) {
            if !active.is_active(region.0) {
                continue;
            }
            npcs.push(crate::snapshot::NpcSnapshot {
                id: npc.id,
                region: region.0,
                pos: pos.0,
                yaw: rot.0,
            });
        }
        // Stable sort by id so binary_search_by_key in `find` works
        // and same-seed sims produce byte-identical snapshots.
        npcs.sort_by_key(|s| s.id.0);

        let snap = crate::snapshot::SimSnapshot {
            tick,
            published_at: std::time::Instant::now(),
            npcs,
        };

        // Rotate the ring: [prev, curr] -> [old_curr, new_snap].
        // `std::mem::take` leaves None in the slot we move out of,
        // then we overwrite both. No allocation other than the
        // snapshot itself.
        self.snapshot_ring[0] = self.snapshot_ring[1].take();
        self.snapshot_ring[1] = Some(snap);
    }

    /// The most recently published snapshot, if any. `None` only
    /// before the first tick.
    pub fn current_snapshot(&self) -> Option<&crate::snapshot::SimSnapshot> {
        self.snapshot_ring[1].as_ref()
    }

    /// The two most recent snapshots as `(prev, curr)`. `None` if
    /// fewer than two ticks have run since sim construction (in
    /// which case the renderer should hold position rather than
    /// guess). The renderer computes its lerp alpha from the two
    /// snapshots' `published_at` instants and interpolates per-NPC
    /// pose.
    pub fn snapshot_pair(
        &self,
    ) -> Option<(&crate::snapshot::SimSnapshot, &crate::snapshot::SimSnapshot)> {
        Some((
            self.snapshot_ring[0].as_ref()?,
            self.snapshot_ring[1].as_ref()?,
        ))
    }

    /// Convenience: compute interpolated render poses for active-
    /// region NPCs within `max_dist_m` of `player_pos` in `region`,
    /// at the renderer's `now` wall clock. Returns an empty Vec
    /// when no snapshot pair is available yet (fresh sim, < 2
    /// ticks). The gdext bridge wraps this for per-frame consumption
    /// from GDScript.
    ///
    /// This is the threaded-sim PR B hot path: one call per render
    /// frame, bulk distance-filter + interp on the Rust side, returns
    /// just `(id, pos, yaw)` rows for the renderer to paint onto
    /// dummy nodes. No `NpcView` clone overhead, no per-NPC bridge
    /// crossings.
    pub fn snapshot_interp_npcs_near(
        &self,
        region: crate::region::RegionId,
        player_pos: [f32; 3],
        max_dist_m: f32,
        now: std::time::Instant,
    ) -> Vec<crate::snapshot::NpcInterpPose> {
        let Some((prev, curr)) = self.snapshot_pair() else {
            return Vec::new();
        };
        crate::snapshot::interp_npcs_near(prev, curr, region, player_pos, max_dist_m, now)
    }

    /// Set the mirror's clock tick (usually to the host's tick from
    /// the most recent delta batch). No-op on authoritative sims —
    /// they own their own clock.
    pub fn set_tick_for_mirror(&mut self, tick: u64) {
        if self
            .world
            .contains_resource::<crate::resources::MirrorMode>()
        {
            self.world.resource_mut::<SimClock>().tick = tick;
        }
    }

    /// Apply a client-originated action to the authoritative sim. See
    /// [`crate::action::ActionKind`] for the variant list. Each arm
    /// routes to the existing mutation method; errors propagate back
    /// so the host wrapper can log them (no validation in slice 1).
    pub fn apply_action(&mut self, steam_id: u64, action: ActionKind) -> Result<()> {
        use ActionKind as A;
        match action {
            A::Move { pos, yaw } => self.move_player(steam_id, pos, yaw),
            A::ChangeRegion { region_name } => {
                let region_id = self
                    .regions()
                    .id_for_name(&region_name)
                    .ok_or_else(|| anyhow::anyhow!("unknown region {region_name}"))?;
                self.change_player_region(steam_id, region_id)
            }
            A::ApplyBandage { part } => self.apply_bandage(steam_id, part),
            A::ApplyTourniquet { part } => self.apply_tourniquet(steam_id, part),
            A::RemoveTourniquet { part } => self.remove_tourniquet(steam_id, part),
            A::ApplyDisinfectant { part } => self.apply_disinfectant(steam_id, part),
            A::ApplyStitch { part } => self.apply_stitch(steam_id, part),
            A::ApplyWoundPack { part } => self.apply_wound_pack(steam_id, part),
            A::ApplyAntibiotics => self.apply_antibiotics(steam_id),
            A::ApplyDrug { drug } => self.apply_drug(steam_id, drug).map(|_| ()),
            A::Eat { kind } => self.eat(steam_id, kind),
            A::Drink { kind } => self.drink(steam_id, kind),
            A::ConsumeSlot {
                slot_idx,
                body_part,
            } => self.consume_from_slot(steam_id, slot_idx as usize, body_part),
            A::DropSlot { slot_idx } => self.drop_item(steam_id, slot_idx as usize),
            A::MoveSlot { from, to } => {
                self.move_between_slots(steam_id, from as usize, to as usize)
            }
            A::MoveBetweenGrids {
                from_grid,
                from_idx,
                to_grid,
            } => self.move_between_grids(steam_id, &from_grid, from_idx as usize, &to_grid),
            A::SalvageSlot { slot_idx } => self.salvage(steam_id, slot_idx as usize).map(|_| ()),
            A::CraftRecipe { recipe_id } => self.craft(steam_id, &recipe_id),
            A::SetNearCampfire { value } => self.set_player_near_campfire(steam_id, value),
            A::SetNearWorkbench { tier } => self.set_player_near_workbench(steam_id, tier),
            A::QueueCraft { recipe_id, count } => {
                self.queue_craft(steam_id, &recipe_id, count).map(|_| ())
            }
            A::CancelCraft { job_id } => self.cancel_craft(steam_id, job_id),
            A::Equip {
                slot_id,
                source_grid,
                source_idx,
            } => self.equip(
                steam_id,
                &crate::items::SlotId::from(slot_id),
                &source_grid,
                source_idx as usize,
            ),
            A::Unequip { slot_id, dest_grid } => {
                self.unequip(steam_id, &crate::items::SlotId::from(slot_id), &dest_grid)
            }
            A::HotbarConsume { idx, body_part } => {
                self.consume_from_hotbar(steam_id, idx, body_part)
            }
            A::ReloadWeapon { slot_id } => {
                self.reload_weapon(steam_id, &crate::items::SlotId::from(slot_id))
            }
            A::EjectMagazine { slot_id } => {
                self.eject_magazine(steam_id, &crate::items::SlotId::from(slot_id))
            }
            A::FireWeapon {
                slot_id,
                aim_yaw,
                aim_pitch,
            } => self
                .fire_weapon(
                    steam_id,
                    &crate::items::SlotId::from(slot_id),
                    aim_yaw,
                    aim_pitch,
                )
                .map(|_| ()),
            A::LoadRoundsIntoMag { slot_id, round_id } => self
                .load_rounds_into_mag(
                    steam_id,
                    &crate::items::SlotId::from(slot_id),
                    &ItemId::from(round_id),
                )
                .map(|_| ()),
            A::LoadRoundsIntoPocketMag {
                pocket_idx,
                round_id,
            } => self
                .load_rounds_into_pocket_mag(steam_id, pocket_idx, &ItemId::from(round_id))
                .map(|_| ()),
            A::GrantItem { item_id, count } => {
                self.grant_item(steam_id, &ItemId::from(item_id), count)
            }
            A::TakeFromContainer {
                container_id,
                source_idx,
            } => self.take_from_container(
                steam_id,
                crate::components::ContainerId(container_id),
                source_idx as usize,
            ),
            A::PutInContainer {
                container_id,
                source_grid,
                source_idx,
            } => self.put_in_container(
                steam_id,
                crate::components::ContainerId(container_id),
                &source_grid,
                source_idx as usize,
            ),
        }
    }
}
