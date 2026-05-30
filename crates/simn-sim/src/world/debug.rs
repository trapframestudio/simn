//! Debug tunables and test-only Sim APIs.
//!
//! Two kinds of things live here:
//!
//! - **Debug tunables** — knobs production code may legitimately
//!   touch (`set_behavior_log`, `set_population_target`,
//!   `scale_all_population_targets`). Used by in-game debug
//!   panels and by tests.
//! - **`*_for_test` helpers** — `#[doc(hidden)]` fast-paths for
//!   tests that would otherwise need to tick the sim forward by
//!   thousands of ticks. Not part of the public game-facing API;
//!   kept public only because integration tests live outside the
//!   crate.
//!
//! Moving these out of `world/mod.rs` keeps the main API surface
//! readable — production callers rarely need to scroll past
//! test-only scaffolding.

use crate::chronicle::{DeathCause, LifeChronicle, LifeRecord};
use crate::components::{
    Actor, ActorKind, Health, InRegion, Lifespan, Npc, NpcGoal, NpcId, Position, Rotation,
};
use crate::delta::WorldDelta;
use crate::region::RegionId;
use crate::resources::{NpcIdCounter, PopulationTargets, SimClock, WorldTime};

use super::{find_npc_in, Sim};

impl Sim {
    /// Toggle the behavior-logging flag. When on, NPC systems emit
    /// `tracing::info!(target: "npc.behavior", ...)` events. We
    /// anchor `last_flush_tick` to the current tick on enable so
    /// the first summary arrives one full `FLUSH_INTERVAL` after
    /// toggling, not immediately (which would then go silent for
    /// another full interval). Counters reset too so the first
    /// summary reflects only behavior observed since the toggle.
    pub fn set_behavior_log(&mut self, enabled: bool) {
        let tick = self.world.resource::<SimClock>().tick;
        let mut log = self.world.resource_mut::<crate::resources::BehaviorLog>();
        log.enabled = enabled;
        if enabled {
            log.last_flush_tick = tick;
            log.reset_counters();
        }
    }

    pub fn behavior_log_enabled(&self) -> bool {
        self.world
            .resource::<crate::resources::BehaviorLog>()
            .enabled
    }

    /// Override a single region/faction population target at runtime.
    /// Takes effect on the next spawn pass (~50 ticks). Used by the
    /// debug density control in Godot and by tests.
    pub fn set_population_target(&mut self, region: RegionId, faction: &str, count: u32) {
        self.world
            .resource_mut::<PopulationTargets>()
            .set(region, faction, count);
    }

    /// Convenience alias used by tests.
    #[doc(hidden)]
    pub fn set_population_target_for_test(&mut self, region: RegionId, faction: &str, count: u32) {
        self.set_population_target(region, faction, count);
    }

    /// Test-only: mutable world handle for tests that need a
    /// `&mut World` (e.g. running an ad-hoc `query::<...>()` loop
    /// to assert against multiple component combinations). Used
    /// by Phase 3C's loot-restock tests; prefer the higher-level
    /// `*_for_test` enumerators when they suffice.
    #[doc(hidden)]
    pub fn world_for_test(&mut self) -> &mut bevy_ecs::prelude::World {
        &mut self.world
    }

    /// Test-only: enumerate every `WorldContainer` in the world as
    /// `(id, region, position, is_public, item_count)` tuples. Used
    /// by Phase 3A's loot-scatter tests to verify seed produces the
    /// expected count + placement pattern per region without
    /// reaching into private fields.
    #[doc(hidden)]
    pub fn all_world_containers_for_test(
        &mut self,
    ) -> Vec<(
        crate::components::ContainerId,
        RegionId,
        [f32; 3],
        bool,
        usize,
    )> {
        use crate::components::{InRegion, Position, WorldContainer};
        let mut out = Vec::new();
        let mut q = self
            .world
            .query::<(&WorldContainer, &Position, &InRegion)>();
        for (wc, p, r) in q.iter(&self.world) {
            out.push((wc.id, r.0, p.0, wc.is_public, wc.grid.items.len()));
        }
        out
    }

    /// Test-only: enumerate every authored `Base` as
    /// `(region, position)` pairs. Pairs with
    /// [`all_world_containers_for_test`] for proximity assertions
    /// in Phase 3A.
    #[doc(hidden)]
    pub fn all_bases_for_test(&mut self) -> Vec<(RegionId, [f32; 3])> {
        use crate::components::{Base, InRegion, Position};
        let mut out = Vec::new();
        let mut q = self.world.query::<(&Base, &Position, &InRegion)>();
        for (_, p, r) in q.iter(&self.world) {
            out.push((r.0, p.0));
        }
        out
    }

    /// Test-only: count `OfflineNpc` entities in the world. Used by
    /// projection round-trip tests to verify online ↔ offline schema
    /// flips don't leak entities.
    #[doc(hidden)]
    pub fn offline_npc_count_for_test(&mut self) -> usize {
        self.world
            .query::<&crate::offline_tier::OfflineNpc>()
            .iter(&self.world)
            .count()
    }

    /// Test-only: count `OfflineNpc` entities in `region`.
    #[doc(hidden)]
    pub fn offline_npc_count_in_region_for_test(&mut self, region: RegionId) -> usize {
        self.world
            .query::<&crate::offline_tier::OfflineNpc>()
            .iter(&self.world)
            .filter(|o| o.region == region)
            .count()
    }

    /// Test-only: clone the `OfflineNpc` carrying `id`, if any. Used
    /// by projection tests to assert preservation of faction /
    /// health-class / group across the boundary.
    #[doc(hidden)]
    pub fn offline_npc_for_test(
        &mut self,
        id: crate::components::NpcId,
    ) -> Option<crate::offline_tier::OfflineNpc> {
        let mut query = self.world.query::<&crate::offline_tier::OfflineNpc>();
        query.iter(&self.world).find(|o| o.id == id).cloned()
    }

    /// Test-only: damage a specific body part on `id`. Wraps the
    /// component-level setter so projection tests can produce a
    /// known `Wounded` / `Critical` state before flipping the
    /// region tier.
    #[doc(hidden)]
    pub fn set_npc_body_part_for_test(
        &mut self,
        id: crate::components::NpcId,
        part: crate::components::BodyPart,
        value: f32,
    ) {
        if let Some(e) = find_npc_in(&mut self.world, id) {
            if let Some(mut bp) = self.world.get_mut::<crate::components::BodyParts>(e) {
                match part {
                    crate::components::BodyPart::Head => bp.head = value,
                    crate::components::BodyPart::Torso => bp.torso = value,
                    crate::components::BodyPart::LeftArm => bp.left_arm = value,
                    crate::components::BodyPart::RightArm => bp.right_arm = value,
                    crate::components::BodyPart::LeftLeg => bp.left_leg = value,
                    crate::components::BodyPart::RightLeg => bp.right_leg = value,
                }
            }
        }
    }

    /// Test-only: mark every region in the current graph as "active"
    /// so `spawn_npcs` will top up populations everywhere on each
    /// tick. Production code never calls this; it's the test
    /// analogue of having a player simultaneously present in every
    /// region. Without this, tests that scale targets and tick
    /// expecting NPCs to spawn naturally find zero — because the
    /// post-Phase-1A spawn gate skips inactive regions and a fresh
    /// `Sim` starts with no active regions.
    #[doc(hidden)]
    pub fn activate_all_regions_for_test(&mut self) {
        let region_ids: Vec<RegionId> = self
            .world
            .resource::<crate::region::RegionGraph>()
            .regions
            .keys()
            .copied()
            .collect();
        let mut ar = self.world.resource_mut::<crate::resources::ActiveRegions>();
        for rid in region_ids {
            ar.regions.insert(rid);
        }
    }

    /// Test-only: directly spawn an `OfflineNpc` of `faction` at
    /// `pos_2d` in `region`. Returns the assigned `NpcId`. Bypasses
    /// projection (which goes online→offline) — useful for movement
    /// tests that want to seed offline NPCs into a region without
    /// first round-tripping through `set_active_region`.
    #[doc(hidden)]
    pub fn spawn_offline_npc_for_test(
        &mut self,
        faction: &str,
        region: RegionId,
        pos_2d: [f32; 2],
    ) -> crate::components::NpcId {
        let id = self.world.resource_mut::<NpcIdCounter>().mint();
        let now = self.current_tick();
        let faction_id = self
            .world
            .resource::<crate::faction::registry::FactionRegistry>()
            .id_of(faction)
            .unwrap_or_else(|| panic!("registry has no faction {faction:?}"));
        let stats = {
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(id.0);
            crate::components::NpcStats::roll(&mut rng, 0.5)
        };
        // Chronicle entry — every NPC (online or offline) has one.
        // Without this, downstream death-event paths
        // (`offline_combat`'s `chronicle.get_mut(id)`) can't record
        // the death and tests that count chronicle entries miss it.
        self.world
            .resource_mut::<crate::chronicle::LifeChronicle>()
            .insert(crate::chronicle::LifeRecord {
                id,
                faction: faction.to_string(),
                birth_tick: now,
                birth_region: region,
                birth_pos: [pos_2d[0], 0.0, pos_2d[1]],
                death_tick: None,
                death_region: None,
                death_cause: None,
                regions_visited: vec![(region, now)],
            });
        let offline = crate::offline_tier::OfflineNpc {
            id,
            region,
            position_2d: pos_2d,
            faction: faction_id,
            group: None,
            health_class: crate::offline_tier::HealthClass::Healthy,
            loadout_class: crate::offline_tier::LoadoutClass::Standard {
                faction: faction_id,
                tier: 1,
            },
            personality_seed: crate::components::NpcCharacter::derive_id(id, faction_id).0,
            stats,
            combat_state: crate::offline_tier::OfflineCombatState::Idle,
            die_at_tick: u64::MAX,
            target_2d: None,
            arrival_offline_tick: 0,
            travel_start_offline_tick: 0,
            travel_start_2d: pos_2d,
            waypoint_chain: Vec::new(),
            waypoint_chain_idx: 0,
            aggro_target: None,
            aggro_last_seen_tick: 0,
        };
        self.world.spawn(offline);
        id
    }

    /// Test-only: iterate every `OfflineNpc` in the world. Yields
    /// `(NpcId, region, position_2d)` per entry. Parallels
    /// `each_npc` (which only sees *online* NPCs) so tests
    /// targeting the offline tier don't get spuriously empty
    /// results because their region happens to be inactive.
    #[doc(hidden)]
    pub fn each_offline_npc_for_test<F: FnMut(crate::components::NpcId, RegionId, [f32; 2])>(
        &mut self,
        mut f: F,
    ) {
        let mut q = self.world.query::<&crate::offline_tier::OfflineNpc>();
        for o in q.iter(&self.world) {
            f(o.id, o.region, o.position_2d);
        }
    }

    /// Test-only: set an `OfflineNpc`'s `health_class`. Used by
    /// offline-tier combat tests to put a victim at `Critical` so a
    /// single combat hit lands the kill, parallel to
    /// `force_npc_hp_for_test` for online NPCs. Returns true if the
    /// id resolves to an offline NPC.
    #[doc(hidden)]
    /// Iteration 5-13 Phase C2: directly set an `OfflineNpc`'s
    /// movement target. Bypasses `pick_offline_target` so a test
    /// can exercise the chain-resolution / segment-walking branch
    /// without needing to set up bases or run the offline tier
    /// long enough for a target to be picked organically. The
    /// next `offline_movement` tick observes the target and
    /// resolves a fresh waypoint chain through the region's graph.
    #[doc(hidden)]
    pub fn set_offline_target_for_test(
        &mut self,
        id: crate::components::NpcId,
        target: Option<[f32; 2]>,
    ) -> bool {
        let mut q = self.world.query::<(
            bevy_ecs::entity::Entity,
            &mut crate::offline_tier::OfflineNpc,
        )>();
        for (_, mut o) in q.iter_mut(&mut self.world) {
            if o.id == id {
                // Reset travel anchors so the next tick re-enters
                // the "target set, no progress yet" state and the
                // waypoint chain resolves cleanly. Without the
                // reset, a previously-arrived NPC's stale
                // arrival_offline_tick would short-circuit movement.
                o.target_2d = target;
                o.travel_start_2d = o.position_2d;
                o.travel_start_offline_tick = 0;
                o.arrival_offline_tick = 0;
                o.waypoint_chain.clear();
                o.waypoint_chain_idx = 0;
                return true;
            }
        }
        false
    }

    /// Iteration 5-13 Phase C2: read an `OfflineNpc`'s current
    /// `position_2d`. Returns `None` if `id` doesn't resolve to
    /// an offline NPC in the world. Used by waypoint-routing
    /// tests to trace the NPC's path.
    #[doc(hidden)]
    pub fn offline_npc_position_for_test(
        &mut self,
        id: crate::components::NpcId,
    ) -> Option<[f32; 2]> {
        let mut q = self.world.query::<&crate::offline_tier::OfflineNpc>();
        q.iter(&self.world)
            .find(|o| o.id == id)
            .map(|o| o.position_2d)
    }

    pub fn force_offline_health_class_for_test(
        &mut self,
        id: crate::components::NpcId,
        class: crate::offline_tier::HealthClass,
    ) -> bool {
        let mut q = self.world.query::<(
            bevy_ecs::entity::Entity,
            &mut crate::offline_tier::OfflineNpc,
        )>();
        for (_, mut o) in q.iter_mut(&mut self.world) {
            if o.id == id {
                o.health_class = class;
                return true;
            }
        }
        false
    }

    /// Scale every existing population target by `factor`. A factor
    /// of 0.5 halves all targets; 2.0 doubles them. Clamped to
    /// [0, 9999] per entry. Used by the in-game density control.
    pub fn scale_all_population_targets(&mut self, factor: f32) {
        let mut targets = self.world.resource_mut::<PopulationTargets>();
        for by_fac in targets.by_region.values_mut() {
            for count in by_fac.values_mut() {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let new = ((*count as f32) * factor).round().clamp(0.0, 9999.0) as u32;
                *count = new;
            }
        }
    }

    /// Test-only: force an NPC's `die_at_tick` to (now + offset_ticks).
    #[doc(hidden)]
    pub fn force_lifespan_for_test(&mut self, id: NpcId, offset_ticks: u64) {
        let now = self.current_tick();
        if let Some(e) = find_npc_in(&mut self.world, id) {
            if let Some(mut l) = self.world.get_mut::<Lifespan>(e) {
                l.die_at_tick = now.wrapping_add(offset_ticks);
            }
        }
    }

    /// Test-only: read an NPC's current goal.
    #[doc(hidden)]
    pub fn npc_goal_for_test(&mut self, id: NpcId) -> Option<NpcGoal> {
        let e = find_npc_in(&mut self.world, id)?;
        self.world.get::<NpcGoal>(e).copied()
    }

    /// Test-only: read the resolver-picked [`ActiveGoal`] for an NPC.
    #[doc(hidden)]
    pub fn npc_active_goal_for_test(&mut self, id: NpcId) -> Option<crate::components::ActiveGoal> {
        let e = find_npc_in(&mut self.world, id)?;
        self.world.get::<crate::components::ActiveGoal>(e).copied()
    }

    /// Test-only: read the registry-keyed `InFaction` set at spawn.
    /// During migration (steps 3-5) NPCs carry this in parallel with
    /// the legacy [`crate::components::InFaction`].
    #[doc(hidden)]
    /// Test-only: directly set an NPC's `Aggro.target`. Bypasses
    /// `npc_aggro`'s perception pass so integration tests can stage
    /// the squad-target-switching scenario without running ticks of
    /// the perception pipeline. Returns `true` if the entity was
    /// found.
    #[doc(hidden)]
    pub fn set_npc_aggro_for_test(&mut self, victim: NpcId, target: NpcId) -> bool {
        let Some(e) = find_npc_in(&mut self.world, victim) else {
            return false;
        };
        let now = self.current_tick();
        self.world.entity_mut(e).insert(crate::components::Aggro {
            target,
            last_seen_tick: now,
        });
        true
    }

    /// Test-only: directly push a damage event onto an NPC's
    /// `RecentAttackers` ring. Bypasses the full combat pipeline so
    /// integration tests can stage threat-board scenarios without
    /// running ticks of `npc_combat`. Returns `true` if the entity
    /// was found and the hit recorded.
    #[doc(hidden)]
    pub fn record_npc_hit_for_test(
        &mut self,
        victim: NpcId,
        attacker: NpcId,
        tick: u64,
        damage: f32,
    ) -> bool {
        let Some(e) = find_npc_in(&mut self.world, victim) else {
            return false;
        };
        let Some(mut recent) = self.world.get_mut::<crate::components::RecentAttackers>(e) else {
            return false;
        };
        recent.record(
            attacker,
            tick,
            damage,
            crate::systems::npc_combat::MAX_RECENT_ATTACKERS,
        );
        true
    }

    pub fn npc_in_faction_for_test(&mut self, id: NpcId) -> Option<crate::components::InFaction> {
        let e = find_npc_in(&mut self.world, id)?;
        self.world.get::<crate::components::InFaction>(e).copied()
    }

    /// Test-only: spawn one NPC of `faction` (registry name) at `pos`
    /// in `region`, optionally as part of `group`. Returns the new
    /// NpcId. The NPC gets an empty `Inventory` — tests that need
    /// loadout contents should call `grant_to_npc_for_test` afterwards.
    #[doc(hidden)]
    pub fn spawn_npc_for_test(
        &mut self,
        faction: &str,
        region: RegionId,
        pos: [f32; 3],
        group: Option<u64>,
    ) -> NpcId {
        let id = self.world.resource_mut::<NpcIdCounter>().mint();
        let now = self.current_tick();
        let faction_id = self
            .world
            .resource::<crate::faction::registry::FactionRegistry>()
            .id_of(faction)
            .unwrap_or_else(|| panic!("registry has no faction {faction:?}"));
        let base_aggression = self
            .world
            .resource::<crate::faction::registry::FactionRegistry>()
            .def(faction_id)
            .base_aggression;
        let bundle = (
            Npc { id },
            Actor {
                kind: ActorKind::Npc,
            },
            crate::components::InFaction(faction_id),
            InRegion(region),
            Position(pos),
            Rotation(0.0),
            Health::new_full(),
            crate::components::BodyParts::new_full(),
            crate::components::LimbStates::default(),
            crate::components::Wounds::default(),
            crate::components::ActiveEffects::default(),
            NpcGoal::Idle {
                until_tick: now + 10,
            },
            Lifespan {
                spawned_tick: now,
                die_at_tick: now.wrapping_add(1_000_000),
            },
            crate::components::Aggression(base_aggression),
            crate::components::RecentAttackers::default(),
        );
        let inv = crate::components::Inventory(crate::components::GridInventory::player_default());
        let active = crate::components::ActiveGoal::default();
        let (archetype, nat_weights, male_w) = {
            let def = self
                .world
                .resource::<crate::faction::registry::FactionRegistry>()
                .def(faction_id);
            (
                def.archetype,
                def.nationality_weights.clone(),
                def.male_name_weight,
            )
        };
        let character = {
            let names = self.world.resource::<crate::names::NameRegistry>();
            crate::components::NpcCharacter::roll(
                id,
                faction_id,
                archetype,
                base_aggression,
                names,
                &nat_weights,
                male_w,
            )
        };
        if let Some(group_id) = group {
            self.world.spawn((
                bundle,
                inv,
                active,
                character,
                crate::components::Group { id: group_id },
            ));
        } else {
            self.world.spawn((bundle, inv, active, character));
        }
        self.world
            .resource_mut::<LifeChronicle>()
            .insert(LifeRecord {
                id,
                faction: faction.to_string(),
                birth_tick: now,
                birth_region: region,
                birth_pos: pos,
                death_tick: None,
                death_region: None,
                death_cause: None,
                regions_visited: vec![(region, now)],
            });
        id
    }

    /// Test-only: read an NPC's pocket grid as a flat stack list.
    /// Returns `None` if the NPC has no `Inventory` component (e.g.
    /// loaded from a pre-PR-4b snapshot). Used by corpse-loot tests.
    #[doc(hidden)]
    pub fn npc_inventory_view_for_test(
        &mut self,
        id: NpcId,
    ) -> Option<Vec<crate::components::ItemInstance>> {
        let e = find_npc_in(&mut self.world, id)?;
        let inv = self.world.get::<crate::components::Inventory>(e)?;
        let mut totals: std::collections::HashMap<crate::items::ItemId, u32> =
            std::collections::HashMap::new();
        for placed in &inv.0.items {
            *totals.entry(placed.stack.id.clone()).or_default() += placed.stack.count;
        }
        Some(
            totals
                .into_iter()
                .map(|(id, count)| crate::components::ItemInstance {
                    id,
                    count,
                    spawned_tick: 0,
                    magazine_state: None,
                })
                .collect(),
        )
    }

    /// Test-only: drop a stack into an NPC's pockets via
    /// `inventory_grid::grant_or_merge`. Used by PR-4b corpse tests
    /// to seed gear without going through the loadout RNG.
    #[doc(hidden)]
    pub fn grant_to_npc_for_test(
        &mut self,
        id: NpcId,
        item: &crate::items::ItemId,
        count: u32,
    ) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let registry = self.world.resource::<crate::items::ItemRegistry>().clone();
        let Some(mut inv) = self.world.get_mut::<crate::components::Inventory>(e) else {
            return false;
        };
        crate::inventory_grid::grant_or_merge(&mut inv.0, &registry, item, count, 0).is_ok()
    }

    /// Test-only: force an NPC's HP to a specific value. Drains the
    /// torso pool to match (NPCs now carry `BodyParts`; `npc_combat`
    /// reads the torso value on each tick and would otherwise re-raise
    /// `Health.current` back to `min(head, torso)` on the next pass).
    #[doc(hidden)]
    pub fn force_npc_hp_for_test(&mut self, id: NpcId, hp: f32) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        // Phase 4A v2: floor *every* body part to `hp` so a single
        // projectile hit — to torso, head, or any limb — kills.
        // Pre-4A v2 we only had to floor torso because the dice
        // damage path applied all hits there. The projectile tick
        // picks a body part geometrically.
        if let Some(mut bp) = self.world.get_mut::<crate::components::BodyParts>(e) {
            let v = hp.max(0.0);
            bp.head = v;
            bp.torso = v;
            bp.left_arm = v;
            bp.right_arm = v;
            bp.left_leg = v;
            bp.right_leg = v;
        }
        if let Some(mut h) = self.world.get_mut::<Health>(e) {
            h.current = hp;
            true
        } else {
            false
        }
    }

    /// Test-only: read an NPC's `Aggro` state, if any.
    #[doc(hidden)]
    pub fn npc_aggro_for_test(&mut self, id: NpcId) -> Option<crate::components::Aggro> {
        let e = find_npc_in(&mut self.world, id)?;
        self.world.get::<crate::components::Aggro>(e).copied()
    }

    /// Test-only: snapshot an NPC's current `Wounds` component, if any.
    /// Returns `None` when the entity is missing entirely; empty `Vec`
    /// when the NPC has the component but no wounds.
    #[doc(hidden)]
    pub fn npc_wounds_for_test(&mut self, id: NpcId) -> Option<crate::components::Wounds> {
        let e = find_npc_in(&mut self.world, id)?;
        self.world.get::<crate::components::Wounds>(e).cloned()
    }

    /// Test-only: read an NPC's `LimbStates` component, if any.
    #[doc(hidden)]
    pub fn npc_limb_states_for_test(&mut self, id: NpcId) -> Option<crate::components::LimbStates> {
        let e = find_npc_in(&mut self.world, id)?;
        self.world.get::<crate::components::LimbStates>(e).copied()
    }

    /// Test-only: read an NPC's `NpcCharacter` (identity + stats).
    #[doc(hidden)]
    pub fn npc_character_for_test(&mut self, id: NpcId) -> Option<crate::components::NpcCharacter> {
        let e = find_npc_in(&mut self.world, id)?;
        self.world
            .get::<crate::components::NpcCharacter>(e)
            .cloned()
    }

    /// Test-only: overwrite the `perception` stat on an NPC's
    /// `NpcCharacter`. Returns `true` if the entity exists and the
    /// component was updated.
    #[doc(hidden)]
    pub fn set_npc_perception_for_test(&mut self, id: NpcId, perception: u8) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let Some(mut character) = self.world.get_mut::<crate::components::NpcCharacter>(e) else {
            return false;
        };
        character.stats.perception = perception;
        true
    }

    /// Test-only: overwrite the `endurance` stat on an NPC's
    /// `NpcCharacter`. Returns `true` if the entity exists and the
    /// component was updated.
    #[doc(hidden)]
    pub fn set_npc_endurance_for_test(&mut self, id: NpcId, endurance: u8) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let Some(mut character) = self.world.get_mut::<crate::components::NpcCharacter>(e) else {
            return false;
        };
        character.stats.endurance = endurance;
        true
    }

    /// Test-only: overwrite the `leadership` stat on an NPC's
    /// `NpcCharacter`. Returns `true` if the entity exists and the
    /// component was updated.
    #[doc(hidden)]
    pub fn set_npc_leadership_for_test(&mut self, id: NpcId, leadership: u8) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let Some(mut character) = self.world.get_mut::<crate::components::NpcCharacter>(e) else {
            return false;
        };
        character.stats.leadership = leadership;
        true
    }

    /// Test-only: clear all personality traits on an NPC's
    /// `NpcCharacter`. Used by goal-arbitration tests that need a
    /// deterministic "no introduced goals" baseline (otherwise the
    /// archetype roll plus personality bias can nominate a
    /// `PersonalityBias` candidate instead of falling back to
    /// `Idle`).
    #[doc(hidden)]
    pub fn clear_npc_personality_for_test(&mut self, id: NpcId) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let Some(mut character) = self.world.get_mut::<crate::components::NpcCharacter>(e) else {
            return false;
        };
        character.personality = crate::components::PersonalityTraits::default();
        true
    }

    /// Test-only: overwrite the personality traits on an NPC's
    /// `NpcCharacter`. Used by personality-driven goal tests that need
    /// a specific trait set (e.g. Socialize requires `social: true`).
    #[doc(hidden)]
    pub fn set_npc_personality_for_test(
        &mut self,
        id: NpcId,
        personality: crate::components::PersonalityTraits,
    ) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let Some(mut character) = self.world.get_mut::<crate::components::NpcCharacter>(e) else {
            return false;
        };
        character.personality = personality;
        true
    }

    /// Test-only: overwrite the `accuracy` stat on an NPC's
    /// `NpcCharacter`. Returns `true` if the entity exists and the
    /// component was updated.
    #[doc(hidden)]
    pub fn set_npc_accuracy_for_test(&mut self, id: NpcId, accuracy: u8) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let Some(mut character) = self.world.get_mut::<crate::components::NpcCharacter>(e) else {
            return false;
        };
        character.stats.accuracy = accuracy;
        true
    }

    /// Test-only: read a player's `LimbStates` component, if any.
    #[doc(hidden)]
    pub fn player_limb_states_for_test(
        &mut self,
        steam_id: u64,
    ) -> Option<crate::components::LimbStates> {
        let e = self.find_player_entity(steam_id)?;
        self.world.get::<crate::components::LimbStates>(e).copied()
    }

    /// Test-only: sever the named body part on an NPC. Flips the
    /// `LimbStates` entry to `Severed`, zeroes the `BodyParts` slot,
    /// and updates aggregate `Health` from the new vital_min.
    /// Severing head or torso drives the death gate; severing a limb
    /// disables it without killing. The full caliber-driven sever
    /// path lands with the projectile + wound-kind pipeline; this is
    /// the test-side shortcut.
    #[doc(hidden)]
    pub fn sever_limb_for_test(&mut self, id: NpcId, part: crate::components::BodyPart) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let vital_min = {
            let Some(mut bp) = self.world.get_mut::<crate::components::BodyParts>(e) else {
                return false;
            };
            *bp.get_mut(part) = 0.0;
            bp.vital_min()
        };
        if let Some(mut states) = self.world.get_mut::<crate::components::LimbStates>(e) {
            states.mark_severed(part);
        }
        if let Some(mut h) = self.world.get_mut::<Health>(e) {
            h.current = vital_min.min(h.max);
        }
        true
    }

    /// Test-only: check whether the NPC carries an `ActiveEffects`
    /// component. Used to verify the humanoid spawn bundle wired it up.
    #[doc(hidden)]
    pub fn npc_has_active_effects_for_test(&mut self, id: NpcId) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        self.world
            .get::<crate::components::ActiveEffects>(e)
            .is_some()
    }

    /// Test-only: read the current `SquadObjective` for a group id.
    #[doc(hidden)]
    pub fn squad_objective_for_test(
        &self,
        group_id: u64,
    ) -> Option<crate::resources::SquadObjective> {
        self.world
            .resource::<crate::resources::SquadObjectives>()
            .by_group
            .get(&group_id)
            .map(|s| s.objective.clone())
    }

    /// Test-only: forcibly inject a `SquadObjective` for a group id,
    /// bypassing the planner. Used by personality-goal tests that need
    /// a specific squad objective (e.g. Socialize requires the squad
    /// to be at Rest, but faction weights may otherwise pick Guard).
    #[doc(hidden)]
    pub fn set_squad_objective_for_test(
        &mut self,
        group_id: u64,
        objective: crate::resources::SquadObjective,
    ) {
        let tick = self.world.resource::<crate::resources::SimClock>().tick;
        let mut objectives = self
            .world
            .resource_mut::<crate::resources::SquadObjectives>();
        objectives.by_group.insert(
            group_id,
            crate::resources::SquadObjectiveState {
                objective,
                set_at_tick: tick,
                recently_visited: Default::default(),
                disperse_target: None,
                last_progress_pos: None,
                last_progress_tick: tick,
                wander_drift_target: None,
                last_stuck_kind: None,
                cohesion_break_disabled_until: 0,
                arrived_at_tick: None,
                last_regroup_exit_tick: None,
                last_drift_heading: None,
            },
        );
    }

    /// Test-only: forcibly set a squad's objective expiry to `tick`.
    /// If the squad currently holds a posted `Guard` (i.e.
    /// `post_key.is_some()`), also vacate the post — posted guards
    /// ignore `expires_at` by design and won't re-roll until
    /// someone else takes the post or the squad dies, so just
    /// zeroing the expiry isn't enough.
    #[doc(hidden)]
    pub fn force_squad_objective_expiry_for_test(&mut self, group_id: u64, tick: u64) {
        // Pull the post key out if the current objective is a
        // posted guard, so we can vacate it after the borrow scope.
        let post_to_clear = self
            .world
            .resource::<crate::resources::SquadObjectives>()
            .by_group
            .get(&group_id)
            .and_then(|s| match &s.objective {
                crate::resources::SquadObjective::Guard {
                    post_key: Some(key),
                    ..
                } => Some(*key),
                _ => None,
            });
        if let Some(state) = self
            .world
            .resource_mut::<crate::resources::SquadObjectives>()
            .by_group
            .get_mut(&group_id)
        {
            use crate::resources::SquadObjective::*;
            match &mut state.objective {
                Patrol { expires_at, .. }
                | Guard { expires_at, .. }
                | Rest { expires_at, .. }
                | Investigate { expires_at, .. }
                | Explore { expires_at, .. }
                | Relieve { expires_at, .. }
                | Wander { expires_at }
                | Regroup { expires_at, .. } => *expires_at = tick,
            }
            // The test helper forces a re-roll; the first-spawn
            // dispersion gate would otherwise hold the squad in
            // Wander until the centroid walked 60–120 m. Clear it
            // so the planner treats this squad as a normal
            // expired-objective candidate.
            state.disperse_target = None;
        }
        // If the objective was a posted Guard, vacate the post too
        // so the planner's `needs_new` check fires (posted guards
        // ignore expires_at).
        if let Some(key) = post_to_clear {
            let mut posts = self.world.resource_mut::<crate::resources::GuardPosts>();
            if posts
                .by_key
                .get(&key)
                .is_some_and(|info| info.group_id == group_id)
            {
                posts.by_key.remove(&key);
            }
        }
    }

    /// Test-only: move an NPC's position directly (e.g. to relocate
    /// a target outside sight without ticking).
    #[doc(hidden)]
    pub fn move_npc_for_test(&mut self, id: NpcId, pos: [f32; 3], region: RegionId) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        if let Some(mut p) = self.world.get_mut::<Position>(e) {
            p.0 = pos;
        }
        if let Some(mut r) = self.world.get_mut::<InRegion>(e) {
            r.0 = region;
        }
        true
    }

    /// Test-only: set an NPC's yaw directly. Used by aggro tests
    /// that need NPCs facing each other so the FOV check passes.
    #[doc(hidden)]
    pub fn set_npc_yaw_for_test(&mut self, id: NpcId, yaw: f32) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        if let Some(mut r) = self.world.get_mut::<Rotation>(e) {
            r.0 = yaw;
            return true;
        }
        false
    }

    /// Test-only: read an NPC's current yaw without going through the
    /// full view layer. Returns `None` if the NPC was not found.
    #[doc(hidden)]
    pub fn npc_yaw_for_test(&mut self, id: NpcId) -> Option<f32> {
        let e = find_npc_in(&mut self.world, id)?;
        self.world.get::<Rotation>(e).map(|r| r.0)
    }

    /// Test-only: force-kill an NPC with the given cause. Mirrors the
    /// production death gate: chronicle write, **corpse container
    /// spawn** (private, holds the NPC's pockets if non-empty), then
    /// despawn + journal `NpcDied`. Used by chronicle and corpse-loot
    /// tests.
    #[doc(hidden)]
    pub fn kill_npc_for_test(&mut self, id: NpcId, cause: DeathCause) -> bool {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return false;
        };
        let region = self.world.get::<InRegion>(e).map(|r| r.0).unwrap_or(0);
        let pos = self
            .world
            .get::<crate::components::Position>(e)
            .map(|p| p.0)
            .unwrap_or([0.0; 3]);
        let inventory_grid = self
            .world
            .get::<crate::components::Inventory>(e)
            .map(|inv| inv.0.clone());
        let tick = self.current_tick();
        self.world
            .resource_mut::<LifeChronicle>()
            .mark_dead(id, tick, region, cause.clone());
        // Spawn corpse container BEFORE despawning the NPC entity, so
        // the new container's id mint and delta land in the same tick
        // as the `NpcDied`. Skips empty inventories (no corpse-noise).
        if let Some(grid) = inventory_grid {
            if !grid.items.is_empty() {
                if let Ok(_cid) =
                    self.spawn_world_container(pos, region, grid.width, grid.height, false)
                {
                    // Populate the freshly-spawned container with the
                    // dead NPC's stacks. We can't pre-populate via
                    // spawn_world_container (it always starts empty),
                    // so do it via the public put-on-grid path: walk
                    // the grid's items and append.
                    if let Some(container_e) =
                        crate::world::containers::find_container_in(&mut self.world, _cid)
                    {
                        if let Some(mut wc) = self
                            .world
                            .get_mut::<crate::components::WorldContainer>(container_e)
                        {
                            wc.grid = grid.clone();
                        }
                    }
                    // Re-emit a single WorldContainerSpawned delta with
                    // the populated grid so mirrors see the actual
                    // contents (the implicit empty one above is a
                    // no-op replay).
                    let _ = self.record_delta(crate::delta::WorldDelta::WorldContainerSpawned {
                        id: _cid,
                        region,
                        pos,
                        is_public: false,
                        initial_grid: grid,
                    });
                }
            }
        }
        self.world.despawn(e);
        let _ = self.record_delta(WorldDelta::NpcDied {
            id,
            region,
            cause,
            tick,
        });
        true
    }

    /// Test-only: force a journal flush+fsync now. Used by tests that
    /// want deterministic on-disk state without calling [`Self::shutdown`].
    #[doc(hidden)]
    pub fn flush_journal_for_test(&mut self) {
        if let Some(ref mut j) = self.journal {
            let _ = j.flush_and_sync();
        }
    }

    /// Test-only: override the snapshot interval so tests don't have
    /// to run for 600 ticks to observe compaction.
    #[doc(hidden)]
    pub fn set_snapshot_interval_for_test(&mut self, ticks: u64) {
        self.snapshot_interval = ticks.max(1);
    }

    /// Test-only: override the inventory weight cap (kg). Lets the
    /// overweight-regen test work against the small default 4×4
    /// pockets grid, where a single item's full stack barely
    /// approaches the production 50 kg default.
    #[doc(hidden)]
    pub fn set_weight_cap_for_test(&mut self, kg: f32) {
        let mut cfg = self
            .world
            .resource_mut::<crate::resources::InventoryConfig>();
        cfg.weight_cap_kg = kg;
    }

    /// Test-only: shorten the bandage→heal timer so tests don't have
    /// to tick the full default 6000 ticks of debug-build sim.
    /// Also sets `heal_ticks_stitched` to half of the new bandaged
    /// timer so stitch's faster-heal property holds in tests too.
    #[doc(hidden)]
    pub fn set_heal_ticks_for_test(&mut self, ticks: u64) {
        let mut cfg = self.world.resource_mut::<crate::resources::MedConfig>();
        cfg.heal_ticks_bandaged = ticks;
        cfg.heal_ticks_stitched = (ticks / 2).max(1);
    }

    /// Test-only: shorten the meds-layer timing constants
    /// proportionally so infection / antibiotics / necrosis tests
    /// don't have to tick 12000+ times. Pass the desired in-test
    /// "infection trigger" tick count; the others are scaled down to
    /// match the default ratios.
    #[doc(hidden)]
    pub fn set_med_timings_for_test(&mut self, infection_trigger_ticks: u64) {
        let mut cfg = self.world.resource_mut::<crate::resources::MedConfig>();
        let base = infection_trigger_ticks.max(1);
        cfg.infection_trigger_ticks = base;
        cfg.antibiotics_clear_ticks = (base / 4).max(1);
        cfg.necrosis_warning_ticks = (base / 2).max(1);
        cfg.necrosis_severe_ticks = (base / 2).max(1);
    }

    /// Test-only: force a specific `WorldTime`. Used by rollover and
    /// persistence tests that shouldn't have to tick forward for
    /// minutes of real time.
    #[doc(hidden)]
    pub fn force_world_time_for_test(&mut self, day: u32, seconds_of_day: f32, day_length: f32) {
        let mut t = self.world.resource_mut::<WorldTime>();
        t.day = day;
        t.seconds_of_day = seconds_of_day;
        t.day_length_seconds = day_length;
    }
}
