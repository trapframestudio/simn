//! Snapshot serialization, entity replay, and the `WorldDelta` apply
//! routine.
//!
//! These are the **free functions** backing `Sim`'s persistence layer.
//! They take `&mut World` directly rather than `&mut Sim` because they
//! run in two contexts: the authoritative save path (`roll_snapshot`)
//! and the mirror sim's external-delta apply (slice-1 replication).
//!
//! - [`serialize_world`] → builds a `SnapshotBody` from the ECS.
//! - [`spawn_serialized`] → inverse: rebuilds an entity from its
//!   serialized form.
//! - [`apply_delta`] → the big match statement that replays a
//!   `WorldDelta` onto a `World`. Used by journal replay (`Sim::load`)
//!   and by client mirrors (`Sim::apply_external_delta`) — one code
//!   path, two callers.
//! - [`find_player_in`] / [`find_npc_in`] → entity-lookup helpers used
//!   across the replay path and the `impl Sim` methods in sibling
//!   modules.

use bevy_ecs::prelude::*;

use crate::chronicle::LifeChronicle;
use crate::components::{
    ActiveEffect, ActiveEffects, Actor, ActorKind, Aggression, Base, BodyParts, Contamination,
    CraftJob, CraftingQueue, DrugTolerance, Equipment, EquippedItem, Group, Health, InFaction,
    InRegion, Inventory, Lifespan, NearCampfire, NearWorkbench, Npc, NpcGoal, NpcId, Pain,
    PlayerOwned, Position, Rotation, Stamina, SurvivalStats, WorldContainer, Wound, WoundTreatment,
    Wounds,
};
use crate::delta::WorldDelta;
use crate::items::{ItemRegistry, RecipeRegistry};
use crate::persistence::snapshot::{SerializedEntity, SnapshotBody};
use crate::region::RegionGraph;
use crate::resources::{
    NpcIdCounter, PopulationTargets, RegionControl, SimClock, WeatherState, WorldTime,
};

/// Serialize the current `World` into a [`SnapshotBody`]. Collects
/// every entity's optional component set into a `SerializedEntity`
/// row, plus pulls world-level resources into the body header.
pub(super) fn serialize_world(world: &mut World) -> SnapshotBody {
    let clock = *world.resource::<SimClock>();
    let region_graph = world.resource::<RegionGraph>().clone();
    let world_time = *world.resource::<WorldTime>();
    let weather = *world.resource::<WeatherState>();
    let region_control = world.resource::<RegionControl>().clone();
    let chronicle = world.resource::<LifeChronicle>().clone();
    let npc_id_counter = *world.resource::<NpcIdCounter>();
    let wound_id_counter = *world.resource::<crate::resources::WoundIdCounter>();
    let population_targets = world.resource::<PopulationTargets>().clone();

    // Collect entity ids first, then inspect each via world.get::<_>(e).
    // Simpler than fighting bevy_ecs 0.18's query variants for an
    // all-optional row.
    let entity_ids: Vec<Entity> = world.query::<Entity>().iter(world).collect();
    let mut entities = Vec::with_capacity(entity_ids.len());
    for e in entity_ids {
        let player = world.get::<PlayerOwned>(e).copied();
        let actor = world.get::<Actor>(e).copied();
        let position = world.get::<Position>(e).copied();
        let rotation = world.get::<Rotation>(e).copied();
        let in_region = world.get::<InRegion>(e).copied();
        let health = world.get::<Health>(e).copied();
        let stamina = world.get::<Stamina>(e).copied();
        let body_parts = world.get::<BodyParts>(e).copied();
        let survival = world.get::<SurvivalStats>(e).copied();
        let wounds = world.get::<Wounds>(e).cloned();
        let pain = world.get::<Pain>(e).copied();
        let contamination = world.get::<Contamination>(e).copied();
        let active_effects = world.get::<ActiveEffects>(e).cloned();
        let drug_tolerance = world.get::<DrugTolerance>(e).cloned();
        let inventory = world.get::<Inventory>(e).cloned();
        let equipment = world.get::<Equipment>(e).cloned();
        let near_campfire = world.get::<NearCampfire>(e).copied();
        let near_workbench = world.get::<NearWorkbench>(e).copied();
        let crafting_queue = world.get::<CraftingQueue>(e).cloned();
        let world_container = world.get::<WorldContainer>(e).cloned();
        // Persist faction by name string (registry edits would
        // break numeric ids). `None` for entities without a faction.
        let in_faction = world.get::<InFaction>(e).copied().map(|f| {
            let reg = world.resource::<crate::faction::registry::FactionRegistry>();
            reg.name_of(f.0).to_string()
        });
        let base = world.get::<Base>(e).copied();
        let npc = world.get::<Npc>(e).copied();
        let npc_goal = world.get::<NpcGoal>(e).copied();
        let lifespan = world.get::<Lifespan>(e).copied();
        let group = world.get::<Group>(e).copied();
        let aggression = world.get::<Aggression>(e).copied();
        let projectile = world.get::<crate::components::Projectile>(e).cloned();
        if player.is_none()
            && actor.is_none()
            && position.is_none()
            && rotation.is_none()
            && in_region.is_none()
            && health.is_none()
            && stamina.is_none()
            && body_parts.is_none()
            && survival.is_none()
            && wounds.is_none()
            && pain.is_none()
            && contamination.is_none()
            && active_effects.is_none()
            && drug_tolerance.is_none()
            && inventory.is_none()
            && equipment.is_none()
            && near_campfire.is_none()
            && near_workbench.is_none()
            && crafting_queue.is_none()
            && world_container.is_none()
            && in_faction.is_none()
            && base.is_none()
            && npc.is_none()
            && npc_goal.is_none()
            && lifespan.is_none()
            && group.is_none()
            && aggression.is_none()
            && projectile.is_none()
        {
            continue;
        }
        entities.push(SerializedEntity {
            player,
            actor,
            position,
            rotation,
            in_region,
            health,
            stamina,
            body_parts,
            survival,
            wounds,
            pain,
            contamination,
            active_effects,
            drug_tolerance,
            inventory,
            equipment,
            near_campfire,
            near_workbench,
            crafting_queue,
            world_container,
            in_faction,
            base,
            npc,
            npc_goal,
            lifespan,
            group,
            aggression,
            projectile,
        });
    }

    // Determinism: bevy's archetype storage iteration order is NOT
    // stable across sim instances (same components + same entities,
    // different storage layouts -> different orders). Sort by a
    // stable per-row key derived from each entity's identifying
    // component (NpcId, steam_id, ContainerId, ProjectileId, base
    // position) so two same-seed sims produce byte-identical
    // snapshots. See `crates/simn-sim/tests/determinism.rs`.
    entities.sort_by_key(entity_sort_key);

    let relation_deltas = world
        .resource::<crate::faction::registry::RelationDeltas>()
        .clone();
    let player_reputation = world
        .resource::<crate::faction::registry::PlayerReputation>()
        .clone();
    SnapshotBody {
        clock,
        region_graph,
        entities,
        world_time,
        region_control,
        chronicle,
        npc_id_counter,
        wound_id_counter,
        effect_id_counter: *world.resource::<crate::resources::EffectIdCounter>(),
        job_id_counter: *world.resource::<crate::resources::JobIdCounter>(),
        projectile_id_counter: *world.resource::<crate::resources::ProjectileIdCounter>(),
        container_id_counter: *world.resource::<crate::resources::ContainerIdCounter>(),
        population_targets,
        weather,
        relation_deltas,
        player_reputation,
        offline_tier_clock: *world.resource::<crate::offline_tier::OfflineTierClock>(),
    }
}

/// Stable sort key for serialized entities, used to make snapshot
/// byte streams deterministic across sim instances (see comment in
/// `serialize_world` above).
///
/// Returns a `(class, id)` tuple where `class` discriminates the
/// entity kind (so e.g. all NPCs sort together, then all bases,
/// etc.) and `id` is a stable per-instance identifier within that
/// class:
/// - Players: `steam_id`.
/// - NPCs: `NpcId`.
/// - World containers: `ContainerId`.
/// - Projectiles: `ProjectileId`.
/// - Bases: a hash of `(region, position bits)` since bases have no
///   stable id today; the same base in two sims produces the same
///   hash because both `region` and `position` are deterministic
///   spawn outputs.
/// - Anything else (player markers without `Actor`, etc.): the
///   debug class `0xFF` and a position-derived hash, with a
///   final fallback of 0.
fn entity_sort_key(se: &SerializedEntity) -> (u8, u64) {
    use std::hash::{Hash, Hasher};
    if let Some(p) = &se.player {
        return (0, p.steam_id);
    }
    if let Some(npc) = &se.npc {
        return (1, npc.id.0);
    }
    if let Some(c) = &se.world_container {
        return (2, c.id.0 as u64);
    }
    if let Some(p) = &se.projectile {
        return (3, p.id.0);
    }
    if let (Some(_b), Some(in_region), Some(pos)) = (&se.base, &se.in_region, &se.position) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        in_region.0.hash(&mut h);
        pos.0[0].to_bits().hash(&mut h);
        pos.0[1].to_bits().hash(&mut h);
        pos.0[2].to_bits().hash(&mut h);
        return (4, h.finish());
    }
    // Fallback: hash whatever position-like data is available so
    // these still sort stably even if the entity has no canonical id.
    if let Some(pos) = &se.position {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        pos.0[0].to_bits().hash(&mut h);
        pos.0[1].to_bits().hash(&mut h);
        pos.0[2].to_bits().hash(&mut h);
        return (0xFF, h.finish());
    }
    (0xFF, 0)
}

/// Inverse of [`serialize_world`] for a single entity: spawns a fresh
/// ECS entity and attaches each component that was present in the
/// serialized row.
pub(super) fn spawn_serialized(world: &mut World, se: SerializedEntity) {
    // Capture NPC-scoped back-compat flags BEFORE we start consuming
    // fields below via `if let Some(_) = se.xxx` moves. Older
    // snapshots may be missing components that NPCs now carry; when
    // loading such a row we default those in later so the NPC still
    // participates in the per-part damage + wound pipelines.
    let npc_needs_body_parts = se.npc.is_some() && se.body_parts.is_none();
    let npc_needs_wounds = se.npc.is_some() && se.wounds.is_none();
    let npc_needs_active_effects = se.npc.is_some() && se.active_effects.is_none();
    // Resolve the faction name string into a FactionId before
    // opening `EntityWorldMut` (registry resource lookup would
    // otherwise borrow-conflict). Names that don't resolve (e.g.
    // mod removed a faction between sessions) drop the component;
    // gameplay decides what to do with faction-less NPCs.
    let derived_in_faction = se.in_faction.as_deref().and_then(|name| {
        let reg = world.resource::<crate::faction::registry::FactionRegistry>();
        reg.id_of(name).map(crate::components::InFaction)
    });
    // Pull `base_aggression` and the archetype for the same reason —
    // `NpcCharacter::roll` needs both, and the registry resource
    // can't be touched once the entity-builder borrow opens below.
    let derived_agg_base = derived_in_faction.map(|f| {
        world
            .resource::<crate::faction::registry::FactionRegistry>()
            .def(f.0)
            .base_aggression
    });
    // Archetype is now driven from `factions.toml` per faction; pull
    // it from the registry once the faction id resolves. Falls back
    // to the legacy name-derived default if `derived_in_faction` is
    // None (faction missing from registry).
    let derived_archetype = derived_in_faction.map(|f| {
        world
            .resource::<crate::faction::registry::FactionRegistry>()
            .def(f.0)
            .archetype
    });
    // Pre-roll the character before opening the entity-builder borrow
    // — `NpcCharacter::roll` needs the `NameRegistry` resource and we
    // can't read resources once `world.spawn(())` locks the world.
    let derived_character = if let (Some(npc), Some(in_fac), Some(agg_base), Some(archetype)) = (
        se.npc,
        derived_in_faction,
        derived_agg_base,
        derived_archetype,
    ) {
        let (nat_weights, male_w) = {
            let def = world
                .resource::<crate::faction::registry::FactionRegistry>()
                .def(in_fac.0);
            (def.nationality_weights.clone(), def.male_name_weight)
        };
        let names = world.resource::<crate::names::NameRegistry>();
        Some(crate::components::NpcCharacter::roll(
            npc.id,
            in_fac.0,
            archetype,
            agg_base,
            names,
            &nat_weights,
            male_w,
        ))
    } else {
        None
    };
    let mut e = world.spawn(());
    if let Some(p) = se.player {
        e.insert(p);
    }
    if let Some(a) = se.actor {
        e.insert(a);
    }
    if let Some(p) = se.position {
        e.insert(p);
    }
    if let Some(r) = se.rotation {
        e.insert(r);
    }
    if let Some(r) = se.in_region {
        e.insert(r);
    }
    if let Some(h) = se.health {
        e.insert(h);
    }
    if let Some(s) = se.stamina {
        e.insert(s);
    }
    if let Some(b) = se.body_parts {
        e.insert(b);
    }
    if let Some(s) = se.survival {
        e.insert(s);
    }
    if let Some(w) = se.wounds {
        e.insert(w);
    }
    if let Some(p) = se.pain {
        e.insert(p);
    }
    if let Some(c) = se.contamination {
        e.insert(c);
    }
    if let Some(ef) = se.active_effects {
        e.insert(ef);
    }
    if let Some(t) = se.drug_tolerance {
        e.insert(t);
    }
    if let Some(inv) = se.inventory {
        e.insert(inv);
    }
    if let Some(eq) = se.equipment {
        e.insert(eq);
    }
    if let Some(nc) = se.near_campfire {
        e.insert(nc);
    }
    if let Some(nw) = se.near_workbench {
        e.insert(nw);
    }
    if let Some(cq) = se.crafting_queue {
        e.insert(cq);
    }
    if let Some(wc) = se.world_container {
        e.insert(wc);
    }
    // `se.in_faction` is the persisted name string; the resolved
    // `derived_in_faction` (Option<InFaction>) is the actual ECS
    // component. Discard the raw string — only the typed component
    // ends up in the world.
    let _ = se.in_faction;
    if let Some(f) = derived_in_faction {
        e.insert(f);
    }
    if let Some(b) = se.base {
        e.insert(b);
    }
    if let Some(n) = se.npc {
        e.insert(n);
        if npc_needs_body_parts {
            e.insert(BodyParts::new_full());
        }
        if npc_needs_wounds {
            e.insert(Wounds::default());
        }
        if npc_needs_active_effects {
            e.insert(ActiveEffects::default());
        }
        // LimbStates + RecentAttackers are transient (not serialized);
        // always attach fresh defaults on load. LimbStates default is
        // six `Intact`; the wound pipeline will flip parts to
        // `Wounded` on the first tick if Wounds carries any open
        // entries.
        e.insert(crate::components::LimbStates::default());
        e.insert(crate::components::RecentAttackers::default());
        // NpcCharacter is also transient — re-roll deterministically
        // from `(npc_id, faction_id)` so the same identity returns on
        // every load. Inline persistence + format bump come when an
        // identity field needs to evolve independently of the spawn
        // contract (e.g., personality drift).
        if let Some(character) = derived_character.clone() {
            e.insert(character);
        }
    }
    if let Some(g) = se.npc_goal {
        e.insert(g);
    }
    if let Some(l) = se.lifespan {
        e.insert(l);
    }
    if let Some(g) = se.group {
        e.insert(g);
    }
    if let Some(a) = se.aggression {
        e.insert(a);
    }
    if let Some(p) = se.projectile {
        e.insert(p);
    }
}

/// Replay a single [`WorldDelta`] into `world`. Used by journal replay
/// (authoritative) and by mirror sims (client-side replication).
/// Unknown variants are handled defensively — any miss triggers a
/// no-op rather than a panic, so forward-compat journals stay safe.
pub(super) fn apply_delta(world: &mut World, delta: &WorldDelta) {
    match delta {
        WorldDelta::SpawnPlayer {
            steam_id,
            region,
            pos,
            yaw,
        } => {
            let existing = find_player_in(world, *steam_id);
            if let Some(e) = existing {
                if let Some(mut p) = world.get_mut::<Position>(e) {
                    p.0 = *pos;
                }
                if let Some(mut r) = world.get_mut::<Rotation>(e) {
                    r.0 = *yaw;
                }
                if let Some(mut r) = world.get_mut::<InRegion>(e) {
                    r.0 = *region;
                }
            } else {
                world.spawn((
                    (
                        PlayerOwned {
                            steam_id: *steam_id,
                        },
                        Actor {
                            kind: ActorKind::Player,
                        },
                        Position(*pos),
                        Rotation(*yaw),
                        InRegion(*region),
                        Health::new_full(),
                        Stamina::new_full(),
                        BodyParts::new_full(),
                        crate::components::LimbStates::default(),
                        SurvivalStats::new_full(),
                        Wounds::default(),
                    ),
                    (
                        Pain::default(),
                        Contamination::default(),
                        ActiveEffects::default(),
                        DrugTolerance::default(),
                        Inventory::default(),
                        NearCampfire::default(),
                    ),
                ));
            }
        }
        WorldDelta::DespawnPlayer { steam_id } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                world.despawn(e);
            }
        }
        WorldDelta::MovePlayer { steam_id, pos, yaw } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut p) = world.get_mut::<Position>(e) {
                    p.0 = *pos;
                }
                if let Some(mut r) = world.get_mut::<Rotation>(e) {
                    r.0 = *yaw;
                }
            }
        }
        WorldDelta::ChangePlayerRegion { steam_id, region } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut r) = world.get_mut::<InRegion>(e) {
                    r.0 = *region;
                }
            }
        }
        WorldDelta::SetHealth { steam_id, current } => {
            // Legacy record (pre-stats-foundation). Replay onto torso so
            // older journals still drive the right death-gate state.
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut bp) = world.get_mut::<BodyParts>(e) {
                    bp.torso = *current;
                }
                if let Some(mut h) = world.get_mut::<Health>(e) {
                    h.current = *current;
                }
            }
        }
        WorldDelta::SetStamina { steam_id, current } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut s) = world.get_mut::<Stamina>(e) {
                    s.current = *current;
                }
            }
        }
        WorldDelta::SetBodyPart {
            steam_id,
            part,
            current,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                let vital_min = if let Some(mut bp) = world.get_mut::<BodyParts>(e) {
                    *bp.get_mut(*part) = *current;
                    Some(bp.vital_min())
                } else {
                    None
                };
                if let Some(v) = vital_min {
                    if let Some(mut h) = world.get_mut::<Health>(e) {
                        h.current = v.min(h.max);
                    }
                }
            }
        }
        WorldDelta::SetSurvivalStat {
            steam_id,
            stat,
            current,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut s) = world.get_mut::<SurvivalStats>(e) {
                    *s.get_mut(*stat) = *current;
                }
            }
        }
        WorldDelta::WoundAdded {
            steam_id,
            wound_id,
            body_part,
            kind,
            severity,
            spawned_tick,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                let wound = Wound {
                    body_part: *body_part,
                    kind: *kind,
                    severity: *severity,
                    spawned_tick: *spawned_tick,
                    treatment: WoundTreatment::Untreated,
                    treatment_changed_tick: *spawned_tick,
                    infected: false,
                    infection_started_tick: None,
                    tourniquet_started_tick: None,
                };
                if let Some(mut wounds) = world.get_mut::<Wounds>(e) {
                    wounds.0.push((*wound_id, wound));
                } else {
                    world.entity_mut(e).insert(Wounds(vec![(*wound_id, wound)]));
                }
                if let Some(mut states) = world.get_mut::<crate::components::LimbStates>(e) {
                    states.mark_wounded(*body_part);
                }
            }
        }
        WorldDelta::WoundTreatmentChanged {
            steam_id,
            wound_id,
            new_state,
            changed_tick,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut wounds) = world.get_mut::<Wounds>(e) {
                    for (id, w) in wounds.0.iter_mut() {
                        if *id == *wound_id {
                            w.treatment = *new_state;
                            w.treatment_changed_tick = *changed_tick;
                            // Mirror the live API's tourniquet bookkeeping.
                            match *new_state {
                                WoundTreatment::Tourniquet => {
                                    w.tourniquet_started_tick = Some(*changed_tick);
                                }
                                WoundTreatment::Untreated
                                | WoundTreatment::Disinfected
                                | WoundTreatment::Bandaged
                                | WoundTreatment::Stitched
                                | WoundTreatment::WoundPacked
                                | WoundTreatment::Healed => {
                                    w.tourniquet_started_tick = None;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        WorldDelta::WoundInfected {
            steam_id,
            wound_id,
            started_tick,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut wounds) = world.get_mut::<Wounds>(e) {
                    for (id, w) in wounds.0.iter_mut() {
                        if *id == *wound_id {
                            w.infected = true;
                            w.infection_started_tick = Some(*started_tick);
                            break;
                        }
                    }
                }
            }
        }
        WorldDelta::EffectApplied {
            steam_id,
            effect_id,
            kind,
            applied_tick,
            duration_ticks,
            intensity,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                let effect = ActiveEffect {
                    id: *effect_id,
                    kind: *kind,
                    applied_tick: *applied_tick,
                    duration_ticks: *duration_ticks,
                    intensity: *intensity,
                };
                if let Some(mut effects) = world.get_mut::<ActiveEffects>(e) {
                    effects.0.push(effect);
                } else {
                    world.entity_mut(e).insert(ActiveEffects(vec![effect]));
                }
            }
        }
        WorldDelta::ToleranceChanged {
            steam_id,
            drug,
            value,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut tol) = world.get_mut::<DrugTolerance>(e) {
                    tol.set(*drug, *value);
                } else {
                    world
                        .entity_mut(e)
                        .insert(DrugTolerance(vec![(*drug, *value)]));
                }
            }
        }
        WorldDelta::RadiationChanged { steam_id, value } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut c) = world.get_mut::<Contamination>(e) {
                    c.radiation = *value;
                } else {
                    world.entity_mut(e).insert(Contamination {
                        radiation: *value,
                        toxicity: 0.0,
                    });
                }
            }
        }
        WorldDelta::ToxicityChanged { steam_id, value } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut c) = world.get_mut::<Contamination>(e) {
                    c.toxicity = *value;
                } else {
                    world.entity_mut(e).insert(Contamination {
                        radiation: 0.0,
                        toxicity: *value,
                    });
                }
            }
        }
        WorldDelta::NpcSpawned {
            id,
            faction,
            region,
            pos,
            yaw,
            die_at_tick,
        } => {
            // Skip if the NPC already exists (replay idempotency).
            if find_npc_in(world, *id).is_none() {
                let spawned_tick = world.resource::<SimClock>().tick;
                let Some(faction_id) = world
                    .resource::<crate::faction::registry::FactionRegistry>()
                    .id_of(faction)
                else {
                    // Faction missing from active registry (e.g. mod
                    // removed it between sessions). Skip the spawn —
                    // no resolution path forward.
                    return;
                };
                let agg_base = world
                    .resource::<crate::faction::registry::FactionRegistry>()
                    .def(faction_id)
                    .base_aggression;
                let (archetype, nat_weights, male_w) = {
                    let def = world
                        .resource::<crate::faction::registry::FactionRegistry>()
                        .def(faction_id);
                    (
                        def.archetype,
                        def.nationality_weights.clone(),
                        def.male_name_weight,
                    )
                };
                let character = crate::components::NpcCharacter::roll(
                    *id,
                    faction_id,
                    archetype,
                    agg_base,
                    world.resource::<crate::names::NameRegistry>(),
                    &nat_weights,
                    male_w,
                );
                world.spawn((
                    Npc { id: *id },
                    Actor {
                        kind: ActorKind::Npc,
                    },
                    InFaction(faction_id),
                    InRegion(*region),
                    Position(*pos),
                    Rotation(*yaw),
                    Health::new_full(),
                    BodyParts::new_full(),
                    crate::components::LimbStates::default(),
                    Wounds::default(),
                    ActiveEffects::default(),
                    NpcGoal::Idle {
                        until_tick: spawned_tick.wrapping_add(20),
                    },
                    Lifespan {
                        spawned_tick,
                        die_at_tick: *die_at_tick,
                    },
                    crate::components::RecentAttackers::default(),
                    character,
                ));
            }
        }
        WorldDelta::SetNpcBodyPart { id, part, current } => {
            if let Some(e) = find_npc_in(world, *id) {
                let vital_min = if let Some(mut bp) = world.get_mut::<BodyParts>(e) {
                    *bp.get_mut(*part) = *current;
                    Some(bp.vital_min())
                } else {
                    None
                };
                if let Some(v) = vital_min {
                    if let Some(mut h) = world.get_mut::<Health>(e) {
                        h.current = v.min(h.max);
                    }
                }
            }
        }
        WorldDelta::NpcChangeRegion { id, region, pos } => {
            if let Some(e) = find_npc_in(world, *id) {
                if let Some(mut r) = world.get_mut::<InRegion>(e) {
                    r.0 = *region;
                }
                if let Some(mut p) = world.get_mut::<Position>(e) {
                    p.0 = *pos;
                }
            }
        }
        WorldDelta::NpcDied { id, .. } => {
            if let Some(e) = find_npc_in(world, *id) {
                world.despawn(e);
            }
        }
        WorldDelta::NpcWoundAdded {
            id,
            wound_id,
            body_part,
            kind,
            severity,
            spawned_tick,
        } => {
            if let Some(e) = find_npc_in(world, *id) {
                let wound = Wound {
                    body_part: *body_part,
                    kind: *kind,
                    severity: *severity,
                    spawned_tick: *spawned_tick,
                    treatment: WoundTreatment::Untreated,
                    treatment_changed_tick: *spawned_tick,
                    infected: false,
                    infection_started_tick: None,
                    tourniquet_started_tick: None,
                };
                if let Some(mut wounds) = world.get_mut::<Wounds>(e) {
                    wounds.0.push((*wound_id, wound));
                } else {
                    world.entity_mut(e).insert(Wounds(vec![(*wound_id, wound)]));
                }
                if let Some(mut states) = world.get_mut::<crate::components::LimbStates>(e) {
                    states.mark_wounded(*body_part);
                }
            }
        }
        WorldDelta::NpcWoundTreatmentChanged {
            id,
            wound_id,
            new_state,
            changed_tick,
        } => {
            if let Some(e) = find_npc_in(world, *id) {
                if let Some(mut wounds) = world.get_mut::<Wounds>(e) {
                    for (wid, w) in wounds.0.iter_mut() {
                        if *wid == *wound_id {
                            w.treatment = *new_state;
                            w.treatment_changed_tick = *changed_tick;
                            match *new_state {
                                WoundTreatment::Tourniquet => {
                                    w.tourniquet_started_tick = Some(*changed_tick);
                                }
                                WoundTreatment::Untreated
                                | WoundTreatment::Disinfected
                                | WoundTreatment::Bandaged
                                | WoundTreatment::Stitched
                                | WoundTreatment::WoundPacked
                                | WoundTreatment::Healed => {
                                    w.tourniquet_started_tick = None;
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        WorldDelta::NpcWoundInfected {
            id,
            wound_id,
            started_tick,
        } => {
            if let Some(e) = find_npc_in(world, *id) {
                if let Some(mut wounds) = world.get_mut::<Wounds>(e) {
                    for (wid, w) in wounds.0.iter_mut() {
                        if *wid == *wound_id {
                            w.infected = true;
                            w.infection_started_tick = Some(*started_tick);
                            break;
                        }
                    }
                }
            }
        }
        WorldDelta::NpcEffectApplied {
            id,
            effect_id,
            kind,
            applied_tick,
            duration_ticks,
            intensity,
        } => {
            if let Some(e) = find_npc_in(world, *id) {
                let effect = ActiveEffect {
                    id: *effect_id,
                    kind: *kind,
                    applied_tick: *applied_tick,
                    duration_ticks: *duration_ticks,
                    intensity: *intensity,
                };
                if let Some(mut effects) = world.get_mut::<ActiveEffects>(e) {
                    effects.0.push(effect);
                } else {
                    world.entity_mut(e).insert(ActiveEffects(vec![effect]));
                }
            }
        }
        WorldDelta::ItemPickedUp {
            steam_id,
            item_id,
            count,
            spawned_tick,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                let registry = world.resource::<ItemRegistry>().clone();
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    super::inventory::merge_item_stack(
                        &mut inv,
                        &registry,
                        item_id.clone(),
                        *count,
                        *spawned_tick,
                    );
                }
            }
        }
        WorldDelta::ItemDropped {
            steam_id, slot_idx, ..
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    if *slot_idx < inv.0.items.len() {
                        inv.0.items.remove(*slot_idx);
                    }
                }
            }
        }
        WorldDelta::ItemMoved {
            steam_id,
            from_slot,
            to_slot,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    if *from_slot < inv.0.items.len() && *to_slot < inv.0.items.len() {
                        inv.0.items.swap(*from_slot, *to_slot);
                    }
                }
            }
        }
        WorldDelta::ItemMovedBetweenGrids {
            steam_id,
            from_grid,
            from_idx,
            to_grid,
            item,
            inner_grid,
            ..
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            // Mirror the host: pull from source, find first-fit on
            // dest, place. The mirror's view should match the host's
            // after the prior delta stream applied; if not, the next
            // snapshot reconciles.
            remove_by_source(world, e, from_grid, *from_idx);
            let registry = world.resource::<ItemRegistry>().clone();
            let dest_str = to_grid.clone();
            if let Some(grid) = grid_for_source_mut(world, e, &dest_str) {
                if let Some((x, y, rot)) = crate::inventory_grid::find_first_fit_any_rotation(
                    grid,
                    &registry,
                    registry.get(&item.id).unwrap_or(&dummy_def(&item.id)),
                ) {
                    let _ = crate::inventory_grid::place_at_with_inner(
                        grid,
                        &registry,
                        item.clone(),
                        inner_grid.clone(),
                        x,
                        y,
                        rot,
                    );
                }
            }
        }
        WorldDelta::ItemConsumed {
            steam_id, slot_idx, ..
        } => {
            // The side-effects (eat / drink / apply_drug / ...) write
            // their own deltas, which are applied independently in the
            // same journal stream. Here we only decrement the stack.
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    if *slot_idx < inv.0.items.len() {
                        inv.0.items[*slot_idx].stack.count =
                            inv.0.items[*slot_idx].stack.count.saturating_sub(1);
                        if inv.0.items[*slot_idx].stack.count == 0 {
                            inv.0.items.remove(*slot_idx);
                        }
                    }
                }
            }
        }
        WorldDelta::ItemsSalvaged {
            steam_id,
            source_slot,
            outputs,
            tick,
        } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    if *source_slot < inv.0.items.len() {
                        inv.0.items[*source_slot].stack.count =
                            inv.0.items[*source_slot].stack.count.saturating_sub(1);
                        if inv.0.items[*source_slot].stack.count == 0 {
                            inv.0.items.remove(*source_slot);
                        }
                    }
                }
                let registry = world.resource::<ItemRegistry>().clone();
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    for stack in outputs {
                        super::inventory::merge_item_stack(
                            &mut inv,
                            &registry,
                            stack.id.clone(),
                            stack.count,
                            *tick,
                        );
                    }
                }
            }
        }
        WorldDelta::ItemsCrafted {
            steam_id,
            recipe_id,
            tick,
        } => {
            let recipe = world.resource::<RecipeRegistry>().get(recipe_id).cloned();
            if let (Some(recipe), Some(e)) = (recipe, find_player_in(world, *steam_id)) {
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    for input in &recipe.inputs {
                        super::inventory::consume_from_stacks(&mut inv, &input.id, input.count);
                    }
                }
                let registry = world.resource::<ItemRegistry>().clone();
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    for out in &recipe.outputs {
                        super::inventory::merge_item_stack(
                            &mut inv,
                            &registry,
                            out.id.clone(),
                            out.count,
                            *tick,
                        );
                    }
                }
            }
        }
        WorldDelta::ItemEquipped {
            steam_id,
            slot_id,
            item,
            inner_grid,
            source_grid,
            source_idx,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            // Remove from source grid (pockets / equipped:<slot>).
            remove_by_source(world, e, source_grid, *source_idx);
            // Insert into Equipment. Initialize weapon_state iff the
            // item is a weapon; the reload delta that follows later
            // in the journal will populate loaded_magazine.
            let weapon_state = super::weapons::init_weapon_state_for(
                &item.id,
                world.resource::<crate::items::ItemRegistry>(),
            );
            let equipped = EquippedItem {
                stack: item.clone(),
                inner_grid: inner_grid.clone(),
                weapon_state,
            };
            if let Some(mut eq) = world.get_mut::<Equipment>(e) {
                eq.0.insert(slot_id.clone(), equipped);
            } else {
                let mut map = std::collections::HashMap::new();
                map.insert(slot_id.clone(), equipped);
                world.entity_mut(e).insert(Equipment(map));
            }
        }
        WorldDelta::ItemUnequipped {
            steam_id,
            slot_id,
            item,
            inner_grid,
            dest_grid,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            // Remove from Equipment.
            if let Some(mut eq) = world.get_mut::<Equipment>(e) {
                eq.0.remove(slot_id);
            }
            // Re-place into the destination grid. Falls through to
            // best-effort: the canonical path on host has already
            // validated the fit, so a None outcome here means the
            // mirror's view differs slightly and the next snapshot
            // will reconcile.
            let registry = world.resource::<ItemRegistry>().clone();
            let grid_ref_source = dest_grid.clone();
            if let Some(grid) = grid_for_source_mut(world, e, &grid_ref_source) {
                if let Some((x, y, rotation)) = crate::inventory_grid::find_first_fit_any_rotation(
                    grid,
                    &registry,
                    registry.get(&item.id).unwrap_or(&dummy_def(&item.id)),
                ) {
                    let _ = crate::inventory_grid::place_at_with_inner(
                        grid,
                        &registry,
                        item.clone(),
                        inner_grid.clone(),
                        x,
                        y,
                        rotation,
                    );
                }
            }
        }
        WorldDelta::WeaponReloaded {
            steam_id,
            slot_id,
            loaded_magazine,
            ejected,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            // Find and remove a matching magazine from pockets. "Match"
            // = same item_id + same loaded_rounds. Identical mags are
            // interchangeable for mirror replay; the host chose one,
            // any matching one here lands in the same world state.
            let pocket_idx = {
                let inv = match world.get::<Inventory>(e) {
                    Some(inv) => inv,
                    None => return,
                };
                let target_rounds = loaded_magazine.loaded_rounds();
                inv.0.items.iter().position(|placed| {
                    placed.stack.id == loaded_magazine.id
                        && placed.stack.loaded_rounds() == target_rounds
                })
            };
            if let Some(idx) = pocket_idx {
                if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                    if idx < inv.0.items.len() {
                        inv.0.items.remove(idx);
                    }
                }
            }
            // Install the loaded magazine into the weapon's state.
            if let Some(mut eq) = world.get_mut::<Equipment>(e) {
                if let Some(equipped) = eq.0.get_mut(slot_id) {
                    let ws = equipped
                        .weapon_state
                        .get_or_insert_with(crate::components::EquippedWeaponState::default);
                    ws.loaded_magazine = Some(loaded_magazine.clone());
                }
            }
            // Return the ejected magazine (if any) to pockets. Replay
            // discards errors: if the mirror's pockets are unexpectedly
            // full, the next snapshot reconciles the state.
            if let Some(old_mag) = ejected {
                let _ = super::weapons::place_mag_in_pockets(world, e, old_mag.clone());
            }
        }
        WorldDelta::WeaponFired {
            steam_id,
            slot_id,
            remaining_rounds,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            if let Some(mag) = super::weapons::loaded_magazine_mut(world, e, slot_id) {
                // Preserve the loaded variant; `WeaponFired` only
                // tells us how many rounds are left after the shot.
                let variant = mag
                    .magazine_state
                    .as_ref()
                    .and_then(|ms| ms.variant.clone());
                mag.magazine_state = Some(crate::components::MagazineState {
                    loaded_rounds: *remaining_rounds,
                    variant,
                });
            }
        }
        WorldDelta::MagazineLoaded {
            steam_id,
            slot_id,
            round_id,
            added,
            total,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            // Consume the ammo stacks that the host consumed. The
            // number matches `added` exactly because caliber +
            // variant gating happened host-side.
            if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                super::inventory::consume_from_stacks(&mut inv, round_id, *added);
            }
            if let Some(mag) = super::weapons::loaded_magazine_mut(world, e, slot_id) {
                mag.magazine_state = Some(crate::components::MagazineState {
                    loaded_rounds: *total,
                    variant: Some(round_id.clone()),
                });
            }
        }
        WorldDelta::PocketMagazineLoaded {
            steam_id,
            pocket_idx,
            round_id,
            added,
            total,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            // Mirror the host-side order: consume ammo stacks
            // first (the ammo row's ItemId differs from the mag
            // row's, so this doesn't invalidate `pocket_idx`),
            // then write the mag state at `pocket_idx`.
            if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                super::inventory::consume_from_stacks(&mut inv, round_id, *added);
            }
            if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                let idx = *pocket_idx as usize;
                if let Some(placed) = inv.0.items.get_mut(idx) {
                    placed.stack.magazine_state = Some(crate::components::MagazineState {
                        loaded_rounds: *total,
                        variant: Some(round_id.clone()),
                    });
                }
            }
        }
        WorldDelta::ProjectileSpawned {
            id,
            source_steam_id,
            source_npc_id: _source_npc_id,
            round_id,
            // Phase 4B v2: variant is FX/AI metadata; the
            // Projectile component re-derives it from `round_id`
            // via the ItemRegistry on tick, so we don't store it
            // on the entity. The delta carries it for mirror
            // clients and journal replay observers.
            variant: _variant,
            origin,
            velocity,
            max_range_m,
            spawned_tick,
        } => {
            // On auth sims, `Sim::fire_weapon` creates the entity
            // directly (host authoritative); this arm is primarily
            // for journal-replay on load + mirror client replay.
            // Idempotency: skip if an entity with the same id
            // already exists (host may have just spawned it).
            let already_present = {
                let mut q = world.query::<(Entity, &crate::components::Projectile)>();
                q.iter(world).any(|(_, p)| p.id == *id)
            };
            if already_present {
                return;
            }
            // Determine region — projectile tracks whichever region
            // the shooter is in. Mirror sims may not have seen the
            // shooter upsert yet; in that case, skip (a later
            // snapshot will reconcile). Drag / mass not in the delta
            // — read them from the registry.
            let shooter_region = {
                let mut q = world.query::<(&crate::components::PlayerOwned, &InRegion)>();
                q.iter(world)
                    .find(|(p, _)| p.steam_id == *source_steam_id)
                    .map(|(_, r)| r.0)
            };
            let Some(region) = shooter_region else {
                return;
            };
            let (mass_kg, drag_k) = {
                let reg = world.resource::<crate::items::ItemRegistry>();
                let ac = reg.get(round_id).and_then(|d| d.ammo_config.as_ref());
                (
                    ac.map(|a| a.mass_g / 1000.0).unwrap_or(0.01),
                    ac.map(|a| a.drag_k).unwrap_or(0.0),
                )
            };
            let proj = crate::components::Projectile {
                id: *id,
                source_steam_id: *source_steam_id,
                // Pre-4A `ProjectileSpawned` deltas don't carry an
                // NPC source — `#[serde(default)]` lands `None`.
                // Phase 4A+ deltas carry the real source so
                // mirror-tier client FX can color NPC vs player
                // tracers.
                source_npc_id: *_source_npc_id,
                round_id: round_id.clone(),
                pos: *origin,
                vel: *velocity,
                distance_traveled_m: 0.0,
                max_range_m: *max_range_m,
                spawned_tick: *spawned_tick,
            };
            let _ = mass_kg; // reserved for future full-mass-aware mirror interp
            let _ = drag_k;
            world.spawn((proj, Position(*origin), Rotation(0.0), InRegion(region)));
        }
        WorldDelta::ProjectileImpacted { id, pos, .. } => {
            // Locate the matching projectile entity by id and
            // despawn. Client-side FX hook lives in the Godot
            // bridge's delta-replay path; the sim's role is just
            // entity bookkeeping so snapshot round-trips don't
            // carry zombie projectiles.
            let mut target: Option<Entity> = None;
            let mut q = world.query::<(Entity, &crate::components::Projectile)>();
            for (entity, proj) in q.iter(world) {
                if proj.id == *id {
                    target = Some(entity);
                    break;
                }
            }
            if let Some(e) = target {
                // Pin position to the impact point first so any
                // consumer reading ECS state lands the FX at the
                // real terminal.
                if let Some(mut p) = world.get_mut::<Position>(e) {
                    p.0 = *pos;
                }
                world.despawn(e);
            }
        }
        WorldDelta::WeaponMagazineEjected {
            steam_id,
            slot_id,
            ejected,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            if let Some(mut eq) = world.get_mut::<Equipment>(e) {
                if let Some(equipped) = eq.0.get_mut(slot_id) {
                    if let Some(ws) = equipped.weapon_state.as_mut() {
                        ws.loaded_magazine = None;
                    }
                }
            }
            if let Some(mag) = ejected {
                let _ = super::weapons::place_mag_in_pockets(world, e, mag.clone());
            }
        }
        WorldDelta::WeaponJammed {
            steam_id,
            slot_id,
            jam,
            condition: _,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            if let Some(mut eq) = world.get_mut::<Equipment>(e) {
                if let Some(equipped) = eq.0.get_mut(slot_id) {
                    let ws = equipped
                        .weapon_state
                        .get_or_insert_with(crate::components::EquippedWeaponState::default);
                    ws.jam_state = *jam;
                }
            }
        }
        WorldDelta::WeaponJamCleared { steam_id, slot_id } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            if let Some(mut eq) = world.get_mut::<Equipment>(e) {
                if let Some(equipped) = eq.0.get_mut(slot_id) {
                    if let Some(ws) = equipped.weapon_state.as_mut() {
                        ws.jam_state = crate::components::JamState::Cleared;
                    }
                }
            }
        }
        WorldDelta::WeaponConditionChanged {
            steam_id,
            slot_id,
            new_condition,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            if let Some(mut eq) = world.get_mut::<Equipment>(e) {
                if let Some(equipped) = eq.0.get_mut(slot_id) {
                    let ws = equipped
                        .weapon_state
                        .get_or_insert_with(crate::components::EquippedWeaponState::default);
                    ws.condition = *new_condition;
                }
            }
        }
        WorldDelta::NearCampfireSet { steam_id, value } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut nc) = world.get_mut::<NearCampfire>(e) {
                    nc.0 = *value;
                } else {
                    world.entity_mut(e).insert(NearCampfire(*value));
                }
            }
        }
        WorldDelta::NearWorkbenchSet { steam_id, tier } => {
            if let Some(e) = find_player_in(world, *steam_id) {
                if let Some(mut nw) = world.get_mut::<NearWorkbench>(e) {
                    nw.0 = *tier;
                } else {
                    world.entity_mut(e).insert(NearWorkbench(*tier));
                }
            }
        }
        WorldDelta::CraftJobQueued {
            steam_id,
            job_id,
            recipe_id,
            count,
            time_ticks_per_unit,
            inputs_consumed,
            started_tick,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                for stack in inputs_consumed {
                    super::inventory::consume_from_stacks(&mut inv, &stack.id, stack.count);
                }
            }
            let job = CraftJob {
                id: *job_id,
                recipe_id: recipe_id.clone(),
                count_remaining: *count,
                ticks_remaining: *time_ticks_per_unit,
                started_tick: *started_tick,
            };
            if let Some(mut cq) = world.get_mut::<CraftingQueue>(e) {
                cq.0.push(job);
            } else {
                world.entity_mut(e).insert(CraftingQueue(vec![job]));
            }
            // Mint on the replay side too so the local counter stays
            // ahead of all observed job ids. (Local mint would re-issue
            // the same id on a subsequent authoritative call otherwise.)
            let mut counter = world.resource_mut::<crate::resources::JobIdCounter>();
            if counter.0 < *job_id {
                counter.0 = *job_id;
            }
        }
        WorldDelta::CraftJobCancelled {
            steam_id,
            job_id,
            refund,
        } => {
            let Some(e) = find_player_in(world, *steam_id) else {
                return;
            };
            if let Some(mut cq) = world.get_mut::<CraftingQueue>(e) {
                cq.0.retain(|j| j.id != *job_id);
            }
            let now = world.resource::<SimClock>().tick;
            let registry = world.resource::<ItemRegistry>().clone();
            if let Some(mut inv) = world.get_mut::<Inventory>(e) {
                for stack in refund {
                    super::inventory::merge_item_stack(
                        &mut inv,
                        &registry,
                        stack.id.clone(),
                        stack.count,
                        now,
                    );
                }
            }
        }
        WorldDelta::NpcPositionBatch { tick: _, updates } => {
            // Apply per-NPC transforms. Ids not known to this sim are
            // silently skipped — either the `NpcSpawned` delta hasn't
            // been applied yet (reorder artifact) or the NPC has
            // despawned locally but the batch still included it.
            for (id, pos, yaw) in updates {
                if let Some(e) = find_npc_in(world, *id) {
                    if let Some(mut p) = world.get_mut::<Position>(e) {
                        p.0 = *pos;
                    }
                    if let Some(mut r) = world.get_mut::<Rotation>(e) {
                        r.0 = *yaw;
                    }
                }
            }
        }
        WorldDelta::WorldContainerSpawned {
            id,
            region,
            pos,
            is_public,
            initial_grid,
        } => {
            // Idempotent: if a container with this id already exists,
            // overwrite its grid + position rather than spawning a duplicate.
            // Keeps mirror replay safe across re-orderings.
            if let Some(e) = super::containers::find_container_in(world, *id) {
                if let Some(mut wc) = world.get_mut::<WorldContainer>(e) {
                    wc.grid = initial_grid.clone();
                    wc.is_public = *is_public;
                }
                if let Some(mut p) = world.get_mut::<Position>(e) {
                    p.0 = *pos;
                }
                if let Some(mut r) = world.get_mut::<InRegion>(e) {
                    r.0 = *region;
                }
            } else {
                // Replay path: fields not on the delta default to
                // the same values `spawn_world_container` uses for
                // ad-hoc spawns. Restock metadata gets re-seeded
                // on the next sweep.
                world.spawn((
                    WorldContainer {
                        id: *id,
                        grid: initial_grid.clone(),
                        is_public: *is_public,
                        faction: None,
                        depth_tier: 1,
                        last_restock_tick: 0,
                        interaction_mode: crate::components::ContainerInteractionMode::Openable,
                    },
                    Position(*pos),
                    InRegion(*region),
                ));
            }
            // Keep the local counter ahead of any observed id so a
            // future authoritative mint can't collide.
            let mut counter = world.resource_mut::<crate::resources::ContainerIdCounter>();
            if counter.0 < id.0 {
                counter.0 = id.0;
            }
        }
        WorldDelta::WorldContainerDespawned { id } => {
            if let Some(e) = super::containers::find_container_in(world, *id) {
                world.despawn(e);
            }
        }
        WorldDelta::WorldContainerItemAdded {
            id,
            item,
            inner_grid,
        } => {
            let Some(e) = super::containers::find_container_in(world, *id) else {
                return;
            };
            let registry = world.resource::<ItemRegistry>().clone();
            let Some(mut wc) = world.get_mut::<WorldContainer>(e) else {
                return;
            };
            // Place via grant_or_merge to find a slot, then attach the
            // travelling inner_grid (loaded backpack / nested container)
            // to the resulting placement. Containers don't merge with
            // existing stacks (each is unique), so the touched index is
            // a fresh placement.
            if let Ok(outcome) = crate::inventory_grid::grant_or_merge(
                &mut wc.grid,
                &registry,
                &item.id,
                item.count,
                item.spawned_tick,
            ) {
                if inner_grid.is_some() {
                    let touched = match outcome {
                        crate::inventory_grid::PlaceOutcome::Placed { touched_indices }
                        | crate::inventory_grid::PlaceOutcome::PartialOrFull {
                            touched_indices,
                            ..
                        } => touched_indices,
                    };
                    if let Some(idx) = touched.last() {
                        if let Some(p) = wc.grid.items.get_mut(*idx) {
                            p.inner_grid = inner_grid.clone();
                        }
                    }
                }
            }
        }
        WorldDelta::WorldContainerItemRemoved {
            id,
            source_idx,
            taken: _,
            inner_grid: _,
        } => {
            let Some(e) = super::containers::find_container_in(world, *id) else {
                return;
            };
            if let Some(mut wc) = world.get_mut::<WorldContainer>(e) {
                let idx = *source_idx as usize;
                if idx < wc.grid.items.len() {
                    wc.grid.items.remove(idx);
                }
            }
        }
        WorldDelta::Tick { tick } => {
            world.resource_mut::<SimClock>().tick = *tick;
        }
        WorldDelta::FactionRelationShift {
            a,
            b,
            delta,
            reason: _,
        } => {
            // Resolve faction names against the active registry. If
            // either name is unknown (modder removed a faction
            // between sessions), drop the shift on replay rather
            // than corrupt the deltas table.
            let registry = world
                .resource::<crate::faction::registry::FactionRegistry>()
                .clone();
            let Some(id_a) = registry.id_of(a) else {
                return;
            };
            let Some(id_b) = registry.id_of(b) else {
                return;
            };
            let mut deltas = world.resource_mut::<crate::faction::registry::RelationDeltas>();
            crate::faction::registry::shift_faction_relation(
                &registry,
                deltas.as_mut(),
                id_a,
                id_b,
                *delta,
            );
        }
        WorldDelta::PlayerRepShift {
            steam_id,
            faction,
            delta,
            reason: _,
        } => {
            let registry = world
                .resource::<crate::faction::registry::FactionRegistry>()
                .clone();
            let Some(id_f) = registry.id_of(faction) else {
                return;
            };
            let mut rep = world.resource_mut::<crate::faction::registry::PlayerReputation>();
            crate::faction::registry::shift_player_rep(
                &registry,
                rep.as_mut(),
                *steam_id,
                id_f,
                *delta,
            );
        }
    }
}

/// Find the `Entity` for a player by Steam ID, if present. `None`
/// when the player isn't in the sim.
pub(super) fn find_player_in(world: &mut World, steam_id: u64) -> Option<Entity> {
    let mut q = world.query::<(Entity, &PlayerOwned)>();
    q.iter(world)
        .find(|(_, p)| p.steam_id == steam_id)
        .map(|(e, _)| e)
}

/// Find the `Entity` for an NPC by id, if present. Crate-public
/// because other systems (e.g. `npc_death_check`) use it to resolve
/// deaths after an out-of-band id lookup.
pub(crate) fn find_npc_in(world: &mut World, id: NpcId) -> Option<Entity> {
    let mut q = world.query::<(Entity, &Npc)>();
    q.iter(world).find(|(_, n)| n.id == id).map(|(e, _)| e)
}

/// Resolve a source-grid string (`"pockets"` or `"equipped:<slot>"`)
/// to a mutable reference into the player's grid, then remove the
/// item at `idx`. No-ops on unknown grid / out-of-range idx.
fn remove_by_source(world: &mut World, e: Entity, source: &str, idx: usize) {
    if source == "pockets" {
        if let Some(mut inv) = world.get_mut::<Inventory>(e) {
            if idx < inv.0.items.len() {
                inv.0.items.remove(idx);
            }
        }
    } else if let Some(slot_str) = source.strip_prefix("equipped:") {
        let slot_id = crate::items::SlotId::from(slot_str);
        if let Some(mut eq) = world.get_mut::<Equipment>(e) {
            if let Some(eq_item) = eq.0.get_mut(&slot_id) {
                if let Some(ref mut grid) = eq_item.inner_grid {
                    if idx < grid.items.len() {
                        grid.items.remove(idx);
                    }
                }
            }
        }
    }
}

/// Resolve a grid-ref string (`"pockets"` or `"equipped:<slot>"`)
/// to the `GridInventory` inside the player entity, for the unequip
/// replay path. Returns `None` if the grid doesn't exist.
fn grid_for_source_mut<'w>(
    world: &'w mut World,
    e: Entity,
    source: &str,
) -> Option<&'w mut crate::components::GridInventory> {
    if source == "pockets" {
        return world.get_mut::<Inventory>(e).map(|inv| {
            // Project the mutable borrow of `Inventory` through to its
            // inner `GridInventory`. Safe because we're returning a
            // reborrow that shares the same lifetime as the caller-
            // held `&mut World`.
            let inv = inv.into_inner();
            &mut inv.0
        });
    }
    if let Some(slot_str) = source.strip_prefix("equipped:") {
        let slot_id = crate::items::SlotId::from(slot_str);
        if let Some(eq) = world.get_mut::<Equipment>(e) {
            let eq = eq.into_inner();
            if let Some(eq_item) = eq.0.get_mut(&slot_id) {
                return eq_item.inner_grid.as_mut();
            }
        }
    }
    None
}

/// Minimal stand-in `ItemDef` for the unequip replay path when the
/// registry lookup misses. Only used for footprint / rotation
/// defaults; never consumed as game data.
fn dummy_def(id: &crate::items::ItemId) -> crate::items::ItemDef {
    crate::items::ItemDef {
        id: id.clone(),
        name: id.0.clone(),
        category: crate::items::ItemCategory::Misc,
        weight: 0.0,
        stack_size: 1,
        perishable_ticks: None,
        consume_action: None,
        salvage: None,
        tool: None,
        size: crate::items::GridSize { w: 1, h: 1 },
        rotatable: false,
        inner_grid: None,
        equip_slots: Vec::new(),
        weapon_config: None,
        magazine_config: None,
        ammo_config: None,
        armor_config: None,
        attachment_config: None,
    }
}
