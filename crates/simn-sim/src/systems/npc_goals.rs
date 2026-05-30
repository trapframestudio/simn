//! NPC movement / behavior executor. Dispatches on the per-NPC
//! [`ActiveGoal`] written by `goal_arbitration`:
//!
//! - [`GoalKind::PursueTarget`] — pursue an aggro target until
//!   within engage range; combat does the work.
//! - [`GoalKind::SquadFollowObjective`] — derive the per-tick
//!   movement target from `SquadObjectives` (Patrol next base, Guard
//!   base pos, Investigate point, Wander long-march, Regroup squad
//!   centroid), with formation offset and faction-driven travel
//!   style.
//! - [`GoalKind::SoloIdleFsm`] — original Idle / MoveTo / RestAt
//!   cycle with the 30%-long-march wander, used when no
//!   higher-priority goal source fires.
//!
//! The branching priority that used to live here (Aggro overrides
//! Group overrides Idle) now lives in `goal_arbitration`. The split
//! keeps "which goal" and "how to satisfy it" at separate layers so
//! blackboard urgencies, scripted claims, personality biases, and
//! survival goals can slot in without growing this match.
//!
//! Pure system: no journal. Movement is deterministic from
//! component + resource state, so reload + re-tick recovers the same
//! trajectory.

use bevy_ecs::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::components::{
    ActiveGoal, Aggro, Base, GoalKind, InFaction, InRegion, Npc, NpcGoal, Path, Position, Rotation,
};

use crate::nav::{NavQueries, TravelStyle};
use crate::resources::{NpcPositionIndex, SimClock, SquadObjective, SquadObjectives};

// Per-tick diagnostic counter for A* calls in this system. Reset at
// the start of `tick_npc_goals`, bumped inside `advance_with_path`,
// logged at end-of-system if non-trivial.
thread_local! {
    static PATHFIND_CALLS_THIS_TICK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

use crate::behavior_config::BehaviorConfig;

const PURSUE_ARRIVE_SQ_M: f32 = 4.0;
const WAYPOINT_ARRIVE_M: f32 = 1.5;
const WAYPOINT_ARRIVE_SQ_M: f32 = WAYPOINT_ARRIVE_M * WAYPOINT_ARRIVE_M;

/// Request for a parallel A* pathfind. Collected during the per-NPC
/// pass over the NPC query and processed in batch via rayon after
/// the mutable iter releases. Letting the A* calls run across all
/// cores in parallel is the main "use the idle cores" win — single-
/// threaded the budget was 8 × 5ms = 40ms, parallel it's ~5-10ms
/// for the same 8 calls on an 8-core machine.
struct PathfindRequest {
    entity: Entity,
    region: crate::region::RegionId,
    from: [f32; 3],
    to: [f32; 3],
    style: TravelStyle,
}

/// Result of a parallel A* pathfind. Applied via `Commands::insert`
/// after the parallel batch completes — overwrites any existing
/// `Path` component. `waypoints == None` ⇒ pathfind failed; we
/// insert an empty-waypoint tombstone to avoid re-running A* every
/// tick for unreachable targets (see `advance_with_path` docs).
struct PathfindResult {
    entity: Entity,
    waypoints: Option<Vec<[f32; 3]>>,
    target: [f32; 3],
}

type NpcGoalRow<'a> = (
    Entity,
    &'a Npc,
    &'a InFaction,
    &'a InRegion,
    Mut<'a, ActiveGoal>,
    Mut<'a, Position>,
    Mut<'a, Rotation>,
    Mut<'a, NpcGoal>,
    Option<Mut<'a, Path>>,
    Option<&'a crate::components::CombatStance>,
    Option<Mut<'a, crate::components::DwellState>>,
    Option<&'a crate::components::BodyParts>,
);

#[allow(clippy::too_many_arguments)]
pub fn tick_npc_goals(
    clock: Res<SimClock>,
    bases: Query<(&Base, &InFaction, &InRegion, &Position)>,
    index: Res<NpcPositionIndex>,
    nav: Res<NavQueries>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    active_regions: Res<crate::resources::ActiveRegions>,
    mut objectives: ResMut<SquadObjectives>,
    mut interaction_areas: ResMut<crate::resources::InteractionAreas>,
    mut world_events: ResMut<crate::world_event_bus::WorldEventQueue>,
    mut npcs: Query<NpcGoalRow, Without<Base>>,
    groups: Query<&crate::components::Group>,
    cover_volumes: Res<crate::cover::CoverVolumes>,
    mut commands: Commands,
    _hash: Res<crate::resources::NpcSpatialHash>,
) {
    let _diag_t = crate::systems::SysTimer::new("tick_npc_goals");
    let _prof_guard = crate::systems::ProfGuard(
        std::time::Instant::now(),
        crate::systems::prof_slots::TICK_NPC_GOALS,
    );
    // Per-tick pathfind counter. Drops a line at end-of-system when
    // the count is non-trivial. The thread_local exists because the
    // budget check happens inside `advance_with_path` which doesn't
    // get a direct `&mut u32` channel through the recursion.
    PATHFIND_CALLS_THIS_TICK.with(|c| c.set(0));
    let bc = BehaviorConfig::load();
    let mv = &bc.movement;
    let ag = &bc.aggro;
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    let step = mv.walk_speed_mps * dt_s;
    let engage_range_m = mv.engage_range_m;
    let engage_range_sq = engage_range_m * engage_range_m;
    let relocate_threshold_sq = (engage_range_m * 0.5) * (engage_range_m * 0.5);
    let squad_arrive_sq = mv.squad_arrive_radius_m * mv.squad_arrive_radius_m;
    let member_arrive_sq = mv.squad_member_arrive_m * mv.squad_member_arrive_m;
    let mut pending_pathfinds: Vec<PathfindRequest> =
        Vec::with_capacity(mv.path_budget_per_tick as usize);

    for (
        entity,
        npc,
        npc_faction,
        npc_region,
        mut active,
        mut pos,
        mut rot,
        mut goal,
        path,
        combat_stance,
        mut dwell_state,
        body_parts,
    ) in npcs.iter_mut()
    {
        // Active-region filter. NPCs in offline regions freeze in
        // place — no movement, no pathfinding, no FSM transitions.
        // Matches the long-term offline-tier intent (regions without
        // observers run an abstract sim, not the online physical
        // path). Iteration cost is still O(N) but per-NPC work
        // collapses to the region check.
        if !active_regions.is_active(npc_region.0) {
            continue;
        }
        let mut path = path;
        // Leg damage cripples movement speed: an NPC limping on a
        // damaged leg covers less ground per tick. Uses the worse
        // of the two legs since a single bad leg slows the whole
        // body. Thresholds match the perception-relevant tiers so
        // movement degrades visibly before the NPC dies.
        let leg_speed_mult = match body_parts {
            Some(bp) => {
                let worst = bp.left_leg.min(bp.right_leg);
                if worst < 25.0 {
                    0.4
                } else if worst < 75.0 {
                    0.7
                } else {
                    1.0
                }
            }
            None => 1.0,
        };
        let step = step * leg_speed_mult;

        match active.kind {
            // Aggro pursuit. Bushwhack toward the live target.
            // Pathfinding lets NPCs detour around cliffs / impassable
            // terrain instead of straight-lining into them. If the
            // target despawned or migrated, fall through to the solo
            // FSM — the arbitrator will pick a new goal next tick.
            GoalKind::PursueTarget { target } => {
                let entry = index
                    .by_id
                    .get(&target)
                    .copied()
                    .filter(|e| e.region == npc_region.0 && e.health > 0.0);
                if let Some(entry) = entry {
                    // Per-NPC engagement-offset spot around the target.
                    // Without this, every pursuer walks to the target's
                    // exact position and they stack on a single point
                    // (the visual "clumping" bug). Each NPC's
                    // deterministic angle is derived from its stable
                    // `NpcId` so it claims the same slot every tick —
                    // no jitter from frame-to-frame angle changes.
                    //
                    // **Approach-only**: NPCs already within
                    // `RELOCATE_THRESHOLD_M` of the target keep their
                    // current position and just fire from there — moving
                    // mid-combat to a "preferred" slot reads as
                    // skittering and would break test scenarios that
                    // place NPCs at point-blank. Only NPCs *approaching*
                    // from far away (e.g. arriving from a base) get
                    // routed to their offset slot so the cluster they
                    // form is spread around the target arc instead of
                    // piled at the closest entry point.
                    let dist_sq_to_target = squared_2d(pos.0, entry.pos);
                    let dist_to_target = dist_sq_to_target.sqrt();

                    // Pursue-progress timeout: if the NPC hasn't gotten
                    // ag.pursue_progress_m closer in ag.pursue_timeout_ticks,
                    // give up — the target is likely unreachable.
                    let pp =
                        active
                            .pursue_progress
                            .get_or_insert(crate::components::PursueProgress {
                                pos: pos.0,
                                tick: clock.tick,
                            });
                    if clock.tick.wrapping_sub(pp.tick) >= ag.pursue_timeout_ticks {
                        let old_dist = squared_2d(pp.pos, entry.pos).sqrt();
                        if old_dist - dist_to_target < ag.pursue_progress_m {
                            commands.entity(entity).remove::<Aggro>();
                            active.pursue_progress = None;
                            continue;
                        }
                        *pp = crate::components::PursueProgress {
                            pos: pos.0,
                            tick: clock.tick,
                        };
                    }

                    // Stance-driven movement override: InCover moves
                    // to cover position, Retreating moves away, Flanking
                    // takes a lateral offset. Falls through to the
                    // existing engagement-slot logic for Approaching/Firing.
                    let stance_override: Option<([f32; 3], f32)> = match combat_stance.copied() {
                        Some(crate::components::CombatStance::InCover { volume_id, .. }) => {
                            cover_volumes
                                .by_region
                                .get(&npc_region.0)
                                .and_then(|vols| vols.iter().find(|v| v.id == volume_id))
                                .map(|v| (v.pos, 4.0))
                        }
                        Some(crate::components::CombatStance::Retreating) => {
                            let away = [
                                pos.0[0] - (entry.pos[0] - pos.0[0]).signum() * 60.0,
                                pos.0[1],
                                pos.0[2] - (entry.pos[2] - pos.0[2]).signum() * 60.0,
                            ];
                            Some((away, 100.0))
                        }
                        Some(crate::components::CombatStance::Flanking) => {
                            let perp_angle = (npc.id.0 as f32)
                                .mul_add(2.399_963_2, 0.0)
                                .rem_euclid(std::f32::consts::TAU);
                            let flank_r = engage_range_m * 0.9;
                            let fx = entry.pos[0] + flank_r * perp_angle.cos();
                            let fz = entry.pos[2] + flank_r * perp_angle.sin();
                            Some(([fx, entry.pos[1], fz], PURSUE_ARRIVE_SQ_M))
                        }
                        Some(crate::components::CombatStance::Suppressed { .. }) => {
                            Some((pos.0, 0.1))
                        }
                        // Approaching + Firing: use the normal
                        // engagement-offset slot logic below (no override).
                        // This keeps NPCs moving — they walk to their
                        // per-NPC engagement slot around the target
                        // instead of freezing in place.
                        _ => None,
                    };

                    // Per-NPC deterministic hash for angle + radius.
                    let id_angle = (npc.id.0 as f32)
                        .mul_add(2.399_963_2, 0.0)
                        .rem_euclid(std::f32::consts::TAU);
                    let id_radius_frac =
                        ((npc.id.0.wrapping_mul(0x517CC1B7) >> 16) & 0xFFFF) as f32 / 65535.0;

                    let (movement_target, arrive_sq) = if let Some((mt, asq)) = stance_override {
                        (mt, asq)
                    } else if dist_sq_to_target < relocate_threshold_sq {
                        // Within close engagement — strafe around
                        // the target. Each NPC has its own phase
                        // (from id) and radius (varied so they don't
                        // all orbit on the same ring).
                        if let Some(crate::components::CombatStance::Firing { since_tick }) =
                            combat_stance.copied()
                        {
                            let in_combat_for = clock.tick.saturating_sub(since_tick);
                            if in_combat_for > 200 {
                                // Per-NPC strafe phase so NPCs don't
                                // move in lockstep.
                                let npc_phase = (npc.id.0 % 7) as f32 * 0.9;
                                let strafe_angle =
                                    id_angle + (clock.tick as f32 / 200.0 + npc_phase) * 0.7;
                                let strafe_r = 4.0 + id_radius_frac * 10.0;
                                let sx = entry.pos[0] + strafe_r * strafe_angle.cos();
                                let sz = entry.pos[2] + strafe_r * strafe_angle.sin();
                                ([sx, entry.pos[1], sz], PURSUE_ARRIVE_SQ_M)
                            } else {
                                (entry.pos, engage_range_sq)
                            }
                        } else {
                            (entry.pos, engage_range_sq)
                        }
                    } else {
                        // Approaching from distance — spread on a
                        // ring with varied radius so NPCs don't
                        // stack at the same distance.
                        let ring_r = engage_range_m * (0.6 + id_radius_frac * 0.6);
                        (
                            [
                                entry.pos[0] + ring_r * id_angle.cos(),
                                entry.pos[1],
                                entry.pos[2] + ring_r * id_angle.sin(),
                            ],
                            PURSUE_ARRIVE_SQ_M,
                        )
                    };
                    advance_with_path(
                        &mut pending_pathfinds,
                        entity,
                        &mut path,
                        movement_target,
                        TravelStyle::Bushwhacker,
                        npc_region.0,
                        &clock,
                        &mut pos,
                        &mut rot,
                        step,
                        arrive_sq,
                    );
                    *goal = NpcGoal::RestAt {
                        until_tick: clock.tick.wrapping_add(20),
                    };
                    continue;
                }
                // Target gone: hold position this tick; arbitrator
                // recomputes next tick and demotes off PursueTarget.
                continue;
            }
            // Squad objective execution. Re-derives the per-tick
            // target from `SquadObjectives` so the formation responds
            // to live patrol-leg state and squad centroid drift.
            GoalKind::SquadFollowObjective => {
                let Ok(g) = groups.get(entity) else {
                    continue;
                };
                let Some(target) = squad_target(&objectives, g.id, &index, pos.0) else {
                    continue;
                };
                let centroid = index
                    .group_centroids
                    .get(&g.id)
                    .map(|c| c.pos)
                    .unwrap_or(pos.0);
                let objective_ref = objectives.by_group.get(&g.id).map(|s| &s.objective);
                let offset = formation_offset(npc.id.0, g.id, centroid, target, objective_ref);
                let effective_target = [target[0] + offset[0], target[1], target[2] + offset[2]];
                // Don't keep walking if already at the formation slot.
                let at_slot = squared_2d(pos.0, effective_target) <= member_arrive_sq;
                if !at_slot {
                    let style = style_for(npc_faction.0, &registry, objective_ref);
                    advance_with_path(
                        &mut pending_pathfinds,
                        entity,
                        &mut path,
                        effective_target,
                        style,
                        npc_region.0,
                        &clock,
                        &mut pos,
                        &mut rot,
                        step,
                        member_arrive_sq,
                    );
                }
                let squad_arrived = squared_2d(target, centroid) <= squad_arrive_sq;
                if squad_arrived {
                    advance_patrol(&mut objectives, g.id);
                    if let Some(state) = objectives.by_group.get(&g.id) {
                        if let SquadObjective::Rest {
                            area_id: Some(id), ..
                        } = &state.objective
                        {
                            if !interaction_areas.is_started(npc.id.0, id) {
                                let kind = interaction_areas
                                    .by_id
                                    .get(id)
                                    .and_then(|&(region, idx)| {
                                        interaction_areas
                                            .by_region
                                            .get(&region)
                                            .and_then(|areas| areas.get(idx))
                                            .map(|a| a.kind.clone())
                                    })
                                    .unwrap_or_default();
                                interaction_areas.mark_started(npc.id.0, id);
                                world_events.push(
                                    crate::world_event_bus::WorldEventKind::InteractionStarted {
                                        npc_id: npc.id,
                                        area_id: id.clone(),
                                        kind,
                                    },
                                    pos.0,
                                    npc_region.0,
                                    clock.tick,
                                    1,
                                );
                            }
                        }
                    }
                    // Dwell duration scales to the in-game day/night
                    // cycle (7200s real = 24h game → 1 real second ≈
                    // 12 game seconds). Guards hold a shift for hours,
                    // resting lasts an hour+, patrol waypoints get a
                    // brief tactical pause.
                    let objective_ref_here = objectives.by_group.get(&g.id).map(|s| &s.objective);
                    let base_dwell = match objective_ref_here {
                        Some(SquadObjective::Guard { .. }) => bc.planning.dwell.guard_ticks,
                        Some(SquadObjective::Rest { .. }) => bc.planning.dwell.rest_ticks,
                        Some(SquadObjective::Patrol { .. }) => {
                            bc.planning.dwell.patrol_waypoint_ticks
                        }
                        _ => bc.planning.dwell.default_ticks,
                    };
                    // Per-NPC ±jitter_frac on dwell so a squad doesn't
                    // exit a Guard/Rest dwell on the same tick. Stable
                    // per (npc.id, objective set_at) so the jitter is
                    // deterministic and doesn't shift mid-dwell.
                    let jitter_frac = bc.planning.dwell.jitter_frac.clamp(0.0, 0.9);
                    let dwell = if jitter_frac > 0.0 && base_dwell > 0 {
                        let salt = objectives
                            .by_group
                            .get(&g.id)
                            .map(|s| s.set_at_tick)
                            .unwrap_or(0);
                        let mut jrng = ChaCha8Rng::seed_from_u64(
                            npc.id.0.wrapping_mul(0xD737_3D85_C24F_3F47) ^ salt,
                        );
                        // Range [1 - frac, 1 + frac].
                        let f = 1.0 + jrng.gen_range(-jitter_frac..=jitter_frac);
                        ((base_dwell as f32) * f).max(1.0) as u64
                    } else {
                        base_dwell
                    };
                    // Communicate the visual pose for renderers: Rest at
                    // campfire / interaction area → sitting; everything
                    // else stays standing. Only re-init `last_shift_tick`
                    // on first arrival (no existing DwellState) — this
                    // block re-runs every tick while squad_arrived, so
                    // unconditionally resetting it would prevent the
                    // shift threshold from ever being crossed.
                    let pose = match objective_ref_here {
                        Some(SquadObjective::Rest { .. }) => crate::components::DwellPose::Sitting,
                        _ => crate::components::DwellPose::Standing,
                    };
                    if let Some(ref mut ds) = dwell_state {
                        ds.pose = pose;
                    } else {
                        commands
                            .entity(entity)
                            .insert(crate::components::DwellState {
                                pose,
                                last_shift_tick: clock.tick,
                            });
                    }
                    // Guard shift: periodically nudge dwelling NPCs at
                    // Guard posts so they visibly shift weight instead of
                    // standing perfectly still for the entire ~20 min
                    // guard tenure. Only applies to Standing pose at
                    // Guard objectives.
                    let shift_interval = bc.planning.dwell.guard_shift_interval_ticks;
                    let shift_radius = bc.planning.dwell.guard_shift_radius_m;
                    if shift_interval > 0
                        && shift_radius > 0.0
                        && pose == crate::components::DwellPose::Standing
                        && matches!(objective_ref_here, Some(SquadObjective::Guard { .. }))
                    {
                        if let Some(ref mut ds) = dwell_state {
                            let due = clock.tick.wrapping_sub(ds.last_shift_tick) >= shift_interval;
                            if due {
                                let mut srng = ChaCha8Rng::seed_from_u64(
                                    npc.id.0.wrapping_mul(0xA24B_AED4_963E_E10B) ^ clock.tick,
                                );
                                let dx = srng.gen_range(-shift_radius..=shift_radius);
                                let dz = srng.gen_range(-shift_radius..=shift_radius);
                                pos.0[0] += dx;
                                pos.0[2] += dz;
                                ds.last_shift_tick = clock.tick;
                            }
                        }
                    }
                    *goal = NpcGoal::RestAt {
                        until_tick: clock.tick.wrapping_add(dwell),
                    };
                } else {
                    *goal = NpcGoal::MoveTo {
                        target: effective_target,
                    };
                }
                continue;
            }
            // Blackboard-urgency reactions. Both kinds carry a
            // world-space position; head there at urgent travel
            // style. Halt at member-arrival radius (matches squad
            // arrival precision; tighter than solo `mv.arrive_radius_m`)
            // and settle into `RestAt` so the arbitrator demotes off
            // the urgency on the next tick once the blackboard entry
            // ages out.
            //
            // Formation offset: without this, every member of a squad
            // reacting to the same `HeardGunshot` / `UnderFireAt` /
            // `DownedAlly` targets the exact same world position,
            // arrives within 1 m, and stacks visibly. Apply the same
            // per-NPC slot offset that `SquadFollowObjective` uses so
            // members spread around the urgency point (circular
            // formation since there's no inherent "travel direction"
            // here — they're converging on a fixed point). Lone NPCs
            // (no Group) collapse to a deterministic per-id jitter
            // via the same offset function.
            //
            // Real take-cover / suppress-back / revive behavior lands
            // with tactical AI; this is the move-to substrate.
            GoalKind::InvestigateAt { pos: target }
            | GoalKind::RegroupOnAlly { pos: target, .. } => {
                let group_id = groups.get(entity).ok().map(|g| g.id).unwrap_or(0);
                let centroid = index
                    .group_centroids
                    .get(&group_id)
                    .map(|c| c.pos)
                    .unwrap_or(pos.0);
                // Use a circular formation (no travel-direction
                // anchor) — pass `target == centroid` to trigger the
                // `len_sq < 1.0` branch in `formation_offset`, which
                // yields the 8-slot circular spread at radius 3 m.
                let offset = formation_offset(npc.id.0, group_id, centroid, centroid, None);
                let effective_target = [target[0] + offset[0], target[1], target[2] + offset[2]];
                advance_with_path(
                    &mut pending_pathfinds,
                    entity,
                    &mut path,
                    effective_target,
                    TravelStyle::Bushwhacker,
                    npc_region.0,
                    &clock,
                    &mut pos,
                    &mut rot,
                    step,
                    member_arrive_sq,
                );
                let arrived = squared_2d(effective_target, pos.0) <= member_arrive_sq;
                if arrived {
                    *goal = NpcGoal::RestAt {
                        until_tick: clock.tick.wrapping_add(40),
                    };
                } else {
                    *goal = NpcGoal::MoveTo {
                        target: effective_target,
                    };
                }
                continue;
            }
            // Socialize: collapse the squad to a tight ring around the
            // group centroid, face inward each tick (so members visibly
            // look at each other), dwell, then return control to the
            // arbitrator. Distinct from `SquadFollowObjective` which
            // holds formation slot positions: Socialize members
            // converge inside the formation, breaking the parade-line
            // look during downtime.
            GoalKind::Socialize { target_pos } => {
                let group_id = groups.get(entity).ok().map(|g| g.id).unwrap_or(0);
                // Each NPC picks a per-id phase on a 2 m social ring so
                // they don't all collapse to the centroid (visual
                // overlap). The ring is tight enough to read as a
                // gathering, loose enough that wound/animation systems
                // can resolve.
                let phase = (npc.id.0 as f32 * 0.61803_f32).fract() * std::f32::consts::TAU;
                let ring_r = 2.0_f32;
                let effective_target = [
                    target_pos[0] + ring_r * phase.cos(),
                    target_pos[1],
                    target_pos[2] + ring_r * phase.sin(),
                ];
                let arrived = squared_2d(effective_target, pos.0) <= member_arrive_sq;
                if !arrived {
                    advance_with_path(
                        &mut pending_pathfinds,
                        entity,
                        &mut path,
                        effective_target,
                        TravelStyle::Bushwhacker,
                        npc_region.0,
                        &clock,
                        &mut pos,
                        &mut rot,
                        step,
                        member_arrive_sq,
                    );
                }
                // Face the gathering centroid each tick — recomputed
                // continuously so as squad-mates shift the visible
                // attention shifts with them.
                let dx = target_pos[0] - pos.0[0];
                let dz = target_pos[2] - pos.0[2];
                if dx * dx + dz * dz > 0.01 {
                    rot.0 = dz.atan2(dx);
                }
                if arrived {
                    // Settle into a RestAt with a Sitting pose so the
                    // renderer plays a gathering animation. Length is
                    // jittered ±25% per NPC so members peel off the
                    // gathering at different times.
                    let mut srng = ChaCha8Rng::seed_from_u64(
                        npc.id.0.wrapping_mul(0x4B65_3D7F_2A91_E0BD) ^ clock.tick,
                    );
                    let base: u64 = 900;
                    let jittered = (base as f32 * srng.gen_range(0.75..=1.25)) as u64;
                    *goal = NpcGoal::RestAt {
                        until_tick: clock.tick.wrapping_add(jittered),
                    };
                    let new_dwell = crate::components::DwellState {
                        pose: crate::components::DwellPose::Sitting,
                        last_shift_tick: clock.tick,
                    };
                    if let Some(ref mut ds) = dwell_state {
                        **ds = new_dwell;
                    } else {
                        commands.entity(entity).insert(new_dwell);
                    }
                } else {
                    *goal = NpcGoal::MoveTo {
                        target: effective_target,
                    };
                }
                let _ = group_id;
                continue;
            }
            // Hunt: curious NPCs walk to a POI (Stash / Lookout /
            // Workbench) they don't own, dwell briefly while inspecting,
            // then clear the goal. The dwell length is jittered per-NPC
            // so squad members peel off the POI at different times.
            GoalKind::Hunt { target_pos, .. } => {
                let group_id = groups.get(entity).ok().map(|g| g.id).unwrap_or(0);
                let centroid = index
                    .group_centroids
                    .get(&group_id)
                    .map(|c| c.pos)
                    .unwrap_or(pos.0);
                let offset = formation_offset(npc.id.0, group_id, centroid, centroid, None);
                let effective_target = [
                    target_pos[0] + offset[0],
                    target_pos[1],
                    target_pos[2] + offset[2],
                ];
                let arrived = squared_2d(effective_target, pos.0) <= member_arrive_sq;
                if !arrived {
                    advance_with_path(
                        &mut pending_pathfinds,
                        entity,
                        &mut path,
                        effective_target,
                        TravelStyle::Bushwhacker,
                        npc_region.0,
                        &clock,
                        &mut pos,
                        &mut rot,
                        step,
                        member_arrive_sq,
                    );
                }
                if arrived {
                    let mut hrng = ChaCha8Rng::seed_from_u64(
                        npc.id.0.wrapping_mul(0x7F26_C0AE_2A7A_C0E5) ^ clock.tick,
                    );
                    let base: u64 = 600;
                    let jittered = (base as f32 * hrng.gen_range(0.7..=1.4)) as u64;
                    *goal = NpcGoal::RestAt {
                        until_tick: clock.tick.wrapping_add(jittered),
                    };
                    let new_dwell = crate::components::DwellState {
                        pose: crate::components::DwellPose::Crouching,
                        last_shift_tick: clock.tick,
                    };
                    if let Some(ref mut ds) = dwell_state {
                        **ds = new_dwell;
                    } else {
                        commands.entity(entity).insert(new_dwell);
                    }
                } else {
                    *goal = NpcGoal::MoveTo {
                        target: effective_target,
                    };
                }
                continue;
            }
            // Loot: greedy NPCs walk to a corpse container and dwell
            // ~15-30s "rifling through pockets". Inventory transfer is
            // a future slice — the visible payoff is the walk + dwell.
            GoalKind::Loot { target_pos, .. } => {
                let arrived = squared_2d(target_pos, pos.0) <= 9.0; // 3m radius
                if !arrived {
                    advance_with_path(
                        &mut pending_pathfinds,
                        entity,
                        &mut path,
                        target_pos,
                        TravelStyle::Bushwhacker,
                        npc_region.0,
                        &clock,
                        &mut pos,
                        &mut rot,
                        step,
                        9.0,
                    );
                }
                if arrived {
                    let mut lrng = ChaCha8Rng::seed_from_u64(
                        npc.id.0.wrapping_mul(0x5E2D_8B4D_C0E3_7F45) ^ clock.tick,
                    );
                    let base: u64 = 400;
                    let jittered = (base as f32 * lrng.gen_range(0.75..=1.5)) as u64;
                    *goal = NpcGoal::RestAt {
                        until_tick: clock.tick.wrapping_add(jittered),
                    };
                    let new_dwell = crate::components::DwellState {
                        pose: crate::components::DwellPose::Crouching,
                        last_shift_tick: clock.tick,
                    };
                    if let Some(ref mut ds) = dwell_state {
                        **ds = new_dwell;
                    } else {
                        commands.entity(entity).insert(new_dwell);
                    }
                } else {
                    *goal = NpcGoal::MoveTo { target: target_pos };
                }
                continue;
            }
            // SeekMedical: critically wounded NPC limping toward the
            // nearest same-faction rest spot. Skips combat decisions
            // (the priority-220 source preempts aggro), so the NPC
            // doesn't pause to fight while fleeing. Once arrived, it
            // dwells in Sitting pose until wounds heal or it dies.
            GoalKind::SeekMedical { target_pos } => {
                let arrived = squared_2d(target_pos, pos.0) <= member_arrive_sq;
                if !arrived {
                    advance_with_path(
                        &mut pending_pathfinds,
                        entity,
                        &mut path,
                        target_pos,
                        TravelStyle::Bushwhacker,
                        npc_region.0,
                        &clock,
                        &mut pos,
                        &mut rot,
                        step,
                        member_arrive_sq,
                    );
                    *goal = NpcGoal::MoveTo { target: target_pos };
                } else {
                    // Stay at the rest spot indefinitely (long until_tick).
                    // The arbiter will demote this candidate as soon as
                    // BodyParts::vital_min() recovers above the threshold.
                    *goal = NpcGoal::RestAt {
                        until_tick: clock.tick.wrapping_add(6000),
                    };
                    let new_dwell = crate::components::DwellState {
                        pose: crate::components::DwellPose::Sitting,
                        last_shift_tick: clock.tick,
                    };
                    if let Some(ref mut ds) = dwell_state {
                        **ds = new_dwell;
                    } else {
                        commands.entity(entity).insert(new_dwell);
                    }
                }
                continue;
            }
            GoalKind::SoloIdleFsm | GoalKind::Bloodsport => {
                // Bloodsport is the only remaining personality drive
                // placeholder — deferred until arena concept lands.
            }
        }

        // Determinism: do NOT mix `entity.to_bits()` into the RNG seed
        // - Bevy's entity ids aren't stable across sim instances, so
        // two same-seed sims would produce different solo movements.
        // `npc.id` is the persistent identifier set by `NpcIdCounter`
        // at spawn; the multiplier breaks the obvious xor symmetry
        // when tick == npc.id.
        let mut rng =
            ChaCha8Rng::seed_from_u64(npc.id.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ clock.tick);
        let target_seed = npc.id.0 ^ (clock.tick / 200);
        let mut target_rng = ChaCha8Rng::seed_from_u64(target_seed);

        let next_goal = match *goal {
            NpcGoal::Idle { until_tick } => {
                if clock.tick >= until_tick {
                    let target = pick_solo_target(
                        &mut target_rng,
                        npc_faction.0,
                        npc_region.0,
                        pos.0,
                        &bases,
                    );
                    Some(NpcGoal::MoveTo { target })
                } else {
                    None
                }
            }
            NpcGoal::MoveTo { target } => {
                let dx = target[0] - pos.0[0];
                let dz = target[2] - pos.0[2];
                let dist = (dx * dx + dz * dz).sqrt();
                if dist <= mv.arrive_radius_m {
                    let rest_ticks = rng.gen_range(100..=400u64);
                    Some(NpcGoal::RestAt {
                        until_tick: clock.tick.wrapping_add(rest_ticks),
                    })
                } else {
                    // Solo NPC walking somewhere uses Mixed style
                    // (default mild road preference). The arrival
                    // radius for solo MoveTo is 10 m (mv.arrive_radius_m),
                    // looser than squad/aggro arrivals so the FSM
                    // transitions into RestAt without dribbling.
                    advance_with_path(
                        &mut pending_pathfinds,
                        entity,
                        &mut path,
                        target,
                        TravelStyle::Mixed,
                        npc_region.0,
                        &clock,
                        &mut pos,
                        &mut rot,
                        step,
                        mv.arrive_radius_m * mv.arrive_radius_m,
                    );
                    None
                }
            }
            NpcGoal::RestAt { until_tick } => {
                if clock.tick >= until_tick {
                    let idle_ticks = rng.gen_range(50..=200u64);
                    Some(NpcGoal::Idle {
                        until_tick: clock.tick.wrapping_add(idle_ticks),
                    })
                } else {
                    None
                }
            }
        };

        if let Some(g) = next_goal {
            *goal = g;
        }
    }

    // Pass 2 (parallel, immutable): run all queued A* calls across
    // CPU cores via rayon. Each `nav.path(...)` is data-independent
    // — the `NavQuery` trait is `Send + Sync` and the underlying
    // grid is read-only during query. The whole batch typically
    // completes in `max_call_cost / cores` ≈ 5-10ms on an 8-core
    // machine vs `budget × max_call_cost` ≈ 40ms single-threaded.
    let pathfind_results: Vec<PathfindResult> = if pending_pathfinds.is_empty() {
        Vec::new()
    } else {
        use rayon::prelude::*;
        pending_pathfinds
            .into_par_iter()
            .map(|req| {
                let waypoints = nav.path(req.region, req.from, req.to, req.style);
                PathfindResult {
                    entity: req.entity,
                    waypoints,
                    target: req.to,
                }
            })
            .collect()
    };

    // Pass 3 (sequential, commands): apply results. Either a fresh
    // successful Path (waypoints.len() ≥ 2) or an empty-waypoint
    // tombstone (failure or degenerate). Both overwrite any
    // existing Path component via `Commands::insert`. The freshly
    // applied Path becomes visible to `advance_with_path` next
    // tick after apply_deferred runs at the next schedule
    // boundary.
    for r in pathfind_results {
        match r.waypoints {
            Some(wp) if wp.len() >= 2 => {
                commands.entity(r.entity).insert(Path {
                    waypoints: wp,
                    current: 1,
                    computed_tick: clock.tick,
                    target: r.target,
                });
            }
            _ => {
                commands.entity(r.entity).insert(Path {
                    waypoints: Vec::new(),
                    current: 0,
                    computed_tick: clock.tick,
                    target: r.target,
                });
            }
        }
    }

    let pathfinds = PATHFIND_CALLS_THIS_TICK.with(|c| c.get());
    if crate::systems::is_verbose_logging() && pathfinds > 10 {
        eprintln!(
            "[tick_npc_goals tick={}] pathfind_calls={}",
            clock.tick, pathfinds
        );
    }

    // NPC-NPC separation is handled by Godot's CharacterBody3D
    // collision (move_and_slide). The sim doesn't need a nudge pass.
}

fn move_toward(
    pos: &mut Position,
    rot: &mut Rotation,
    target: [f32; 3],
    step: f32,
    arrive_sq: f32,
) {
    let dx = target[0] - pos.0[0];
    let dz = target[2] - pos.0[2];
    let dist_sq = dx * dx + dz * dz;
    if dist_sq <= arrive_sq {
        return;
    }
    let dist = dist_sq.sqrt();
    let nx = dx / dist;
    let nz = dz / dist;
    pos.0[0] += nx * step;
    pos.0[2] += nz * step;
    rot.0 = nz.atan2(nx);
}

/// Step toward `target` using the cached [`Path`] when one exists,
/// else enqueue a parallel A* request so the system can batch-run
/// pathfinds across cores after the mutable iter releases, else
/// straight-line so the NPC isn't completely stuck.
///
/// Arrival semantics live with the caller: `arrive_sq_to_target` is
/// the squared distance to `target` at which the NPC stops (engage
/// range, formation arrival, etc.). Path waypoint arrival is
/// independent — when the NPC is within [`WAYPOINT_ARRIVE_M`] of the
/// current waypoint, it advances to the next; on path exhaustion it
/// straight-lines for the final approach.
///
/// **Why no inline A*.** Each `nav.path()` call costs ~1-5ms in the
/// happy case and up to `~MAX_NODES_EXPANDED × ~1µs` in the cap-
/// hitting case. Inline single-threaded that meant any tick where
/// many NPCs need new paths blew the frame budget. The system-level
/// `tick_npc_goals` now collects all requests into `pending`, runs
/// them via `rayon::par_iter` after the mutable NPC iter releases,
/// and applies results via `Commands::insert` in a third pass. NPCs
/// whose request was queued straight-line this tick; the fresh path
/// arrives on the next tick.
#[allow(clippy::too_many_arguments)]
fn advance_with_path(
    pending: &mut Vec<PathfindRequest>,
    entity: Entity,
    path: &mut Option<Mut<'_, Path>>,
    target: [f32; 3],
    style: TravelStyle,
    region: crate::region::RegionId,
    clock: &SimClock,
    pos: &mut Position,
    rot: &mut Rotation,
    step: f32,
    arrive_sq_to_target: f32,
) {
    // Already arrived at target? Stop moving along the path, but
    // KEEP the Path component cached so the next-tick recompute
    // check has something to compare against. NPCs whose formation
    // target jitters by sub-`PATH_RECOMPUTE_DIST_M` per tick used to
    // re-run A* every cycle (arrive → drop → next-tick no path →
    // recompute → arrive); keeping the cached path lets the drift
    // check short-circuit until the target genuinely moves >8m.
    if squared_2d(pos.0, target) <= arrive_sq_to_target {
        return;
    }

    // Should we recompute? Cases: no path, target drifted, path is stale.
    let bcfg = BehaviorConfig::load();
    let recompute_dist_sq =
        bcfg.movement.path_recompute_dist_m * bcfg.movement.path_recompute_dist_m;
    let needs_recompute = match path.as_deref() {
        None => true,
        Some(p) => {
            squared_2d(p.target, target) > recompute_dist_sq
                || clock.tick.wrapping_sub(p.computed_tick) > bcfg.movement.path_max_age_ticks
        }
    };

    if needs_recompute {
        let already = PATHFIND_CALLS_THIS_TICK.with(|c| c.get());
        if already >= bcfg.movement.path_budget_per_tick {
            move_toward(pos, rot, target, step, arrive_sq_to_target);
            return;
        }
        PATHFIND_CALLS_THIS_TICK.with(|c| c.set(already + 1));
        pending.push(PathfindRequest {
            entity,
            region,
            from: pos.0,
            to: target,
            style,
        });
        // Straight-line this tick. Next tick the freshly-computed
        // (or tombstoned) Path lands via `Commands::insert` and we
        // walk it normally.
        move_toward(pos, rot, target, step, arrive_sq_to_target);
        return;
    }

    // Walk along the cached path.
    let Some(p) = path.as_deref_mut() else {
        move_toward(pos, rot, target, step, arrive_sq_to_target);
        return;
    };

    // Empty-waypoint Path = pathfinding-failure tombstone (see the
    // `_ =>` branch above). Don't remove it on this tick — that's
    // the whole point. Just straight-line. The recompute check
    // above will refresh the tombstone when target drifts or it
    // ages out, at which point we retry A*.
    if p.waypoints.is_empty() {
        move_toward(pos, rot, target, step, arrive_sq_to_target);
        return;
    }

    while (p.current as usize) < p.waypoints.len() {
        let wp = p.waypoints[p.current as usize];
        if squared_2d(pos.0, wp) <= WAYPOINT_ARRIVE_SQ_M {
            p.current = p.current.saturating_add(1);
            continue;
        }
        // Step toward the waypoint. Target arrival is checked against
        // the original target so we don't overshoot when the last
        // waypoint sits just past the engagement / formation circle.
        move_toward(pos, rot, wp, step, 0.0);
        return;
    }

    // Path exhausted (walked past the last waypoint) but we haven't
    // arrived yet — straight-line for the final approach. KEEP the
    // Path component so subsequent ticks see a cached `target` and
    // skip the recompute unless the goal genuinely drifts. Without
    // this retention the next tick would have `Path == None` and
    // re-run A*, which dominated `tick_npc_goals` cost in the
    // pre-fix profile.
    move_toward(pos, rot, target, step, arrive_sq_to_target);
}

/// Pick a [`TravelStyle`] from faction archetype + active squad
/// objective. Roughly:
///
/// - **Patrol**: organized factions hug roads (PWA / Linemen / Federal /
///   RevereGuard / CorporateResearch); off-road factions stay
///   bushwhackers even on patrol (Wanderers, NoosphereWorshippers).
/// - **Investigate / Explore**: bushwhacker for everyone (no roads to
///   trust at unknown coords).
/// - **Other objectives or none**: faction default (Mixed for
///   organized, Bushwhacker for off-roaders).
fn style_for(
    faction: crate::faction::registry::FactionId,
    registry: &crate::faction::registry::FactionRegistry,
    objective: Option<&SquadObjective>,
) -> TravelStyle {
    // Read the per-faction `road_friendly` flag from the registry —
    // configurable in `factions.toml` per the registry plan, no
    // hardcoded list here.
    let road_friendly = registry.def(faction).road_friendly;
    match objective {
        Some(SquadObjective::Patrol { .. }) if road_friendly => TravelStyle::RoadHugger,
        Some(SquadObjective::Investigate { .. } | SquadObjective::Explore { .. }) => {
            TravelStyle::Bushwhacker
        }
        _ => {
            if road_friendly {
                TravelStyle::Mixed
            } else {
                TravelStyle::Bushwhacker
            }
        }
    }
}

fn squared_2d(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dz = a[2] - b[2];
    dx * dx + dz * dz
}

/// Resolve the per-tick movement target for a member of `group_id`.
/// `here` is the member's own position (used for the long-march
/// fallback if there's no objective).
fn squad_target(
    objectives: &SquadObjectives,
    group_id: u64,
    index: &NpcPositionIndex,
    here: [f32; 3],
) -> Option<[f32; 3]> {
    let state = objectives.by_group.get(&group_id)?;
    // First-spawn dispersion overrides the active objective's
    // nominal target until the centroid has walked out from the
    // spawn point. Cleared by `squad_planner` once arrived.
    if let Some(dt) = state.disperse_target {
        return Some(dt);
    }
    match &state.objective {
        SquadObjective::Patrol {
            route, current_idx, ..
        } => {
            route.get(*current_idx).copied().or_else(|| {
                // Empty/exhausted route — fall back to centroid.
                index.group_centroids.get(&group_id).map(|c| c.pos)
            })
        }
        SquadObjective::Guard { base_pos, .. } => Some(*base_pos),
        SquadObjective::Rest { base_pos, .. } => Some(*base_pos),
        SquadObjective::Explore { portal_pos, .. } => Some(*portal_pos),
        SquadObjective::Relieve { dest_pos, .. } => Some(*dest_pos),
        SquadObjective::Investigate { target, .. } => Some(*target),
        SquadObjective::Wander { .. } => {
            // Prefer the squad-planner's `wander_drift_target` so the
            // squad actually meanders to new ground every 30–60 s
            // instead of standing on the centroid. Falls back to the
            // centroid for any transient window where the drift
            // target hasn't been seeded yet (one planner tick at
            // most).
            let _ = here;
            if let Some(t) = state.wander_drift_target {
                return Some(t);
            }
            let centroid = index.group_centroids.get(&group_id)?.pos;
            Some([centroid[0], 0.0, centroid[2]])
        }
        SquadObjective::Regroup { rally_pos, .. } => Some(*rally_pos),
    }
}

/// Per-member offset from the squad's raw objective target so
/// members occupy distinct positions and don't stack on the
/// anchor point. Offsets are stable per-NPC (derived from
/// `npc_id`) so a given member always takes the same slot across
/// ticks and you don't see them jitter between positions.
///
/// **Squad phase (2026-05-23)**: when multiple squads pick the
/// same anchor (e.g. several Rest objectives target the same
/// authored base, or several Patrol legs share a waypoint),
/// each squad's 8-slot ring would otherwise pile on the same 8
/// positions. Each squad now gets a `group_id`-derived angular
/// phase + radius shift so adjacent squads occupy interleaved
/// rings around the shared anchor instead of stacking. Within a
/// squad the slot layout is unchanged.
///
/// - **Rest / Guard / Regroup** — circular spread, 8 slots, 4-6m
///   radius (squad-phase modulated). Gives everyone their own space.
/// - **Patrol / Explore / Investigate** — line formation
///   perpendicular to travel direction, ±3m spacing from center
///   slot. Squad phase rotates the perpendicular so converging
///   patrols stay distinct.
/// - **Wander / none** — small random-looking jitter (still
///   stable per-NPC) so members spread loosely without a
///   formation.
fn formation_offset(
    npc_id: u64,
    group_id: u64,
    centroid: [f32; 3],
    target: [f32; 3],
    objective: Option<&SquadObjective>,
) -> [f32; 3] {
    let slot = (npc_id as u32) % 8;
    // Squad-phase angle in [0, TAU). Wide spread between adjacent
    // group ids via a 32-bit prime multiplier (golden-ratio
    // descendant). Keeps the formation deterministic and avoids
    // overlap when multiple squads share an anchor.
    let group_hash = (group_id as u32).wrapping_mul(2_654_435_761);
    let squad_phase = (group_hash as f32 / u32::MAX as f32) * std::f32::consts::TAU;
    // Wider radial variation per squad so when many squads claim
    // the same anchor (e.g. several Rest objectives at the same
    // faction Outpost — usually unavoidable when a region only has
    // 1-2 same-faction bases) the squads' rings land on distinct
    // radii instead of stacking on the same ~4-7 m band. With
    // `% 6 × 4.0` the per-squad radii span 0..20 m, so 6+ co-located
    // squads spread out over a 20 m disk rather than a tight clump.
    let squad_radius_jitter = ((group_id as u32 % 6) as f32) * 4.0;
    match objective {
        // Guard rings spread wider than Rest/Regroup so a 3-NPC
        // squad doesn't visibly clump at a single guard point.
        // Per-slot radial jitter (npc_id-keyed) breaks the
        // perfect-circle look so the formation reads as
        // independent positions rather than a UFO ring.
        Some(SquadObjective::Guard { .. }) => {
            // Use a continuous per-NPC angle instead of the 8-slot
            // wraparound — a 12-NPC guard squad had ≥4 members
            // share each slot via `npc_id % 8` and physically pile
            // at the same world position. Hashing the full id gives
            // every member a distinct angle on the ring regardless
            // of squad size.
            let id_hash = (npc_id as u32).wrapping_mul(2_246_822_519);
            let npc_angle = ((id_hash & 0xFFFF) as f32 / 65535.0) * std::f32::consts::TAU;
            let angle = squad_phase + npc_angle;
            // Per-NPC radius jitter in [-2, +2] m so members at
            // similar angles sit at slightly different distances.
            let r_jitter = ((id_hash >> 16) & 0xFF) as f32 / 255.0 * 4.0 - 2.0;
            let r = (10.0 + squad_radius_jitter + r_jitter).max(6.0);
            [angle.cos() * r, 0.0, angle.sin() * r]
        }
        // Rest spreads wider than Regroup since several squads
        // routinely converge on the same outpost (only same-
        // faction base in many regions) — formation rings need to
        // interleave over a larger disk than the 4–24 m band the
        // shared branch produces.
        Some(SquadObjective::Rest { .. }) => {
            let angle = squad_phase + (slot as f32) * (std::f32::consts::TAU / 8.0);
            let id_hash = (npc_id as u32).wrapping_mul(2_246_822_519);
            let r_jitter = ((id_hash >> 8) & 0xFF) as f32 / 255.0 * 4.0 - 2.0;
            let r = (10.0 + squad_radius_jitter + r_jitter).max(6.0);
            [angle.cos() * r, 0.0, angle.sin() * r]
        }
        Some(SquadObjective::Regroup { .. }) => {
            let angle = squad_phase + (slot as f32) * (std::f32::consts::TAU / 8.0);
            let r = 4.0 + squad_radius_jitter;
            [angle.cos() * r, 0.0, angle.sin() * r]
        }
        Some(SquadObjective::Patrol { .. })
        | Some(SquadObjective::Explore { .. })
        | Some(SquadObjective::Investigate { .. })
        | Some(SquadObjective::Relieve { .. }) => {
            let dx = target[0] - centroid[0];
            let dz = target[2] - centroid[2];
            let len_sq = dx * dx + dz * dz;
            if len_sq < 1.0 {
                // Nearly at target already — just spread circularly.
                let angle = squad_phase + (slot as f32) * (std::f32::consts::TAU / 8.0);
                return [
                    angle.cos() * (3.0 + squad_radius_jitter),
                    0.0,
                    angle.sin() * (3.0 + squad_radius_jitter),
                ];
            }
            let len = len_sq.sqrt();
            // Right-hand perpendicular to travel direction.
            let px = -dz / len;
            let pz = dx / len;
            // Slot 0 at center, then alternating left/right outward.
            // Squad phase nudges the line laterally so adjacent
            // squads don't share the exact same column.
            let lateral_base = match slot {
                0 => 0.0,
                1 => 2.5,
                2 => -2.5,
                3 => 5.0,
                4 => -5.0,
                5 => 7.5,
                6 => -7.5,
                _ => 0.0,
            };
            let lateral = lateral_base + squad_phase.sin() * 2.0;
            // Also stagger along travel axis so the line reads as
            // a V rather than a hard rank.
            let forward_stagger = match slot {
                0 => 0.0,
                1 | 2 => -1.5,
                3 | 4 => -3.0,
                5 | 6 => -4.5,
                _ => 0.0,
            } + squad_phase.cos() * 1.5;
            let fx = dx / len;
            let fz = dz / len;
            [
                px * lateral + fx * forward_stagger,
                0.0,
                pz * lateral + fz * forward_stagger,
            ]
        }
        _ => {
            // Wander / unknown — small stable jitter.
            let angle = squad_phase + (slot as f32) * (std::f32::consts::TAU / 8.0);
            let r = 3.0 + squad_radius_jitter;
            [angle.cos() * r, 0.0, angle.sin() * r]
        }
    }
}

fn advance_patrol(objectives: &mut SquadObjectives, group_id: u64) {
    if let Some(state) = objectives.by_group.get_mut(&group_id) {
        if let SquadObjective::Patrol {
            route, current_idx, ..
        } = &mut state.objective
        {
            if !route.is_empty() {
                *current_idx = (*current_idx + 1) % route.len();
            }
        }
    }
}

fn pick_solo_target(
    rng: &mut ChaCha8Rng,
    npc_faction: crate::faction::registry::FactionId,
    npc_region: crate::region::RegionId,
    here: [f32; 3],
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
) -> [f32; 3] {
    let bcfg = BehaviorConfig::load();
    if rng.gen_bool(bcfg.movement.long_march_prob) {
        let r = bcfg.movement.long_march_radius_m;
        return [
            here[0] + rng.gen_range(-r..r),
            0.0,
            here[2] + rng.gen_range(-r..r),
        ];
    }
    let mut same_faction: Vec<[f32; 3]> = Vec::new();
    let mut any_base: Vec<[f32; 3]> = Vec::new();
    for (_, f, r, p) in bases.iter() {
        if r.0 != npc_region {
            continue;
        }
        any_base.push(p.0);
        if f.0 == npc_faction {
            same_faction.push(p.0);
        }
    }
    let pool = if !same_faction.is_empty() {
        same_faction
    } else if !any_base.is_empty() {
        any_base
    } else {
        return [
            here[0] + rng.gen_range(-200.0..200.0),
            0.0,
            here[2] + rng.gen_range(-200.0..200.0),
        ];
    };
    pool[rng.gen_range(0..pool.len())]
}
