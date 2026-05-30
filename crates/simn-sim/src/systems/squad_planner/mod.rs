//! Squad-level objective planner.
//!
//! For each `Group` of NPCs:
//! 1. If the squad has no objective, an expired one, or a satisfied
//!    one, roll a new objective from a per-faction archetype table.
//! 2. If the squad's spread (max member distance from centroid)
//!    exceeds `COHESION_BREAK_M`, override to `Regroup` for ~5s so
//!    stragglers catch up.
//!
//! Aggro preempts: if any squadmate has `Aggro`, skip the planner
//! for that group — combat behavior in `tick_npc_goals` runs
//! instead. Once everyone deaggros, the planner picks a fresh
//! objective on the next pass.
//!
//! Stale group ids (no living members) are pruned each pass.
//!
//! This is **placeholder** for the parked tactical AI's real
//! GOAP+squad+brain integration. Per-faction weights are tuned by
//! eye, not playtested; brain stance will eventually drive them.

use bevy_ecs::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, VecDeque};

use crate::components::{
    Aggro, Base, BaseKind, Group, InFaction, InRegion, Npc, NpcCharacter, PersonalityTraits,
    Position,
};
use crate::squad_blackboard::{BlackboardKey, SquadBlackboards};

use crate::helpers::quantize_post_pos;
use crate::region::{RegionGraph, RegionId};
use crate::resources::{
    BehaviorLog, GuardPostInfo, GuardPosts, NpcPositionIndex, RegionControl, SimClock,
    SquadObjective, SquadObjectiveState, SquadObjectives,
};

use crate::behavior_config::BehaviorConfig;

fn pcfg() -> crate::behavior_config::PlanningConfig {
    BehaviorConfig::load().planning
}

const POST_MIN_AGE_FOR_RELIEF_TICKS: u64 = 1200;
/// Distance within which a Relieve squadmate is considered to have
/// arrived at the post.
const RELIEVE_ARRIVAL_RADIUS_M: f32 = 25.0;
const RELIEVE_ARRIVAL_RADIUS_SQ_M: f32 = RELIEVE_ARRIVAL_RADIUS_M * RELIEVE_ARRIVAL_RADIUS_M;
/// Squad centroid within this distance of the active objective's
/// target counts as "arrived". Used by stuck-detection to skip the
/// force-expire path: a squad sitting at its target hasn't moved
/// but isn't stuck — they're doing the objective. Picked loose
/// (~25 m) because formation_offset rings can put the centroid
/// 15–25 m off the nominal target for multi-NPC squads.
const ARRIVED_AT_TARGET_M: f32 = 25.0;
const ARRIVED_AT_TARGET_SQ_M: f32 = ARRIVED_AT_TARGET_M * ARRIVED_AT_TARGET_M;

/// Wander drift target picked this far from the squad centroid.
/// Bumped to 150–500 m (2026-05-25) because shorter legs left
/// every squad orbiting their spawn base — even with the drift
/// rotating, the centroid never escaped the base's gravitational
/// well. Longer legs force squads to actually traverse the
/// region between drift points so the map populates instead of
/// piling at one outpost.
#[allow(dead_code)]
const WANDER_DRIFT_MIN_M: f32 = 150.0;
#[allow(dead_code)]
const WANDER_DRIFT_MAX_M: f32 = 500.0;
/// Squad centroid within this distance of the wander_drift_target
/// counts as "arrived" — pick a new leg.
const WANDER_DRIFT_ARRIVE_M: f32 = 20.0;
const WANDER_DRIFT_ARRIVE_SQ_M: f32 = WANDER_DRIFT_ARRIVE_M * WANDER_DRIFT_ARRIVE_M;
// drift_reroll loaded from behavior.toml.

/// Per-squad cohesion-threshold multiplier from the squad's mean
/// `leadership` stat (averaged across members carrying `NpcCharacter`).
/// Linear in `[0.7, 1.3]` over the mean leadership `0..=100`, 1.0× at
/// 50. A squad of unled grunts has a tighter ~56 m leash; a squad
/// with a high-leadership veteran in tow can stretch out to ~104 m
/// before the planner drops them into `Regroup`. Squads with no
/// `NpcCharacter` carriers (legacy / bare-spawn paths) collapse to
/// the flat baseline 1.0.
pub fn cohesion_multiplier_for_leadership(mean_leadership: u8) -> f32 {
    COHESION_MULT_BIAS + COHESION_MULT_SLOPE * f32::from(mean_leadership)
}

const COHESION_MULT_BIAS: f32 = 0.7;
const COHESION_MULT_SLOPE: f32 = 0.006;

/// Per-trait fraction across a squad's members. `disciplined = 0.6`
/// means 60% of the squad has the disciplined trait set. Used by
/// [`pick_objective`] to bias the objective utility scores
/// continuously rather than via a hard "majority" threshold.
/// Empty squads (no `NpcCharacter` carriers) collapse to all zeros,
/// which the scoring path treats as no personality bias —
/// equivalent to the legacy weighted-random behavior.
#[derive(Clone, Copy, Debug, Default)]
pub struct SquadPersonality {
    pub aggressive: f32,
    pub cautious: f32,
    pub curious: f32,
    pub greedy: f32,
    pub loyal: f32,
    pub bloodthirsty: f32,
    pub social: f32,
    pub solitary: f32,
    pub disciplined: f32,
    pub reckless: f32,
}

impl SquadPersonality {
    fn accumulate(&mut self, traits: &PersonalityTraits) {
        if traits.aggressive {
            self.aggressive += 1.0;
        }
        if traits.cautious {
            self.cautious += 1.0;
        }
        if traits.curious {
            self.curious += 1.0;
        }
        if traits.greedy {
            self.greedy += 1.0;
        }
        if traits.loyal {
            self.loyal += 1.0;
        }
        if traits.bloodthirsty {
            self.bloodthirsty += 1.0;
        }
        if traits.social {
            self.social += 1.0;
        }
        if traits.solitary {
            self.solitary += 1.0;
        }
        if traits.disciplined {
            self.disciplined += 1.0;
        }
        if traits.reckless {
            self.reckless += 1.0;
        }
    }

    fn normalize(&mut self, member_count: u32) {
        if member_count == 0 {
            return;
        }
        let n = member_count as f32;
        self.aggressive /= n;
        self.cautious /= n;
        self.curious /= n;
        self.greedy /= n;
        self.loyal /= n;
        self.bloodthirsty /= n;
        self.social /= n;
        self.solitary /= n;
        self.disciplined /= n;
        self.reckless /= n;
    }
}
const PATROL_ROUTE_LEN: usize = 3;
const RECENT_VISITED_CAP: usize = 5;
/// Round position to an integer-meter cell for the recently-visited
/// dedupe key (so jitter near the same base counts as the same place).
const RECENT_QUANTIZE_M: f32 = 50.0;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn squad_planner(
    clock: Res<SimClock>,
    graph: Res<RegionGraph>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    deltas: Res<crate::faction::registry::RelationDeltas>,
    bases: Query<(&Base, &InFaction, &InRegion, &Position)>,
    npcs: Query<(
        &Npc,
        &InFaction,
        &InRegion,
        &Position,
        &Group,
        Option<&Aggro>,
        Option<&NpcCharacter>,
    )>,
    index: Res<NpcPositionIndex>,
    control: Res<RegionControl>,
    blackboards: Res<SquadBlackboards>,
    active_regions: Res<crate::resources::ActiveRegions>,
    mut objectives: ResMut<SquadObjectives>,
    mut posts: ResMut<GuardPosts>,
    mut log: ResMut<BehaviorLog>,
    mut interaction_areas: ResMut<crate::resources::InteractionAreas>,
    mut activity_points: ResMut<crate::resources::ActivityPoints>,
    mut world_events: ResMut<crate::world_event_bus::WorldEventQueue>,
) {
    let _diag_t = crate::systems::SysTimer::new("squad_planner");
    let _prof_guard = crate::systems::ProfGuard(
        std::time::Instant::now(),
        crate::systems::prof_slots::SQUAD_PLANNER,
    );
    let now = clock.tick;
    let bp = pcfg();
    let planner_interval = bp.planner_interval_ticks;
    let guard_tenure = bp.guard_tenure_ticks;
    let investigate_dwell = bp.investigate_arrival_dwell_ticks;
    let regroup_duration = bp.cohesion.regroup_duration_ticks;
    let failed_regroup_disable = bp.cohesion.failed_regroup_disable_ticks;
    let disperse_min = bp.dispersion.min_dist_m;
    let disperse_max = bp.dispersion.max_dist_m;
    let disperse_arrive_sq = bp.dispersion.arrive_radius_m * bp.dispersion.arrive_radius_m;
    let stuck_progress_sq = bp.stuck_detection.progress_m * bp.stuck_detection.progress_m;
    let stuck_timeout = bp.stuck_detection.timeout_ticks;
    let drift_reroll = bp.wander_drift_reroll_ticks;

    // Per-tick: aggregate squad mean leadership AND personality for
    // the cohesion check + objective reroll. With the per-group
    // staggering below (each group rerolls on its own slot within
    // the planner_interval cycle), some group is "due" every
    // tick — so personality has to be available every tick too.
    // The total work over a full 200-tick cycle is the same as the
    // old burst-on-planner-tick approach; we just spread it.
    //
    // Skips NPCs in offline regions — those squads freeze in place
    // until a player re-enters.
    let mut leadership_sum: HashMap<u64, (u32, u32)> = HashMap::new();
    let mut personality_sum: HashMap<u64, (SquadPersonality, u32)> = HashMap::new();
    for (_, _, region, _, g, _, character) in npcs.iter() {
        if !active_regions.is_active(region.0) {
            continue;
        }
        if let Some(c) = character {
            let entry = leadership_sum.entry(g.id).or_insert((0, 0));
            entry.0 += u32::from(c.stats.leadership);
            entry.1 += 1;
            let pentry = personality_sum
                .entry(g.id)
                .or_insert((SquadPersonality::default(), 0));
            pentry.0.accumulate(&c.personality);
            pentry.1 += 1;
        }
    }
    let mean_leadership: HashMap<u64, u8> = leadership_sum
        .into_iter()
        .map(|(g, (sum, n))| (g, (sum / n.max(1)) as u8))
        .collect();
    let squad_personality: HashMap<u64, SquadPersonality> = personality_sum
        .into_iter()
        .map(|(g, (mut p, n))| {
            p.normalize(n);
            (g, p)
        })
        .collect();

    // Cohesion always runs every tick (cheap).
    cohesion_pass(now, &index, &mean_leadership, &mut objectives);
    // Regroup resolution: once a squad has actually gathered at its
    // rally_pos, exit Regroup early — otherwise squads sit at rally
    // for the full 30 s waiting for `expires_at` even though
    // everyone arrived in 5 s. Reads the same NpcPositionIndex
    // entries we already walked in `cohesion_pass`.
    regroup_early_exit_pass(now, &index, &mut objectives);

    // The per-group reroll slot: each group's stable phase within
    // the planner_interval cycle. Spreads ~1/200 of groups
    // per tick instead of all-on-the-same-tick — eliminates the
    // periodic 30-second spike we used to see every 10 seconds
    // when squad_planner re-rolled every group simultaneously.
    let current_slot = now % planner_interval;

    // Group NPCs by group_id to know member count and a representative
    // faction/region per group. Skip offline-region NPCs so the
    // planner only considers groups currently visible to a player.
    let mut groups: HashMap<u64, GroupSummary> = HashMap::new();
    for (_, fac, reg, _, g, aggro, _) in npcs.iter() {
        if !active_regions.is_active(reg.0) {
            continue;
        }
        let centroid = index
            .group_centroids
            .get(&g.id)
            .map(|c| c.pos)
            .unwrap_or([0.0, 0.0, 0.0]);
        let summary = groups.entry(g.id).or_insert(GroupSummary {
            faction: fac.0,
            region: reg.0,
            member_count: 0,
            any_aggroed: false,
            centroid,
        });
        summary.member_count += 1;
        if aggro.is_some() {
            summary.any_aggroed = true;
        }
    }

    // Prune stale objectives for groups that no longer exist or are dead.
    // Release their AP claims first so the points become available.
    let dead_groups: Vec<u64> = objectives
        .by_group
        .keys()
        .filter(|gid| !groups.get(gid).map(|s| s.member_count > 0).unwrap_or(false))
        .copied()
        .collect();
    for gid in &dead_groups {
        if objectives.by_group.contains_key(gid) {
            let region = groups.get(gid).map(|s| s.region).unwrap_or(0);
            activity_points.release_group(region, *gid);
        }
    }
    objectives
        .by_group
        .retain(|gid, _| !dead_groups.contains(gid));
    // Prune posts whose holder died. Another squad with `Relieve`
    // en route will get its objective re-rolled because the post
    // disappeared.
    posts.by_key.retain(|_, info| {
        groups
            .get(&info.group_id)
            .map(|s| s.member_count > 0)
            .unwrap_or(false)
    });

    // Handle relief arrivals: a `Relieve` squad within radius of
    // its target post swaps with the holder. Done before the
    // needs-new check so we don't roll a fresh objective for the
    // arriving squad on the same tick.
    handle_relief_arrivals(now, &groups, &mut objectives, &mut posts, &log);

    let mut rng = ChaCha8Rng::seed_from_u64(now.wrapping_mul(0x517C_C1B7_2722_0A95));

    // Per-tick anchor reservation table. Seeds from squads' CURRENT
    // non-expiring objectives so the planner avoids handing out an
    // already-occupied anchor to a different squad. Populated as
    // each pick lands so subsequent same-tick picks see it. Same
    // pattern as `GuardPosts` but covers Rest (which uses base
    // positions, not guard-posts) — without this, several squads
    // rolling Rest on the same tick all picked the same nearest
    // same-faction base and piled. Quantized at 10 m granularity
    // to absorb sub-cell jitter when reading a centroid back.
    let mut taken_rest_anchors: std::collections::HashSet<(RegionId, [i32; 3])> =
        std::collections::HashSet::new();
    // Mirror table for Relieve objectives. Each squad currently
    // walking to relieve a post claims its `post_key`; subsequent
    // same-tick picks see the claim and don't dogpile the same
    // destination. Without this, multiple squads converge on one
    // post, only one gets to swap into Guard, and the rest sit at
    // the dest_pos with stale Relieve objectives bumping into the
    // newly-installed guards.
    let mut relief_targeted: std::collections::HashSet<(RegionId, [i32; 3])> =
        std::collections::HashSet::new();
    for (gid, state) in &objectives.by_group {
        if let SquadObjective::Rest { base_pos, .. } = &state.objective {
            let _ = gid;
            taken_rest_anchors.insert((
                // Region looked up via the group's summary if present.
                groups.get(gid).map(|s| s.region).unwrap_or(0),
                quantize_anchor(*base_pos),
            ));
        }
        if let SquadObjective::Relieve { post_key, .. } = &state.objective {
            relief_targeted.insert(*post_key);
        }
    }

    // Determinism: iterate groups in stable order so the tick-seeded
    // RNG produces the same objective rolls across same-seed sims.
    // HashMap iteration is NOT stable across instances.
    let mut groups_sorted: Vec<(&u64, &GroupSummary)> = groups.iter().collect();
    groups_sorted.sort_by_key(|(gid, _)| **gid);
    // First-spawn seed pass — runs EVERY tick (not gated on the
    // per-group slot), so a freshly-spawned squad gets a
    // disperse_target on the same tick it appears. Without this,
    // a squad whose `group_id % 200` is far from `current_slot`
    // would sit idle at its spawn point for up to 10 s waiting
    // for its first planner-slot visit. That looked like "NPCs
    // stuck near bases doing nothing" — by the time the slot
    // came up they'd already drifted onto the user's screen.
    for (group_id, summary) in groups_sorted.iter() {
        if objectives.by_group.contains_key(group_id) {
            continue;
        }
        if summary.any_aggroed {
            continue;
        }
        use rand::Rng;
        // Mix `now` into the seed so a squad that gets re-seeded
        // after an offline→online round-trip (their old objective
        // entry was pruned in `retain` when they despawned) picks
        // a fresh disperse angle instead of the same direction
        // they were originally seeded with. Without this, repeated
        // tier transitions would oscillate squads along the same
        // axis from the same anchor.
        let mut seed_rng = ChaCha8Rng::seed_from_u64(
            group_id
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(now.wrapping_mul(0xBF58_476D_1CE4_E5B9))
                .wrapping_add(0xD1B5_4A32_D192_ED03),
        );
        let angle = seed_rng.gen::<f32>() * std::f32::consts::TAU;
        let dist = disperse_min + seed_rng.gen::<f32>() * (disperse_max - disperse_min);
        let target = [
            summary.centroid[0] + angle.cos() * dist,
            summary.centroid[1],
            summary.centroid[2] + angle.sin() * dist,
        ];
        // Assign a patrol zone if the squad doesn't have one yet.
        // Pick the least-populated zone to spread squads evenly.
        // Short-lived Wander for first-spawn dispersion. Expires
        // after 1000 ticks (~50s real) — long enough for the
        // 80-150m dispersion walk at 3 m/s, short enough that
        // squads get a real objective within a minute of spawn.
        objectives.by_group.insert(
            **group_id,
            SquadObjectiveState {
                objective: SquadObjective::Wander {
                    expires_at: now + 1000,
                },
                set_at_tick: now,
                recently_visited: VecDeque::new(),
                disperse_target: Some(target),
                last_progress_pos: Some(summary.centroid),
                last_progress_tick: now,
                wander_drift_target: None,
                last_stuck_kind: None,
                cohesion_break_disabled_until: 0,
                arrived_at_tick: None,
                last_regroup_exit_tick: None,
                last_drift_heading: None,
            },
        );
    }
    // Wander drift refresh — runs every tick for every Wander
    // squad (no slot gate). The drift target needs to roll over
    // the moment the centroid arrives or the leg goes stale;
    // gating this on slots meant a squad with an expired drift
    // target could sit at the old point for up to 10 s before
    // refreshing, which read as "stuck in Wander near the base".
    // Non-Wander squads get their stale drift target cleared
    // here too so an old leg doesn't persist into a fresh
    // objective.
    for (group_id, summary) in groups_sorted.iter() {
        let Some(state) = objectives.by_group.get_mut(group_id) else {
            continue;
        };
        if !matches!(&state.objective, SquadObjective::Wander { .. }) {
            state.wander_drift_target = None;
            state.last_drift_heading = None;
            continue;
        }
        let needs_new_leg = match state.wander_drift_target {
            None => true,
            Some(t) => {
                let dx = summary.centroid[0] - t[0];
                let dz = summary.centroid[2] - t[2];
                let arrived = dx * dx + dz * dz <= WANDER_DRIFT_ARRIVE_SQ_M;
                let drift_set_tick = state.last_progress_tick;
                let stale = now.saturating_sub(drift_set_tick) >= drift_reroll;
                arrived || stale
            }
        };
        if !needs_new_leg {
            continue;
        }
        use rand::Rng;
        let mut leg_rng = ChaCha8Rng::seed_from_u64(
            group_id
                .wrapping_mul(0xBF58_476D_1CE4_E5B9)
                .wrapping_add(now),
        );
        // Wander drift: pick a random absolute world position within
        // the region extents (±1800m from origin). This sends squads
        // to genuinely different parts of the map instead of orbiting
        // the centroid. Heading continuity still applies for smooth
        // path changes, but the base position is world-random.
        #[allow(dead_code)]
        const REGION_HALF_EXTENT: f32 = 1800.0;
        // Zone-seeded drift: each squad gets a "home zone" derived
        // from its group_id. The drift target is a random point
        // within that zone. Squads spread because their zone seeds
        // are distinct. Zone = 500m × 500m grid cell.
        // Pick a random drift target at least 100m from the centroid.
        // Try up to 5 times to find a distant-enough point; if all
        // fail, use the last candidate regardless (better than stuck).
        let min_dist_sq = 100.0_f32 * 100.0;
        let zone_size = 500.0_f32;
        let zone_seed = *group_id;
        let zx = ((zone_seed.wrapping_mul(0x9E37_79B9) >> 16) as i32 % 8 - 4) as f32;
        let zz = ((zone_seed.wrapping_mul(0x7F4A_7C15) >> 16) as i32 % 8 - 4) as f32;
        let mut tx = 0.0_f32;
        let mut tz = 0.0_f32;
        for _attempt in 0..5 {
            tx = zx * zone_size + leg_rng.gen_range(0.0..zone_size);
            tz = zz * zone_size + leg_rng.gen_range(0.0..zone_size);
            let ddx = tx - summary.centroid[0];
            let ddz = tz - summary.centroid[2];
            if ddx * ddx + ddz * ddz >= min_dist_sq {
                break;
            }
        }
        let heading = (tz - summary.centroid[2]).atan2(tx - summary.centroid[0]);
        state.wander_drift_target = Some([tx, summary.centroid[1], tz]);
        state.last_drift_heading = Some(heading);
        state.last_progress_tick = now;
    }

    for (group_id, summary) in groups_sorted {
        // Per-group temporal staggering. Each group only re-evaluates
        // its objective on its own slot within the
        // planner_interval cycle. Without this, every group
        // whose objective expired hit `pick_objective` on the same
        // tick — cascading into 500+ pathfind recomputes for the
        // affected NPCs and a 30-second hang every 10 seconds of
        // play. The principle: each NPC/group acts as its own
        // player making real-time decisions, not a step in a batch.
        if (group_id % planner_interval) != current_slot {
            continue;
        }
        if summary.any_aggroed {
            continue;
        }
        // Dispersion mid-walk: state exists with
        // `disperse_target = Some(_)` → if the centroid hasn't
        // reached it yet, hold the Wander objective and skip the
        // re-roll. Once arrived, clear `disperse_target` and fall
        // through to the regular `needs_new` path so the next
        // planner pass picks a proper objective from the current
        // (post-disperse) position. The first-time seed lives
        // above the slot gate; only the "are we still walking?"
        // check runs here.
        let disperse_state = objectives
            .by_group
            .get(group_id)
            .and_then(|s| s.disperse_target);
        if let Some(dt) = disperse_state {
            let dx = summary.centroid[0] - dt[0];
            let dz = summary.centroid[2] - dt[2];
            let dist_sq = dx * dx + dz * dz;
            if dist_sq > disperse_arrive_sq {
                // Still walking out from spawn — but check stuck
                // progress so an unreachable disperse_target (e.g.
                // random angle landed in a blocked area) doesn't
                // pin the squad in Wander forever.
                if let Some(state) = objectives.by_group.get_mut(group_id) {
                    let progressed = match state.last_progress_pos {
                        None => true,
                        Some(prev) => {
                            let ddx = summary.centroid[0] - prev[0];
                            let ddz = summary.centroid[2] - prev[2];
                            ddx * ddx + ddz * ddz >= stuck_progress_sq
                        }
                    };
                    if progressed {
                        state.last_progress_pos = Some(summary.centroid);
                        state.last_progress_tick = now;
                        continue;
                    } else if now.saturating_sub(state.last_progress_tick) > stuck_timeout {
                        // Disperse target unreachable. Drop it and
                        // fall through to the normal `needs_new`
                        // path so the planner picks a fresh
                        // objective from where the squad is now.
                        state.disperse_target = None;
                        state.last_progress_pos = Some(summary.centroid);
                        state.last_progress_tick = now;
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                // Arrived. Clear so the rest of the loop treats this
                // squad as a normal needs-new candidate.
                if let Some(state) = objectives.by_group.get_mut(group_id) {
                    state.disperse_target = None;
                }
            }
        }
        // Stuck-detection: for movement-oriented objectives, track
        // centroid progress between planner passes. If the squad
        // hasn't moved `STUCK_PROGRESS_M` in `stuck_timeout`, force
        // a re-roll and push the dead target into recently_visited
        // so we don't immediately re-pick the same destination.
        // Stationary objectives (Rest/Guard/Regroup) skip the
        // check — they're supposed to be still. **Arrived-and-
        // milling** also skips: a squad sitting within `ARRIVED_M`
        // of its objective's target hasn't moved but isn't stuck
        // — they're doing the objective. Without this check,
        // Investigate squads that arrive at their target trip
        // stuck-detection ~30 s after arrival and re-roll prematurely.
        let mut stuck_dead_target: Option<[f32; 3]> = None;
        if let Some(state) = objectives.by_group.get_mut(group_id) {
            let movement_objective = matches!(
                &state.objective,
                SquadObjective::Patrol { .. }
                    | SquadObjective::Investigate { .. }
                    | SquadObjective::Explore { .. }
                    | SquadObjective::Relieve { .. }
                    | SquadObjective::Wander { .. }
            );
            if movement_objective {
                let arrived_at_target =
                    position_of_objective(&state.objective).is_some_and(|tgt| {
                        let dx = summary.centroid[0] - tgt[0];
                        let dz = summary.centroid[2] - tgt[2];
                        dx * dx + dz * dz <= ARRIVED_AT_TARGET_SQ_M
                    });
                let progressed = match state.last_progress_pos {
                    None => true, // First observation — count as progress.
                    Some(prev) => {
                        let dx = summary.centroid[0] - prev[0];
                        let dz = summary.centroid[2] - prev[2];
                        dx * dx + dz * dz >= stuck_progress_sq
                    }
                };
                // Track arrival transitions so the planner can cap
                // arrival-dwell for objectives that would otherwise
                // freeze a squad at the target until the full
                // dwell timer (Investigate especially — 4 min of
                // NPCs standing still reads as broken).
                if arrived_at_target && state.arrived_at_tick.is_none() {
                    state.arrived_at_tick = Some(now);
                } else if !arrived_at_target && state.arrived_at_tick.is_some() {
                    state.arrived_at_tick = None;
                }
                if progressed || arrived_at_target {
                    state.last_progress_pos = Some(summary.centroid);
                    state.last_progress_tick = now;
                } else if now.saturating_sub(state.last_progress_tick) > stuck_timeout {
                    // Force expiry. Capture the current objective's
                    // target so we can ban it from the next roll,
                    // and remember the kind so the immediate re-roll
                    // doesn't re-pick the same broken plan.
                    stuck_dead_target = position_of_objective(&state.objective);
                    state.last_stuck_kind = Some(objective_kind_tag(&state.objective));
                    match &mut state.objective {
                        SquadObjective::Patrol { expires_at, .. }
                        | SquadObjective::Investigate { expires_at, .. }
                        | SquadObjective::Explore { expires_at, .. }
                        | SquadObjective::Relieve { expires_at, .. }
                        | SquadObjective::Wander { expires_at } => *expires_at = now,
                        _ => {}
                    }
                    // Reset progress tracking so the next objective
                    // starts with a clean window.
                    state.last_progress_pos = Some(summary.centroid);
                    state.last_progress_tick = now;
                }
            }
        }
        let needs_new = match objectives.by_group.get(group_id) {
            None => true,
            Some(state) => match &state.objective {
                SquadObjective::Regroup { expires_at, .. } => *expires_at <= now,
                // Posted guard: re-rolls on relief or squad death,
                // OR after `guard_tenure` of holding the
                // post so squads don't sit at one spot forever.
                // Without the tenure cap, posted guards looked
                // permanently stuck even after the rest of the
                // anti-stuck work landed.
                SquadObjective::Guard {
                    post_key: Some(key),
                    ..
                } => {
                    let lost_claim = posts
                        .by_key
                        .get(key)
                        .is_none_or(|info| info.group_id != *group_id);
                    let tenure_expired = posts
                        .by_key
                        .get(key)
                        .is_some_and(|info| now.saturating_sub(info.since_tick) >= guard_tenure);
                    lost_claim || tenure_expired
                }
                SquadObjective::Investigate { expires_at, .. } => {
                    // Cap Investigate's arrival-dwell at
                    // investigate_dwell so squads
                    // don't stand frozen at a target for the full
                    // 4 min after arriving. The walk + look-around
                    // total dwell stays ≤ the full expires_at
                    // either way.
                    let dwell_expired = state
                        .arrived_at_tick
                        .is_some_and(|t| now.saturating_sub(t) >= investigate_dwell);
                    *expires_at <= now || dwell_expired
                }
                _ => state.objective.expires_at() <= now,
            },
        };
        // Pre-stash the stuck target into recently_visited so the
        // re-roll inside `pick_objective` (via recently_visited
        // filtering) avoids re-picking it.
        if let Some(dead) = stuck_dead_target {
            if let Some(state) = objectives.by_group.get_mut(group_id) {
                push_recent(&mut state.recently_visited, dead);
            }
        }
        if !needs_new {
            continue;
        }
        // Detect a Regroup that failed to gather everyone before
        // its natural 30 s expiry. The early-exit pass would have
        // collapsed `expires_at` to a much earlier tick on
        // success — a Regroup that ran for ~regroup_duration
        // hit the natural timeout, which means at least one
        // outlier never reached rally. Disable cohesion-break
        // detection for `failed_regroup_disable` (~5 min) so
        // the squad commits to its next objective instead of
        // immediately re-Regrouping on the same unreachable
        // straggler. Sets the flag on the soon-to-be-replaced
        // state; the value carries through the `remove(group_id)`
        // + insert below because we read it back before mutating.
        let regroup_failed = matches!(
            objectives.by_group.get(group_id).map(|s| &s.objective),
            Some(SquadObjective::Regroup { .. })
        ) && objectives
            .by_group
            .get(group_id)
            .is_some_and(|s| now.saturating_sub(s.set_at_tick) >= regroup_duration);
        let mut state = objectives
            .by_group
            .remove(group_id)
            .unwrap_or(SquadObjectiveState {
                objective: SquadObjective::Wander {
                    expires_at: now + pcfg().objective_default_duration_ticks,
                },
                set_at_tick: now,
                recently_visited: VecDeque::new(),
                disperse_target: None,
                last_progress_pos: None,
                last_progress_tick: 0,
                wander_drift_target: None,
                last_stuck_kind: None,
                cohesion_break_disabled_until: 0,
                arrived_at_tick: None,
                last_regroup_exit_tick: None,
                last_drift_heading: None,
            });
        if regroup_failed {
            state.cohesion_break_disabled_until = now.saturating_add(failed_regroup_disable);
        }
        let personality = squad_personality.get(group_id).copied().unwrap_or_default();
        let bb_signals = read_blackboard_signals(&blackboards, *group_id);
        let banned_kind = state.last_stuck_kind;
        let new_obj = pick_objective(
            &mut rng,
            summary,
            now,
            &state.recently_visited,
            &bases,
            &graph,
            &posts,
            &control,
            &registry,
            &deltas,
            *group_id,
            &personality,
            &bb_signals,
            &mut interaction_areas,
            &mut activity_points,
            &taken_rest_anchors,
            &relief_targeted,
            banned_kind,
        );
        // Track newly-rolled Relieve so subsequent same-tick picks
        // dedupe against it.
        if let SquadObjective::Relieve { post_key, .. } = &new_obj {
            relief_targeted.insert(*post_key);
        }
        // One-shot ban — clear so the next pick is unrestricted.
        state.last_stuck_kind = None;
        // Reserve the rest anchor so subsequent same-tick picks see
        // it. Vacating happens implicitly: the prior-objective swap
        // above just dropped the old state, and next-tick's reseed
        // rebuilds the set from current objectives.
        if let SquadObjective::Rest { base_pos, .. } = &new_obj {
            taken_rest_anchors.insert((summary.region, quantize_anchor(*base_pos)));
        }
        // Push current Patrol/Guard/Investigate destination into
        // recently_visited (if it had one) before swapping.
        if let Some(visited_pos) = position_of_objective(&state.objective) {
            push_recent(&mut state.recently_visited, visited_pos);
        }
        // If this squad was holding a post, vacate it so someone
        // else can claim it.
        if let SquadObjective::Guard {
            post_key: Some(key),
            ..
        } = &state.objective
        {
            if posts
                .by_key
                .get(key)
                .is_some_and(|info| info.group_id == *group_id)
            {
                posts.by_key.remove(key);
            }
        }
        // Iteration 5-13 Phase D3: if this squad was at a
        // designer-placed rest area, free the reservation so
        // another squad can claim the spot. Cohesion-driven
        // Regroup overrides also reach here when the planner
        // tries to swap *back* to a fresh objective; the
        // `state.objective` we're holding is the to-be-replaced
        // one in all paths. Drain Started markers AND emit
        // matching InteractionEnded events in the same pass so
        // PDA toasts have a paired bracket.
        if let SquadObjective::Rest {
            area_id: Some(id), ..
        } = &state.objective
        {
            interaction_areas.release_internal(id);
            let leaving = interaction_areas.drain_started_for_area(id);
            for npc_id_raw in leaving {
                world_events.push(
                    crate::world_event_bus::WorldEventKind::InteractionEnded {
                        npc_id: crate::components::NpcId(npc_id_raw),
                        area_id: id.clone(),
                    },
                    summary.centroid,
                    summary.region,
                    now,
                    1,
                );
            }
        }
        // If the new objective is a posted Guard, register it.
        if let SquadObjective::Guard {
            post_key: Some(key),
            ..
        } = &new_obj
        {
            posts.by_key.insert(
                *key,
                GuardPostInfo {
                    group_id: *group_id,
                    since_tick: now,
                    faction: summary.faction,
                },
            );
        }
        // Detect Regroup → other transition so `cohesion_pass` can
        // apply its post-Regroup cooldown on the next tick. We
        // check the OLD objective (about to be replaced).
        if matches!(state.objective, SquadObjective::Regroup { .. }) {
            state.last_regroup_exit_tick = Some(now);
        }
        // Release activity point claims from the old objective so
        // other squads can use them.
        activity_points.release_group(summary.region, *group_id);
        state.objective = new_obj;
        state.set_at_tick = now;
        if log.enabled {
            *log.objectives
                .entry(objective_kind_str(&state.objective))
                .or_insert(0) += 1;
        }
        objectives.by_group.insert(*group_id, state);
    }
}

/// Scan Relieve objectives and, for any squad whose centroid sits
/// within the relief radius of its target post, execute the swap:
/// evict the old holder (their Guard objective becomes `None` so
/// they re-roll next pass), install the arriving squad as the new
/// posted Guard, reset the `since_tick`.
fn handle_relief_arrivals(
    now: u64,
    groups: &HashMap<u64, GroupSummary>,
    objectives: &mut SquadObjectives,
    posts: &mut GuardPosts,
    log: &BehaviorLog,
) {
    // Collect arrivals first to avoid mutating during iteration.
    // Carries the new holder's faction so the post can be re-tagged
    // on takeover — relief should never quietly hand a faction's
    // post to another faction.
    type ArrivalRow = (
        u64,
        (RegionId, [i32; 3]),
        [f32; 3],
        crate::faction::registry::FactionId,
    );
    let mut arrivals: Vec<ArrivalRow> = Vec::new();
    for (group_id, state) in &objectives.by_group {
        let SquadObjective::Relieve {
            post_key, dest_pos, ..
        } = &state.objective
        else {
            continue;
        };
        let Some(summary) = groups.get(group_id) else {
            continue;
        };
        if summary.any_aggroed {
            continue;
        }
        // Must still be in the region that owns the post.
        if summary.region != post_key.0 {
            continue;
        }
        let dx = summary.centroid[0] - dest_pos[0];
        let dz = summary.centroid[2] - dest_pos[2];
        if dx * dx + dz * dz > RELIEVE_ARRIVAL_RADIUS_SQ_M {
            continue;
        }
        arrivals.push((*group_id, *post_key, *dest_pos, summary.faction));
    }

    for (new_holder, key, dest_pos, new_faction) in arrivals {
        // Evict the current holder, if any.
        let old_holder = posts.by_key.get(&key).map(|info| info.group_id);
        if let Some(old) = old_holder {
            if let Some(old_state) = objectives.by_group.get_mut(&old) {
                // Clear their Guard objective so the next planner
                // pass picks something new for them.
                if matches!(
                    old_state.objective,
                    SquadObjective::Guard {
                        post_key: Some(_),
                        ..
                    }
                ) {
                    old_state.objective = SquadObjective::Wander {
                        // Short expiry so they immediately re-roll
                        // (the main loop will see this as expired
                        // on the next planner pass).
                        expires_at: now,
                    };
                    // Seed a wander drift target ~80 m from the
                    // post on a deterministic per-(group, tick)
                    // angle so the evicted squad immediately walks
                    // away from the relief party instead of milling
                    // at the post for the up-to-10s slot delay
                    // before its next planner pass picks a real
                    // objective.
                    use rand::Rng;
                    let mut leg_rng = ChaCha8Rng::seed_from_u64(
                        old.wrapping_mul(0xBF58_476D_1CE4_E5B9)
                            .wrapping_add(now)
                            .wrapping_add(0x94D0_4955_5CB6_8B27),
                    );
                    let angle = leg_rng.gen::<f32>() * std::f32::consts::TAU;
                    let drift = [
                        dest_pos[0] + angle.cos() * 80.0,
                        dest_pos[1],
                        dest_pos[2] + angle.sin() * 80.0,
                    ];
                    old_state.wander_drift_target = Some(drift);
                }
            }
        }
        // Install the new posted guard.
        posts.by_key.insert(
            key,
            GuardPostInfo {
                group_id: new_holder,
                since_tick: now,
                faction: new_faction,
            },
        );
        if let Some(state) = objectives.by_group.get_mut(&new_holder) {
            state.objective = SquadObjective::Guard {
                base_pos: dest_pos,
                // Posted guards ignore time; set far-future anyway
                // so the expires-check doesn't trip.
                expires_at: u64::MAX,
                post_key: Some(key),
            };
            state.set_at_tick = now;
        }
        if log.enabled {
            tracing::debug!(
                target: "npc.behavior",
                "relief group={:x} took post {:?}",
                new_holder & 0xFFFF,
                key
            );
        }
    }
}

fn objective_kind_str(o: &SquadObjective) -> &'static str {
    match o {
        SquadObjective::Patrol { .. } => "patrol",
        SquadObjective::Guard {
            post_key: Some(_), ..
        } => "guard_post",
        SquadObjective::Guard { .. } => "guard",
        SquadObjective::Rest { .. } => "rest",
        SquadObjective::Investigate { .. } => "investigate",
        SquadObjective::Explore { .. } => "explore",
        SquadObjective::Relieve { .. } => "relieve",
        SquadObjective::Wander { .. } => "wander",
        SquadObjective::Regroup { .. } => "regroup",
    }
}

/// Radius (meters) within which every squad member must sit to
/// consider a Regroup "complete". A member further than this from
/// `rally_pos` is still considered straggling. Picked well under
/// `COHESION_BREAK_M` so a freshly-exited Regroup squad has slack
/// before the cohesion check fires again.
const REGROUP_GATHERED_RADIUS_M: f32 = 20.0;
const REGROUP_GATHERED_RADIUS_SQ_M: f32 = REGROUP_GATHERED_RADIUS_M * REGROUP_GATHERED_RADIUS_M;

/// Walk Regroup squads; if every member sits within
/// `REGROUP_GATHERED_RADIUS_M` of the rally point, force
/// `expires_at` to `now` so the main planner loop's `needs_new`
/// check fires this tick and the squad rolls a fresh objective
/// without waiting the rest of the dwell window. Squads with a
/// straggler keep waiting — the planner falls through to the
/// existing timeout-driven re-roll at 30 s, and the cooldown
/// keeps cohesion from immediately re-triggering.
fn regroup_early_exit_pass(now: u64, index: &NpcPositionIndex, objectives: &mut SquadObjectives) {
    let mut to_expire: Vec<u64> = Vec::new();
    for (group_id, state) in objectives.by_group.iter() {
        let SquadObjective::Regroup { rally_pos, .. } = &state.objective else {
            continue;
        };
        let mut all_gathered = true;
        let mut member_seen = false;
        for entry in index.by_id.values() {
            if entry.group != Some(*group_id) {
                continue;
            }
            member_seen = true;
            let dx = entry.pos[0] - rally_pos[0];
            let dz = entry.pos[2] - rally_pos[2];
            if dx * dx + dz * dz > REGROUP_GATHERED_RADIUS_SQ_M {
                all_gathered = false;
                break;
            }
        }
        if member_seen && all_gathered {
            to_expire.push(*group_id);
        }
    }
    for group_id in to_expire {
        if let Some(state) = objectives.by_group.get_mut(&group_id) {
            if let SquadObjective::Regroup { expires_at, .. } = &mut state.objective {
                *expires_at = now;
            }
        }
    }
}

fn cohesion_pass(
    now: u64,
    index: &NpcPositionIndex,
    mean_leadership: &HashMap<u64, u8>,
    objectives: &mut SquadObjectives,
) {
    // Walk NPCs once, tracking max-sq-from-centroid per group — the
    // old loop was O(groups × npcs) per tick (scanned every NPC for
    // every group). One pass is O(npcs) and the centroid lookup is
    // a HashMap hit.
    let mut worst_per_group: std::collections::HashMap<u64, f32> =
        std::collections::HashMap::with_capacity(index.group_centroids.len());
    for entry in index.by_id.values() {
        let Some(group_id) = entry.group else {
            continue;
        };
        let Some(centroid) = index.group_centroids.get(&group_id) else {
            continue;
        };
        if centroid.member_count < 2 {
            continue;
        }
        let dx = entry.pos[0] - centroid.pos[0];
        let dz = entry.pos[2] - centroid.pos[2];
        let d_sq = dx * dx + dz * dz;
        let slot = worst_per_group.entry(group_id).or_insert(0.0);
        if d_sq > *slot {
            *slot = d_sq;
        }
    }

    for (group_id, worst_sq) in &worst_per_group {
        // Per-squad leadership leash: leader-rich squads stretch
        // farther before regrouping. Squads without `NpcCharacter`
        // carriers fall back to the flat 80 m baseline.
        let break_sq = match mean_leadership.get(group_id) {
            Some(&m) => {
                let mult = cohesion_multiplier_for_leadership(m);
                let break_m = pcfg().cohesion.break_distance_m * mult;
                break_m * break_m
            }
            None => {
                let m = pcfg().cohesion.break_distance_m;
                m * m
            }
        };
        if *worst_sq <= break_sq {
            continue;
        }
        // Already regrouping? Don't reset the timer.
        if let Some(state) = objectives.by_group.get(group_id) {
            if matches!(state.objective, SquadObjective::Regroup { .. }) {
                continue;
            }
            // Cohesion-break suppression after a failed Regroup.
            // Set by the planner main loop when a Regroup hits its
            // natural timeout without all members gathering — gives
            // the squad ~5 min to commit to a movement objective
            // instead of immediately re-Regrouping on the same
            // unreachable straggler.
            if now < state.cohesion_break_disabled_until {
                continue;
            }
            // Post-Regroup cooldown. Without this, a squad just out
            // of Regroup whose follow-up objective causes any
            // movement (which it always does, except parked Guard)
            // re-trips the cohesion-break check within a few ticks
            // — so squads thrash between Regroup and their real
            // goal. The cooldown gives the squad time to actually
            // commit to a goal + traverse some distance before
            // re-evaluating cohesion. Gated on the dedicated
            // `last_regroup_exit_tick` field so a freshly-spawned
            // squad isn't accidentally treated as "post-Regroup".
            let on_cooldown = state
                .last_regroup_exit_tick
                .is_some_and(|t| now.saturating_sub(t) < pcfg().cohesion.regroup_cooldown_ticks);
            if matches!(
                state.objective,
                SquadObjective::Patrol { .. }
                    | SquadObjective::Investigate { .. }
                    | SquadObjective::Explore { .. }
                    | SquadObjective::Relieve { .. }
                    | SquadObjective::Wander { .. }
            ) && on_cooldown
            {
                continue;
            }
        }
        let mut state = objectives
            .by_group
            .remove(group_id)
            .unwrap_or(SquadObjectiveState {
                objective: SquadObjective::Wander {
                    expires_at: now + pcfg().objective_default_duration_ticks,
                },
                set_at_tick: now,
                recently_visited: VecDeque::new(),
                disperse_target: None,
                last_progress_pos: None,
                last_progress_tick: 0,
                wander_drift_target: None,
                last_stuck_kind: None,
                cohesion_break_disabled_until: 0,
                arrived_at_tick: None,
                last_regroup_exit_tick: None,
                last_drift_heading: None,
            });
        // Snapshot the centroid NOW so the rally is fixed for the
        // duration of this Regroup. Without this snapshot the rally
        // would track the live `group_centroids`, which is recomputed
        // each tick from member positions — moving members drag the
        // centroid with them and the squad spirals inward to a single
        // point. Frozen rally → squad converges on a stable spot and
        // stops.
        let rally_pos = index
            .group_centroids
            .get(group_id)
            .map(|c| c.pos)
            .unwrap_or([0.0, 0.0, 0.0]);
        state.objective = SquadObjective::Regroup {
            rally_pos,
            expires_at: now + pcfg().cohesion.regroup_duration_ticks,
        };
        state.set_at_tick = now;
        objectives.by_group.insert(*group_id, state);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct GroupSummary {
    pub(crate) faction: crate::faction::registry::FactionId,
    pub(crate) region: RegionId,
    pub(crate) member_count: u32,
    pub(crate) any_aggroed: bool,
    pub(crate) centroid: [f32; 3],
}

#[allow(clippy::too_many_arguments)]
/// Per-squad blackboard signals consumed by `pick_objective` to
/// bias the utility scoring. Substrate-driven: each field is
/// derived from a [`BlackboardKey`] entry whose presence
/// influences which objectives the squad finds appealing this
/// pass. Empty-blackboard squads (default) get no signal bias —
/// the legacy faction-weight + personality scoring runs alone.
#[derive(Default, Clone, Copy, Debug)]
pub struct BlackboardSignals {
    /// Squad heard a gunshot recently → boost Investigate.
    pub heard_gunshot: bool,
    /// Squad has a last-known enemy position cached → boost
    /// Investigate (chase the lead).
    pub last_known_enemy_pos: bool,
    /// Squad is taking incoming fire → boost Investigate / Guard
    /// (cover-up posture). Aggro-driven combat also handles this,
    /// but on the planner pass we want to reflect that the squad
    /// shouldn't be picking Rest / Explore right now.
    pub under_fire: bool,
}

fn read_blackboard_signals(bb: &SquadBlackboards, group_id: u64) -> BlackboardSignals {
    let Some(group) = bb.get(group_id) else {
        return BlackboardSignals::default();
    };
    BlackboardSignals {
        heard_gunshot: group.get(&BlackboardKey::HeardGunshot).is_some(),
        last_known_enemy_pos: group.get(&BlackboardKey::LastKnownEnemyPos).is_some(),
        under_fire: group.get(&BlackboardKey::UnderFireAt).is_some(),
    }
}

/// Score one objective kind from base weight × personality fractions
/// × blackboard signals. Public for testing — the consumer is
/// `pick_objective`, but unit tests want to spot-check the
/// trait/signal nudges without setting up a full Sim. The base weight is the per-faction
/// archetype value (preserving the legacy faction-flavor table).
/// Personality fractions multiply by `1 + (max_bonus * fraction)`
/// per-trait — a 100% disciplined squad gets the full disciplined
/// boost; a 50% squad gets half. Blackboard signals apply
/// situational boosts (HeardGunshot → Investigate ↑↑).
///
/// Returned utility is in arbitrary units; only the relative
/// ranking across kinds matters. Picked by max in [`pick_objective`]
/// with a deterministic tiebreak by enum order.
pub fn objective_utility(
    kind: ObjKind,
    base_weight: u32,
    p: &SquadPersonality,
    bb: &BlackboardSignals,
    has_territorial_standing: bool,
) -> f32 {
    let base = base_weight as f32;
    if base <= 0.0 {
        return 0.0;
    }
    // Personality multipliers amplified ~3x vs. the original tuning so
    // trait differences are visible in playtest. Negative-damping
    // factors can now drive their term below zero (e.g. a fully
    // aggressive squad's Rest factor); the final `mult.max(0.0)` floor
    // catches that, meaning strongly off-trait objectives collapse to
    // zero utility and are effectively never picked. That is the
    // intent — a 100% aggressive squad should not rest.
    let mult: f32 = match kind {
        ObjKind::Patrol => {
            (1.0 + 0.90 * p.disciplined) * (1.0 - 0.60 * p.solitary) * (1.0 + 0.30 * p.loyal)
        }
        ObjKind::Guard => {
            // Guard is reserved for territorial standing; outside our
            // own region the kind falls through to Investigate (see
            // pick_objective). Still let the trait math run so the
            // utility ordering reflects what the squad WOULD pick if
            // they had standing.
            let standing_mult = if has_territorial_standing { 1.0 } else { 0.4 };
            standing_mult
                * (1.0 + 1.20 * p.disciplined)
                * (1.0 + 0.90 * p.loyal)
                * (1.0 - 1.20 * p.curious)
        }
        ObjKind::Investigate => {
            let mut m =
                (1.0 + 1.50 * p.curious) * (1.0 + 0.90 * p.aggressive) * (1.0 - 0.60 * p.cautious);
            if bb.heard_gunshot {
                m *= 1.50;
            }
            if bb.last_known_enemy_pos {
                m *= 1.30;
            }
            if bb.under_fire {
                m *= 1.20;
            }
            m
        }
        ObjKind::Rest => {
            let mut m =
                (1.0 + 0.90 * p.cautious) * (1.0 - 1.20 * p.aggressive) * (1.0 + 0.60 * p.social);
            // Don't rest under fire / when chasing a lead.
            if bb.under_fire || bb.heard_gunshot {
                m *= 0.30;
            }
            m
        }
        ObjKind::Explore => {
            let mut m = (1.0 + 1.20 * p.curious) * (1.0 + 0.60 * p.solitary);
            if bb.under_fire {
                m *= 0.50;
            }
            m
        }
        ObjKind::Wander => {
            let mut m =
                (1.0 + 0.90 * p.solitary) * (1.0 + 0.60 * p.curious) * (1.0 - 0.60 * p.disciplined);
            if bb.under_fire {
                m *= 0.50;
            }
            m
        }
    };
    base * mult.max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn pick_objective(
    rng: &mut ChaCha8Rng,
    summary: &GroupSummary,
    now: u64,
    recent: &VecDeque<[i32; 3]>,
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
    graph: &RegionGraph,
    posts: &GuardPosts,
    control: &RegionControl,
    registry: &crate::faction::registry::FactionRegistry,
    deltas: &crate::faction::registry::RelationDeltas,
    group_id: u64,
    personality: &SquadPersonality,
    bb_signals: &BlackboardSignals,
    interaction_areas: &mut crate::resources::InteractionAreas,
    activity_points: &mut crate::resources::ActivityPoints,
    taken_rest_anchors: &std::collections::HashSet<(RegionId, [i32; 3])>,
    relief_targeted: &std::collections::HashSet<(RegionId, [i32; 3])>,
    banned_kind: Option<crate::resources::SquadObjectiveKindTag>,
) -> SquadObjective {
    // A faction has standing to guard/relieve in a region only if
    // it's the primary controller or an active contester. Outside
    // of those, Guard rolls fall through to something more
    // faction-appropriate (Investigate / Patrol / Wander) instead
    // of setting up shop in someone else's backyard.
    let summary_faction_name = registry.name_of(summary.faction);
    let has_territorial_standing = control
        .by_region
        .get(&summary.region)
        .map(|state| {
            state.primary.as_deref() == Some(summary_faction_name)
                || state.contested_by.iter().any(|f| f == summary_faction_name)
        })
        .unwrap_or(false);
    // Utility scoring: per-objective base weight (faction archetype)
    // × personality multipliers × blackboard signal multipliers ×
    // a small per-squad noise term. Without the noise, two squads
    // of the same faction with similar personality + blackboard
    // state always pick the same kind — visually "all federal
    // squads patrol simultaneously". The noise is stable per
    // (group_id, set_at_tick) so a freshly-rolled objective doesn't
    // jitter, but distinct groups get distinct utility orderings.
    let w = weights_for(registry, summary.faction);
    let mut kinds: [(ObjKind, u32); 6] = [
        (ObjKind::Patrol, w.0),
        (ObjKind::Guard, w.1),
        (ObjKind::Investigate, w.2),
        (ObjKind::Rest, w.3),
        (ObjKind::Explore, w.4),
        (ObjKind::Wander, w.5),
    ];
    // Personality floor: a squad whose membership is heavily slanted
    // toward a trait should have at least a minimum base weight on the
    // matching objectives, even if the faction archetype zeros them
    // out. Lets a majority-curious sub-squad of a Guard-heavy faction
    // still go investigate; a social squad gets Rest options; a greedy
    // squad gets Wander options for the open-world roaming that loot
    // routes are built on.
    const PERSONALITY_FLOOR_THRESHOLD: f32 = 0.6;
    const PERSONALITY_FLOOR_WEIGHT: u32 = 2;
    if personality.curious > PERSONALITY_FLOOR_THRESHOLD && kinds[2].1 < PERSONALITY_FLOOR_WEIGHT {
        kinds[2].1 = PERSONALITY_FLOOR_WEIGHT;
    }
    if personality.social > PERSONALITY_FLOOR_THRESHOLD && kinds[3].1 < PERSONALITY_FLOOR_WEIGHT {
        kinds[3].1 = PERSONALITY_FLOOR_WEIGHT;
    }
    if personality.curious > PERSONALITY_FLOOR_THRESHOLD && kinds[4].1 < PERSONALITY_FLOOR_WEIGHT {
        kinds[4].1 = PERSONALITY_FLOOR_WEIGHT;
    }
    if personality.solitary > PERSONALITY_FLOOR_THRESHOLD && kinds[5].1 < PERSONALITY_FLOOR_WEIGHT {
        kinds[5].1 = PERSONALITY_FLOOR_WEIGHT;
    }
    if personality.greedy > PERSONALITY_FLOOR_THRESHOLD && kinds[5].1 < PERSONALITY_FLOOR_WEIGHT {
        kinds[5].1 = PERSONALITY_FLOOR_WEIGHT;
    }
    // Per-(group, tick, kind) noise in roughly [0, 1.0). 1.0 of
    // utility is enough to flip "tied" picks while not overpowering
    // strong-signal cases (e.g. blackboard `under_fire` multipliers
    // in the 5-10x range still dominate).
    let mut best = (ObjKind::Wander, f32::NEG_INFINITY);
    for (k, base) in kinds {
        // Skip the kind that just got the squad stuck so the
        // re-roll picks something different. The ban is one-shot
        // — caller clears `last_stuck_kind` after the pick.
        if let Some(banned) = banned_kind {
            if objkind_matches_tag(k, banned) {
                continue;
            }
        }
        let u = objective_utility(k, base, personality, bb_signals, has_territorial_standing);
        let noise = squad_objective_noise(group_id, now, k);
        if u + noise > best.1 {
            best = (k, u + noise);
        }
    }
    let kind = best.0;
    // `rng` is still consumed downstream by `build_*` (patrol route
    // shuffling, investigate target offset, etc.). The kind pick
    // itself is now deterministic from `(group_id, tick, weights,
    // personality, bb_signals)` — no shared-RNG draw.

    match kind {
        ObjKind::Patrol => try_patrol_from_activity_points(activity_points, summary, group_id, now)
            .or_else(|| build_patrol(rng, summary, now, recent, bases))
            .unwrap_or_else(|| wander(now)),
        ObjKind::Guard => {
            if !has_territorial_standing {
                build_investigate(rng, summary, now, bases, registry, deltas)
            } else {
                // Try authored activity points first, then legacy base-position guards
                try_guard_from_activity_points(activity_points, summary, group_id, now)
                    .or_else(|| build_guard(rng, summary, now, bases, posts, group_id, recent))
                    .or_else(|| build_relieve(rng, summary, now, posts, group_id, relief_targeted))
                    .unwrap_or_else(|| wander(now))
            }
        }
        ObjKind::Investigate => build_investigate(rng, summary, now, bases, registry, deltas),
        ObjKind::Rest => try_rest_from_activity_points(activity_points, summary, group_id, now)
            .or_else(|| {
                build_rest(
                    rng,
                    summary,
                    now,
                    bases,
                    interaction_areas,
                    taken_rest_anchors,
                    recent,
                )
            })
            .unwrap_or_else(|| wander(now)),
        ObjKind::Explore => build_explore(rng, summary, now, graph).unwrap_or_else(|| wander(now)),
        ObjKind::Wander => wander(now),
    }
}

/// Public-for-testing analog of [`SquadObjective`]'s tag set —
/// `pick_objective` chooses one of these via utility scoring before
/// the matching `build_*` helper instantiates the variant. The full
/// `SquadObjective` enum keeps its payload (positions, post keys,
/// expiries); `ObjKind` is just the discriminant for scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjKind {
    Patrol,
    Guard,
    Investigate,
    Rest,
    Explore,
    Wander,
}

/// Default per-faction objective weights, looked up by registry
/// name. (Patrol, Guard, Investigate, Rest, Explore, Wander).
/// Unknown / mod-defined factions fall back to a balanced 1/1/1/1/1/1
/// split. Subfactions (Linemen, Choir, Cartel, …) inherit their
/// parent's tuning via the registry's parent walk unless they have
/// their own override row here. Future commit moves these onto
/// `FactionDef` so they're configurable in `factions.toml`.
fn weights_for_known(name: &str) -> Option<(u32, u32, u32, u32, u32, u32)> {
    // Tuned for clearer faction identity. Tuples are
    // (Patrol, Guard, Investigate, Rest, Explore, Wander).
    match name {
        "pwa" => Some((4, 6, 2, 3, 2, 1)),         // territorial defenders
        "linemen" => Some((2, 8, 2, 3, 1, 0)),     // entrenched grid operators
        "federal" => Some((2, 3, 5, 3, 2, 1)),     // investigate-heavy law enforcement
        "ghost_teams" => Some((1, 1, 7, 2, 4, 0)), // direct-action investigators
        "aegis_pacific" => Some((1, 2, 5, 3, 2, 1)),
        "recovery_division" => Some((1, 1, 7, 2, 4, 0)), // hunting witnesses
        "revere_guard" => Some((2, 1, 3, 2, 6, 1)),      // infiltrators roam hard
        "bandits" => Some((2, 1, 3, 3, 3, 3)),           // opportunistic drifters
        "cartel" => Some((2, 2, 3, 3, 5, 1)),            // contractor-style mobility
        "attuned" => Some((2, 4, 2, 3, 2, 1)),
        "choir" => Some((1, 7, 2, 3, 1, 1)), // hold sacred ground
        "gulf_compact" => Some((2, 2, 2, 2, 5, 2)), // contractors rove
        "registry" => Some((2, 3, 4, 2, 3, 0)), // enforcement sweeps
        "wanderers" => Some((0, 0, 1, 3, 6, 5)), // drifters live to explore
        "merged" => Some((0, 9, 0, 0, 0, 0)),
        _ => None,
    }
}

fn weights_for(
    reg: &crate::faction::registry::FactionRegistry,
    id: crate::faction::registry::FactionId,
) -> (u32, u32, u32, u32, u32, u32) {
    reg.resolve_with_parent_walk(id, weights_for_known)
        .unwrap_or((1, 1, 1, 1, 1, 1))
}

pub(super) mod builders;
pub(crate) use builders::*;
