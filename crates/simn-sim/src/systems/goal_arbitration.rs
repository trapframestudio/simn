//! Per-tick goal resolver. Collects candidate goals from every
//! source (aggro, squad objective, blackboard urgency, scripted
//! claim, …), picks max by priority + recency tiebreak, and writes
//! the winner into [`ActiveGoal`] for the executor to consume.
//!
//! Today's sources: `IndividualAggro`, `SquadAggro`, `SquadObjective`,
//! `Idle`. Stages 2+ add `BlackboardUrgency`, `IndividualSurvival`,
//! `PersonalityBias`, and the flank-bonus rule that lets
//! `IndividualAggro` outrank `SquadAggro` when a different attacker
//! engages a member from the side/rear.
//!
//! ## Priority table
//!
//! Numbers are illustrative; real values land via playtest. Stable
//! ordering matters more than the exact gap between rows — the
//! resolver picks the candidate with the highest `priority`, with
//! the older `created_tick` winning a tie so re-derivation doesn't
//! preempt itself.
//!
//! | Source                            | Priority |
//! |-----------------------------------|----------|
//! | `ScriptedClaim`                   | 240      |
//! | `IndividualSurvival`              | 220      |
//! | `BlackboardUrgency::DownedAlly`   | 180      |
//! | `SquadAggro`                      | 160      |
//! | `IndividualAggro`                 | 150      |
//! | `BlackboardUrgency::UnderFireAt`  | 140      |
//! | `SquadObjective`                  | 80       |
//! | `PersonalityBias`                 | 60       |
//! | `BlackboardUrgency::HeardGunshot` | 40       |
//! | `Idle`                            | 0        |
//!
//! ### "Don't pull a working squad off-task" rule
//!
//! Distant gunshots are flavorful, not commanding. Earlier `HeardGunshot`
//! sat at 100 — just barely beating the 20-point hysteresis vs an 80
//! `SquadObjective` — so any audible shot pulled a regrouping squad
//! off-task. Now it's 40, below `SquadObjective` baseline, so a squad
//! mid-task (Regroup / Patrol / Guard / Investigate) finishes what
//! it's doing. Idle / personality-biased NPCs still react. Combat
//! distractions (`UnderFireAt`, `DownedAlly`, visible aggro) remain
//! above squad-objective priority — those *are* important.
//! ## Hysteresis
//!
//! Without dampening, frequently-flipping inputs (a target on the
//! edge of perception, a blackboard that re-fires every few ticks)
//! cause goal-thrash. The resolver requires a new candidate to beat
//! the current `ActiveGoal.priority` by at least
//! [`HYSTERESIS_PRIO_DELTA`] before preempting. Same-source
//! re-derivations update `expires_at` instead of replacing.
//!
//! ## Determinism
//!
//! Iteration order over the NPC query is unstable (Bevy archetype
//! storage order isn't guaranteed across sim instances). Arbitration
//! itself is purely-functional per-NPC and consumes no shared RNG, so
//! the order doesn't matter today. If a future source consumes RNG
//! at this stage, sort by `NpcId` first — the determinism harness
//! will catch the regression.

use bevy_ecs::prelude::*;

use crate::components::{
    ActiveGoal, Aggro, GoalKind, GoalSource, Group, InRegion, Npc, NpcCharacter, NpcId,
    PersonalityTraits,
};
use crate::resources::{
    ActiveRegions, NpcPositionIndex, SimClock, SquadObjective, SquadObjectives,
};
use crate::squad_blackboard::{BlackboardKey, BlackboardValue, SquadBlackboards};

#[allow(dead_code)]
const PRIO_SCRIPTED_CLAIM: u8 = 240;
const PRIO_INDIVIDUAL_SURVIVAL: u8 = 220;
const PRIO_BLACKBOARD_DOWNED_ALLY: u8 = 180;
const PRIO_SQUAD_AGGRO: u8 = 160;
const PRIO_INDIVIDUAL_AGGRO: u8 = 150;
/// Suppressing fire on the squad. Slotted just under
/// `IndividualAggro` so a visible target still wins; this fires
/// when the squad is being shot at but has no aggro target yet
/// (suppression from cover, ambush opener).
const PRIO_BLACKBOARD_UNDER_FIRE: u8 = 140;
/// Same-faction intel: a squad-mate or another same-faction squad
/// has aggro on an enemy nearby. Below `UnderFireAt` (the squad
/// itself is taking damage = more urgent) but above
/// `SquadObjective` so a Resting/Wandering squad will respond to
/// nearby combat even without personal LOS. Hysteresis still
/// applies — see `apply_with_hysteresis`.
const PRIO_BLACKBOARD_LAST_KNOWN_ENEMY: u8 = 130;
// SOLO_BORROW_RADIUS_M and RESPONDER_CAP_PER_TARGET loaded from behavior.toml.
const PRIO_SQUAD_OBJECTIVE: u8 = 80;
/// Distant gunshots — flavorful, not commanding. Sits below
/// `SquadObjective` so a working squad doesn't break formation for
/// curiosity. Above `Idle` and `PersonalityBias` so idle NPCs still
/// react. Combat distractions remain via `UnderFireAt` (NPC took a
/// hit) and aggro (NPC sees a hostile).
const PRIO_BLACKBOARD_GUNSHOT: u8 = 40;
#[allow(dead_code)]
const PRIO_PERSONALITY_BIAS: u8 = 60;
const PRIO_IDLE: u8 = 0;

/// Minimum priority delta a new candidate must beat the current
/// `ActiveGoal` by to preempt it. Same-source re-derivations bypass
/// this — they refresh `expires_at` rather than replacing.
use crate::behavior_config::BehaviorConfig;

const COMMITMENT_BYPASS_PRIO: u8 = 140;

/// Per-objective multiplier applied to a squad-following NPC's
/// `SquadObjective` candidate priority based on the NPC's personality
/// traits. The output is in roughly `[0.3, 2.5]` — wide enough that
/// trait differences re-rank decisively within the SquadObjective
/// tier. Combat and survival sources are explicitly NOT modulated by
/// personality; the result is clamped by `biased_priority` to never
/// preempt aggro priorities (149 ceiling).
///
/// Trait nudges per objective (~3x amplified vs the original tuning):
///
/// - `Patrol`: disciplined +60%, curious +30%, solitary -40%.
/// - `Guard`: disciplined +90%, loyal +60%, curious -50%.
/// - `Rest`: cautious +60%, aggressive -50%.
/// - `Investigate`: curious +150%, cautious -25%, aggressive -50%.
/// - `Explore` / `Wander`: curious +120%, solitary +60%, disciplined -25%.
/// - `Relieve` / `Regroup`: loyal +60%, solitary -50%.
pub fn personality_bias_for_objective(
    traits: &PersonalityTraits,
    objective: &SquadObjective,
) -> f32 {
    let mut m: f32 = 1.0;
    match objective {
        SquadObjective::Patrol { .. } => {
            if traits.disciplined {
                m *= 1.60;
            }
            if traits.curious {
                m *= 1.30;
            }
            if traits.solitary {
                m *= 0.60;
            }
        }
        SquadObjective::Guard { .. } => {
            if traits.disciplined {
                m *= 1.90;
            }
            if traits.loyal {
                m *= 1.60;
            }
            if traits.curious {
                m *= 0.50;
            }
        }
        SquadObjective::Rest { .. } => {
            if traits.cautious {
                m *= 1.60;
            }
            if traits.aggressive {
                m *= 0.50;
            }
        }
        SquadObjective::Investigate { .. } => {
            if traits.curious {
                m *= 2.50;
            }
            if traits.cautious {
                m *= 0.75;
            }
            if traits.aggressive {
                m *= 0.50;
            }
        }
        SquadObjective::Explore { .. } | SquadObjective::Wander { .. } => {
            if traits.curious {
                m *= 2.20;
            }
            if traits.solitary {
                m *= 1.60;
            }
            if traits.disciplined {
                m *= 0.75;
            }
        }
        SquadObjective::Relieve { .. } | SquadObjective::Regroup { .. } => {
            if traits.loyal {
                m *= 1.60;
            }
            if traits.solitary {
                m *= 0.50;
            }
        }
    }
    m
}

/// Apply the personality-bias multiplier to a base priority and
/// clamp back into `u8` space. Held to ≤ `PRIO_INDIVIDUAL_AGGRO - 1`
/// (149) so a personality-boosted SquadObjective can never preempt
/// an aggro pursuit; combat lanes stay reserved. `pub` so the
/// personality-traits test suite can verify the clamp directly.
pub fn biased_priority(base: u8, multiplier: f32) -> u8 {
    let scaled = (f32::from(base) * multiplier).round();
    scaled.clamp(0.0, f32::from(PRIO_INDIVIDUAL_AGGRO - 1)) as u8
}

/// One nominee. The resolver collects these per-NPC, picks the max,
/// applies hysteresis vs the existing `ActiveGoal`, and writes the
/// winner. `created_tick` is the tick the candidate's source first
/// fired — used as the recency tiebreak.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    source: GoalSource,
    kind: GoalKind,
    priority: u8,
    created_tick: u64,
    expires_at: Option<u64>,
}

/// Run per tick after `npc_aggro` and `squad_planner`. Reads source
/// state, writes [`ActiveGoal`] in place. Inserts a default
/// `ActiveGoal` if missing (defensive — `npc_spawn` already includes
/// it, but a load-from-snapshot of older worlds may not).
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn goal_arbitration(
    clock: Res<SimClock>,
    objectives: Res<SquadObjectives>,
    index: Res<NpcPositionIndex>,
    blackboards: Res<SquadBlackboards>,
    active_regions: Res<ActiveRegions>,
    activity_points: Res<crate::resources::ActivityPoints>,
    corpse_index: Res<crate::resources::CorpseIndex>,
    faction_registry: Res<crate::faction::registry::FactionRegistry>,
    faction_deltas: Res<crate::faction::registry::RelationDeltas>,
    // Index of (group_id → faction) so solo-NPC propagation below
    // can match by faction without a second pass over the
    // population. Built from a side query so the main per-NPC
    // mut-iter doesn't hold an extra borrow.
    group_factions_q: Query<(&Group, &crate::components::InFaction), With<Npc>>,
    npc_factions_q: Query<(&crate::components::NpcId, &crate::components::InFaction), With<Npc>>,
    mut npcs: Query<(
        Entity,
        &Npc,
        &crate::components::InFaction,
        &InRegion,
        Option<&Aggro>,
        Option<&Group>,
        Option<&NpcCharacter>,
        Option<&crate::components::BodyParts>,
        Option<&mut ActiveGoal>,
    )>,
    mut commands: Commands,
) {
    let _diag_t = crate::systems::SysTimer::new("goal_arbitration");
    let _prof_guard = crate::systems::ProfGuard(
        std::time::Instant::now(),
        crate::systems::prof_slots::GOAL_ARBITRATION,
    );
    let now = clock.tick;
    let arb = BehaviorConfig::load().arbitration;
    // Per-group urgency cache. The blackboard is per-group, but
    // arbitration runs per-NPC; without caching, a 5-NPC squad reads
    // the same blackboard 5 times. Build the urgency-candidate list
    // lazily on first access per group and reuse for subsequent
    // squadmates.
    let mut per_group_urgency: std::collections::HashMap<u64, Vec<Candidate>> =
        std::collections::HashMap::new();
    // Side index: group_id → faction. Solo NPCs use this to find
    // same-faction squads worth borrowing intel from. Built from
    // a single linear scan of grouped NPCs (faction is squad-level
    // by construction, so the first one wins).
    let mut group_faction: std::collections::HashMap<u64, crate::faction::registry::FactionId> =
        std::collections::HashMap::new();
    for (g, f) in group_factions_q.iter() {
        group_faction.entry(g.id).or_insert(f.0);
    }
    // Side index: npc_id → faction. Solo NPCs index into this for
    // their own faction without holding an extra borrow on the
    // mut npcs query.
    let mut npc_faction: std::collections::HashMap<
        crate::components::NpcId,
        crate::faction::registry::FactionId,
    > = std::collections::HashMap::new();
    for (id, f) in npc_factions_q.iter() {
        npc_faction.insert(*id, f.0);
    }
    // Responder cap pre-pass. Scan the existing-tick `ActiveGoal`s
    // off the npcs query (read-only borrow ends before the mut
    // pass below) and count how many distinct responders per
    // `(faction, quantized_target_50m)` are already on it. Used to
    // gate the `LastKnownEnemy` candidate so a single enemy
    // doesn't drag every same-faction squad within 200 m off-task.
    // A responder = group_id for grouped NPCs, or `npc_id.0` (with
    // the high bit set) for solo NPCs.
    type RespKey = (crate::faction::registry::FactionId, [i32; 2]);
    let mut responders: std::collections::HashMap<RespKey, std::collections::HashSet<u64>> =
        std::collections::HashMap::new();
    {
        // Local read-only borrow on `npcs` to seed the table. Bevy
        // allows the borrow because we're not holding it past the
        // block — the mut-iter below picks up cleanly.
        let mut read_q = npcs.transmute_lens::<(
            &Npc,
            &crate::components::InFaction,
            Option<&Group>,
            Option<&ActiveGoal>,
        )>();
        for (n, faction, g, ag) in read_q.query().iter() {
            let Some(ag) = ag else { continue };
            let target_pos = match ag.kind {
                GoalKind::PursueTarget { target } => index.by_id.get(&target).map(|e| e.pos),
                GoalKind::InvestigateAt { pos } | GoalKind::RegroupOnAlly { pos, .. } => Some(pos),
                _ => None,
            };
            let Some(tp) = target_pos else { continue };
            let cell = [(tp[0] / 50.0).floor() as i32, (tp[2] / 50.0).floor() as i32];
            let responder_id = g.map(|g| g.id).unwrap_or(n.id.0 | (1u64 << 63));
            responders
                .entry((faction.0, cell))
                .or_default()
                .insert(responder_id);
        }
    }
    for (entity, npc, faction, region, aggro, group, character, body_parts, existing) in
        npcs.iter_mut()
    {
        // Active-region filter. Offline-region NPCs keep their
        // previous `ActiveGoal` (or default Idle) — arbitration is
        // skipped until a player re-enters the region.
        if !active_regions.is_active(region.0) {
            continue;
        }
        let mut candidates: Vec<Candidate> = Vec::with_capacity(4);

        // IndividualSurvival: NPCs with critically-wounded vitals
        // (head or torso < `SEEK_MEDICAL_VITAL_THRESHOLD`) break off
        // and head for the nearest same-faction RestSpot / Campfire
        // activity point. Sits at priority 220 — preempts everything
        // except ScriptedClaim and DownedAlly. If no rest spot is
        // reachable, no candidate is pushed and the NPC keeps doing
        // whatever it was doing (fight or flight from elsewhere).
        const SEEK_MEDICAL_VITAL_THRESHOLD: f32 = 25.0;
        const SEEK_MEDICAL_SEARCH_RADIUS_M: f32 = 200.0;
        const SEEK_MEDICAL_SEARCH_RADIUS_SQ_M: f32 =
            SEEK_MEDICAL_SEARCH_RADIUS_M * SEEK_MEDICAL_SEARCH_RADIUS_M;
        if let Some(bp) = body_parts {
            if bp.vital_min() < SEEK_MEDICAL_VITAL_THRESHOLD {
                if let Some(self_pos) = index.by_id.get(&npc.id).map(|e| e.pos) {
                    let mut best: Option<(f32, [f32; 3])> = None;
                    for pt in activity_points.points_in_region(region.0) {
                        if !pt.kind.is_rest() {
                            continue;
                        }
                        if let Some(f) = pt.faction {
                            if f != faction.0 {
                                continue;
                            }
                        }
                        let dx = pt.pos[0] - self_pos[0];
                        let dz = pt.pos[2] - self_pos[2];
                        let d_sq = dx * dx + dz * dz;
                        if d_sq > SEEK_MEDICAL_SEARCH_RADIUS_SQ_M {
                            continue;
                        }
                        if best.map(|b| d_sq < b.0).unwrap_or(true) {
                            best = Some((d_sq, pt.pos));
                        }
                    }
                    if let Some((_d_sq, target_pos)) = best {
                        candidates.push(Candidate {
                            source: GoalSource::IndividualSurvival,
                            kind: GoalKind::SeekMedical { target_pos },
                            priority: PRIO_INDIVIDUAL_SURVIVAL,
                            created_tick: now,
                            expires_at: None,
                        });
                    }
                }
            }
        }

        // Aggro candidates. If the target is alive in the same region,
        // emit IndividualAggro; if the NPC is also in a Group, emit
        // SquadAggro on top — squad coordination is the default per
        // user direction (see `goal-arbitration-plan.md` §6).
        if let Some(ag) = aggro {
            let target_live = index
                .by_id
                .get(&ag.target)
                .copied()
                .map(|e| e.region == region.0 && e.health > 0.0)
                .unwrap_or(false);
            if target_live {
                let kind = GoalKind::PursueTarget { target: ag.target };
                candidates.push(Candidate {
                    source: GoalSource::IndividualAggro,
                    kind,
                    priority: PRIO_INDIVIDUAL_AGGRO,
                    created_tick: ag.last_seen_tick,
                    expires_at: None,
                });
                if let Some(g) = group {
                    // Focus fire: if the squad's ThreatList has a top
                    // entry, all members target that instead of their
                    // individual Aggro target. Concentrates fire.
                    let focus_target = blackboards
                        .get(g.id)
                        .and_then(|bb| bb.get(&BlackboardKey::ThreatList))
                        .and_then(|entry| {
                            if let crate::squad_blackboard::BlackboardValue::Threats(threats) =
                                &entry.value
                            {
                                threats.first().map(|t| t.target_id)
                            } else {
                                None
                            }
                        });
                    let focus_kind = if let Some(tid) = focus_target {
                        let live = index
                            .by_id
                            .get(&tid)
                            .copied()
                            .map(|e| e.region == region.0 && e.health > 0.0)
                            .unwrap_or(false);
                        if live {
                            GoalKind::PursueTarget { target: tid }
                        } else {
                            kind
                        }
                    } else {
                        kind
                    };
                    candidates.push(Candidate {
                        source: GoalSource::SquadAggro,
                        kind: focus_kind,
                        priority: PRIO_SQUAD_AGGRO,
                        created_tick: ag.last_seen_tick,
                        expires_at: None,
                    });
                }
            }
        }

        // Blackboard-urgency candidates. Reads the squad's shared
        // store for `DownedAlly` / `UnderFireAt` / `HeardGunshot`
        // entries (written by `world_event_bus` drain and the
        // `npc_combat` damage path) and nominates per-kind
        // candidates with priorities slotted around combat. Only
        // grouped NPCs participate — ungrouped lone NPCs (Wanderers,
        // some Looters) don't share a blackboard and react only via
        // their individual aggro path. Cached per-group above.
        // Helper: cap-aware filter. For `LastKnownEnemy` candidates,
        // count how many distinct responders (groups or solos) are
        // already on the same `(faction, target_cell_50m)`. If at
        // or above `RESPONDER_CAP_PER_TARGET` AND this responder
        // isn't already in the set, drop the candidate so the
        // squad sticks with its current objective. Other candidate
        // kinds (DownedAlly, UnderFireAt, HeardGunshot, …) pass
        // through unchanged.
        let self_faction = Some(faction.0);
        let self_responder_id = group.map(|g| g.id).unwrap_or(npc.id.0 | (1u64 << 63));
        let cap_ok = |c: &Candidate| -> bool {
            if c.priority != PRIO_BLACKBOARD_LAST_KNOWN_ENEMY {
                return true;
            }
            let GoalKind::InvestigateAt { pos } = c.kind else {
                return true;
            };
            let Some(f) = self_faction else {
                return true;
            };
            let cell = [
                (pos[0] / 50.0).floor() as i32,
                (pos[2] / 50.0).floor() as i32,
            ];
            let set = responders.get(&(f, cell));
            let count = set.map(|s| s.len()).unwrap_or(0);
            // Already on it → always allowed (we'd just refresh
            // our own goal).
            if set.is_some_and(|s| s.contains(&self_responder_id)) {
                return true;
            }
            count < arb.responder_cap_per_target as usize
        };

        if let Some(g) = group {
            let cached = per_group_urgency.entry(g.id).or_insert_with(|| {
                let mut v = Vec::with_capacity(3);
                push_blackboard_candidates(&blackboards, g.id, &mut v);
                v
            });
            candidates.extend(cached.iter().filter(|c| cap_ok(c)).copied());
        } else {
            // Solo NPC: borrow intel from same-faction squad
            // blackboards within `SOLO_BORROW_RADIUS_M`. Without
            // this, ungrouped NPCs miss the faction-aggro
            // propagation entirely and stand idle while a
            // squad-mate fights 30 m away. Limited to the highest-
            // priority candidate across all reachable groups —
            // adding all of them would double-count the same
            // intel.
            if let Some(self_entry) = index.by_id.get(&npc.id).copied() {
                if let Some(&self_faction) = npc_faction.get(&npc.id) {
                    let r_sq = arb.solo_borrow_radius_m * arb.solo_borrow_radius_m;
                    let mut best: Option<Candidate> = None;
                    for (gid, centroid) in index.group_centroids.iter() {
                        if centroid.region != region.0 {
                            continue;
                        }
                        let dx = centroid.pos[0] - self_entry.pos[0];
                        let dz = centroid.pos[2] - self_entry.pos[2];
                        if dx * dx + dz * dz > r_sq {
                            continue;
                        }
                        if group_faction.get(gid).copied() != Some(self_faction) {
                            continue;
                        }
                        let cached = per_group_urgency.entry(*gid).or_insert_with(|| {
                            let mut v = Vec::with_capacity(3);
                            push_blackboard_candidates(&blackboards, *gid, &mut v);
                            v
                        });
                        for c in cached.iter() {
                            if !cap_ok(c) {
                                continue;
                            }
                            if best.is_none_or(|b| c.priority > b.priority) {
                                best = Some(*c);
                            }
                        }
                    }
                    if let Some(c) = best {
                        candidates.push(c);
                    }
                }
            }
        }

        // Squad objective candidate. Personality biases the
        // priority within the squad-objective lane only — combat
        // and survival lanes are not modulated.
        if let Some(g) = group {
            if let Some(state) = objectives.by_group.get(&g.id) {
                let priority = match character {
                    Some(c) => {
                        let mult = personality_bias_for_objective(&c.personality, &state.objective);
                        biased_priority(PRIO_SQUAD_OBJECTIVE, mult)
                    }
                    None => PRIO_SQUAD_OBJECTIVE,
                };
                candidates.push(Candidate {
                    source: GoalSource::SquadObjective,
                    kind: GoalKind::SquadFollowObjective,
                    priority,
                    created_tick: state.set_at_tick,
                    expires_at: None,
                });
            }
        }

        // Personality-introduced candidates. Each trait's introduced
        // drive is resolved into a concrete `GoalKind` here using
        // contextual lookups (group centroid for Socialize, corpse
        // position for Loot, activity point catalog for Hunt). Drives
        // for which we can't find a valid target are skipped — there's
        // no "wander aimlessly" fallback at this tier.
        //
        // Socialize is the only personality drive resolved in this
        // phase. It's gated to:
        //   - Squad objective is Rest, AND
        //   - Group centroid is within `SOCIALIZE_ARRIVE_RADIUS_M` of
        //     the Rest target (squad has actually arrived).
        // Nominated at `PRIO_SOCIALIZE` (85) which is just above the
        // baseline SquadObjective so social NPCs visibly break their
        // formation slot to converge into a tight ring — but well
        // below combat priorities (140+) so a firefight still wins.
        const SOCIALIZE_ARRIVE_RADIUS_M: f32 = 22.0;
        const PRIO_SOCIALIZE: u8 = PRIO_SQUAD_OBJECTIVE + 5;
        if let Some(c) = character {
            for drive in c.personality.introduces_drives() {
                use crate::components::PersonalityDrive::*;
                match drive {
                    Socialize => {
                        let Some(g) = group else { continue };
                        let Some(state) = objectives.by_group.get(&g.id) else {
                            continue;
                        };
                        let SquadObjective::Rest { base_pos, .. } = state.objective else {
                            continue;
                        };
                        let Some(centroid) = index.group_centroids.get(&g.id) else {
                            continue;
                        };
                        let dx = centroid.pos[0] - base_pos[0];
                        let dz = centroid.pos[2] - base_pos[2];
                        if dx * dx + dz * dz > SOCIALIZE_ARRIVE_RADIUS_M * SOCIALIZE_ARRIVE_RADIUS_M
                        {
                            continue;
                        }
                        candidates.push(Candidate {
                            source: GoalSource::PersonalityBias,
                            kind: GoalKind::Socialize {
                                target_pos: base_pos,
                            },
                            priority: PRIO_SOCIALIZE,
                            created_tick: now,
                            expires_at: None,
                        });
                    }
                    Hunt => {
                        // Curious NPCs head for nearby Stash / Lookout /
                        // Workbench activity points that aren't owned by
                        // their faction (or are unfactioned). Gated to
                        // Rest / Wander squad objectives (or solo NPCs)
                        // so a working squad doesn't wander off mid-task.
                        let allowed_now = match group {
                            Some(g) => objectives
                                .by_group
                                .get(&g.id)
                                .map(|s| {
                                    matches!(
                                        s.objective,
                                        SquadObjective::Rest { .. } | SquadObjective::Wander { .. }
                                    )
                                })
                                .unwrap_or(true),
                            None => true,
                        };
                        if !allowed_now {
                            continue;
                        }
                        let Some(self_pos) = index.by_id.get(&npc.id).map(|e| e.pos) else {
                            continue;
                        };
                        const HUNT_SIGHT_RADIUS_M: f32 = 80.0;
                        const HUNT_SIGHT_RADIUS_SQ_M: f32 =
                            HUNT_SIGHT_RADIUS_M * HUNT_SIGHT_RADIUS_M;
                        let points = activity_points.points_in_region(region.0);
                        let mut best: Option<(f32, [f32; 3], u64)> = None;
                        for pt in points {
                            if !matches!(
                                pt.kind,
                                crate::resources::ActivityKind::Stash
                                    | crate::resources::ActivityKind::Lookout
                                    | crate::resources::ActivityKind::Workbench
                            ) {
                                continue;
                            }
                            if let Some(f) = pt.faction {
                                if f == faction.0 {
                                    continue;
                                }
                            }
                            let dx = pt.pos[0] - self_pos[0];
                            let dz = pt.pos[2] - self_pos[2];
                            let d_sq = dx * dx + dz * dz;
                            if d_sq > HUNT_SIGHT_RADIUS_SQ_M {
                                continue;
                            }
                            if best.map(|b| d_sq < b.0).unwrap_or(true) {
                                best = Some((d_sq, pt.pos, pt.id));
                            }
                        }
                        if let Some((_d_sq, target_pos, ap_id)) = best {
                            // Solo NPCs nominate at the bias tier (60);
                            // grouped NPCs at Rest/Wander get a 5-point
                            // bump (65) so they actually break formation.
                            // Both stay well below combat priorities.
                            let prio = if group.is_some() {
                                PRIO_PERSONALITY_BIAS + 5
                            } else {
                                PRIO_PERSONALITY_BIAS
                            };
                            candidates.push(Candidate {
                                source: GoalSource::PersonalityBias,
                                kind: GoalKind::Hunt {
                                    target_pos,
                                    activity_point_id: Some(ap_id),
                                },
                                priority: prio,
                                created_tick: now,
                                expires_at: None,
                            });
                        }
                    }
                    Loot => {
                        // Greedy NPCs head for nearby corpse containers
                        // when off-duty (no group, or squad at Rest /
                        // Wander). Same-faction corpses are skipped —
                        // looting a squadmate is a faction taboo, not
                        // a default-greedy behavior. The arbiter picks
                        // the nearest qualifying corpse within sight
                        // radius; the executor walks there and dwells.
                        let allowed_now = match group {
                            Some(g) => objectives
                                .by_group
                                .get(&g.id)
                                .map(|s| {
                                    matches!(
                                        s.objective,
                                        SquadObjective::Rest { .. } | SquadObjective::Wander { .. }
                                    )
                                })
                                .unwrap_or(true),
                            None => true,
                        };
                        if !allowed_now {
                            continue;
                        }
                        let Some(self_pos) = index.by_id.get(&npc.id).map(|e| e.pos) else {
                            continue;
                        };
                        const LOOT_SIGHT_RADIUS_M: f32 = 80.0;
                        const LOOT_SIGHT_RADIUS_SQ_M: f32 =
                            LOOT_SIGHT_RADIUS_M * LOOT_SIGHT_RADIUS_M;
                        let mut best: Option<(f32, [f32; 3], crate::components::ContainerId)> =
                            None;
                        for (cid, entry) in corpse_index.by_container.iter() {
                            if entry.region != region.0 {
                                continue;
                            }
                            // Skip same-faction corpses; that's taboo
                            // not greed. Use the registry's relation
                            // walk so any "not-hostile" pair (e.g.
                            // sub-factions of the same parent) also
                            // skips — it'd be weird for a Linemen
                            // squad to loot a Linemen corpse.
                            if entry.faction == faction.0 {
                                continue;
                            }
                            let rel = crate::faction::registry::faction_relation(
                                &faction_registry,
                                &faction_deltas,
                                faction.0,
                                entry.faction,
                            );
                            if matches!(
                                rel,
                                crate::faction::Relation::Warm | crate::faction::Relation::Friendly
                            ) {
                                continue;
                            }
                            let dx = entry.pos[0] - self_pos[0];
                            let dz = entry.pos[2] - self_pos[2];
                            let d_sq = dx * dx + dz * dz;
                            if d_sq > LOOT_SIGHT_RADIUS_SQ_M {
                                continue;
                            }
                            if best.map(|b| d_sq < b.0).unwrap_or(true) {
                                best = Some((d_sq, entry.pos, *cid));
                            }
                        }
                        if let Some((_d_sq, target_pos, container_id)) = best {
                            let prio = if group.is_some() {
                                PRIO_PERSONALITY_BIAS + 10
                            } else {
                                PRIO_PERSONALITY_BIAS + 5
                            };
                            candidates.push(Candidate {
                                source: GoalSource::PersonalityBias,
                                kind: GoalKind::Loot {
                                    target_pos,
                                    target_container: Some(container_id.0 as u64),
                                },
                                priority: prio,
                                created_tick: now,
                                expires_at: None,
                            });
                        }
                    }
                    // Bloodsport deferred until arena concept lands.
                    Bloodsport => {}
                }
            }
        }

        // Idle fallback.
        candidates.push(Candidate {
            source: GoalSource::Idle,
            kind: GoalKind::SoloIdleFsm,
            priority: PRIO_IDLE,
            created_tick: now,
            expires_at: None,
        });

        let winner = pick_winner(&candidates).expect("Idle candidate is always present");

        // If we just committed to a LastKnownEnemy or PursueTarget
        // goal, bump the responders set so subsequent same-tick
        // arbitration (for other NPCs of the same faction) sees us
        // and the cap holds within the tick — otherwise three
        // squads' first members all pass the cap simultaneously
        // and we'd end up over the limit.
        if let Some(f) = self_faction {
            let target_pos = match winner.kind {
                GoalKind::PursueTarget { target } => index.by_id.get(&target).map(|e| e.pos),
                GoalKind::InvestigateAt { pos } | GoalKind::RegroupOnAlly { pos, .. } => Some(pos),
                _ => None,
            };
            if let Some(tp) = target_pos {
                let cell = [(tp[0] / 50.0).floor() as i32, (tp[2] / 50.0).floor() as i32];
                responders
                    .entry((f, cell))
                    .or_default()
                    .insert(self_responder_id);
            }
        }

        match existing {
            Some(mut current) => {
                // If the current goal's source no longer produces a
                // matching candidate, the goal is "orphaned" — e.g.
                // PursueTarget after Aggro decays (target dead, no
                // candidate generated). Hysteresis would block the
                // transition forever because nothing can beat the
                // stale priority. Replace unconditionally.
                let current_still_sourced = candidates
                    .iter()
                    .any(|c| c.source == current.source && c.kind == current.kind);
                if current_still_sourced {
                    apply_with_hysteresis(&mut current, winner, now);
                } else {
                    let commit = if winner.source == GoalSource::SquadObjective {
                        now + arb.commitment_ticks
                    } else {
                        0
                    };
                    *current = ActiveGoal {
                        source: winner.source,
                        kind: winner.kind,
                        priority: winner.priority,
                        created_tick: now,
                        expires_at: winner.expires_at,
                        pursue_progress: None,
                        committed_until_tick: commit,
                    };
                }
            }
            None => {
                let commit = if winner.source == GoalSource::SquadObjective {
                    now + arb.commitment_ticks
                } else {
                    0
                };
                commands.entity(entity).insert(ActiveGoal {
                    source: winner.source,
                    kind: winner.kind,
                    priority: winner.priority,
                    created_tick: now,
                    expires_at: winner.expires_at,
                    pursue_progress: None,
                    committed_until_tick: commit,
                });
            }
        }
    }
}

/// Walk the squad's blackboard and push `BlackboardUrgency`
/// candidates for the urgency keys this system recognizes. Only
/// `Position`-valued entries produce candidates; entries with
/// missing or mistyped values are silently skipped (defensive
/// against future writers that mis-spell the contract).
///
/// Determinism: at most one `DownedAlly` candidate is pushed per
/// tick — the one with the smallest `NpcId` if several entries
/// coexist (priorities tie at 180 and `pick_winner`'s
/// fallback-to-last-element on ties depends on iteration order,
/// which `HashMap::iter` doesn't guarantee). `UnderFireAt` and
/// `HeardGunshot` use single keys so only one entry can exist.
fn push_blackboard_candidates(
    blackboards: &SquadBlackboards,
    group_id: u64,
    candidates: &mut Vec<Candidate>,
) {
    let Some(bb) = blackboards.get(group_id) else {
        return;
    };

    // DownedAlly: scan all entries with that key, pick the
    // lowest-id one as the representative. The squad reacts to one
    // body at a time; the rest fade out via their own TTL.
    let mut best_downed: Option<(NpcId, [f32; 3], u64, Option<u64>)> = None;
    for (key, entry) in bb.iter() {
        if let BlackboardKey::DownedAlly { id } = key {
            if let BlackboardValue::Position(pos) = entry.value {
                let expires_at = entry.written_tick.checked_add(entry.ttl_ticks as u64);
                let candidate_tuple = (*id, pos, entry.written_tick, expires_at);
                best_downed = Some(match best_downed {
                    Some(current) if current.0 .0 <= id.0 => current,
                    _ => candidate_tuple,
                });
            }
        }
    }
    if let Some((id, pos, created_tick, expires_at)) = best_downed {
        candidates.push(Candidate {
            source: GoalSource::BlackboardUrgency,
            kind: GoalKind::RegroupOnAlly { id, pos },
            priority: PRIO_BLACKBOARD_DOWNED_ALLY,
            created_tick,
            expires_at,
        });
    }

    // UnderFireAt: singleton key. Position is the attacker's last
    // known pos written by `npc_combat` on damage application.
    if let Some(entry) = bb.get(&BlackboardKey::UnderFireAt) {
        if let BlackboardValue::Position(pos) = entry.value {
            candidates.push(Candidate {
                source: GoalSource::BlackboardUrgency,
                kind: GoalKind::InvestigateAt { pos },
                priority: PRIO_BLACKBOARD_UNDER_FIRE,
                created_tick: entry.written_tick,
                expires_at: entry.written_tick.checked_add(entry.ttl_ticks as u64),
            });
        }
    }

    // HeardGunshot: singleton key. Position is the gunshot origin
    // written by `world_event_bus::apply_to_blackboard` on drain.
    if let Some(entry) = bb.get(&BlackboardKey::HeardGunshot) {
        if let BlackboardValue::Position(pos) = entry.value {
            candidates.push(Candidate {
                source: GoalSource::BlackboardUrgency,
                kind: GoalKind::InvestigateAt { pos },
                priority: PRIO_BLACKBOARD_GUNSHOT,
                created_tick: entry.written_tick,
                expires_at: entry.written_tick.checked_add(entry.ttl_ticks as u64),
            });
        }
    }

    // LastKnownEnemyPos: faction-wide aggro propagation. The bus
    // routes `EnemySighted` to every hostile-to-target squad
    // within audible range, which writes this key on their
    // blackboard. Without an arbitration candidate the intel
    // sat unused — a wandering same-faction squad would walk
    // past a teammate-vs-enemy firefight 30 m away with no
    // reaction. Push as an InvestigateAt at priority
    // `PRIO_BLACKBOARD_LAST_KNOWN_ENEMY = 130` so the squad
    // breaks off non-combat objectives (Rest, Wander, Patrol)
    // to converge on the enemy's last-known position. The
    // squad's own `npc_aggro` perception will upgrade this to
    // a real `SquadAggro`/`IndividualAggro` once they get LOS.
    if let Some(entry) = bb.get(&BlackboardKey::LastKnownEnemyPos) {
        if let BlackboardValue::Position(pos) = entry.value {
            candidates.push(Candidate {
                source: GoalSource::BlackboardUrgency,
                kind: GoalKind::InvestigateAt { pos },
                priority: PRIO_BLACKBOARD_LAST_KNOWN_ENEMY,
                created_tick: entry.written_tick,
                expires_at: entry.written_tick.checked_add(entry.ttl_ticks as u64),
            });
        }
    }
}

/// Pick the highest-priority candidate, breaking ties by older
/// `created_tick` (long-running plans aren't preempted by their own
/// re-derivation). Returns `None` only if the list is empty — the
/// resolver always pushes an Idle candidate so this shouldn't happen.
fn pick_winner(candidates: &[Candidate]) -> Option<Candidate> {
    candidates.iter().copied().max_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then(b.created_tick.cmp(&a.created_tick))
    })
}

/// Update `current` in place. Same-source re-derivations refresh
/// `expires_at` (and only that) so a cooperating long-running plan
/// can rethink its TTL each tick. Different-source winners must beat
/// the existing priority by at least `HYSTERESIS_PRIO_DELTA` to
/// preempt; otherwise the existing goal sticks.
fn apply_with_hysteresis(current: &mut ActiveGoal, winner: Candidate, now: u64) {
    let arb = BehaviorConfig::load().arbitration;
    let same_source_kind = current.source == winner.source && current.kind == winner.kind;
    if same_source_kind {
        current.expires_at = winner.expires_at;
        return;
    }
    if now < current.committed_until_tick && winner.priority < COMMITMENT_BYPASS_PRIO {
        return;
    }
    let preempts = winner.priority >= current.priority.saturating_add(arb.hysteresis_prio_delta);
    if preempts {
        let commit = if winner.source == GoalSource::SquadObjective {
            now + arb.commitment_ticks
        } else {
            0
        };
        *current = ActiveGoal {
            source: winner.source,
            kind: winner.kind,
            priority: winner.priority,
            created_tick: now,
            expires_at: winner.expires_at,
            pursue_progress: None,
            committed_until_tick: commit,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::NpcId;

    fn cand(source: GoalSource, kind: GoalKind, priority: u8, created_tick: u64) -> Candidate {
        Candidate {
            source,
            kind,
            priority,
            created_tick,
            expires_at: None,
        }
    }

    #[test]
    fn pick_winner_picks_highest_priority() {
        let cs = vec![
            cand(GoalSource::Idle, GoalKind::SoloIdleFsm, 0, 0),
            cand(
                GoalSource::IndividualAggro,
                GoalKind::PursueTarget { target: NpcId(1) },
                150,
                10,
            ),
            cand(
                GoalSource::SquadObjective,
                GoalKind::SquadFollowObjective,
                80,
                5,
            ),
        ];
        let winner = pick_winner(&cs).unwrap();
        assert_eq!(winner.source, GoalSource::IndividualAggro);
    }

    #[test]
    fn pick_winner_squad_aggro_outranks_individual_aggro() {
        // Both candidates exist for grouped NPCs with aggro. Squad
        // coordination is the default — the user-locked decision.
        let cs = vec![
            cand(
                GoalSource::IndividualAggro,
                GoalKind::PursueTarget { target: NpcId(1) },
                150,
                10,
            ),
            cand(
                GoalSource::SquadAggro,
                GoalKind::PursueTarget { target: NpcId(1) },
                160,
                10,
            ),
        ];
        let winner = pick_winner(&cs).unwrap();
        assert_eq!(winner.source, GoalSource::SquadAggro);
    }

    #[test]
    fn pick_winner_breaks_ties_by_older_created_tick() {
        // Same priority — the older one wins so a long-running plan
        // isn't preempted by its own re-derivation each tick.
        let cs = vec![
            cand(
                GoalSource::SquadObjective,
                GoalKind::SquadFollowObjective,
                80,
                100,
            ),
            cand(
                GoalSource::SquadObjective,
                GoalKind::SquadFollowObjective,
                80,
                10,
            ),
        ];
        let winner = pick_winner(&cs).unwrap();
        assert_eq!(winner.created_tick, 10);
    }

    #[test]
    fn hysteresis_blocks_low_delta_preemption() {
        // Existing IndividualAggro at 150 should not be preempted by
        // SquadObjective at 80 — that's a -70 delta, way below the
        // +20 hysteresis threshold.
        let mut current = ActiveGoal {
            source: GoalSource::IndividualAggro,
            kind: GoalKind::PursueTarget { target: NpcId(1) },
            priority: 150,
            created_tick: 5,
            expires_at: None,
            pursue_progress: None,
            committed_until_tick: 0,
        };
        let winner = cand(
            GoalSource::SquadObjective,
            GoalKind::SquadFollowObjective,
            80,
            10,
        );
        apply_with_hysteresis(&mut current, winner, 100);
        assert_eq!(current.source, GoalSource::IndividualAggro);
        assert_eq!(current.priority, 150);
    }

    #[test]
    fn hysteresis_allows_high_delta_preemption() {
        let mut current = ActiveGoal {
            source: GoalSource::SquadObjective,
            kind: GoalKind::SquadFollowObjective,
            priority: 80,
            created_tick: 5,
            expires_at: None,
            pursue_progress: None,
            committed_until_tick: 0,
        };
        let winner = cand(
            GoalSource::SquadAggro,
            GoalKind::PursueTarget { target: NpcId(2) },
            160,
            50,
        );
        apply_with_hysteresis(&mut current, winner, 100);
        assert_eq!(current.source, GoalSource::SquadAggro);
        assert_eq!(current.priority, 160);
        assert_eq!(current.created_tick, 100, "preemption stamps now");
    }

    #[test]
    fn same_source_refreshes_in_place() {
        let mut current = ActiveGoal {
            source: GoalSource::SquadObjective,
            kind: GoalKind::SquadFollowObjective,
            priority: 80,
            created_tick: 5,
            expires_at: None,
            pursue_progress: None,
            committed_until_tick: 0,
        };
        let winner = Candidate {
            source: GoalSource::SquadObjective,
            kind: GoalKind::SquadFollowObjective,
            priority: 80,
            created_tick: 90,
            expires_at: Some(200),
        };
        apply_with_hysteresis(&mut current, winner, 100);
        assert_eq!(current.created_tick, 5, "long-running plan preserved");
        assert_eq!(current.expires_at, Some(200), "TTL refreshed");
    }

    fn pos(x: f32, z: f32) -> [f32; 3] {
        [x, 0.0, z]
    }

    #[test]
    fn blackboard_empty_pushes_nothing() {
        let bb = SquadBlackboards::default();
        let mut out = Vec::new();
        push_blackboard_candidates(&bb, 1, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn blackboard_heard_gunshot_emits_investigate_at_gunshot_prio() {
        let mut bb = SquadBlackboards::default();
        bb.write(
            1,
            BlackboardKey::HeardGunshot,
            BlackboardValue::Position(pos(10.0, 20.0)),
            50,
            100,
        );
        let mut out = Vec::new();
        push_blackboard_candidates(&bb, 1, &mut out);
        assert_eq!(out.len(), 1);
        let c = out[0];
        assert_eq!(c.source, GoalSource::BlackboardUrgency);
        assert_eq!(c.priority, PRIO_BLACKBOARD_GUNSHOT);
        assert!(matches!(
            c.kind,
            GoalKind::InvestigateAt { pos: p } if p == pos(10.0, 20.0)
        ));
        assert_eq!(c.created_tick, 50);
        assert_eq!(c.expires_at, Some(150));
    }

    #[test]
    fn blackboard_under_fire_outranks_heard_gunshot() {
        let mut bb = SquadBlackboards::default();
        bb.write(
            7,
            BlackboardKey::HeardGunshot,
            BlackboardValue::Position(pos(1.0, 0.0)),
            10,
            100,
        );
        bb.write(
            7,
            BlackboardKey::UnderFireAt,
            BlackboardValue::Position(pos(2.0, 0.0)),
            12,
            100,
        );
        let mut out = Vec::new();
        push_blackboard_candidates(&bb, 7, &mut out);
        assert_eq!(out.len(), 2);
        let max_prio = out.iter().map(|c| c.priority).max().unwrap();
        assert_eq!(max_prio, PRIO_BLACKBOARD_UNDER_FIRE);
    }

    #[test]
    fn blackboard_downed_ally_emits_regroup_with_id() {
        let mut bb = SquadBlackboards::default();
        bb.write(
            3,
            BlackboardKey::DownedAlly { id: NpcId(42) },
            BlackboardValue::Position(pos(5.0, 5.0)),
            20,
            600,
        );
        let mut out = Vec::new();
        push_blackboard_candidates(&bb, 3, &mut out);
        assert_eq!(out.len(), 1);
        let c = out[0];
        assert_eq!(c.source, GoalSource::BlackboardUrgency);
        assert_eq!(c.priority, PRIO_BLACKBOARD_DOWNED_ALLY);
        let GoalKind::RegroupOnAlly { id, pos: p } = c.kind else {
            panic!("expected RegroupOnAlly, got {:?}", c.kind);
        };
        assert_eq!(id, NpcId(42));
        assert_eq!(p, pos(5.0, 5.0));
    }

    #[test]
    fn blackboard_multiple_downed_allies_picks_lowest_id() {
        // Two DownedAlly entries on the same blackboard. Both have
        // priority 180; tiebreak must be deterministic. The helper
        // picks the lowest NpcId.
        let mut bb = SquadBlackboards::default();
        bb.write(
            9,
            BlackboardKey::DownedAlly { id: NpcId(7) },
            BlackboardValue::Position(pos(1.0, 0.0)),
            10,
            600,
        );
        bb.write(
            9,
            BlackboardKey::DownedAlly { id: NpcId(3) },
            BlackboardValue::Position(pos(2.0, 0.0)),
            12,
            600,
        );
        bb.write(
            9,
            BlackboardKey::DownedAlly { id: NpcId(11) },
            BlackboardValue::Position(pos(3.0, 0.0)),
            14,
            600,
        );
        let mut out = Vec::new();
        push_blackboard_candidates(&bb, 9, &mut out);
        assert_eq!(out.len(), 1);
        let GoalKind::RegroupOnAlly { id, .. } = out[0].kind else {
            panic!();
        };
        assert_eq!(id, NpcId(3), "lowest-id ally should win deterministically");
    }

    #[test]
    fn blackboard_non_position_values_skipped() {
        // A future writer might land a wrong-type value (Bool, etc.)
        // on these keys; the helper silently ignores rather than
        // panicking.
        let mut bb = SquadBlackboards::default();
        bb.write(
            1,
            BlackboardKey::HeardGunshot,
            BlackboardValue::Bool(true),
            10,
            100,
        );
        bb.write(
            1,
            BlackboardKey::UnderFireAt,
            BlackboardValue::Tick(0),
            10,
            100,
        );
        bb.write(
            1,
            BlackboardKey::DownedAlly { id: NpcId(1) },
            BlackboardValue::Float(0.0),
            10,
            600,
        );
        let mut out = Vec::new();
        push_blackboard_candidates(&bb, 1, &mut out);
        assert!(
            out.is_empty(),
            "non-Position values shouldn't produce candidates"
        );
    }
}
