//! Squad threat-board aggregation.
//!
//! Per tick, this system:
//!
//! 1. Sweeps each NPC's [`RecentAttackers`] — drops entries past
//!    [`THREAT_TTL_TICKS`].
//! 2. For every `Group`, aggregates members' surviving
//!    `RecentAttackers` into a single
//!    [`BlackboardValue::Threats`] entry on the squad blackboard,
//!    keyed by attacker `NpcId` and scored by
//!    `damage × recency × proximity`.
//!
//! See `docs/book/src/planning/threat-board-plan.md` for the
//! design. Goal arbitration reads the resulting `ThreatList` to
//! switch a squad's `Aggro.target` when a new attacker dominates
//! (lands in step 3).
//!
//! ## Determinism
//!
//! Iterates groups + members in a stable order so a same-seed sim
//! produces byte-identical threat boards. The aggregation does no
//! RNG draws, so iteration order doesn't matter for randomness —
//! but the resulting `Vec<ThreatEntry>` ordering is observable
//! (it's serialized into the squad blackboard, which can be
//! inspected) and a stable sort by `attacker_id` keeps the output
//! reproducible.

use bevy_ecs::prelude::*;
use std::collections::HashMap;

use crate::components::{Group, Npc, NpcId, RecentAttackers};
use crate::resources::{NpcPositionIndex, SimClock};
use crate::squad_blackboard::{BlackboardKey, BlackboardValue, SquadBlackboards, ThreatEntry};
use crate::systems::npc_combat::{MAX_RECENT_ATTACKERS, THREAT_TTL_TICKS};

/// Inside this radius, attackers count at full proximity weight.
const PROX_FULL_RADIUS_M: f32 = 30.0;
/// Past this radius, attackers count at zero proximity weight.
/// Proximity factor falls linearly from `1.0` at `PROX_FULL_RADIUS_M`
/// to `0.0` at `PROX_FADE_RADIUS_M`.
const PROX_FADE_RADIUS_M: f32 = 80.0;

/// TTL the threat board entry persists in the blackboard. Re-written
/// every tick the system runs, so the actual lifetime is "until the
/// next sweep skips this group" — same as the underlying
/// `RecentAttackers` retention. We set the value generously so a
/// blackboard sweep doesn't drop it between writes.
const THREAT_BOARD_BB_TTL_TICKS: u32 = THREAT_TTL_TICKS as u32 + 10;

#[allow(clippy::type_complexity)]
pub fn sweep_threats(
    clock: Res<SimClock>,
    mut npcs: Query<(
        &Npc,
        Option<&Group>,
        &crate::components::InRegion,
        &crate::components::Position,
        &mut RecentAttackers,
    )>,
    index: Res<NpcPositionIndex>,
    active_regions: Res<crate::resources::ActiveRegions>,
    mut blackboards: ResMut<SquadBlackboards>,
) {
    let _diag_t = crate::systems::SysTimer::new("sweep_threats");
    let now = clock.tick;
    let cutoff = now.saturating_sub(THREAT_TTL_TICKS);

    // Pass 1: sweep per-NPC ring buffers; collect a (group_id,
    // member_pos, recent_events) triple for grouped NPCs so we can
    // aggregate without a second mutable borrow.
    //
    // Active-region filter: offline-region NPCs skip the sweep and
    // the aggregation. Their `RecentAttackers` rings drift until the
    // region re-activates, which is fine — threat state is
    // perception-derived and decays automatically via TTL.
    //
    // Determinism: group iteration order matters because the threat
    // list is observable. Sort by group_id when emitting.
    let mut by_group: HashMap<u64, Vec<(NpcId, [f32; 3], Vec<crate::components::AttackerHit>)>> =
        HashMap::new();
    for (npc, group, region, pos, mut recent) in npcs.iter_mut() {
        if !active_regions.is_active(region.0) {
            continue;
        }
        recent.sweep(cutoff);
        let Some(g) = group else { continue };
        // Clone the events list (small, ≤ MAX_RECENT_ATTACKERS) so
        // the aggregation pass below can iterate without a borrow.
        if recent.events.is_empty() {
            continue;
        }
        by_group
            .entry(g.id)
            .or_default()
            .push((npc.id, pos.0, recent.events.clone()));
    }

    // Pass 2: aggregate per group. Sum damage per attacker_id across
    // all squadmates, score with recency × proximity, sort descending.
    let mut group_ids: Vec<u64> = by_group.keys().copied().collect();
    group_ids.sort_unstable();
    for gid in group_ids {
        let members = &by_group[&gid];
        // Aggregate damage per attacker across all squadmates.
        let mut acc: HashMap<NpcId, AggAcc> = HashMap::with_capacity(MAX_RECENT_ATTACKERS);
        for (_member_id, member_pos, events) in members {
            for hit in events {
                let entry = acc.entry(hit.attacker_id).or_insert(AggAcc {
                    damage: 0.0,
                    last_seen_tick: 0,
                    nearest_to_member_dist_sq: f32::INFINITY,
                });
                entry.damage += hit.damage;
                entry.last_seen_tick = entry.last_seen_tick.max(hit.tick);
                // Proximity is computed against the closest member,
                // since the squad threat board represents "the
                // threat at the squad's nearest face." If the
                // attacker isn't in the position index (offline
                // tier, despawned), skip the proximity update.
                if let Some(target) = index.by_id.get(&hit.attacker_id) {
                    let dx = target.pos[0] - member_pos[0];
                    let dz = target.pos[2] - member_pos[2];
                    let d2 = dx * dx + dz * dz;
                    if d2 < entry.nearest_to_member_dist_sq {
                        entry.nearest_to_member_dist_sq = d2;
                    }
                }
            }
        }

        // Score + serialize. Sort by attacker_id first so the
        // top-scored ordering is deterministic on ties.
        let mut entries: Vec<(NpcId, AggAcc)> = acc.into_iter().collect();
        entries.sort_by_key(|(id, _)| *id);
        let mut threats: Vec<ThreatEntry> = entries
            .into_iter()
            .map(|(target_id, agg)| {
                let recency = recency_factor(now, agg.last_seen_tick);
                let proximity = proximity_factor(agg.nearest_to_member_dist_sq);
                ThreatEntry {
                    target_id,
                    score: agg.damage * recency * proximity,
                    last_seen_tick: agg.last_seen_tick,
                }
            })
            .collect();
        // Highest score first. Stable sort preserves the attacker_id
        // tiebreak from the previous sort.
        threats.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if threats.is_empty() {
            // Empty threat list — drop any stale entry from the
            // blackboard rather than writing an empty Vec.
            continue;
        }
        blackboards.write(
            gid,
            BlackboardKey::ThreatList,
            BlackboardValue::Threats(threats),
            now,
            THREAT_BOARD_BB_TTL_TICKS,
        );
    }
}

struct AggAcc {
    damage: f32,
    last_seen_tick: u64,
    /// Squared distance from the closest squadmate to the attacker.
    /// Squared so we don't sqrt during the inner aggregation loop.
    nearest_to_member_dist_sq: f32,
}

/// Hysteresis: a new top-threat must beat the current `Aggro.target`'s
/// score in the squad threat board by this MULTIPLIER OR by this
/// ABSOLUTE delta to flip the target. Prevents thrashing between
/// similar-scored attackers when they trade fire on the squad.
pub const THREAT_SWITCH_MULTIPLIER: f32 = 1.5;
pub const THREAT_SWITCH_ABSOLUTE_DELTA: f32 = 2.0;

/// Cut a squad's individual `Aggro.target` over to the squad threat
/// board's top entry when the new threat dominates by enough margin
/// to clear hysteresis. Runs after `sweep_threats` (so the board is
/// current) and before `goal_arbitration` (so the resolver sees the
/// updated target this tick). Squad-coordinated focus fire emerges
/// from this single rule: every member ends up pursuing whoever the
/// squad as a whole is most threatened by.
///
/// Lone NPCs (no `Group`) are unaffected; they keep their
/// `npc_aggro`-acquired target unchanged.
#[allow(clippy::type_complexity)]
pub fn apply_threat_priority(
    blackboards: Res<SquadBlackboards>,
    mut aggroed: Query<(&Group, &mut crate::components::Aggro)>,
) {
    for (group, mut aggro) in aggroed.iter_mut() {
        let Some(bb) = blackboards.get(group.id) else {
            continue;
        };
        let Some(entry) = bb.get(&BlackboardKey::ThreatList) else {
            continue;
        };
        let BlackboardValue::Threats(threats) = &entry.value else {
            continue;
        };
        let Some(top) = threats.first() else {
            continue;
        };
        if top.target_id == aggro.target {
            // Already on the squad's top threat; nothing to do.
            continue;
        }
        // Find the current target's score in the same threat list
        // for the hysteresis comparison. If they're not in the list
        // (the squad isn't being threatened by the current target —
        // perception still says they're enemy but no recent damage),
        // any non-zero top threat preempts.
        let current_score = threats
            .iter()
            .find(|t| t.target_id == aggro.target)
            .map(|t| t.score)
            .unwrap_or(0.0);
        let dominates_relative = top.score >= current_score * THREAT_SWITCH_MULTIPLIER;
        let dominates_absolute = top.score >= current_score + THREAT_SWITCH_ABSOLUTE_DELTA;
        if dominates_relative || dominates_absolute {
            aggro.target = top.target_id;
            aggro.last_seen_tick = top.last_seen_tick;
        }
    }
}

/// Linear falloff: `1.0` at write tick, `0.0` at `last + TTL`.
fn recency_factor(now: u64, last_seen_tick: u64) -> f32 {
    let elapsed = now.saturating_sub(last_seen_tick) as f32;
    let ttl = THREAT_TTL_TICKS as f32;
    ((ttl - elapsed) / ttl).clamp(0.0, 1.0)
}

/// `1.0` inside engage range, linear falloff to `0.0` past sight.
fn proximity_factor(dist_sq: f32) -> f32 {
    if !dist_sq.is_finite() {
        // Attacker not in position index → unknown distance. Treat
        // as far so a stale memory of damage doesn't dominate.
        return 0.0;
    }
    let dist = dist_sq.sqrt();
    if dist <= PROX_FULL_RADIUS_M {
        return 1.0;
    }
    if dist >= PROX_FADE_RADIUS_M {
        return 0.0;
    }
    let t = (dist - PROX_FULL_RADIUS_M) / (PROX_FADE_RADIUS_M - PROX_FULL_RADIUS_M);
    1.0 - t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recency_at_write_is_one() {
        assert!((recency_factor(100, 100) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recency_at_ttl_is_zero() {
        assert!((recency_factor(100 + THREAT_TTL_TICKS, 100)).abs() < 1e-6);
    }

    #[test]
    fn recency_falls_linearly() {
        // Halfway through TTL → 0.5 factor.
        let half = THREAT_TTL_TICKS / 2;
        assert!((recency_factor(100 + half, 100) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn proximity_inside_engage_is_one() {
        let r = PROX_FULL_RADIUS_M - 5.0;
        assert!((proximity_factor(r * r) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn proximity_past_sight_is_zero() {
        let r = PROX_FADE_RADIUS_M + 5.0;
        assert!(proximity_factor(r * r).abs() < 1e-6);
    }

    #[test]
    fn proximity_falls_linearly_in_band() {
        let mid = (PROX_FULL_RADIUS_M + PROX_FADE_RADIUS_M) * 0.5;
        let f = proximity_factor(mid * mid);
        assert!((f - 0.5).abs() < 1e-6, "got {f}");
    }

    #[test]
    fn proximity_unknown_position_is_zero() {
        assert_eq!(proximity_factor(f32::INFINITY), 0.0);
    }
}
