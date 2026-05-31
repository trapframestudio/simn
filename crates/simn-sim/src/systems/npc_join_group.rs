//! "Solo NPC joins a friendly squad" placeholder.
//!
//! Every `JOIN_INTERVAL_TICKS`, walk every NPC that doesn't have a
//! `Group`. For each, find the closest squad in the same region
//! whose faction is the same or `Warm`-related, within
//! `JOIN_RADIUS_M`. With probability `JOIN_PROB_NUM/JOIN_PROB_DEN`
//! per check, attach the solo to that squad (insert `Group { id }`).
//!
//! Nomads stay solo by design (their faction archetype is the
//! drifter — letting them group up would erase their flavor).
//! Everyone else can pick up stragglers.
//!
//! Placeholder. Real squad merging / faction politics lands with
//! the brain layer.

use bevy_ecs::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::components::{Group, InFaction, InRegion, Npc, Position};
use crate::faction::Relation;
use crate::resources::SimClock;

/// Run once per ~5s.
const JOIN_INTERVAL_TICKS: u64 = 100;
/// Solo must be within this distance of a squad member to consider joining.
const JOIN_RADIUS_M: f32 = 30.0;
const JOIN_RADIUS_SQ_M: f32 = JOIN_RADIUS_M * JOIN_RADIUS_M;
/// Roll probability per qualifying check. 1/3 chance each pass.
const JOIN_PROB_NUM: u32 = 1;
const JOIN_PROB_DEN: u32 = 3;

#[allow(clippy::type_complexity)]
pub fn npc_join_group(
    clock: Res<SimClock>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    deltas: Res<crate::faction::registry::RelationDeltas>,
    npcs: Query<(
        Entity,
        &Npc,
        &InFaction,
        &InRegion,
        &Position,
        Option<&Group>,
    )>,
    mut commands: Commands,
) {
    let _diag_t = crate::systems::SysTimer::new("npc_join_group");
    if clock.tick == 0 || !clock.tick.is_multiple_of(JOIN_INTERVAL_TICKS) {
        return;
    }

    // Snapshot once so the O(n×k) walk doesn't fight the borrow checker.
    let snap: Vec<(
        Entity,
        crate::faction::registry::FactionId,
        u32,
        [f32; 3],
        Option<u64>,
    )> = npcs
        .iter()
        .map(|(e, _, f, r, p, g)| (e, f.0, r.0, p.0, g.map(|g| g.id)))
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(clock.tick.wrapping_mul(0xA5A5_5A5A_F00D_BABE));

    let baseline_id = registry.player_baseline();
    for (entity, faction, region, pos, group) in &snap {
        if group.is_some() {
            continue;
        }
        if Some(*faction) == baseline_id {
            // The player-baseline faction stays solo (loners don't
            // auto-join squads). Config-driven via `player_baseline`.
            continue;
        }
        // Find nearest same-region group member with a friendly relation.
        let mut best: Option<(f32, u64)> = None;
        for (e2, f2, r2, p2, g2) in &snap {
            if e2 == entity {
                continue;
            }
            if r2 != region {
                continue;
            }
            let Some(gid) = g2 else { continue };
            // Same faction or warm-related → friendly enough to join.
            let rel = crate::faction::registry::faction_relation(&registry, &deltas, *faction, *f2);
            if rel != Relation::Warm {
                continue;
            }
            let dx = p2[0] - pos[0];
            let dz = p2[2] - pos[2];
            let d_sq = dx * dx + dz * dz;
            if d_sq > JOIN_RADIUS_SQ_M {
                continue;
            }
            if best.map(|(b, _)| d_sq < b).unwrap_or(true) {
                best = Some((d_sq, *gid));
            }
        }
        let Some((_, gid)) = best else { continue };
        if !rng.gen_ratio(JOIN_PROB_NUM, JOIN_PROB_DEN) {
            continue;
        }
        commands.entity(*entity).insert(Group { id: gid });
    }
}
