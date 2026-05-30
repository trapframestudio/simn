//! Portal-arrival crossing for exploring squads.
//!
//! Replaces the old random-teleport migration. Squads with a
//! `SquadObjective::Explore` walk to their region's portal under
//! `tick_npc_goals`; when any member gets within
//! `PORTAL_CROSS_RADIUS_M` of the portal, every non-aggroed
//! squadmate in that region is relocated to the destination
//! region's reciprocal portal (with a small scatter so they don't
//! stack). The `Explore` objective is cleared after crossing so
//! the planner rolls a fresh objective on the next pass.

use bevy_ecs::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use crate::chronicle::LifeChronicle;
use crate::components::{Aggro, Group, InRegion, Npc, Position};
use crate::delta::WorldDelta;
use crate::region::{RegionGraph, RegionId};
use crate::resources::{BehaviorLog, PendingDeltas, SimClock, SquadObjective, SquadObjectives};

/// Any squadmate within this of the portal triggers a crossing.
/// Bumped from 10m → 25m so typical squad spread (members within
/// ~15m of centroid) reliably triggers without needing the whole
/// squad to stack exactly on the portal.
const PORTAL_CROSS_RADIUS_M: f32 = 25.0;
const PORTAL_CROSS_RADIUS_SQ_M: f32 = PORTAL_CROSS_RADIUS_M * PORTAL_CROSS_RADIUS_M;
const ARRIVAL_SCATTER_M: f32 = 12.0;

struct Crossing {
    dest_region: RegionId,
    arrival_pos: [f32; 3],
}

#[allow(clippy::type_complexity)]
pub fn npc_portal_cross(
    clock: Res<SimClock>,
    graph: Res<RegionGraph>,
    mut objectives: ResMut<SquadObjectives>,
    mut npcs: Query<(
        &Npc,
        Option<&Group>,
        Option<&Aggro>,
        &mut InRegion,
        &mut Position,
    )>,
    mut chronicle: ResMut<LifeChronicle>,
    mut pending: ResMut<PendingDeltas>,
    mut log: ResMut<BehaviorLog>,
) {
    let _diag_t = crate::systems::SysTimer::new("npc_portal_cross");
    let now = clock.tick;

    // Pass 1 (read-only): find groups with a member in range of
    // their Explore portal. Snapshot positions so we can mutate
    // below without borrow fights.
    struct Snap {
        group: Option<u64>,
        region: RegionId,
        pos: [f32; 3],
    }
    let snap: Vec<Snap> = npcs
        .iter()
        .map(|(_, g, _, r, p)| Snap {
            group: g.map(|g| g.id),
            region: r.0,
            pos: p.0,
        })
        .collect();

    let mut crossings: HashMap<u64, Crossing> = HashMap::new();
    for s in &snap {
        let Some(group_id) = s.group else { continue };
        if crossings.contains_key(&group_id) {
            continue;
        }
        let Some(state) = objectives.by_group.get(&group_id) else {
            continue;
        };
        let SquadObjective::Explore {
            dest_region,
            portal_pos,
            ..
        } = state.objective
        else {
            continue;
        };
        // Squad must still be in the region whose portal it picked.
        let Some(this_region) = graph.get(s.region) else {
            continue;
        };
        let Some(my_portal) = this_region.transitions.get(&dest_region) else {
            continue;
        };
        // Sanity: portal recorded in objective should match the
        // graph (could drift if the graph reloaded mid-objective).
        if (my_portal[0] - portal_pos[0]).abs() > 1.0 || (my_portal[2] - portal_pos[2]).abs() > 1.0
        {
            continue;
        }
        let dx = s.pos[0] - portal_pos[0];
        let dz = s.pos[2] - portal_pos[2];
        if dx * dx + dz * dz > PORTAL_CROSS_RADIUS_SQ_M {
            continue;
        }
        // Reciprocal portal on the other side.
        let Some(dest) = graph.get(dest_region) else {
            continue;
        };
        let Some(arrival_pos) = dest.transitions.get(&s.region).copied() else {
            continue;
        };
        crossings.insert(
            group_id,
            Crossing {
                dest_region,
                arrival_pos,
            },
        );
    }

    if crossings.is_empty() {
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(now.wrapping_mul(0x9E37_79B9_7F4A_7C15));

    // Pass 2: mutate. Move every non-aggroed member of a crossing
    // group to its destination portal with a small scatter; emit
    // the delta + chronicle per member.
    for (npc, group, aggro, mut region, mut pos) in npcs.iter_mut() {
        if aggro.is_some() {
            continue;
        }
        let Some(g) = group else { continue };
        let Some(crossing) = crossings.get(&g.id) else {
            continue;
        };
        let ox: f32 = rng.gen_range(-ARRIVAL_SCATTER_M..ARRIVAL_SCATTER_M);
        let oz: f32 = rng.gen_range(-ARRIVAL_SCATTER_M..ARRIVAL_SCATTER_M);
        let new_pos = [
            crossing.arrival_pos[0] + ox,
            crossing.arrival_pos[1],
            crossing.arrival_pos[2] + oz,
        ];
        region.0 = crossing.dest_region;
        pos.0 = new_pos;

        if let Some(rec) = chronicle.records.get_mut(&npc.id) {
            rec.regions_visited.push((crossing.dest_region, now));
        }
        pending.push(WorldDelta::NpcChangeRegion {
            id: npc.id,
            region: crossing.dest_region,
            pos: new_pos,
        });
        if log.enabled {
            log.migrations += 1;
            *log.migrations_by_region
                .entry(crossing.dest_region)
                .or_insert(0) += 1;
        }
    }

    // Clear the Explore objective on each crossed group so the
    // planner rolls something fresh next pass instead of pointing
    // members back at the old region's portal forever.
    for gid in crossings.keys() {
        objectives.by_group.remove(gid);
    }
}
