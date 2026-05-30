//! Build the per-tick `NpcPositionIndex` so other systems can look
//! up an NPC's position/region/health by `NpcId` without needing a
//! second query borrow on the NPC table (which trips bevy_ecs's
//! query-conflict guard).
//!
//! Also computes per-`Group` centroids so cohesion checks downstream
//! don't need to re-iterate NPCs.

use bevy_ecs::prelude::*;
use std::collections::HashMap;

use crate::components::{Group, Health, InRegion, Npc, Position};
use crate::resources::{GroupCentroid, NpcPositionEntry, NpcPositionIndex};

pub fn index_npc_positions(
    npcs: Query<(&Npc, &InRegion, &Position, &Health, Option<&Group>)>,
    mut index: ResMut<NpcPositionIndex>,
) {
    let _diag_t = crate::systems::SysTimer::new("index_npc_positions");
    let prof_t = std::time::Instant::now();
    index.by_id.clear();
    index.group_centroids.clear();

    // Walk once to fill by_id and accumulate centroid sums.
    let mut sums: HashMap<u64, ([f64; 3], u32, crate::region::RegionId)> = HashMap::new();
    for (n, r, p, h, g) in npcs.iter() {
        let group = g.map(|g| g.id);
        index.by_id.insert(
            n.id,
            NpcPositionEntry {
                pos: p.0,
                region: r.0,
                health: h.current,
                group,
            },
        );
        if let Some(gid) = group {
            let entry = sums.entry(gid).or_insert(([0.0; 3], 0, r.0));
            entry.0[0] += p.0[0] as f64;
            entry.0[1] += p.0[1] as f64;
            entry.0[2] += p.0[2] as f64;
            entry.1 += 1;
            entry.2 = r.0;
        }
    }
    for (gid, (sum, count, region)) in sums {
        let n = count as f64;
        index.group_centroids.insert(
            gid,
            GroupCentroid {
                pos: [
                    (sum[0] / n) as f32,
                    (sum[1] / n) as f32,
                    (sum[2] / n) as f32,
                ],
                region,
                member_count: count,
            },
        );
    }
    crate::systems::record_perception_slot(
        crate::systems::prof_slots::POSITION_INDEX,
        prof_t.elapsed(),
    );
}
