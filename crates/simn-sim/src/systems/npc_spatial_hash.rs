//! Per-tick rebuild of [`NpcSpatialHash`].
//!
//! Pure system. Runs after [`super::index_npc_positions`] and before
//! [`super::npc_aggro`] so Pass 2 of aggro can iterate cells instead
//! of the O(Σ n_r²) snapshot pair scan.
//!
//! Architecturally parallel to `index_npc_positions`: both rebuild a
//! read-only snapshot of every NPC from the ECS query, keyed
//! differently. `NpcPositionIndex` is keyed by `NpcId` (lookup a
//! target by id); `NpcSpatialHash` is keyed by `(region, cell)`
//! (find all NPCs near a point).
//!
//! The hash stays stale by one tick for movement (positions update
//! in `tick_npc_goals` after this system has already run), which is
//! fine — aggro was already tolerant of 20Hz sight-radius granularity.

use bevy_ecs::prelude::{Entity, Query, ResMut};

use crate::components::{InFaction, InRegion, Npc, Position, Rotation};
use crate::resources::{NpcSpatialHash, SpatialEntry};

pub fn rebuild_spatial_hash(
    npcs: Query<(Entity, &Npc, &InFaction, &InRegion, &Position, &Rotation)>,
    mut hash: ResMut<NpcSpatialHash>,
) {
    let _diag_t = crate::systems::SysTimer::new("rebuild_spatial_hash");
    let prof_t = std::time::Instant::now();
    hash.clear();
    for (entity, npc, faction, region, pos, rot) in &npcs {
        let entry = SpatialEntry {
            npc_id: npc.id,
            entity,
            pos: pos.0,
            yaw: rot.0,
            faction: faction.0,
            region: region.0,
        };
        hash.grid_mut(region.0).insert(entry);
    }
    crate::systems::record_perception_slot(
        crate::systems::prof_slots::SPATIAL_HASH,
        prof_t.elapsed(),
    );
}
