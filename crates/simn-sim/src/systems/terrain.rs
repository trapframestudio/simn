//! Per-tick terrain Y-clamping for NPCs.
//!
//! For each NPC in a region whose `TerrainMaps` entry is populated,
//! snaps `Position.y` to the heightmap sample at `(x, z)`. NPCs in
//! regions without attached terrain are left alone (legacy flat-floor
//! behavior).
//!
//! Runs at the end of the NPC tick chain — after `tick_npc_goals`
//! (which may have walked the NPC to new XZ) and after `spawn_npcs`
//! (so newly-spawned NPCs land on the ground in the same tick they
//! enter the world).
//!
//! Bases are clamped one-shot inside `Sim::attach_region_terrain`
//! rather than via this system, since they don't move.

use bevy_ecs::prelude::*;

use crate::components::{InRegion, Npc, Position};
use crate::resources::TerrainMaps;

pub fn clamp_npc_terrain_y(
    terrains: Res<TerrainMaps>,
    mut q: Query<(&InRegion, &mut Position), With<Npc>>,
) {
    let _diag_t = crate::systems::SysTimer::new("clamp_npc_terrain_y");
    for (region, mut pos) in q.iter_mut() {
        if let Some(y) = terrains.ground_at(region.0, pos.0[0], pos.0[2]) {
            pos.0[1] = y;
        }
    }
}
