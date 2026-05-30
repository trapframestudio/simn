//! Per-tick NPC transform broadcast for host → client replication.
//!
//! Emits one `WorldDelta::NpcPositionBatch` covering every NPC in
//! an active region (one the local player is in). Inactive-region
//! NPCs are omitted — the client mirror sim doesn't visualize them
//! anyway, and replicating positions for thousands of distant NPCs
//! is pure cost. Slice-2 per-region subscription on the wire is a
//! separate, finer-grained filter; this is the server-side prefilter.
//!
//! The client's mirror sim applies the batch to keep pill positions
//! aligned with the host without running the NPC-mutating systems
//! (which would diverge because their RNG seeds depend on
//! `Entity::to_bits()`, which isn't stable across sim instances).
//!
//! Not journaled — the authoritative journal re-derives positions from
//! the NPC schedule on replay. This system only populates the per-tick
//! broadcast buffer via `PendingDeltas`, which `Sim::tick` drains into
//! `last_tick_deltas`.

use bevy_ecs::prelude::{Query, Res, ResMut};

use crate::components::{InRegion, Npc, Position, Rotation};
use crate::delta::WorldDelta;
use crate::resources::{ActiveRegions, PendingDeltas, SimClock};

pub fn broadcast_npc_positions(
    q: Query<(&Npc, &InRegion, &Position, &Rotation)>,
    active_regions: Res<ActiveRegions>,
    mut pending: ResMut<PendingDeltas>,
    clock: Res<SimClock>,
) {
    let _diag_t = crate::systems::SysTimer::new("broadcast_npc_positions");
    // No active region (sim before player join / after they leave)
    // → skip entirely. No mirror consumer in that state.
    if active_regions.regions.is_empty() {
        return;
    }
    let updates: Vec<(crate::components::NpcId, [f32; 3], f32)> = q
        .iter()
        .filter_map(|(npc, region, pos, rot)| {
            active_regions
                .is_active(region.0)
                .then_some((npc.id, pos.0, rot.0))
        })
        .collect();
    if updates.is_empty() {
        return;
    }
    pending.push(WorldDelta::NpcPositionBatch {
        tick: clock.tick,
        updates,
    });
}
