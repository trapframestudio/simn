//! Lifespan-based death.
//!
//! When an NPC's tick exceeds `Lifespan::die_at_tick`, the NPC dies
//! of natural causes: chronicle is updated, entity despawned, journal
//! delta queued via `PendingDeltas`. Real combat death (HP → 0) lands
//! when combat does and uses the same chronicle write path.
//!
//! Two passes because bevy_ecs needs disjoint borrows: first collect
//! the (entity, id, region) triples to kill, then mutate world state.

use bevy_ecs::prelude::*;

use crate::chronicle::{DeathCause, LifeChronicle};
use crate::components::{InFaction, InRegion, Inventory, Lifespan, Npc, Position};
use crate::delta::WorldDelta;
use crate::resources::{
    ContainerIdCounter, CorpseIndex, CorpseIndexEntry, PendingDeltas, SimClock,
};
use crate::world::containers::spawn_corpse_container;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn age_npcs(
    clock: Res<SimClock>,
    npcs: Query<(
        Entity,
        &Npc,
        &InFaction,
        &InRegion,
        &Lifespan,
        &Position,
        Option<&Inventory>,
    )>,
    mut commands: Commands,
    mut chronicle: ResMut<LifeChronicle>,
    mut pending: ResMut<PendingDeltas>,
    mut container_counter: ResMut<ContainerIdCounter>,
    mut corpse_index: ResMut<CorpseIndex>,
) {
    let _diag_t = crate::systems::SysTimer::new("age_npcs");
    let now = clock.tick;
    for (entity, npc, faction, region, lifespan, pos, inventory) in npcs.iter() {
        if now < lifespan.die_at_tick {
            continue;
        }
        chronicle.mark_dead(npc.id, now, region.0, DeathCause::NaturalCauses);
        if let Some(inv) = inventory {
            let cid = spawn_corpse_container(
                &mut commands,
                &mut container_counter,
                &mut pending,
                pos.0,
                region.0,
                inv.0.clone(),
            );
            if let Some(cid) = cid {
                corpse_index.by_container.insert(
                    cid,
                    CorpseIndexEntry {
                        pos: pos.0,
                        region: region.0,
                        faction: faction.0,
                        spawned_tick: now,
                    },
                );
            }
        }
        commands.entity(entity).despawn();
        pending.push(WorldDelta::NpcDied {
            id: npc.id,
            region: region.0,
            cause: DeathCause::NaturalCauses,
            tick: now,
        });
    }
}
