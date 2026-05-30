//! Combat death gate.
//!
//! Each tick: scan NPCs whose `Health.current <= 0`. For each, write
//! the chronicle death record (`DeathCause::Combat { killer_faction }`
//! using the `LastDamager` hint stamped by `npc_combat`), despawn
//! the entity, push a `WorldDelta::NpcDied` so the journal
//! preserves the death across crashes, and push an
//! [`AllyDown`](crate::world_event_bus::WorldEventKind::AllyDown)
//! event to the world event bus so same-faction squads pick up a
//! [`DownedAlly`](crate::squad_blackboard::BlackboardKey::DownedAlly)
//! blackboard entry on the next tick.
//!
//! `LastDamager` falls back to `Other` if absent (shouldn't happen
//! once aggro+combat is in play, but defensive).
//!
//! `CorpseSpotted` is observer-driven (an ally has to *see* the
//! body) and so is not emitted from here — it lives with a future
//! corpse-perception pass that scans for nearby `WorldContainer`s of
//! `Corpse` kind.

use bevy_ecs::prelude::*;

use crate::chronicle::{DeathCause, LifeChronicle};
use crate::components::{Health, InFaction, InRegion, Inventory, LastDamager, Npc, Position};
use crate::delta::WorldDelta;
use crate::resources::{
    ContainerIdCounter, CorpseIndex, CorpseIndexEntry, PendingDeltas, SimClock,
};
use crate::world::containers::spawn_corpse_container;
use crate::world_event_bus::{WorldEventKind, WorldEventQueue};

/// TTL (ticks) for `AllyDown` events. Drain runs next tick and
/// writes a longer-lived `DownedAlly` blackboard entry on listeners
/// (currently 600 ticks ≈ 30 s, defined in
/// [`crate::world_event_bus::apply_to_blackboard`]).
const ALLY_DOWN_EVENT_TTL_TICKS: u32 = 2;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn npc_death_check(
    clock: Res<SimClock>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    npcs: Query<(
        Entity,
        &Npc,
        &InFaction,
        &InRegion,
        &Health,
        &Position,
        Option<&Inventory>,
        Option<&LastDamager>,
    )>,
    mut commands: Commands,
    mut chronicle: ResMut<LifeChronicle>,
    mut pending: ResMut<PendingDeltas>,
    mut container_counter: ResMut<ContainerIdCounter>,
    mut event_queue: ResMut<WorldEventQueue>,
    mut corpse_index: ResMut<CorpseIndex>,
) {
    let _diag_t = crate::systems::SysTimer::new("npc_death_check");
    let now = clock.tick;
    // Determinism: query iteration order is archetype-storage order,
    // which isn't stable across sim instances. The order of
    // `WorldEvent` ids assigned to AllyDown pushes (and the
    // `PendingDeltas` journal writes alongside them) needs to be
    // identical between two same-seed sims; sort by NpcId once.
    let mut dead: Vec<_> = npcs
        .iter()
        .filter(|(_, _, _, _, h, _, _, _)| h.current <= 0.0)
        .collect();
    dead.sort_by_key(|(_, npc, _, _, _, _, _, _)| npc.id);
    for (entity, npc, faction, region, _health, pos, inventory, damager) in dead {
        let cause = match damager {
            Some(d) => DeathCause::Combat {
                killer_faction: registry.name_of(d.faction).to_string(),
            },
            None => DeathCause::Other,
        };
        // Skip if this NPC's chronicle record is already marked
        // dead (defensive — same entity shouldn't enter this loop
        // twice in one tick, but a stale `WoundSeverity::Fatal`
        // could).
        if let Some(rec) = chronicle.get(npc.id) {
            if rec.death_tick.is_some() {
                continue;
            }
        }
        chronicle.mark_dead(npc.id, now, region.0, cause.clone());
        // Convert pockets → corpse container BEFORE despawning. Empty
        // inventories produce no container (no corpse-noise on NPCs
        // that rolled empty loadouts). When a container is spawned,
        // mirror it into `CorpseIndex` and tag the entity with a
        // `CorpseMarker` so the loot arbiter can find it fast.
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
            cause,
            tick: now,
        });
        // Notify the deceased's faction: same-faction squads within
        // audible radius pick up a `DownedAlly { id }` blackboard
        // entry on next tick (audience filter handles the faction
        // gate; spatial decay scales TTL per listener).
        event_queue.push(
            WorldEventKind::AllyDown {
                id: npc.id,
                faction: faction.0,
            },
            pos.0,
            region.0,
            now,
            ALLY_DOWN_EVENT_TTL_TICKS,
        );
    }
}

/// Age entries out of [`CorpseIndex`] so the loot arbiter doesn't
/// keep targeting corpses that have despawned or have grown stale.
/// Containers don't despawn today (they persist as drops), but
/// well-aged corpses are no longer interesting Loot targets — once a
/// faction has had time to sweep them, the arbiter should consider
/// fresher targets instead. Sweep is cheap (`O(corpses)`) and runs
/// once per tick in the lifecycle segment.
const CORPSE_INDEX_TTL_TICKS: u64 = 12000; // ~10 min real

pub fn prune_corpse_index(clock: Res<SimClock>, mut corpse_index: ResMut<CorpseIndex>) {
    let now = clock.tick;
    corpse_index
        .by_container
        .retain(|_, e| now.saturating_sub(e.spawned_tick) < CORPSE_INDEX_TTL_TICKS);
}
