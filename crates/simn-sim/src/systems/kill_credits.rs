//! Drain `PendingKillCredits` and promote killer NPCs by lived
//! experience.
//!
//! `npc_combat` pushes a credit when its damage application brings a
//! target's `vital_min` to zero. This system runs right after, walks
//! every NPC with a credit waiting, calls
//! [`crate::components::NpcCharacter::record_kill`], and clears the
//! resource for the next tick.
//!
//! Why a separate system: bevy's query-aliasing rules make it
//! awkward to mutate one NPC's `NpcCharacter` while iterating the
//! shooters query that needs the same component for accuracy
//! lookup. Splitting into "produce credits in npc_combat, consume
//! in apply_kill_credits" sidesteps the conflict entirely.

use bevy_ecs::prelude::*;

use crate::components::{Npc, NpcCharacter};
use crate::resources::PendingKillCredits;

pub fn apply_kill_credits(
    mut credits: ResMut<PendingKillCredits>,
    mut npcs: Query<(&Npc, &mut NpcCharacter)>,
) {
    if credits.credits.is_empty() {
        return;
    }
    let drained = credits.drain();
    for (npc, mut character) in npcs.iter_mut() {
        if let Some(count) = drained.get(&npc.id) {
            for _ in 0..*count {
                character.record_kill();
            }
        }
    }
}
