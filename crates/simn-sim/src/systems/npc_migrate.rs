//! Rare per-tick chance for an NPC to migrate to a graph-neighbor
//! region. When it fires we update `InRegion`, drop the NPC roughly
//! near a same-faction base in the new region (or origin if none),
//! append a `regions_visited` entry to the chronicle, and queue a
//! `WorldDelta::NpcChangeRegion` for the journal.

use bevy_ecs::prelude::*;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::chronicle::LifeChronicle;
use crate::components::{Aggro, Base, InFaction, InRegion, Npc, Position};
use crate::delta::WorldDelta;
use crate::region::{RegionGraph, RegionId};
use crate::resources::{PendingDeltas, SimClock};

/// Inverse-probability per tick. 1 / 5000 at 20Hz ≈ one migration
/// every ~250 seconds (~4 minutes) of real time per NPC.
const MIGRATION_INV_PROB: u32 = 5000;

type MigrateRow<'a> = (
    Entity,
    &'a Npc,
    &'a InFaction,
    Mut<'a, InRegion>,
    Mut<'a, Position>,
    Option<&'a Aggro>,
);

pub fn migrate_npcs(
    clock: Res<SimClock>,
    graph: Res<RegionGraph>,
    bases: Query<(&Base, &InFaction, &InRegion, &Position)>,
    mut npcs: Query<MigrateRow, Without<Base>>,
    mut chronicle: ResMut<LifeChronicle>,
    mut pending: ResMut<PendingDeltas>,
) {
    let now = clock.tick;
    for (entity, npc, faction, mut region, mut pos, aggro) in npcs.iter_mut() {
        // Don't randomly leave a region in the middle of a firefight.
        if aggro.is_some() {
            continue;
        }
        // Determinism: see `npc_goals.rs` solo-FSM comment. Drop
        // `entity.to_bits()` from the seed - it isn't stable across
        // sim instances - and rely on `npc.id` + `now` for variance.
        let _ = entity;
        let mut rng = ChaCha8Rng::seed_from_u64(
            npc.id.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ now.wrapping_mul(7919),
        );
        if !rng.gen_ratio(1, MIGRATION_INV_PROB) {
            continue;
        }
        let Some(reg) = graph.get(region.0) else {
            continue;
        };
        if reg.neighbors.is_empty() {
            continue;
        }
        let new_region: RegionId = *reg.neighbors.choose(&mut rng).expect("non-empty");
        let new_pos = pick_landing_pos(&mut rng, faction.0, new_region, &bases);

        region.0 = new_region;
        pos.0 = new_pos;

        if let Some(rec) = chronicle.records.get_mut(&npc.id) {
            rec.regions_visited.push((new_region, now));
        }

        pending.push(WorldDelta::NpcChangeRegion {
            id: npc.id,
            region: new_region,
            pos: new_pos,
        });
    }
}

fn pick_landing_pos(
    rng: &mut ChaCha8Rng,
    faction: crate::faction::registry::FactionId,
    region: RegionId,
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
) -> [f32; 3] {
    let mut same_faction: Vec<[f32; 3]> = Vec::new();
    let mut any: Vec<[f32; 3]> = Vec::new();
    for (_, f, r, p) in bases.iter() {
        if r.0 != region {
            continue;
        }
        any.push(p.0);
        if f.0 == faction {
            same_faction.push(p.0);
        }
    }
    let pool = if !same_faction.is_empty() {
        same_faction
    } else if !any.is_empty() {
        any
    } else {
        return [0.0, 0.0, 0.0];
    };
    let base = pool[rng.gen_range(0..pool.len())];
    [
        base[0] + rng.gen_range(-15.0..15.0),
        0.0,
        base[2] + rng.gen_range(-15.0..15.0),
    ]
}
