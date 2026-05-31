//! Population top-up system, spawning in **groups** by default.
//!
//! Every `SPAWN_INTERVAL_TICKS` ticks: walk `PopulationTargets` and,
//! for each (region, faction) under-target, spawn a *squad* near a
//! same-faction base. Squad size depends on the faction (Coalition patrols
//! 4–6, Looter gangs 2–4, Nomads solo, etc.). All members share a
//! `Group` so `tick_npc_goals` can give them the same patrol target
//! — they walk together rather than dispersing.
//!
//! Lifespan rolls per-NPC across a wide range so the squad doesn't
//! all die in the same minute.

use bevy_ecs::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use crate::chronicle::{LifeChronicle, LifeRecord};
use crate::components::{
    ActiveEffects, ActiveGoal, Actor, ActorKind, Aggression, Base, BodyParts, Group, Health,
    InFaction, InRegion, Inventory, Lifespan, LimbStates, Npc, NpcCharacter, NpcGoal, Position,
    RecentAttackers, Rotation, Wounds,
};
use crate::delta::WorldDelta;

use crate::items::ItemRegistry;
use crate::npc_loadouts::NpcLoadoutRegistry;
use crate::region::RegionId;
use crate::resources::{ActiveRegions, NpcIdCounter, PendingDeltas, PopulationTargets, SimClock};

const SPAWN_INTERVAL_TICKS: u64 = 50;

/// Lifespan range in ticks (~25–66 in-game minutes at 20Hz).
const LIFESPAN_MIN_TICKS: u64 = 30_000;
const LIFESPAN_MAX_TICKS: u64 = 80_000;

/// Roll a squad size for a faction from its config-driven `squad_size`
/// range (`factions.toml`). Subfactions inherit their parent's range,
/// then a global default, via the registry. The engine no longer
/// hardcodes per-faction squad sizes.
fn squad_size_for_id(
    reg: &crate::faction::registry::FactionRegistry,
    id: crate::faction::registry::FactionId,
    rng: &mut ChaCha8Rng,
) -> usize {
    let s = reg.squad_size(id);
    let min = s.min.max(1);
    let max = s.max.max(min);
    rng.gen_range(min..=max) as usize
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_npcs(
    clock: Res<SimClock>,
    targets: Res<PopulationTargets>,
    active_regions: Res<ActiveRegions>,
    items: Res<ItemRegistry>,
    loadouts: Res<NpcLoadoutRegistry>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    names: Res<crate::names::NameRegistry>,
    bases: Query<(&Base, &InFaction, &InRegion, &Position)>,
    npcs: Query<(&Npc, &InFaction, &InRegion)>,
    mut counter: ResMut<NpcIdCounter>,
    mut chronicle: ResMut<LifeChronicle>,
    mut pending: ResMut<PendingDeltas>,
    mut commands: Commands,
) {
    let _diag_t = crate::systems::SysTimer::new("spawn_npcs");
    // Run every tick (not just every SPAWN_INTERVAL_TICKS) so we can
    // pace squad spawns globally. The old "bulk spawn every 50 ticks"
    // path spawned up to ~1280 NPCs in one tick on a cold start,
    // visible as a multi-hundred-ms hitch. With per-tick pacing we
    // spawn at most `MAX_SQUADS_PER_TICK` squads globally each tick
    // and reach target population in seconds without any spike.
    let _ = SPAWN_INTERVAL_TICKS; // retained for plan-doc references
    if clock.tick == 0 {
        return;
    }

    // Tally live populations. Keyed by registry name string so the
    // tally lines up with `PopulationTargets`.
    let mut live: HashMap<(RegionId, String), u32> = HashMap::new();
    let mut total_live: u32 = 0;
    for (_, f, r) in npcs.iter() {
        let name = registry.name_of(f.0).to_string();
        *live.entry((r.0, name)).or_default() += 1;
        total_live += 1;
    }
    // DIAGNOSTIC: print live count + target sum so we can see if
    // population is over-generating. Gated by the global verbose
    // flag — defaults off; toggle from the dev panel's World tab.
    if crate::systems::is_verbose_logging() {
        let total_target: u32 = targets.by_region.values().flat_map(|m| m.values()).sum();
        eprintln!(
            "[spawn_npcs tick={}] live={} target_sum={} per_region={:?}",
            clock.tick,
            total_live,
            total_target,
            live.iter()
                .map(|((r, f), c)| ((*r, f.as_str()), *c))
                .collect::<Vec<_>>()
        );
    }
    let _ = total_live;

    let mut rng = ChaCha8Rng::seed_from_u64(clock.tick.wrapping_mul(0x9E37_79B9_7F4A_7C15));

    // Global per-tick cap on squad spawns. The principle: every
    // entity acts as its own player making real-time decisions, so
    // we never burst-create thousands of entities on a single tick.
    // At ~4 NPCs/squad × 8 squads/tick × 20Hz = ~640 NPCs/sec
    // spawn rate. Cold-start to 3600 target NPCs takes ~6 seconds
    // of sim time, with each individual tick staying under budget.
    const MAX_SQUADS_PER_TICK: usize = 8;
    let mut squads_spawned_this_tick: usize = 0;

    // Iterate targets in a stable order. `HashMap` iteration is
    // not stable across sim instances (each map has a per-instance
    // RandomState), and `spawn_npcs` consumes a tick-seeded RNG —
    // mixing those two is a determinism trap. Sort by (region, faction
    // name) before consuming the RNG so two same-seed sims spawn the
    // same squads in the same order. Caught by `tests/determinism.rs`.
    let mut by_region_sorted: Vec<(&RegionId, &std::collections::HashMap<String, u32>)> =
        targets.by_region.iter().collect();
    by_region_sorted.sort_by_key(|(rid, _)| **rid);
    for (region_id, by_fac) in by_region_sorted {
        // Phase 1A gate: only top up populations in active regions.
        // Pre-seeded NPCs in offline regions persist; lifespan-deaths
        // there will be picked up by the offline-tier population
        // dynamics in Phase 1E. Without this gate, walking into a
        // fresh region triggers a multi-second spawn flood because
        // the per-tick budget walks every region's deficit.
        if !active_regions.is_active(*region_id) {
            continue;
        }
        // Iteration 5-14 Phase C: don't spawn squads in regions
        // that have zero `Base` entities. Scene-authored regions
        // (test_map_1..4) start empty; bases get registered when
        // the map loads via `Sim::register_authored_base`. Until
        // then spawning falls back to a random `[-100..100]` pool
        // (see `pick_spawn_pos`) which clusters every squad in a
        // 200 m radius — the O(N²) combat pass then takes the
        // sim's per-tick budget out behind the woodshed (10+ min
        // for what used to be 11 s). Skip the region until at
        // least one base exists.
        let has_base_in_region = bases.iter().any(|(_, _, r, _)| r.0 == *region_id);
        if !has_base_in_region {
            continue;
        }
        let mut by_fac_sorted: Vec<(&String, &u32)> = by_fac.iter().collect();
        by_fac_sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (faction_name, target) in by_fac_sorted {
            // Skip factions not in the active registry — modders
            // who removed a faction from the TOML would otherwise
            // strand spawns at unresolvable names.
            let Some(faction_id) = registry.id_of(faction_name) else {
                continue;
            };
            let mut current = live
                .get(&(*region_id, faction_name.clone()))
                .copied()
                .unwrap_or(0);
            let mut squads_spawned = 0usize;
            while current < *target && squads_spawned_this_tick < MAX_SQUADS_PER_TICK {
                let needed = (*target - current) as usize;
                let squad_size = squad_size_for_id(&registry, faction_id, &mut rng).min(needed);
                spawn_one_squad(
                    *region_id,
                    faction_name,
                    faction_id,
                    squad_size,
                    clock.tick,
                    &mut rng,
                    &bases,
                    &mut counter,
                    &mut chronicle,
                    &mut pending,
                    &mut commands,
                    &items,
                    &loadouts,
                    &registry,
                    &names,
                );
                current = current.saturating_add(squad_size as u32);
                squads_spawned += 1;
                squads_spawned_this_tick += 1;
            }
            if crate::systems::is_verbose_logging() && squads_spawned > 0 {
                eprintln!(
                    "[spawn_npcs tick={} region={} faction={}] spawned {} squads → current={} (target={}) tick_budget={}/{}",
                    clock.tick,
                    region_id,
                    faction_name,
                    squads_spawned,
                    current,
                    target,
                    squads_spawned_this_tick,
                    MAX_SQUADS_PER_TICK,
                );
            }
            // Early-out the outer loops once the per-tick budget is
            // exhausted — saves walking the remaining (region, faction)
            // pairs each tick for no benefit.
            if squads_spawned_this_tick >= MAX_SQUADS_PER_TICK {
                return;
            }
        }
    }
}

/// Spawn a single squad of `squad_size` NPCs of `faction_name` /
/// `faction_id` in `region_id`. Used by both `spawn_npcs` (per-tick
/// budgeted top-up) and `bulk_seed_npcs` (one-shot init). The two
/// callers differ only in their budget loop and which regions they
/// iterate; the per-NPC bundle is identical.
#[allow(clippy::too_many_arguments)]
fn spawn_one_squad(
    region_id: RegionId,
    faction_name: &str,
    faction_id: crate::faction::registry::FactionId,
    squad_size: usize,
    clock_tick: u64,
    rng: &mut ChaCha8Rng,
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
    counter: &mut NpcIdCounter,
    chronicle: &mut LifeChronicle,
    pending: &mut PendingDeltas,
    commands: &mut Commands,
    items: &ItemRegistry,
    loadouts: &NpcLoadoutRegistry,
    registry: &crate::faction::registry::FactionRegistry,
    names: &crate::names::NameRegistry,
) {
    // Pick one base for the whole squad to anchor on.
    let anchor = pick_spawn_pos(rng, faction_id, region_id, bases);
    // Stable group id derived from this spawn pass + region.
    let group_id = clock_tick
        .wrapping_mul(31)
        .wrapping_add(region_id as u64)
        .wrapping_add(rng.gen::<u64>());
    let squad_form_up_ticks = rng.gen_range(60..=120u64);
    let squad_idle_until = clock_tick.wrapping_add(squad_form_up_ticks);

    for _ in 0..squad_size {
        let id = counter.mint();
        let lifespan_ticks = rng.gen_range(LIFESPAN_MIN_TICKS..=LIFESPAN_MAX_TICKS);
        let die_at_tick = clock_tick.wrapping_add(lifespan_ticks);
        let yaw: f32 = rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
        // Spawn jitter so a squad's members start dispersed around
        // the anchor, not piled on the base centroid (the legacy
        // `±6 m` jitter let every squad visually pile at its
        // faction's HQ). `±25 m` puts a squad of 8 on a ~50 m
        // spread — well below the 80 m cohesion-break, so the
        // planner-`Regroup` thrash from squads spawning *past*
        // cohesion is avoided while still reading as dispersed.
        let pos = [
            anchor[0] + rng.gen_range(-25.0..25.0),
            0.0,
            anchor[2] + rng.gen_range(-25.0..25.0),
        ];

        // Per-NPC aggression: registry baseline ± 0.15
        // jitter, clamped to [0, 1].
        let agg_base = registry.def(faction_id).base_aggression;
        let agg = (agg_base + rng.gen_range(-0.15..=0.15)).clamp(0.0, 1.0);

        // NpcCharacter — deterministic identity + stat block +
        // personality from `(npc_id, faction_id, archetype)`. Re-rolled
        // identically on snapshot reload, no inline persistence
        // required.
        let def = registry.def(faction_id);
        let archetype = def.archetype;
        let nat_weights = def.nationality_weights.clone();
        let male_name_weight = def.male_name_weight;
        let character = NpcCharacter::roll(
            id,
            faction_id,
            archetype,
            agg_base,
            names,
            &nat_weights,
            male_name_weight,
        );

        let loadout_grid = loadouts.build_inventory(faction_name, items, rng);
        let bundle = (
            Npc { id },
            Actor {
                kind: ActorKind::Npc,
            },
            InFaction(faction_id),
            InRegion(region_id),
            Position(pos),
            Rotation(yaw),
            Health::new_full(),
            BodyParts::new_full(),
            LimbStates::default(),
            Wounds::default(),
            ActiveEffects::default(),
            NpcGoal::Idle {
                until_tick: squad_idle_until,
            },
            Lifespan {
                spawned_tick: clock_tick,
                die_at_tick,
            },
            Aggression(agg),
            RecentAttackers::default(),
        );
        let inv = Inventory(loadout_grid);
        let active_goal = ActiveGoal::default();
        // Every spawned NPC gets a Group — multi-member squads
        // share the same id, solos get a unique synthetic id
        // (derived from their NpcId) so they participate in
        // squad_planner like a 1-member squad. Without this,
        // Nomads / Merged / any solo-by-faction spawn was
        // left out of objective rolling entirely and just idled
        // at spawn until a blackboard signal pulled them in.
        let resolved_group_id = if squad_size > 1 {
            group_id
        } else {
            // Solo synthetic id with the high bit set so it can't
            // collide with the tick-derived multi-member ids
            // (`spawn_squad` builds those from `clock_tick.wrapping_shl(8) | …`).
            id.0 | (1u64 << 63)
        };
        commands.spawn((
            bundle,
            inv,
            active_goal,
            character,
            Group {
                id: resolved_group_id,
            },
        ));

        chronicle.insert(LifeRecord {
            id,
            faction: faction_name.to_string(),
            birth_tick: clock_tick,
            birth_region: region_id,
            birth_pos: pos,
            death_tick: None,
            death_region: None,
            death_cause: None,
            regions_visited: vec![(region_id, clock_tick)],
        });

        pending.push(WorldDelta::NpcSpawned {
            id,
            faction: faction_name.to_string(),
            region: region_id,
            pos,
            yaw,
            die_at_tick,
        });
    }
}

/// One-shot: fill every region's `PopulationTargets` in a single
/// pass, bypassing the per-tick squad budget that paces incremental
/// spawning. Called once at world init from
/// `Sim::initial_bulk_seed_npcs`. Without this, a fresh sim starts
/// with zero NPCs and the budget loop in `spawn_npcs` takes seconds
/// of wall-clock time per region to fill — visible as a spawn flood
/// the first time a player walks into each region.
///
/// Determinism: seeded from a stable mix of region + faction id so
/// two same-seed worlds bulk-seed identically. The world has no
/// `SimClock::tick` to key off (this runs at tick 0), so we use the
/// seed-from-world-content path that `seed_random_world_content`
/// already established for reproducibility.
#[allow(clippy::too_many_arguments)]
pub fn bulk_seed_npcs(
    targets: Res<PopulationTargets>,
    items: Res<ItemRegistry>,
    loadouts: Res<NpcLoadoutRegistry>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    names: Res<crate::names::NameRegistry>,
    bases: Query<(&Base, &InFaction, &InRegion, &Position)>,
    mut counter: ResMut<NpcIdCounter>,
    mut chronicle: ResMut<LifeChronicle>,
    mut pending: ResMut<PendingDeltas>,
    mut commands: Commands,
) {
    let _diag_t = crate::systems::SysTimer::new("bulk_seed_npcs");

    // Sorted iteration for cross-instance determinism (HashMap
    // iteration order is per-instance). Same sort key as
    // `spawn_npcs`.
    let mut by_region_sorted: Vec<(&RegionId, &std::collections::HashMap<String, u32>)> =
        targets.by_region.iter().collect();
    by_region_sorted.sort_by_key(|(rid, _)| **rid);

    let mut total_spawned: u32 = 0;
    for (region_id, by_fac) in by_region_sorted {
        // Iteration 5-14 Phase C: skip regions with zero bases.
        // Same rationale as the gate in `spawn_npcs` above —
        // without bases the spawner clusters every NPC at origin
        // via the `pick_spawn_pos` fallback, and the resulting
        // dogpile blows the per-tick budget.
        let has_base_in_region = bases.iter().any(|(_, _, r, _)| r.0 == *region_id);
        if !has_base_in_region {
            continue;
        }
        let mut by_fac_sorted: Vec<(&String, &u32)> = by_fac.iter().collect();
        by_fac_sorted.sort_by(|a, b| a.0.cmp(b.0));
        for (faction_name, target) in by_fac_sorted {
            let Some(faction_id) = registry.id_of(faction_name) else {
                continue;
            };
            // Per-(region, faction) RNG so spawn order within one
            // pair is deterministic regardless of the order we walked
            // earlier pairs. Mirrors the determinism contract laid
            // out in `world_seed.rs`.
            let mut rng = ChaCha8Rng::seed_from_u64(
                (*region_id as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(faction_id.0 as u64),
            );
            let mut current: u32 = 0;
            while current < *target {
                let needed = (*target - current) as usize;
                let squad_size = squad_size_for_id(&registry, faction_id, &mut rng).min(needed);
                spawn_one_squad(
                    *region_id,
                    faction_name,
                    faction_id,
                    squad_size,
                    0, // clock_tick — world is at tick 0 during bulk seed
                    &mut rng,
                    &bases,
                    &mut counter,
                    &mut chronicle,
                    &mut pending,
                    &mut commands,
                    &items,
                    &loadouts,
                    &registry,
                    &names,
                );
                current = current.saturating_add(squad_size as u32);
                total_spawned = total_spawned.saturating_add(squad_size as u32);
            }
        }
    }

    tracing::info!(
        "bulk_seed_npcs: spawned {} NPCs across all regions",
        total_spawned
    );
}

fn pick_spawn_pos(
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
    } else {
        return [
            rng.gen_range(-1500.0..1500.0),
            0.0,
            rng.gen_range(-1500.0..1500.0),
        ];
    };
    let base = pool[rng.gen_range(0..pool.len())];
    // Spread spawns around the base instead of stacking at its
    // exact position. 50-150m jitter so squads start dispersed.
    let jitter = 50.0 + rng.gen::<f32>() * 100.0;
    let angle = rng.gen::<f32>() * std::f32::consts::TAU;
    [
        base[0] + angle.cos() * jitter,
        base[1],
        base[2] + angle.sin() * jitter,
    ]
}
