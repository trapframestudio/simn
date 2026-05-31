//! Deterministic random world content seeder.
//!
//! Populates `RegionControl` and spawns a handful of `Base` entities
//! per region using weighted-random faction distribution. Used at
//! sim-init when there's no authored map content; replaced wholesale
//! when the real region map lands.
//!
//! Uses `rand_chacha::ChaCha8Rng` so the same seed produces the same
//! world on every platform / Rust version. `Sim::new` calls this
//! once; `Sim::load` does not (snapshots already contain the seeded
//! state).

use bevy_ecs::prelude::World;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Half-extent of the test maps in meters. Bases place inside a
/// slightly smaller box so they don't sit on the edge.
pub const TEST_MAP_HALF_EXTENT_M: f32 = 2500.0;
pub const BASE_PLACEMENT_HALF_EXTENT_M: f32 = 2300.0;

use crate::components::{
    Base, BaseKind, GridInventory, Health, InFaction, InRegion, Position, WorldContainer,
};
use crate::loot_containers::{LootContainerDef, LootContainerRegistry};
use crate::loot_pools::LootPoolRegistry;
use crate::region::{RegionGraph, RegionId};
use crate::resources::{ContainerIdCounter, PopulationTargets, RegionControl, RegionControlState};

/// Default seed for fresh sims when no caller-supplied seed is given.
/// Deliberately fixed so dev demos are reproducible; production
/// servers can pass a real seed.
pub const DEFAULT_SEED: u64 = 1;

pub fn seed_random_world_content(world: &mut World, graph: &RegionGraph, seed: u64) {
    tracing::info!(
        "world_seed: stratified placement on 7x7 grid, ±{}m, seed={}",
        BASE_PLACEMENT_HALF_EXTENT_M,
        seed
    );
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Sort region IDs so iteration is deterministic regardless of
    // the underlying HashMap order. Real DEM-backed maps (corbett,
    // latourell, …) are excluded here — they opt in via the
    // `procedurally_seeded` flag on `Region`, which defaults to
    // false. That keeps procedural factions/bases/NPCs quarantined
    // to the synthetic test maps until hand-authored content lands
    // per DESIGN.md §3.4+.
    //
    // Iteration 5-14 Phase C: regions with `scene_authored_pois ==
    // true` keep iterating here (so `RegionControl` +
    // `PopulationTargets` still seed below) but the base/camp
    // scatter inside the loop is gated off — the POI baker +
    // `Sim::register_authored_base` populates them from the scene
    // tree instead.
    let mut region_ids: Vec<RegionId> = graph
        .regions
        .iter()
        .filter(|(_, r)| r.procedurally_seeded)
        .map(|(id, _)| *id)
        .collect();
    region_ids.sort_unstable();

    let mut control = RegionControl::default();
    // (region, faction-name, base-kind, position).
    let mut bases: Vec<(RegionId, String, BaseKind, Position)> = Vec::new();

    // 7×7 stratified grid for base placement. Each cell is
    // 2*BASE_PLACEMENT_HALF_EXTENT_M / 7 ≈ 657m wide. Picking from
    // distinct cells per region prevents the visible clumping of
    // pure-uniform sampling and gives roughly one POI per 0.4 km²
    // — dense enough to be useful as NPC traversal anchors when
    // they land.
    const STRATA: i32 = 7;
    let cell_size = (2.0 * BASE_PLACEMENT_HALF_EXTENT_M) / STRATA as f32;

    for region_id in region_ids {
        let primary = pick_primary_faction(&mut rng).to_string();
        let mut state = RegionControlState::uncontested(&primary);

        // TEMP (evaluation): every region is contested, with 2–4
        // additional factions present. Guarantees aggro encounters
        // on every map. Real contest rates come out of the brain
        // layer later.
        let extras = rng.gen_range(2..=4usize);
        for _ in 0..extras {
            let challenger = pick_primary_faction(&mut rng).to_string();
            if challenger != primary && !state.contested_by.iter().any(|c| c == &challenger) {
                state.contested_by.push(challenger);
            }
        }
        state.tension = 0.4 + 0.2 * state.contested_by.len() as f32;
        control.by_region.insert(region_id, state.clone());

        // Iteration 5-14 Phase C: scene-authored regions skip the
        // base + camp scatter. The POI baker (Phase D) emits scene-
        // tree `PoiMarker3D` nodes; the GDScript spawner (Phase E)
        // walks them on map load and calls
        // `Sim::register_authored_base`. RegionControl +
        // PopulationTargets (above + below) still seed normally.
        let region_is_scene_authored = graph
            .regions
            .get(&region_id)
            .map(|r| r.scene_authored_pois)
            .unwrap_or(false);
        if region_is_scene_authored {
            continue;
        }

        let base_count = rng.gen_range(25..=40usize);
        let mut available_cells: Vec<(i32, i32)> = (0..STRATA)
            .flat_map(|x| (0..STRATA).map(move |z| (x, z)))
            .collect();
        available_cells.shuffle(&mut rng);
        for &(cx, cz) in available_cells.iter().take(base_count) {
            let owner: String = if state.contested_by.is_empty() || rng.gen_bool(0.70) {
                primary.clone()
            } else {
                state
                    .contested_by
                    .choose(&mut rng)
                    .cloned()
                    .expect("non-empty contested_by")
            };
            let owner_id = world
                .resource::<crate::faction::registry::FactionRegistry>()
                .id_of(&owner);
            let kind = match owner_id {
                Some(id) => pick_base_kind_for_id(
                    &mut rng,
                    world.resource::<crate::faction::registry::FactionRegistry>(),
                    id,
                ),
                None => BaseKind::Outpost,
            };
            let cell_x0 = -BASE_PLACEMENT_HALF_EXTENT_M + cx as f32 * cell_size;
            let cell_z0 = -BASE_PLACEMENT_HALF_EXTENT_M + cz as f32 * cell_size;
            let inset = cell_size * 0.10;
            let pos = Position([
                rng.gen_range((cell_x0 + inset)..(cell_x0 + cell_size - inset)),
                0.0,
                rng.gen_range((cell_z0 + inset)..(cell_z0 + cell_size - inset)),
            ]);
            bases.push((region_id, owner, kind, pos));
        }

        // Neutral, non-contestable campsites: stored under
        // `nomads` as a placeholder neutral owner; `RegionControl`
        // doesn't count them so they stay off the contest map.
        let campsite_count = rng.gen_range(4..=7usize);
        for &(cx, cz) in available_cells.iter().skip(base_count).take(campsite_count) {
            let cell_x0 = -BASE_PLACEMENT_HALF_EXTENT_M + cx as f32 * cell_size;
            let cell_z0 = -BASE_PLACEMENT_HALF_EXTENT_M + cz as f32 * cell_size;
            let inset = cell_size * 0.10;
            let pos = Position([
                rng.gen_range((cell_x0 + inset)..(cell_x0 + cell_size - inset)),
                0.0,
                rng.gen_range((cell_z0 + inset)..(cell_z0 + cell_size - inset)),
            ]);
            bases.push((region_id, "nomads".to_string(), BaseKind::CampSite, pos));
        }
    }

    // Seed initial population targets per region, derived from
    // who controls it. Primary faction gets a healthy live count,
    // each contesting faction a smaller presence. Tuned for
    // playable framerate at the current iteration's online-tier
    // cost. Bigger pop comes back once distance-tier projection
    // (online cone gated by player draw radius) lands — that
    // un-gates the architectural ceiling on `npc_aggro` etc.
    let mut targets = PopulationTargets::default();
    for (region_id, state) in &control.by_region {
        if let Some(primary) = &state.primary {
            targets.set(*region_id, primary, 80);
        }
        for fac in &state.contested_by {
            targets.set(*region_id, fac, 40);
        }
    }
    world.insert_resource(targets);

    world.insert_resource(control);

    // Track placed-base positions per region for the loot-scatter
    // and activity-point passes below.
    let mut base_positions_by_region: std::collections::BTreeMap<RegionId, Vec<[f32; 3]>> =
        std::collections::BTreeMap::new();
    let mut base_metadata_by_region: BaseMetadata = std::collections::BTreeMap::new();
    for (region_id, owner, kind, pos) in bases {
        let Some(owner_id) = world
            .resource::<crate::faction::registry::FactionRegistry>()
            .id_of(&owner)
        else {
            continue;
        };
        base_positions_by_region
            .entry(region_id)
            .or_default()
            .push(pos.0);
        let is_campsite = matches!(kind, BaseKind::CampSite);
        base_metadata_by_region
            .entry(region_id)
            .or_default()
            .push((pos.0, owner_id, is_campsite));
        world.spawn((
            Base { kind },
            InFaction(owner_id),
            InRegion(region_id),
            pos,
            Health::new_full(),
        ));
    }

    // Scatter activity points + cover volumes so procedural maps
    // have the same behavioral hooks scene-authored maps get.
    seed_activity_points_and_cover(world, &mut rng, &base_metadata_by_region);

    // Phase 3A: scatter loot containers per region. Per-region
    // count is 8-15; each container is placed near a random
    // existing base (within `LOOT_BASE_RADIUS_M`) so the world
    // tells a story — caches cluster around faction strongholds
    // — rather than the engine sprinkling them on flat plains.
    // Kind selection is weighted by `spawn_weight` from the TOML
    // registry. Containers start empty; Phase 3B fills them.
    seed_loot_containers(world, &mut rng, &base_positions_by_region);
}

type BaseMetadata = std::collections::BTreeMap<
    RegionId,
    Vec<([f32; 3], crate::faction::registry::FactionId, bool)>,
>;

fn seed_activity_points_and_cover(
    world: &mut World,
    rng: &mut ChaCha8Rng,
    base_metadata: &BaseMetadata,
) {
    use crate::cover::{CoverHeight, CoverMaterialId, CoverVolume};
    use crate::resources::{ActivityKind, ActivityPoint};

    let mut ap_count = 0usize;
    let mut cover_count = 0usize;
    let mut route_count = 0usize;

    {
        let mut aps = world.resource_mut::<crate::resources::ActivityPoints>();

        for (&region_id, bases) in base_metadata.iter() {
            // -- Guard + rest APs near each base/campsite --
            for &(pos, faction_id, is_campsite) in bases {
                if is_campsite {
                    // Campfire AP at the campsite itself.
                    let id = aps.next_id();
                    aps.by_region
                        .entry(region_id)
                        .or_default()
                        .push(ActivityPoint {
                            id,
                            kind: ActivityKind::Campfire,
                            pos,
                            facing_yaw: 0.0,
                            faction: None,
                            radius_m: 15.0,
                            capacity: 6,
                            priority: 8,
                            loop_id: None,
                            occupants: Vec::new(),
                            claimed_by_groups: Vec::new(),
                        });
                    ap_count += 1;
                    // RestSpot offset from the campfire.
                    let id = aps.next_id();
                    let dx = rng.gen_range(-15.0_f32..15.0);
                    let dz = rng.gen_range(-15.0_f32..15.0);
                    aps.by_region
                        .entry(region_id)
                        .or_default()
                        .push(ActivityPoint {
                            id,
                            kind: ActivityKind::RestSpot,
                            pos: [pos[0] + dx, pos[1], pos[2] + dz],
                            facing_yaw: 0.0,
                            faction: None,
                            radius_m: 10.0,
                            capacity: 4,
                            priority: 6,
                            loop_id: None,
                            occupants: Vec::new(),
                            claimed_by_groups: Vec::new(),
                        });
                    ap_count += 1;
                } else {
                    // Faction base: 2-3 guard points around the perimeter.
                    let guard_count = rng.gen_range(2..=3u32);
                    for i in 0..guard_count {
                        let angle = (i as f32 / guard_count as f32) * std::f32::consts::TAU
                            + rng.gen_range(-0.3_f32..0.3);
                        let dist = rng.gen_range(15.0_f32..40.0);
                        let gx = pos[0] + angle.cos() * dist;
                        let gz = pos[2] + angle.sin() * dist;
                        let kind = if rng.gen_bool(0.7) {
                            ActivityKind::GuardStatic
                        } else {
                            ActivityKind::GuardPerimeter
                        };
                        let id = aps.next_id();
                        aps.by_region
                            .entry(region_id)
                            .or_default()
                            .push(ActivityPoint {
                                id,
                                kind,
                                pos: [gx, 0.0, gz],
                                facing_yaw: angle + std::f32::consts::PI,
                                faction: Some(faction_id),
                                radius_m: 20.0,
                                capacity: 3,
                                priority: 10,
                                loop_id: None,
                                occupants: Vec::new(),
                                claimed_by_groups: Vec::new(),
                            });
                        ap_count += 1;
                    }
                }
            }

            // -- Lookout APs scattered across the map --
            let lookout_count = rng.gen_range(3..=5u32);
            for _ in 0..lookout_count {
                let lx = rng.gen_range(-BASE_PLACEMENT_HALF_EXTENT_M..BASE_PLACEMENT_HALF_EXTENT_M);
                let lz = rng.gen_range(-BASE_PLACEMENT_HALF_EXTENT_M..BASE_PLACEMENT_HALF_EXTENT_M);
                let id = aps.next_id();
                aps.by_region
                    .entry(region_id)
                    .or_default()
                    .push(ActivityPoint {
                        id,
                        kind: ActivityKind::Lookout,
                        pos: [lx, 0.0, lz],
                        facing_yaw: rng.gen_range(0.0_f32..std::f32::consts::TAU),
                        faction: None,
                        radius_m: 15.0,
                        capacity: 2,
                        priority: 5,
                        loop_id: None,
                        occupants: Vec::new(),
                        claimed_by_groups: Vec::new(),
                    });
                ap_count += 1;
            }

            // -- Patrol routes connecting same-faction bases --
            let mut faction_bases: std::collections::BTreeMap<
                crate::faction::registry::FactionId,
                Vec<[f32; 3]>,
            > = std::collections::BTreeMap::new();
            for &(pos, fid, is_camp) in bases {
                if !is_camp {
                    faction_bases.entry(fid).or_default().push(pos);
                }
            }
            for (fid, positions) in &faction_bases {
                if positions.len() < 2 {
                    continue;
                }
                let routes_for_faction = rng.gen_range(1..=2u32).min(positions.len() as u32 / 2);
                for r in 0..routes_for_faction {
                    let start = (r as usize * 2) % positions.len();
                    let end = (r as usize * 2 + 1) % positions.len();
                    if start == end {
                        continue;
                    }
                    let waypoints = vec![positions[start], positions[end]];
                    let route_id = format!("proc_{region_id}_{fid:?}_{r}");
                    aps.routes_by_region.entry(region_id).or_default().push(
                        crate::resources::PatrolRoute {
                            id: route_id,
                            waypoints,
                            faction: Some(*fid),
                            is_loop: true,
                            priority: 5,
                            claimed_by_group: None,
                        },
                    );
                    route_count += 1;
                }
            }
        }
    } // drop aps borrow

    {
        let mut covers = world.resource_mut::<crate::cover::CoverVolumes>();
        for (&region_id, bases) in base_metadata.iter() {
            // 2-3 cover volumes per base.
            for &(pos, _, _) in bases {
                let n = rng.gen_range(2..=3u32);
                for _ in 0..n {
                    let dx = rng.gen_range(-25.0_f32..25.0);
                    let dz = rng.gen_range(-25.0_f32..25.0);
                    let id = covers.next_id();
                    let height = if rng.gen_bool(0.6) {
                        CoverHeight::High
                    } else {
                        CoverHeight::Low
                    };
                    let (mat, thick) = if rng.gen_bool(0.5) {
                        (CoverMaterialId::Concrete, 250.0)
                    } else {
                        (CoverMaterialId::WoodThick, 120.0)
                    };
                    covers
                        .by_region
                        .entry(region_id)
                        .or_default()
                        .push(CoverVolume {
                            id,
                            region: region_id,
                            pos: [pos[0] + dx, 0.8, pos[2] + dz],
                            half_extents: [
                                rng.gen_range(0.5_f32..2.0),
                                rng.gen_range(0.5_f32..1.5),
                                rng.gen_range(0.5_f32..2.0),
                            ],
                            rotation: [0.0, 0.0, 0.0, 1.0],
                            material_id: mat,
                            height,
                            thickness_mm: thick,
                            destructible: false,
                            health: 100.0,
                            max_health: 100.0,
                        });
                    cover_count += 1;
                }
            }

            // 15-25 free-standing cover positions across the map.
            let freestanding = rng.gen_range(15..=25u32);
            for _ in 0..freestanding {
                let cx = rng.gen_range(-BASE_PLACEMENT_HALF_EXTENT_M..BASE_PLACEMENT_HALF_EXTENT_M);
                let cz = rng.gen_range(-BASE_PLACEMENT_HALF_EXTENT_M..BASE_PLACEMENT_HALF_EXTENT_M);
                let id = covers.next_id();
                let height = match rng.gen_range(0..3u32) {
                    0 => CoverHeight::Low,
                    1 => CoverHeight::High,
                    _ => CoverHeight::Full,
                };
                let (mat, thick) = match rng.gen_range(0..4u32) {
                    0 => (CoverMaterialId::Earth, 400.0),
                    1 => (CoverMaterialId::Concrete, 300.0),
                    2 => (CoverMaterialId::VehicleBody, 80.0),
                    _ => (CoverMaterialId::WoodThick, 120.0),
                };
                covers
                    .by_region
                    .entry(region_id)
                    .or_default()
                    .push(CoverVolume {
                        id,
                        region: region_id,
                        pos: [cx, 0.6, cz],
                        half_extents: [
                            rng.gen_range(0.8_f32..3.0),
                            rng.gen_range(0.4_f32..1.5),
                            rng.gen_range(0.8_f32..3.0),
                        ],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        material_id: mat,
                        height,
                        thickness_mm: thick,
                        destructible: false,
                        health: 100.0,
                        max_health: 100.0,
                    });
                cover_count += 1;
            }
        }
    } // drop covers borrow

    tracing::info!(
        "world_seed: scattered {} activity points, {} patrol routes, {} cover volumes",
        ap_count,
        route_count,
        cover_count,
    );
}

/// Half-extent of the random offset applied to a loot container
/// from its anchor base, in meters. Keeps caches inside the
/// faction's footprint but jittered enough that they're not all
/// stacked on the base centroid.
const LOOT_BASE_RADIUS_M: f32 = 80.0;

/// Per-region container count range. Tuned so a busy zone has a
/// dozen-ish containers — enough for the player to find some on
/// every map without the world feeling littered.
const LOOT_PER_REGION_MIN: usize = 8;
const LOOT_PER_REGION_MAX: usize = 15;

fn seed_loot_containers(
    world: &mut World,
    rng: &mut ChaCha8Rng,
    base_positions_by_region: &std::collections::BTreeMap<RegionId, Vec<[f32; 3]>>,
) {
    let registry = world.resource::<LootContainerRegistry>().clone();
    let pool_registry = world.resource::<LootPoolRegistry>().clone();
    let item_registry = world.resource::<crate::items::ItemRegistry>().clone();
    let region_factions: std::collections::HashMap<RegionId, String> = world
        .resource::<RegionControl>()
        .by_region
        .iter()
        .filter_map(|(rid, state)| state.primary.as_ref().map(|p| (*rid, p.clone())))
        .collect();
    if registry.is_empty() {
        // Empty TOML or load failure — log and skip rather than
        // failing sim init. Same posture as
        // `seed_random_world_content`'s mod-faction skip.
        tracing::info!("world_seed: loot_containers registry empty, skipping scatter");
        return;
    }
    let mut spawned_total = 0usize;
    let mut items_placed = 0usize;
    // Plan placements + capture each container's kind so the
    // second pass can run the initial-content roll without
    // re-resolving the registry.
    let mut planned: Vec<(RegionId, [f32; 3], LootContainerDef)> = Vec::new();
    for (region_id, bases) in base_positions_by_region.iter() {
        if bases.is_empty() {
            continue;
        }
        let count = rng.gen_range(LOOT_PER_REGION_MIN..=LOOT_PER_REGION_MAX);
        for _ in 0..count {
            // Anchor on a random base, jitter inside a small XZ
            // disk. Y stays at 0 — ground sampling happens at
            // render time (terrain isn't loaded at sim init).
            let anchor = bases[rng.gen_range(0..bases.len())];
            let dx = rng.gen_range(-LOOT_BASE_RADIUS_M..=LOOT_BASE_RADIUS_M);
            let dz = rng.gen_range(-LOOT_BASE_RADIUS_M..=LOOT_BASE_RADIUS_M);
            let pos = [anchor[0] + dx, anchor[1], anchor[2] + dz];
            let kind = registry.weighted_pick(rng).expect("registry non-empty");
            planned.push((*region_id, pos, kind.clone()));
        }
    }
    // Spawn pass — mint id, roll contents, place items, attach
    // the entity. We use the same ChaCha8Rng stream as everything
    // else in this seed pass, so identical seeds produce
    // identical container contents (within a run); new saves get
    // fresh seeds and therefore fresh loot.
    for (region_id, pos, kind) in planned {
        let id = world.resource_mut::<ContainerIdCounter>().mint();
        let mut grid = GridInventory::new(kind.grid.w, kind.grid.h);
        // Faction defaults to the region's primary controller
        // (state.primary). Falls back to "nomads" — the
        // neutral pool — if the region has no recorded primary.
        let faction = region_factions
            .get(&region_id)
            .cloned()
            .unwrap_or_else(|| "nomads".to_string());
        let depth_tier: u8 = 1; // Surface tier until zones author tiers.
        items_placed += roll_initial_container_contents(
            &mut grid,
            &kind,
            &faction,
            depth_tier,
            &pool_registry,
            &item_registry,
            rng,
        );
        let component = WorldContainer {
            id,
            grid,
            is_public: kind.is_public,
            faction: Some(faction),
            depth_tier,
            last_restock_tick: 0,
            interaction_mode: crate::components::ContainerInteractionMode::Openable,
        };
        world.spawn((component, Position(pos), InRegion(region_id)));
        spawned_total += 1;
    }
    tracing::info!(
        "world_seed: scattered {} loot containers ({} items placed) across {} regions",
        spawned_total,
        items_placed,
        base_positions_by_region.len(),
    );
}

/// Roll initial contents into `grid` for a freshly-spawned
/// container. Returns the number of items successfully placed.
///
/// Walks the kind's `items_per_roll` range and, for each slot,
/// picks a family weighted by the kind's `family_weights`, then
/// rolls one item from `(faction, depth_tier, family)`. Items
/// that don't exist in the `ItemRegistry` are silently skipped —
/// keeps pool TOML edits from breaking sim init when an entry
/// references an unknown id.
///
/// Used by both `seed_loot_containers` (initial roll at world
/// gen) and the Phase 3C restock sweep (partial top-up while
/// the world is live).
pub(crate) fn roll_initial_container_contents(
    grid: &mut GridInventory,
    kind: &LootContainerDef,
    faction: &str,
    depth_tier: u8,
    pool_registry: &LootPoolRegistry,
    item_registry: &crate::items::ItemRegistry,
    rng: &mut ChaCha8Rng,
) -> usize {
    let n = kind.roll_items_per_roll(rng) as usize;
    let mut placed = 0;
    for _ in 0..n {
        let Some(family) = kind.pick_family(rng) else {
            continue;
        };
        let Some(rolled) = pool_registry.roll_one(rng, faction, depth_tier, family) else {
            continue;
        };
        // Skip silently if the registry doesn't know the id —
        // happens when pool TOML references an item that's been
        // renamed or removed without updating the pool entry.
        if item_registry.get(&rolled.id).is_none() {
            continue;
        }
        if crate::inventory_grid::grant_or_merge(grid, item_registry, &rolled.id, rolled.count, 0)
            .is_ok()
        {
            placed += 1;
        }
    }
    placed
}

/// Weighted picker for "who runs this region." Top-level factions
/// only — Vanguard / Directorate Recon / Registry / Consortium Recovery /
/// Devout / Looters / Smugglers are subfactions and never primary
/// region controllers (they spawn within their parent's territory).
/// `merged` is excluded — endgame, single fixed site.
fn pick_primary_faction(rng: &mut ChaCha8Rng) -> &'static str {
    const WEIGHTED: &[(&str, u32)] = &[
        ("coalition", 30),
        ("directorate", 12),
        ("the_order", 12),
        ("raiders", 14),
        ("homesteaders", 9),
        ("consortium", 12),
        ("syndicate", 4),
        ("nomads", 5),
    ];
    let total: u32 = WEIGHTED.iter().map(|(_, w)| *w).sum();
    let mut roll = rng.gen_range(0..total);
    for (name, w) in WEIGHTED {
        if roll < *w {
            return name;
        }
        roll -= *w;
    }
    "nomads" // unreachable in practice
}

/// Pick a base kind, weighted by the owner faction's config-driven
/// `base_kinds` (factions.toml), so the world reads right. Subfactions
/// inherit their parent's weights, then a generic default, via the
/// registry. The engine no longer hardcodes per-faction base-kind mixes.
fn pick_base_kind_for_id(
    rng: &mut ChaCha8Rng,
    reg: &crate::faction::registry::FactionRegistry,
    owner: crate::faction::registry::FactionId,
) -> BaseKind {
    let weights = reg.base_kind_weights(owner);
    let total: u32 = weights.iter().map(|(_, w)| *w).sum();
    if total == 0 {
        return BaseKind::Outpost;
    }
    let mut roll = rng.gen_range(0..total);
    for (k, w) in &weights {
        if roll < *w {
            return *k;
        }
        roll -= *w;
    }
    BaseKind::Outpost
}
