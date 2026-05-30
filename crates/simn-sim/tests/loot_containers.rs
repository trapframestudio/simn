//! Phase 3A — loot container registry + world-seed scatter.
//!
//! Covers:
//! - TOML parse via `LootContainerRegistry::load()` finds the three
//!   shipped kinds.
//! - `weighted_pick` returns kinds in proportion to `spawn_weight`
//!   over a large sample (sanity check the picker isn't constant).
//! - `Sim::new_in_memory_with_seed` produces 8-15 containers per
//!   seeded region — empty grids, private (not kit-pool), positioned
//!   close to authored bases.

use simn_sim::loot_containers::LootContainerRegistry;
use simn_sim::region::{Region, RegionGraph};
use simn_sim::Sim;

/// Build a fresh on-disk-less sim with a graph that exercises the
/// procedural container scatter. Mirrors `default_test_graph`'s
/// map_a but flips `scene_authored_pois = false` so Phase C's gate
/// doesn't elide the loot scatter (it anchors to procedurally-
/// seeded bases). Seed is fixed so the assertions can be exact.
fn fresh_sim(seed: u64) -> Sim {
    Sim::new_in_memory_with_seed(legacy_procedural_graph(), seed)
}

fn legacy_procedural_graph() -> RegionGraph {
    // Same shape as `default_test_graph`'s 4-region 2×2 layout but
    // with `scene_authored_pois = false` on every region. Keeps the
    // pre-iteration-5-14 procedural scatter path exercised
    // independent of the test-map authoring rework.
    let mut g = RegionGraph::new();
    for (id, name, scene) in [
        (1, "map_a", "res://scenes/test/test_map_1.tscn"),
        (2, "map_b", "res://scenes/test/test_map_2.tscn"),
        (3, "map_c", "res://scenes/test/test_map_3.tscn"),
        (4, "map_d", "res://scenes/test/test_map_4.tscn"),
    ] {
        g.insert(Region {
            id,
            name: name.into(),
            map_scene: scene.into(),
            neighbors: vec![],
            transitions: Default::default(),
            procedurally_seeded: true,
            scene_authored_pois: false,
        });
    }
    g
}

#[test]
fn registry_loads_the_three_default_kinds() {
    let r = LootContainerRegistry::load();
    assert!(
        !r.is_empty(),
        "loot_containers.toml should have at least one kind"
    );
    // The three shipped kinds must be present. If we rename / add
    // entries, update this list — it's the contract for downstream
    // code (3B pool tables key on these ids).
    let ids: Vec<&str> = r.iter().map(|d| d.id.as_str()).collect();
    for required in ["small_crate", "medium_stash", "large_cache"] {
        assert!(
            ids.contains(&required),
            "missing required kind {required} in {ids:?}",
        );
    }
    // Grid dimensions must be > 0 — a zero grid would mean
    // `WorldContainer::new` produces a 0-capacity container.
    for d in r.iter() {
        assert!(d.grid.w > 0 && d.grid.h > 0, "kind {} has zero grid", d.id);
    }
}

#[test]
fn weighted_pick_is_not_constant() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let r = LootContainerRegistry::load();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for _ in 0..1000 {
        let kind = r.weighted_pick(&mut rng).expect("non-empty registry");
        *counts.entry(kind.id.clone()).or_default() += 1;
    }
    // At least two distinct kinds should appear over 1000 rolls —
    // catches a regression where the picker collapses to a single
    // entry.
    assert!(
        counts.len() >= 2,
        "weighted_pick produced only {} distinct kind(s): {:?}",
        counts.len(),
        counts,
    );
    // `small_crate` (weight 70 in the shipped TOML) should dominate.
    let small = counts.get("small_crate").copied().unwrap_or(0);
    assert!(
        small > 500,
        "expected `small_crate` to dominate (>500/1000); got {small}",
    );
}

#[test]
fn world_seed_scatters_containers_near_bases() {
    let mut sim = fresh_sim(7);

    let containers = sim.all_world_containers_for_test();
    let bases = sim.all_bases_for_test();
    assert!(
        !containers.is_empty(),
        "no containers spawned at all — seed_random_world_content didn't run scatter",
    );

    // Per-region container count must be in the seeded range.
    let mut per_region: std::collections::BTreeMap<_, usize> = std::collections::BTreeMap::new();
    for (_, region, _, _, _) in &containers {
        *per_region.entry(*region).or_default() += 1;
    }
    for (region, count) in &per_region {
        assert!(
            (8..=15).contains(count),
            "region {region:?} has {count} containers; expected 8-15",
        );
    }

    // Index bases by region so the proximity check is O(n).
    let mut bases_by_region: std::collections::BTreeMap<_, Vec<[f32; 3]>> =
        std::collections::BTreeMap::new();
    for (region, pos) in bases {
        bases_by_region.entry(region).or_default().push(pos);
    }

    // Authored radius is 80 m + base placement scatter; allow a
    // generous 150 m so the test stays resilient to minor balance
    // tweaks.
    const NEAR_BASE_RADIUS_M: f32 = 150.0;
    for (_, region, pos, _, _) in &containers {
        let region_bases = bases_by_region
            .get(region)
            .expect("every region with containers should have bases");
        let near = region_bases.iter().any(|b| {
            let dx = b[0] - pos[0];
            let dz = b[2] - pos[2];
            (dx * dx + dz * dz).sqrt() <= NEAR_BASE_RADIUS_M
        });
        assert!(
            near,
            "container at {pos:?} in region {region:?} is >{NEAR_BASE_RADIUS_M}m from every base",
        );
    }
}

#[test]
fn scattered_containers_default_private() {
    // Phase 3C amended the original "start empty" contract:
    // containers now eager-roll initial contents at world-gen
    // (so a fresh save has loot to find before the player ever
    // picks anything up). The "private by default" half of the
    // contract still holds — public-flagged crates are
    // authored-only.
    let mut sim = fresh_sim(11);
    let containers = sim.all_world_containers_for_test();
    assert!(!containers.is_empty(), "no containers spawned");
    for (id, _, _, is_public, _items) in containers {
        assert!(
            !is_public,
            "container {id:?} should default private (not kit-pool)",
        );
    }
}

#[test]
fn scatter_is_deterministic_given_same_seed() {
    let mut sim_a = fresh_sim(123);
    let mut sim_b = fresh_sim(123);

    let positions = |s: &mut Sim| -> Vec<[i32; 3]> {
        let mut out: Vec<[i32; 3]> = s
            .all_world_containers_for_test()
            .into_iter()
            // Round to mm so float equality doesn't bite.
            .map(|(_, _, p, _, _)| {
                [
                    (p[0] * 1000.0).round() as i32,
                    (p[1] * 1000.0).round() as i32,
                    (p[2] * 1000.0).round() as i32,
                ]
            })
            .collect();
        out.sort_unstable();
        out
    };
    let a = positions(&mut sim_a);
    let b = positions(&mut sim_b);
    assert_eq!(a, b, "same seed must produce identical container scatter",);
}
