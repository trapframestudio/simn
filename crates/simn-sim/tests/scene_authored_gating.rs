//! Iteration 5-14 Phase C: tests for the `scene_authored_pois` gate.
//!
//! When a region has `procedurally_seeded = true` *AND*
//! `scene_authored_pois = true`, the world seeder still sets up
//! `RegionControl` + `PopulationTargets` (so the squad planner and
//! NPC spawner have something to work with) but does NOT scatter
//! random bases/camps — those come from the GDScript-side
//! `Sim::register_authored_base` calls instead.

use simn_sim::components::{Base, InRegion};
use simn_sim::region::{Region, RegionGraph, RegionId};
use simn_sim::Sim;

/// Builds a one-region graph with both flags true. Mirrors the
/// `default_test_graph` map_a..d shape from `region.rs::default_test_graph`.
fn one_scene_authored_region() -> RegionGraph {
    let mut g = RegionGraph::new();
    g.insert(Region {
        id: 1,
        name: "map_a".into(),
        map_scene: "res://scenes/test/test_map_1.tscn".into(),
        neighbors: vec![],
        transitions: Default::default(),
        procedurally_seeded: true,
        scene_authored_pois: true,
    });
    g
}

fn one_procedural_region() -> RegionGraph {
    let mut g = RegionGraph::new();
    g.insert(Region {
        id: 1,
        name: "map_a".into(),
        map_scene: "res://scenes/test/test_map_1.tscn".into(),
        neighbors: vec![],
        transitions: Default::default(),
        procedurally_seeded: true,
        scene_authored_pois: false,
    });
    g
}

const TEST_REGION_ID: RegionId = 1;

#[test]
fn scene_authored_region_has_no_seeded_bases() {
    let mut sim = Sim::new_in_memory(one_scene_authored_region());
    let world = sim.world_for_test();
    let mut q = world.query::<(&Base, &InRegion)>();
    let count = q.iter(world).filter(|(_, r)| r.0 == TEST_REGION_ID).count();
    assert_eq!(
        count, 0,
        "scene-authored region must have zero procedurally-seeded bases",
    );
}

#[test]
fn procedural_region_still_seeds_bases() {
    let mut sim = Sim::new_in_memory(one_procedural_region());
    let world = sim.world_for_test();
    let mut q = world.query::<(&Base, &InRegion)>();
    let count = q.iter(world).filter(|(_, r)| r.0 == TEST_REGION_ID).count();
    assert!(
        count >= 25,
        "legacy procedural region must still scatter bases (>=25), got {count}",
    );
}

#[test]
fn scene_authored_region_has_population_targets() {
    // `new_in_memory` clears `PopulationTargets` post-seed so test
    // sims don't auto-spawn thousands of NPCs. Use the path-backed
    // constructor for this assertion — it preserves whatever the
    // seeder set up.
    let dir = tempfile::TempDir::new().unwrap();
    let paths = simn_sim::SavePaths::in_dir(dir.path());
    let sim = Sim::new(paths, one_scene_authored_region()).unwrap();
    let targets = sim.population_targets();
    let region_targets = targets
        .by_region
        .get(&TEST_REGION_ID)
        .expect("scene-authored region must have PopulationTargets seeded");
    assert!(
        !region_targets.is_empty(),
        "population targets must contain at least one faction allocation",
    );
}

#[test]
fn scene_authored_region_has_region_control() {
    let sim = Sim::new_in_memory(one_scene_authored_region());
    let controls = sim.region_controls();
    let state = controls
        .get(&TEST_REGION_ID)
        .expect("scene-authored region must have RegionControl seeded");
    assert!(
        state.primary.is_some(),
        "RegionControl must pick a primary faction even on scene-authored",
    );
}
