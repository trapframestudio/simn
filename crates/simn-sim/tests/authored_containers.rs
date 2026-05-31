//! Phase 3D — `Sim::register_authored_container` integration tests.
//!
//! Coverage:
//! - Registers a small / medium / large kind and verifies grid
//!   dimensions match the catalog.
//! - Eager rolls fill the grid with items (most rolls produce
//!   non-zero items on the shipped pools).
//! - `interaction_mode = Breakable` survives onto the spawned
//!   component.
//! - Same seed → same contents; different seeds → different
//!   contents (the per-marker deterministic property the GDScript
//!   walker relies on).
//! - Unknown kind id → error, no container spawned.

use simn_sim::components::{ContainerInteractionMode, GridInventory, WorldContainer};
use simn_sim::region::RegionGraph;
use simn_sim::Sim;

fn fresh_sim(seed: u64) -> Sim {
    Sim::new_in_memory_with_seed(RegionGraph::default_test_graph(), seed)
}

fn first_region(_sim: &mut Sim) -> simn_sim::region::RegionId {
    // Iteration 5-14 Phase C: test maps no longer auto-seed
    // procedural loot containers (the scatter anchors to bases
    // which the new `scene_authored_pois` gate skips). Pick a
    // known-good region id directly — `default_test_graph` always
    // has map_a as region id 1.
    1
}

#[test]
fn registers_small_crate_with_correct_grid() {
    let mut sim = fresh_sim(101);
    let region = first_region(&mut sim);
    let id = sim
        .register_authored_container(
            "small_crate",
            region,
            [10.0, 0.0, 10.0],
            /*is_public=*/ false,
            Some("coalition".to_string()),
            /*depth_tier=*/ 1,
            ContainerInteractionMode::Openable,
            /*seed=*/ 12345,
        )
        .expect("registration should succeed");
    let grid = sim
        .container_view(id)
        .expect("container should be queryable post-register");
    assert_eq!(grid.width, 4, "small_crate is a 4x4 grid");
    assert_eq!(grid.height, 4, "small_crate is a 4x4 grid");
}

#[test]
fn registers_large_cache_with_correct_grid() {
    let mut sim = fresh_sim(102);
    let region = first_region(&mut sim);
    let id = sim
        .register_authored_container(
            "large_cache",
            region,
            [0.0, 0.0, 0.0],
            false,
            Some("coalition".to_string()),
            1,
            ContainerInteractionMode::Openable,
            999,
        )
        .expect("registration should succeed");
    let grid = sim.container_view(id).expect("container present");
    assert_eq!(grid.width, 8, "large_cache is 8 wide");
    assert_eq!(grid.height, 10, "large_cache is 10 tall");
}

#[test]
fn unknown_kind_id_returns_error() {
    let mut sim = fresh_sim(103);
    let region = first_region(&mut sim);
    let result = sim.register_authored_container(
        "not_a_real_kind",
        region,
        [0.0; 3],
        false,
        None,
        1,
        ContainerInteractionMode::Openable,
        1,
    );
    assert!(result.is_err(), "unknown kind should be rejected");
}

#[test]
fn breakable_mode_persists_on_spawned_component() {
    let mut sim = fresh_sim(104);
    let region = first_region(&mut sim);
    let id = sim
        .register_authored_container(
            "medium_stash",
            region,
            [5.0, 0.0, 5.0],
            false,
            Some("raiders".to_string()),
            1,
            ContainerInteractionMode::Breakable,
            42,
        )
        .expect("registration should succeed");

    // Walk the world for the entity to inspect its component
    // directly — `container_view` only surfaces the grid, not
    // the mode.
    let world = sim.world_for_test();
    let mut q = world.query::<&WorldContainer>();
    let mut found_mode: Option<ContainerInteractionMode> = None;
    for wc in q.iter(world) {
        if wc.id.0 as i64 == i64::from(id.0) {
            found_mode = Some(wc.interaction_mode);
            break;
        }
    }
    assert_eq!(
        found_mode,
        Some(ContainerInteractionMode::Breakable),
        "interaction_mode should flow onto the spawned component",
    );
}

#[test]
fn same_seed_produces_identical_authored_contents() {
    // Two fresh sims with the same world seed + same registration
    // seed → same content roll. Verifies the determinism the
    // GDScript walker relies on (hash of container_id_str →
    // stable per-marker contents across reloads).
    let mut sim_a = fresh_sim(50);
    let mut sim_b = fresh_sim(50);
    let region_a = first_region(&mut sim_a);
    let region_b = first_region(&mut sim_b);
    assert_eq!(region_a, region_b);

    let id_a = sim_a
        .register_authored_container(
            "small_crate",
            region_a,
            [0.0; 3],
            false,
            Some("coalition".to_string()),
            1,
            ContainerInteractionMode::Openable,
            777,
        )
        .unwrap();
    let id_b = sim_b
        .register_authored_container(
            "small_crate",
            region_b,
            [0.0; 3],
            false,
            Some("coalition".to_string()),
            1,
            ContainerInteractionMode::Openable,
            777,
        )
        .unwrap();
    let stringify = |g: GridInventory| -> Vec<(String, u32)> {
        g.items
            .iter()
            .map(|p| (p.stack.id.0.clone(), p.stack.count))
            .collect()
    };
    let a = stringify(sim_a.container_view(id_a).unwrap());
    let b = stringify(sim_b.container_view(id_b).unwrap());
    assert_eq!(
        a, b,
        "same registration seed must produce identical authored contents",
    );
}

#[test]
fn different_seeds_produce_different_authored_contents() {
    let mut sim = fresh_sim(60);
    let region = first_region(&mut sim);
    let id_a = sim
        .register_authored_container(
            "medium_stash",
            region,
            [0.0; 3],
            false,
            Some("coalition".to_string()),
            1,
            ContainerInteractionMode::Openable,
            1,
        )
        .unwrap();
    let id_b = sim
        .register_authored_container(
            "medium_stash",
            region,
            [0.0; 3],
            false,
            Some("coalition".to_string()),
            1,
            ContainerInteractionMode::Openable,
            2,
        )
        .unwrap();
    let a = sim.container_view(id_a).unwrap();
    let b = sim.container_view(id_b).unwrap();
    // Almost-zero chance two different seeds produce identical
    // grids — pool has dozens of entries and roll picks ~4-7
    // items, so collision probability is ε.
    let str_a: Vec<_> = a
        .items
        .iter()
        .map(|p| (p.stack.id.0.clone(), p.stack.count))
        .collect();
    let str_b: Vec<_> = b
        .items
        .iter()
        .map(|p| (p.stack.id.0.clone(), p.stack.count))
        .collect();
    assert_ne!(
        str_a, str_b,
        "different seeds should produce different content rolls",
    );
}
