//! Iteration 5-14 Phase B: tests for `Sim::register_authored_base`.
//!
//! The procedural-scatter path in `world_seed.rs` is being replaced
//! (Phase C) by scene-authored `PoiMarker3D` markers — this method
//! is the sim-side endpoint that the GDScript-side enumerator
//! (`base_spawner.gd`, Phase E) calls per marker.

use simn_sim::components::{Base, BaseKind, Position};
use simn_sim::region::{RegionGraph, RegionId};
use simn_sim::Sim;

const TEST_REGION_ID: RegionId = 1;

fn build_sim() -> Sim {
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

#[test]
fn register_spawns_full_component_tuple() {
    let mut sim = build_sim();
    let coalition = sim
        .faction_registry()
        .id_of("coalition")
        .expect("factions.toml must define `coalition`");
    let entity = sim
        .register_authored_base(
            TEST_REGION_ID,
            [100.0, 0.0, 200.0],
            BaseKind::Outpost,
            coalition,
        )
        .expect("register");
    // Verify the spawn shape via world_for_test access.
    let world = sim.world_for_test();
    let base = world.get::<Base>(entity).expect("Base component spawned");
    assert_eq!(base.kind, BaseKind::Outpost);
    let pos = world
        .get::<Position>(entity)
        .expect("Position component spawned");
    // Y is whatever the test sim resolved (no terrain attached → 0 fallback).
    assert!((pos.0[0] - 100.0).abs() < 0.01);
    assert!((pos.0[2] - 200.0).abs() < 0.01);
}

#[test]
fn unknown_region_errors() {
    let mut sim = build_sim();
    let coalition = sim.faction_registry().id_of("coalition").unwrap();
    let bad_region: RegionId = 999;
    let result =
        sim.register_authored_base(bad_region, [0.0, 0.0, 0.0], BaseKind::Outpost, coalition);
    assert!(
        result.is_err(),
        "unknown region must error rather than spawn",
    );
}

#[test]
fn camp_site_has_no_nav_footprint() {
    // CampSite returns None from nav_footprint_xz_m, so registering
    // one does NOT stamp a blocked cell — verify the cell at the
    // base center is still walkable after register. (Other kinds
    // stamp a 3-8m half-extent and block their center; this asserts
    // the CampSite-specific contract.)
    let mut sim = build_sim();
    // Attach a flat heightmap so the nav grid exists.
    use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
    use simn_terrain::{Heightmap, TerrainMetadata};
    let metadata = TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: "test".into(),
        width: 64,
        height: 64,
        spacing_m: 2.0,
        vert_min_m: 0.0,
        vert_max_m: 100.0,
        origin_utm_zone: "10N".into(),
        origin_utm_easting: 0.0,
        origin_utm_northing: 0.0,
        blake3: String::new(),
        features_blake3: String::new(),
        region_size_m: 2048.0,
        playable_extent_x_m: 0.0,
        playable_extent_z_m: 0.0,
        nav_mask_format_version: 0,
        nav_mask_blake3: String::new(),
    };
    let hm = Heightmap::from_raw(metadata, vec![50.0_f32; 64 * 64]).unwrap();
    sim.attach_region_terrain(TEST_REGION_ID, hm).unwrap();
    // After attach, the cell at (10, _, 10) is walkable.
    assert!(sim.is_traversable(TEST_REGION_ID, [10.0, 0.0, 10.0]));
    // Register a CampSite at that location — nomads is the
    // neutral placeholder faction per the world_seed convention.
    let nomads = sim.faction_registry().id_of("nomads").unwrap();
    sim.register_authored_base(
        TEST_REGION_ID,
        [10.0, 0.0, 10.0],
        BaseKind::CampSite,
        nomads,
    )
    .expect("register camp");
    // Cell is STILL walkable — CampSite doesn't stamp.
    assert!(
        sim.is_traversable(TEST_REGION_ID, [10.0, 0.0, 10.0]),
        "CampSite must not block its center cell",
    );
}

#[test]
fn structured_kind_stamps_nav_footprint() {
    // Mirror of `camp_site_has_no_nav_footprint` but with Outpost,
    // which has a 5m half-extent → the center cell ends up blocked.
    let mut sim = build_sim();
    use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
    use simn_terrain::{Heightmap, TerrainMetadata};
    let metadata = TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: "test".into(),
        width: 64,
        height: 64,
        spacing_m: 2.0,
        vert_min_m: 0.0,
        vert_max_m: 100.0,
        origin_utm_zone: "10N".into(),
        origin_utm_easting: 0.0,
        origin_utm_northing: 0.0,
        blake3: String::new(),
        features_blake3: String::new(),
        region_size_m: 2048.0,
        playable_extent_x_m: 0.0,
        playable_extent_z_m: 0.0,
        nav_mask_format_version: 0,
        nav_mask_blake3: String::new(),
    };
    let hm = Heightmap::from_raw(metadata, vec![50.0_f32; 64 * 64]).unwrap();
    sim.attach_region_terrain(TEST_REGION_ID, hm).unwrap();
    assert!(sim.is_traversable(TEST_REGION_ID, [10.0, 0.0, 10.0]));
    let coalition = sim.faction_registry().id_of("coalition").unwrap();
    sim.register_authored_base(
        TEST_REGION_ID,
        [10.0, 0.0, 10.0],
        BaseKind::Outpost,
        coalition,
    )
    .expect("register outpost");
    assert!(
        !sim.is_traversable(TEST_REGION_ID, [10.0, 0.0, 10.0]),
        "Outpost must stamp a nav footprint at its center",
    );
}
