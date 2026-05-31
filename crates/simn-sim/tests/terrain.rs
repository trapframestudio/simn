//! Sim ↔ terrain integration tests.
//!
//! Constructs a tiny synthetic heightmap via `Heightmap::from_raw`,
//! attaches it to a region, and verifies that:
//!
//! - Existing bases get Y-snapped on attach.
//! - NPCs spawn-then-tick land at terrain Y.
//! - Unknown-region attach fails cleanly.
//! - Regions without attached terrain leave Y untouched (legacy).

use simn_sim::{RegionGraph, Sim, TerrainMaps};
use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
use simn_terrain::{Heightmap, TerrainMetadata};

/// Build a small heightmap whose Y is exactly `expected_y` everywhere
/// inside the playable extent. The flat span avoids any sampling
/// surprises when a test asks "what's the ground at (x, z)?".
fn flat_heightmap(map_id: &str, extent_m: f32, expected_y_m: f32) -> Heightmap {
    let spacing_m = 100.0_f32;
    let samples_per_side = (extent_m / spacing_m).round() as u32;
    // Use a vert range that easily contains expected_y.
    let vert_max = (expected_y_m * 2.0).max(100.0);
    let metadata = TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: map_id.into(),
        width: samples_per_side,
        height: samples_per_side,
        spacing_m,
        vert_min_m: 0.0,
        vert_max_m: vert_max,
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
    let samples = vec![expected_y_m; (samples_per_side * samples_per_side) as usize];
    Heightmap::from_raw(metadata, samples).unwrap()
}

#[test]
fn attach_unknown_region_errors() {
    let mut sim = Sim::new_in_memory(legacy_procedural_graph());
    let hm = flat_heightmap("ghost", 5000.0, 0.0);
    let err = sim
        .attach_region_terrain(9999, hm)
        .err()
        .expect("attach should reject unknown region");
    assert!(err.to_string().contains("unknown region"), "got: {err}");
}

fn legacy_procedural_graph() -> RegionGraph {
    // Iteration 5-14 Phase C: `default_test_graph` flags map_a..d as
    // `scene_authored_pois = true`, so the procedural base scatter
    // doesn't run. Terrain tests rely on auto-scattered bases for
    // Y-snap assertions; this helper builds the same 4-region 2×2
    // layout with the gate off.
    use simn_sim::Region;
    let mut g = RegionGraph::new();
    for (id, name, scene) in [
        (1u32, "map_a", "res://scenes/test/test_map_1.tscn"),
        (2u32, "map_b", "res://scenes/test/test_map_2.tscn"),
        (3u32, "map_c", "res://scenes/test/test_map_3.tscn"),
        (4u32, "map_d", "res://scenes/test/test_map_4.tscn"),
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
fn attach_clamps_existing_bases_to_ground() {
    let mut sim = Sim::new_in_memory(legacy_procedural_graph());

    // Region 1 ("map_a") is seeded with bases at Y=0 by world_seed.
    // After attaching a flat heightmap at Y=42, all of those should
    // jump to Y=42.
    let hm = flat_heightmap("map_a", 5000.0, 42.0);
    sim.attach_region_terrain(1, hm).unwrap();

    let mut bases_seen = 0usize;
    for view in sim.bases_in_region(1) {
        // f32 storage is bit-exact for flat heightmaps; tight
        // tolerance just guards against sampler arithmetic drift.
        assert!(
            (view.pos[1] - 42.0).abs() < 1e-4,
            "base at {:?} not clamped to 42m (got {})",
            view.pos,
            view.pos[1]
        );
        bases_seen += 1;
    }
    assert!(bases_seen > 0, "expected world_seed to have placed bases");

    // Verify TerrainMaps resource is populated.
    assert!(sim.has_terrain(1));
    assert!(!sim.has_terrain(2)); // didn't attach map_b
}

#[test]
fn npcs_clamp_to_terrain_per_tick() {
    let mut sim = Sim::new_in_memory(legacy_procedural_graph());
    // `new_in_memory` clears all population targets; opt back into
    // a small nomads population per region — enough for the clamp
    // assertions to find some, cheap enough to tick.
    for r in [1u32, 2, 3, 4] {
        sim.set_population_target_for_test(r, "nomads", 5);
    }
    sim.set_active_region(1);

    // Tick the sim a bit so NPCs spawn into region 1.
    for _ in 0..150 {
        sim.tick().unwrap();
    }

    // Confirm there are NPCs to clamp.
    let npcs_before = sim.npcs_in_region(1);
    assert!(
        !npcs_before.is_empty(),
        "expected world_seed/spawn to have produced NPCs"
    );

    // Attach a flat-at-77m heightmap, then tick once so the
    // clamp_npc_terrain_y system runs against the new state.
    let hm = flat_heightmap("map_a", 5000.0, 77.0);
    sim.attach_region_terrain(1, hm).unwrap();
    sim.tick().unwrap();

    let npcs_after = sim.npcs_in_region(1);
    assert!(!npcs_after.is_empty());
    for view in &npcs_after {
        assert!(
            (view.pos[1] - 77.0).abs() < 1e-2,
            "npc at {:?} not clamped to 77m (got {})",
            view.pos,
            view.pos[1]
        );
    }
}

#[test]
fn regions_without_terrain_leave_y_untouched() {
    let mut sim = Sim::new_in_memory(legacy_procedural_graph());
    // `new_in_memory` clears all population targets; opt back into
    // a small nomads population per region — enough for the clamp
    // assertions to find some, cheap enough to tick.
    for r in [1u32, 2, 3, 4] {
        sim.set_population_target_for_test(r, "nomads", 5);
    }
    sim.set_active_region(2);

    for _ in 0..150 {
        sim.tick().unwrap();
    }

    // No terrain attached for region 2 → all NPC Y should remain 0
    // (the spawn default).
    let npcs = sim.npcs_in_region(2);
    assert!(!npcs.is_empty());
    for view in &npcs {
        assert!(
            view.pos[1].abs() < 1e-2,
            "npc Y in unattached region drifted: {}",
            view.pos[1]
        );
    }
}

#[test]
fn terrain_maps_default_is_empty() {
    let tm = TerrainMaps::default();
    assert!(tm.ground_at(1, 0.0, 0.0).is_none());
    assert!(!tm.has(1));
}
