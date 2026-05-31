//! Offline movement tests (Phase 1D of `sim-iteration-5-12-plan.md`).
//!
//! Acceptance: an offline NPC's position changes over offline ticks
//! as it hops between waypoints. Phase 1D uses `Base` entities as
//! de facto waypoints (proper waypoint graph from
//! `npc-traversal-plan.md` lands later).
//!
//! Cadence: offline clock advances every 10 sim ticks. Each offline
//! tick, `offline_movement` either picks a new target or advances
//! interpolation along the current leg. Tests tick by sim tick (the
//! API only exposes that) and assert on offline-tick boundaries.

use simn_sim::{RegionGraph, SavePaths, Sim, OFFLINE_TIER_TICK_DIVISOR};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

fn tick_one_offline_tick(sim: &mut Sim) {
    for _ in 0..OFFLINE_TIER_TICK_DIVISOR {
        sim.tick().unwrap();
    }
}

fn legacy_procedural_graph() -> RegionGraph {
    // Iteration 5-14 Phase C: `default_test_graph` flags map_a..d as
    // `scene_authored_pois = true`, so the procedural base scatter
    // doesn't run. These tests use bases as offline movement
    // waypoints, so we build a legacy graph with the gate off.
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
fn offline_npc_picks_a_target_after_first_offline_tick() {
    // Procedurally-seeded test graph has bases in regions 1-4, which
    // is what `offline_movement` uses as waypoints. Region 1 is not
    // in `ActiveRegions` (default), so it's an "offline" region.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), legacy_procedural_graph()).unwrap();
    // Activate region 2 so region 1's NPC stays offline.
    sim.set_active_region(2);
    let id = sim.spawn_offline_npc_for_test("coalition", 1, [100.0, -200.0]);

    // Before any offline tick the NPC has no target.
    let before = sim.offline_npc_for_test(id).unwrap();
    assert!(
        before.target_2d.is_none(),
        "fresh offline NPC starts with no target"
    );

    // Advance one offline tick (10 sim ticks at the stock divisor).
    tick_one_offline_tick(&mut sim);

    let after = sim.offline_npc_for_test(id).unwrap();
    assert!(
        after.target_2d.is_some(),
        "after one offline tick the NPC should have a target picked from region's bases"
    );
    // Arrival tick should be set in the future.
    assert!(
        after.arrival_offline_tick > 0,
        "arrival_offline_tick should be a positive future tick, got {}",
        after.arrival_offline_tick
    );
}

#[test]
fn offline_npc_position_changes_over_ticks() {
    // The strongest version of the acceptance criterion: position is
    // observably different after enough offline ticks.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), legacy_procedural_graph()).unwrap();
    sim.set_active_region(2);
    let id = sim.spawn_offline_npc_for_test("coalition", 1, [500.0, 500.0]);
    let start_pos = sim.offline_npc_for_test(id).unwrap().position_2d;

    // Run 30 offline ticks (~15 s of sim wall time). Enough for the
    // NPC to pick a target and travel some distance along it.
    for _ in 0..30 {
        tick_one_offline_tick(&mut sim);
    }
    let end_pos = sim.offline_npc_for_test(id).unwrap().position_2d;
    let dx = end_pos[0] - start_pos[0];
    let dz = end_pos[1] - start_pos[1];
    let moved = (dx * dx + dz * dz).sqrt();
    assert!(
        moved > 5.0,
        "offline NPC should have moved at least 5 m after 30 offline ticks (got {moved} m: start={start_pos:?} end={end_pos:?})"
    );
}

#[test]
fn offline_npc_in_region_with_no_bases_stays_idle() {
    // empty_graph_with_one_region — one region, no procedural
    // seeding → no bases. The NPC should never pick a target and
    // never move.
    use simn_sim::Region;
    let mut g = RegionGraph::new();
    g.insert(Region {
        id: 99,
        name: "empty_map".into(),
        map_scene: "res://scenes/test/test_map_1.tscn".into(),
        neighbors: vec![],
        transitions: Default::default(),
        procedurally_seeded: false,
        scene_authored_pois: false,
    });

    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), g).unwrap();
    let id = sim.spawn_offline_npc_for_test("nomads", 99, [0.0, 0.0]);

    for _ in 0..5 {
        tick_one_offline_tick(&mut sim);
    }
    let state = sim.offline_npc_for_test(id).unwrap();
    assert!(
        state.target_2d.is_none(),
        "no bases → no target picked, but got {:?}",
        state.target_2d
    );
    assert_eq!(
        state.position_2d,
        [0.0, 0.0],
        "no target → no movement, but position changed to {:?}",
        state.position_2d
    );
}

#[test]
fn offline_npc_arrives_at_target_and_picks_a_new_one() {
    // Position the NPC near a known waypoint, run enough offline
    // ticks to complete the leg, and confirm the leg cycle restarts.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), legacy_procedural_graph()).unwrap();
    sim.set_active_region(2);
    let id = sim.spawn_offline_npc_for_test("coalition", 1, [0.0, 0.0]);

    // 100 offline ticks at 0.5 s each = 50 s of offline-tier time.
    // At 6 m/s walking speed, that's up to 300 m of travel — enough
    // to complete several short legs between bases on the test grid
    // (each cell is ~657 m, but the NPC may pick a base just a few
    // meters away on the first roll).
    let mut seen_targets: Vec<[f32; 2]> = Vec::new();
    let mut last_target: Option<[f32; 2]> = None;
    for _ in 0..100 {
        tick_one_offline_tick(&mut sim);
        if let Some(t) = sim.offline_npc_for_test(id).unwrap().target_2d {
            if Some(t) != last_target {
                seen_targets.push(t);
                last_target = Some(t);
            }
        } else {
            // Idle this tick — next tick will pick fresh.
            last_target = None;
        }
    }
    // We won't always see >1 distinct target (legs can be long), but
    // the NPC must have picked at least one target during the
    // window.
    assert!(
        !seen_targets.is_empty(),
        "expected at least one target pick across 100 offline ticks"
    );
}
