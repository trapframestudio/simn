//! Per-NPC `perception` stat → `npc_aggro` sight radius. First behavior
//! integration of the `NpcCharacter` substrate (see
//! `docs/book/src/planning/npc-character-authoring-plan.md` §6.1).

use simn_sim::{sight_radius_for_perception, RegionGraph, Sim};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn sight_radius_scaling_endpoints() {
    // Linear in [0.6, 1.4] across perception 0..=100, centered on
    // 1.0 at perception 50.
    let base = 80.0;
    let lo = sight_radius_for_perception(0, base);
    let mid = sight_radius_for_perception(50, base);
    let hi = sight_radius_for_perception(100, base);
    assert!((lo - 48.0).abs() < 0.001, "lo={}", lo);
    assert!((mid - 80.0).abs() < 0.001, "mid={}", mid);
    assert!((hi - 112.0).abs() < 0.001, "hi={}", hi);
}

#[test]
fn sight_radius_monotonic_in_perception() {
    // Higher perception ⇒ larger sight radius. No regressions.
    let base = 80.0;
    let mut last = -1.0_f32;
    for p in 0..=100 {
        let r = sight_radius_for_perception(p, base);
        assert!(
            r > last,
            "non-monotonic at perception={}: {} <= {}",
            p,
            r,
            last
        );
        last = r;
    }
}

#[test]
fn high_perception_acquires_at_long_range_low_does_not() {
    // Place a hostile pair at 95m apart — beyond the default 80m
    // baseline, beyond a perception-30 NPC's ~57m, but within a
    // perception-100 NPC's 112m. The high-perception spotter should
    // acquire `Aggro`; the low-perception target should not (the
    // pair is in range for the spotter but not for the target).
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    // PWA spots Looters at 95m. Both placed facing each other so
    // FOV doesn't gate anyone out.
    let spotter = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let target = sim.spawn_npc_for_test("looters", 1, [95.0, 0.0, 0.0], None);
    sim.set_npc_perception_for_test(spotter, 100);
    sim.set_npc_perception_for_test(target, 30);
    // Face each other — z-axis-aligned setup, yaw 0 = +X. Spotter
    // is at origin facing +X (toward target); target is at +95X
    // facing -X (toward spotter).
    sim.move_npc_for_test(spotter, [0.0, 0.0, 0.0], 1);
    sim.move_npc_for_test(target, [95.0, 0.0, 0.0], 1);
    sim.set_npc_yaw_for_test(spotter, 0.0);
    sim.set_npc_yaw_for_test(target, std::f32::consts::PI);
    // Tick once so npc_aggro runs.
    sim.tick().unwrap();
    let npcs = sim.npcs_in_region(1);
    let spotter_view = npcs.iter().find(|n| n.id == spotter).unwrap();
    let target_view = npcs.iter().find(|n| n.id == target).unwrap();
    // High-perception spotter should have aggro on target.
    assert_eq!(
        spotter_view.aggro_target, target.0,
        "high-perception spotter should acquire aggro at 95m"
    );
    // Low-perception target should NOT have aggro on spotter — at
    // perception 30 their sight is ~67m, which falls short of 95m.
    assert_eq!(
        target_view.aggro_target, 0,
        "low-perception target should not see at 95m"
    );
}

#[test]
fn equal_perception_at_close_range_is_symmetric() {
    // Both NPCs have the same perception and are within everyone's
    // range — both should acquire aggro (the existing behavior).
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let a = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let b = sim.spawn_npc_for_test("looters", 1, [30.0, 0.0, 0.0], None);
    sim.set_npc_perception_for_test(a, 50);
    sim.set_npc_perception_for_test(b, 50);
    sim.set_npc_yaw_for_test(a, 0.0);
    sim.set_npc_yaw_for_test(b, std::f32::consts::PI);
    sim.tick().unwrap();
    let npcs = sim.npcs_in_region(1);
    let a_view = npcs.iter().find(|n| n.id == a).unwrap();
    let b_view = npcs.iter().find(|n| n.id == b).unwrap();
    assert_eq!(a_view.aggro_target, b.0);
    assert_eq!(b_view.aggro_target, a.0);
}

#[test]
fn far_pair_below_min_perception_neither_aggros() {
    // 130m apart, both at perception 0 (~48m sight). Neither
    // should acquire aggro at any tick.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let a = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let b = sim.spawn_npc_for_test("looters", 1, [130.0, 0.0, 0.0], None);
    sim.set_npc_perception_for_test(a, 0);
    sim.set_npc_perception_for_test(b, 0);
    sim.set_npc_yaw_for_test(a, 0.0);
    sim.set_npc_yaw_for_test(b, std::f32::consts::PI);
    sim.tick().unwrap();
    let npcs = sim.npcs_in_region(1);
    let a_view = npcs.iter().find(|n| n.id == a).unwrap();
    let b_view = npcs.iter().find(|n| n.id == b).unwrap();
    assert_eq!(a_view.aggro_target, 0);
    assert_eq!(b_view.aggro_target, 0);
}
