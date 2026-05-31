//! Per-squad mean `leadership` → cohesion-break threshold. Third
//! behavior integration of the `NpcCharacter` substrate (see
//! `docs/book/src/planning/npc-character-authoring-plan.md` §6.1).

use simn_sim::systems::cohesion_multiplier_for_leadership;
use simn_sim::{RegionGraph, Sim, SquadObjective};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn cohesion_multiplier_endpoints() {
    let lo = cohesion_multiplier_for_leadership(0);
    let mid = cohesion_multiplier_for_leadership(50);
    let hi = cohesion_multiplier_for_leadership(100);
    assert!((lo - 0.7).abs() < 0.001, "lo={}", lo);
    assert!((mid - 1.0).abs() < 0.001, "mid={}", mid);
    assert!((hi - 1.3).abs() < 0.001, "hi={}", hi);
}

#[test]
fn cohesion_multiplier_monotonic_in_leadership() {
    let mut last = -1.0_f32;
    for l in 0..=100 {
        let m = cohesion_multiplier_for_leadership(l);
        assert!(m > last, "non-monotonic at {}: {} <= {}", l, m, last);
        last = m;
    }
}

#[test]
fn low_leadership_squad_regroups_at_90m_spread() {
    // Two-NPC squad. Place them 180m apart so each is 90m from the
    // shared centroid. Leadership 0 ⇒ multiplier 0.7 ⇒ threshold
    // 56m. 90m > 56m ⇒ Regroup should fire.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 7777;
    let a = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));
    let b = sim.spawn_npc_for_test("coalition", 1, [180.0, 0.0, 0.0], Some(group_id));
    sim.set_npc_leadership_for_test(a, 0);
    sim.set_npc_leadership_for_test(b, 0);
    // Re-position to confirm spread (debug spawn places at [0;3]
    // first, then we move). Move both so the centroid is at [90,0,0]
    // and each is 90m from it.
    sim.move_npc_for_test(a, [0.0, 0.0, 0.0], 1);
    sim.move_npc_for_test(b, [180.0, 0.0, 0.0], 1);
    sim.tick().unwrap();
    let obj = sim.squad_objective_for_test(group_id);
    assert!(
        matches!(obj, Some(SquadObjective::Regroup { .. })),
        "low-leadership squad at 90m spread should Regroup, got {:?}",
        obj
    );
}

#[test]
fn high_leadership_squad_holds_at_90m_spread() {
    // Same 90m-from-centroid scenario, but leadership 100 ⇒
    // multiplier 1.3 ⇒ threshold 104m. 90m < 104m ⇒ no Regroup.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 8888;
    let a = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));
    let b = sim.spawn_npc_for_test("coalition", 1, [180.0, 0.0, 0.0], Some(group_id));
    sim.set_npc_leadership_for_test(a, 100);
    sim.set_npc_leadership_for_test(b, 100);
    sim.move_npc_for_test(a, [0.0, 0.0, 0.0], 1);
    sim.move_npc_for_test(b, [180.0, 0.0, 0.0], 1);
    sim.tick().unwrap();
    let obj = sim.squad_objective_for_test(group_id);
    assert!(
        !matches!(obj, Some(SquadObjective::Regroup { .. })),
        "high-leadership squad at 90m spread should NOT Regroup, got {:?}",
        obj
    );
}

#[test]
fn baseline_threshold_still_holds_below_56m() {
    // Sanity: a 50m spread is within the threshold even at the
    // tightest (low-leadership) leash. No Regroup expected.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 9999;
    let a = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));
    let b = sim.spawn_npc_for_test("coalition", 1, [100.0, 0.0, 0.0], Some(group_id));
    sim.set_npc_leadership_for_test(a, 0);
    sim.set_npc_leadership_for_test(b, 0);
    sim.move_npc_for_test(a, [0.0, 0.0, 0.0], 1);
    sim.move_npc_for_test(b, [100.0, 0.0, 0.0], 1);
    sim.tick().unwrap();
    let obj = sim.squad_objective_for_test(group_id);
    assert!(
        !matches!(obj, Some(SquadObjective::Regroup { .. })),
        "50m-from-centroid (within tightest 56m leash) should NOT \
         Regroup, got {:?}",
        obj
    );
}
