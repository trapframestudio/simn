//! Integration test for the threaded-sim PR A scaffold: snapshot
//! publishing + the `(prev, curr)` pair API. Confirms that
//! `Sim::tick` rotates the ring correctly and that the snapshot
//! reflects authoritative NPC positions for active-region NPCs
//! only (offline-region NPCs are frozen by the tier filter, and
//! correspondingly omitted from the published snapshot).
//!
//! See `docs/book/src/planning/threaded-sim-plan.md` §4 for the
//! contract this test pins down.

use simn_sim::{RegionGraph, Sim};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn snapshot_pair_unavailable_before_two_ticks() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    // Fresh sim: no snapshots yet.
    assert!(sim.snapshot_pair().is_none());
    assert!(sim.current_snapshot().is_none());

    // After one tick: current populated, but no pair yet (prev is
    // still None — the ring needs two rotations before pair() is
    // usable).
    sim.tick().unwrap();
    assert!(sim.current_snapshot().is_some());
    assert!(sim.snapshot_pair().is_none());

    // After two ticks: pair is available.
    sim.tick().unwrap();
    let (prev, curr) = sim.snapshot_pair().expect("pair after two ticks");
    assert!(curr.tick > prev.tick);
}

#[test]
fn snapshot_includes_only_active_region_npcs() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    // active region = 1
    let n1 = sim.spawn_npc_for_test("pwa", 1, [10.0, 0.0, 10.0], None);
    let n2 = sim.spawn_npc_for_test("pwa", 2, [20.0, 0.0, 20.0], None);

    sim.tick().unwrap();
    sim.tick().unwrap();

    let curr = sim.current_snapshot().expect("snapshot present");
    // n1 in active region → present; n2 in offline region → omitted.
    assert!(
        curr.find(n1).is_some(),
        "active-region NPC must be in snapshot"
    );
    assert!(
        curr.find(n2).is_none(),
        "offline-region NPC must be omitted from snapshot"
    );
}

#[test]
fn snapshot_advances_published_at_across_ticks() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let _ = sim.spawn_npc_for_test("pwa", 1, [10.0, 0.0, 10.0], None);

    sim.tick().unwrap();
    // Force a small wall-clock gap so consecutive snapshots have
    // distinct `published_at` instants.
    std::thread::sleep(std::time::Duration::from_millis(2));
    sim.tick().unwrap();

    let (prev, curr) = sim.snapshot_pair().expect("pair after two ticks");
    assert!(curr.published_at > prev.published_at, "wall clock advances");
    assert_eq!(curr.tick, prev.tick + 1, "tick number advances by 1");
}

#[test]
fn snapshot_npcs_sorted_by_id() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    // Spawn several so we can confirm ordering. Ids assigned in
    // spawn order, so ascending by spawn time.
    let _a = sim.spawn_npc_for_test("pwa", 1, [10.0, 0.0, 0.0], None);
    let _b = sim.spawn_npc_for_test("pwa", 1, [20.0, 0.0, 0.0], None);
    let _c = sim.spawn_npc_for_test("pwa", 1, [30.0, 0.0, 0.0], None);

    sim.tick().unwrap();

    let curr = sim.current_snapshot().expect("snapshot present");
    let ids: Vec<u64> = curr.npcs.iter().map(|s| s.id.0).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted, "snapshot npcs must be sorted by id");
}
