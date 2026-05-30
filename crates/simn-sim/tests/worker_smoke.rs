//! Smoke test for threaded-sim PR C step 3 — `SimWorker`.
//!
//! Exercises the worker thread end-to-end: spawn, send
//! commands, observe ticks advancing via the published
//! snapshot + view, shut down cleanly. Validates the
//! cross-thread plumbing in isolation before step 4 wires
//! the bridge onto it.
//!
//! Timing-sensitive — uses real wall clock (20 Hz ⇒ 50 ms
//! per tick). The harness polls a published cell with a
//! generous timeout (5 s) so CI variance doesn't false-fail.

use std::time::{Duration, Instant};

use simn_sim::action::ActionKind;
use simn_sim::worker::{SimCommand, SimWorker};
use simn_sim::{RegionGraph, Sim};

fn quiet_sim() -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

/// Wait until the worker's published view tick reaches or
/// exceeds `target`, or fail after `timeout`. Returns the
/// final observed tick.
fn wait_for_tick(worker: &SimWorker, target: u64, timeout: Duration) -> u64 {
    let start = Instant::now();
    loop {
        if let Some(view) = worker.view() {
            if view.tick >= target {
                return view.tick;
            }
        }
        if start.elapsed() > timeout {
            panic!(
                "worker did not reach tick {target} within {:?}; last view: {:?}",
                timeout,
                worker.view().map(|v| v.tick)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn worker_spawn_and_shutdown() {
    let worker = SimWorker::spawn(quiet_sim());
    // Give the worker enough time to complete at least one
    // tick (50 ms cadence) so we know the loop is actually
    // running rather than wedged on spawn.
    wait_for_tick(&worker, 1, Duration::from_secs(2));
    worker.shutdown().expect("clean shutdown");
}

#[test]
fn worker_publishes_view_and_snapshots() {
    let worker = SimWorker::spawn(quiet_sim());
    // After 2 ticks, snapshot pair should be available
    // (needs prev + curr).
    wait_for_tick(&worker, 2, Duration::from_secs(2));
    let view = worker.view().expect("view present after 2 ticks");
    assert!(
        view.tick >= 2,
        "view tick should advance; got {}",
        view.tick
    );
    let pair = worker
        .snapshots()
        .expect("snapshot pair present after 2 ticks");
    assert!(
        pair.curr.tick > pair.prev.tick,
        "pair should have prev<curr; got prev={} curr={}",
        pair.prev.tick,
        pair.curr.tick
    );
    worker.shutdown().expect("clean shutdown");
}

#[test]
fn worker_processes_upsert_and_move_commands() {
    let worker = SimWorker::spawn(quiet_sim());
    // Wait for the worker to come online.
    wait_for_tick(&worker, 1, Duration::from_secs(2));
    // Spawn a player via command.
    worker
        .send(SimCommand::UpsertPlayer {
            steam_id: 42,
            region: 1,
            pos: [0.0, 0.0, 0.0],
            yaw: 0.0,
        })
        .expect("upsert command sent");
    // Wait for next tick to process.
    let post_upsert_tick = worker.view().map(|v| v.tick).unwrap_or(0);
    wait_for_tick(&worker, post_upsert_tick + 2, Duration::from_secs(2));
    let view = worker.view().expect("view present");
    let p = view
        .players
        .get(&42)
        .expect("player 42 should appear in view");
    assert_eq!(p.pos, [0.0, 0.0, 0.0]);

    // Now move the player and check the view updates.
    worker
        .send(SimCommand::Action {
            steam_id: 42,
            kind: ActionKind::Move {
                pos: [10.0, 0.0, -5.0],
                yaw: 1.5,
            },
        })
        .expect("move command sent");
    let post_move_tick = worker.view().map(|v| v.tick).unwrap_or(0);
    wait_for_tick(&worker, post_move_tick + 2, Duration::from_secs(2));
    let view = worker.view().expect("view present");
    let p = view
        .players
        .get(&42)
        .expect("player 42 still in view after move");
    assert_eq!(p.pos, [10.0, 0.0, -5.0]);
    assert!((p.yaw - 1.5).abs() < 1e-6);

    worker.shutdown().expect("clean shutdown");
}

#[test]
fn worker_inspect_returns_query_result() {
    let worker = SimWorker::spawn(quiet_sim());
    wait_for_tick(&worker, 1, Duration::from_secs(2));
    let tick = worker
        .inspect(|sim| sim.current_tick())
        .expect("inspect ran");
    assert!(tick >= 1, "inspect should see at least one tick: {tick}");
    worker.shutdown().expect("clean shutdown");
}

#[test]
fn worker_inspect_can_mutate_then_observe() {
    let worker = SimWorker::spawn(quiet_sim());
    wait_for_tick(&worker, 1, Duration::from_secs(2));
    // Use inspect for both the mutation and the read. This
    // is how the migration escape hatch handles a `SimHost`
    // call site where no `SimCommand` variant exists yet.
    let inserted = worker
        .inspect(|sim| {
            sim.upsert_player(77, 1, [4.0, 0.0, 8.0], 0.7).unwrap();
            sim.player_view(77).map(|v| v.pos)
        })
        .expect("inspect ran");
    assert_eq!(inserted, Some([4.0, 0.0, 8.0]));
    worker.shutdown().expect("clean shutdown");
}

#[test]
fn worker_view_tick_advances_over_time() {
    let worker = SimWorker::spawn(quiet_sim());
    // Reach a baseline.
    wait_for_tick(&worker, 3, Duration::from_secs(2));
    let t1 = worker.view().expect("view").tick;
    // Wait at least 250 ms (5 sim ticks at 20 Hz) and confirm
    // the published tick has advanced by ≥4. Tolerance for
    // CI scheduler noise.
    std::thread::sleep(Duration::from_millis(250));
    let t2 = worker.view().expect("view").tick;
    assert!(
        t2 >= t1 + 4,
        "tick should advance ≥4 over 250ms; t1={} t2={}",
        t1,
        t2
    );
    worker.shutdown().expect("clean shutdown");
}
