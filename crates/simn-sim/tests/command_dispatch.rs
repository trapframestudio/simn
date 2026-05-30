//! Dispatcher test for threaded-sim PR C step 2 — `SimCommand`.
//!
//! Pins the contract that `dispatch_command(&mut sim, cmd)`
//! drives the same mutation as the equivalent direct
//! `Sim::*` call. Step 3 wires this dispatcher into the
//! worker thread; step 4 flips `SimHost` call sites from
//! direct mutation to `cmd_tx.send(...)`. Both depend on the
//! dispatcher being correct in isolation, which is what this
//! test covers.

use simn_sim::action::ActionKind;
use simn_sim::worker::{dispatch_command, SimCommand};
use simn_sim::{RegionGraph, Sim};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn dispatch_upsert_player_spawns_entity() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    assert!(sim.player_view(42).is_none());
    dispatch_command(
        &mut sim,
        SimCommand::UpsertPlayer {
            steam_id: 42,
            region: 1,
            pos: [1.0, 0.0, 2.0],
            yaw: 0.0,
        },
    )
    .unwrap();
    let view = sim.player_view(42).expect("player spawned");
    assert_eq!(view.pos, [1.0, 0.0, 2.0]);
}

#[test]
fn dispatch_action_move_updates_position() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    dispatch_command(
        &mut sim,
        SimCommand::UpsertPlayer {
            steam_id: 7,
            region: 1,
            pos: [0.0, 0.0, 0.0],
            yaw: 0.0,
        },
    )
    .unwrap();
    dispatch_command(
        &mut sim,
        SimCommand::Action {
            steam_id: 7,
            kind: ActionKind::Move {
                pos: [10.0, 0.0, -5.0],
                yaw: 1.5,
            },
        },
    )
    .unwrap();
    let view = sim.player_view(7).expect("player exists");
    assert_eq!(view.pos, [10.0, 0.0, -5.0]);
    assert!((view.yaw - 1.5).abs() < 1e-6);
}

#[test]
fn dispatch_remove_player_clears_entity() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    dispatch_command(
        &mut sim,
        SimCommand::UpsertPlayer {
            steam_id: 99,
            region: 1,
            pos: [0.0, 0.0, 0.0],
            yaw: 0.0,
        },
    )
    .unwrap();
    assert!(sim.player_view(99).is_some());
    dispatch_command(&mut sim, SimCommand::RemovePlayer { steam_id: 99 }).unwrap();
    assert!(sim.player_view(99).is_none());
}

#[test]
fn dispatch_set_active_region_changes_focus() {
    let _dir = TempDir::new().unwrap();
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    dispatch_command(&mut sim, SimCommand::SetActiveRegion { region: 2 }).unwrap();
    // No public read of ActiveRegions today (it's a resource);
    // the proxy is that tick() runs without panic and that
    // subsequent SetActiveRegion calls also succeed.
    sim.tick().unwrap();
    dispatch_command(&mut sim, SimCommand::SetActiveRegion { region: 1 }).unwrap();
    sim.tick().unwrap();
}

#[test]
fn dispatch_action_for_unknown_player_errors_but_doesnt_panic() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    // Move on a non-existent steam_id should return an error,
    // not panic — worker loop logs and keeps going.
    let result = dispatch_command(
        &mut sim,
        SimCommand::Action {
            steam_id: 999,
            kind: ActionKind::Move {
                pos: [0.0, 0.0, 0.0],
                yaw: 0.0,
            },
        },
    );
    assert!(result.is_err());
}
