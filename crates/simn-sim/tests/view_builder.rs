//! Builder test for the threaded-sim PR C step 1 — `SimView`.
//!
//! Validates that `worker::build_sim_view(&mut sim)` produces a
//! denormalized view whose fields match the same per-call reads
//! the renderer + HUD do today. Pinning this here lets later
//! steps (worker thread, `ArcSwap` publish) refactor the
//! plumbing without worrying about silently dropping a field.

use simn_sim::worker::build_sim_view;
use simn_sim::{RegionGraph, Sim};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn empty_sim_builds_view_with_no_players() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    sim.tick().unwrap();
    let view = build_sim_view(&mut sim);
    assert_eq!(view.tick, sim.current_tick());
    assert!(view.players.is_empty());
    // Empty-world chronicle: nobody alive, nobody ever spawned in
    // this fresh in-memory sim with no population targets.
    assert_eq!(view.chronicle_summary.total_ever_spawned, 0);
}

#[test]
fn view_tick_matches_current_tick_after_advance() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    for _ in 0..5 {
        sim.tick().unwrap();
    }
    let view = build_sim_view(&mut sim);
    assert_eq!(view.tick, sim.current_tick());
}

#[test]
fn view_world_time_and_weather_match_sim() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    sim.tick().unwrap();
    let view = build_sim_view(&mut sim);
    let expected_time = sim.world_time();
    let expected_weather = sim.weather();
    assert_eq!(view.world_time.day, expected_time.day);
    assert_eq!(view.world_time.seconds_of_day, expected_time.seconds_of_day);
    assert_eq!(view.weather.current, expected_weather.current);
}

#[test]
fn view_includes_every_connected_player() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    sim.upsert_player(11, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
    sim.upsert_player(22, 1, [10.0, 0.0, 5.0], 1.5).unwrap();
    sim.tick().unwrap();
    let view = build_sim_view(&mut sim);
    assert_eq!(view.players.len(), 2);
    let p11 = view.players.get(&11).expect("player 11 in view");
    assert_eq!(p11.steam_id, 11);
    assert_eq!(p11.pos, [0.0, 0.0, 0.0]);
    let p22 = view.players.get(&22).expect("player 22 in view");
    assert_eq!(p22.pos, [10.0, 0.0, 5.0]);
    assert!((p22.yaw - 1.5).abs() < 1e-6);
}

#[test]
fn view_player_state_matches_player_view() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    sim.upsert_player(7, 1, [3.0, 0.0, -2.0], 0.5).unwrap();
    sim.tick().unwrap();
    let view = build_sim_view(&mut sim);
    let direct = sim.player_view(7).expect("direct player_view");
    let viewed = view.players.get(&7).expect("player in view");
    // Spot-check the high-traffic fields HUDs read each frame.
    assert_eq!(viewed.pos, direct.pos);
    assert_eq!(viewed.region, direct.region);
    assert_eq!(viewed.health.current, direct.health.current);
    assert_eq!(viewed.stamina.current, direct.stamina.current);
    assert_eq!(viewed.body_parts.head, direct.body_parts.head);
}

#[test]
fn dropped_player_drops_from_view() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    sim.upsert_player(99, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
    sim.tick().unwrap();
    assert!(build_sim_view(&mut sim).players.contains_key(&99));
    sim.remove_player(99).unwrap();
    sim.tick().unwrap();
    assert!(!build_sim_view(&mut sim).players.contains_key(&99));
}
