//! PDA event log tests (Phase 1F of `sim-iteration-5-12-plan.md`).
//!
//! Cross-tier event surface: offline-tier kills, gunfire, and base-
//! flips should land in the `PdaEventLog` so the client PDA can
//! poll and render them as toasts. Tests assert the entries appear
//! with the expected shape and that the `seq`-bookmark contract
//! works (a client tracking `last_seen` doesn't re-receive events).

use simn_sim::{PdaEvent, RegionGraph, SavePaths, Sim, OFFLINE_TIER_TICK_DIVISOR};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

fn tick_offline_ticks(sim: &mut Sim, n: u64) {
    for _ in 0..(n * OFFLINE_TIER_TICK_DIVISOR) {
        sim.tick().unwrap();
    }
}

#[test]
fn offline_combat_death_lands_in_pda_log() {
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(2);

    // Two hostile clusters in region 1 (offline).
    for i in 0..6 {
        sim.spawn_offline_npc_for_test("coalition", 1, [(i as f32) * 3.0, 0.0]);
        sim.spawn_offline_npc_for_test("raiders", 1, [(i as f32) * 3.0 + 1.0, 20.0]);
    }

    let initial_high_water = sim.pda_log_high_water();
    tick_offline_ticks(&mut sim, 120);

    let events = sim.recent_pda_events_since(initial_high_water);
    let combat_kill = events.iter().find(|e| {
        matches!(
            &e.event,
            PdaEvent::OfflineCombatDeath { killed_faction, killer_faction, .. }
                if (killed_faction == "coalition" && killer_faction == "raiders")
                    || (killed_faction == "raiders" && killer_faction == "coalition")
        )
    });
    assert!(
        combat_kill.is_some(),
        "expected OfflineCombatDeath entry in PDA log after 120 offline ticks, got {} entries: {:?}",
        events.len(),
        events.iter().map(|e| &e.event).collect::<Vec<_>>()
    );
}

#[test]
fn offline_gunfire_coalesces_to_one_pda_entry_per_region_per_tick() {
    // Eight engagement pairs would push 8 raw `Gunshot` events to
    // the bus; we want exactly one `OfflineGunfire` PDA entry per
    // region per offline tick (so toasts don't flood).
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(2);

    for i in 0..8 {
        sim.spawn_offline_npc_for_test("coalition", 1, [(i as f32) * 4.0, 0.0]);
        sim.spawn_offline_npc_for_test("raiders", 1, [(i as f32) * 4.0 + 1.0, 30.0]);
    }

    let initial_high_water = sim.pda_log_high_water();
    // One offline tick — should push at most one gunfire entry per
    // region (we're only fighting in region 1).
    tick_offline_ticks(&mut sim, 1);

    let events = sim.recent_pda_events_since(initial_high_water);
    let gunfire_entries: Vec<_> = events
        .iter()
        .filter(|e| matches!(&e.event, PdaEvent::OfflineGunfire { region: 1 }))
        .collect();
    assert_eq!(
        gunfire_entries.len(),
        1,
        "expected exactly one coalesced gunfire entry for region 1 in one offline tick, got {} (all events: {:?})",
        gunfire_entries.len(),
        events.iter().map(|e| &e.event).collect::<Vec<_>>()
    );
}

#[test]
fn pda_log_seq_bookmark_prevents_redelivery() {
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(2);

    for i in 0..4 {
        sim.spawn_offline_npc_for_test("coalition", 1, [(i as f32) * 3.0, 0.0]);
        sim.spawn_offline_npc_for_test("raiders", 1, [(i as f32) * 3.0 + 1.0, 25.0]);
    }

    // First poll — get everything from boot.
    tick_offline_ticks(&mut sim, 5);
    let first_poll = sim.recent_pda_events_since(0);
    let new_bookmark = first_poll.last().map(|e| e.seq).unwrap_or(0);

    // Second poll using the bookmark — no overlap with first.
    let second_poll = sim.recent_pda_events_since(new_bookmark);
    let first_seqs: std::collections::HashSet<u64> = first_poll.iter().map(|e| e.seq).collect();
    for entry in &second_poll {
        assert!(
            !first_seqs.contains(&entry.seq),
            "seq {} returned twice — bookmark contract broken",
            entry.seq
        );
    }
}

#[test]
fn pda_log_high_water_grows_as_events_arrive() {
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(2);
    let initial = sim.pda_log_high_water();

    for i in 0..4 {
        sim.spawn_offline_npc_for_test("coalition", 1, [(i as f32) * 3.0, 0.0]);
        sim.spawn_offline_npc_for_test("raiders", 1, [(i as f32) * 3.0 + 1.0, 20.0]);
    }
    tick_offline_ticks(&mut sim, 30);

    let after = sim.pda_log_high_water();
    assert!(
        after > initial,
        "high-water should advance as events land (initial={initial} after={after})"
    );
}
