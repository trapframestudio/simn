//! End-to-end persistence tests.

use simn_sim::{
    BodyPart, BodyParts, Health, RegionGraph, SavePaths, Sim, Stamina, SurvivalStat, SurvivalStats,
};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

#[test]
fn save_load_roundtrip() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.upsert_player(100, 1, [1.0, 2.0, 3.0], 0.1).unwrap();
        sim.upsert_player(200, 1, [4.0, 5.0, 6.0], 0.2).unwrap();
        sim.upsert_player(300, 2, [7.0, 8.0, 9.0], 0.3).unwrap();
        for _ in 0..10 {
            sim.tick().unwrap();
        }
        sim.shutdown().unwrap();
    }

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    assert_eq!(sim.current_tick(), 10);
    let p1 = sim.player_view(100).unwrap();
    assert_eq!(p1.pos, [1.0, 2.0, 3.0]);
    assert_eq!(p1.region, 1);
    assert!((p1.yaw - 0.1).abs() < f32::EPSILON);
    assert_eq!(sim.player_view(200).unwrap().pos, [4.0, 5.0, 6.0]);
    assert_eq!(sim.player_view(300).unwrap().region, 2);
}

#[test]
fn journal_replay_after_crash() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    // Simulate a crash: write state, tick 50x, drop without shutdown.
    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.upsert_player(42, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
        for i in 1..=50 {
            let x = i as f32;
            sim.move_player(42, [x, 0.0, 0.0], 0.0).unwrap();
            sim.tick().unwrap();
        }
        // Flush so the journal is on disk even without shutdown.
        // (A crash before any fsync would still recover an earlier
        // state; we explicitly flush here so the test is deterministic.)
        sim.flush_journal_for_test();
        // No shutdown() — simulate crash.
        drop(sim);
    }

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let p = sim.player_view(42).unwrap();
    assert_eq!(p.pos, [50.0, 0.0, 0.0]);
    assert_eq!(sim.current_tick(), 50);
}

/// Regression gate: every schedule that runs after `tick()` must have
/// access to its resources. The bug class this catches — adding a new
/// resource to `build_world` (the fresh-sim path) and forgetting to
/// mirror it into `Sim::load` — has bitten us multiple times. The
/// existing `journal_replay_after_crash` loads but doesn't tick after
/// the load, so a missing resource only surfaces in production when
/// the worker thread panics on its first tick. This test loads and
/// then ticks; if any system can't find its `ResMut<…>`, the schedule
/// panics and the test fails.
#[test]
fn load_then_tick_runs_all_schedules() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.upsert_player(7, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
        for _ in 0..5 {
            sim.tick().unwrap();
        }
        sim.shutdown().unwrap();
    }
    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    // Several ticks so every schedule segment runs (player, NPC index,
    // NPC threats, NPC aggro, NPC planning, NPC lifecycle, offline
    // loot) — any missing resource panics inside the schedule.
    for _ in 0..5 {
        sim.tick().unwrap();
    }
    assert!(sim.current_tick() >= 10);
}

#[test]
fn torn_journal_tail_is_skipped() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let sp = paths(&dir);

    {
        let mut sim = Sim::new(sp.clone(), graph.clone()).unwrap();
        sim.upsert_player(1, 1, [1.0, 0.0, 0.0], 0.0).unwrap();
        sim.tick().unwrap();
        sim.upsert_player(1, 1, [2.0, 0.0, 0.0], 0.0).unwrap();
        sim.tick().unwrap();
        sim.flush_journal_for_test();
    }

    // Corrupt the journal: append 12 random bytes (shorter than any
    // valid record prefix+payload+crc).
    {
        let mut f = OpenOptions::new().write(true).open(&sp.journal).unwrap();
        f.seek(SeekFrom::End(0)).unwrap();
        f.write_all(&[
            0xab, 0xcd, 0xef, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
        ])
        .unwrap();
    }

    let mut sim = Sim::load_or_new(sp, graph).unwrap();
    // Last intact state was pos=[2,0,0] at tick 2.
    let p = sim.player_view(1).unwrap();
    assert_eq!(p.pos, [2.0, 0.0, 0.0]);
}

#[test]
fn region_navigation_persists() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.upsert_player(7, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
        sim.change_player_region(7, 2).unwrap();
        sim.tick().unwrap();
        sim.shutdown().unwrap();
    }

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    assert_eq!(sim.player_view(7).unwrap().region, 2);
}

#[test]
fn unknown_region_rejected() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    assert!(sim.upsert_player(1, 999, [0.0; 3], 0.0).is_err());
}

#[test]
fn vitals_roundtrip() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
        sim.apply_damage(1, 30.0).unwrap();
        // Step 2: 30 damage spawns a heavy Bleed wound that would drain
        // ~0.5 HP across the tick loop below. Tourniquet stops it so
        // this test continues to assert the HP-math invariant
        // (damage − heal = 20).
        sim.apply_tourniquet(1, BodyPart::Torso).unwrap();
        sim.heal(1, 10.0).unwrap();
        for _ in 0..5 {
            sim.tick().unwrap();
        }
        sim.shutdown().unwrap();
    }

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(
        (v.health.current - (Health::DEFAULT_MAX - 20.0)).abs() < 0.01,
        "health was {}",
        v.health.current
    );
    // Stamina should have regen'd from the full value; roughly, at least.
    assert!(v.stamina.current <= v.stamina.max);
    assert!(v.stamina.current >= Stamina::DEFAULT_MAX);
}

#[test]
fn damage_clamps_to_zero() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage(1, 10_000.0).unwrap();
    assert_eq!(sim.player_view(1).unwrap().health.current, 0.0);
}

#[test]
fn heal_clamps_to_max() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.heal(1, 10_000.0).unwrap();
    let v = sim.player_view(1).unwrap();
    assert_eq!(v.health.current, v.health.max);
}

#[test]
fn stamina_regens_each_tick() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.set_stamina(1, 0.0).unwrap();
    for _ in 0..10 {
        sim.tick().unwrap();
    }
    let v = sim.player_view(1).unwrap();
    assert!(v.stamina.current > 0.0, "stamina was {}", v.stamina.current);
    assert!(v.stamina.current <= v.stamina.max);
}

#[test]
fn world_time_advances() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    let before = sim.world_time();
    for _ in 0..20 {
        sim.tick().unwrap();
    }
    let after = sim.world_time();
    let expected_delta = 20.0 * 0.050; // 20 ticks * 50ms
    assert!((after.seconds_of_day - before.seconds_of_day - expected_delta).abs() < 0.01);
}

#[test]
fn world_time_rolls_day() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.force_world_time_for_test(0, 1439.95, 1440.0);
    for _ in 0..10 {
        sim.tick().unwrap();
    }
    let t = sim.world_time();
    assert_eq!(t.day, 1, "day should have rolled; got {:?}", t);
}

#[test]
fn world_time_persists() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.force_world_time_for_test(3, 720.0, 1440.0);
        sim.shutdown().unwrap();
    }

    let sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let t = sim.world_time();
    assert_eq!(t.day, 3);
    assert!((t.seconds_of_day - 720.0).abs() < 0.1);
}

#[test]
fn snapshot_compaction() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let sp = paths(&dir);
    let mut sim = Sim::new(sp.clone(), graph).unwrap();
    sim.set_snapshot_interval_for_test(5);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    for _ in 0..5 {
        sim.tick().unwrap();
    }
    // Journal should have been rotated; we verify by reading its
    // snapshot_tick and ensuring it now points at tick 5.
    let st = simn_sim::persistence::journal::read_journal_snapshot_tick(&sp.journal).unwrap();
    assert_eq!(st, 5);
}

#[test]
fn apply_damage_routes_to_torso() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage(1, 50.0).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!((v.body_parts.torso - 50.0).abs() < 0.01);
    assert!((v.body_parts.head - BodyParts::DEFAULT_MAX).abs() < 0.01);
    assert!((v.health.current - 50.0).abs() < 0.01);
}

#[test]
fn body_parts_roundtrip() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
        sim.apply_damage_to_part(1, BodyPart::Head, 10.0).unwrap();
        sim.apply_damage_to_part(1, BodyPart::Torso, 20.0).unwrap();
        sim.apply_damage_to_part(1, BodyPart::LeftArm, 30.0)
            .unwrap();
        sim.apply_damage_to_part(1, BodyPart::RightArm, 40.0)
            .unwrap();
        sim.apply_damage_to_part(1, BodyPart::LeftLeg, 50.0)
            .unwrap();
        sim.apply_damage_to_part(1, BodyPart::RightLeg, 60.0)
            .unwrap();
        // Step 2: above-threshold damage spawns Bleed wounds. Stop them
        // so the 10-tick loop doesn't drift HP — this test is about
        // the persistence roundtrip, not bleed mechanics. Tourniquet
        // works on any severity; bandage would not, since most of
        // these are heavy bleeds.
        for part in [
            BodyPart::Head,
            BodyPart::Torso,
            BodyPart::LeftArm,
            BodyPart::RightArm,
            BodyPart::LeftLeg,
            BodyPart::RightLeg,
        ] {
            sim.apply_tourniquet(1, part).unwrap();
        }
        for _ in 0..10 {
            sim.tick().unwrap();
        }
        sim.shutdown().unwrap();
    }

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!((v.body_parts.head - 90.0).abs() < 0.01);
    assert!((v.body_parts.torso - 80.0).abs() < 0.01);
    assert!((v.body_parts.left_arm - 70.0).abs() < 0.01);
    assert!((v.body_parts.right_arm - 60.0).abs() < 0.01);
    assert!((v.body_parts.left_leg - 50.0).abs() < 0.01);
    assert!((v.body_parts.right_leg - 40.0).abs() < 0.01);
    // Aggregate health mirrors min(head, torso) = min(90, 80) = 80.
    assert!((v.health.current - 80.0).abs() < 0.01);
}

#[test]
fn head_damage_kills() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::Head, 200.0).unwrap();
    let v = sim.player_view(1).unwrap();
    assert_eq!(v.body_parts.head, 0.0);
    assert!(!v.body_parts.is_alive());
    assert_eq!(v.health.current, 0.0);
}

#[test]
fn limb_damage_disables_not_kills() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::LeftLeg, 200.0)
        .unwrap();
    let v = sim.player_view(1).unwrap();
    assert_eq!(v.body_parts.left_leg, 0.0);
    assert!(v.body_parts.limb_disabled(BodyPart::LeftLeg));
    assert!(v.body_parts.is_alive());
    // Aggregate health unaffected by a limb at 0.
    assert!((v.health.current - Health::DEFAULT_MAX).abs() < 0.01);
}

#[test]
fn survival_stats_drain() {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    let before = sim.player_view(1).unwrap().survival;
    for _ in 0..200 {
        sim.tick().unwrap();
    }
    let after = sim.player_view(1).unwrap().survival;
    assert!(
        after.hunger < before.hunger,
        "hunger should drain: before={} after={}",
        before.hunger,
        after.hunger
    );
    assert!(after.thirst < before.thirst);
    assert!(after.fatigue < before.fatigue);
    // Thirst drains faster than hunger per spec §3.3.
    assert!(
        before.thirst - after.thirst > before.hunger - after.hunger,
        "thirst should drain faster than hunger"
    );
}

#[test]
fn survival_persists_across_reload() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
        sim.set_survival_stat(1, SurvivalStat::Hunger, 42.0)
            .unwrap();
        sim.set_survival_stat(1, SurvivalStat::Thirst, 17.0)
            .unwrap();
        sim.set_survival_stat(1, SurvivalStat::Fatigue, 88.0)
            .unwrap();
        sim.shutdown().unwrap();
    }

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!((v.survival.hunger - 42.0).abs() < 0.01);
    assert!((v.survival.thirst - 17.0).abs() < 0.01);
    assert!((v.survival.fatigue - 88.0).abs() < 0.01);
}

#[test]
fn consume_clamps_to_full() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.set_survival_stat(1, SurvivalStat::Hunger, 50.0)
        .unwrap();
    sim.consume(1, 200.0, 0.0, 0.0).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!((v.survival.hunger - SurvivalStats::FULL).abs() < 0.01);
}

#[test]
fn low_hunger_halves_regen() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph).unwrap();
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    // Starve below threshold but well above HP-drain threshold so the
    // torso doesn't take HP damage during the test (which would not
    // affect stamina but would muddy the regen-only intent).
    sim.set_survival_stat(1, SurvivalStat::Hunger, 25.0)
        .unwrap();
    sim.set_stamina(1, 0.0).unwrap();
    for _ in 0..20 {
        sim.tick().unwrap();
    }
    let v = sim.player_view(1).unwrap();
    // Full-rate regen would be 15.0 * 0.05 * 20 = 15.0 stamina.
    // Halved: ~7.5. Allow a wide window for clamp + edge effects.
    assert!(
        v.stamina.current > 5.0 && v.stamina.current < 11.0,
        "expected halved regen ≈ 7.5, got {}",
        v.stamina.current
    );
}

#[test]
fn starvation_drains_hp_slowly() {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.set_survival_stat(1, SurvivalStat::Hunger, 0.0).unwrap();
    sim.set_survival_stat(1, SurvivalStat::Thirst, 0.0).unwrap();
    let before = sim.player_view(1).unwrap().health.current;
    for _ in 0..100 {
        sim.tick().unwrap();
    }
    let after = sim.player_view(1).unwrap().health.current;
    assert!(
        after < before,
        "starvation should drain HP: before={before} after={after}"
    );
    // 100 ticks = 5 sec real. Two penalties × 0.25 hp/sec × 5 sec = 2.5 hp lost.
    // Should be nowhere near lethal.
    assert!(
        before - after < 10.0,
        "starvation should be slow: lost {} hp in 100 ticks",
        before - after
    );
}
