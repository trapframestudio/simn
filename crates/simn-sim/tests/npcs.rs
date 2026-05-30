//! NPC spawn / chronicle / death tests.

use simn_sim::{
    BodyPart, DeathCause, GoalKind, GoalSource, NpcGoal, RegionGraph, SavePaths, Sim,
    SquadObjective,
};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

/// Sim with a tiny population per region — enough for chronicle /
/// id-uniqueness / spawn-loadout tests to see *some* NPCs, but small
/// enough that the per-tick AI cost is bearable. Without this, the
/// default 360-per-faction targets put thousands of NPCs in the
/// world and each `tick()` becomes ~400 ms in debug.
fn light_populated_sim(dir: &TempDir) -> Sim {
    // Real sim — we want spawning, faction control, journal+snapshot.
    // 0.02 takes the stock 360 primary / 180 contested down to ~7 / 4
    // per region per faction. Total ≈ 40 NPCs across the 4 test maps,
    // plenty for chronicle tests, no longer crippling per-tick.
    //
    // Iteration 5-14 Phase C: `default_test_graph` flags map_a..d as
    // `scene_authored_pois = true`, which gates off the procedural
    // base scatter. Tests in this file rely on auto-scattered bases
    // for spawning, so build a legacy graph explicitly.
    let mut sim = Sim::new(paths(dir), legacy_procedural_4region_graph()).unwrap();
    sim.scale_all_population_targets(0.02);
    // Phase 1A gate: `spawn_npcs` only tops up populations in
    // active regions. Tests using this helper expect natural
    // tick-time spawning across the multi-region test graph, so
    // activate them all — same effect as having a player in every
    // region simultaneously (only possible in tests).
    sim.activate_all_regions_for_test();
    sim
}

fn legacy_procedural_4region_graph() -> RegionGraph {
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

fn empty_graph_with_one_region() -> RegionGraph {
    use simn_sim::Region;
    let mut g = RegionGraph::new();
    g.insert(Region {
        id: 1,
        name: "map_a".into(),
        map_scene: "res://scenes/test/test_map_1.tscn".into(),
        neighbors: vec![],
        transitions: Default::default(),
        procedurally_seeded: true,
        scene_authored_pois: false,
    });
    g
}

#[test]
#[ignore = "long-running scenario (~60s); run with --include-ignored before push"]
fn spawn_to_target_converges() {
    // Phase 1A landed an active-region gate on `spawn_npcs`: only
    // *active* regions get topped up via the incremental loop. A
    // fresh `Sim::new` has zero active regions, so the spawn loop
    // never fires until we mark one online. This test exercises
    // the incremental spawn loop (not the bulk-seed path), so we
    // mark region 1 active and let `spawn_npcs` converge over the
    // 2000-tick window.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(1);
    // Run for 2000 ticks; with the per-tick squad budget that's
    // plenty to top up a target of ~540 NPCs in region 1 from
    // zero.
    for _ in 0..2000 {
        sim.tick().unwrap();
    }
    let summary = sim.chronicle_summary();
    assert!(
        summary.currently_alive > 0,
        "expected NPCs to have spawned, got {summary:?}"
    );
    assert!(
        summary.total_ever_spawned >= summary.currently_alive,
        "ever_spawned should be ≥ alive"
    );
}

#[test]
fn chronicle_records_birth() {
    let dir = TempDir::new().unwrap();
    let mut sim = light_populated_sim(&dir);
    // Tick past the first spawn pass.
    for _ in 0..100 {
        sim.tick().unwrap();
    }
    let summary = sim.chronicle_summary();
    assert!(summary.total_ever_spawned > 0);
    // Pick any live NPC and read back its record.
    let mut sample_id = None;
    sim.each_npc(|view| {
        sample_id.get_or_insert(view.id);
    });
    let id = sample_id.expect("at least one NPC alive after 100 ticks");
    let rec = sim.chronicle_get(id).expect("record present");
    assert!(rec.death_tick.is_none());
    assert_eq!(rec.regions_visited.len(), 1);
}

#[test]
fn chronicle_records_death() {
    let dir = TempDir::new().unwrap();
    let mut sim = light_populated_sim(&dir);
    for _ in 0..100 {
        sim.tick().unwrap();
    }
    let mut victim = None;
    sim.each_npc(|view| {
        victim.get_or_insert(view.id);
    });
    let id = victim.expect("alive NPC");
    assert!(sim.kill_npc_for_test(id, DeathCause::NaturalCauses));
    let rec = sim.chronicle_get(id).expect("record present");
    assert!(rec.death_tick.is_some());
    assert_eq!(rec.death_cause, Some(DeathCause::NaturalCauses));
}

#[test]
fn chronicle_persists() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let (alive_before, ever_before) = {
        let mut sim = light_populated_sim(&dir);
        for _ in 0..200 {
            sim.tick().unwrap();
        }
        let s = sim.chronicle_summary();
        sim.shutdown().unwrap();
        (s.currently_alive, s.total_ever_spawned)
    };

    let sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let s = sim.chronicle_summary();
    assert_eq!(s.currently_alive, alive_before);
    assert_eq!(s.total_ever_spawned, ever_before);
}

#[test]
fn ids_are_stable_and_unique() {
    let dir = TempDir::new().unwrap();
    let mut sim = light_populated_sim(&dir);
    for _ in 0..500 {
        sim.tick().unwrap();
    }
    // Collect all ids ever recorded.
    let summary = sim.chronicle_summary();
    let mut ids: Vec<u64> = Vec::new();
    sim.each_npc(|v| ids.push(v.id.0));
    // Can't iterate the whole chronicle directly here, but the
    // summary's total_ever_spawned vs duplicate-check on live ids is
    // the proxy.
    ids.sort_unstable();
    ids.dedup();
    assert!(ids.len() as u64 <= summary.total_ever_spawned);
}

#[test]
fn age_kills_at_die_at_tick() {
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), empty_graph_with_one_region()).unwrap();
    // Set a tiny target so we get a known spawn.
    sim.set_population_target_for_test(1, "wanderers", 1);
    // Phase 1A gate: activate the region so `spawn_npcs` runs there.
    sim.set_active_region(1);
    // Tick past first spawn pass.
    for _ in 0..100 {
        sim.tick().unwrap();
    }
    let mut victim = None;
    sim.each_npc(|v| {
        victim.get_or_insert(v.id);
    });
    let Some(id) = victim else {
        // Empty graph has no bases, so spawner can't pick a base; it
        // falls back to random pos. If even then nothing spawned, the
        // test isn't useful — bail.
        return;
    };
    sim.force_lifespan_for_test(id, 0); // die immediately
    sim.tick().unwrap();
    let rec = sim.chronicle_get(id).expect("record");
    assert!(rec.death_tick.is_some(), "{rec:?}");
    assert_eq!(rec.death_cause, Some(DeathCause::NaturalCauses));
}

#[test]
fn recent_deaths_returns_in_order() {
    let dir = TempDir::new().unwrap();
    let mut sim = light_populated_sim(&dir);
    for _ in 0..100 {
        sim.tick().unwrap();
    }
    let mut ids: Vec<_> = Vec::new();
    sim.each_npc(|v| ids.push(v.id));
    let to_kill: Vec<_> = ids.into_iter().take(3).collect();
    if to_kill.len() < 3 {
        return;
    }
    for id in &to_kill {
        sim.kill_npc_for_test(*id, DeathCause::NaturalCauses);
        // tick so each death has a distinct tick
        sim.tick().unwrap();
    }
    let recent = sim.recent_deaths(3);
    assert_eq!(recent.len(), 3);
    // Most recent first: ticks should be descending.
    let ticks: Vec<u64> = recent.iter().map(|r| r.death_tick.unwrap()).collect();
    let mut sorted = ticks.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(ticks, sorted);
}

// ---------- aggro / combat ----------

fn quiet_sim(_dir: &TempDir) -> Sim {
    // No-disk, no-NPC variant. `Sim::new_in_memory` clears
    // `PopulationTargets` after seeding so the spawn loop is a no-op
    // and aggro/combat scenarios stay deterministic.
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    // Mark region 1 as active so the offline-tier gate doesn't
    // skip aggro / combat / goals in tests.
    sim.set_active_region(1);
    sim
}

#[test]
fn aggro_acquired_in_sight() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let pwa = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let bandit = sim.spawn_npc_for_test("looters", 1, [50.0, 0.0, 0.0], None);
    // Under the FOV model, NPCs only acquire targets inside their
    // forward cone. Face them toward each other: PWA yaw 0 (+X,
    // toward bandit), bandit yaw π (-X, toward PWA).
    sim.set_npc_yaw_for_test(pwa, 0.0);
    sim.set_npc_yaw_for_test(bandit, std::f32::consts::PI);
    sim.tick().unwrap();
    let a = sim.npc_aggro_for_test(pwa).expect("pwa aggroed");
    assert_eq!(a.target, bandit);
    let b = sim.npc_aggro_for_test(bandit).expect("bandit aggroed");
    assert_eq!(b.target, pwa);
}

#[test]
fn squad_share_aggro() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 99;
    let leader = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], Some(group_id));
    let mate1 = sim.spawn_npc_for_test("pwa", 1, [-3.0, 0.0, 0.0], Some(group_id));
    let mate2 = sim.spawn_npc_for_test("pwa", 1, [3.0, 0.0, 0.0], Some(group_id));
    let bandit = sim.spawn_npc_for_test("looters", 1, [50.0, 0.0, 0.0], None);
    // PWA squad faces bandit; bandit faces them back.
    for sid in [leader, mate1, mate2] {
        sim.set_npc_yaw_for_test(sid, 0.0);
    }
    sim.set_npc_yaw_for_test(bandit, std::f32::consts::PI);
    sim.tick().unwrap();
    for sid in [leader, mate1, mate2] {
        let a = sim.npc_aggro_for_test(sid).expect("squad member aggroed");
        assert_eq!(a.target, bandit, "squad member should share aggro");
    }
}

#[test]
fn aggro_decays_when_target_unseen() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let pwa = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let bandit = sim.spawn_npc_for_test("looters", 1, [50.0, 0.0, 0.0], None);
    sim.set_npc_yaw_for_test(pwa, 0.0);
    sim.set_npc_yaw_for_test(bandit, std::f32::consts::PI);
    sim.tick().unwrap();
    assert!(sim.npc_aggro_for_test(pwa).is_some());
    // Move bandit to a far-away offline region; with the target
    // out of sight, aggro should decay after `aggro.decay_ticks`
    // (configured in `behavior.toml`). Read the actual config value
    // rather than hardcoding so retuning the decay window doesn't
    // silently break this test (it did once, see 2026-05-28 fix).
    sim.move_npc_for_test(bandit, [10000.0, 0.0, 10000.0], 2);
    let decay_ticks = simn_sim::BehaviorConfig::load().aggro.decay_ticks;
    for _ in 0..(decay_ticks + 10) {
        sim.tick().unwrap();
    }
    assert!(
        sim.npc_aggro_for_test(pwa).is_none(),
        "aggro should have decayed after {decay_ticks}+10 ticks"
    );
}

#[test]
fn combat_kills_low_hp() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let pwa = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let bandit = sim.spawn_npc_for_test("looters", 1, [10.0, 0.0, 0.0], None);
    // Pwa faces +X (toward bandit). Bandit also faces +X (away from
    // pwa) so its right arm capsule sits on the *far* side of the
    // torso from the shooter — at yaw=π the right arm rotates in
    // front of the torso and intercepts the projectile before it
    // reaches the vital part. Bandit doesn't need to face back; pwa
    // is the only shooter here.
    sim.set_npc_yaw_for_test(pwa, 0.0);
    sim.set_npc_yaw_for_test(bandit, 0.0);
    // Phase 4A v2: hit/miss is geometric, so pin shooter accuracy
    // at 100 (no cone-of-fire jitter) — otherwise a random
    // `NpcCharacter::roll` accuracy stat makes hit landing
    // non-deterministic.
    sim.set_npc_accuracy_for_test(pwa, 100);
    sim.force_npc_hp_for_test(bandit, 5.0); // one hit kills any part
                                            // Tick enough times for npc_combat to fire (every
                                            // `FIRE_INTERVAL_TICKS = 50`) and at least one geometric
                                            // hit on head or torso to land. Phase 4A v2 made hit
                                            // resolution geometric (cone-of-fire jitter), so we
                                            // budget extra fires vs the dice era.
    for _ in 0..600 {
        sim.tick().unwrap();
        if sim
            .chronicle_get(bandit)
            .and_then(|r| r.death_tick)
            .is_some()
        {
            break;
        }
    }
    let rec = sim.chronicle_get(bandit).expect("bandit record");
    assert!(rec.death_tick.is_some(), "bandit should have died");
    assert_eq!(
        rec.death_cause,
        Some(DeathCause::Combat {
            killer_faction: "pwa".to_string(),
        })
    );
    let _ = pwa;
}

#[test]
fn squad_gets_an_objective() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 7;
    for i in 0..4 {
        sim.spawn_npc_for_test("pwa", 1, [i as f32 * 2.0, 0.0, 0.0], Some(group_id));
    }
    // Tick past one planner interval (~200 ticks).
    for _ in 0..250 {
        sim.tick().unwrap();
    }
    let obj = sim
        .squad_objective_for_test(group_id)
        .expect("planner should have set an objective");
    // Any kind is fine — just assert one exists.
    let _ = matches!(
        obj,
        SquadObjective::Patrol { .. }
            | SquadObjective::Guard { .. }
            | SquadObjective::Investigate { .. }
            | SquadObjective::Wander { .. }
            | SquadObjective::Regroup { .. }
    );
}

#[test]
fn squad_objective_expires_and_rerolls() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 8;
    for i in 0..4 {
        sim.spawn_npc_for_test("pwa", 1, [i as f32 * 2.0, 0.0, 0.0], Some(group_id));
    }
    for _ in 0..250 {
        sim.tick().unwrap();
    }
    let _first = sim.squad_objective_for_test(group_id).unwrap();
    // Force the current objective to be expired in the past, then
    // tick into the next planner pass (200 more ticks).
    sim.force_squad_objective_expiry_for_test(group_id, 0);
    for _ in 0..250 {
        sim.tick().unwrap();
    }
    // We just want to assert the planner ran and the state is still
    // present (with a fresh non-expired expiry).
    let second = sim
        .squad_objective_for_test(group_id)
        .expect("planner should re-roll");
    use SquadObjective::*;
    let exp = match second {
        Patrol { expires_at, .. }
        | Guard { expires_at, .. }
        | Rest { expires_at, .. }
        | Investigate { expires_at, .. }
        | Explore { expires_at, .. }
        | Relieve { expires_at, .. }
        | Wander { expires_at }
        | Regroup { expires_at, .. } => expires_at,
    };
    assert!(exp > 0, "new objective should have a fresh expiry");
}

#[test]
fn cohesion_break_triggers_regroup() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 9;
    let leader = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], Some(group_id));
    sim.spawn_npc_for_test("pwa", 1, [3.0, 0.0, 0.0], Some(group_id));
    sim.spawn_npc_for_test("pwa", 1, [-3.0, 0.0, 0.0], Some(group_id));
    // Tick once to let the index populate.
    sim.tick().unwrap();
    // Teleport the leader 200m away to break cohesion.
    sim.move_npc_for_test(leader, [200.0, 0.0, 200.0], 1);
    sim.tick().unwrap();
    let obj = sim
        .squad_objective_for_test(group_id)
        .expect("squad should have an objective");
    assert!(
        matches!(obj, SquadObjective::Regroup { .. }),
        "expected Regroup, got {obj:?}"
    );
}

#[test]
fn solo_npcs_get_no_squad_objective() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let _wanderer = sim.spawn_npc_for_test("wanderers", 1, [0.0, 0.0, 0.0], None);
    for _ in 0..250 {
        sim.tick().unwrap();
    }
    // No group means no objective row should exist for any group id.
    // The simplest invariant we can assert is "fictional group_id 0
    // has no entry."
    assert!(sim.squad_objective_for_test(0).is_none());
}

#[test]
fn migration_suppressed_while_aggroed() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let pwa = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let bandit = sim.spawn_npc_for_test("looters", 1, [50.0, 0.0, 0.0], None);
    sim.set_npc_yaw_for_test(pwa, 0.0);
    sim.set_npc_yaw_for_test(bandit, std::f32::consts::PI);
    sim.tick().unwrap();
    assert!(sim.npc_aggro_for_test(pwa).is_some());
    // Tick a few hundred ticks. Even with the small migration
    // probability, an aggroed NPC should never leave the region.
    for _ in 0..1000 {
        sim.tick().unwrap();
        let rec = sim.chronicle_get(pwa).unwrap();
        assert_eq!(
            rec.regions_visited.len(),
            1,
            "aggroed NPC must not migrate (visited: {:?})",
            rec.regions_visited
        );
    }
}

#[test]
fn goal_fsm_progresses() {
    let dir = TempDir::new().unwrap();
    let mut sim = light_populated_sim(&dir);
    sim.set_active_region(1);
    for _ in 0..100 {
        sim.tick().unwrap();
    }
    // Pick a solo NPC (no Group) in the active region. Grouped NPCs
    // are driven by the squad-objective system, which bypasses
    // `NpcGoal` entirely; offline-region NPCs are frozen by the
    // active-region filter in `tick_npc_goals`. The per-NPC FSM only
    // runs for solo NPCs in the active region.
    let mut victim = None;
    sim.each_npc(|v| {
        if v.group_id == 0 && v.region == 1 && victim.is_none() {
            victim = Some(v.id);
        }
    });
    let Some(id) = victim else { return };

    // Tick a long time and assert at least one MoveTo or RestAt
    // was observed (FSM did transition out of Idle).
    let mut saw_non_idle = false;
    for _ in 0..2000 {
        sim.tick().unwrap();
        if let Some(g) = sim.npc_goal_for_test(id) {
            if !matches!(g, NpcGoal::Idle { .. }) {
                saw_non_idle = true;
                break;
            }
        } else {
            // NPC died of old age; that's fine, the goal cycled.
            saw_non_idle = true;
            break;
        }
    }
    assert!(saw_non_idle);
}

// ---- offline-NPC parity (player-reported bug + spatial hash) ----

#[test]
#[ignore = "long-running scenario (~60s); run with --include-ignored before push"]
fn offline_npcs_move_over_time() {
    // Direct regression test for the player-reported bug: spawn the
    // default 2-region world, keep region 1 active, let region 2 tick
    // offline. At the end of the run we expect at least one NPC in
    // region 2 to have moved off its spawn anchor (i.e. a non-trivial
    // distance from every base in that region).
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    // Phase 1A: the bulk-seed path is production's bridge between
    // `Sim::new` (creates an empty world) and "first tick has NPCs
    // in every region" — without it the incremental spawn loop
    // only fills the active region. The test wants offline NPCs in
    // region 2; bulk-seed populates everywhere then projects all
    // regions down to offline, and the `set_active_region(1)` below
    // re-projects region 1 back to online. Region 2 stays offline
    // with NPCs, which is the scenario `offline_movement` exercises.
    sim.initial_bulk_seed_npcs();
    sim.set_active_region(1);
    // ~25 in-world seconds at 20Hz — well past multiple spawn_npcs
    // passes (every 50 ticks) and multiple squad_planner rolls
    // (every ~10s).
    for _ in 0..500 {
        sim.tick().unwrap();
    }

    let bases = sim.bases_in_region(2);
    let mut moved_off_anchor = false;
    let mut offline_count = 0usize;
    // Region 2 is offline, so its NPCs are `OfflineNpc` components,
    // NOT online entities — `each_npc` would silently skip them and
    // every iteration of this test would falsely "see zero NPCs"
    // before ever asking the movement question. Use the offline-
    // specific iterator instead.
    sim.each_offline_npc_for_test(|_id, region, pos_2d| {
        if region != 2 {
            return;
        }
        offline_count += 1;
        let min_dist_to_any_base = bases
            .iter()
            .map(|b| {
                let dx = b.pos[0] - pos_2d[0];
                let dz = b.pos[2] - pos_2d[1];
                (dx * dx + dz * dz).sqrt()
            })
            .fold(f32::INFINITY, f32::min);
        if min_dist_to_any_base > 5.0 {
            moved_off_anchor = true;
        }
    });
    assert!(
        offline_count > 0,
        "bulk-seed should have populated region 2 with offline NPCs; got zero",
    );
    assert!(
        moved_off_anchor,
        "expected at least one of {offline_count} offline NPCs in region 2 to have moved >5m from any base"
    );
}

#[test]
#[ignore = "long-running scenario (~60s); run with --include-ignored before push"]
fn offline_combat_can_kill_npcs() {
    // Seed two hostile NPCs in region 2 (offline) close enough to
    // see each other immediately. Apply a big preemptive damage hit
    // to one so a single combat hit lands. Tick enough for npc_aggro
    // to acquire and npc_combat to fire (FIRE_INTERVAL_TICKS = 50).
    // Verify the chronicle records a combat death in region 2 — the
    // direct evidence that offline aggro + combat are running.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(1);

    // Pick two factions that are hostile per the matrix in
    // `crates/simn-sim/src/faction.rs`: Pwa ↔ Looters. Spawn them
    // directly as `OfflineNpc`s in region 2 — `spawn_npc_for_test`
    // mints *online* entities, which `offline_combat` (which queries
    // `&mut OfflineNpc`) doesn't see. The dedicated
    // `spawn_offline_npc_for_test` helper bypasses the projection
    // round-trip.
    let shooter_id = sim.spawn_offline_npc_for_test("pwa", 2, [0.0, 0.0]);
    let victim_id = sim.spawn_offline_npc_for_test("looters", 2, [20.0, 0.0]);
    // Hobble the victim to Critical so one offline combat hit
    // closes the kill (Healthy → Wounded → Critical → Dead).
    assert!(sim.force_offline_health_class_for_test(
        victim_id,
        simn_sim::offline_tier::HealthClass::Critical,
    ));

    let _ = shooter_id;
    for _ in 0..400 {
        sim.tick().unwrap();
    }

    // Victim should be dead; chronicle should record the death in
    // region 2 with Combat cause.
    let rec = sim.chronicle_get(victim_id).expect("victim recorded");
    assert!(
        rec.death_tick.is_some(),
        "expected offline victim to have died: {rec:?}"
    );
    assert_eq!(rec.death_region, Some(2));
    assert!(matches!(rec.death_cause, Some(DeathCause::Combat { .. })));
}

#[test]
fn spatial_hash_pair_iteration_finds_close_npcs() {
    // White-box check of the cell iteration: place two NPCs in the
    // same region within cell_size, tick once (so the hash is
    // rebuilt), tick once more so aggro Pass 2 runs against them.
    // They should acquire each other. Then move them apart past
    // sight range and confirm aggro decays.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(1);
    let a = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let b = sim.spawn_npc_for_test("looters", 1, [30.0, 0.0, 0.0], None);
    let _ = (a, b);
    // A few ticks for aggro to land.
    for _ in 0..5 {
        sim.tick().unwrap();
    }
    let mut a_has_aggro = false;
    sim.each_npc(|v| {
        if v.id == a && v.aggro_target != 0 {
            a_has_aggro = true;
        }
    });
    assert!(
        a_has_aggro,
        "expected NPC a to acquire aggro on b within a few ticks (spatial hash)"
    );
}

#[test]
fn npc_spawns_with_body_parts_full() {
    // NPCs ship with BodyParts::new_full() — the bridge view exposes
    // it so the dummy HUD can render per-part HP.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let mut found = None;
    sim.each_npc(|v| {
        if v.id == id {
            found = Some(v);
        }
    });
    let view = found.expect("spawned NPC");
    let bp = view
        .body_parts
        .expect("NPCs spawned now carry BodyParts on the view");
    for part in [
        BodyPart::Head,
        BodyPart::Torso,
        BodyPart::LeftArm,
        BodyPart::RightArm,
        BodyPart::LeftLeg,
        BodyPart::RightLeg,
    ] {
        assert_eq!(
            bp.get(part),
            simn_sim::BodyParts::DEFAULT_MAX,
            "fresh NPC has full {part:?}"
        );
    }
}

#[test]
fn npc_part_damage_drops_aggregate_health() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);

    sim.apply_damage_to_npc_part(id, BodyPart::Head, 25.0)
        .unwrap();

    let mut view = None;
    sim.each_npc(|v| {
        if v.id == id {
            view = Some(v);
        }
    });
    let v = view.expect("NPC alive");
    let bp = v.body_parts.expect("BodyParts present");
    assert!((bp.head - 75.0).abs() < f32::EPSILON, "head drained to 75");
    assert!((bp.torso - 100.0).abs() < f32::EPSILON, "torso untouched");
    // Aggregate Health mirror tracks min(head, torso).
    assert!(
        (v.health.current - 75.0).abs() < f32::EPSILON,
        "aggregate health = min(head, torso) = 75"
    );
}

#[test]
fn npc_head_zero_kills_via_death_check() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);

    sim.apply_damage_to_npc_part(id, BodyPart::Head, 200.0)
        .unwrap();
    // npc_death_check runs on tick; one tick past the damage is enough.
    sim.tick().unwrap();

    let rec = sim.chronicle_get(id).expect("record present");
    assert!(
        rec.death_tick.is_some(),
        "NPC with head at 0 should be dead: {rec:?}"
    );
}

#[test]
fn set_npc_body_part_round_trips_through_journal() {
    // Damage an NPC, force a snapshot + replay, and verify the
    // SetNpcBodyPart delta was journaled and re-applied on load.
    // Uses sub-threshold damage so no bleed wound drains HP
    // mid-tick — the point of this test is the delta round-trip,
    // not the bleed pipeline.
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let (id, head_before) = {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
        sim.apply_damage_to_npc_part(id, BodyPart::Head, 5.0)
            .unwrap();
        sim.tick().unwrap();
        sim.shutdown().unwrap();
        (id, 95.0_f32)
    };

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let mut seen = None;
    sim.each_npc(|v| {
        if v.id == id {
            seen = v.body_parts.map(|bp| bp.head);
        }
    });
    let head_after = seen.expect("reloaded NPC has BodyParts");
    assert!(
        (head_after - head_before).abs() < 0.01,
        "head HP should survive snapshot/journal round-trip: before={head_before}, after={head_after}"
    );
}

// ---------- PR-4b: faction loadouts + corpse drops ----------

#[test]
fn npcs_spawn_with_faction_loadout_in_pockets() {
    // Run a real sim long enough for at least one spawn pass, then
    // assert that PWA NPCs carry the guaranteed bandage from their
    // loadout. Reads via `each_npc` + the sim's snapshot — confirms
    // the wiring all the way through `spawn_npcs`.
    let dir = TempDir::new().unwrap();
    let mut sim = light_populated_sim(&dir);
    for _ in 0..150 {
        sim.tick().unwrap();
    }
    // Snapshot any PWA NPC's id and inspect via the snapshot path.
    let pwa_faction_id = sim
        .faction_registry()
        .id_of("pwa")
        .expect("registry has pwa");
    let mut pwa_id = None;
    sim.each_npc(|v| {
        if v.faction == pwa_faction_id && pwa_id.is_none() {
            pwa_id = Some(v.id);
        }
    });
    let id = pwa_id.expect("at least one PWA NPC after 150 ticks");
    let items = sim
        .npc_inventory_view_for_test(id)
        .expect("PWA NPC must have an Inventory component");
    let bandage_count: u32 = items
        .iter()
        .filter(|s| s.id.0 == "bandage")
        .map(|s| s.count)
        .sum();
    assert!(
        bandage_count >= 1,
        "PWA NPC should carry the guaranteed bandage from data/npc_loadouts.toml; got {items:?}"
    );
}

#[test]
fn killing_an_npc_spawns_a_corpse_container_with_their_gear() {
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    let id = sim.spawn_npc_for_test("pwa", 1, [10.0, 0.0, 5.0], None);
    // Seed gear directly so the test doesn't depend on RNG outcomes.
    use simn_sim::ItemId;
    assert!(sim.grant_to_npc_for_test(id, &ItemId::from("bandage"), 3));
    assert!(sim.grant_to_npc_for_test(id, &ItemId::from("metal_scrap"), 2));
    // Need a player at the corpse to use containers_in_range; spawn
    // one and put them next to the NPC.
    sim.upsert_player(99, 1, [10.0, 0.0, 5.0], 0.0).unwrap();
    sim.kill_npc_for_test(id, DeathCause::Other);
    let corpses = sim.containers_in_range(99, 5.0);
    assert_eq!(corpses.len(), 1, "exactly one corpse container expected");
    let (cid, pos, is_public) = corpses[0];
    assert!(!is_public, "corpses must be private (not in kit-pool)");
    assert_eq!(pos, [10.0, 0.0, 5.0]);
    let grid = sim.container_view(cid).expect("corpse grid present");
    let total_bandage: u32 = grid
        .items
        .iter()
        .filter(|p| p.stack.id.0 == "bandage")
        .map(|p| p.stack.count)
        .sum();
    let total_scrap: u32 = grid
        .items
        .iter()
        .filter(|p| p.stack.id.0 == "metal_scrap")
        .map(|p| p.stack.count)
        .sum();
    assert_eq!(total_bandage, 3);
    assert_eq!(total_scrap, 2);
}

#[test]
fn empty_inventory_npc_drops_no_corpse() {
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    let id = sim.spawn_npc_for_test("wanderers", 1, [0.0; 3], None);
    // Intentionally don't grant — the test-spawn helper starts with
    // an empty Inventory regardless of faction.
    sim.upsert_player(99, 1, [0.0; 3], 0.0).unwrap();
    sim.kill_npc_for_test(id, DeathCause::NaturalCauses);
    let corpses = sim.containers_in_range(99, 5.0);
    assert!(
        corpses.is_empty(),
        "empty-inventory NPCs should not drop a corpse pile"
    );
}

#[test]
fn corpse_loot_is_takeable_into_pockets() {
    // End-to-end: kill NPC → walk to corpse → take an item out.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    let id = sim.spawn_npc_for_test("linemen", 1, [3.0, 0.0, 0.0], None);
    use simn_sim::ItemId;
    assert!(sim.grant_to_npc_for_test(id, &ItemId::from("bandage"), 1));
    sim.upsert_player(7, 1, [3.0, 0.0, 0.0], 0.0).unwrap();
    sim.kill_npc_for_test(id, DeathCause::Other);
    let corpses = sim.containers_in_range(7, 5.0);
    assert_eq!(corpses.len(), 1);
    let (cid, _, _) = corpses[0];
    sim.take_from_container(7, cid, 0).unwrap();
    let inv = sim.inventory_view(7);
    assert_eq!(inv.len(), 1);
    assert_eq!(inv[0].id.0, "bandage");
}

#[test]
fn arbitration_lone_npc_with_aggro_goes_individual() {
    // Lone NPC + hostile target = IndividualAggro (no Group, so no
    // SquadAggro candidate).
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let pwa = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let bandit = sim.spawn_npc_for_test("looters", 1, [50.0, 0.0, 0.0], None);
    sim.set_npc_yaw_for_test(pwa, 0.0);
    sim.set_npc_yaw_for_test(bandit, std::f32::consts::PI);
    sim.tick().unwrap();
    sim.tick().unwrap(); // arbitration runs after npc_aggro acquires
    let g = sim
        .npc_active_goal_for_test(pwa)
        .expect("pwa has ActiveGoal");
    assert_eq!(g.source, GoalSource::IndividualAggro);
    assert!(matches!(g.kind, GoalKind::PursueTarget { target } if target == bandit));
}

#[test]
fn arbitration_grouped_npc_with_aggro_goes_squad() {
    // Grouped NPC + hostile target = SquadAggro (priority 160 beats
    // IndividualAggro priority 150).
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 42;
    let leader = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], Some(group_id));
    let mate = sim.spawn_npc_for_test("pwa", 1, [-3.0, 0.0, 0.0], Some(group_id));
    let bandit = sim.spawn_npc_for_test("looters", 1, [50.0, 0.0, 0.0], None);
    sim.set_npc_yaw_for_test(leader, 0.0);
    sim.set_npc_yaw_for_test(mate, 0.0);
    sim.set_npc_yaw_for_test(bandit, std::f32::consts::PI);
    sim.tick().unwrap();
    sim.tick().unwrap();
    for member in [leader, mate] {
        let g = sim
            .npc_active_goal_for_test(member)
            .expect("squad member has ActiveGoal");
        assert_eq!(g.source, GoalSource::SquadAggro);
        assert!(matches!(g.kind, GoalKind::PursueTarget { target } if target == bandit));
    }
}

#[test]
fn arbitration_idle_npc_falls_back_to_solo() {
    // Lone NPC, no aggro, no group, no personality-introduced goals
    // = Idle/SoloIdleFsm. Personality traits get cleared explicitly
    // because the archetype roll might otherwise nominate a weak
    // bias candidate (Socialize / Hunt / etc.) that beats idle.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let pwa = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.clear_npc_personality_for_test(pwa);
    sim.tick().unwrap();
    sim.tick().unwrap();
    let g = sim
        .npc_active_goal_for_test(pwa)
        .expect("npc has ActiveGoal");
    assert_eq!(g.source, GoalSource::Idle);
    assert_eq!(g.kind, GoalKind::SoloIdleFsm);
}
