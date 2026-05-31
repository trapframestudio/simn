//! Offline combat tests (Phase 1E of `sim-iteration-5-12-plan.md`).
//!
//! Acceptance: two opposing-faction squads tick offline for 60 s of
//! sim time → at least one death + chronicle entry + `AllyDown` event
//! pushed to the world event bus.

use simn_sim::{DeathCause, RegionGraph, SavePaths, Sim, OFFLINE_TIER_TICK_DIVISOR};
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
fn two_hostile_offline_squads_eventually_produce_a_death() {
    // Region 1 is offline (we activate region 2). Plant 8 coalition + 8
    // raiders in close range so engagement fires and the dice cycle
    // runs hot. Coalition vs raiders are Hostile per `factions.toml`.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(2);

    // Cluster both squads near each other inside `OFFLINE_ENGAGEMENT_RADIUS_M`
    // so the dice fire every offline tick.
    for i in 0..8 {
        let dx = (i as f32) * 4.0;
        sim.spawn_offline_npc_for_test("coalition", 1, [dx, 0.0]);
        sim.spawn_offline_npc_for_test("raiders", 1, [dx + 1.0, 50.0]);
    }

    // 120 offline ticks ≈ 60 s of sim wall time. With per-tick hit
    // chance ~0.05 and three HealthClass steps to kill, expected
    // first death lands well before 60 ticks; 120 gives plenty of
    // headroom for the dice variance.
    tick_offline_ticks(&mut sim, 120);

    // At least one death recorded.
    let deaths = sim.recent_deaths(64);
    assert!(
        !deaths.is_empty(),
        "expected at least one offline combat death after 120 offline ticks, chronicle had 0"
    );
    let combat_death = deaths
        .iter()
        .find(|r| matches!(r.death_cause, Some(DeathCause::Combat { .. })));
    assert!(
        combat_death.is_some(),
        "at least one death should be tagged as Combat (got {:?})",
        deaths.iter().map(|r| &r.death_cause).collect::<Vec<_>>()
    );
}

#[test]
fn allied_factions_dont_fight_offline() {
    // Same-faction NPCs never engage even when close together.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(2);

    for i in 0..6 {
        sim.spawn_offline_npc_for_test("coalition", 1, [(i as f32) * 3.0, 0.0]);
    }

    tick_offline_ticks(&mut sim, 60);

    let deaths = sim.recent_deaths(32);
    let combat_deaths: Vec<_> = deaths
        .iter()
        .filter(|r| matches!(r.death_cause, Some(DeathCause::Combat { .. })))
        .collect();
    assert!(
        combat_deaths.is_empty(),
        "same-faction NPCs shouldn't kill each other offline, got {} combat deaths",
        combat_deaths.len()
    );
}

#[test]
fn offline_combat_degrades_health_class_before_killing() {
    // Single pair, single damage roll over time — observe Healthy
    // → Wounded → Critical → death transitions. We can't deterministic-
    // ally pin a single roll, but with 30 offline ticks and 0.05 hit
    // chance, at least one degradation is overwhelmingly likely
    // (P(no hit) = 0.95^30 ≈ 0.21).
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(2);

    let a = sim.spawn_offline_npc_for_test("coalition", 1, [0.0, 0.0]);
    let b = sim.spawn_offline_npc_for_test("raiders", 1, [5.0, 0.0]);

    // Before any ticks both NPCs are Healthy.
    assert!(matches!(
        sim.offline_npc_for_test(a).unwrap().health_class,
        simn_sim::HealthClass::Healthy
    ));
    assert!(matches!(
        sim.offline_npc_for_test(b).unwrap().health_class,
        simn_sim::HealthClass::Healthy
    ));

    // 30 offline ticks: ~95% chance of seeing at least one
    // degradation. We do 60 to make this effectively certain
    // (P(no hit in 60) = 0.95^60 ≈ 0.046).
    tick_offline_ticks(&mut sim, 60);

    let a_state = sim.offline_npc_for_test(a);
    let b_state = sim.offline_npc_for_test(b);
    // At least one of them: either died (no longer in offline schema)
    // OR has Wounded / Critical HealthClass.
    let a_progressed = a_state
        .as_ref()
        .map(|o| !matches!(o.health_class, simn_sim::HealthClass::Healthy))
        .unwrap_or(true);
    let b_progressed = b_state
        .as_ref()
        .map(|o| !matches!(o.health_class, simn_sim::HealthClass::Healthy))
        .unwrap_or(true);
    assert!(
        a_progressed || b_progressed,
        "after 60 offline ticks at least one combatant should be wounded/critical/dead"
    );
}

#[test]
fn neutral_factions_dont_engage_offline() {
    // Some faction pairs default to Neutral, not Hostile (e.g. Coalition
    // vs Directorate — verify by reading the registry). If we pick a
    // known non-hostile pair, no combat should fire.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    sim.set_active_region(2);

    // Use nomads vs coalition — looking at factions.toml, nomads are
    // typically Cold/Neutral with most factions. If nomads turn
    // out to be hostile to coalition, this assertion will need a different
    // pair; the test still verifies the "non-hostile → no engagement"
    // contract.
    let registry = sim.faction_registry();
    let coalition = registry.id_of("coalition").unwrap();
    let wand = registry.id_of("nomads").unwrap();
    let relation = sim.faction_relation(coalition, wand);
    assert!(
        !matches!(relation, simn_sim::Relation::Hostile),
        "test premise: coalition<->nomads must be non-hostile (got {relation:?}); \
         pick a different non-hostile pair if the registry changed"
    );

    for i in 0..6 {
        sim.spawn_offline_npc_for_test("coalition", 1, [(i as f32) * 3.0, 0.0]);
        sim.spawn_offline_npc_for_test("nomads", 1, [(i as f32) * 3.0 + 1.0, 0.0]);
    }

    tick_offline_ticks(&mut sim, 60);

    let deaths = sim.recent_deaths(32);
    let combat_deaths: Vec<_> = deaths
        .iter()
        .filter(|r| matches!(r.death_cause, Some(DeathCause::Combat { .. })))
        .collect();
    assert!(
        combat_deaths.is_empty(),
        "non-hostile factions shouldn't engage, got {} combat deaths",
        combat_deaths.len()
    );
}
