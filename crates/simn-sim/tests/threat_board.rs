//! Squad threat board integration tests. Per
//! `docs/book/src/planning/threat-board-plan.md` step 2 — covers
//! the sweep system that aggregates per-NPC `RecentAttackers` into
//! the per-squad `BlackboardKey::ThreatList`.

use simn_sim::{BlackboardKey, BlackboardValue, RegionGraph, Sim};
use tempfile::TempDir;

/// Quiet sim with no auto-spawning. `Sim::new_in_memory` clears
/// `PopulationTargets` after seeding so the spawn loop is a no-op.
fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn squad_threat_board_aggregates_member_attackers() {
    // Two grouped Coalition NPCs each take damage from a different
    // attacker. After one tick (sweep_threats runs at the top),
    // the squad's ThreatList should contain BOTH attackers.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 99;
    let mate1 = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));
    let mate2 = sim.spawn_npc_for_test("coalition", 1, [10.0, 0.0, 0.0], Some(group_id));
    // Spawn the attackers so the position index has them — proximity
    // factor reads from there.
    let attacker_a = sim.spawn_npc_for_test("looters", 1, [5.0, 0.0, 5.0], None);
    let attacker_b = sim.spawn_npc_for_test("looters", 1, [15.0, 0.0, 5.0], None);

    let now = sim.current_tick();
    sim.record_npc_hit_for_test(mate1, attacker_a, now, 25.0);
    sim.record_npc_hit_for_test(mate2, attacker_b, now, 10.0);

    sim.tick().unwrap();

    let bb = sim.squad_blackboard(group_id).expect("group has bb");
    let entry = bb.get(&BlackboardKey::ThreatList).expect("threat list set");
    let BlackboardValue::Threats(threats) = &entry.value else {
        panic!("expected Threats value, got {:?}", entry.value);
    };
    assert_eq!(threats.len(), 2, "both attackers in threat board");
    let ids: Vec<_> = threats.iter().map(|t| t.target_id).collect();
    assert!(ids.contains(&attacker_a));
    assert!(ids.contains(&attacker_b));
}

#[test]
fn squad_threat_board_top_score_is_highest_damage() {
    // Same setup as above but attacker A does 3× the damage of B.
    // A should top the list by score (assuming similar proximity).
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 42;
    let mate = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));
    // Place both attackers at the same distance so proximity factors
    // are identical and damage drives the score ordering.
    let attacker_a = sim.spawn_npc_for_test("looters", 1, [5.0, 0.0, 5.0], None);
    let attacker_b = sim.spawn_npc_for_test("looters", 1, [-5.0, 0.0, 5.0], None);

    let now = sim.current_tick();
    sim.record_npc_hit_for_test(mate, attacker_a, now, 90.0);
    sim.record_npc_hit_for_test(mate, attacker_b, now, 30.0);

    sim.tick().unwrap();

    let bb = sim.squad_blackboard(group_id).expect("group has bb");
    let entry = bb.get(&BlackboardKey::ThreatList).expect("threat list set");
    let BlackboardValue::Threats(threats) = &entry.value else {
        panic!("expected Threats");
    };
    assert_eq!(
        threats.first().unwrap().target_id,
        attacker_a,
        "highest damage = top score"
    );
}

#[test]
fn solo_npc_has_no_threat_board() {
    // No Group → no aggregation; sweep should not write anything
    // for an unaffiliated NPC even with damage on file.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let solo = sim.spawn_npc_for_test("nomads", 1, [0.0, 0.0, 0.0], None);
    let attacker = sim.spawn_npc_for_test("looters", 1, [5.0, 0.0, 0.0], None);
    sim.record_npc_hit_for_test(solo, attacker, sim.current_tick(), 50.0);

    sim.tick().unwrap();

    // The grouped sibling test (`squad_threat_board_aggregates_member_attackers`)
    // proves a hit on a *grouped* member produces a ThreatList on that
    // group's blackboard. The invariant here is the mirror: an
    // unaffiliated NPC (no Group) must NOT produce any threat-board
    // entry — the sweep gates on `Some(Group)` and skips ungrouped
    // NPCs. Solo NPCs surface as group_id 0 in the view; the spawn
    // above is the only NPC in the world with damage on file, so if
    // any board exists for group 0 the gate has regressed.
    let mut solo_group = None;
    sim.each_npc(|v| {
        if v.id == solo {
            solo_group = Some(v.group_id);
        }
    });
    let gid = solo_group.expect("solo NPC must be alive after the sweep");
    assert_eq!(
        gid, 0,
        "an NPC spawned with no group must report group_id 0"
    );
    assert!(
        sim.squad_blackboard(gid).is_none(),
        "ungrouped NPC must not produce a squad threat board, but group {gid} has one",
    );
}

#[test]
fn aggregated_score_falls_off_with_recency() {
    // Hit recorded long ago should score lower than a recent hit
    // of identical damage from a different attacker.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 7;
    let mate = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));
    let stale_attacker = sim.spawn_npc_for_test("looters", 1, [5.0, 0.0, 5.0], None);
    let fresh_attacker = sim.spawn_npc_for_test("looters", 1, [-5.0, 0.0, 5.0], None);

    // Record the stale hit way in the past via a synthetic tick. We
    // can't easily fast-forward the clock in tests; instead, push
    // the stale hit at tick 0, advance ~half the TTL, then push the
    // fresh hit. Recency factor on stale will be ~0.5; on fresh
    // ~1.0 → fresh wins.
    let stale_tick = sim.current_tick();
    sim.record_npc_hit_for_test(mate, stale_attacker, stale_tick, 50.0);
    // Tick forward to age the stale event. THREAT_TTL_TICKS = 600;
    // 100 ticks ≈ 1/6 of TTL → stale recency ≈ 0.83.
    for _ in 0..100 {
        sim.tick().unwrap();
    }
    let fresh_tick = sim.current_tick();
    sim.record_npc_hit_for_test(mate, fresh_attacker, fresh_tick, 50.0);

    sim.tick().unwrap();

    let bb = sim.squad_blackboard(group_id).expect("group has bb");
    let entry = bb.get(&BlackboardKey::ThreatList).expect("threat list set");
    let BlackboardValue::Threats(threats) = &entry.value else {
        panic!("expected Threats");
    };
    assert_eq!(
        threats.first().unwrap().target_id,
        fresh_attacker,
        "fresh hit should outrank stale despite equal damage",
    );
}

#[test]
fn squad_aggro_switches_to_top_threat_when_dominant() {
    // Squad currently aggro'd on attacker_b. attacker_a starts
    // doing 3× the damage. After the next sweep + apply_threat_priority
    // tick, the squad should switch focus to attacker_a.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 11;
    let mate1 = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));
    let mate2 = sim.spawn_npc_for_test("coalition", 1, [10.0, 0.0, 0.0], Some(group_id));
    let attacker_a = sim.spawn_npc_for_test("looters", 1, [5.0, 0.0, 5.0], None);
    let attacker_b = sim.spawn_npc_for_test("looters", 1, [-5.0, 0.0, 5.0], None);

    // Squad starts aggro'd on B (e.g., perception saw B first).
    sim.set_npc_aggro_for_test(mate1, attacker_b);
    sim.set_npc_aggro_for_test(mate2, attacker_b);
    // B has done some prior damage (so it's in the threat list).
    let now = sim.current_tick();
    sim.record_npc_hit_for_test(mate1, attacker_b, now, 30.0);
    // A starts hitting hard.
    sim.record_npc_hit_for_test(mate1, attacker_a, now, 90.0);
    sim.record_npc_hit_for_test(mate2, attacker_a, now, 90.0);

    sim.tick().unwrap();

    let aggro1 = sim.npc_aggro_for_test(mate1).expect("mate1 aggro");
    let aggro2 = sim.npc_aggro_for_test(mate2).expect("mate2 aggro");
    assert_eq!(
        aggro1.target, attacker_a,
        "mate1 should switch to dominant new threat"
    );
    assert_eq!(
        aggro2.target, attacker_a,
        "mate2 should switch to dominant new threat"
    );
}

#[test]
fn squad_aggro_holds_under_hysteresis() {
    // Squad aggro'd on A (100 dmg). New attacker B does 70 dmg —
    // 0.7× ratio fails the 1.5× threshold and the +2 absolute delta
    // is satisfied (70 vs 100 → no, 70 < 102), so squad should HOLD.
    // Wait: 70 < 100, so the hysteresis check is "top.score >=
    // current.score * 1.5" or "top.score >= current.score + 2.0".
    // top=70, current=100 → top is LOWER, never preempts. Test
    // confirms the obvious case.
    //
    // The interesting case: top=120, current=100. 1.5× → no
    // (120 < 150); +2 absolute → no (120 < 102 wait that's >). So
    // 120 vs 100 has 120 ≥ 102 absolute delta → DOES switch. Need
    // to reduce delta. Try top=101, current=100: 101 < 150 (no),
    // 101 < 102 (no). Holds. That's the hysteresis case.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let group_id = 13;
    let mate = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));
    let attacker_a = sim.spawn_npc_for_test("looters", 1, [5.0, 0.0, 5.0], None);
    let attacker_b = sim.spawn_npc_for_test("looters", 1, [-5.0, 0.0, 5.0], None);

    sim.set_npc_aggro_for_test(mate, attacker_a);
    let now = sim.current_tick();
    // A: 100 damage. B: 101 damage. Hysteresis: 101 < 150 (no
    // multiplier) AND 101 < 102 (no absolute) → hold.
    sim.record_npc_hit_for_test(mate, attacker_a, now, 100.0);
    sim.record_npc_hit_for_test(mate, attacker_b, now, 101.0);

    sim.tick().unwrap();

    let aggro = sim.npc_aggro_for_test(mate).expect("mate aggro");
    assert_eq!(
        aggro.target, attacker_a,
        "hysteresis should hold under tiny score delta"
    );
}

#[test]
fn ungrouped_npc_unaffected_by_threat_board() {
    // Solo NPC has no Group, so apply_threat_priority should leave
    // its Aggro.target alone even if a (nonexistent) threat board
    // would have flipped it. Sanity check that the system gates on
    // Group correctly.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let solo = sim.spawn_npc_for_test("nomads", 1, [0.0, 0.0, 0.0], None);
    let original_target = sim.spawn_npc_for_test("looters", 1, [5.0, 0.0, 0.0], None);
    let other = sim.spawn_npc_for_test("looters", 1, [-5.0, 0.0, 0.0], None);

    sim.set_npc_aggro_for_test(solo, original_target);
    sim.record_npc_hit_for_test(solo, other, sim.current_tick(), 1000.0);

    sim.tick().unwrap();

    let aggro = sim.npc_aggro_for_test(solo).expect("solo aggro");
    assert_eq!(
        aggro.target, original_target,
        "solo NPC keeps Aggro.target unchanged",
    );
}
