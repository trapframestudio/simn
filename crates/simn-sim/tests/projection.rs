//! Projection tests (Phase 1C of `sim-iteration-5-12-plan.md`).
//!
//! The invariant under test: an NPC is an online entity iff its
//! region is in `ActiveRegions`. Crossing the boundary projects state
//! one way and despawns the source side.
//!
//! Specifically:
//! - `set_active_region(X)` projects every online NPC in regions
//!   leaving `ActiveRegions` to `OfflineNpc`.
//! - Same call projects every `OfflineNpc` in X back up to the full
//!   online schema.
//! - Identity (NpcId, faction, group) survives the round-trip.
//! - Health-class survives: a wounded NPC re-materializes wounded.

use simn_sim::{BodyPart, HealthClass, RegionGraph, SavePaths, Sim};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

/// One-region fresh sim with no procedural seeding noise — we
/// `spawn_npc_for_test` the NPC we care about so the assertions stay
/// scoped to a single deterministic entity.
fn fresh_sim(dir: &TempDir) -> Sim {
    Sim::new(paths(dir), RegionGraph::default_test_graph()).unwrap()
}

#[test]
fn set_active_region_demotes_inactive_region_npcs_to_offline() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    // Spawn an NPC in region 2 and activate region 1. Region 2's
    // NPC should be in the offline schema after the activation
    // because the invariant is "online entity ⇔ region active."
    let id = sim.spawn_npc_for_test("coalition", 2, [10.0, 0.0, -5.0], None);
    sim.set_active_region(2);
    assert!(
        sim.offline_npc_for_test(id).is_none(),
        "after set_active_region(2), the NPC should be online (offline_npc_for_test returns None)"
    );

    // Flip to region 1. Region 2 transitions offline → its NPC
    // becomes an OfflineNpc.
    sim.set_active_region(1);
    let offline = sim
        .offline_npc_for_test(id)
        .expect("NPC should be projected to offline after deactivation");
    assert_eq!(offline.region, 2, "region preserved across projection");
    assert!(
        matches!(offline.health_class, HealthClass::Healthy),
        "fresh NPC starts at full HP → HealthClass::Healthy, got {:?}",
        offline.health_class
    );
}

#[test]
fn online_offline_online_round_trip_preserves_identity_and_health_class() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_active_region(1);
    // Spawn fresh in the active region.
    let id = sim.spawn_npc_for_test("coalition_vanguard", 1, [0.0, 0.0, 0.0], None);
    // Damage one leg so the NPC will project as Wounded.
    sim.set_npc_body_part_for_test(id, BodyPart::LeftLeg, 40.0);

    // Capture the online faction before projection.
    let faction_before = sim.npc_in_faction_for_test(id).expect("InFaction present");

    // Flip: region 1 → offline (NPC goes offline), region 2 → online.
    sim.set_active_region(2);
    let offline = sim
        .offline_npc_for_test(id)
        .expect("NPC should be offline after deactivation");
    assert_eq!(offline.id, id, "NpcId preserved");
    assert_eq!(offline.faction, faction_before.0, "faction preserved");
    assert!(
        matches!(offline.health_class, HealthClass::Wounded),
        "wounded NPC projects as Wounded, got {:?}",
        offline.health_class
    );

    // Flip back: region 1 → online.
    sim.set_active_region(1);
    // No `OfflineNpc` should remain for `id` after re-projection.
    assert!(
        sim.offline_npc_for_test(id).is_none(),
        "NPC should be back online after re-activation"
    );
    // Online entity should still exist with the same id + faction.
    let faction_after = sim
        .npc_in_faction_for_test(id)
        .expect("NPC re-materialized with same id");
    assert_eq!(
        faction_after.0, faction_before.0,
        "faction preserved across full round-trip"
    );

    // Body parts re-materialize from HealthClass. A `Wounded` re-
    // projection should still produce at least one limb below the
    // wounded-threshold but vital parts intact.
    let mut bp_seen = None;
    sim.each_npc(|v| {
        if v.id == id {
            bp_seen = v.body_parts;
        }
    });
    let bp = bp_seen.expect("re-materialized NPC has BodyParts");
    let min_part = bp
        .head
        .min(bp.torso)
        .min(bp.left_arm)
        .min(bp.right_arm)
        .min(bp.left_leg)
        .min(bp.right_leg);
    assert!(
        min_part < 75.0,
        "Wounded re-projection should produce some damaged limb, all-min was {min_part}"
    );
    assert!(
        bp.head >= 25.0 && bp.torso >= 25.0,
        "Wounded class shouldn't materialize critical-vital damage (head={}, torso={})",
        bp.head,
        bp.torso
    );
}

#[test]
fn projection_preserves_squad_group_id() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_active_region(1);
    let group_id: u64 = 0x1234_5678_DEAD_BEEF;
    let id = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], Some(group_id));

    sim.set_active_region(2);
    let offline = sim
        .offline_npc_for_test(id)
        .expect("offline projection should exist");
    assert_eq!(offline.group, Some(group_id), "group preserved offline");

    sim.set_active_region(1);
    // After re-projection, the NPC's online entity should have the
    // Group component back. We don't have a direct accessor for
    // `Group` in `world::debug`, but the NpcId returning in
    // `each_npc` confirms the entity exists; squad cohesion is
    // exercised in `npcs.rs` and `npc_aggro.rs` once Phase 1E
    // wires offline_combat.
    let mut still_exists = false;
    sim.each_npc(|v| {
        if v.id == id {
            still_exists = true;
        }
    });
    assert!(
        still_exists,
        "NPC should re-materialize online after region re-activation"
    );
}

#[test]
fn critical_npc_projects_critical() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_active_region(1);
    let id = sim.spawn_npc_for_test("directorate", 1, [0.0, 0.0, 0.0], None);
    // Torso below 25% → Critical per `body_parts_to_health_class`.
    sim.set_npc_body_part_for_test(id, BodyPart::Torso, 10.0);

    sim.set_active_region(2);
    let offline = sim
        .offline_npc_for_test(id)
        .expect("offline projection should exist");
    assert!(
        matches!(offline.health_class, HealthClass::Critical),
        "torso=10 should produce Critical, got {:?}",
        offline.health_class
    );
}

#[test]
fn redundant_set_active_region_is_a_no_op() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_active_region(1);
    let id = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], None);
    // Calling set_active_region(1) again should not project anything;
    // the NPC should remain online.
    sim.set_active_region(1);
    sim.set_active_region(1);
    assert!(
        sim.offline_npc_for_test(id).is_none(),
        "NPC stays online when set_active_region is called redundantly"
    );
}
