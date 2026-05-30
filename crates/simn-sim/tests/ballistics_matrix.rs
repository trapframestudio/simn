//! Pen-vs-armor damage matrix tests. For each round class (HP /
//! FMJ / AP) × armor class combination, fire a projectile at a
//! test NPC and assert the damage lands in the expected band. We
//! use ranges (`> 0.8 * max`, `< 0.2 * max`, etc.) instead of
//! exact floats so tuning passes don't cascade into test
//! breakage.
//!
//! The tests plant the projectile directly via the
//! `ProjectileSpawned` delta (same replay path commit 2's
//! projectile tests use) so they don't depend on the fire path
//! being wired in a particular way.

use simn_sim::{BodyPart, ItemId, ProjectileId, RegionGraph, Sim, WorldDelta};
use tempfile::TempDir;

fn fresh_sim(_dir: &TempDir) -> Sim {
    Sim::new_in_memory(RegionGraph::default_test_graph())
}
fn id(s: &str) -> ItemId {
    ItemId::from(s)
}

/// Spawn a projectile with the given round aimed straight at the
/// NPC at `(0, head_height, npc_z)` and tick the sim forward
/// enough to resolve the impact. Returns the `ProjectileImpacted`
/// delta (host-generated).
fn fire_straight_at_part(
    round_id: &str,
    target_y: f32,
    npc_armor: &[&str],
    npc_head_gear: Option<&str>,
    part: BodyPart,
) -> (f32, bool) {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.upsert_player(1, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
    let npc = sim.spawn_npc_for_test("looters", 1, [0.0, 0.0, 5.0], None);

    // Equip armor on the NPC if any.
    if !npc_armor.is_empty() || npc_head_gear.is_some() {
        sim.equip_test_armor_on_npc(npc, npc_armor, npc_head_gear)
            .expect("equip test armor");
    }

    // Spawn a projectile via the delta path. The replay arm
    // spawns the Projectile entity in-world for us.
    let pid = ProjectileId(7);
    let spawn = WorldDelta::ProjectileSpawned {
        id: pid,
        source_steam_id: 1,
        source_npc_id: None,
        round_id: id(round_id),
        variant: simn_sim::AmmoVariant::default(),
        origin: [0.0, target_y, 0.0],
        velocity: [0.0, 0.0, 880.0], // near muzzle velocity so energy falloff is negligible at 5m
        max_range_m: 100.0,
        spawned_tick: sim.current_tick(),
    };
    sim.apply_external_delta(&spawn);

    // Tick until the impact delta appears.
    for _ in 0..5 {
        sim.tick().unwrap();
    }
    let deltas = sim.drain_tick_deltas();
    for delta in &deltas {
        if let WorldDelta::ProjectileImpacted {
            id: did,
            hit_npc: Some(hit_id),
            body_part: Some(p),
            damage_applied,
            penetrated,
            ..
        } = delta
        {
            if *did == pid && *hit_id == npc && *p == part {
                return (*damage_applied, *penetrated);
            }
        }
    }
    panic!(
        "no matching impact on part {:?} with round {:?} and armor={:?}",
        part, round_id, npc_armor
    );
}

#[test]
fn hp_vs_bare_torso_does_full_damage() {
    let (dmg, pen) = fire_straight_at_part(
        "round_5_45x39_hp",
        1.2, // torso height
        &[],
        None,
        BodyPart::Torso,
    );
    assert!(pen, "HP through bare flesh penetrates");
    // damage_soft = 55.0 * torso_mult 1.0 * falloff ~1.0
    assert!(
        dmg > 45.0,
        "HP vs bare torso should ~= damage_soft; got {dmg}"
    );
}

#[test]
fn hp_vs_class2_vest_does_near_zero() {
    let (dmg, pen) = fire_straight_at_part(
        "round_5_45x39_hp",
        1.2,
        &["armor_ballistic_rig"], // class 2
        None,
        BodyPart::Torso,
    );
    assert!(!pen, "HP blocked by class-2 armor");
    // pen_eff = 1 - 2 = -1 → ratio = 0.75 → dmg_blunt 12 * 0.75 = 9 * torso 1.0 ≈ 9
    assert!(
        dmg < 12.0,
        "HP vs class-2 vest should be blunt-only; got {dmg}"
    );
}

#[test]
fn fmj_vs_class1_vest_does_reduced_damage() {
    // 5.45 FMJ has pen_class 2 vs armor_soft_vest class 1:
    // pen_eff = +1, so FMJ penetrates. Damage = damage_soft 38 *
    // torso 1.0.
    let (dmg, pen) = fire_straight_at_part(
        "round_5_45x39",
        1.2,
        &["armor_soft_vest"], // class 1
        None,
        BodyPart::Torso,
    );
    assert!(pen, "FMJ penetrates class-1 vest");
    assert!(
        dmg > 30.0,
        "FMJ vs class-1 vest should do ~damage_soft (38); got {dmg}"
    );
}

#[test]
fn fmj_vs_class3_plates_is_blocked() {
    let (dmg, pen) = fire_straight_at_part(
        "round_5_45x39",
        1.2,
        &["armor_plate_carrier"], // class 3
        None,
        BodyPart::Torso,
    );
    assert!(!pen, "FMJ blocked by class-3 plates");
    // pen_eff = 2 - 3 = -1 → ratio 0.75 → blunt 15 * 0.75 = 11.25
    assert!(
        dmg < 13.0,
        "FMJ vs class-3 plates should be blunt-only; got {dmg}"
    );
}

#[test]
fn ap_vs_class3_plates_penetrates_reduced() {
    let (dmg, pen) = fire_straight_at_part(
        "round_5_45x39_ap",
        1.2,
        &["armor_plate_carrier"], // class 3
        None,
        BodyPart::Torso,
    );
    assert!(pen, "AP punches through class-3 plates");
    // damage_soft 32 * torso 1.0
    assert!(
        dmg > 28.0,
        "AP vs class-3 plates should deal ~damage_soft (32); got {dmg}"
    );
}

#[test]
fn ap_vs_class4_exo_is_blocked() {
    let (dmg, pen) = fire_straight_at_part(
        "round_5_45x39_ap",
        1.2,
        &["armor_heavy_exo"], // class 4
        None,
        BodyPart::Torso,
    );
    assert!(
        pen,
        "AP pen_class 4 vs armor class 4 → pen_eff 0, still penetrates"
    );
    // damage_soft 32 * torso 1.0 ≈ 32
    assert!(dmg > 28.0, "AP vs class-4 exo penetrates; got {dmg}");
}

#[test]
fn headshot_no_helmet_multiplies_damage() {
    let (dmg, pen) = fire_straight_at_part("round_5_45x39", 1.75, &[], None, BodyPart::Head);
    assert!(pen);
    // damage_soft 38 * head_mult 2.5 = 95
    assert!(
        dmg > 80.0,
        "FMJ headshot should apply head soft multiplier; got {dmg}"
    );
}

#[test]
fn hp_headshot_class2_helmet_blocks() {
    let (dmg, pen) = fire_straight_at_part(
        "round_5_45x39_hp",
        1.75,
        &[],
        Some("helmet_6b47"), // class 2 on head
        BodyPart::Head,
    );
    assert!(!pen, "HP (pen 1) vs class-2 helmet is blocked");
    // blunt 12 * ratio 0.75 * head_mult 2.5 = 22.5
    assert!(
        dmg < 30.0,
        "HP vs class-2 helmet should be modest blunt; got {dmg}"
    );
}
