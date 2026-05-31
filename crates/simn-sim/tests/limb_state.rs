//! `LimbStates` integration tests. Per
//! `docs/book/src/planning/dismemberment-plan.md` — the wound pipeline
//! flips parts to `Wounded` on damage and back to `Intact` when the
//! last open wound resolves; sever flips to `Severed` and is permanent.

use simn_sim::{BodyPart, BodyParts, LimbState, RegionGraph, Sim};
use tempfile::TempDir;

fn fresh_sim(_dir: &TempDir) -> Sim {
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

/// Quiet sim with population targets zeroed. Used by NPC-side tests so
/// `force_npc_hp_for_test` / `sever_limb_for_test` operate on a single
/// known NPC without auto-spawned squads polluting state. The
/// `new_in_memory` constructor already clears `PopulationTargets`,
/// so the per-region zero loop here is redundant — kept for the doc
/// signal, but the loop is a no-op.
fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn fresh_player_starts_all_intact() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    let states = sim
        .player_limb_states_for_test(1)
        .expect("player has LimbStates");
    for part in BodyPart::ALL {
        assert!(
            matches!(states.get(part), LimbState::Intact),
            "{:?} should start Intact",
            part
        );
    }
}

#[test]
fn small_damage_does_not_flip_state() {
    // Sub-threshold damage doesn't spawn a wound, so the limb stays
    // Intact even though HP dropped.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::Torso, 5.0).unwrap();
    let states = sim.player_limb_states_for_test(1).unwrap();
    assert!(matches!(states.torso, LimbState::Intact));
}

#[test]
fn wound_marks_limb_wounded() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::LeftArm, 20.0)
        .unwrap();
    let states = sim.player_limb_states_for_test(1).unwrap();
    assert!(matches!(states.left_arm, LimbState::Wounded));
    assert!(matches!(states.torso, LimbState::Intact));
}

#[test]
fn multiple_wounds_same_limb_stay_wounded() {
    // Two wounds on the same part — state stays `Wounded`, doesn't
    // double-flip or break.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::Torso, 15.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::Torso, 15.0).unwrap();
    let states = sim.player_limb_states_for_test(1).unwrap();
    assert!(matches!(states.torso, LimbState::Wounded));
}

#[test]
fn heal_clears_limb_to_intact() {
    // Damage → bandage → tick past heal timer → wound dropped → state
    // back to Intact.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_heal_ticks_for_test(40);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::LeftArm, 15.0)
        .unwrap();
    assert!(matches!(
        sim.player_limb_states_for_test(1).unwrap().left_arm,
        LimbState::Wounded
    ));

    sim.apply_bandage(1, BodyPart::LeftArm).unwrap();

    let mut healed = false;
    for _ in 0..200 {
        sim.tick().unwrap();
        let v = sim.player_view(1).unwrap();
        if v.wounds.is_empty() {
            healed = true;
            break;
        }
    }
    assert!(healed, "bandaged wound never cleared");
    let states = sim.player_limb_states_for_test(1).unwrap();
    assert!(
        matches!(states.left_arm, LimbState::Intact),
        "left_arm should be Intact after heal, got {:?}",
        states.left_arm
    );
}

#[test]
fn other_limbs_stay_intact_when_one_wounded() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::RightLeg, 30.0)
        .unwrap();
    let states = sim.player_limb_states_for_test(1).unwrap();
    assert!(matches!(states.right_leg, LimbState::Wounded));
    for part in [
        BodyPart::Head,
        BodyPart::Torso,
        BodyPart::LeftArm,
        BodyPart::RightArm,
        BodyPart::LeftLeg,
    ] {
        assert!(
            matches!(states.get(part), LimbState::Intact),
            "{:?} should be Intact, got {:?}",
            part,
            states.get(part)
        );
    }
}

#[test]
fn sever_arm_flips_state_and_zeroes_hp() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], None);
    assert!(sim.sever_limb_for_test(id, BodyPart::LeftArm));
    let states = sim.npc_limb_states_for_test(id).unwrap();
    assert!(matches!(states.left_arm, LimbState::Severed));
    let view = sim
        .npcs_in_region(1)
        .into_iter()
        .find(|v| v.id == id)
        .unwrap();
    let bp = view.body_parts.unwrap();
    assert!(
        bp.left_arm <= 0.01,
        "left_arm HP should be 0, got {}",
        bp.left_arm
    );
    // Head + torso untouched → NPC alive.
    assert!(view.health.current > 0.0, "limb sever should not kill NPC");
}

#[test]
fn sever_head_kills_npc() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], None);
    assert!(sim.sever_limb_for_test(id, BodyPart::Head));
    let states = sim.npc_limb_states_for_test(id).unwrap();
    assert!(matches!(states.head, LimbState::Severed));
    let view = sim
        .npcs_in_region(1)
        .into_iter()
        .find(|v| v.id == id)
        .unwrap();
    // Head zeroed → vital_min == 0 → aggregate Health.current == 0.
    assert!(
        view.health.current <= 0.01,
        "head sever should drop aggregate health to 0, got {}",
        view.health.current
    );
}

#[test]
fn sever_overrides_wounded() {
    // Wound a limb (Wounded), then sever it (Severed). Severed wins.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::LeftArm, 20.0)
        .unwrap();
    assert!(matches!(
        sim.npc_limb_states_for_test(id).unwrap().left_arm,
        LimbState::Wounded
    ));
    assert!(sim.sever_limb_for_test(id, BodyPart::LeftArm));
    assert!(matches!(
        sim.npc_limb_states_for_test(id).unwrap().left_arm,
        LimbState::Severed
    ));
}

#[test]
fn sever_persists_when_remaining_wound_clears() {
    // Sever a limb, then bandage + heal whatever wound is left on
    // it (if any) — Severed must NOT downgrade to Intact via the
    // Wounded → Intact recompute path.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    sim.set_heal_ticks_for_test(40);
    let id = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], None);
    // Pre-wound the limb so a wound exists, then sever it. Sever
    // doesn't drop the existing wound; the recompute path will run
    // when the bandage heals the wound and it's retained-out.
    sim.apply_damage_to_npc_part(id, BodyPart::LeftArm, 15.0)
        .unwrap();
    assert!(sim.sever_limb_for_test(id, BodyPart::LeftArm));
    sim.apply_bandage_npc(id, BodyPart::LeftArm).unwrap();
    let mut wound_cleared = false;
    for _ in 0..200 {
        sim.tick().unwrap();
        if sim.npc_wounds_for_test(id).is_some_and(|w| w.0.is_empty()) {
            wound_cleared = true;
            break;
        }
    }
    assert!(wound_cleared, "bandaged wound never cleared");
    let states = sim.npc_limb_states_for_test(id).unwrap();
    assert!(
        matches!(states.left_arm, LimbState::Severed),
        "Severed must be permanent; got {:?}",
        states.left_arm
    );
}

#[test]
fn npc_combat_wound_marks_limb_wounded() {
    // Real combat path: NPC vs NPC, the npc_combat system writes the
    // wound + states. Verify the LimbStates flip flows through.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let attacker = sim.spawn_npc_for_test("looters", 1, [0.0, 0.0, 0.0], None);
    let target = sim.spawn_npc_for_test("coalition", 1, [3.0, 0.0, 0.0], None);
    sim.set_npc_aggro_for_test(attacker, target);
    // Tick forward until the combat system has fired at least one
    // wound-grade hit. npc_combat's per-attack damage roll is 5..=20
    // (enough to clear the light wound threshold), so a single hit
    // is sufficient.
    let mut wounded = false;
    for _ in 0..200 {
        sim.tick().unwrap();
        let states = sim.npc_limb_states_for_test(target).unwrap();
        if matches!(states.torso, LimbState::Wounded | LimbState::Severed) {
            wounded = true;
            break;
        }
    }
    assert!(wounded, "target limb never flipped to Wounded after combat");
}

#[test]
fn body_parts_default_max_unchanged() {
    // Sanity check that we didn't accidentally change the spawn HP
    // contract while extending the state model.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!((v.body_parts.head - BodyParts::DEFAULT_MAX).abs() < 0.01);
    assert!((v.body_parts.torso - BodyParts::DEFAULT_MAX).abs() < 0.01);
}
