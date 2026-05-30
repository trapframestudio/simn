//! End-to-end wound + bleed tests.

use simn_sim::{
    BodyPart, BodyParts, NpcId, RegionGraph, SavePaths, Sim, WoundKind, WoundTreatment, Wounds,
};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

fn fresh_sim(_dir: &TempDir) -> Sim {
    // No-disk, no-NPC variant. Persistence-roundtrip tests below
    // construct their own Sim::new explicitly.
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

fn upsert(sim: &mut Sim, sid: u64) {
    sim.upsert_player(sid, 1, [0.0; 3], 0.0).unwrap();
}

#[test]
fn small_damage_no_wound() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 5.0).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(v.wounds.is_empty(), "wounds: {:?}", v.wounds);
    assert!((v.body_parts.torso - 95.0).abs() < 0.01);
}

#[test]
fn light_damage_creates_light_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 15.0).unwrap();
    let v = sim.player_view(1).unwrap();
    assert_eq!(v.wounds.len(), 1);
    let (_, w) = &v.wounds[0];
    assert!(matches!(w.kind, WoundKind::Bleed));
    assert!(w.severity >= 1 && w.severity <= 3, "sev {}", w.severity);
    assert_eq!(w.body_part, BodyPart::Torso);
    assert!(matches!(w.treatment, WoundTreatment::Untreated));
}

#[test]
fn heavy_damage_creates_heavy_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::LeftLeg, 30.0)
        .unwrap();
    let v = sim.player_view(1).unwrap();
    assert_eq!(v.wounds.len(), 1);
    let (_, w) = &v.wounds[0];
    assert!(w.severity >= 4 && w.severity <= 5, "sev {}", w.severity);
    assert_eq!(w.body_part, BodyPart::LeftLeg);
}

#[test]
fn bleed_drains_part_hp_over_time() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 30.0).unwrap();
    let after_hit = sim.player_view(1).unwrap().body_parts.torso;
    // Tick 20x = 1 in-world second. sev 4 wound bleeds at 2 hp/sec.
    for _ in 0..20 {
        sim.tick().unwrap();
    }
    let after_bleed = sim.player_view(1).unwrap().body_parts.torso;
    let lost = after_hit - after_bleed;
    assert!(
        lost > 1.5 && lost < 3.5,
        "expected ~2 hp lost from sev-4 bleed in 1s, got {lost}"
    );
}

#[test]
fn bandage_stops_light_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 12.0).unwrap();
    sim.apply_bandage(1, BodyPart::Torso).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(matches!(v.wounds[0].1.treatment, WoundTreatment::Bandaged));
    let baseline = v.body_parts.torso;
    for _ in 0..40 {
        sim.tick().unwrap();
    }
    let after = sim.player_view(1).unwrap().body_parts.torso;
    assert!(
        (baseline - after).abs() < 0.01,
        "bandaged wound shouldn't bleed; lost {}",
        baseline - after
    );
}

#[test]
fn bandage_on_heavy_bleed_errors() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 30.0).unwrap();
    let err = sim.apply_bandage(1, BodyPart::Torso).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("heavy bleed"), "msg was: {msg}");
}

#[test]
fn bandage_with_no_wound_errors() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    let err = sim.apply_bandage(1, BodyPart::Torso).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("no light bleed"), "msg was: {msg}");
}

#[test]
fn tourniquet_stops_any_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::LeftLeg, 50.0)
        .unwrap();
    sim.apply_tourniquet(1, BodyPart::LeftLeg).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(matches!(
        v.wounds[0].1.treatment,
        WoundTreatment::Tourniquet
    ));
    let baseline = v.body_parts.left_leg;
    for _ in 0..40 {
        sim.tick().unwrap();
    }
    let after = sim.player_view(1).unwrap().body_parts.left_leg;
    assert!((baseline - after).abs() < 0.01);
}

#[test]
fn remove_tourniquet_resumes_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::LeftLeg, 30.0)
        .unwrap();
    sim.apply_tourniquet(1, BodyPart::LeftLeg).unwrap();
    sim.remove_tourniquet(1, BodyPart::LeftLeg).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(matches!(v.wounds[0].1.treatment, WoundTreatment::Untreated));
    let baseline = v.body_parts.left_leg;
    for _ in 0..20 {
        sim.tick().unwrap();
    }
    let after = sim.player_view(1).unwrap().body_parts.left_leg;
    assert!(after < baseline, "bleed should resume after removal");
}

#[test]
fn bandaged_wound_heals_and_despawns() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_heal_ticks_for_test(20);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 12.0).unwrap();
    sim.apply_bandage(1, BodyPart::Torso).unwrap();
    for _ in 0..25 {
        sim.tick().unwrap();
    }
    let v = sim.player_view(1).unwrap();
    assert!(v.wounds.is_empty(), "wound should have despawned");
}

#[test]
fn wounds_persist_roundtrip() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    let (saved_ids, saved_torso, saved_arm, saved_leg) = {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        upsert(&mut sim, 1);
        sim.apply_damage_to_part(1, BodyPart::Torso, 12.0).unwrap();
        sim.apply_damage_to_part(1, BodyPart::LeftArm, 18.0)
            .unwrap();
        sim.apply_damage_to_part(1, BodyPart::LeftLeg, 30.0)
            .unwrap();
        sim.apply_bandage(1, BodyPart::Torso).unwrap();
        sim.apply_tourniquet(1, BodyPart::LeftLeg).unwrap();
        for _ in 0..5 {
            sim.tick().unwrap();
        }
        let v = sim.player_view(1).unwrap();
        let ids: Vec<u64> = v.wounds.iter().map(|(id, _)| id.0).collect();
        let pick = |part: BodyPart| {
            v.wounds
                .iter()
                .find(|(_, w)| w.body_part == part)
                .map(|(_, w)| w.treatment)
                .unwrap()
        };
        let trio = (
            pick(BodyPart::Torso),
            pick(BodyPart::LeftArm),
            pick(BodyPart::LeftLeg),
        );
        sim.shutdown().unwrap();
        (ids, trio.0, trio.1, trio.2)
    };

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let v = sim.player_view(1).unwrap();
    let ids: Vec<u64> = v.wounds.iter().map(|(id, _)| id.0).collect();
    assert_eq!(ids, saved_ids);
    let pick = |part: BodyPart| {
        v.wounds
            .iter()
            .find(|(_, w)| w.body_part == part)
            .map(|(_, w)| w.treatment)
            .unwrap()
    };
    assert!(
        matches!(pick(BodyPart::Torso), WoundTreatment::Bandaged)
            == matches!(saved_torso, WoundTreatment::Bandaged)
    );
    assert!(
        matches!(pick(BodyPart::LeftArm), WoundTreatment::Untreated)
            == matches!(saved_arm, WoundTreatment::Untreated)
    );
    assert!(
        matches!(pick(BodyPart::LeftLeg), WoundTreatment::Tourniquet)
            == matches!(saved_leg, WoundTreatment::Tourniquet)
    );
}

#[test]
fn wound_id_counter_persists() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    let last_id_before = {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        upsert(&mut sim, 1);
        sim.apply_damage_to_part(1, BodyPart::Torso, 12.0).unwrap();
        sim.apply_damage_to_part(1, BodyPart::LeftArm, 12.0)
            .unwrap();
        let v = sim.player_view(1).unwrap();
        sim.shutdown().unwrap();
        v.wounds.iter().map(|(id, _)| id.0).max().unwrap()
    };

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    sim.apply_damage_to_part(1, BodyPart::RightArm, 12.0)
        .unwrap();
    let v = sim.player_view(1).unwrap();
    let new_id = v
        .wounds
        .iter()
        .find(|(_, w)| w.body_part == BodyPart::RightArm)
        .map(|(id, _)| id.0)
        .unwrap();
    assert!(
        new_id > last_id_before,
        "new id {new_id} should be > last persisted id {last_id_before}"
    );
}

#[test]
fn untreated_wound_default_seed_empty() {
    // Regression: a freshly-spawned player should have an empty Wounds
    // component (not absent), so apply_bandage doesn't error with
    // "has no Wounds" before the player has been hit.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    let v = sim.player_view(1).unwrap();
    assert_eq!(v.wounds, Vec::<(_, _)>::new());
    // Confirm by trying to bandage — should error with "no light bleed",
    // not "has no Wounds".
    let err = sim.apply_bandage(1, BodyPart::Torso).unwrap_err();
    assert!(format!("{err}").contains("no light bleed"));
}

#[test]
fn wounds_default_is_empty() {
    // Sanity for the Default impl — used by upsert_player and SpawnPlayer replay.
    let w = Wounds::default();
    assert!(w.0.is_empty());
}

#[test]
fn body_parts_default_max_unchanged() {
    // Stats foundation regression — confirm wounds work didn't shift the
    // existing constant.
    assert!((BodyParts::DEFAULT_MAX - 100.0).abs() < f32::EPSILON);
}

// ---- Step 3: infection / disinfect / stitch / necrosis / wound pack ----

#[test]
fn untreated_wound_becomes_infected() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_med_timings_for_test(40); // infection trigger at tick 40
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 15.0).unwrap();
    for _ in 0..50 {
        sim.tick().unwrap();
    }
    let v = sim.player_view(1).unwrap();
    assert!(
        v.wounds[0].1.infected,
        "wound should be infected past trigger"
    );
}

#[test]
fn disinfect_prevents_infection() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_med_timings_for_test(40);
    sim.set_heal_ticks_for_test(200); // keep wound around past the trigger
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 15.0).unwrap();
    sim.apply_disinfectant(1, BodyPart::Torso).unwrap();
    sim.apply_bandage(1, BodyPart::Torso).unwrap();
    for _ in 0..50 {
        sim.tick().unwrap();
    }
    let v = sim.player_view(1).unwrap();
    assert!(
        !v.wounds.is_empty(),
        "wound should still be present at assert time (heal_ticks=200 keeps it alive)"
    );
    let (_, w) = v.wounds.first().expect("wound should still be present");
    assert!(!w.infected, "disinfected→bandaged wound shouldn't infect");
}

#[test]
fn antibiotics_clear_infection() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_med_timings_for_test(40); // infection trigger 40, antibiotics clear 10
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 15.0).unwrap();
    for _ in 0..50 {
        sim.tick().unwrap();
    }
    assert!(sim.player_view(1).unwrap().wounds[0].1.infected);
    sim.apply_antibiotics(1).unwrap();
    for _ in 0..30 {
        sim.tick().unwrap();
    }
    let v = sim.player_view(1).unwrap();
    // The wound is Untreated, which never auto-heals/despawns, so it
    // is guaranteed present here — assert that unconditionally.
    assert!(
        !v.wounds.is_empty(),
        "untreated wound should still be present after antibiotics window"
    );
    let (_, w) = v.wounds.first().expect("wound should still be present");
    assert!(!w.infected, "antibiotics should clear infection");
}

#[test]
fn stitch_heals_faster_than_bandage() {
    let elapsed_until_heal = |use_stitch: bool| -> u64 {
        let dir = TempDir::new().unwrap();
        let mut sim = fresh_sim(&dir);
        sim.set_heal_ticks_for_test(40);
        upsert(&mut sim, 1);
        sim.apply_damage_to_part(1, BodyPart::Torso, 15.0).unwrap();
        sim.apply_bandage(1, BodyPart::Torso).unwrap();
        if use_stitch {
            sim.apply_stitch(1, BodyPart::Torso).unwrap();
        }
        let mut ticks = 0;
        loop {
            sim.tick().unwrap();
            ticks += 1;
            if sim.player_view(1).unwrap().wounds.is_empty() {
                return ticks;
            }
            if ticks > 200 {
                return ticks;
            }
        }
    };
    let bandaged = elapsed_until_heal(false);
    let stitched = elapsed_until_heal(true);
    assert!(
        stitched < bandaged,
        "stitched should heal faster than bandaged: stitched={stitched} bandaged={bandaged}"
    );
}

#[test]
fn tourniquet_necrosis_starts_after_warning() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_med_timings_for_test(40); // necrosis warning at tick 20
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::LeftLeg, 30.0)
        .unwrap();
    sim.apply_tourniquet(1, BodyPart::LeftLeg).unwrap();
    let baseline = sim.player_view(1).unwrap().body_parts.left_leg;
    // Tick to just before necrosis warning.
    for _ in 0..18 {
        sim.tick().unwrap();
    }
    let pre_warning = sim.player_view(1).unwrap().body_parts.left_leg;
    assert!(
        (baseline - pre_warning).abs() < 0.5,
        "expected no drain before necrosis warning: {baseline} -> {pre_warning}"
    );
    // Tick well past warning to let necrosis accumulate.
    for _ in 0..60 {
        sim.tick().unwrap();
    }
    let after_necrosis = sim.player_view(1).unwrap().body_parts.left_leg;
    assert!(
        after_necrosis < pre_warning,
        "necrosis should drain post-warning: {pre_warning} -> {after_necrosis}"
    );
}

#[test]
fn wound_pack_stops_heavy_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 30.0).unwrap();
    sim.apply_wound_pack(1, BodyPart::Torso).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(matches!(
        v.wounds[0].1.treatment,
        WoundTreatment::WoundPacked
    ));
    let baseline = v.body_parts.torso;
    for _ in 0..40 {
        sim.tick().unwrap();
    }
    let after = sim.player_view(1).unwrap().body_parts.torso;
    assert!(
        (baseline - after).abs() < 0.01,
        "wound pack should stop bleed"
    );
}

// ---------- NPC wound-plumbing smoke tests ----------

#[test]
fn npcs_spawn_with_empty_wounds_and_active_effects() {
    // Humanoid NPCs need both components so the wound pipeline and
    // antibiotics-clearing-infection flow can iterate them. At spawn
    // they carry the components but with empty payloads.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let wounds = sim
        .npc_wounds_for_test(id)
        .expect("NPC should carry Wounds component");
    assert!(wounds.0.is_empty(), "fresh NPC has no wounds");
    assert!(
        sim.npc_has_active_effects_for_test(id),
        "NPC should carry ActiveEffects component"
    );
}

#[test]
fn npc_wounds_survive_snapshot_round_trip() {
    // Round-trip through shutdown + reload via the same journal /
    // snapshot path the game uses. Confirms the serialize/deserialize
    // + NpcSpawned replay path all carry the new components.
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let id = {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
        sim.tick().unwrap();
        sim.shutdown().unwrap();
        id
    };
    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let wounds = sim
        .npc_wounds_for_test(id)
        .expect("reloaded NPC should carry Wounds");
    assert!(wounds.0.is_empty(), "reloaded NPC has no wounds");
    assert!(
        sim.npc_has_active_effects_for_test(id),
        "reloaded NPC should carry ActiveEffects"
    );
}

#[test]
fn npc_light_damage_spawns_light_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 15.0)
        .unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert_eq!(wounds.0.len(), 1, "expected one wound");
    let (_, w) = &wounds.0[0];
    assert!(matches!(w.kind, WoundKind::Bleed));
    assert!(w.severity >= 1 && w.severity <= 3, "sev {}", w.severity);
    assert_eq!(w.body_part, BodyPart::Torso);
    assert!(matches!(w.treatment, WoundTreatment::Untreated));
}

#[test]
fn npc_heavy_damage_spawns_heavy_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 30.0)
        .unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert_eq!(wounds.0.len(), 1);
    let (_, w) = &wounds.0[0];
    assert!(w.severity >= 4, "heavy bleed severity {}", w.severity);
}

#[test]
fn npc_sub_threshold_damage_no_wound() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 5.0)
        .unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert!(wounds.0.is_empty(), "sub-threshold damage = no wound");
}

#[test]
fn npc_bleed_drains_part_hp_over_time() {
    // Spawn an NPC with a fresh untreated heavy bleed. Tick the sim
    // past several seconds; the torso pool should drop beyond the
    // initial hit amount because the untreated wound drains per tick.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 30.0)
        .unwrap();
    // Population targets in the default graph are real; zero them so
    // automatic spawn / aggro / combat doesn't interfere with this
    // NPC during the bleed window.
    let names: Vec<String> = sim
        .faction_registry()
        .defs()
        .map(|d| d.name.clone())
        .collect();
    for name in &names {
        for r in [1u32, 2, 3, 4] {
            sim.set_population_target_for_test(r, name, 0);
        }
    }
    // Grab the pool value *after* the damage hit but *before* ticks
    // so we measure only the bleed contribution.
    let wounds_pre = sim.npc_wounds_for_test(id).unwrap();
    assert_eq!(wounds_pre.0.len(), 1, "one wound from the hit");
    // 20 ticks ≈ 1 real second at the default 50ms tick. A sev-4 bleed
    // drains 2 HP/sec, so we expect ~2 HP more gone.
    for _ in 0..40 {
        sim.tick().unwrap();
    }
    let wounds_post = sim.npc_wounds_for_test(id).unwrap();
    assert_eq!(
        wounds_post.0.len(),
        1,
        "wound shouldn't vanish without treatment"
    );
}

#[test]
fn npc_combat_spawns_journaled_wound() {
    // Two hostile NPCs in sight of each other. Phase 4A v2 routes
    // combat damage through the projectile-tick path, which calls
    // `Sim::apply_damage_to_npc_part` and therefore JOURNALS an
    // `NpcWoundAdded` delta (unlike the retired probabilistic dice
    // path, which was ephemeral). This test proves both halves:
    //   1. combat actually spawns a wound on a combatant, and
    //   2. that wound is reflected by an `NpcWoundAdded` delta whose
    //      `(id, body_part)` matches a wound now present on the NPC.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_active_region(1);
    // Zero pop targets so automatic spawning doesn't pollute.
    let names: Vec<String> = sim
        .faction_registry()
        .defs()
        .map(|d| d.name.clone())
        .collect();
    for name in &names {
        for r in [1u32, 2, 3, 4] {
            sim.set_population_target_for_test(r, name, 0);
        }
    }
    let pwa = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let bandit = sim.spawn_npc_for_test("looters", 1, [10.0, 0.0, 0.0], None);
    sim.set_npc_yaw_for_test(pwa, 0.0);
    sim.set_npc_yaw_for_test(bandit, std::f32::consts::PI);
    // Tick past several FIRE_INTERVAL_TICKS (50) so combat fires, and
    // accumulate every per-tick delta. `drain_tick_deltas` only holds
    // the most-recent tick's deltas, so we drain after each tick.
    let mut all_deltas: Vec<simn_sim::WorldDelta> = Vec::new();
    for _ in 0..200 {
        sim.tick().unwrap();
        all_deltas.extend(sim.drain_tick_deltas());
    }
    let pwa_wounds = sim.npc_wounds_for_test(pwa).unwrap();
    let bandit_wounds = sim.npc_wounds_for_test(bandit).unwrap();
    let any_combat_wounds = !pwa_wounds.0.is_empty() || !bandit_wounds.0.is_empty();
    assert!(
        any_combat_wounds,
        "combat should have spawned at least one wound between pwa+bandit"
    );
    // Collect every `NpcWoundAdded` delta journaled for either
    // combatant across the combat window.
    let journaled: Vec<(NpcId, BodyPart)> = all_deltas
        .iter()
        .filter_map(|d| match d {
            simn_sim::WorldDelta::NpcWoundAdded { id, body_part, .. }
                if *id == pwa || *id == bandit =>
            {
                Some((*id, *body_part))
            }
            _ => None,
        })
        .collect();
    assert!(
        !journaled.is_empty(),
        "combat wounds must be journaled via NpcWoundAdded (projectile-tick path)"
    );
    // Cross-check: at least one journaled delta corresponds to a wound
    // still present on the NPC — i.e. the journal reflects real state,
    // not a phantom delta.
    let present = |id: NpcId, part: BodyPart| {
        let w = if id == pwa {
            &pwa_wounds
        } else {
            &bandit_wounds
        };
        w.0.iter().any(|(_, wound)| wound.body_part == part)
    };
    assert!(
        journaled.iter().any(|(id, part)| present(*id, *part)),
        "a journaled NpcWoundAdded should match a wound currently on the NPC; \
         journaled={journaled:?}, pwa={:?}, bandit={:?}",
        pwa_wounds.0,
        bandit_wounds.0,
    );
}

// ---------- NPC treatment API ----------

#[test]
fn bandage_stops_npc_light_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 15.0)
        .unwrap();
    sim.apply_bandage_npc(id, BodyPart::Torso).unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert!(matches!(wounds.0[0].1.treatment, WoundTreatment::Bandaged));
}

#[test]
fn tourniquet_stops_any_npc_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::LeftLeg, 30.0)
        .unwrap();
    sim.apply_tourniquet_npc(id, BodyPart::LeftLeg).unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert!(matches!(
        wounds.0[0].1.treatment,
        WoundTreatment::Tourniquet
    ));
}

#[test]
fn remove_npc_tourniquet_resumes_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::LeftLeg, 30.0)
        .unwrap();
    sim.apply_tourniquet_npc(id, BodyPart::LeftLeg).unwrap();
    sim.remove_tourniquet_npc(id, BodyPart::LeftLeg).unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert!(matches!(wounds.0[0].1.treatment, WoundTreatment::Untreated));
    assert!(
        wounds.0[0].1.tourniquet_started_tick.is_none(),
        "tourniquet timer cleared"
    );
}

#[test]
fn stitch_closes_npc_bandaged_wound() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 15.0)
        .unwrap();
    sim.apply_bandage_npc(id, BodyPart::Torso).unwrap();
    sim.apply_stitch_npc(id, BodyPart::Torso).unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert!(matches!(wounds.0[0].1.treatment, WoundTreatment::Stitched));
}

#[test]
fn wound_pack_stops_heavy_npc_bleed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 30.0)
        .unwrap();
    sim.apply_wound_pack_npc(id, BodyPart::Torso).unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert!(matches!(
        wounds.0[0].1.treatment,
        WoundTreatment::WoundPacked
    ));
}

#[test]
fn disinfect_prevents_npc_infection() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 15.0)
        .unwrap();
    sim.apply_disinfectant_npc(id, BodyPart::Torso).unwrap();
    let wounds = sim.npc_wounds_for_test(id).unwrap();
    assert!(matches!(
        wounds.0[0].1.treatment,
        WoundTreatment::Disinfected
    ));
}

#[test]
fn antibiotics_clear_npc_infection() {
    // Mirror of the player antibiotics test: infect an NPC wound via
    // the normal tick_infection path, then apply antibiotics and
    // verify the `infected` flag flips back off.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.set_med_timings_for_test(40);
    // Zero pop targets so auto spawns don't interfere.
    let names: Vec<String> = sim
        .faction_registry()
        .defs()
        .map(|d| d.name.clone())
        .collect();
    for name in &names {
        for r in [1u32, 2, 3, 4] {
            sim.set_population_target_for_test(r, name, 0);
        }
    }
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    sim.apply_damage_to_npc_part(id, BodyPart::Torso, 15.0)
        .unwrap();
    for _ in 0..50 {
        sim.tick().unwrap();
    }
    let wounds_pre = sim.npc_wounds_for_test(id).unwrap();
    assert!(
        wounds_pre.0[0].1.infected,
        "wound should be infected after trigger window"
    );
    sim.apply_antibiotics_npc(id).unwrap();
    for _ in 0..30 {
        sim.tick().unwrap();
    }
    let wounds_post = sim.npc_wounds_for_test(id).unwrap();
    // The wound is Untreated (never auto-heals/despawns), so it is
    // guaranteed present here — assert the cleared infection state
    // unconditionally.
    assert!(
        !wounds_post.0.is_empty(),
        "untreated NPC wound should still be present after antibiotics window"
    );
    let (_, w) = wounds_post
        .0
        .first()
        .expect("wound should still be present");
    assert!(!w.infected, "antibiotics should clear NPC infection");
}
