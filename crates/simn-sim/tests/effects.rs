//! Drug effects, tolerance, overdose, withdrawal.

use simn_sim::{BodyPart, DrugKind, DrugOutcome, EffectKind, RegionGraph, SavePaths, Sim, Stamina};
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
fn first_dose_is_safe() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    for drug in [
        DrugKind::Painkiller,
        DrugKind::Morphine,
        DrugKind::StimCocktail,
        DrugKind::Adrenaline,
        DrugKind::AntiRad,
        DrugKind::AntiTox,
    ] {
        let mut s = fresh_sim(&TempDir::new().unwrap());
        upsert(&mut s, 1);
        let outcome = s.apply_drug(1, drug).unwrap();
        assert_eq!(outcome, DrugOutcome::Effect, "{drug:?} first dose");
    }
}

#[test]
fn painkiller_reduces_pain() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_damage_to_part(1, BodyPart::Torso, 30.0).unwrap();
    sim.tick().unwrap(); // let tick_pain derive
    let pain_before = sim.player_view(1).unwrap().pain.0;
    assert!(
        pain_before > 10.0,
        "expected pain from heavy wound, got {pain_before}"
    );
    sim.apply_drug(1, DrugKind::Painkiller).unwrap();
    sim.tick().unwrap();
    let pain_after = sim.player_view(1).unwrap().pain.0;
    assert!(
        pain_after < pain_before,
        "painkiller should reduce pain: {pain_before} -> {pain_after}"
    );
}

#[test]
fn morphine_reduces_pain_more_than_painkiller() {
    let baseline = |drug: DrugKind| -> f32 {
        let dir = TempDir::new().unwrap();
        let mut sim = fresh_sim(&dir);
        upsert(&mut sim, 1);
        sim.apply_damage_to_part(1, BodyPart::Torso, 30.0).unwrap();
        sim.apply_damage_to_part(1, BodyPart::LeftArm, 30.0)
            .unwrap();
        sim.tick().unwrap();
        sim.apply_drug(1, drug).unwrap();
        sim.tick().unwrap();
        sim.player_view(1).unwrap().pain.0
    };
    let with_painkiller = baseline(DrugKind::Painkiller);
    let with_morphine = baseline(DrugKind::Morphine);
    assert!(
        with_morphine < with_painkiller,
        "morphine ({with_morphine}) should be lower than painkiller ({with_painkiller})"
    );
}

#[test]
fn stim_boosts_regen() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.set_stamina(1, 0.0).unwrap();
    sim.apply_drug(1, DrugKind::StimCocktail).unwrap();
    for _ in 0..20 {
        sim.tick().unwrap();
    }
    let s = sim.player_view(1).unwrap().stamina.current;
    let baseline_only = Stamina::DEFAULT_REGEN * 0.05 * 20.0; // 15 stamina
    assert!(
        s > baseline_only,
        "stim should accelerate regen; got {s}, baseline ~{baseline_only}"
    );
}

#[test]
fn adrenaline_revives_at_low_hp() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    // Push torso near zero via several damage hits (HP only — we
    // don't want bleed to dominate the test).
    sim.apply_damage_to_part(1, BodyPart::Torso, 95.0).unwrap();
    sim.apply_tourniquet(1, BodyPart::Torso).unwrap(); // kill the bleed
    let before = sim.player_view(1).unwrap().body_parts.torso;
    assert!(
        before < 10.0,
        "setup: torso should be near zero, got {before}"
    );
    sim.apply_drug(1, DrugKind::Adrenaline).unwrap();
    let after = sim.player_view(1).unwrap().body_parts.torso;
    assert!(
        after >= 30.0,
        "adrenaline should revive to ~30%, got {after}"
    );
}

#[test]
fn tolerance_increments_per_use() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_drug(1, DrugKind::Painkiller).unwrap();
    let t1 = sim.player_view(1).unwrap();
    let tol1 = t1
        .drug_tolerance
        .iter()
        .find(|(d, _)| *d == DrugKind::Painkiller)
        .map(|(_, v)| *v)
        .unwrap();
    sim.apply_drug(1, DrugKind::Painkiller).unwrap();
    let t2 = sim.player_view(1).unwrap();
    let tol2 = t2
        .drug_tolerance
        .iter()
        .find(|(d, _)| *d == DrugKind::Painkiller)
        .map(|(_, v)| *v)
        .unwrap();
    assert!(tol2 > tol1, "tolerance should increment: {tol1} -> {tol2}");
}

#[test]
fn tolerance_decays_over_time() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.apply_drug(1, DrugKind::Painkiller).unwrap();
    sim.apply_drug(1, DrugKind::Painkiller).unwrap();
    sim.apply_drug(1, DrugKind::Painkiller).unwrap();
    let tol_before = sim.player_view(1).unwrap().drug_tolerance[0].1;
    // Just verify decay direction; the rate is already pinned by the
    // MedConfig default and exercised end-to-end elsewhere.
    for _ in 0..200 {
        sim.tick().unwrap();
    }
    let tol_after = sim.player_view(1).unwrap().drug_tolerance[0].1;
    assert!(
        tol_after < tol_before,
        "tolerance should decay over time: {tol_before} -> {tol_after}"
    );
}

#[test]
fn third_morphine_overdoses_when_tolerance_high() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    // Morphine: +30 tolerance per use, threshold 75. After 3 doses
    // (90 tolerance) and another while still active → overdose.
    let o1 = sim.apply_drug(1, DrugKind::Morphine).unwrap();
    let o2 = sim.apply_drug(1, DrugKind::Morphine).unwrap();
    let o3 = sim.apply_drug(1, DrugKind::Morphine).unwrap();
    let o4 = sim.apply_drug(1, DrugKind::Morphine).unwrap();
    assert_eq!(o1, DrugOutcome::Effect);
    assert_eq!(o2, DrugOutcome::Effect);
    assert_eq!(o3, DrugOutcome::Effect);
    assert_eq!(
        o4,
        DrugOutcome::Overdose,
        "4th dose at tolerance 90 should overdose"
    );
}

#[test]
fn overdose_disorients_player() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    for _ in 0..4 {
        sim.apply_drug(1, DrugKind::Morphine).unwrap();
    }
    let v = sim.player_view(1).unwrap();
    assert!(
        v.active_effects
            .iter()
            .any(|e| matches!(e.kind, EffectKind::OverdoseDisorientation)),
        "expected disorientation effect after overdose"
    );
}

#[test]
fn anti_rad_reduces_radiation_with_tox_cost() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.set_radiation(1, 60.0).unwrap();
    sim.set_toxicity(1, 0.0).unwrap();
    sim.apply_drug(1, DrugKind::AntiRad).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(v.contamination.radiation < 60.0, "rad should drop");
    assert!(
        v.contamination.toxicity > 0.0,
        "tox should rise (spec §4.4 trade-off)"
    );
}

#[test]
fn anti_tox_reduces_toxicity() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.set_toxicity(1, 60.0).unwrap();
    sim.apply_drug(1, DrugKind::AntiTox).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(v.contamination.toxicity < 60.0);
}

#[test]
fn high_contamination_drains_hp() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.set_radiation(1, 90.0).unwrap();
    let before = sim.player_view(1).unwrap().body_parts.torso;
    for _ in 0..200 {
        sim.tick().unwrap();
    }
    let after = sim.player_view(1).unwrap().body_parts.torso;
    assert!(
        after < before,
        "high rad should drain HP: {before} -> {after}"
    );
}

#[test]
fn effects_persist_roundtrip() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let saved_count;
    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        upsert(&mut sim, 1);
        sim.apply_drug(1, DrugKind::Painkiller).unwrap();
        sim.apply_drug(1, DrugKind::StimCocktail).unwrap(); // also schedules a FatigueRebound
        let v = sim.player_view(1).unwrap();
        saved_count = v.active_effects.len();
        assert!(
            saved_count >= 3,
            "expected painkiller + stim + rebound, got {saved_count}"
        );
        sim.shutdown().unwrap();
    }

    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let v = sim.player_view(1).unwrap();
    assert_eq!(
        v.active_effects.len(),
        saved_count,
        "effects preserved across reload"
    );
}
