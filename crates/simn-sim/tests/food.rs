//! Food + water consumption profiles.

use simn_sim::{FoodKind, RegionGraph, Sim, SurvivalStat, WaterKind};
use tempfile::TempDir;

fn fresh_sim(_dir: &TempDir) -> Sim {
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

fn upsert_drained(sim: &mut Sim, sid: u64) {
    sim.upsert_player(sid, 1, [0.0; 3], 0.0).unwrap();
    sim.set_survival_stat(sid, SurvivalStat::Hunger, 0.0)
        .unwrap();
    sim.set_survival_stat(sid, SurvivalStat::Thirst, 0.0)
        .unwrap();
    sim.set_radiation(sid, 0.0).unwrap();
    sim.set_toxicity(sid, 0.0).unwrap();
}

#[test]
fn eat_preserved_ration_restores_hunger() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert_drained(&mut sim, 1);
    sim.eat(1, FoodKind::PreservedRation).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(
        (v.survival.hunger - 30.0).abs() < 0.01,
        "hunger {}",
        v.survival.hunger
    );
    // PreservedRation has -5 thirst (salty); from 0 stays clamped at 0.
    assert!(v.survival.thirst <= 0.01);
}

#[test]
fn eat_raw_meat_adds_toxicity() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert_drained(&mut sim, 1);
    sim.eat(1, FoodKind::RawMeat).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(
        v.contamination.toxicity > 0.0,
        "raw meat should add tox: {}",
        v.contamination.toxicity
    );
    assert!(v.survival.hunger > 0.0);
}

#[test]
fn eat_cooked_meat_safe() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert_drained(&mut sim, 1);
    sim.eat(1, FoodKind::CookedMeat).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(v.contamination.toxicity < 0.01, "cooked meat is safe");
    assert!(v.contamination.radiation < 0.01);
}

#[test]
fn eat_contaminated_food_adds_rad_and_tox() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert_drained(&mut sim, 1);
    sim.eat(1, FoodKind::ContaminatedFood).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(v.contamination.radiation > 0.0);
    assert!(v.contamination.toxicity > 0.0);
}

#[test]
fn drink_dirty_water_adds_rad_and_tox() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert_drained(&mut sim, 1);
    sim.drink(1, WaterKind::DirtyWater).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(v.survival.thirst > 0.0);
    assert!(v.contamination.radiation > 0.0);
    assert!(v.contamination.toxicity > 0.0);
}

#[test]
fn drink_clean_water_safe() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert_drained(&mut sim, 1);
    sim.drink(1, WaterKind::CleanWater).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(v.survival.thirst >= 60.0 - 0.01);
    assert!(v.contamination.radiation < 0.01);
}

#[test]
fn drink_energy_drink_grants_stim() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert_drained(&mut sim, 1);
    sim.drink(1, WaterKind::EnergyDrink).unwrap();
    let v = sim.player_view(1).unwrap();
    assert!(
        v.active_effects
            .iter()
            .any(|e| matches!(e.kind, simn_sim::EffectKind::StimCocktail)),
        "energy drink should grant a Stim effect"
    );
}
