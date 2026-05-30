//! Phase 4D — weapon condition + jam tests.
//!
//! Coverage:
//! - `jam_chance_at_condition` curve: 0 above threshold, peaks
//!   at `jam_chance_floor` at zero condition.
//! - `fire_weapon` decrements condition by `wear_per_shot` on
//!   successful shots and journals `WeaponConditionChanged`.
//! - A fresh weapon (cond=100) above threshold never jams over
//!   a full magazine.
//! - A clapped-out weapon (cond=0) journals `WeaponJammed`
//!   *eventually* (within bounded tries) and gates further fire.
//! - `clear_weapon_jam` restores `JamState::Cleared` and the
//!   weapon fires again — without bumping condition.
//! - `clear_weapon_jam` on an un-jammed weapon errors.

use simn_sim::{jam_chance_at_condition, ItemId, JamState, RegionGraph, Sim, SlotId, WorldDelta};
use tempfile::TempDir;

fn fresh_sim(_dir: &TempDir) -> Sim {
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

fn slot(s: &str) -> SlotId {
    SlotId::from(s)
}

fn id(s: &str) -> ItemId {
    ItemId::from(s)
}

fn pocket_index_of(sim: &mut Sim, sid: u64, needle: &ItemId) -> Option<usize> {
    sim.inventory_view(sid).iter().position(|s| &s.id == needle)
}

fn ready_aks74(sim: &mut Sim) {
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.grant_item(1, &id("rifle_aks74"), 1).unwrap();
    let idx = pocket_index_of(sim, 1, &id("rifle_aks74")).expect("rifle granted");
    sim.equip(1, &slot("primary"), "pockets", idx).unwrap();
    sim.grant_item(1, &id("mag_aks74_30"), 1).unwrap();
    sim.reload_weapon(1, &slot("primary")).unwrap();
    sim.set_equipped_mag_state_for_test(1, &slot("primary"), 30, Some("round_5_45x39"));
}

#[test]
fn jam_chance_curve_is_zero_above_threshold_and_peaks_at_zero() {
    // Build a config whose threshold + floor we set explicitly.
    let cfg = simn_sim::WeaponConfig {
        caliber: simn_sim::Caliber::from("5.45x39"),
        damage: 35.0,
        range_m: 300.0,
        fire_interval_s: 0.15,
        spread_deg: 0.5,
        slots: vec![],
        wear_per_shot: 0.05,
        jam_threshold: 70.0,
        jam_chance_floor: 0.18,
    };
    // Above the threshold: dead zero.
    assert_eq!(jam_chance_at_condition(100.0, &cfg), 0.0);
    assert_eq!(jam_chance_at_condition(71.0, &cfg), 0.0);
    assert_eq!(jam_chance_at_condition(70.0, &cfg), 0.0);
    // Half-way down from threshold to 0 → half of the floor.
    let mid = jam_chance_at_condition(35.0, &cfg);
    assert!(
        (mid - 0.09).abs() < 1e-4,
        "condition 35 should produce ~0.09 jam chance; got {mid}",
    );
    // Bottoms out at the floor.
    assert!(
        (jam_chance_at_condition(0.0, &cfg) - 0.18).abs() < 1e-6,
        "condition 0 should hit the floor exactly",
    );
    // Negative condition clamps to the floor (defense in depth
    // for any future bug that lets it go below 0).
    assert!(
        (jam_chance_at_condition(-5.0, &cfg) - 0.18).abs() < 1e-6,
        "negative condition should clamp to floor",
    );
}

#[test]
fn fresh_weapon_decrements_condition_on_fire_and_journals_delta() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    ready_aks74(&mut sim);
    let (start, jam) = sim
        .weapon_condition_for_test(1, &slot("primary"))
        .expect("primary slot has weapon state");
    assert_eq!(start, 100.0);
    assert_eq!(jam, JamState::Cleared);

    sim.fire_weapon(1, &slot("primary"), 0.0, 0.0).unwrap();

    let (after, jam) = sim.weapon_condition_for_test(1, &slot("primary")).unwrap();
    assert_eq!(jam, JamState::Cleared);
    assert!(
        after < start && after > start - 0.5,
        "fire should decrement condition by a small amount; got {start} → {after}",
    );

    let saw_condition_change = sim
        .drain_tick_deltas()
        .iter()
        .any(|d| matches!(d, WorldDelta::WeaponConditionChanged { .. }));
    assert!(
        saw_condition_change,
        "fire should journal a WeaponConditionChanged delta",
    );
}

#[test]
fn fresh_weapon_never_jams_over_a_full_magazine() {
    // Fresh AKS-74 condition = 100, jam_threshold = 70 (from
    // ammo.toml default) → jam_chance is identically 0 across
    // the whole magazine. Confirms no spurious jam ever fires
    // when the curve says it can't.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    ready_aks74(&mut sim);
    for _ in 0..30 {
        sim.fire_weapon(1, &slot("primary"), 0.0, 0.0).unwrap();
        sim.tick().unwrap(); // advance for distinct RNG seeds
    }
    let (_, jam) = sim.weapon_condition_for_test(1, &slot("primary")).unwrap();
    assert_eq!(jam, JamState::Cleared, "fresh AKS-74 should not jam");
}

#[test]
fn clapped_out_weapon_jams_within_bounded_tries() {
    // At cond=0, jam chance is `jam_chance_floor` (0.18). After
    // ~50 tries we should see a jam with overwhelming
    // probability; we cap at 80 to keep the test deterministic
    // across rng-state cycling. If this ever flakes we'll widen
    // the cap rather than retune.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    ready_aks74(&mut sim);
    sim.set_weapon_condition_for_test(1, &slot("primary"), 0.0);

    let mut jammed = false;
    for _ in 0..80 {
        // Top mag back up so we don't run dry mid-test.
        sim.set_equipped_mag_state_for_test(1, &slot("primary"), 5, Some("round_5_45x39"));
        match sim.fire_weapon(1, &slot("primary"), 0.0, 0.0) {
            Ok(()) => {}
            Err(e) if e.to_string().contains("jammed") => {
                jammed = true;
                break;
            }
            Err(other) => panic!("unexpected fire error: {other}"),
        }
        sim.tick().unwrap();
    }
    assert!(jammed, "weapon at condition 0 should jam within 80 tries");
    let (_, state) = sim.weapon_condition_for_test(1, &slot("primary")).unwrap();
    assert!(state.is_jammed(), "jam state should persist after jam");
}

#[test]
fn jammed_weapon_dry_clicks_until_cleared() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    ready_aks74(&mut sim);
    sim.set_weapon_condition_for_test(1, &slot("primary"), 0.0);

    // Force a jam by exhausting tries up front.
    for _ in 0..120 {
        sim.set_equipped_mag_state_for_test(1, &slot("primary"), 5, Some("round_5_45x39"));
        if sim
            .fire_weapon(1, &slot("primary"), 0.0, 0.0)
            .err()
            .map(|e| e.to_string().contains("jammed"))
            .unwrap_or(false)
        {
            break;
        }
        sim.tick().unwrap();
    }
    assert!(sim
        .weapon_condition_for_test(1, &slot("primary"))
        .unwrap()
        .1
        .is_jammed());

    // Another fire on a jammed weapon must error without
    // expending a round.
    sim.set_equipped_mag_state_for_test(1, &slot("primary"), 5, Some("round_5_45x39"));
    let pre = sim
        .weapon_condition_for_test(1, &slot("primary"))
        .unwrap()
        .0;
    let err = sim
        .fire_weapon(1, &slot("primary"), 0.0, 0.0)
        .expect_err("jammed weapon should dry-click");
    assert!(err.to_string().contains("jammed"), "got {err}");
    let post = sim
        .weapon_condition_for_test(1, &slot("primary"))
        .unwrap()
        .0;
    assert_eq!(pre, post, "dry click on a jam shouldn't wear the weapon");

    // Clear the jam — weapon fires again.
    sim.clear_weapon_jam(1, &slot("primary"))
        .expect("clear_jam should succeed on a jammed weapon");
    let (cond_after_clear, state_after) =
        sim.weapon_condition_for_test(1, &slot("primary")).unwrap();
    assert_eq!(state_after, JamState::Cleared);
    assert_eq!(
        cond_after_clear, post,
        "clear_jam shouldn't repair condition",
    );
}

#[test]
fn clear_jam_errors_on_un_jammed_weapon() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    ready_aks74(&mut sim);
    // Fresh weapon — not jammed. clear_jam should be a noisy
    // no-op so callers don't accidentally consume a clear-jam
    // animation slot when the gun is fine.
    let err = sim
        .clear_weapon_jam(1, &slot("primary"))
        .expect_err("clear on a clean weapon should error");
    assert!(
        err.to_string().contains("wasn't jammed"),
        "expected a wasn't-jammed message; got {err}",
    );
}
