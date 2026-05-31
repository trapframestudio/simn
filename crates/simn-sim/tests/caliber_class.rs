//! `CaliberClass` taxonomy + per-ammo TOML configurability. Per
//! `weapons-plan.md` §4 + `dismemberment-plan.md` §5.

use simn_sim::{audible_radius_m, CaliberAudibleBand, CaliberClass, ItemRegistry, WorldEventKind};

#[test]
fn audible_radius_orders_by_class_loudness() {
    let pdw = audible_radius_m(&WorldEventKind::Gunshot {
        caliber_class: CaliberClass::PDW,
    });
    let pistol = audible_radius_m(&WorldEventKind::Gunshot {
        caliber_class: CaliberClass::Pistol,
    });
    let shotgun = audible_radius_m(&WorldEventKind::Gunshot {
        caliber_class: CaliberClass::Shotgun,
    });
    let intermediate = audible_radius_m(&WorldEventKind::Gunshot {
        caliber_class: CaliberClass::Intermediate,
    });
    let full = audible_radius_m(&WorldEventKind::Gunshot {
        caliber_class: CaliberClass::FullPowerRifle,
    });
    let magnum = audible_radius_m(&WorldEventKind::Gunshot {
        caliber_class: CaliberClass::Magnum,
    });
    let am = audible_radius_m(&WorldEventKind::Gunshot {
        caliber_class: CaliberClass::AntiMateriel,
    });
    // Sanity: monotonic up the loudness ladder.
    assert!(pdw < pistol);
    assert!(pistol < shotgun);
    assert!(shotgun < intermediate);
    assert!(intermediate < full);
    assert!(full < magnum);
    assert!(magnum < am);
}

#[test]
fn audible_band_groups_classes_into_legacy_bands() {
    assert_eq!(
        CaliberClass::Pistol.audible_band(),
        CaliberAudibleBand::Light
    );
    assert_eq!(CaliberClass::PDW.audible_band(), CaliberAudibleBand::Light);
    assert_eq!(
        CaliberClass::Shotgun.audible_band(),
        CaliberAudibleBand::Light
    );
    assert_eq!(
        CaliberClass::Intermediate.audible_band(),
        CaliberAudibleBand::Medium
    );
    assert_eq!(
        CaliberClass::FullPowerRifle.audible_band(),
        CaliberAudibleBand::Heavy
    );
    assert_eq!(
        CaliberClass::Magnum.audible_band(),
        CaliberAudibleBand::Heavy
    );
    assert_eq!(
        CaliberClass::AntiMateriel.audible_band(),
        CaliberAudibleBand::Heavy
    );
}

#[test]
fn ammo_config_carries_caliber_class_from_toml() {
    let registry = ItemRegistry::load();
    let pistol_round = registry
        .get(&simn_sim::ItemId("round_9x18".to_string()))
        .expect("9x18 in registry");
    let ammo = pistol_round
        .ammo_config
        .as_ref()
        .expect("9x18 has AmmoConfig");
    assert_eq!(ammo.caliber_class, CaliberClass::Pistol);

    let rifle = registry
        .get(&simn_sim::ItemId("round_5_45x39".to_string()))
        .expect("5.45 in registry");
    let ammo = rifle.ammo_config.as_ref().unwrap();
    assert_eq!(ammo.caliber_class, CaliberClass::Intermediate);

    let shotgun = registry
        .get(&simn_sim::ItemId("round_12ga_buckshot".to_string()))
        .expect("12ga in registry");
    let ammo = shotgun.ammo_config.as_ref().unwrap();
    assert_eq!(ammo.caliber_class, CaliberClass::Shotgun);
}

#[test]
fn roster_covers_every_caliber_class() {
    // Spot-check that the expanded ammo TOML reaches every
    // CaliberClass variant. Catches a regression where a roster
    // expansion drops one of the seven buckets.
    let registry = ItemRegistry::load();
    let mut seen: std::collections::HashSet<CaliberClass> = std::collections::HashSet::new();
    for def in registry.iter() {
        if let Some(ammo) = &def.ammo_config {
            seen.insert(ammo.caliber_class);
        }
    }
    use CaliberClass::*;
    for cls in [
        Pistol,
        PDW,
        Intermediate,
        FullPowerRifle,
        Magnum,
        AntiMateriel,
        Shotgun,
    ] {
        assert!(
            seen.contains(&cls),
            "ammo roster missing CaliberClass::{:?}",
            cls
        );
    }
}

#[test]
fn modern_calibers_spot_check() {
    // A handful of canonical rounds with their expected caliber
    // classes, sourced from `weapons-plan.md` §4.2 + the the genre
    // expansion. Catches typos in `caliber_class` keys.
    let registry = ItemRegistry::load();
    let cases: &[(&str, CaliberClass)] = &[
        ("round_9x19", CaliberClass::Pistol),
        ("round_45acp", CaliberClass::Pistol),
        ("round_22lr", CaliberClass::Pistol),
        ("round_57x28", CaliberClass::PDW),
        ("round_46x30", CaliberClass::PDW),
        ("round_556x45_m193", CaliberClass::Intermediate),
        ("round_762x39", CaliberClass::Intermediate),
        ("round_9x39_sp5", CaliberClass::Intermediate),
        ("round_300blk", CaliberClass::Intermediate),
        ("round_762x54r", CaliberClass::FullPowerRifle),
        ("round_308win", CaliberClass::FullPowerRifle),
        ("round_338lapua", CaliberClass::Magnum),
        ("round_50bmg", CaliberClass::AntiMateriel),
        ("round_127x108", CaliberClass::AntiMateriel),
        ("round_145x114", CaliberClass::AntiMateriel),
        ("round_20ga_buck", CaliberClass::Shotgun),
        ("round_410_slug", CaliberClass::Shotgun),
    ];
    for (id, expected) in cases {
        let def = registry
            .get(&simn_sim::ItemId(id.to_string()))
            .unwrap_or_else(|| panic!("{} should be in registry", id));
        let ammo = def
            .ammo_config
            .as_ref()
            .unwrap_or_else(|| panic!("{} should have ammo_config", id));
        assert_eq!(
            ammo.caliber_class, *expected,
            "{} expected {:?}, got {:?}",
            id, expected, ammo.caliber_class
        );
    }
}

#[test]
fn subsonic_rounds_have_velocity_below_speed_of_sound() {
    // Sanity: rounds explicitly designed as subsonic (9x39, .300 BLK
    // subsonic, dragon's breath shotgun) must have velocity < 340
    // m/s. Prevents a velocity typo from accidentally creating a
    // "subsonic" round that's actually supersonic.
    let registry = ItemRegistry::load();
    let subsonic_ids = [
        "round_9x39_sp5",
        "round_9x39_sp6",
        "round_300blk_sub",
        "round_12ga_dragons_breath",
    ];
    for id in subsonic_ids {
        let def = registry.get(&simn_sim::ItemId(id.to_string())).unwrap();
        let ammo = def.ammo_config.as_ref().unwrap();
        assert!(
            ammo.muzzle_velocity_mps < 340.0,
            "{} declared subsonic but velocity = {}",
            id,
            ammo.muzzle_velocity_mps
        );
    }
}
