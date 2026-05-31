//! Phase 4A v1 — NPC fire spawns visible (cosmetic) projectile
//! entities alongside the existing dice-damage path.
//!
//! Coverage:
//! - `Sim::npc_fire_projectile` mints a `Projectile` entity with
//!   `source_npc_id = Some(...)` and the configured shooter
//!   region.
//! - The projectile carries the default NPC round id and a
//!   velocity in the rough target direction.
//! - Accuracy 100 produces a velocity vector nearly parallel to
//!   the shooter→target line; accuracy 0 produces visible jitter.
//! - The projectile tick's NPC-source carve-out: NPC-fired
//!   projectiles fly to range without applying damage (the
//!   dice path in `npc_combat` is the v1 damage source).

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use simn_sim::components::{NpcId, Position, Projectile};
use simn_sim::items::ItemId;
use simn_sim::region::RegionGraph;
use simn_sim::Sim;

/// Convenience for the existing tests — the rifle-caliber default
/// that NPC fire used in Phase 4A v1 before per-faction selection.
fn default_round() -> ItemId {
    ItemId::from("round_5_45x39")
}

fn fresh_sim(seed: u64) -> Sim {
    Sim::new_in_memory_with_seed(RegionGraph::default_test_graph(), seed)
}

fn first_region(_sim: &mut Sim) -> simn_sim::region::RegionId {
    // Iteration 5-14 Phase C: `default_test_graph` test maps no
    // longer auto-seed bases (the gate skips that for scene-
    // authored regions). Region id 1 (map_a) is always valid in
    // the default graph.
    1
}

fn projectile_snapshot(sim: &mut Sim) -> Vec<(Projectile, [f32; 3])> {
    let world = sim.world_for_test();
    let mut q = world.query::<(&Projectile, &Position)>();
    q.iter(world).map(|(p, pos)| (p.clone(), pos.0)).collect()
}

#[test]
fn npc_fire_spawns_projectile_with_npc_source() {
    let mut sim = fresh_sim(7);
    let region = first_region(&mut sim);
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    sim.npc_fire_projectile(
        NpcId(1),
        [0.0, 1.7, 0.0],
        region,
        [50.0, 1.7, 0.0],
        /*accuracy=*/ 80,
        default_round(),
        &mut rng,
    )
    .expect("fire should succeed for the default round");

    let projs = projectile_snapshot(&mut sim);
    assert_eq!(projs.len(), 1, "exactly one projectile spawned");
    let (proj, _pos) = &projs[0];
    assert_eq!(proj.source_steam_id, 0, "NPC projectiles carry steam_id=0");
    assert_eq!(
        proj.source_npc_id,
        Some(NpcId(1)),
        "source_npc_id should round-trip the shooter",
    );
    // Velocity should be roughly along +X (target was at +50 on
    // X axis); accuracy 80 keeps yaw jitter under ~1°.
    let vx = proj.vel[0];
    let vz = proj.vel[2];
    assert!(vx > 0.0, "x velocity should be positive (target is +X)");
    assert!(
        vx.abs() > vz.abs() * 5.0,
        "high-accuracy shot should be ~aligned with +X (vx={vx}, vz={vz})",
    );
}

#[test]
fn high_accuracy_has_less_yaw_spread_than_low_accuracy() {
    let mut sim_hi = fresh_sim(13);
    let mut sim_lo = fresh_sim(13);
    let region = first_region(&mut sim_hi);

    // Fire 200 shots each with same target + same RNG seed but
    // different accuracy. Measure the spread of yaw deviations
    // from the target direction.
    let collect = |sim: &mut Sim, accuracy: u8| -> f32 {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let mut sum_sq = 0.0_f32;
        const N: u32 = 200;
        for i in 0..N {
            sim.npc_fire_projectile(
                NpcId(u64::from(i + 1)),
                [0.0, 1.7, 0.0],
                region,
                [0.0, 1.7, 100.0], // target dead ahead on +Z
                accuracy,
                default_round(),
                &mut rng,
            )
            .unwrap();
        }
        let world = sim.world_for_test();
        let mut q = world.query::<&Projectile>();
        for p in q.iter(world) {
            // Yaw error: angle between vel-xz and ideal (+Z).
            let yaw = p.vel[0].atan2(p.vel[2]);
            sum_sq += yaw * yaw;
        }
        (sum_sq / f32::from(u16::try_from(N).unwrap_or(1))).sqrt()
    };

    let rms_hi = collect(&mut sim_hi, 100);
    let rms_lo = collect(&mut sim_lo, 0);
    assert!(
        rms_lo > rms_hi * 5.0,
        "low-accuracy spread should dominate high-accuracy spread (rms_hi={rms_hi:.4}, rms_lo={rms_lo:.4})",
    );
    // Sanity: accuracy 100 should give near-zero spread.
    assert!(
        rms_hi < 0.005,
        "accuracy 100 should produce near-zero yaw spread; got rms={rms_hi:.4}",
    );
}

#[test]
fn npc_projectiles_damage_online_npcs() {
    // Phase 4A v2 — NPC projectiles now DO apply damage on hit
    // (the v1 cosmetic-only carve-out was lifted). Fire a
    // sustained stream at a hostile-faction online NPC at close
    // range with acc=100; expect HP loss + a wound entry.
    use simn_sim::components::BodyPart;
    let mut sim = fresh_sim(21);
    sim.activate_all_regions_for_test();
    let shooter = sim.spawn_npc_for_test("coalition", 1, [0.0, 0.0, 0.0], None);
    let target = sim.spawn_npc_for_test("looters", 1, [10.0, 0.0, 0.0], None);
    sim.set_npc_accuracy_for_test(shooter, 100);
    sim.set_npc_yaw_for_test(shooter, std::f32::consts::FRAC_PI_2);
    let _ = (shooter, BodyPart::Torso);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    // shooter / target positions are foot-level (Y=0); the
    // helper internally offsets to the muzzle (Y=muzzle_up) and
    // aims at the target's center mass.
    for _ in 0..50 {
        sim.npc_fire_projectile(
            shooter,
            [0.0, 0.0, 0.0],
            1,
            [10.0, 0.0, 0.0],
            100,
            default_round(),
            &mut rng,
        )
        .unwrap();
    }
    // Tick projectiles through their full range.
    for _ in 0..40 {
        sim.tick().unwrap();
    }
    // All projectiles should be gone.
    assert_eq!(projectile_snapshot(&mut sim).len(), 0);
    // Target should have taken damage somewhere. The projectile
    // path can land on any body part (the hitbox closest to the
    // swept segment wins) — limb hits don't drop overall Health
    // because `vital_min` reads head + torso only. So we assert
    // on the body parts themselves: at least one part shows wear
    // below its DEFAULT_MAX.
    //
    // If the target was killed outright (vital part dropped to
    // zero) it'll have been despawned by `npc_death_check`. In
    // that case the chronicle records a Combat death.
    use simn_sim::components::BodyPart as BP;
    let mut parts_summary: Option<(f32, f32, f32, f32, f32, f32)> = None;
    let mut target_alive = false;
    sim.each_npc(|v| {
        if v.id == target {
            target_alive = true;
            if let Some(bp) = v.body_parts {
                parts_summary = Some((
                    bp.head,
                    bp.torso,
                    bp.left_arm,
                    bp.right_arm,
                    bp.left_leg,
                    bp.right_leg,
                ));
            }
        }
    });
    let _ = BP::Torso;
    let max = simn_sim::components::BodyParts::DEFAULT_MAX;

    // Exactly one of two outcomes is valid, and BOTH branches make a
    // hard assertion (no vacuous path):
    //   (a) the target survived 50 close-range acc=100 shots and shows
    //       wear on at least one body part, OR
    //   (b) the target was killed and despawned, with the chronicle
    //       recording a Combat death.
    let damage_observed = match parts_summary {
        Some((h, t, la, ra, ll, rl)) => {
            h < max - f32::EPSILON
                || t < max - f32::EPSILON
                || la < max - f32::EPSILON
                || ra < max - f32::EPSILON
                || ll < max - f32::EPSILON
                || rl < max - f32::EPSILON
        }
        None => false,
    };
    let combat_death = sim
        .chronicle_get(target)
        .map(|rec| matches!(rec.death_cause, Some(simn_sim::DeathCause::Combat { .. })))
        .unwrap_or(false);

    if target_alive {
        // Target survived: damage on a body part is the only valid
        // signal here.
        assert!(
            damage_observed,
            "alive target should show projectile damage on at least one body part; \
             parts_summary={parts_summary:?}",
        );
    } else {
        // Target despawned (killed): the chronicle must record a
        // Combat death.
        assert!(
            combat_death,
            "target despawned but chronicle didn't record a combat death",
        );
    }
    // Whichever branch we took, exactly one of the two signals holds.
    assert!(
        damage_observed ^ combat_death,
        "exactly one of {{damage observed, combat death}} should hold \
         (damage_observed={damage_observed}, combat_death={combat_death})",
    );
}

// --- Phase 4B v1: faction-flavored round selection -----------------

#[test]
fn faction_default_round_matches_authored_table() {
    use simn_sim::default_npc_round_for_faction;
    // Spot-check the shipped mapping. Coalition / Directorate / Consortium fire
    // rifle-caliber intermediates; raiders + nomads fire
    // pistol-caliber 9×18; Order fire 7.62×39; unknown factions
    // fall back to 5.45×39 (neutral default).
    let cases: &[(&str, &str)] = &[
        ("coalition", "round_5_45x39"),
        ("directorate", "round_556x45_m193"),
        ("consortium", "round_556x45_m193"),
        ("homesteaders", "round_5_45x39"),
        ("the_order", "round_762x39"),
        ("syndicate", "round_9x19"),
        ("raiders", "round_9x18"),
        ("nomads", "round_9x18"),
        ("coalition_vanguard", "round_5_45x39"),
        ("not_a_real_faction", "round_5_45x39"), // fallback
    ];
    for (faction, expected) in cases {
        let actual = default_npc_round_for_faction(faction);
        assert_eq!(
            actual.0, *expected,
            "{} should fire {} but got {}",
            faction, expected, actual.0,
        );
    }
}

#[test]
fn each_authored_faction_round_has_ammo_config() {
    use simn_sim::default_npc_round_for_faction;
    // Catches typos / pool drift: every faction's authored
    // default round must exist in the item registry and have a
    // populated `ammo_config` block. Otherwise `npc_fire_projectile`
    // would log warnings + drop the shot at runtime.
    let mut sim = fresh_sim(33);
    let world = sim.world_for_test();
    let registry = world.resource::<simn_sim::items::ItemRegistry>().clone();
    for faction in [
        "coalition",
        "directorate",
        "consortium",
        "homesteaders",
        "the_order",
        "syndicate",
        "raiders",
        "nomads",
        "coalition_vanguard",
    ] {
        let round = default_npc_round_for_faction(faction);
        let def = registry.get(&round).unwrap_or_else(|| {
            panic!("faction {faction}'s round {round:?} missing from ammo.toml")
        });
        assert!(
            def.ammo_config.is_some(),
            "faction {faction}'s round {:?} has no ammo_config — fire path can't resolve a muzzle velocity",
            round,
        );
    }
}

#[test]
fn npc_fire_with_pistol_caliber_produces_lower_muzzle_velocity() {
    // Raiders fire 9×18 (~315 m/s muzzle velocity), Coalition fire
    // 5.45×39 (~880 m/s). Confirms the velocity actually lands
    // on the projectile from the round's ammo_config, not a
    // hardcoded constant.
    let mut sim = fresh_sim(77);
    let region = first_region(&mut sim);
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    sim.npc_fire_projectile(
        NpcId(1),
        [0.0, 1.7, 0.0],
        region,
        [100.0, 1.7, 0.0],
        100,
        ItemId::from("round_9x18"),
        &mut rng,
    )
    .unwrap();
    sim.npc_fire_projectile(
        NpcId(2),
        [0.0, 1.7, 0.0],
        region,
        [100.0, 1.7, 0.0],
        100,
        ItemId::from("round_5_45x39"),
        &mut rng,
    )
    .unwrap();
    let projs = projectile_snapshot(&mut sim);
    assert_eq!(projs.len(), 2);
    let speed = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let pistol = speed(projs[0].0.vel);
    let rifle = speed(projs[1].0.vel);
    assert!(
        rifle > pistol * 1.5,
        "rifle muzzle velocity ({rifle:.0} m/s) should be >1.5× pistol ({pistol:.0} m/s)",
    );
}

// --- Phase 4B v2: ammo variant tag --------------------------------

#[test]
fn ammo_variant_default_is_fmj_for_legacy_data() {
    // The `#[serde(default)]` on `AmmoConfig.variant` should
    // mean ammo entries that don't declare a variant parse to
    // `AmmoVariant::Fmj`. Baseline rounds (`round_9x18`,
    // `round_5_45x39`, etc.) intentionally omit the field to
    // keep the TOML minimal — confirm they decode correctly.
    use simn_sim::AmmoVariant;
    let mut sim = fresh_sim(0);
    let registry = sim
        .world_for_test()
        .resource::<simn_sim::items::ItemRegistry>()
        .clone();
    for round in [
        "round_9x18",
        "round_5_45x39",
        "round_762x39",
        "round_556x45_m193",
        "round_45acp",
    ] {
        let ac = registry
            .get(&ItemId::from(round))
            .and_then(|def| def.ammo_config.as_ref())
            .unwrap_or_else(|| panic!("{round} missing ammo_config"));
        assert_eq!(
            ac.variant,
            AmmoVariant::Fmj,
            "{round} should default to FMJ when variant is omitted; got {:?}",
            ac.variant,
        );
    }
}

#[test]
fn ammo_variant_tags_parse_correctly_for_hp_ap_overpressure_tracer() {
    // Hand-picked rounds across the four non-FMJ variants. If any
    // of these drift in the TOML the variant tag → caliber-name
    // mapping breaks for AI / loot / FX consumers downstream.
    use simn_sim::AmmoVariant;
    let mut sim = fresh_sim(0);
    let registry = sim
        .world_for_test()
        .resource::<simn_sim::items::ItemRegistry>()
        .clone();
    let cases: &[(&str, AmmoVariant)] = &[
        ("round_9x18_hp", AmmoVariant::Hp),
        ("round_556x45_jhp", AmmoVariant::Hp),
        ("round_9x18_ap", AmmoVariant::Ap),
        ("round_556x45_m855a1", AmmoVariant::Ap),
        ("round_45acp_p", AmmoVariant::Overpressure),
        ("round_5_45x39_t", AmmoVariant::Tracer),
        ("round_556x45_tracer", AmmoVariant::Tracer),
        ("round_762x54r_t46", AmmoVariant::Tracer),
    ];
    for (round, expected) in cases {
        let ac = registry
            .get(&ItemId::from(*round))
            .and_then(|def| def.ammo_config.as_ref())
            .unwrap_or_else(|| panic!("{round} missing ammo_config"));
        assert_eq!(
            ac.variant, *expected,
            "{round} should tag {:?}; got {:?}",
            expected, ac.variant,
        );
    }
}

#[test]
fn projectile_spawned_delta_carries_resolved_variant() {
    // Fire two projectiles — one FMJ, one HP — and confirm the
    // last `WorldDelta::ProjectileSpawned` records reflect the
    // round's variant tag. Verifies the resolution path through
    // `Sim::resolve_round_variant` + delta emission.
    use simn_sim::AmmoVariant;
    let mut sim = fresh_sim(99);
    let region = first_region(&mut sim);
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    sim.npc_fire_projectile(
        NpcId(1),
        [0.0, 1.7, 0.0],
        region,
        [50.0, 1.7, 0.0],
        100,
        ItemId::from("round_5_45x39"), // FMJ
        &mut rng,
    )
    .unwrap();
    sim.npc_fire_projectile(
        NpcId(2),
        [0.0, 1.7, 0.0],
        region,
        [50.0, 1.7, 0.0],
        100,
        ItemId::from("round_9x18_hp"), // HP
        &mut rng,
    )
    .unwrap();
    sim.npc_fire_projectile(
        NpcId(3),
        [0.0, 1.7, 0.0],
        region,
        [50.0, 1.7, 0.0],
        100,
        ItemId::from("round_5_45x39_t"), // Tracer
        &mut rng,
    )
    .unwrap();
    let deltas = sim.drain_tick_deltas();
    let variants: Vec<AmmoVariant> = deltas
        .iter()
        .filter_map(|d| match d {
            simn_sim::WorldDelta::ProjectileSpawned { variant, .. } => Some(*variant),
            _ => None,
        })
        .collect();
    assert_eq!(
        variants,
        vec![AmmoVariant::Fmj, AmmoVariant::Hp, AmmoVariant::Tracer],
        "spawn deltas should carry the resolved variant tag in fire order",
    );
}
