//! Phase 3B — loot pool registry + quest-reward bias.
//!
//! Coverage:
//! - Registry parses the shipped TOML without errors.
//! - `lookup` finds exact entries and falls back through
//!   `(faction, tier, family) → (faction, 1, family) →
//!   (nomads, tier, family) → (nomads, 1, family)`.
//! - `roll_one` produces items inside the configured weight band
//!   over a 1000-iteration sample.
//! - `roll_quest_reward` with K=5 skews the distribution toward
//!   rare entries vs K=1.
//! - `LootContainerRegistry::weighted_pick_for_difficulty` picks
//!   small crates at difficulty 1 and large caches at difficulty 5.
//! - `LootContainerDef::pick_family` honours the family weights.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use simn_sim::loot_containers::LootContainerRegistry;
use simn_sim::loot_pools::{quest_lottery_k, LootPoolRegistry};
use std::collections::HashMap;

#[test]
fn pool_registry_loads_with_pwa_bandits_wanderers_attuned() {
    let r = LootPoolRegistry::load();
    assert!(!r.is_empty(), "loot_pools.toml should have pools");
    // Sanity: every shipped (faction, tier 1) × family combo must
    // resolve to a pool (either directly or via nomads
    // fallback).
    for faction in ["coalition", "raiders", "nomads", "the_order"] {
        for family in [
            "weapons",
            "magazines",
            "ammo",
            "armor",
            "medical",
            "food",
            "tools",
            "junk",
        ] {
            let pool = r.lookup(faction, 1, family);
            assert!(
                pool.is_some(),
                "no pool / fallback for ({faction}, 1, {family})",
            );
            let pool = pool.unwrap();
            assert!(
                !pool.entries.is_empty(),
                "({faction}, 1, {family}) pool has no entries",
            );
            for e in &pool.entries {
                assert!(e.count_min <= e.count_max, "{} bad count range", e.id);
            }
        }
    }
}

#[test]
fn lookup_falls_back_to_wanderers_on_unknown_faction() {
    let r = LootPoolRegistry::load();
    // `coalition` knows about `weapons`. `mod_a` doesn't exist; fallback
    // resolves to nomads tier 1.
    let coalition = r
        .lookup("coalition", 1, "weapons")
        .expect("coalition weapons present");
    let unknown = r
        .lookup("a_mod_faction_we_dont_have", 1, "weapons")
        .expect("fallback should resolve");
    assert_eq!(unknown.faction, "nomads");
    assert_ne!(coalition.faction, unknown.faction);
}

#[test]
fn lookup_falls_back_to_tier_1_within_faction() {
    let r = LootPoolRegistry::load();
    // tier 2 isn't shipped for coalition weapons; should fall back to coalition
    // tier 1 (NOT nomads tier 2 / 1).
    let resolved = r
        .lookup("coalition", 2, "weapons")
        .expect("fallback should resolve");
    assert_eq!(resolved.faction, "coalition");
    assert_eq!(resolved.depth_tier, 1);
}

#[test]
fn roll_one_distribution_tracks_weights_within_tolerance() {
    let r = LootPoolRegistry::load();
    let pool = r
        .lookup("coalition", 1, "ammo")
        .expect("coalition ammo pool exists");
    let total_w: u32 = pool.entries.iter().map(|e| e.weight).sum();

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let mut counts: HashMap<String, u32> = HashMap::new();
    let n: u32 = 5000;
    for _ in 0..n {
        let item = r
            .roll_one(&mut rng, "coalition", 1, "ammo")
            .expect("non-empty pool");
        *counts.entry(item.id.0.clone()).or_default() += 1;
    }
    // Each entry should land within ±5 percentage points of its
    // expected share over 5000 rolls.
    for entry in &pool.entries {
        let observed = *counts.get(&entry.id).unwrap_or(&0) as f64 / n as f64;
        let expected = entry.weight as f64 / total_w as f64;
        let diff = (observed - expected).abs();
        assert!(
            diff < 0.05,
            "{}: observed {:.3} vs expected {:.3} (diff {:.3} > 0.05)",
            entry.id,
            observed,
            expected,
            diff,
        );
    }
}

#[test]
fn quest_reward_bias_skews_toward_rare_entries() {
    let r = LootPoolRegistry::load();
    let pool = r
        .lookup("coalition", 1, "ammo")
        .expect("coalition ammo pool exists");
    // Pick the rarest (lowest-weight) entry id for the assertion.
    let rare = pool
        .entries
        .iter()
        .min_by_key(|e| e.weight)
        .expect("non-empty")
        .id
        .clone();

    let n: u32 = 5000;

    // K=1 — normal weighted pick.
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut k1_rare = 0u32;
    for _ in 0..n {
        let item = r
            .roll_quest_reward(&mut rng, "coalition", 1, "ammo", 1)
            .unwrap();
        if item.id.0 == rare {
            k1_rare += 1;
        }
    }

    // K=5 — strong rare bias.
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut k5_rare = 0u32;
    for _ in 0..n {
        let item = r
            .roll_quest_reward(&mut rng, "coalition", 1, "ammo", 5)
            .unwrap();
        if item.id.0 == rare {
            k5_rare += 1;
        }
    }

    assert!(
        k5_rare > k1_rare,
        "quest reward at K=5 should pull rare entry more often than K=1 (saw {k5_rare} vs {k1_rare})",
    );
    // Loose lower bound — K=5 should pull the rare entry at
    // *least* 2× as often as K=1 on a 5-entry pool. This is well
    // below the theoretical lift (~5× for K=5 on lowest-weight)
    // but keeps the test resilient to seed variance.
    assert!(
        k5_rare >= k1_rare * 2,
        "K=5 bias too weak: {k5_rare} vs {k1_rare}",
    );
}

#[test]
fn quest_lottery_k_difficulty_curve() {
    assert_eq!(quest_lottery_k(0), 1);
    assert_eq!(quest_lottery_k(1), 1);
    assert_eq!(quest_lottery_k(2), 1);
    assert_eq!(quest_lottery_k(3), 2);
    assert_eq!(quest_lottery_k(4), 3);
    assert_eq!(quest_lottery_k(5), 5);
    assert_eq!(quest_lottery_k(99), 5);
}

#[test]
fn difficulty_picks_small_crates_at_low_and_large_at_high() {
    let r = LootContainerRegistry::load();
    let mut rng = ChaCha8Rng::seed_from_u64(7);

    // Difficulty 1 should mostly produce small_crate.
    let mut small_d1 = 0u32;
    let mut large_d1 = 0u32;
    for _ in 0..1000 {
        let kind = r.weighted_pick_for_difficulty(&mut rng, 1).unwrap();
        match kind.id.as_str() {
            "small_crate" => small_d1 += 1,
            "large_cache" => large_d1 += 1,
            _ => {}
        }
    }
    assert!(
        small_d1 > large_d1 * 3,
        "difficulty 1 should pick small_crate >>> large_cache (saw {small_d1} vs {large_d1})",
    );

    // Difficulty 5 should mostly produce large_cache.
    let mut small_d5 = 0u32;
    let mut large_d5 = 0u32;
    for _ in 0..1000 {
        let kind = r.weighted_pick_for_difficulty(&mut rng, 5).unwrap();
        match kind.id.as_str() {
            "small_crate" => small_d5 += 1,
            "large_cache" => large_d5 += 1,
            _ => {}
        }
    }
    assert!(
        large_d5 > small_d5 * 3,
        "difficulty 5 should pick large_cache >>> small_crate (saw {large_d5} vs {small_d5})",
    );
}

#[test]
fn pick_family_honors_kind_weights() {
    let r = LootContainerRegistry::load();
    let small = r.get("small_crate").expect("small_crate present");
    let mut rng = ChaCha8Rng::seed_from_u64(13);

    let mut counts: HashMap<String, u32> = HashMap::new();
    for _ in 0..2000 {
        if let Some(fam) = small.pick_family(&mut rng) {
            *counts.entry(fam.to_string()).or_default() += 1;
        }
    }
    // small_crate's heaviest family is `ammo` (weight 30).
    // `weapons` isn't in its weights at all, so it should never
    // appear.
    let ammo = *counts.get("ammo").unwrap_or(&0);
    let weapons = *counts.get("weapons").unwrap_or(&0);
    assert!(
        ammo > 400,
        "ammo under-represented in small_crate rolls: {ammo}"
    );
    assert_eq!(
        weapons, 0,
        "weapons shouldn't be in small_crate's family pool"
    );
}

#[test]
fn items_per_roll_respects_range() {
    let r = LootContainerRegistry::load();
    let medium = r.get("medium_stash").expect("medium_stash present");
    let mut rng = ChaCha8Rng::seed_from_u64(21);
    let mut lo_seen = u32::MAX;
    let mut hi_seen = 0u32;
    for _ in 0..1000 {
        let n = medium.roll_items_per_roll(&mut rng);
        lo_seen = lo_seen.min(n);
        hi_seen = hi_seen.max(n);
    }
    // medium_stash ships items_per_roll = [4, 7].
    assert!(
        lo_seen >= 4 && hi_seen <= 7,
        "items_per_roll out of range: saw [{lo_seen}, {hi_seen}]",
    );
}
