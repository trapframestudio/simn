//! `NameRegistry` substrate. Per
//! `docs/book/src/planning/npc-character-authoring-plan.md` step 2 —
//! every NPC rolls a multicultural first + last name from one of
//! eight nationality buckets at character-derive time.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use simn_sim::{
    NameRegistry, NationalityBucket, NpcCharacter, NpcId, PersonalityArchetype, RegionGraph, Sim,
};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn registry_loads_all_buckets() {
    let reg = NameRegistry::load();
    for &b in &NationalityBucket::ALL {
        let firsts = reg.first_names(b);
        let lasts = reg.last_names(b);
        assert!(
            firsts.len() >= 30,
            "{:?} should ship with at least 30 first names, got {}",
            b,
            firsts.len()
        );
        assert!(
            lasts.len() >= 30,
            "{:?} should ship with at least 30 last names, got {}",
            b,
            lasts.len()
        );
    }
}

#[test]
fn registry_first_names_are_non_empty_and_distinct() {
    // Sanity: blank lines / dupes shouldn't survive the loader. We
    // don't enforce strict uniqueness across buckets (some
    // first-name overlap between cultures is expected) but within a
    // bucket we shouldn't have trivial duplicates.
    let reg = NameRegistry::load();
    for &b in &NationalityBucket::ALL {
        let firsts = reg.first_names(b);
        for n in &firsts {
            assert!(!n.is_empty(), "{:?}: empty first name", b);
        }
        let mut sorted: Vec<&str> = firsts.iter().copied().collect();
        sorted.sort();
        let len_before = sorted.len();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            len_before,
            "{:?} has duplicate first names",
            b
        );
    }
}

#[test]
fn example_pack_buckets_are_distinct() {
    // Regression guard. The embedded example pack must give each bucket
    // a different name pool. If buckets ever collapse to identical files
    // (as a lazy placeholder would), the per-bucket and weighting tests
    // become vacuous, so assert the cross-bucket union exceeds any
    // single bucket.
    let reg = NameRegistry::load();
    let single = reg.first_names(NationalityBucket::American).len();
    let union: std::collections::BTreeSet<String> = NationalityBucket::ALL
        .iter()
        .flat_map(|&b| reg.first_names(b).into_iter().map(str::to_string))
        .collect();
    assert!(
        union.len() > single,
        "buckets share one pool ({} unique across all vs {} per bucket); \
         the example pack must give buckets distinct names",
        union.len(),
        single
    );
}

#[test]
fn roll_produces_first_last_format() {
    let reg = NameRegistry::load();
    let mut rng = ChaCha8Rng::seed_from_u64(1234);
    let (_bucket, name) = reg.roll(&mut rng);
    let parts: Vec<_> = name.splitn(2, ' ').collect();
    assert_eq!(
        parts.len(),
        2,
        "name should be 'First Last', got {:?}",
        name
    );
    assert!(!parts[0].is_empty());
    assert!(!parts[1].is_empty());
}

#[test]
fn roll_is_deterministic_for_seed() {
    let reg = NameRegistry::load();
    let a = {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        reg.roll(&mut rng)
    };
    let b = {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        reg.roll(&mut rng)
    };
    assert_eq!(a, b);
}

#[test]
fn roll_covers_all_buckets_over_population() {
    // Sample 500 rolls; with uniform default weights all 8 buckets
    // should appear at least once. Catches a regression where one
    // bucket file is empty / mis-keyed.
    let reg = NameRegistry::load();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..500 {
        let (b, _) = reg.roll(&mut rng);
        seen.insert(b);
        if seen.len() == NationalityBucket::ALL.len() {
            break;
        }
    }
    assert_eq!(
        seen.len(),
        NationalityBucket::ALL.len(),
        "not all buckets surfaced over 500 rolls; got {:?}",
        seen
    );
}

#[test]
fn fresh_npc_has_name_and_nationality() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("coalition", 1, [0.0; 3], None);
    let c = sim.npc_character_for_test(id).expect("npc has character");
    assert!(!c.name.is_empty(), "name shouldn't be empty");
    assert!(
        c.name.contains(' '),
        "name should be 'First Last' shape, got {:?}",
        c.name
    );
    // nationality is one of the 8 buckets — implicit if the field
    // even exists in the deserialized struct.
    let _ = c.nationality;
}

#[test]
fn from_name_round_trips_all_buckets() {
    for &b in &NationalityBucket::ALL {
        assert_eq!(NationalityBucket::from_name(b.name()), Some(b));
    }
    assert_eq!(NationalityBucket::from_name("not_a_bucket"), None);
}

#[test]
fn roll_for_faction_skews_toward_weighted_bucket() {
    // Smugglers's TOML lean is `latin_american = 7` against ~8 other
    // buckets at low weights. Over a sample, Latin-American names
    // should dominate.
    let reg = NameRegistry::load();
    let mut weights: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    weights.insert("latin_american".to_string(), 7);
    weights.insert("american".to_string(), 1);
    let mut rng = ChaCha8Rng::seed_from_u64(2024);
    let mut latin = 0;
    let mut american = 0;
    for _ in 0..400 {
        let (b, _) = reg.roll_for_faction(&mut rng, &weights);
        match b {
            NationalityBucket::LatinAmerican => latin += 1,
            NationalityBucket::American => american += 1,
            _ => {} // weighted-out bucket should never surface
        }
    }
    assert!(
        latin > american,
        "expected latin_american dominance, got {} vs {}",
        latin,
        american
    );
    assert!(
        latin >= 250,
        "latin_american should claim most rolls, got {}/400",
        latin
    );
}

#[test]
fn roll_for_faction_empty_weights_falls_back_to_uniform() {
    // The common case: faction with no nationality_weights TOML entry
    // gets the uniform global distribution. Just checking it doesn't
    // panic and produces a name from any bucket.
    let reg = NameRegistry::load();
    let weights: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let (_b, name) = reg.roll_for_faction(&mut rng, &weights);
    assert!(name.contains(' '));
}

#[test]
fn unknown_bucket_keys_drop_silently() {
    // If a faction TOML has a typo, the bucket is dropped from the
    // weight list — no panic, just falls back to uniform if nothing
    // remains.
    let reg = NameRegistry::load();
    let mut weights: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    weights.insert("typo_bucket".to_string(), 9);
    weights.insert("another_typo".to_string(), 4);
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let (_b, name) = reg.roll_for_faction(&mut rng, &weights);
    assert!(name.contains(' '));
}

#[test]
fn npc_name_deterministic_from_identity() {
    // Two separate sims, same NPC id + faction — the rolled name
    // should match exactly. Confirms re-roll on snapshot reload
    // yields the same name.
    let pwa_arche = PersonalityArchetype::Disciplined;
    let names = NameRegistry::load();
    let dir = TempDir::new().unwrap();
    let sim = quiet_sim(&dir);
    let coalition = sim.faction_registry().id_of("coalition").unwrap();
    let weights = std::collections::HashMap::new();
    let nid = NpcId(123);
    let a = NpcCharacter::roll(nid, coalition, pwa_arche, 0.6, &names, &weights, None);
    let b = NpcCharacter::roll(nid, coalition, pwa_arche, 0.6, &names, &weights, None);
    assert_eq!(a.name, b.name);
    assert_eq!(a.nationality, b.nationality);
}
