//! `NpcCharacter` substrate tests. Per
//! `docs/book/src/planning/npc-character-authoring-plan.md` step 1 —
//! deterministic identity (`CharacterId`) + stat block (`NpcStats`)
//! seeded from `(npc_id, faction_id)`.

use simn_sim::{NameRegistry, NpcCharacter, NpcStats, PersonalityArchetype, RegionGraph, Sim};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn fresh_npc_has_character() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0; 3], None);
    let c = sim
        .npc_character_for_test(id)
        .expect("npc has NpcCharacter");
    // Stats live in the documented 0..=100 range.
    let s = c.stats;
    for v in [
        s.accuracy,
        s.perception,
        s.stealth,
        s.strength,
        s.endurance,
        s.marksmanship,
        s.leadership,
        s.luck,
    ] {
        assert!(v <= 100, "stat out of range: {}", v);
    }
}

#[test]
fn character_id_stable_per_identity() {
    // Same (npc_id, faction_id) → same CharacterId. Different
    // factions or different ids → different CharacterId.
    let dir = TempDir::new().unwrap();
    let sim = quiet_sim(&dir);
    let registry = sim.faction_registry();
    let pwa = registry.id_of("pwa").unwrap();
    let looters = registry.id_of("looters").unwrap();
    let nid = simn_sim::NpcId(7);
    let a = NpcCharacter::derive_id(nid, pwa);
    let b = NpcCharacter::derive_id(nid, pwa);
    assert_eq!(a, b, "same inputs → same id");
    let c = NpcCharacter::derive_id(nid, looters);
    assert_ne!(a, c, "different faction → different id");
    let d = NpcCharacter::derive_id(simn_sim::NpcId(8), pwa);
    assert_ne!(a, d, "different npc_id → different id");
}

#[test]
fn stats_deterministic_from_identity() {
    // Two rolls with the same (npc_id, faction_id, base_aggression)
    // produce identical stats.
    let dir = TempDir::new().unwrap();
    let sim = quiet_sim(&dir);
    let pwa = sim.faction_registry().id_of("pwa").unwrap();
    let arche = PersonalityArchetype::from_faction_name("pwa");
    let names = NameRegistry::load();
    let nid = simn_sim::NpcId(42);
    let weights = std::collections::HashMap::new();
    let a = NpcCharacter::roll(nid, pwa, arche, 0.6, &names, &weights, None);
    let b = NpcCharacter::roll(nid, pwa, arche, 0.6, &names, &weights, None);
    assert_eq!(a, b);
}

#[test]
fn aggression_nudges_combat_stats_upward() {
    // High base_aggression should produce higher mean accuracy +
    // marksmanship across a population than low base_aggression.
    // We sample 30 NPCs at each level so the law of large numbers
    // smooths over per-NPC variance.
    let dir = TempDir::new().unwrap();
    let sim = quiet_sim(&dir);
    let pwa = sim.faction_registry().id_of("pwa").unwrap();
    let arche = PersonalityArchetype::from_faction_name("pwa");
    let names = NameRegistry::load();

    let mut hi_sum: u32 = 0;
    let mut lo_sum: u32 = 0;
    let weights = std::collections::HashMap::new();
    for i in 0..30u64 {
        let nid = simn_sim::NpcId(1_000 + i);
        let hi = NpcCharacter::roll(nid, pwa, arche, 1.0, &names, &weights, None);
        let lo = NpcCharacter::roll(nid, pwa, arche, 0.0, &names, &weights, None);
        hi_sum += u32::from(hi.stats.accuracy) + u32::from(hi.stats.marksmanship);
        lo_sum += u32::from(lo.stats.accuracy) + u32::from(lo.stats.marksmanship);
    }
    // The combat nudge is +20 per stat at aggression=1.0, so over
    // 30 NPCs × 2 stats the gap should be ~1200. We accept any
    // positive gap as the property check; large-margin assertion
    // makes for noisy tests if the constant changes later.
    assert!(
        hi_sum > lo_sum,
        "aggressive faction should have higher mean accuracy + marksmanship — hi={}, lo={}",
        hi_sum,
        lo_sum
    );
}

#[test]
fn spawned_npc_character_id_matches_derive() {
    // The live spawn path (`spawn_npc_for_test`) and the standalone
    // `NpcCharacter::derive_id` path must agree — that's what makes
    // the "re-roll on snapshot reload" pattern safe.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0; 3], None);
    let pwa = sim.faction_registry().id_of("pwa").unwrap();
    let expected = NpcCharacter::derive_id(id, pwa);
    let c = sim.npc_character_for_test(id).unwrap();
    assert_eq!(c.character_id, expected);
}

#[test]
fn npc_stats_size_is_bounded() {
    // Sanity check that NpcStats is u8 × 8 = 8 bytes. The plan
    // budget is ~500 bytes per character including future fields,
    // so the stat block staying tiny matters.
    assert!(std::mem::size_of::<NpcStats>() <= 16);
}
