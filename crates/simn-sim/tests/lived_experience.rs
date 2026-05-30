//! `NpcCharacter::record_kill` lived-experience rank promotion. Per
//! `docs/book/src/planning/npc-character-authoring-plan.md` step 2:
//! kills accumulated buff effective competence and promote the
//! NPC through rank tiers over their lifetime.

use simn_sim::{
    NameRegistry, NpcCharacter, NpcId, NpcRank, PersonalityArchetype, RegionGraph, Sim,
};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

fn fresh_character() -> NpcCharacter {
    let dir = TempDir::new().unwrap();
    let sim = quiet_sim(&dir);
    let pwa = sim.faction_registry().id_of("pwa").unwrap();
    let arche = PersonalityArchetype::from_faction_name("pwa");
    let names = NameRegistry::load();
    let weights = std::collections::HashMap::new();
    NpcCharacter::roll(NpcId(1), pwa, arche, 0.6, &names, &weights, None)
}

#[test]
fn fresh_character_has_zero_kills() {
    let c = fresh_character();
    assert_eq!(c.kills, 0);
    assert_eq!(c.effective_competence(), c.stats.combat_competence());
}

#[test]
fn record_kill_increments_count() {
    let mut c = fresh_character();
    c.record_kill();
    c.record_kill();
    c.record_kill();
    assert_eq!(c.kills, 3);
}

#[test]
fn record_kill_promotes_rank_eventually() {
    // Even a low-stat NPC should reach Master / Legend after enough
    // kills (≥ 50 by the current KILL_COMPETENCE_BUFF = 3 → +150
    // competence headroom).
    let mut c = fresh_character();
    let initial_rank = c.rank;
    for _ in 0..60 {
        c.record_kill();
    }
    assert!(
        c.rank >= initial_rank,
        "rank shouldn't regress: {:?} → {:?}",
        initial_rank,
        c.rank
    );
    assert!(
        c.rank >= NpcRank::Veteran,
        "60 kills should reach at least Veteran from any starting tier; got {:?}",
        c.rank
    );
}

#[test]
fn effective_competence_caps_at_500() {
    // u16::MAX kills × 3 buff would overflow u32 if unguarded; the
    // function clamps to 500 so the rank function stays in its
    // documented domain.
    let mut c = fresh_character();
    c.kills = u16::MAX;
    assert_eq!(c.effective_competence(), 500);
}

#[test]
fn rank_promotion_is_pure_function_of_kills_and_stats() {
    // Two characters with the same base stats and the same kill
    // count must produce identical effective_competence and rank.
    let dir = TempDir::new().unwrap();
    let sim = quiet_sim(&dir);
    let pwa = sim.faction_registry().id_of("pwa").unwrap();
    let arche = PersonalityArchetype::from_faction_name("pwa");
    let names = NameRegistry::load();
    let weights = std::collections::HashMap::new();
    let mut a = NpcCharacter::roll(NpcId(7), pwa, arche, 0.6, &names, &weights, None);
    let mut b = NpcCharacter::roll(NpcId(7), pwa, arche, 0.6, &names, &weights, None);
    for _ in 0..15 {
        a.record_kill();
        b.record_kill();
    }
    assert_eq!(a.kills, b.kills);
    assert_eq!(a.effective_competence(), b.effective_competence());
    assert_eq!(a.rank, b.rank);
}
