//! `PersonalityTraits` substrate + goal-arbitration personality bias.
//! Per `docs/book/src/planning/npc-character-authoring-plan.md` step 2
//! and §5.

use simn_sim::systems::personality_bias_for_objective;
use simn_sim::{
    NameRegistry, NpcCharacter, NpcId, PersonalityArchetype, PersonalityTraits, RegionGraph, Sim,
};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn fresh_npc_has_personality_traits() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("coalition", 1, [0.0; 3], None);
    let c = sim.npc_character_for_test(id).expect("npc has character");
    // Trait struct is the right shape — unwrap and read all 10
    // fields without panicking.
    let p = c.personality;
    let _ = (
        p.aggressive,
        p.cautious,
        p.curious,
        p.greedy,
        p.loyal,
        p.bloodthirsty,
        p.social,
        p.solitary,
        p.disciplined,
        p.reckless,
    );
}

#[test]
fn personality_deterministic_from_identity() {
    // Same (npc_id, faction_id, archetype) → identical personality.
    let dir = TempDir::new().unwrap();
    let sim = quiet_sim(&dir);
    let faction = sim.faction_registry().id_of("coalition").unwrap();
    let arche = PersonalityArchetype::Disciplined;
    let names = NameRegistry::load();
    let weights = std::collections::HashMap::new();
    let nid = NpcId(42);
    let a = NpcCharacter::roll(nid, faction, arche, 0.6, &names, &weights, None);
    let b = NpcCharacter::roll(nid, faction, arche, 0.6, &names, &weights, None);
    assert_eq!(a.personality, b.personality);
}

#[test]
fn archetype_skews_observed_traits_across_population() {
    // Sample 100 NPCs of each archetype and check the dominant
    // trait fires more often than its inverse.
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);

    let mut disciplined_yes_count = 0;
    let mut greedy_yes_count = 0;
    for _ in 0..200 {
        let t = PersonalityArchetype::Disciplined.roll_traits(&mut rng);
        if t.disciplined {
            disciplined_yes_count += 1;
        }
        let t = PersonalityArchetype::Greedy.roll_traits(&mut rng);
        if t.greedy {
            greedy_yes_count += 1;
        }
    }
    // Disciplined-archetype NPCs are 0.85 likely to have
    // disciplined=true → expect ~170/200. Greedy NPCs 0.85 likely to
    // have greedy=true. Wide tolerance, just need to be ≥ 130 and
    // distinct from a coin-flip baseline.
    assert!(
        disciplined_yes_count >= 130,
        "Disciplined archetype should skew toward disciplined=true; got {}/200",
        disciplined_yes_count
    );
    assert!(
        greedy_yes_count >= 130,
        "Greedy archetype should skew toward greedy=true; got {}/200",
        greedy_yes_count
    );
}

#[test]
fn curious_boosts_investigate_priority() {
    // Curious personality on an Investigate objective should land
    // a multiplier > 1.0; Aggressive on the same objective < 1.0.
    use simn_sim::SquadObjective;
    let curious = PersonalityTraits {
        curious: true,
        ..Default::default()
    };
    let aggressive = PersonalityTraits {
        aggressive: true,
        ..Default::default()
    };
    let obj = SquadObjective::Investigate {
        target: [0.0; 3],
        expires_at: 0,
    };
    let m_curious = personality_bias_for_objective(&curious, &obj);
    let m_aggressive = personality_bias_for_objective(&aggressive, &obj);
    assert!(
        m_curious > 1.0,
        "curious × Investigate should boost: {}",
        m_curious
    );
    assert!(
        m_aggressive < 1.0,
        "aggressive × Investigate should dampen: {}",
        m_aggressive
    );
    assert!(m_curious > m_aggressive);
}

#[test]
fn disciplined_boosts_guard_priority() {
    use simn_sim::SquadObjective;
    let disciplined = PersonalityTraits {
        disciplined: true,
        loyal: true,
        ..Default::default()
    };
    let drifter = PersonalityTraits {
        curious: true,
        solitary: true,
        ..Default::default()
    };
    let obj = SquadObjective::Guard {
        base_pos: [0.0; 3],
        expires_at: 0,
        post_key: None,
    };
    let m_disciplined = personality_bias_for_objective(&disciplined, &obj);
    let m_drifter = personality_bias_for_objective(&drifter, &obj);
    assert!(m_disciplined > m_drifter);
}

#[test]
fn no_traits_set_yields_baseline_multiplier() {
    use simn_sim::SquadObjective;
    let blank = PersonalityTraits::default();
    let obj = SquadObjective::Patrol {
        route: vec![[0.0; 3]],
        current_idx: 0,
        expires_at: 0,
    };
    let m = personality_bias_for_objective(&blank, &obj);
    assert!(
        (m - 1.0).abs() < 0.001,
        "blank traits should be 1.0×, got {}",
        m
    );
}

#[test]
fn introduces_drives_curious_yields_hunt() {
    let traits = PersonalityTraits {
        curious: true,
        ..Default::default()
    };
    let drives = traits.introduces_drives();
    assert!(drives.contains(&simn_sim::components::PersonalityDrive::Hunt));
}

#[test]
fn introduces_drives_greedy_yields_loot() {
    let traits = PersonalityTraits {
        greedy: true,
        ..Default::default()
    };
    let drives = traits.introduces_drives();
    assert!(drives.contains(&simn_sim::components::PersonalityDrive::Loot));
}

#[test]
fn introduces_drives_bloodthirsty_yields_bloodsport() {
    let traits = PersonalityTraits {
        bloodthirsty: true,
        ..Default::default()
    };
    let drives = traits.introduces_drives();
    assert!(drives.contains(&simn_sim::components::PersonalityDrive::Bloodsport));
}

#[test]
fn introduces_drives_social_yields_socialize() {
    let traits = PersonalityTraits {
        social: true,
        ..Default::default()
    };
    let drives = traits.introduces_drives();
    assert!(drives.contains(&simn_sim::components::PersonalityDrive::Socialize));
}

#[test]
fn introduces_drives_blank_traits_is_empty() {
    let traits = PersonalityTraits::default();
    assert!(traits.introduces_drives().is_empty());
}

#[test]
fn introduces_drives_multiple_traits_yield_multiple() {
    let traits = PersonalityTraits {
        curious: true,
        greedy: true,
        bloodthirsty: true,
        social: true,
        ..Default::default()
    };
    let drives = traits.introduces_drives();
    assert_eq!(drives.len(), 4);
}

#[test]
fn personality_bias_does_not_preempt_aggro_lane() {
    // Combat lanes are reserved. Even at the most-extreme bias boost,
    // the priority that lands in `ActiveGoal` must stay strictly below
    // `PRIO_INDIVIDUAL_AGGRO` (150). After the 2026-05-27 amplification
    // the raw multiplier exceeds the safe range — the arbiter relies
    // on `biased_priority`'s clamp to keep combat lanes reserved, so
    // assert through that function (not the raw multiply) to verify
    // the actual contract.
    use simn_sim::systems::biased_priority;
    use simn_sim::SquadObjective;
    let max_boost = PersonalityTraits {
        curious: true,
        solitary: true,
        ..Default::default()
    };
    let obj = SquadObjective::Wander { expires_at: 0 };
    let m = personality_bias_for_objective(&max_boost, &obj);
    // The raw scaled value SHOULD exceed 150 — that's what makes the
    // clamp load-bearing. If this assertion ever fails, someone has
    // de-amplified the bias and the clamp is no longer necessary.
    let raw = (80.0_f32 * m).round();
    assert!(
        raw >= 150.0,
        "expected raw bias to exceed aggro tier so the clamp is exercised; got {raw}"
    );
    // The clamped value the arbiter actually applies must stay below
    // the aggro lane.
    let applied = biased_priority(80, m);
    assert!(
        applied < 150,
        "biased_priority must not preempt aggro lane (150); got {applied}"
    );
}
