//! `NpcRank` substrate. Universal STALKER-style threat tier
//! derived from `NpcStats::combat_competence`. Per
//! `docs/book/src/planning/npc-character-authoring-plan.md` step 2.

use simn_sim::{
    NameRegistry, NpcCharacter, NpcId, NpcRank, NpcStats, PersonalityArchetype, RegionGraph, Sim,
};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

fn stats_at(score: u8) -> NpcStats {
    NpcStats {
        accuracy: score,
        perception: score,
        marksmanship: score,
        endurance: score,
        luck: score,
        // these don't contribute to combat_competence
        stealth: 50,
        strength: 50,
        leadership: 50,
    }
}

#[test]
fn from_stats_threshold_endpoints() {
    // 5 stats × value = combat_competence. Pick values that land
    // each rank just past its floor.
    // Score 0 and 200 (5 × 40) both land below the Experienced floor.
    assert_eq!(NpcRank::from_stats(&stats_at(0)), NpcRank::Rookie);
    assert_eq!(NpcRank::from_stats(&stats_at(40)), NpcRank::Rookie);
    // 280/5 = 56 → Experienced floor.
    assert_eq!(NpcRank::from_stats(&stats_at(56)), NpcRank::Experienced);
    // 350/5 = 70 → Veteran floor.
    assert_eq!(NpcRank::from_stats(&stats_at(70)), NpcRank::Veteran);
    // 410/5 = 82 → Master floor.
    assert_eq!(NpcRank::from_stats(&stats_at(82)), NpcRank::Master);
    // 460/5 = 92 → Legend floor.
    assert_eq!(NpcRank::from_stats(&stats_at(92)), NpcRank::Legend);
    // Cap.
    assert_eq!(NpcRank::from_stats(&stats_at(100)), NpcRank::Legend);
}

#[test]
fn from_stats_monotonic_in_competence() {
    // Sweeping competence upward should never produce a lower rank.
    let mut last = NpcRank::Rookie;
    for v in 0..=100u8 {
        let r = NpcRank::from_stats(&stats_at(v));
        assert!(
            r >= last,
            "rank regressed at score={}: {:?} < {:?}",
            v,
            r,
            last
        );
        last = r;
    }
}

#[test]
fn combat_competence_excludes_utility_stats() {
    // strength / leadership / stealth do not contribute. Bumping
    // them shouldn't change the score.
    let base = NpcStats {
        accuracy: 50,
        perception: 50,
        marksmanship: 50,
        endurance: 50,
        luck: 50,
        stealth: 0,
        strength: 0,
        leadership: 0,
    };
    let buffed = NpcStats {
        stealth: 100,
        strength: 100,
        leadership: 100,
        ..base
    };
    assert_eq!(base.combat_competence(), buffed.combat_competence());
    assert_eq!(base.combat_competence(), 250);
}

#[test]
fn rank_labels_are_stable() {
    assert_eq!(NpcRank::Rookie.label(), "Rookie");
    assert_eq!(NpcRank::Experienced.label(), "Experienced");
    assert_eq!(NpcRank::Veteran.label(), "Veteran");
    assert_eq!(NpcRank::Master.label(), "Master");
    assert_eq!(NpcRank::Legend.label(), "Legend");
}

#[test]
fn fresh_npc_has_rank_consistent_with_stats() {
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let id = sim.spawn_npc_for_test("pwa", 1, [0.0; 3], None);
    let c = sim.npc_character_for_test(id).expect("npc has character");
    assert_eq!(c.rank, NpcRank::from_stats(&c.stats));
}

#[test]
fn rank_distribution_skews_low_for_pwa() {
    // Sample 200 NPCs; expect Rookie + Experienced to dominate, with
    // Master + Legend rare. PWA's combat-stat nudge (+0.6 × 20 ≈ 12)
    // pulls the typical sum upward but most NPCs still sit below the
    // 410 Master floor. Wide tolerance — this guards against a
    // floor-table miscalibration that flips the distribution.
    let pwa = PersonalityArchetype::from_faction_name("pwa");
    let dir = TempDir::new().unwrap();
    let sim = quiet_sim(&dir);
    let pwa_id = sim.faction_registry().id_of("pwa").unwrap();
    let names = NameRegistry::load();
    let weights = std::collections::HashMap::new();
    let mut counts = [0u32; 5];
    for i in 0..200u64 {
        let c = NpcCharacter::roll(NpcId(10_000 + i), pwa_id, pwa, 0.6, &names, &weights, None);
        counts[c.rank as usize] += 1;
    }
    let rookie = counts[0];
    let experienced = counts[1];
    let master_plus = counts[3] + counts[4];
    assert!(
        rookie + experienced >= 130,
        "expected most NPCs in Rookie+Experienced, got {} of 200",
        rookie + experienced
    );
    assert!(
        master_plus < 50,
        "Master/Legend should be rare, got {}/200",
        master_plus
    );
}
