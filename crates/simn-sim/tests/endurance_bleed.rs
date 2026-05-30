//! Per-NPC `endurance` stat → bleed-rate damping. Second behavior
//! integration of the `NpcCharacter` substrate (see
//! `docs/book/src/planning/npc-character-authoring-plan.md` §6.1).

use simn_sim::systems::bleed_rate_multiplier;
use simn_sim::{BodyPart, RegionGraph, Sim};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn bleed_multiplier_endpoints() {
    // Linear in [0.7, 1.3] inverted across endurance 0..=100,
    // 1.0 at endurance 50.
    let lo = bleed_rate_multiplier(0);
    let mid = bleed_rate_multiplier(50);
    let hi = bleed_rate_multiplier(100);
    assert!((lo - 1.3).abs() < 0.001, "lo={}", lo);
    assert!((mid - 1.0).abs() < 0.001, "mid={}", mid);
    assert!((hi - 0.7).abs() < 0.001, "hi={}", hi);
}

#[test]
fn bleed_multiplier_monotonic_in_endurance() {
    // Higher endurance ⇒ smaller multiplier (less drain). No
    // regressions across the integer domain.
    let mut last = f32::INFINITY;
    for e in 0..=100 {
        let m = bleed_rate_multiplier(e);
        assert!(
            m < last,
            "non-monotonic at endurance={}: {} >= {}",
            e,
            m,
            last
        );
        last = m;
    }
}

#[test]
fn high_endurance_npc_bleeds_slower_than_low() {
    // Two NPCs, identical wound, ticked the same number of times.
    // The high-endurance NPC should retain more torso HP at the end.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let tough = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let frail = sim.spawn_npc_for_test("pwa", 1, [10.0, 0.0, 0.0], None);
    sim.set_npc_endurance_for_test(tough, 100);
    sim.set_npc_endurance_for_test(frail, 0);
    // Apply the same wound to both — 30 damage each, which spawns a
    // heavy bleed wound (severity 4-5). Use the journaled per-NPC
    // damage entry point so a Wound is added too.
    sim.apply_damage_to_npc_part(tough, BodyPart::Torso, 30.0)
        .unwrap();
    sim.apply_damage_to_npc_part(frail, BodyPart::Torso, 30.0)
        .unwrap();
    // Tick forward for 10 in-game seconds (~200 ticks at 50ms tick
    // dt). Severity-4 bleed × 2.0 HP/sec × 10s = 20 HP base; with
    // multipliers tough takes 14 HP, frail takes 26 HP.
    for _ in 0..200 {
        sim.tick().unwrap();
    }
    let npcs = sim.npcs_in_region(1);
    let tough_view = npcs.iter().find(|n| n.id == tough).unwrap();
    let frail_view = npcs.iter().find(|n| n.id == frail).unwrap();
    let tough_torso = tough_view.body_parts.unwrap().torso;
    let frail_torso = frail_view.body_parts.unwrap().torso;
    assert!(
        tough_torso > frail_torso,
        "tough torso ({}) should be > frail torso ({}) — endurance \
         multiplier inverted?",
        tough_torso,
        frail_torso
    );
    // Sanity: gap should be roughly proportional to the multiplier
    // ratio (1.3 / 0.7 ≈ 1.86×). Guard against trivially-passing
    // tests where both NPCs already hit the HP floor.
    assert!(
        tough_torso > 50.0,
        "tough torso ({}) shouldn't have bled out — test exposes \
         a multiplier bug",
        tough_torso
    );
}

#[test]
fn player_bleed_unchanged_by_npc_character_absence() {
    // Players don't have NpcCharacter, so the multiplier is the
    // baseline 1.0. Confirm a wounded player drains at the legacy
    // rate — same wound, same tick budget, expected drain unchanged.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    sim.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    sim.apply_damage_to_part(1, BodyPart::Torso, 30.0).unwrap();
    let pre_view = sim.player_view(1).unwrap();
    let pre_torso = pre_view.body_parts.torso;
    for _ in 0..200 {
        sim.tick().unwrap();
    }
    let post_view = sim.player_view(1).unwrap();
    let post_torso = post_view.body_parts.torso;
    let drained = pre_torso - post_torso;
    // Severity 4-5 wound × ~10s of bleed at multiplier 1.0:
    // expect ~20 HP drained (severity 4 × 0.5 = 2.0 HP/s × 10s).
    // Allow a wide tolerance — exact severity rolls into 4 or 5.
    assert!(
        drained > 15.0 && drained < 30.0,
        "player drained {} HP, expected ~20 — baseline multiplier \
         broken?",
        drained
    );
}
