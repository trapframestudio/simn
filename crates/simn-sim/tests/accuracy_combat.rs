//! Per-NPC `accuracy` stat → `npc_combat` hit-chance multiplier.
//! Fourth behavior integration of the `NpcCharacter` substrate (see
//! `docs/book/src/planning/npc-character-authoring-plan.md` §6.1).

use simn_sim::systems::accuracy_hit_multiplier;
use simn_sim::{BodyPart, RegionGraph, Sim};
use tempfile::TempDir;

fn quiet_sim(_dir: &TempDir) -> Sim {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.set_active_region(1);
    sim
}

#[test]
fn accuracy_multiplier_endpoints() {
    let lo = accuracy_hit_multiplier(0);
    let mid = accuracy_hit_multiplier(50);
    let hi = accuracy_hit_multiplier(100);
    assert!((lo - 0.7).abs() < 0.001, "lo={}", lo);
    assert!((mid - 1.0).abs() < 0.001, "mid={}", mid);
    assert!((hi - 1.3).abs() < 0.001, "hi={}", hi);
}

#[test]
fn accuracy_multiplier_monotonic() {
    let mut last = -1.0_f32;
    for a in 0..=100 {
        let m = accuracy_hit_multiplier(a);
        assert!(m > last, "non-monotonic at {}: {} <= {}", a, m, last);
        last = m;
    }
}

#[test]
fn accuracy_wiring_smoke_test() {
    // Smoke-check that the accuracy multiplier is plumbed into
    // `npc_combat`'s hit-chance formula at all. A high-accuracy
    // shooter at the FIRST fire interval should successfully damage
    // a hostile target placed under their aggro at point-blank
    // range. We don't compare absolute hit counts here — the per-NPC
    // wandering and the shared tick-seeded RNG make multi-pair
    // ratio tests flaky. A behavioral A/B comparison waits for a
    // controlled-fire test harness (or projectile ballistics, where
    // accuracy → aim cone instead of hit-chance multiplier).
    //
    // PWA vs Looters: hostile per the canonical relation table.
    // Required for `npc_aggro` to actually evaluate the pair (and
    // populate `LosCache`) — same-faction pairs short-circuit before
    // LOS is sampled, so the post-PR-#TBD LOS gate in `npc_combat`
    // would block the shot.
    let dir = TempDir::new().unwrap();
    let mut sim = quiet_sim(&dir);
    let shooter = sim.spawn_npc_for_test("pwa", 1, [0.0, 0.0, 0.0], None);
    let target = sim.spawn_npc_for_test("looters", 1, [5.0, 0.0, 0.0], None);
    sim.set_npc_accuracy_for_test(shooter, 100);
    sim.set_npc_aggro_for_test(shooter, target);
    sim.move_npc_for_test(shooter, [0.0, 0.0, 0.0], 1);
    sim.move_npc_for_test(target, [5.0, 0.0, 0.0], 1);
    // FIRE_INTERVAL_TICKS=50; one fire interval × ~0.91 hit rate
    // (close range × 100 accuracy) leaves the target wounded at
    // very high probability. Loop a handful of fire intervals to
    // make false negatives vanishingly small — at 200 ticks the
    // target is either (a) wounded and tracked or (b) dead and
    // despawned. Both are valid "wiring works" outcomes; pre-
    // iteration-5-14 the procedurally-seeded bases tended to pull
    // shooter + target apart and (a) was the typical result, but
    // post-Phase-C the test maps have no procedural scatter so
    // both NPCs camp at point-blank and (b) is common. Accept
    // either signal.
    for _ in 0..400 {
        sim.tick().unwrap();
    }
    let mut found_target_hp: Option<f32> = None;
    sim.each_npc(|v| {
        if v.id == target {
            found_target_hp = Some(v.health.current);
        }
    });
    match found_target_hp {
        Some(hp) => assert!(
            hp < 100.0 - f32::EPSILON,
            "target still at full HP — npc_combat never fired or accuracy multiplier zeroed it",
        ),
        None => {
            // Target despawned mid-combat — they took enough damage
            // to die, which is an even stronger signal that the
            // wiring works.
        }
    }
    let _ = (shooter, BodyPart::Torso);
}
