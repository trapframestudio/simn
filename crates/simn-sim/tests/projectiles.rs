//! End-to-end projectile tick tests. These run the full host
//! path — spawn a `Projectile` entity (via the delta-replay seam
//! below, since commit 3 wires the real `Sim::fire_weapon` path),
//! call `Sim::tick` a few times, assert on emitted deltas + ECS
//! state.

use simn_sim::{ItemId, ProjectileId, RegionGraph, SavePaths, Sim, WorldDelta};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

fn fresh_sim(_dir: &TempDir) -> Sim {
    // No-disk, no-NPC variant. Persistence-roundtrip test below
    // constructs its own Sim::new explicitly.
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

/// Spawn a projectile entity by replaying a `ProjectileSpawned`
/// delta. The delta arm in `apply_delta` is idempotent + works
/// on auth sims too; it's the same entry point a load-from-
/// journal would use to restore an in-flight projectile.
fn spawn_via_delta(
    sim: &mut Sim,
    id: u64,
    round_id: &str,
    origin: [f32; 3],
    velocity: [f32; 3],
    max_range_m: f32,
) -> ProjectileId {
    sim.upsert_player(1, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
    let pid = ProjectileId(id);
    let delta = WorldDelta::ProjectileSpawned {
        id: pid,
        source_steam_id: 1,
        source_npc_id: None,
        round_id: ItemId::from(round_id),
        variant: simn_sim::AmmoVariant::default(),
        origin,
        velocity,
        max_range_m,
        spawned_tick: sim.current_tick(),
    };
    sim.apply_external_delta(&delta);
    pid
}

#[test]
fn projectile_drops_under_gravity_over_half_second() {
    // Fire horizontally (+Z) at 100 m/s; after ~0.5s the bullet
    // should have dropped ~1.2m (0.5 * 9.81 * 0.5²) minus small
    // drag corrections. Assert within a generous band because
    // drag + integrator drift add noise.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    let _id = spawn_via_delta(
        &mut sim,
        1,
        "round_5_45x39",
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 100.0],
        1000.0,
    );
    // Tick 10 times at 50ms each (Phase 1 default) ≈ 0.5 s.
    for _ in 0..10 {
        sim.tick().unwrap();
    }
    // Drain deltas; there should be no ProjectileImpacted yet
    // (no NPCs in range, and max_range far exceeds traveled).
    let deltas: Vec<_> = sim.drain_tick_deltas();
    assert!(
        !deltas
            .iter()
            .any(|d| matches!(d, WorldDelta::ProjectileImpacted { .. })),
        "unexpected impact — no targets in scene"
    );
    // Verify Y dropped.
    let ys: Vec<f32> = sim
        .collect_projectile_positions_for_test()
        .iter()
        .map(|p| p[1])
        .collect();
    let y = ys
        .first()
        .copied()
        .expect("projectile still alive after 0.5s flight");
    assert!(
        y < 10.0 && y > 6.0,
        "projectile Y should fall from 10 into [6, 10); got {y}"
    );
}

#[test]
fn projectile_exits_world_despawns_with_null_impact() {
    // Short max_range so we hit the out-of-range branch fast.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    spawn_via_delta(
        &mut sim,
        42,
        "round_5_45x39",
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 100.0],
        5.0,
    );
    // Tick until the projectile runs out of range (< 2 ticks at
    // 100 m/s over a 5m cap — drain a few to be safe).
    for _ in 0..5 {
        sim.tick().unwrap();
    }
    let deltas = sim.drain_tick_deltas();
    let impacts: Vec<&WorldDelta> = deltas
        .iter()
        .filter(|d| matches!(d, WorldDelta::ProjectileImpacted { .. }))
        .collect();
    assert_eq!(impacts.len(), 1, "expected exactly one impact delta");
    match impacts[0] {
        WorldDelta::ProjectileImpacted {
            hit_npc,
            damage_applied,
            penetrated,
            ..
        } => {
            assert!(hit_npc.is_none(), "no NPC in scene — impact is range-out");
            assert_eq!(*damage_applied, 0.0);
            assert!(!*penetrated);
        }
        _ => unreachable!(),
    }
    // Projectile entity should be gone.
    assert!(
        sim.collect_projectile_positions_for_test().is_empty(),
        "projectile entity should be despawned after out-of-range"
    );
}

#[test]
fn projectile_hits_head_on_straight_shot() {
    // Plant an NPC, fire a level round at head height.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    sim.upsert_player(1, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
    let npc_id = sim.spawn_npc_for_test("looters", 1, [0.0, 0.0, 5.0], None);
    spawn_via_delta(
        &mut sim,
        7,
        "round_5_45x39",
        [0.0, 1.75, 0.0],
        [0.0, 0.0, 100.0],
        100.0,
    );
    for _ in 0..3 {
        sim.tick().unwrap();
    }
    let deltas = sim.drain_tick_deltas();
    let impact = deltas
        .iter()
        .find_map(|d| match d {
            WorldDelta::ProjectileImpacted {
                hit_npc: Some(id),
                body_part: Some(part),
                damage_applied,
                ..
            } => Some((*id, *part, *damage_applied)),
            _ => None,
        })
        .expect("expected a ProjectileImpacted with an NPC target");
    assert_eq!(impact.0, npc_id);
    assert_eq!(
        impact.1,
        simn_sim::BodyPart::Head,
        "level shot at y=1.75 should hit head"
    );
    assert!(
        impact.2 > 0.0,
        "some damage should apply (commit 2: flat damage_soft)"
    );
}

#[test]
fn snapshot_round_trips_projectile_entity() {
    let dir = TempDir::new().unwrap();
    {
        let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
        spawn_via_delta(
            &mut sim,
            99,
            "round_5_45x39",
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 50.0],
            500.0,
        );
        // Flush with shutdown so the snapshot lands on disk.
        sim.shutdown().unwrap();
    }
    let mut sim = Sim::load(paths(&dir)).unwrap();
    let positions = sim.collect_projectile_positions_for_test();
    assert_eq!(
        positions.len(),
        1,
        "projectile should survive snapshot round-trip"
    );
}
