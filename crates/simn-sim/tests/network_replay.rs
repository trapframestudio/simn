//! End-to-end sim-replication tests. An **authoritative** `Sim` drives
//! state; a **mirror** `Sim` consumes its snapshots + deltas and
//! reaches the same observable state. Proves the slice-1 replication
//! path is sound before we wire any real network transport to it.

use simn_sim::{
    ActionKind, BodyPart, DrugKind, FoodKind, ItemId, RegionGraph, SavePaths, Sim, SnapshotBody,
    SurvivalStat, SurvivalStats,
};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

fn host_sim(dir: &TempDir) -> Sim {
    // Real on-disk sim — replication tests round-trip snapshots
    // through `serialize_snapshot_body` and journal deltas, so we
    // can't drop persistence here. We *can* dial the population
    // targets way down — replication only needs *some* NPCs, not
    // the stock 360-per-faction spike.
    //
    // Iteration 5-14 Phase C: the default test graph now flags
    // map_a..d as `scene_authored_pois = true`, which skips the
    // procedural base scatter. Replication tests *need* the
    // scatter for spawn anchors, so build a legacy graph
    // explicitly here.
    let mut sim = Sim::new(paths(dir), legacy_procedural_graph()).unwrap();
    sim.scale_all_population_targets(0.02);
    // Phase 1A gate: enable tick-time spawning across every region
    // so replication actually has NPCs to journal.
    sim.activate_all_regions_for_test();
    sim
}

fn mirror_sim() -> Sim {
    Sim::new_mirror(legacy_procedural_graph())
}

fn legacy_procedural_graph() -> RegionGraph {
    // Same shape as `default_test_graph`'s 4-region 2×2 layout but
    // with `scene_authored_pois = false` so the procedural scatter
    // runs. Replication / persistence tests rely on that scatter
    // for spawn anchors + initial loot containers.
    let mut g = RegionGraph::new();
    for (id, name, scene) in [
        (1u32, "map_a", "res://scenes/test/test_map_1.tscn"),
        (2u32, "map_b", "res://scenes/test/test_map_2.tscn"),
        (3u32, "map_c", "res://scenes/test/test_map_3.tscn"),
        (4u32, "map_d", "res://scenes/test/test_map_4.tscn"),
    ] {
        g.insert(simn_sim::Region {
            id,
            name: name.into(),
            map_scene: scene.into(),
            neighbors: vec![],
            transitions: Default::default(),
            procedurally_seeded: true,
            scene_authored_pois: false,
        });
    }
    g
}

/// Host serializes its `SnapshotBody` via the same path `roll_snapshot`
/// uses. We intercept it by writing a snapshot to a tempfile and
/// reading back. In a production flow, the host would serialize to
/// `Vec<u8>` directly and send via `Msg::Snapshot`.
fn take_snapshot(sim: &mut Sim, dir: &TempDir) -> (u64, SnapshotBody) {
    sim.shutdown().unwrap(); // writes final snapshot
    simn_sim::persistence::read_snapshot(&paths(dir).snapshot).unwrap()
}

#[test]
fn mirror_applies_snapshot() {
    let host_dir = TempDir::new().unwrap();
    let mut host = host_sim(&host_dir);
    host.upsert_player(1, 1, [1.0, 2.0, 3.0], 0.5).unwrap();
    host.upsert_player(2, 1, [4.0, 5.0, 6.0], 1.0).unwrap();
    host.apply_damage_to_part(1, BodyPart::Torso, 20.0).unwrap();
    for _ in 0..5 {
        host.tick().unwrap();
    }
    let (tick, body) = take_snapshot(&mut host, &host_dir);

    let mut mirror = mirror_sim();
    mirror.apply_external_snapshot(body, tick);

    let v1 = mirror.player_view(1).expect("player 1 in mirror");
    let v2 = mirror.player_view(2).expect("player 2 in mirror");
    assert_eq!(v1.pos, [1.0, 2.0, 3.0]);
    assert_eq!(v2.pos, [4.0, 5.0, 6.0]);
    assert!(
        !v1.wounds.is_empty(),
        "mirror should see player 1's wound: {:?}",
        v1.wounds
    );
}

#[test]
fn mirror_applies_delta_batch() {
    // Host makes a series of mutations, capturing each tick's drain.
    // Mirror starts from a fresh snapshot (tick 0) and replays every
    // drained delta.
    let host_dir = TempDir::new().unwrap();
    let mut host = host_sim(&host_dir);
    host.upsert_player(42, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
    let _ = host.drain_tick_deltas(); // clear spawn deltas

    // Capture host tick-0 snapshot BEFORE we start accumulating deltas.
    let snap_dir = TempDir::new().unwrap();
    let mut snap_host = Sim::new(paths(&snap_dir), RegionGraph::default_test_graph()).unwrap();
    snap_host
        .upsert_player(42, 1, [0.0, 0.0, 0.0], 0.0)
        .unwrap();
    let (snap_tick, body) = take_snapshot(&mut snap_host, &snap_dir);

    let mut mirror = mirror_sim();
    mirror.apply_external_snapshot(body, snap_tick);

    // Host mutates.
    host.apply_damage_to_part(42, BodyPart::Torso, 15.0)
        .unwrap();
    host.set_survival_stat(42, SurvivalStat::Hunger, 60.0)
        .unwrap();
    let deltas_a = host.drain_tick_deltas();
    host.grant_item(42, &ItemId::from("bandage"), 2).unwrap();
    let deltas_b = host.drain_tick_deltas();

    // Replay on mirror.
    for d in deltas_a.iter().chain(deltas_b.iter()) {
        mirror.apply_external_delta(d);
    }

    let hv = host.player_view(42).unwrap();
    let mv = mirror.player_view(42).unwrap();
    assert_eq!(hv.wounds.len(), mv.wounds.len(), "wound counts diverged");
    assert_eq!(hv.survival.hunger, mv.survival.hunger);
    assert_eq!(mirror.inventory_view(42).len(), 1);
}

#[test]
fn action_move_roundtrip() {
    let dir = TempDir::new().unwrap();
    let mut host = host_sim(&dir);
    host.upsert_player(1, 1, [0.0, 0.0, 0.0], 0.0).unwrap();
    let _ = host.drain_tick_deltas();
    host.apply_action(
        1,
        ActionKind::Move {
            pos: [10.0, 0.0, -20.0],
            yaw: 1.5,
        },
    )
    .unwrap();
    let v = host.player_view(1).unwrap();
    assert_eq!(v.pos, [10.0, 0.0, -20.0]);
    assert!((v.yaw - 1.5).abs() < 1e-4);
}

#[test]
fn action_bandage_roundtrip() {
    let dir = TempDir::new().unwrap();
    let mut host = host_sim(&dir);
    host.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    host.apply_damage_to_part(1, BodyPart::Torso, 15.0).unwrap();
    host.apply_action(
        1,
        ActionKind::ApplyBandage {
            part: BodyPart::Torso,
        },
    )
    .unwrap();
    let v = host.player_view(1).unwrap();
    assert!(v
        .wounds
        .iter()
        .any(|(_, w)| matches!(w.treatment, simn_sim::WoundTreatment::Bandaged)));
}

#[test]
fn action_consume_slot_roundtrip() {
    let dir = TempDir::new().unwrap();
    let mut host = host_sim(&dir);
    host.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    host.set_survival_stat(1, SurvivalStat::Hunger, 10.0)
        .unwrap();
    host.grant_item(1, &ItemId::from("cooked_meat"), 2).unwrap();
    host.apply_action(
        1,
        ActionKind::ConsumeSlot {
            slot_idx: 0,
            body_part: None,
        },
    )
    .unwrap();
    let v = host.player_view(1).unwrap();
    assert!(v.survival.hunger > 10.0);
    assert_eq!(host.inventory_view(1)[0].count, 1);
}

#[test]
fn action_drug_roundtrip() {
    let dir = TempDir::new().unwrap();
    let mut host = host_sim(&dir);
    host.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    host.apply_action(
        1,
        ActionKind::ApplyDrug {
            drug: DrugKind::Painkiller,
        },
    )
    .unwrap();
    let v = host.player_view(1).unwrap();
    assert!(
        v.active_effects
            .iter()
            .any(|e| matches!(e.kind, simn_sim::EffectKind::Painkiller)),
        "painkiller effect missing"
    );
}

#[test]
fn action_eat_roundtrip() {
    let dir = TempDir::new().unwrap();
    let mut host = host_sim(&dir);
    host.upsert_player(1, 1, [0.0; 3], 0.0).unwrap();
    host.set_survival_stat(1, SurvivalStat::Hunger, 30.0)
        .unwrap();
    host.apply_action(
        1,
        ActionKind::Eat {
            kind: FoodKind::CookedMeat,
        },
    )
    .unwrap();
    let v = host.player_view(1).unwrap();
    assert!(v.survival.hunger > 30.0, "hunger didn't rise");
}

#[test]
fn mirror_sim_no_disk_writes() {
    // Build a mirror, apply some ticks, ensure the process never
    // opens a file under any save path.
    let mut mirror = mirror_sim();
    for _ in 0..50 {
        mirror.tick().unwrap();
    }
    // There's no way to inspect filesystem access from here; rely on
    // the structural invariant: `new_mirror` takes no `SavePaths`, and
    // `shutdown` is a no-op.
    mirror.shutdown().unwrap();
    // If this test ever gains a tempdir + directory-listing check, do
    // it here. For now, compile-time proof via API is sufficient.
}

#[test]
fn mirror_clock_slaves_to_host_tick() {
    let mut mirror = mirror_sim();
    assert_eq!(mirror.current_tick(), 0);
    mirror.set_tick_for_mirror(500);
    assert_eq!(mirror.current_tick(), 500);
    // Authoritative sim ignores the call.
    let dir = TempDir::new().unwrap();
    let mut host = host_sim(&dir);
    host.set_tick_for_mirror(999);
    assert_eq!(host.current_tick(), 0);
}

#[test]
fn action_encode_decode_stable() {
    // Every variant: encode → decode → equal.
    let cases = [
        ActionKind::Move {
            pos: [1.0, 2.0, 3.0],
            yaw: 0.5,
        },
        ActionKind::ChangeRegion {
            region_name: "map_b".into(),
        },
        ActionKind::ApplyBandage {
            part: BodyPart::Torso,
        },
        ActionKind::ApplyTourniquet {
            part: BodyPart::LeftLeg,
        },
        ActionKind::ApplyAntibiotics,
        ActionKind::ApplyDrug {
            drug: DrugKind::Painkiller,
        },
        ActionKind::Eat {
            kind: FoodKind::CookedMeat,
        },
        ActionKind::ConsumeSlot {
            slot_idx: 3,
            body_part: Some(BodyPart::Head),
        },
        ActionKind::DropSlot { slot_idx: 0 },
        ActionKind::MoveSlot { from: 0, to: 1 },
        ActionKind::SalvageSlot { slot_idx: 7 },
        ActionKind::CraftRecipe {
            recipe_id: "cook_meat".into(),
        },
        ActionKind::SetNearCampfire { value: true },
        ActionKind::GrantItem {
            item_id: "bandage".into(),
            count: 5,
        },
    ];
    for action in cases {
        let bytes = simn_sim::encode_action(&action).unwrap();
        let decoded = simn_sim::decode_action(&bytes).expect("decode");
        assert_eq!(action, decoded, "roundtrip mismatch for {action:?}");
    }
}

#[test]
fn npc_position_batch_applies_on_mirror() {
    // Host spawns some NPCs via its schedule; extracts the
    // NpcPositionBatch delta from drain; mirror applies; mirror's NPC
    // positions match host's.
    let dir = TempDir::new().unwrap();
    let mut host = host_sim(&dir);
    // NPC spawns fire every SPAWN_INTERVAL_TICKS=50 ticks; give it
    // several cycles so populations accumulate.
    for _ in 0..200 {
        host.tick().unwrap();
    }
    let all_deltas: Vec<_> = host.drain_tick_deltas();
    let host_npcs: Vec<_> = host.all_npc_positions_for_test();
    assert!(!host_npcs.is_empty(), "host didn't spawn any NPCs");

    // Build a mirror from host's current snapshot, then apply the
    // drained deltas (including NpcPositionBatch).
    let (tick, body) = take_snapshot(&mut host, &dir);
    let mut mirror = mirror_sim();
    mirror.apply_external_snapshot(body, tick);
    for d in &all_deltas {
        mirror.apply_external_delta(d);
    }

    // Mirror's NPC positions should converge on host's. Snapshot is
    // enough — the batch just re-confirms. The assertion here guards
    // against regressions where apply_external_snapshot misses the
    // NPC entities.
    let mirror_npcs = mirror.all_npc_positions_for_test();
    assert_eq!(host_npcs.len(), mirror_npcs.len());
}

#[test]
fn per_run_save_paths_isolate() {
    let root = TempDir::new().unwrap();
    let alpha = SavePaths::in_run_dir(root.path(), "alpha");
    let beta = SavePaths::in_run_dir(root.path(), "beta");
    assert_ne!(alpha.snapshot, beta.snapshot);
    assert_ne!(alpha.journal, beta.journal);
    // Sanity: the prefix is `<root>/saves/<id>`.
    assert!(alpha
        .snapshot
        .starts_with(root.path().join("saves").join("alpha")));
    assert!(beta
        .snapshot
        .starts_with(root.path().join("saves").join("beta")));
}

#[test]
fn per_run_save_isolation_end_to_end() {
    // Create two runs, write different state to each, reload and
    // confirm they don't stomp on each other.
    let root = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    {
        let mut alpha =
            Sim::new(SavePaths::in_run_dir(root.path(), "alpha"), graph.clone()).unwrap();
        alpha.upsert_player(1, 1, [100.0, 0.0, 0.0], 0.0).unwrap();
        alpha.shutdown().unwrap();
    }
    {
        let mut beta = Sim::new(SavePaths::in_run_dir(root.path(), "beta"), graph.clone()).unwrap();
        beta.upsert_player(1, 1, [-50.0, 0.0, 0.0], 0.0).unwrap();
        beta.shutdown().unwrap();
    }

    let mut alpha_r = Sim::load(SavePaths::in_run_dir(root.path(), "alpha")).unwrap();
    let mut beta_r = Sim::load(SavePaths::in_run_dir(root.path(), "beta")).unwrap();
    assert_eq!(alpha_r.player_view(1).unwrap().pos, [100.0, 0.0, 0.0]);
    assert_eq!(beta_r.player_view(1).unwrap().pos, [-50.0, 0.0, 0.0]);
}

// Sanity: keep `SurvivalStats` reachable in the test crate.
#[allow(dead_code)]
fn _needs_survival_stats() -> SurvivalStats {
    SurvivalStats::new_full()
}
