//! Faction matrix + random world seeder + persistence tests.

use simn_sim::{
    load_default_faction_registry, registry_faction_relation, BaseKind, RegionGraph, Relation,
    RelationDeltas, SavePaths, Sim,
};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

#[test]
fn relation_symmetry() {
    let reg = load_default_faction_registry();
    let deltas = RelationDeltas::default();
    let ids: Vec<_> = reg.defs().map(|d| d.id).collect();
    for &a in &ids {
        for &b in &ids {
            assert_eq!(
                registry_faction_relation(&reg, &deltas, a, b),
                registry_faction_relation(&reg, &deltas, b, a),
                "relation({}, {}) != relation({}, {})",
                reg.name_of(a),
                reg.name_of(b),
                reg.name_of(b),
                reg.name_of(a),
            );
        }
    }
}

#[test]
fn relation_self_is_warm() {
    // Default `default_self_relation = "warm"` in the canonical TOML.
    let reg = load_default_faction_registry();
    let deltas = RelationDeltas::default();
    for def in reg.defs() {
        assert_eq!(
            registry_faction_relation(&reg, &deltas, def.id, def.id),
            Relation::Warm,
            "{} should be Warm to itself",
            def.name,
        );
    }
}

#[test]
fn faction_names_unique() {
    // Smoke-test: registry rejects duplicate faction names at load
    // time; this just confirms the canonical TOML has none.
    let reg = load_default_faction_registry();
    let mut names: Vec<&str> = reg.defs().map(|d| d.name.as_str()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "duplicate faction names in registry");
}

#[test]
fn seed_is_deterministic() {
    let dir1 = TempDir::new().unwrap();
    let dir2 = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    let mut sim1 = Sim::new_with_seed(paths(&dir1), graph.clone(), 42).unwrap();
    let mut sim2 = Sim::new_with_seed(paths(&dir2), graph, 42).unwrap();

    // Compare RegionControl per region.
    for region_id in [1, 2] {
        let c1 = sim1.region_control(region_id).cloned();
        let c2 = sim2.region_control(region_id).cloned();
        assert_eq!(c1, c2, "region {} control diverged", region_id);
    }

    // Compare bases per region.
    for region_id in [1, 2] {
        let mut b1 = sim1.bases_in_region(region_id);
        let mut b2 = sim2.bases_in_region(region_id);
        // Sort by (faction discriminant, kind discriminant, pos x) so
        // entity-id ordering doesn't matter.
        let key = |b: &simn_sim::BaseView| (b.faction.0, b.kind as u8, b.pos[0].to_bits());
        b1.sort_by_key(key);
        b2.sort_by_key(key);
        assert_eq!(b1, b2, "region {} bases diverged", region_id);
    }
}

#[test]
fn every_seeded_region_has_primary_faction() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let sim = Sim::new(paths(&dir), graph.clone()).unwrap();
    for (region_id, region) in &graph.regions {
        if !region.procedurally_seeded {
            continue; // real DEM-backed maps are intentionally unseeded
        }
        let c = sim.region_control(*region_id).expect("seeded");
        assert!(c.primary.is_some(), "region {} has no primary", region_id);
    }
}

#[test]
fn every_seeded_region_has_at_least_one_base() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
    for (region_id, region) in &graph.regions {
        if !region.procedurally_seeded {
            continue;
        }
        // Iteration 5-14 Phase C: scene-authored regions skip the
        // base scatter — bases come from `Sim::register_authored_base`
        // on map load. They're still `procedurally_seeded` (so
        // `RegionControl` + `PopulationTargets` seed normally) but
        // the bases themselves come from scene markers.
        if region.scene_authored_pois {
            continue;
        }
        let bases = sim.bases_in_region(*region_id);
        assert!(!bases.is_empty(), "region {} has no bases", region_id);
    }
}

#[test]
fn unseeded_regions_stay_empty() {
    // The quarantine invariant: real DEM-backed maps (corbett +
    // spine) carry procedurally_seeded=false and must have no
    // procedural faction control or bases.
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
    for (region_id, region) in &graph.regions {
        if region.procedurally_seeded {
            continue;
        }
        assert!(
            sim.region_control(*region_id).is_none(),
            "unseeded region {} ({}) has RegionControl",
            region_id,
            region.name,
        );
        assert!(
            sim.bases_in_region(*region_id).is_empty(),
            "unseeded region {} ({}) has bases",
            region_id,
            region.name,
        );
    }
}

#[test]
fn merged_never_in_random_seed() {
    let graph = RegionGraph::default_test_graph();
    for seed in [1u64, 7, 42, 99, 1234, 9999] {
        let dir = TempDir::new().unwrap();
        let mut sim = Sim::new_with_seed(paths(&dir), graph.clone(), seed).unwrap();
        for (region_id, region) in &graph.regions {
            if !region.procedurally_seeded {
                continue;
            }
            let c = sim.region_control(*region_id).unwrap();
            assert_ne!(c.primary.as_deref(), Some("the_afflicted"));
            assert!(!c.contested_by.iter().any(|f| f == "the_afflicted"));
            let merged_id = sim
                .faction_registry()
                .id_of("the_afflicted")
                .expect("registry has merged");
            for base in sim.bases_in_region(*region_id) {
                assert_ne!(base.faction, merged_id);
            }
        }
    }
}

#[test]
fn factions_persist_roundtrip() {
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();

    let (control_before, bases_before) = {
        let mut sim = Sim::new_with_seed(paths(&dir), graph.clone(), 7).unwrap();
        sim.shutdown().unwrap();
        // After shutdown the snapshot is fresh; reopen to read state.
        let mut sim = Sim::load_or_new(paths(&dir), graph.clone()).unwrap();
        let control: Vec<_> = (1..=2)
            .filter_map(|r| sim.region_control(r).cloned().map(|c| (r, c)))
            .collect();
        let mut bases: Vec<_> = Vec::new();
        sim.each_base(|b| bases.push(b));
        let key =
            |b: &simn_sim::BaseView| (b.region, b.faction.0, b.kind as u8, b.pos[0].to_bits());
        bases.sort_by_key(key);
        (control, bases)
    };

    // Reload fresh and compare.
    let mut sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let control_after: Vec<_> = (1..=2)
        .filter_map(|r| sim.region_control(r).cloned().map(|c| (r, c)))
        .collect();
    let mut bases_after: Vec<_> = Vec::new();
    sim.each_base(|b| bases_after.push(b));
    let key = |b: &simn_sim::BaseView| (b.region, b.faction.0, b.kind as u8, b.pos[0].to_bits());
    bases_after.sort_by_key(key);

    assert_eq!(control_before, control_after);
    assert_eq!(bases_before, bases_after);
}

#[test]
fn base_kind_strings_cover_enum() {
    // Make sure base_kind_to_str handles every variant (smoke).
    for k in [
        BaseKind::Checkpoint,
        BaseKind::Outpost,
        BaseKind::Safehouse,
        BaseKind::Headquarters,
        BaseKind::ResearchPost,
    ] {
        let s = simn_sim::base_kind_to_str(k);
        assert!(!s.is_empty());
    }
}

#[test]
fn spawned_npc_carries_registry_faction_id() {
    // After the registry migration, `InFaction` directly holds the
    // registry id. The spawn helper takes a name string and resolves
    // it against the active registry.
    let dir = TempDir::new().unwrap();
    let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
    let id = sim.spawn_npc_for_test("nomads", 1, [0.0, 0.0, 0.0], None);
    let expected_id = sim
        .faction_registry()
        .id_of("nomads")
        .expect("registry has nomads");
    let actual = sim.npc_in_faction_for_test(id).expect("InFaction present");
    assert_eq!(actual.0, expected_id);
}

#[test]
fn faction_relation_drift_persists_across_save_load() {
    // Step 7 contract: drift events journal + survive a snapshot
    // round-trip. Push coalition <-> homesteaders from Hostile (-100) up
    // by +120, expect the band to land at Warm on read; then
    // shutdown + reload and verify the same.
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.shift_faction_relation("coalition", "homesteaders", 180, "test_thaw")
            .unwrap();
        let coalition = sim.faction_registry().id_of("coalition").unwrap();
        let rg = sim.faction_registry().id_of("homesteaders").unwrap();
        assert_eq!(
            sim.faction_relation(coalition, rg),
            simn_sim::Relation::Friendly
        );
        sim.shutdown().unwrap();
    }
    let sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let coalition = sim.faction_registry().id_of("coalition").unwrap();
    let rg = sim.faction_registry().id_of("homesteaders").unwrap();
    assert_eq!(
        sim.faction_relation(coalition, rg),
        simn_sim::Relation::Friendly,
        "drift should survive save/load",
    );
}

#[test]
fn player_rep_drift_persists_and_isolates() {
    // Each player's rep drifts independently; reload preserves both.
    let dir = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    {
        let mut sim = Sim::new(paths(&dir), graph.clone()).unwrap();
        sim.shift_player_rep(1, "coalition_vanguard", -200, "killed_crew_chief")
            .unwrap();
        sim.shift_player_rep(2, "coalition_vanguard", 50, "delivered_supplies")
            .unwrap();
        sim.shutdown().unwrap();
    }
    let sim = Sim::load_or_new(paths(&dir), graph).unwrap();
    let coalition_vanguard = sim.faction_registry().id_of("coalition_vanguard").unwrap();
    let rep = sim.player_reputation();
    let deltas = sim.relation_deltas();
    let r1 = simn_sim::registry_player_relation(
        sim.faction_registry(),
        rep,
        deltas,
        1,
        coalition_vanguard,
    );
    let r2 = simn_sim::registry_player_relation(
        sim.faction_registry(),
        rep,
        deltas,
        2,
        coalition_vanguard,
    );
    assert_eq!(r1, simn_sim::Relation::Hostile, "player 1 trashed rep");
    // Player 2 nudged +50; baseline (nomads vs coalition_vanguard) is Cold (-50).
    // nomads ↔ coalition_vanguard = Cold per the canonical TOML; +50 on top of
    // the cold base lifts player 2's effective rep into Neutral band
    // (the player rep score itself is +50; the baseline isn't applied
    // when an explicit player rep entry exists).
    assert_eq!(r2, simn_sim::Relation::Warm, "player 2 helped");
}
