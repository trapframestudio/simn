//! End-to-end weapon equip + reload + magazine eject tests.
//!
//! All tuning (caliber, capacity) flows through `items.toml`. The
//! tests only read ids and assert behavior; caliber values and
//! capacities come from [`Sim::item_def`] so data drift breaks the
//! loader tests in `tests/inventory.rs` rather than here.

use simn_sim::{ItemId, MagazineState, RegionGraph, SavePaths, Sim, SlotId};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

fn fresh_sim(_dir: &TempDir) -> Sim {
    // No-disk, no-NPC variant. Persistence-roundtrip tests below
    // construct their own Sim::new explicitly.
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

fn upsert(sim: &mut Sim, sid: u64) {
    sim.upsert_player(sid, 1, [0.0; 3], 0.0).unwrap();
}

fn id(s: &str) -> ItemId {
    ItemId::from(s)
}

fn slot(s: &str) -> SlotId {
    SlotId::from(s)
}

/// Find the pocket index of the first item whose id matches `needle`.
fn pocket_index_of(sim: &mut Sim, sid: u64, needle: &ItemId) -> Option<usize> {
    sim.inventory_view(sid).iter().position(|s| &s.id == needle)
}

/// Equip the weapon at `item_id` to `slot_id`, granting it first if
/// needed. Panics on any step failure — these are test helpers.
fn grant_and_equip_weapon(sim: &mut Sim, sid: u64, item_id: &str, slot_id: &str) {
    sim.grant_item(sid, &id(item_id), 1).unwrap();
    let idx = pocket_index_of(sim, sid, &id(item_id)).expect("weapon in pockets");
    sim.equip(sid, &slot(slot_id), "pockets", idx).unwrap();
}

#[test]
fn reload_from_empty_weapon_loads_full_mag() {
    // Rifle equipped, one empty mag in pockets. After reload, the
    // weapon holds the mag at 0 rounds (capacity-unaware; ammo loads
    // separately in the rounds slice).
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "rifle_aks74", "primary");
    sim.grant_item(1, &id("mag_aks74_30"), 1).unwrap();
    assert_eq!(sim.inventory_view(1).len(), 1, "mag in pockets pre-reload");

    sim.reload_weapon(1, &slot("primary")).unwrap();

    let eq = sim.equipment_view(1);
    let ws = eq
        .get(&slot("primary"))
        .expect("rifle still equipped")
        .weapon_state
        .as_ref()
        .expect("rifle has weapon_state");
    let loaded = ws.loaded_magazine.as_ref().expect("mag loaded");
    assert_eq!(loaded.id, id("mag_aks74_30"));
    assert_eq!(loaded.loaded_rounds(), 0, "fresh-granted mag starts empty");
    assert!(
        sim.inventory_view(1).is_empty(),
        "pockets empty after mag moved to weapon"
    );
}

#[test]
fn reload_with_partial_mag_in_weapon_preserves_remaining_rounds() {
    // Rifle equipped. Load one mag, seed it to 15 rounds, then load
    // the second (0-round) mag. The 15-round mag must land in pockets
    // intact — reload never silently drops ammo.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "rifle_aks74", "primary");
    sim.grant_item(1, &id("mag_aks74_30"), 2).unwrap();

    sim.reload_weapon(1, &slot("primary")).unwrap();
    sim.set_equipped_mag_rounds_for_test(1, &slot("primary"), 15);
    sim.reload_weapon(1, &slot("primary")).unwrap();

    let eq = sim.equipment_view(1);
    let on_weapon = eq
        .get(&slot("primary"))
        .unwrap()
        .weapon_state
        .as_ref()
        .unwrap()
        .loaded_magazine
        .as_ref()
        .unwrap();
    let pocket_mag = sim
        .inventory_view(1)
        .into_iter()
        .find(|s| s.id == id("mag_aks74_30"))
        .expect("ejected mag landed in pockets");
    assert_eq!(
        on_weapon.loaded_rounds() + pocket_mag.loaded_rounds(),
        15,
        "total rounds preserved across swap"
    );
}

#[test]
fn reload_fails_with_no_matching_caliber() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "rifle_aks74", "primary");
    // Pistol mag in pockets — wrong caliber.
    sim.grant_item(1, &id("mag_makarov_8"), 1).unwrap();

    let err = sim.reload_weapon(1, &slot("primary")).unwrap_err();
    assert!(
        err.to_string().contains("no magazine"),
        "error should mention missing magazine; got {err}"
    );
}

#[test]
fn reload_fails_on_empty_slot() {
    // Equip something to *a* slot so the Equipment component exists,
    // then try reloading a different, empty slot.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "pistol_makarov", "sidearm");
    let err = sim.reload_weapon(1, &slot("primary")).unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "empty slot error expected; got {err}"
    );
}

#[test]
fn eject_magazine_returns_current_mag_to_pockets() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "pistol_makarov", "sidearm");
    sim.grant_item(1, &id("mag_makarov_8"), 1).unwrap();
    sim.reload_weapon(1, &slot("sidearm")).unwrap();
    assert!(sim.inventory_view(1).is_empty(), "mag in weapon");

    sim.eject_magazine(1, &slot("sidearm")).unwrap();

    let eq = sim.equipment_view(1);
    let ws = eq
        .get(&slot("sidearm"))
        .unwrap()
        .weapon_state
        .as_ref()
        .unwrap();
    assert!(ws.loaded_magazine.is_none(), "mag removed from weapon");
    let inv = sim.inventory_view(1);
    assert_eq!(inv.len(), 1);
    assert_eq!(inv[0].id, id("mag_makarov_8"));
}

#[test]
fn eject_on_empty_weapon_is_idempotent_noop() {
    // Weapon equipped, no mag loaded. Eject must not error and must
    // not add phantom mags.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "shotgun_saiga", "primary");
    sim.eject_magazine(1, &slot("primary")).unwrap();
    assert!(sim.inventory_view(1).is_empty());
    let eq = sim.equipment_view(1);
    assert!(eq
        .get(&slot("primary"))
        .unwrap()
        .weapon_state
        .as_ref()
        .and_then(|ws| ws.loaded_magazine.as_ref())
        .is_none());
}

#[test]
fn reload_survives_snapshot_round_trip() {
    // Equip + load, shutdown to flush the snapshot, re-load from
    // disk, verify weapon state persisted.
    let dir = TempDir::new().unwrap();
    {
        let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
        upsert(&mut sim, 42);
        grant_and_equip_weapon(&mut sim, 42, "rifle_aks74", "primary");
        sim.grant_item(42, &id("mag_aks74_30"), 1).unwrap();
        sim.reload_weapon(42, &slot("primary")).unwrap();
        sim.set_equipped_mag_rounds_for_test(42, &slot("primary"), 22);
        sim.shutdown().unwrap();
    }
    let mut sim = Sim::load(paths(&dir)).unwrap();
    let eq = sim.equipment_view(42);
    let ws = eq
        .get(&slot("primary"))
        .expect("rifle still equipped after reload")
        .weapon_state
        .as_ref()
        .expect("weapon_state survived snapshot");
    let loaded = ws.loaded_magazine.as_ref().expect("mag survived snapshot");
    assert_eq!(loaded.id, id("mag_aks74_30"));
    assert_eq!(
        loaded.magazine_state,
        Some(MagazineState {
            loaded_rounds: 22,
            variant: None
        })
    );
}

#[test]
fn fire_decrements_loaded_rounds() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "rifle_aks74", "primary");
    sim.grant_item(1, &id("mag_aks74_30"), 1).unwrap();
    sim.reload_weapon(1, &slot("primary")).unwrap();
    sim.set_equipped_mag_state_for_test(1, &slot("primary"), 5, Some("round_5_45x39"));

    sim.fire_weapon(1, &slot("primary"), 0.0, 0.0).unwrap();

    let eq = sim.equipment_view(1);
    let loaded = eq
        .get(&slot("primary"))
        .unwrap()
        .weapon_state
        .as_ref()
        .unwrap()
        .loaded_magazine
        .as_ref()
        .unwrap();
    assert_eq!(
        loaded.magazine_state,
        Some(MagazineState {
            loaded_rounds: 4,
            variant: Some(simn_sim::ItemId::from("round_5_45x39")),
        })
    );
}

#[test]
fn fire_empty_mag_errors_no_round_consumed() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "pistol_makarov", "sidearm");
    sim.grant_item(1, &id("mag_makarov_8"), 1).unwrap();
    sim.reload_weapon(1, &slot("sidearm")).unwrap(); // empty mag loaded
    sim.set_equipped_mag_state_for_test(1, &slot("sidearm"), 0, Some("round_9x18"));

    let err = sim.fire_weapon(1, &slot("sidearm"), 0.0, 0.0).unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "empty-mag dry-click expected; got {err}"
    );
    // Rounds still 0, still loaded.
    let eq = sim.equipment_view(1);
    let mag = eq
        .get(&slot("sidearm"))
        .unwrap()
        .weapon_state
        .as_ref()
        .unwrap()
        .loaded_magazine
        .as_ref()
        .unwrap();
    assert_eq!(mag.loaded_rounds(), 0);
}

#[test]
fn fire_without_variant_dry_clicks() {
    // A fresh-reloaded mag has `variant = None` until the player
    // explicitly loads rounds (commit 5). Fire must dry-click, not
    // pick an arbitrary variant.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "pistol_makarov", "sidearm");
    sim.grant_item(1, &id("mag_makarov_8"), 1).unwrap();
    sim.reload_weapon(1, &slot("sidearm")).unwrap();
    sim.set_equipped_mag_rounds_for_test(1, &slot("sidearm"), 5);
    // variant is still None; rounds are 5.
    let err = sim.fire_weapon(1, &slot("sidearm"), 0.0, 0.0).unwrap_err();
    assert!(
        err.to_string().contains("variant"),
        "no-variant dry-click expected; got {err}"
    );
}

#[test]
fn fire_survives_snapshot_round_trip() {
    let dir = TempDir::new().unwrap();
    {
        let mut sim = Sim::new(paths(&dir), RegionGraph::default_test_graph()).unwrap();
        upsert(&mut sim, 99);
        grant_and_equip_weapon(&mut sim, 99, "rifle_aks74", "primary");
        sim.grant_item(99, &id("mag_aks74_30"), 1).unwrap();
        sim.reload_weapon(99, &slot("primary")).unwrap();
        sim.set_equipped_mag_state_for_test(99, &slot("primary"), 10, Some("round_5_45x39"));
        sim.fire_weapon(99, &slot("primary"), 0.0, 0.0).unwrap();
        sim.fire_weapon(99, &slot("primary"), 0.0, 0.0).unwrap();
        sim.shutdown().unwrap();
    }
    let mut sim = Sim::load(paths(&dir)).unwrap();
    let eq = sim.equipment_view(99);
    let mag = eq
        .get(&slot("primary"))
        .unwrap()
        .weapon_state
        .as_ref()
        .unwrap()
        .loaded_magazine
        .as_ref()
        .unwrap();
    assert_eq!(
        mag.magazine_state,
        Some(MagazineState {
            loaded_rounds: 8,
            variant: Some(simn_sim::ItemId::from("round_5_45x39")),
        }),
        "two fires consumed two rounds; variant + state persisted"
    );
}

#[test]
fn fire_no_mag_errors() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "pistol_makarov", "sidearm");
    // No reload — sidearm has no mag loaded.

    let err = sim.fire_weapon(1, &slot("sidearm"), 0.0, 0.0).unwrap_err();
    assert!(
        err.to_string().contains("no magazine") || err.to_string().contains("no magazine loaded"),
        "no-mag dry-click expected; got {err}"
    );
}

#[test]
fn load_rounds_fills_empty_mag_with_variant() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "rifle_aks74", "primary");
    sim.grant_item(1, &id("mag_aks74_30"), 1).unwrap();
    sim.reload_weapon(1, &slot("primary")).unwrap();
    // Grant 10 rounds of AP ammo in pockets.
    sim.grant_item(1, &id("round_5_45x39_ap"), 10).unwrap();

    let loaded = sim
        .load_rounds_into_mag(1, &slot("primary"), &id("round_5_45x39_ap"))
        .unwrap();
    assert_eq!(loaded, 10, "all 10 pocket rounds go into the empty mag");

    let eq = sim.equipment_view(1);
    let mag = eq
        .get(&slot("primary"))
        .unwrap()
        .weapon_state
        .as_ref()
        .unwrap()
        .loaded_magazine
        .as_ref()
        .unwrap();
    assert_eq!(mag.loaded_rounds(), 10);
    assert_eq!(
        mag.magazine_state
            .as_ref()
            .and_then(|ms| ms.variant.as_ref()),
        Some(&id("round_5_45x39_ap")),
    );
}

#[test]
fn load_rounds_clamps_to_capacity() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "pistol_makarov", "sidearm");
    sim.grant_item(1, &id("mag_makarov_8"), 1).unwrap();
    sim.reload_weapon(1, &slot("sidearm")).unwrap();
    // Grant 50 FMJ; mag capacity is 8.
    sim.grant_item(1, &id("round_9x18"), 50).unwrap();

    let loaded = sim
        .load_rounds_into_mag(1, &slot("sidearm"), &id("round_9x18"))
        .unwrap();
    assert_eq!(loaded, 8, "mag tops out at capacity");
    let pocket: u32 = sim
        .inventory_view(1)
        .iter()
        .filter(|s| s.id == id("round_9x18"))
        .map(|s| s.count)
        .sum();
    assert_eq!(pocket, 42, "remaining 42 rounds stay in pockets");
}

#[test]
fn load_rounds_rejects_caliber_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "pistol_makarov", "sidearm");
    sim.grant_item(1, &id("mag_makarov_8"), 1).unwrap();
    sim.reload_weapon(1, &slot("sidearm")).unwrap();
    sim.grant_item(1, &id("round_5_45x39"), 30).unwrap();

    let err = sim
        .load_rounds_into_mag(1, &slot("sidearm"), &id("round_5_45x39"))
        .unwrap_err();
    assert!(
        err.to_string().contains("caliber"),
        "caliber mismatch expected; got {err}"
    );
}

#[test]
fn load_rounds_rejects_variant_flip_on_partial_mag() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "rifle_aks74", "primary");
    sim.grant_item(1, &id("mag_aks74_30"), 1).unwrap();
    sim.reload_weapon(1, &slot("primary")).unwrap();
    sim.grant_item(1, &id("round_5_45x39_ap"), 5).unwrap();
    // Load 5 AP.
    sim.load_rounds_into_mag(1, &slot("primary"), &id("round_5_45x39_ap"))
        .unwrap();
    // Now try to load HP on top — should reject.
    sim.grant_item(1, &id("round_5_45x39_hp"), 5).unwrap();
    let err = sim
        .load_rounds_into_mag(1, &slot("primary"), &id("round_5_45x39_hp"))
        .unwrap_err();
    assert!(
        err.to_string().contains("already holds"),
        "variant flip rejection expected; got {err}"
    );
}

#[test]
fn fire_spawns_projectile_with_mag_variant_stats() {
    // AP-loaded mag vs HP-loaded mag fire projectiles whose
    // spawn deltas carry the variant's ammo id; downstream
    // pen/damage formula uses that id. This test just verifies
    // the variant travels through the fire path into the delta.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    grant_and_equip_weapon(&mut sim, 1, "rifle_aks74", "primary");
    sim.grant_item(1, &id("mag_aks74_30"), 1).unwrap();
    sim.reload_weapon(1, &slot("primary")).unwrap();
    sim.grant_item(1, &id("round_5_45x39_ap"), 30).unwrap();
    sim.load_rounds_into_mag(1, &slot("primary"), &id("round_5_45x39_ap"))
        .unwrap();
    sim.fire_weapon(1, &slot("primary"), 0.0, 0.0).unwrap();
    let spawned = sim
        .drain_tick_deltas()
        .into_iter()
        .find(|d| matches!(d, simn_sim::WorldDelta::ProjectileSpawned { .. }))
        .expect("ProjectileSpawned emitted");
    match spawned {
        simn_sim::WorldDelta::ProjectileSpawned { round_id, .. } => {
            assert_eq!(round_id, id("round_5_45x39_ap"));
        }
        _ => unreachable!(),
    }
}

#[test]
fn load_rounds_into_pocket_mag_fills_empty_mag() {
    // Grant a mag + ammo to pockets (no weapon equip needed for
    // pocket loading), load rounds into the mag.
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.grant_item(1, &id("mag_aks74_30"), 1).unwrap();
    sim.grant_item(1, &id("round_5_45x39_ap"), 12).unwrap();

    // Find the pocket index of the mag.
    let view = sim.inventory_view(1);
    let pocket_idx = view
        .iter()
        .position(|s| s.id == id("mag_aks74_30"))
        .expect("mag in pockets") as u32;

    let loaded = sim
        .load_rounds_into_pocket_mag(1, pocket_idx, &id("round_5_45x39_ap"))
        .unwrap();
    assert_eq!(loaded, 12, "all 12 pocket rounds load");

    let view = sim.inventory_view(1);
    let mag = view
        .iter()
        .find(|s| s.id == id("mag_aks74_30"))
        .expect("mag still there");
    assert_eq!(mag.loaded_rounds(), 12);
    assert_eq!(
        mag.magazine_state
            .as_ref()
            .and_then(|ms| ms.variant.as_ref()),
        Some(&id("round_5_45x39_ap")),
    );
    // Ammo should be gone from pockets.
    assert!(view.iter().all(|s| s.id != id("round_5_45x39_ap")));
}

#[test]
fn load_rounds_into_pocket_mag_rejects_caliber_mismatch() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.grant_item(1, &id("mag_makarov_8"), 1).unwrap();
    sim.grant_item(1, &id("round_5_45x39"), 10).unwrap();
    let view = sim.inventory_view(1);
    let pocket_idx = view
        .iter()
        .position(|s| s.id == id("mag_makarov_8"))
        .unwrap() as u32;
    let err = sim
        .load_rounds_into_pocket_mag(1, pocket_idx, &id("round_5_45x39"))
        .unwrap_err();
    assert!(
        err.to_string().contains("caliber"),
        "caliber mismatch error expected; got {err}"
    );
}

#[test]
fn load_rounds_into_pocket_mag_rejects_variant_flip() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.grant_item(1, &id("mag_aks74_30"), 1).unwrap();
    sim.grant_item(1, &id("round_5_45x39_ap"), 5).unwrap();
    sim.grant_item(1, &id("round_5_45x39_hp"), 5).unwrap();
    let view = sim.inventory_view(1);
    let pocket_idx = view
        .iter()
        .position(|s| s.id == id("mag_aks74_30"))
        .unwrap() as u32;
    // Load AP first.
    sim.load_rounds_into_pocket_mag(1, pocket_idx, &id("round_5_45x39_ap"))
        .unwrap();
    // Now try HP — should reject.
    let err = sim
        .load_rounds_into_pocket_mag(1, pocket_idx, &id("round_5_45x39_hp"))
        .unwrap_err();
    assert!(
        err.to_string().contains("already holds"),
        "variant flip rejection expected; got {err}"
    );
}

#[test]
fn load_rounds_into_pocket_mag_clamps_to_capacity() {
    let dir = TempDir::new().unwrap();
    let mut sim = fresh_sim(&dir);
    upsert(&mut sim, 1);
    sim.grant_item(1, &id("mag_makarov_8"), 1).unwrap();
    sim.grant_item(1, &id("round_9x18"), 50).unwrap();
    let view = sim.inventory_view(1);
    let pocket_idx = view
        .iter()
        .position(|s| s.id == id("mag_makarov_8"))
        .unwrap() as u32;
    let loaded = sim
        .load_rounds_into_pocket_mag(1, pocket_idx, &id("round_9x18"))
        .unwrap();
    assert_eq!(loaded, 8, "capped at mag capacity (8)");
    let view = sim.inventory_view(1);
    let ammo: u32 = view
        .iter()
        .filter(|s| s.id == id("round_9x18"))
        .map(|s| s.count)
        .sum();
    assert_eq!(ammo, 42);
}

#[test]
fn load_rounds_into_pocket_mag_replicates_through_mirror() {
    // Host loads a pocket mag; mirror sees the MagazineLoaded
    // delta apply identically — mag state + pocket ammo both
    // match on the client.
    let dir = TempDir::new().unwrap();
    let mut host = fresh_sim(&dir);
    upsert(&mut host, 7);
    host.grant_item(7, &id("mag_aks74_30"), 1).unwrap();
    host.grant_item(7, &id("round_5_45x39_ap"), 30).unwrap();

    let snapshot = host.serialize_snapshot_body();
    let host_tick = host.current_tick();
    let mut mirror = Sim::new_mirror(RegionGraph::default_test_graph());
    mirror.apply_external_snapshot(snapshot, host_tick);
    let _ = host.drain_tick_deltas();

    let view = host.inventory_view(7);
    let pocket_idx = view
        .iter()
        .position(|s| s.id == id("mag_aks74_30"))
        .unwrap() as u32;
    host.load_rounds_into_pocket_mag(7, pocket_idx, &id("round_5_45x39_ap"))
        .unwrap();
    for d in host.drain_tick_deltas() {
        mirror.apply_external_delta(&d);
    }

    let mirror_view = mirror.inventory_view(7);
    let mirror_mag = mirror_view
        .iter()
        .find(|s| s.id == id("mag_aks74_30"))
        .expect("mag on mirror");
    assert_eq!(mirror_mag.loaded_rounds(), 30);
    assert_eq!(
        mirror_mag
            .magazine_state
            .as_ref()
            .and_then(|ms| ms.variant.as_ref()),
        Some(&id("round_5_45x39_ap")),
    );
    // Ammo also consumed on mirror.
    assert!(
        mirror_view.iter().all(|s| s.id != id("round_5_45x39_ap")),
        "mirror pockets should have no leftover AP rounds"
    );
}

#[test]
fn fire_replicates_through_mirror_sim() {
    let dir = TempDir::new().unwrap();
    let mut host = fresh_sim(&dir);
    upsert(&mut host, 7);
    grant_and_equip_weapon(&mut host, 7, "pistol_makarov", "sidearm");
    host.grant_item(7, &id("mag_makarov_8"), 1).unwrap();
    host.reload_weapon(7, &slot("sidearm")).unwrap();
    host.set_equipped_mag_state_for_test(7, &slot("sidearm"), 4, Some("round_9x18"));

    let snapshot = host.serialize_snapshot_body();
    let host_tick = host.current_tick();
    let mut mirror = Sim::new_mirror(RegionGraph::default_test_graph());
    mirror.apply_external_snapshot(snapshot, host_tick);
    let _ = host.drain_tick_deltas();

    host.fire_weapon(7, &slot("sidearm"), 0.0, 0.0).unwrap();
    for d in host.drain_tick_deltas() {
        mirror.apply_external_delta(&d);
    }

    let eq = mirror.equipment_view(7);
    let mag = eq
        .get(&slot("sidearm"))
        .unwrap()
        .weapon_state
        .as_ref()
        .unwrap()
        .loaded_magazine
        .as_ref()
        .unwrap();
    assert_eq!(
        mag.magazine_state,
        Some(MagazineState {
            loaded_rounds: 3,
            variant: Some(simn_sim::ItemId::from("round_9x18")),
        }),
        "mirror reflects post-fire count + preserves variant"
    );
}

#[test]
fn reload_replicates_through_mirror_sim() {
    // Authoritative sim reloads → drain the delta → apply on a
    // mirror seeded from the host snapshot. Mirror must end in the
    // same equipped/pocket state.
    let dir = TempDir::new().unwrap();
    let mut host = fresh_sim(&dir);
    upsert(&mut host, 7);
    grant_and_equip_weapon(&mut host, 7, "pistol_makarov", "sidearm");
    host.grant_item(7, &id("mag_makarov_8"), 1).unwrap();

    // Stand up a mirror from a mid-flight snapshot: post-equip,
    // pre-reload.
    let snapshot = host.serialize_snapshot_body();
    let host_tick = host.current_tick();
    let mut mirror = Sim::new_mirror(RegionGraph::default_test_graph());
    mirror.apply_external_snapshot(snapshot, host_tick);
    // Drain any spurious deltas from host equip/grant.
    let _ = host.drain_tick_deltas();

    host.reload_weapon(7, &slot("sidearm")).unwrap();
    for d in host.drain_tick_deltas() {
        mirror.apply_external_delta(&d);
    }

    let eq = mirror.equipment_view(7);
    let ws = eq
        .get(&slot("sidearm"))
        .expect("mirror: sidearm still equipped")
        .weapon_state
        .as_ref()
        .expect("mirror: weapon_state present");
    assert!(
        ws.loaded_magazine.is_some(),
        "mirror received loaded magazine"
    );
    assert!(
        mirror.inventory_view(7).is_empty(),
        "mirror pockets drained to match host"
    );
}
