//! Iteration 5-13 Phase D2 tests for the `InteractionAreas`
//! resource and the `Sim::attach_region_interaction_areas` /
//! `reserve_interaction_area` / `release_interaction_area` API.
//!
//! Phase D3 will add objective-side tests that exercise the squad
//! planner picking a `Rest` area, and the
//! `InteractionStarted`/`InteractionEnded` event firing. This file
//! covers the substrate: registry round-trip, capacity, release,
//! faction filter.

use std::collections::HashMap;

use simn_sim::region::{RegionGraph, RegionId};
use simn_sim::resources::InteractionArea;
use simn_sim::Sim;

const TEST_REGION_ID: RegionId = 1;

fn build_sim() -> Sim {
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

fn area(id: &str, kind: &str, capacity: u32) -> InteractionArea {
    InteractionArea {
        id: id.into(),
        kind: kind.into(),
        pos: [0.0, 0.0, 0.0],
        extents: [1.5, 1.5],
        faction: None,
        capacity,
        occupants: 0,
        tags: HashMap::new(),
    }
}

#[test]
fn register_and_query_areas() {
    let mut sim = build_sim();
    let rest = area("camp_rest_1", "rest", 2);
    let work = area("camp_work_1", "work", 1);
    sim.attach_region_interaction_areas(TEST_REGION_ID, vec![rest.clone(), work.clone()])
        .expect("attach");

    let in_region = sim.interaction_areas_in_region(TEST_REGION_ID);
    assert_eq!(in_region.len(), 2, "both areas should be in region's set",);
    let ids: Vec<&str> = in_region.iter().map(|a| a.id.as_str()).collect();
    assert!(ids.contains(&"camp_rest_1"));
    assert!(ids.contains(&"camp_work_1"));

    // Replacing replaces wholesale.
    let replacement = area("camp_socialize_1", "socialize", 4);
    sim.attach_region_interaction_areas(TEST_REGION_ID, vec![replacement.clone()])
        .expect("replace");
    let in_region = sim.interaction_areas_in_region(TEST_REGION_ID);
    assert_eq!(in_region.len(), 1);
    assert_eq!(in_region[0].id, "camp_socialize_1");

    // Prior ids must be removed from `by_id` so reserve fails for them.
    assert!(
        !sim.reserve_interaction_area("camp_rest_1", None),
        "stale id from prior attach must not resolve",
    );
}

#[test]
fn reserve_respects_capacity() {
    let mut sim = build_sim();
    sim.attach_region_interaction_areas(TEST_REGION_ID, vec![area("bench_a", "work", 2)])
        .expect("attach");

    assert!(sim.reserve_interaction_area("bench_a", None));
    assert!(sim.reserve_interaction_area("bench_a", None));
    assert!(
        !sim.reserve_interaction_area("bench_a", None),
        "third reservation must fail when capacity is 2",
    );
    let in_region = sim.interaction_areas_in_region(TEST_REGION_ID);
    assert_eq!(in_region[0].occupants, 2);
}

#[test]
fn release_frees_slot() {
    let mut sim = build_sim();
    sim.attach_region_interaction_areas(TEST_REGION_ID, vec![area("bench_b", "work", 1)])
        .expect("attach");

    assert!(sim.reserve_interaction_area("bench_b", None));
    assert!(!sim.reserve_interaction_area("bench_b", None));

    assert!(sim.release_interaction_area("bench_b"));
    assert!(
        sim.reserve_interaction_area("bench_b", None),
        "release should free the only slot back up",
    );

    // Releasing a never-reserved area saturates rather than underflows.
    assert!(sim.release_interaction_area("bench_b"));
    assert!(sim.release_interaction_area("bench_b"));
    let in_region = sim.interaction_areas_in_region(TEST_REGION_ID);
    assert_eq!(in_region[0].occupants, 0, "occupants saturate at zero");
}

#[test]
fn faction_filter_restricts_reservation() {
    let mut sim = build_sim();
    let registry = sim.faction_registry();
    // `factions.toml` ships with at least `looters` + `coalition` + `directorate`.
    // Pick any two distinct ids; if they aren't present, the test
    // surfaces the registry change loudly rather than passing
    // silently.
    let primary = registry
        .id_of("directorate")
        .expect("factions.toml must define `directorate`");
    let outsider = registry
        .id_of("looters")
        .expect("factions.toml must define `looters`");
    assert_ne!(primary, outsider);

    let mut restricted = area("federal_only_post", "guard_post", 1);
    restricted.faction = Some(primary);
    sim.attach_region_interaction_areas(TEST_REGION_ID, vec![restricted])
        .expect("attach");

    assert!(
        !sim.reserve_interaction_area("federal_only_post", Some(outsider)),
        "outsider faction must be rejected",
    );
    assert!(
        !sim.reserve_interaction_area("federal_only_post", None),
        "no-faction context must be rejected when area is faction-locked",
    );
    assert!(
        sim.reserve_interaction_area("federal_only_post", Some(primary)),
        "matching faction must succeed",
    );
}

#[test]
fn started_set_dedupes_and_drains() {
    use simn_sim::resources::InteractionAreas;
    let mut store = InteractionAreas::default();
    assert!(!store.is_started(42, "camp"));
    store.mark_started(42, "camp");
    store.mark_started(42, "camp"); // dedupe — second mark is a no-op
    store.mark_started(7, "camp");
    assert!(store.is_started(42, "camp"));
    assert!(store.is_started(7, "camp"));
    let drained = store.drain_started_for_area("camp");
    assert_eq!(drained.len(), 2, "all started NPCs come back from drain");
    assert!(drained.contains(&42));
    assert!(drained.contains(&7));
    assert!(
        !store.is_started(42, "camp"),
        "drain clears the set so the next attach starts fresh",
    );
    // Draining a never-marked id is a no-op (returns empty vec).
    assert!(store.drain_started_for_area("nope").is_empty());
}

#[test]
fn attach_clears_started_for_replaced_areas() {
    let mut sim = build_sim();
    sim.attach_region_interaction_areas(TEST_REGION_ID, vec![area("camp", "rest", 2)])
        .expect("attach");
    // Hand-mark the started set; mimics what tick_npc_goals does
    // on first arrival. We use the sim-internal API via the
    // resource's public methods.
    sim.world_for_test()
        .resource_mut::<simn_sim::resources::InteractionAreas>()
        .mark_started(99, "camp");
    assert!(sim
        .world_for_test()
        .resource::<simn_sim::resources::InteractionAreas>()
        .is_started(99, "camp"));
    // Re-attaching (even with the same id) clears the marker —
    // the contract is that Started entries don't survive a
    // region's marker enumeration.
    sim.attach_region_interaction_areas(TEST_REGION_ID, vec![area("camp", "rest", 2)])
        .expect("re-attach");
    assert!(!sim
        .world_for_test()
        .resource::<simn_sim::resources::InteractionAreas>()
        .is_started(99, "camp"));
}

#[test]
fn duplicate_ids_keep_last() {
    let mut sim = build_sim();
    let first = area("dup", "rest", 1);
    let mut second = area("dup", "rest", 4);
    second.tags.insert("variant".into(), "B".into());
    sim.attach_region_interaction_areas(TEST_REGION_ID, vec![first, second])
        .expect("attach");
    let in_region = sim.interaction_areas_in_region(TEST_REGION_ID);
    assert_eq!(in_region.len(), 1, "duplicate id collapses to one entry");
    assert_eq!(
        in_region[0].capacity, 4,
        "later declaration wins on duplicate",
    );
    assert_eq!(in_region[0].tags.get("variant"), Some(&"B".to_string()));
}
