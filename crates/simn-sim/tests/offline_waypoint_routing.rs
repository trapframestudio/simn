//! Iteration 5-13 Phase C2 end-to-end test: offline NPCs path
//! along the `WaypointGraph` instead of bee-lining through painted
//! / stamped blocks.
//!
//! Builds a synthetic two-base region with a painter-blocked
//! corridor between the bases, attaches a `Heightmap` carrying the
//! mask, spawns an offline NPC near the left base targeting the
//! right base, ticks the offline sim, and asserts the NPC's
//! `position_2d` traces a path *around* the wall (via the open
//! top/bottom of the map) instead of straight through it.

use simn_sim::offline_tier::{
    offline_tick_just_advanced, OfflineNpc, OFFLINE_TIER_TICK_DIVISOR, OFFLINE_WALK_SPEED_M_PER_S,
};
use simn_sim::region::{RegionGraph, RegionId};
use simn_sim::{NavOverride, Sim};
use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
use simn_terrain::{Heightmap, TerrainMetadata};

const W: u32 = 64;
const H: u32 = 64;
const SPACING_M: f32 = 2.0;
const TEST_REGION_ID: RegionId = 1;

/// Build a 64×64 flat heightmap with a vertical block painted from
/// heightmap cols 30..34 and rows 10..54 (leaves a ~10-row gap at
/// each end of the map). World extent = 126 m × 126 m centered on
/// origin; the wall sits roughly at world x ∈ [-3, 5].
fn build_blocked_heightmap() -> Heightmap {
    let metadata = TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: "offline_waypoint_test".into(),
        width: W,
        height: H,
        spacing_m: SPACING_M,
        vert_min_m: 0.0,
        vert_max_m: 100.0,
        origin_utm_zone: "10N".into(),
        origin_utm_easting: 0.0,
        origin_utm_northing: 0.0,
        blake3: String::new(),
        features_blake3: String::new(),
        region_size_m: 2048.0,
        playable_extent_x_m: 0.0,
        playable_extent_z_m: 0.0,
        nav_mask_format_version: 0,
        nav_mask_blake3: String::new(),
    };
    let samples = vec![50.0_f32; (W * H) as usize];
    let mut mask = vec![NavOverride::Default as u8; (W * H) as usize];
    for row in 10..54u32 {
        for col in 30..34u32 {
            mask[(row * W + col) as usize] = NavOverride::ForceBlocked as u8;
        }
    }
    Heightmap::from_raw_with_layers(metadata, samples, None, Some(mask))
        .expect("test heightmap with painted wall")
}

#[test]
fn offline_npc_routes_along_chain_around_painted_block() {
    let mut sim = Sim::new_in_memory(RegionGraph::default_test_graph());
    sim.attach_region_terrain(TEST_REGION_ID, build_blocked_heightmap())
        .expect("attach terrain");

    // Spawn one offline NPC near the left side of the map. Targets
    // are picked from `Base` entities; the default test graph
    // doesn't have any bases unless we add them explicitly, so
    // we'll use `spawn_offline_npc_for_test` to bypass spawning +
    // target-pick and just check the chain resolution + traversal.
    let npc_id = sim.spawn_offline_npc_for_test("coalition", TEST_REGION_ID, [-40.0, 0.0]);

    // Directly set the NPC's target to the far right and let the
    // movement system resolve a chain through the graph. This
    // tests the "have target → walk chain" branch without needing
    // to also exercise `pick_offline_target`.
    sim.set_offline_target_for_test(npc_id, Some([40.0, 0.0]));

    // Tick enough cycles to traverse ~80 m at 6 m/s = ~13 s sim
    // time = ~260 sim ticks (offline movement runs every
    // OFFLINE_PROCESS_INTERVAL_TICKS == 4 ticks). 2000 ticks is
    // overkill but cheap.
    let mut visited: Vec<[f32; 2]> = Vec::new();
    for _ in 0..2000 {
        sim.tick().unwrap();
        if let Some(pos) = sim.offline_npc_position_for_test(npc_id) {
            visited.push(pos);
        }
    }

    // No visited position may lie inside the painted block. The
    // wall covers world XZ in roughly x ∈ [-3, 5] × z ∈ [-43, 45].
    // Use a conservative inner box (x ∈ [-1, 3], z ∈ [-35, 35])
    // that's definitely inside the painted band regardless of
    // sub-cell alignment.
    let block_x_min = -1.0;
    let block_x_max = 3.0;
    let block_z_min = -35.0;
    let block_z_max = 35.0;
    for &pos in &visited {
        let in_x = pos[0] > block_x_min && pos[0] < block_x_max;
        let in_z = pos[1] > block_z_min && pos[1] < block_z_max;
        assert!(
            !(in_x && in_z),
            "offline NPC entered painted block at {:?}",
            pos
        );
    }

    // And — sanity — the NPC made non-trivial progress (not stuck
    // at the start). At least one visited position should be on
    // the *right* side of the wall, since the chain routes through
    // the top or bottom gap.
    let any_right_of_wall = visited.iter().any(|p| p[0] > 10.0);
    assert!(
        any_right_of_wall,
        "offline NPC never made it past the wall over 2000 ticks"
    );

    let _ = (
        offline_tick_just_advanced as fn(_, _) -> _,
        OFFLINE_TIER_TICK_DIVISOR,
        OFFLINE_WALK_SPEED_M_PER_S,
        std::any::TypeId::of::<OfflineNpc>(),
    );
}
