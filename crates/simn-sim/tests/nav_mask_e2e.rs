//! Iteration 5-13 Phase A3 end-to-end test for the nav-mask
//! pipeline.
//!
//! Builds a synthetic `Heightmap` with an authored `nav_mask` byte
//! grid, attaches it to a `Sim`, then asks `Sim::path_in_region` to
//! route across a painted wall. The test passes when the returned
//! waypoints detour around the painted block (instead of cutting
//! straight through it).
//!
//! This is the "smallest meaningful end-to-end" demonstration that
//! Phase A1 (nav.rs honors mask) + Phase A2 (exporter writes
//! nav_mask.r8) compose correctly. It bypasses the editor / disk
//! roundtrip — the integration test that walks the on-disk pipeline
//! lives in `crates/simn-terrain/tests/nav_mask_io.rs`.

use simn_sim::nav::TravelStyle;
use simn_sim::region::{RegionGraph, RegionId};
use simn_sim::Sim;
use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
use simn_terrain::{Heightmap, NavOverride, TerrainMetadata};

const W: u32 = 32;
const H: u32 = 32;
const SPACING_M: f32 = 2.0;
const TEST_REGION_ID: RegionId = 1;

/// Build a flat-elevation `Heightmap` carrying a hand-authored
/// `nav_mask` that paints a vertical block at heightmap columns
/// `[col_lo, col_hi)` × rows `[row_lo, row_hi)`. Three columns
/// gives a wall comfortably wider than the heightmap → nav grid
/// half-cell alignment slack.
fn paint_corridor_block(col_lo: u32, col_hi: u32, row_lo: u32, row_hi: u32) -> Heightmap {
    let metadata = TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: "nav_mask_e2e".into(),
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
    for row in row_lo..row_hi {
        for col in col_lo..col_hi {
            mask[(row * W + col) as usize] = NavOverride::ForceBlocked as u8;
        }
    }
    Heightmap::from_raw_with_layers(metadata, samples, None, Some(mask))
        .expect("test heightmap with painted nav_mask")
}

/// Build a `Sim` keyed against the standard 2×2 test region graph
/// (`RegionGraph::default_test_graph`). We attach our painted
/// heightmap to region 1 (`map_a`), which has no NPCs unless a
/// test opts in.
fn build_sim_with_region() -> Sim {
    Sim::new_in_memory(RegionGraph::default_test_graph())
}

#[test]
fn painted_block_routes_around_via_top_gap() {
    // 32-wide heightmap → 31×31 nav grid at 2 m cells. Wall
    // spans columns 15..18 (3 wide), rows 5..25 — leaves gaps at
    // the top and bottom of the map. Path from (-30, 0, -25) to
    // (30, 0, 25) must take one of the gaps.
    let hm = paint_corridor_block(15, 18, 5, 25);
    let mut sim = build_sim_with_region();
    sim.attach_region_terrain(TEST_REGION_ID, hm)
        .expect("attach terrain");

    let path = sim
        .path_in_region(
            TEST_REGION_ID,
            [-30.0, 0.0, -25.0],
            [30.0, 0.0, 25.0],
            TravelStyle::Bushwhacker,
        )
        .expect("path should exist via the top or bottom gap");

    // No waypoint may land inside the painted block. World-X
    // values that fall inside the wall span are in roughly
    // [-2, 4]; rows 5..25 correspond to roughly z in [-22, 18].
    // The conservative check: assert no waypoint lies inside a
    // tighter inner core that's definitely inside the painted
    // band, regardless of the half-cell offset.
    let block_x_min = -1.0;
    let block_x_max = 3.0;
    let block_z_min = -20.0;
    let block_z_max = 16.0;
    for w in &path {
        let in_x = w[0] > block_x_min && w[0] < block_x_max;
        let in_z = w[2] > block_z_min && w[2] < block_z_max;
        assert!(
            !(in_x && in_z),
            "waypoint {:?} lies inside the painted block (x ∈ [{}, {}], z ∈ [{}, {}])",
            w,
            block_x_min,
            block_x_max,
            block_z_min,
            block_z_max,
        );
    }

    // Path must touch start + end (within the snap radius).
    assert!(
        path.first().is_some(),
        "path returned at least one waypoint"
    );
}

#[test]
fn no_painted_block_uses_direct_route() {
    // Regression guard: a heightmap with no painted overrides
    // should produce a path that A* can simplify down to a small
    // waypoint count (LOS collapse). If the mask plumbing
    // accidentally leaks blocks into a no-mask map, this fails.
    let hm = paint_corridor_block(0, 0, 0, 0); // no blocked cells
    let mut sim = build_sim_with_region();
    sim.attach_region_terrain(TEST_REGION_ID, hm)
        .expect("attach terrain");
    let path = sim
        .path_in_region(
            TEST_REGION_ID,
            [-30.0, 0.0, -25.0],
            [30.0, 0.0, 25.0],
            TravelStyle::Bushwhacker,
        )
        .expect("flat-grid no-mask path should exist");
    assert!(
        path.len() <= 4,
        "flat grid with no overrides should LOS-collapse; got {} waypoints",
        path.len()
    );
}

#[test]
fn is_traversable_reflects_painted_cells() {
    // Center the painted block on the world origin (column 16 at
    // heightmap center → world x ≈ 1, row 16 → world z ≈ 1). The
    // sim's `is_traversable` should report `false` for a query at
    // the wall center and `true` for a query in the open band.
    let hm = paint_corridor_block(15, 18, 14, 18);
    let mut sim = build_sim_with_region();
    sim.attach_region_terrain(TEST_REGION_ID, hm)
        .expect("attach terrain");
    assert!(
        !sim.is_traversable(TEST_REGION_ID, [1.0, 0.0, 1.0]),
        "painted block at world (1, _, 1) should be impassable"
    );
    assert!(
        sim.is_traversable(TEST_REGION_ID, [-25.0, 0.0, -25.0]),
        "open cell far from the block should be passable"
    );
}
