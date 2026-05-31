//! Rust-side pathfinding for online-tier NPC movement.
//!
//! Single source of truth for "where can a humanoid stand?" lives in
//! `simn-sim` (this module) — not Godot's `NavigationServer3D`. Reasons
//! locked 2026-05-05 in `docs/book/src/planning/npc-traversal-plan.md`
//! §1: replay determinism (Godot's nav is a black box that drifts
//! across engine versions), dedicated-server posture (no Godot at
//! all), and latency.
//!
//! Phase 1 scope: uniform-grid A* over heightmap-derived traversability
//! (slope + feature-class gating). Static-obstacle integration (OSM
//! buildings, hand-placed obstacles) is a phase-2 follow-up; the
//! [`NavQuery`] trait abstracts over implementations so consumers
//! don't change when the obstacle bake lands.
//!
//! ## Coordinates
//!
//! All public APIs use **region-local world coordinates** — the same
//! frame `Position` lives in. A 5 km map spans `[-extent/2, +extent/2]`
//! on each axis, with `(0, 0)` at the region center. The grid origin
//! lives at the NW corner; conversion happens inside [`GridNavQuery`].
//!
//! ## Determinism
//!
//! A* tie-breaking is stable: the priority queue's secondary key is
//! `(cell.z, cell.x)`, so equal-`f` nodes resolve in NW-to-SE order
//! every run. No RNG inside path queries.

use std::collections::HashMap;
use std::sync::Arc;

use bevy_ecs::prelude::Resource;
use simn_terrain::{FeatureClass, Heightmap, NavOverride};

use crate::region::RegionId;

/// Default grid cell size in meters. Matches the heightmap spacing on
/// production maps (2 m). Tunable per-region, but cells smaller than
/// the heightmap spacing don't gain accuracy and cells much larger
/// lose obstacle resolution.
pub const DEFAULT_CELL_SIZE_M: f32 = 2.0;

/// Slope above which a cell is impassable. ~35° matches "steep but not
/// vertical" terrain a fit human can't traverse without climbing gear.
/// Stored as cosine of the angle from vertical for cheap comparison
/// against `normal.y` (no `acos` per cell).
pub const DEFAULT_MAX_SLOPE_COS: f32 = 0.819_152_f32; // cos(35°) ≈ 0.819

/// Maximum number of A* nodes expanded before giving up. Bounds
/// worst-case query cost. Original 50_000 was a fraction-of-grid cap
/// but in practice every cap-hit cost ~60ms per call in debug builds
/// — at our per-tick path budget of 8 calls that's ~480ms per tick
/// just from failures, breaking framerate. The current 5_000 cap is
/// enough to find paths up to ~70 cells (~140 m) along the optimal
/// direction on the test maps; unreachable targets bail in ~5ms
/// instead of ~60ms. Real-world long-range paths get tombstoned by
/// the caller (`advance_with_path`) so unreachable goals don't retry
/// every tick. Bump back up when production maps need longer paths
/// and the cost is amortized via the rayon parallel-pathfinding pass.
pub const MAX_NODES_EXPANDED: usize = 5_000;

/// How an NPC prefers to traverse the world. Drives per-cell cost
/// multipliers in [`NavQuery::path`] so different factions /
/// objectives produce visibly different routes between the same
/// start + goal.
///
/// Determinism note: per-style cost multipliers are integer constants;
/// no floating-point math leaks into A* tie-breaking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TravelStyle {
    /// Strong road preference. Patrols, military movement,
    /// long-distance vehicle convoys (when those land). Walks roads
    /// even when off-road would be shorter; takes ~2× cost penalty
    /// for forest / shrubland / wetland.
    RoadHugger,
    /// Mild road preference. Default for most NPCs - takes roads
    /// when nearby but doesn't detour for them.
    #[default]
    Mixed,
    /// Cross-country, no road preference. Nomads, hunters, NPCs
    /// avoiding patrols. Takes the geometrically shortest traversable
    /// route.
    Bushwhacker,
}

impl TravelStyle {
    /// Per-cell cost multiplier (in hundredths). 100 = neutral; 50 =
    /// half cost (preferred); 200 = double cost (avoided). Multiplied
    /// against the directional cost (orthogonal=100, diagonal=141)
    /// then divided by 100, so all arithmetic stays integer.
    fn cell_cost_mult(self, class: u8) -> u32 {
        // `class` is the FeatureClass discriminant byte (or 0xFF for
        // a blocked cell - never reached by A* since blocked cells
        // aren't enumerated as successors). Hand-rolled match instead
        // of FeatureClass::from_u8 because we want this branch to
        // inline cleanly inside the A* hot loop.
        const PAVED_ROAD: u8 = 21;
        const UNPAVED_ROAD: u8 = 22;
        const TRAIL: u8 = 23;
        const FOREST: u8 = 2;
        const SHRUBLAND: u8 = 3;
        const WETLAND: u8 = 9;
        match (self, class) {
            (Self::RoadHugger, PAVED_ROAD) => 50,
            (Self::RoadHugger, UNPAVED_ROAD) => 60,
            (Self::RoadHugger, TRAIL) => 70,
            (Self::RoadHugger, FOREST | SHRUBLAND) => 200,
            (Self::RoadHugger, WETLAND) => 250,
            (Self::Mixed, PAVED_ROAD) => 85,
            (Self::Mixed, UNPAVED_ROAD) => 90,
            (Self::Mixed, TRAIL) => 95,
            (Self::Bushwhacker, _) => 100,
            _ => 100,
        }
    }
}

/// String parser for the Godot bridge. Accepts the variant names
/// case-insensitively. Unknown strings fall back to [`TravelStyle::default`].
pub fn travel_style_from_str(s: &str) -> TravelStyle {
    match s.to_ascii_lowercase().as_str() {
        "road" | "roadhugger" | "road_hugger" => TravelStyle::RoadHugger,
        "mixed" => TravelStyle::Mixed,
        "bush" | "bushwhacker" => TravelStyle::Bushwhacker,
        _ => TravelStyle::default(),
    }
}

/// A query interface for path lookups. Both online (uniform-grid A*)
/// and offline (waypoint graph, future) consumers implement this.
pub trait NavQuery: Send + Sync {
    /// Find a path from `from` to `to` in region-local world coords,
    /// weighted by the caller's [`TravelStyle`] preference (which
    /// makes road-followers and bushwhackers produce visibly different
    /// routes). Returns simplified waypoints including start and end.
    /// `None` if unreachable, the query exceeded [`MAX_NODES_EXPANDED`],
    /// or either endpoint snaps to no traversable cell within
    /// [`SNAP_RADIUS_CELLS`] of itself.
    fn path(&self, from: [f32; 3], to: [f32; 3], style: TravelStyle) -> Option<Vec<[f32; 3]>>;

    /// Cheap query: is `pos` on a traversable cell?
    fn is_traversable(&self, pos: [f32; 3]) -> bool;

    /// Grid dimensions for debug visualization.
    fn dims(&self) -> (u32, u32);

    /// Per-cell traversability for debug viz. `true` = passable.
    /// Indexed `z * width + x`. Borrowed; cheap.
    fn traversability(&self) -> &[bool];
}

/// Snap a query endpoint to the nearest traversable cell within this
/// many cells (radius). Lets path queries succeed when the start or
/// end happens to land on a non-traversable cell (e.g. an NPC sitting
/// half-on-half-off a steep cell, or a target marker placed in water).
const SNAP_RADIUS_CELLS: u32 = 3;

/// Iteration 5-13 Phase B2: a single AABB nav-override stamped
/// into a [`GridNavQuery`] at attach time. Designers place these
/// in scenes via `NavObstacleMarker3D` (godot/scripts/world/);
/// the bridge enumerates them per region and hands an array to
/// [`Sim::attach_region_terrain_with_obstacles`]. Sim-internal
/// type; never persisted (obstacles live in the scene, not in
/// `nav_mask.r8`).
#[derive(Clone, Copy, Debug)]
pub struct NavObstacle {
    /// World-space XZ center of the obstacle.
    pub center: [f32; 2],
    /// XZ half-size (extents). The sim consumes only XZ; Y on the
    /// authoring marker is gizmo-only.
    pub extents: [f32; 2],
    /// What overlapping nav cells become. Only `ForceBlocked` and
    /// `ForceWalkable` are meaningful here — a `Default` obstacle
    /// is a no-op and the apply loop drops it.
    pub kind: NavOverride,
}

/// Uniform-grid pathfinder backed by heightmap traversability.
///
/// One instance per region. Built lazily by [`NavQueries::build_for`]
/// from a region's [`Heightmap`]; rebuilt only on explicit re-bake.
///
/// Per-cell ground Y is precomputed at build time (single sample per
/// cell center) and stored alongside traversability, so runtime path
/// queries don't need a back-reference to the heightmap. Cuts the
/// lifetime / sharing concerns at a small (~4 MB / 2000×2000 grid)
/// memory cost.
///
/// `Clone` derived so [`NavQueries::apply_obstacles`] can use
/// `Arc::make_mut` for the zero-copy fast path at attach time.
/// Cloning a 2000×2000 grid is ~4 MB — only paid when the renderer
/// is currently holding a snapshot, which is rare.
#[derive(Clone)]
pub struct GridNavQuery {
    cells: Vec<bool>,
    /// FeatureClass discriminant per cell (the `as u8` of
    /// [`simn_terrain::FeatureClass`]). Drives `TravelStyle`
    /// cost multipliers at query time. `0` (Unknown) for blocked
    /// cells - they're never enumerated as successors anyway.
    cell_class: Vec<u8>,
    /// Iteration 5-13 Phase A1: per-cell designer override byte,
    /// cached from `Heightmap::nav_override_at` at build time
    /// (decoded as `NavOverride as u8`). Phase B2 reads this in
    /// `apply_obstacles` to enforce the painter-wins merge rule
    /// (POI `block` doesn't downgrade a painter `ForceWalkable`).
    /// `0` (Default) for every cell on maps with no painted mask.
    cell_override: Vec<u8>,
    cell_y: Vec<f32>,
    width: u32,
    height: u32,
    cell_size_m: f32,
    /// World-local XZ of cell `(0, 0)` (NW corner). `Position` adds
    /// half-extent on each axis to align with the heightmap's NW
    /// origin, then divides by `cell_size_m` to land on a cell index.
    nw_origin: [f32; 2],
}

impl GridNavQuery {
    /// Build a grid nav query from a region's heightmap. Walks the
    /// heightmap once, classifying each cell as traversable or not
    /// based on slope + feature class, and precomputing per-cell
    /// ground Y. ~250 ms on a 2000×2000 grid.
    pub fn from_heightmap(heightmap: &Heightmap, cell_size_m: f32, max_slope_cos: f32) -> Self {
        let extent = heightmap.extent_m();
        let width = (extent[0] / cell_size_m).floor() as u32;
        let height = (extent[1] / cell_size_m).floor() as u32;
        let total = (width as usize) * (height as usize);
        let mut cells = vec![false; total];
        let mut cell_class = vec![0u8; total];
        let mut cell_override = vec![NavOverride::Default as u8; total];
        let mut cell_y = vec![0.0_f32; total];

        let nw_origin = [-extent[0] * 0.5, -extent[1] * 0.5];

        // Heightmap pixel grid is `W_hm × H_hm`; nav grid is coarser
        // (cell_size_m ≥ heightmap spacing). Map each nav cell's
        // center to one heightmap pixel for the nav_mask lookup.
        let hm_w = heightmap.width() as f32;
        let hm_h = heightmap.height() as f32;
        let spacing_m = heightmap.metadata().spacing_m;

        for cz in 0..height {
            for cx in 0..width {
                // Sample at cell center so impassable features
                // aligned with cell edges don't slip through.
                let world_x = nw_origin[0] + (cx as f32 + 0.5) * cell_size_m;
                let world_z = nw_origin[1] + (cz as f32 + 0.5) * cell_size_m;

                // sample_* in simn-terrain takes NW-origin coords
                // (0 to extent), so add half-extent to convert from
                // world-centered.
                let hm_x = world_x + extent[0] * 0.5;
                let hm_z = world_z + extent[1] * 0.5;

                let idx = (cz as usize) * (width as usize) + cx as usize;
                let class = heightmap.sample_feature(hm_x, hm_z);

                // Iteration 5-13 Phase A1: consult the designer-painted
                // nav override at this nav cell's center pixel. Default
                // → existing slope/feature logic. ForceBlocked → cell
                // is always blocked. ForceWalkable → cell is always
                // walkable (overrides Water / Cliff / steep slope).
                // The override byte is also cached in `cell_override`
                // so Phase B2's `apply_obstacles` can enforce the
                // painter-wins-over-POI merge rule.
                let hm_col = (hm_x / spacing_m).floor().clamp(0.0, hm_w - 1.0) as usize;
                let hm_row = (hm_z / spacing_m).floor().clamp(0.0, hm_h - 1.0) as usize;
                let override_ = heightmap.nav_override_at(hm_col, hm_row);
                cell_override[idx] = override_ as u8;

                cells[idx] = match override_ {
                    NavOverride::ForceBlocked => false,
                    NavOverride::ForceWalkable => true,
                    NavOverride::Default => {
                        cell_passable(class, heightmap, hm_x, hm_z, max_slope_cos)
                    }
                };
                cell_class[idx] = class as u8;
                cell_y[idx] = heightmap.sample(hm_x, hm_z);
            }
        }

        Self {
            cells,
            cell_class,
            cell_override,
            cell_y,
            width,
            height,
            cell_size_m,
            nw_origin,
        }
    }

    fn cell_index(&self, cx: u32, cz: u32) -> usize {
        (cz as usize) * (self.width as usize) + cx as usize
    }

    fn passable(&self, cx: u32, cz: u32) -> bool {
        if cx >= self.width || cz >= self.height {
            return false;
        }
        self.cells[self.cell_index(cx, cz)]
    }

    fn world_to_cell(&self, world_x: f32, world_z: f32) -> Option<(u32, u32)> {
        let local_x = world_x - self.nw_origin[0];
        let local_z = world_z - self.nw_origin[1];
        if local_x < 0.0 || local_z < 0.0 {
            return None;
        }
        let cx = (local_x / self.cell_size_m) as u32;
        let cz = (local_z / self.cell_size_m) as u32;
        if cx >= self.width || cz >= self.height {
            return None;
        }
        Some((cx, cz))
    }

    fn cell_to_world(&self, cx: u32, cz: u32) -> [f32; 3] {
        let world_x = self.nw_origin[0] + (cx as f32 + 0.5) * self.cell_size_m;
        let world_z = self.nw_origin[1] + (cz as f32 + 0.5) * self.cell_size_m;
        let world_y = self.cell_y[self.cell_index(cx, cz)];
        [world_x, world_y, world_z]
    }

    /// Iteration 5-13 Phase A1: designer-painted [`NavOverride`] at
    /// `(cx, cz)`. Cached from `Heightmap::nav_override_at` at build
    /// time. Returns [`NavOverride::Default`] for out-of-bounds
    /// cells. Phase B2's `apply_obstacles` consumes this to enforce
    /// the painter-wins-over-POI merge rule.
    pub fn cell_override(&self, cx: u32, cz: u32) -> NavOverride {
        if cx >= self.width || cz >= self.height {
            return NavOverride::Default;
        }
        match self.cell_override[self.cell_index(cx, cz)] {
            1 => NavOverride::ForceBlocked,
            2 => NavOverride::ForceWalkable,
            _ => NavOverride::Default,
        }
    }

    /// Iteration 5-13 Phase B2: stamp a set of in-memory obstacles
    /// into the grid. Called after `from_heightmap` (so painter
    /// overrides land first), typically by
    /// [`Sim::attach_region_terrain_with_obstacles`].
    ///
    /// **Merge rule** (painter wins):
    /// - POI `ForceBlocked` ⇒ cell becomes blocked **unless** the
    ///   painter declared `ForceWalkable` for it (cached in
    ///   `cell_override`). Designer intent is the more deliberate
    ///   signal.
    /// - POI `ForceWalkable` ⇒ cell becomes walkable regardless;
    ///   rare, for catwalk overlays attached to placed structures.
    /// - POI `Default` (or any other byte that decoded to Default
    ///   on the marker) is a no-op and silently skipped.
    ///
    /// Each obstacle stamps every nav cell whose center sits inside
    /// its `[center ± extents]` AABB. Sub-cell obstacles miss every
    /// cell center and become no-ops — the authoring-side
    /// configuration warning on `NavObstacleMarker3D` flags this
    /// for designers before runtime.
    pub fn apply_obstacles(&mut self, obstacles: &[NavObstacle]) {
        for obs in obstacles {
            // Translate the obstacle's world AABB into nav cell
            // index ranges. nw_origin is the world-space XZ of cell
            // (0, 0)'s NW corner, so `(world - nw) / cell_size` is
            // the cell index. Floor + ceil to get inclusive bounds
            // for any cell whose center sits inside the AABB.
            let min_x = obs.center[0] - obs.extents[0];
            let max_x = obs.center[0] + obs.extents[0];
            let min_z = obs.center[1] - obs.extents[1];
            let max_z = obs.center[1] + obs.extents[1];
            let cx_lo = (((min_x - self.nw_origin[0]) / self.cell_size_m).floor() as i32).max(0);
            let cx_hi = (((max_x - self.nw_origin[0]) / self.cell_size_m).ceil() as i32).max(0);
            let cz_lo = (((min_z - self.nw_origin[1]) / self.cell_size_m).floor() as i32).max(0);
            let cz_hi = (((max_z - self.nw_origin[1]) / self.cell_size_m).ceil() as i32).max(0);
            let cx_hi = (cx_hi as u32).min(self.width.saturating_sub(1));
            let cz_hi = (cz_hi as u32).min(self.height.saturating_sub(1));
            let cx_lo = cx_lo as u32;
            let cz_lo = cz_lo as u32;
            if cx_lo > cx_hi || cz_lo > cz_hi {
                continue;
            }
            for cz in cz_lo..=cz_hi {
                for cx in cx_lo..=cx_hi {
                    let idx = self.cell_index(cx, cz);
                    // Use the cached painter override for the
                    // merge rule rather than walking back to the
                    // heightmap. `cell_override[idx]` was stamped
                    // by `from_heightmap` (Phase A1).
                    let painter = self.cell_override[idx];
                    match obs.kind {
                        NavOverride::ForceBlocked => {
                            // Painter ForceWalkable wins over POI block.
                            if painter != NavOverride::ForceWalkable as u8 {
                                self.cells[idx] = false;
                            }
                        }
                        NavOverride::ForceWalkable => {
                            // POI walkable overlays everything.
                            self.cells[idx] = true;
                        }
                        NavOverride::Default => {
                            // No-op — a Default obstacle has nothing
                            // to say.
                        }
                    }
                }
            }
        }
    }

    /// Snap `(cx, cz)` to the nearest traversable cell within
    /// [`SNAP_RADIUS_CELLS`]. Returns the original cell if it's already
    /// passable. Spiral search; deterministic order (NW-first).
    fn snap_to_traversable(&self, cx: u32, cz: u32) -> Option<(u32, u32)> {
        if self.passable(cx, cz) {
            return Some((cx, cz));
        }
        for r in 1..=SNAP_RADIUS_CELLS {
            let r = r as i32;
            for dz in -r..=r {
                for dx in -r..=r {
                    // Only check ring cells at this radius.
                    if dx.abs() != r && dz.abs() != r {
                        continue;
                    }
                    let nx = cx as i32 + dx;
                    let nz = cz as i32 + dz;
                    if nx < 0 || nz < 0 {
                        continue;
                    }
                    if self.passable(nx as u32, nz as u32) {
                        return Some((nx as u32, nz as u32));
                    }
                }
            }
        }
        None
    }

    /// Walk the cell-path; if a straight line between waypoint i and
    /// waypoint i+2 stays on traversable cells, drop waypoint i+1.
    /// Iterates until no more drops. Produces fewer, more direct
    /// waypoints — typically ~3-5x reduction on long paths.
    fn simplify_los(&self, cells: &[(u32, u32)], style: TravelStyle) -> Vec<(u32, u32)> {
        if cells.len() <= 2 {
            return cells.to_vec();
        }
        let mut result = Vec::with_capacity(cells.len());
        result.push(cells[0]);
        // Index in `cells` of the most recently committed waypoint.
        let mut last_kept = 0;
        for i in 1..cells.len() {
            // If a straight line from the last committed waypoint to
            // cells[i+1] stays on cells of equal cost under `style`,
            // drop cells[i].
            if i + 1 < cells.len() && self.line_collapsible(cells[last_kept], cells[i + 1], style) {
                continue;
            }
            result.push(cells[i]);
            last_kept = i;
        }
        result
    }

    /// Bresenham-style line walk from `a` to `b`. Returns true if the
    /// line stays on traversable cells AND - for non-`Bushwhacker`
    /// styles - all cells on the line have the same cost multiplier
    /// as the start cell under `style`. This preserves waypoints at
    /// cost-class transitions (road-hugger keeps the corner where its
    /// route enters / leaves a road), while still collapsing same-cost
    /// runs (long stretches on the same road simplify to two
    /// waypoints).
    ///
    /// For `Bushwhacker` (uniform 100% cost), this reduces to plain
    /// passability — any clear line collapses, matching the v1
    /// behavior.
    fn line_collapsible(&self, a: (u32, u32), b: (u32, u32), style: TravelStyle) -> bool {
        let target_mult = match style {
            TravelStyle::Bushwhacker => None,
            _ => Some(style.cell_cost_mult(self.cell_class[self.cell_index(a.0, a.1)])),
        };
        let mut x0 = a.0 as i32;
        let mut z0 = a.1 as i32;
        let x1 = b.0 as i32;
        let z1 = b.1 as i32;
        let dx = (x1 - x0).abs();
        let dz = -(z1 - z0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sz = if z0 < z1 { 1 } else { -1 };
        let mut err = dx + dz;
        loop {
            if !self.passable(x0 as u32, z0 as u32) {
                return false;
            }
            if let Some(target) = target_mult {
                let class = self.cell_class[self.cell_index(x0 as u32, z0 as u32)];
                if style.cell_cost_mult(class) != target {
                    return false;
                }
            }
            if x0 == x1 && z0 == z1 {
                return true;
            }
            let e2 = 2 * err;
            if e2 >= dz {
                err += dz;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                z0 += sz;
            }
        }
    }
}

/// Cell-passability decision: water and cliff are out, anything past
/// the slope threshold is out, everything else passes. Phase 1
/// approximation; phase 2 layers static-obstacle bake on top.
fn cell_passable(
    class: FeatureClass,
    heightmap: &Heightmap,
    hm_x: f32,
    hm_z: f32,
    max_slope_cos: f32,
) -> bool {
    if matches!(class, FeatureClass::Water | FeatureClass::Cliff) {
        return false;
    }
    let normal = heightmap.sample_normal(hm_x, hm_z);
    // normal.y is cosine of angle from vertical. Larger = flatter.
    normal[1] >= max_slope_cos
}

impl NavQuery for GridNavQuery {
    fn path(&self, from: [f32; 3], to: [f32; 3], style: TravelStyle) -> Option<Vec<[f32; 3]>> {
        let start_cell = self.world_to_cell(from[0], from[2])?;
        let goal_cell = self.world_to_cell(to[0], to[2])?;
        let start = self.snap_to_traversable(start_cell.0, start_cell.1)?;
        let goal = self.snap_to_traversable(goal_cell.0, goal_cell.1)?;

        if start == goal {
            return Some(vec![from, to]);
        }

        // Capture for the closure (pathfinding crate takes a callable).
        let width = self.width;
        let height = self.height;
        let cells = &self.cells;
        let cell_class = &self.cell_class;
        let goal_x = goal.0 as i32;
        let goal_z = goal.1 as i32;

        // Cost scale: 100 = orthogonal step, 141 = diagonal (≈ √2 ×
        // 100). Integer costs keep A* tie-breaking deterministic
        // without floating-point comparison drift. Per-cell cost
        // multipliers (driven by `style`) multiply against this then
        // divide by 100 to keep ints; minimum mult observed (50%
        // for road-hugger on paved road) keeps the heuristic
        // admissible.
        const ORTHO_COST: u32 = 100;
        const DIAG_COST: u32 = 141;
        // Smallest possible cost-mult any cell can produce. Used by
        // the heuristic to stay admissible across all styles.
        const MIN_CELL_MULT: u32 = 50;

        // Bounded-cost search: cap node expansion to MAX_NODES_EXPANDED.
        // We use astar from the `pathfinding` crate, but break out early
        // by tracking expansion count via the successors closure.
        let mut nodes_expanded = 0usize;
        let result = pathfinding::prelude::astar(
            &start,
            |&(cx, cz)| {
                nodes_expanded += 1;
                if nodes_expanded > MAX_NODES_EXPANDED {
                    return Vec::new(); // No successors -> path fails.
                }
                let mut succ: Vec<((u32, u32), u32)> = Vec::with_capacity(8);
                // 8-connectivity. Yield in stable NW→SE order so
                // equal-cost ties resolve identically across runs.
                for (dx, dz, dir_cost) in [
                    (-1i32, -1i32, DIAG_COST),
                    (0, -1, ORTHO_COST),
                    (1, -1, DIAG_COST),
                    (-1, 0, ORTHO_COST),
                    (1, 0, ORTHO_COST),
                    (-1, 1, DIAG_COST),
                    (0, 1, ORTHO_COST),
                    (1, 1, DIAG_COST),
                ] {
                    let nx = cx as i32 + dx;
                    let nz = cz as i32 + dz;
                    if nx < 0 || nz < 0 || nx >= width as i32 || nz >= height as i32 {
                        continue;
                    }
                    let idx = (nz as usize) * (width as usize) + nx as usize;
                    if !cells[idx] {
                        continue;
                    }
                    let mult = style.cell_cost_mult(cell_class[idx]);
                    let edge_cost = (dir_cost * mult) / 100;
                    succ.push(((nx as u32, nz as u32), edge_cost));
                }
                succ
            },
            |&(cx, cz)| {
                // Octile heuristic, scaled by the smallest-possible
                // cell mult so it stays admissible. (A* requires
                // h(n) <= true cost; underestimating with the floor
                // keeps it valid for any TravelStyle.)
                let dx = (cx as i32 - goal_x).unsigned_abs();
                let dz = (cz as i32 - goal_z).unsigned_abs();
                let (lo, hi) = if dx < dz { (dx, dz) } else { (dz, dx) };
                let unweighted = hi * ORTHO_COST + lo * (DIAG_COST - ORTHO_COST);
                (unweighted * MIN_CELL_MULT) / 100
            },
            |&node| node == goal,
        );

        let (cell_path, _cost) = result?;
        if nodes_expanded > MAX_NODES_EXPANDED {
            return None;
        }

        // Simplify via line-of-sight collapse.
        let simplified = self.simplify_los(&cell_path, style);

        // Lift to 3D world coords. Replace first/last with caller's
        // exact `from`/`to` so the returned path starts/ends where
        // requested, not at cell centers.
        let mut waypoints: Vec<[f32; 3]> = simplified
            .iter()
            .map(|&(cx, cz)| self.cell_to_world(cx, cz))
            .collect();
        if let Some(first) = waypoints.first_mut() {
            *first = from;
        }
        if let Some(last) = waypoints.last_mut() {
            *last = to;
        }
        Some(waypoints)
    }

    fn is_traversable(&self, pos: [f32; 3]) -> bool {
        match self.world_to_cell(pos[0], pos[2]) {
            Some((cx, cz)) => self.passable(cx, cz),
            None => false,
        }
    }

    fn dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn traversability(&self) -> &[bool] {
        &self.cells
    }
}

/// Iteration 5-13 Phase C1: sparse waypoint graph derived from a
/// [`GridNavQuery`].
///
/// One node per `spacing_m` × `spacing_m` region of walkable cells
/// (default 32 m → ~16 nav cells at 2 m). Edges between adjacent
/// nodes (8-connectivity at the sample stride) whose Bresenham-line
/// traversal stays on walkable cells. Cost = straight-line
/// distance in meters.
///
/// The graph backs the offline tier's path planning (Phase C2 of
/// the iteration plan): offline NPCs hop along the graph instead
/// of bee-lining between Base positions. Cheap to build (sub-100ms
/// on a 1000×1000 grid), cheap to query (small node count keeps A*
/// linear in practice), and rebuilt alongside the underlying grid
/// on every `attach_region_terrain` — no snapshot persistence
/// required (content, not state).
#[derive(Clone, Debug, Default)]
pub struct WaypointGraph {
    /// Per-node world-space XZ (meters). Indexed by `u32` for cheap
    /// edge keys.
    pub nodes: Vec<[f32; 2]>,
    /// Adjacency: `node_idx → [(neighbor_idx, cost_m)]`. Cost is
    /// straight-line distance between the two node positions.
    pub edges: HashMap<u32, Vec<(u32, f32)>>,
}

impl WaypointGraph {
    /// Build the graph from a region's [`GridNavQuery`]. `spacing_m`
    /// is the desired distance between adjacent nodes (snapped up to
    /// the nearest multiple of `grid.cell_size_m`).
    ///
    /// Algorithm:
    /// 1. Walk the grid on a `stride` cell-step pattern. Each sampled
    ///    cell that's walkable becomes a node at its world center.
    /// 2. For each node, attempt 8-connectivity edges to its
    ///    neighbors at the same stride. Bresenham-trace each candidate
    ///    line on the underlying nav grid; if every traversed cell is
    ///    walkable, add a bidirectional edge.
    ///
    /// Determinism: iteration walks NW→SE; edge insertion order is
    /// stable. No RNG.
    pub fn build_from_grid(grid: &GridNavQuery, spacing_m: f32) -> Self {
        let cell_size = grid.cell_size_m.max(0.001);
        let stride = ((spacing_m / cell_size).round() as i32).max(1) as u32;
        let w = grid.width;
        let h = grid.height;

        // Pass 1: sample walkable cells at the stride.
        let mut nodes: Vec<[f32; 2]> = Vec::new();
        // node_idx_at[i] = Some(idx_in_nodes) if cell (cx, cz) where
        // cx = (i % stride_w) * stride and cz = (i / stride_w) * stride
        // is a node. Used by pass 2's neighbor lookup.
        let stride_w = w.div_ceil(stride);
        let stride_h = h.div_ceil(stride);
        let mut node_idx_at: Vec<Option<u32>> =
            vec![None; (stride_w as usize) * (stride_h as usize)];

        for sz in 0..stride_h {
            for sx in 0..stride_w {
                let cx = sx * stride;
                let cz = sz * stride;
                if cx >= w || cz >= h {
                    continue;
                }
                if !grid.passable(cx, cz) {
                    continue;
                }
                let world_x = grid.nw_origin[0] + (cx as f32 + 0.5) * cell_size;
                let world_z = grid.nw_origin[1] + (cz as f32 + 0.5) * cell_size;
                let idx = nodes.len() as u32;
                nodes.push([world_x, world_z]);
                node_idx_at[(sz as usize) * (stride_w as usize) + (sx as usize)] = Some(idx);
            }
        }

        // Pass 2: 8-connectivity edges with Bresenham traversal check.
        let mut edges: HashMap<u32, Vec<(u32, f32)>> = HashMap::with_capacity(nodes.len());
        for sz in 0..stride_h {
            for sx in 0..stride_w {
                let Some(from_idx) =
                    node_idx_at[(sz as usize) * (stride_w as usize) + (sx as usize)]
                else {
                    continue;
                };
                let from_cell = (sx * stride, sz * stride);
                for (dx, dz) in [
                    (-1i32, -1i32),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                ] {
                    let nsx = sx as i32 + dx;
                    let nsz = sz as i32 + dz;
                    if nsx < 0 || nsz < 0 || nsx >= stride_w as i32 || nsz >= stride_h as i32 {
                        continue;
                    }
                    let Some(to_idx) =
                        node_idx_at[(nsz as usize) * (stride_w as usize) + (nsx as usize)]
                    else {
                        continue;
                    };
                    let to_cell = ((nsx as u32) * stride, (nsz as u32) * stride);
                    if !walkable_line(grid, from_cell, to_cell) {
                        continue;
                    }
                    let from = nodes[from_idx as usize];
                    let to = nodes[to_idx as usize];
                    let cost = ((to[0] - from[0]).powi(2) + (to[1] - from[1]).powi(2)).sqrt();
                    edges.entry(from_idx).or_default().push((to_idx, cost));
                }
            }
        }

        Self { nodes, edges }
    }

    /// Index of the node nearest `pos_2d` (linear scan; node counts
    /// are small per region). `None` when the graph has no nodes.
    pub fn nearest_node(&self, pos_2d: [f32; 2]) -> Option<u32> {
        let mut best: Option<(u32, f32)> = None;
        for (i, n) in self.nodes.iter().enumerate() {
            let d2 = (n[0] - pos_2d[0]).powi(2) + (n[1] - pos_2d[1]).powi(2);
            match best {
                Some((_, b)) if b <= d2 => {}
                _ => best = Some((i as u32, d2)),
            }
        }
        best.map(|(i, _)| i)
    }

    /// Path through the graph from `start` node to `goal` node.
    /// Returns the chain of node indices (start..=goal) or `None`
    /// if unreachable. Uses straight-line distance as the heuristic.
    pub fn path(&self, start: u32, goal: u32) -> Option<Vec<u32>> {
        if start == goal {
            return Some(vec![start]);
        }
        let goal_pos = *self.nodes.get(goal as usize)?;
        let result = pathfinding::prelude::astar(
            &start,
            |&n| {
                self.edges
                    .get(&n)
                    .map(|adj| {
                        adj.iter()
                            .map(|&(neighbor, cost)| {
                                // pathfinding expects integer costs;
                                // scale by 1000 for sub-mm precision.
                                (neighbor, (cost * 1000.0) as u32)
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            },
            |&n| {
                let p = self.nodes[n as usize];
                let d = ((p[0] - goal_pos[0]).powi(2) + (p[1] - goal_pos[1]).powi(2)).sqrt();
                (d * 1000.0) as u32
            },
            |&n| n == goal,
        );
        result.map(|(nodes, _cost)| nodes)
    }

    /// Cheap reachability check: BFS from `start` to `goal`. Used by
    /// `pick_offline_target` (Phase C2) to filter candidates without
    /// paying the full A* cost.
    pub fn reachable(&self, start: u32, goal: u32) -> bool {
        if start == goal {
            return true;
        }
        let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut frontier: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        frontier.push_back(start);
        seen.insert(start);
        while let Some(cur) = frontier.pop_front() {
            let Some(adj) = self.edges.get(&cur) else {
                continue;
            };
            for &(nb, _) in adj {
                if nb == goal {
                    return true;
                }
                if seen.insert(nb) {
                    frontier.push_back(nb);
                }
            }
        }
        false
    }
}

/// Iteration 5-13 Phase C1: Bresenham-traverse the integer cell
/// segment from `a` to `b` (inclusive). Returns `true` if every
/// touched cell is `passable()`; `false` on the first impassable
/// cell. Used by [`WaypointGraph::build_from_grid`] to decide
/// whether two sample-stride neighbors warrant an edge.
fn walkable_line(grid: &GridNavQuery, a: (u32, u32), b: (u32, u32)) -> bool {
    let mut x0 = a.0 as i32;
    let mut z0 = a.1 as i32;
    let x1 = b.0 as i32;
    let z1 = b.1 as i32;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dz = -(z1 - z0).abs();
    let sz = if z0 < z1 { 1 } else { -1 };
    let mut err = dx + dz;
    loop {
        if x0 < 0 || z0 < 0 {
            return false;
        }
        if !grid.passable(x0 as u32, z0 as u32) {
            return false;
        }
        if x0 == x1 && z0 == z1 {
            return true;
        }
        let e2 = 2 * err;
        if e2 >= dz {
            err += dz;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            z0 += sz;
        }
    }
}

/// Default sample spacing for the offline waypoint graph. 32 m
/// keeps node counts well under 4k on typical region sizes (a
/// 1000×1000 grid → ~63×63 = ~3.9k nodes) while preserving enough
/// granularity that offline NPCs read as "walking through the
/// region" rather than teleporting between waypoints.
pub const DEFAULT_WAYPOINT_SPACING_M: f32 = 32.0;

/// Per-region nav query registry. Built lazily as regions attach
/// terrain; rebuilt only on explicit re-bake (no live invalidation).
#[derive(Resource, Default)]
pub struct NavQueries {
    by_region: HashMap<RegionId, Arc<GridNavQuery>>,
    /// Iteration 5-13 Phase C1: per-region offline waypoint graph
    /// built alongside the grid in [`Self::build_for`]. Sparse
    /// (default 32 m sample spacing); offline NPCs hop along it
    /// in [`crate::offline_tier::offline_movement`].
    waypoints: HashMap<RegionId, Arc<WaypointGraph>>,
}

impl NavQueries {
    /// Construct a [`GridNavQuery`] from a region's heightmap and
    /// register it. Replaces any prior entry for the same region.
    /// Caller is `Sim::attach_region_terrain`; the heightmap is
    /// consulted only during build (per-cell traversability + Y
    /// snapshot) and not retained, so the caller is free to move
    /// or drop the heightmap afterward.
    pub fn build_for(&mut self, region: RegionId, heightmap: &Heightmap) {
        let query =
            GridNavQuery::from_heightmap(heightmap, DEFAULT_CELL_SIZE_M, DEFAULT_MAX_SLOPE_COS);
        // Iteration 5-13 Phase C1: build the offline waypoint graph
        // alongside the grid so `attach_region_terrain` lands both
        // in one pass. The graph references the grid only at build
        // time and stores its own positions + edges; no lifetime
        // back-reference.
        let waypoints = WaypointGraph::build_from_grid(&query, DEFAULT_WAYPOINT_SPACING_M);
        self.by_region.insert(region, Arc::new(query));
        self.waypoints.insert(region, Arc::new(waypoints));
    }

    /// Iteration 5-13 Phase B2: stamp obstacles into the region's
    /// grid. Called after [`Self::build_for`] by
    /// [`crate::Sim::attach_region_terrain_with_obstacles`].
    /// No-op if the region has no grid registered or the slice
    /// is empty.
    pub fn apply_obstacles(&mut self, region: RegionId, obstacles: &[NavObstacle]) {
        if obstacles.is_empty() {
            return;
        }
        let Some(arc) = self.by_region.get_mut(&region) else {
            return;
        };
        // Grids are stored as `Arc<GridNavQuery>` so the renderer
        // can hold cheap snapshots; `make_mut` clones lazily on
        // multi-ref. At attach time there are typically no
        // outstanding clones, so this is the zero-copy fast path.
        Arc::make_mut(arc).apply_obstacles(obstacles);
    }

    /// Drop the nav query for a region (e.g. on region unload).
    pub fn detach(&mut self, region: RegionId) {
        self.by_region.remove(&region);
        self.waypoints.remove(&region);
    }

    pub fn has(&self, region: RegionId) -> bool {
        self.by_region.contains_key(&region)
    }

    pub fn get(&self, region: RegionId) -> Option<&Arc<GridNavQuery>> {
        self.by_region.get(&region)
    }

    /// Iteration 5-13 Phase C1: region's offline waypoint graph,
    /// if one has been built. `None` when the region has no nav
    /// data attached. Offline-tier consumers (`offline_movement`
    /// in Phase C2) borrow this for cheap reachability + path
    /// queries.
    pub fn waypoints(&self, region: RegionId) -> Option<&Arc<WaypointGraph>> {
        self.waypoints.get(&region)
    }

    /// Path query convenience that resolves the region first.
    pub fn path(
        &self,
        region: RegionId,
        from: [f32; 3],
        to: [f32; 3],
        style: TravelStyle,
    ) -> Option<Vec<[f32; 3]>> {
        self.by_region
            .get(&region)
            .and_then(|q| q.path(from, to, style))
    }

    /// Traversability convenience.
    pub fn is_traversable(&self, region: RegionId, pos: [f32; 3]) -> bool {
        self.by_region
            .get(&region)
            .map(|q| q.is_traversable(pos))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
    use simn_terrain::TerrainMetadata;

    /// Build a flat heightmap with `vert_min == vert_max` so every
    /// cell has zero slope (normal points straight up). All cells
    /// passable unless we override features. Cells = `width * height`.
    fn flat_heightmap(width: u32, height: u32, spacing_m: f32) -> Heightmap {
        flat_heightmap_with_layers(width, height, spacing_m, None, None)
    }

    /// Iteration 5-13 Phase A1: flat-heightmap helper with optional
    /// `features.r8` and `nav_mask.r8` byte layers. Layers must be
    /// `width * height` bytes when present. Used by the nav-mask
    /// tests below to paint corridors of `ForceBlocked` /
    /// `ForceWalkable` cells without touching disk.
    fn flat_heightmap_with_layers(
        width: u32,
        height: u32,
        spacing_m: f32,
        features: Option<Vec<u8>>,
        nav_mask: Option<Vec<u8>>,
    ) -> Heightmap {
        let metadata = TerrainMetadata {
            format_version: CURRENT_FORMAT_VERSION,
            map_id: "test".into(),
            width,
            height,
            spacing_m,
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
        let samples = vec![50.0_f32; (width * height) as usize];
        Heightmap::from_raw_with_layers(metadata, samples, features, nav_mask)
            .expect("flat heightmap with layers")
    }

    #[test]
    fn path_on_flat_grid() {
        let hm = flat_heightmap(64, 64, 2.0);
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        // Region origin at (0,0), extent 128×128, so corners at ±64.
        let path = nav
            .path(
                [-50.0, 0.0, -50.0],
                [50.0, 0.0, 50.0],
                TravelStyle::Bushwhacker,
            )
            .unwrap();
        assert!(path.len() >= 2, "path has at least start + end");
        assert_eq!(path.first().unwrap()[0], -50.0);
        assert_eq!(path.first().unwrap()[2], -50.0);
        assert_eq!(path.last().unwrap()[0], 50.0);
        assert_eq!(path.last().unwrap()[2], 50.0);
        // Straight diagonal should LOS-collapse to ~2 waypoints.
        assert!(
            path.len() <= 4,
            "flat-grid diagonal should simplify, got {} waypoints",
            path.len()
        );
    }

    #[test]
    fn unreachable_returns_none() {
        let hm = flat_heightmap(16, 16, 2.0);
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        // Wall off all but one cell by directly mutating.
        let total = (nav.width as usize) * (nav.height as usize);
        let mut cells = vec![false; total];
        cells[0] = true; // only NW cell passable
        let nav = GridNavQuery {
            cells,
            cell_class: nav.cell_class,
            cell_override: nav.cell_override,
            cell_y: nav.cell_y,
            width: nav.width,
            height: nav.height,
            cell_size_m: nav.cell_size_m,
            nw_origin: nav.nw_origin,
        };
        // Start in the one passable cell, target far.
        let path = nav.path(
            [-15.0, 0.0, -15.0],
            [15.0, 0.0, 15.0],
            TravelStyle::Bushwhacker,
        );
        assert!(path.is_none());
    }

    #[test]
    fn path_around_obstacle() {
        // 21-wide heightmap → (W-1)*spacing = 40 m extent → 20×20 nav grid.
        let hm = flat_heightmap(21, 21, 2.0);
        let mut nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let w = nav.width as usize;
        // Wall down the middle (column 10, rows 5..15) → forces detour.
        for cz in 5..15 {
            let idx = cz * w + 10;
            nav.cells[idx] = false;
        }
        // Start west of wall, target east of wall.
        let path = nav
            .path(
                [-15.0, 0.0, 0.0],
                [15.0, 0.0, 0.0],
                TravelStyle::Bushwhacker,
            )
            .unwrap();
        // Path should not pass through column 10, rows 5..15.
        for waypoint in &path {
            let cx = ((waypoint[0] - nav.nw_origin[0]) / nav.cell_size_m) as u32;
            let cz = ((waypoint[2] - nav.nw_origin[1]) / nav.cell_size_m) as u32;
            if cx == 10 && (5..15).contains(&cz) {
                panic!("path crossed wall at ({cx}, {cz}): {waypoint:?}");
            }
        }
        assert!(path.len() >= 3, "detour requires at least one bend");
    }

    #[test]
    fn is_traversable_basic() {
        let hm = flat_heightmap(16, 16, 2.0);
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        assert!(nav.is_traversable([0.0, 0.0, 0.0]));
        // Out of bounds → not traversable.
        assert!(!nav.is_traversable([1000.0, 0.0, 0.0]));
    }

    #[test]
    fn pathfinding_is_deterministic() {
        // Run the same query twice on identical fresh navs; assert
        // identical waypoint sequences. Catches RNG leaks or unstable
        // tie-breaking.
        let hm = flat_heightmap(32, 32, 2.0);
        let nav1 = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let nav2 = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let from = [-25.0, 0.0, -10.0];
        let to = [25.0, 0.0, 20.0];
        let p1 = nav1.path(from, to, TravelStyle::Mixed).unwrap();
        let p2 = nav2.path(from, to, TravelStyle::Mixed).unwrap();
        assert_eq!(p1.len(), p2.len(), "waypoint count differs");
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a, b, "waypoint differs between runs");
        }
    }

    #[test]
    fn navqueries_register_and_query() {
        let hm = flat_heightmap(16, 16, 2.0);
        let mut nq = NavQueries::default();
        nq.build_for(7, &hm);
        assert!(nq.has(7));
        assert!(nq.is_traversable(7, [0.0, 0.0, 0.0]));
        let path = nq
            .path(
                7,
                [-10.0, 0.0, -10.0],
                [10.0, 0.0, 10.0],
                TravelStyle::Bushwhacker,
            )
            .unwrap();
        assert!(path.len() >= 2);
        nq.detach(7);
        assert!(!nq.has(7));
    }

    /// Road-hugger should detour through cheap road cells when bushwhacker
    /// takes a straight off-road line. Uses a fixture grid where a row
    /// of "PavedRoad" cells (class byte = 21) connects start to goal
    /// orthogonally, while the diagonal cuts through "Forest" (class 2).
    #[test]
    fn road_hugger_prefers_road_over_diagonal() {
        let hm = flat_heightmap(21, 21, 2.0);
        let mut nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let w = nav.width as usize;
        // Mark every cell as Forest by default.
        for c in nav.cell_class.iter_mut() {
            *c = 2; // Forest
        }
        // Carve an L-shaped paved-road corridor: start row → one column
        // → goal row. The L is longer than the diagonal, but with road
        // cost-mult 50% and forest 200%, the road wins.
        let start_row = 5usize;
        let goal_row = 15usize;
        let corner_col = 5usize;
        for col in 0..=corner_col {
            nav.cell_class[start_row * w + col] = 21; // PavedRoad
        }
        for row in start_row..=goal_row {
            nav.cell_class[row * w + corner_col] = 21;
        }
        for col in corner_col..20 {
            nav.cell_class[goal_row * w + col] = 21;
        }

        let from = [-18.0, 0.0, -10.0]; // ~ cell (1, 5)
        let to = [18.0, 0.0, 10.0]; // ~ cell (19, 15)
        let bush = nav.path(from, to, TravelStyle::Bushwhacker).unwrap();
        let road = nav.path(from, to, TravelStyle::RoadHugger).unwrap();

        // Bushwhacker should go diagonal-ish; road-hugger should hug
        // the L corridor. Distinct routes -> distinct waypoint sets.
        // Stronger assertion: at least one road-hugger waypoint sits
        // on the corridor, and at least one bushwhacker waypoint does
        // not.
        let on_road_corridor = |w_pos: [f32; 3]| -> bool {
            let cx = ((w_pos[0] - nav.nw_origin[0]) / nav.cell_size_m) as usize;
            let cz = ((w_pos[2] - nav.nw_origin[1]) / nav.cell_size_m) as usize;
            cz < 21 && cx < 21 && nav.cell_class[cz * w + cx] == 21
        };
        let road_uses_corridor = road.iter().any(|w| on_road_corridor(*w));
        assert!(
            road_uses_corridor,
            "road-hugger should route through the road corridor: {road:?}"
        );
        // The two paths shouldn't be identical (they'd be the same
        // sequence if travel style were ignored).
        assert_ne!(bush, road, "bushwhacker and road-hugger should differ");
    }

    // --- Iteration 5-13 Phase A1: nav-mask designer-override tests ---

    /// Encode a vertical wall of `ForceBlocked` cells spanning
    /// heightmap columns `[col_lo, col_hi)`, rows `[row_lo, row_hi)`.
    /// Multi-column walls survive the half-cell heightmap → nav
    /// alignment without leaving a single-cell escape gap.
    fn paint_vertical_block(
        width: u32,
        height: u32,
        col_lo: u32,
        col_hi: u32,
        row_lo: u32,
        row_hi: u32,
    ) -> Vec<u8> {
        let mut mask = vec![NavOverride::Default as u8; (width * height) as usize];
        for row in row_lo..row_hi {
            for col in col_lo..col_hi {
                mask[(row * width + col) as usize] = NavOverride::ForceBlocked as u8;
            }
        }
        mask
    }

    #[test]
    fn nav_mask_force_block_routes_around() {
        // 32-wide heightmap (62 m extent) → 31×31 nav grid at 2 m
        // cells. Paint a 3-column-wide vertical block at heightmap
        // columns 15..18, spanning rows 5..25. The wall splits the
        // grid into a left and right half with gaps at top + bottom.
        // A* from far-left to far-right must detour through one of
        // the gaps.
        let mask = paint_vertical_block(32, 32, 15, 18, 5, 25);
        let hm = flat_heightmap_with_layers(32, 32, 2.0, None, Some(mask));
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);

        // Walk every nav cell in the wall band and confirm A1's
        // build path actually flagged them blocked. This is the
        // load-bearing assertion — if it passes, the path-detour
        // check below is just confirming A* respects `passable()`.
        let mut found_blocked = 0;
        for cz in 0..nav.height {
            for cx in 0..nav.width {
                if nav.cell_override(cx, cz) == NavOverride::ForceBlocked {
                    found_blocked += 1;
                    assert!(
                        !nav.passable(cx, cz),
                        "ForceBlocked cell ({cx}, {cz}) must be impassable"
                    );
                }
            }
        }
        assert!(
            found_blocked >= 30,
            "expected ~60 blocked nav cells in the band (3 cols × 20 rows), got {found_blocked}",
        );

        // A* must produce *some* path (gaps at top + bottom) and
        // none of its waypoints can land on a ForceBlocked cell.
        let path = nav
            .path(
                [-30.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                TravelStyle::Bushwhacker,
            )
            .expect("path should exist via the top or bottom gap");
        for w in &path {
            if let Some((cx, cz)) = nav.world_to_cell(w[0], w[2]) {
                assert_ne!(
                    nav.cell_override(cx, cz),
                    NavOverride::ForceBlocked,
                    "waypoint {:?} (cell {},{}) lies in the painted block",
                    w,
                    cx,
                    cz
                );
            }
        }
    }

    #[test]
    fn nav_mask_force_walkable_overrides_blocked_feature() {
        // 16×16 heightmap, every cell tagged FeatureClass::Cliff
        // (discriminant per `FeatureClass::Cliff as u8`). Without
        // any override, every cell is impassable. Painting
        // `ForceWalkable` over every cell flips them all back to
        // passable — proving the override beats the feature class.
        let cliff = FeatureClass::Cliff as u8;
        let features = vec![cliff; 16 * 16];
        // Sanity check: cliff alone produces an unreachable grid.
        let hm_cliff_only = flat_heightmap_with_layers(16, 16, 2.0, Some(features.clone()), None);
        let nav_blocked = GridNavQuery::from_heightmap(&hm_cliff_only, 2.0, DEFAULT_MAX_SLOPE_COS);
        let path_blocked = nav_blocked.path(
            [-10.0, 0.0, -10.0],
            [10.0, 0.0, 10.0],
            TravelStyle::Bushwhacker,
        );
        assert!(
            path_blocked.is_none(),
            "all-cliff grid with no override should produce no path"
        );

        // Now paint ForceWalkable over every cell.
        let mask = vec![NavOverride::ForceWalkable as u8; 16 * 16];
        let hm = flat_heightmap_with_layers(16, 16, 2.0, Some(features), Some(mask));
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let path = nav.path(
            [-10.0, 0.0, -10.0],
            [10.0, 0.0, 10.0],
            TravelStyle::Bushwhacker,
        );
        assert!(
            path.is_some(),
            "ForceWalkable should override Cliff and produce a path"
        );

        // And every grid cell now reports ForceWalkable on the
        // cached override byte.
        for cz in 0..nav.height {
            for cx in 0..nav.width {
                assert_eq!(nav.cell_override(cx, cz), NavOverride::ForceWalkable);
            }
        }
    }

    // --- Iteration 5-13 Phase B2: NavObstacle merge-rule tests ---

    #[test]
    fn poi_block_overlays_walkable_cells() {
        // Flat heightmap → every cell starts walkable. Stamp a
        // ForceBlocked obstacle covering a few cells; assert they
        // flip to impassable.
        let hm = flat_heightmap(16, 16, 2.0);
        let mut nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        // World extent = (16-1)*2 = 30 m. Nav origin at (-15, -15).
        // Cells are 2 m. Obstacle at world (5, _, 5) with extents
        // (3, _, 3) covers world XZ in [2, 8] × [2, 8]. That's
        // 4 nav cells per axis = 16 cells total.
        let obs = NavObstacle {
            center: [5.0, 5.0],
            extents: [3.0, 3.0],
            kind: NavOverride::ForceBlocked,
        };
        let before_blocked = nav.cells.iter().filter(|&&c| !c).count();
        nav.apply_obstacles(&[obs]);
        let after_blocked = nav.cells.iter().filter(|&&c| !c).count();
        assert!(
            after_blocked - before_blocked >= 9,
            "expected at least 9 cells flipped to blocked, got {}",
            after_blocked - before_blocked
        );
        // The cell at the obstacle center must be blocked.
        let center = nav.world_to_cell(5.0, 5.0).expect("center in grid");
        assert!(!nav.passable(center.0, center.1));
    }

    #[test]
    fn painter_force_walkable_wins_over_poi_block() {
        // Paint a single ForceWalkable cell at heightmap col=4,
        // row=4 (world XZ ≈ (8-15, 8-15) = (-7, -7) — depends on
        // alignment, but inside the obstacle AABB below). Then
        // stamp a ForceBlocked obstacle that fully covers that
        // cell. The painter override must win.
        let mut mask = vec![NavOverride::Default as u8; 16 * 16];
        // Cover a 2×2 block of ForceWalkable cells so at least one
        // nav cell center lands on a painted cell regardless of
        // half-cell alignment.
        for row in 4..=6 {
            for col in 4..=6 {
                mask[row * 16 + col] = NavOverride::ForceWalkable as u8;
            }
        }
        let hm = flat_heightmap_with_layers(16, 16, 2.0, None, Some(mask));
        let mut nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        // Confirm the painted cells start walkable (and reported
        // as ForceWalkable on cell_override).
        let mut force_walkable_count = 0;
        for cz in 0..nav.height {
            for cx in 0..nav.width {
                if nav.cell_override(cx, cz) == NavOverride::ForceWalkable {
                    force_walkable_count += 1;
                    assert!(
                        nav.passable(cx, cz),
                        "painter ForceWalkable cell ({cx}, {cz}) should be passable"
                    );
                }
            }
        }
        assert!(
            force_walkable_count > 0,
            "expected at least one ForceWalkable cell from the paint"
        );

        // Stamp an obstacle that fully overlaps the painted region.
        // World cell 4 (heightmap col 4) sits at world x ≈ 4*2 - 15 = -7;
        // cell 6 at world x = -3. So the painted region spans
        // roughly [-7, -3] in world XZ. Cover a wider AABB to
        // ensure full overlap.
        let obs = NavObstacle {
            center: [-5.0, -5.0],
            extents: [5.0, 5.0],
            kind: NavOverride::ForceBlocked,
        };
        nav.apply_obstacles(&[obs]);

        // Painter ForceWalkable cells must still be walkable.
        for cz in 0..nav.height {
            for cx in 0..nav.width {
                if nav.cell_override(cx, cz) == NavOverride::ForceWalkable {
                    assert!(
                        nav.passable(cx, cz),
                        "painter ForceWalkable cell ({cx}, {cz}) must survive POI block (merge rule)"
                    );
                }
            }
        }
    }

    #[test]
    fn poi_block_propagates_through_aabb_cells() {
        // Single obstacle with extents (4, _, 4) at 2 m cells →
        // covers a 4×4 area in world space, which is at minimum a
        // 4×4 cell block (16 cells) when AABB-aligned, or up to 5×5
        // (25) due to apply_obstacles using ceil on the upper bound.
        let hm = flat_heightmap(16, 16, 2.0);
        let mut nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let obs = NavObstacle {
            center: [0.0, 0.0],
            extents: [4.0, 4.0],
            kind: NavOverride::ForceBlocked,
        };
        let before = nav.cells.iter().filter(|&&c| !c).count();
        nav.apply_obstacles(&[obs]);
        let after = nav.cells.iter().filter(|&&c| !c).count();
        assert!(
            after - before >= 16,
            "8m × 8m obstacle at 2m cells should block ≥ 16 cells; got {}",
            after - before
        );
    }

    #[test]
    fn poi_walkable_carves_through_cliff_feature() {
        // All-cliff grid → fully blocked by default. POI walkable
        // overlay flips the covered cells back to passable.
        let cliff = FeatureClass::Cliff as u8;
        let features = vec![cliff; 16 * 16];
        let hm = flat_heightmap_with_layers(16, 16, 2.0, Some(features), None);
        let mut nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        // No cell should be passable yet.
        assert!(nav.cells.iter().all(|&c| !c));
        // Stamp a walkable obstacle covering the world origin.
        nav.apply_obstacles(&[NavObstacle {
            center: [0.0, 0.0],
            extents: [3.0, 3.0],
            kind: NavOverride::ForceWalkable,
        }]);
        let center = nav.world_to_cell(0.0, 0.0).expect("center in grid");
        assert!(
            nav.passable(center.0, center.1),
            "POI ForceWalkable should override Cliff feature"
        );
    }

    #[test]
    fn nav_mask_absent_is_noop_back_compat() {
        // A heightmap with no nav_mask should produce the same grid
        // it always has — passable everywhere flat, no override on
        // any cell. Regression guard so the A1 change doesn't drift
        // the baseline build behavior for maps that never paint.
        let hm = flat_heightmap(16, 16, 2.0);
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let path = nav.path(
            [-10.0, 0.0, -10.0],
            [10.0, 0.0, 10.0],
            TravelStyle::Bushwhacker,
        );
        assert!(path.is_some(), "flat grid with no mask is passable");
        for cz in 0..nav.height {
            for cx in 0..nav.width {
                assert_eq!(
                    nav.cell_override(cx, cz),
                    NavOverride::Default,
                    "every cell should default-override on a mask-less map"
                );
            }
        }
    }

    // --- Iteration 5-13 Phase C1: WaypointGraph tests -------------

    #[test]
    fn waypoint_graph_connects_adjacent_nodes_on_open_grid() {
        // 32×32 flat heightmap → 31×31 nav grid. Sample every 8 m
        // (4 cells) → ~8×8 = 64 nodes (some sample cells overshoot
        // the grid edge and produce None). Every node should
        // 8-connectivity-edge to its grid neighbors.
        let hm = flat_heightmap(32, 32, 2.0);
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let graph = WaypointGraph::build_from_grid(&nav, 8.0);
        assert!(
            graph.nodes.len() >= 49,
            "expected ≥ 49 nodes, got {}",
            graph.nodes.len()
        );
        assert!(!graph.edges.is_empty(), "edges should be populated");
        // Each node (except corners) should have multiple edges.
        let avg_edges: f32 = graph
            .edges
            .values()
            .map(|adj| adj.len() as f32)
            .sum::<f32>()
            / graph.edges.len() as f32;
        assert!(
            avg_edges >= 4.0,
            "open grid should average ≥ 4 edges per node; got {avg_edges}",
        );
    }

    #[test]
    fn waypoint_graph_skips_pairs_with_blocked_cell_between() {
        // Paint a vertical wall splitting the grid. Two nodes
        // straddling the wall must NOT have a direct edge — the
        // Bresenham trace hits an impassable cell.
        let mut mask = vec![NavOverride::Default as u8; 32 * 32];
        // Wall at heightmap col 15, full height.
        for row in 0..32u32 {
            mask[(row * 32 + 15) as usize] = NavOverride::ForceBlocked as u8;
        }
        let hm = flat_heightmap_with_layers(32, 32, 2.0, None, Some(mask));
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let graph = WaypointGraph::build_from_grid(&nav, 4.0); // tight stride
                                                               // Locate two nodes that straddle the wall horizontally at
                                                               // the same z. The wall is at world x ≈ 0 (col 15 → world
                                                               // x = 15*2 - extent/2 ≈ -1, depending on alignment). Find
                                                               // a node with x < -3 and another with x > 3 at the same z.
        let left = graph
            .nodes
            .iter()
            .enumerate()
            .find(|(_, n)| n[0] < -8.0 && n[1].abs() < 1.0);
        let right = graph
            .nodes
            .iter()
            .enumerate()
            .find(|(_, n)| n[0] > 8.0 && n[1].abs() < 1.0);
        if let (Some((li, _)), Some((ri, _))) = (left, right) {
            // Cross-wall direct edge must not exist.
            let edges_from_left = graph.edges.get(&(li as u32));
            let has_direct = edges_from_left
                .map(|adj| adj.iter().any(|&(n, _)| n == ri as u32))
                .unwrap_or(false);
            assert!(
                !has_direct,
                "cross-wall direct edge ({li} → {ri}) should not exist"
            );
        }
        // Graph should still be reachable through the top/bottom
        // gap (no gap since wall is full-height → graph is
        // disconnected into two components, no path exists).
        // Confirming the disconnected case is the next test
        // (waypoint_graph_handles_islands).
    }

    #[test]
    fn waypoint_graph_handles_islands() {
        // Full-height wall → two disconnected components. `reachable`
        // returns false across them.
        let mut mask = vec![NavOverride::Default as u8; 32 * 32];
        for row in 0..32u32 {
            mask[(row * 32 + 15) as usize] = NavOverride::ForceBlocked as u8;
        }
        let hm = flat_heightmap_with_layers(32, 32, 2.0, None, Some(mask));
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let graph = WaypointGraph::build_from_grid(&nav, 4.0);
        let left = graph.nearest_node([-15.0, 0.0]).expect("left node");
        let right = graph.nearest_node([15.0, 0.0]).expect("right node");
        assert!(
            !graph.reachable(left, right),
            "left + right separated by a full-height wall should be unreachable"
        );
        assert!(graph.path(left, right).is_none());
        // Sanity: a node should be reachable from itself.
        assert!(graph.reachable(left, left));
    }

    #[test]
    fn waypoint_graph_path_returns_node_chain() {
        // Open grid; ask for a path; result is a node chain from
        // start to goal inclusive.
        let hm = flat_heightmap(32, 32, 2.0);
        let nav = GridNavQuery::from_heightmap(&hm, 2.0, DEFAULT_MAX_SLOPE_COS);
        let graph = WaypointGraph::build_from_grid(&nav, 8.0);
        let start = graph.nearest_node([-20.0, -20.0]).expect("start node");
        let goal = graph.nearest_node([20.0, 20.0]).expect("goal node");
        let path = graph
            .path(start, goal)
            .expect("path should exist on open grid");
        assert_eq!(*path.first().unwrap(), start);
        assert_eq!(*path.last().unwrap(), goal);
        assert!(path.len() >= 2, "non-trivial path should have ≥ 2 nodes");
    }

    #[test]
    fn waypoint_graph_built_alongside_grid_via_navqueries() {
        // Smoke: NavQueries::build_for stores both the grid and
        // the waypoint graph; both are retrievable.
        let hm = flat_heightmap(16, 16, 2.0);
        let mut queries = NavQueries::default();
        queries.build_for(42, &hm);
        assert!(queries.get(42).is_some(), "grid stored");
        assert!(
            queries.waypoints(42).is_some(),
            "waypoint graph stored alongside grid"
        );
        assert!(
            !queries.waypoints(42).unwrap().nodes.is_empty(),
            "waypoint graph has nodes on a flat heightmap"
        );
    }
}
