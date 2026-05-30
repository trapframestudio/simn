//! The region graph — a directed graph of named regions that NPCs can
//! traverse and that players anchor their transforms against.
//!
//! Region IDs are small integers (stable within a save); region names
//! are what GDScript and the map system refer to when calling into the
//! sim. The graph is seeded at sim-start (either from the snapshot
//! file or a code-defined default) and is treated as immutable
//! thereafter for this slice.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type RegionId = u32;

/// A single named region.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Region {
    pub id: RegionId,
    pub name: String,
    /// `res://`-qualified path to the Godot scene this region maps to.
    /// GDScript reads this to load the right scene when the local
    /// player's region changes.
    pub map_scene: String,
    /// IDs of directly-reachable neighbor regions. Travel between
    /// non-adjacent regions isn't modeled yet.
    pub neighbors: Vec<RegionId>,
    /// Portal positions per neighbor: `neighbors[i]` → this region's
    /// local position of the crossing that *leads to* that neighbor.
    /// NPCs wanting to migrate to a neighbor walk here; when they're
    /// within `PORTAL_CROSS_RADIUS_M` the `npc_portal_cross` system
    /// relocates them to the neighbor's reciprocal portal.
    #[serde(default, serialize_with = "crate::det_serde::sorted_map")]
    pub transitions: HashMap<RegionId, [f32; 3]>,
    /// Whether `world_seed` should populate this region with bases,
    /// factions, and NPC population targets at sim init. Test maps
    /// flip this on; real DEM-backed maps stay `false` until they're
    /// ready for content (DESIGN.md §3.4+ hand-authored content
    /// replaces procedural seeding wholesale).
    ///
    /// `#[serde(default)]` so pre-flag snapshots load with this
    /// false — which matches the quarantine intent for any region
    /// that existed before this field.
    #[serde(default)]
    pub procedurally_seeded: bool,
    /// Iteration 5-14 Phase C. Set true on regions that ship
    /// scene-authored `PoiMarker3D` markers (test_map_1..4 after
    /// the POI baker runs). When true, `world_seed`'s base/camp
    /// scatter pass skips the region — bases come from the
    /// GDScript-driven `Sim::register_authored_base` calls
    /// (`base_spawner.gd`, Phase E) instead. The
    /// `PopulationTargets` + `RegionControl` setup still runs, so
    /// NPCs still spawn and factions still contest.
    ///
    /// `#[serde(default)]` for snapshot back-compat — pre-flag
    /// snapshots load this as `false`, which preserves the legacy
    /// behavior of "let `world_seed` scatter bases".
    #[serde(default)]
    pub scene_authored_pois: bool,
}

#[derive(Resource, Serialize, Deserialize, Clone, Debug, Default)]
pub struct RegionGraph {
    #[serde(serialize_with = "crate::det_serde::sorted_map")]
    pub regions: HashMap<RegionId, Region>,
}

impl RegionGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, region: Region) {
        self.regions.insert(region.id, region);
    }

    pub fn get(&self, id: RegionId) -> Option<&Region> {
        self.regions.get(&id)
    }

    pub fn id_for_name(&self, name: &str) -> Option<RegionId> {
        self.regions.values().find(|r| r.name == name).map(|r| r.id)
    }

    /// Hard-coded four-region 2×2 grid matching the current test
    /// maps. Each region has two neighbors (via its east/south or
    /// west/north edges). Portal positions sit close to the map
    /// edge that leads to that neighbor, at a consistent offset
    /// that matches the `TransitionCube` nodes placed in each
    /// scene.
    ///
    /// Layout:
    ///
    /// ```text
    ///  map_a (NW) ─── map_b (NE)
    ///    │              │
    ///  map_c (SW) ─── map_d (SE)
    /// ```
    ///
    /// Each portal sits on the shared edge at `±PORTAL_EDGE_OFFSET_M`
    /// from the map origin. (Maps are 5km wide; portals inset from
    /// the absolute edge so NPCs don't spawn on the mountain ring.)
    pub fn default_test_graph() -> Self {
        // Distance from center to each portal — well inside the
        // playable area so reached by walk from a neighboring base.
        const P: f32 = 2000.0;
        let mut g = Self::new();

        // map_a (NW): east → map_b (+x), south → map_c (+z)
        let mut a_trans = HashMap::new();
        a_trans.insert(2u32, [P, 0.0, 0.0]);
        a_trans.insert(3u32, [0.0, 0.0, P]);
        g.insert(Region {
            id: 1,
            name: "map_a".into(),
            map_scene: "res://scenes/test/test_map_1.tscn".into(),
            neighbors: vec![2, 3],
            transitions: a_trans,
            procedurally_seeded: true,
            scene_authored_pois: true,
        });

        // map_b (NE): west → map_a (-x), south → map_d (+z)
        let mut b_trans = HashMap::new();
        b_trans.insert(1u32, [-P, 0.0, 0.0]);
        b_trans.insert(4u32, [0.0, 0.0, P]);
        g.insert(Region {
            id: 2,
            name: "map_b".into(),
            map_scene: "res://scenes/test/test_map_2.tscn".into(),
            neighbors: vec![1, 4],
            transitions: b_trans,
            procedurally_seeded: true,
            scene_authored_pois: true,
        });

        // map_c (SW): north → map_a (-z), east → map_d (+x)
        let mut c_trans = HashMap::new();
        c_trans.insert(1u32, [0.0, 0.0, -P]);
        c_trans.insert(4u32, [P, 0.0, 0.0]);
        g.insert(Region {
            id: 3,
            name: "map_c".into(),
            map_scene: "res://scenes/test/test_map_3.tscn".into(),
            neighbors: vec![1, 4],
            transitions: c_trans,
            procedurally_seeded: true,
            scene_authored_pois: true,
        });

        // map_d (SE): north → map_b (-z), west → map_c (-x)
        let mut d_trans = HashMap::new();
        d_trans.insert(2u32, [0.0, 0.0, -P]);
        d_trans.insert(3u32, [-P, 0.0, 0.0]);
        g.insert(Region {
            id: 4,
            name: "map_d".into(),
            map_scene: "res://scenes/test/test_map_4.tscn".into(),
            neighbors: vec![2, 3],
            transitions: d_trans,
            procedurally_seeded: true,
            scene_authored_pois: true,
        });

        // Corbett (id 5) ↔ Latourell (id 6). Gorge corridor spine,
        // west-to-east. Each map is rectangular and centered on origin
        // in scene coords, so portals sit at ±(extent_x/2 − 300) on
        // the east/west edge, z=0 (mid-height).
        let mut corbett_trans = HashMap::new();
        // Corbett is 5000×3500 m → east edge at x=+2500 portal at +2200.
        corbett_trans.insert(6u32, [2200.0, 0.0, 0.0]);
        // Branches south: Sandy portal south-center, Bull Run
        // south-west so the two don't stack.
        corbett_trans.insert(13u32, [800.0, 0.0, 1450.0]);
        corbett_trans.insert(14u32, [-1800.0, 0.0, 1450.0]);
        g.insert(Region {
            id: 5,
            name: "corbett".into(),
            map_scene: "res://scenes/maps/corbett.tscn".into(),
            neighbors: vec![6, 13, 14],
            transitions: corbett_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut latourell_trans = HashMap::new();
        // Latourell is 5000 m wide → west edge at x=−2500, portal at −2200.
        latourell_trans.insert(5u32, [-2200.0, 0.0, 0.0]);
        // East edge portal at +2200, leading to Multnomah.
        latourell_trans.insert(7u32, [2200.0, 0.0, 0.0]);
        g.insert(Region {
            id: 6,
            name: "latourell".into(),
            map_scene: "res://scenes/maps/latourell.tscn".into(),
            neighbors: vec![5, 7],
            transitions: latourell_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut multnomah_trans = HashMap::new();
        // Multnomah is 7500 m wide → west edge at x=−3750, portal at −3450.
        multnomah_trans.insert(6u32, [-3450.0, 0.0, 0.0]);
        // East edge portal leading to Bonneville.
        multnomah_trans.insert(8u32, [3450.0, 0.0, 0.0]);
        g.insert(Region {
            id: 7,
            name: "multnomah".into(),
            map_scene: "res://scenes/maps/multnomah.tscn".into(),
            neighbors: vec![6, 8],
            transitions: multnomah_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut bonneville_trans = HashMap::new();
        // Bonneville is 5000 m wide → west edge at x=−2500, portal at −2200.
        bonneville_trans.insert(7u32, [-2200.0, 0.0, 0.0]);
        bonneville_trans.insert(9u32, [2200.0, 0.0, 0.0]);
        g.insert(Region {
            id: 8,
            name: "bonneville".into(),
            map_scene: "res://scenes/maps/bonneville.tscn".into(),
            neighbors: vec![7, 9],
            transitions: bonneville_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut cascade_locks_trans = HashMap::new();
        // Cascade Locks is 4500×4000 → west at x=−2250 portal −1950;
        // east to hood_river at +1950; south edge z=+2000 portal +1700
        // leading to Eagle Creek wilderness branch.
        cascade_locks_trans.insert(8u32, [-1950.0, 0.0, 0.0]);
        cascade_locks_trans.insert(10u32, [1950.0, 0.0, 0.0]);
        cascade_locks_trans.insert(15u32, [0.0, 0.0, 1700.0]);
        g.insert(Region {
            id: 9,
            name: "cascade_locks".into(),
            map_scene: "res://scenes/maps/cascade_locks.tscn".into(),
            neighbors: vec![8, 10, 15],
            transitions: cascade_locks_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut hood_river_trans = HashMap::new();
        // Hood River is 5500×4500 → W/E at ±2450, N/S at ±1950 (for
        // 300 m edge inset). South branch → Hood River Valley;
        // north branch → White Salmon (across the Columbia).
        hood_river_trans.insert(9u32, [-2450.0, 0.0, 0.0]);
        hood_river_trans.insert(11u32, [2450.0, 0.0, 0.0]);
        hood_river_trans.insert(16u32, [0.0, 0.0, 1950.0]);
        hood_river_trans.insert(17u32, [0.0, 0.0, -1950.0]);
        g.insert(Region {
            id: 10,
            name: "hood_river".into(),
            map_scene: "res://scenes/maps/hood_river.tscn".into(),
            neighbors: vec![9, 11, 16, 17],
            transitions: hood_river_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut mosier_trans = HashMap::new();
        // Mosier is 6000 m wide → west edge at x=−3000, portal at −2700.
        mosier_trans.insert(10u32, [-2700.0, 0.0, 0.0]);
        mosier_trans.insert(12u32, [2700.0, 0.0, 0.0]);
        g.insert(Region {
            id: 11,
            name: "mosier".into(),
            map_scene: "res://scenes/maps/mosier.tscn".into(),
            neighbors: vec![10, 12],
            transitions: mosier_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut the_dalles_trans = HashMap::new();
        // The Dalles is 7000×5500 m → west edge at x=−3500, portal at −3200.
        // Klickitat Hold sits across the Columbia on the WA side at the
        // north edge (z=−2750), portal at −2450. Celilo picks up at
        // the east edge (dam + switchyard + converter + drowned falls
        // split into its own map — see tools/bakes/celilo.toml).
        // Umatilla sits far to the east on the Columbia Plateau — no
        // direct geography, SE-corner portal abstracts the dry-country
        // overland route past Celilo (DESIGN.md §3.3).
        the_dalles_trans.insert(11u32, [-3200.0, 0.0, 0.0]);
        the_dalles_trans.insert(19u32, [0.0, 0.0, -2450.0]);
        the_dalles_trans.insert(22u32, [3200.0, 0.0, 0.0]);
        the_dalles_trans.insert(20u32, [3200.0, 0.0, 2000.0]);
        g.insert(Region {
            id: 12,
            name: "the_dalles".into(),
            map_scene: "res://scenes/maps/the_dalles.tscn".into(),
            neighbors: vec![11, 19, 20, 22],
            transitions: the_dalles_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        // Branch maps begin at id 13. Each connects to its spine
        // parent via a perpendicular-edge portal (south edge for most,
        // north for white_salmon).

        let mut sandy_trans = HashMap::new();
        // Sandy is 5000×5000 m → north edge at z=−2500, portal at −2200
        // leading back to Corbett.
        sandy_trans.insert(5u32, [0.0, 0.0, -2200.0]);
        g.insert(Region {
            id: 13,
            name: "sandy".into(),
            map_scene: "res://scenes/maps/sandy.tscn".into(),
            neighbors: vec![5],
            transitions: sandy_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut bull_run_trans = HashMap::new();
        // Bull Run is 5500×5000 m. Its NE corner is roughly under
        // Corbett's SW area, so the return portal sits in the NE
        // quadrant — inset 300 m from each edge.
        bull_run_trans.insert(5u32, [2450.0, 0.0, -2200.0]);
        g.insert(Region {
            id: 14,
            name: "bull_run".into(),
            map_scene: "res://scenes/maps/bull_run.tscn".into(),
            neighbors: vec![5],
            transitions: bull_run_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut eagle_creek_trans = HashMap::new();
        // Eagle Creek is 4500×5000 → north edge at z=−2500, portal at
        // −2200 leading back to Cascade Locks.
        eagle_creek_trans.insert(9u32, [0.0, 0.0, -2200.0]);
        g.insert(Region {
            id: 15,
            name: "eagle_creek".into(),
            map_scene: "res://scenes/maps/eagle_creek.tscn".into(),
            neighbors: vec![9],
            transitions: eagle_creek_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut hrv_trans = HashMap::new();
        // Hood River Valley is 10000×17000 m — orchard valley running
        // N-S with Odell at the N, Parkdale mid-valley, Cooper Spur
        // ridge at the E. North-edge portal (z=−8200) teleports to
        // Hood River; south-edge portal (z=+8200) teleports to Mt.
        // Hood. West-edge portal (x=−4700) leads to the Lost Lake
        // transit map (the Mt. Hood east-approach corridor).
        hrv_trans.insert(10u32, [0.0, 0.0, -8200.0]);
        hrv_trans.insert(18u32, [0.0, 0.0, 8200.0]);
        hrv_trans.insert(23u32, [-4700.0, 0.0, 0.0]);
        g.insert(Region {
            id: 16,
            name: "hood_river_valley".into(),
            map_scene: "res://scenes/maps/hood_river_valley.tscn".into(),
            neighbors: vec![10, 18, 23],
            transitions: hrv_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut white_salmon_trans = HashMap::new();
        // White Salmon is 5500×4000 m → south edge at z=+2000, portal
        // at +1700 leading back to Hood River (across the Columbia).
        white_salmon_trans.insert(10u32, [0.0, 0.0, 1700.0]);
        g.insert(Region {
            id: 17,
            name: "white_salmon".into(),
            map_scene: "res://scenes/maps/white_salmon.tscn".into(),
            neighbors: vec![10],
            transitions: white_salmon_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut mt_hood_trans = HashMap::new();
        // Mt. Hood is 8000×8000 m → north edge at z=−4000, portal at
        // −3700 leading back to Hood River Valley. (Geographic gap
        // between HRV's south edge and Mt. Hood's north edge is
        // abstracted — sim teleports on portal entry.)
        mt_hood_trans.insert(16u32, [0.0, 0.0, -3700.0]);
        g.insert(Region {
            id: 18,
            name: "mt_hood".into(),
            map_scene: "res://scenes/maps/mt_hood.tscn".into(),
            neighbors: vec![16],
            transitions: mt_hood_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut klickitat_hold_trans = HashMap::new();
        // Klickitat Hold is 6000×5000 m → south edge at z=+2500, portal
        // at +2200 leading back to The Dalles across the Columbia.
        // Final branch map; sits on the WA side covering the lower
        // Klickitat River canyon from Lyle north into the basalt bluffs.
        klickitat_hold_trans.insert(12u32, [0.0, 0.0, 2200.0]);
        g.insert(Region {
            id: 19,
            name: "klickitat_hold".into(),
            map_scene: "res://scenes/maps/klickitat_hold.tscn".into(),
            neighbors: vec![12],
            transitions: klickitat_hold_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        // Endgame outdoor maps — Umatilla + Hanford Spur. Both live on
        // the Columbia Plateau in UTM 11N (first bakes east of the
        // standard UTM 10N zone). Umatilla is the overland-east
        // branch from The Dalles; Hanford Spur connects north of
        // Umatilla via the rebuilt Citizen tunnel (DESIGN.md §3.5).

        let mut umatilla_trans = HashMap::new();
        // Umatilla is 8000×6000 m. West edge at x=−4000 returns to The
        // Dalles; north edge at z=−3000 surfaces the Hanford Spur
        // tunnel (lateral communication corridor per DESIGN §3.5).
        umatilla_trans.insert(12u32, [-3700.0, 0.0, 0.0]);
        umatilla_trans.insert(21u32, [0.0, 0.0, -2700.0]);
        g.insert(Region {
            id: 20,
            name: "umatilla".into(),
            map_scene: "res://scenes/maps/umatilla.tscn".into(),
            neighbors: vec![12, 21],
            transitions: umatilla_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut hanford_spur_trans = HashMap::new();
        // Hanford Spur is 7000×6000 m. South edge at z=+3000 returns
        // to Umatilla via the tunnel. Future: a second portal to
        // facility_7 once the indoor map lands.
        hanford_spur_trans.insert(20u32, [0.0, 0.0, 2700.0]);
        g.insert(Region {
            id: 21,
            name: "hanford_spur".into(),
            map_scene: "res://scenes/maps/hanford_spur.tscn".into(),
            neighbors: vec![20],
            transitions: hanford_spur_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut celilo_trans = HashMap::new();
        // Celilo is 13500×8500 m. West edge at x=−6750, portal at
        // −6450 leading back to The Dalles. Captures The Dalles Dam,
        // Big Eddy Substation, Celilo Converter Station, and the
        // drowned Celilo Falls at the far east end — the endgame
        // power-infrastructure corridor split off from the_dalles.
        celilo_trans.insert(12u32, [-6450.0, 0.0, 0.0]);
        g.insert(Region {
            id: 22,
            name: "celilo".into(),
            map_scene: "res://scenes/maps/celilo.tscn".into(),
            neighbors: vec![12],
            transitions: celilo_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        let mut lost_lake_trans = HashMap::new();
        // Lost Lake is 5000×5000 m — the smaller transit map between
        // the HRV food-hub and Mt. Hood's east approach. East-edge
        // portal (x=+2200) teleports back to HRV.
        lost_lake_trans.insert(16u32, [2200.0, 0.0, 0.0]);
        g.insert(Region {
            id: 23,
            name: "lost_lake".into(),
            map_scene: "res://scenes/maps/lost_lake.tscn".into(),
            neighbors: vec![16],
            transitions: lost_lake_trans,
            procedurally_seeded: false,
            scene_authored_pois: false,
        });

        g
    }
}
