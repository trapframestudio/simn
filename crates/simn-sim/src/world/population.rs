//! Population management, region lifecycle, terrain attach, and
//! nav/event/blackboard/LOS accessors on `Sim`.

use anyhow::Result;
use bevy_ecs::prelude::*;

use crate::components::{Base, InRegion, Position};
use crate::nav::NavQuery as _;
use crate::region::RegionGraph;
use crate::resources::ActiveRegions;

use super::Sim;

impl Sim {
    /// One-shot bulk-seed: walk every region's `PopulationTargets`
    /// and spawn the full target population in a single pass,
    /// bypassing the per-tick squad budget. Call on fresh worlds
    /// (after `Sim::new`); a no-op on snapshot-loaded worlds because
    /// the snapshot already carries the seeded NPCs.
    ///
    /// Without this, walking into a fresh region triggers a
    /// multi-second spawn flood as `spawn_npcs` paces in NPCs at
    /// 8 squads/tick. With it, every region is at its target from
    /// tick 0 and `spawn_npcs` only handles incremental replenishment.
    ///
    /// Production callers (SimHost) invoke this immediately after
    /// `Sim::new` / `Sim::new_with_seed`. Tests that want pacing
    /// behavior skip it; tests using `set_population_target_for_test`
    /// or `scale_all_population_targets` should call it (or not)
    /// based on whether they want NPCs at tick 0.
    pub fn initial_bulk_seed_npcs(&mut self) {
        use bevy_ecs::system::RunSystemOnce;
        // Idempotency guard: skip if the world already has NPCs.
        // Covers two paths:
        //   - snapshot-loaded sim: deserialized NPCs are already in
        //     ECS, no need to re-seed (would produce duplicates).
        //   - re-entrant call within one session: caller already
        //     ran the bulk seed, second call no-ops.
        // A fresh `Sim::new` always satisfies this branch (zero NPCs).
        let any_npc_exists = self
            .world
            .query::<&crate::components::Npc>()
            .iter(&self.world)
            .next()
            .is_some();
        if any_npc_exists {
            return;
        }
        if let Err(e) = (&mut self.world).run_system_once(crate::systems::npc_spawn::bulk_seed_npcs)
        {
            tracing::warn!("initial_bulk_seed_npcs: run_system_once failed: {e:?}");
        }
        // Phase 1C: enforce the invariant "an NPC is an online entity
        // iff its region is in ActiveRegions" from tick 0. The bulk
        // seed creates online entities in every region; immediately
        // demote them all to offline. SimHost's first
        // `set_active_region(start_region)` re-projects just that
        // region back up. Without this, every map outside the active
        // region carries thousands of frozen online entities the
        // hot-systems gate happens to skip — works today, breaks the
        // moment Phase 1E offline systems try to run.
        let region_ids: Vec<crate::region::RegionId> = self
            .world
            .resource::<crate::region::RegionGraph>()
            .regions
            .keys()
            .copied()
            .collect();
        for r in region_ids {
            crate::offline_tier::project_online_to_offline(&mut self.world, r);
        }
    }

    /// Mark a region as "online" (full per-NPC simulation). Call
    /// when the local player enters a region. Multiple regions can
    /// be active simultaneously (future multiplayer).
    ///
    /// Phase 1C: projecting state across the tier boundary on each
    /// transition. Previously-active regions get their online NPCs
    /// collapsed to `OfflineNpc` (inventory + per-limb body-parts
    /// dropped, body-part state summarized to `HealthClass`). The
    /// newly-active region gets its `OfflineNpc`s re-materialized
    /// into full online entities (body-parts from class, inventory
    /// re-rolled from faction loadout tables, `NpcCharacter` re-
    /// derived from `(npc_id, faction_id)`). The invariant is "an
    /// NPC is an online entity iff its region is in
    /// `ActiveRegions`."
    pub fn set_active_region(&mut self, region: crate::region::RegionId) {
        use std::collections::HashSet;
        let current: HashSet<crate::region::RegionId> =
            self.world.resource::<ActiveRegions>().regions.clone();
        let new_active: HashSet<crate::region::RegionId> = std::iter::once(region).collect();
        if current == new_active {
            return;
        }
        let going_offline: Vec<_> = current.difference(&new_active).copied().collect();
        let going_online: Vec<_> = new_active.difference(&current).copied().collect();
        for r in going_offline {
            crate::offline_tier::project_online_to_offline(&mut self.world, r);
        }
        for r in going_online {
            crate::offline_tier::project_offline_to_online(&mut self.world, r);
        }
        *self.world.resource_mut::<ActiveRegions>() = ActiveRegions {
            regions: new_active,
        };
    }

    /// True if the given region has an attached heightmap.
    pub fn has_terrain(&self, region: crate::region::RegionId) -> bool {
        self.world
            .resource::<crate::resources::TerrainMaps>()
            .has(region)
    }

    /// Find a navigable path between two points in the same region,
    /// weighted by the caller's travel-style preference (drives the
    /// per-cell cost multipliers - road-hugger and bushwhacker
    /// produce visibly different routes between the same endpoints).
    /// Returns waypoints (including start and end) on the navmesh
    /// grid. `None` if the region has no nav data, the path can't be
    /// found, or either endpoint is too far from any traversable cell.
    /// See [`crate::nav`] for the underlying contract.
    pub fn path_in_region(
        &self,
        region: crate::region::RegionId,
        from: [f32; 3],
        to: [f32; 3],
        style: crate::nav::TravelStyle,
    ) -> Option<Vec<[f32; 3]>> {
        self.world
            .resource::<crate::nav::NavQueries>()
            .path(region, from, to, style)
    }

    /// Cheap query: is `pos` on a traversable cell in `region`?
    /// Returns `false` if the region has no nav data attached.
    pub fn is_traversable(&self, region: crate::region::RegionId, pos: [f32; 3]) -> bool {
        self.world
            .resource::<crate::nav::NavQueries>()
            .is_traversable(region, pos)
    }

    /// Grid dimensions for a region's nav query, useful for editor
    /// debug overlays. `None` when the region has no nav data.
    pub fn nav_grid_dims(&self, region: crate::region::RegionId) -> Option<(u32, u32)> {
        let nq = self.world.resource::<crate::nav::NavQueries>();
        nq.get(region).map(|q| q.dims())
    }

    /// Push a world-bus event from external code (e.g. tests, a
    /// scripted-quest runner, the future contestation tick). The
    /// event lands on the [`crate::world_event_bus::WorldEventQueue`]
    /// and is delivered to listeners by the next-tick drain. Returns
    /// the assigned event id.
    pub fn push_world_event(
        &mut self,
        kind: crate::world_event_bus::WorldEventKind,
        position: [f32; 3],
        region: crate::region::RegionId,
        ttl_ticks: u32,
    ) -> u64 {
        let now = self.world.resource::<crate::resources::SimClock>().tick;
        self.world
            .resource_mut::<crate::world_event_bus::WorldEventQueue>()
            .push(kind, position, region, now, ttl_ticks)
    }

    /// Number of events currently queued on the world bus (between
    /// drains). Test / instrumentation helper.
    pub fn world_event_queue_len(&self) -> usize {
        self.world
            .resource::<crate::world_event_bus::WorldEventQueue>()
            .len()
    }

    /// Read the squad blackboard for `group_id`. `None` if the group
    /// has no entries (or was pruned by the per-tick TTL sweep).
    /// Borrowing API; callers should clone individual entries if they
    /// need to outlive the borrow. See [`crate::squad_blackboard`] for
    /// the resource contract.
    pub fn squad_blackboard(
        &self,
        group_id: u64,
    ) -> Option<&crate::squad_blackboard::GroupBlackboard> {
        self.world
            .resource::<crate::squad_blackboard::SquadBlackboards>()
            .get(group_id)
    }

    /// Read a cached line-of-sight exposure value for the (observer,
    /// target) pair, populated this tick by `npc_aggro` if the pair
    /// passed the FOV gate. `None` if the pair wasn't evaluated this
    /// tick (e.g. out of perception range, no FOV overlap, or the
    /// cache was cleared after writes by a stale-eviction pass).
    ///
    /// LOS is asymmetric: `Sim::los_exposure(a, b)` and
    /// `Sim::los_exposure(b, a)` look up different entries.
    pub fn los_exposure(
        &self,
        observer: crate::components::NpcId,
        target: crate::components::NpcId,
    ) -> Option<f32> {
        self.world
            .resource::<crate::los_cache::LosCache>()
            .get(observer, target)
    }

    /// Per-cell traversability snapshot for a region's nav query.
    /// Returns a freshly-allocated `Vec<bool>` of length
    /// `width * height` (NW-origin, row-major). `None` when the
    /// region has no nav data attached. Caller is editor / debug
    /// code; the Godot bridge converts to `PackedByteArray` at the
    /// boundary. Cheap on small grids; on a 2000×2000 grid this
    /// allocates 4 MB, so don't call every frame.
    pub fn nav_traversability(&self, region: crate::region::RegionId) -> Option<Vec<bool>> {
        let nq = self.world.resource::<crate::nav::NavQueries>();
        nq.get(region).map(|q| q.traversability().to_vec())
    }

    /// Attach a heightmap to a region. Existing bases in the region
    /// have their `Position.y` snapped to ground; subsequent NPC
    /// spawns and per-tick NPC movement get clamped automatically by
    /// the `clamp_npc_terrain_y` system.
    ///
    /// World coordinates are centered on the region origin (matching
    /// the Godot scene convention); the heightmap's NW-corner sample
    /// space is offset internally by half-extent.
    pub fn attach_region_terrain(
        &mut self,
        region: crate::region::RegionId,
        heightmap: simn_terrain::Heightmap,
    ) -> Result<()> {
        // Thin back-compat wrapper around the new obstacle-aware
        // variant. Pre-iteration-5-13 callers continue working
        // unchanged; the obstacle-aware path is opt-in for callers
        // (the Godot bridge) that walk `nav_obstacle_markers` and
        // pass a non-empty slice.
        self.attach_region_terrain_with_obstacles(region, heightmap, &[])
    }

    /// Iteration 5-13 Phase B2: obstacle-aware variant of
    /// [`Self::attach_region_terrain`]. Builds the
    /// [`crate::nav::GridNavQuery`] from the heightmap (honoring
    /// painter overrides via `nav_mask.r8`), then stamps the
    /// supplied [`crate::nav::NavObstacle`]s on top with the
    /// painter-wins merge rule (see
    /// [`crate::nav::GridNavQuery::apply_obstacles`]).
    ///
    /// Obstacles are **transient** — they live in the in-memory nav
    /// grid for the lifetime of the region attach. Detaching plus
    /// reattaching the region drops them; passing a new slice
    /// replaces the previous set. Scene-side authoring
    /// (`NavObstacleMarker3D`) is the source of truth, and Godot
    /// re-enumerates on every region attach.
    pub fn attach_region_terrain_with_obstacles(
        &mut self,
        region: crate::region::RegionId,
        heightmap: simn_terrain::Heightmap,
        obstacles: &[crate::nav::NavObstacle],
    ) -> Result<()> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }

        // Build the nav-query grid first (borrows the heightmap once)
        // before moving the heightmap into TerrainMaps. Per
        // `nav::GridNavQuery::from_heightmap` the build is read-only
        // and snapshot-style — the resulting query owns its own data
        // and doesn't keep a reference back.
        self.world
            .resource_mut::<crate::nav::NavQueries>()
            .build_for(region, &heightmap);

        // Iteration 5-13 follow-up. Collect the per-region base
        // footprints — each `BaseKind` carries a conservative
        // half-extent (`BaseKind::nav_footprint_xz_m`) that
        // stamps "structure exists here" onto the nav grid.
        // Bases get Y-snapped further down; here we only need
        // their XZ. CampSite returns `None` and is skipped.
        let mut base_obstacles: Vec<crate::nav::NavObstacle> = Vec::new();
        {
            let mut q = self.world.query::<(&Base, &InRegion, &Position)>();
            for (base, in_region, pos) in q.iter(&self.world) {
                if in_region.0 != region {
                    continue;
                }
                if let Some(extents) = base.kind.nav_footprint_xz_m() {
                    base_obstacles.push(crate::nav::NavObstacle {
                        center: [pos.0[0], pos.0[2]],
                        extents,
                        kind: simn_terrain::NavOverride::ForceBlocked,
                    });
                }
            }
        }

        // Stamp caller-provided + base-derived obstacles into the
        // freshly-built grid before TerrainMaps takes ownership of
        // the heightmap. Caller obstacles applied first so any
        // painter-`ForceWalkable` rule still wins over both
        // (B2 merge contract).
        if !obstacles.is_empty() {
            self.world
                .resource_mut::<crate::nav::NavQueries>()
                .apply_obstacles(region, obstacles);
        }
        if !base_obstacles.is_empty() {
            self.world
                .resource_mut::<crate::nav::NavQueries>()
                .apply_obstacles(region, &base_obstacles);
        }

        self.world
            .resource_mut::<crate::resources::TerrainMaps>()
            .attach(region, heightmap);

        // Snap existing bases in this region to ground. Two-phase to
        // avoid holding both a `TerrainMaps` borrow and a query at once.
        let mut bases: Vec<(Entity, f32, f32)> = Vec::new();
        {
            let mut q = self.world.query::<(Entity, &Base, &InRegion, &Position)>();
            for (e, _base, in_region, pos) in q.iter(&self.world) {
                if in_region.0 == region {
                    bases.push((e, pos.0[0], pos.0[2]));
                }
            }
        }
        let updates: Vec<(Entity, f32)> = {
            let terrains = self.world.resource::<crate::resources::TerrainMaps>();
            bases
                .into_iter()
                .filter_map(|(e, x, z)| terrains.ground_at(region, x, z).map(|y| (e, y)))
                .collect()
        };
        for (e, y) in updates {
            if let Some(mut p) = self.world.get_mut::<Position>(e) {
                p.0[1] = y;
            }
        }

        Ok(())
    }

    pub fn regions(&self) -> &RegionGraph {
        self.world.resource::<RegionGraph>()
    }
}
