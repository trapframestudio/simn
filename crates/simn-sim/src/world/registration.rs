//! Scene-authored entity registration on `Sim`: interaction areas
//! and faction bases.

use anyhow::Result;

use super::Sim;

impl Sim {
    /// Iteration 5-13 Phase D2: replace `region`'s set of
    /// interaction areas. Clears the prior set + the corresponding
    /// rows in `by_id`. Designed to be called once per region on
    /// map load by the Godot bridge (`game_session.gd` walks
    /// `interaction_area_markers` and ships the dicts).
    ///
    /// Auto-derives `id` from the position when the marker left
    /// `area_id` empty: `auto:<region_name>:<x>_<z>` keyed on
    /// integer-rounded XZ. Modders who want stable cross-region
    /// references should set the marker's `area_id` instead.
    ///
    /// Duplicate ids within the same call: the *last* wins (later
    /// markers shadow earlier ones). A warn log fires.
    pub fn attach_region_interaction_areas(
        &mut self,
        region: crate::region::RegionId,
        areas: Vec<crate::resources::InteractionArea>,
    ) -> Result<()> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }
        let mut store = self
            .world
            .resource_mut::<crate::resources::InteractionAreas>();
        // Drop the prior set for this region from both maps.
        // Also drop any Started entries pointing at ids that are
        // about to disappear — otherwise a stale id from a prior
        // attach could leak into a fresh tick's event stream.
        if let Some(prior) = store.by_region.remove(&region) {
            for area in &prior {
                if let Some((existing_region, _)) = store.by_id.get(&area.id) {
                    if *existing_region == region {
                        store.by_id.remove(&area.id);
                    }
                }
                store.started.remove(&area.id);
            }
        }
        // Rebuild from the new set, deduplicating ids by last-wins.
        let mut deduped: Vec<crate::resources::InteractionArea> = Vec::with_capacity(areas.len());
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Iterate in reverse so the *last* declaration wins on
        // duplicate id (matches GuardPosts.set semantics).
        for area in areas.into_iter().rev() {
            if !seen.insert(area.id.clone()) {
                tracing::warn!(
                    "attach_region_interaction_areas: duplicate id {:?} in region {} — \
                     keeping the last occurrence",
                    area.id,
                    region
                );
                continue;
            }
            deduped.push(area);
        }
        deduped.reverse(); // Restore source order for stable indices.
        for (idx, area) in deduped.iter().enumerate() {
            store.by_id.insert(area.id.clone(), (region, idx));
        }
        store.by_region.insert(region, deduped);
        Ok(())
    }

    /// Iteration 5-13 Phase D2: try to reserve a slot at the
    /// interaction area named `area_id`. Returns `true` on success
    /// (occupancy bumped); `false` when the area doesn't exist, is
    /// at capacity, or the caller's faction doesn't match the
    /// area's restriction.
    ///
    /// `faction = None` means "no faction context" (e.g. a system
    /// reserving on behalf of an entity without an `InFaction`);
    /// the call succeeds only if the area itself has no faction
    /// restriction.
    pub fn reserve_interaction_area(
        &mut self,
        area_id: &str,
        faction: Option<crate::faction::registry::FactionId>,
    ) -> bool {
        // Faction-filter is the external-caller's check; if it
        // fails we don't even attempt to reserve. Internal callers
        // (`squad_planner::build_rest`) pre-filter then call
        // `reserve_internal` directly so the faction check isn't
        // repeated. Single source of truth for the bookkeeping
        // lives on the resource.
        let required = {
            let store = self.world.resource::<crate::resources::InteractionAreas>();
            let Some(&(region, idx)) = store.by_id.get(area_id) else {
                return false;
            };
            let Some(area) = store
                .by_region
                .get(&region)
                .and_then(|areas| areas.get(idx))
            else {
                return false;
            };
            area.faction
        };
        if let Some(required) = required {
            match faction {
                Some(f) if f == required => {}
                _ => return false,
            }
        }
        let mut store = self
            .world
            .resource_mut::<crate::resources::InteractionAreas>();
        store.reserve_internal(area_id)
    }

    /// Iteration 5-13 Phase D2: release a previously-reserved slot
    /// at `area_id`. Saturates at zero (releasing a never-reserved
    /// area is a no-op rather than an underflow). Returns `true`
    /// when the area exists, `false` when it doesn't.
    pub fn release_interaction_area(&mut self, area_id: &str) -> bool {
        let mut store = self
            .world
            .resource_mut::<crate::resources::InteractionAreas>();
        store.release_internal(area_id)
    }

    /// Iteration 5-13 Phase D2: list every interaction area in a
    /// region. Borrows the resource; cheap. Used by tests and by
    /// Phase D3's squad-planner utility scoring.
    pub fn interaction_areas_in_region(
        &self,
        region: crate::region::RegionId,
    ) -> &[crate::resources::InteractionArea] {
        self.world
            .resource::<crate::resources::InteractionAreas>()
            .by_region
            .get(&region)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Iteration 5-14 Phase B: spawn a scene-authored faction base.
    /// Designer-placed `PoiMarker3D` nodes in the test map scenes
    /// flow through here; the procedural-scatter path in
    /// `world_seed::seed_random_world_content` is now gated off for
    /// regions with `Region::scene_authored_pois = true` (Phase C),
    /// so this is the *only* path bases get into the world for those
    /// regions.
    ///
    /// Mirrors the spawn tuple used by the procedural path
    /// (`world_seed.rs::seed_random_world_content` lines 181–187):
    /// `(Base, InFaction, InRegion, Position, Health)`. Y-snaps to
    /// terrain if a heightmap is attached, otherwise honors the
    /// caller-supplied Y.
    ///
    /// Also stamps a small nav-obstacle footprint matching
    /// `BaseKind::nav_footprint_xz_m()` so squads naturally route
    /// around it — same contract the procedural path gets via
    /// `attach_region_terrain_with_obstacles`. `CampSite` has no
    /// footprint and is a no-op for the obstacle stamp.
    ///
    /// Errors on an unknown region; succeeds otherwise.
    pub fn register_authored_base(
        &mut self,
        region: crate::region::RegionId,
        pos: [f32; 3],
        kind: crate::components::BaseKind,
        faction: crate::faction::registry::FactionId,
    ) -> Result<bevy_ecs::entity::Entity> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }
        // Y-snap if terrain is attached. Mirrors the snap pass in
        // `attach_region_terrain_with_obstacles`.
        let snapped_y = self
            .world
            .resource::<crate::resources::TerrainMaps>()
            .ground_at(region, pos[0], pos[2])
            .unwrap_or(pos[1]);
        let final_pos = [pos[0], snapped_y, pos[2]];
        let entity = self
            .world
            .spawn((
                crate::components::Base { kind },
                crate::components::InFaction(faction),
                crate::components::InRegion(region),
                crate::components::Position(final_pos),
                crate::components::Health::new_full(),
            ))
            .id();
        // Stamp the per-kind nav footprint into the region grid so
        // pathfinding routes around the structure. `CampSite` returns
        // `None` from `nav_footprint_xz_m` and is skipped.
        if let Some(extents) = kind.nav_footprint_xz_m() {
            self.world
                .resource_mut::<crate::nav::NavQueries>()
                .apply_obstacles(
                    region,
                    &[crate::nav::NavObstacle {
                        center: [final_pos[0], final_pos[2]],
                        extents,
                        kind: simn_terrain::NavOverride::ForceBlocked,
                    }],
                );
        }
        Ok(entity)
    }

    // ── Activity points ─────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn register_activity_point(
        &mut self,
        region: crate::region::RegionId,
        kind: crate::resources::ActivityKind,
        pos: [f32; 3],
        facing_yaw: f32,
        faction: Option<crate::faction::registry::FactionId>,
        radius_m: f32,
        capacity: u8,
        priority: i8,
        loop_id: Option<String>,
    ) -> Result<u64> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }
        let mut store = self
            .world
            .resource_mut::<crate::resources::ActivityPoints>();
        let id = store.next_id();
        let point = crate::resources::ActivityPoint {
            id,
            kind,
            pos,
            facing_yaw,
            faction,
            radius_m,
            capacity,
            priority,
            loop_id,
            occupants: Vec::new(),
            claimed_by_groups: Vec::new(),
        };
        store.by_region.entry(region).or_default().push(point);
        Ok(id)
    }

    pub fn register_patrol_route(
        &mut self,
        region: crate::region::RegionId,
        route_id: String,
        waypoints: Vec<[f32; 3]>,
        faction: Option<crate::faction::registry::FactionId>,
        is_loop: bool,
        priority: i8,
    ) -> Result<()> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }
        let mut store = self
            .world
            .resource_mut::<crate::resources::ActivityPoints>();
        let route = crate::resources::PatrolRoute {
            id: route_id,
            waypoints,
            faction,
            is_loop,
            priority,
            claimed_by_group: None,
        };
        store
            .routes_by_region
            .entry(region)
            .or_default()
            .push(route);
        Ok(())
    }

    pub fn clear_activity_points_for_region(&mut self, region: crate::region::RegionId) {
        self.world
            .resource_mut::<crate::resources::ActivityPoints>()
            .clear_region(region);
    }

    // ── Authored spawn points ───────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn register_spawn_point(
        &mut self,
        region: crate::region::RegionId,
        pos: [f32; 3],
        faction: crate::faction::registry::FactionId,
        spawn_rate_per_min: f32,
        max_concurrent: u8,
        squad_size: (u8, u8),
        spread_radius_m: f32,
        loadout_tier: u8,
        enabled: bool,
        initial_delay_ticks: u64,
    ) -> Result<u64> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }
        let mut store = self
            .world
            .resource_mut::<crate::resources::AuthoredSpawnPoints>();
        let id = store.next_id();
        let point = crate::resources::AuthoredSpawnPoint {
            id,
            region,
            pos,
            faction,
            spawn_rate_per_min,
            max_concurrent,
            squad_size,
            spread_radius_m,
            loadout_tier,
            enabled,
            active_squads: Vec::new(),
            last_spawn_tick: 0,
            initial_delay_ticks,
        };
        store.by_region.entry(region).or_default().push(point);
        Ok(id)
    }

    pub fn clear_spawn_points_for_region(&mut self, region: crate::region::RegionId) {
        self.world
            .resource_mut::<crate::resources::AuthoredSpawnPoints>()
            .clear_region(region);
    }

    // ── Cover volumes ───────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn register_cover_volume(
        &mut self,
        region: crate::region::RegionId,
        pos: [f32; 3],
        half_extents: [f32; 3],
        rotation: [f32; 4],
        material_id: crate::cover::CoverMaterialId,
        height: crate::cover::CoverHeight,
        thickness_mm: f32,
        destructible: bool,
        health: f32,
    ) -> Result<u64> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }
        let mut store = self.world.resource_mut::<crate::cover::CoverVolumes>();
        let id = store.next_id();
        let vol = crate::cover::CoverVolume {
            id,
            region,
            pos,
            half_extents,
            rotation,
            material_id,
            height,
            thickness_mm,
            destructible,
            health,
            max_health: health,
        };
        store.by_region.entry(region).or_default().push(vol);
        Ok(id)
    }

    pub fn clear_cover_volumes_for_region(&mut self, region: crate::region::RegionId) {
        self.world
            .resource_mut::<crate::cover::CoverVolumes>()
            .clear_region(region);
    }
}
