//! World-container API on `Sim` — ground drops, scene-placed crates,
//! NPC corpses (PR-4b). All container operations route through here so
//! the journal stays consistent and the kit-pool path can find them.
//!
//! Public vs. private (`WorldContainer.is_public`) controls whether
//! a container's contents count toward the **crafting kit-pool** —
//! a parts bin chained to a workbench (`is_public = true`) does;
//! a player's stash (`is_public = false`) doesn't, even if the
//! crafter is standing next to it. The kit-pool walk lives in
//! `world::inventory::collect_shared_inventories` and the helper
//! [`Sim::collect_public_container_grids`] below.

use anyhow::Result;
use bevy_ecs::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::components::{
    ContainerId, ContainerInteractionMode, GridInventory, InRegion, Inventory, Position,
    WorldContainer,
};
use crate::delta::WorldDelta;
use crate::inventory_grid::{self, PlaceOutcome};
use crate::items::ItemRegistry;
use crate::loot_containers::LootContainerRegistry;
use crate::loot_pools::LootPoolRegistry;
use crate::region::RegionId;
use crate::resources::{ContainerIdCounter, SimClock};

use super::Sim;

/// Default footprint for a container spawned by `Sim::drop_item` —
/// big enough to hold a few stacks dropped at once without immediately
/// overflowing into a second ground container, small enough to feel
/// like a "pile at your feet" rather than a stash.
const GROUND_CONTAINER_W: u32 = 4;
const GROUND_CONTAINER_H: u32 = 4;

/// Snap radius for merging a freshly-dropped item into an existing
/// nearby ground container. Anything within this XZ distance of the
/// player counts as "the same pile" — saves the world from being
/// littered with N single-stack containers when a player shifts and
/// drops repeatedly.
const GROUND_MERGE_RADIUS_M: f32 = 1.5;

impl Sim {
    /// Spawn a fresh world container at `pos` in `region`, with
    /// `(width × height)` cells and the supplied `is_public` flag.
    /// Returns the new id. Journals `WorldContainerSpawned`.
    ///
    /// Used by:
    /// - `drop_item` (private, small footprint)
    /// - scene-placed crates (public for shared bench bins; private
    ///   for player stashes — caller's choice)
    /// - PR-4b NPC death conversion (private, footprint sized to the
    ///   NPC's loadout)
    pub fn spawn_world_container(
        &mut self,
        pos: [f32; 3],
        region: RegionId,
        width: u32,
        height: u32,
        is_public: bool,
    ) -> Result<ContainerId> {
        if width == 0 || height == 0 {
            return Err(anyhow::anyhow!(
                "spawn_world_container: zero dimension ({width}×{height})"
            ));
        }
        let id = self.world.resource_mut::<ContainerIdCounter>().mint();
        let grid = GridInventory::new(width, height);
        let component = WorldContainer {
            id,
            grid: grid.clone(),
            is_public,
            // Caller-spawned containers (drops, corpses, ad-hoc
            // scenes) don't carry a faction by default — they
            // don't participate in the restock sweep.
            faction: None,
            depth_tier: 1,
            last_restock_tick: 0,
            interaction_mode: crate::components::ContainerInteractionMode::Openable,
        };
        self.world
            .spawn((component, Position(pos), InRegion(region)));
        self.record_delta(WorldDelta::WorldContainerSpawned {
            id,
            region,
            pos,
            is_public,
            initial_grid: grid,
        })?;
        Ok(id)
    }

    /// Phase 3D — spawn a world container from a hand-placed
    /// `LootContainerMarker3D` (or any caller passing the same
    /// data shape). Resolves grid size from `LootContainerRegistry`,
    /// rolls eager initial contents from `LootPoolRegistry` against
    /// the supplied `(faction, depth_tier)`, and stamps the
    /// container's `interaction_mode` so future damage routing
    /// (BREAKABLE) has the data it needs.
    ///
    /// `seed` lets the caller derive deterministic content rolls
    /// from authored ids — e.g. hash of `container_id_str` so the
    /// same map always rolls the same authored content within a
    /// save, but new saves with different seeds get different
    /// rolls (the seed is meant to be a save-time-varying value).
    /// `0` falls back to a fresh `thread_rng`-style state from
    /// `tick()`.
    ///
    /// Returns the new container id, or `Err` on unknown kind /
    /// zero-size grid / unknown region.
    #[allow(clippy::too_many_arguments)]
    pub fn register_authored_container(
        &mut self,
        kind_id: &str,
        region: RegionId,
        pos: [f32; 3],
        is_public: bool,
        faction: Option<String>,
        depth_tier: u8,
        interaction_mode: ContainerInteractionMode,
        seed: u64,
    ) -> Result<ContainerId> {
        // Resolve the kind from the registry.
        let kind = self
            .world
            .resource::<LootContainerRegistry>()
            .get(kind_id)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!("register_authored_container: unknown kind id {kind_id:?}")
            })?;
        let pool_registry = self.world.resource::<LootPoolRegistry>().clone();
        let item_registry = self.world.resource::<ItemRegistry>().clone();
        // Seed the roll RNG. If the caller passed 0 we mix the
        // sim tick + kind id so test runs that don't care about
        // determinism still get something.
        let effective_seed = if seed != 0 {
            seed
        } else {
            let tick = self.world.resource::<SimClock>().tick;
            // FNV-ish folding of the kind id keeps the seed
            // sensitive to kind without dragging in a hash crate.
            let mut salt: u64 = 0xCBF2_9CE4_8422_2325;
            for b in kind_id.as_bytes() {
                salt ^= u64::from(*b);
                salt = salt.wrapping_mul(0x100_0000_01B3);
            }
            tick.wrapping_add(salt)
        };
        let mut rng = ChaCha8Rng::seed_from_u64(effective_seed);

        // Mint id + roll contents.
        let id = self.world.resource_mut::<ContainerIdCounter>().mint();
        let mut grid = GridInventory::new(kind.grid.w, kind.grid.h);
        let roll_faction = faction.clone().unwrap_or_else(|| "nomads".to_string());
        let _placed = crate::world_seed::roll_initial_container_contents(
            &mut grid,
            &kind,
            &roll_faction,
            depth_tier,
            &pool_registry,
            &item_registry,
            &mut rng,
        );

        let component = WorldContainer {
            id,
            grid: grid.clone(),
            is_public,
            faction,
            depth_tier,
            last_restock_tick: 0,
            interaction_mode,
        };
        self.world
            .spawn((component, Position(pos), InRegion(region)));
        self.record_delta(WorldDelta::WorldContainerSpawned {
            id,
            region,
            pos,
            is_public,
            initial_grid: grid,
        })?;
        Ok(id)
    }

    /// Remove a world container from the sim. Caller is responsible
    /// for any item migration (e.g. NPC corpse cleanup auto-drops
    /// remaining contents into a smaller pile, in PR-4b). Journals
    /// `WorldContainerDespawned`.
    pub fn despawn_world_container(&mut self, id: ContainerId) -> Result<()> {
        let Some(e) = find_container_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!(
                "despawn_world_container: unknown id {id:?}"
            ));
        };
        self.world.despawn(e);
        self.record_delta(WorldDelta::WorldContainerDespawned { id })?;
        Ok(())
    }

    /// Read-only snapshot of a container's grid. `None` for unknown id.
    pub fn container_view(&mut self, id: ContainerId) -> Option<GridInventory> {
        let e = find_container_in(&mut self.world, id)?;
        self.world.get::<WorldContainer>(e).map(|c| c.grid.clone())
    }

    /// Container's current world position + region + is_public flag.
    /// `None` for unknown id.
    pub fn container_position(&mut self, id: ContainerId) -> Option<(RegionId, [f32; 3], bool)> {
        let e = find_container_in(&mut self.world, id)?;
        let pos = self.world.get::<Position>(e)?.0;
        let region = self.world.get::<InRegion>(e)?.0;
        let is_public = self.world.get::<WorldContainer>(e)?.is_public;
        Some((region, pos, is_public))
    }

    /// Containers in the same region as the player, within `radius` m
    /// (XZ distance). Returns `(id, position, is_public)` triples so
    /// the caller can decide what to do — the looting UI shows
    /// nearby ones, the kit-pool only walks `is_public = true` ones.
    pub fn containers_in_range(
        &mut self,
        steam_id: u64,
        radius: f32,
    ) -> Vec<(ContainerId, [f32; 3], bool)> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Vec::new();
        };
        let Some(pos) = self.world.get::<Position>(e).copied() else {
            return Vec::new();
        };
        let Some(region) = self.world.get::<InRegion>(e).copied() else {
            return Vec::new();
        };
        let r2 = radius * radius;
        let mut out = Vec::new();
        let mut q = self
            .world
            .query::<(&WorldContainer, &Position, &InRegion)>();
        for (wc, p, r) in q.iter(&self.world) {
            if r.0 != region.0 {
                continue;
            }
            let dx = p.0[0] - pos.0[0];
            let dz = p.0[2] - pos.0[2];
            if dx * dx + dz * dz <= r2 {
                out.push((wc.id, p.0, wc.is_public));
            }
        }
        out
    }

    /// Take the item at `source_idx` out of container `id` and grant
    /// it to the player's pockets. Journals
    /// `WorldContainerItemRemoved` then `ItemPickedUp`. Returns
    /// `Err` on unknown player / unknown container / out-of-range
    /// idx / pockets full (item left in the container).
    pub fn take_from_container(
        &mut self,
        steam_id: u64,
        id: ContainerId,
        source_idx: usize,
    ) -> Result<()> {
        let Some(player_e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let container_e = find_container_in(&mut self.world, id)
            .ok_or_else(|| anyhow::anyhow!("take_from_container: unknown container {id:?}"))?;
        // Bounds check + remove from container.
        let registry = self.world.resource::<ItemRegistry>().clone();
        let tick = self.world.resource::<crate::resources::SimClock>().tick;
        let removed = {
            let mut wc = self
                .world
                .get_mut::<WorldContainer>(container_e)
                .ok_or_else(|| anyhow::anyhow!("container missing component"))?;
            if source_idx >= wc.grid.items.len() {
                return Err(anyhow::anyhow!(
                    "take_from_container: source_idx {source_idx} out of range"
                ));
            }
            wc.grid.items.remove(source_idx)
        };
        // Try to place into pockets.
        let outcome = {
            let mut inv = self
                .world
                .get_mut::<Inventory>(player_e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            inventory_grid::grant_or_merge(
                &mut inv.0,
                &registry,
                &removed.stack.id,
                removed.stack.count,
                removed.stack.spawned_tick.max(tick),
            )
            .map_err(|e| anyhow::anyhow!("take_from_container placement: {e}"))?
        };
        if let PlaceOutcome::PartialOrFull { remaining, .. } = outcome {
            if remaining > 0 {
                // Pockets couldn't fit it; put it back in the
                // container at the same x/y/rotation it had.
                let mut wc = self.world.get_mut::<WorldContainer>(container_e).unwrap();
                wc.grid.items.insert(source_idx, removed);
                return Err(anyhow::anyhow!(
                    "take_from_container: pockets full ({remaining} units couldn't fit)"
                ));
            }
        }
        // Journal both sides.
        let count = removed.stack.count;
        self.record_delta(WorldDelta::WorldContainerItemRemoved {
            id,
            source_idx: source_idx as u32,
            taken: removed.stack.clone(),
            inner_grid: removed.inner_grid.clone(),
        })?;
        self.record_delta(WorldDelta::ItemPickedUp {
            steam_id,
            item_id: removed.stack.id.clone(),
            count,
            spawned_tick: removed.stack.spawned_tick,
        })?;
        Ok(())
    }

    /// Move the item at `(source_grid, source_idx)` from the player
    /// into container `id`. Source-grid strings match the equip API:
    /// `"pockets"` or `"equipped:<slot_id>"`. Journals
    /// `WorldContainerItemAdded`. Returns `Err` on missing source /
    /// missing container / no room in container.
    pub fn put_in_container(
        &mut self,
        steam_id: u64,
        id: ContainerId,
        source_grid: &str,
        source_idx: usize,
    ) -> Result<()> {
        let Some(player_e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let container_e = find_container_in(&mut self.world, id)
            .ok_or_else(|| anyhow::anyhow!("put_in_container: unknown container {id:?}"))?;
        // Pull the placement off the source grid (pockets or an
        // equipped container's inner grid).
        let placed =
            super::inventory::take_placed_item(&mut self.world, player_e, source_grid, source_idx)?;
        // Try to place into the destination container's grid.
        let registry = self.world.resource::<ItemRegistry>().clone();
        let outcome = {
            let mut wc = self
                .world
                .get_mut::<WorldContainer>(container_e)
                .ok_or_else(|| anyhow::anyhow!("container missing component"))?;
            inventory_grid::grant_or_merge(
                &mut wc.grid,
                &registry,
                &placed.stack.id,
                placed.stack.count,
                placed.stack.spawned_tick,
            )
            .map_err(|e| anyhow::anyhow!("put_in_container placement: {e}"))?
        };
        if let PlaceOutcome::PartialOrFull { remaining, .. } = outcome {
            if remaining > 0 {
                // Container couldn't fit; put back in source.
                super::inventory::put_placed_back(&mut self.world, player_e, source_grid, placed)?;
                return Err(anyhow::anyhow!(
                    "put_in_container: container full ({remaining} units couldn't fit)"
                ));
            }
        }
        self.record_delta(WorldDelta::WorldContainerItemAdded {
            id,
            item: placed.stack,
            inner_grid: placed.inner_grid,
        })?;
        Ok(())
    }

    /// Drop the slot at `slot_idx` from the player's inventory into a
    /// **personal** ground container at the player's position. If a
    /// nearby (within `GROUND_MERGE_RADIUS_M`) personal container
    /// already exists, the stack merges into that one instead of
    /// spawning a new pile. Journals `ItemDropped` + either a
    /// `WorldContainerSpawned` (with the dropped stack as the
    /// initial grid) or a `WorldContainerItemAdded` (if merging).
    pub fn drop_item_to_ground(&mut self, steam_id: u64, slot_idx: usize) -> Result<()> {
        let Some(player_e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let (player_pos, player_region) = {
            let pos = self
                .world
                .get::<Position>(player_e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Position"))?
                .0;
            let region = self
                .world
                .get::<InRegion>(player_e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no InRegion"))?
                .0;
            (pos, region)
        };
        // Take the stack out of pockets.
        let placed = {
            let mut inv = self
                .world
                .get_mut::<Inventory>(player_e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            if slot_idx >= inv.0.items.len() {
                return Err(anyhow::anyhow!(
                    "drop_item_to_ground: slot_idx {slot_idx} out of range"
                ));
            }
            inv.0.items.remove(slot_idx)
        };
        let count = placed.stack.count;
        self.record_delta(WorldDelta::ItemDropped {
            steam_id,
            slot_idx,
            count,
        })?;
        // Find a nearby personal container to merge into; if not,
        // spawn a fresh one.
        let merge_target = find_nearby_personal_container(
            &mut self.world,
            player_pos,
            player_region,
            GROUND_MERGE_RADIUS_M,
        );
        let registry = self.world.resource::<ItemRegistry>().clone();
        match merge_target {
            Some(container_id) => {
                let container_e =
                    find_container_in(&mut self.world, container_id).ok_or_else(|| {
                        anyhow::anyhow!("merge target {container_id:?} vanished mid-drop")
                    })?;
                {
                    let mut wc = self.world.get_mut::<WorldContainer>(container_e).unwrap();
                    let _ = inventory_grid::grant_or_merge(
                        &mut wc.grid,
                        &registry,
                        &placed.stack.id,
                        placed.stack.count,
                        placed.stack.spawned_tick,
                    )
                    .map_err(|e| anyhow::anyhow!("drop merge placement: {e}"))?;
                }
                self.record_delta(WorldDelta::WorldContainerItemAdded {
                    id: container_id,
                    item: placed.stack,
                    inner_grid: placed.inner_grid,
                })?;
            }
            None => {
                // Fresh container.
                let id = self.spawn_world_container(
                    player_pos,
                    player_region,
                    GROUND_CONTAINER_W,
                    GROUND_CONTAINER_H,
                    false,
                )?;
                let container_e = find_container_in(&mut self.world, id).unwrap();
                {
                    let mut wc = self.world.get_mut::<WorldContainer>(container_e).unwrap();
                    let _ = inventory_grid::grant_or_merge(
                        &mut wc.grid,
                        &registry,
                        &placed.stack.id,
                        placed.stack.count,
                        placed.stack.spawned_tick,
                    )
                    .map_err(|e| anyhow::anyhow!("drop spawn placement: {e}"))?;
                }
                self.record_delta(WorldDelta::WorldContainerItemAdded {
                    id,
                    item: placed.stack,
                    inner_grid: placed.inner_grid,
                })?;
            }
        }
        Ok(())
    }

    /// Collect grid clones for every **public** [`WorldContainer`] in
    /// the same region as `crafter_e` and within `radius_m`. Used by
    /// the crafting kit-pool to extend "what counts at the bench"
    /// from player inventories alone to nearby parts bins / shared
    /// crates. Returns clones — modifying them is a no-op on the
    /// world.
    pub(super) fn collect_public_container_grids(
        world: &mut World,
        crafter_e: Entity,
        radius_m: f32,
    ) -> Vec<GridInventory> {
        let Some(pos) = world.get::<Position>(crafter_e).copied() else {
            return Vec::new();
        };
        let Some(region) = world.get::<InRegion>(crafter_e).copied() else {
            return Vec::new();
        };
        let r2 = radius_m * radius_m;
        let mut out = Vec::new();
        let mut q = world.query::<(&WorldContainer, &Position, &InRegion)>();
        for (wc, p, r) in q.iter(world) {
            if !wc.is_public || r.0 != region.0 {
                continue;
            }
            let dx = p.0[0] - pos.0[0];
            let dz = p.0[2] - pos.0[2];
            if dx * dx + dz * dz <= r2 {
                out.push(wc.grid.clone());
            }
        }
        out
    }
}

/// Locate the container entity owning `id`. `None` if no such
/// container is in the world.
pub(super) fn find_container_in(world: &mut World, id: ContainerId) -> Option<Entity> {
    let mut q = world.query::<(Entity, &WorldContainer)>();
    q.iter(world).find(|(_, c)| c.id == id).map(|(e, _)| e)
}

/// Spawn a private corpse [`WorldContainer`] from a system context
/// (NPC death gates in `systems/npc_death_check.rs` and
/// `systems/npc_age.rs`). Mints an id, spawns the container entity
/// with the supplied grid as initial state, and queues a single
/// `WorldContainerSpawned` delta so mirrors see the corpse.
///
/// `inventory_grid` is taken by value because it's the dead NPC's
/// pockets — that grid moves into the container and the NPC entity
/// is despawned by the caller. If the grid is empty, no container is
/// spawned and no delta emitted (skips corpse-noise for NPCs that
/// rolled empty loadouts).
pub(crate) fn spawn_corpse_container(
    commands: &mut Commands,
    counter: &mut ContainerIdCounter,
    pending: &mut crate::resources::PendingDeltas,
    pos: [f32; 3],
    region: RegionId,
    inventory_grid: GridInventory,
) -> Option<ContainerId> {
    if inventory_grid.items.is_empty() {
        return None;
    }
    let id = counter.mint();
    let component = WorldContainer {
        id,
        grid: inventory_grid.clone(),
        is_public: false,
        // Corpses don't restock — contents are whatever the NPC
        // had at time of death. Phase 1F's loot-and-economy plan
        // §2 explicitly distinguishes corpse loot from container
        // loot.
        faction: None,
        depth_tier: 1,
        last_restock_tick: 0,
        interaction_mode: crate::components::ContainerInteractionMode::Openable,
    };
    commands.spawn((component, Position(pos), InRegion(region)));
    pending.push(crate::delta::WorldDelta::WorldContainerSpawned {
        id,
        region,
        pos,
        is_public: false,
        initial_grid: inventory_grid,
    });
    Some(id)
}

/// Find a personal (non-public) ground container within `radius` of
/// `pos` in the same `region`. Returns the closest one if multiple
/// match; `None` if none. Used by `drop_item_to_ground` so successive
/// drops merge into a single pile instead of littering the ground.
fn find_nearby_personal_container(
    world: &mut World,
    pos: [f32; 3],
    region: RegionId,
    radius: f32,
) -> Option<ContainerId> {
    let r2 = radius * radius;
    let mut closest: Option<(f32, ContainerId)> = None;
    let mut q = world.query::<(&WorldContainer, &Position, &InRegion)>();
    for (wc, p, r) in q.iter(world) {
        if wc.is_public || r.0 != region {
            continue;
        }
        let dx = p.0[0] - pos[0];
        let dz = p.0[2] - pos[2];
        let d2 = dx * dx + dz * dz;
        if d2 > r2 {
            continue;
        }
        if closest.map(|(best, _)| d2 < best).unwrap_or(true) {
            closest = Some((d2, wc.id));
        }
    }
    closest.map(|(_, id)| id)
}
