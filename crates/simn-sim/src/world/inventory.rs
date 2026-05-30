//! Inventory / items / crafting / salvage API on `Sim`.
//!
//! Every mutation emits a `WorldDelta` via `Sim::record_delta` so the
//! journal and the per-tick broadcast buffer stay in lockstep. Free
//! helpers (`merge_item_stack`, `consume_from_stacks`) are exposed
//! `pub(super)` so `persistence::apply_delta` can replay the same
//! inventory transforms during journal replay / client-side
//! application.
//!
//! Survives the post-Step-4 rewrite pattern: mutation methods route
//! through `self.find_player_entity` → `self.world.get_mut::<Inventory>(e)`
//! → merge or slot-edit → journal. The item definitions + recipes
//! come from read-only `ItemRegistry` / `RecipeRegistry` resources
//! loaded at `Sim::new` (not mutated at runtime except via
//! `set_perishable_ticks_for_test`).

use anyhow::Result;

use std::collections::HashMap;

use crate::components::{
    BodyPart, CraftJob, CraftingQueue, Equipment, EquippedItem, GridInventory, InRegion, Inventory,
    ItemInstance, NearCampfire, NearWorkbench, Npc, Position, Rotation,
};
use crate::delta::WorldDelta;
use crate::inventory_grid::{self, PlaceOutcome};
use crate::items::{
    ConsumeAction, CraftStation, EquipmentSlotRegistry, ItemDef, ItemId, ItemRegistry, ItemStack,
    KitRequirement, Recipe, RecipeRegistry, SalvageOutput, SlotId, ToolTier,
};
use crate::region::RegionId;
use crate::resources::{JobIdCounter, SimClock};

use super::Sim;

impl Sim {
    /// Look up an item definition by id. Returns `None` for unknown ids.
    pub fn item_def(&self, id: &ItemId) -> Option<&ItemDef> {
        self.world.resource::<ItemRegistry>().get(id)
    }

    /// Iterate every item the sim knows about.
    pub fn items(&self) -> impl Iterator<Item = &ItemDef> {
        self.world.resource::<ItemRegistry>().iter()
    }

    /// Read-only access to the full item registry. Used by the
    /// threaded-sim worker to clone an `Arc<ItemRegistry>` once
    /// at spawn for cross-thread inventory / equipment dict
    /// conversions (the registry is immutable for the session).
    pub fn item_registry(&self) -> &ItemRegistry {
        self.world.resource::<ItemRegistry>()
    }

    /// Look up a recipe by id. Returns `None` for unknown ids.
    pub fn recipe(&self, id: &str) -> Option<&Recipe> {
        self.world.resource::<RecipeRegistry>().get(id)
    }

    /// Iterate every recipe the sim knows about.
    pub fn recipes(&self) -> impl Iterator<Item = &Recipe> {
        self.world.resource::<RecipeRegistry>().iter()
    }

    /// Read-only access to the equipment-slot registry. The UI layer
    /// pulls this once to lay out the paper doll.
    pub fn equipment_slots(&self) -> &EquipmentSlotRegistry {
        self.world.resource::<EquipmentSlotRegistry>()
    }

    /// Clone the player's inventory as a flat slot list, dropping the
    /// grid position metadata. Empty for an unknown player. **Back-compat
    /// shape** — the new grid model keeps `(x, y, rotation)` per stack;
    /// callers that need that detail use [`Self::inventory_view_grid`].
    pub fn inventory_view(&mut self, steam_id: u64) -> Vec<ItemInstance> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Vec::new();
        };
        self.world
            .get::<Inventory>(e)
            .map(|inv| inv.0.items.iter().map(|p| p.stack.clone()).collect())
            .unwrap_or_default()
    }

    /// Clone the player's inventory as the full [`GridInventory`] —
    /// includes width/height + per-stack `(x, y, rotation)`. UI consumes
    /// this when rendering the grid; tests / craft pre-flight that only
    /// care about counts use [`Self::inventory_view`].
    pub fn inventory_view_grid(&mut self, steam_id: u64) -> GridInventory {
        let Some(e) = self.find_player_entity(steam_id) else {
            return GridInventory::player_default();
        };
        self.world
            .get::<Inventory>(e)
            .map(|inv| inv.0.clone())
            .unwrap_or_else(GridInventory::player_default)
    }

    /// Total carried weight: `sum(count × def.weight)`. Step 5 capped via
    /// [`crate::resources::InventoryConfig`] — over-cap halves stamina
    /// regen.
    pub fn inventory_weight(&mut self, steam_id: u64) -> f32 {
        let Some(e) = self.find_player_entity(steam_id) else {
            return 0.0;
        };
        let Some(inv) = self.world.get::<Inventory>(e) else {
            return 0.0;
        };
        let reg = self.world.resource::<ItemRegistry>();
        inv.0
            .items
            .iter()
            .map(|p| {
                reg.get(&p.stack.id)
                    .map(|d| d.weight * p.stack.count as f32)
                    .unwrap_or(0.0)
            })
            .sum()
    }

    /// True if the player is currently flagged as "near a campfire"
    /// (debug-only context for crafting; see [`Self::set_player_near_campfire`]).
    pub fn near_campfire(&mut self, steam_id: u64) -> bool {
        self.find_player_entity(steam_id)
            .and_then(|e| self.world.get::<NearCampfire>(e))
            .map(|nc| nc.0)
            .unwrap_or(false)
    }

    /// Tier of the nearest workbench the player is flagged as standing
    /// next to. `None` = no workbench in range. Set via
    /// [`Self::set_player_near_workbench`] (debug today; scene-driven
    /// once workbench entities land).
    pub fn near_workbench(&mut self, steam_id: u64) -> Option<ToolTier> {
        self.find_player_entity(steam_id)
            .and_then(|e| self.world.get::<NearWorkbench>(e))
            .and_then(|nw| nw.0)
    }

    /// Grant (or pick up) `count` of `id` into the player's inventory.
    /// Stacks merge into existing slots of the same id up to
    /// `def.stack_size`; overflow creates new slots. Perishable items
    /// only merge when `spawned_tick` matches so older stacks expire
    /// first. Journals `ItemPickedUp`.
    pub fn grant_item(&mut self, steam_id: u64, id: &ItemId, count: u32) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        // Validate item id exists before touching anything.
        if self.world.resource::<ItemRegistry>().get(id).is_none() {
            return Err(anyhow::anyhow!("unknown item {:?}", id));
        }
        let tick = self.world.resource::<SimClock>().tick;
        // Two-phase borrow: snapshot inventory + registry refs separately
        // so the placement engine (which needs &Inventory + &ItemRegistry)
        // can run without a borrow conflict, then write back.
        let outcome = {
            let registry = self.world.resource::<ItemRegistry>().clone();
            let mut inv = self
                .world
                .get_mut::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            inventory_grid::grant_or_merge(&mut inv.0, &registry, id, count, tick)
                .map_err(|e| anyhow::anyhow!("grant_item placement: {e}"))?
        };
        if let PlaceOutcome::PartialOrFull { remaining, .. } = outcome {
            if remaining > 0 {
                tracing::warn!(
                    target: "inventory",
                    "grant_item: dropped {remaining} of {count} {:?} — no room",
                    id
                );
            }
        }
        self.record_delta(WorldDelta::ItemPickedUp {
            steam_id,
            item_id: id.clone(),
            count,
            spawned_tick: tick,
        })?;
        Ok(())
    }

    /// Convenience alias for [`Self::grant_item`] — the pickup verb
    /// reads better from engine-side code even though Step 4 doesn't
    /// yet have ground-item entities to pick up from.
    pub fn pickup(&mut self, steam_id: u64, id: &ItemId, count: u32) -> Result<()> {
        self.grant_item(steam_id, id, count)
    }

    /// Remove the slot at `slot_idx` from the player's pockets and
    /// drop it onto the ground at the player's feet. If a personal
    /// ground container is already nearby it merges into that pile;
    /// otherwise a fresh private [`crate::components::WorldContainer`]
    /// is spawned. Journals `ItemDropped` plus either
    /// `WorldContainerSpawned` + `WorldContainerItemAdded` or just
    /// `WorldContainerItemAdded` for the merge case. Delegates to
    /// [`Self::drop_item_to_ground`].
    pub fn drop_item(&mut self, steam_id: u64, slot_idx: usize) -> Result<()> {
        self.drop_item_to_ground(steam_id, slot_idx)
    }

    /// Swap the stacks at `from` and `to`. Both indices must point to
    /// existing slots. Journals `ItemMoved`.
    pub fn move_between_slots(&mut self, steam_id: u64, from: usize, to: usize) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        {
            let mut inv = self
                .world
                .get_mut::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            if from >= inv.0.items.len() || to >= inv.0.items.len() {
                return Err(anyhow::anyhow!("move_between_slots: index out of range"));
            }
            inv.0.items.swap(from, to);
        }
        self.record_delta(WorldDelta::ItemMoved {
            steam_id,
            from_slot: from,
            to_slot: to,
        })?;
        Ok(())
    }

    /// Move the item at `(from_grid, from_idx)` into `to_grid` at the
    /// first free spot (rotation tried first, then 90° if rotatable).
    /// `from_grid` / `to_grid` are `"pockets"` or `"equipped:<slot>"`.
    /// The item's `inner_grid` (loaded backpack, mag with rounds)
    /// travels with it. Restores to source on placement failure —
    /// nothing leaks.
    ///
    /// Use [`Self::move_between_slots`] for same-pockets swaps.
    pub fn move_between_grids(
        &mut self,
        steam_id: u64,
        from_grid: &str,
        from_idx: usize,
        to_grid: &str,
    ) -> Result<()> {
        if from_grid == to_grid {
            return Err(anyhow::anyhow!(
                "move_between_grids: same grid {:?} (use move_between_slots)",
                from_grid
            ));
        }
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let placed = take_placed_item(&mut self.world, e, from_grid, from_idx)?;
        let registry = self.world.resource::<ItemRegistry>().clone();
        let Some(def) = registry.get(&placed.stack.id).cloned() else {
            let item_id = placed.stack.id.clone();
            let _ = put_placed_back(&mut self.world, e, from_grid, placed);
            return Err(anyhow::anyhow!(
                "move_between_grids: unknown item {:?}",
                item_id
            ));
        };
        let Some(dest) = grid_mut_by_source(&mut self.world, e, to_grid) else {
            let _ = put_placed_back(&mut self.world, e, from_grid, placed);
            return Err(anyhow::anyhow!(
                "move_between_grids: dest grid {:?} not found",
                to_grid
            ));
        };
        let Some((x, y, rotation)) =
            crate::inventory_grid::find_first_fit_any_rotation(dest, &registry, &def)
        else {
            let _ = put_placed_back(&mut self.world, e, from_grid, placed);
            return Err(anyhow::anyhow!(
                "move_between_grids: no room in {:?}",
                to_grid
            ));
        };
        let stack = placed.stack.clone();
        let inner_grid = placed.inner_grid.clone();
        let to_idx = crate::inventory_grid::place_at_with_inner(
            dest,
            &registry,
            stack.clone(),
            inner_grid.clone(),
            x,
            y,
            rotation,
        )
        .map_err(|err| anyhow::anyhow!("move_between_grids placement: {err}"))?;
        self.record_delta(WorldDelta::ItemMovedBetweenGrids {
            steam_id,
            from_grid: from_grid.to_string(),
            from_idx,
            to_grid: to_grid.to_string(),
            to_idx,
            item: stack,
            inner_grid,
        })?;
        Ok(())
    }

    /// Apply the item in `slot_idx` to the player. Reads the item's
    /// [`ConsumeAction`] and routes to the matching existing `Sim`
    /// method (`eat` / `drink` / `apply_drug` / `apply_bandage` / …).
    /// `body_part` is required for wound-treatment items; for others
    /// it's ignored. If the underlying action errors (e.g. "no wound
    /// to bandage"), the item is **not** consumed.
    pub fn consume_from_slot(
        &mut self,
        steam_id: u64,
        slot_idx: usize,
        body_part: Option<BodyPart>,
    ) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let (item_id, action) = {
            let inv = self
                .world
                .get::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            if slot_idx >= inv.0.items.len() {
                return Err(anyhow::anyhow!(
                    "consume_from_slot: slot {slot_idx} out of range"
                ));
            }
            let item_id = inv.0.items[slot_idx].stack.id.clone();
            let reg = self.world.resource::<ItemRegistry>();
            let def = reg
                .get(&item_id)
                .ok_or_else(|| anyhow::anyhow!("unknown item {:?}", item_id))?;
            let action = def
                .consume_action
                .ok_or_else(|| anyhow::anyhow!("item {:?} is not consumable", item_id))?;
            (item_id, action)
        };
        if action.needs_body_part() && body_part.is_none() {
            return Err(anyhow::anyhow!(
                "consume_from_slot: item {:?} requires a body part",
                item_id
            ));
        }
        // Route to existing API. If it errors, early-return without
        // decrementing — the item stays in the slot for retry.
        match action {
            ConsumeAction::Eat { food_kind } => self.eat(steam_id, food_kind)?,
            ConsumeAction::Drink { water_kind } => self.drink(steam_id, water_kind)?,
            ConsumeAction::ApplyDrug { drug } => {
                let _ = self.apply_drug(steam_id, drug)?;
            }
            ConsumeAction::ApplyBandage => self.apply_bandage(steam_id, body_part.unwrap())?,
            ConsumeAction::ApplyTourniquet => {
                self.apply_tourniquet(steam_id, body_part.unwrap())?
            }
            ConsumeAction::ApplyDisinfectant => {
                self.apply_disinfectant(steam_id, body_part.unwrap())?
            }
            ConsumeAction::ApplyStitch => self.apply_stitch(steam_id, body_part.unwrap())?,
            ConsumeAction::ApplyWoundPack => self.apply_wound_pack(steam_id, body_part.unwrap())?,
            ConsumeAction::ApplyAntibiotics => self.apply_antibiotics(steam_id)?,
        }
        {
            let mut inv = self.world.get_mut::<Inventory>(e).unwrap();
            if slot_idx < inv.0.items.len() {
                inv.0.items[slot_idx].stack.count =
                    inv.0.items[slot_idx].stack.count.saturating_sub(1);
                if inv.0.items[slot_idx].stack.count == 0 {
                    inv.0.items.remove(slot_idx);
                }
            }
        }
        self.record_delta(WorldDelta::ItemConsumed {
            steam_id,
            slot_idx,
            body_part,
        })?;
        Ok(())
    }

    /// Salvage the item in `slot_idx` into its component outputs (see
    /// [`crate::items::SalvageRecipe`]). Requires the recipe's
    /// `tool_required` in the player's inventory. Rolls one unit of
    /// each output in `[min, max]` deterministically (seeded by tick +
    /// slot). Returns the produced outputs. Journals `ItemsSalvaged`
    /// with the actual rolled list so replay is stable.
    pub fn salvage(&mut self, steam_id: u64, slot_idx: usize) -> Result<Vec<ItemStack>> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let tick = self.world.resource::<SimClock>().tick;
        let (item_id, recipe) = {
            let inv = self
                .world
                .get::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            if slot_idx >= inv.0.items.len() {
                return Err(anyhow::anyhow!("salvage: slot {slot_idx} out of range"));
            }
            let item_id = inv.0.items[slot_idx].stack.id.clone();
            let reg = self.world.resource::<ItemRegistry>();
            let def = reg
                .get(&item_id)
                .ok_or_else(|| anyhow::anyhow!("unknown item {:?}", item_id))?;
            let recipe = def
                .salvage
                .clone()
                .ok_or_else(|| anyhow::anyhow!("item {:?} has no salvage recipe", item_id))?;
            (item_id, recipe)
        };
        if let Some(tool) = &recipe.tool_required {
            let has_tool = self
                .world
                .get::<Inventory>(e)
                .map(|inv| {
                    inv.0
                        .items
                        .iter()
                        .any(|p| &p.stack.id == tool && p.stack.count > 0)
                })
                .unwrap_or(false);
            if !has_tool {
                return Err(anyhow::anyhow!(
                    "salvage {item_id:?}: requires {tool:?} in inventory"
                ));
            }
        }
        let outputs = roll_salvage_outputs(&recipe.outputs, tick, slot_idx);
        {
            let mut inv = self.world.get_mut::<Inventory>(e).unwrap();
            inv.0.items[slot_idx].stack.count = inv.0.items[slot_idx].stack.count.saturating_sub(1);
            if inv.0.items[slot_idx].stack.count == 0 {
                inv.0.items.remove(slot_idx);
            }
        }
        // Grant outputs via a direct merge (no per-output journal) so
        // the atomic `ItemsSalvaged` record is the single authority.
        for stack in &outputs {
            self.merge_direct(steam_id, &stack.id, stack.count, tick)?;
        }
        self.record_delta(WorldDelta::ItemsSalvaged {
            steam_id,
            source_slot: slot_idx,
            outputs: outputs.clone(),
            tick,
        })?;
        Ok(outputs)
    }

    /// Craft the recipe `recipe_id` instantly (no queue). Checks
    /// tool + kit + context + inputs, consumes inputs FIFO across
    /// stacks, grants outputs with `spawned_tick = now`. Journals
    /// `ItemsCrafted`.
    ///
    /// Tool + kit requirements are satisfied by **any nearby
    /// co-op player's inventory** within
    /// [`CRAFTING_SHARE_RADIUS_M`] in the same region — you don't
    /// have to hold your crewmate's gunsmith kit to use it while
    /// standing at the same bench. Inputs are still consumed from
    /// the crafter's own inventory.
    pub fn craft(&mut self, steam_id: u64, recipe_id: &str) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let tick = self.world.resource::<SimClock>().tick;
        let recipe = self
            .world
            .resource::<RecipeRegistry>()
            .get(recipe_id)
            .ok_or_else(|| anyhow::anyhow!("unknown recipe {recipe_id}"))?
            .clone();
        let shared = collect_shared_inventories(&mut self.world, e);
        if let Some(tool) = &recipe.required_tool {
            if !any_inventory_has_item(&shared, tool) {
                return Err(anyhow::anyhow!(
                    "craft {recipe_id}: requires {tool:?} in group inventory"
                ));
            }
        }
        if let Some(ctx) = recipe.required_context {
            if !player_has_station(&self.world, e, ctx) {
                return Err(anyhow::anyhow!(
                    "craft {recipe_id}: requires {ctx:?} station"
                ));
            }
        }
        if let Some(kit) = recipe.required_kit {
            if !any_inventory_has_kit(&shared, &self.world, kit) {
                return Err(anyhow::anyhow!("craft {recipe_id}: requires {kit:?}"));
            }
        }
        {
            let inv = self
                .world
                .get::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            for input in &recipe.inputs {
                let have = inventory_grid::count_of(&inv.0, &input.id);
                if have < input.count {
                    return Err(anyhow::anyhow!(
                        "craft {recipe_id}: need {} of {:?}, have {}",
                        input.count,
                        input.id,
                        have
                    ));
                }
            }
        }
        {
            let mut inv = self.world.get_mut::<Inventory>(e).unwrap();
            for input in &recipe.inputs {
                inventory_grid::consume_from_grid(&mut inv.0, &input.id, input.count);
            }
        }
        for out in &recipe.outputs {
            self.merge_direct(steam_id, &out.id, out.count, tick)?;
        }
        self.record_delta(WorldDelta::ItemsCrafted {
            steam_id,
            recipe_id: recipe_id.to_string(),
            tick,
        })?;
        Ok(())
    }

    /// Toggle the debug "near campfire" flag on the player entity.
    /// Step 5 replaces this with real campfire entities + proximity
    /// checks; the [`Recipe::required_context`] field survives that
    /// transition. Journals `NearCampfireSet`.
    pub fn set_player_near_campfire(&mut self, steam_id: u64, value: bool) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        if let Some(mut nc) = self.world.get_mut::<NearCampfire>(e) {
            nc.0 = value;
        } else {
            self.world.entity_mut(e).insert(NearCampfire(value));
        }
        self.record_delta(WorldDelta::NearCampfireSet { steam_id, value })?;
        Ok(())
    }

    /// Test-only: pull every NPC's `(id, pos, yaw, region)`. Used by
    /// network-replay tests to assert mirror convergence.
    #[doc(hidden)]
    pub fn all_npc_positions_for_test(
        &mut self,
    ) -> Vec<(crate::components::NpcId, [f32; 3], f32, RegionId)> {
        let mut q = self
            .world
            .query::<(&Npc, &Position, &Rotation, &InRegion)>();
        q.iter(&self.world)
            .map(|(n, p, r, reg)| (n.id, p.0, r.0, reg.0))
            .collect()
    }

    /// Test-only: override an item's `perishable_ticks` in the loaded
    /// registry so expiry tests don't wait 30 in-world minutes. The
    /// change lives only in this `Sim` instance's registry; items.toml
    /// is untouched.
    #[doc(hidden)]
    pub fn set_perishable_ticks_for_test(&mut self, id: &str, ticks: u64) {
        let mut reg = self.world.resource_mut::<ItemRegistry>();
        reg.set_perishable_for_test(&ItemId::from(id), Some(ticks));
    }

    /// Internal: merge a stack into the player's inventory without
    /// journaling. Used by `salvage` and `craft` which emit a single
    /// atomic delta instead of per-output `ItemPickedUp`s.
    fn merge_direct(
        &mut self,
        steam_id: u64,
        id: &ItemId,
        count: u32,
        spawned_tick: u64,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let registry = self.world.resource::<ItemRegistry>().clone();
        let mut inv = self
            .world
            .get_mut::<Inventory>(e)
            .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
        let outcome =
            inventory_grid::grant_or_merge(&mut inv.0, &registry, id, count, spawned_tick)
                .map_err(|e| anyhow::anyhow!("merge_direct: {e}"))?;
        if let PlaceOutcome::PartialOrFull { remaining, .. } = outcome {
            if remaining > 0 {
                tracing::warn!(
                    target: "inventory",
                    "merge_direct: dropped {remaining} of {count} {:?} — no room",
                    id
                );
            }
        }
        Ok(())
    }

    /// Check whether the player could queue (or instantly craft) the
    /// given recipe right now. Returns a [`CraftabilityReport`] that
    /// names every missing precondition (so UI can show "need 2 more
    /// metal_scrap + advanced gunsmith kit"), never panics for
    /// unknown recipes (returns `ok = false` with a blank report —
    /// caller is expected to validate the id separately if it cares).
    pub fn can_craft(&mut self, steam_id: u64, recipe_id: &str) -> CraftabilityReport {
        let Some(e) = self.find_player_entity(steam_id) else {
            return CraftabilityReport::default();
        };
        let Some(recipe) = self
            .world
            .resource::<RecipeRegistry>()
            .get(recipe_id)
            .cloned()
        else {
            return CraftabilityReport::default();
        };
        let inv = self.world.get::<Inventory>(e).cloned().unwrap_or_default();
        let shared = collect_shared_inventories(&mut self.world, e);

        let inputs: Vec<InputStatus> = recipe
            .inputs
            .iter()
            .map(|stack| {
                let have = inventory_grid::count_of(&inv.0, &stack.id);
                InputStatus {
                    id: stack.id.clone(),
                    need: stack.count,
                    have,
                }
            })
            .collect();

        let missing_tool = recipe
            .required_tool
            .as_ref()
            .filter(|t| !any_inventory_has_item(&shared, t))
            .cloned();

        let missing_kit = recipe
            .required_kit
            .filter(|kit| !any_inventory_has_kit(&shared, &self.world, *kit));

        let wrong_station = recipe
            .required_context
            .filter(|ctx| !player_has_station(&self.world, e, *ctx));

        let enough_inputs = inputs.iter().all(|s| s.have >= s.need);
        let ok = enough_inputs
            && missing_tool.is_none()
            && missing_kit.is_none()
            && wrong_station.is_none();

        CraftabilityReport {
            ok,
            inputs,
            missing_tool,
            missing_kit,
            wrong_station,
        }
    }

    /// Queue `count` copies of `recipe_id` on the player's crafting
    /// queue. Validates tool/kit/context/inputs up front; consumes
    /// **all** required inputs × count immediately (spec §7.5 — locks
    /// materials at queue time to prevent cancel-re-queue duping).
    /// Returns the new job's id. Emits `CraftJobQueued`.
    ///
    /// Per-unit completions happen deterministically inside the tick
    /// schedule via `tick_crafting_queue` and are **not** journaled —
    /// replay re-derives them from queue state + clock, same pattern
    /// as `tick_perishables`.
    pub fn queue_craft(&mut self, steam_id: u64, recipe_id: &str, count: u32) -> Result<u32> {
        if count == 0 {
            return Err(anyhow::anyhow!("queue_craft count must be > 0"));
        }
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let recipe = self
            .world
            .resource::<RecipeRegistry>()
            .get(recipe_id)
            .ok_or_else(|| anyhow::anyhow!("unknown recipe {recipe_id}"))?
            .clone();

        if let Some(ctx) = recipe.required_context {
            if !player_has_station(&self.world, e, ctx) {
                return Err(anyhow::anyhow!(
                    "queue_craft {recipe_id}: requires {ctx:?} station"
                ));
            }
        }
        let shared = collect_shared_inventories(&mut self.world, e);
        if let Some(kit) = recipe.required_kit {
            if !any_inventory_has_kit(&shared, &self.world, kit) {
                return Err(anyhow::anyhow!("queue_craft {recipe_id}: requires {kit:?}"));
            }
        }
        if let Some(tool) = &recipe.required_tool {
            if !any_inventory_has_item(&shared, tool) {
                return Err(anyhow::anyhow!(
                    "queue_craft {recipe_id}: requires {tool:?} in group inventory"
                ));
            }
        }

        // Total inputs required across all units.
        let total_inputs: Vec<ItemStack> = recipe
            .inputs
            .iter()
            .map(|s| ItemStack {
                id: s.id.clone(),
                count: s.count.saturating_mul(count),
            })
            .collect();

        {
            let inv = self
                .world
                .get::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            for input in &total_inputs {
                let have = inventory_grid::count_of(&inv.0, &input.id);
                if have < input.count {
                    return Err(anyhow::anyhow!(
                        "queue_craft {recipe_id}: need {} of {:?}, have {}",
                        input.count,
                        input.id,
                        have
                    ));
                }
            }
        }

        // Consume all materials up front.
        {
            let mut inv = self.world.get_mut::<Inventory>(e).unwrap();
            for input in &total_inputs {
                inventory_grid::consume_from_grid(&mut inv.0, &input.id, input.count);
            }
        }

        let started_tick = self.world.resource::<SimClock>().tick;
        let job_id = self.world.resource_mut::<JobIdCounter>().mint();
        let time_ticks_per_unit = recipe.time_ticks;
        let job = CraftJob {
            id: job_id,
            recipe_id: recipe_id.to_string(),
            count_remaining: count,
            ticks_remaining: time_ticks_per_unit,
            started_tick,
        };
        if let Some(mut cq) = self.world.get_mut::<CraftingQueue>(e) {
            cq.0.push(job);
        } else {
            self.world.entity_mut(e).insert(CraftingQueue(vec![job]));
        }

        self.record_delta(WorldDelta::CraftJobQueued {
            steam_id,
            job_id,
            recipe_id: recipe_id.to_string(),
            count,
            time_ticks_per_unit,
            inputs_consumed: total_inputs,
            started_tick,
        })?;
        Ok(job_id)
    }

    /// Cancel a queued craft job. Refunds inputs for the units that
    /// haven't started yet (the in-progress unit, if any, is forfeit
    /// per the delta contract — simpler than fractional refunds).
    /// Emits `CraftJobCancelled`. No-op if the job id isn't on the
    /// player's queue.
    pub fn cancel_craft(&mut self, steam_id: u64, job_id: u32) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };

        // Read the job, figure out refund, then remove it.
        let Some((idx, recipe_id, count_remaining, ticks_remaining)) =
            self.world.get::<CraftingQueue>(e).and_then(|cq| {
                cq.0.iter().enumerate().find_map(|(i, j)| {
                    if j.id == job_id {
                        Some((i, j.recipe_id.clone(), j.count_remaining, j.ticks_remaining))
                    } else {
                        None
                    }
                })
            })
        else {
            return Err(anyhow::anyhow!(
                "cancel_craft: job {job_id} not found for player {steam_id}"
            ));
        };

        let recipe = self
            .world
            .resource::<RecipeRegistry>()
            .get(&recipe_id)
            .cloned();

        // Refund = whole recipe × (count_remaining - 1) if the head
        // unit has started ticking down (forfeit), else × count_remaining.
        let refund_units = if let Some(ref r) = recipe {
            if ticks_remaining > 0 && ticks_remaining < r.time_ticks {
                count_remaining.saturating_sub(1)
            } else {
                count_remaining
            }
        } else {
            0
        };

        let refund: Vec<ItemStack> = recipe
            .as_ref()
            .map(|r| {
                r.inputs
                    .iter()
                    .map(|s| ItemStack {
                        id: s.id.clone(),
                        count: s.count.saturating_mul(refund_units),
                    })
                    .filter(|s| s.count > 0)
                    .collect()
            })
            .unwrap_or_default();

        // Remove job from queue.
        if let Some(mut cq) = self.world.get_mut::<CraftingQueue>(e) {
            cq.0.remove(idx);
        }

        // Grant refund. Perishable outputs start fresh at now.
        let now = self.world.resource::<SimClock>().tick;
        for stack in &refund {
            self.merge_direct(steam_id, &stack.id, stack.count, now)?;
        }

        self.record_delta(WorldDelta::CraftJobCancelled {
            steam_id,
            job_id,
            refund,
        })?;
        Ok(())
    }

    /// Read-only snapshot of the player's current crafting queue.
    /// Empty for an unknown player or one with no jobs.
    pub fn crafting_queue(&mut self, steam_id: u64) -> Vec<CraftJob> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Vec::new();
        };
        self.world
            .get::<CraftingQueue>(e)
            .map(|cq| cq.0.clone())
            .unwrap_or_default()
    }

    /// Debug / test helper — set the `NearWorkbench(tier)` flag on
    /// the player. Production will drive this from a scene-side
    /// proximity system reading placed workbench entities; until
    /// then tests and the debug overlay call this directly. Journals
    /// `NearWorkbenchSet`.
    pub fn set_player_near_workbench(
        &mut self,
        steam_id: u64,
        tier: Option<ToolTier>,
    ) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        if let Some(mut nw) = self.world.get_mut::<NearWorkbench>(e) {
            nw.0 = tier;
        } else {
            self.world.entity_mut(e).insert(NearWorkbench(tier));
        }
        self.record_delta(WorldDelta::NearWorkbenchSet { steam_id, tier })?;
        Ok(())
    }

    // -------- Equipment / paper doll / hotbar --------

    /// Read-only view of the player's current equipment loadout.
    /// Empty map for an unknown player or one with nothing equipped.
    pub fn equipment_view(&mut self, steam_id: u64) -> HashMap<SlotId, EquippedItem> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return HashMap::new();
        };
        self.world
            .get::<Equipment>(e)
            .map(|eq| eq.0.clone())
            .unwrap_or_default()
    }

    /// Move the `source_idx`-th item out of `source_grid` and into
    /// the paper-doll slot `slot_id`. Fails if:
    ///
    /// - the slot id is unknown (registry miss)
    /// - the item's category isn't in the slot's `accepts` list AND
    ///   the item's `equip_slots` whitelist doesn't name this slot
    /// - the slot is already occupied (caller must unequip first)
    /// - `source_grid` names a grid that doesn't exist, or the index
    ///   is out of range
    ///
    /// The moved item keeps its `inner_grid` payload (a loaded
    /// backpack stays loaded through equip). Journals `ItemEquipped`.
    ///
    /// `source_grid` strings:
    /// - `"pockets"` — the player's `Inventory` grid
    /// - `"equipped:<other_slot_id>"` — a nested grid inside a
    ///   different equipped container (unload a kit from your
    ///   backpack directly onto the belt)
    pub fn equip(
        &mut self,
        steam_id: u64,
        slot_id: &SlotId,
        source_grid: &str,
        source_idx: usize,
    ) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        // Validate slot exists.
        let registry = self.world.resource::<EquipmentSlotRegistry>();
        if registry.get(slot_id).is_none() {
            return Err(anyhow::anyhow!("equip: unknown slot {:?}", slot_id));
        }
        if self
            .world
            .get::<Equipment>(e)
            .map(|eq| eq.0.contains_key(slot_id))
            .unwrap_or(false)
        {
            return Err(anyhow::anyhow!(
                "equip: slot {:?} already occupied",
                slot_id
            ));
        }
        // Pull the placement off the source grid.
        let placed = take_placed_item(&mut self.world, e, source_grid, source_idx)?;
        // Validate item can equip to this slot.
        {
            let item_reg = self.world.resource::<ItemRegistry>();
            let slot_reg = self.world.resource::<EquipmentSlotRegistry>();
            let def = item_reg
                .get(&placed.stack.id)
                .ok_or_else(|| anyhow::anyhow!("equip: unknown item {:?}", placed.stack.id))?;
            if !slot_reg.can_equip(slot_id, def) {
                // Put it back — placement contract: equip never loses items.
                let item_id = placed.stack.id.clone();
                let _ = put_placed_back(&mut self.world, e, source_grid, placed);
                return Err(anyhow::anyhow!(
                    "equip: item {:?} doesn't fit slot {:?}",
                    item_id,
                    slot_id
                ));
            }
        }
        // Initialize `weapon_state` iff the item is a weapon (has a
        // `weapon_config` block in its def). No magazine loaded until
        // the player runs reload.
        let weapon_state = super::weapons::init_weapon_state_for(
            &placed.stack.id,
            self.world.resource::<ItemRegistry>(),
        );
        // Stow in Equipment.
        let equipped = EquippedItem {
            stack: placed.stack.clone(),
            inner_grid: placed.inner_grid.clone(),
            weapon_state,
        };
        if let Some(mut eq) = self.world.get_mut::<Equipment>(e) {
            eq.0.insert(slot_id.clone(), equipped);
        } else {
            let mut map = HashMap::new();
            map.insert(slot_id.clone(), equipped);
            self.world.entity_mut(e).insert(Equipment(map));
        }
        self.record_delta(WorldDelta::ItemEquipped {
            steam_id,
            slot_id: slot_id.clone(),
            item: placed.stack,
            inner_grid: placed.inner_grid,
            source_grid: source_grid.to_string(),
            source_idx,
        })?;
        Ok(())
    }

    /// Pull the item at `slot_id` off the paper doll and drop it
    /// into `dest_grid` at the first free spot. Fails if:
    ///
    /// - the slot is empty
    /// - `dest_grid` doesn't exist
    /// - no free cells large enough for the item's footprint (caller
    ///   can drop-to-ground instead in PR-4)
    ///
    /// Journals `ItemUnequipped`. The equipped item's `inner_grid`
    /// (if any) travels with it.
    pub fn unequip(&mut self, steam_id: u64, slot_id: &SlotId, dest_grid: &str) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        // Pull out of Equipment.
        let eq_item = {
            let mut eq = self
                .world
                .get_mut::<Equipment>(e)
                .ok_or_else(|| anyhow::anyhow!("unequip: no Equipment component"))?;
            eq.0.remove(slot_id)
                .ok_or_else(|| anyhow::anyhow!("unequip: slot {:?} is empty", slot_id))?
        };
        // Place into destination grid.
        let registry = self.world.resource::<ItemRegistry>().clone();
        let def = registry
            .get(&eq_item.stack.id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("unequip: unknown item {:?}", eq_item.stack.id))?;

        let item_id = eq_item.stack.id.clone();
        let Some(grid) = grid_mut_by_source(&mut self.world, e, dest_grid) else {
            // Dest grid missing — put the item back on the paper doll
            // to keep the API total (no lost items).
            if let Some(mut eq) = self.world.get_mut::<Equipment>(e) {
                eq.0.insert(slot_id.clone(), eq_item);
            }
            return Err(anyhow::anyhow!(
                "unequip: dest grid {:?} not found",
                dest_grid
            ));
        };
        let Some((x, y, rotation)) =
            crate::inventory_grid::find_first_fit_any_rotation(grid, &registry, &def)
        else {
            // No room. Put it back.
            if let Some(mut eq) = self.world.get_mut::<Equipment>(e) {
                eq.0.insert(slot_id.clone(), eq_item);
            }
            return Err(anyhow::anyhow!(
                "unequip: no room in {:?} for {:?}",
                dest_grid,
                item_id
            ));
        };
        let stack = eq_item.stack.clone();
        let inner_grid = eq_item.inner_grid.clone();
        crate::inventory_grid::place_at_with_inner(
            grid,
            &registry,
            stack.clone(),
            inner_grid.clone(),
            x,
            y,
            rotation,
        )
        .map_err(|e| anyhow::anyhow!("unequip placement: {e}"))?;
        self.record_delta(WorldDelta::ItemUnequipped {
            steam_id,
            slot_id: slot_id.clone(),
            item: stack,
            inner_grid,
            dest_grid: dest_grid.to_string(),
        })?;
        Ok(())
    }

    /// Fire the `consume_from_slot` flow for the belt slot bound to
    /// hotbar index `idx` (1-based, matching
    /// [`EquipmentSlotDef::hotbar_index`]). Does nothing if no slot
    /// claims that index or the slot is empty. Returns an error if
    /// the underlying consume action errors (e.g. bandage with no
    /// wound) — the item is **not** consumed in that case.
    pub fn consume_from_hotbar(
        &mut self,
        steam_id: u64,
        idx: u8,
        body_part: Option<crate::components::BodyPart>,
    ) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let slot_id = {
            let reg = self.world.resource::<EquipmentSlotRegistry>();
            let Some(def) = reg.by_hotbar_index(idx) else {
                return Err(anyhow::anyhow!("hotbar index {idx} not bound to any slot"));
            };
            def.id.clone()
        };
        // The hotbar slot holds a single stack (1×1). Consume one
        // unit; the slot empties if the stack is now zero.
        let stack_id = {
            let Some(eq) = self.world.get::<Equipment>(e) else {
                return Err(anyhow::anyhow!("hotbar: no Equipment on player"));
            };
            let Some(eq_item) = eq.0.get(&slot_id) else {
                return Err(anyhow::anyhow!(
                    "hotbar: slot {:?} (index {idx}) is empty",
                    slot_id
                ));
            };
            eq_item.stack.id.clone()
        };
        let action = {
            let item_reg = self.world.resource::<ItemRegistry>();
            let def = item_reg
                .get(&stack_id)
                .ok_or_else(|| anyhow::anyhow!("hotbar: unknown item {:?}", stack_id))?;
            def.consume_action
                .ok_or_else(|| anyhow::anyhow!("hotbar: item {:?} is not consumable", stack_id))?
        };
        if action.needs_body_part() && body_part.is_none() {
            return Err(anyhow::anyhow!(
                "hotbar: item {:?} requires a body part",
                stack_id
            ));
        }
        match action {
            ConsumeAction::Eat { food_kind } => self.eat(steam_id, food_kind)?,
            ConsumeAction::Drink { water_kind } => self.drink(steam_id, water_kind)?,
            ConsumeAction::ApplyDrug { drug } => {
                let _ = self.apply_drug(steam_id, drug)?;
            }
            ConsumeAction::ApplyBandage => self.apply_bandage(steam_id, body_part.unwrap())?,
            ConsumeAction::ApplyTourniquet => {
                self.apply_tourniquet(steam_id, body_part.unwrap())?
            }
            ConsumeAction::ApplyDisinfectant => {
                self.apply_disinfectant(steam_id, body_part.unwrap())?
            }
            ConsumeAction::ApplyStitch => self.apply_stitch(steam_id, body_part.unwrap())?,
            ConsumeAction::ApplyWoundPack => self.apply_wound_pack(steam_id, body_part.unwrap())?,
            ConsumeAction::ApplyAntibiotics => self.apply_antibiotics(steam_id)?,
        }
        // Decrement the hotbar stack. The underlying Sim method
        // succeeded, so now we consume one unit.
        if let Some(mut eq) = self.world.get_mut::<Equipment>(e) {
            if let Some(eq_item) = eq.0.get_mut(&slot_id) {
                eq_item.stack.count = eq_item.stack.count.saturating_sub(1);
                if eq_item.stack.count == 0 {
                    eq.0.remove(&slot_id);
                }
            }
        }
        Ok(())
    }
}

/// Pull the `source_idx`-th placed item out of the grid named by
/// `source_grid` on the player entity. Returns the removed
/// [`PlacedItem`]. Caller is responsible for either re-placing it
/// somewhere or emitting a journal record.
pub(super) fn take_placed_item(
    world: &mut bevy_ecs::world::World,
    e: bevy_ecs::entity::Entity,
    source_grid: &str,
    source_idx: usize,
) -> Result<crate::components::PlacedItem> {
    let Some(grid) = grid_mut_by_source(world, e, source_grid) else {
        return Err(anyhow::anyhow!(
            "source grid {:?} not found on player",
            source_grid
        ));
    };
    if source_idx >= grid.items.len() {
        return Err(anyhow::anyhow!(
            "source idx {source_idx} out of range in grid {:?}",
            source_grid
        ));
    }
    Ok(grid.items.remove(source_idx))
}

/// Put a [`PlacedItem`] back into `source_grid` at its original
/// position. Used by `equip` when a later validation step fails and
/// the item needs to be restored — we don't leak items on a partial
/// failure. Ignores overlap errors (the slot was just vacated).
pub(super) fn put_placed_back(
    world: &mut bevy_ecs::world::World,
    e: bevy_ecs::entity::Entity,
    source_grid: &str,
    placed: crate::components::PlacedItem,
) -> Result<()> {
    let Some(grid) = grid_mut_by_source(world, e, source_grid) else {
        return Err(anyhow::anyhow!(
            "put_back: source grid {:?} missing",
            source_grid
        ));
    };
    grid.items.push(placed);
    Ok(())
}

/// Resolve a source-grid string to `&mut GridInventory` on the
/// player entity. `"pockets"` returns the `Inventory` component's
/// inner grid; `"equipped:<slot>"` returns the nested grid inside the
/// equipped container at that slot.
fn grid_mut_by_source<'w>(
    world: &'w mut bevy_ecs::world::World,
    e: bevy_ecs::entity::Entity,
    source: &str,
) -> Option<&'w mut crate::components::GridInventory> {
    if source == "pockets" {
        return world.get_mut::<Inventory>(e).map(|inv| {
            let inv = inv.into_inner();
            &mut inv.0
        });
    }
    if let Some(slot_str) = source.strip_prefix("equipped:") {
        let slot_id = SlotId::from(slot_str);
        if let Some(eq) = world.get_mut::<Equipment>(e) {
            let eq = eq.into_inner();
            if let Some(eq_item) = eq.0.get_mut(&slot_id) {
                return eq_item.inner_grid.as_mut();
            }
        }
    }
    None
}

/// One recipe input's satisfaction status — exposed via
/// [`CraftabilityReport`] so UIs can render `"metal_scrap: 3 / 5"`
/// next to each line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputStatus {
    pub id: ItemId,
    pub need: u32,
    pub have: u32,
}

/// Output of [`Sim::can_craft`]: structured rejection reasons so the
/// recipe browser can show "requires advanced gunsmith kit" without
/// parsing error strings. `ok = true` ⇔ every reason field is
/// `None`/empty and every input has `have >= need`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CraftabilityReport {
    pub ok: bool,
    pub inputs: Vec<InputStatus>,
    pub missing_tool: Option<ItemId>,
    pub missing_kit: Option<KitRequirement>,
    pub wrong_station: Option<CraftStation>,
}

/// Radius (meters, XZ) within which nearby co-op players' inventories
/// count for kit / tool checks. Approximates "everyone standing at
/// the same workbench" — once actual workbench entities land, the
/// radius anchor shifts from the crafter's position to the bench's
/// and this constant goes away.
pub const CRAFTING_SHARE_RADIUS_M: f32 = 6.0;

/// True if the player is standing in range of the named crafting
/// station. Benches are cumulative (an Advanced bench satisfies
/// Basic recipes); Campfire requires a campfire specifically.
fn player_has_station(
    world: &bevy_ecs::world::World,
    e: bevy_ecs::entity::Entity,
    ctx: CraftStation,
) -> bool {
    match ctx {
        CraftStation::Campfire => world.get::<NearCampfire>(e).map(|nc| nc.0).unwrap_or(false),
        CraftStation::BasicBench | CraftStation::AdvancedBench | CraftStation::ExpertBench => {
            let needed = ctx.bench_rank().unwrap_or(0);
            let have = world
                .get::<NearWorkbench>(e)
                .and_then(|nw| nw.0.map(tier_rank))
                .unwrap_or(0);
            have >= needed
        }
    }
}

/// Collect every nearby player's **accessible grids** (pockets +
/// every equipped container's inner grid + every nested container
/// sitting inside a worn container) whose owner is in the same
/// region and within [`CRAFTING_SHARE_RADIUS_M`] of `crafter_e`.
///
/// Used by kit / tool checks so coop partners at the same workbench
/// can pool kits without handing items back and forth — and so a
/// toolkit stashed in your backpack still counts as "on the bench"
/// without having to dig it out to pockets first. Each returned
/// `GridInventory` is a cloned snapshot; mutating them won't
/// affect the world.
fn collect_shared_inventories(
    world: &mut bevy_ecs::world::World,
    crafter_e: bevy_ecs::entity::Entity,
) -> Vec<GridInventory> {
    let Some(pos) = world.get::<Position>(crafter_e).copied() else {
        return Vec::new();
    };
    let Some(region) = world.get::<InRegion>(crafter_e).copied() else {
        return Vec::new();
    };
    let r2 = CRAFTING_SHARE_RADIUS_M * CRAFTING_SHARE_RADIUS_M;
    let mut out = Vec::new();
    let mut q = world.query::<(
        &crate::components::PlayerOwned,
        &InRegion,
        &Position,
        &Inventory,
        Option<&Equipment>,
    )>();
    for (_po, in_region, p, inv, eq) in q.iter(world) {
        if in_region.0 != region.0 {
            continue;
        }
        let dx = p.0[0] - pos.0[0];
        let dz = p.0[2] - pos.0[2];
        if dx * dx + dz * dz > r2 {
            continue;
        }
        // Pockets.
        append_grid_recursive(&inv.0, &mut out);
        // Every equipped container's inner grid.
        if let Some(eq) = eq {
            for eq_item in eq.0.values() {
                if let Some(ref nested) = eq_item.inner_grid {
                    append_grid_recursive(nested, &mut out);
                }
            }
        }
    }
    // Plus every nearby **public** WorldContainer (parts bin chained
    // to the workbench, shared crew crate). Private containers
    // (player stashes, ground drops) are deliberately excluded — see
    // mechanics/crafting.md.
    for grid in Sim::collect_public_container_grids(world, crafter_e, CRAFTING_SHARE_RADIUS_M) {
        append_grid_recursive(&grid, &mut out);
    }
    out
}

/// Push `grid` onto `out` and recurse into any container items
/// inside it that carry their own `inner_grid`. Captures "kit in
/// a kit bag inside your backpack" style nesting in one pass.
fn append_grid_recursive(grid: &GridInventory, out: &mut Vec<GridInventory>) {
    out.push(grid.clone());
    for placed in &grid.items {
        if let Some(ref nested) = placed.inner_grid {
            append_grid_recursive(nested, out);
        }
    }
}

/// True if any grid in `grids` contains a tool / kit item that
/// satisfies the recipe's [`KitRequirement`] — any [`ItemDef`] whose
/// [`crate::items::ToolSpec`] matches `specialty` with
/// `tier >= min_tier`. Higher-tier kits cover lower-tier
/// requirements within the same specialty. The caller uses
/// [`collect_shared_inventories`] to flatten pockets + equipped
/// containers + group-nearby inventories into the grid list.
fn any_inventory_has_kit(
    grids: &[GridInventory],
    world: &bevy_ecs::world::World,
    kit: KitRequirement,
) -> bool {
    let reg = world.resource::<ItemRegistry>();
    grids.iter().any(|grid| {
        grid.items.iter().any(|placed| {
            if placed.stack.count == 0 {
                return false;
            }
            let Some(def) = reg.get(&placed.stack.id) else {
                return false;
            };
            let Some(tool) = def.tool else {
                return false;
            };
            tool.specialty == kit.specialty && tier_rank(tool.tier) >= tier_rank(kit.min_tier)
        })
    })
}

/// True if any grid in `grids` contains at least one of the named
/// item id. Used for recipe `required_tool` checks, which name a
/// specific item (e.g. `cookware`) rather than a specialty+tier.
fn any_inventory_has_item(grids: &[GridInventory], id: &ItemId) -> bool {
    grids.iter().any(|grid| {
        grid.items
            .iter()
            .any(|p| &p.stack.id == id && p.stack.count > 0)
    })
}

fn tier_rank(t: ToolTier) -> u8 {
    match t {
        ToolTier::Basic => 1,
        ToolTier::Advanced => 2,
        ToolTier::Expert => 3,
    }
}

/// Merge `count` of `id` into `inv`, respecting `stack_size` and the
/// perishable age-mixing rule, AND the new grid placement constraints
/// (footprint, rotation, free-slot scan). Wrapper around
/// [`crate::inventory_grid::grant_or_merge`] kept under the legacy
/// free-function name so `persistence::apply_delta` and other callers
/// don't need to know about the placement engine. Returns the
/// outcome so callers can detect partial drops in low-room cases.
pub(crate) fn merge_item_stack(
    inv: &mut Inventory,
    registry: &ItemRegistry,
    id: ItemId,
    count: u32,
    spawned_tick: u64,
) -> PlaceOutcome {
    inventory_grid::grant_or_merge(&mut inv.0, registry, &id, count, spawned_tick).unwrap_or(
        PlaceOutcome::PartialOrFull {
            placed: 0,
            remaining: count,
            touched_indices: Vec::new(),
        },
    )
}

/// Consume `need` of `id` FIFO across `inv`. Removes emptied stacks.
/// Caller has already confirmed the total is available. Returns the
/// actual count consumed (may be less than `need` if the grid didn't
/// have enough — caller hasn't been doing that check today, kept for
/// safety).
pub(super) fn consume_from_stacks(inv: &mut Inventory, id: &ItemId, need: u32) -> u32 {
    inventory_grid::consume_from_grid(&mut inv.0, id, need)
}

/// Roll salvage outputs deterministically for a (tick, slot) pair.
/// Uses a ChaCha8 PRNG seeded from the pair so replay (when re-rolled
/// on the same tick + slot) would land on the same values — though
/// replay normally uses the journaled outputs directly.
fn roll_salvage_outputs(outputs: &[SalvageOutput], tick: u64, slot_idx: usize) -> Vec<ItemStack> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
        tick.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(slot_idx as u64),
    );
    let mut out = Vec::new();
    for o in outputs {
        let count = if o.max > o.min {
            rng.gen_range(o.min..=o.max)
        } else {
            o.min
        };
        if count > 0 {
            out.push(ItemStack {
                id: o.id.clone(),
                count,
            });
        }
    }
    out
}
