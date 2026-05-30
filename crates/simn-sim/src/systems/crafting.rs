//! Per-tick crafting-queue progression.
//!
//! Pure system. For each player with a non-empty [`CraftingQueue`],
//! decrement the active (front) job's `ticks_remaining`. When it hits
//! zero:
//!
//! 1. Look up the recipe and merge each output into the player's
//!    [`Inventory`] via the shared
//!    [`crate::world::inventory::merge_item_stack`] helper — same
//!    stack-size + perishable-age rules the pickup path uses, stamped
//!    at the current tick so any perishable outputs start aging from
//!    completion.
//! 2. Decrement `count_remaining`. If it's still > 0, reset
//!    `ticks_remaining` to the recipe's `time_ticks` so the next unit
//!    begins. Else pop the job; the next queued job (if any) becomes
//!    active next tick.
//!
//! Deliberately not journaled. Per-unit completions are a
//! deterministic function of (queue state, `SimClock::tick`,
//! recipe catalog), same pattern as
//! [`super::inventory::tick_perishables`]. Host and mirror sims both
//! run this system and reach identical outputs without per-unit wire
//! traffic.
//!
//! Only `CraftJobQueued` / `CraftJobCancelled` (player-initiated
//! lifecycle events) travel via [`crate::delta::WorldDelta`]; those
//! stock `Inventory` with the material debit / refund so the
//! deterministic-ness of this system starts from equivalent
//! pre-conditions on every replay path.

use bevy_ecs::prelude::{Query, Res, With};

use crate::components::{CraftingQueue, Inventory, PlayerOwned};
use crate::items::{ItemRegistry, RecipeRegistry};
use crate::resources::SimClock;
use crate::world::merge_item_stack;

pub fn tick_crafting_queue(
    mut q: Query<(&mut CraftingQueue, &mut Inventory), With<PlayerOwned>>,
    items: Res<ItemRegistry>,
    recipes: Res<RecipeRegistry>,
    clock: Res<SimClock>,
) {
    let now = clock.tick;
    for (mut queue, mut inv) in &mut q {
        if queue.0.is_empty() {
            continue;
        }
        let job = &mut queue.0[0];
        if job.ticks_remaining > 0 {
            job.ticks_remaining -= 1;
        }
        if job.ticks_remaining > 0 {
            continue;
        }
        // Unit complete. Look up the recipe; a missing entry (e.g. a
        // deleted recipe referenced by an old save) drops the job
        // silently rather than looping.
        let Some(recipe) = recipes.get(&job.recipe_id) else {
            queue.0.remove(0);
            continue;
        };
        let outputs = recipe.outputs.clone();
        let next_unit_ticks = recipe.time_ticks;
        job.count_remaining = job.count_remaining.saturating_sub(1);
        let finished = job.count_remaining == 0;
        if !finished {
            job.ticks_remaining = next_unit_ticks;
        }

        for stack in outputs {
            if items.get(&stack.id).is_none() {
                continue;
            }
            merge_item_stack(&mut inv, &items, stack.id.clone(), stack.count, now);
        }

        if finished {
            queue.0.remove(0);
        }
    }
}
