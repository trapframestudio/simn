//! Per-tick perishable aging.
//!
//! Pure system: scans player inventories for stacks whose item def has
//! `perishable_ticks` set, and removes any whose
//! `clock.tick - spawned_tick >= perishable_ticks`. Not journaled —
//! expiry is a deterministic function of the stack's mint tick and the
//! current sim tick, same pattern as `tick_active_effects` retiring
//! expired drug effects.
//!
//! Replay therefore reconstructs the same inventory without an
//! explicit `ItemExpired` delta: on load, stacks past their expiry
//! tick are dropped the next time this system runs.
//!
//! Player-only (via `With<PlayerOwned>`); NPC inventories land later.

use bevy_ecs::prelude::{Query, Res, With};

use crate::components::{Inventory, PlayerOwned};
use crate::items::ItemRegistry;
use crate::resources::SimClock;

pub fn tick_perishables(
    mut q: Query<&mut Inventory, With<PlayerOwned>>,
    registry: Res<ItemRegistry>,
    clock: Res<SimClock>,
) {
    let now = clock.tick;
    for mut inv in &mut q {
        inv.0.items.retain(|placed| {
            let Some(def) = registry.get(&placed.stack.id) else {
                return true;
            };
            let Some(max_age) = def.perishable_ticks else {
                return true;
            };
            now.saturating_sub(placed.stack.spawned_tick) < max_age
        });
    }
}
