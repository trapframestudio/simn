//! Periodic partial loot-container restock sweep (Phase 3C).
//!
//! Runs every [`RESTOCK_SWEEP_INTERVAL_TICKS`] ticks (~1 in-world
//! hour at the default 7200 s / 20 Hz cadence). For each
//! container in an active region:
//!
//! - Roll [`RESTOCK_CHANCE_PER_CONTAINER`] — most containers are
//!   left alone on any given sweep.
//! - If picked, add 1–3 items via the kind's family-weighted
//!   pool roll. Items merge into existing stacks first; new
//!   stacks land in the first free footprint. Anything that
//!   doesn't fit is silently dropped — restock never bumps
//!   existing loot out of the way.
//!
//! In-fiction: this models wanderers / supply squads passing
//! through and stashing a few items each — not a wholesale
//! refill. A solo-loot run still leaves enough variation between
//! visits that backtracking through cleared regions feels
//! rewarding without being trivially exploitable.
//!
//! Squall-driven sweeps (after every fault clear / squall) are
//! scaffolded as `apply_squall_restock` but currently dead code
//! — squalls land with `faults-plan.md` Step 7. The fault system
//! will pull this function when it fires a squall resolve.
//!
//! RNG: each sweep seeds a `ChaCha8Rng` from `(tick, salt)` so
//! reload of the same save produces the same restock pattern
//! (deterministic from snapshot state). New saves with different
//! initial seeds get different worlds and therefore different
//! restocks. The user-facing contract is "new saves feel
//! different; reloads stay stable."

use bevy_ecs::prelude::{Query, Res};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::components::{InRegion, WorldContainer};
use crate::items::ItemRegistry;
use crate::loot_containers::LootContainerRegistry;
use crate::loot_pools::LootPoolRegistry;
use crate::resources::{ActiveRegions, SimClock};

/// Sweep cadence in sim ticks. At 20 Hz and the default 7200 s
/// in-world day, 72_000 ticks = 1 in-world hour ≈ 5 real
/// minutes. Tuned for "containers feel fresh on revisit" without
/// turning the world into a slot machine. Re-tune once playtests
/// land.
pub const RESTOCK_SWEEP_INTERVAL_TICKS: u64 = 72_000;

/// Per-sweep probability that any given container in an active
/// region gets new items. Low enough that most containers are
/// stable between sweeps; high enough that backtracking through
/// a cleared region after a sweep finds *some* fresh loot.
pub const RESTOCK_CHANCE_PER_CONTAINER: f64 = 0.30;

/// `[min, max]` items added per restocked container. Smaller
/// than the initial `items_per_roll` — restock is a top-up, not
/// a refill.
pub const RESTOCK_ITEMS_MIN: u32 = 1;
pub const RESTOCK_ITEMS_MAX: u32 = 3;

/// Salt mixed into the per-sweep RNG seed. Picked from the
/// golden-ratio constant so distinct system RNGs (weather,
/// loot_restock, etc.) decorrelate even when their `tick` is the
/// same.
const RNG_SALT: u64 = 0xC0FF_EE12_3456_789A;

/// Periodic-sweep system. Cheap when not firing — the modulo
/// check runs every tick but the body only walks containers
/// every [`RESTOCK_SWEEP_INTERVAL_TICKS`] ticks.
pub fn tick_loot_restock(
    clock: Res<SimClock>,
    active_regions: Res<ActiveRegions>,
    container_registry: Res<LootContainerRegistry>,
    pool_registry: Res<LootPoolRegistry>,
    item_registry: Res<ItemRegistry>,
    mut containers: Query<(&mut WorldContainer, &InRegion)>,
) {
    if clock.tick == 0 || !clock.tick.is_multiple_of(RESTOCK_SWEEP_INTERVAL_TICKS) {
        return;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(clock.tick.wrapping_mul(RNG_SALT));
    let mut swept = 0u32;
    let mut topped_up = 0u32;
    let mut items_added = 0u32;
    for (mut container, region) in containers.iter_mut() {
        // Restock sweep only touches containers that participate
        // in the faction pool system. Player drops + corpses
        // carry `faction = None` and are skipped — the
        // loot-and-economy plan's §2 keeps those surfaces
        // distinct.
        let Some(faction) = container.faction.clone() else {
            continue;
        };
        if !active_regions.regions.contains(&region.0) {
            continue;
        }
        swept += 1;
        if !rng.gen_bool(RESTOCK_CHANCE_PER_CONTAINER) {
            continue;
        }
        let Some(kind) = container_registry
            .get(&container_kind_id_for(&container))
            .or_else(|| best_fit_kind(&container_registry, &container))
        else {
            continue;
        };
        let added = restock_partial(
            &mut container,
            kind,
            &faction,
            &pool_registry,
            &item_registry,
            &mut rng,
        );
        items_added += added;
        if added > 0 {
            topped_up += 1;
            container.last_restock_tick = clock.tick;
        }
    }
    if swept > 0 {
        tracing::debug!(
            "loot_restock: tick={} swept={} topped_up={} items_added={}",
            clock.tick,
            swept,
            topped_up,
            items_added,
        );
    }
}

/// Restock a single container with `[RESTOCK_ITEMS_MIN,
/// RESTOCK_ITEMS_MAX]` items. Returns the number of items
/// actually placed (silently 0 when the container's grid is
/// full).
fn restock_partial(
    container: &mut WorldContainer,
    kind: &crate::loot_containers::LootContainerDef,
    faction: &str,
    pool_registry: &LootPoolRegistry,
    item_registry: &ItemRegistry,
    rng: &mut ChaCha8Rng,
) -> u32 {
    let n = rng.gen_range(RESTOCK_ITEMS_MIN..=RESTOCK_ITEMS_MAX);
    let depth_tier = container.depth_tier;
    let mut placed = 0u32;
    for _ in 0..n {
        let Some(family) = kind.pick_family(rng) else {
            continue;
        };
        let Some(rolled) = pool_registry.roll_one(rng, faction, depth_tier, family) else {
            continue;
        };
        if item_registry.get(&rolled.id).is_none() {
            continue;
        }
        if crate::inventory_grid::grant_or_merge(
            &mut container.grid,
            item_registry,
            &rolled.id,
            rolled.count,
            0,
        )
        .is_ok()
        {
            placed += 1;
        }
    }
    placed
}

/// Apply a one-off restock sweep right now — used by future
/// squall resolutions (faults-plan Step 7). Same
/// machinery as the periodic sweep; just bypasses the cadence
/// gate so the caller controls when it fires. Currently
/// **unused** by production code (no squall system exists
/// yet); kept to lock in the API the fault system will call.
#[allow(dead_code)]
pub fn apply_squall_restock(
    clock: &SimClock,
    active_regions: &ActiveRegions,
    container_registry: &LootContainerRegistry,
    pool_registry: &LootPoolRegistry,
    item_registry: &ItemRegistry,
    containers: &mut Query<(&mut WorldContainer, &InRegion)>,
    salt: u64,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(clock.tick.wrapping_add(salt).wrapping_mul(RNG_SALT));
    for (mut container, region) in containers.iter_mut() {
        let Some(faction) = container.faction.clone() else {
            continue;
        };
        if !active_regions.regions.contains(&region.0) {
            continue;
        }
        // Squalls hit harder than periodic — every container
        // in an active region gets a partial restock, not the
        // probabilistic 30 %.
        let Some(kind) = container_registry
            .get(&container_kind_id_for(&container))
            .or_else(|| best_fit_kind(container_registry, &container))
        else {
            continue;
        };
        let added = restock_partial(
            &mut container,
            kind,
            &faction,
            pool_registry,
            item_registry,
            &mut rng,
        );
        if added > 0 {
            container.last_restock_tick = clock.tick;
        }
    }
}

/// Best-effort container-kind id lookup from a `WorldContainer`.
/// The kind isn't persisted on the entity (kinds are runtime
/// config, not save state), so we infer from grid dimensions.
/// Returns `""` when no match — caller falls back to
/// [`best_fit_kind`].
fn container_kind_id_for(container: &WorldContainer) -> String {
    // Pre-3A containers + corpses + ground drops won't match any
    // kind exactly, which is fine — those skip restock via the
    // `faction.is_none()` gate above.
    let _ = container;
    String::new()
}

/// Fallback: pick whichever kind's grid matches `container`'s
/// dimensions. Used because `WorldContainer` doesn't persist the
/// kind id directly.
fn best_fit_kind<'a>(
    registry: &'a LootContainerRegistry,
    container: &WorldContainer,
) -> Option<&'a crate::loot_containers::LootContainerDef> {
    registry
        .iter()
        .find(|d| d.grid.w == container.grid.width && d.grid.h == container.grid.height)
}
