//! grid-based grid inventory placement engine.
//!
//! Pure functions over [`crate::components::GridInventory`] — no ECS,
//! no `Sim`, no `World`. Every mutation goes through this module so
//! the overlap / bounds / rotation rules stay in one place.
//!
//! ## Model
//!
//! - Each grid is `width × height` cells, origin top-left.
//! - Each item occupies a rectangle: `def.size` (w×h), or its rotated
//!   transpose if the [`PlacedItem::rotation`] is `Deg90`.
//! - No two items may overlap.
//! - Items stay inside the grid (`x + w <= grid.width`,
//!   `y + h <= grid.height`).
//! - Stacks merge **across positions** when the item id and
//!   `spawned_tick` match (perishable rule), respecting
//!   `def.stack_size`. The placement engine prefers merging into an
//!   existing stack before allocating a new cell.
//!
//! ## Performance
//!
//! Naive O(N × cells) overlap check per placement query — fine for
//! the grid sizes we use (player pockets 4×4, backpacks up to ~10×10,
//! world crates up to ~12×12, total items per grid in the dozens).
//! If profiling ever flags this, swap to a bitmap occupancy mask.

use crate::components::{GridInventory, ItemInstance, ItemRotation, PlacedItem};
use crate::items::{GridSize, ItemDef, ItemId, ItemRegistry};

/// What [`grant_or_merge`] / [`place_new`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaceOutcome {
    /// Every requested unit landed somewhere — either merged into
    /// existing stacks or placed in newly-allocated cells. The vec
    /// lists all items that were touched (mutated count, or newly
    /// inserted), useful for partial-success reporting.
    Placed { touched_indices: Vec<usize> },
    /// At least one unit couldn't fit. `placed` is how many actually
    /// landed; `remaining` is the leftover count the caller must
    /// either reject or stash elsewhere.
    PartialOrFull {
        placed: u32,
        remaining: u32,
        touched_indices: Vec<usize>,
    },
}

/// Reason a placement attempt failed (besides "no room", which is a
/// `Partial` outcome on the merge path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlacementError {
    OutOfBounds {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        gw: u32,
        gh: u32,
    },
    Overlap(usize),
    NotRotatable,
    BadIndex(usize),
    UnknownItem(ItemId),
}

impl std::fmt::Display for PlacementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds { x, y, w, h, gw, gh } => write!(
                f,
                "placement out of bounds: ({x},{y}) + ({w}×{h}) doesn't fit in {gw}×{gh} grid"
            ),
            Self::Overlap(i) => write!(f, "placement overlaps existing item at index {i}"),
            Self::NotRotatable => write!(f, "rotation requested but item is not rotatable"),
            Self::BadIndex(i) => write!(f, "item index {i} out of range"),
            Self::UnknownItem(id) => write!(f, "unknown item id {id:?}"),
        }
    }
}

impl std::error::Error for PlacementError {}

/// Effective footprint of an item after applying rotation.
pub fn footprint(def: &ItemDef, rotation: ItemRotation) -> GridSize {
    match rotation {
        ItemRotation::Deg0 => def.size,
        ItemRotation::Deg90 => def.size.rotated(),
    }
}

/// True if the rectangle `(x, y, w, h)` fits inside the grid bounds.
fn in_bounds(grid: &GridInventory, x: u32, y: u32, w: u32, h: u32) -> bool {
    let Some(right) = x.checked_add(w) else {
        return false;
    };
    let Some(bottom) = y.checked_add(h) else {
        return false;
    };
    right <= grid.width && bottom <= grid.height
}

/// True if rectangles `(ax, ay, aw, ah)` and `(bx, by, bw, bh)` share
/// any cell.
#[allow(clippy::too_many_arguments)]
fn rects_overlap(ax: u32, ay: u32, aw: u32, ah: u32, bx: u32, by: u32, bw: u32, bh: u32) -> bool {
    let a_right = ax + aw;
    let a_bottom = ay + ah;
    let b_right = bx + bw;
    let b_bottom = by + bh;
    ax < b_right && bx < a_right && ay < b_bottom && by < a_bottom
}

/// Check whether placing an item with footprint `(w, h)` at `(x, y)`
/// would be legal — within bounds and not overlapping any existing
/// placed item. Pass `ignore_idx = Some(i)` to skip the `i`-th item
/// in the overlap test (used by the move-within-grid path so an item
/// doesn't conflict with itself).
#[allow(clippy::too_many_arguments)]
pub fn fits_at(
    grid: &GridInventory,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    ignore_idx: Option<usize>,
) -> Result<(), PlacementError> {
    if !in_bounds(grid, x, y, w, h) {
        return Err(PlacementError::OutOfBounds {
            x,
            y,
            w,
            h,
            gw: grid.width,
            gh: grid.height,
        });
    }
    for (i, p) in grid.items.iter().enumerate() {
        if Some(i) == ignore_idx {
            continue;
        }
        // Existing item's footprint — we don't re-check rotation here
        // because `p.rotation` already determines effective `(w, h)`,
        // but we need to look up its `def` to know `def.size`. To
        // keep this function registry-free, the caller is expected
        // to use `fits_at_with_def` if they need that lookup. For
        // legality checks we pull each existing item's footprint via
        // its `rotation` and a small assumption: the existing
        // placement is also legal so we can re-compute.
        let (pw, ph) = placed_footprint_unchecked(p);
        if rects_overlap(x, y, w, h, p.x, p.y, pw, ph) {
            return Err(PlacementError::Overlap(i));
        }
    }
    Ok(())
}

/// Best-effort placed-item footprint: read the cached `(w, h)` we
/// stored alongside the position. Today we don't store size
/// separately and require a registry lookup — but for overlap checks
/// we can derive it by walking back through every placed item's
/// stack id and looking up the def. Since we don't have the registry
/// in this function's signature, callers that need overlap checks on
/// rotated items use [`fits_at_with_registry`].
///
/// **Limitation today**: this fallback assumes 1×1. The
/// registry-aware path is the correct one — kept here so
/// registry-free callers (rare) don't crash.
fn placed_footprint_unchecked(_p: &PlacedItem) -> (u32, u32) {
    (1, 1)
}

/// Registry-aware version of [`fits_at`]: looks up each existing
/// item's def to compute its real footprint. Use this in the host
/// path; tests use [`fits_at`] when the items are known 1×1.
#[allow(clippy::too_many_arguments)]
pub fn fits_at_with_registry(
    grid: &GridInventory,
    registry: &ItemRegistry,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    ignore_idx: Option<usize>,
) -> Result<(), PlacementError> {
    if !in_bounds(grid, x, y, w, h) {
        return Err(PlacementError::OutOfBounds {
            x,
            y,
            w,
            h,
            gw: grid.width,
            gh: grid.height,
        });
    }
    for (i, p) in grid.items.iter().enumerate() {
        if Some(i) == ignore_idx {
            continue;
        }
        let (pw, ph) = match registry.get(&p.stack.id) {
            Some(def) => {
                let size = footprint(def, p.rotation);
                (size.w, size.h)
            }
            None => (1, 1),
        };
        if rects_overlap(x, y, w, h, p.x, p.y, pw, ph) {
            return Err(PlacementError::Overlap(i));
        }
    }
    Ok(())
}

/// Top-left scan for the first `(x, y)` where a `(w, h)` rectangle
/// would fit. Returns `None` if the grid has no room at all.
pub fn find_first_fit(
    grid: &GridInventory,
    registry: &ItemRegistry,
    w: u32,
    h: u32,
) -> Option<(u32, u32)> {
    if w == 0 || h == 0 || w > grid.width || h > grid.height {
        return None;
    }
    for y in 0..=(grid.height - h) {
        for x in 0..=(grid.width - w) {
            if fits_at_with_registry(grid, registry, x, y, w, h, None).is_ok() {
                return Some((x, y));
            }
        }
    }
    None
}

/// Try Deg0 first, then Deg90 if `def.rotatable`. Returns the
/// rotation that worked along with the position.
pub fn find_first_fit_any_rotation(
    grid: &GridInventory,
    registry: &ItemRegistry,
    def: &ItemDef,
) -> Option<(u32, u32, ItemRotation)> {
    let s = def.size;
    if let Some((x, y)) = find_first_fit(grid, registry, s.w, s.h) {
        return Some((x, y, ItemRotation::Deg0));
    }
    if def.rotatable && def.size.w != def.size.h {
        let s = def.size.rotated();
        if let Some((x, y)) = find_first_fit(grid, registry, s.w, s.h) {
            return Some((x, y, ItemRotation::Deg90));
        }
    }
    None
}

/// Place a brand-new stack at a specific spot. Validates fit,
/// overlap, and rotation legality. Does NOT attempt to merge with
/// existing stacks — use [`grant_or_merge`] for the pickup path.
/// Returns the new item's index in `grid.items`.
pub fn place_at(
    grid: &mut GridInventory,
    registry: &ItemRegistry,
    stack: ItemInstance,
    x: u32,
    y: u32,
    rotation: ItemRotation,
) -> Result<usize, PlacementError> {
    let def = registry
        .get(&stack.id)
        .ok_or_else(|| PlacementError::UnknownItem(stack.id.clone()))?;
    if rotation == ItemRotation::Deg90 && !def.rotatable {
        return Err(PlacementError::NotRotatable);
    }
    let size = footprint(def, rotation);
    fits_at_with_registry(grid, registry, x, y, size.w, size.h, None)?;
    // If the item is a container, initialize a fresh inner grid of
    // the declared dimensions. Items you're placing for the first
    // time (pickup / craft output / salvage output) start empty;
    // the equip path uses `place_at_with_inner` to preserve a
    // pre-existing nested grid on unequip.
    let inner_grid = def
        .inner_grid
        .map(|s| GridInventory::new(s.w.max(1), s.h.max(1)));
    grid.items.push(PlacedItem {
        stack,
        x,
        y,
        rotation,
        inner_grid,
    });
    Ok(grid.items.len() - 1)
}

/// Place an item + an already-populated `inner_grid`, e.g. when
/// unequipping a loaded backpack back into pockets — the nested grid
/// travels with the item rather than being reset.
pub fn place_at_with_inner(
    grid: &mut GridInventory,
    registry: &ItemRegistry,
    stack: ItemInstance,
    inner_grid: Option<GridInventory>,
    x: u32,
    y: u32,
    rotation: ItemRotation,
) -> Result<usize, PlacementError> {
    let def = registry
        .get(&stack.id)
        .ok_or_else(|| PlacementError::UnknownItem(stack.id.clone()))?;
    if rotation == ItemRotation::Deg90 && !def.rotatable {
        return Err(PlacementError::NotRotatable);
    }
    let size = footprint(def, rotation);
    fits_at_with_registry(grid, registry, x, y, size.w, size.h, None)?;
    grid.items.push(PlacedItem {
        stack,
        x,
        y,
        rotation,
        inner_grid,
    });
    Ok(grid.items.len() - 1)
}

/// Remove the item at `idx` and return it.
pub fn remove(grid: &mut GridInventory, idx: usize) -> Result<PlacedItem, PlacementError> {
    if idx >= grid.items.len() {
        return Err(PlacementError::BadIndex(idx));
    }
    Ok(grid.items.remove(idx))
}

/// Move an item to a new position / rotation within the same grid.
/// Validates the destination ignoring the item itself.
pub fn move_within(
    grid: &mut GridInventory,
    registry: &ItemRegistry,
    idx: usize,
    new_x: u32,
    new_y: u32,
    new_rotation: ItemRotation,
) -> Result<(), PlacementError> {
    if idx >= grid.items.len() {
        return Err(PlacementError::BadIndex(idx));
    }
    let stack_id = grid.items[idx].stack.id.clone();
    let def = registry
        .get(&stack_id)
        .ok_or(PlacementError::UnknownItem(stack_id))?;
    if new_rotation == ItemRotation::Deg90 && !def.rotatable {
        return Err(PlacementError::NotRotatable);
    }
    let size = footprint(def, new_rotation);
    fits_at_with_registry(grid, registry, new_x, new_y, size.w, size.h, Some(idx))?;
    let it = &mut grid.items[idx];
    it.x = new_x;
    it.y = new_y;
    it.rotation = new_rotation;
    Ok(())
}

/// Add `count` of `id` to the grid using the merge-then-place
/// strategy:
///
/// 1. Walk existing placed stacks. For each that matches `(id,
///    spawned_tick)` (or any spawned_tick if non-perishable), pour
///    units in until the stack hits `def.stack_size`.
/// 2. While units remain, allocate new positions via
///    [`find_first_fit_any_rotation`]. Each new placement is also
///    capped at `def.stack_size`.
/// 3. If room runs out before all units land, return
///    [`PlaceOutcome::PartialOrFull`] with `remaining > 0`.
///
/// This is the entry point used by the pickup / craft-output /
/// salvage-output / refund paths — same shape as the legacy
/// `merge_item_stack` but grid-aware.
pub fn grant_or_merge(
    grid: &mut GridInventory,
    registry: &ItemRegistry,
    id: &ItemId,
    count: u32,
    spawned_tick: u64,
) -> Result<PlaceOutcome, PlacementError> {
    if count == 0 {
        return Ok(PlaceOutcome::Placed {
            touched_indices: Vec::new(),
        });
    }
    let def = registry
        .get(id)
        .ok_or_else(|| PlacementError::UnknownItem(id.clone()))?
        .clone();
    let stack_size = def.stack_size.max(1);
    let is_perishable = def.perishable_ticks.is_some();
    let mut remaining = count;
    let mut touched: Vec<usize> = Vec::new();

    // 1. Merge into matching existing stacks.
    for (i, p) in grid.items.iter_mut().enumerate() {
        if remaining == 0 {
            break;
        }
        if p.stack.id != *id {
            continue;
        }
        if is_perishable && p.stack.spawned_tick != spawned_tick {
            continue;
        }
        let space = stack_size.saturating_sub(p.stack.count);
        if space == 0 {
            continue;
        }
        let take = remaining.min(space);
        p.stack.count = p.stack.count.saturating_add(take);
        remaining = remaining.saturating_sub(take);
        touched.push(i);
    }

    // 2. Allocate fresh placements for whatever's left.
    while remaining > 0 {
        let Some((x, y, rotation)) = find_first_fit_any_rotation(grid, registry, &def) else {
            break;
        };
        let take = remaining.min(stack_size);
        let new_idx = place_at(
            grid,
            registry,
            ItemInstance {
                id: id.clone(),
                count: take,
                spawned_tick,
                magazine_state: None,
            },
            x,
            y,
            rotation,
        )?;
        touched.push(new_idx);
        remaining = remaining.saturating_sub(take);
    }

    if remaining > 0 {
        Ok(PlaceOutcome::PartialOrFull {
            placed: count - remaining,
            remaining,
            touched_indices: touched,
        })
    } else {
        Ok(PlaceOutcome::Placed {
            touched_indices: touched,
        })
    }
}

/// Subtract `count` of `id` from the grid, FIFO across stacks. Drops
/// any stack whose count hits zero. Returns the actual count
/// consumed (may be less than `count` if the grid didn't have enough).
pub fn consume_from_grid(grid: &mut GridInventory, id: &ItemId, count: u32) -> u32 {
    let mut remaining = count;
    let mut consumed = 0u32;
    let mut idx = 0;
    while idx < grid.items.len() && remaining > 0 {
        if grid.items[idx].stack.id != *id {
            idx += 1;
            continue;
        }
        let take = remaining.min(grid.items[idx].stack.count);
        grid.items[idx].stack.count -= take;
        remaining -= take;
        consumed += take;
        if grid.items[idx].stack.count == 0 {
            grid.items.remove(idx);
        } else {
            idx += 1;
        }
    }
    consumed
}

/// Total number of `id` units present across all stacks in the grid.
/// Used by `can_craft` / pre-flight checks.
pub fn count_of(grid: &GridInventory, id: &ItemId) -> u32 {
    grid.items
        .iter()
        .filter(|p| &p.stack.id == id)
        .map(|p| p.stack.count)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::items::{GridSize, ItemCategory};

    fn def(id: &str, w: u32, h: u32, rotatable: bool, stack_size: u32) -> ItemDef {
        ItemDef {
            id: ItemId::from(id),
            name: id.to_string(),
            category: ItemCategory::Misc,
            weight: 0.0,
            stack_size,
            perishable_ticks: None,
            consume_action: None,
            salvage: None,
            tool: None,
            size: GridSize { w, h },
            rotatable,
            inner_grid: None,
            equip_slots: Vec::new(),
            weapon_config: None,
            magazine_config: None,
            ammo_config: None,
            armor_config: None,
            attachment_config: None,
        }
    }

    fn registry_with(defs: Vec<ItemDef>) -> ItemRegistry {
        let mut r = ItemRegistry::empty_for_test();
        for d in defs {
            r.insert_for_test(d);
        }
        r
    }

    fn stack(id: &str, count: u32) -> ItemInstance {
        ItemInstance {
            id: ItemId::from(id),
            count,
            spawned_tick: 0,
            magazine_state: None,
        }
    }

    #[test]
    fn place_at_succeeds_in_bounds_no_overlap() {
        let mut g = GridInventory::new(4, 4);
        let r = registry_with(vec![def("bandage", 1, 1, false, 20)]);
        let idx = place_at(&mut g, &r, stack("bandage", 5), 0, 0, ItemRotation::Deg0).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(g.items.len(), 1);
    }

    #[test]
    fn place_at_rejects_out_of_bounds() {
        let mut g = GridInventory::new(4, 4);
        let r = registry_with(vec![def("rifle", 1, 4, false, 1)]);
        // 1×4 placed at y=2 → would extend to y=6, past height 4.
        let err = place_at(&mut g, &r, stack("rifle", 1), 0, 2, ItemRotation::Deg0).unwrap_err();
        assert!(matches!(err, PlacementError::OutOfBounds { .. }));
    }

    #[test]
    fn place_at_rejects_overlap() {
        let mut g = GridInventory::new(4, 4);
        let r = registry_with(vec![def("box_2x2", 2, 2, false, 1)]);
        place_at(&mut g, &r, stack("box_2x2", 1), 0, 0, ItemRotation::Deg0).unwrap();
        let err = place_at(&mut g, &r, stack("box_2x2", 1), 1, 1, ItemRotation::Deg0).unwrap_err();
        assert!(matches!(err, PlacementError::Overlap(0)));
    }

    #[test]
    fn rotation_swaps_footprint() {
        let mut g = GridInventory::new(4, 1);
        let r = registry_with(vec![def("rifle", 1, 4, true, 1)]);
        // 1×4 doesn't fit in 4×1, but rotated to 4×1 it does.
        let err = place_at(&mut g, &r, stack("rifle", 1), 0, 0, ItemRotation::Deg0).unwrap_err();
        assert!(matches!(err, PlacementError::OutOfBounds { .. }));
        let idx = place_at(&mut g, &r, stack("rifle", 1), 0, 0, ItemRotation::Deg90).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn rotation_rejected_when_item_not_rotatable() {
        let mut g = GridInventory::new(4, 1);
        let r = registry_with(vec![def("brick", 1, 4, false, 1)]);
        let err = place_at(&mut g, &r, stack("brick", 1), 0, 0, ItemRotation::Deg90).unwrap_err();
        assert!(matches!(err, PlacementError::NotRotatable));
    }

    #[test]
    fn find_first_fit_walks_top_left_to_bottom_right() {
        let mut g = GridInventory::new(4, 4);
        let r = registry_with(vec![def("box", 2, 2, false, 1)]);
        place_at(&mut g, &r, stack("box", 1), 0, 0, ItemRotation::Deg0).unwrap();
        // Next 2×2 should land at (2, 0), not (0, 2), per top-left scan.
        let pos = find_first_fit(&g, &r, 2, 2);
        assert_eq!(pos, Some((2, 0)));
    }

    #[test]
    fn grant_or_merge_into_existing_stack() {
        let mut g = GridInventory::new(4, 4);
        let r = registry_with(vec![def("bandage", 1, 1, false, 20)]);
        place_at(&mut g, &r, stack("bandage", 5), 0, 0, ItemRotation::Deg0).unwrap();
        let outcome = grant_or_merge(&mut g, &r, &ItemId::from("bandage"), 10, 0).unwrap();
        assert!(matches!(outcome, PlaceOutcome::Placed { .. }));
        assert_eq!(g.items.len(), 1);
        assert_eq!(g.items[0].stack.count, 15);
    }

    #[test]
    fn grant_or_merge_overflows_into_new_slot() {
        let mut g = GridInventory::new(4, 4);
        let r = registry_with(vec![def("bandage", 1, 1, false, 20)]);
        place_at(&mut g, &r, stack("bandage", 18), 0, 0, ItemRotation::Deg0).unwrap();
        let outcome = grant_or_merge(&mut g, &r, &ItemId::from("bandage"), 5, 0).unwrap();
        assert!(matches!(outcome, PlaceOutcome::Placed { .. }));
        assert_eq!(g.items.len(), 2);
        assert_eq!(g.items[0].stack.count, 20);
        assert_eq!(g.items[1].stack.count, 3);
    }

    #[test]
    fn grant_or_merge_partial_when_grid_full() {
        let mut g = GridInventory::new(2, 2);
        let r = registry_with(vec![def("box", 2, 2, false, 1)]);
        place_at(&mut g, &r, stack("box", 1), 0, 0, ItemRotation::Deg0).unwrap();
        // Try to add another 2×2 box; no room.
        let outcome = grant_or_merge(&mut g, &r, &ItemId::from("box"), 1, 0).unwrap();
        match outcome {
            PlaceOutcome::PartialOrFull {
                placed, remaining, ..
            } => {
                assert_eq!(placed, 0);
                assert_eq!(remaining, 1);
            }
            _ => panic!("expected partial outcome, got {:?}", outcome),
        }
    }

    #[test]
    fn perishable_stacks_dont_merge_across_ticks() {
        let mut g = GridInventory::new(4, 4);
        let mut d = def("raw_meat", 1, 1, false, 10);
        d.perishable_ticks = Some(100);
        let r = registry_with(vec![d]);
        let s_old = ItemInstance {
            id: ItemId::from("raw_meat"),
            count: 3,
            spawned_tick: 0,
            magazine_state: None,
        };
        place_at(&mut g, &r, s_old, 0, 0, ItemRotation::Deg0).unwrap();
        // New batch at tick 50 — should NOT merge into the t=0 stack.
        let outcome = grant_or_merge(&mut g, &r, &ItemId::from("raw_meat"), 4, 50).unwrap();
        assert!(matches!(outcome, PlaceOutcome::Placed { .. }));
        assert_eq!(g.items.len(), 2);
        assert_eq!(g.items[0].stack.count, 3);
        assert_eq!(g.items[1].stack.count, 4);
        assert_eq!(g.items[1].stack.spawned_tick, 50);
    }

    #[test]
    fn consume_from_grid_drains_fifo_and_drops_empty() {
        let mut g = GridInventory::new(4, 4);
        let r = registry_with(vec![def("bandage", 1, 1, false, 20)]);
        place_at(&mut g, &r, stack("bandage", 4), 0, 0, ItemRotation::Deg0).unwrap();
        place_at(&mut g, &r, stack("bandage", 6), 1, 0, ItemRotation::Deg0).unwrap();
        let took = consume_from_grid(&mut g, &ItemId::from("bandage"), 7);
        assert_eq!(took, 7);
        assert_eq!(g.items.len(), 1);
        assert_eq!(g.items[0].stack.count, 3);
    }

    #[test]
    fn move_within_validates_new_spot() {
        let mut g = GridInventory::new(4, 4);
        let r = registry_with(vec![def("box", 2, 2, false, 1)]);
        place_at(&mut g, &r, stack("box", 1), 0, 0, ItemRotation::Deg0).unwrap();
        place_at(&mut g, &r, stack("box", 1), 2, 0, ItemRotation::Deg0).unwrap();
        // Move first box to (1, 1) — would overlap second.
        let err = move_within(&mut g, &r, 0, 1, 1, ItemRotation::Deg0).unwrap_err();
        assert!(matches!(err, PlacementError::Overlap(1)));
        // Move first box to (0, 2) — fits.
        move_within(&mut g, &r, 0, 0, 2, ItemRotation::Deg0).unwrap();
        assert_eq!(g.items[0].x, 0);
        assert_eq!(g.items[0].y, 2);
    }
}
