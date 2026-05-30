//! Player-initiated actions that flow from client → host over the
//! network and get dispatched into the host's authoritative sim.
//!
//! On clients, the `SimHost` Godot wrapper catches each mutating
//! `#[func]` call and, when the local role is `Client`, packages the
//! equivalent [`ActionKind`] into a `Msg::Action` instead of mutating
//! locally. The host receives the message, decodes the action, and
//! calls [`crate::Sim::apply_action`] — which dispatches to the same
//! mutation methods the host itself would call directly.
//!
//! The host's existing mutation methods emit `WorldDelta`s into the
//! per-tick buffer, which `SimHost` broadcasts to every peer (including
//! the action's originator). So from the client's point of view, the
//! flow is:
//!
//! 1. Client calls `sim_host.apply_bandage(sid, "torso")`.
//! 2. Client's `apply_bandage` wrapper sees Client role → encodes an
//!    `ActionKind::ApplyBandage { part: Torso }` into `Msg::Action`
//!    → sends to host.
//! 3. Host decodes → `sim.apply_action(sid, ActionKind::ApplyBandage { ... })`.
//! 4. Host's sim mutates, journals a `WoundTreatmentChanged` delta.
//! 5. Tick end: host drains the delta → broadcasts.
//! 6. Client's mirror applies the `WoundTreatmentChanged` delta → ECS
//!    reflects the bandage.
//!
//! **Validation is deferred to slice 3.** Today the host trusts any
//! `ActionKind` from any peer as legitimate. Before 12-player +
//! production traffic, we need invariants (only the right `steam_id`
//! can act on itself, rate limits, plausible input ranges, etc.).

use serde::{Deserialize, Serialize};

use crate::components::{BodyPart, DrugKind, FoodKind, WaterKind};

/// One discrete player action. Variants mirror the mutating `Sim`
/// methods; each routes to its matching API in
/// [`crate::Sim::apply_action`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionKind {
    /// Move the player. High-frequency (20Hz from client).
    Move {
        pos: [f32; 3],
        yaw: f32,
    },
    /// Cross a region boundary. Low-frequency.
    ChangeRegion {
        region_name: String,
    },
    /// Treatment actions. All take a `BodyPart`.
    ApplyBandage {
        part: BodyPart,
    },
    ApplyTourniquet {
        part: BodyPart,
    },
    RemoveTourniquet {
        part: BodyPart,
    },
    ApplyDisinfectant {
        part: BodyPart,
    },
    ApplyStitch {
        part: BodyPart,
    },
    ApplyWoundPack {
        part: BodyPart,
    },
    ApplyAntibiotics,
    /// Drug + food + drink — parts of the meds stack.
    ApplyDrug {
        drug: DrugKind,
    },
    Eat {
        kind: FoodKind,
    },
    Drink {
        kind: WaterKind,
    },
    /// Inventory verbs.
    ConsumeSlot {
        slot_idx: u32,
        body_part: Option<BodyPart>,
    },
    DropSlot {
        slot_idx: u32,
    },
    MoveSlot {
        from: u32,
        to: u32,
    },
    /// Move the item at `(from_grid, from_idx)` into `to_grid` at the
    /// first free spot. `from_grid` / `to_grid` are `"pockets"` or
    /// `"equipped:<slot_id>"` (same convention as `Equip.source_grid`).
    /// Rejected when `from_grid == to_grid` (use `MoveSlot` for in-grid
    /// swaps), source is empty, or dest can't fit. The item's nested
    /// `inner_grid` (loaded backpack, mag with rounds) travels with it.
    MoveBetweenGrids {
        from_grid: String,
        from_idx: u32,
        to_grid: String,
    },
    SalvageSlot {
        slot_idx: u32,
    },
    CraftRecipe {
        recipe_id: String,
    },
    SetNearCampfire {
        value: bool,
    },
    /// Step 5: workbench proximity flag. `tier = None` clears it.
    SetNearWorkbench {
        tier: Option<crate::items::ToolTier>,
    },
    /// Step 5: queue N units of `recipe_id` on the player's crafting
    /// queue. Materials lock up front (host-validated).
    QueueCraft {
        recipe_id: String,
        count: u32,
    },
    /// Step 5: cancel a previously queued craft job.
    CancelCraft {
        job_id: u32,
    },
    /// PR-2 equipment: move an item from `source_grid` at `source_idx`
    /// into the paper-doll slot. `source_grid` is `"pockets"` or
    /// `"equipped:<slot_id>"`.
    Equip {
        slot_id: String,
        source_grid: String,
        source_idx: u32,
    },
    /// PR-2 equipment: pull the item at `slot_id` off the paper doll
    /// into `dest_grid` (same string form as `Equip.source_grid`).
    Unequip {
        slot_id: String,
        dest_grid: String,
    },
    /// PR-2 hotbar: fire `consume_from_hotbar` for the belt slot at
    /// `idx` (1-based).
    HotbarConsume {
        idx: u8,
        body_part: Option<BodyPart>,
    },
    /// Weapons phase 1: reload the weapon in `slot_id`. Host pulls a
    /// matching-caliber magazine from the player's pockets grid
    /// (preferring the most-loaded one if multiple are available),
    /// installs it on the equipped weapon, and ejects any prior
    /// magazine back to inventory with its `loaded_rounds`
    /// preserved.
    ReloadWeapon {
        slot_id: String,
    },
    /// Weapons phase 1: eject the current magazine from the weapon
    /// in `slot_id` back to the player's pockets (preserving the
    /// loaded-round count).
    EjectMagazine {
        slot_id: String,
    },
    /// Weapons phase 2: fire the weapon in `slot_id` with the
    /// shooter's current aim. The host spawns a `Projectile` entity
    /// at the player's muzzle origin with velocity derived from
    /// `(aim_yaw, aim_pitch)` × the loaded round's
    /// `muzzle_velocity_mps`; hit resolution happens in
    /// `tick_projectiles`. Dry-click (no round consumed, no
    /// projectile) if the weapon has no mag or the mag is empty.
    /// Legacy pre-phase-2 action payloads deserialize cleanly
    /// (yaw + pitch default to 0 = facing +Z).
    FireWeapon {
        slot_id: String,
        #[serde(default)]
        aim_yaw: f32,
        #[serde(default)]
        aim_pitch: f32,
    },
    /// Weapons phase 2: load pocket ammo into the mag loaded at
    /// `slot_id`. `round_id` names an `AmmoConfig` item; load
    /// tops the mag up to its capacity, consuming from matching
    /// pocket stacks. Rejected on caliber mismatch or partial-mag
    /// variant flip.
    LoadRoundsIntoMag {
        slot_id: String,
        round_id: String,
    },
    /// Weapons phase 2: load pocket ammo into a magazine at
    /// `pocket_idx` in the player's pockets grid. Same validation
    /// as `LoadRoundsIntoMag`, but targets a pre-reload mag so
    /// players can prep spares through the inventory UI.
    LoadRoundsIntoPocketMag {
        pocket_idx: u32,
        round_id: String,
    },
    /// Debug / dev only. Slice 1 keeps this because the debug overlay
    /// cycles test items via `G`; production slice 3 strips it.
    GrantItem {
        item_id: String,
        count: u32,
    },
    /// PR-4c looting: pull the item at `source_idx` out of
    /// `WorldContainer(container_id)` into the player's pockets.
    TakeFromContainer {
        container_id: u32,
        source_idx: u32,
    },
    /// PR-4c looting: push the item at `(source_grid, source_idx)`
    /// from the player into `WorldContainer(container_id)`.
    /// `source_grid` matches the equip API: `"pockets"` or
    /// `"equipped:<slot_id>"`.
    PutInContainer {
        container_id: u32,
        source_grid: String,
        source_idx: u32,
    },
}

/// Serialize an [`ActionKind`] to the opaque byte blob that rides in
/// `Msg::Action`.
pub fn encode_action(action: &ActionKind) -> anyhow::Result<Vec<u8>> {
    Ok(bincode::serialize(action)?)
}

/// Inverse of [`encode_action`]. Returns `None` on decode failure
/// (forward-compat: unknown variants are dropped rather than erroring).
pub fn decode_action(bytes: &[u8]) -> Option<ActionKind> {
    bincode::deserialize(bytes).ok()
}
