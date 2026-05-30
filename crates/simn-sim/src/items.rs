//! Item definitions + recipe definitions + their registry resources.
//!
//! Items and recipes are **data** — loaded once from TOML (bundled
//! via `include_str!`) at `Sim::new` / `Sim::load` and exposed via
//! read-only registry resources. They're not snapshotted: on load,
//! the same TOML produces the same registry deterministically, so
//! there's nothing to save.
//!
//! `ConsumeAction` is the bridge from a TOML item to the existing
//! (already-landed) `Sim` API: each variant names an existing sim
//! method. `Sim::consume_from_slot` reads the item's action and
//! routes to `eat` / `drink` / `apply_drug` / `apply_bandage` /
//! etc. — no behavior duplication.
//!
//! Spec reference: `docs/survival-and-crafting-plan.md` §1 "every
//! mechanic is data-driven" and §4–§7 for the item / recipe shape.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::components::{DrugKind, FoodKind, WaterKind};

/// Stable string identifier for an item. The value matches the `id`
/// key in `content/items.toml`; references from other items (salvage
/// outputs, recipe inputs, required tools) are by this id.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ItemId(pub String);

impl From<&str> for ItemId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ItemId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Top-level item record. Loaded from TOML; fields correspond
/// directly to the `[[items]]` block keys.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ItemDef {
    pub id: ItemId,
    pub name: String,
    pub category: ItemCategory,
    #[serde(default)]
    pub weight: f32,
    #[serde(default = "default_stack_size")]
    pub stack_size: u32,
    #[serde(default)]
    pub perishable_ticks: Option<u64>,
    #[serde(default)]
    pub consume_action: Option<ConsumeAction>,
    #[serde(default)]
    pub salvage: Option<SalvageRecipe>,
    /// If set, this item counts as a **toolkit / specialty kit** for
    /// recipe [`KitRequirement`] checks — e.g. `{ specialty = "gunsmith",
    /// tier = "advanced" }` on an Advanced Gunsmith Kit. Higher tiers
    /// cover everything lower tiers cover within the same specialty
    /// (a single `expert` kit satisfies `basic` / `advanced` / `expert`
    /// requirements). Stacking two different specialties requires two
    /// items; the GAMMA-style progression of
    /// general → gunsmith / armor / weapon / drug is expressed in
    /// [`Specialty`].
    #[serde(default)]
    pub tool: Option<ToolSpec>,
    /// Tarkov/STALKER-style grid footprint: how many cells wide × tall
    /// this item occupies in a [`crate::components::GridInventory`].
    /// Defaults to 1×1 (one cell, no rotation needed). Set explicitly
    /// for items larger than a single cell — e.g. an AK rifle is
    /// `{ w = 1, h = 4 }` (one cell wide, four cells tall when held
    /// vertically).
    #[serde(default = "default_grid_size")]
    pub size: GridSize,
    /// True if the item can be rotated 90° in the grid (so a `1×4`
    /// item also fits a `4×1` slot). Defaults to false — symmetric
    /// items don't gain anything from rotation, and small items
    /// (1×1, 2×2) where rotation is meaningless can stay flagged
    /// false to avoid pointless UI affordances.
    #[serde(default)]
    pub rotatable: bool,
    /// If `Some`, this item is itself a container — equipping it (or
    /// dropping it on the ground / opening a corpse) exposes a
    /// nested [`crate::components::GridInventory`] of these
    /// dimensions. Backpacks, rigs, weapon cases, medical pouches.
    #[serde(default)]
    pub inner_grid: Option<GridSize>,
    /// Equipment-slot whitelist — which slot ids (from
    /// `equipment_slots.toml`) this item fits in. Empty = not
    /// equippable. When set, it overrides the slot's
    /// [`EquipmentSlotDef::accepts`] category match, letting the
    /// catalog pin specific items to specific slots without
    /// widening a category. Typical use: a two-handed rifle only
    /// fits `primary`, not `sidearm`.
    #[serde(default)]
    pub equip_slots: Vec<SlotId>,
    /// Weapons-phase-1 config block. Present only on weapon items
    /// (category = `WeaponPrimary` / `WeaponSecondary` / `Sidearm`
    /// / `Melee`). All numeric tuning (damage, range, fire rate,
    /// spread) lives here — never hardcoded in engine code.
    /// Phase 2 extends this with ballistic fields (muzzle velocity,
    /// drag coefficient).
    #[serde(default)]
    pub weapon_config: Option<WeaponConfig>,
    /// Weapons-phase-1 config block for magazine items (category =
    /// `Magazine`). Caliber gates which weapons this mag fits;
    /// capacity gates how many rounds reload pulls from inventory.
    #[serde(default)]
    pub magazine_config: Option<MagazineConfig>,
    /// Config block for ammo items (category = `Ammo`). Caliber
    /// gates which magazines this round loads into. Phase 2 adds
    /// ballistic + penetration fields so different variants of the
    /// same caliber (HP / FMJ / AP) behave distinctly under the
    /// penetration-vs-armor formula.
    #[serde(default)]
    pub ammo_config: Option<AmmoConfig>,
    /// Weapons-phase-2 config block for wearable armor (category =
    /// `ArmorVest` / `HeadGear`). `protection_class` gates which
    /// round variants can penetrate; `coverage` lists which body
    /// parts this armor protects. The sim scans the wearer's
    /// `Equipment` for all armor items covering the hit part and
    /// takes the maximum protection class.
    #[serde(default)]
    pub armor_config: Option<ArmorConfig>,
    /// Phase 4C: attachment config block. Only set on items that
    /// are *themselves* attachments (scopes, rails, suppressors,
    /// adapters). Declares which mount surface tag this attachment
    /// consumes and what new tag(s) it provides downstream.
    /// Validated by `validate_attachment_chain` at attach time.
    /// See `docs/book/src/planning/weapons-plan.md` §3.
    #[serde(default)]
    pub attachment_config: Option<AttachmentConfig>,
}

/// Weapon tuning parameters. All values read from `items.toml` —
/// engine code never supplies defaults that would mask missing
/// config. A weapon without a `weapon_config` block simply can't
/// fire (the fire path returns an error that surfaces to the
/// player as a dry click).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct WeaponConfig {
    /// Caliber tag — matches the `caliber` field on
    /// [`MagazineConfig`] to gate what magazine fits. Arbitrary
    /// string (`"9x18"`, `"5.45x39"`, `"12ga"`) so modders can
    /// declare new calibers without touching engine code.
    pub caliber: Caliber,
    /// HP per hit, passed to `Sim::apply_damage_to_npc_part` on
    /// successful shot.
    pub damage: f32,
    /// Hard raycast cutoff (m). Bullets beyond this distance miss.
    pub range_m: f32,
    /// Seconds between successive shots. LMB is gated on this
    /// cooldown.
    pub fire_interval_s: f32,
    /// Half-angle (degrees) of the uniform random spread cone
    /// applied to the fired direction. 0 = perfectly on-axis.
    pub spread_deg: f32,
    /// Phase 4C: attachment slots on this weapon. Each slot has
    /// an id (`muzzle`, `optic_rail`, `magwell`, ...) and a list
    /// of mount-surface tags that fit it (`threaded_14x1_lh`,
    /// `dovetail_side`, `picatinny`, ...). Attachments declare
    /// which tag they `consumes_tag`; when an attachment's
    /// consumed tag matches any tag in any of the weapon's
    /// initial slot tag-lists *or* any tag provided by an
    /// already-attached attachment, the attach is valid. See
    /// `validate_attachment_chain` for the resolver.
    ///
    /// Defaults to empty for weapons that don't yet author
    /// attachment data (most of the GAMMA roster pre-4C). Empty
    /// slots = no attachments allowed.
    #[serde(default)]
    pub slots: Vec<WeaponSlot>,
    /// Phase 4D: condition lost per shot fired (in 0..100
    /// condition units). Average GAMMA-era authoring: 0.05 per
    /// shot for milspec rifles, 0.12 for cheap pistols, 0.20+
    /// for clapped-out hand-loads. With v1's single aggregate
    /// condition, this is the whole wear model — per-part
    /// rates (§5.2 of `weapons-plan.md`) land in a later
    /// iteration. Defaults to 0.05 so legacy weapons without
    /// authoring still see *some* wear and the jam path
    /// exercises in tests; mod data overrides per row.
    #[serde(default = "default_wear_per_shot")]
    pub wear_per_shot: f32,
    /// Phase 4D: condition value (in 0..100) at which jam
    /// probability begins ramping above zero. Above this,
    /// `jam_chance_at_condition` returns 0; below this, the
    /// curve rises linearly to `jam_chance_floor` at
    /// `condition == 0`. Tuned at 70 for a Kalashnikov-class
    /// reliability profile (jam-free above 70 % condition).
    /// Modders tighten or loosen this per weapon archetype.
    #[serde(default = "default_jam_threshold")]
    pub jam_threshold: f32,
    /// Phase 4D: peak jam probability per shot at
    /// `condition == 0` (0..1). The curve is linear from 0 at
    /// `jam_threshold` to this value at 0 condition. Default
    /// 0.18 — at fully clapped-out (cond=0) an AK still fires
    /// 4/5 shots without jamming, in line with the GAMMA
    /// jam-economy feel. Mods may push it higher for fragile
    /// archetypes (matchlock, prototype designs).
    #[serde(default = "default_jam_chance_floor")]
    pub jam_chance_floor: f32,
}

fn default_wear_per_shot() -> f32 {
    0.05
}

fn default_jam_threshold() -> f32 {
    70.0
}

fn default_jam_chance_floor() -> f32 {
    0.18
}

/// Phase 4D: jam-probability curve.
///
/// Linear from 0 at `jam_threshold` to `jam_chance_floor` at 0.
/// Above the threshold the function returns 0 (no jam ever).
/// Clamped to `[0, jam_chance_floor]` so out-of-range conditions
/// can't produce silly probabilities.
pub fn jam_chance_at_condition(condition: f32, cfg: &WeaponConfig) -> f32 {
    if condition >= cfg.jam_threshold {
        return 0.0;
    }
    if condition <= 0.0 {
        return cfg.jam_chance_floor;
    }
    let t = (cfg.jam_threshold - condition) / cfg.jam_threshold.max(f32::EPSILON);
    (cfg.jam_chance_floor * t).clamp(0.0, cfg.jam_chance_floor)
}

/// Phase 4C: a single attachment slot on a weapon. The slot's
/// `tags` are the mount-surface fingerprints that the weapon
/// exposes natively (no attachments installed). Slot ids are
/// authoring-side (`muzzle`, `optic_rail`, `handguard`, ...)
/// and don't need to be unique across weapons; the tag list is
/// what the resolver matches against.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct WeaponSlot {
    /// Stable slot id (`muzzle`, `optic_rail`, `magwell`, ...).
    /// Authoring hint + diagnostic only — the resolver matches
    /// on tags, not slot ids.
    pub id: SlotId,
    /// Mount-surface tags this slot exposes natively. See
    /// `weapons-plan.md` §3.3 for the canonical vocabulary
    /// (mounting surfaces, muzzle threads, stock interfaces,
    /// magazine wells).
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Phase 4C: per-item attachment config block. Items with this
/// block set are *attachments* — they consume one mount-surface
/// tag and optionally provide one or more new tags downstream.
/// See `weapons-plan.md` §3.1 for the model and §3.3 for the
/// starter tag vocabulary.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct AttachmentConfig {
    /// Mount-surface tag this attachment requires from the
    /// weapon (or a previously-installed attachment).
    pub consumes_tag: String,
    /// New tag(s) this attachment exposes downstream. A
    /// dovetail→picatinny adapter consumes `dovetail_side` and
    /// provides `[picatinny]`. A scope consumes `picatinny`
    /// (or whatever) and provides nothing.
    #[serde(default)]
    pub provides_tags: Vec<String>,
    /// Phase 4C is data-only: effect strings are authored for
    /// future stat aggregation (§3.4 of `weapons-plan.md`) but
    /// not yet consumed by the sim. Effect keys are free-form
    /// strings (`recoil_control`, `barrel_wear_mult`, ...);
    /// values are signed floats. Future work plumbs these
    /// through `EquippedWeaponState`.
    #[serde(default)]
    pub effects: std::collections::HashMap<String, f32>,
}

/// Phase 4C: reasons an attempted attachment is invalid.
///
/// Returned by [`validate_attachment_chain`] so callers can
/// surface a specific error to the UI / log without re-walking
/// the chain. Variants intentionally name the *first* failing
/// attachment so a long invalid chain produces one clear error,
/// not a torrent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentError {
    /// Item id wasn't found in the registry.
    UnknownItem(ItemId),
    /// Item exists but isn't an attachment (missing
    /// `attachment_config`).
    NotAnAttachment(ItemId),
    /// Item exists but isn't a weapon (missing `weapon_config`).
    NotAWeapon(ItemId),
    /// The attachment's `consumes_tag` isn't exposed by the
    /// weapon's slots or any earlier attachment's
    /// `provides_tags`. The chain is invalid at this attachment.
    NoMatchingSlot {
        attachment: ItemId,
        needed_tag: String,
    },
    /// Two attachments both consume the same tag. The second
    /// one would conflict — `attached` is the earlier one that
    /// already claimed the tag.
    TagAlreadyConsumed {
        attachment: ItemId,
        attached: ItemId,
        tag: String,
    },
}

/// Phase 4C: walk an attachment chain and validate every step.
///
/// Algorithm: start with the weapon's slot tags as the available
/// tag pool. For each attachment in order:
///
/// 1. Resolve the attachment's `attachment_config`. Reject if
///    missing.
/// 2. Check `consumes_tag` is in the pool. Reject if not.
/// 3. Remove the consumed tag (a slot can only host one
///    attachment).
/// 4. Add the attachment's `provides_tags` to the pool.
///
/// Returns the final tag pool on success (useful for "what can
/// still be added" UI / loot rolls). Returns the first error
/// encountered on failure — callers don't get a list of every
/// problem, just the one that blocked the chain.
///
/// The same tag can appear on multiple slots or in multiple
/// `provides_tags`. The resolver removes only one occurrence
/// per consume — so a weapon with two `picatinny` slots can
/// host two `picatinny`-consuming attachments.
pub fn validate_attachment_chain(
    registry: &ItemRegistry,
    weapon_id: &ItemId,
    attachments: &[ItemId],
) -> Result<Vec<String>, AttachmentError> {
    let weapon = registry
        .get(weapon_id)
        .ok_or_else(|| AttachmentError::UnknownItem(weapon_id.clone()))?;
    let weapon_config = weapon
        .weapon_config
        .as_ref()
        .ok_or_else(|| AttachmentError::NotAWeapon(weapon_id.clone()))?;

    // Initial tag pool = every tag from every slot, flattened.
    let mut pool: Vec<String> = weapon_config
        .slots
        .iter()
        .flat_map(|s| s.tags.iter().cloned())
        .collect();
    // Track which attachment claimed each tag so a duplicate
    // consume can name the earlier claimant.
    let mut claimed_by: std::collections::HashMap<String, ItemId> =
        std::collections::HashMap::new();

    for attach_id in attachments {
        let def = registry
            .get(attach_id)
            .ok_or_else(|| AttachmentError::UnknownItem(attach_id.clone()))?;
        let cfg = def
            .attachment_config
            .as_ref()
            .ok_or_else(|| AttachmentError::NotAnAttachment(attach_id.clone()))?;

        let pos = match pool.iter().position(|t| t == &cfg.consumes_tag) {
            Some(p) => p,
            None => {
                if let Some(earlier) = claimed_by.get(&cfg.consumes_tag) {
                    return Err(AttachmentError::TagAlreadyConsumed {
                        attachment: attach_id.clone(),
                        attached: earlier.clone(),
                        tag: cfg.consumes_tag.clone(),
                    });
                }
                return Err(AttachmentError::NoMatchingSlot {
                    attachment: attach_id.clone(),
                    needed_tag: cfg.consumes_tag.clone(),
                });
            }
        };
        let consumed = pool.swap_remove(pos);
        claimed_by.insert(consumed, attach_id.clone());
        for new_tag in &cfg.provides_tags {
            pool.push(new_tag.clone());
        }
    }

    Ok(pool)
}

/// Magazine tuning parameters. See [`ItemDef::magazine_config`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MagazineConfig {
    /// Caliber tag — must match a weapon's
    /// [`WeaponConfig::caliber`] for reload to accept the mag.
    pub caliber: Caliber,
    /// Maximum rounds this magazine holds. Reload tops
    /// [`crate::components::MagazineState::loaded_rounds`] up to
    /// this value.
    pub capacity: u32,
}

/// Ammo tuning parameters. See [`ItemDef::ammo_config`].
///
/// Phase 2 ballistics: each round carries its ballistic profile
/// (mass, muzzle velocity, drag) plus its terminal behavior
/// (penetration class, soft/blunt damage, reference energy for
/// range falloff). The ballistics tick uses these directly.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AmmoConfig {
    /// Caliber tag — must match [`MagazineConfig::caliber`] for
    /// the round to load into a magazine during reload.
    pub caliber: Caliber,
    /// Projectile mass (grams). Feeds kinetic-energy calc
    /// (`E = ½ m v²`).
    pub mass_g: f32,
    /// Muzzle velocity (m/s). Projectile spawn velocity magnitude.
    pub muzzle_velocity_mps: f32,
    /// Pre-computed drag scalar `0.5 * rho * Cd * A / mass` (1/m).
    /// Used in the tick as `drag_accel = v̂ * speed² * drag_k`.
    pub drag_k: f32,
    /// Penetration class (integer). Compared against
    /// [`ArmorConfig::protection_class`] in the damage formula.
    /// `0` = hollow-point softball; `4+` = heavy armor piercing.
    pub penetration_class: u8,
    /// HP dealt on a clean (penetrating) torso hit.
    /// Body-part multipliers scale this.
    pub damage_soft: f32,
    /// HP dealt on a blocked hit — blunt trauma. Scaled down by
    /// how many classes short of penetration the round was.
    pub damage_blunt: f32,
    /// Kinetic-energy reference for range-falloff scaling.
    /// `damage *= clamp(E_impact / reference_energy_j, floor, 1.0)`.
    pub reference_energy_j: f32,
    /// Audibility / wound-resolution bucket. TOML-configurable per
    /// caliber so adding a new round to `items.toml` doesn't
    /// require Rust changes — see
    /// `crate::world_event_bus::CaliberClass`. Defaults to `pistol`
    /// when omitted (the most-conservative-radius bucket).
    #[serde(default)]
    pub caliber_class: crate::world_event_bus::CaliberClass,
    /// Phase 4B v2: variant family tag. FMJ baseline, HP for
    /// expanding soft-point, AP for armor-piercing, Tracer for
    /// visible-trace ammo, Overpressure for +P/+P+ proof-load.
    /// Damage numbers are already authored per-row in TOML (so
    /// the variant tag doesn't multiply); the field exists for:
    /// (1) **client FX** — tracer color, impact sparks, casing
    /// ejection sound profile per variant; (2) **AI hints** —
    /// future loot economy + faction loadout preferences can
    /// read this without a string-parse on the round id; (3)
    /// **player UX** — inventory icons / tooltips group by
    /// variant. Defaults to FMJ for legacy data + ammo entries
    /// that don't bother to declare.
    #[serde(default)]
    pub variant: AmmoVariant,
}

/// Phase 4B v2 ammo variant family. Each value tags the broad
/// terminal behavior + visual + sonic profile of a round.
///
/// Concrete numbers (damage, penetration, mass, muzzle velocity)
/// are still per-row in `ammo.toml`. The variant tag is for
/// client FX, AI/loot hints, and player UX only.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum AmmoVariant {
    /// Full Metal Jacket — baseline lead-core, copper-jacketed
    /// ball ammo. Default variant for any round that doesn't
    /// declare otherwise.
    #[default]
    Fmj,
    /// Hollow Point / Soft Point / JHP — expanding round. Higher
    /// damage_soft, lower penetration_class. Wound profile is
    /// "permanent cavity" rather than "icepick channel".
    Hp,
    /// Armor Piercing — hardened-core / tungsten / steel
    /// penetrator. Higher penetration_class, often slightly
    /// lower damage_soft.
    Ap,
    /// Tracer — visible incendiary trace at long range. Damage
    /// profile typically matches FMJ; the variant tag is the
    /// FX hook (client renders a brighter tracer + spec ignites
    /// dry foliage on impact, future work).
    Tracer,
    /// Overpressure / +P+ / proof-load — increased muzzle
    /// velocity and chamber pressure. Tags the round as
    /// disproportionately wearing on weapon parts (Phase 4D
    /// jam economy reads this).
    Overpressure,
}

impl AmmoVariant {
    /// Snake-case identifier for bridge / UI surfaces. Matches the
    /// TOML serde representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            AmmoVariant::Fmj => "fmj",
            AmmoVariant::Hp => "hp",
            AmmoVariant::Ap => "ap",
            AmmoVariant::Tracer => "tracer",
            AmmoVariant::Overpressure => "overpressure",
        }
    }
}

/// Armor tuning parameters. See [`ItemDef::armor_config`].
///
/// `protection_class` is the integer armor-class rating compared
/// against [`AmmoConfig::penetration_class`]. `coverage` lists
/// which body parts the armor protects — a plate carrier covers
/// `[torso]`; an exosuit may cover `[torso, head]`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ArmorConfig {
    pub protection_class: u8,
    pub coverage: Vec<crate::components::BodyPart>,
}

/// Caliber tag. Modders add new calibers by inventing new strings
/// and using them on a weapon + magazine + ammo triple in
/// `items.toml` — engine code never hardcodes caliber values.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Caliber(pub String);

impl From<&str> for Caliber {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for Caliber {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Stable identifier for an equipment slot. Matches the `id` field in
/// `equipment_slots.toml`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub String);

impl From<&str> for SlotId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for SlotId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// One paper-doll slot. Loaded from `equipment_slots.toml`.
///
/// The modularity contract: engine code never names a specific slot.
/// All slot behavior falls out of this struct's fields — `accepts`
/// gates what items can equip, `position` drives the UI layout,
/// `is_hotbar` + `hotbar_index` wire number-key quick-use to slots
/// that opt in.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EquipmentSlotDef {
    pub id: SlotId,
    /// Human-readable label for the UI layer.
    pub label: String,
    /// Category whitelist. Items whose
    /// [`ItemDef::category`] is in this list can equip to the slot
    /// — unless [`ItemDef::equip_slots`] is non-empty, in which case
    /// that whitelist overrides and the item only fits explicitly
    /// listed slots.
    #[serde(default)]
    pub accepts: Vec<ItemCategory>,
    /// Paper-doll grid position. The UI layer interprets `(x, y)` as
    /// a cell on the doll-layout grid — the sim doesn't care about
    /// the numbers, only that they're stable.
    #[serde(default)]
    pub position: SlotPosition,
    /// Paper-doll cell footprint. The UI renders the slot panel
    /// `w` cells wide and `h` cells tall — so a "primary weapon"
    /// slot at 3×1 reads as a rifle silhouette, an armor vest at
    /// 2×2 reads as a chest piece, etc. Defaults to 1×1 (single
    /// square). Sim-irrelevant; pure UI metadata, matches
    /// [`Self::position`].
    #[serde(default = "default_slot_size")]
    pub size: SlotSize,
    /// True if this slot participates in the hotbar. Belt / pouch
    /// slots set it; head / torso don't.
    #[serde(default)]
    pub is_hotbar: bool,
    /// `1..=N` hotbar index when `is_hotbar`. Ignored otherwise.
    /// Each hotbar index must be unique across the registry.
    #[serde(default)]
    pub hotbar_index: u8,
}

/// Paper-doll grid position for a slot. Pure UI metadata.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SlotPosition {
    pub x: u32,
    pub y: u32,
}

/// Paper-doll cell footprint for a slot. Pure UI metadata; the sim
/// doesn't care. Defaults to 1×1 for legacy entries that pre-date
/// the `size` field.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotSize {
    pub w: u32,
    pub h: u32,
}

fn default_slot_size() -> SlotSize {
    SlotSize { w: 1, h: 1 }
}

impl Default for SlotSize {
    fn default() -> Self {
        default_slot_size()
    }
}

/// 2D grid footprint: width × height in cells. Used both on
/// [`ItemDef::size`] (item shape) and [`ItemDef::inner_grid`]
/// (container's inner-grid shape).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridSize {
    pub w: u32,
    pub h: u32,
}

impl GridSize {
    pub const fn new(w: u32, h: u32) -> Self {
        Self { w, h }
    }

    /// Cell count covered by this footprint.
    pub fn area(&self) -> u32 {
        self.w.saturating_mul(self.h)
    }

    /// Footprint when rotated 90° — width and height swap.
    pub fn rotated(self) -> Self {
        Self {
            w: self.h,
            h: self.w,
        }
    }
}

fn default_stack_size() -> u32 {
    1
}

fn default_grid_size() -> GridSize {
    GridSize { w: 1, h: 1 }
}

/// Broad category used for filtering, UI sections, equipment-slot
/// gating, and future trader-value tables (spec §6.3). Item behavior
/// is driven by `consume_action` / `salvage` / `equip_slots`, not by
/// category — this is metadata.
///
/// Equipment categories (`HeadGear` / `Eyes` / `ArmorVest` /
/// `ChestRig` / `Backpack` / `WeaponPrimary` / `Sidearm` / `Melee`)
/// are matched against [`EquipmentSlotDef::accepts`] to decide which
/// slots accept an item — in addition to the per-item
/// [`ItemDef::equip_slots`] override. The modularity contract: a new
/// kind of gear means extending this enum *and* adding the slot-row
/// in `equipment_slots.toml`; the engine code stays item-agnostic.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ItemCategory {
    Food,
    Drink,
    Medical,
    Drug,
    Junk,
    Component,
    Tool,
    Misc,
    HeadGear,
    Eyes,
    ArmorVest,
    ChestRig,
    Backpack,
    WeaponPrimary,
    WeaponSecondary,
    Sidearm,
    Melee,
    /// Magazine for a weapon. Holds rounds in its
    /// [`crate::components::MagazineState`] (loaded_rounds count).
    /// Caliber gating via [`MagazineConfig::caliber`].
    Magazine,
    /// Loose rounds of a given caliber. Stackable. Reload consumes
    /// these to top up a magazine's `loaded_rounds`.
    Ammo,
    /// Phase 4C: weapon attachment (optic, suppressor, adapter,
    /// rail, grip, ...). Items in this category must carry an
    /// `attachment_config` block declaring which mount tag they
    /// consume and what they provide downstream. Loose item in
    /// the inventory; only "installs" onto a weapon via the
    /// attachment-chain validator. Stat effects and slot-graph
    /// runtime application are deferred to Phase 5.
    Attachment,
}

/// Specifies which `Sim` API a `consume_from_slot` call should route
/// to for this item. Serialized as a tagged enum in TOML — each
/// `[[items]]` block uses `consume_action = { kind = "...", ... }`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConsumeAction {
    Eat { food_kind: FoodKind },
    Drink { water_kind: WaterKind },
    ApplyDrug { drug: DrugKind },
    ApplyBandage,
    ApplyTourniquet,
    ApplyDisinfectant,
    ApplyStitch,
    ApplyWoundPack,
    ApplyAntibiotics,
}

impl ConsumeAction {
    /// True if this action requires the caller to specify a
    /// `BodyPart` (bandage / tourniquet / disinfect / stitch /
    /// wound pack target a limb).
    pub fn needs_body_part(&self) -> bool {
        matches!(
            self,
            ConsumeAction::ApplyBandage
                | ConsumeAction::ApplyTourniquet
                | ConsumeAction::ApplyDisinfectant
                | ConsumeAction::ApplyStitch
                | ConsumeAction::ApplyWoundPack
        )
    }
}

/// Per-item salvage recipe (spec §6.1). `tool_required` references
/// another item id (commonly `field_toolkit`); if absent, anything
/// can salvage. `time_ticks` is reserved for Step 5's crafting queue
/// — Step 4 runs salvage instantly.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SalvageRecipe {
    #[serde(default)]
    pub tool_required: Option<ItemId>,
    #[serde(default)]
    pub time_ticks: u64,
    pub outputs: Vec<SalvageOutput>,
}

/// One line of salvage output. Rolled as `rand_range(min..=max)` —
/// 0 is a valid minimum (the output might not appear on a given
/// roll).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SalvageOutput {
    pub id: ItemId,
    pub min: u32,
    pub max: u32,
}

/// A crafting recipe (spec §7.2). Inputs are consumed; outputs
/// are produced. `required_tool` must be present by exact item id
/// (legacy, still used by Step 4 recipes). `required_kit` is the
/// GAMMA-style specialty+tier requirement — inventory must contain
/// at least one [`ItemDef`] with matching [`ToolSpec`] where
/// `tier >= min_tier`. `required_context` is the
/// crafting-station requirement: Campfire or one of the three bench
/// tiers. Higher bench tiers satisfy lower-tier requirements.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Recipe {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub required_tool: Option<ItemId>,
    #[serde(default)]
    pub required_kit: Option<KitRequirement>,
    #[serde(default)]
    pub required_context: Option<CraftStation>,
    #[serde(default)]
    pub time_ticks: u64,
    pub inputs: Vec<ItemStack>,
    pub outputs: Vec<ItemStack>,
}

/// Alias for back-compat with Step 4 naming. Prefer [`CraftStation`]
/// going forward.
pub type RecipeContext = CraftStation;

/// Crafting-station requirement: the fixed-location workspace the
/// recipe needs. Campfire is a free-standing light source; the three
/// bench tiers (GAMMA-style basic → advanced → expert progression)
/// are distinct placeable entities. Recipe checks are
/// **cumulative**: a recipe that requires `BasicBench` is satisfied
/// by any bench tier ≥ Basic the player is standing near; a recipe
/// that requires `Campfire` is satisfied only by a campfire (benches
/// don't substitute).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CraftStation {
    Campfire,
    BasicBench,
    AdvancedBench,
    ExpertBench,
}

/// Three-step progression matching GAMMA / Anomaly: each higher tier
/// satisfies every lower-tier requirement within the same
/// [`Specialty`]. Stored on both [`ToolSpec`] (the item) and
/// [`KitRequirement`] (the recipe).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ToolTier {
    Basic,
    Advanced,
    Expert,
}

/// Axes of the crafting / repair progression. Matches GAMMA's
/// specialty-kit ladder 1:1.
///
/// - `General` — universal toolkit path (Basic/Advanced/Expert
///   Toolkit). Bandages, basic repairs, generic crafts, and the
///   "craft the specialty kit" recipes sit here — in GAMMA, Advanced
///   Tools are the ingredient that crafts a Heavy Armor Repair Kit,
///   Expert Tools for Exosuit repair, etc. Recipes that produce a
///   specialty kit naturally name a `General` tier as their prereq.
/// - `Gunsmith` — weapon disassembly / part fitting / modding /
///   ammo crafting.
/// - `ArmorRepair` — armor condition restoration. The tier gates
///   armor **class**: Basic → light, Advanced → medium, Expert →
///   heavy / exosuit.
/// - `WeaponRepair` — weapon condition restoration, distinct from
///   Gunsmith (condition vs. part swaps). Tier gates weapon
///   **class**: Basic → TYPE A/B, Advanced → TYPE C, Expert →
///   TYPE D.
/// - `DrugMaking` — antibiotics, stims, anti-rad, anti-tox, advanced
///   chem synthesis.
/// - `Shards` — shard combination / upgrade / breaking (homage to
///   GAMMA's Artefact Melter). Single-tier today; future expansion
///   could split basic / advanced melters for rarer shard lines.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Specialty {
    General,
    Gunsmith,
    ArmorRepair,
    WeaponRepair,
    DrugMaking,
    Shards,
}

/// A tool item's specialty + tier badge. Set on toolkits and
/// specialty kits via `tool = { specialty = "...", tier = "..." }`
/// in items.toml. The item is **not** consumed on craft — it only
/// needs to be present in inventory (one per crafter, regardless of
/// how many recipes you queue).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ToolSpec {
    pub specialty: Specialty,
    pub tier: ToolTier,
}

/// Recipe-side declaration: "this recipe needs a `{specialty}` kit of
/// at least `{min_tier}`." Checked by scanning the crafter's
/// inventory for any [`ItemDef`] whose `tool` matches. Example: a
/// repair recipe for an advanced rifle sets
/// `required_kit = { specialty = "gunsmith", min_tier = "advanced" }`;
/// a basic gunsmith kit would fail, an advanced or expert gunsmith
/// kit succeeds.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KitRequirement {
    pub specialty: Specialty,
    pub min_tier: ToolTier,
}

impl CraftStation {
    /// Bench-tier ordering used for cumulative proximity checks —
    /// standing near an Advanced bench satisfies `BasicBench` recipes.
    /// Campfire is unordered (separate kind of station).
    pub fn bench_rank(self) -> Option<u8> {
        match self {
            CraftStation::BasicBench => Some(1),
            CraftStation::AdvancedBench => Some(2),
            CraftStation::ExpertBench => Some(3),
            CraftStation::Campfire => None,
        }
    }
}

/// A count of one item — used in recipe inputs / outputs. Not a
/// runtime inventory instance (that's `crate::components::ItemInstance`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ItemStack {
    pub id: ItemId,
    pub count: u32,
}

// -------- Registries (loaded once, read-only at runtime) --------

/// Read-only registry of every `ItemDef` the sim knows about. Loaded
/// by `Sim::new` / `Sim::load` from the embedded `items.toml`;
/// parsing errors are hard errors at boot.
#[derive(Resource, Clone, Debug)]
pub struct ItemRegistry {
    by_id: HashMap<ItemId, ItemDef>,
}

#[derive(Deserialize)]
struct ItemsFile {
    items: Vec<ItemDef>,
}

impl ItemRegistry {
    /// Parse the bundled item catalog. Item defs are split across
    /// per-category files in `crates/simn-sim/content/items/` (food,
    /// medical, salvage, tools, containers, weapons, magazines,
    /// ammo, armor) so each file stays readable as the catalog
    /// grows. The loader concatenates them into one TOML stream
    /// before parsing — order doesn't matter, IDs are unique
    /// across the whole set. Panics on malformed TOML or duplicate
    /// IDs since item data is author-controlled and the sim can't
    /// run with a broken catalog.
    ///
    /// The parsed catalog is cached process-wide via a `OnceLock`
    /// — every `load()` call after the first returns a clone of the
    /// cached registry, skipping the TOML parse + validation. This
    /// matters because every `Sim::new` calls `load()` and a
    /// 200-test suite would otherwise re-parse 1300+ lines of TOML
    /// 200 times. Clone cost is dominated by the `HashMap` walk,
    /// which is much cheaper than the parse.
    pub fn load() -> Self {
        static CACHE: std::sync::OnceLock<ItemRegistry> = std::sync::OnceLock::new();
        CACHE
            .get_or_init(|| Self::parse(&crate::ContentSource::Embedded))
            .clone()
    }

    /// Load from an explicit content source. `Embedded` routes through
    /// the process-wide cache; a `Dir` pack parses fresh (a game loads
    /// once at boot, so re-parse cost is irrelevant and it sidesteps
    /// the "first pack wins" hazard of the global cache).
    pub fn load_from(src: &crate::ContentSource) -> Self {
        match src {
            crate::ContentSource::Embedded => Self::load(),
            other => Self::parse(other),
        }
    }

    fn parse(src: &crate::ContentSource) -> Self {
        const FILES: [&str; 10] = [
            "items/food.toml",
            "items/medical.toml",
            "items/salvage.toml",
            "items/tools.toml",
            "items/containers.toml",
            "items/weapons.toml",
            "items/magazines.toml",
            "items/ammo.toml",
            "items/armor.toml",
            "items/attachments.toml",
        ];
        let mut raw = String::new();
        for f in FILES {
            raw.push_str(
                &src.read_str(f)
                    .unwrap_or_else(|e| panic!("items content load failed: {e}")),
            );
            raw.push('\n');
        }
        let parsed: ItemsFile =
            toml::from_str(&raw).expect("items/*.toml parse failed — author error");
        let mut by_id = HashMap::with_capacity(parsed.items.len());
        for def in parsed.items {
            if by_id.contains_key(&def.id) {
                panic!("items/*.toml: duplicate id {:?}", def.id);
            }
            by_id.insert(def.id.clone(), def);
        }
        Self { by_id }
    }

    pub fn get(&self, id: &ItemId) -> Option<&ItemDef> {
        self.by_id.get(id)
    }

    pub fn contains(&self, id: &ItemId) -> bool {
        self.by_id.contains_key(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ItemDef> {
        self.by_id.values()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Test-only: override an item's `perishable_ticks` in place.
    /// Lets expiry tests avoid waiting real-world minutes.
    #[doc(hidden)]
    pub fn set_perishable_for_test(&mut self, id: &ItemId, ticks: Option<u64>) {
        if let Some(def) = self.by_id.get_mut(id) {
            def.perishable_ticks = ticks;
        }
    }

    /// Test-only: empty registry to seed with synthetic items.
    /// Used by the [`crate::inventory_grid`] tests so they don't have
    /// to drag in the bundled `items.toml`.
    #[doc(hidden)]
    pub fn empty_for_test() -> Self {
        Self {
            by_id: HashMap::new(),
        }
    }

    /// Test-only: insert a synthetic [`ItemDef`].
    #[doc(hidden)]
    pub fn insert_for_test(&mut self, def: ItemDef) {
        self.by_id.insert(def.id.clone(), def);
    }
}

impl Default for ItemRegistry {
    fn default() -> Self {
        Self::load()
    }
}

/// Read-only registry of every `Recipe` the sim knows about.
#[derive(Resource, Clone, Debug)]
pub struct RecipeRegistry {
    by_id: HashMap<String, Recipe>,
}

#[derive(Deserialize)]
struct RecipesFile {
    recipes: Vec<Recipe>,
}

impl RecipeRegistry {
    /// Parse + cache; see [`ItemRegistry::load`] for why.
    pub fn load() -> Self {
        static CACHE: std::sync::OnceLock<RecipeRegistry> = std::sync::OnceLock::new();
        CACHE
            .get_or_init(|| Self::parse(&crate::ContentSource::Embedded))
            .clone()
    }

    /// Load from an explicit content source; see [`ItemRegistry::load_from`].
    pub fn load_from(src: &crate::ContentSource) -> Self {
        match src {
            crate::ContentSource::Embedded => Self::load(),
            other => Self::parse(other),
        }
    }

    fn parse(src: &crate::ContentSource) -> Self {
        let raw = src
            .read_str("recipes.toml")
            .unwrap_or_else(|e| panic!("recipes content load failed: {e}"));
        let parsed: RecipesFile =
            toml::from_str(&raw).expect("recipes.toml parse failed — author error");
        let mut by_id = HashMap::with_capacity(parsed.recipes.len());
        for recipe in parsed.recipes {
            if by_id.contains_key(&recipe.id) {
                panic!("recipes.toml: duplicate id {:?}", recipe.id);
            }
            by_id.insert(recipe.id.clone(), recipe);
        }
        Self { by_id }
    }

    pub fn get(&self, id: &str) -> Option<&Recipe> {
        self.by_id.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Recipe> {
        self.by_id.values()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

impl Default for RecipeRegistry {
    fn default() -> Self {
        Self::load()
    }
}

/// Read-only registry of every paper-doll equipment slot the sim
/// knows about. Loaded once from `content/equipment_slots.toml` at
/// `Sim::new` / `Sim::load`. Same modularity pattern as
/// [`ItemRegistry`] / [`RecipeRegistry`] — adding a new slot is a
/// TOML-only change.
#[derive(Resource, Clone, Debug)]
pub struct EquipmentSlotRegistry {
    slots: Vec<EquipmentSlotDef>,
    by_id: HashMap<SlotId, usize>,
    by_hotbar_index: HashMap<u8, usize>,
}

#[derive(Deserialize)]
struct SlotsFile {
    slots: Vec<EquipmentSlotDef>,
}

impl EquipmentSlotRegistry {
    /// Parse the bundled `equipment_slots.toml`. Panics on malformed
    /// TOML or on duplicate slot id / hotbar index — these are
    /// author-controlled invariants and the sim can't run with a
    /// broken layout. Parsed result is cached process-wide; see
    /// [`ItemRegistry::load`] for the rationale.
    pub fn load() -> Self {
        static CACHE: std::sync::OnceLock<EquipmentSlotRegistry> = std::sync::OnceLock::new();
        CACHE
            .get_or_init(|| Self::parse(&crate::ContentSource::Embedded))
            .clone()
    }

    /// Load from an explicit content source; see [`ItemRegistry::load_from`].
    pub fn load_from(src: &crate::ContentSource) -> Self {
        match src {
            crate::ContentSource::Embedded => Self::load(),
            other => Self::parse(other),
        }
    }

    fn parse(src: &crate::ContentSource) -> Self {
        let raw = src
            .read_str("equipment_slots.toml")
            .unwrap_or_else(|e| panic!("equipment_slots content load failed: {e}"));
        let parsed: SlotsFile =
            toml::from_str(&raw).expect("equipment_slots.toml parse failed — author error");
        let mut by_id = HashMap::with_capacity(parsed.slots.len());
        let mut by_hotbar_index = HashMap::new();
        for (i, def) in parsed.slots.iter().enumerate() {
            if by_id.contains_key(&def.id) {
                panic!("equipment_slots.toml: duplicate slot id {:?}", def.id);
            }
            by_id.insert(def.id.clone(), i);
            if def.is_hotbar {
                if def.hotbar_index == 0 {
                    panic!(
                        "equipment_slots.toml: slot {:?} has is_hotbar=true but hotbar_index=0",
                        def.id
                    );
                }
                if by_hotbar_index.contains_key(&def.hotbar_index) {
                    panic!(
                        "equipment_slots.toml: duplicate hotbar_index {} (slot {:?})",
                        def.hotbar_index, def.id
                    );
                }
                by_hotbar_index.insert(def.hotbar_index, i);
            }
        }
        Self {
            slots: parsed.slots,
            by_id,
            by_hotbar_index,
        }
    }

    pub fn get(&self, id: &SlotId) -> Option<&EquipmentSlotDef> {
        self.by_id.get(id).and_then(|&i| self.slots.get(i))
    }

    pub fn iter(&self) -> impl Iterator<Item = &EquipmentSlotDef> {
        self.slots.iter()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Look up a slot by its hotbar index (1-based). Returns `None`
    /// for indices that no slot claims.
    pub fn by_hotbar_index(&self, idx: u8) -> Option<&EquipmentSlotDef> {
        self.by_hotbar_index
            .get(&idx)
            .and_then(|&i| self.slots.get(i))
    }

    /// Check whether `def` (an [`ItemDef`]) can equip to the slot
    /// `slot_id`. Rules:
    ///
    /// 1. If `def.equip_slots` is non-empty, the slot id must appear
    ///    in it (whitelist overrides category matching).
    /// 2. Else, the slot must accept `def.category` via
    ///    [`EquipmentSlotDef::accepts`].
    ///
    /// Returns `None` if either the slot id or the def is unknown.
    pub fn can_equip(&self, slot_id: &SlotId, def: &ItemDef) -> bool {
        let Some(slot) = self.get(slot_id) else {
            return false;
        };
        if !def.equip_slots.is_empty() {
            return def.equip_slots.iter().any(|s| s == slot_id);
        }
        slot.accepts.contains(&def.category)
    }

    #[doc(hidden)]
    pub fn empty_for_test() -> Self {
        Self {
            slots: Vec::new(),
            by_id: HashMap::new(),
            by_hotbar_index: HashMap::new(),
        }
    }

    #[doc(hidden)]
    pub fn insert_for_test(&mut self, def: EquipmentSlotDef) {
        let idx = self.slots.len();
        self.by_id.insert(def.id.clone(), idx);
        if def.is_hotbar && def.hotbar_index > 0 {
            self.by_hotbar_index.insert(def.hotbar_index, idx);
        }
        self.slots.push(def);
    }
}

impl Default for EquipmentSlotRegistry {
    fn default() -> Self {
        Self::load()
    }
}
