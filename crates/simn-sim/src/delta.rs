//! Journal delta records.
//!
//! Each `WorldDelta` is one atomic mutation of the simulation state.
//! Applying a series of deltas on top of a snapshot reconstructs the
//! state exactly. Keep this enum additive — unknown variants on load
//! should be skipped (forward-compat), not error out, so clients with
//! older code can still read newer journals.

use serde::{Deserialize, Serialize};

use crate::chronicle::DeathCause;
use crate::components::{
    BodyPart, DrugKind, EffectId, EffectKind, NpcId, SurvivalStat, WoundId, WoundKind,
    WoundTreatment,
};

use crate::items::{ItemId, ItemStack};
use crate::region::RegionId;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum WorldDelta {
    /// A new player entity was added to the sim.
    SpawnPlayer {
        steam_id: u64,
        region: RegionId,
        pos: [f32; 3],
        yaw: f32,
    },
    /// A player entity was removed (disconnect).
    DespawnPlayer { steam_id: u64 },
    /// Player transform updated within a region.
    MovePlayer {
        steam_id: u64,
        pos: [f32; 3],
        yaw: f32,
    },
    /// Player crossed a region boundary.
    ChangePlayerRegion { steam_id: u64, region: RegionId },
    /// Player's aggregate health was set to a specific value (post-clamp).
    /// Retained for forward-compat replay of pre-stats-foundation
    /// journals; new code emits `SetBodyPart` instead and the aggregate
    /// is derived from body parts.
    SetHealth { steam_id: u64, current: f32 },
    /// Player's stamina was set to a specific value (post-clamp).
    /// Per-tick regen is *not* journaled — only explicit drains/sets.
    SetStamina { steam_id: u64, current: f32 },
    /// One body-part HP was set to a specific value (post-clamp).
    /// The aggregate `Health` mirror is rederived from parts on replay.
    SetBodyPart {
        steam_id: u64,
        part: BodyPart,
        current: f32,
    },
    /// One survival meter was set to a specific value (post-clamp to
    /// `[0, SurvivalStats::FULL]`). Per-tick drain is *not* journaled —
    /// drain is a deterministic function of last-known value + elapsed
    /// in-world time, so the snapshot is enough.
    SetSurvivalStat {
        steam_id: u64,
        stat: SurvivalStat,
        current: f32,
    },
    /// A new wound was added to a player. Spawned by
    /// `Sim::apply_damage_to_part` when damage exceeds the wound
    /// threshold. Replay re-creates the wound on the player's
    /// `Wounds` component.
    WoundAdded {
        steam_id: u64,
        wound_id: WoundId,
        body_part: BodyPart,
        kind: WoundKind,
        severity: u8,
        spawned_tick: u64,
    },
    /// Treatment state of an existing wound changed (bandage applied,
    /// tourniquet on/off). Per-tick auto-heal of `Bandaged` → `Healed`
    /// is *not* journaled — it's a deterministic function of
    /// `treatment_changed_tick` + elapsed ticks, derived on replay.
    WoundTreatmentChanged {
        steam_id: u64,
        wound_id: WoundId,
        new_state: WoundTreatment,
        changed_tick: u64,
    },
    /// A wound flipped to `infected` (driven by `tick_infection`'s
    /// untreated-wound timer). Journaled because the trigger is
    /// time-based and we want replay to land at the same tick.
    WoundInfected {
        steam_id: u64,
        wound_id: WoundId,
        started_tick: u64,
    },
    /// A new active effect (drug, status) attached to a player. Spawned
    /// by `Sim::apply_drug`, `Sim::eat`/`drink` (when the food grants
    /// an effect), or by `Sim::apply_drug` again on the overdose path
    /// (where the spawned `kind` is `OverdoseDisorientation`).
    /// `intensity` magnitude depends on the effect kind.
    EffectApplied {
        steam_id: u64,
        effect_id: EffectId,
        kind: EffectKind,
        applied_tick: u64,
        duration_ticks: u64,
        intensity: f32,
    },
    /// A drug tolerance counter was set to a new value. Per-tick decay
    /// is *not* journaled — it's pure (deterministic from snapshot +
    /// elapsed ticks). Only explicit bumps (apply_drug) journal.
    ToleranceChanged {
        steam_id: u64,
        drug: DrugKind,
        value: f32,
    },
    /// A player's radiation level was set (post-clamp `[0, 100]`).
    /// Passive decay is pure; only explicit application/consumption
    /// journals.
    RadiationChanged { steam_id: u64, value: f32 },
    /// A player's toxicity level was set (post-clamp `[0, 100]`).
    /// Same shape as `RadiationChanged`.
    ToxicityChanged { steam_id: u64, value: f32 },
    /// A new NPC entered the world. `faction` is the registry name
    /// string (`"coalition"`) so journals stay valid across registry edits.
    NpcSpawned {
        id: NpcId,
        faction: String,
        region: RegionId,
        pos: [f32; 3],
        yaw: f32,
        die_at_tick: u64,
    },
    /// An NPC moved between regions.
    NpcChangeRegion {
        id: NpcId,
        region: RegionId,
        pos: [f32; 3],
    },
    /// An NPC died. The chronicle keeps the record around; the entity
    /// is gone.
    NpcDied {
        id: NpcId,
        region: RegionId,
        cause: DeathCause,
        tick: u64,
    },
    /// One body-part HP was set to a specific value on an NPC
    /// (post-clamp). Aggregate `Health` mirror is rederived from parts
    /// on replay. Mirrors `SetBodyPart` for players. Emitted by
    /// `Sim::apply_damage_to_npc_part` and `Sim::heal_npc_part`;
    /// `npc_combat`'s per-tick probabilistic damage does not journal
    /// (same trade-off as before — recovered from the next snapshot).
    SetNpcBodyPart {
        id: NpcId,
        part: BodyPart,
        current: f32,
    },
    /// A new wound was added to an NPC. Mirrors `WoundAdded` for
    /// players. Spawned by `Sim::apply_damage_to_npc_part` when
    /// damage exceeds the wound threshold. `npc_combat`'s own wound
    /// spawns (from the probabilistic NPC-vs-NPC hit model) are
    /// NOT journaled — they're ephemeral, same policy as the torso
    /// HP drain that already rides that path.
    NpcWoundAdded {
        id: NpcId,
        wound_id: WoundId,
        body_part: BodyPart,
        kind: WoundKind,
        severity: u8,
        spawned_tick: u64,
    },
    /// Treatment state of an existing NPC wound changed. Mirror of
    /// `WoundTreatmentChanged`. Per-tick auto-heal of `Bandaged` →
    /// `Healed` is NOT journaled — it's a deterministic function of
    /// `treatment_changed_tick` + clock.
    NpcWoundTreatmentChanged {
        id: NpcId,
        wound_id: WoundId,
        new_state: WoundTreatment,
        changed_tick: u64,
    },
    /// An NPC's wound flipped to `infected` (driven by `tick_infection`).
    /// Kept for parity with `WoundInfected` though the system derives
    /// infection purely from treatment + clock; this variant is
    /// reserved for tooling / future explicit infection journaling.
    NpcWoundInfected {
        id: NpcId,
        wound_id: WoundId,
        started_tick: u64,
    },
    /// A new active effect attached to an NPC. Mirrors `EffectApplied`
    /// for players. In this slice only `AntibioticsActive` is emitted
    /// (via `Sim::apply_antibiotics_npc`); the variant stays generic
    /// across `EffectKind` so future NPC effect work doesn't need a
    /// second delta.
    NpcEffectApplied {
        id: NpcId,
        effect_id: EffectId,
        kind: EffectKind,
        applied_tick: u64,
        duration_ticks: u64,
        intensity: f32,
    },
    /// An item stack was added to a player's inventory (pickup or debug
    /// grant). `spawned_tick` is the mint timestamp used by perishable
    /// aging; `slot_hint` is advisory for UIs — replay places stacks by
    /// normal merge/split rules, not by slot index.
    ItemPickedUp {
        steam_id: u64,
        item_id: ItemId,
        count: u32,
        spawned_tick: u64,
    },
    /// An item stack was removed from a player's inventory (drop). Step 4
    /// does not create ground items — the stack vanishes; world
    /// containers + ground-item entities land in a later PR.
    ItemDropped {
        steam_id: u64,
        slot_idx: usize,
        count: u32,
    },
    /// Two inventory slots swapped. Slot indices reference the player's
    /// `Inventory` vector at the time of replay.
    ItemMoved {
        steam_id: u64,
        from_slot: usize,
        to_slot: usize,
    },
    /// Item moved between two distinct grids on the player (pockets ↔
    /// equipped-container inner grid, or two equipped inner grids).
    /// `item` + `inner_grid` ride along so client mirrors can replay
    /// without seeing the host's RNG; the host's first-fit position is
    /// re-derived deterministically on the mirror's view of the dest
    /// grid.
    ItemMovedBetweenGrids {
        steam_id: u64,
        from_grid: String,
        from_idx: usize,
        to_grid: String,
        to_idx: usize,
        item: crate::components::ItemInstance,
        inner_grid: Option<crate::components::GridInventory>,
    },
    /// One unit of a stack was consumed via `consume_from_slot` (eat /
    /// drink / apply_drug / apply_bandage / ...). The underlying sim
    /// side-effects (survival stat changes, effects, wound treatments)
    /// are journaled separately as their own deltas — this record is
    /// just the inventory decrement so replay matches.
    ItemConsumed {
        steam_id: u64,
        slot_idx: usize,
        body_part: Option<BodyPart>,
    },
    /// Junk was salvaged. `outputs` is the actual rolled result so replay
    /// is deterministic without re-running the salvage RNG.
    ItemsSalvaged {
        steam_id: u64,
        source_slot: usize,
        outputs: Vec<ItemStack>,
        tick: u64,
    },
    /// A recipe was crafted. Inputs were consumed FIFO from matching
    /// stacks; outputs minted with `spawned_tick = tick`.
    ItemsCrafted {
        steam_id: u64,
        recipe_id: String,
        tick: u64,
    },
    /// Player equipped an item into `slot_id`. `source_grid` names the
    /// grid the item came from (`"pockets"` or a nested
    /// `"equipped:<slot_id>"` reference — see
    /// [`crate::world::Sim::equip`]). `source_idx` is the items-vec
    /// index inside that grid. If `inner_grid` is `Some`, the
    /// equipped item is a container and the attached grid travels
    /// with it.
    ItemEquipped {
        steam_id: u64,
        slot_id: crate::items::SlotId,
        item: crate::components::ItemInstance,
        inner_grid: Option<crate::components::GridInventory>,
        source_grid: String,
        source_idx: usize,
    },
    /// Player unequipped the item at `slot_id` back into `dest_grid`.
    /// If the destination grid had no room for the bare stack, the
    /// item sits at the named `(dest_x, dest_y)` anyway; UI callers
    /// are expected to pre-validate. `inner_grid` (if `Some`) is the
    /// container payload that came off with the item.
    ItemUnequipped {
        steam_id: u64,
        slot_id: crate::items::SlotId,
        item: crate::components::ItemInstance,
        inner_grid: Option<crate::components::GridInventory>,
        dest_grid: String,
    },
    /// PR-4: a new world container appeared (ground drop, scene-placed
    /// crate, NPC corpse). The grid starts as captured here; later
    /// adds / removes journal individually.
    WorldContainerSpawned {
        id: crate::components::ContainerId,
        region: RegionId,
        pos: [f32; 3],
        is_public: bool,
        initial_grid: crate::components::GridInventory,
    },
    /// PR-4: a world container was removed (drop picked up clean,
    /// corpse TTL'd out, scene unloaded). Mirror sims drop the entity.
    WorldContainerDespawned { id: crate::components::ContainerId },
    /// PR-4: an item moved INTO a container. `item` carries the stack
    /// and any nested inner grid; mirrors apply via `grant_or_merge`
    /// on the container's grid.
    WorldContainerItemAdded {
        id: crate::components::ContainerId,
        item: crate::components::ItemInstance,
        inner_grid: Option<crate::components::GridInventory>,
    },
    /// PR-4: an item moved OUT of a container. The taken stack is
    /// described so mirrors can update both sides without re-running
    /// the container's FIFO consume algorithm.
    WorldContainerItemRemoved {
        id: crate::components::ContainerId,
        source_idx: u32,
        taken: crate::components::ItemInstance,
        inner_grid: Option<crate::components::GridInventory>,
    },
    /// Weapons phase 1: player reloaded the weapon at `slot_id`.
    /// Self-describing — the full post-reload `loaded_magazine`
    /// (with its `magazine_state.loaded_rounds`) is in the delta so
    /// replay reproduces the exact state. `ejected` is the mag
    /// returned to pockets (if any); empty on first reload of a
    /// fresh weapon.
    WeaponReloaded {
        steam_id: u64,
        slot_id: crate::items::SlotId,
        loaded_magazine: crate::components::ItemInstance,
        #[serde(default)]
        ejected: Option<crate::components::ItemInstance>,
    },
    /// Weapons phase 1: player ejected the magazine at `slot_id`
    /// without replacing it. `ejected` is the mag that came out
    /// (if any) — replay returns it to pockets.
    WeaponMagazineEjected {
        steam_id: u64,
        slot_id: crate::items::SlotId,
        ejected: Option<crate::components::ItemInstance>,
    },
    /// Weapons phase 1: player fired the weapon at `slot_id`. One
    /// round consumed from the loaded magazine; `remaining_rounds`
    /// is the post-fire count. The bridge-side raycast resolves the
    /// hit separately (routed through `apply_damage_to_npc_part` if
    /// it lands).
    WeaponFired {
        steam_id: u64,
        slot_id: crate::items::SlotId,
        remaining_rounds: u32,
    },
    /// Phase 4D: a fire attempt rolled a jam instead of expending
    /// a round. The weapon's `jam_state` transitioned to a
    /// non-cleared kind; subsequent fire calls dry-click until
    /// `clear_weapon_jam` runs. Carries the condition at jam
    /// time + the jam kind so client UX can pick the right
    /// clear-jam prompt + audio cue.
    WeaponJammed {
        steam_id: u64,
        slot_id: crate::items::SlotId,
        jam: crate::components::JamState,
        condition: f32,
    },
    /// Phase 4D: a `clear_weapon_jam` action ran successfully.
    /// `jam_state` is back to `Cleared`; the weapon is firable
    /// again next call. Condition is unaffected — clearing a
    /// jam doesn't repair wear.
    WeaponJamCleared {
        steam_id: u64,
        slot_id: crate::items::SlotId,
    },
    /// Phase 4D: condition on the loaded weapon at `slot_id`
    /// changed (typically from per-shot wear). Mirror clients
    /// apply the new value so HUD condition meters tick down
    /// in lockstep with the auth sim.
    WeaponConditionChanged {
        steam_id: u64,
        slot_id: crate::items::SlotId,
        new_condition: f32,
    },
    /// Weapons phase 2: player topped up the magazine loaded at
    /// `slot_id` with `added` rounds of `round_id`, bringing it
    /// to `total`. Ammo stacks in pockets were consumed by the
    /// matching count. Client-side replay mutates the mag state +
    /// consumes the ammo stack via `consume_from_stacks`.
    MagazineLoaded {
        steam_id: u64,
        slot_id: crate::items::SlotId,
        round_id: crate::items::ItemId,
        added: u32,
        total: u32,
    },
    /// Weapons phase 2: player topped up a magazine sitting in
    /// pockets (not installed in a weapon slot). `pocket_idx`
    /// references the `Inventory.items` slot; stable within the
    /// replay window because `consume_from_stacks` touches only
    /// the ammo row, never the mag row.
    PocketMagazineLoaded {
        steam_id: u64,
        pocket_idx: u32,
        round_id: crate::items::ItemId,
        added: u32,
        total: u32,
    },
    /// Weapons phase 2: a `Projectile` entity was spawned. Mirrors
    /// replay this by creating a client-side tracer that animates
    /// from `origin` along `velocity`; the tracer resolves when
    /// the matching [`WorldDelta::ProjectileImpacted`] arrives.
    ProjectileSpawned {
        id: crate::components::ProjectileId,
        source_steam_id: u64,
        /// Phase 4A v1: `Some` when the projectile was fired by
        /// an NPC. `#[serde(default)]` for pre-4A snapshots /
        /// mirror replay.
        #[serde(default)]
        source_npc_id: Option<crate::components::NpcId>,
        round_id: crate::items::ItemId,
        /// Phase 4B v2: variant tag (FMJ / HP / AP / Tracer /
        /// Overpressure) resolved from the round's
        /// `AmmoConfig.variant` at fire time. Client uses this to
        /// drive tracer color, casing-eject SFX, and (future)
        /// impact-FX selection without doing an item-registry
        /// lookup per spawn. `#[serde(default)]` resolves to
        /// `Fmj` on pre-4B v2 snapshots in flight.
        #[serde(default)]
        variant: crate::items::AmmoVariant,
        origin: [f32; 3],
        velocity: [f32; 3],
        max_range_m: f32,
        spawned_tick: u64,
    },
    /// Weapons phase 2: a projectile reached its impact point —
    /// either a body part on an NPC, or the terminal of its max
    /// range. `damage_applied` is the post-armor HP already applied
    /// via `apply_damage_to_npc_part`; `penetrated` indicates
    /// whether the round breached armor (true) or was reduced to
    /// blunt trauma (false). NPC miss (`hit_npc = None`) occurs
    /// when the projectile runs out of range — the client uses the
    /// terminal `pos` for a ground impact FX.
    ProjectileImpacted {
        id: crate::components::ProjectileId,
        pos: [f32; 3],
        #[serde(default)]
        hit_npc: Option<crate::components::NpcId>,
        /// Phase 4A v2: when the projectile struck the player,
        /// carries their steam_id; mutually exclusive with
        /// `hit_npc`. `#[serde(default)]` for pre-4A
        /// snapshots that never had a player-hit variant.
        #[serde(default)]
        hit_player_steam_id: Option<u64>,
        #[serde(default)]
        body_part: Option<crate::components::BodyPart>,
        damage_applied: f32,
        penetrated: bool,
    },
    /// Debug-only: transient "player is standing near a campfire" flag
    /// flipped. Journaled so save/restore preserves the setting across a
    /// crafting session. Step 5 replaces this with scene-placed campfire
    /// entities + proximity checks.
    NearCampfireSet { steam_id: u64, value: bool },
    /// Like [`WorldDelta::NearCampfireSet`] but for the tiered workbench
    /// proximity flag. Scene-placed workbench entities + a proximity
    /// system will drive this in production; the debug setter is
    /// behind `Sim::set_player_near_workbench_for_test` until then.
    NearWorkbenchSet {
        steam_id: u64,
        tier: Option<crate::items::ToolTier>,
    },
    /// A crafting job was queued. Inputs were already consumed at
    /// queue time (spec §7.5 — materials lock up front to prevent
    /// cancel-re-queue duping). `inputs_consumed` captures the exact
    /// stacks removed so clients can mirror the inventory mutation
    /// without re-running the FIFO consume algorithm. Per-unit
    /// completions are NOT journaled — they're a deterministic
    /// function of queue state + clock, same pattern as
    /// `tick_perishables`.
    CraftJobQueued {
        steam_id: u64,
        job_id: u32,
        recipe_id: String,
        count: u32,
        time_ticks_per_unit: u64,
        inputs_consumed: Vec<ItemStack>,
        started_tick: u64,
    },
    /// A crafting job was cancelled before completion. Remaining
    /// unconsumed inputs (for the units that hadn't run yet) are
    /// refunded to inventory as listed. The job's current in-progress
    /// unit is **forfeit** — materials for it are lost (simpler than
    /// fractional refunds).
    CraftJobCancelled {
        steam_id: u64,
        job_id: u32,
        refund: Vec<ItemStack>,
    },
    /// Batched per-tick NPC transform updates, emitted by the host to
    /// drive client-side pill motion. NPC movement systems
    /// (`tick_npc_goals`, `npc_migrate`, etc.) run **only on the
    /// authoritative sim** — running them on a client would diverge
    /// from host state because their RNG seeds mix in `Entity::to_bits()`
    /// which differs between instances. So host owns the positions and
    /// broadcasts them every tick.
    ///
    /// Not written to the host's journal — it's a deterministic function
    /// of the host's schedule output. Replay on the authoritative side
    /// recomputes positions by running the schedule; only the
    /// cross-peer wire path needs this variant.
    NpcPositionBatch {
        tick: u64,
        updates: Vec<(NpcId, [f32; 3], f32)>,
    },
    /// Tick boundary marker. Lets replay know "everything before this
    /// belongs to tick N" for future rewind/replay tooling.
    Tick { tick: u64 },
    /// A faction-vs-faction relation drift was applied. Both faction
    /// names are registry name strings; the delta is the signed
    /// nudge (`i16`, accumulator saturates on read). `reason` is a
    /// free-form string for the chronicle UI
    /// ("killed_lineman_crew_chief", "freed_attuned_prisoner") — no
    /// machine-readable schema, just human/log breadcrumbs.
    FactionRelationShift {
        a: String,
        b: String,
        delta: i16,
        reason: String,
    },
    /// A per-player faction reputation drift. Same shape as
    /// `FactionRelationShift` but isolated to one player's rep —
    /// see `PlayerReputation`. Player A's troublemaking does not
    /// move player B's standing.
    PlayerRepShift {
        steam_id: u64,
        faction: String,
        delta: i16,
        reason: String,
    },
}
