//! Weapons — Phase 1: equipped-weapon state, reload, magazine
//! eject. All tuning (damage, range, fire rate, spread, magazine
//! capacity, caliber) flows from `items.toml` via the
//! [`crate::items::ItemDef`] `weapon_config` / `magazine_config` /
//! `ammo_config` blocks. Engine code never supplies defaults that
//! could shadow missing TOML fields; a weapon without a
//! `weapon_config` block can't fire, a magazine without a
//! `magazine_config` block can't be loaded.
//!
//! Phase 2 extends this module with projectile simulation + ballistic
//! drop. Phase 3 adds the attachment slot-tag graph on top of
//! [`crate::components::EquippedWeaponState`]. This file stays
//! small; the data model grows in `items.rs` / `components.rs`.
//!
//! Spec reference: `docs/book/src/planning/weapons-plan.md` §2 (data
//! model), §6 (magazine-as-container).
//!
//! **Fire path** (lives in `crates/simn-godot` for now): the client
//! raycasts, decrements `loaded_rounds` via `Sim::fire_weapon`
//! (commit 3), and damage routes through the existing
//! `apply_damage_to_npc_part` path.

use anyhow::Result;
use bevy_ecs::prelude::*;

use crate::components::{
    Equipment, EquippedWeaponState, Inventory, ItemInstance, MagazineState, Position,
};
use crate::delta::WorldDelta;
use crate::inventory_grid;
use crate::items::{ItemId, ItemRegistry, SlotId, WeaponConfig};

use super::Sim;

impl Sim {
    /// Reload the weapon at `slot_id`. The flow:
    ///
    /// 1. Resolve the player + their weapon equipped at `slot_id`.
    ///    Error if no weapon is there or the item has no
    ///    `weapon_config` (can't be a weapon without one).
    /// 2. Remove any magazine currently loaded (keeping its
    ///    `magazine_state.loaded_rounds` intact).
    /// 3. Scan the player's pockets grid for a magazine whose
    ///    `magazine_config.caliber` matches the weapon's
    ///    `weapon_config.caliber`. Prefer the most-loaded one so
    ///    players default to tactical reloads (swap for the fullest
    ///    mag); tie-break by grid placement order.
    /// 4. Remove that magazine from pockets; install it as the
    ///    weapon's `loaded_magazine`.
    /// 5. If there was an old magazine, return it to pockets via
    ///    `grant_or_merge` so the player keeps it.
    /// 6. Journal `WorldDelta::WeaponReloaded`.
    ///
    /// Errors if no matching magazine is available, if the slot
    /// isn't a weapon, or if the player doesn't exist.
    pub fn reload_weapon(&mut self, steam_id: u64, slot_id: &SlotId) -> Result<()> {
        let e = self
            .find_player_entity(steam_id)
            .ok_or_else(|| anyhow::anyhow!("unknown player {steam_id}"))?;

        // Read the weapon's required caliber from its ItemDef →
        // weapon_config.caliber. Engine never hardcodes a caliber.
        let weapon_caliber = weapon_config_at(&self.world, e, slot_id, "reload")?
            .caliber
            .clone();

        // Find the best matching magazine in pockets. Preference:
        // most-loaded, then first-placed.
        let best_pocket_idx = {
            let inv = self
                .world
                .get::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            let reg = self.world.resource::<ItemRegistry>();
            let mut best: Option<(usize, u32)> = None;
            for (idx, placed) in inv.0.items.iter().enumerate() {
                let Some(def) = reg.get(&placed.stack.id) else {
                    continue;
                };
                let Some(mc) = def.magazine_config.as_ref() else {
                    continue;
                };
                if mc.caliber != weapon_caliber {
                    continue;
                }
                let loaded = placed.stack.loaded_rounds();
                match best {
                    Some((_, best_loaded)) if best_loaded >= loaded => {}
                    _ => best = Some((idx, loaded)),
                }
            }
            best.map(|(idx, _)| idx).ok_or_else(|| {
                anyhow::anyhow!("reload: no magazine in caliber {:?}", weapon_caliber)
            })?
        };

        // Pull the mag out of pockets (detaches the PlacedItem).
        let pulled = {
            let mut inv = self
                .world
                .get_mut::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Inventory"))?;
            inventory_grid::remove(&mut inv.0, best_pocket_idx)
                .map_err(|err| anyhow::anyhow!("reload: mag removal failed: {err:?}"))?
        };
        let mut new_mag = pulled.stack;
        // Ensure the pulled mag has a magazine_state; if it was a
        // freshly-granted empty mag missing the field, default to
        // empty + no variant. Phase 2 requires explicit ammo loading
        // before the mag has a firable variant.
        if new_mag.magazine_state.is_none() {
            new_mag.magazine_state = Some(MagazineState::default());
        }

        // Swap the new mag in, keeping the old one aside.
        let ejected = {
            let mut eq = self
                .world
                .get_mut::<Equipment>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Equipment"))?;
            let equipped =
                eq.0.get_mut(slot_id)
                    .ok_or_else(|| anyhow::anyhow!("reload: slot {slot_id:?} disappeared"))?;
            let ws = equipped
                .weapon_state
                .get_or_insert_with(EquippedWeaponState::default);
            ws.loaded_magazine.replace(new_mag.clone())
        };

        // If there was an ejected mag, put it back in pockets. Keep
        // the pulled spawned_tick so perishable aging (future rule)
        // doesn't reset.
        if let Some(old_mag) = ejected.clone() {
            place_mag_in_pockets(&mut self.world, e, old_mag)?;
        }

        self.record_delta(WorldDelta::WeaponReloaded {
            steam_id,
            slot_id: slot_id.clone(),
            loaded_magazine: new_mag,
            ejected,
        })?;
        Ok(())
    }

    /// Top up the magazine currently loaded in `slot_id` with
    /// rounds of the given `round_id`, consuming ammo stacks from
    /// the player's pockets. Returns the number of rounds actually
    /// loaded.
    ///
    /// Rules:
    /// - The mag must be present and its caliber must match the
    ///   ammo's caliber.
    /// - If the mag already has a variant set and it differs from
    ///   `round_id`, load is rejected (player must fire-out or
    ///   eject-and-reload-empty first — real-gun model).
    /// - Stops when the mag's `magazine_config.capacity` is
    ///   reached or pockets run out of matching rounds.
    ///
    /// Journals `WorldDelta::MagazineLoaded { … }` on any non-zero
    /// load; returns `Ok(0)` with no journal on a zero-effect call.
    ///
    /// See also [`Self::load_rounds_into_pocket_mag`] for loading
    /// a magazine that isn't installed in a weapon — the same
    /// validation is applied via `validate_mag_load`.
    pub fn load_rounds_into_mag(
        &mut self,
        steam_id: u64,
        slot_id: &SlotId,
        round_id: &ItemId,
    ) -> Result<u32> {
        let e = self
            .find_player_entity(steam_id)
            .ok_or_else(|| anyhow::anyhow!("unknown player {steam_id}"))?;

        // Resolve the target mag's id + current state. Splits the
        // borrow so the `ItemRegistry` read below doesn't overlap
        // with the mag mutable borrow.
        let (mag_id, current_rounds, current_variant) = {
            let mag = loaded_magazine_mut(&mut self.world, e, slot_id)
                .ok_or_else(|| anyhow::anyhow!("load_rounds: no magazine loaded"))?;
            let state = mag.magazine_state.clone().unwrap_or_default();
            (mag.id.clone(), state.loaded_rounds, state.variant)
        };

        // Validate + resolve capacity (caliber match + variant-flip
        // check live here so the equipped-mag and pocket-mag paths
        // share a single rule set).
        let capacity = {
            let reg = self.world.resource::<ItemRegistry>();
            validate_mag_load(reg, &mag_id, round_id, current_rounds, &current_variant)?
        };

        let room = capacity.saturating_sub(current_rounds);
        if room == 0 {
            return Ok(0);
        }

        // Consume rounds from pockets (up to `room`).
        let loaded = {
            let mut inv = self
                .world
                .get_mut::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("load_rounds: no Inventory"))?;
            super::inventory::consume_from_stacks(&mut inv, round_id, room)
        };
        if loaded == 0 {
            return Ok(0);
        }

        // Write the new mag state.
        {
            let mag = loaded_magazine_mut(&mut self.world, e, slot_id)
                .ok_or_else(|| anyhow::anyhow!("load_rounds: mag disappeared mid-load"))?;
            mag.magazine_state = Some(MagazineState {
                loaded_rounds: current_rounds + loaded,
                variant: Some(round_id.clone()),
            });
        }

        self.record_delta(WorldDelta::MagazineLoaded {
            steam_id,
            slot_id: slot_id.clone(),
            round_id: round_id.clone(),
            added: loaded,
            total: current_rounds + loaded,
        })?;
        Ok(loaded)
    }

    /// Top up a magazine sitting at `pocket_idx` in the player's
    /// pockets grid with rounds of `round_id`. Same validation
    /// rules as [`Self::load_rounds_into_mag`] (caliber match,
    /// no partial-mag variant flip), but targets a pocket mag so
    /// players can pre-load spares before inserting them.
    ///
    /// Returns the number of rounds actually loaded. Journals
    /// `WorldDelta::PocketMagazineLoaded` on any non-zero load.
    ///
    /// `pocket_idx` is stable within this call — pockets are
    /// compacted only by the ammo-stack consume step, which
    /// targets a different `ItemId` than the magazine, so the
    /// magazine itself never moves index during loading.
    pub fn load_rounds_into_pocket_mag(
        &mut self,
        steam_id: u64,
        pocket_idx: u32,
        round_id: &ItemId,
    ) -> Result<u32> {
        let e = self
            .find_player_entity(steam_id)
            .ok_or_else(|| anyhow::anyhow!("unknown player {steam_id}"))?;

        // Read the target mag's id + state.
        let (mag_id, current_rounds, current_variant) = {
            let inv = self
                .world
                .get::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("load_rounds: no Inventory"))?;
            let idx = pocket_idx as usize;
            let placed = inv
                .0
                .items
                .get(idx)
                .ok_or_else(|| anyhow::anyhow!("load_rounds: pocket_idx {idx} out of range"))?;
            let stack = &placed.stack;
            let state = stack.magazine_state.clone().unwrap_or_default();
            (stack.id.clone(), state.loaded_rounds, state.variant)
        };

        // Validate + resolve capacity.
        let capacity = {
            let reg = self.world.resource::<ItemRegistry>();
            validate_mag_load(reg, &mag_id, round_id, current_rounds, &current_variant)?
        };

        let room = capacity.saturating_sub(current_rounds);
        if room == 0 {
            return Ok(0);
        }

        // Consume matching ammo from pockets. The ammo stack is a
        // different item id than the magazine so this doesn't
        // invalidate `pocket_idx`.
        let loaded = {
            let mut inv = self
                .world
                .get_mut::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("load_rounds: no Inventory"))?;
            super::inventory::consume_from_stacks(&mut inv, round_id, room)
        };
        if loaded == 0 {
            return Ok(0);
        }

        // Write the new mag state back.
        {
            let mut inv = self
                .world
                .get_mut::<Inventory>(e)
                .ok_or_else(|| anyhow::anyhow!("load_rounds: no Inventory"))?;
            let idx = pocket_idx as usize;
            let placed = inv.0.items.get_mut(idx).ok_or_else(|| {
                anyhow::anyhow!("load_rounds: pocket_idx {idx} disappeared mid-load")
            })?;
            placed.stack.magazine_state = Some(MagazineState {
                loaded_rounds: current_rounds + loaded,
                variant: Some(round_id.clone()),
            });
        }

        self.record_delta(WorldDelta::PocketMagazineLoaded {
            steam_id,
            pocket_idx,
            round_id: round_id.clone(),
            added: loaded,
            total: current_rounds + loaded,
        })?;
        Ok(loaded)
    }

    /// Eject the magazine in `slot_id` back to pockets without
    /// loading a replacement. No-op (returns Ok) if the slot has a
    /// weapon with no mag loaded; errors only on missing player /
    /// missing weapon / unknown slot.
    pub fn eject_magazine(&mut self, steam_id: u64, slot_id: &SlotId) -> Result<()> {
        let e = self
            .find_player_entity(steam_id)
            .ok_or_else(|| anyhow::anyhow!("unknown player {steam_id}"))?;

        let ejected = {
            let mut eq = self
                .world
                .get_mut::<Equipment>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Equipment"))?;
            let equipped =
                eq.0.get_mut(slot_id)
                    .ok_or_else(|| anyhow::anyhow!("eject: slot {slot_id:?} is empty"))?;
            let Some(ws) = equipped.weapon_state.as_mut() else {
                return Err(anyhow::anyhow!(
                    "eject: item in slot {slot_id:?} isn't a weapon"
                ));
            };
            ws.loaded_magazine.take()
        };

        if let Some(mag) = ejected.clone() {
            place_mag_in_pockets(&mut self.world, e, mag)?;
        }

        self.record_delta(WorldDelta::WeaponMagazineEjected {
            steam_id,
            slot_id: slot_id.clone(),
            ejected,
        })?;
        Ok(())
    }

    /// Phase 4D: clear a jam on the weapon at `slot_id`. Resets
    /// `jam_state` to `Cleared` and journals
    /// `WorldDelta::WeaponJamCleared`. Errors if the slot is
    /// empty, the item isn't a weapon, or the weapon wasn't
    /// jammed (caller-side surface: clear-jam input on an
    /// un-jammed weapon is a no-op error, not a panic). Does
    /// **not** repair condition — clearing a jam costs time +
    /// surface UX but doesn't restore wear.
    pub fn clear_weapon_jam(&mut self, steam_id: u64, slot_id: &SlotId) -> Result<()> {
        let e = self
            .find_player_entity(steam_id)
            .ok_or_else(|| anyhow::anyhow!("unknown player {steam_id}"))?;
        let was_jammed = {
            let mut eq = self
                .world
                .get_mut::<Equipment>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Equipment"))?;
            let equipped =
                eq.0.get_mut(slot_id)
                    .ok_or_else(|| anyhow::anyhow!("clear_jam: slot {slot_id:?} is empty"))?;
            let Some(ws) = equipped.weapon_state.as_mut() else {
                return Err(anyhow::anyhow!(
                    "clear_jam: item in slot {slot_id:?} isn't a weapon"
                ));
            };
            let was = ws.jam_state.is_jammed();
            ws.jam_state = crate::components::JamState::Cleared;
            was
        };
        if !was_jammed {
            return Err(anyhow::anyhow!("clear_jam: weapon wasn't jammed"));
        }
        self.record_delta(WorldDelta::WeaponJamCleared {
            steam_id,
            slot_id: slot_id.clone(),
        })?;
        Ok(())
    }

    /// Fire the weapon at `slot_id` with the shooter's current aim
    /// (`aim_yaw` and `aim_pitch` in radians). The sim:
    ///
    /// 1. Looks up the weapon's `WeaponConfig` (range, fire rate, …).
    /// 2. Decrements the loaded magazine; dry-clicks if empty or
    ///    the mag has no variant set. Commit 5 makes the variant-
    ///    missing case a first-class dry click; commit 3 keeps it
    ///    to "the mag has a variant, fire a round of that variant."
    /// 3. Spawns a `Projectile` entity with velocity derived from
    ///    the shooter's aim + the round's muzzle velocity. The
    ///    projectile ticks sim-side in `tick_projectiles`.
    /// 4. Journals `WeaponFired` (for HUD) and `ProjectileSpawned`
    ///    (for client-side tracer FX).
    ///
    /// Dry-click returns `Err` so the bridge can distinguish
    /// successful fire from silent-trigger-pull and flash the HUD.
    /// The GDScript bridge no longer consumes a `WeaponConfig`
    /// return value — hit resolution lives entirely in the sim.
    pub fn fire_weapon(
        &mut self,
        steam_id: u64,
        slot_id: &SlotId,
        aim_yaw: f32,
        aim_pitch: f32,
    ) -> Result<()> {
        let e = self
            .find_player_entity(steam_id)
            .ok_or_else(|| anyhow::anyhow!("unknown player {steam_id}"))?;

        // Pre-validate the weapon (existence + weapon_config). Resolves
        // empty-slot / unknown-item / non-weapon before mag state.
        let weapon_config = weapon_config_at(&self.world, e, slot_id, "fire")?.clone();

        // Phase 4D: jam state gate. If the weapon is currently
        // jammed, fire is a dry click — player must clear the
        // jam first. This sits *before* the mag-state read so
        // a jammed weapon doesn't expend a round on every
        // trigger pull.
        if let Some(eq) = self
            .world
            .get::<crate::components::Equipment>(e)
            .and_then(|eq| eq.0.get(slot_id).cloned())
        {
            if let Some(ws) = eq.weapon_state.as_ref() {
                if ws.jam_state.is_jammed() {
                    return Err(anyhow::anyhow!("fire: weapon is jammed"));
                }
            }
        }

        // Phase 4D: roll for a jam *before* expending a round.
        // Probability comes from `jam_chance_at_condition` against
        // the weapon's current `EquippedWeaponState.condition`.
        // A jammed roll transitions the jam state, emits a
        // `WeaponJammed` delta, and returns Err so the bridge can
        // surface the dry-click + UI prompt. Condition is read
        // here; the wear decrement runs on a *successful* fire
        // below so a jammed round doesn't count as fired.
        let condition = self
            .world
            .get::<crate::components::Equipment>(e)
            .and_then(|eq| eq.0.get(slot_id).cloned())
            .and_then(|eq| eq.weapon_state.map(|ws| ws.condition))
            .unwrap_or(100.0);
        let jam_chance = crate::items::jam_chance_at_condition(condition, &weapon_config);
        if jam_chance > 0.0 {
            let roll: f32 = {
                use rand::SeedableRng;
                let tick = self.current_tick();
                // Salt with steam_id so two players firing on the same
                // tick don't roll identical jams. Cheap deterministic
                // mix; same pattern as the projectile / spawn RNGs.
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
                    tick.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(steam_id),
                );
                rand::Rng::gen(&mut rng)
            };
            if roll < jam_chance {
                // Pick a jam kind by condition — heavier wear
                // biases toward FTE / stovepipe. Tuned to feel
                // gradient rather than rigorous part-source.
                let jam = if condition < 20.0 {
                    crate::components::JamState::FailureToExtract
                } else if condition < 45.0 {
                    crate::components::JamState::Stovepipe
                } else {
                    crate::components::JamState::FailureToFeed
                };
                if let Some(mut eq) = self.world.get_mut::<crate::components::Equipment>(e) {
                    if let Some(equipped) = eq.0.get_mut(slot_id) {
                        let ws = equipped
                            .weapon_state
                            .get_or_insert_with(crate::components::EquippedWeaponState::default);
                        ws.jam_state = jam;
                    }
                }
                self.record_delta(WorldDelta::WeaponJammed {
                    steam_id,
                    slot_id: slot_id.clone(),
                    jam,
                    condition,
                })?;
                return Err(anyhow::anyhow!("fire: weapon jammed"));
            }
        }

        // Decrement the loaded mag's rounds + capture its variant
        // (for projectile ballistics). Dry-click on no-mag /
        // empty-mag / no-variant.
        let (remaining, round_id) = {
            let mag = loaded_magazine_mut(&mut self.world, e, slot_id)
                .ok_or_else(|| anyhow::anyhow!("fire: no magazine loaded"))?;
            let state = mag.magazine_state.get_or_insert(MagazineState::default());
            if state.loaded_rounds == 0 {
                return Err(anyhow::anyhow!("fire: magazine empty"));
            }
            let Some(round_id) = state.variant.clone() else {
                return Err(anyhow::anyhow!(
                    "fire: magazine has no ammo variant loaded (commit 5 wires load_rounds_into_mag)"
                ));
            };
            state.loaded_rounds -= 1;
            (state.loaded_rounds, round_id)
        };

        // Phase 4D: apply wear on a successful fire. Floor at 0
        // so a clapped-out weapon never goes "negative" condition.
        let new_condition = {
            let mut new = condition;
            if let Some(mut eq) = self.world.get_mut::<crate::components::Equipment>(e) {
                if let Some(equipped) = eq.0.get_mut(slot_id) {
                    if let Some(ws) = equipped.weapon_state.as_mut() {
                        ws.condition = (ws.condition - weapon_config.wear_per_shot).max(0.0);
                        new = ws.condition;
                    }
                }
            }
            new
        };
        if (new_condition - condition).abs() > f32::EPSILON {
            self.record_delta(WorldDelta::WeaponConditionChanged {
                steam_id,
                slot_id: slot_id.clone(),
                new_condition,
            })?;
        }

        // Look up the round's ballistic profile. If the mag's
        // variant references an unknown round (modder forgot to
        // ship the ammo def), dry-click rather than panic.
        let muzzle_velocity = {
            let reg = self.world.resource::<ItemRegistry>();
            reg.get(&round_id)
                .and_then(|def| def.ammo_config.as_ref())
                .map(|ac| ac.muzzle_velocity_mps)
                .ok_or_else(|| {
                    anyhow::anyhow!("fire: unknown or non-ammo variant {:?}", round_id)
                })?
        };

        // Resolve shooter position + muzzle offset from
        // `BallisticsConfig`. Muzzle lives `forward_m` ahead of the
        // shooter and `up_m` above feet, rotated by aim_yaw.
        let (origin, velocity) = {
            let bc = self.world.resource::<crate::resources::BallisticsConfig>();
            let shooter_pos = self
                .world
                .get::<Position>(e)
                .map(|p| p.0)
                .ok_or_else(|| anyhow::anyhow!("fire: shooter has no Position"))?;
            let (sin_yaw, cos_yaw) = aim_yaw.sin_cos();
            let (sin_pitch, cos_pitch) = aim_pitch.sin_cos();
            // Forward vector in world space (Godot: +Z = forward,
            // +X = right, +Y = up). Pitch tilts forward up/down.
            let fwd = [cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw];
            let origin = [
                shooter_pos[0] + fwd[0] * bc.muzzle_forward_m,
                shooter_pos[1] + bc.muzzle_up_m + fwd[1] * bc.muzzle_forward_m,
                shooter_pos[2] + fwd[2] * bc.muzzle_forward_m,
            ];
            let velocity = [
                fwd[0] * muzzle_velocity,
                fwd[1] * muzzle_velocity,
                fwd[2] * muzzle_velocity,
            ];
            (origin, velocity)
        };

        let shooter_region = self
            .world
            .get::<crate::components::InRegion>(e)
            .map(|r| r.0)
            .ok_or_else(|| anyhow::anyhow!("fire: shooter has no InRegion"))?;

        let spawned_tick = self.current_tick();
        let projectile_id = self
            .world
            .resource_mut::<crate::resources::ProjectileIdCounter>()
            .mint();

        // Spawn the projectile entity (host-side) and journal both
        // deltas: `WeaponFired` for HUD reactivity, `ProjectileSpawned`
        // for client tracer FX.
        self.world.spawn((
            crate::components::Projectile {
                id: projectile_id,
                source_steam_id: steam_id,
                source_npc_id: None,
                round_id: round_id.clone(),
                pos: origin,
                vel: velocity,
                distance_traveled_m: 0.0,
                max_range_m: weapon_config.range_m,
                spawned_tick,
            },
            Position(origin),
            crate::components::Rotation(aim_yaw),
            crate::components::InRegion(shooter_region),
        ));
        self.record_delta(WorldDelta::WeaponFired {
            steam_id,
            slot_id: slot_id.clone(),
            remaining_rounds: remaining,
        })?;
        let variant = self.resolve_round_variant(&round_id);
        self.record_delta(WorldDelta::ProjectileSpawned {
            id: projectile_id,
            source_steam_id: steam_id,
            source_npc_id: None,
            round_id,
            variant,
            origin,
            velocity,
            max_range_m: weapon_config.range_m,
            spawned_tick,
        })?;
        Ok(())
    }

    /// Phase 4B v2: resolve a round id's variant tag (FMJ / HP /
    /// AP / Tracer / Overpressure) from the item registry. Falls
    /// back to FMJ when the round can't be found or has no
    /// `ammo_config` block — same default as the
    /// `AmmoVariant::default()` and the legacy-snapshot serde
    /// default, so missing-data clients see a sane variant
    /// rather than a panic.
    pub(crate) fn resolve_round_variant(&self, round_id: &ItemId) -> crate::items::AmmoVariant {
        self.world
            .resource::<ItemRegistry>()
            .get(round_id)
            .and_then(|def| def.ammo_config.as_ref())
            .map(|ac| ac.variant)
            .unwrap_or_default()
    }

    /// Test-only: directly set `loaded_rounds` on the magazine
    /// currently loaded into the weapon at `slot_id`. Panics on any
    /// missing state. Real gameplay never calls this — the rounds
    /// count changes through fire + ammo-load flows in phase-1-rounds
    /// and phase-2. Exposed here because loader/reload tests need a
    /// way to seed a partial mag without reaching into private
    /// state.
    pub fn set_equipped_mag_rounds_for_test(
        &mut self,
        steam_id: u64,
        slot_id: &SlotId,
        rounds: u32,
    ) {
        let e = self
            .find_player_entity(steam_id)
            .expect("set_equipped_mag_rounds_for_test: player missing");
        let mag = loaded_magazine_mut(&mut self.world, e, slot_id)
            .expect("set_equipped_mag_rounds_for_test: no mag loaded");
        // Preserve existing variant if any; test helper only changes
        // the round count.
        let variant = mag
            .magazine_state
            .as_ref()
            .and_then(|ms| ms.variant.clone());
        mag.magazine_state = Some(MagazineState {
            loaded_rounds: rounds,
            variant,
        });
    }

    /// Test-only: set both the loaded rounds AND the ammo variant
    /// on the magazine currently loaded into the weapon at
    /// `slot_id`. Phase 2 fire requires a variant; use this helper
    /// to seed a mag without going through the full load_rounds
    /// ammo-pull flow.
    #[doc(hidden)]
    pub fn set_equipped_mag_state_for_test(
        &mut self,
        steam_id: u64,
        slot_id: &SlotId,
        rounds: u32,
        variant: Option<&str>,
    ) {
        let e = self
            .find_player_entity(steam_id)
            .expect("set_equipped_mag_state_for_test: player missing");
        let mag = loaded_magazine_mut(&mut self.world, e, slot_id)
            .expect("set_equipped_mag_state_for_test: no mag loaded");
        mag.magazine_state = Some(MagazineState {
            loaded_rounds: rounds,
            variant: variant.map(ItemId::from),
        });
    }

    /// Phase 4D: test-only setter for the weapon's current
    /// `condition` (0.0–100.0). Used in `tests/weapons_4d.rs`
    /// to put a weapon at sub-threshold condition and exercise
    /// the jam path without firing thousands of rounds first.
    #[doc(hidden)]
    pub fn set_weapon_condition_for_test(
        &mut self,
        steam_id: u64,
        slot_id: &SlotId,
        condition: f32,
    ) {
        let e = self
            .find_player_entity(steam_id)
            .expect("set_weapon_condition_for_test: player missing");
        let mut eq = self
            .world
            .get_mut::<Equipment>(e)
            .expect("set_weapon_condition_for_test: no Equipment");
        let equipped =
            eq.0.get_mut(slot_id)
                .expect("set_weapon_condition_for_test: empty slot");
        let ws = equipped
            .weapon_state
            .get_or_insert_with(crate::components::EquippedWeaponState::default);
        ws.condition = condition.clamp(0.0, 100.0);
    }

    /// Phase 4D: read the weapon's current condition + jam state
    /// at `slot_id`. Test-only. Mutably borrows the sim because
    /// `find_player_entity` runs a Bevy query; no actual state
    /// mutation happens here.
    #[doc(hidden)]
    pub fn weapon_condition_for_test(
        &mut self,
        steam_id: u64,
        slot_id: &SlotId,
    ) -> Option<(f32, crate::components::JamState)> {
        let e = self.find_player_entity(steam_id)?;
        let eq = self.world.get::<Equipment>(e)?;
        let equipped = eq.0.get(slot_id)?;
        let ws = equipped.weapon_state.as_ref()?;
        Some((ws.condition, ws.jam_state))
    }

    /// Phase 4A v1 — spawn a cosmetic projectile from an NPC at
    /// the supplied target position. The projectile flies via the
    /// existing tick (gravity, drag, range despawn) but does **not**
    /// apply damage on hit; NPC-vs-NPC damage still routes through
    /// `npc_combat`'s dice path at fire time. 4A v2 will migrate
    /// damage onto the projectile-hit branch and remove the dice
    /// step.
    ///
    /// Caller supplies the shooter's NPC id, its world position +
    /// region, the target world position, an `accuracy` stat
    /// (0..=100) which drives cone-of-fire jitter, and the
    /// `round_id` to fire. Faction-flavored NPC rounds come from
    /// [`default_npc_round_for_faction`] in Phase 4B v1 —
    /// `round_5_45x39` (intermediate caliber) is the fallback.
    ///
    /// Returns `Ok(())` on a successful spawn, `Err` if the round
    /// def / ammo_config is missing (which would be a registry
    /// problem). `WeaponFired` is **not** journaled — NPC fire
    /// has no per-player HUD reactivity to drive — but the
    /// `ProjectileSpawned` delta IS journaled so client mirrors
    /// see the tracer.
    #[allow(clippy::too_many_arguments)]
    pub fn npc_fire_projectile(
        &mut self,
        shooter_id: crate::components::NpcId,
        shooter_pos: [f32; 3],
        shooter_region: crate::region::RegionId,
        target_pos: [f32; 3],
        accuracy: u8,
        round_id: ItemId,
        rng: &mut impl rand::Rng,
    ) -> Result<()> {
        // Resolve ballistic profile from the round def.
        let muzzle_velocity = {
            let reg = self.world.resource::<ItemRegistry>();
            reg.get(&round_id)
                .and_then(|def| def.ammo_config.as_ref())
                .map(|ac| ac.muzzle_velocity_mps)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "npc_fire_projectile: round {:?} missing or has no ammo_config",
                        round_id
                    )
                })?
        };

        // Aim from shooter muzzle toward target with cone-of-fire
        // jitter scaled by accuracy. Accuracy 100 = 0 rad jitter
        // (perfect aim); accuracy 0 = MAX_CONE_RAD jitter (very
        // bad). Apply jitter as a small yaw + pitch perturbation,
        // not a uniform-on-sphere — most real spread is roll
        // (left/right) anyway. Was 0.18 rad (~10°) which made
        // consecutive NPC shots fan visibly across a wide arc —
        // 0.08 rad (~4.5°) at acc=0 still varies hit-or-miss for
        // bad shooters without reading as "spraying sideways".
        const MAX_CONE_RAD: f32 = 0.08;
        let acc_f = f32::from(accuracy.min(100)) / 100.0;
        let cone = MAX_CONE_RAD * (1.0 - acc_f);

        let bc = self.world.resource::<crate::resources::BallisticsConfig>();
        let muzzle_forward = bc.muzzle_forward_m;
        let muzzle_up = bc.muzzle_up_m;

        // Aim from the *muzzle* (shooter Y + muzzle_up) to the
        // target's center mass — not muzzle-to-feet (which lands
        // a marginal grazing shot on the leg capsule's bottom
        // edge). `target_pos` is assumed to be a foot/ground
        // position; offset by `TARGET_CENTER_MASS_Y` so the
        // shot lands in the torso unless jitter pushes it.
        const TARGET_CENTER_MASS_Y: f32 = 1.2;
        let muzzle_y = shooter_pos[1] + muzzle_up;
        let aim_target_y = target_pos[1] + TARGET_CENTER_MASS_Y;
        let dx = target_pos[0] - shooter_pos[0];
        let dy = aim_target_y - muzzle_y;
        let dz = target_pos[2] - shooter_pos[2];
        let aim_yaw = dx.atan2(dz);
        let horiz = (dx * dx + dz * dz).sqrt().max(0.001);
        let aim_pitch = dy.atan2(horiz);

        // Symmetric jitter in [-cone/2, +cone/2].
        let yaw_jitter = (rng.gen::<f32>() - 0.5) * cone;
        let pitch_jitter = (rng.gen::<f32>() - 0.5) * cone;
        let fire_yaw = aim_yaw + yaw_jitter;
        let fire_pitch = aim_pitch + pitch_jitter;

        let (sin_yaw, cos_yaw) = fire_yaw.sin_cos();
        let (sin_pitch, cos_pitch) = fire_pitch.sin_cos();
        let fwd = [cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw];
        let origin = [
            shooter_pos[0] + fwd[0] * muzzle_forward,
            shooter_pos[1] + muzzle_up + fwd[1] * muzzle_forward,
            shooter_pos[2] + fwd[2] * muzzle_forward,
        ];
        let velocity = [
            fwd[0] * muzzle_velocity,
            fwd[1] * muzzle_velocity,
            fwd[2] * muzzle_velocity,
        ];

        let spawned_tick = self.current_tick();
        let projectile_id = self
            .world
            .resource_mut::<crate::resources::ProjectileIdCounter>()
            .mint();

        // Max range floor — NPC weapons don't have a configured
        // weapon_config yet (4B). Use the npc_combat sight radius
        // ceiling so the projectile despawns roughly where the
        // dice path would have stopped firing anyway.
        const NPC_PROJECTILE_RANGE_M: f32 = 150.0;

        self.world.spawn((
            crate::components::Projectile {
                id: projectile_id,
                source_steam_id: 0,
                source_npc_id: Some(shooter_id),
                round_id: round_id.clone(),
                pos: origin,
                vel: velocity,
                distance_traveled_m: 0.0,
                max_range_m: NPC_PROJECTILE_RANGE_M,
                spawned_tick,
            },
            Position(origin),
            crate::components::Rotation(fire_yaw),
            crate::components::InRegion(shooter_region),
        ));
        let variant = self.resolve_round_variant(&round_id);
        self.record_delta(WorldDelta::ProjectileSpawned {
            id: projectile_id,
            source_steam_id: 0,
            source_npc_id: Some(shooter_id),
            round_id,
            variant,
            origin,
            velocity,
            max_range_m: NPC_PROJECTILE_RANGE_M,
            spawned_tick,
        })?;
        Ok(())
    }
}

/// Phase 4B v1 — faction-flavored NPC round selection. Each
/// faction's "default round" maps to the caliber its loadout
/// most often carries. Coalition / Directorate / Consortium / Homesteaders /
/// Order fire rifle-caliber intermediates; raiders / nomads
/// fire pistol-caliber cheap ammo; Syndicate splits.
///
/// Returns an `ItemId` that must exist in `ammo.toml` — caller
/// (`npc_fire_projectile`) validates by reading the round's
/// `ammo_config`, falling back to logging on missing/malformed
/// rounds rather than panicking.
///
/// Phase 4B v2 will drive this from `factions.toml` so modders
/// can author faction → round mappings without touching Rust.
/// Until then this lookup is the single source of truth.
pub fn default_npc_round_for_faction(faction: &str) -> ItemId {
    let id = match faction {
        "coalition" => "round_5_45x39",
        "directorate" => "round_556x45_m193",
        "consortium" => "round_556x45_m193",
        "homesteaders" => "round_5_45x39",
        "the_order" => "round_762x39",
        "syndicate" => "round_9x19",
        "raiders" => "round_9x18",
        "nomads" => "round_9x18",
        "coalition_vanguard" => "round_5_45x39",
        // Unknown factions (mods, future additions) fall back to
        // the intermediate-caliber neutral default. Same posture
        // as the loot-pool registry's nomads fallback.
        _ => "round_5_45x39",
    };
    ItemId::from(id)
}

/// Walk `Equipment -> slot -> weapon_state -> loaded_magazine` and
/// return a mutable ref to the loaded magazine `ItemInstance`, or
/// `None` if any link is missing. Shared between the `WeaponFired`
/// delta-replay arm and the test-only `set_equipped_mag_rounds_for_test`
/// helper; keeps the four-deep `if-let` chain in one place.
pub(super) fn loaded_magazine_mut<'w>(
    world: &'w mut World,
    e: Entity,
    slot_id: &SlotId,
) -> Option<&'w mut ItemInstance> {
    let eq = world.get_mut::<Equipment>(e)?.into_inner();
    let equipped = eq.0.get_mut(slot_id)?;
    let ws = equipped.weapon_state.as_mut()?;
    ws.loaded_magazine.as_mut()
}

/// Validate that `round_id` can load into `mag_id` given the mag's
/// current (rounds, variant). Returns the mag's capacity on success.
/// Shared by both load-rounds paths (equipped-mag and pocket-mag)
/// so the caliber-match + partial-mag-variant-flip rules live in
/// one place.
fn validate_mag_load(
    registry: &ItemRegistry,
    mag_id: &ItemId,
    round_id: &ItemId,
    current_rounds: u32,
    current_variant: &Option<ItemId>,
) -> Result<u32> {
    let ammo_def = registry
        .get(round_id)
        .ok_or_else(|| anyhow::anyhow!("load_rounds: unknown ammo {:?}", round_id))?;
    let ammo_cfg = ammo_def
        .ammo_config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("load_rounds: {:?} isn't ammo", round_id))?;
    let mag_def = registry
        .get(mag_id)
        .ok_or_else(|| anyhow::anyhow!("load_rounds: unknown mag {:?}", mag_id))?;
    let mag_cfg = mag_def
        .magazine_config
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("load_rounds: {:?} isn't a magazine", mag_id))?;
    if mag_cfg.caliber != ammo_cfg.caliber {
        return Err(anyhow::anyhow!(
            "load_rounds: mag caliber {:?} doesn't match ammo {:?}",
            mag_cfg.caliber,
            ammo_cfg.caliber
        ));
    }
    if current_rounds > 0 {
        if let Some(existing) = current_variant {
            if existing != round_id {
                return Err(anyhow::anyhow!(
                    "load_rounds: mag already holds {:?}; fire out / eject before loading {:?}",
                    existing,
                    round_id
                ));
            }
        }
    }
    Ok(mag_cfg.capacity)
}

/// Initial `EquippedWeaponState` for an item being placed into a
/// paper-doll slot. Returns `Some(default)` if the item's `ItemDef`
/// carries a `weapon_config` block (so the slot should track
/// loaded-magazine state), `None` otherwise. Used by both the live
/// equip path in `world::inventory::Sim::equip` and the journal-
/// replay arm for `WorldDelta::ItemEquipped` so both routes land the
/// same shape.
pub(super) fn init_weapon_state_for(
    id: &ItemId,
    registry: &ItemRegistry,
) -> Option<EquippedWeaponState> {
    registry
        .get(id)
        .and_then(|def| def.weapon_config.as_ref())
        .map(|_| EquippedWeaponState::default())
}

/// Resolve the `WeaponConfig` of the item equipped at `slot_id` on
/// `player_entity`, borrowing from the registry. Errors walk the
/// chain: no Equipment → slot empty → unknown item → item isn't a
/// weapon (no `weapon_config` block).
///
/// `action_label` is spliced into the error text (`"reload: …"` vs
/// `"fire: …"`) so callers can keep their tagging without repeating
/// the lookup chain.
fn weapon_config_at<'w>(
    world: &'w World,
    player_entity: Entity,
    slot_id: &SlotId,
    action_label: &str,
) -> Result<&'w WeaponConfig> {
    let eq = world
        .get::<Equipment>(player_entity)
        .ok_or_else(|| anyhow::anyhow!("{action_label}: player has no Equipment"))?;
    let equipped =
        eq.0.get(slot_id)
            .ok_or_else(|| anyhow::anyhow!("{action_label}: slot {slot_id:?} is empty"))?;
    let reg = world.resource::<ItemRegistry>();
    let def = reg
        .get(&equipped.stack.id)
        .ok_or_else(|| anyhow::anyhow!("{action_label}: unknown item {:?}", equipped.stack.id))?;
    def.weapon_config.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "{action_label}: item {:?} in slot {slot_id:?} isn't a weapon",
            equipped.stack.id
        )
    })
}

/// Place `mag` back in the player's pockets grid. Magazines never
/// stack (each instance carries its own loaded-round count), so this
/// always tries a fresh placement via `find_first_fit_any_rotation`
/// → `place_at`. If pockets are full, surfaces an error — host-side
/// callers (reload / eject) propagate it; the delta-replay path in
/// `persistence::apply_delta` discards it (the mirror state is
/// already divergent and the next snapshot will reconcile).
///
/// Shared by [`Sim::reload_weapon`] / [`Sim::eject_magazine`] and
/// the `WeaponReloaded` / `WeaponMagazineEjected` replay arms so
/// the placement logic lives in exactly one place.
pub(super) fn place_mag_in_pockets(
    world: &mut World,
    player_entity: Entity,
    mag: ItemInstance,
) -> Result<()> {
    let reg = world.resource::<ItemRegistry>().clone();
    let def = reg
        .get(&mag.id)
        .ok_or_else(|| anyhow::anyhow!("place_mag: unknown item {:?}", mag.id))?
        .clone();
    let mut inv = world
        .get_mut::<Inventory>(player_entity)
        .ok_or_else(|| anyhow::anyhow!("place_mag: player has no Inventory"))?;
    let (x, y, rotation) = inventory_grid::find_first_fit_any_rotation(&inv.0, &reg, &def)
        .ok_or_else(|| {
            anyhow::anyhow!("place_mag: pockets full; drop-on-ground lands with the loot slice")
        })?;
    inventory_grid::place_at(&mut inv.0, &reg, mag, x, y, rotation)
        .map_err(|e| anyhow::anyhow!("place_mag: placement failed: {e:?}"))?;
    Ok(())
}
