//! Wound treatment API on `Sim`.
//!
//! All seven treatment paths (bandage / tourniquet / remove-tourniquet
//! / disinfectant / stitch / wound-pack / antibiotics) plus the
//! read-only `wounds_on_player` accessor. Every mutation emits a
//! `WoundTreatmentChanged` delta (or `EffectApplied` for antibiotics)
//! so the journal + broadcast buffer stay in sync.
//!
//! **Treatment pipelines:**
//! - Light bleed (severity ≤ 3): `Untreated → [Disinfected] → Bandaged → [Stitched] → Healed`
//! - Heavy bleed (severity ≥ 4): `Untreated → Tourniquet | WoundPacked → [Stitched] → Healed`
//!
//! Auto-healing (bandaged/stitched → healed) is driven by
//! `age_and_heal_wounds` in the tick schedule. This module only
//! handles player-initiated treatment transitions.

use anyhow::Result;

use crate::components::{
    ActiveEffect, ActiveEffects, BodyPart, EffectKind, NpcId, Wound, WoundId, WoundKind,
    WoundTreatment, Wounds,
};
use crate::delta::WorldDelta;
use crate::resources::SimClock;

use super::{find_npc_in, Sim};

impl Sim {
    /// Apply a bandage to the most-severe **light** Bleed wound
    /// (severity ≤ 3) on the given body part. Accepts both `Untreated`
    /// and `Disinfected` source states; the latter is the
    /// no-infection-risk path. Bandage is the wrong tool for heavy
    /// bleed (severity ≥ 4) — that requires a tourniquet or wound
    /// pack first; this returns explicit `Err` so the triage UI can
    /// surface the right hint. Journaled.
    pub fn apply_bandage(&mut self, steam_id: u64, part: BodyPart) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let target_id_and_state = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Wounds"))?;
            let mut chosen: Option<usize> = None;
            let mut chosen_sev: u8 = 0;
            for (i, (_, w)) in wounds.0.iter().enumerate() {
                if w.body_part != part
                    || !matches!(w.kind, WoundKind::Bleed)
                    || !matches!(
                        w.treatment,
                        WoundTreatment::Untreated | WoundTreatment::Disinfected
                    )
                    || w.severity > 3
                {
                    continue;
                }
                if w.severity > chosen_sev {
                    chosen_sev = w.severity;
                    chosen = Some(i);
                }
            }
            let Some(idx) = chosen else {
                let any_heavy = wounds.0.iter().any(|(_, w)| {
                    w.body_part == part
                        && matches!(w.kind, WoundKind::Bleed)
                        && matches!(
                            w.treatment,
                            WoundTreatment::Untreated | WoundTreatment::Disinfected
                        )
                        && w.severity >= 4
                });
                if any_heavy {
                    return Err(anyhow::anyhow!(
                        "bandage cannot treat heavy bleed on {part:?}; apply tourniquet or wound pack first"
                    ));
                }
                return Err(anyhow::anyhow!("no light bleed to bandage on {part:?}"));
            };
            let (id, w) = &mut wounds.0[idx];
            w.treatment = WoundTreatment::Bandaged;
            w.treatment_changed_tick = now;
            (*id, WoundTreatment::Bandaged)
        };
        self.record_delta(WorldDelta::WoundTreatmentChanged {
            steam_id,
            wound_id: target_id_and_state.0,
            new_state: target_id_and_state.1,
            changed_tick: now,
        })?;
        Ok(())
    }

    /// Apply antiseptic to all `Untreated` Bleed wounds on the given
    /// part — flips them to `Disinfected`, which prevents infection
    /// from setting in. Subsequent `apply_bandage` works on either
    /// state. Idempotent. Journals one record per affected wound.
    pub fn apply_disinfectant(&mut self, steam_id: u64, part: BodyPart) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Wounds"))?;
            let mut out = Vec::new();
            for (id, w) in wounds.0.iter_mut() {
                if w.body_part == part
                    && matches!(w.kind, WoundKind::Bleed)
                    && matches!(w.treatment, WoundTreatment::Untreated)
                {
                    w.treatment = WoundTreatment::Disinfected;
                    w.treatment_changed_tick = now;
                    out.push(*id);
                }
            }
            out
        };
        if changed.is_empty() {
            return Err(anyhow::anyhow!(
                "no untreated wounds to disinfect on {part:?}"
            ));
        }
        for id in changed {
            self.record_delta(WorldDelta::WoundTreatmentChanged {
                steam_id,
                wound_id: id,
                new_state: WoundTreatment::Disinfected,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// Apply a stitch to all `Bandaged`, `Tourniquet`, or `WoundPacked`
    /// wounds on the given part — flips them to `Stitched`, which
    /// halves the heal time. Closes both the light-bleed and
    /// heavy-bleed pipelines. Idempotent. Journals per affected wound.
    pub fn apply_stitch(&mut self, steam_id: u64, part: BodyPart) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Wounds"))?;
            let mut out = Vec::new();
            for (id, w) in wounds.0.iter_mut() {
                if w.body_part == part
                    && matches!(
                        w.treatment,
                        WoundTreatment::Bandaged
                            | WoundTreatment::Tourniquet
                            | WoundTreatment::WoundPacked
                    )
                {
                    w.treatment = WoundTreatment::Stitched;
                    w.treatment_changed_tick = now;
                    // Stitch closes a tourniqueted wound — necrosis stops.
                    w.tourniquet_started_tick = None;
                    out.push(*id);
                }
            }
            out
        };
        if changed.is_empty() {
            return Err(anyhow::anyhow!(
                "no bandaged/tourniqueted/wound-packed wound to stitch on {part:?}"
            ));
        }
        for id in changed {
            self.record_delta(WorldDelta::WoundTreatmentChanged {
                steam_id,
                wound_id: id,
                new_state: WoundTreatment::Stitched,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// Apply a wound pack (pressure dressing) to **untreated** Bleed
    /// wounds on the given part — flips them straight to `WoundPacked`
    /// regardless of severity. The no-cost alternative to a tourniquet
    /// for heavy bleed: same effect (stops bleed) but without the
    /// necrosis timer. Idempotent.
    pub fn apply_wound_pack(&mut self, steam_id: u64, part: BodyPart) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Wounds"))?;
            let mut out = Vec::new();
            for (id, w) in wounds.0.iter_mut() {
                if w.body_part == part
                    && matches!(w.kind, WoundKind::Bleed)
                    && matches!(
                        w.treatment,
                        WoundTreatment::Untreated | WoundTreatment::Disinfected
                    )
                {
                    w.treatment = WoundTreatment::WoundPacked;
                    w.treatment_changed_tick = now;
                    out.push(*id);
                }
            }
            out
        };
        if changed.is_empty() {
            return Err(anyhow::anyhow!("no untreated wounds to pack on {part:?}"));
        }
        for id in changed {
            self.record_delta(WorldDelta::WoundTreatmentChanged {
                steam_id,
                wound_id: id,
                new_state: WoundTreatment::WoundPacked,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// Apply antibiotics — spawns an `AntibioticsActive` effect that
    /// `tick_infection` consumes to clear infection on every infected
    /// wound that's been infected for ≥ `MedConfig::antibiotics_clear_ticks`.
    /// Returns `Err` only if the player doesn't exist; applying with
    /// no infected wounds is a no-op (the effect just expires unused).
    pub fn apply_antibiotics(&mut self, steam_id: u64) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let cfg = *self.world.resource::<crate::resources::MedConfig>();
        let id = self
            .world
            .resource_mut::<crate::resources::EffectIdCounter>()
            .mint();
        let effect = ActiveEffect {
            id,
            kind: EffectKind::AntibioticsActive,
            applied_tick: now,
            duration_ticks: cfg.antibiotics_clear_ticks + 1, // active long enough to clear
            intensity: 1.0,
        };
        if let Some(mut effects) = self.world.get_mut::<ActiveEffects>(e) {
            effects.0.push(effect);
        } else {
            self.world.entity_mut(e).insert(ActiveEffects(vec![effect]));
        }
        self.record_delta(WorldDelta::EffectApplied {
            steam_id,
            effect_id: id,
            kind: EffectKind::AntibioticsActive,
            applied_tick: now,
            duration_ticks: effect.duration_ticks,
            intensity: 1.0,
        })?;
        Ok(())
    }

    /// Apply a tourniquet to **all** untreated/disinfected Bleed
    /// wounds on the given body part. Stops bleed regardless of
    /// severity (the emergency option). Starts the necrosis timer
    /// (`tourniquet_started_tick = now`) — see [`crate::resources::MedConfig::necrosis_warning_ticks`].
    /// Idempotent. Journals one record per affected wound.
    pub fn apply_tourniquet(&mut self, steam_id: u64, part: BodyPart) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Wounds"))?;
            let mut out = Vec::new();
            for (id, w) in wounds.0.iter_mut() {
                if w.body_part == part
                    && matches!(w.kind, WoundKind::Bleed)
                    && matches!(
                        w.treatment,
                        WoundTreatment::Untreated | WoundTreatment::Disinfected
                    )
                {
                    w.treatment = WoundTreatment::Tourniquet;
                    w.treatment_changed_tick = now;
                    w.tourniquet_started_tick = Some(now);
                    out.push(*id);
                }
            }
            out
        };
        for id in changed {
            self.record_delta(WorldDelta::WoundTreatmentChanged {
                steam_id,
                wound_id: id,
                new_state: WoundTreatment::Tourniquet,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// Remove a tourniquet from every wound on the given part — they
    /// revert to `Untreated` and resume bleeding (and clear the
    /// necrosis timer). Step 6 protocol: tourniquet → stitch (which
    /// closes the wound and clears the timer in one step). Removing
    /// without stitching is the "wait, I bandaged below the
    /// tourniquet" path.
    pub fn remove_tourniquet(&mut self, steam_id: u64, part: BodyPart) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Wounds"))?;
            let mut out = Vec::new();
            for (id, w) in wounds.0.iter_mut() {
                if w.body_part == part && matches!(w.treatment, WoundTreatment::Tourniquet) {
                    w.treatment = WoundTreatment::Untreated;
                    w.treatment_changed_tick = now;
                    w.tourniquet_started_tick = None;
                    out.push(*id);
                }
            }
            out
        };
        for id in changed {
            self.record_delta(WorldDelta::WoundTreatmentChanged {
                steam_id,
                wound_id: id,
                new_state: WoundTreatment::Untreated,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// Read-only list of wounds on a player, in storage order. Empty
    /// vec for an uninjured player or unknown steam_id.
    pub fn wounds_on_player(&mut self, steam_id: u64) -> Vec<(WoundId, Wound)> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Vec::new();
        };
        self.world
            .get::<Wounds>(e)
            .map(|w| w.0.clone())
            .unwrap_or_default()
    }

    // ---------- NPC treatment API ----------
    //
    // Every method below mirrors its player twin (`apply_bandage`,
    // `apply_tourniquet`, etc.) byte-for-byte with the entity resolver
    // swapped (`find_npc_in` instead of `find_player_entity`) and
    // journaling a parallel `NpcWoundTreatmentChanged` / `NpcEffectApplied`
    // variant. Same error message shapes, same treatment ordering
    // rules, same per-wound journal record count.

    /// NPC twin of [`Self::apply_bandage`].
    pub fn apply_bandage_npc(&mut self, id: NpcId, part: BodyPart) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let target_id_and_state = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("npc {id:?} has no Wounds"))?;
            let mut chosen: Option<usize> = None;
            let mut chosen_sev: u8 = 0;
            for (i, (_, w)) in wounds.0.iter().enumerate() {
                if w.body_part != part
                    || !matches!(w.kind, WoundKind::Bleed)
                    || !matches!(
                        w.treatment,
                        WoundTreatment::Untreated | WoundTreatment::Disinfected
                    )
                    || w.severity > 3
                {
                    continue;
                }
                if w.severity > chosen_sev {
                    chosen_sev = w.severity;
                    chosen = Some(i);
                }
            }
            let Some(idx) = chosen else {
                let any_heavy = wounds.0.iter().any(|(_, w)| {
                    w.body_part == part
                        && matches!(w.kind, WoundKind::Bleed)
                        && matches!(
                            w.treatment,
                            WoundTreatment::Untreated | WoundTreatment::Disinfected
                        )
                        && w.severity >= 4
                });
                if any_heavy {
                    return Err(anyhow::anyhow!(
                        "bandage cannot treat heavy bleed on {part:?}; apply tourniquet or wound pack first"
                    ));
                }
                return Err(anyhow::anyhow!("no light bleed to bandage on {part:?}"));
            };
            let (wid, w) = &mut wounds.0[idx];
            w.treatment = WoundTreatment::Bandaged;
            w.treatment_changed_tick = now;
            (*wid, WoundTreatment::Bandaged)
        };
        self.record_delta(WorldDelta::NpcWoundTreatmentChanged {
            id,
            wound_id: target_id_and_state.0,
            new_state: target_id_and_state.1,
            changed_tick: now,
        })?;
        Ok(())
    }

    /// NPC twin of [`Self::apply_disinfectant`].
    pub fn apply_disinfectant_npc(&mut self, id: NpcId, part: BodyPart) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("npc {id:?} has no Wounds"))?;
            let mut out = Vec::new();
            for (wid, w) in wounds.0.iter_mut() {
                if w.body_part == part
                    && matches!(w.kind, WoundKind::Bleed)
                    && matches!(w.treatment, WoundTreatment::Untreated)
                {
                    w.treatment = WoundTreatment::Disinfected;
                    w.treatment_changed_tick = now;
                    out.push(*wid);
                }
            }
            out
        };
        if changed.is_empty() {
            return Err(anyhow::anyhow!(
                "no untreated wounds to disinfect on {part:?}"
            ));
        }
        for wid in changed {
            self.record_delta(WorldDelta::NpcWoundTreatmentChanged {
                id,
                wound_id: wid,
                new_state: WoundTreatment::Disinfected,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// NPC twin of [`Self::apply_stitch`].
    pub fn apply_stitch_npc(&mut self, id: NpcId, part: BodyPart) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("npc {id:?} has no Wounds"))?;
            let mut out = Vec::new();
            for (wid, w) in wounds.0.iter_mut() {
                if w.body_part == part
                    && matches!(
                        w.treatment,
                        WoundTreatment::Bandaged
                            | WoundTreatment::Tourniquet
                            | WoundTreatment::WoundPacked
                    )
                {
                    w.treatment = WoundTreatment::Stitched;
                    w.treatment_changed_tick = now;
                    w.tourniquet_started_tick = None;
                    out.push(*wid);
                }
            }
            out
        };
        if changed.is_empty() {
            return Err(anyhow::anyhow!(
                "no bandaged/tourniqueted/wound-packed wound to stitch on {part:?}"
            ));
        }
        for wid in changed {
            self.record_delta(WorldDelta::NpcWoundTreatmentChanged {
                id,
                wound_id: wid,
                new_state: WoundTreatment::Stitched,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// NPC twin of [`Self::apply_wound_pack`].
    pub fn apply_wound_pack_npc(&mut self, id: NpcId, part: BodyPart) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("npc {id:?} has no Wounds"))?;
            let mut out = Vec::new();
            for (wid, w) in wounds.0.iter_mut() {
                if w.body_part == part
                    && matches!(w.kind, WoundKind::Bleed)
                    && matches!(
                        w.treatment,
                        WoundTreatment::Untreated | WoundTreatment::Disinfected
                    )
                {
                    w.treatment = WoundTreatment::WoundPacked;
                    w.treatment_changed_tick = now;
                    out.push(*wid);
                }
            }
            out
        };
        if changed.is_empty() {
            return Err(anyhow::anyhow!("no untreated wounds to pack on {part:?}"));
        }
        for wid in changed {
            self.record_delta(WorldDelta::NpcWoundTreatmentChanged {
                id,
                wound_id: wid,
                new_state: WoundTreatment::WoundPacked,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// NPC twin of [`Self::apply_antibiotics`]. Spawns an
    /// `AntibioticsActive` effect on the NPC; `tick_infection` (now
    /// iterates NPCs too) consumes it to clear infection on every
    /// infected wound.
    pub fn apply_antibiotics_npc(&mut self, id: NpcId) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let cfg = *self.world.resource::<crate::resources::MedConfig>();
        let effect_id = self
            .world
            .resource_mut::<crate::resources::EffectIdCounter>()
            .mint();
        let effect = ActiveEffect {
            id: effect_id,
            kind: EffectKind::AntibioticsActive,
            applied_tick: now,
            duration_ticks: cfg.antibiotics_clear_ticks + 1,
            intensity: 1.0,
        };
        if let Some(mut effects) = self.world.get_mut::<ActiveEffects>(e) {
            effects.0.push(effect);
        } else {
            self.world.entity_mut(e).insert(ActiveEffects(vec![effect]));
        }
        self.record_delta(WorldDelta::NpcEffectApplied {
            id,
            effect_id,
            kind: EffectKind::AntibioticsActive,
            applied_tick: now,
            duration_ticks: effect.duration_ticks,
            intensity: 1.0,
        })?;
        Ok(())
    }

    /// NPC twin of [`Self::apply_tourniquet`].
    pub fn apply_tourniquet_npc(&mut self, id: NpcId, part: BodyPart) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("npc {id:?} has no Wounds"))?;
            let mut out = Vec::new();
            for (wid, w) in wounds.0.iter_mut() {
                if w.body_part == part
                    && matches!(w.kind, WoundKind::Bleed)
                    && matches!(
                        w.treatment,
                        WoundTreatment::Untreated | WoundTreatment::Disinfected
                    )
                {
                    w.treatment = WoundTreatment::Tourniquet;
                    w.treatment_changed_tick = now;
                    w.tourniquet_started_tick = Some(now);
                    out.push(*wid);
                }
            }
            out
        };
        for wid in changed {
            self.record_delta(WorldDelta::NpcWoundTreatmentChanged {
                id,
                wound_id: wid,
                new_state: WoundTreatment::Tourniquet,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// NPC twin of [`Self::remove_tourniquet`].
    pub fn remove_tourniquet_npc(&mut self, id: NpcId, part: BodyPart) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let now = self.world.resource::<SimClock>().tick;
        let changed: Vec<WoundId> = {
            let mut wounds = self
                .world
                .get_mut::<Wounds>(e)
                .ok_or_else(|| anyhow::anyhow!("npc {id:?} has no Wounds"))?;
            let mut out = Vec::new();
            for (wid, w) in wounds.0.iter_mut() {
                if w.body_part == part && matches!(w.treatment, WoundTreatment::Tourniquet) {
                    w.treatment = WoundTreatment::Untreated;
                    w.treatment_changed_tick = now;
                    w.tourniquet_started_tick = None;
                    out.push(*wid);
                }
            }
            out
        };
        for wid in changed {
            self.record_delta(WorldDelta::NpcWoundTreatmentChanged {
                id,
                wound_id: wid,
                new_state: WoundTreatment::Untreated,
                changed_tick: now,
            })?;
        }
        Ok(())
    }

    /// Read-only list of wounds on an NPC, in storage order. Empty
    /// vec for an uninjured NPC or unknown id.
    pub fn wounds_on_npc(&mut self, id: NpcId) -> Vec<(WoundId, Wound)> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Vec::new();
        };
        self.world
            .get::<Wounds>(e)
            .map(|w| w.0.clone())
            .unwrap_or_default()
    }
}
