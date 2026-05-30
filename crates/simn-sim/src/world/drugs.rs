//! Drug application API on `Sim` — the entry point for the meds
//! stack's active effects.
//!
//! `apply_drug` handles tolerance bumping, overdose detection, the
//! primary-effect spawn, the deferred crash-phase spawn (for Stim /
//! Adrenaline), and the stat-mutation path for immediate-effect drugs
//! (AntiRad / AntiTox / Adrenaline revive). Tolerance + effect
//! lifecycle tuning lives in [`crate::resources::MedConfig`].
//!
//! Tolerance decay + effect aging happen in the tick schedule
//! (`decay_drug_tolerance`, `tick_active_effects`) — this module only
//! handles the discrete `apply_drug` event.

use anyhow::Result;
use bevy_ecs::prelude::*;

use crate::components::{
    ActiveEffect, ActiveEffects, BodyPart, BodyParts, Contamination, DrugKind, DrugTolerance,
    EffectKind,
};
use crate::delta::WorldDelta;
use crate::resources::SimClock;
use crate::systems::meds::{
    crash_kind, default_active_duration_ticks, default_crash_duration_ticks, default_intensity,
    default_tolerance_gain, has_active,
};

use super::{DrugOutcome, Sim};

impl Sim {
    /// Apply a drug. Single-use is always safe. Returns
    /// [`DrugOutcome::Overdose`] if `tolerance > MedConfig::overdose_threshold`
    /// AND another dose of the same drug is currently active (the
    /// player stacked too aggressively). On overdose: an
    /// `OverdoseDisorientation` effect spawns instead of the normal
    /// drug effect; tolerance still bumps. On normal application: the
    /// primary effect spawns, plus a deferred crash effect for drugs
    /// that have one (Adrenaline, Stim). Tolerance bumps either way.
    /// Anti-rad / anti-tox apply their stat change immediately.
    /// Adrenaline checks `vital_min < 10` and bumps body parts toward
    /// the revive threshold.
    pub fn apply_drug(&mut self, steam_id: u64, drug: DrugKind) -> Result<DrugOutcome> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let cfg = *self.world.resource::<crate::resources::MedConfig>();
        let now = self.world.resource::<SimClock>().tick;

        let tol_now = self
            .world
            .get::<DrugTolerance>(e)
            .map(|t| t.get(drug))
            .unwrap_or(0.0);
        let already_active = self
            .world
            .get::<ActiveEffects>(e)
            .map(|ef| has_active(ef, drug.primary_effect(), now))
            .unwrap_or(false);

        let overdose = tol_now > cfg.overdose_threshold && already_active;

        // Bump tolerance regardless of overdose path.
        let new_tol = (tol_now + default_tolerance_gain(drug)).clamp(0.0, 100.0);
        if let Some(mut tol) = self.world.get_mut::<DrugTolerance>(e) {
            tol.set(drug, new_tol);
        } else {
            self.world
                .entity_mut(e)
                .insert(DrugTolerance(vec![(drug, new_tol)]));
        }
        self.record_delta(WorldDelta::ToleranceChanged {
            steam_id,
            drug,
            value: new_tol,
        })?;

        if overdose {
            self.spawn_effect(
                steam_id,
                e,
                EffectKind::OverdoseDisorientation,
                now,
                600, // 2 in-world min disorientation
                1.0,
            )?;
            return Ok(DrugOutcome::Overdose);
        }

        let kind = drug.primary_effect();
        let active_dur = default_active_duration_ticks(drug);
        let intensity = default_intensity(drug);

        // Anti-rad / anti-tox apply their stat change immediately —
        // their "active" effect is just a marker.
        match drug {
            DrugKind::AntiRad => {
                let cur = self
                    .world
                    .get::<Contamination>(e)
                    .map(|c| c.radiation)
                    .unwrap_or(0.0);
                self.set_radiation(steam_id, (cur - 30.0).max(0.0))?;
                // Spec §4.4: anti-rad raises tox slightly.
                let cur_tox = self
                    .world
                    .get::<Contamination>(e)
                    .map(|c| c.toxicity)
                    .unwrap_or(0.0);
                self.set_toxicity(steam_id, (cur_tox + 5.0).clamp(0.0, 100.0))?;
            }
            DrugKind::AntiTox => {
                let cur = self
                    .world
                    .get::<Contamination>(e)
                    .map(|c| c.toxicity)
                    .unwrap_or(0.0);
                self.set_toxicity(steam_id, (cur - 30.0).max(0.0))?;
            }
            DrugKind::Adrenaline => {
                // Revive: if vitals are critically low, bump torso to
                // 30% of max so the player isn't bleeding out the next tick.
                let bp_now = self.world.get::<BodyParts>(e).copied();
                if let Some(bp) = bp_now {
                    if bp.vital_min() < 10.0 {
                        let target = BodyParts::DEFAULT_MAX * 0.30;
                        if bp.head < target {
                            self.heal_part(steam_id, BodyPart::Head, target - bp.head)?;
                        }
                        if bp.torso < target {
                            self.heal_part(steam_id, BodyPart::Torso, target - bp.torso)?;
                        }
                    }
                }
            }
            _ => {}
        }

        self.spawn_effect(steam_id, e, kind, now, active_dur, intensity)?;

        // Schedule the crash phase as a deferred effect.
        if let Some(ck) = crash_kind(drug) {
            let crash_dur = default_crash_duration_ticks(drug);
            if crash_dur > 0 {
                self.spawn_effect(steam_id, e, ck, now + active_dur, crash_dur, 1.0)?;
            }
        }

        Ok(DrugOutcome::Effect)
    }

    /// Internal: mint an `EffectId`, push the `ActiveEffect`, journal.
    fn spawn_effect(
        &mut self,
        steam_id: u64,
        entity: Entity,
        kind: EffectKind,
        applied_tick: u64,
        duration_ticks: u64,
        intensity: f32,
    ) -> Result<()> {
        let id = self
            .world
            .resource_mut::<crate::resources::EffectIdCounter>()
            .mint();
        let effect = ActiveEffect {
            id,
            kind,
            applied_tick,
            duration_ticks,
            intensity,
        };
        if let Some(mut effects) = self.world.get_mut::<ActiveEffects>(entity) {
            effects.0.push(effect);
        } else {
            self.world
                .entity_mut(entity)
                .insert(ActiveEffects(vec![effect]));
        }
        self.record_delta(WorldDelta::EffectApplied {
            steam_id,
            effect_id: id,
            kind,
            applied_tick,
            duration_ticks,
            intensity,
        })?;
        Ok(())
    }
}
