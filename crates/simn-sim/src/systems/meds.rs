//! Drug effects, pain, and tolerance.
//!
//! All systems here are **pure**: outputs are a deterministic function
//! of component-and-clock state, so no journaling. The persisted
//! `ActiveEffects` and `DrugTolerance` components plus the `SimClock`
//! are enough for replay to re-derive everything.
//!
//! Per `docs/book/src/mechanics/drugs-and-effects.md`, the design tilt
//! is "punishing but balanced for intermediate players":
//! - First dose of any drug never overdoses (tolerance starts at 0).
//! - Stacking 3+ doses in succession triggers overdose; the
//!   disorientation effect lasts ~2 in-world min and is recoverable.
//! - Withdrawal triggers only after sustained heavy use AND a no-dose
//!   window of 4+ in-world hours; lifts when tolerance < 25.
//! - Tolerance decays at 25 units / in-world hour — about 5 real
//!   minutes from a moderate-use level back to safe.
//!
//! See `MedConfig` in `resources.rs` for the tunable thresholds.

use bevy_ecs::prelude::{Query, Res, With};

use crate::components::{
    ActiveEffects, DrugKind, DrugTolerance, EffectKind, Pain, PlayerOwned, WoundKind,
    WoundTreatment, Wounds,
};
use crate::resources::{MedConfig, SimClock};

/// Pain points contributed per severity for an Untreated wound.
/// Bandaged/Stitched/Tourniquet attenuate via the weight table below.
pub const PAIN_PER_SEVERITY: f32 = 5.0;

/// Default per-drug tolerance gain on each `apply_drug` call.
pub fn default_tolerance_gain(drug: DrugKind) -> f32 {
    match drug {
        DrugKind::Painkiller => 15.0,
        DrugKind::Morphine => 30.0,
        DrugKind::Adrenaline => 50.0,
        DrugKind::StimCocktail => 20.0,
        DrugKind::AntiRad => 20.0,
        DrugKind::AntiTox => 20.0,
    }
}

/// Default active-phase duration of each drug, in ticks (50ms each).
pub fn default_active_duration_ticks(drug: DrugKind) -> u64 {
    match drug {
        DrugKind::Painkiller => 1500,   // 5 in-world min
        DrugKind::Morphine => 600,      // 2 in-world min
        DrugKind::Adrenaline => 150,    // 30 in-world sec
        DrugKind::StimCocktail => 1500, // 5 in-world min
        DrugKind::AntiRad => 1,         // instant; effect is the immediate stat change
        DrugKind::AntiTox => 1,
    }
}

/// Default crash/rebound phase duration (ticks). 0 means no crash phase.
pub fn default_crash_duration_ticks(drug: DrugKind) -> u64 {
    match drug {
        DrugKind::Adrenaline => 600,   // 2 in-world min crash
        DrugKind::StimCocktail => 600, // 2 in-world min fatigue rebound
        _ => 0,
    }
}

/// Map a drug to its crash-phase effect kind, when applicable.
pub fn crash_kind(drug: DrugKind) -> Option<EffectKind> {
    match drug {
        DrugKind::Adrenaline => Some(EffectKind::AdrenalineCrash),
        DrugKind::StimCocktail => Some(EffectKind::FatigueRebound),
        _ => None,
    }
}

/// Default primary-effect intensity (the magnitude consumed by
/// downstream systems — pain reduction, regen multiplier, etc.).
pub fn default_intensity(drug: DrugKind) -> f32 {
    match drug {
        DrugKind::Painkiller => 25.0,  // pain reduced by 25
        DrugKind::Morphine => 75.0,    // pain reduced by 75
        DrugKind::Adrenaline => 1.0,   // marker; the revive logic gates on presence
        DrugKind::StimCocktail => 1.5, // regen × 1.5 + stamina_max +30 cosmetic
        DrugKind::AntiRad => 30.0,     // -30 radiation immediately
        DrugKind::AntiTox => 30.0,     // -30 toxicity immediately
    }
}

/// True if `effects` contains an active (non-expired) effect of the
/// given kind at `now`.
pub fn has_active(effects: &ActiveEffects, kind: EffectKind, now: u64) -> bool {
    effects
        .0
        .iter()
        .any(|e| e.kind == kind && now.saturating_sub(e.applied_tick) < e.duration_ticks)
}

/// Sum the intensity of every active effect of the given kind.
pub fn sum_intensity(effects: &ActiveEffects, kind: EffectKind, now: u64) -> f32 {
    effects
        .0
        .iter()
        .filter(|e| e.kind == kind && now.saturating_sub(e.applied_tick) < e.duration_ticks)
        .map(|e| e.intensity)
        .sum()
}

/// Sum the intensity across multiple kinds (e.g. Painkiller +
/// Morphine for total pain relief).
pub fn sum_intensity_any(effects: &ActiveEffects, kinds: &[EffectKind], now: u64) -> f32 {
    kinds.iter().map(|k| sum_intensity(effects, *k, now)).sum()
}

/// Retire effects whose `applied_tick + duration_ticks` is past `now`.
/// Pure: deterministic from snapshot state. Withdrawal-spawn happens
/// in `Sim::tick_active_effects_withdrawal_check` (a separate non-pure
/// step that runs in the schedule), keeping this system noise-free.
pub fn tick_active_effects(
    mut q: Query<&mut ActiveEffects, With<PlayerOwned>>,
    clock: Res<SimClock>,
) {
    let now = clock.tick;
    for mut effects in &mut q {
        effects
            .0
            .retain(|e| now.saturating_sub(e.applied_tick) < e.duration_ticks);
    }
}

/// Decay each drug's tolerance toward 0 at the configured rate.
/// Pure.
pub fn decay_drug_tolerance(
    mut q: Query<&mut DrugTolerance, With<PlayerOwned>>,
    clock: Res<SimClock>,
    cfg: Res<MedConfig>,
) {
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    let decay = cfg.tolerance_decay_per_in_world_sec * dt_s;
    for mut tol in &mut q {
        for (_, v) in tol.0.iter_mut() {
            *v = (*v - decay).max(0.0);
        }
    }
}

/// Derive the player's `Pain` from active wounds and active painkillers.
/// Pure. Stored on the entity so other systems (regen, future aim
/// shake) read a single value without iterating wounds again.
pub fn tick_pain(
    mut q: Query<(&Wounds, &ActiveEffects, &mut Pain), With<PlayerOwned>>,
    clock: Res<SimClock>,
) {
    let now = clock.tick;
    for (wounds, effects, mut pain) in &mut q {
        let mut raw = 0.0_f32;
        for (_, w) in &wounds.0 {
            if !matches!(w.kind, WoundKind::Bleed) {
                continue; // Future kinds add their own pain weight.
            }
            let weight = match w.treatment {
                WoundTreatment::Untreated | WoundTreatment::Disinfected => 1.0,
                WoundTreatment::Bandaged => 0.5,
                WoundTreatment::Stitched => 0.25,
                WoundTreatment::Tourniquet | WoundTreatment::WoundPacked => 0.25,
                WoundTreatment::Healed => 0.0,
            };
            raw += f32::from(w.severity) * PAIN_PER_SEVERITY * weight;
        }
        let relief = sum_intensity_any(
            effects,
            &[EffectKind::Painkiller, EffectKind::Morphine],
            now,
        );
        pain.0 = (raw - relief).clamp(0.0, 100.0);
    }
}
