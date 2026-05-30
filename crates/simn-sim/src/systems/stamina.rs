//! Passive stamina regeneration.
//!
//! Pure per-tick system: adds `regen_per_sec * dt` to every entity
//! with a [`Stamina`] component, clamped at `max`. Not journaled —
//! regen is a deterministic function of last-known stamina and
//! elapsed ticks, so the snapshot + `SimClock::tick` are enough to
//! recover the right value. On crash, up to one snapshot interval of
//! regen drift is lost; documented, fine for now.
//!
//! Modifiers:
//! - `SurvivalStats` (hunger < 30 OR thirst < 50): halve regen
//!   (per `docs/book/src/planning/survival-and-crafting-plan.md` §3.3).
//! - `Pain > MedConfig::pain_regen_threshold` (default 50): halve
//!   regen (§3.3 / §4.4).
//! - Inventory weight > `InventoryConfig::weight_cap_kg`: multiply
//!   regen by `overweight_regen_mult` (default 0.5). Same shape as
//!   the low-hunger penalty; encumbered players recover slower. The
//!   cap is soft — pickup itself never fails, the hit shows up here.
//! - Active `StimCocktail` effect: multiply regen by intensity.
//! - Active `Withdrawal` effect: subtract 5 from `regen_per_sec`
//!   before scaling.
//! - Active `OverdoseDisorientation`: halve regen.
//! - Active `AdrenalineCrash`: cap regen at 0 for the crash window.
//!
//! NPCs (no `Pain` / `ActiveEffects` components) regen at full rate.

use bevy_ecs::prelude::{Query, Res};

use crate::components::{ActiveEffects, EffectKind, Inventory, Pain, Stamina, SurvivalStats};
use crate::items::ItemRegistry;
use crate::resources::{InventoryConfig, MedConfig, SimClock};
use crate::systems::meds::{has_active, sum_intensity};
use crate::systems::survival::{HUNGER_REGEN_PENALTY_THRESHOLD, THIRST_REGEN_PENALTY_THRESHOLD};

#[allow(clippy::type_complexity)]
pub fn regen_stamina(
    mut q: Query<(
        &mut Stamina,
        Option<&SurvivalStats>,
        Option<&Pain>,
        Option<&ActiveEffects>,
        Option<&Inventory>,
    )>,
    clock: Res<SimClock>,
    cfg: Res<MedConfig>,
    inv_cfg: Res<InventoryConfig>,
    items: Res<ItemRegistry>,
) {
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    let now = clock.tick;
    for (mut s, survival, pain, effects, inventory) in &mut q {
        let mut base = s.regen_per_sec;
        if let Some(eff) = effects {
            if has_active(eff, EffectKind::Withdrawal, now) {
                base = (base - 5.0).max(0.0);
            }
            if has_active(eff, EffectKind::AdrenalineCrash, now) {
                s.current = s.current.min(s.max); // no regen this tick
                continue;
            }
        }
        let mut mult = 1.0_f32;
        if let Some(sv) = survival {
            if sv.hunger < HUNGER_REGEN_PENALTY_THRESHOLD
                || sv.thirst < THIRST_REGEN_PENALTY_THRESHOLD
            {
                mult *= 0.5;
            }
        }
        if let Some(p) = pain {
            if p.0 > cfg.pain_regen_threshold {
                mult *= 0.5;
            }
        }
        if let Some(inv) = inventory {
            let carried: f32 = inv
                .0
                .items
                .iter()
                .map(|placed| {
                    items
                        .get(&placed.stack.id)
                        .map(|d| d.weight * placed.stack.count as f32)
                        .unwrap_or(0.0)
                })
                .sum();
            if carried > inv_cfg.weight_cap_kg {
                mult *= inv_cfg.overweight_regen_mult;
            }
        }
        if let Some(eff) = effects {
            let stim = sum_intensity(eff, EffectKind::StimCocktail, now);
            if stim > 0.0 {
                mult *= stim;
            }
            if has_active(eff, EffectKind::OverdoseDisorientation, now) {
                mult *= 0.5;
            }
        }
        let next = s.current + base * mult * dt_s;
        s.current = next.min(s.max);
    }
}
