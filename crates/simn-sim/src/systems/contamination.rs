//! Radiation + Toxicity passive decay and HP-gate.
//!
//! Pure per-tick system, parallel to `apply_survival_effects` for the
//! hunger/thirst HP gate. Both meters decay slowly toward zero; above
//! `MedConfig::contamination_hp_threshold` (default 80) each
//! contributes a slow HP drain on the torso. Anti-rad / anti-tox drugs
//! reduce these explicitly via `Sim::set_radiation` / `set_toxicity`,
//! which journal — that path is the one the player can act on.

use bevy_ecs::prelude::{Query, Res, With};

use crate::components::{BodyParts, Contamination, Health, PlayerOwned};
use crate::resources::{MedConfig, SimClock};

pub fn tick_contamination(
    mut q: Query<(&mut Contamination, &mut BodyParts, Option<&mut Health>), With<PlayerOwned>>,
    clock: Res<SimClock>,
    cfg: Res<MedConfig>,
) {
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    let rad_decay = cfg.radiation_decay_per_in_world_sec * dt_s;
    let tox_decay = cfg.toxicity_decay_per_in_world_sec * dt_s;
    let hp_drain = cfg.contamination_hp_drain_per_sec * dt_s;
    for (mut c, mut parts, health) in &mut q {
        c.radiation = (c.radiation - rad_decay).clamp(0.0, 100.0);
        c.toxicity = (c.toxicity - tox_decay).clamp(0.0, 100.0);

        let mut loss = 0.0;
        if c.radiation > cfg.contamination_hp_threshold {
            loss += hp_drain;
        }
        if c.toxicity > cfg.contamination_hp_threshold {
            loss += hp_drain;
        }
        if loss > 0.0 {
            parts.torso = (parts.torso - loss).max(0.0);
            let vital = parts.vital_min();
            if let Some(mut h) = health {
                h.current = vital.min(h.max);
            }
        }
    }
}
