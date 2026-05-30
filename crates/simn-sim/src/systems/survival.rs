//! Per-tick survival drain + degraded-function effects.
//!
//! Two pure systems:
//!
//! - [`drain_survival_stats`] decrements hunger/thirst/fatigue at
//!   configurable per-in-world-second rates. Players only (filtered by
//!   `PlayerOwned`). Matches `docs/survival-and-crafting-plan.md` §3.2:
//!   a full hunger bar drains over ~8 in-game hours of active play.
//!   With the default `WorldTime::DEFAULT_DAY_LENGTH` (7200 in-world
//!   seconds = 1 in-world day), 8 in-game hours = 2400 in-world
//!   seconds, giving `100 / 2400 ≈ 0.0417` hunger units per in-world
//!   second. Thirst drains slightly faster (water matters more, per
//!   §3.3); fatigue ramps on action and is currently passive (sprint
//!   cost lands with the movement layer in Step 2+).
//! - [`apply_survival_effects`] couples low survival meters to other
//!   stats per §3.3: HP trickle when hunger or thirst bottom out. The
//!   stamina-regen halving is handled inside [`super::stamina`] so
//!   regen and survival reads happen in one query pass.
//!
//! Both systems are pure: outputs are a deterministic function of the
//! component's last-known value plus elapsed ticks, so they don't
//! journal. On crash, up to one snapshot interval (~30s) of drain is
//! lost; documented and acceptable.

use bevy_ecs::prelude::{Query, Res, With};

use crate::components::{BodyParts, Health, PlayerOwned, SurvivalStats};
use crate::resources::SimClock;

/// Hunger drain in points per in-world second. 100 / 2400 ≈ 0.0417,
/// matching "~8 in-game hours per full bar" with a 7200-second
/// in-world day.
pub const HUNGER_DRAIN_PER_SEC: f32 = 100.0 / 2400.0;
/// Thirst drains a little faster than hunger — water matters more.
/// Tunable; ~6 in-game hours per full bar at default day length.
pub const THIRST_DRAIN_PER_SEC: f32 = 100.0 / 1800.0;
/// Fatigue passive drain. Sprinting and combat will add bursts on top
/// once the movement / action layers exist.
pub const FATIGUE_DRAIN_PER_SEC: f32 = 100.0 / 4800.0;

/// Hunger threshold below which stamina regen is halved. §3.3.
pub const HUNGER_REGEN_PENALTY_THRESHOLD: f32 = 30.0;
/// Thirst threshold below which stamina regen is halved.
pub const THIRST_REGEN_PENALTY_THRESHOLD: f32 = 50.0;
/// Hunger threshold below which the player takes slow HP damage.
pub const HUNGER_HP_DRAIN_THRESHOLD: f32 = 10.0;
/// Thirst threshold below which the player takes slow HP damage.
pub const THIRST_HP_DRAIN_THRESHOLD: f32 = 20.0;
/// HP loss from low hunger or thirst, per in-world second. Tuned so
/// the player still has minutes to find food/water — never instant
/// death from a survival stat alone (§3.3).
pub const STARVE_HP_LOSS_PER_SEC: f32 = 0.25;

/// Drain hunger, thirst, and fatigue toward zero each tick. Player
/// entities only.
pub fn drain_survival_stats(
    mut q: Query<&mut SurvivalStats, With<PlayerOwned>>,
    clock: Res<SimClock>,
) {
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    for mut s in &mut q {
        s.hunger = (s.hunger - HUNGER_DRAIN_PER_SEC * dt_s).max(0.0);
        s.thirst = (s.thirst - THIRST_DRAIN_PER_SEC * dt_s).max(0.0);
        s.fatigue = (s.fatigue - FATIGUE_DRAIN_PER_SEC * dt_s).max(0.0);
    }
}

/// Slow HP trickle when hunger or thirst bottom out. Operates on the
/// torso slot of [`BodyParts`] (and mirrors to aggregate `Health`) —
/// keeps the death path consistent with regular damage, but limited
/// enough that no survival stat alone kills in under a minute.
pub fn apply_survival_effects(
    mut q: Query<(&SurvivalStats, &mut BodyParts, Option<&mut Health>), With<PlayerOwned>>,
    clock: Res<SimClock>,
) {
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    for (stats, mut parts, health) in &mut q {
        let mut loss = 0.0;
        if stats.hunger < HUNGER_HP_DRAIN_THRESHOLD {
            loss += STARVE_HP_LOSS_PER_SEC * dt_s;
        }
        if stats.thirst < THIRST_HP_DRAIN_THRESHOLD {
            loss += STARVE_HP_LOSS_PER_SEC * dt_s;
        }
        if loss <= 0.0 {
            continue;
        }
        parts.torso = (parts.torso - loss).max(0.0);
        let vital = parts.vital_min();
        if let Some(mut h) = health {
            h.current = vital.min(h.max);
        }
    }
}
