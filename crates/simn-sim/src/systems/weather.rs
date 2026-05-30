//! Global weather transitions.
//!
//! Placeholder: a single world-wide weather state rolls to a new
//! `Weather` variant every in-game hour (scaled to sim ticks).
//! Transitions are seeded from the sim tick, so replays are
//! deterministic. Real weather (per-region fronts, wind, pressure)
//! lands with the per-region state slice later.
//!
//! The sky controller on the Godot side reads `WeatherState` each
//! frame and lerps visuals, so the change from clear → overcast
//! looks gradual even though the sim flips in one tick.

use bevy_ecs::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::resources::{SimClock, Weather, WeatherState, WorldTime};

/// In-game seconds between weather rolls. Each roll picks a new
/// `next` from the transition table; when `transitions_at_tick`
/// arrives, `current` flips to `next`. Picking two rolls ahead
/// means the sky controller has a known future target it can lerp
/// toward (nicer visuals than snap-switching).
const ROLL_INTERVAL_IN_GAME_SECONDS: f32 = 1800.0; // 30 in-game minutes

pub fn advance_weather(
    clock: Res<SimClock>,
    time: Res<WorldTime>,
    mut state: ResMut<WeatherState>,
) {
    // Convert 30 in-game minutes to sim ticks. sim dt is fixed_dt_ms.
    // in-game seconds per real second = DEFAULT_DAY_LENGTH-compressed
    // factor. seconds_of_day advances by (86400 / day_length_seconds)
    // in-game seconds per real second.
    let real_seconds_per_sim_tick = f32::from(clock.fixed_dt_ms as u16) / 1000.0;
    let in_game_per_real = if time.day_length_seconds > 0.0 {
        86400.0 / time.day_length_seconds
    } else {
        1.0
    };
    let in_game_per_tick = real_seconds_per_sim_tick * in_game_per_real;
    if in_game_per_tick <= 0.0 {
        return;
    }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let roll_interval_ticks = ((ROLL_INTERVAL_IN_GAME_SECONDS / in_game_per_tick).max(1.0)) as u64;

    // First-ever transition: pick an initial target.
    if state.transitions_at_tick == 0 && clock.tick > 0 {
        state.transitions_at_tick = clock.tick.saturating_add(roll_interval_ticks);
        let mut rng = ChaCha8Rng::seed_from_u64(clock.tick.wrapping_mul(0x9E37_79B1));
        state.next = pick_next(state.current, &mut rng);
        return;
    }

    // Time to flip? Advance current → next and re-roll next.
    if clock.tick >= state.transitions_at_tick && state.transitions_at_tick != 0 {
        state.current = state.next;
        let mut rng = ChaCha8Rng::seed_from_u64(clock.tick.wrapping_mul(0xA5A5_5A5A));
        state.next = pick_next(state.current, &mut rng);
        state.transitions_at_tick = clock.tick.saturating_add(roll_interval_ticks);
    }
}

/// Markov-ish transition weights. PNW-climate placeholder:
///
/// - The common axis is `PartlyCloudy ↔ Overcast ↔ Drizzle ↔
///   LightRain ↔ HeavyRain ↔ Thunderstorm` — one step at a time,
///   with a strong bias toward staying put (weather doesn't flip
///   every 30 minutes in reality; most transitions are gradual).
/// - `MarineLayer` branches off Clear / PartlyCloudy on the
///   overnight/dawn side; it burns off back to Clear/Partly more
///   often than it thickens to Overcast.
/// - `Fog` is the denser inland cousin — reachable from Overcast
///   or MarineLayer, escapes to Overcast.
/// - `Windstorm` associates with frontal systems: reachable from
///   LightRain/HeavyRain/Overcast, resolves back through HeavyRain
///   or Overcast.
/// - `SmokeHaze` is a dry-season attractor; reachable only from
///   Clear/PartlyCloudy and it *sticks* — it takes a real front to
///   clear (strongest escape is back to Overcast, hinting at a
///   front arriving).
///
/// Seasonal biasing and time-of-day effects (marine-layer at dawn,
/// thunderstorms in summer afternoons) are placeholder-parked —
/// those fold into the #20 per-region-state slice with a season
/// resource.
fn pick_next(current: Weather, rng: &mut ChaCha8Rng) -> Weather {
    let weights: &[(Weather, u32)] = match current {
        Weather::Clear => &[
            (Weather::Clear, 5),
            (Weather::PartlyCloudy, 4),
            (Weather::MarineLayer, 2),
            (Weather::SmokeHaze, 1),
        ],
        Weather::PartlyCloudy => &[
            (Weather::Clear, 3),
            (Weather::PartlyCloudy, 5),
            (Weather::Overcast, 3),
            (Weather::MarineLayer, 1),
        ],
        Weather::Overcast => &[
            (Weather::PartlyCloudy, 2),
            (Weather::Overcast, 6),
            (Weather::Fog, 1),
            (Weather::Drizzle, 3),
            (Weather::Windstorm, 1),
        ],
        Weather::MarineLayer => &[
            (Weather::Clear, 3),
            (Weather::PartlyCloudy, 3),
            (Weather::MarineLayer, 4),
            (Weather::Overcast, 2),
            (Weather::Fog, 1),
        ],
        Weather::Fog => &[
            (Weather::Fog, 5),
            (Weather::Overcast, 3),
            (Weather::MarineLayer, 2),
            (Weather::Drizzle, 1),
        ],
        Weather::Drizzle => &[
            (Weather::Overcast, 3),
            (Weather::Drizzle, 5),
            (Weather::LightRain, 3),
            (Weather::Fog, 1),
        ],
        Weather::LightRain => &[
            (Weather::Drizzle, 3),
            (Weather::LightRain, 5),
            (Weather::HeavyRain, 2),
            (Weather::Overcast, 1),
            (Weather::Windstorm, 1),
        ],
        Weather::HeavyRain => &[
            (Weather::LightRain, 3),
            (Weather::HeavyRain, 4),
            (Weather::Thunderstorm, 2),
            (Weather::Windstorm, 2),
            (Weather::Overcast, 1),
        ],
        Weather::Windstorm => &[
            (Weather::Windstorm, 3),
            (Weather::HeavyRain, 3),
            (Weather::LightRain, 2),
            (Weather::Overcast, 2),
        ],
        Weather::Thunderstorm => &[
            (Weather::Thunderstorm, 2),
            (Weather::HeavyRain, 4),
            (Weather::LightRain, 2),
        ],
        Weather::SmokeHaze => &[
            (Weather::SmokeHaze, 7),
            (Weather::Clear, 1),
            (Weather::Overcast, 2),
        ],
    };
    let total: u32 = weights.iter().map(|(_, w)| *w).sum();
    if total == 0 {
        return current;
    }
    let mut roll = rng.gen_range(0..total);
    for (w, n) in weights {
        if roll < *n {
            return *w;
        }
        roll -= *n;
    }
    current
}
