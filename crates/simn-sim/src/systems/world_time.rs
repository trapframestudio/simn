//! Advances [`WorldTime`] each tick.
//!
//! Pure per-tick system: deterministic from the resource's current
//! value and `SimClock::fixed_dt_ms`. Not journaled; recovered from
//! the latest snapshot on load (so a crash costs up to a snapshot
//! interval of time-of-day drift, which is invisible to day/night
//! rendering).

use bevy_ecs::prelude::{Res, ResMut};

use crate::resources::{SimClock, WorldTime};

pub fn advance_world_time(mut time: ResMut<WorldTime>, clock: Res<SimClock>) {
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    time.seconds_of_day += dt_s;
    while time.seconds_of_day >= time.day_length_seconds {
        time.seconds_of_day -= time.day_length_seconds;
        time.day = time.day.wrapping_add(1);
    }
}
