//! Advances the [`SimClock`] by one tick.

use bevy_ecs::prelude::ResMut;

use crate::resources::SimClock;

pub fn advance_clock(mut clock: ResMut<SimClock>) {
    clock.tick = clock.tick.wrapping_add(1);
}
