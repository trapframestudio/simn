//! Patrol zone system. Divides each region into a grid of patrol
//! zones and assigns each squad to a zone. All objectives (patrol,
//! guard, rest, wander) are constrained to the assigned zone.
//! This is the structural fix for base-gravity-well clumping:
//! squads spread across the map because zones are distinct.

use bevy_ecs::prelude::*;
use std::collections::HashMap;

use crate::region::RegionId;

const ZONE_SIZE_M: f32 = 500.0;
const HALF_REGION_EXTENT: f32 = 1800.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ZoneId {
    pub region: RegionId,
    pub x: i32,
    pub z: i32,
}

impl ZoneId {
    pub fn from_pos(region: RegionId, pos: [f32; 3]) -> Self {
        Self {
            region,
            x: (pos[0] / ZONE_SIZE_M).floor() as i32,
            z: (pos[2] / ZONE_SIZE_M).floor() as i32,
        }
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.x as f32 + 0.5) * ZONE_SIZE_M,
            0.0,
            (self.z as f32 + 0.5) * ZONE_SIZE_M,
        ]
    }

    pub fn random_point_in(&self, rng: &mut impl rand::Rng) -> [f32; 3] {
        let x = self.x as f32 * ZONE_SIZE_M + rng.gen_range(0.0..ZONE_SIZE_M);
        let z = self.z as f32 * ZONE_SIZE_M + rng.gen_range(0.0..ZONE_SIZE_M);
        [x, 0.0, z]
    }
}

#[derive(Resource, Default)]
pub struct PatrolZones {
    pub assignments: HashMap<u64, ZoneId>,
    zone_squad_count: HashMap<ZoneId, u32>,
}

impl PatrolZones {
    pub fn assign(&mut self, group_id: u64, zone: ZoneId) {
        if let Some(old) = self.assignments.insert(group_id, zone) {
            if let Some(count) = self.zone_squad_count.get_mut(&old) {
                *count = count.saturating_sub(1);
            }
        }
        *self.zone_squad_count.entry(zone).or_insert(0) += 1;
    }

    pub fn release(&mut self, group_id: u64) {
        if let Some(zone) = self.assignments.remove(&group_id) {
            if let Some(count) = self.zone_squad_count.get_mut(&zone) {
                *count = count.saturating_sub(1);
            }
        }
    }

    pub fn zone_for(&self, group_id: u64) -> Option<ZoneId> {
        self.assignments.get(&group_id).copied()
    }

    pub fn squad_count_in(&self, zone: &ZoneId) -> u32 {
        self.zone_squad_count.get(zone).copied().unwrap_or(0)
    }

    /// Pick the least-populated zone in the region for a new squad.
    pub fn pick_zone(&self, region: RegionId, rng: &mut impl rand::Rng) -> ZoneId {
        let grid_half = (HALF_REGION_EXTENT / ZONE_SIZE_M).ceil() as i32;
        let mut best_zone = ZoneId { region, x: 0, z: 0 };
        let mut best_count = u32::MAX;
        let mut candidates = 0u32;

        for x in -grid_half..grid_half {
            for z in -grid_half..grid_half {
                let zone = ZoneId { region, x, z };
                let count = self.squad_count_in(&zone);
                if count < best_count {
                    best_count = count;
                    best_zone = zone;
                    candidates = 1;
                } else if count == best_count {
                    candidates += 1;
                    if rng.gen_range(0..candidates) == 0 {
                        best_zone = zone;
                    }
                }
            }
        }
        best_zone
    }
}
