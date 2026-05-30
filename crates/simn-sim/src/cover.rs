//! Cover/concealment system: authored cover volumes with material-
//! based projectile penetration. Designer places `CoverVolumeMarker3D`
//! nodes in the scene; the bridge registers them as `CoverVolume`s.
//! Combat resolution queries `check_cover_between` to gate damage.
//!
//! Materials are loaded from `content/cover_materials.toml`.

use bevy_ecs::prelude::Resource;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::region::RegionId;

// ── Material definitions ────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverMaterialId {
    Concrete,
    Brick,
    SteelThick,
    SteelThin,
    WoodThick,
    WoodThin,
    Sandbag,
    Earth,
    Glass,
    Vegetation,
    VehicleBody,
}

impl CoverMaterialId {
    pub fn all() -> &'static [CoverMaterialId] {
        use CoverMaterialId::*;
        &[
            Concrete,
            Brick,
            SteelThick,
            SteelThin,
            WoodThick,
            WoodThin,
            Sandbag,
            Earth,
            Glass,
            Vegetation,
            VehicleBody,
        ]
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct CoverMaterialDef {
    pub name: String,
    pub id: CoverMaterialId,
    pub protection_class: u8,
    pub provides_cover: bool,
    pub provides_concealment: bool,
    pub ricochet_chance: f32,
    pub spall_factor: f32,
    pub durability_per_mm: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct CoverMaterialsFile {
    material: Vec<CoverMaterialDef>,
}

static MATERIAL_TABLE: OnceLock<HashMap<CoverMaterialId, CoverMaterialDef>> = OnceLock::new();

pub fn material_table() -> &'static HashMap<CoverMaterialId, CoverMaterialDef> {
    // Cover materials are an embedded-only carve-out: `material_def`
    // is called on the penetration/tick path with no content handle,
    // so the table is process-global and read from the embedded pack.
    MATERIAL_TABLE.get_or_init(|| {
        let toml_str = crate::ContentSource::Embedded
            .read_str("cover_materials.toml")
            .expect("embedded cover_materials.toml present");
        let file: CoverMaterialsFile =
            toml::from_str(&toml_str).expect("cover_materials.toml parse");
        file.material.into_iter().map(|m| (m.id, m)).collect()
    })
}

pub fn material_def(id: CoverMaterialId) -> Option<&'static CoverMaterialDef> {
    material_table().get(&id)
}

// ── Penetration resolution ──────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PenetrationResult {
    Stopped,
    PartialPenetration { spall_damage_frac: f32 },
    FullPenetration { residual_class: u8 },
}

pub fn can_penetrate(
    projectile_pen_class: u8,
    material_id: CoverMaterialId,
    thickness_mm: f32,
    angle_of_incidence: f32,
) -> PenetrationResult {
    let Some(mat) = material_def(material_id) else {
        return PenetrationResult::FullPenetration {
            residual_class: projectile_pen_class,
        };
    };

    if !mat.provides_cover {
        return PenetrationResult::FullPenetration {
            residual_class: projectile_pen_class,
        };
    }

    let effective_thickness = thickness_mm / angle_of_incidence.cos().abs().max(0.1);
    let thickness_factor = (effective_thickness / 100.0).min(2.0);
    let effective_protection = mat.protection_class as f32 + thickness_factor;

    if (projectile_pen_class as f32) >= effective_protection + 1.0 {
        let residual = projectile_pen_class.saturating_sub(mat.protection_class);
        PenetrationResult::FullPenetration {
            residual_class: residual,
        }
    } else if (projectile_pen_class as f32) >= effective_protection - 0.5 {
        PenetrationResult::PartialPenetration {
            spall_damage_frac: mat.spall_factor,
        }
    } else {
        PenetrationResult::Stopped
    }
}

// ── Cover volume data ───────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoverHeight {
    Low,
    High,
    Full,
}

#[derive(Clone, Debug)]
pub struct CoverVolume {
    pub id: u64,
    pub region: RegionId,
    pub pos: [f32; 3],
    pub half_extents: [f32; 3],
    pub rotation: [f32; 4],
    pub material_id: CoverMaterialId,
    pub height: CoverHeight,
    pub thickness_mm: f32,
    pub destructible: bool,
    pub health: f32,
    pub max_health: f32,
}

#[derive(Resource, Default)]
pub struct CoverVolumes {
    pub by_region: HashMap<RegionId, Vec<CoverVolume>>,
    next_id: u64,
    /// NPC currently using each cover volume. Prevents multiple
    /// NPCs from claiming the same cover and stacking.
    pub occupied_by: HashMap<u64, crate::components::NpcId>,
}

/// Result of a ray-vs-cover intersection test.
#[derive(Clone, Debug)]
pub struct CoverHit {
    pub volume_id: u64,
    pub material_id: CoverMaterialId,
    pub thickness_mm: f32,
    pub angle: f32,
}

impl CoverVolumes {
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn clear_region(&mut self, region: RegionId) {
        self.by_region.remove(&region);
    }

    /// Test whether any cover volumes lie between shooter and target
    /// in the given region. Returns all intersected volumes with
    /// material + thickness for penetration calculation.
    pub fn check_cover_between(
        &self,
        region: RegionId,
        shooter_pos: [f32; 3],
        target_pos: [f32; 3],
    ) -> Vec<CoverHit> {
        let Some(volumes) = self.by_region.get(&region) else {
            return Vec::new();
        };

        let mut hits = Vec::new();
        let dx = target_pos[0] - shooter_pos[0];
        let dy = target_pos[1] - shooter_pos[1];
        let dz = target_pos[2] - shooter_pos[2];
        let ray_len_sq = dx * dx + dy * dy + dz * dz;
        if ray_len_sq < 0.01 {
            return hits;
        }
        let ray_len = ray_len_sq.sqrt();
        let rd = [dx / ray_len, dy / ray_len, dz / ray_len];

        for vol in volumes {
            if vol.destructible && vol.health <= 0.0 {
                continue;
            }
            if ray_aabb_intersects(shooter_pos, rd, ray_len, vol.pos, vol.half_extents) {
                let angle = ray_aabb_angle(shooter_pos, rd, vol.pos, vol.half_extents);
                hits.push(CoverHit {
                    volume_id: vol.id,
                    material_id: vol.material_id,
                    thickness_mm: vol.thickness_mm,
                    angle,
                });
            }
        }
        hits
    }

    /// Find the nearest unoccupied cover volume. Skips volumes
    /// already claimed by another NPC (prevents stacking) and
    /// volumes directly past the enemy.
    pub fn nearest_cover(
        &self,
        region: RegionId,
        pos: [f32; 3],
        threat_dir: [f32; 2],
        max_dist: f32,
        _height_pref: CoverHeight,
    ) -> Option<&CoverVolume> {
        self.nearest_cover_for(region, pos, threat_dir, max_dist, _height_pref, None)
    }

    /// Same as `nearest_cover` but allows specifying the requesting
    /// NPC so their own occupied cover isn't rejected.
    pub fn nearest_cover_for(
        &self,
        region: RegionId,
        pos: [f32; 3],
        threat_dir: [f32; 2],
        max_dist: f32,
        _height_pref: CoverHeight,
        self_id: Option<crate::components::NpcId>,
    ) -> Option<&CoverVolume> {
        let volumes = self.by_region.get(&region)?;
        let max_dist_sq = max_dist * max_dist;
        let mut best: Option<(&CoverVolume, f32)> = None;

        for vol in volumes {
            if vol.destructible && vol.health <= 0.0 {
                continue;
            }
            if let Some(occupant) = self.occupied_by.get(&vol.id) {
                if self_id != Some(*occupant) {
                    continue;
                }
            }
            let dx = vol.pos[0] - pos[0];
            let dz = vol.pos[2] - pos[2];
            let dist_sq = dx * dx + dz * dz;
            if dist_sq > max_dist_sq {
                continue;
            }
            let len = dist_sq.sqrt().max(0.01);
            let to_cover_norm = [dx / len, dz / len];
            let dot = to_cover_norm[0] * threat_dir[0] + to_cover_norm[1] * threat_dir[1];
            if dot < -0.8 {
                continue;
            }
            if best.is_none() || dist_sq < best.unwrap().1 {
                best = Some((vol, dist_sq));
            }
        }
        best.map(|(v, _)| v)
    }

    pub fn claim_cover(&mut self, volume_id: u64, npc: crate::components::NpcId) {
        self.occupied_by.insert(volume_id, npc);
    }

    pub fn release_cover(&mut self, volume_id: u64, npc: crate::components::NpcId) {
        if self.occupied_by.get(&volume_id) == Some(&npc) {
            self.occupied_by.remove(&volume_id);
        }
    }

    pub fn release_all_for_npc(&mut self, npc: crate::components::NpcId) {
        self.occupied_by.retain(|_, v| *v != npc);
    }

    /// Apply damage to a destructible cover volume. Returns true if
    /// the volume was destroyed by this hit.
    pub fn damage_cover(&mut self, region: RegionId, volume_id: u64, damage: f32) -> bool {
        let Some(volumes) = self.by_region.get_mut(&region) else {
            return false;
        };
        for vol in volumes.iter_mut() {
            if vol.id == volume_id && vol.destructible {
                vol.health = (vol.health - damage).max(0.0);
                return vol.health <= 0.0;
            }
        }
        false
    }
}

// ── Ray-AABB intersection (axis-aligned, no rotation for v1) ────

fn ray_aabb_intersects(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    ray_len: f32,
    aabb_center: [f32; 3],
    aabb_half: [f32; 3],
) -> bool {
    let mut tmin = 0.0_f32;
    let mut tmax = ray_len;

    for i in 0..3 {
        let inv_d = if ray_dir[i].abs() > 1e-8 {
            1.0 / ray_dir[i]
        } else {
            1e8_f32.copysign(ray_dir[i])
        };
        let t1 = (aabb_center[i] - aabb_half[i] - ray_origin[i]) * inv_d;
        let t2 = (aabb_center[i] + aabb_half[i] - ray_origin[i]) * inv_d;
        let (t_near, t_far) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        tmin = tmin.max(t_near);
        tmax = tmax.min(t_far);
        if tmin > tmax {
            return false;
        }
    }
    true
}

fn ray_aabb_angle(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    aabb_center: [f32; 3],
    _aabb_half: [f32; 3],
) -> f32 {
    let to_center = [
        aabb_center[0] - ray_origin[0],
        aabb_center[1] - ray_origin[1],
        aabb_center[2] - ray_origin[2],
    ];
    let len =
        (to_center[0] * to_center[0] + to_center[1] * to_center[1] + to_center[2] * to_center[2])
            .sqrt()
            .max(0.001);
    let norm = [to_center[0] / len, to_center[1] / len, to_center[2] / len];
    let dot = ray_dir[0] * norm[0] + ray_dir[1] * norm[1] + ray_dir[2] * norm[2];
    dot.abs().acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_table_loads() {
        let table = material_table();
        assert!(table.contains_key(&CoverMaterialId::Concrete));
        assert!(table.contains_key(&CoverMaterialId::Glass));
        assert!(table.len() >= 10);
    }

    #[test]
    fn concrete_stops_pistol() {
        let result = can_penetrate(1, CoverMaterialId::Concrete, 300.0, 0.0);
        assert_eq!(result, PenetrationResult::Stopped);
    }

    #[test]
    fn rifle_penetrates_wood_thin() {
        let result = can_penetrate(3, CoverMaterialId::WoodThin, 20.0, 0.0);
        assert!(matches!(result, PenetrationResult::FullPenetration { .. }));
    }

    #[test]
    fn glass_is_concealment_only() {
        let result = can_penetrate(0, CoverMaterialId::Glass, 10.0, 0.0);
        assert!(matches!(result, PenetrationResult::FullPenetration { .. }));
    }

    #[test]
    fn ray_hits_aabb() {
        let origin = [0.0, 1.0, 0.0];
        let dir = [1.0, 0.0, 0.0];
        let center = [5.0, 1.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        assert!(ray_aabb_intersects(origin, dir, 10.0, center, half));
    }

    #[test]
    fn ray_misses_aabb() {
        let origin = [0.0, 1.0, 0.0];
        let dir = [0.0, 0.0, 1.0];
        let center = [5.0, 1.0, 0.0];
        let half = [1.0, 1.0, 1.0];
        assert!(!ray_aabb_intersects(origin, dir, 10.0, center, half));
    }

    #[test]
    fn cover_between_finds_volume() {
        let mut cv = CoverVolumes::default();
        let region = 0u32;
        let id = cv.next_id();
        cv.by_region.entry(region).or_default().push(CoverVolume {
            id,
            region,
            pos: [5.0, 1.0, 0.0],
            half_extents: [1.0, 2.0, 2.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            material_id: CoverMaterialId::Concrete,
            height: CoverHeight::High,
            thickness_mm: 300.0,
            destructible: false,
            health: 100.0,
            max_health: 100.0,
        });
        let hits = cv.check_cover_between(region, [0.0, 1.0, 0.0], [10.0, 1.0, 0.0]);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].material_id, CoverMaterialId::Concrete);
    }

    #[test]
    fn destructible_cover_destroyed() {
        let mut cv = CoverVolumes::default();
        let region = 0u32;
        let id = cv.next_id();
        cv.by_region.entry(region).or_default().push(CoverVolume {
            id,
            region,
            pos: [5.0, 1.0, 0.0],
            half_extents: [1.0, 1.0, 1.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            material_id: CoverMaterialId::Glass,
            height: CoverHeight::High,
            thickness_mm: 10.0,
            destructible: true,
            health: 5.0,
            max_health: 5.0,
        });
        let destroyed = cv.damage_cover(region, id, 10.0);
        assert!(destroyed);
        let hits = cv.check_cover_between(region, [0.0, 1.0, 0.0], [10.0, 1.0, 0.0]);
        assert!(hits.is_empty());
    }
}
