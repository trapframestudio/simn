//! Host-side projectile tick — gravity + drag integration and
//! swept-ray hit resolution against humanoid hitboxes.
//!
//! Runs once per `Sim::tick` after the main schedule but before
//! the snapshot interval check. Not a Bevy `System` because it
//! needs to mutate `Equipment` / `BodyParts` / `Wounds` through
//! the existing `Sim::apply_damage_to_npc_part` method (which
//! journals `SetNpcBodyPart` + `NpcWoundAdded` for us). Keeping
//! the tick as a free function on `&mut Sim` gives us that
//! single-source-of-truth behavior for free.
//!
//! Mirror sims skip this — projectiles are host-authoritative.
//! Clients read `ProjectileSpawned` / `ProjectileImpacted` deltas
//! as pure FX hooks; the mirror sim never owns `Projectile`
//! entities.

use anyhow::Result;
use bevy_ecs::prelude::*;

use crate::components::{
    BodyPart, Equipment, InRegion, LastDamager, Npc, NpcId, PlayerOwned, Position, Projectile,
    RecentAttackers, Rotation,
};
use crate::delta::WorldDelta;
use crate::items::{ArmorConfig, ItemId, ItemRegistry};
use crate::resources::{BallisticsConfig, SimClock};

use super::hitbox;
use super::Sim;

impl Sim {
    /// Advance every in-flight projectile by one sim tick. Called
    /// from `Sim::tick` on authoritative sims (mirror sims skip —
    /// they replay impact deltas as FX). Per projectile:
    ///
    /// 1. Integrate drag + gravity.
    /// 2. Walk NPC candidates in the spatial-hash cells along the
    ///    swept segment; first swept-ray hit wins.
    /// 3. On hit: compute damage (phase-3 wires the full formula;
    ///    this commit does flat `damage_soft` so the system is
    ///    testable end-to-end), call
    ///    `apply_damage_to_npc_part`, emit `ProjectileImpacted`,
    ///    despawn the projectile entity.
    /// 4. On out-of-range: emit `ProjectileImpacted { hit_npc:
    ///    None, ... }` at the terminal position, despawn.
    ///
    /// Commit 3 swaps the flat damage for the full pen-vs-armor
    /// formula + `BallisticsConfig`-driven constants.
    pub(crate) fn tick_projectiles(&mut self) -> Result<()> {
        // Collect per-projectile plans to avoid mutating while
        // iterating the query. Each plan records what to do after
        // this tick: just-advance, hit an NPC, or despawn.
        /// Target struck by a projectile this tick.
        enum HitTarget {
            Npc(NpcId),
            Player(u64),
        }

        enum Plan {
            /// Keep flying — update `pos`, `vel`, `distance`.
            Advance {
                entity: Entity,
                new_pos: [f32; 3],
                new_vel: [f32; 3],
                new_distance: f32,
            },
            /// Impact at `hit_pos` on `target`'s `part`. Despawns
            /// the projectile entity, applies damage, emits delta.
            /// Phase 4A v2: `source_npc_id` drives attribution
            /// writes (LastDamager / RecentAttackers / blackboard
            /// UnderFireAt / kill credit) when the shooter was an
            /// NPC. Player-fired projectiles carry `None` and skip
            /// those writes — the player path tracks its own
            /// damage events separately.
            Hit {
                entity: Entity,
                projectile_id: crate::components::ProjectileId,
                round_id: ItemId,
                hit_pos: [f32; 3],
                target: HitTarget,
                part: BodyPart,
                source_npc_id: Option<NpcId>,
                impact_speed_mps: f32,
            },
            /// Out of range. Despawns the entity, emits a
            /// null-target impact delta.
            Despawn {
                entity: Entity,
                projectile_id: crate::components::ProjectileId,
                terminal_pos: [f32; 3],
            },
        }

        let dt_s = f32::from(
            u16::try_from(self.world.resource::<SimClock>().fixed_dt_ms).unwrap_or(u16::MAX),
        ) / 1000.0;
        let gravity = self.world.resource::<BallisticsConfig>().gravity_mps2;

        // Snapshot NPC candidates (entity, npc_id, pos, yaw, region)
        // before mutating. We need region to filter projectiles to
        // same-region NPCs only (cross-region hits don't make sense
        // and the spatial hash is keyed by region anyway). Sorted
        // by `NpcId` so two same-seed sims resolve same-tick ties
        // identically — bevy archetype iteration isn't stable
        // across sim instances.
        let mut npc_snapshot: Vec<(
            Entity,
            crate::components::NpcId,
            [f32; 3],
            f32,
            crate::region::RegionId,
        )> = {
            let mut q = self
                .world
                .query::<(Entity, &Npc, &Position, &Rotation, &InRegion)>();
            q.iter(&self.world)
                .map(|(e, n, p, r, reg)| (e, n.id, p.0, r.0, reg.0))
                .collect()
        };
        npc_snapshot.sort_by_key(|(_, id, _, _, _)| *id);

        // Phase 4A v2: snapshot player candidates too so NPC-fired
        // projectiles can hit the player. `PlayerOwned.steam_id`
        // doubles as the exclude key for player-fired self-hits.
        let mut player_snapshot: Vec<(Entity, u64, [f32; 3], f32, crate::region::RegionId)> = {
            let mut q = self
                .world
                .query::<(Entity, &PlayerOwned, &Position, &Rotation, &InRegion)>();
            q.iter(&self.world)
                .map(|(e, po, p, r, reg)| (e, po.steam_id, p.0, r.0, reg.0))
                .collect()
        };
        player_snapshot.sort_by_key(|(_, sid, _, _, _)| *sid);

        // Snapshot projectiles. We only need the fields for the
        // integration + hit test; the ECS entity id lets us despawn.
        let proj_snapshot: Vec<(Entity, Projectile, crate::region::RegionId)> = {
            let mut q = self.world.query::<(Entity, &Projectile, &InRegion)>();
            q.iter(&self.world)
                .map(|(e, p, r)| (e, p.clone(), r.0))
                .collect()
        };

        let mut plans: Vec<Plan> = Vec::with_capacity(proj_snapshot.len());
        for (entity, proj, proj_region) in proj_snapshot {
            // Integrate drag + gravity for one fixed dt. See
            // `docs/book/src/planning/weapons-plan.md` §4.3.
            // `drag_k` comes from the round's `AmmoConfig`;
            // defaulting to 0 for unknown rounds means no drag
            // (but the spawn path validates the round exists before
            // minting a projectile, so unknown rounds shouldn't
            // reach here).
            let drag_k = {
                let reg = self.world.resource::<ItemRegistry>();
                reg.get(&proj.round_id)
                    .and_then(|d| d.ammo_config.as_ref())
                    .map(|a| a.drag_k)
                    .unwrap_or(0.0)
            };
            let speed = magnitude(proj.vel);
            let dir = if speed > 0.0 {
                [
                    proj.vel[0] / speed,
                    proj.vel[1] / speed,
                    proj.vel[2] / speed,
                ]
            } else {
                [0.0, 0.0, 0.0]
            };
            let drag_accel = speed * speed * drag_k;
            let new_vel = [
                proj.vel[0] - dir[0] * drag_accel * dt_s,
                proj.vel[1] - dir[1] * drag_accel * dt_s - gravity * dt_s,
                proj.vel[2] - dir[2] * drag_accel * dt_s,
            ];
            let segment = [new_vel[0] * dt_s, new_vel[1] * dt_s, new_vel[2] * dt_s];
            let segment_len = magnitude(segment);
            let unit_segment = if segment_len > 0.0 {
                [
                    segment[0] / segment_len,
                    segment[1] / segment_len,
                    segment[2] / segment_len,
                ]
            } else {
                [0.0, 0.0, 0.0]
            };
            let new_pos = [
                proj.pos[0] + segment[0],
                proj.pos[1] + segment[1],
                proj.pos[2] + segment[2],
            ];
            let new_distance = proj.distance_traveled_m + segment_len;

            // Hit test: walk NPC + player candidates in the same
            // region. Phase 4A v2 lifted the NPC-source carve-out
            // and added players as targets so NPC fire can damage
            // the player (and player fire damages NPCs as before).
            // Self-hit prevention: NPC-fired projectiles skip the
            // shooter's own NpcId; player-fired skip the shooter's
            // steam_id.
            let mut best_hit: Option<(f32, HitTarget, BodyPart, [f32; 3])> = None;
            for (_, npc_id, npc_pos, npc_yaw, npc_region) in &npc_snapshot {
                if *npc_region != proj_region {
                    continue;
                }
                if proj.source_npc_id == Some(*npc_id) {
                    // Self-hit prevention for NPC fire.
                    continue;
                }
                if let Some((part, t)) = hitbox::ray_hits_humanoid(
                    proj.pos,
                    unit_segment,
                    segment_len,
                    *npc_pos,
                    *npc_yaw,
                ) {
                    let hit_pos = [
                        proj.pos[0] + unit_segment[0] * t,
                        proj.pos[1] + unit_segment[1] * t,
                        proj.pos[2] + unit_segment[2] * t,
                    ];
                    match best_hit {
                        Some((best_t, _, _, _)) if best_t <= t => {}
                        _ => best_hit = Some((t, HitTarget::Npc(*npc_id), part, hit_pos)),
                    }
                }
            }
            for (_, steam_id, p_pos, p_yaw, p_region) in &player_snapshot {
                if *p_region != proj_region {
                    continue;
                }
                if proj.source_steam_id != 0 && proj.source_steam_id == *steam_id {
                    // Self-hit prevention for player fire.
                    continue;
                }
                if let Some((part, t)) =
                    hitbox::ray_hits_humanoid(proj.pos, unit_segment, segment_len, *p_pos, *p_yaw)
                {
                    let hit_pos = [
                        proj.pos[0] + unit_segment[0] * t,
                        proj.pos[1] + unit_segment[1] * t,
                        proj.pos[2] + unit_segment[2] * t,
                    ];
                    match best_hit {
                        Some((best_t, _, _, _)) if best_t <= t => {}
                        _ => best_hit = Some((t, HitTarget::Player(*steam_id), part, hit_pos)),
                    }
                }
            }

            // Cover penetration check: test the swept segment
            // against registered cover volumes. If cover intercepts
            // the ray before the humanoid hit, the projectile may be
            // stopped, partially penetrate (spall), or fully
            // penetrate and continue to the target.
            let cover_blocked = if let Some((best_t, _, _, _)) = best_hit {
                let cover_vols = self.world.resource::<crate::cover::CoverVolumes>();
                let hits = cover_vols.check_cover_between(
                    proj_region,
                    proj.pos,
                    [
                        proj.pos[0] + unit_segment[0] * best_t,
                        proj.pos[1] + unit_segment[1] * best_t,
                        proj.pos[2] + unit_segment[2] * best_t,
                    ],
                );
                let pen_class = {
                    let items = self.world.resource::<crate::items::ItemRegistry>();
                    items
                        .get(&proj.round_id)
                        .and_then(|d| d.ammo_config.as_ref())
                        .map(|a| a.penetration_class)
                        .unwrap_or(1)
                };
                let mut blocked = false;
                for cover_hit in &hits {
                    let result = crate::cover::can_penetrate(
                        pen_class,
                        cover_hit.material_id,
                        cover_hit.thickness_mm,
                        cover_hit.angle,
                    );
                    match result {
                        crate::cover::PenetrationResult::Stopped => {
                            blocked = true;
                            break;
                        }
                        crate::cover::PenetrationResult::PartialPenetration { .. } => {
                            blocked = true;
                            break;
                        }
                        crate::cover::PenetrationResult::FullPenetration { .. } => {
                            // Projectile punches through, continues
                        }
                    }
                }
                // Damage destructible cover volumes that were hit
                if !hits.is_empty() {
                    let mut cover_vols_mut =
                        self.world.resource_mut::<crate::cover::CoverVolumes>();
                    for cover_hit in &hits {
                        cover_vols_mut.damage_cover(
                            proj_region,
                            cover_hit.volume_id,
                            pen_class as f32 * 2.0,
                        );
                    }
                }
                blocked
            } else {
                false
            };

            if let Some((_, target, part, hit_pos)) = best_hit {
                if cover_blocked {
                    // Cover stopped the projectile — impact on cover,
                    // not the target. Despawn with a null-target delta
                    // so tracers terminate visually.
                    let cover_impact_pos = hit_pos;
                    plans.push(Plan::Despawn {
                        entity,
                        projectile_id: proj.id,
                        terminal_pos: cover_impact_pos,
                    });
                } else {
                    plans.push(Plan::Hit {
                        entity,
                        projectile_id: proj.id,
                        round_id: proj.round_id.clone(),
                        hit_pos,
                        target,
                        part,
                        source_npc_id: proj.source_npc_id,
                        impact_speed_mps: magnitude(new_vel),
                    });
                }
            } else if new_distance >= proj.max_range_m {
                plans.push(Plan::Despawn {
                    entity,
                    projectile_id: proj.id,
                    terminal_pos: new_pos,
                });
            } else {
                plans.push(Plan::Advance {
                    entity,
                    new_pos,
                    new_vel,
                    new_distance,
                });
            }
        }

        // Apply plans.
        for plan in plans {
            match plan {
                Plan::Advance {
                    entity,
                    new_pos,
                    new_vel,
                    new_distance,
                } => {
                    if let Some(mut proj) = self.world.get_mut::<Projectile>(entity) {
                        proj.pos = new_pos;
                        proj.vel = new_vel;
                        proj.distance_traveled_m = new_distance;
                    }
                    if let Some(mut pos) = self.world.get_mut::<Position>(entity) {
                        pos.0 = new_pos;
                    }
                }
                Plan::Hit {
                    entity,
                    projectile_id,
                    round_id,
                    hit_pos,
                    target,
                    part,
                    source_npc_id,
                    impact_speed_mps,
                } => {
                    let (hit_npc, hit_player_steam_id, damage, penetrated) = match target {
                        HitTarget::Npc(npc_id) => {
                            let (damage, penetrated) = compute_hit_damage(
                                &mut self.world,
                                &round_id,
                                npc_id,
                                part,
                                impact_speed_mps,
                            );
                            self.apply_damage_to_npc_part(npc_id, part, damage)?;
                            // NPC-source attribution (LastDamager /
                            // RecentAttackers / blackboard / kill
                            // credit). Migrated from the old
                            // npc_combat dice path so projectile +
                            // melee + future damage sources land at
                            // the same attribution seam.
                            if let Some(attacker) = source_npc_id {
                                let attacker_pos = self.world.get::<Position>(entity).map(|p| p.0);
                                self.apply_npc_attribution_for_hit(
                                    npc_id,
                                    attacker,
                                    attacker_pos.unwrap_or(hit_pos),
                                    damage,
                                )?;
                            }
                            (Some(npc_id), None, damage, penetrated)
                        }
                        HitTarget::Player(steam_id) => {
                            // Player damage uses the same pen-vs-armor
                            // formula via `compute_hit_damage_player`.
                            let (damage, penetrated) = compute_hit_damage_player(
                                &mut self.world,
                                &round_id,
                                steam_id,
                                part,
                                impact_speed_mps,
                            );
                            self.apply_damage_to_part(steam_id, part, damage)?;
                            (None, Some(steam_id), damage, penetrated)
                        }
                    };
                    self.record_delta(WorldDelta::ProjectileImpacted {
                        id: projectile_id,
                        pos: hit_pos,
                        hit_npc,
                        hit_player_steam_id,
                        body_part: Some(part),
                        damage_applied: damage,
                        penetrated,
                    })?;
                    self.world.despawn(entity);
                }
                Plan::Despawn {
                    entity,
                    projectile_id,
                    terminal_pos,
                } => {
                    self.record_delta(WorldDelta::ProjectileImpacted {
                        id: projectile_id,
                        pos: terminal_pos,
                        hit_npc: None,
                        hit_player_steam_id: None,
                        body_part: None,
                        damage_applied: 0.0,
                        penetrated: false,
                    })?;
                    self.world.despawn(entity);
                }
            }
        }
        Ok(())
    }
}

#[inline]
fn magnitude(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Compute final damage for a projectile → NPC body-part hit under
/// the Phase 2 pen-vs-armor formula. Returns `(damage,
/// penetrated)`. See `docs/book/src/planning/weapons-plan.md` §6.
///
/// Formula (integer pen classes):
/// ```text
/// pen_effective = round.penetration_class - armor.protection_class
/// if pen_effective >= 0:
///     damage = round.damage_soft * part.soft_multiplier
///     penetrated = true
/// else:
///     ratio = max(0.0, 1.0 + pen_effective * blocked_ratio_per_class_short)
///     damage = round.damage_blunt * ratio * part.soft_multiplier
///     penetrated = false
/// damage *= clamp(E_impact / round.reference_energy_j, floor, 1.0)
/// ```
///
/// Commit 3 applies the range-falloff factor using the round's
/// `reference_energy_j` compared to a naïve "kinetic energy at
/// muzzle = reference" assumption (projectile has no retained-
/// energy tracking yet). Future slice wires the falloff to the
/// actual in-flight velocity.
fn compute_hit_damage(
    world: &mut World,
    round_id: &ItemId,
    npc: crate::components::NpcId,
    part: BodyPart,
    impact_speed_mps: f32,
) -> (f32, bool) {
    // Clone the values we need off the registry up-front so we can
    // take a mutable borrow of World for the Equipment query below
    // without overlapping immutable borrows.
    let (ammo, soft_mult, blocked_ratio, energy_floor) = {
        let reg = world.resource::<ItemRegistry>();
        let Some(round_def) = reg.get(round_id) else {
            return (0.0, false);
        };
        let Some(ammo) = round_def.ammo_config.as_ref() else {
            return (0.0, false);
        };
        let bc = world.resource::<BallisticsConfig>();
        (
            ammo.clone(),
            bc.body_part_soft_multipliers.get(part),
            bc.blocked_damage_ratio_per_class_short,
            bc.retained_energy_floor,
        )
    };

    let armor_class = armor_class_at_part(world, npc, part);

    let pen_effective = i32::from(ammo.penetration_class) - i32::from(armor_class);
    let (damage_raw, penetrated) = if pen_effective >= 0 {
        (ammo.damage_soft * soft_mult, true)
    } else {
        let ratio = (1.0 + pen_effective as f32 * blocked_ratio).max(0.0);
        (ammo.damage_blunt * ratio * soft_mult, false)
    };

    // Range falloff from actual retained kinetic energy. The
    // projectile's velocity decays via drag each tick; impact_speed
    // is the velocity at the moment of hit. `E_impact / E_ref`
    // gives a natural distance-based damage curve: point-blank
    // ≈ 1.0×, max range approaches the floor.
    let mass_kg = ammo.mass_g / 1000.0;
    let v = if impact_speed_mps > 0.0 {
        impact_speed_mps
    } else {
        ammo.muzzle_velocity_mps
    };
    let e_impact = 0.5 * mass_kg * v * v;
    let falloff = (e_impact / ammo.reference_energy_j).clamp(energy_floor, 1.0);

    (damage_raw * falloff, penetrated)
}

/// Scan `Equipment` on `npc` (if any) for armor items whose
/// `ArmorConfig.coverage` includes `part`, and return the maximum
/// `protection_class`. `0` when no armor covers the part.
///
/// Clones the `ItemRegistry` once up-front so the Equipment query
/// borrow doesn't overlap with registry lookups.
fn armor_class_at_part(world: &mut World, npc: crate::components::NpcId, part: BodyPart) -> u8 {
    let reg = world.resource::<ItemRegistry>().clone();
    let mut q = world.query::<(Entity, &Npc)>();
    let mut entity: Option<Entity> = None;
    for (e, n) in q.iter(world) {
        if n.id == npc {
            entity = Some(e);
            break;
        }
    }
    let Some(entity) = entity else {
        return 0;
    };
    let Some(eq) = world.get::<Equipment>(entity) else {
        return 0;
    };
    let mut best: u8 = 0;
    for equipped in eq.0.values() {
        let Some(def) = reg.get(&equipped.stack.id) else {
            continue;
        };
        let Some(armor) = def.armor_config.as_ref() else {
            continue;
        };
        if armor_covers(armor, part) && armor.protection_class > best {
            best = armor.protection_class;
        }
    }
    best
}

fn armor_covers(armor: &ArmorConfig, part: BodyPart) -> bool {
    armor.coverage.contains(&part)
}

/// Phase 4A v2 — player-target hit damage. Mirrors
/// `compute_hit_damage` but resolves armor through the player's
/// `Equipment` instead of an NPC's. Falloff math is identical.
fn compute_hit_damage_player(
    world: &mut World,
    round_id: &ItemId,
    steam_id: u64,
    part: BodyPart,
    impact_speed_mps: f32,
) -> (f32, bool) {
    let (ammo, soft_mult, blocked_ratio, energy_floor) = {
        let reg = world.resource::<ItemRegistry>();
        let Some(round_def) = reg.get(round_id) else {
            return (0.0, false);
        };
        let Some(ammo) = round_def.ammo_config.as_ref() else {
            return (0.0, false);
        };
        let bc = world.resource::<BallisticsConfig>();
        (
            ammo.clone(),
            bc.body_part_soft_multipliers.get(part),
            bc.blocked_damage_ratio_per_class_short,
            bc.retained_energy_floor,
        )
    };
    let armor_class = armor_class_at_part_player(world, steam_id, part);
    let pen_effective = i32::from(ammo.penetration_class) - i32::from(armor_class);
    let (damage_raw, penetrated) = if pen_effective >= 0 {
        (ammo.damage_soft * soft_mult, true)
    } else {
        let ratio = (1.0 + pen_effective as f32 * blocked_ratio).max(0.0);
        (ammo.damage_blunt * ratio * soft_mult, false)
    };
    let mass_kg = ammo.mass_g / 1000.0;
    let v = if impact_speed_mps > 0.0 {
        impact_speed_mps
    } else {
        ammo.muzzle_velocity_mps
    };
    let e_impact = 0.5 * mass_kg * v * v;
    let falloff = (e_impact / ammo.reference_energy_j).clamp(energy_floor, 1.0);
    (damage_raw * falloff, penetrated)
}

fn armor_class_at_part_player(world: &mut World, steam_id: u64, part: BodyPart) -> u8 {
    let reg = world.resource::<ItemRegistry>().clone();
    let mut q = world.query::<(Entity, &PlayerOwned)>();
    let mut entity: Option<Entity> = None;
    for (e, po) in q.iter(world) {
        if po.steam_id == steam_id {
            entity = Some(e);
            break;
        }
    }
    let Some(entity) = entity else {
        return 0;
    };
    let Some(eq) = world.get::<Equipment>(entity) else {
        return 0;
    };
    let mut best: u8 = 0;
    for equipped in eq.0.values() {
        let Some(def) = reg.get(&equipped.stack.id) else {
            continue;
        };
        let Some(armor) = def.armor_config.as_ref() else {
            continue;
        };
        if armor_covers(armor, part) && armor.protection_class > best {
            best = armor.protection_class;
        }
    }
    best
}

impl Sim {
    /// Phase 4A v2 — attribution writes when an NPC-source
    /// projectile hits an NPC. Migrated out of `npc_combat`'s
    /// dice path so projectile + future melee damage paths share
    /// the same seam. Writes: `LastDamager` on the victim,
    /// pushes into the victim's `RecentAttackers` ring, queues a
    /// kill credit if the hit dropped HP to zero, and writes
    /// `UnderFireAt` to the victim's squad blackboard.
    pub(super) fn apply_npc_attribution_for_hit(
        &mut self,
        victim: NpcId,
        attacker: NpcId,
        attacker_pos: [f32; 3],
        damage: f32,
    ) -> Result<()> {
        const MAX_RECENT_ATTACKERS: usize = 8;
        const UNDER_FIRE_BB_TTL: u32 = 100;

        let tick = self.world.resource::<crate::resources::SimClock>().tick;
        let Some(victim_entity) = super::persistence::find_npc_in(&mut self.world, victim) else {
            return Ok(());
        };
        // Attacker faction id for LastDamager.faction.
        // `FactionId` has no `Default` — fall back to the
        // first/`0` id when the attacker entity has vanished
        // (rare race: NPC killed between fire and impact).
        let attacker_faction = super::persistence::find_npc_in(&mut self.world, attacker)
            .and_then(|e| self.world.get::<crate::components::InFaction>(e).copied())
            .map(|f| f.0)
            .unwrap_or(crate::faction::registry::FactionId(0));
        // Did the hit kill the victim?
        let died = self
            .world
            .get::<crate::components::Health>(victim_entity)
            .map(|h| h.current <= 0.0)
            .unwrap_or(false);
        // RecentAttackers — push or accumulate.
        if let Some(mut recent) = self.world.get_mut::<RecentAttackers>(victim_entity) {
            recent.record(attacker, tick, damage, MAX_RECENT_ATTACKERS);
        }
        // LastDamager — overwrites any previous entry.
        self.world.entity_mut(victim_entity).insert(LastDamager {
            attacker_id: Some(attacker),
            faction: attacker_faction,
            tick,
        });
        // Kill credit on the killing blow.
        if died {
            self.world
                .resource_mut::<crate::resources::PendingKillCredits>()
                .credit(attacker);
        }
        // Squad blackboard write — UnderFireAt for the victim's
        // squad if it has one.
        let group = self
            .world
            .get::<crate::components::Group>(victim_entity)
            .copied();
        if let Some(group) = group {
            self.world
                .resource_mut::<crate::squad_blackboard::SquadBlackboards>()
                .write(
                    group.id,
                    crate::squad_blackboard::BlackboardKey::UnderFireAt,
                    crate::squad_blackboard::BlackboardValue::Position(attacker_pos),
                    tick,
                    UNDER_FIRE_BB_TTL,
                );
        }
        // Reactive aggro: victim immediately faces and aggros the
        // attacker. Without this, NPCs facing away from the shooter
        // never acquire aggro (FOV cone misses) and stand there
        // getting shot.
        if !died {
            let victim_pos = self
                .world
                .get::<Position>(victim_entity)
                .map(|p| p.0)
                .unwrap_or([0.0; 3]);
            let face_dx = attacker_pos[0] - victim_pos[0];
            let face_dz = attacker_pos[2] - victim_pos[2];
            if face_dx * face_dx + face_dz * face_dz > 0.01 {
                if let Some(mut r) = self
                    .world
                    .get_mut::<crate::components::Rotation>(victim_entity)
                {
                    r.0 = face_dz.atan2(face_dx);
                }
            }
            if self
                .world
                .get::<crate::components::Aggro>(victim_entity)
                .is_none()
            {
                self.world
                    .entity_mut(victim_entity)
                    .insert(crate::components::Aggro {
                        target: attacker,
                        last_seen_tick: tick,
                    });
            }
        }
        Ok(())
    }
}

impl Sim {
    /// Public wrapper around the armor lookup helper for tests +
    /// the docs site. Forwards to the private free function so the
    /// implementation stays in one place.
    #[doc(hidden)]
    pub fn armor_class_at_part_for_test(
        &mut self,
        npc: crate::components::NpcId,
        part: BodyPart,
    ) -> u8 {
        armor_class_at_part(&mut self.world, npc, part)
    }

    /// Test-only: equip the given armor item ids on an NPC. Each
    /// string must be a valid `ItemDef` id with an `armor_config`
    /// block. `torso_and_body` is a list of items that go in the
    /// `armor_vest` slot (only the first actually equips; others
    /// are ignored — there's one vest slot). `head_gear` goes in
    /// the `head` slot.
    ///
    /// Used by `tests/ballistics_matrix.rs`; a real gameplay
    /// equip flow lives in `Sim::equip` for player entities,
    /// but NPCs get no equip UI yet so this is the single seam.
    #[doc(hidden)]
    pub fn equip_test_armor_on_npc(
        &mut self,
        npc: crate::components::NpcId,
        torso_and_body: &[&str],
        head_gear: Option<&str>,
    ) -> anyhow::Result<()> {
        let mut q = self
            .world
            .query::<(bevy_ecs::prelude::Entity, &crate::components::Npc)>();
        let mut target: Option<bevy_ecs::prelude::Entity> = None;
        for (e, n) in q.iter(&self.world) {
            if n.id == npc {
                target = Some(e);
                break;
            }
        }
        let Some(entity) = target else {
            return Err(anyhow::anyhow!("equip_test_armor_on_npc: unknown npc"));
        };
        let mut map: std::collections::HashMap<
            crate::items::SlotId,
            crate::components::EquippedItem,
        > = std::collections::HashMap::new();
        if let Some(item_id) = torso_and_body.first() {
            map.insert(
                crate::items::SlotId::from("armor_vest"),
                crate::components::EquippedItem {
                    stack: crate::components::ItemInstance {
                        id: ItemId::from(*item_id),
                        count: 1,
                        spawned_tick: self.current_tick(),
                        magazine_state: None,
                    },
                    inner_grid: None,
                    weapon_state: None,
                },
            );
        }
        if let Some(item_id) = head_gear {
            map.insert(
                crate::items::SlotId::from("head"),
                crate::components::EquippedItem {
                    stack: crate::components::ItemInstance {
                        id: ItemId::from(item_id),
                        count: 1,
                        spawned_tick: self.current_tick(),
                        magazine_state: None,
                    },
                    inner_grid: None,
                    weapon_state: None,
                },
            );
        }
        self.world
            .entity_mut(entity)
            .insert(crate::components::Equipment(map));
        Ok(())
    }
}

impl Sim {
    /// Test-only: snapshot every in-flight `Projectile` entity's
    /// current world position. Tests use this to assert on drop /
    /// despawn without reaching into `self.world`.
    #[doc(hidden)]
    pub fn collect_projectile_positions_for_test(&mut self) -> Vec<[f32; 3]> {
        let mut q = self.world.query::<(&Projectile, &Position)>();
        q.iter(&self.world).map(|(_, pos)| pos.0).collect()
    }
}
