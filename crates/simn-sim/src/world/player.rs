//! Player CRUD, damage/heal, and world-time accessors on `Sim`.

use anyhow::Result;
use bevy_ecs::prelude::*;

use crate::components::*;
use crate::delta::WorldDelta;
use crate::region::RegionId;
use crate::resources::*;
use crate::systems::wounds::severity_from_damage;

use super::{find_npc_in, PlayerView, Sim};

impl Sim {
    /// Spawn or update the player entity for `steam_id`. If the player
    /// already exists, this is an idempotent move. Otherwise it
    /// creates a new entity with all the expected components.
    pub fn upsert_player(
        &mut self,
        steam_id: u64,
        region: RegionId,
        pos: [f32; 3],
        yaw: f32,
    ) -> Result<()> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }
        let existing = self.find_player_entity(steam_id);
        let delta = match existing {
            Some(e) => {
                {
                    let mut pos_ref = self
                        .world
                        .get_mut::<Position>(e)
                        .expect("player has Position");
                    pos_ref.0 = pos;
                }
                {
                    let mut rot_ref = self
                        .world
                        .get_mut::<Rotation>(e)
                        .expect("player has Rotation");
                    rot_ref.0 = yaw;
                }
                {
                    let mut region_ref = self
                        .world
                        .get_mut::<InRegion>(e)
                        .expect("player has InRegion");
                    region_ref.0 = region;
                }
                WorldDelta::SpawnPlayer {
                    steam_id,
                    region,
                    pos,
                    yaw,
                }
            }
            None => {
                self.world.spawn((
                    (
                        PlayerOwned { steam_id },
                        Actor {
                            kind: ActorKind::Player,
                        },
                        Position(pos),
                        Rotation(yaw),
                        InRegion(region),
                        Health::new_full(),
                        Stamina::new_full(),
                        BodyParts::new_full(),
                        crate::components::LimbStates::default(),
                        SurvivalStats::new_full(),
                        Wounds::default(),
                    ),
                    (
                        Pain::default(),
                        Contamination::default(),
                        ActiveEffects::default(),
                        DrugTolerance::default(),
                        crate::components::Inventory::default(),
                        crate::components::NearCampfire::default(),
                    ),
                ));
                WorldDelta::SpawnPlayer {
                    steam_id,
                    region,
                    pos,
                    yaw,
                }
            }
        };
        self.record_delta(delta)?;
        Ok(())
    }

    pub fn remove_player(&mut self, steam_id: u64) -> Result<()> {
        if let Some(e) = self.find_player_entity(steam_id) {
            self.world.despawn(e);
            self.record_delta(WorldDelta::DespawnPlayer { steam_id })?;
        }
        Ok(())
    }

    pub fn move_player(&mut self, steam_id: u64, pos: [f32; 3], yaw: f32) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        // Ensure the player's region is marked active every tick.
        // ActiveRegions is transient (not saved), so on a save-resume
        // it starts empty. upsert/change_region set it on region
        // entry, but move_player runs each physics tick and is the
        // only call guaranteed to fire continuously.
        if let Some(r) = self.world.get::<InRegion>(e) {
            let region = r.0;
            let mut ar = self.world.resource_mut::<ActiveRegions>();
            if !ar.is_active(region) {
                ar.regions.clear();
                ar.regions.insert(region);
            }
        }
        self.world
            .get_mut::<Position>(e)
            .expect("player has Position")
            .0 = pos;
        self.world
            .get_mut::<Rotation>(e)
            .expect("player has Rotation")
            .0 = yaw;
        self.record_delta(WorldDelta::MovePlayer { steam_id, pos, yaw })?;
        Ok(())
    }

    pub fn change_player_region(&mut self, steam_id: u64, region: RegionId) -> Result<()> {
        if self.regions().get(region).is_none() {
            return Err(anyhow::anyhow!("unknown region {region}"));
        }
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        self.world
            .get_mut::<InRegion>(e)
            .expect("player has InRegion")
            .0 = region;
        self.record_delta(WorldDelta::ChangePlayerRegion { steam_id, region })?;
        Ok(())
    }

    /// Every connected player's `steam_id`, sorted ascending.
    /// Used by `worker::build_sim_view` to enumerate players
    /// without exposing the ECS world; also handy for any caller
    /// that wants to fan out per-player ops in deterministic
    /// order. Takes `&mut self` because the ECS query handle
    /// requires mutable access.
    pub fn connected_player_ids(&mut self) -> Vec<u64> {
        let mut q = self.world.query::<&PlayerOwned>();
        let mut ids: Vec<u64> = q.iter(&self.world).map(|p| p.steam_id).collect();
        ids.sort_unstable();
        ids
    }

    /// Read-only view of a player's state. Takes `&mut self` because
    /// `bevy_ecs::World::query` needs to build a `QueryState`; no
    /// mutation happens.
    pub fn player_view(&mut self, steam_id: u64) -> Option<PlayerView> {
        let e = self.find_player_entity(steam_id)?;
        let pos = self.world.get::<Position>(e)?.0;
        let yaw = self.world.get::<Rotation>(e)?.0;
        let region = self.world.get::<InRegion>(e)?.0;
        let health = self
            .world
            .get::<Health>(e)
            .copied()
            .unwrap_or(Health::new_full());
        let stamina = self
            .world
            .get::<Stamina>(e)
            .copied()
            .unwrap_or(Stamina::new_full());
        let body_parts = self
            .world
            .get::<BodyParts>(e)
            .copied()
            .unwrap_or(BodyParts::new_full());
        let survival = self
            .world
            .get::<SurvivalStats>(e)
            .copied()
            .unwrap_or(SurvivalStats::new_full());
        let wounds = self
            .world
            .get::<Wounds>(e)
            .map(|w| w.0.clone())
            .unwrap_or_default();
        let pain = self.world.get::<Pain>(e).copied().unwrap_or_default();
        let contamination = self
            .world
            .get::<Contamination>(e)
            .copied()
            .unwrap_or_default();
        let active_effects = self
            .world
            .get::<ActiveEffects>(e)
            .map(|e| e.0.clone())
            .unwrap_or_default();
        let drug_tolerance = self
            .world
            .get::<DrugTolerance>(e)
            .map(|t| t.0.clone())
            .unwrap_or_default();
        Some(PlayerView {
            steam_id,
            region,
            pos,
            yaw,
            health,
            stamina,
            body_parts,
            survival,
            wounds,
            pain,
            contamination,
            active_effects,
            drug_tolerance,
        })
    }

    /// Current in-world clock (day + time-of-day).
    pub fn world_time(&self) -> WorldTime {
        *self.world.resource::<WorldTime>()
    }

    /// Global weather state (current, upcoming, transition tick).
    pub fn weather(&self) -> WeatherState {
        *self.world.resource::<WeatherState>()
    }

    /// Force the current weather to `w` immediately. The next
    /// transition re-rolls normally from this new state.
    pub fn set_weather(&mut self, w: crate::resources::Weather) {
        let mut state = self.world.resource_mut::<WeatherState>();
        state.current = w;
        state.next = w;
        // Reset transition so the normal system picks a fresh next.
        state.transitions_at_tick = 0;
    }

    /// Set the in-world time of day. `hour` is 0..23, `minute` is
    /// 0..59. Keeps the current day count and day-length unchanged.
    pub fn set_time_of_day(&mut self, hour: u32, minute: u32) {
        let mut t = self.world.resource_mut::<WorldTime>();
        let frac = (hour as f32 * 60.0 + minute as f32) / 1440.0;
        t.seconds_of_day = frac * t.day_length_seconds;
    }

    /// Advance the in-world clock by `hours` hours (can be fractional).
    /// Handles day rollover.
    pub fn advance_time(&mut self, hours: f32) {
        let mut t = self.world.resource_mut::<WorldTime>();
        let frac = hours / 24.0;
        let advance = frac * t.day_length_seconds;
        t.seconds_of_day += advance;
        while t.seconds_of_day >= t.day_length_seconds {
            t.seconds_of_day -= t.day_length_seconds;
            t.day += 1;
        }
    }

    /// Apply un-located damage to a player. Routes to the torso —
    /// head/torso are the only body parts that gate death, and the
    /// torso is the most common hit location for non-located damage
    /// (falls, environmental, debug). Journaled as a `SetBodyPart`
    /// record. Aggregate `Health` mirror is updated to reflect the new
    /// `min(head, torso)`.
    pub fn apply_damage(&mut self, steam_id: u64, amount: f32) -> Result<()> {
        self.apply_damage_to_part(steam_id, BodyPart::Torso, amount)
    }

    /// Heal un-located damage on a player. Routes to the torso (mirror
    /// of [`Self::apply_damage`]). Journaled.
    pub fn heal(&mut self, steam_id: u64, amount: f32) -> Result<()> {
        self.heal_part(steam_id, BodyPart::Torso, amount)
    }

    /// Apply damage to a specific body part. Clamped to `[0, max]`.
    /// Updates the aggregate `Health.current` mirror to
    /// `min(body_parts.head, body_parts.torso)`. Journaled as a
    /// `SetBodyPart` record.
    ///
    /// If `amount` exceeds [`crate::systems::wounds::WOUND_THRESHOLD_LIGHT`],
    /// a `Bleed` wound is also spawned on the part with severity scaled
    /// from the damage; the wound is journaled as a separate `WoundAdded`
    /// record so replay reproduces it exactly.
    pub fn apply_damage_to_part(
        &mut self,
        steam_id: u64,
        part: BodyPart,
        amount: f32,
    ) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let (new_part, vital_min) = {
            let mut bp = self
                .world
                .get_mut::<BodyParts>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no BodyParts"))?;
            let slot = bp.get_mut(part);
            *slot = (*slot - amount).clamp(0.0, BodyParts::DEFAULT_MAX);
            (*slot, bp.vital_min())
        };
        if let Some(mut h) = self.world.get_mut::<Health>(e) {
            h.current = vital_min.min(h.max);
        }
        self.record_delta(WorldDelta::SetBodyPart {
            steam_id,
            part,
            current: new_part,
        })?;

        // Spawn a Bleed wound for above-threshold hits. Severity is
        // derived from damage magnitude per `severity_from_damage`.
        if let Some(severity) = severity_from_damage(amount) {
            let spawned_tick = self.world.resource::<SimClock>().tick;
            let wound_id = self
                .world
                .resource_mut::<crate::resources::WoundIdCounter>()
                .mint();
            let wound = Wound {
                body_part: part,
                kind: WoundKind::Bleed,
                severity,
                spawned_tick,
                treatment: WoundTreatment::Untreated,
                treatment_changed_tick: spawned_tick,
                infected: false,
                infection_started_tick: None,
                tourniquet_started_tick: None,
            };
            if let Some(mut wounds) = self.world.get_mut::<Wounds>(e) {
                wounds.0.push((wound_id, wound));
            } else {
                self.world
                    .entity_mut(e)
                    .insert(Wounds(vec![(wound_id, wound)]));
            }
            if let Some(mut states) = self.world.get_mut::<crate::components::LimbStates>(e) {
                states.mark_wounded(part);
            }
            self.record_delta(WorldDelta::WoundAdded {
                steam_id,
                wound_id,
                body_part: part,
                kind: WoundKind::Bleed,
                severity,
                spawned_tick,
            })?;
        }
        Ok(())
    }

    /// Heal a specific body part. Clamped to `[0, max]`. Updates the
    /// aggregate `Health.current` mirror. Journaled.
    pub fn heal_part(&mut self, steam_id: u64, part: BodyPart, amount: f32) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let (new_part, vital_min) = {
            let mut bp = self
                .world
                .get_mut::<BodyParts>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no BodyParts"))?;
            let slot = bp.get_mut(part);
            *slot = (*slot + amount).clamp(0.0, BodyParts::DEFAULT_MAX);
            (*slot, bp.vital_min())
        };
        if let Some(mut h) = self.world.get_mut::<Health>(e) {
            h.current = vital_min.min(h.max);
        }
        self.record_delta(WorldDelta::SetBodyPart {
            steam_id,
            part,
            current: new_part,
        })?;
        Ok(())
    }

    /// Apply damage to one of an NPC's body parts. Parallel to
    /// [`Self::apply_damage_to_part`] for players. Clamps to `[0, max]`,
    /// updates the aggregate `Health.current` mirror to the new
    /// `min(head, torso)`, and journals a [`WorldDelta::SetNpcBodyPart`].
    ///
    /// If `amount` exceeds [`crate::systems::wounds::WOUND_THRESHOLD_LIGHT`],
    /// a `Bleed` wound is also spawned on the part with severity
    /// scaled from the damage magnitude; the wound is journaled as a
    /// separate `NpcWoundAdded` record so replay reproduces it.
    pub fn apply_damage_to_npc_part(
        &mut self,
        id: NpcId,
        part: BodyPart,
        amount: f32,
    ) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let (new_part, vital_min) = {
            let mut bp = self
                .world
                .get_mut::<BodyParts>(e)
                .ok_or_else(|| anyhow::anyhow!("npc {id:?} has no BodyParts"))?;
            let slot = bp.get_mut(part);
            *slot = (*slot - amount).clamp(0.0, BodyParts::DEFAULT_MAX);
            (*slot, bp.vital_min())
        };
        if let Some(mut h) = self.world.get_mut::<Health>(e) {
            h.current = vital_min.min(h.max);
        }
        self.record_delta(WorldDelta::SetNpcBodyPart {
            id,
            part,
            current: new_part,
        })?;

        // Spawn a Bleed wound for above-threshold hits. Severity is
        // derived from damage magnitude per `severity_from_damage`.
        // Mirrors the player path at `apply_damage_to_part`.
        if let Some(severity) = severity_from_damage(amount) {
            let spawned_tick = self.world.resource::<SimClock>().tick;
            let wound_id = self
                .world
                .resource_mut::<crate::resources::WoundIdCounter>()
                .mint();
            let wound = Wound {
                body_part: part,
                kind: WoundKind::Bleed,
                severity,
                spawned_tick,
                treatment: WoundTreatment::Untreated,
                treatment_changed_tick: spawned_tick,
                infected: false,
                infection_started_tick: None,
                tourniquet_started_tick: None,
            };
            if let Some(mut wounds) = self.world.get_mut::<Wounds>(e) {
                wounds.0.push((wound_id, wound));
            } else {
                self.world
                    .entity_mut(e)
                    .insert(Wounds(vec![(wound_id, wound)]));
            }
            if let Some(mut states) = self.world.get_mut::<crate::components::LimbStates>(e) {
                states.mark_wounded(part);
            }
            self.record_delta(WorldDelta::NpcWoundAdded {
                id,
                wound_id,
                body_part: part,
                kind: WoundKind::Bleed,
                severity,
                spawned_tick,
            })?;
        }
        Ok(())
    }

    /// Heal one of an NPC's body parts. Mirror of
    /// [`Self::apply_damage_to_npc_part`].
    pub fn heal_npc_part(&mut self, id: NpcId, part: BodyPart, amount: f32) -> Result<()> {
        let Some(e) = find_npc_in(&mut self.world, id) else {
            return Err(anyhow::anyhow!("unknown npc {id:?}"));
        };
        let (new_part, vital_min) = {
            let mut bp = self
                .world
                .get_mut::<BodyParts>(e)
                .ok_or_else(|| anyhow::anyhow!("npc {id:?} has no BodyParts"))?;
            let slot = bp.get_mut(part);
            *slot = (*slot + amount).clamp(0.0, BodyParts::DEFAULT_MAX);
            (*slot, bp.vital_min())
        };
        if let Some(mut h) = self.world.get_mut::<Health>(e) {
            h.current = vital_min.min(h.max);
        }
        self.record_delta(WorldDelta::SetNpcBodyPart {
            id,
            part,
            current: new_part,
        })?;
        Ok(())
    }

    /// Set a player's stamina to a specific value, clamped to `[0, max]`.
    /// Journaled. Regen keeps running on top of whatever you set.
    pub fn set_stamina(&mut self, steam_id: u64, value: f32) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let new = {
            let mut s = self
                .world
                .get_mut::<Stamina>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Stamina"))?;
            s.current = value.max(0.0).min(s.max);
            s.current
        };
        self.record_delta(WorldDelta::SetStamina {
            steam_id,
            current: new,
        })?;
        Ok(())
    }

    /// Iterate all known players. Order is unspecified. Takes
    /// `&mut self` for the same reason as [`Self::player_view`].
    #[allow(clippy::type_complexity)]
    pub fn each_player<F: FnMut(PlayerView)>(&mut self, mut f: F) {
        let mut q = self.world.query::<(
            &PlayerOwned,
            &Position,
            &Rotation,
            &InRegion,
            Option<&Health>,
            Option<&Stamina>,
            Option<&BodyParts>,
            Option<&SurvivalStats>,
            Option<&Wounds>,
            Option<&Pain>,
            Option<&Contamination>,
            Option<&ActiveEffects>,
            Option<&DrugTolerance>,
        )>();
        for (p, pos, rot, r, h, s, bp, sv, w, pn, cn, ef, dt) in q.iter(&self.world) {
            f(PlayerView {
                steam_id: p.steam_id,
                region: r.0,
                pos: pos.0,
                yaw: rot.0,
                health: h.copied().unwrap_or(Health::new_full()),
                stamina: s.copied().unwrap_or(Stamina::new_full()),
                body_parts: bp.copied().unwrap_or(BodyParts::new_full()),
                survival: sv.copied().unwrap_or(SurvivalStats::new_full()),
                wounds: w.map(|w| w.0.clone()).unwrap_or_default(),
                pain: pn.copied().unwrap_or_default(),
                contamination: cn.copied().unwrap_or_default(),
                active_effects: ef.map(|e| e.0.clone()).unwrap_or_default(),
                drug_tolerance: dt.map(|t| t.0.clone()).unwrap_or_default(),
            });
        }
    }

    pub(super) fn find_player_entity(&mut self, steam_id: u64) -> Option<Entity> {
        let mut q = self.world.query::<(Entity, &PlayerOwned)>();
        q.iter(&self.world)
            .find(|(_, p)| p.steam_id == steam_id)
            .map(|(e, _)| e)
    }
}
