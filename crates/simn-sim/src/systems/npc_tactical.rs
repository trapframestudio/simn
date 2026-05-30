//! Tactical combat brain. Runs for every NPC with `Aggro` and
//! decides their `CombatStance` via the GOAP planner. The stance
//! drives movement in `tick_npc_goals` and fire gating in `npc_combat`.
//!
//! Performance-critical path: runs per aggroed NPC per tick. Key
//! invariants:
//! - `nearest_cover()` is called at most ONCE per NPC per tick (cached)
//! - Suppression detection uses a single pass with no heap allocation
//! - GOAP replanning triggers only when the plan empties, not on a timer

use bevy_ecs::prelude::*;

use crate::components::{
    Aggro, CombatRole, CombatStance, GoapPlanComp, Group, Health, InFaction, InRegion, Npc,
    NpcCharacter, Position, RecentAttackers,
};
use crate::cover::{CoverHeight, CoverVolumes};
use crate::resources::{ActiveRegions, NpcPositionIndex, SimClock};
use crate::squad_blackboard::{BlackboardKey, SquadBlackboards};
use crate::world_event_bus::{ChatterIntent, WorldEventKind, WorldEventQueue};

use crate::behavior_config::BehaviorConfig;

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn npc_tactical(
    clock: Res<SimClock>,
    active_regions: Res<ActiveRegions>,
    index: Res<NpcPositionIndex>,
    mut cover_volumes: ResMut<CoverVolumes>,
    blackboards: Res<SquadBlackboards>,
    faction_registry: Res<crate::faction::registry::FactionRegistry>,
    mut event_queue: ResMut<WorldEventQueue>,
    mut query: Query<(
        Entity,
        &Npc,
        &InFaction,
        &InRegion,
        &Position,
        &Health,
        &Aggro,
        Option<&RecentAttackers>,
        Option<&mut CombatStance>,
        Option<&NpcCharacter>,
        Option<&CombatRole>,
        Option<&Group>,
        Option<&mut GoapPlanComp>,
    )>,
    mut commands: Commands,
) {
    let now = clock.tick;
    let bc = BehaviorConfig::load();
    let cc = &bc.combat;
    let engage_range_sq = bc.movement.engage_range_m * bc.movement.engage_range_m;

    // Collect cover claim/release actions to apply after the
    // query iteration (can't mutate cover_volumes during the loop).
    let mut cover_claims: Vec<(u64, crate::components::NpcId)> = Vec::new();
    let mut cover_releases: Vec<crate::components::NpcId> = Vec::new();

    for (
        entity,
        _npc,
        npc_faction,
        region,
        pos,
        health,
        aggro,
        attackers,
        stance,
        character,
        role,
        group,
        mut goap_plan,
    ) in query.iter_mut()
    {
        if !active_regions.is_active(region.0) {
            continue;
        }

        let target_entry = index
            .by_id
            .get(&aggro.target)
            .copied()
            .filter(|e| e.region == region.0 && e.health > 0.0);

        let Some(target) = target_entry else {
            if stance.is_some() {
                commands.entity(entity).remove::<CombatStance>();
            }
            continue;
        };

        // --- Threat assessment (zero-alloc) ---
        let dx = target.pos[0] - pos.0[0];
        let dz = target.pos[2] - pos.0[2];
        let dist_sq = dx * dx + dz * dz;
        let threat_dir = if dist_sq > 0.01 {
            let d = dist_sq.sqrt();
            [dx / d, dz / d]
        } else {
            [1.0, 0.0]
        };

        // Single-pass suppression: count hits AND distinct attackers
        // without allocating a HashSet. Uses a u64 bitmap for attacker
        // ID deduplication (hash-collisions are acceptable for a
        // threshold check).
        let (recent_hits, distinct_attackers) = attackers
            .map(|ra| {
                let mut hit_count = 0usize;
                let mut id_bits = 0u64;
                let mut distinct = 0usize;
                for h in &ra.events {
                    if now.saturating_sub(h.tick) < cc.suppression_window_ticks {
                        hit_count += 1;
                        let bit = 1u64 << (h.attacker_id.0 % 64);
                        if id_bits & bit == 0 {
                            id_bits |= bit;
                            distinct += 1;
                        }
                    }
                }
                (hit_count, distinct)
            })
            .unwrap_or((0, 0));

        let is_suppressed = recent_hits >= cc.suppression_hit_threshold
            || distinct_attackers >= cc.suppression_attacker_threshold;
        let taking_fire = attackers
            .map(|ra| ra.events.iter().any(|h| now.saturating_sub(h.tick) < 60))
            .unwrap_or(false);
        let is_cautious = character.map(|c| c.personality.cautious).unwrap_or(false);
        let health_frac = health.current / health.max;

        // --- Role assignment (first combat entry only) ---
        if role.is_none() {
            if let Some(ch) = character {
                commands
                    .entity(entity)
                    .insert(CombatRole::assign(&ch.stats, &ch.personality));
            }
        }
        let current_role = role.copied().unwrap_or(CombatRole::Support);

        // --- Squad state ---
        let squad_should_retreat = group
            .map(|g| {
                blackboards
                    .get(g.id)
                    .map(|bb| {
                        bb.iter()
                            .filter(|(k, _)| matches!(k, BlackboardKey::DownedAlly { .. }))
                            .count()
                            >= 2
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        // --- Cover lookup (ONCE per NPC per tick) ---
        let search_radius = if is_cautious {
            cc.cover_search_radius_m * 1.5
        } else {
            cc.cover_search_radius_m
        };
        let nearby_cover_id: Option<u64> = cover_volumes
            .nearest_cover_for(
                region.0,
                pos.0,
                threat_dir,
                search_radius,
                CoverHeight::High,
                Some(_npc.id),
            )
            .map(|v| v.id);
        let cover_available = nearby_cover_id.is_some();

        // --- GOAP planning ---
        let mut ws = crate::goap::WorldState(crate::goap::HAS_TARGET | crate::goap::HAS_AMMO);
        if dist_sq <= engage_range_sq {
            ws = ws.set(crate::goap::IN_RANGE);
        }
        if cover_available {
            ws = ws.set(crate::goap::COVER_AVAILABLE);
        }
        if matches!(
            stance.as_deref().copied(),
            Some(CombatStance::InCover { .. })
        ) {
            ws = ws.set(crate::goap::IN_COVER).set(crate::goap::HAS_LOS);
        }
        if matches!(
            stance.as_deref().copied(),
            Some(CombatStance::Firing { .. })
        ) {
            ws = ws.set(crate::goap::HAS_LOS).set(crate::goap::IN_RANGE);
        }
        // NPCs within engage range can see the target (LOS) regardless
        // of stance — they don't need to be in Firing or InCover to
        // have line of sight. This lets them shoot while moving.
        if dist_sq <= engage_range_sq {
            ws = ws.set(crate::goap::HAS_LOS);
        }
        if taking_fire {
            ws = ws.set(crate::goap::TAKING_FIRE);
        }
        if is_suppressed {
            ws = ws.set(crate::goap::IS_SUPPRESSED);
        }
        if health_frac < cc.retreat_health_frac {
            ws = ws.set(crate::goap::HEALTH_LOW);
        }
        if squad_should_retreat {
            ws = ws.set(crate::goap::SQUAD_RETREATING);
        }

        // Replan when: plan empty, plan stale (>200 ticks of same
        // stance), or key world state changed since last plan.
        let replan_bits = crate::goap::TAKING_FIRE
            | crate::goap::HEALTH_LOW
            | crate::goap::IS_SUPPRESSED
            | crate::goap::COVER_AVAILABLE;
        let ws_key = ws.0 & replan_bits;
        let needs_replan = match &goap_plan {
            Some(p) => {
                p.actions.is_empty()
                    || p.is_stale(now, 200)
                    || (p.last_world_state & replan_bits) != ws_key
            }
            None => true,
        };

        if needs_replan {
            let faction_name = faction_registry.name_of(npc_faction.0);
            let has_ally_down = group
                .map(|g| {
                    blackboards
                        .get(g.id)
                        .map(|bb| {
                            bb.iter()
                                .any(|(k, _)| matches!(k, BlackboardKey::DownedAlly { .. }))
                        })
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            let goals = crate::goap_actions::combat_goals(
                health_frac,
                taking_fire,
                has_ally_down && current_role == CombatRole::Medic,
                squad_should_retreat,
            );
            let actions = crate::goap_actions::combat_actions(faction_name, current_role);
            if let Some(plan) = crate::goap::plan(ws, &goals, &actions, 6) {
                let comp = GoapPlanComp {
                    actions: plan.actions,
                    planned_at_tick: now,
                    last_world_state: ws.0,
                };
                match goap_plan {
                    Some(ref mut existing) => **existing = comp,
                    None => {
                        commands.entity(entity).insert(comp);
                    }
                }
            }
        }

        // --- Map GOAP action → CombatStance ---
        let goap_action = goap_plan.as_deref().and_then(|p| p.current_action());
        let peek_offset = (_npc.id.0 % 4) * cc.peek_interval_ticks / 4;

        let new_stance = if squad_should_retreat {
            CombatStance::Retreating
        } else if is_suppressed
            && !matches!(stance.as_deref().copied(), Some(CombatStance::Retreating))
        {
            CombatStance::Suppressed {
                until_tick: now + cc.suppression_duration_ticks,
            }
        } else if let Some(action) = goap_action {
            match action {
                "MoveToCover" => {
                    if let Some(vol_id) = nearby_cover_id {
                        CombatStance::InCover {
                            volume_id: vol_id,
                            peek_until_tick: 0,
                            next_peek_tick: now + cc.peek_interval_ticks / 2 + peek_offset,
                        }
                    } else {
                        CombatStance::Firing { since_tick: now }
                    }
                }
                "PeekFromCover" | "Shoot" | "Suppress" => {
                    if let Some(CombatStance::InCover { volume_id, .. }) =
                        stance.as_deref().copied()
                    {
                        let (peek_dur, peek_int) = role_peek_timing(current_role);
                        CombatStance::InCover {
                            volume_id,
                            peek_until_tick: now + peek_dur,
                            next_peek_tick: now + peek_int + peek_offset,
                        }
                    } else {
                        CombatStance::Firing { since_tick: now }
                    }
                }
                "Advance" => {
                    if current_role == CombatRole::Pointman
                        && health_frac > cc.pointman_push_health_frac
                    {
                        CombatStance::Firing { since_tick: now }
                    } else {
                        CombatStance::Approaching
                    }
                }
                "Flank" => CombatStance::Flanking,
                "Retreat" => CombatStance::Retreating,
                "Reload" | "HealAlly" => {
                    if let Some(CombatStance::InCover {
                        volume_id,
                        peek_until_tick: _,
                        next_peek_tick,
                    }) = stance.as_deref().copied()
                    {
                        CombatStance::InCover {
                            volume_id,
                            peek_until_tick: 0,
                            next_peek_tick,
                        }
                    } else if let Some(vol_id) = nearby_cover_id {
                        CombatStance::InCover {
                            volume_id: vol_id,
                            peek_until_tick: 0,
                            next_peek_tick: now + cc.peek_interval_ticks + peek_offset,
                        }
                    } else {
                        CombatStance::Approaching
                    }
                }
                _ => stance
                    .as_deref()
                    .copied()
                    .unwrap_or(CombatStance::Approaching),
            }
        } else {
            // No GOAP plan — role-driven fallback.
            match stance.as_deref().copied() {
                None => CombatStance::Approaching,
                Some(CombatStance::Approaching) if dist_sq <= engage_range_sq => {
                    // Flankers move to a lateral position instead of
                    // standing and firing from the front.
                    if current_role == CombatRole::Flanker {
                        CombatStance::Flanking
                    } else if let Some(vol_id) = nearby_cover_id {
                        CombatStance::InCover {
                            volume_id: vol_id,
                            peek_until_tick: 0,
                            next_peek_tick: now + cc.peek_interval_ticks / 2 + peek_offset,
                        }
                    } else {
                        CombatStance::Firing { since_tick: now }
                    }
                }
                Some(CombatStance::Suppressed { until_tick }) if now >= until_tick => {
                    // After suppression, reposition — don't pop
                    // back up in the same spot.
                    if current_role == CombatRole::Flanker {
                        CombatStance::Flanking
                    } else if let Some(vol_id) = nearby_cover_id {
                        CombatStance::InCover {
                            volume_id: vol_id,
                            peek_until_tick: 0,
                            next_peek_tick: now + cc.peek_interval_ticks / 2 + peek_offset,
                        }
                    } else {
                        CombatStance::Firing { since_tick: now }
                    }
                }
                Some(s) => s,
            }
        };

        // Advance plan on stance transition
        if goap_action.is_some() {
            let transitioned_goap = stance
                .as_deref()
                .copied()
                .map(|o| std::mem::discriminant(&o) != std::mem::discriminant(&new_stance))
                .unwrap_or(true);
            if transitioned_goap {
                if let Some(ref mut plan) = goap_plan {
                    plan.advance();
                }
            }
        }

        // --- Chatter on stance transitions ---
        let old_stance = stance.as_deref().copied();
        let transitioned = old_stance
            .map(|o| std::mem::discriminant(&o) != std::mem::discriminant(&new_stance))
            .unwrap_or(true);
        if transitioned {
            let chatter = match (&old_stance, &new_stance) {
                (None, CombatStance::Approaching) => Some((ChatterIntent::Alarm, 2)),
                (_, CombatStance::Suppressed { .. }) => Some((ChatterIntent::Callout, 1)),
                (_, CombatStance::Flanking) => Some((ChatterIntent::Callout, 1)),
                (_, CombatStance::Retreating) => Some((ChatterIntent::Callout, 2)),
                _ => None,
            };
            if let Some((intent, ttl)) = chatter {
                event_queue.push(
                    WorldEventKind::Chatter {
                        speaker: _npc.id,
                        intent,
                    },
                    pos.0,
                    region.0,
                    now,
                    ttl,
                );
            }
        }

        // Track cover claims for post-loop application.
        let entering_cover = matches!(new_stance, CombatStance::InCover { .. });
        let leaving_cover = stance
            .as_deref()
            .copied()
            .map(|s| matches!(s, CombatStance::InCover { .. }) && !entering_cover)
            .unwrap_or(false);
        if entering_cover {
            if let CombatStance::InCover { volume_id, .. } = new_stance {
                cover_claims.push((volume_id, _npc.id));
            }
        }
        if leaving_cover {
            cover_releases.push(_npc.id);
        }

        match stance {
            Some(mut s) => *s = new_stance,
            None => {
                commands.entity(entity).insert(new_stance);
            }
        }
    }

    // Apply cover claims/releases after the query iteration.
    for npc_id in cover_releases {
        cover_volumes.release_all_for_npc(npc_id);
    }
    for (vol_id, npc_id) in cover_claims {
        cover_volumes.claim_cover(vol_id, npc_id);
    }
}

fn role_peek_timing(role: CombatRole) -> (u64, u64) {
    let cc = BehaviorConfig::load().combat;
    let peek = cc.peek_duration_ticks;
    let interval = cc.peek_interval_ticks;
    match role {
        CombatRole::Pointman => (peek + 10, interval - 10),
        CombatRole::Support => (peek + 20, interval + 10),
        CombatRole::Flanker => (peek, interval - 5),
        CombatRole::Medic => (peek.saturating_sub(5), interval + 20),
    }
}
