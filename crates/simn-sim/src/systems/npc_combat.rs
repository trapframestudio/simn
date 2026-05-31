//! Aggro-driven raycast-style combat between NPCs.
//!
//! Every `FIRE_INTERVAL_TICKS`: for each NPC with `Aggro`, look up
//! the target. If both alive in the same region within sight AND
//! the per-tick `LosCache` shows a fresh exposure ≥
//! [`LOS_FIRE_THRESHOLD`], roll a distance-based hit; on hit, deal
//! `[DAMAGE_MIN, DAMAGE_MAX]` to the target's torso via `BodyParts`,
//! keep the aggregate `Health` mirror in sync, and stamp the target
//! with a `LastDamager` so `npc_death_check` can credit the kill.
//!
//! On every shot (whether or not it hits), the shooter also:
//! - pushes a [`WorldEventKind::Gunshot`] into the
//!   [`WorldEventQueue`], so other squads' blackboards pick up a
//!   `HeardGunshot` write next tick (audible-radius gated per
//!   caliber); and
//! - on a hit, writes [`BlackboardKey::UnderFireAt`] to the target's
//!   group blackboard pointing at the shooter's position, so the
//!   victim's squad can react via goal arbitration next tick.
//!
//! ## LOS gate
//!
//! `npc_aggro` populates the [`LosCache`] for any pair that passed
//! the FOV cone. `npc_combat` reads the same cache: no entry (target
//! is behind the shooter, out of FOV, or aggro decayed without
//! refresh) → no shot. Entry present but exposure <
//! [`LOS_FIRE_THRESHOLD`] → no shot. This is the interim stopgap
//! against wall-piercing dice rolls; the long-term replacement is
//! the projectile-collision path from [`physical-combat-plan.md`]
//! per the 2026-05-05 scope-reduction decision in
//! `combat-los-plan.md`.
//!
//! Damage **does not** journal — it's treated like the per-tick pure
//! systems (regen_stamina, advance_world_time): recovered from the
//! latest snapshot on load, with up to one snapshot interval of
//! drift. `npc_death_check` *does* journal `NpcDied` since that's a
//! discrete event (and the chronicle entry must survive). Combat in
//! flight on a crash resumes with HP from the last snapshot;
//! perception re-acquires.
//!
//! "Raycast" here is just a same-region distance + probability roll —
//! no geometry test. Real geometry-aware combat lands with the
//! tactical-AI / tactical-map slice (`docs/walkthrough/tactical-ai.md`).
//! NPC-vs-NPC damage always lands on the torso in this probabilistic
//! model; player weapon raycasts go through
//! `Sim::apply_damage_to_npc_part` and can resolve any body part.

use bevy_ecs::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::components::{Aggression, Aggro, InFaction, InRegion, Npc, NpcCharacter, Position};
use crate::los_cache::LosCache;
use crate::resources::{ActiveRegions, NpcPositionIndex, SimClock};
use crate::world_event_bus::{CaliberClass, WorldEventKind, WorldEventQueue};

const FIRE_INTERVAL_TICKS: u64 = 50;
const SIGHT_RADIUS_M: f32 = 80.0;
const SIGHT_RADIUS_SQ_M: f32 = SIGHT_RADIUS_M * SIGHT_RADIUS_M;

/// Per-shooter hit-chance multiplier from the `accuracy` stat. Linear
/// in `[0.7, 1.3]` over accuracy `0..=100`, 1.0× at 50.
///
/// **Phase 4A v2:** retained for legacy callers + the
/// `accuracy_combat_endpoints` unit test, but no longer consumed
/// by `npc_combat` — fire decisions are pass/fail per shooter
/// and accuracy now drives cone-of-fire jitter in
/// `Sim::npc_fire_projectile`.
pub fn accuracy_hit_multiplier(accuracy: u8) -> f32 {
    ACCURACY_MULT_BIAS + ACCURACY_MULT_SLOPE * f32::from(accuracy)
}

const ACCURACY_MULT_BIAS: f32 = 0.7;
const ACCURACY_MULT_SLOPE: f32 = 0.006;

/// How long a damage event survives in `RecentAttackers` /
/// the squad threat board before the sweep system drops it. ~30s
/// at 20 Hz — long enough for an NPC to "remember" being shot at
/// across cover transitions but not so long that an old hit keeps
/// dominating threat scoring.
pub const THREAT_TTL_TICKS: u64 = 600;
/// FIFO cap on the per-NPC `RecentAttackers` ring. Hits from
/// existing attackers accumulate damage in place (so a stream of
/// fire from one shooter doesn't crowd out other threats from the
/// cap). Tuned for typical squad combat: a 5-NPC squad facing 4
/// attackers fits comfortably under the cap with one slot left for
/// the next emerging threat. Used by the projectile-tick
/// attribution helper.
pub const MAX_RECENT_ATTACKERS: usize = 8;

/// Minimum `LosCache` exposure needed to fire on a target.
const LOS_FIRE_THRESHOLD: f32 = 0.33;

/// Fallback caliber class when a round's `ammo_config` can't be
/// resolved. Phase 4B v1's faction → round mapping should hit
/// real entries; this is drift insurance.
const DEFAULT_NPC_CALIBER: CaliberClass = CaliberClass::Intermediate;

/// TTL (ticks) for `Gunshot` events on the bus. Drain runs the next
/// tick and writes `HeardGunshot` to listeners' blackboards (with
/// their own per-key TTL); the bus event itself is short-lived.
const GUNSHOT_EVENT_TTL_TICKS: u32 = 2;

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn npc_combat(
    clock: Res<SimClock>,
    shooters: Query<(
        &Npc,
        &InFaction,
        &InRegion,
        &Position,
        &Aggro,
        Option<&Aggression>,
        Option<&NpcCharacter>,
        Option<&crate::components::CombatStance>,
    )>,
    index: Res<NpcPositionIndex>,
    los_cache: Res<LosCache>,
    active_regions: Res<ActiveRegions>,
    mut event_queue: ResMut<WorldEventQueue>,
    mut pending_shots: ResMut<crate::resources::PendingNpcShots>,
    faction_registry: Res<crate::faction::registry::FactionRegistry>,
    item_registry: Res<crate::items::ItemRegistry>,
) {
    let _diag_t = crate::systems::SysTimer::new("npc_combat");
    let _prof_guard = crate::systems::ProfGuard(
        std::time::Instant::now(),
        crate::systems::prof_slots::NPC_COMBAT,
    );
    if clock.tick == 0 || !clock.tick.is_multiple_of(FIRE_INTERVAL_TICKS) {
        return;
    }
    // Active-region filter: combat skips entire offline regions.
    // Offline regions have no observer; their NPCs' combat lands as
    // dice in the eventual offline-tier sim, not here. Until that
    // exists, online combat in offline regions is wasted work.
    if active_regions.regions.is_empty() {
        // No active region (sim before player joins / after they
        // leave). Don't do any per-NPC combat work.
        return;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(clock.tick.wrapping_mul(0xCAFE_BABE_DEAD_BEEF));

    // Determinism: Bevy's query iteration is in archetype storage
    // order, which isn't stable across sim instances. Collect into a
    // Vec and sort by NpcId before consuming the tick-seeded RNG so
    // fire-decision rolls land in the same order on every run. Caught
    // by `tests/determinism.rs::ticked_sim_snapshots_match_at_intervals`.
    let mut shooters_sorted: Vec<_> = shooters.iter().collect();
    shooters_sorted.sort_by_key(|(npc, _, _, _, _, _, _, _)| npc.id);

    for (
        shooter_npc,
        shooter_fac,
        shooter_region,
        shooter_pos,
        ag,
        aggression,
        character,
        stance,
    ) in shooters_sorted
    {
        // Active-region filter (per-shooter). Skips combat for
        // shooters in offline regions — see comment above.
        if !active_regions.is_active(shooter_region.0) {
            continue;
        }
        if let Some(s) = stance {
            if !s.can_fire(clock.tick) {
                continue;
            }
        }
        let Some(entry) = index.by_id.get(&ag.target).copied() else {
            continue;
        };
        if entry.region != shooter_region.0 || entry.health <= 0.0 {
            continue;
        }
        let dx = entry.pos[0] - shooter_pos.0[0];
        let dz = entry.pos[2] - shooter_pos.0[2];
        let dist_sq = dx * dx + dz * dz;
        if dist_sq > SIGHT_RADIUS_SQ_M {
            continue;
        }
        // Interim LOS gate (`combat-los-plan.md` §6 stopgap). Read
        // the per-tick LosCache that `npc_aggro` populated during
        // its FOV pass; no entry (FOV failed) or low exposure → no
        // shot. Replaces wall-piercing dice combat until the
        // projectile-collision path from `physical-combat-plan.md`
        // takes over.
        let exposure = los_cache.get(shooter_npc.id, ag.target).unwrap_or(0.0);
        if exposure < LOS_FIRE_THRESHOLD {
            continue;
        }
        // Phase 4A v2: aggression governs fire cadence (skip the
        // shot sometimes) instead of hit chance — hit/miss is now
        // a geometric question resolved by the projectile tick.
        // 1.0 = always fire when eligible; 0.5 = fire half the
        // intervals (matches the dice-era effective damage rate).
        if let Some(a) = aggression {
            let fire_chance = 0.5 + 0.5 * a.0.clamp(0.0, 1.0);
            if rng.gen::<f32>() >= fire_chance {
                continue;
            }
        }
        let _ = dist_sq;
        // Phase 4B v1: round id varies by shooter faction so
        // tracer color / impact FX / audible Gunshot bands reflect
        // who's shooting. Raiders fire pistol-caliber, Coalition fires
        // intermediate rifle, etc. Caliber class for the Gunshot
        // event comes from the round's authored `ammo_config`;
        // falls back to the legacy default if the registry can't
        // resolve the round (modded data + drift insurance).
        let shooter_faction_name = faction_registry.name_of(shooter_fac.0);
        let round_id = crate::world::weapons::default_npc_round_for_faction(shooter_faction_name);
        let caliber_class = item_registry
            .get(&round_id)
            .and_then(|def| def.ammo_config.as_ref())
            .map(|ac| ac.caliber_class)
            .unwrap_or(DEFAULT_NPC_CALIBER);
        // Every shot — hit or miss — is audible. Push a Gunshot
        // event at the shooter's position; the drain delivers
        // `HeardGunshot` to listening squads on the next tick.
        event_queue.push(
            WorldEventKind::Gunshot { caliber_class },
            shooter_pos.0,
            shooter_region.0,
            clock.tick,
            GUNSHOT_EVENT_TTL_TICKS,
        );
        // Phase 4A v2: damage flows from the projectile-tick hit
        // branch — npc_combat is purely a fire-decision system
        // now. Push the intent; `Sim::tick`'s drain spawns the
        // projectile, and the projectile-tick resolves hit /
        // damage / attribution (LastDamager, RecentAttackers,
        // kill credits, blackboard `UnderFireAt`) at impact time.
        pending_shots.push(crate::resources::NpcShotIntent {
            shooter_id: shooter_npc.id,
            shooter_pos: shooter_pos.0,
            shooter_region: shooter_region.0,
            target_pos: entry.pos,
            accuracy: character.map(|c| c.stats.accuracy).unwrap_or(50),
            round_id,
        });
        let _ = shooter_fac;
    }
}
