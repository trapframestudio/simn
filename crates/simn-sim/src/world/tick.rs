//! Tick loop, schedule construction, and per-tick perf reporting.

use anyhow::{Context, Result};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::Schedule;

use crate::delta::WorldDelta;
use crate::resources::PendingDeltas;
use crate::systems::{
    advance_clock, advance_weather, advance_world_time, age_and_heal_wounds, age_npcs,
    apply_bleed_damage, apply_survival_effects, decay_drug_tolerance, drain_survival_stats,
    goal_arbitration, index_npc_positions, npc_aggro, npc_combat, npc_death_check, npc_join_group,
    npc_portal_cross, npc_treat_wounds, rebuild_spatial_hash, regen_stamina, spawn_npcs,
    squad_planner, tick_active_effects, tick_contamination, tick_crafting_queue, tick_infection,
    tick_necrosis, tick_npc_goals, tick_pain, tick_perishables,
};

use super::{Sim, TickPerfReport, TickSegments, TICK_PERF_WINDOW};

impl Sim {
    /// Advance one simulation tick. Runs the schedule, drains any
    /// system-emitted deltas to the journal + broadcast buffer,
    /// flushes the journal (fsync'd periodically), and rolls a
    /// snapshot if we've crossed the snapshot interval.
    ///
    /// On mirror sims (no journal / no save) this still runs the
    /// mirror schedule (pure per-tick player systems) so things like
    /// stamina regen, pain decay, world time advance, and perishable
    /// expiry animate between host deltas.
    pub fn tick(&mut self) -> Result<()> {
        // DIAGNOSTIC tick timer. Gated by the global verbose-logging
        // flag (see [`crate::systems::set_verbose_logging`]). Logs
        // any tick that runs >25ms (half a sim-tick budget at 20Hz)
        // so outliers stand out. `SIMN_TICK_VERBOSE=1` additionally
        // logs every tick — kept as an env knob since "log every
        // tick" is a profiling concern, not a normal-play one.
        let _diag_verbose_ticks = std::env::var("SIMN_TICK_VERBOSE").is_ok();
        let _diag_start = crate::systems::is_verbose_logging().then(std::time::Instant::now);
        // Always-on rolling perf timer (cheap; ~1 µs per tick) so
        // the bridge's `tick_perf()` can report avg/p99 without
        // requiring a process restart with `SIMN_VERBOSE=1`.
        let perf_start = std::time::Instant::now();

        // Five-segment schedule. Each runs sequentially in `tick()`;
        // ordering across segments is preserved by the call sequence
        // (bevy's auto-ordering only reaches within a single
        // Schedule). Per-segment timing surfaces via `tick_perf()`.
        let mut segments = TickSegments::default();

        let t0 = std::time::Instant::now();
        self.schedule_player.run(&mut self.world);
        segments.player = t0.elapsed();

        let t1a = std::time::Instant::now();
        self.schedule_npc_index.run(&mut self.world);
        segments.npc_index = t1a.elapsed();
        let slots = crate::systems::drain_perception_slots();
        segments.clear_los = slots[crate::systems::prof_slots::CLEAR_LOS];
        segments.sweep_bb = slots[crate::systems::prof_slots::SWEEP_BB];
        segments.position_index = slots[crate::systems::prof_slots::POSITION_INDEX];
        segments.drain_events = slots[crate::systems::prof_slots::DRAIN_EVENTS];
        segments.spatial_hash = slots[crate::systems::prof_slots::SPATIAL_HASH];
        segments.event_count = crate::systems::drain_event_count() as u32;

        let t1b = std::time::Instant::now();
        self.schedule_npc_threats.run(&mut self.world);
        segments.npc_threats = t1b.elapsed();

        let t1c = std::time::Instant::now();
        self.schedule_npc_aggro.run(&mut self.world);
        segments.npc_aggro = t1c.elapsed();

        segments.npc_perception = segments.npc_index + segments.npc_threats + segments.npc_aggro;

        let t2 = std::time::Instant::now();
        self.schedule_npc_planning.run(&mut self.world);
        segments.npc_planning = t2.elapsed();
        // Drain planning sub-slots (same thread-local array as
        // perception's). Each system records its elapsed via
        // `ProfGuard` on drop.
        let plan_slots = crate::systems::drain_perception_slots();
        segments.squad_planner = plan_slots[crate::systems::prof_slots::SQUAD_PLANNER];
        segments.goal_arbitration = plan_slots[crate::systems::prof_slots::GOAL_ARBITRATION];
        segments.tick_npc_goals = plan_slots[crate::systems::prof_slots::TICK_NPC_GOALS];
        segments.npc_combat = plan_slots[crate::systems::prof_slots::NPC_COMBAT];

        let t3 = std::time::Instant::now();
        self.schedule_npc_lifecycle.run(&mut self.world);
        segments.npc_lifecycle = t3.elapsed();

        let t4 = std::time::Instant::now();
        self.schedule_offline_loot.run(&mut self.world);
        segments.offline_loot = t4.elapsed();

        let schedule_elapsed = segments.player
            + segments.npc_perception
            + segments.npc_planning
            + segments.npc_lifecycle
            + segments.offline_loot;

        let proj_t = std::time::Instant::now();
        // Advance in-flight projectiles. Host-authoritative — skipped
        // on mirror sims since they don't own `Projectile` entities
        // (they replay spawn/impact deltas as FX only).
        if !self
            .world
            .contains_resource::<crate::resources::MirrorMode>()
        {
            // Phase 4A v1: drain the NPC shot intents queued by
            // `npc_combat` and spawn the (cosmetic) projectile
            // entities BEFORE the projectile-tick pass so tracers
            // get one tick of advancement on the same frame they
            // appear. RNG salted from sim tick + a constant — same
            // pattern as other tick-derived RNGs.
            let intents = self
                .world
                .resource_mut::<crate::resources::PendingNpcShots>()
                .drain();
            if !intents.is_empty() {
                use rand::SeedableRng;
                let tick = self.world.resource::<crate::resources::SimClock>().tick;
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
                    tick.wrapping_mul(0xA1F2_3D5E_7B9C_4061),
                );
                for intent in intents {
                    if let Err(e) = self.npc_fire_projectile(
                        intent.shooter_id,
                        intent.shooter_pos,
                        intent.shooter_region,
                        intent.target_pos,
                        intent.accuracy,
                        intent.round_id.clone(),
                        &mut rng,
                    ) {
                        tracing::debug!("npc_fire_projectile drain: {e:#}");
                    }
                }
            }
            self.tick_projectiles()?;
        }
        let proj_elapsed = proj_t.elapsed();

        // Drain deltas pushed by ECS systems this tick (NPC spawns,
        // region migrations, deaths, NPC position batch) + tally into
        // `BehaviorLog` if enabled.
        let pending = self.world.resource_mut::<PendingDeltas>().drain();
        let log_enabled = self
            .world
            .resource::<crate::resources::BehaviorLog>()
            .enabled;
        if log_enabled {
            let mut log = self.world.resource_mut::<crate::resources::BehaviorLog>();
            for delta in &pending {
                tally_delta(delta, &mut log);
            }
        }
        for delta in pending {
            self.record_delta(delta).context("append system delta")?;
        }

        // Flush batched summary every FLUSH_INTERVAL ticks.
        {
            let t = self.current_tick();
            let mut log = self.world.resource_mut::<crate::resources::BehaviorLog>();
            if log.enabled
                && t > 0
                && t.wrapping_sub(log.last_flush_tick)
                    >= crate::resources::BehaviorLog::FLUSH_INTERVAL
            {
                emit_summary(t, &log);
                log.last_flush_tick = t;
                log.reset_counters();
            }
        }

        let tick = self.current_tick();
        if let Some(ref mut journal) = self.journal {
            journal
                .append(&WorldDelta::Tick { tick })
                .context("append tick marker")?;
            journal.maybe_fsync()?;
        }

        if tick > 0 && tick.is_multiple_of(self.snapshot_interval) && self.save_paths.is_some() {
            self.roll_snapshot(tick)?;
        }

        // Publish a render-facing snapshot of active-region NPC poses.
        // Rotates the 2-slot ring: `[prev, curr]` becomes
        // `[old_curr, new_snapshot]`. Renderer reads the pair via
        // `snapshot_pair()` and lerps. See snapshot.rs for the
        // contract.
        self.publish_snapshot(tick);

        segments.total = perf_start.elapsed();
        if self.tick_perf_history.len() == TICK_PERF_WINDOW {
            self.tick_perf_history.pop_front();
        }
        self.tick_perf_history.push_back(segments);

        if let Some(start) = _diag_start {
            let total = start.elapsed();
            if _diag_verbose_ticks || total.as_millis() > 25 {
                eprintln!(
                    "[sim.tick tick={}] total={:?} schedule={:?} projectiles={:?}",
                    tick, total, schedule_elapsed, proj_elapsed
                );
            }
        }

        Ok(())
    }

    /// Rolling snapshot of recent tick perf. Returns
    /// [`TickPerfReport`] with avg + p99 for each segment + total
    /// over the last [`TICK_PERF_WINDOW`] ticks (or fewer, if the
    /// sim hasn't accumulated that many yet). All zeros before the
    /// first tick completes. Cheap — sorts a stack copy of the
    /// durations to pick out the p99 per segment.
    pub fn recent_tick_perf(&self) -> TickPerfReport {
        let n = self.tick_perf_history.len();
        let mut r = TickPerfReport {
            samples: n,
            ..Default::default()
        };
        if n == 0 {
            return r;
        }
        let to_ms = |d: std::time::Duration| d.as_secs_f32() * 1_000.0;
        let p99_idx = ((n as f32 * 0.99) as usize).min(n - 1);
        let mut buf: Vec<f32> = Vec::with_capacity(n);
        let mut pull = |get: fn(&TickSegments) -> std::time::Duration| -> (f32, f32) {
            buf.clear();
            buf.extend(self.tick_perf_history.iter().map(|s| to_ms(get(s))));
            let avg = buf.iter().sum::<f32>() / n as f32;
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (avg, buf[p99_idx])
        };
        (r.avg_player_ms, r.p99_player_ms) = pull(|s| s.player);
        (r.avg_npc_index_ms, r.p99_npc_index_ms) = pull(|s| s.npc_index);
        (r.avg_clear_los_ms, r.p99_clear_los_ms) = pull(|s| s.clear_los);
        (r.avg_sweep_bb_ms, r.p99_sweep_bb_ms) = pull(|s| s.sweep_bb);
        (r.avg_position_index_ms, r.p99_position_index_ms) = pull(|s| s.position_index);
        (r.avg_drain_events_ms, r.p99_drain_events_ms) = pull(|s| s.drain_events);
        (r.avg_spatial_hash_ms, r.p99_spatial_hash_ms) = pull(|s| s.spatial_hash);
        let total_ev: u64 = self
            .tick_perf_history
            .iter()
            .map(|s| s.event_count as u64)
            .sum();
        r.avg_event_count = total_ev as f32 / n as f32;
        r.max_event_count = self
            .tick_perf_history
            .iter()
            .map(|s| s.event_count)
            .max()
            .unwrap_or(0);
        (r.avg_npc_threats_ms, r.p99_npc_threats_ms) = pull(|s| s.npc_threats);
        (r.avg_npc_aggro_ms, r.p99_npc_aggro_ms) = pull(|s| s.npc_aggro);
        (r.avg_npc_perception_ms, r.p99_npc_perception_ms) = pull(|s| s.npc_perception);
        (r.avg_npc_planning_ms, r.p99_npc_planning_ms) = pull(|s| s.npc_planning);
        (r.avg_squad_planner_ms, r.p99_squad_planner_ms) = pull(|s| s.squad_planner);
        (r.avg_goal_arbitration_ms, r.p99_goal_arbitration_ms) = pull(|s| s.goal_arbitration);
        (r.avg_tick_npc_goals_ms, r.p99_tick_npc_goals_ms) = pull(|s| s.tick_npc_goals);
        (r.avg_npc_combat_ms, r.p99_npc_combat_ms) = pull(|s| s.npc_combat);
        (r.avg_npc_lifecycle_ms, r.p99_npc_lifecycle_ms) = pull(|s| s.npc_lifecycle);
        (r.avg_offline_loot_ms, r.p99_offline_loot_ms) = pull(|s| s.offline_loot);
        (r.avg_total_ms, r.p99_total_ms) = pull(|s| s.total);
        r
    }
}

/// Update `BehaviorLog` counters from one delta, and emit an
/// individual tracing line for rare/interesting events (deaths,
/// migrations). Called per-delta when logging is enabled.
fn tally_delta(delta: &WorldDelta, log: &mut crate::resources::BehaviorLog) {
    match delta {
        WorldDelta::NpcSpawned { faction, .. } => {
            log.spawns += 1;
            *log.spawns_by_faction.entry(faction.clone()).or_insert(0) += 1;
        }
        WorldDelta::NpcChangeRegion { region, .. } => {
            log.migrations += 1;
            *log.migrations_by_region.entry(*region).or_insert(0) += 1;
        }
        WorldDelta::NpcDied { cause, .. } => {
            log.deaths += 1;
            *log.deaths_by_cause
                .entry(crate::chronicle::death_cause_to_str(cause))
                .or_insert(0) += 1;
        }
        _ => {}
    }
}

/// Emit the periodic summary line over the interval just closed.
fn emit_summary(tick: u64, log: &crate::resources::BehaviorLog) {
    if log.spawns == 0
        && log.deaths == 0
        && log.migrations == 0
        && log.aggro_acquisitions == 0
        && log.objectives.is_empty()
    {
        return;
    }
    let objs = format_counts_str(&log.objectives);
    let spawns_fac = format_faction_counts(&log.spawns_by_faction);
    let death_kinds = format_counts_str(&log.deaths_by_cause);
    tracing::info!(
        target: "npc.behavior",
        "tick={} spawns={}{} deaths={}{} migrations={} aggro={} objectives=[{}]",
        tick,
        log.spawns,
        if spawns_fac.is_empty() { String::new() } else { format!("({spawns_fac})") },
        log.deaths,
        if death_kinds.is_empty() { String::new() } else { format!("({death_kinds})") },
        log.migrations,
        log.aggro_acquisitions,
        objs,
    );
}

fn format_counts_str(m: &std::collections::HashMap<&'static str, u32>) -> String {
    let mut v: Vec<(&&str, &u32)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    v.iter()
        .map(|(k, n)| format!("{k}:{n}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_faction_counts(m: &std::collections::HashMap<String, u32>) -> String {
    let mut v: Vec<(&String, &u32)> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    v.iter()
        .take(4)
        .map(|(f, n)| format!("{f}:{n}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Empty schedule placeholder; used by the mirror builder for
/// segments that don't run on client-side sims (NPC pipeline +
/// offline tier are host-authoritative).
pub(crate) fn build_schedule_empty() -> Schedule {
    Schedule::default()
}

/// Segment 1 — player-facing tick: clock, world time, weather,
/// survival, meds, wounds, contamination, stamina. Chained so a
/// wound's bleed updates happen before stamina regen reads pain in
/// the same tick. Identical between authoritative and mirror sims.
pub(crate) fn build_schedule_player() -> Schedule {
    let mut schedule = Schedule::default();
    // Split into two chains because bevy's `.chain()` tuple bound
    // caps at 16 systems. First chain wraps everything through
    // wound healing; second chain depends on the first via
    // `after(age_and_heal_wounds)` so ordering is preserved.
    schedule.add_systems(
        (
            advance_clock,
            advance_world_time,
            advance_weather,
            drain_survival_stats,
            apply_survival_effects,
            tick_active_effects,
            decay_drug_tolerance,
            tick_infection,
            tick_necrosis,
            apply_bleed_damage,
            age_and_heal_wounds,
        )
            .chain(),
    );
    schedule.add_systems(
        (
            npc_treat_wounds,
            tick_pain,
            tick_contamination,
            tick_perishables,
            tick_crafting_queue,
            regen_stamina,
        )
            .chain()
            .after(age_and_heal_wounds),
    );
    schedule
}

/// Same as [`build_schedule_player`] but for mirror sims — the
/// chain itself is bit-identical; the alias clarifies intent at
/// call sites.
pub(crate) fn build_schedule_player_mirror() -> Schedule {
    build_schedule_player()
}

/// Segment 2a — position index, spatial hash rebuild, world-event
/// drain, LOS-cache clear, squad-blackboard sweep. Linear in NPC
/// count; the cheap setup the dense pair-scan downstream depends on.
pub(crate) fn build_schedule_npc_index() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            crate::los_cache::clear_los_cache,
            crate::squad_blackboard::sweep_squad_blackboards,
            index_npc_positions,
            crate::world_event_bus::drain_world_events,
            rebuild_spatial_hash,
        )
            .chain(),
    );
    schedule
}

/// Segment 2b — threat-board sweep. Runs after the position index
/// is refreshed (proximity scoring keys off it) but before any
/// consumer (goal_arbitration / npc_combat).
pub(crate) fn build_schedule_npc_threats() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(crate::systems::sweep_threats);
    schedule
}

/// Segment 2c — dense pair-scan (`npc_aggro`) and squad-priority
/// resolution. `npc_aggro` runs first because it can newly-acquire
/// Aggro for NPCs that just spotted an enemy; threat priority only
/// matters when an Aggro already exists.
pub(crate) fn build_schedule_npc_aggro() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems((npc_aggro, crate::systems::apply_threat_priority).chain());
    schedule
}

/// Segment 3 — NPC planning + combat: squad planner, goal
/// arbitration, pathfind + movement (`tick_npc_goals`), combat
/// resolution. Pathfind cascade is rayon-parallel but still
/// spike-prone.
pub(crate) fn build_schedule_npc_planning() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            squad_planner,
            goal_arbitration,
            crate::systems::npc_tactical,
        )
            .chain(),
    );
    schedule.add_systems(
        (tick_npc_goals, npc_combat)
            .chain()
            .after(crate::systems::npc_tactical),
    );
    schedule
}

/// Segment 4 — NPC lifecycle + replication broadcast: kill
/// credits, death check, regroup, portal cross, age, spawn,
/// clamp Y, broadcast positions for replication.
pub(crate) fn build_schedule_npc_lifecycle() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            crate::systems::apply_kill_credits,
            npc_death_check,
            npc_join_group,
            npc_portal_cross,
            age_npcs,
            crate::systems::prune_corpse_index,
            spawn_npcs,
            crate::systems::base_capture_check,
            crate::systems::clamp_npc_terrain_y,
            crate::systems::broadcast_npc_positions,
        )
            .chain(),
    );
    schedule
}

/// Mirror-side variant of segment 4 — only `index_npc_positions`
/// runs so client-side minimap / proximity queries work; every
/// other NPC-mutating system is host-authoritative.
pub(crate) fn build_schedule_npc_index_only_mirror() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(index_npc_positions);
    schedule
}

/// Segment 5 — offline tier (2 Hz heartbeat) + loot restock
/// cadence. Cheap per-tick: both gate internally on cadence
/// counters so the real work fires every N ticks.
///
/// Original `.after(advance_clock)` from the monolithic schedule
/// no longer applies — `advance_clock` lives in segment 1
/// (player), which always runs before segment 5 due to the
/// in-`tick()` call ordering. Bevy's auto-ordering inside a
/// single `Schedule` doesn't reach across schedule boundaries.
pub(crate) fn build_schedule_offline_loot() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.add_systems(
        (
            crate::offline_tier::tick_offline_clock,
            crate::offline_tier::offline_movement,
            crate::offline_tier::offline_combat,
        )
            .chain(),
    );
    schedule.add_systems(crate::systems::tick_loot_restock);
    schedule
}
