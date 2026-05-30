//! Perception, aggro acquisition, decay, and squad sharing.
//!
//! Every tick:
//! 1. Build an O(n) snapshot of (entity, npc_id, faction, region,
//!    pos, group) for every live NPC.
//! 2. For each NPC currently with `Aggro`, refresh `last_seen_tick`
//!    if the target is still in sight; clear if not (decay).
//! 3. Iterate [`NpcSpatialHash`] — for every region-grid, for every
//!    cell, pair-scan within-cell + 4 directional neighbors. For each
//!    hostile pair within `SIGHT_RADIUS_M` in the same region, set
//!    `Aggro` on both. Propagate to all squadmates sharing the same
//!    `Group.id`.
//!
//! The spatial hash drops the pair scan from `O(Σ n_r²)` to
//! ~`O(N)` in practice — cell-size = 100m vs sight-radius = 80m means
//! a within-cell + 4-neighbor scan finds every possible sight-pair
//! while comparing an order of magnitude fewer actual pairs. That's
//! what unlocked running aggro/combat in offline regions too.

use bevy_ecs::prelude::*;

use crate::components::{Aggro, Group, InFaction, InRegion, Npc, NpcCharacter, Position, Rotation};
use crate::faction::Relation;
use crate::los_cache::LosCache;
use crate::perception::{in_fov, sight_radius_for_perception, LosService, PerceptionConfig};
use crate::region::RegionId;
use crate::resources::{BehaviorLog, NpcSpatialHash, SimClock, SpatialEntry};
use crate::squad_blackboard::{BlackboardKey, BlackboardValue, SquadBlackboards};
use crate::world_event_bus::{WorldEventKind, WorldEventQueue};

/// TTL for `LastKnownEnemyPos` / `LastKnownEnemyId` blackboard
/// entries. Matches `AGGRO_DECAY_TICKS` so the blackboard outlives
/// individual aggro decay (a squadmate may lose direct aggro but the
/// shared "we saw them recently" memory persists).
fn aggro_decay_ticks() -> u64 {
    crate::behavior_config::BehaviorConfig::load()
        .aggro
        .decay_ticks
}
#[allow(non_snake_case)]
fn AGGRO_DECAY_TICKS() -> u64 {
    aggro_decay_ticks()
}
#[allow(non_snake_case)]
fn AGGRO_BB_TTL_TICKS() -> u32 {
    aggro_decay_ticks() as u32
}

/// Cadence (in sim ticks) at which `npc_aggro`'s Pass 2 acquisition
/// scan runs. Pass 1 (decay / refresh) still runs every tick so
/// already-aggroed NPCs keep their `last_seen_tick` fresh; Pass 2
/// is the expensive pair scan and skipping ticks is invisible to
/// gameplay because `AGGRO_DECAY_TICKS = 200` (~10 s). One
/// acquisition opportunity per `PASS_2_TICK_INTERVAL` ticks
/// (~150 ms at 20 Hz) is well inside the decay window.
///
/// The `LosCache` retention window mirrors this: cache entries
/// stay valid for `PASS_2_TICK_INTERVAL` ticks after they're
/// written so consumers (`npc_combat`'s fire gate) can read a
/// fresh exposure on ticks when Pass 2 didn't run. See
/// [`crate::los_cache::clear_los_cache`].
pub const PASS_2_TICK_INTERVAL: u64 = 3;

/// Snapshot-size threshold above which the Pass 2 cadence gate
/// kicks in. Below this, Pass 2 runs every tick — the dense-pair
/// scan cost is negligible at low pop, and tests that tick once
/// then expect immediate aggro acquisition continue to work
/// without changes. Picked generously above realistic test
/// scenarios but well below playtest pop.
const PASS_2_DENSE_THRESHOLD: usize = 64;

/// Per-NPC snapshot used by Pass 1 (decay/refresh of existing aggro)
/// and by `queue_aggro` (squad-share iteration). Pass 2 reads from
/// [`NpcSpatialHash`] directly via [`SpatialEntry`], which carries
/// the pair-scan-relevant fields (pos/yaw/faction/region); `Snap`
/// only needs the fields the other two paths consult.
#[derive(Clone, Copy)]
pub(crate) struct Snap {
    entity: Entity,
    npc_id: crate::components::NpcId,
    region: RegionId,
    pos: [f32; 3],
    group: Option<u64>,
    /// Pre-computed `sight_radius_m^2` for this NPC, factoring in
    /// the perception stat. Pass 1 reads it as the observer's own
    /// range; Pass 2 looks it up by NpcId via `sight_sq_by_id` for
    /// the asymmetric pair check.
    sight_sq: f32,
}

/// Reusable scratch buffers for `npc_aggro`. Long-lived ECS resource
/// — the system clears each buffer at tick start and refills, rather
/// than allocating four fresh containers per tick (a measurable hit
/// at full population: ~6,400 inserts plus a ~100 KB `Vec<Snap>`
/// alloc/drop on every tick before caching landed). Pre-grow happens
/// implicitly when the system extends past the current capacity.
#[derive(bevy_ecs::prelude::Resource, Default)]
pub struct AggroScratch {
    snapshot: Vec<Snap>,
    by_id: std::collections::HashMap<crate::components::NpcId, Snap>,
    /// `Entity → Snap` index. `try_acquire_pair` looks up the
    /// spotter's `Snap` (specifically the `group` field) on every
    /// successful acquisition; doing that via `snapshot.iter()
    /// .find(|s| s.entity == ...)` was a linear O(N) scan over up
    /// to 720 snapshot entries per find, and at dense pop it cost
    /// ~30 ms / tick alone. The map costs one HashMap insert per
    /// NPC at snapshot build time and one lookup per acquisition.
    by_entity: std::collections::HashMap<Entity, Snap>,
    sight_sq_by_id: std::collections::HashMap<crate::components::NpcId, f32>,
    current_target: std::collections::HashMap<Entity, crate::components::NpcId>,
    pending: Vec<(Entity, Aggro)>,
}

impl AggroScratch {
    fn clear(&mut self) {
        self.snapshot.clear();
        self.by_id.clear();
        self.by_entity.clear();
        self.sight_sq_by_id.clear();
        self.current_target.clear();
        self.pending.clear();
    }
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn npc_aggro(
    clock: Res<SimClock>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    deltas: Res<crate::faction::registry::RelationDeltas>,
    npcs: Query<(
        Entity,
        &Npc,
        &InFaction,
        &InRegion,
        &Position,
        &Rotation,
        Option<&Group>,
        Option<&NpcCharacter>,
    )>,
    mut commands: Commands,
    mut existing_aggro: Query<&mut Aggro>,
    mut log: ResMut<BehaviorLog>,
    config: Res<PerceptionConfig>,
    los: Res<LosService>,
    mut los_cache: ResMut<LosCache>,
    mut blackboards: ResMut<SquadBlackboards>,
    mut event_queue: ResMut<WorldEventQueue>,
    hash: Res<NpcSpatialHash>,
    active_regions: Res<crate::resources::ActiveRegions>,
    mut scratch: ResMut<AggroScratch>,
) {
    let _diag_t = crate::systems::SysTimer::new("npc_aggro");
    let now = clock.tick;
    let scratch = &mut *scratch;
    scratch.clear();

    // Snapshot filter: only build entries for NPCs that any pass
    // of this system might actually do work for. That's:
    //   - NPCs in active regions (Pass 2 acquires aggro for them).
    //   - NPCs with existing `Aggro` (Pass 1 refreshes/decays them
    //     regardless of region — aggro on offline targets decays
    //     correctly via the same code path).
    // At full population this drops the snapshot from ~3600 to
    // ~900 entries (1 active region of 4) and proportionally
    // reduces the by_id / sight_sq_by_id HashMap inserts. Per the
    // "every NPC acts as its own player" principle: offline-region
    // NPCs without aggro don't make perception decisions until
    // their region comes online.
    for (e, n, _f, r, p, _rot, g, character) in npcs.iter() {
        let in_active = active_regions.is_active(r.0);
        let has_aggro = existing_aggro.get(e).is_ok();
        if !in_active && !has_aggro {
            continue;
        }
        let perception = character.map(|c| c.stats.perception);
        let sight_radius = match perception {
            Some(p) => sight_radius_for_perception(p, config.sight_radius_m),
            None => config.sight_radius_m,
        };
        scratch.snapshot.push(Snap {
            entity: e,
            npc_id: n.id,
            region: r.0,
            pos: p.0,
            group: g.map(|g| g.id),
            sight_sq: sight_radius * sight_radius,
        });
    }

    // Build NpcId → Snap and NpcId → sight_sq indices for cheap
    // lookups in Pass 1 / Pass 2. Both reuse the scratch HashMaps
    // (cleared at the top of this tick); their bucket arrays grow
    // once to fit population and stay sized across ticks.
    for s in &scratch.snapshot {
        scratch.by_id.insert(s.npc_id, *s);
        scratch.by_entity.insert(s.entity, *s);
        scratch.sight_sq_by_id.insert(s.npc_id, s.sight_sq);
    }

    // Pass 1: decay or refresh existing aggro. Runs for every
    // region — cheap (O(aggroed NPCs), not pair-scan). Also builds a
    // map of entity → current target so Pass 2 can skip re-inserting
    // identical aggro.
    for s in &scratch.snapshot {
        let Ok(mut ag) = existing_aggro.get_mut(s.entity) else {
            continue;
        };
        // The observer's own perception-scaled sight radius gates
        // "do I still see my target". A stat 30 grunt loses sight
        // earlier than a stat 90 sniper would in the same scenario.
        match scratch.by_id.get(&ag.target) {
            Some(t) if t.region == s.region && within(s.pos, t.pos, s.sight_sq) => {
                ag.last_seen_tick = now;
                scratch.current_target.insert(s.entity, ag.target);
            }
            _ => {
                if now.saturating_sub(ag.last_seen_tick) > AGGRO_DECAY_TICKS() {
                    commands.entity(s.entity).remove::<Aggro>();
                } else {
                    scratch.current_target.insert(s.entity, ag.target);
                }
            }
        }
    }

    // Pass 2: new acquisitions, driven by the spatial hash. For
    // each region-grid, for each cell, pair-scan within-cell + 4
    // directional neighbor cells. The neighbor pattern covers all 8
    // directions while pairing each cell-pair exactly once.
    //
    // **Parallel with rayon** (2026-05-23): each cell's pair work is
    // independent — within-cell pairs touch only that cell's
    // entries; outgoing neighbor pairs from cell `C` are owned only
    // by `C`'s iteration (the reverse direction lives in the other
    // cell's iteration in the original sequential layout, but each
    // cell-pair is still visited exactly once). Side effects
    // (`los_cache.put`, `blackboards.write`, `event_queue.push`,
    // `pending.push`) accumulate into a per-cell `CellWork` and the
    // sequential merge below drains them in **sorted cell order** so
    // the "last write wins" determinism guarantee that pre-2026-05
    // tests rely on is preserved.
    //
    // Active-region filter: this is the dominant cost in npc_aggro
    // (within-cell `O(K²)` plus the 4-of-8 neighbor pair-scans).
    // Non-active regions have no observer — their NPCs don't need
    // online perception, and any aggro acquired there would be
    // invisible until a player enters. Skipping the outer region
    // loop drops aggro cost roughly proportionally to inactive-
    // region NPC count.
    // Pass 2 cadence gate (adaptive). The acquisition pair-scan is
    // the dense hot path; at 720 NPCs running it every tick costs
    // ~40-70 ms. Skipping it 2 out of every 3 ticks drops cost to
    // ~13-23 ms and is **invisible to gameplay**: AGGRO_DECAY_TICKS
    // = 200, so a target stays aggro'd ~10 s after last sight. A
    // 3-tick (150 ms) gap between acquisition opportunities is well
    // inside that window. Pass 1 (decay/refresh) still runs every
    // tick so already-aggroed NPCs keep their `last_seen_tick`
    // fresh while their target is in sight.
    //
    // The gate is skipped entirely at low NPC count so the per-tick
    // Pass 2 cost in tests / small scenarios is unchanged (the
    // pair scan at < `PASS_2_THRESHOLD` is microseconds anyway, and
    // tests that tick once + expect aggro to land continue to work
    // without changes).
    if scratch.snapshot.len() >= PASS_2_DENSE_THRESHOLD && !now.is_multiple_of(PASS_2_TICK_INTERVAL)
    {
        return;
    }

    let los_pass_through = los.provider.is_pass_through_on_current_thread();
    let mut regions_sorted: Vec<(&RegionId, &crate::resources::SpatialGrid)> =
        hash.by_region.iter().collect();
    regions_sorted.sort_by_key(|(rid, _)| **rid);
    // Build the flat, sorted list of cells to process. Each entry is
    // `(region_id, cell_xz, cell_indices, region_grid)`. The grid
    // ref carries the per-region entries vec for neighbor-cell
    // lookups inside the parallel closure.
    let mut cell_inputs: Vec<(
        RegionId,
        (i32, i32),
        &Vec<u32>,
        &crate::resources::SpatialGrid,
    )> = Vec::new();
    for (rid, grid) in &regions_sorted {
        if !active_regions.is_active(**rid) {
            continue;
        }
        for (cell, indices) in &grid.cells {
            cell_inputs.push((**rid, *cell, indices, *grid));
        }
    }
    cell_inputs.sort_by_key(|(rid, cell, _, _)| (*rid, cell.0, cell.1));

    let snapshot_ref: &[Snap] = &scratch.snapshot;
    let by_entity_ref = &scratch.by_entity;
    let sight_sq_ref = &scratch.sight_sq_by_id;
    let current_target_ref = &scratch.current_target;
    let los_ref = &*los;
    let config_ref = &*config;
    let blackboards_ref: &SquadBlackboards = &blackboards;
    // Pre-compute the F×F hostility matrix once per tick so the
    // pair scan never calls `faction_relation` — which routes
    // through `canonical_pair_names`, allocating two `String`s on
    // every call to hash into `RelationDeltas.by_pair`. At ~9 k
    // pairs that's ~18 k String allocations per tick. The matrix
    // is `O(F²)` to build (≤ 16 cells), `O(1)` to query.
    let n_factions = registry.count();
    let mut hostility: Vec<bool> = vec![false; n_factions * n_factions];
    for fa in 0..n_factions {
        for fb in 0..n_factions {
            #[allow(clippy::cast_possible_truncation)]
            let rel = crate::faction::registry::faction_relation(
                &registry,
                &deltas,
                crate::faction::registry::FactionId(fa as u32),
                crate::faction::registry::FactionId(fb as u32),
            );
            hostility[fa * n_factions + fb] = rel == Relation::Hostile;
        }
    }
    let hostility_ref: &[bool] = &hostility;

    use rayon::prelude::*;
    let cell_outputs: Vec<CellWork> = cell_inputs
        .par_iter()
        .map(|(_rid, cell, indices, grid)| {
            let mut out = CellWork::default();
            // Within-cell pairs.
            for ii in 0..indices.len() {
                for jj in (ii + 1)..indices.len() {
                    let a = grid.entries[indices[ii] as usize];
                    let b = grid.entries[indices[jj] as usize];
                    try_acquire_pair(
                        a,
                        b,
                        now,
                        sight_sq_ref,
                        config_ref.sight_radius_m,
                        config_ref,
                        los_ref,
                        los_pass_through,
                        current_target_ref,
                        snapshot_ref,
                        by_entity_ref,
                        blackboards_ref,
                        &mut out,
                        hostility_ref,
                        n_factions,
                    );
                }
            }
            // Neighbor-cell pairs. Only 4 of the 8 directions so
            // each cell pair is visited exactly once across the
            // (sorted) outer loop.
            for (dx, dz) in NEIGHBOR_OFFSETS {
                let ncell = (cell.0 + dx, cell.1 + dz);
                let Some(nindices) = grid.cells.get(&ncell) else {
                    continue;
                };
                for &ai in indices.iter() {
                    for &bi in nindices {
                        let a = grid.entries[ai as usize];
                        let b = grid.entries[bi as usize];
                        try_acquire_pair(
                            a,
                            b,
                            now,
                            sight_sq_ref,
                            config_ref.sight_radius_m,
                            config_ref,
                            los_ref,
                            los_pass_through,
                            current_target_ref,
                            snapshot_ref,
                            by_entity_ref,
                            blackboards_ref,
                            &mut out,
                            hostility_ref,
                            n_factions,
                        );
                    }
                }
            }
            out
        })
        .collect();

    // Sequential merge — drain in `cell_inputs` order (already
    // sorted by `(region_id, cell.x, cell.z)`) so the side-effect
    // ordering matches the pre-parallel implementation. Determinism
    // tests in `simn-sim/tests/determinism.rs` flag a regression if
    // this ordering ever stops matching.
    for work in cell_outputs {
        for (a_id, b_id, exp, when) in work.los_inserts {
            los_cache.put(a_id, b_id, exp, when);
        }
        for op in work.bb_writes {
            write_aggro_to_blackboard(
                &mut blackboards,
                Some(op.group_id),
                op.target_id,
                op.target_pos,
                op.now,
            );
        }
        for op in work.events {
            event_queue.push(
                WorldEventKind::EnemySighted {
                    target_id: op.target_id,
                    target_faction: op.target_faction,
                },
                op.target_pos,
                op.region,
                op.now,
                EVENT_TTL_TICKS,
            );
        }
        for (e, ag) in work.pending {
            scratch.pending.push((e, ag));
        }
    }
    let count = scratch.pending.len() as u32;
    for (e, ag) in scratch.pending.drain(..) {
        commands.entity(e).insert(ag);
    }
    if log.enabled && count > 0 {
        log.aggro_acquisitions = log.aggro_acquisitions.saturating_add(count);
    }
}

/// Per-cell side-effect buffer used by the parallel pair-scan. Each
/// rayon worker fills its own `CellWork`; the sequential merge below
/// drains them in deterministic cell order.
#[derive(Default)]
struct CellWork {
    /// `los_cache.put(a, b, exposure, now)` ops to apply.
    los_inserts: Vec<(crate::components::NpcId, crate::components::NpcId, f32, u64)>,
    pending: Vec<(Entity, Aggro)>,
    bb_writes: Vec<BBWriteOp>,
    events: Vec<EventOp>,
}

struct BBWriteOp {
    group_id: u64,
    target_id: crate::components::NpcId,
    target_pos: [f32; 3],
    now: u64,
}

struct EventOp {
    target_id: crate::components::NpcId,
    target_faction: crate::faction::registry::FactionId,
    target_pos: [f32; 3],
    region: RegionId,
    now: u64,
}

/// 4-of-8 directional neighbors for the cell pair scan. Pairing each
/// cell with `(+1, 0)`, `(+1, +1)`, `(0, +1)`, `(-1, +1)` reaches
/// every possible pair of adjacent cells exactly once across the
/// grid iteration (the reverse directions are covered by being the
/// "other" cell in some other iteration).
const NEIGHBOR_OFFSETS: [(i32, i32); 4] = [(1, 0), (1, 1), (0, 1), (-1, 1)];

#[allow(clippy::too_many_arguments)]
fn try_acquire_pair(
    a: SpatialEntry,
    b: SpatialEntry,
    now: u64,
    sight_sq_by_id: &std::collections::HashMap<crate::components::NpcId, f32>,
    base_sight_radius_m: f32,
    config: &PerceptionConfig,
    los: &LosService,
    los_pass_through: bool,
    current_target: &std::collections::HashMap<Entity, crate::components::NpcId>,
    snapshot: &[Snap],
    by_entity: &std::collections::HashMap<Entity, Snap>,
    blackboards: &SquadBlackboards,
    work: &mut CellWork,
    hostility: &[bool],
    n_factions: usize,
) {
    let base_sq = base_sight_radius_m * base_sight_radius_m;
    let a_sight_sq = sight_sq_by_id.get(&a.npc_id).copied().unwrap_or(base_sq);
    let b_sight_sq = sight_sq_by_id.get(&b.npc_id).copied().unwrap_or(base_sq);
    // Pre-cull: if neither NPC could see the other at this distance,
    // skip everything else. Use the larger of the two ranges so we
    // don't drop pairs where the asymmetric check would still fire.
    let max_sight_sq = a_sight_sq.max(b_sight_sq);
    if !within(a.pos, b.pos, max_sight_sq) {
        return;
    }
    let ai = a.faction.0 as usize;
    let bi = b.faction.0 as usize;
    if ai >= n_factions || bi >= n_factions || !hostility[ai * n_factions + bi] {
        return;
    }
    // Per-direction sight gates: each NPC's own perception-scaled
    // range applies to whether THEY can see the other. Combined with
    // the FOV cone check below, this lets a high-perception sniper
    // spot a low-perception grunt at a distance the inverse pair
    // can't reach.
    let a_in_range = within(a.pos, b.pos, a_sight_sq);
    let b_in_range = within(b.pos, a.pos, b_sight_sq);
    let a_sees_b = a_in_range && in_fov(a.pos, a.yaw, b.pos, config.fov_deg);
    let b_sees_a = b_in_range && in_fov(b.pos, b.yaw, a.pos, config.fov_deg);
    if !a_sees_b && !b_sees_a {
        return;
    }
    // LOS. Sample per direction that needs it. Early-out on the
    // cheap default provider (AlwaysVisibleLos ≡ 1.0) AND on the
    // `los_pass_through` hint (e.g. GodotLosProvider when called
    // from a rayon worker — short-circuits to 1.0 anyway, so
    // skipping the call avoids the cost of acquiring the provider's
    // internal mutex across N parallel workers).
    let (a_exposed, b_exposed) = if config.los_enabled && !los_pass_through {
        let eye_a = raise(a.pos, config.eye_height_m);
        let eye_b = raise(b.pos, config.eye_height_m);
        let ex_ab = if a_sees_b {
            los.provider.exposure(eye_a, b.pos, a.region, config)
        } else {
            0.0
        };
        let ex_ba = if b_sees_a {
            los.provider.exposure(eye_b, a.pos, b.region, config)
        } else {
            0.0
        };
        (ex_ab, ex_ba)
    } else {
        (
            if a_sees_b { 1.0 } else { 0.0 },
            if b_sees_a { 1.0 } else { 0.0 },
        )
    };
    // Stash both directions in the per-tick LOS cache so downstream
    // consumers (cover queries, future combat resolution) can read
    // the same numbers without re-raycasting. Only writes pairs that
    // actually passed the FOV gate; the `else 0.0` branches above
    // mean a value of zero may also represent "wasn't asked", so
    // skip writing those. Cache writes still fire under
    // `los_pass_through` (where exposure was hard-coded to 1.0) —
    // `npc_combat`'s LOS gate keys off cache presence and would
    // refuse to fire on a hostile if the cache stays empty, so the
    // parallel fast-path must still populate it.
    if a_sees_b {
        work.los_inserts.push((a.npc_id, b.npc_id, a_exposed, now));
    }
    if b_sees_a {
        work.los_inserts.push((b.npc_id, a.npc_id, b_exposed, now));
    }
    let a_ok = a_exposed >= config.exposure_required;
    let b_ok = b_exposed >= config.exposure_required;
    if !a_ok && !b_ok {
        return;
    }
    let a_has = current_target.get(&a.entity) == Some(&b.npc_id);
    let b_has = current_target.get(&b.entity) == Some(&a.npc_id);
    if a_ok && !a_has {
        // Look up the spotter's full Snap (with group) for squad-share.
        if let Some(spotter) = by_entity.get(&a.entity).copied() {
            queue_aggro(
                &mut work.pending,
                snapshot,
                spotter,
                b.npc_id,
                now,
                current_target,
            );
            // Per-squad event throttle. The bus event broadcasts to
            // ~all hostile-faction groups within 200 m; doing that
            // every time a squadmate's FOV cone catches a new target
            // is wasted work because consumers only need to know
            // "your squad has spotted enemies recently", not the
            // play-by-play of which specific target each frame. If
            // the spotter's squad blackboard already has a fresh
            // `LastKnownEnemyId` (any target), we skip the bus emit
            // entirely. The direct `bb_write` below still fires so
            // the spotter's own squad sees the new target.
            //
            // Loss: hostile-faction squads outside the spotter
            // squad's blackboard reach get fewer notifications, but
            // their own perception still picks up enemies through
            // the same `npc_aggro` pair-scan — alerts arrive on
            // their own cadence.
            let already_engaged = spotter.group.is_some_and(|gid| {
                blackboards
                    .get(gid)
                    .and_then(|g| g.get(&BlackboardKey::LastKnownEnemyId))
                    .is_some_and(|e| e.is_fresh(now))
            });
            if let Some(gid) = spotter.group {
                work.bb_writes.push(BBWriteOp {
                    group_id: gid,
                    target_id: b.npc_id,
                    target_pos: b.pos,
                    now,
                });
            }
            if !already_engaged {
                work.events.push(EventOp {
                    target_id: b.npc_id,
                    target_faction: b.faction,
                    target_pos: b.pos,
                    region: a.region,
                    now,
                });
            }
        }
    }
    if b_ok && !b_has {
        if let Some(spotter) = by_entity.get(&b.entity).copied() {
            queue_aggro(
                &mut work.pending,
                snapshot,
                spotter,
                a.npc_id,
                now,
                current_target,
            );
            let already_engaged = spotter.group.is_some_and(|gid| {
                blackboards
                    .get(gid)
                    .and_then(|g| g.get(&BlackboardKey::LastKnownEnemyId))
                    .is_some_and(|e| e.is_fresh(now))
            });
            if let Some(gid) = spotter.group {
                work.bb_writes.push(BBWriteOp {
                    group_id: gid,
                    target_id: a.npc_id,
                    target_pos: a.pos,
                    now,
                });
            }
            if !already_engaged {
                work.events.push(EventOp {
                    target_id: a.npc_id,
                    target_faction: a.faction,
                    target_pos: a.pos,
                    region: b.region,
                    now,
                });
            }
        }
    }
}

/// Default TTL on bus events emitted from aggro acquisition. 1 tick
/// is enough — the drain runs the next tick, applies the event into
/// listening blackboards (which carry their own TTL), then the bus
/// drops the event.
const EVENT_TTL_TICKS: u32 = 2;

/// Stash the new aggro target's id + position on the spotter's
/// squad blackboard so squadmates (and the future goal arbitrator)
/// can react without their own perception sample. No-op for ungrouped
/// spotters.
fn write_aggro_to_blackboard(
    blackboards: &mut SquadBlackboards,
    spotter_group: Option<u64>,
    target_id: crate::components::NpcId,
    target_pos: [f32; 3],
    now: u64,
) {
    let Some(group_id) = spotter_group else {
        return;
    };
    blackboards.write(
        group_id,
        BlackboardKey::LastKnownEnemyId,
        BlackboardValue::NpcRef(target_id),
        now,
        AGGRO_BB_TTL_TICKS(),
    );
    blackboards.write(
        group_id,
        BlackboardKey::LastKnownEnemyPos,
        BlackboardValue::Position(target_pos),
        now,
        AGGRO_BB_TTL_TICKS(),
    );
}

fn within(a: [f32; 3], b: [f32; 3], sq: f32) -> bool {
    let dx = a[0] - b[0];
    let dz = a[2] - b[2];
    dx * dx + dz * dz <= sq
}

fn raise(p: [f32; 3], dy: f32) -> [f32; 3] {
    [p[0], p[1] + dy, p[2]]
}

fn queue_aggro(
    pending: &mut Vec<(Entity, Aggro)>,
    snapshot: &[Snap],
    spotter: Snap,
    target: crate::components::NpcId,
    now: u64,
    current: &std::collections::HashMap<Entity, crate::components::NpcId>,
) {
    let ag = Aggro {
        target,
        last_seen_tick: now,
    };
    if current.get(&spotter.entity) != Some(&target) {
        pending.push((spotter.entity, ag));
    }
    // Squad-share: spotter's squadmates adopt the same target, but
    // only if they don't already have an aggro of their own. This
    // stops a roving engagement from churning every squadmate's
    // target every tick as different opponents are spotted in
    // sequence. Members keep their first real target until decay.
    if let Some(group_id) = spotter.group {
        for s in snapshot {
            if s.entity == spotter.entity || s.group != Some(group_id) {
                continue;
            }
            if current.contains_key(&s.entity) {
                continue;
            }
            pending.push((s.entity, ag));
        }
    }
}
