//! Offline-tier parallel schema (Phase 1B of
//! `docs/book/src/planning/sim-iteration-5-12-plan.md`).
//!
//! When a region has no observers, its NPCs should *simulate* —
//! moving along the waypoint graph, resolving combat via dice,
//! shifting territory — rather than just *freeze*. This module
//! defines the lightweight schema that holds that state and the
//! slow clock that drives it. The actual movement / combat / event-
//! emission systems land in Phase 1D, 1E, 1F. For now this is
//! plumbing: the parallel component, the resource, the schedule slot.
//!
//! Design source: `docs/book/src/planning/offline-tier-plan.md` §3.
//!
//! Why parallel components instead of stripped versions of the
//! online ones? An offline NPC has no `BodyParts`, no `Inventory`,
//! no `ActiveGoal`, no path. Reusing those types would mean keeping
//! them allocated and zero-initialised for thousands of inactive
//! NPCs; the savings only land if the abstract tier has its own
//! storage. Projection between tiers (Phase 1C) translates one
//! schema to the other on the region-activation boundary.

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use crate::components::{NpcId, NpcStats};
use crate::faction::registry::FactionId;
use crate::region::RegionId;
use crate::resources::SimClock;

// Note on serde: `FactionId` is intentionally not `Serialize`/`Deserialize`
// (interned id, not stable across registry edits — snapshots use the
// faction name string instead, like `SerializedEntity::in_faction`).
// Phase 1C will add a parallel `SerializedOfflineNpc` shape that
// converts `FactionId` ↔ name string at the snapshot boundary, mirroring
// the pattern in `crate::persistence::snapshot::SerializedEntity`. For
// Phase 1B the in-ECS component just doesn't derive serde — none of the
// scheduled offline-tier systems touch persistence yet.

/// Coarse-grain health for an offline-tier NPC. Replaces the
/// per-limb `BodyParts` of the online tier with three buckets the
/// dice combat (Phase 1E) can branch on.
///
/// - `Healthy` → all body parts > 75% in online; full combat
///   effectiveness offline.
/// - `Wounded` → any limb 25-75%; reduced accuracy + slower
///   movement offline.
/// - `Critical` → multi-part damage or any vital part < 25%;
///   active bleed in projection back to online tier.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HealthClass {
    Healthy,
    Wounded,
    Critical,
}

// `LoadoutClass` and `OfflineNpc` carry `FactionId`, which is not
// `Serialize`/`Deserialize` by design — see the module-level note.
// The derives are intentionally absent until Phase 1C wires up the
// name-string snapshot path.

/// Coarse-grain equipment for an offline-tier NPC. Replaces the
/// `Inventory` grid with a faction + tier bucket that projection
/// (Phase 1C) re-rolls into specific items when the region goes
/// online.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoadoutClass {
    /// Standard faction kit at the given gear tier (1 = newbie,
    /// 5 = veteran). Specific weapons/ammo re-rolled from faction
    /// loadout tables when the NPC re-enters the online tier.
    Standard { faction: FactionId, tier: u8 },
    /// Faction elite kit (top-tier weapons + armor for the faction).
    Elite { faction: FactionId },
    /// Mixed kit — wanderers, looters, anyone who took what was on
    /// the ground. No faction-coherent loadout table; projection
    /// rolls from a "scrap" pool.
    Improvised,
}

/// Offline-tier combat state. Mirrors the abstract "are you fighting
/// right now" flag without tracking the projectile / aggro pair
/// state the online tier carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OfflineCombatState {
    Idle,
    Engaged {
        opponent: NpcId,
        since_tick: u64,
    },
    /// Fleeing until the given offline-tier tick. After that we
    /// re-evaluate: idle if no enemies in range, re-engage
    /// otherwise.
    Routed {
        until_offline_tick: u64,
    },
}

/// Lightweight per-NPC state for regions with no observers. Replaces
/// `Npc + InFaction + InRegion + Position + BodyParts + Inventory +
/// ActiveGoal + Aggression + NpcGoal + …` (~12 components) with one.
///
/// Projection (Phase 1C) translates between this and the online
/// schema at region-activation boundaries. Chronicle (`LifeChronicle`)
/// is shared and lives on whichever side the NPC is currently on.
///
/// `personality_seed` keeps identity stable across tier transitions
/// — same NPC, same name, same personality, same stats whether they
/// just left the online tier or just entered it. Mirrors the
/// existing online-tier `NpcCharacter` deterministic-roll pattern.
#[derive(Component, Clone, Debug)]
pub struct OfflineNpc {
    pub id: NpcId,
    pub region: RegionId,
    /// World XZ position in meters. Y is stripped — the heightmap
    /// re-resolves it on projection back to online.
    pub position_2d: [f32; 2],
    pub faction: FactionId,
    pub group: Option<u64>,
    pub health_class: HealthClass,
    pub loadout_class: LoadoutClass,
    pub personality_seed: u64,
    pub stats: NpcStats,
    pub combat_state: OfflineCombatState,
    /// Tick the NPC dies of old age. Carries over from the online
    /// `Lifespan` component so age-out still happens in offline
    /// regions (Phase 1E's offline event emit pushes a chronicle
    /// entry).
    pub die_at_tick: u64,
    /// Phase 1D movement target. `None` = currently idle (next
    /// offline tick will pick a new target). XZ world meters.
    pub target_2d: Option<[f32; 2]>,
    /// Offline-tier tick at which `target_2d` is reached. Position
    /// interpolates linearly between `(travel_start_pos,
    /// travel_start_offline_tick)` and `(target_2d,
    /// arrival_offline_tick)` while in transit.
    pub arrival_offline_tick: u64,
    /// Offline-tier tick when the current leg started — anchors the
    /// linear interpolation.
    pub travel_start_offline_tick: u64,
    /// Position when the current leg started.
    pub travel_start_2d: [f32; 2],
    /// Iteration 5-13 Phase C2: optional chain of waypoint-graph
    /// node indices resolved from the per-region `WaypointGraph`.
    /// When non-empty, the offline movement system advances along
    /// the chain segment by segment instead of bee-lining straight
    /// to `target_2d`. Empty for legacy NPCs (no graph available
    /// at attach time) and after the chain is exhausted —
    /// movement falls back to the existing bee-line path so the
    /// system stays robust against missing data.
    pub waypoint_chain: Vec<u32>,
    /// Index of the next chain node to walk toward. `0` means the
    /// NPC is on its way to `waypoint_chain[0]`.
    pub waypoint_chain_idx: u32,
    /// Aggro target id, preserved across the tier transition so an
    /// NPC mid-firefight doesn't lose its target the moment its
    /// region goes offline. Restored as an `Aggro` component when
    /// the region comes back online. `None` if the NPC wasn't
    /// aggroed at projection time. Decays via the offline-tier's
    /// own combat logic — see `offline_combat`.
    pub aggro_target: Option<NpcId>,
    /// Tick of the most recent sight of `aggro_target`. Mirrors the
    /// online `Aggro::last_seen_tick` so decay timing carries over
    /// the tier boundary — an NPC that just lost sight of its
    /// target right before the region went offline doesn't get a
    /// fresh 200-tick lease when the region re-activates.
    pub aggro_last_seen_tick: u64,
}

/// Slow clock for the offline tier. Advances once every
/// `OFFLINE_TIER_TICK_DIVISOR` sim ticks (default 10 → 2 Hz at the
/// stock 20 Hz sim rate). The slower cadence is a conscious cost
/// cut and matches the abstract grain of the simulation: a squad
/// hopping waypoints in 30 s of real time has 60 offline ticks
/// to make ~6 decisions, which is the right resolution.
#[derive(Resource, Serialize, Deserialize, Clone, Copy, Debug, Default)]
pub struct OfflineTierClock {
    pub tick: u64,
    /// Sim tick at which the offline tier last advanced. Used by
    /// the heartbeat to detect divisor crossings; an explicit field
    /// (rather than `sim_tick % 10 == 0`) keeps the cadence robust
    /// across pause/resume and `SimClock` reset corner cases.
    pub last_sim_tick: u64,
}

/// How many sim ticks elapse between offline-tier ticks. 10 sim
/// ticks at 20 Hz = 0.5 s of offline-tier wall time, i.e. 2 Hz.
/// This is the *clock* cadence (`tick_offline_clock` advances at
/// this rate). The actual offline-tier work systems run at the
/// (lower) `OFFLINE_PROCESS_INTERVAL_TICKS` rate.
pub const OFFLINE_TIER_TICK_DIVISOR: u64 = 10;

/// How many sim ticks elapse between offline-tier work bursts
/// (movement / combat / base-dominance). 4 ticks at 20 Hz = 5 Hz
/// total, so with N regions in round-robin each region updates at
/// `5 / N` Hz. At the design target of ~4 procedurally-seeded
/// regions per map that's ~1.25 Hz per-region — slow but enough
/// for "abstract dice simulation" granularity, which is the
/// design intent. Lowering this is free perf; raising it is the
/// trade-off if combat needs to resolve faster.
pub const OFFLINE_PROCESS_INTERVAL_TICKS: u64 = 4;

/// Heartbeat system: advances `OfflineTierClock` once every
/// `OFFLINE_TIER_TICK_DIVISOR` sim ticks. Runs every sim tick (cheap
/// — just an integer compare) and gates the slow tier on its own
/// counter. Phase 1D's `offline_movement`, Phase 1E's
/// `offline_combat`, and Phase 1F's `offline_event_emit` will all
/// run *after* this in the schedule and read
/// `offline_clock.tick > prev_tick` to decide whether to do work
/// this sim tick. Letting each consumer make that decision keeps
/// them composable without a dedicated system-set gate.
pub fn tick_offline_clock(sim_clock: Res<SimClock>, mut offline_clock: ResMut<OfflineTierClock>) {
    let _diag_t = crate::systems::SysTimer::new("tick_offline_clock");
    // Bump once per divisor crossing. Using "current - last >= divisor"
    // rather than "% == 0" so a tick gap (pause / restart) doesn't
    // miss a beat.
    if sim_clock.tick.wrapping_sub(offline_clock.last_sim_tick) >= OFFLINE_TIER_TICK_DIVISOR
        || (offline_clock.last_sim_tick == 0 && sim_clock.tick >= OFFLINE_TIER_TICK_DIVISOR)
    {
        offline_clock.tick = offline_clock.tick.wrapping_add(1);
        offline_clock.last_sim_tick = sim_clock.tick;
    }
}

/// True if the offline clock advanced this sim tick. Phase 1D-1F
/// consumers read this to decide whether to do their per-offline-
/// tick work, rather than each computing `% DIVISOR` independently.
pub fn offline_tick_just_advanced(sim_clock: &SimClock, offline_clock: &OfflineTierClock) -> bool {
    sim_clock.tick == offline_clock.last_sim_tick && offline_clock.tick > 0
}

// --- Projection (Phase 1C) ---------------------------------------------------
//
// The invariant: an NPC is an online entity iff its region is in
// `ActiveRegions`. Crossing the boundary projects state one way and
// despawns the source side — there's never a window where both
// schemas hold the same NPC.
//
// Online → offline collapses ~12 components into one `OfflineNpc`
// plus a faction-keyed `LoadoutClass` summary. The inventory is
// destroyed (per `offline-tier-plan.md` §5): when the NPC re-
// projects to online, fresh items roll from faction loadout tables.
// Body-part fidelity collapses to a `HealthClass` enum.
//
// Offline → online does the inverse: re-roll an inventory from the
// faction's `NpcLoadoutRegistry`, materialize `BodyParts` matching
// `HealthClass`, re-derive `NpcCharacter` from `(npc_id, faction_id)`
// (deterministic, no state needs to be preserved across the
// transition). Squad cohesion survives via the `group` field.

use crate::components::{
    ActiveEffects, ActiveGoal, Actor, ActorKind, Aggression, BodyParts, Group, Health, InFaction,
    InRegion, Inventory, Lifespan, LimbStates, Npc, NpcCharacter, NpcGoal, Position,
    RecentAttackers, Rotation, Wounds,
};

/// Derive a coarse `HealthClass` from the per-limb `BodyParts` state.
///
/// Thresholds match `offline-tier-plan.md` §5:
/// - Head/torso < 25 → `Critical` (vital damage)
/// - Any limb < 25 → `Critical` (multi-part interpretation)
/// - Any part 25-75 → `Wounded`
/// - All parts ≥ 75 → `Healthy`
///
/// `BodyParts::DEFAULT_MAX` is 100, so 25/75 are direct cutoffs.
pub fn body_parts_to_health_class(bp: &BodyParts) -> HealthClass {
    let critical_threshold = BodyParts::DEFAULT_MAX * 0.25;
    let wounded_threshold = BodyParts::DEFAULT_MAX * 0.75;
    if bp.head < critical_threshold || bp.torso < critical_threshold {
        return HealthClass::Critical;
    }
    let min_limb = bp
        .left_arm
        .min(bp.right_arm)
        .min(bp.left_leg)
        .min(bp.right_leg);
    if min_limb < critical_threshold {
        return HealthClass::Critical;
    }
    let min_all = bp.head.min(bp.torso).min(min_limb);
    if min_all < wounded_threshold {
        HealthClass::Wounded
    } else {
        HealthClass::Healthy
    }
}

/// Materialize a plausible `BodyParts` distribution from a coarse
/// `HealthClass`. Deterministic per `(npc_id, class)` so re-
/// projecting the same NPC twice produces the same wound layout.
fn health_class_to_body_parts(class: HealthClass, npc_id: crate::components::NpcId) -> BodyParts {
    use rand::{Rng, SeedableRng};
    let mut bp = BodyParts::new_full();
    if matches!(class, HealthClass::Healthy) {
        return bp;
    }
    // Per-class salt so Wounded and Critical pick different RNG
    // streams for the same NpcId.
    let salt: u64 = match class {
        HealthClass::Wounded => 0x57A1_FE03_BA53_57A1,
        HealthClass::Critical => 0x77A1_FE03_C817_1CA1,
        HealthClass::Healthy => 0,
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(npc_id.0.wrapping_add(salt));
    let max = BodyParts::DEFAULT_MAX;
    match class {
        HealthClass::Wounded => {
            // One limb at 30-60% — body still functional, just hurt.
            let limb = rng.gen_range(0..4);
            let damaged = max * rng.gen_range(0.30..0.60);
            match limb {
                0 => bp.left_arm = damaged,
                1 => bp.right_arm = damaged,
                2 => bp.left_leg = damaged,
                _ => bp.right_leg = damaged,
            }
        }
        HealthClass::Critical => {
            // Vital + a limb. The vital draws low enough to satisfy
            // the health-class threshold (`< 25%`).
            let vital_damaged = max * rng.gen_range(0.10..0.20);
            if rng.gen_bool(0.5) {
                bp.torso = vital_damaged;
            } else {
                bp.head = vital_damaged;
            }
            let limb_damaged = max * rng.gen_range(0.10..0.40);
            let limb = rng.gen_range(0..4);
            match limb {
                0 => bp.left_arm = limb_damaged,
                1 => bp.right_arm = limb_damaged,
                2 => bp.left_leg = limb_damaged,
                _ => bp.right_leg = limb_damaged,
            }
        }
        HealthClass::Healthy => unreachable!(),
    }
    bp
}

/// Pick a `LoadoutClass` for an online NPC about to go offline.
/// Coarse faction-name lookup for Phase 1C; Phase 1E may refine to
/// consult actual equipped items.
fn faction_to_loadout_class(
    faction: crate::faction::registry::FactionId,
    registry: &crate::faction::registry::FactionRegistry,
) -> LoadoutClass {
    let name = registry.name_of(faction);
    match name {
        "wanderers" | "bandits" => LoadoutClass::Improvised,
        "ghost_teams" | "recovery_division" | "choir" | "registry" => {
            LoadoutClass::Elite { faction }
        }
        _ => LoadoutClass::Standard { faction, tier: 1 },
    }
}

/// Project every online NPC in `region` down to an `OfflineNpc`.
/// Used by `Sim::set_active_region` when a region transitions from
/// active to inactive. Inventory + per-limb body-part state are
/// dropped; they re-materialize when the region next goes online.
pub fn project_online_to_offline(world: &mut World, region: RegionId) {
    let _diag_t = crate::systems::SysTimer::new("project_online_to_offline");

    let mut targets: Vec<bevy_ecs::entity::Entity> = Vec::new();
    {
        let mut query = world.query::<(bevy_ecs::entity::Entity, &Npc, &InRegion)>();
        for (entity, _npc, r) in query.iter(world) {
            if r.0 == region {
                targets.push(entity);
            }
        }
    }

    let registry = world
        .resource::<crate::faction::registry::FactionRegistry>()
        .clone();

    let mut to_spawn: Vec<OfflineNpc> = Vec::with_capacity(targets.len());
    for entity in &targets {
        let Some(npc) = world.get::<Npc>(*entity).copied() else {
            continue;
        };
        let Some(faction) = world.get::<InFaction>(*entity).copied() else {
            continue;
        };
        let Some(pos) = world.get::<Position>(*entity).copied() else {
            continue;
        };
        let body = world.get::<BodyParts>(*entity).copied();
        let group = world.get::<Group>(*entity).copied();
        let character = world.get::<NpcCharacter>(*entity).cloned();
        let lifespan = world.get::<Lifespan>(*entity).copied();
        let aggro = world.get::<crate::components::Aggro>(*entity).copied();

        let health_class = body
            .map(|bp| body_parts_to_health_class(&bp))
            .unwrap_or(HealthClass::Healthy);
        let loadout_class = faction_to_loadout_class(faction.0, &registry);
        let stats = character.as_ref().map(|c| c.stats).unwrap_or_else(|| {
            // No character component? Roll a stub deterministic from
            // id. Covers tests that spawn bare NPCs via
            // `spawn_npc_for_test`; production NPCs always have one.
            use rand::SeedableRng;
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(npc.id.0);
            crate::components::NpcStats::roll(&mut rng, 0.5)
        });
        let personality_seed = character
            .as_ref()
            .map(|c| c.character_id.0)
            .unwrap_or_else(|| NpcCharacter::derive_id(npc.id, faction.0).0);

        let pos_2d = [pos.0[0], pos.0[2]];
        to_spawn.push(OfflineNpc {
            id: npc.id,
            region,
            position_2d: pos_2d,
            faction: faction.0,
            group: group.map(|g| g.id),
            health_class,
            loadout_class,
            personality_seed,
            stats,
            // Carry an in-progress engagement across the boundary
            // so `offline_movement` doesn't immediately route the
            // NPC to a random base and break the firefight. Cleared
            // by `offline_combat` once the opponent dies or aggro
            // ages out (see `OFFLINE_ENGAGE_STALE_TICKS`).
            combat_state: match aggro {
                Some(a) => OfflineCombatState::Engaged {
                    opponent: a.target,
                    since_tick: a.last_seen_tick,
                },
                None => OfflineCombatState::Idle,
            },
            die_at_tick: lifespan.map(|l| l.die_at_tick).unwrap_or(u64::MAX),
            // Idle by default — `offline_movement` will pick a target
            // on the next offline tick.
            target_2d: None,
            arrival_offline_tick: 0,
            travel_start_offline_tick: 0,
            travel_start_2d: pos_2d,
            waypoint_chain: Vec::new(),
            waypoint_chain_idx: 0,
            aggro_target: aggro.map(|a| a.target),
            aggro_last_seen_tick: aggro.map(|a| a.last_seen_tick).unwrap_or(0),
        });
    }

    for entity in targets {
        world.despawn(entity);
    }
    for offline in to_spawn {
        world.spawn(offline);
    }
}

/// Project every offline NPC in `region` back up to the full online
/// schema. Used by `Sim::set_active_region` when a region transitions
/// from inactive to active. Body parts re-materialize from the
/// `HealthClass` enum; inventory rolls fresh from the faction's
/// loadout tables; `NpcCharacter` re-derives from `(npc_id,
/// faction_id)` deterministically.
pub fn project_offline_to_online(world: &mut World, region: RegionId) {
    let _diag_t = crate::systems::SysTimer::new("project_offline_to_online");

    let mut targets: Vec<bevy_ecs::entity::Entity> = Vec::new();
    let mut offline_state: Vec<OfflineNpc> = Vec::new();
    {
        let mut query = world.query::<(bevy_ecs::entity::Entity, &OfflineNpc)>();
        for (entity, offline) in query.iter(world) {
            if offline.region == region {
                targets.push(entity);
                offline_state.push(offline.clone());
            }
        }
    }

    let registry = world
        .resource::<crate::faction::registry::FactionRegistry>()
        .clone();
    let items = world.resource::<crate::items::ItemRegistry>().clone();
    let loadouts = world
        .resource::<crate::npc_loadouts::NpcLoadoutRegistry>()
        .clone();
    let names = world.resource::<crate::names::NameRegistry>().clone();

    for entity in targets {
        world.despawn(entity);
    }

    for offline in offline_state {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(offline.personality_seed);
        let faction_name = registry.name_of(offline.faction).to_string();
        let inv = loadouts.build_inventory(&faction_name, &items, &mut rng);

        let body = health_class_to_body_parts(offline.health_class, offline.id);
        let def = registry.def(offline.faction);
        let agg_base = def.base_aggression;
        let archetype = def.archetype;
        let nat_weights = def.nationality_weights.clone();
        let male_w = def.male_name_weight;
        let character = NpcCharacter::roll(
            offline.id,
            offline.faction,
            archetype,
            agg_base,
            &names,
            &nat_weights,
            male_w,
        );

        // Per-NPC spawn jitter (±15 m XZ, deterministic from
        // npc_id). Without this, every offline NPC that was
        // "between hops" (target_2d = None at projection time)
        // projects back at the exact same waypoint position — so
        // a full squad lands on one point and the player sees
        // a giant clump until peer-separation + spawn-disperse
        // pull them apart. The jitter pre-spreads them so they
        // start out in their formation slots, not stacked.
        use rand::Rng;
        let mut jitter_rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(
            offline.id.0.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        let jitter_x = jitter_rng.gen_range(-15.0..15.0);
        let jitter_z = jitter_rng.gen_range(-15.0..15.0);
        let spawn_pos = [
            offline.position_2d[0] + jitter_x,
            0.0,
            offline.position_2d[1] + jitter_z,
        ];
        let bundle = (
            Npc { id: offline.id },
            Actor {
                kind: ActorKind::Npc,
            },
            InFaction(offline.faction),
            InRegion(offline.region),
            Position(spawn_pos),
            Rotation(0.0),
            Health::new_full(),
            body,
            LimbStates::default(),
            Wounds::default(),
            ActiveEffects::default(),
            NpcGoal::Idle { until_tick: 0 },
            Lifespan {
                spawned_tick: 0,
                die_at_tick: offline.die_at_tick,
            },
            Aggression(agg_base),
            RecentAttackers::default(),
        );
        let active_goal = ActiveGoal::default();
        let entity = if let Some(gid) = offline.group {
            world
                .spawn((
                    bundle,
                    Inventory(inv),
                    active_goal,
                    character,
                    Group { id: gid },
                ))
                .id()
        } else {
            world
                .spawn((bundle, Inventory(inv), active_goal, character))
                .id()
        };
        // Restore aggro across the tier boundary. The NPC was mid-
        // engagement when its region went offline; perception will
        // re-acquire LOS on the next `npc_aggro` tick, but seeding
        // the `Aggro` component now means goal arbitration picks
        // Pursue immediately rather than dropping back to Idle for
        // 200 ticks. The id is preserved verbatim — if the target
        // is no longer online here, aggro decays naturally.
        if let Some(target) = offline.aggro_target {
            world.entity_mut(entity).insert(crate::components::Aggro {
                target,
                last_seen_tick: offline.aggro_last_seen_tick,
            });
        }
    }
}

// --- Offline movement (Phase 1D) --------------------------------------------
//
// Offline NPCs hop between bases in their region. A real waypoint
// graph (per `npc-traversal-plan.md`) doesn't exist yet — for Phase
// 1D we treat every `Base` entity as a de facto waypoint. The system
// is shaped to swap in a proper graph once the navigation work lands:
// the per-NPC fields (`target_2d`, `arrival_offline_tick`,
// `travel_start_*`) describe a single linear leg, agnostic to how
// the target gets picked.
//
// Cadence: runs every sim tick, gated on `offline_tick_just_advanced`
// so the per-NPC work only happens 2 Hz (every 10 sim ticks). Each
// offline tick:
//   - NPCs with no target: pick a same-faction base in their region
//     as the new target. Empty region → idle. Single base region →
//     orbit back to it after each leg (still cheap, looks natural).
//   - NPCs with a target: linearly interpolate `position_2d` based on
//     elapsed offline ticks vs `arrival_offline_tick`. On arrival,
//     drop the target so the next tick picks a new one.
//
// Speed: stock walking pace ~6 m/s in the online tier; offline scales
// this same number through the slower clock. A 200 m hop takes 33 s
// of real time / 66 offline ticks.

/// Stock offline-tier walking speed in m/s. Matches the online-tier
/// patrol-walking pace; offline interpolation just samples this
/// continuously between ticks rather than doing per-frame
/// physics-y steering.
pub const OFFLINE_WALK_SPEED_M_PER_S: f32 = 6.0;

/// Per-offline-tick wall time at the stock 20 Hz sim rate. Used to
/// convert distance → offline-tick count for `arrival_offline_tick`.
const OFFLINE_TICK_SECONDS: f32 = (OFFLINE_TIER_TICK_DIVISOR as f32) / 20.0; // 10 / 20 = 0.5 s

/// Pick a target for an offline NPC that just finished its previous
/// leg (or never had one). Returns `None` if no candidate waypoint
/// exists in the region. Phase 1D uses same-faction `Base` entities
/// as waypoints; falls back to any `Base` in region; falls back to
/// `None` (NPC stays idle) when the region has no bases.
///
/// Iteration 5-13 Phase C2: when `waypoints` is `Some`, candidates
/// are filtered by graph reachability from `current_pos` — NPCs
/// stop picking bases that are on the wrong side of a painted /
/// stamped wall. If the filter empties the pool, falls back to
/// the unfiltered pool so a stranded NPC still picks something
/// rather than freezing.
fn pick_offline_target(
    rng: &mut rand_chacha::ChaCha8Rng,
    npc_region: RegionId,
    npc_faction: FactionId,
    bases: &[(FactionId, RegionId, [f32; 2])],
    current_pos: [f32; 2],
    waypoints: Option<&crate::nav::WaypointGraph>,
) -> Option<[f32; 2]> {
    use rand::seq::SliceRandom;
    // Build candidate list. Prefer same-faction; fall back to any-
    // faction in the same region.
    let mut same_faction: Vec<[f32; 2]> = Vec::new();
    let mut any_in_region: Vec<[f32; 2]> = Vec::new();
    for (fac, region, pos) in bases.iter().copied() {
        if region != npc_region {
            continue;
        }
        any_in_region.push(pos);
        if fac == npc_faction {
            same_faction.push(pos);
        }
    }
    let raw_pool = if !same_faction.is_empty() {
        same_faction
    } else if !any_in_region.is_empty() {
        any_in_region
    } else {
        return None;
    };

    // Phase C2: filter pool by waypoint-graph reachability when a
    // graph is available. If filter empties the pool, fall back to
    // the unfiltered pool (better some target than freezing).
    let pool = if let Some(graph) = waypoints {
        if let Some(start_node) = graph.nearest_node(current_pos) {
            let reachable: Vec<[f32; 2]> = raw_pool
                .iter()
                .copied()
                .filter(|&candidate| {
                    graph
                        .nearest_node(candidate)
                        .map(|goal| graph.reachable(start_node, goal))
                        .unwrap_or(false)
                })
                .collect();
            if reachable.is_empty() {
                raw_pool
            } else {
                reachable
            }
        } else {
            raw_pool
        }
    } else {
        raw_pool
    };
    // Avoid picking the current position (NPC sitting on a base) —
    // otherwise the next leg is zero-length and the NPC oscillates.
    // Cheap: just pick once and pick again if it's too close.
    let mut pick = *pool.choose(rng).expect("non-empty");
    if pool.len() > 1 && dist_2d(pick, current_pos) < 1.0 {
        pick = *pool.choose(rng).expect("non-empty");
    }
    Some(pick)
}

fn dist_2d(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dz = a[1] - b[1];
    (dx * dx + dz * dz).sqrt()
}

/// Movement tick for the offline tier. Gated on the offline clock
/// (only does work the sim tick the offline clock advances). For
/// each `OfflineNpc`, advance its current leg or pick a new one.
///
/// Iteration order: sorted by `NpcId` to keep RNG consumption
/// deterministic across sim instances — see
/// `crate::systems::npc_spawn::spawn_npcs` and
/// `tests/determinism.rs` for the established pattern.
#[allow(clippy::too_many_arguments)]
pub fn offline_movement(
    sim_clock: Res<SimClock>,
    offline_clock: Res<OfflineTierClock>,
    // Iteration 5-13 Phase C2: per-region waypoint graphs built
    // alongside the grid. NPC target selection filters by graph
    // reachability; movement walks segment-by-segment along the
    // resolved chain instead of bee-lining.
    nav: Res<crate::nav::NavQueries>,
    bases_q: Query<(
        &crate::components::Base,
        &crate::components::InFaction,
        &crate::components::InRegion,
        &crate::components::Position,
    )>,
    mut offline_q: Query<(bevy_ecs::entity::Entity, &mut OfflineNpc)>,
) {
    let _diag_t = crate::systems::SysTimer::new("offline_movement");
    let now_sim = sim_clock.tick;
    // Spread the work across ticks: only one region per *process*
    // tick, and we only process every `OFFLINE_PROCESS_INTERVAL_TICKS`
    // sim ticks. Combined effect: total offline-tier worker load
    // drops to ~`1 / INTERVAL` of the naive "every tick" path,
    // dialed for the design target of ~800 NPCs / region without
    // tank.
    if now_sim == 0 || !now_sim.is_multiple_of(OFFLINE_PROCESS_INTERVAL_TICKS) {
        return;
    }
    let now_offline = offline_clock.tick;
    // Phase 2 perf: spread work across sim ticks. We pick one
    // region per sim tick in round-robin order so the worker
    // doesn't burst on every offline-tick boundary. Each region
    // updates at ~5 Hz (with ~4 regions and 20 Hz sim) — same
    // visual fluidity as the old all-at-once heartbeat path but
    // distributed evenly.
    use std::collections::HashMap;
    let mut by_region: HashMap<RegionId, Vec<(NpcId, bevy_ecs::entity::Entity)>> = HashMap::new();
    for (e, o) in offline_q.iter() {
        by_region.entry(o.region).or_default().push((o.id, e));
    }
    if by_region.is_empty() {
        return;
    }
    let mut region_keys: Vec<RegionId> = by_region.keys().copied().collect();
    region_keys.sort_unstable();
    let region = region_keys[(now_sim as usize) % region_keys.len()];
    let mut order = match by_region.remove(&region) {
        Some(v) => v,
        None => return,
    };
    // Sort by NpcId for deterministic RNG-seed order.
    order.sort_by_key(|(id, _)| id.0);

    // Snapshot bases for the current region only — cheaper than
    // walking all bases every tick.
    let bases: Vec<(FactionId, RegionId, [f32; 2])> = bases_q
        .iter()
        .filter(|(_b, _f, r, _p)| r.0 == region)
        .map(|(_b, faction, _r, pos)| (faction.0, region, [pos.0[0], pos.0[2]]))
        .collect();

    let waypoints = nav.waypoints(region).map(|arc| arc.as_ref());

    for (_npc_id, entity) in order {
        let Ok((_, mut offline)) = offline_q.get_mut(entity) else {
            continue;
        };
        // Engaged NPCs hold position — the offline-tier dice
        // exchange resolves on its own cadence and routing them
        // off to a random base would break the firefight the
        // moment the region went offline. `offline_combat`
        // transitions back to `Idle` once the opponent dies or
        // aggro stales out.
        if matches!(offline.combat_state, OfflineCombatState::Engaged { .. }) {
            // Clear any in-flight travel so we freeze where we are
            // — the next non-Engaged offline tick will pick a
            // fresh target from the current position.
            if offline.target_2d.is_some() {
                offline.target_2d = None;
                offline.waypoint_chain.clear();
                offline.waypoint_chain_idx = 0;
            }
            continue;
        }
        match offline.target_2d {
            // No target → pick one + resolve waypoint chain.
            None => {
                use rand::SeedableRng;
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(
                    offline
                        .id
                        .0
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(now_offline),
                );
                let cur = offline.position_2d;
                let Some(target) = pick_offline_target(
                    &mut rng,
                    offline.region,
                    offline.faction,
                    &bases,
                    cur,
                    waypoints,
                ) else {
                    continue; // idle — no waypoints in region.
                };
                // Phase C2: resolve a chain through the graph
                // when available. The chain's hops drive
                // segment-by-segment movement; if no chain (no
                // graph, or path failed across an island), fall
                // back to bee-line by leaving the chain empty.
                let chain = waypoints
                    .and_then(|g| {
                        let start_node = g.nearest_node(cur)?;
                        let goal_node = g.nearest_node(target)?;
                        if start_node == goal_node {
                            return None;
                        }
                        g.path(start_node, goal_node)
                    })
                    .unwrap_or_default();
                let total_dist = chain_total_distance(chain.as_slice(), waypoints, cur, target);
                let leg_seconds = total_dist / OFFLINE_WALK_SPEED_M_PER_S;
                let leg_ticks = (leg_seconds / OFFLINE_TICK_SECONDS).ceil().max(1.0) as u64;
                offline.travel_start_2d = cur;
                offline.travel_start_offline_tick = now_offline;
                offline.target_2d = Some(target);
                offline.arrival_offline_tick = now_offline.saturating_add(leg_ticks);
                offline.waypoint_chain = chain;
                offline.waypoint_chain_idx = 0;
            }
            // Have a target → interpolate or arrive.
            Some(target) => {
                if now_offline >= offline.arrival_offline_tick {
                    offline.position_2d = target;
                    offline.target_2d = None;
                    offline.waypoint_chain.clear();
                    offline.waypoint_chain_idx = 0;
                } else {
                    let total = offline
                        .arrival_offline_tick
                        .saturating_sub(offline.travel_start_offline_tick)
                        .max(1);
                    let elapsed = now_offline.saturating_sub(offline.travel_start_offline_tick);
                    let t = (elapsed as f32 / total as f32).clamp(0.0, 1.0);
                    let start = offline.travel_start_2d;

                    // Phase C2: when a chain is present, walk
                    // along its segments instead of bee-lining
                    // start→target. The chain is interpreted as
                    // start → chain[0] → chain[1] → ... → target,
                    // each segment weighted by its distance.
                    offline.position_2d = if offline.waypoint_chain.is_empty() {
                        // Legacy / fallback path: straight line.
                        [
                            start[0] + (target[0] - start[0]) * t,
                            start[1] + (target[1] - start[1]) * t,
                        ]
                    } else {
                        position_along_chain(start, &offline.waypoint_chain, target, waypoints, t)
                    };
                }
            }
        }
    }
}

/// Iteration 5-13 Phase C2: compute the total length (meters) of a
/// chain that starts at `start`, walks through every node in
/// `chain` (referenced into `waypoints`), and ends at `target`.
/// When `chain` is empty or `waypoints` is missing, the answer is
/// the straight-line distance `start → target`.
fn chain_total_distance(
    chain: &[u32],
    waypoints: Option<&crate::nav::WaypointGraph>,
    start: [f32; 2],
    target: [f32; 2],
) -> f32 {
    let Some(graph) = waypoints else {
        return dist_2d(start, target);
    };
    if chain.is_empty() {
        return dist_2d(start, target);
    }
    let mut acc = 0.0;
    let mut prev = start;
    for &idx in chain {
        if let Some(p) = graph.nodes.get(idx as usize).copied() {
            acc += dist_2d(prev, p);
            prev = p;
        }
    }
    acc += dist_2d(prev, target);
    acc
}

/// Iteration 5-13 Phase C2: interpolate a position along the
/// chain `start → chain[0] → ... → chain[N-1] → target` at
/// fractional progress `t` in `[0, 1]`. Each segment is weighted
/// by its distance; `t == 0` returns `start`, `t == 1` returns
/// `target`.
fn position_along_chain(
    start: [f32; 2],
    chain: &[u32],
    target: [f32; 2],
    waypoints: Option<&crate::nav::WaypointGraph>,
    t: f32,
) -> [f32; 2] {
    let Some(graph) = waypoints else {
        return [
            start[0] + (target[0] - start[0]) * t,
            start[1] + (target[1] - start[1]) * t,
        ];
    };
    // Build the full segment list: start → chain[0] → ... → target.
    let mut points: Vec<[f32; 2]> = Vec::with_capacity(chain.len() + 2);
    points.push(start);
    for &idx in chain {
        if let Some(p) = graph.nodes.get(idx as usize).copied() {
            points.push(p);
        }
    }
    points.push(target);
    let total: f32 = points
        .windows(2)
        .map(|w| dist_2d(w[0], w[1]))
        .sum::<f32>()
        .max(f32::EPSILON);
    let mut remaining = t.clamp(0.0, 1.0) * total;
    for w in points.windows(2) {
        let seg_len = dist_2d(w[0], w[1]);
        if seg_len <= 0.0 {
            continue;
        }
        if remaining <= seg_len {
            let u = remaining / seg_len;
            return [
                w[0][0] + (w[1][0] - w[0][0]) * u,
                w[0][1] + (w[1][1] - w[0][1]) * u,
            ];
        }
        remaining -= seg_len;
    }
    target
}

// --- Offline combat + age-out (Phase 1E) ------------------------------------
//
// At 2 Hz, opposing-faction `OfflineNpc`s within `OFFLINE_ENGAGEMENT_RADIUS_M`
// roll a dice exchange. Outcomes degrade the defender's `HealthClass`
// (Healthy → Wounded → Critical → death). Events propagate to the
// online world via `WorldEventQueue` (already routed since PR #145):
//
// - `Gunshot { Intermediate }` on every engagement (rough audibility
//   approximation; refines once `LoadoutClass` carries weapon kind).
// - `AllyDown { id, faction }` on every combat death.
//
// Same system handles offline age-out: NPCs with `die_at_tick <=
// sim_clock.tick` die naturally and pick up a chronicle entry. No
// `AllyDown` for natural deaths — squad blackboards shouldn't react
// to old-age.

/// Engagement radius for offline combat. Wider than the online sight
/// radius (~80 m) to compensate for the slower offline tick rate —
/// at 2 Hz, an aggressor closing 80 m would miss several engagement
/// windows. Tuning lives here for the next playtest pass.
pub const OFFLINE_ENGAGEMENT_RADIUS_M: f32 = 150.0;

/// Approximate ticks-to-kill at full pair engagement. With per-tick
/// `OFFLINE_COMBAT_HIT_CHANCE = 0.05` and three HealthClass steps
/// (Healthy → Wounded → Critical → Dead), expected total = 60
/// offline ticks ≈ 30 s of sim time. Matches the design target in
/// `offline-tier-plan.md` §8 "Dice cadence within offline combat."
const OFFLINE_COMBAT_HIT_CHANCE: f32 = 0.05;

/// Sim ticks of "no proximity to opponent" before an `Engaged`
/// offline NPC ages back to `Idle`. Mirrors the 200-tick online
/// aggro decay (~10 s at 20 Hz) so cross-tier behaviour stays
/// consistent: an NPC that hasn't seen / been near its opponent
/// for 10 s drops the engagement and resumes wandering. Without
/// this an opponent's death (or projection out of the region)
/// would freeze the survivor in `Engaged` forever — `offline_movement`
/// skips Engaged NPCs, so they'd stand still until the region
/// re-activated.
const OFFLINE_ENGAGE_STALE_TICKS: u64 = 200;

/// Combat tick for the offline tier. Runs every sim tick (NOT
/// gated on the heartbeat) but only processes **one region per
/// tick** in a round-robin schedule, so the worker spreads the
/// O(n²) → O(n) bucketed scan evenly across the offline-tier
/// period instead of cramming everything into one tick every 0.5s.
///
/// Spatial bucketing: for each region, NPCs are placed into 150 m
/// cells (matching `OFFLINE_ENGAGEMENT_RADIUS_M`). Each NPC checks
/// pairs only within its cell and 8 neighbors, so the pair-scan
/// cost is ~O(n) instead of O(n²). With 800 NPCs in a region this
/// drops the per-tick cost from ~320k comparisons to ~7k.
///
/// Determinism: pairs iterated in `(min_id, max_id)`-sorted order,
/// RNG seeded per pair off `(pair_key, sim_tick)`. Round-robin
/// region order is a stable sort of `RegionId`.
#[allow(clippy::too_many_arguments)]
pub fn offline_combat(
    sim_clock: Res<SimClock>,
    _offline_clock: Res<OfflineTierClock>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    deltas: Res<crate::faction::registry::RelationDeltas>,
    mut event_bus: ResMut<crate::world_event_bus::WorldEventQueue>,
    mut chronicle: ResMut<crate::chronicle::LifeChronicle>,
    mut pda_log: ResMut<crate::pda_log::PdaEventLog>,
    mut commands: Commands,
    bases_q: Query<(
        &crate::components::Base,
        &crate::components::InFaction,
        &crate::components::InRegion,
        &crate::components::Position,
    )>,
    mut offline_q: Query<(bevy_ecs::entity::Entity, &mut OfflineNpc)>,
) {
    let _diag_t = crate::systems::SysTimer::new("offline_combat");
    let now_sim = sim_clock.tick;
    // Match `offline_movement`'s rate-limit so the combined offline-
    // tier worker burden stays well under the tick budget even at
    // 800 NPCs / region.
    if now_sim == 0 || !now_sim.is_multiple_of(OFFLINE_PROCESS_INTERVAL_TICKS) {
        return;
    }

    // Round-robin pick: which region gets processed this sim tick.
    // We bucket offline NPCs by region first, then index the
    // sorted region list with `now_sim % len`. Each region gets a
    // tick worth of work every N sim ticks; with N ≤ ~10 regions
    // this still works out to ≥ 2 Hz per-region update — same
    // cadence as the old heartbeat-gated path, but no spike.
    #[derive(Clone)]
    struct Snap {
        entity: bevy_ecs::entity::Entity,
        id: NpcId,
        faction: FactionId,
        pos: [f32; 2],
        health_class: HealthClass,
        die_at_tick: u64,
        accuracy: u8,
    }
    use std::collections::HashMap;
    let mut by_region: HashMap<RegionId, Vec<Snap>> = HashMap::new();
    for (entity, o) in offline_q.iter() {
        by_region.entry(o.region).or_default().push(Snap {
            entity,
            id: o.id,
            faction: o.faction,
            pos: o.position_2d,
            health_class: o.health_class,
            die_at_tick: o.die_at_tick,
            accuracy: o.stats.accuracy,
        });
    }
    if by_region.is_empty() {
        return;
    }
    let mut region_keys: Vec<RegionId> = by_region.keys().copied().collect();
    region_keys.sort_unstable();
    let region = region_keys[(now_sim as usize) % region_keys.len()];
    let mut snaps = match by_region.remove(&region) {
        Some(v) => v,
        None => return,
    };
    // Sort by NpcId for deterministic iteration / RNG-seed order.
    snaps.sort_by_key(|s| s.id.0);
    // Free the by_region HashMap immediately; everything else
    // operates on the single-region `snaps` Vec.
    drop(by_region);

    // Track per-NPC damage applied this tick + deaths so each NPC
    // only degrades by at most one class per tick (avoid a triple
    // hit in a single tick).
    let mut new_class: HashMap<NpcId, HealthClass> = HashMap::new();
    let mut dead_this_tick: Vec<(NpcId, FactionId, RegionId, [f32; 2], DeathCauseKind)> =
        Vec::new();
    // Coalesce per-tick offline gunfire to one PDA entry per region —
    // raw `Gunshot` bus events fire per pair, which would flood the
    // toast queue. AI still sees every individual shot via the bus.
    let mut gunfire_regions: std::collections::HashSet<RegionId> = std::collections::HashSet::new();

    enum DeathCauseKind {
        Combat { killer_faction: FactionId },
        Natural,
    }

    // Age-out pass first — natural-cause deaths land in chronicle
    // before combat events that might also kill the same NPC.
    for s in snaps.iter() {
        if now_sim >= s.die_at_tick {
            dead_this_tick.push((s.id, s.faction, region, s.pos, DeathCauseKind::Natural));
        }
    }

    // Build spatial bucket: 150 m cells matching the engagement
    // radius. Each NPC checks pairs within its own cell + 8
    // neighbors. With ~800 NPCs in a 4500 m region this drops the
    // pair-scan from ~320k comparisons to ~7k — and we only do
    // ONE region per sim tick, so the per-tick cost stays small.
    let cell_size = OFFLINE_ENGAGEMENT_RADIUS_M;
    let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, s) in snaps.iter().enumerate() {
        let cx = (s.pos[0] / cell_size).floor() as i32;
        let cz = (s.pos[1] / cell_size).floor() as i32;
        cells.entry((cx, cz)).or_default().push(i);
    }

    // Pre-build a HashSet of dead ids so the per-pair "is dead"
    // check is O(1) instead of O(deaths). With most NPCs alive
    // this matters less but compounds with cell scans.
    let mut already_dead: std::collections::HashSet<NpcId> =
        dead_this_tick.iter().map(|(id, ..)| *id).collect();

    let r2 = cell_size * cell_size;
    // Sort cell keys for deterministic RNG-seed order.
    let mut cell_keys: Vec<(i32, i32)> = cells.keys().copied().collect();
    cell_keys.sort_unstable();
    let mut seen_pair: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
    // Track which (npc, opponent) pairs exchanged fire this tick.
    // Drives the `combat_state` refresh below: a hit refreshes
    // `since_tick` to `now_sim` so the stale-out clock restarts.
    // Both directions are recorded — order matters since the
    // map is keyed by the *self* id.
    let mut engaged_refresh: HashMap<NpcId, NpcId> = HashMap::new();
    for (cx, cz) in cell_keys {
        let own = match cells.get(&(cx, cz)) {
            Some(v) => v.clone(),
            None => continue,
        };
        // Inspect own cell + 8 neighbors (full Moore neighborhood).
        // `seen_pair` dedups across multiple cell-iteration paths
        // that would otherwise visit the same pair twice.
        let neighbors: [(i32, i32); 9] = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (0, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];
        for (dx, dz) in neighbors {
            let key = (cx + dx, cz + dz);
            let other = match cells.get(&key) {
                Some(v) => v.clone(),
                None => continue,
            };
            for &i_a in own.iter() {
                for &i_b in other.iter() {
                    if i_a == i_b {
                        continue;
                    }
                    let a = &snaps[i_a];
                    let b = &snaps[i_b];
                    // Dedup the (a, b) / (b, a) pair via a canonical
                    // (lo, hi) HashSet entry. Cross-cell iteration
                    // would otherwise visit the same pair twice.
                    let lo = a.id.0.min(b.id.0);
                    let hi = a.id.0.max(b.id.0);
                    if !seen_pair.insert((lo, hi)) {
                        continue;
                    }
                    if a.faction == b.faction {
                        continue;
                    }
                    if already_dead.contains(&a.id) || already_dead.contains(&b.id) {
                        continue;
                    }
                    let relation = crate::faction::registry::faction_relation(
                        &registry, &deltas, a.faction, b.faction,
                    );
                    if !matches!(relation, crate::faction::Relation::Hostile) {
                        continue;
                    }
                    let ddx = a.pos[0] - b.pos[0];
                    let ddz = a.pos[1] - b.pos[1];
                    if ddx * ddx + ddz * ddz > r2 {
                        continue;
                    }
                    // Per-pair RNG keyed off canonical pair + sim_tick.
                    // Cheaper integer-hash gen_bool than re-seeding a
                    // ChaCha8 per pair — we trust the pair-key entropy.
                    use rand::{Rng, SeedableRng};
                    let seed = lo
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(hi.wrapping_mul(0xBF58_476D_1CE4_E5B9))
                        .wrapping_add(now_sim);
                    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                    let mid_xz = [(a.pos[0] + b.pos[0]) * 0.5, (a.pos[1] + b.pos[1]) * 0.5];
                    event_bus.push(
                        crate::world_event_bus::WorldEventKind::Gunshot {
                            caliber_class: crate::world_event_bus::CaliberClass::Intermediate,
                        },
                        [mid_xz[0], 0.0, mid_xz[1]],
                        region,
                        now_sim,
                        /* ttl_ticks = */ 4,
                    );
                    gunfire_regions.insert(region);
                    // Mark both sides as Engaged-this-tick so the
                    // stale-out clock restarts. New engagements
                    // (Idle → Engaged) and existing ones both flow
                    // through this path — `offline_movement` then
                    // freezes their travel until aggro ages out.
                    engaged_refresh.insert(a.id, b.id);
                    engaged_refresh.insert(b.id, a.id);
                    let a_hit = OFFLINE_COMBAT_HIT_CHANCE + (f32::from(a.accuracy) - 50.0) * 0.001;
                    let b_hit = OFFLINE_COMBAT_HIT_CHANCE + (f32::from(b.accuracy) - 50.0) * 0.001;
                    let a_hits_b = rng.gen_bool(a_hit.clamp(0.0, 0.5) as f64);
                    let b_hits_a = rng.gen_bool(b_hit.clamp(0.0, 0.5) as f64);
                    if a_hits_b {
                        let cur = *new_class.get(&b.id).unwrap_or(&b.health_class);
                        if let Some(next) = degrade_health_class(cur) {
                            new_class.insert(b.id, next);
                        } else {
                            dead_this_tick.push((
                                b.id,
                                b.faction,
                                region,
                                b.pos,
                                DeathCauseKind::Combat {
                                    killer_faction: a.faction,
                                },
                            ));
                            already_dead.insert(b.id);
                        }
                    }
                    if b_hits_a {
                        let cur = *new_class.get(&a.id).unwrap_or(&a.health_class);
                        if let Some(next) = degrade_health_class(cur) {
                            new_class.insert(a.id, next);
                        } else {
                            dead_this_tick.push((
                                a.id,
                                a.faction,
                                region,
                                a.pos,
                                DeathCauseKind::Combat {
                                    killer_faction: b.faction,
                                },
                            ));
                            already_dead.insert(a.id);
                        }
                    }
                }
            }
        }
    }

    // Apply engagement refresh + stale-out. Iterates the live
    // offline_q in this region; touches `combat_state` directly.
    // Order: refresh first (so newly-engaged + ongoing-engaged
    // get since_tick = now_sim), then stale-out (so anything that
    // didn't refresh this tick AND hasn't refreshed in
    // OFFLINE_ENGAGE_STALE_TICKS ages back to Idle).
    let region_entities: Vec<bevy_ecs::entity::Entity> = snaps.iter().map(|s| s.entity).collect();
    for entity in region_entities {
        let Ok((_, mut offline)) = offline_q.get_mut(entity) else {
            continue;
        };
        if let Some(opponent) = engaged_refresh.get(&offline.id).copied() {
            offline.combat_state = OfflineCombatState::Engaged {
                opponent,
                since_tick: now_sim,
            };
            offline.aggro_target = Some(opponent);
            offline.aggro_last_seen_tick = now_sim;
        } else if let OfflineCombatState::Engaged { since_tick, .. } = offline.combat_state {
            if now_sim.saturating_sub(since_tick) > OFFLINE_ENGAGE_STALE_TICKS {
                offline.combat_state = OfflineCombatState::Idle;
                offline.aggro_target = None;
            }
        }
    }

    // Apply HealthClass updates first (skip those marked dead).
    let dead_ids: std::collections::HashSet<NpcId> =
        dead_this_tick.iter().map(|(id, ..)| *id).collect();
    if !new_class.is_empty() {
        // Iterate ECS to update components directly. Cheap O(degraded).
        let entities_to_update: Vec<(bevy_ecs::entity::Entity, HealthClass)> = snaps
            .iter()
            .filter_map(|s| {
                if dead_ids.contains(&s.id) {
                    return None;
                }
                new_class.get(&s.id).map(|c| (s.entity, *c))
            })
            .collect();
        for (entity, hc) in entities_to_update {
            if let Ok((_, mut o)) = offline_q.get_mut(entity) {
                o.health_class = hc;
            }
        }
    }

    // Apply deaths.
    for (id, faction, region, pos, cause) in &dead_this_tick {
        let id = *id;
        let faction = *faction;
        let region = *region;
        let pos = *pos;
        // Find entity for despawn.
        if let Some(s) = snaps.iter().find(|s| s.id == id) {
            commands.entity(s.entity).despawn();
        }
        // Chronicle update — routes through `mark_dead` so the
        // summary cache stays in sync.
        let death_cause = match &cause {
            DeathCauseKind::Combat { killer_faction } => crate::chronicle::DeathCause::Combat {
                killer_faction: registry.name_of(*killer_faction).to_string(),
            },
            DeathCauseKind::Natural => crate::chronicle::DeathCause::NaturalCauses,
        };
        chronicle.mark_dead(id, now_sim, region, death_cause);
        // Bus event — combat deaths only.
        if let DeathCauseKind::Combat { killer_faction } = cause {
            let killer_faction = *killer_faction;
            event_bus.push(
                crate::world_event_bus::WorldEventKind::AllyDown { id, faction },
                [pos[0], 0.0, pos[1]],
                region,
                now_sim,
                /* ttl_ticks = */ 4,
            );
            // Phase 1F: surface the kill to the player's PDA toast
            // queue. Natural-causes deaths don't get a PDA entry —
            // age-out should be invisible to the player.
            pda_log.push(
                crate::pda_log::PdaEvent::OfflineCombatDeath {
                    killed_faction: registry.name_of(faction).to_string(),
                    killer_faction: registry.name_of(killer_faction).to_string(),
                    region,
                },
                now_sim,
            );
        }
    }

    // Phase 1F: emit a coalesced "Gunfire in [region]" PDA entry,
    // but debounce so a region that's continuously firing doesn't
    // flood the toast queue. We only push if the most recent
    // OfflineGunfire entry for the region (if any) is older than
    // `GUNFIRE_PDA_COOLDOWN_TICKS` sim ticks ago — matches the
    // pre-spread heartbeat cadence (~0.5 s).
    const GUNFIRE_PDA_COOLDOWN_TICKS: u64 = 10;
    for region in gunfire_regions {
        let recent = pda_log.all().rev().find(|e| {
            matches!(
                &e.event,
                crate::pda_log::PdaEvent::OfflineGunfire { region: r } if *r == region,
            )
        });
        let on_cooldown = recent
            .map(|e| now_sim.saturating_sub(e.tick) < GUNFIRE_PDA_COOLDOWN_TICKS)
            .unwrap_or(false);
        if on_cooldown {
            continue;
        }
        pda_log.push(crate::pda_log::PdaEvent::OfflineGunfire { region }, now_sim);
    }

    // Phase 1F: BaseFlip heuristic — scan bases against the current
    // offline-NPC population. If a base's local majority faction
    // changed since last check, emit a BaseFlip event. Tracks state
    // in the static `last_dominance` map keyed off Base entity. This
    // is a placeholder until contestation (`contestation-plan.md`)
    // lands — that system will own ownership flips properly. The
    // 80 m radius is intentionally tight so a single fly-by doesn't
    // count as "took the base."
    const BASE_DOMINANCE_RADIUS_M: f32 = 80.0;
    let bd2 = BASE_DOMINANCE_RADIUS_M * BASE_DOMINANCE_RADIUS_M;
    // Per-tick region rotation: we only have `snaps` for the
    // single region we're processing, so filter bases to that
    // region up front.
    let bases_snap: Vec<(FactionId, [f32; 2])> = bases_q
        .iter()
        .filter(|(_b, _f, r, _p)| r.0 == region)
        .map(|(_b, faction, _r, pos)| (faction.0, [pos.0[0], pos.0[2]]))
        .collect();
    for (base_owner, base_pos) in bases_snap {
        let base_region = region;
        // Count offline NPCs in range by faction.
        let mut faction_counts: HashMap<FactionId, u32> = HashMap::new();
        for s in &snaps {
            // Skip NPCs that died this tick.
            if already_dead.contains(&s.id) {
                continue;
            }
            let dx = s.pos[0] - base_pos[0];
            let dz = s.pos[1] - base_pos[1];
            if dx * dx + dz * dz > bd2 {
                continue;
            }
            *faction_counts.entry(s.faction).or_default() += 1;
        }
        // Dominant faction = highest count, requires at least 2 NPCs
        // to count as "contesting" (one is just a passerby).
        let dominant: Option<FactionId> = faction_counts
            .iter()
            .filter(|(_, &c)| c >= 2)
            .max_by_key(|(_, c)| **c)
            .map(|(f, _)| *f);
        // If dominant differs from base's authored owner AND dominant
        // is hostile to the owner, surface as a BaseFlip. Only fires
        // ONCE per dominant per offline tick (no per-base state
        // tracking yet — a "did this flip JUST happen" check would
        // need an extra resource and the placeholder semantics don't
        // warrant it). The dedup at the PDA UI layer suppresses
        // duplicates from showing as toasts.
        if let Some(new_owner_id) = dominant {
            if new_owner_id != base_owner {
                let rel = crate::faction::registry::faction_relation(
                    &registry,
                    &deltas,
                    new_owner_id,
                    base_owner,
                );
                if matches!(rel, crate::faction::Relation::Hostile) {
                    // Bus event for AI listeners.
                    event_bus.push(
                        crate::world_event_bus::WorldEventKind::BaseFlip {
                            new_owner: new_owner_id,
                            old_owner: Some(base_owner),
                        },
                        [base_pos[0], 0.0, base_pos[1]],
                        base_region,
                        now_sim,
                        /* ttl_ticks = */ 4,
                    );
                    // PDA toast.
                    pda_log.push(
                        crate::pda_log::PdaEvent::BaseFlip {
                            new_owner: registry.name_of(new_owner_id).to_string(),
                            old_owner: Some(registry.name_of(base_owner).to_string()),
                            region: base_region,
                        },
                        now_sim,
                    );
                }
            }
        }
    }

    // Trim aged entries from the PDA log.
    pda_log.evict_old(now_sim);
}

/// Degrade `class` by one step. Returns `None` if already at
/// `Critical` (caller treats `None` as death).
fn degrade_health_class(class: HealthClass) -> Option<HealthClass> {
    match class {
        HealthClass::Healthy => Some(HealthClass::Wounded),
        HealthClass::Wounded => Some(HealthClass::Critical),
        HealthClass::Critical => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heartbeat_advances_every_divisor_ticks() {
        // Build a minimal world with just the two clock resources.
        let mut world = bevy_ecs::world::World::new();
        world.insert_resource(SimClock::new());
        world.insert_resource(OfflineTierClock::default());

        // Manually advance the sim clock and run the heartbeat as a
        // one-shot system between each step.
        use bevy_ecs::system::RunSystemOnce;
        for sim_tick in 1u64..=25 {
            world.resource_mut::<SimClock>().tick = sim_tick;
            (&mut world).run_system_once(tick_offline_clock).unwrap();
        }

        let offline = *world.resource::<OfflineTierClock>();
        // sim_tick 10 → offline=1, 20 → offline=2; sim_tick 25 hasn't
        // reached 30 yet so offline stays at 2.
        assert_eq!(
            offline.tick, 2,
            "expected 2 offline ticks after 25 sim ticks at divisor=10, got {offline:?}"
        );
        assert_eq!(offline.last_sim_tick, 20);
    }

    #[test]
    fn heartbeat_no_op_when_sim_clock_paused() {
        let mut world = bevy_ecs::world::World::new();
        world.insert_resource(SimClock::new());
        world.insert_resource(OfflineTierClock::default());
        use bevy_ecs::system::RunSystemOnce;

        // Advance once to tick 10, then re-run heartbeat 5 times at
        // the same sim tick. Offline clock should advance exactly once.
        world.resource_mut::<SimClock>().tick = 10;
        for _ in 0..5 {
            (&mut world).run_system_once(tick_offline_clock).unwrap();
        }
        assert_eq!(world.resource::<OfflineTierClock>().tick, 1);
    }
}
