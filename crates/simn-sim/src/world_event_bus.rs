//! AI-strategic event bus — the propagation cousin of the encounter
//! dispatcher.
//!
//! Where [`encounter-dispatcher-plan.md`](../planning/encounter-dispatcher-plan.md)
//! routes hand-authored encounter triggers to gameplay runners, this
//! module routes *simulated* events between AI agents. NPC fires a
//! gun → a `Gunshot` event lands on the queue → at the next tick,
//! the drain delivers it to every nearby squad blackboard via
//! spatial decay + per-kind faction-audience filter. Reactive squad
//! behavior (relief, investigate, mourn) follows from blackboard
//! reads, not from squad-internal pair scans.
//!
//! ## Push / drain cadence
//!
//! - **Push** (any tick): producers (`npc_aggro`, future
//!   `npc_combat` / `npc_death_check` / `contestation`) call
//!   [`WorldEventQueue::push`] inside their normal pass.
//! - **Drain** (tick start): [`drain_world_events`] runs early in
//!   the NPC tick chain — alongside `clear_los_cache` and
//!   `sweep_squad_blackboards`. Empties the queue, applies each
//!   event to relevant squad blackboards, then drops events past
//!   their TTL.
//!
//! The 1-tick delay between producer and consumer is deliberate.
//! Same-tick fan-out would require ordering producers before drain,
//! and consumers after — pinning the schedule shape brittlely. The
//! delay is undetectable at 20 Hz (50 ms).
//!
//! ## No de-duplication
//!
//! Decided 2026-05-05: if 5 squad members fire in one tick, that's
//! 5 events. Believability over compute. The drain is O(events ×
//! groups) per tick; with ~hundreds of events and ~tens of groups
//! that's bounded.
//!
//! ## Phase 1
//!
//! Pure primitive — one emitter wired (`npc_aggro` pushes
//! `EnemySighted` on new acquisition); blackboard writers for the
//! other event kinds land in their own PRs (combat → Gunshot,
//! death-check → AllyDown, contestation → BaseFlip, etc.). No Godot
//! bridge yet.
//!
//! See `docs/book/src/planning/world-event-bus-plan.md`.

use std::collections::HashMap;

use bevy_ecs::prelude::{Query, Res, ResMut, Resource};

use crate::components::{Group, InFaction, Npc, NpcId};
use crate::faction::Relation;
use crate::region::RegionId;
use crate::resources::{NpcPositionIndex, SimClock};
use crate::squad_blackboard::{BlackboardKey, BlackboardValue, SquadBlackboards};

/// Caliber-class taxonomy. Locked per `weapons-plan.md` §4 and the
/// `dismemberment-plan.md` §5 sever-threshold table. Seven classes
/// covering everything from PDWs to anti-materiel rifles plus
/// shotguns as their own bucket. `audible_radius_m`, the future
/// `resolve_wound_kind` (dismemberment step 2), and `npc_combat`
/// aim-cone scaling all key off this.
///
/// The legacy three-band names (`Light` / `Medium` / `Heavy`) used
/// by older callsites map onto the new model:
/// - `Light` → `Pistol` (pistols, SMGs grouped together by audibility).
/// - `Medium` → `Intermediate` (assault rifles, AKs).
/// - `Heavy` → `FullPowerRifle` (battle rifles, marksman rifles).
///
/// Use [`Self::audible_band`] to recover the old three-band axis when
/// a downstream system only cares about audible-radius bucketing.
///
/// Configurable per-caliber via `items.toml`'s `[ammo]` entries —
/// each `AmmoConfig` carries a `caliber_class` tag that downstream
/// systems (audibility, future `resolve_wound_kind`) read directly.
#[derive(
    serde::Serialize, serde::Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum CaliberClass {
    /// 9mm, .45 ACP — pistols + SMGs.
    #[default]
    Pistol,
    /// 4.6×30, 5.7×28 — Personal-Defense Weapons. Smaller than
    /// pistol audibility, supersonic short-range carbines.
    #[serde(rename = "pdw")]
    PDW,
    /// 5.45×39, 5.56×45, 7.62×39 — assault rifles, the workhorse
    /// caliber band.
    Intermediate,
    /// .308, 7.62×54R — battle rifles, marksman rifles.
    FullPowerRifle,
    /// .338 Lapua — magnum-class precision rifles.
    Magnum,
    /// .50 BMG, 14.5×114 — anti-materiel rifles.
    AntiMateriel,
    /// 12ga, 20ga — shotguns. Audibility distinct from pistol band
    /// (lower-frequency boom carries).
    Shotgun,
}

/// Three-band audibility bucket retained for legacy mapping.
/// `audible_radius_m` uses the seven-class enum directly; this
/// helper exists for callers (older tests, the 3-band telemetry in
/// some debug overlays) that haven't migrated yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaliberAudibleBand {
    Light,
    Medium,
    Heavy,
}

impl CaliberClass {
    /// Map to the legacy three-band axis. Useful when a consumer
    /// only cares about "how loud" a class is, not which specific
    /// caliber fired.
    pub fn audible_band(self) -> CaliberAudibleBand {
        match self {
            Self::Pistol | Self::PDW | Self::Shotgun => CaliberAudibleBand::Light,
            Self::Intermediate => CaliberAudibleBand::Medium,
            Self::FullPowerRifle | Self::Magnum | Self::AntiMateriel => CaliberAudibleBand::Heavy,
        }
    }
}

/// Chatter event flavor. Drives stealth (overhearing patrols) and
/// near-NPC flavor; producers for chatter are out-of-scope for this
/// PR but the variant ships now so the queue's enum is stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatterIntent {
    /// Patrol banter, idle conversation. Player flavor + stealth
    /// "I hear voices" cue.
    IdleConversation,
    /// "Contact!", "Reload!", "Flank left!" — squad coordination.
    Callout,
    /// "Enemy spotted!" — drives squad alertness.
    Alarm,
    /// Post-combat, ally-down chatter.
    Mourning,
}

/// Closed enum of bus event kinds. New variants land here as the AI
/// systems that need them ship; mods that want their own events use
/// the [`WorldEventKind::ModExtension`] variant (mirrors the squad
/// blackboard's `Custom` key).
#[derive(Clone, Debug)]
pub enum WorldEventKind {
    /// A weapon was fired. `caliber_class` flavors the audible
    /// radius (Heavy carries further than Light).
    Gunshot { caliber_class: CaliberClass },
    /// A grenade / vehicle / charge detonated. `magnitude` is a
    /// rough relative scale (1.0 = frag grenade).
    Explosion { magnitude: f32 },
    /// A friendly went down. `id` is the dead NPC; `faction` filters
    /// the audience to allies.
    AllyDown {
        id: NpcId,
        faction: crate::faction::registry::FactionId,
    },
    /// A spotter saw a hostile. Audience: factions hostile to
    /// `target_faction`.
    EnemySighted {
        target_id: NpcId,
        target_faction: crate::faction::registry::FactionId,
    },
    /// A corpse / loot pile was visible. Audience: same faction
    /// (mourn / loot) — not the killers, who already know.
    CorpseSpotted {
        faction: crate::faction::registry::FactionId,
    },
    /// A base changed ownership. Global within the new owner's
    /// faction; ignored by other audiences for now.
    BaseFlip {
        new_owner: crate::faction::registry::FactionId,
        old_owner: Option<crate::faction::registry::FactionId>,
    },
    /// A player was sighted. Hostile factions react.
    PlayerSighted { player_id: u64 },
    /// A squad portal-crossed. Mostly informational; keeps a
    /// hook for future "track migrating squad" behaviors.
    PortalUsed {
        from: RegionId,
        to: RegionId,
        faction: crate::faction::registry::FactionId,
    },
    /// An NPC said / shouted something audible. See [`ChatterIntent`].
    Chatter {
        speaker: NpcId,
        intent: ChatterIntent,
    },
    /// Iteration 5-13 Phase D3: an NPC arrived at a designer-
    /// placed interaction area (`InteractionArea`) and is starting
    /// to occupy it. `area_id` is the stable per-area key shipped
    /// from `InteractionAreaMarker3D` (or auto-derived); `kind`
    /// is the free-form descriptor (`"rest"`, `"work"`, etc.).
    /// Emitted once per (npc, area) pair on arrival; the
    /// `InteractionEnded` counterpart closes the pair on
    /// objective-change / leave / death.
    InteractionStarted {
        npc_id: NpcId,
        area_id: String,
        kind: String,
    },
    /// Iteration 5-13 Phase D3: paired with [`Self::InteractionStarted`].
    /// Fires when an NPC leaves the area (objective swap, walked
    /// out of extents, squad died). The PDA log can use these
    /// to time-bracket "squad X rested at the river camp from
    /// tick A to tick B" toasts.
    InteractionEnded { npc_id: NpcId, area_id: String },
    /// Mod-defined event. `mod_id` namespaces the kind so two mods
    /// can use the same `name` without collision. Bus drains
    /// pass-through; mod systems own their own consumers.
    ModExtension {
        mod_id: u32,
        name: std::borrow::Cow<'static, str>,
    },
}

/// One queued event. Producers fill this; the drain consumes.
#[derive(Clone, Debug)]
pub struct WorldEvent {
    pub id: u64,
    pub kind: WorldEventKind,
    /// World-local position the event occurred at. Used by the
    /// spatial-decay filter and (when applicable) written to
    /// listening squads' blackboards.
    pub position: [f32; 3],
    /// Region the event occurred in.
    pub region: RegionId,
    /// Tick the event was pushed at.
    pub created_tick: u64,
    /// Drain drops the event after `created_tick + ttl_ticks`.
    /// Default 1 tick — events are AI inputs, not durable state.
    pub ttl_ticks: u32,
}

/// FIFO event queue. Resource lives in the ECS world; producers push
/// into it from any system; drain runs at the head of each tick.
#[derive(Resource, Default)]
pub struct WorldEventQueue {
    events: Vec<WorldEvent>,
    next_id: u64,
}

impl WorldEventQueue {
    /// Push an event with `ttl_ticks` lifetime. Returns the assigned
    /// event id (mostly useful for tests / instrumentation).
    pub fn push(
        &mut self,
        kind: WorldEventKind,
        position: [f32; 3],
        region: RegionId,
        created_tick: u64,
        ttl_ticks: u32,
    ) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.events.push(WorldEvent {
            id,
            kind,
            position,
            region,
            created_tick,
            ttl_ticks,
        });
        id
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Drain helper used by [`drain_world_events`]. Empties the
    /// queue and returns the events; drops events past TTL.
    fn take_active(&mut self, now: u64) -> Vec<WorldEvent> {
        let drained = std::mem::take(&mut self.events);
        drained
            .into_iter()
            .filter(|e| now <= e.created_tick.saturating_add(e.ttl_ticks as u64))
            .collect()
    }
}

/// Audible radius (meters) for an event kind. Linear falloff above —
/// past the radius, listeners don't notice. Tunable; numbers picked
/// for "realistic-but-gameyish" per the locked decision (2026-05-05).
pub fn audible_radius_m(kind: &WorldEventKind) -> f32 {
    match kind {
        WorldEventKind::Gunshot { caliber_class } => match caliber_class {
            // Pistol-class: pistols, SMGs.
            CaliberClass::Pistol => 180.0,
            // PDWs are quieter than pistols on average.
            CaliberClass::PDW => 150.0,
            // Shotguns: lower-frequency boom carries differently
            // from a pistol crack but ends up similar at the
            // gameplay-relevant distance band.
            CaliberClass::Shotgun => 200.0,
            // Intermediate (5.56, 5.45, 7.62×39): the assault-rifle
            // workhorse band.
            CaliberClass::Intermediate => 250.0,
            // Full-power rifle (.308, 7.62×54R): battle / marksman.
            CaliberClass::FullPowerRifle => 350.0,
            // Magnum (.338 Lapua): louder than full-power.
            CaliberClass::Magnum => 450.0,
            // Anti-materiel (.50 BMG, 14.5×114): unmistakable.
            CaliberClass::AntiMateriel => 600.0,
        },
        WorldEventKind::Explosion { magnitude } => 400.0 * magnitude.max(0.5),
        WorldEventKind::AllyDown { .. } => 150.0,
        WorldEventKind::EnemySighted { .. } => 200.0,
        WorldEventKind::CorpseSpotted { .. } => 60.0,
        WorldEventKind::BaseFlip { .. } => f32::INFINITY,
        WorldEventKind::PlayerSighted { .. } => 250.0,
        WorldEventKind::PortalUsed { .. } => 50.0,
        WorldEventKind::Chatter { .. } => 30.0,
        WorldEventKind::ModExtension { .. } => 100.0,
        // Interaction events are PDA-toast inputs, not AI-audible
        // shouts — short audible radius keeps them from polluting
        // squad blackboards that happen to be nearby.
        WorldEventKind::InteractionStarted { .. } | WorldEventKind::InteractionEnded { .. } => 0.0,
    }
}

/// Who hears a given event.
#[derive(Clone, Copy, Debug)]
pub enum Audience {
    /// Anyone within radius regardless of faction.
    Anyone,
    /// Only same-faction listeners (e.g. ally-down).
    SameFaction(crate::faction::registry::FactionId),
    /// Only listeners hostile to `target` (e.g. enemy-sighted).
    HostileTo(crate::faction::registry::FactionId),
    /// Global within a specific faction (e.g. base-flip notifying
    /// the new owner faction across the whole map).
    GlobalFaction(crate::faction::registry::FactionId),
}

impl Audience {
    // Kept for tests + future direct callers. The hot drain path
    // inlines this via a pre-computed hostility matrix to skip the
    // `RelationDeltas` String allocations.
    #[cfg_attr(not(test), allow(dead_code))]
    fn passes(
        self,
        listener_faction: crate::faction::registry::FactionId,
        registry: &crate::faction::registry::FactionRegistry,
        deltas: &crate::faction::registry::RelationDeltas,
    ) -> bool {
        use crate::faction::registry::faction_relation;
        match self {
            Self::Anyone => true,
            Self::SameFaction(f) => listener_faction == f,
            Self::HostileTo(target) => {
                faction_relation(registry, deltas, listener_faction, target) == Relation::Hostile
            }
            Self::GlobalFaction(f) => listener_faction == f,
        }
    }

    fn is_global(self) -> bool {
        matches!(self, Self::GlobalFaction(_))
    }
}

/// Resolve who hears an event of `kind`.
pub fn audience_for(kind: &WorldEventKind) -> Audience {
    match kind {
        WorldEventKind::Gunshot { .. }
        | WorldEventKind::Explosion { .. }
        | WorldEventKind::Chatter { .. }
        | WorldEventKind::ModExtension { .. } => Audience::Anyone,
        WorldEventKind::AllyDown { faction, .. } | WorldEventKind::CorpseSpotted { faction } => {
            Audience::SameFaction(*faction)
        }
        WorldEventKind::EnemySighted { target_faction, .. } => Audience::HostileTo(*target_faction),
        WorldEventKind::BaseFlip { new_owner, .. } => Audience::GlobalFaction(*new_owner),
        WorldEventKind::PortalUsed { faction, .. } => Audience::SameFaction(*faction),
        WorldEventKind::PlayerSighted { .. } => Audience::Anyone, // hostile-to-players: TBD
        // Interaction events flow through the bus so the PDA log
        // can subscribe, but they don't have a blackboard
        // audience — the dispatch is a no-op for nearby squads.
        WorldEventKind::InteractionStarted { .. } | WorldEventKind::InteractionEnded { .. } => {
            Audience::Anyone
        }
    }
}

/// Translate a delivered event into a blackboard write on the
/// listener's group. Only kinds with a corresponding blackboard key
/// produce writes; the rest are pass-through (the event still
/// dispatched, just no blackboard side-effect).
fn apply_to_blackboard(
    event: &WorldEvent,
    bb: &mut SquadBlackboards,
    group_id: u64,
    now: u64,
    relevance: f32,
) {
    // Per-kind TTLs roughly match the umbrella's blackboard plan.
    // Relevance scales TTL slightly so faraway events fade faster
    // than nearby ones (linear with distance).
    match &event.kind {
        WorldEventKind::Gunshot { .. } => {
            let ttl = (50.0 * relevance.max(0.2)) as u32;
            bb.write(
                group_id,
                BlackboardKey::HeardGunshot,
                BlackboardValue::Position(event.position),
                now,
                ttl,
            );
        }
        WorldEventKind::AllyDown { id, .. } => {
            bb.write(
                group_id,
                BlackboardKey::DownedAlly { id: *id },
                BlackboardValue::Position(event.position),
                now,
                600, // ~30s at 20 Hz
            );
        }
        WorldEventKind::EnemySighted {
            target_id,
            target_faction: _,
        } => {
            // Dedupe: if this group already has a fresh
            // `LastKnownEnemyId` pointing at the same target, both
            // writes would be no-ops (same TTL 200, same value). At
            // dense pop the same hostile pair gets broadcast every
            // tick that anyone re-acquires aggro on it — short-
            // circuit here so the HashMap writes (and the
            // `RelationDeltas` String-allocating lookups they
            // trigger upstream) don't burn the tick budget. Cost:
            // one HashMap lookup; saves two HashMap inserts.
            if let Some(group) = bb.get(group_id) {
                if let Some(existing) = group.get(&BlackboardKey::LastKnownEnemyId) {
                    if existing.value == BlackboardValue::NpcRef(*target_id)
                        && existing.is_fresh(now)
                    {
                        return;
                    }
                }
            }
            bb.write(
                group_id,
                BlackboardKey::LastKnownEnemyId,
                BlackboardValue::NpcRef(*target_id),
                now,
                200,
            );
            bb.write(
                group_id,
                BlackboardKey::LastKnownEnemyPos,
                BlackboardValue::Position(event.position),
                now,
                200,
            );
        }
        // Other kinds: bus dispatched (audience matched), but no
        // blackboard key defined yet. Future PRs add the missing
        // keys + writers.
        _ => {}
    }
}

/// Cell size (meters) for the per-tick group spatial bucket built
/// by [`drain_world_events`]. Picked to match the largest audible
/// radius in [`audible_radius_m`] (200 m for `EnemySighted` /
/// `BaseFlip`-derived broadcasts) so a 3×3-cell scan around the
/// event point reaches every group inside the audible disc. Larger
/// radii fall back to the per-pair distance check at the bottom
/// of the inner loop — no semantics are lost, just (rare) extra
/// iteration. Smaller cells would shrink per-event candidate sets
/// further but inflate the per-tick HashMap rebuild cost; 200 m
/// was the sweet spot empirically at 720-NPC density.
const EVENT_BUCKET_CELL_M: f32 = 200.0;

fn event_bucket_cell(pos: [f32; 3]) -> (i32, i32) {
    let cx = (pos[0] / EVENT_BUCKET_CELL_M).floor() as i32;
    let cz = (pos[2] / EVENT_BUCKET_CELL_M).floor() as i32;
    (cx, cz)
}

/// Per-tick drain. Runs early in the NPC tick chain so any blackboard
/// reads later in the same tick see freshly-delivered events.
#[allow(clippy::too_many_arguments)]
pub fn drain_world_events(
    clock: Res<SimClock>,
    registry: Res<crate::faction::registry::FactionRegistry>,
    deltas: Res<crate::faction::registry::RelationDeltas>,
    active_regions: Res<crate::resources::ActiveRegions>,
    mut queue: ResMut<WorldEventQueue>,
    mut blackboards: ResMut<SquadBlackboards>,
    index: Res<NpcPositionIndex>,
    npcs: Query<(&Npc, &InFaction, Option<&Group>)>,
) {
    let _diag_t = crate::systems::SysTimer::new("drain_world_events");
    let prof_t = std::time::Instant::now();
    let now = clock.tick;
    let events = queue.take_active(now);
    // TEMP perf instrumentation — stash this tick's event count
    // into the thread-local profile slots so the bridge can read it.
    crate::systems::record_event_count(events.len());
    if events.is_empty() {
        crate::systems::record_perception_slot(
            crate::systems::prof_slots::DRAIN_EVENTS,
            prof_t.elapsed(),
        );
        return;
    }

    // Pre-bin groups by region so per-event iteration only touches
    // groups relevant to that event's region (the dominant case
    // — only `BaseFlip` is global). Without this we walked every
    // group in every region per event, which at full population
    // (3.6k NPCs / ~900 groups across 4 regions) cost ~14ms/tick
    // even when none of the recipients could react (most groups
    // were in offline regions whose consumers — `goal_arbitration`,
    // `squad_planner` — short-circuit on the active-region filter
    // anyway). Per the "every NPC acts as its own player" principle:
    // events delivered to offline-region groups can't drive
    // observable behavior until that region comes online, so we
    // skip writes there entirely. Blackboard entries are derived
    // state and re-built on tier-transition.
    // Spatial-bin groups by `(region, cell_x, cell_z)` with
    // `cell_size = EVENT_BUCKET_CELL_M` (matches the largest audible
    // radius). Each event then iterates only the 3×3 cells around
    // its position — for `EnemySighted` (radius 200 m) that's at
    // most ~10 candidate groups instead of the whole region's ~200
    // groups, which is the dominant cost at dense pop. Global
    // audiences (`BaseFlip`) bypass the spatial filter via the
    // `region_iter` branch below.
    type GroupEntry = (u64, crate::faction::registry::FactionId, [f32; 3]);
    let mut groups_by_cell: HashMap<(RegionId, i32, i32), Vec<GroupEntry>> = HashMap::new();
    let mut group_faction: HashMap<u64, crate::faction::registry::FactionId> =
        HashMap::with_capacity(16);
    for (_npc, faction, group) in npcs.iter() {
        if let Some(g) = group {
            group_faction.entry(g.id).or_insert(faction.0);
        }
    }
    for (group_id, centroid) in index.group_centroids.iter() {
        // Skip groups whose region is offline. Consumers don't read
        // those blackboards anyway, and including them inflates the
        // inner loop by ~4× at full population (1 active vs 4 total
        // regions).
        if !active_regions.is_active(centroid.region) {
            continue;
        }
        let Some(&fac) = group_faction.get(group_id) else {
            continue;
        };
        let cell = event_bucket_cell(centroid.pos);
        groups_by_cell
            .entry((centroid.region, cell.0, cell.1))
            .or_default()
            .push((*group_id, fac, centroid.pos));
    }

    // Pre-compute the full hostility matrix for this tick. Without
    // this, every `aud.passes(...)` call routes to `faction_relation`,
    // which calls `canonical_pair_names` and *allocates two Strings*
    // per call to hash into `RelationDeltas.by_pair`. At dense pop
    // (720 NPCs × ~1000 events/tick) we were doing 4 k+ String
    // allocations per tick just to know hostility — and that was
    // already *after* hoisting from the inner group loop. The matrix
    // is `O(F²)` to build (≤ 16 cells today), `O(1)` to query.
    let n_factions = registry.count();
    let mut hostility: Vec<bool> = vec![false; n_factions * n_factions];
    for a in 0..n_factions {
        for b in 0..n_factions {
            #[allow(clippy::cast_possible_truncation)]
            let rel = crate::faction::registry::faction_relation(
                &registry,
                &deltas,
                crate::faction::registry::FactionId(a as u32),
                crate::faction::registry::FactionId(b as u32),
            );
            hostility[a * n_factions + b] = rel == Relation::Hostile;
        }
    }
    let is_hostile =
        |a: crate::faction::registry::FactionId, b: crate::faction::registry::FactionId| -> bool {
            let ai = a.0 as usize;
            let bi = b.0 as usize;
            if ai < n_factions && bi < n_factions {
                hostility[ai * n_factions + bi]
            } else {
                false
            }
        };

    // Per-event audience cache, derived from the hostility matrix.
    // Cheap (≤ 4 factions, O(1) lookup per faction) so we can rebuild
    // per event without touching the registry.
    let mut audience_cache: Vec<crate::faction::registry::FactionId> = Vec::with_capacity(8);
    let all_faction_ids: Vec<crate::faction::registry::FactionId> = (0..n_factions as u32)
        .map(crate::faction::registry::FactionId)
        .collect();
    for event in events {
        let aud = audience_for(&event.kind);
        let radius = audible_radius_m(&event.kind);
        let radius_sq = if radius.is_finite() {
            radius * radius
        } else {
            f32::INFINITY
        };

        // Re-fill audience_cache for this event using the precomputed
        // matrix — no `RelationDeltas` lookup, no String alloc.
        audience_cache.clear();
        for fid in &all_faction_ids {
            let passes = match aud {
                Audience::Anyone => true,
                Audience::SameFaction(f) | Audience::GlobalFaction(f) => *fid == f,
                Audience::HostileTo(target) => is_hostile(*fid, target),
            };
            if passes {
                audience_cache.push(*fid);
            }
        }

        // Build the list of candidate cells to walk. For global
        // audiences (BaseFlip), every active-region cell is a
        // candidate. For local audiences, only the 3×3 cells around
        // the event position — at `EVENT_BUCKET_CELL_M = 200 m` per
        // cell, that covers any radius ≤ 200 m exactly, and any
        // larger radius via the per-pair distance check below.
        let mut candidates: Vec<&Vec<_>> = Vec::with_capacity(16);
        if aud.is_global() {
            candidates.extend(groups_by_cell.values());
        } else {
            let ecell = event_bucket_cell(event.position);
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if let Some(list) =
                        groups_by_cell.get(&(event.region, ecell.0 + dx, ecell.1 + dz))
                    {
                        candidates.push(list);
                    }
                }
            }
            if candidates.is_empty() {
                continue;
            }
        }

        for list in candidates {
            for (group_id, listener_faction, pos) in list {
                if !audience_cache.contains(listener_faction) {
                    continue;
                }
                let dx = pos[0] - event.position[0];
                let dz = pos[2] - event.position[2];
                let dist_sq = dx * dx + dz * dz;
                if !aud.is_global() && dist_sq > radius_sq {
                    continue;
                }
                // Linear relevance falloff: 1.0 at the event point,
                // 0.0 at the radius edge. Global events are full
                // relevance everywhere.
                let relevance = if aud.is_global() {
                    1.0
                } else {
                    let dist = dist_sq.sqrt();
                    (1.0 - (dist / radius)).max(0.0)
                };
                apply_to_blackboard(&event, &mut blackboards, *group_id, now, relevance);
            }
        }
    }
    crate::systems::record_perception_slot(
        crate::systems::prof_slots::DRAIN_EVENTS,
        prof_t.elapsed(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faction::registry::{load_default, FactionId, FactionRegistry, RelationDeltas};

    fn fixture() -> (FactionRegistry, RelationDeltas, FactionId, FactionId) {
        let reg = load_default();
        let deltas = RelationDeltas::default();
        let coalition = reg.id_of("coalition").unwrap();
        let looters = reg.id_of("looters").unwrap();
        (reg, deltas, coalition, looters)
    }

    #[test]
    fn push_assigns_unique_ids() {
        let mut q = WorldEventQueue::default();
        let a = q.push(
            WorldEventKind::Gunshot {
                caliber_class: CaliberClass::Intermediate,
            },
            [0.0, 0.0, 0.0],
            1,
            100,
            5,
        );
        let b = q.push(
            WorldEventKind::Gunshot {
                caliber_class: CaliberClass::Intermediate,
            },
            [10.0, 0.0, 0.0],
            1,
            100,
            5,
        );
        assert_ne!(a, b);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn take_active_drops_expired() {
        let mut q = WorldEventQueue::default();
        // Created at tick 5, TTL 5 -> expires at tick 10.
        q.push(
            WorldEventKind::Gunshot {
                caliber_class: CaliberClass::Pistol,
            },
            [0.0, 0.0, 0.0],
            1,
            5,
            5,
        );
        // Now = 11 -> past TTL.
        let active = q.take_active(11);
        assert!(active.is_empty(), "expired events should be dropped");
        assert!(q.is_empty(), "queue is drained either way");
    }

    #[test]
    fn take_active_keeps_fresh() {
        let mut q = WorldEventQueue::default();
        q.push(
            WorldEventKind::Gunshot {
                caliber_class: CaliberClass::Pistol,
            },
            [0.0, 0.0, 0.0],
            1,
            5,
            5,
        );
        // Now = 8 -> within TTL.
        let active = q.take_active(8);
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn audience_filters_correctly() {
        let (reg, deltas, coalition, looters) = fixture();

        // Same-faction event (AllyDown) only passes to same faction.
        let aud = audience_for(&WorldEventKind::AllyDown {
            id: NpcId(1),
            faction: coalition,
        });
        assert!(aud.passes(coalition, &reg, &deltas));
        assert!(!aud.passes(looters, &reg, &deltas));

        // EnemySighted reaches factions hostile to the target's faction.
        let aud = audience_for(&WorldEventKind::EnemySighted {
            target_id: NpcId(2),
            target_faction: looters,
        });
        // Coalition is hostile to Looters per the canonical relation table.
        assert!(aud.passes(coalition, &reg, &deltas));
        // Looters hearing "enemy sighted at one of us" makes no sense.
        assert!(!aud.passes(looters, &reg, &deltas));

        // BaseFlip is global within the new owner's faction.
        let aud = audience_for(&WorldEventKind::BaseFlip {
            new_owner: coalition,
            old_owner: None,
        });
        assert!(aud.is_global());
        assert!(aud.passes(coalition, &reg, &deltas));
        assert!(!aud.passes(looters, &reg, &deltas));

        // Gunshot is everyone.
        let aud = audience_for(&WorldEventKind::Gunshot {
            caliber_class: CaliberClass::Intermediate,
        });
        assert!(aud.passes(coalition, &reg, &deltas));
        assert!(aud.passes(looters, &reg, &deltas));
    }

    #[test]
    fn audible_radius_scales_with_caliber() {
        let l = audible_radius_m(&WorldEventKind::Gunshot {
            caliber_class: CaliberClass::Pistol,
        });
        let m = audible_radius_m(&WorldEventKind::Gunshot {
            caliber_class: CaliberClass::Intermediate,
        });
        let h = audible_radius_m(&WorldEventKind::Gunshot {
            caliber_class: CaliberClass::FullPowerRifle,
        });
        assert!(l < m);
        assert!(m < h);
    }

    #[test]
    fn base_flip_is_infinite_radius() {
        let (_reg, _deltas, coalition, _looters) = fixture();
        let r = audible_radius_m(&WorldEventKind::BaseFlip {
            new_owner: coalition,
            old_owner: None,
        });
        assert!(r.is_infinite());
    }
}
