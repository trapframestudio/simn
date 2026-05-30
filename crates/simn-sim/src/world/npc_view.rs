//! NPC/base view queries, chronicle, PDA log, region control,
//! and goal-tag helpers on `Sim`.

use crate::chronicle::{ChronicleSummary, LifeChronicle, LifeRecord};
use crate::components::*;
use crate::faction::Relation;
use crate::region::RegionId;
use crate::resources::*;

use super::{BaseView, NpcView, Sim};

/// Short string tag for an NPC's current FSM state, for dev overlays.
/// Aggro overrides whatever the goal FSM says, since the combat
/// branch in `tick_npc_goals` effectively runs a pursue-to-engage
/// loop on top. For grouped NPCs we surface the squad objective
/// (guard_post / patrol / rest / explore / …) — that's the actual
/// behavior the player sees — and fall back to the per-NPC FSM
/// only for solos.
fn goal_tag(
    goal: &NpcGoal,
    active: Option<&crate::components::ActiveGoal>,
    aggroed: bool,
    group_id: Option<u64>,
    objectives: &crate::resources::SquadObjectives,
) -> &'static str {
    // Personality drives and survival are surfaced from ActiveGoal
    // first — they preempt SquadFollowObjective in the arbiter, so
    // showing the squad objective ("rest") for an NPC that's actually
    // off looting a corpse would mislead debug viewers.
    if let Some(tag) = active.and_then(active_goal_override_tag) {
        return tag;
    }
    if aggroed {
        return "pursue";
    }
    if let Some(gid) = group_id {
        if let Some(state) = objectives.by_group.get(&gid) {
            return squad_objective_tag(&state.objective);
        }
    }
    match goal {
        NpcGoal::Idle { .. } => "idle",
        NpcGoal::MoveTo { .. } => "move",
        NpcGoal::RestAt { .. } => "rest",
    }
}

/// Variant of [`goal_tag`] that reads from a pre-extracted
/// `group_id → SquadObjective` snapshot instead of the full
/// `SquadObjectives` resource. Lets the view-collection path skip a
/// resource-shaped clone on the 20 Hz dummy-sync hot path.
fn derive_goal_tag(
    goal: &NpcGoal,
    active: Option<&crate::components::ActiveGoal>,
    aggroed: bool,
    group_id: u64,
    objectives: &std::collections::HashMap<u64, crate::resources::SquadObjective>,
) -> &'static str {
    if let Some(tag) = active.and_then(active_goal_override_tag) {
        return tag;
    }
    if aggroed {
        return "pursue";
    }
    if group_id != 0 {
        if let Some(o) = objectives.get(&group_id) {
            return squad_objective_tag(o);
        }
    }
    match goal {
        NpcGoal::Idle { .. } => "idle",
        NpcGoal::MoveTo { .. } => "move",
        NpcGoal::RestAt { .. } => "rest",
    }
}

/// Short tag for the arbiter source that produced the active goal.
/// Surfaced in the debug HUD so we can tell at a glance whether an
/// NPC is on-task (squad_obj) vs reacting to a distraction
/// (blackboard_*) vs being driven by personality.
fn goal_source_tag(source: crate::components::GoalSource) -> &'static str {
    use crate::components::GoalSource::*;
    match source {
        ScriptedClaim => "scripted",
        IndividualSurvival => "survival",
        SquadAggro => "aggro_squad",
        IndividualAggro => "aggro_solo",
        BlackboardUrgency => "blackboard",
        SquadObjective => "squad_obj",
        PersonalityBias => "personality",
        Idle => "idle",
    }
}

/// Returns a tag for the personality-driven / survival
/// `ActiveGoal.kind`s that take precedence over the squad objective.
/// `None` means "no override — let the legacy chain pick a tag from
/// squad objective or NpcGoal."
fn active_goal_override_tag(active: &crate::components::ActiveGoal) -> Option<&'static str> {
    use crate::components::GoalKind;
    match active.kind {
        GoalKind::Hunt { .. } => Some("hunt"),
        GoalKind::Socialize { .. } => Some("socialize"),
        GoalKind::Loot { .. } => Some("loot"),
        GoalKind::Bloodsport => Some("bloodsport"),
        GoalKind::SeekMedical { .. } => Some("seek_medical"),
        GoalKind::InvestigateAt { .. } => Some("investigate_at"),
        GoalKind::RegroupOnAlly { .. } => Some("regroup_on_ally"),
        GoalKind::PursueTarget { .. } | GoalKind::SquadFollowObjective | GoalKind::SoloIdleFsm => {
            None
        }
    }
}

fn squad_objective_tag(o: &crate::resources::SquadObjective) -> &'static str {
    use crate::resources::SquadObjective::*;
    match o {
        Patrol { .. } => "patrol",
        Guard {
            post_key: Some(_), ..
        } => "guard_post",
        Guard { .. } => "guard",
        Rest { .. } => "rest",
        Investigate { .. } => "investigate",
        Explore { .. } => "explore",
        Relieve { .. } => "relieve",
        Wander { .. } => "wander",
        Regroup { .. } => "regroup",
    }
}

fn stance_tag(s: &crate::components::CombatStance) -> &'static str {
    use crate::components::CombatStance::*;
    match s {
        Approaching => "approaching",
        InCover { .. } => "in_cover",
        Firing { .. } => "firing",
        Suppressed { .. } => "suppressed",
        Flanking => "flanking",
        Retreating => "retreating",
    }
}

fn role_tag(r: &crate::components::CombatRole) -> &'static str {
    use crate::components::CombatRole::*;
    match r {
        Pointman => "pointman",
        Support => "support",
        Flanker => "flanker",
        Medic => "medic",
    }
}

/// Map the (NpcGoal, DwellState) pair to a short pose tag for
/// renderers. Only RestAt + present `DwellState` produces a pose;
/// active movement/idle states return `None` so the renderer falls
/// back to the default locomotion blend.
fn dwell_pose_tag(
    goal: &crate::components::NpcGoal,
    dwell: Option<&crate::components::DwellState>,
) -> Option<&'static str> {
    use crate::components::DwellPose::*;
    use crate::components::NpcGoal;
    let NpcGoal::RestAt { .. } = goal else {
        return None;
    };
    Some(match dwell?.pose {
        Standing => "standing",
        Sitting => "sitting",
        Crouching => "crouching",
    })
}

impl Sim {
    /// Per-region control state. `None` if the region wasn't seeded
    /// or doesn't exist in the graph.
    pub fn region_control(&self, region: RegionId) -> Option<&RegionControlState> {
        self.world
            .resource::<RegionControl>()
            .by_region
            .get(&region)
    }

    /// Convenience: look up region control by name.
    pub fn region_control_by_name(&self, name: &str) -> Option<&RegionControlState> {
        let id = self.regions().id_for_name(name)?;
        self.region_control(id)
    }

    /// Full region-control snapshot, used by the threaded-sim view
    /// builder to fold per-region faction state into `SimView` so
    /// the debug overlay reads it lock-free instead of round-
    /// tripping through `inspect` once per frame.
    pub fn region_controls(&self) -> &std::collections::HashMap<RegionId, RegionControlState> {
        &self.world.resource::<RegionControl>().by_region
    }

    /// All bases currently in `region`. Order is unspecified.
    pub fn bases_in_region(&mut self, region: RegionId) -> Vec<BaseView> {
        let mut out = Vec::new();
        let mut q = self
            .world
            .query::<(&Base, &InFaction, &Position, &InRegion, &Health)>();
        for (b, f, pos, r, h) in q.iter(&self.world) {
            if r.0 == region {
                out.push(BaseView {
                    region: r.0,
                    pos: pos.0,
                    kind: b.kind,
                    faction: f.0,
                    health: *h,
                });
            }
        }
        out
    }

    /// Iterate every base in the world.
    pub fn each_base<F: FnMut(BaseView)>(&mut self, mut f: F) {
        let mut q = self
            .world
            .query::<(&Base, &InFaction, &Position, &InRegion, &Health)>();
        for (b, fac, pos, r, h) in q.iter(&self.world) {
            f(BaseView {
                region: r.0,
                pos: pos.0,
                kind: b.kind,
                faction: fac.0,
                health: *h,
            });
        }
    }

    /// Look up the relation between two factions, applying registry
    /// inheritance + runtime drift. The accepted spectrum is
    /// `Hostile / Cold / Neutral / Warm / Friendly`.
    pub fn faction_relation(
        &self,
        a: crate::faction::registry::FactionId,
        b: crate::faction::registry::FactionId,
    ) -> Relation {
        crate::faction::registry::faction_relation(
            self.world
                .resource::<crate::faction::registry::FactionRegistry>(),
            self.world
                .resource::<crate::faction::registry::RelationDeltas>(),
            a,
            b,
        )
    }

    // ---------- NPCs ----------

    /// All live NPCs currently in `region`. Order is unspecified.
    ///
    /// **Marshaling cost.** Each `NpcView` clones the NPC's `Wounds`
    /// vec and `name` String — fine for a one-off inspector pull, but
    /// hot when called every tick. For the 20Hz dummy-sync path the
    /// renderer goes through [`Self::npcs_near`], which adds a
    /// server-side distance filter so only NPCs within the player's
    /// draw range pay the per-NPC clone + bridge marshaling cost.
    pub fn npcs_in_region(&mut self, region: RegionId) -> Vec<NpcView> {
        self.collect_npcs_in_region(region, None)
    }

    /// Like [`Self::npcs_in_region`] but server-side distance-filtered
    /// against `player_pos`. Skips per-NPC `NpcView` construction
    /// (and the wounds/name clones inside) for NPCs outside
    /// `max_dist_m` in the XZ plane. With ~800 NPCs in a region and
    /// a 100 m draw radius, this typically drops marshaling work by
    /// 10–20× versus the unfiltered call.
    pub fn npcs_near(
        &mut self,
        region: RegionId,
        player_pos: [f32; 3],
        max_dist_m: f32,
    ) -> Vec<NpcView> {
        let max_sq = max_dist_m * max_dist_m;
        self.collect_npcs_in_region(region, Some((player_pos, max_sq)))
    }

    fn collect_npcs_in_region(
        &mut self,
        region: RegionId,
        filter: Option<([f32; 3], f32)>,
    ) -> Vec<NpcView> {
        let mut out = Vec::new();
        // `query` takes `&mut World` so we can't keep an immutable
        // `&SquadObjectives` ref live across the loop. `goal_tag`
        // needs the resource only to read each NPC's group state, so
        // we pull the small `by_group` map out by reference via a
        // scoped borrow before starting the query, then re-take the
        // resource view inside the loop via a re-resource read.
        //
        // The previous implementation cloned the entire `SquadObjectives`
        // HashMap up front; that was unnecessary work every call (the
        // 20Hz dummy sync was hitting it hard). The current shape
        // takes a snapshot of just the keys+values we need: small
        // `(group_id, SquadObjective)` map keyed by `u64`. Cheaper
        // than the full resource clone and matches what `goal_tag`
        // reads.
        let objectives_snapshot: std::collections::HashMap<u64, crate::resources::SquadObjective> =
            self.world
                .resource::<crate::resources::SquadObjectives>()
                .by_group
                .iter()
                .map(|(k, v)| (*k, v.objective.clone()))
                .collect();
        let mut q = self.world.query::<(
            &Npc,
            &InFaction,
            &InRegion,
            &Position,
            &Rotation,
            &Health,
            Option<&BodyParts>,
            Option<&Wounds>,
            &NpcGoal,
            Option<&Group>,
            Option<&Aggro>,
            Option<&crate::components::NpcCharacter>,
            (
                Option<&crate::components::CombatStance>,
                Option<&crate::components::CombatRole>,
                Option<&crate::components::DwellState>,
                Option<&crate::components::ActiveGoal>,
            ),
        )>();
        for (
            n,
            f,
            r,
            pos,
            rot,
            h,
            bp,
            wounds,
            goal,
            group,
            aggro,
            character,
            (stance, role, dwell, active),
        ) in q.iter(&self.world)
        {
            if r.0 != region {
                continue;
            }
            // Server-side distance gate. Cheap squared-distance check
            // before the per-NPC clones land in `NpcView`.
            if let Some((origin, max_sq)) = filter {
                let dx = pos.0[0] - origin[0];
                let dz = pos.0[2] - origin[2];
                if dx * dx + dz * dz > max_sq {
                    continue;
                }
            }
            let group_id = group.map(|g| g.id).unwrap_or(0);
            let goal_tag = derive_goal_tag(
                goal,
                active,
                aggro.is_some(),
                group_id,
                &objectives_snapshot,
            );
            out.push(NpcView {
                id: n.id,
                faction: f.0,
                region: r.0,
                pos: pos.0,
                yaw: rot.0,
                health: *h,
                body_parts: bp.copied(),
                wounds: wounds.map(|w| w.0.clone()).unwrap_or_default(),
                goal: goal_tag,
                group_id,
                aggro_target: aggro.map(|a| a.target.0).unwrap_or(0),
                name: character.map(|c| c.name.clone()).unwrap_or_default(),
                nationality: character.map(|c| c.nationality),
                rank: character.map(|c| c.rank),
                combat_stance: stance.map(stance_tag),
                combat_role: role.map(role_tag),
                dwell_pose: dwell_pose_tag(goal, dwell),
                goal_source: active.map(|a| goal_source_tag(a.source)).unwrap_or("?"),
                goal_priority: active.map(|a| a.priority).unwrap_or(0),
            });
        }
        out
    }

    /// Iterate every live NPC in the world. Order is unspecified.
    pub fn each_npc<F: FnMut(NpcView)>(&mut self, mut f: F) {
        let objectives = self
            .world
            .resource::<crate::resources::SquadObjectives>()
            .by_group
            .clone();
        let objectives_res = crate::resources::SquadObjectives {
            by_group: objectives,
        };
        let mut q = self.world.query::<(
            &Npc,
            &InFaction,
            &InRegion,
            &Position,
            &Rotation,
            &Health,
            Option<&BodyParts>,
            Option<&Wounds>,
            &NpcGoal,
            Option<&Group>,
            Option<&Aggro>,
            Option<&crate::components::NpcCharacter>,
            (
                Option<&crate::components::CombatStance>,
                Option<&crate::components::CombatRole>,
                Option<&crate::components::DwellState>,
                Option<&crate::components::ActiveGoal>,
            ),
        )>();
        for (
            n,
            fac,
            r,
            pos,
            rot,
            h,
            bp,
            wounds,
            goal,
            group,
            aggro,
            character,
            (stance, role, dwell, active),
        ) in q.iter(&self.world)
        {
            f(NpcView {
                id: n.id,
                faction: fac.0,
                region: r.0,
                pos: pos.0,
                yaw: rot.0,
                health: *h,
                body_parts: bp.copied(),
                wounds: wounds.map(|w| w.0.clone()).unwrap_or_default(),
                goal: goal_tag(
                    goal,
                    active,
                    aggro.is_some(),
                    group.map(|g| g.id),
                    &objectives_res,
                ),
                group_id: group.map(|g| g.id).unwrap_or(0),
                aggro_target: aggro.map(|a| a.target.0).unwrap_or(0),
                name: character.map(|c| c.name.clone()).unwrap_or_default(),
                nationality: character.map(|c| c.nationality),
                rank: character.map(|c| c.rank),
                combat_stance: stance.map(stance_tag),
                combat_role: role.map(role_tag),
                dwell_pose: dwell_pose_tag(goal, dwell),
                goal_source: active.map(|a| goal_source_tag(a.source)).unwrap_or("?"),
                goal_priority: active.map(|a| a.priority).unwrap_or(0),
            });
        }
    }

    /// Aggregate stats over the chronicle (alive + ever). Reads
    /// the cached summary (O(1) + a clone of the small per-faction
    /// table) rather than scanning every record.
    pub fn chronicle_summary(&self) -> ChronicleSummary {
        self.world.resource::<LifeChronicle>().summary().clone()
    }

    /// One specific NPC's permanent record, alive or dead.
    pub fn chronicle_get(&self, id: NpcId) -> Option<&LifeRecord> {
        self.world.resource::<LifeChronicle>().get(id)
    }

    /// Most recent deaths first, capped at `limit`.
    pub fn recent_deaths(&self, limit: usize) -> Vec<&LifeRecord> {
        self.world.resource::<LifeChronicle>().recent_deaths(limit)
    }

    /// PDA event log — events with `seq > since_seq`, ordered oldest
    /// first. The client tracks its own last-seen `seq` bookmark; on
    /// first poll it should pass `Sim::pda_log_high_water()` (or 0
    /// for "show me everything since the world booted").
    pub fn recent_pda_events_since(&self, since_seq: u64) -> Vec<crate::pda_log::PdaLogEntry> {
        self.world
            .resource::<crate::pda_log::PdaEventLog>()
            .since(since_seq)
            .cloned()
            .collect()
    }

    /// Current highest `seq` in the PDA log. Clients seed their
    /// `last_seen` bookmark from this after a snapshot load so they
    /// don't re-toast events that landed before the player joined.
    pub fn pda_log_high_water(&self) -> u64 {
        self.world
            .resource::<crate::pda_log::PdaEventLog>()
            .high_water()
    }

    /// View-snapshot helper used by `worker::view::build_sim_view`
    /// to fold the PDA log into the published `SimView` each tick.
    /// Returning `(entries, high_water)` together saves an extra
    /// resource lookup. Clone is bounded (≤256 entries, ~16 KB).
    /// Without this, `recent_pda_events_since` on the bridge side
    /// would block on `worker.inspect` 4× per second from the PDA
    /// toast poller, costing ~5-10 FPS on the main thread.
    pub fn pda_log_view_snapshot(&self) -> (Vec<crate::pda_log::PdaLogEntry>, u64) {
        let log = self.world.resource::<crate::pda_log::PdaEventLog>();
        (log.all().cloned().collect(), log.high_water())
    }

    /// View-snapshot helper: collect every NPC within
    /// `ONLINE_NEAR_RADIUS_M` of any connected player in an active
    /// region, keyed by region. The bridge's `npcs_near` reads
    /// this off the published `SimView` so the per-frame rendering
    /// path is lock-free.
    ///
    /// Distance prefilter (500 m): caching every NPC in every
    /// active region would clone ~800 `NpcView`s per tick
    /// (~80-100 KB). Most are far outside the player's draw
    /// distance — only ~50 are within `NPC_DRAW_DISTANCE_M`.
    /// Pre-filtering on the worker side cuts the per-tick clone
    /// cost ~16× and the main-thread filter/marshal cost the same.
    /// NPCs in active regions but beyond 500 m still simulate
    /// (the active-region tier filter only excludes offline
    /// regions) — they're just not exposed to the renderer.
    ///
    /// Multi-player union: when multiple players are in the same
    /// region we keep NPCs within range of *any* of them, dedup
    /// via a `HashSet<NpcId>`.
    pub fn active_region_npc_views(&mut self) -> std::collections::HashMap<RegionId, Vec<NpcView>> {
        let active: std::collections::HashSet<RegionId> = self
            .world
            .resource::<crate::resources::ActiveRegions>()
            .regions
            .clone();
        let mut by_region: std::collections::HashMap<RegionId, Vec<NpcView>> =
            std::collections::HashMap::new();
        if active.is_empty() {
            return by_region;
        }
        // Collect each connected player's (region, pos) into owned
        // data so subsequent `&mut self` npcs_near calls don't
        // conflict with the player-view borrow.
        let player_anchors: Vec<(RegionId, [f32; 3])> = self
            .connected_player_ids()
            .into_iter()
            .filter_map(|sid| {
                self.player_view(sid)
                    .filter(|pv| active.contains(&pv.region))
                    .map(|pv| (pv.region, pv.pos))
            })
            .collect();
        for (region, pos) in player_anchors {
            let mut seen: std::collections::HashSet<crate::components::NpcId> = by_region
                .get(&region)
                .map(|v| v.iter().map(|n| n.id).collect())
                .unwrap_or_default();
            for v in self.npcs_near(region, pos, crate::world::ONLINE_NEAR_RADIUS_M) {
                if seen.insert(v.id) {
                    by_region.entry(region).or_default().push(v);
                }
            }
        }
        by_region
    }

    /// Per-region per-faction desired live count. Tunable later.
    pub fn population_targets(&self) -> &PopulationTargets {
        self.world.resource::<PopulationTargets>()
    }
}
