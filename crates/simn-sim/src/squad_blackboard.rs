//! Per-`Group` shared-memory store for squad coordination.
//!
//! Today, the only thing squadmates share beyond `Group.id` is an
//! inherited `Aggro` target via a hard-coded special case in
//! `npc_aggro`. The blackboard generalizes that pattern: any
//! squadmate (or any AI system) can write a *fact* about the squad's
//! world (heard a gunshot, ally went down, taking fire from a
//! direction, designated rally point), and any reader (squad
//! planner, goal arbitration, tactical AI) can consult it.
//!
//! ## Tick discipline
//!
//! Entries carry a `ttl_ticks`; the [`sweep_squad_blackboards`] system
//! runs early in the NPC tick chain and drops any entries past their
//! TTL. Empty group blackboards are also pruned (group dissolved or
//! all entries expired). Cache writes happen during `npc_aggro` /
//! `npc_combat` / `npc_death_check` / world event bus delivery; reads
//! happen later in the same tick by the planner / goal resolver.
//!
//! ## Persistence
//!
//! Blackboards are derived state — they're rebuilt from world state +
//! recent events on tier transition (decided 2026-05-05; matches
//! STALKER's design). NOT journaled. This module never serialises
//! anything; reload from snapshot starts every group's blackboard
//! empty.
//!
//! ## Modding
//!
//! [`BlackboardKey`] is a closed enum for engine-level facts (the AI
//! systems shipped in this crate). Mods that want their own keys use
//! the `Custom { mod_id, name }` variant — same write/read path,
//! just keyed on a stable mod id + interned name.
//!
//! ## Phase 1
//!
//! Pure primitive. One writer hooked up (`npc_aggro` writes
//! `LastKnownEnemyId` + `LastKnownEnemyPos` on new aggro acquisition)
//! so the resource has something tangible in it; no readers yet. The
//! upcoming squad planner / goal arbitration / world event bus PRs
//! consume it.
//!
//! See `docs/book/src/planning/squad-blackboard-plan.md`.

use std::borrow::Cow;
use std::collections::HashMap;

use bevy_ecs::prelude::{Res, ResMut, Resource};

use crate::components::NpcId;
use crate::resources::SimClock;

/// Stable identifier for a mod-defined blackboard key. Matches the
/// shape modders use elsewhere in the engine (`mod_id` is the stable
/// short id under which the mod is loaded).
pub type ModId = u32;

/// Typed blackboard keys. Closed enum for engine-level facts so the
/// compiler catches typos and the schema stays reviewable. Mods
/// extend via [`BlackboardKey::Custom`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlackboardKey {
    /// Last known position of the squad's current target. Refreshed
    /// every tick aggro is held; decays with the entry's TTL.
    LastKnownEnemyPos,
    /// `NpcId` of the squad's current target.
    LastKnownEnemyId,
    /// A specific squadmate has gone down. Carries the dead NPC's id
    /// so revive / mourn objectives can target the right body.
    DownedAlly { id: NpcId },
    /// Squad heard a gunshot recently (from event bus delivery).
    HeardGunshot,
    /// Squad is taking incoming fire. Drives a "go to cover"
    /// objective.
    UnderFireAt,
    /// Suppressing fire from a specific octant (0=N, 1=NE, ... 7=NW).
    /// Drives flank / shift maneuvers.
    Suppressed { from_dir: u8 },
    /// `NpcId` of the squad leader.
    LeaderId,
    /// Squad-designated rally point. Members converge here on
    /// regroup / retreat.
    RallyPoint,
    /// This squad is on the way to reinforce another group.
    Reinforcing { target_group: u64 },
    /// Aggregated multi-target threat list for the squad. Built each
    /// tick from members' [`crate::components::RecentAttackers`] by
    /// `sweep_threats`. Drives target switching in
    /// `goal_arbitration` so a squad can concentrate fire on the
    /// highest-scored attacker even when several are doing damage.
    /// Value is [`BlackboardValue::Threats`].
    ThreatList,
    /// Mod-defined extension. `mod_id` namespaces the key so two
    /// mods can use the same `name` without collision.
    Custom {
        mod_id: ModId,
        name: Cow<'static, str>,
    },
}

/// Typed blackboard values. Same closed-enum philosophy as
/// [`BlackboardKey`]: mods that want exotic value types can pack
/// them into the variants below or convert via `Bool`/`Float` /
/// position triples.
#[derive(Clone, Debug, PartialEq)]
pub enum BlackboardValue {
    Position([f32; 3]),
    NpcRef(NpcId),
    GroupRef(u64),
    Tick(u64),
    Float(f32),
    Bool(bool),
    /// Aggregated threat list for [`BlackboardKey::ThreatList`].
    /// Sorted by `score` descending so the top entry is the squad's
    /// preferred target. Empty when no member has recent attackers.
    Threats(Vec<ThreatEntry>),
}

/// One entry in the squad's threat board. Score = aggregated damage
/// × recency falloff × proximity factor; see
/// `docs/book/src/planning/threat-board-plan.md` §5.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThreatEntry {
    pub target_id: NpcId,
    pub score: f32,
    pub last_seen_tick: u64,
}

/// One blackboard fact.
#[derive(Clone, Debug, PartialEq)]
pub struct BlackboardEntry {
    pub value: BlackboardValue,
    pub written_tick: u64,
    /// How long after `written_tick` the entry is considered fresh.
    /// `0` means "this tick only" — sweeps drop it next tick.
    pub ttl_ticks: u32,
}

impl BlackboardEntry {
    /// Whether this entry is still fresh at `now`. `now` <
    /// `written_tick + ttl_ticks` (saturating) → fresh.
    pub fn is_fresh(&self, now: u64) -> bool {
        now < self.written_tick.saturating_add(self.ttl_ticks as u64)
    }
}

/// One squad's blackboard. Wraps a `HashMap<BlackboardKey,
/// BlackboardEntry>` with TTL-aware insertion + eviction helpers.
#[derive(Default, Clone, Debug)]
pub struct GroupBlackboard {
    entries: HashMap<BlackboardKey, BlackboardEntry>,
}

impl GroupBlackboard {
    pub fn get(&self, key: &BlackboardKey) -> Option<&BlackboardEntry> {
        self.entries.get(key)
    }

    pub fn set(&mut self, key: BlackboardKey, entry: BlackboardEntry) {
        self.entries.insert(key, entry);
    }

    /// Convenience setter that builds the entry from value + TTL.
    pub fn write(
        &mut self,
        key: BlackboardKey,
        value: BlackboardValue,
        written_tick: u64,
        ttl_ticks: u32,
    ) {
        self.entries.insert(
            key,
            BlackboardEntry {
                value,
                written_tick,
                ttl_ticks,
            },
        );
    }

    pub fn clear_key(&mut self, key: &BlackboardKey) -> Option<BlackboardEntry> {
        self.entries.remove(key)
    }

    /// Drop entries past their TTL. Returns the number of entries
    /// removed (test / instrumentation).
    pub fn sweep_expired(&mut self, now: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, e| e.is_fresh(now));
        before - self.entries.len()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&BlackboardKey, &BlackboardEntry)> {
        self.entries.iter()
    }
}

/// Per-`Group` blackboard registry. Resource lives in the ECS world.
#[derive(Resource, Default, Clone)]
pub struct SquadBlackboards {
    by_group: HashMap<u64, GroupBlackboard>,
}

impl SquadBlackboards {
    /// Read the blackboard for `group_id`. `None` if the group has
    /// no entries (or was pruned by the sweep).
    pub fn get(&self, group_id: u64) -> Option<&GroupBlackboard> {
        self.by_group.get(&group_id)
    }

    /// Mutable access for writers; lazily creates the group's
    /// blackboard on first write.
    pub fn entry_mut(&mut self, group_id: u64) -> &mut GroupBlackboard {
        self.by_group.entry(group_id).or_default()
    }

    /// Convenience: write a single entry to `group_id`'s blackboard.
    pub fn write(
        &mut self,
        group_id: u64,
        key: BlackboardKey,
        value: BlackboardValue,
        written_tick: u64,
        ttl_ticks: u32,
    ) {
        self.entry_mut(group_id)
            .write(key, value, written_tick, ttl_ticks);
    }

    /// Drop a whole group's blackboard. Use when a squad dissolves
    /// (last member died / left). Sweep also handles this when the
    /// group's entry-set goes empty.
    pub fn drop_group(&mut self, group_id: u64) {
        self.by_group.remove(&group_id);
    }

    /// Per-tick sweep: walk every group, drop expired entries, and
    /// drop now-empty group blackboards. Cheap (O(active groups *
    /// avg entries-per-group); both are small).
    pub fn sweep(&mut self, now: u64) {
        let mut empty_groups: Vec<u64> = Vec::new();
        for (gid, bb) in self.by_group.iter_mut() {
            bb.sweep_expired(now);
            if bb.is_empty() {
                empty_groups.push(*gid);
            }
        }
        for gid in empty_groups {
            self.by_group.remove(&gid);
        }
    }

    /// Total number of groups with at least one entry.
    pub fn group_count(&self) -> usize {
        self.by_group.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_group.is_empty()
    }
}

/// Schedule-side wrapper that evicts expired entries. Runs early in
/// the NPC tick chain (alongside `clear_los_cache`) so writes during
/// the same tick land into a freshly-swept cache.
pub fn sweep_squad_blackboards(clock: Res<SimClock>, mut bb: ResMut<SquadBlackboards>) {
    let t = std::time::Instant::now();
    bb.sweep(clock.tick);
    crate::systems::record_perception_slot(crate::systems::prof_slots::SWEEP_BB, t.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(x: f32, z: f32) -> BlackboardValue {
        BlackboardValue::Position([x, 0.0, z])
    }

    #[test]
    fn write_and_read() {
        let mut bb = SquadBlackboards::default();
        bb.write(
            42,
            BlackboardKey::LastKnownEnemyPos,
            pos(10.0, 20.0),
            100,
            50,
        );
        let entry = bb
            .get(42)
            .and_then(|g| g.get(&BlackboardKey::LastKnownEnemyPos))
            .expect("entry present");
        assert_eq!(entry.value, pos(10.0, 20.0));
        assert_eq!(entry.written_tick, 100);
        assert_eq!(entry.ttl_ticks, 50);
    }

    #[test]
    fn missing_group_returns_none() {
        let bb = SquadBlackboards::default();
        assert!(bb.get(99).is_none());
    }

    #[test]
    fn sweep_drops_expired_entries() {
        let mut bb = SquadBlackboards::default();
        // Written at tick 10, TTL 5 -> expires at tick 15.
        bb.write(1, BlackboardKey::HeardGunshot, pos(0.0, 0.0), 10, 5);
        // Sweep at tick 14: still fresh.
        bb.sweep(14);
        assert!(bb.get(1).is_some());
        // Sweep at tick 15: expired.
        bb.sweep(15);
        assert!(bb.get(1).is_none(), "expired entry should drop the group");
    }

    #[test]
    fn sweep_drops_empty_groups() {
        let mut bb = SquadBlackboards::default();
        bb.write(7, BlackboardKey::HeardGunshot, pos(1.0, 1.0), 0, 1);
        assert_eq!(bb.group_count(), 1);
        // Sweep at tick 1 -> entry expires (1 == written + ttl).
        bb.sweep(1);
        assert_eq!(bb.group_count(), 0);
    }

    #[test]
    fn explicit_drop_group_removes_blackboard() {
        let mut bb = SquadBlackboards::default();
        bb.write(3, BlackboardKey::LeaderId, BlackboardValue::Tick(5), 0, 100);
        assert!(bb.get(3).is_some());
        bb.drop_group(3);
        assert!(bb.get(3).is_none());
    }

    #[test]
    fn entry_is_fresh_uses_saturating_math() {
        // TTL = u32::MAX; written at tick u64::MAX - 10. Should still
        // be considered fresh because of saturating add.
        let entry = BlackboardEntry {
            value: BlackboardValue::Bool(true),
            written_tick: u64::MAX - 10,
            ttl_ticks: u32::MAX,
        };
        // saturating_add caps at u64::MAX; now = u64::MAX is NOT < u64::MAX.
        assert!(!entry.is_fresh(u64::MAX));
        // now = u64::MAX - 11 IS < u64::MAX (saturated).
        assert!(entry.is_fresh(u64::MAX - 11));
    }

    #[test]
    fn custom_keys_are_independent_per_mod() {
        let mut bb = SquadBlackboards::default();
        let key_a = BlackboardKey::Custom {
            mod_id: 1,
            name: Cow::Borrowed("foo"),
        };
        let key_b = BlackboardKey::Custom {
            mod_id: 2,
            name: Cow::Borrowed("foo"),
        };
        bb.write(0, key_a.clone(), BlackboardValue::Float(1.0), 0, 10);
        bb.write(0, key_b.clone(), BlackboardValue::Float(2.0), 0, 10);
        let g = bb.get(0).unwrap();
        assert_eq!(g.get(&key_a).unwrap().value, BlackboardValue::Float(1.0));
        assert_eq!(g.get(&key_b).unwrap().value, BlackboardValue::Float(2.0));
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn write_overwrites_existing_key() {
        let mut bb = SquadBlackboards::default();
        bb.write(5, BlackboardKey::HeardGunshot, pos(1.0, 1.0), 0, 10);
        bb.write(5, BlackboardKey::HeardGunshot, pos(9.0, 9.0), 5, 10);
        let entry = bb
            .get(5)
            .and_then(|g| g.get(&BlackboardKey::HeardGunshot))
            .unwrap();
        assert_eq!(entry.value, pos(9.0, 9.0));
        assert_eq!(entry.written_tick, 5);
    }
}
