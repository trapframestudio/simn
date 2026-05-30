//! Per-tick line-of-sight (LOS) cache.
//!
//! `npc_aggro` already runs a per-pair exposure raycast through
//! [`crate::perception::LosService`]; without this cache, downstream
//! consumers (cover-system queries, future tactical AI peek-shoot
//! decisions, the planned hitscan/projectile combat path) would have
//! to recompute the same rays. This module stores the directional
//! exposure values for the current tick so consumers can look them up
//! cheaply and tactical AI doesn't double-spend raycasts on pairs
//! aggro already evaluated.
//!
//! ## Tick discipline
//!
//! Entries are valid only for the tick they were computed in. A
//! [`clear_los_cache`] system runs early in the tick schedule (before
//! [`crate::systems::npc_aggro`]) and drops any entries from prior
//! ticks. Cache writes happen during `npc_aggro`'s pair-scan; reads
//! happen later in the same tick (combat resolution, cover queries).
//!
//! ## Direction-keyed
//!
//! LOS is asymmetric: `eye(A) -> B` and `eye(B) -> A` traverse
//! different rays (different start points + obstacles between eye
//! height and ground may differ). Cache stores both directions as
//! separate entries; callers ask "can OBSERVER see TARGET?" and get
//! the right value back.
//!
//! ## Phase 1
//!
//! Pure primitive — no consumers in this PR. Slot the cache in now so
//! the upcoming cover-system + physical-combat work can plug in
//! without touching `npc_aggro`. See
//! `docs/book/src/planning/combat-los-plan.md`.

use std::collections::HashMap;

use bevy_ecs::prelude::{Res, ResMut, Resource};

use crate::components::NpcId;
use crate::resources::SimClock;

/// One cached exposure result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LosEntry {
    /// Exposure in `0.0..=1.0`. `1.0` = fully visible; `0.0` =
    /// fully blocked.
    pub exposure: f32,
    /// Tick when the entry was written. Older entries are dropped by
    /// [`clear_los_cache`] at tick start.
    pub computed_tick: u64,
}

/// Per-tick LOS exposure cache. Keyed `(observer, target)` so the
/// asymmetric direction is captured.
#[derive(Resource, Default, Clone)]
pub struct LosCache {
    entries: HashMap<(NpcId, NpcId), LosEntry>,
}

impl LosCache {
    /// Read a cached exposure. `None` if the pair hasn't been
    /// evaluated this tick.
    pub fn get(&self, observer: NpcId, target: NpcId) -> Option<f32> {
        self.entries
            .get(&(observer, target))
            .map(|entry| entry.exposure)
    }

    /// Read the full entry (exposure + tick).
    pub fn entry(&self, observer: NpcId, target: NpcId) -> Option<LosEntry> {
        self.entries.get(&(observer, target)).copied()
    }

    /// Write a freshly-computed exposure value.
    pub fn put(&mut self, observer: NpcId, target: NpcId, exposure: f32, tick: u64) {
        self.entries.insert(
            (observer, target),
            LosEntry {
                exposure,
                computed_tick: tick,
            },
        );
    }

    /// Drop entries older than `current_tick`. Default usage:
    /// `cache.clear_stale(now)` at the top of each tick keeps only
    /// entries written in the new tick (none yet at that point) plus
    /// any callers that wrote ahead-of-time. In practice this just
    /// empties the cache once per tick, but the explicit comparison
    /// leaves a path open to bumping retention if a system needs
    /// previous-tick LOS for hysteresis.
    pub fn clear_stale(&mut self, current_tick: u64) {
        self.entries
            .retain(|_, entry| entry.computed_tick >= current_tick);
    }

    /// Drop entries older than `max_age` ticks. Used when the
    /// writer (`npc_aggro` Pass 2) runs on a coarser cadence than
    /// the consumers (`npc_combat`'s LOS gate); entries written at
    /// tick `T` stay valid for ticks `T..T+max_age`, so consumers
    /// reading on an in-between tick get the most recent reading
    /// instead of a cleared cache. With `max_age == 1` this
    /// degenerates to [`Self::clear_stale`].
    pub fn retain_newer_than(&mut self, current_tick: u64, max_age: u64) {
        self.entries
            .retain(|_, entry| current_tick.saturating_sub(entry.computed_tick) < max_age);
    }

    /// Number of currently-cached entries. Test / instrumentation
    /// helper; not load-bearing for runtime correctness.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache has any entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Schedule-side wrapper that evicts pre-tick entries. Runs before
/// [`crate::systems::npc_aggro`] in the NPC tick chain so the cache
/// is empty when aggro starts writing.
pub fn clear_los_cache(clock: Res<SimClock>, mut cache: ResMut<LosCache>) {
    let t = std::time::Instant::now();
    // Retain entries written within the last
    // `PASS_2_TICK_INTERVAL` ticks. `npc_aggro` only repopulates
    // the cache every `PASS_2_TICK_INTERVAL` ticks (acquisition is
    // the expensive pair scan; keeping it on every-tick cadence
    // was the bulk of the sim-tick cost at 720 NPCs), so consumers
    // like `npc_combat` need the prior reading to stay valid on
    // ticks where Pass 2 didn't run. The world moves slowly enough
    // at 20 Hz that a 3-tick (~150 ms) old LOS read is
    // behaviorally indistinguishable from a fresh one.
    cache.retain_newer_than(clock.tick, crate::systems::npc_aggro::PASS_2_TICK_INTERVAL);
    crate::systems::record_perception_slot(crate::systems::prof_slots::CLEAR_LOS, t.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::NpcId;

    fn id(n: u64) -> NpcId {
        NpcId(n)
    }

    #[test]
    fn put_and_get_round_trip() {
        let mut c = LosCache::default();
        assert!(c.is_empty());
        c.put(id(1), id(2), 0.6, 100);
        assert_eq!(c.get(id(1), id(2)), Some(0.6));
        // Direction matters: (2,1) is a separate entry.
        assert!(c.get(id(2), id(1)).is_none());
    }

    #[test]
    fn entry_returns_full_record() {
        let mut c = LosCache::default();
        c.put(id(7), id(9), 0.42, 50);
        let e = c.entry(id(7), id(9)).expect("entry present");
        assert_eq!(e.exposure, 0.42);
        assert_eq!(e.computed_tick, 50);
    }

    #[test]
    fn clear_stale_drops_old_entries_keeps_current() {
        let mut c = LosCache::default();
        c.put(id(1), id(2), 0.4, 10);
        c.put(id(3), id(4), 0.7, 11);
        c.clear_stale(11);
        // tick 10 entry is gone, tick 11 stays.
        assert!(c.get(id(1), id(2)).is_none());
        assert_eq!(c.get(id(3), id(4)), Some(0.7));
    }

    #[test]
    fn clear_stale_empty_cache_is_noop() {
        let mut c = LosCache::default();
        c.clear_stale(42);
        assert!(c.is_empty());
    }

    #[test]
    fn put_overwrites_existing_entry() {
        let mut c = LosCache::default();
        c.put(id(1), id(2), 0.3, 5);
        c.put(id(1), id(2), 0.9, 6);
        let e = c.entry(id(1), id(2)).unwrap();
        assert_eq!(e.exposure, 0.9);
        assert_eq!(e.computed_tick, 6);
    }
}
