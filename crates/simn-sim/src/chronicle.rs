//! The permanent record of every NPC the world has ever produced.
//!
//! `LifeRecord`s are written when an NPC is born, updated on
//! significant life events (region migration, death), and *kept* in
//! the chronicle after the entity itself is despawned. They're how
//! later systems answer "who lived on the Western Line last week",
//! "how many Linemen has the Valley killed since launch", etc.
//!
//! Backed by a `BTreeMap<NpcId, LifeRecord>` so iteration order is
//! deterministic without sorting at every query.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use bevy_ecs::prelude::Resource;

use crate::components::NpcId;

use crate::region::RegionId;

/// What ended an NPC's life. `Combat` carries the faction whose
/// member was responsible — useful for chronicle queries like "how
/// many Linemen has the Western Line's PWA killed since launch."
/// `Copy` removed when `Combat` started carrying a `String`
/// (faction name). Death events are infrequent; the `clone()` cost
/// is negligible vs the gain from name-string portability across
/// registry edits.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum DeathCause {
    /// Lifespan expired.
    NaturalCauses,
    /// Killed by an NPC of the named faction. Stored as the
    /// registry name string (`"pwa"`) so chronicles stay valid
    /// across registry edits.
    Combat { killer_faction: String },
    /// Catch-all for non-combat deaths we don't model yet.
    Other,
}

pub fn death_cause_to_str(c: &DeathCause) -> &'static str {
    match c {
        DeathCause::NaturalCauses => "natural_causes",
        DeathCause::Combat { .. } => "combat",
        DeathCause::Other => "other",
    }
}

/// Permanent record for one NPC, alive or dead.
///
/// Additive shape: new fields use `#[serde(default)]` so older
/// snapshots still load. Order of declaration matters for `bincode`
/// compatibility.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LifeRecord {
    pub id: NpcId,
    /// Faction name (registry id as string, e.g. `"pwa"`).
    pub faction: String,

    pub birth_tick: u64,
    pub birth_region: RegionId,
    pub birth_pos: [f32; 3],

    pub death_tick: Option<u64>,
    pub death_region: Option<RegionId>,
    pub death_cause: Option<DeathCause>,

    /// `(region, entered_at_tick)` log. The first entry mirrors the
    /// birth region; subsequent entries land on each migration.
    #[serde(default)]
    pub regions_visited: Vec<(RegionId, u64)>,
}

impl LifeRecord {
    pub fn is_alive(&self) -> bool {
        self.death_tick.is_none()
    }
}

#[derive(Resource, Serialize, Deserialize, Clone, Debug, Default)]
pub struct LifeChronicle {
    pub records: BTreeMap<NpcId, LifeRecord>,
    /// Running totals updated incrementally by [`insert`] and
    /// [`mark_dead`]. Skipped during serialization (records are the
    /// canonical source); rebuilt by [`rebuild_summary_cache`] after
    /// snapshot load. Returned by [`summary`] without iteration.
    #[serde(skip)]
    summary_cache: ChronicleSummary,
}

impl LifeChronicle {
    pub fn insert(&mut self, record: LifeRecord) {
        let faction = record.faction.clone();
        let alive = record.is_alive();
        let prev = self.records.insert(record.id, record);
        // If we replaced an existing record (rare — IDs are
        // monotonically minted), unwind its contribution before
        // applying the new one. Keeps the cache exact under any
        // future use of `insert`.
        if let Some(old) = prev {
            self.unapply(&old);
        }
        self.summary_cache.total_ever_spawned += 1;
        let s = self.summary_cache.by_faction.entry(faction).or_default();
        if alive {
            self.summary_cache.currently_alive += 1;
            s.alive = s.alive.saturating_add(1);
        } else {
            s.dead = s.dead.saturating_add(1);
        }
    }

    pub fn get(&self, id: NpcId) -> Option<&LifeRecord> {
        self.records.get(&id)
    }

    pub fn get_mut(&mut self, id: NpcId) -> Option<&mut LifeRecord> {
        self.records.get_mut(&id)
    }

    /// Mark an NPC as dead in the chronicle and update the cached
    /// summary in step. No-op if the record is missing or already
    /// dead — callers can route every death through this method
    /// without an upfront alive-check.
    ///
    /// All death sites (npc_age, npc_death_check, offline combat)
    /// must use this rather than poking `death_tick` via `get_mut`,
    /// or the summary cache will drift.
    pub fn mark_dead(&mut self, id: NpcId, tick: u64, region: RegionId, cause: DeathCause) -> bool {
        let Some(rec) = self.records.get_mut(&id) else {
            return false;
        };
        if rec.death_tick.is_some() {
            return false;
        }
        rec.death_tick = Some(tick);
        rec.death_region = Some(region);
        rec.death_cause = Some(cause);
        let faction = rec.faction.clone();
        if let Some(s) = self.summary_cache.by_faction.get_mut(&faction) {
            if s.alive > 0 {
                s.alive -= 1;
            }
            s.dead = s.dead.saturating_add(1);
        }
        if self.summary_cache.currently_alive > 0 {
            self.summary_cache.currently_alive -= 1;
        }
        true
    }

    fn unapply(&mut self, rec: &LifeRecord) {
        self.summary_cache.total_ever_spawned =
            self.summary_cache.total_ever_spawned.saturating_sub(1);
        if let Some(s) = self.summary_cache.by_faction.get_mut(&rec.faction) {
            if rec.is_alive() {
                if s.alive > 0 {
                    s.alive -= 1;
                }
            } else if s.dead > 0 {
                s.dead -= 1;
            }
        }
        if rec.is_alive() && self.summary_cache.currently_alive > 0 {
            self.summary_cache.currently_alive -= 1;
        }
    }

    /// Reconstruct the cached summary from the full record set.
    /// Called once after snapshot load (the cache is `#[serde(skip)]`
    /// so it arrives default-zero). O(n) over records but only paid
    /// at load — every subsequent `summary()` call is O(1).
    pub fn rebuild_summary_cache(&mut self) {
        let mut cache = ChronicleSummary::default();
        for r in self.records.values() {
            cache.total_ever_spawned += 1;
            let s = cache.by_faction.entry(r.faction.clone()).or_default();
            if r.is_alive() {
                cache.currently_alive += 1;
                s.alive = s.alive.saturating_add(1);
            } else {
                s.dead = s.dead.saturating_add(1);
            }
        }
        self.summary_cache = cache;
    }

    /// Most recent deaths first, capped at `limit`. Iterates the full
    /// chronicle; if it grows past low-tens-of-thousands we'll add an
    /// index, but at that scale the snapshot will need attention too.
    pub fn recent_deaths(&self, limit: usize) -> Vec<&LifeRecord> {
        let mut deaths: Vec<&LifeRecord> = self
            .records
            .values()
            .filter(|r| r.death_tick.is_some())
            .collect();
        deaths.sort_by_key(|r| std::cmp::Reverse(r.death_tick.unwrap_or(0)));
        deaths.truncate(limit);
        deaths
    }

    /// `(total_ever_spawned, currently_alive)` plus per-faction
    /// alive/dead splits. O(1) — returns the incrementally
    /// maintained cache, no record iteration.
    pub fn summary(&self) -> &ChronicleSummary {
        &self.summary_cache
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChronicleSummary {
    pub total_ever_spawned: u64,
    pub currently_alive: u64,
    pub by_faction: std::collections::HashMap<String, FactionStats>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FactionStats {
    pub alive: u32,
    pub dead: u32,
}
