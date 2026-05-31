//! Loot pool registry + content roll (Phase 3B).
//!
//! Pools are keyed by `(faction, depth_tier, family)` and live in
//! `content/items/loot_pools.toml`. Each pool lists weighted entries;
//! a single roll picks one entry weighted by `weight`, then picks a
//! stack size uniformly in `[count_min, count_max]`.
//!
//! Two roll APIs:
//!
//! - [`LootPoolRegistry::roll_one`] — picks one entry. The
//!   primitive used by both the procedural restock path (3C) and
//!   the quest-reward path below.
//! - [`LootPoolRegistry::roll_quest_reward`] — picks one entry
//!   using **best-of-K** semantics: roll K times and keep the
//!   rarest (lowest-weight) candidate. K scales with quest
//!   difficulty so harder quests skew toward rare entries without
//!   touching the underlying weights or duplicating pools per
//!   difficulty tier.
//!
//! Container-kind selection by quest difficulty lives separately
//! on [`crate::loot_containers::LootContainerRegistry`] —
//! `weighted_pick_for_difficulty` biases toward the larger
//! container kinds as difficulty climbs.
//!
//! Pool lookup falls back through this chain when an exact match
//! is missing — keeps TOML editable without forcing every
//! `(faction, tier, family)` cell to be authored:
//!
//! ```text
//! (faction, tier, family) → (faction, 1, family) →
//! ("nomads", tier, family) → ("nomads", 1, family) → empty
//! ```

use bevy_ecs::prelude::Resource;
use rand::Rng;
use serde::Deserialize;
use std::sync::OnceLock;

use crate::items::ItemId;

/// One weighted entry inside a pool. `weight` drives selection
/// likelihood; `count_min`/`count_max` bracket the stack size
/// returned when the entry is picked.
#[derive(Clone, Debug, Deserialize)]
pub struct PoolEntry {
    pub id: String,
    pub weight: u32,
    #[serde(default = "one")]
    pub count_min: u32,
    #[serde(default = "one")]
    pub count_max: u32,
}

fn one() -> u32 {
    1
}

/// One pool — the unit `LootPoolRegistry` indexes by
/// `(faction, depth_tier, family)`.
#[derive(Clone, Debug, Deserialize)]
pub struct LootPool {
    pub faction: String,
    pub depth_tier: u8,
    pub family: String,
    pub entries: Vec<PoolEntry>,
}

#[derive(Deserialize)]
struct Wire {
    #[serde(default, rename = "pool")]
    pools: Vec<LootPool>,
}

/// One rolled item: an item id plus the stack size to grant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RolledItem {
    pub id: ItemId,
    pub count: u32,
}

/// Process-wide registry of loot pools. Loaded from
/// `content/items/loot_pools.toml`; same `OnceLock` caching as the
/// other content registries — TOML parse runs at most once per
/// process.
#[derive(Resource, Clone, Debug, Default)]
pub struct LootPoolRegistry {
    pools: Vec<LootPool>,
}

impl LootPoolRegistry {
    /// Load `content/items/loot_pools.toml`. Returns an empty
    /// registry on missing file or parse failure — callers treat
    /// "no pool" as "no loot" rather than panicking sim init.
    pub fn load() -> Self {
        static CACHE: OnceLock<Vec<LootPool>> = OnceLock::new();
        let pools = CACHE
            .get_or_init(|| Self::parse_pools(&crate::ContentSource::Embedded))
            .clone();
        Self { pools }
    }

    /// Load from an explicit content source; see [`crate::items::ItemRegistry::load_from`].
    pub fn load_from(src: &crate::ContentSource) -> Self {
        match src {
            crate::ContentSource::Embedded => Self::load(),
            other => Self {
                pools: Self::parse_pools(other),
            },
        }
    }

    /// A missing or malformed file yields an empty registry — callers
    /// treat "no pool" as "no loot" rather than panicking sim init.
    fn parse_pools(src: &crate::ContentSource) -> Vec<LootPool> {
        match src.read_str_opt("loot/loot_pools.toml") {
            Ok(Some(text)) => match toml::from_str::<Wire>(&text) {
                Ok(wire) => wire.pools,
                Err(e) => {
                    tracing::warn!("loot_pools.toml parse failed ({e}); using empty registry");
                    Vec::new()
                }
            },
            Ok(None) => Vec::new(),
            Err(e) => {
                tracing::warn!("loot_pools.toml read failed: {e}; using empty registry");
                Vec::new()
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pools.len()
    }

    /// Look up a pool with the full fallback chain (see module
    /// docs). Returns `None` only if neither the exact match nor
    /// any fallback exists.
    pub fn lookup(&self, faction: &str, depth_tier: u8, family: &str) -> Option<&LootPool> {
        // Exact.
        if let Some(p) = self.find_exact(faction, depth_tier, family) {
            return Some(p);
        }
        // Same faction, tier 1.
        if depth_tier != 1 {
            if let Some(p) = self.find_exact(faction, 1, family) {
                return Some(p);
            }
        }
        // Nomads at the requested tier.
        if faction != "nomads" {
            if let Some(p) = self.find_exact("nomads", depth_tier, family) {
                return Some(p);
            }
            // Nomads tier 1 — the floor.
            if depth_tier != 1 {
                if let Some(p) = self.find_exact("nomads", 1, family) {
                    return Some(p);
                }
            }
        }
        None
    }

    fn find_exact(&self, faction: &str, depth_tier: u8, family: &str) -> Option<&LootPool> {
        self.pools
            .iter()
            .find(|p| p.faction == faction && p.depth_tier == depth_tier && p.family == family)
    }

    /// Roll one weighted entry from `(faction, depth_tier, family)`.
    /// Stack size is picked uniformly in `[count_min, count_max]`.
    /// Returns `None` when the pool (or any fallback) is empty.
    pub fn roll_one(
        &self,
        rng: &mut impl Rng,
        faction: &str,
        depth_tier: u8,
        family: &str,
    ) -> Option<RolledItem> {
        let pool = self.lookup(faction, depth_tier, family)?;
        self.pick_weighted(rng, pool).map(|e| RolledItem {
            id: ItemId(e.id.clone()),
            count: rng.gen_range(e.count_min.min(e.count_max)..=e.count_min.max(e.count_max)),
        })
    }

    /// Quest-reward roll — **best of K**, where K = `lottery_k`.
    /// Picks K candidates with the normal weighted picker, then
    /// returns the one with the **lowest** `weight` (rarest). The
    /// stack count for that item still rolls in its declared
    /// `[count_min, count_max]`.
    ///
    /// K=1 collapses to [`roll_one`]; higher K skews aggressively
    /// toward rare entries. Difficulty → K mapping is the
    /// caller's choice (see [`quest_lottery_k`]).
    pub fn roll_quest_reward(
        &self,
        rng: &mut impl Rng,
        faction: &str,
        depth_tier: u8,
        family: &str,
        lottery_k: u32,
    ) -> Option<RolledItem> {
        let pool = self.lookup(faction, depth_tier, family)?;
        let k = lottery_k.max(1);
        let mut best: Option<&PoolEntry> = None;
        for _ in 0..k {
            let candidate = self.pick_weighted(rng, pool)?;
            best = match best {
                None => Some(candidate),
                Some(prev) if candidate.weight < prev.weight => Some(candidate),
                Some(prev) => Some(prev),
            };
        }
        best.map(|e| RolledItem {
            id: ItemId(e.id.clone()),
            count: rng.gen_range(e.count_min.min(e.count_max)..=e.count_min.max(e.count_max)),
        })
    }

    fn pick_weighted<'a>(&self, rng: &mut impl Rng, pool: &'a LootPool) -> Option<&'a PoolEntry> {
        if pool.entries.is_empty() {
            return None;
        }
        let total: u64 = pool.entries.iter().map(|e| u64::from(e.weight)).sum();
        if total == 0 {
            // Every weight zero — uniform fallback.
            let i = rng.gen_range(0..pool.entries.len());
            return Some(&pool.entries[i]);
        }
        let mut roll = rng.gen_range(0..total);
        for e in &pool.entries {
            let w = u64::from(e.weight);
            if roll < w {
                return Some(e);
            }
            roll -= w;
        }
        pool.entries.last()
    }
}

/// Lottery-K curve for quest reward rolls. Difficulty 1 → K=1
/// (normal weighted pick); difficulty 5 → K=5 (strong rare bias).
/// Difficulty 0 / out-of-range clamps to 1.
pub fn quest_lottery_k(difficulty: u8) -> u32 {
    match difficulty {
        0 | 1 => 1,
        2 => 1,
        3 => 2,
        4 => 3,
        _ => 5,
    }
}
