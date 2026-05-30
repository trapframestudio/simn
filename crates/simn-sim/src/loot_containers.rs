//! Loot container registry — scene-placed `WorldContainer` kinds.
//!
//! Loaded once per process from `content/loot_containers.toml`. The
//! catalog drives `world_seed`'s scatter pass (Phase 3A) and will
//! be the anchor surface for the Phase 3B faction × depth pool
//! tables + Phase 3C restock. The TOML schema:
//!
//! ```toml
//! [[containers]]
//! id = "small_crate"
//! name = "Small Crate"
//! grid = { w = 4, h = 4 }
//! spawn_weight = 70
//! is_public = false
//! ```
//!
//! Public (`is_public = true`) crates count toward the crafting
//! kit-pool — author them sparingly for explicit shared bench
//! bins. Default false: a generic loot crate is the player's
//! reward, not a permanent shared workbench resource.
//!
//! The registry is content config, not save state — it's reloaded
//! from disk on every `Sim::new` / `Sim::load`. Adding / editing /
//! removing container kinds in TOML lands without a save migration.

use bevy_ecs::prelude::Resource;
use serde::Deserialize;
use std::sync::OnceLock;

use crate::items::GridSize;

/// One container kind. Field names match the TOML keys verbatim.
#[derive(Clone, Debug, Deserialize)]
pub struct LootContainerDef {
    pub id: String,
    pub name: String,
    pub grid: GridSize,
    #[serde(default = "default_weight")]
    pub spawn_weight: u32,
    #[serde(default)]
    pub is_public: bool,
    /// `[min, max]` items rolled when restocking this kind
    /// (Phase 3C consumes this). Defaults to `[1, 1]` if absent —
    /// a single-item crate is the minimum-viable fill.
    #[serde(default = "default_items_per_roll")]
    pub items_per_roll: [u32; 2],
    /// Family → weight map for restock rolls. Empty = "no family
    /// bias, fall back to a uniform pick across the loot-pool
    /// registry's known families for this faction".
    #[serde(default)]
    pub family_weights: std::collections::HashMap<String, u32>,
    /// `[difficulty, weight]` pairs driving quest-reward container
    /// kind selection. Empty = "this kind never appears as a quest
    /// reward" (still picked by procedural scatter via
    /// `spawn_weight`).
    #[serde(default)]
    pub difficulty_weights: Vec<[u32; 2]>,
}

fn default_weight() -> u32 {
    1
}

fn default_items_per_roll() -> [u32; 2] {
    [1, 1]
}

#[derive(Deserialize)]
struct Wire {
    #[serde(default)]
    containers: Vec<LootContainerDef>,
}

/// Process-wide registry of loot-container kinds.
///
/// Cloneable into Bevy as a resource. The underlying `Vec` is
/// cached in a `OnceLock` so the TOML parse runs at most once per
/// process — same pattern as [`crate::items::ItemRegistry`].
#[derive(Resource, Clone, Debug, Default)]
pub struct LootContainerRegistry {
    defs: Vec<LootContainerDef>,
}

impl LootContainerRegistry {
    /// Load from `content/loot_containers.toml` on disk. Returns an
    /// empty registry if the file is missing or malformed — callers
    /// (just `world_seed`) treat an empty registry as "no scatter
    /// containers this run" rather than failing sim init.
    pub fn load() -> Self {
        static CACHE: OnceLock<Vec<LootContainerDef>> = OnceLock::new();
        let defs = CACHE
            .get_or_init(|| Self::parse_defs(&crate::ContentSource::Embedded))
            .clone();
        Self { defs }
    }

    /// Load from an explicit content source; see [`crate::items::ItemRegistry::load_from`].
    pub fn load_from(src: &crate::ContentSource) -> Self {
        match src {
            crate::ContentSource::Embedded => Self::load(),
            other => Self {
                defs: Self::parse_defs(other),
            },
        }
    }

    /// A missing or malformed file yields an empty registry — callers
    /// (just `world_seed`) treat that as "no scatter containers this
    /// run" rather than failing sim init.
    fn parse_defs(src: &crate::ContentSource) -> Vec<LootContainerDef> {
        match src.read_str_opt("loot_containers.toml") {
            Ok(Some(text)) => match toml::from_str::<Wire>(&text) {
                Ok(wire) => wire.containers,
                Err(e) => {
                    tracing::warn!("loot_containers.toml parse failed ({e}); using empty registry");
                    Vec::new()
                }
            },
            Ok(None) => Vec::new(),
            Err(e) => {
                tracing::warn!("loot_containers.toml read failed: {e}; using empty registry");
                Vec::new()
            }
        }
    }

    /// Iterate every defined container kind. Order matches the
    /// TOML source.
    pub fn iter(&self) -> impl Iterator<Item = &LootContainerDef> {
        self.defs.iter()
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    /// Look a kind up by id. `None` if the id isn't in the catalog
    /// — caller decides whether that's a soft skip or a hard error.
    pub fn get(&self, id: &str) -> Option<&LootContainerDef> {
        self.defs.iter().find(|d| d.id == id)
    }

    /// Pick a kind weighted by `spawn_weight` using `rng`. Returns
    /// `None` only if the registry is empty.
    pub fn weighted_pick(&self, rng: &mut impl rand::Rng) -> Option<&LootContainerDef> {
        if self.defs.is_empty() {
            return None;
        }
        let total: u64 = self.defs.iter().map(|d| u64::from(d.spawn_weight)).sum();
        if total == 0 {
            // Every weight is zero — fall back to uniform pick so
            // a malformed TOML doesn't silently freeze scatter.
            let idx = rng.gen_range(0..self.defs.len());
            return Some(&self.defs[idx]);
        }
        let mut roll = rng.gen_range(0..total);
        for d in &self.defs {
            let w = u64::from(d.spawn_weight);
            if roll < w {
                return Some(d);
            }
            roll -= w;
        }
        // Unreachable in practice — kept for safety; pick the last.
        self.defs.last()
    }

    /// Quest-reward kind picker — weighted by each kind's
    /// `difficulty_weights` entry for the requested `difficulty`.
    /// Difficulty 1 (trivial) skews to small crates; difficulty 5
    /// (very hard) skews to large caches.
    ///
    /// Returns `None` only if **no** kind has an entry for the
    /// requested difficulty (genuinely impossible to roll a reward
    /// at that tier). Out-of-range difficulties (0, >5) clamp to
    /// the nearest configured tier.
    pub fn weighted_pick_for_difficulty(
        &self,
        rng: &mut impl rand::Rng,
        difficulty: u32,
    ) -> Option<&LootContainerDef> {
        if self.defs.is_empty() {
            return None;
        }
        let clamped = difficulty.clamp(1, 5);
        // Collect (def, weight) for the requested difficulty.
        let candidates: Vec<(&LootContainerDef, u32)> = self
            .defs
            .iter()
            .filter_map(|d| {
                d.difficulty_weights
                    .iter()
                    .find(|pair| pair[0] == clamped)
                    .map(|pair| (d, pair[1]))
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let total: u64 = candidates.iter().map(|(_, w)| u64::from(*w)).sum();
        if total == 0 {
            let i = rng.gen_range(0..candidates.len());
            return Some(candidates[i].0);
        }
        let mut roll = rng.gen_range(0..total);
        for (def, w) in &candidates {
            let w = u64::from(*w);
            if roll < w {
                return Some(*def);
            }
            roll -= w;
        }
        candidates.last().map(|(d, _)| *d)
    }
}

impl LootContainerDef {
    /// Pick a family weighted by this kind's `family_weights`.
    /// Returns `None` if the map is empty — caller decides whether
    /// to fall back to a default family list or skip the roll.
    pub fn pick_family(&self, rng: &mut impl rand::Rng) -> Option<&str> {
        if self.family_weights.is_empty() {
            return None;
        }
        // Sorted-by-key iteration so the picker is deterministic
        // across HashMap reseeds — same seed always picks the same
        // family.
        let mut entries: Vec<(&String, &u32)> = self.family_weights.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let total: u64 = entries.iter().map(|(_, w)| u64::from(**w)).sum();
        if total == 0 {
            let i = rng.gen_range(0..entries.len());
            return Some(entries[i].0.as_str());
        }
        let mut roll = rng.gen_range(0..total);
        for (fam, w) in &entries {
            let w = u64::from(**w);
            if roll < w {
                return Some(fam.as_str());
            }
            roll -= w;
        }
        entries.last().map(|(fam, _)| fam.as_str())
    }

    /// Random item count within `items_per_roll`. Useful for the
    /// 3C restock loop ("roll this many items for this crate").
    pub fn roll_items_per_roll(&self, rng: &mut impl rand::Rng) -> u32 {
        let lo = self.items_per_roll[0].min(self.items_per_roll[1]);
        let hi = self.items_per_roll[0].max(self.items_per_roll[1]).max(lo);
        rng.gen_range(lo..=hi)
    }
}
