//! Factions, faction relations, and string codecs.
//!
//! All faction data lives in the [`registry`] module — the canonical
//! roster, relation matrix, default loadouts, base aggression, and
//! road-friendliness all flow from `crates/simn-sim/src/factions.toml`
//! at sim startup. There is no closed `Faction` enum; gameplay
//! references factions by [`registry::FactionId`] (a small interned
//! `u32`) for hot-path equality, or by registry name string for
//! save / wire serialization.
//!
//! Roster + lore reference: `docs/book/src/lore/factions/`. Naming
//! rule from DESIGN.md §5.3 is hard: faction names in code mirror
//! the canonical lore names exactly.
//!
//! ## Relation spectrum
//!
//! Five anchors ordered as a number line from "shoot on sight" to
//! "come to your aid": [`Relation::Hostile`] / `Cold` / `Neutral` /
//! `Warm` / `Friendly`. Internally a continuous `i16` score so
//! playthrough events can drift relations without re-architecting.
//! Anchor scores: `-100 / -50 / 0 / +50 / +100`. The registry stores
//! scores and snaps to bands on read via [`band_from_score`].

use serde::{Deserialize, Serialize};

pub mod registry;

/// Five-step ordered spectrum of inter-faction (or player-to-faction)
/// affinity. Ordered as a number line from "shoot on sight" to "come
/// to your aid":
///
/// ```text
/// Hostile   Cold   Neutral   Warm   Friendly
///   -100    -50      0       +50      +100
/// ```
///
/// Numeric anchors are exposed via [`anchor_score`]; the registry
/// stores continuous `i16` scores and snaps to bands via
/// [`band_from_score`] so playthrough drift can be expressed without
/// re-architecting. Variants are declared in spectrum order so
/// `(a as u8) < (b as u8)` matches "a is more hostile than b."
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Relation {
    /// Active opposition. Shoot on sight.
    Hostile,
    /// Functional, transactional, publicly civil.
    Cold,
    /// Default for unrelated parties; covers the old `Detente`
    /// (official coolness with heavy unofficial trade) collapsed in
    /// per the 2026-05-06 spectrum lock.
    Neutral,
    /// Aligned interests; trade easily, friendly toward the player.
    Warm,
    /// Active alliance — comes to the player's aid in combat.
    Friendly,
}

/// Score anchor for a named relation band. Scores between anchors
/// are valid; [`band_from_score`] returns the nearest enum.
pub const fn anchor_score(r: Relation) -> i16 {
    match r {
        Relation::Hostile => -100,
        Relation::Cold => -50,
        Relation::Neutral => 0,
        Relation::Warm => 50,
        Relation::Friendly => 100,
    }
}

/// Inclusive lower bound the registry clamps drift scores to.
pub const SCORE_MIN: i16 = -100;
/// Inclusive upper bound the registry clamps drift scores to.
pub const SCORE_MAX: i16 = 100;

/// Snap a continuous score to the nearest [`Relation`] band. Bands
/// center on each anchor with the boundary midway between adjacent
/// anchors (`±25`, `±75`), so a small drift doesn't immediately
/// re-classify but a deliberate ±50 shift moves to the next band.
pub fn band_from_score(score: i16) -> Relation {
    match score {
        s if s <= -75 => Relation::Hostile,
        s if s <= -25 => Relation::Cold,
        s if s < 25 => Relation::Neutral,
        s if s < 75 => Relation::Warm,
        _ => Relation::Friendly,
    }
}

/// Render a [`Relation`] band as a lowercase ascii string for UI /
/// log lines. Inverse: [`relation_from_str`].
pub fn relation_to_str(r: Relation) -> &'static str {
    match r {
        Relation::Hostile => "hostile",
        Relation::Cold => "cold",
        Relation::Neutral => "neutral",
        Relation::Warm => "warm",
        Relation::Friendly => "friendly",
    }
}

/// Inverse of [`relation_to_str`]. Used by the TOML loader and any
/// caller persisting the named band.
pub fn relation_from_str(s: &str) -> Option<Relation> {
    Some(match s {
        "hostile" => Relation::Hostile,
        "cold" => Relation::Cold,
        "neutral" => Relation::Neutral,
        "warm" => Relation::Warm,
        "friendly" => Relation::Friendly,
        _ => return None,
    })
}
