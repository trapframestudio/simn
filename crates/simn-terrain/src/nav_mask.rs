//! Designer-painted nav-override mask. **Phase 2A of
//! [`npc-traversal-plan.md`](../../docs/book/src/planning/npc-traversal-plan.md).**
//!
//! The sim's [`crate::Heightmap`] already drives a per-cell
//! traversability decision via slope + [`crate::FeatureClass`]. Two
//! failure modes that slope+class can't fix on its own:
//!
//! - **False negative.** A `Cliff` or steep slope cell the designer
//!   wants walkable anyway (goat path, ford, scripted route).
//! - **False positive.** Geometrically open terrain that's intentionally
//!   off-limits (fenced compound interior, ravine bottom, story-critical
//!   no-go).
//!
//! [`NavOverride`] is the designer's escape hatch for either case.
//! Painted in Terrain3D (slot 14 = block, slot 15 = walkable), exported
//! to canonical `nav_mask.r8` next to `features.r8`, consumed by
//! [`crate::Heightmap::nav_override_at`] which the sim's
//! `GridNavQuery::from_heightmap` honors per cell.
//!
//! The file format is a single byte per nav cell, row-major NW-up,
//! length `width * height`, no header. Length + format-version +
//! blake3 hash live in `terrain.toml`. See the iteration plan
//! `docs/book/src/planning/sim-iteration-5-13-plan.md` for the
//! full contract.

use std::sync::atomic::{AtomicBool, Ordering};

/// Per-cell designer override on traversability.
///
/// Encoded as a single byte in `nav_mask.r8`. [`decode`] maps the
/// raw byte to the enum; unknown bytes degrade to [`NavOverride::Default`]
/// with one warn-log per process lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum NavOverride {
    /// No override — defer to slope + [`crate::FeatureClass`].
    #[default]
    Default = 0,
    /// Painter says "NPCs cannot enter this cell" regardless of
    /// slope / class. Set by Terrain3D slot 14 (`nav_block`) or by
    /// programmatic POI obstacle stamping.
    ForceBlocked = 1,
    /// Painter says "NPCs can enter this cell" regardless of slope
    /// / `Cliff` / `Water` class. Set by Terrain3D slot 15
    /// (`nav_walkable`). Wins over POI `block` in the merge rule
    /// (designer intent is the more deliberate signal).
    ForceWalkable = 2,
}

impl NavOverride {
    /// `true` for `ForceBlocked` only — the gating override.
    pub fn is_blocked(self) -> bool {
        matches!(self, NavOverride::ForceBlocked)
    }

    /// `true` for `ForceWalkable` only — the carving override.
    pub fn is_walkable_override(self) -> bool {
        matches!(self, NavOverride::ForceWalkable)
    }
}

/// Canonical file format version for `nav_mask.r8`.
///
/// Stored in `terrain.toml` as `nav_mask_format_version`. Loader
/// rejects unknown versions to catch drift after a future bump.
pub const NAV_MASK_FORMAT_VERSION: u8 = 1;

static UNKNOWN_BYTE_WARNED: AtomicBool = AtomicBool::new(false);

/// Decode a single `nav_mask.r8` byte into a [`NavOverride`].
///
/// Unknown values (anything ≥ 3 in v1) degrade to [`NavOverride::Default`]
/// and trigger one `tracing::warn` per process lifetime — drift
/// insurance against a designer's painter accidentally stamping a
/// high-weight slot that wasn't part of the schema.
pub fn decode(byte: u8) -> NavOverride {
    match byte {
        0 => NavOverride::Default,
        1 => NavOverride::ForceBlocked,
        2 => NavOverride::ForceWalkable,
        other => {
            if !UNKNOWN_BYTE_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    byte = other,
                    "nav_mask.r8 contained unknown byte; treating as Default. \
                     Further warnings suppressed."
                );
            }
            NavOverride::Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_known_values() {
        assert_eq!(decode(0), NavOverride::Default);
        assert_eq!(decode(1), NavOverride::ForceBlocked);
        assert_eq!(decode(2), NavOverride::ForceWalkable);
    }

    #[test]
    fn decode_unknown_byte_is_default() {
        // Reset the warn-flag so this test sees the warn fire. We
        // can't actually assert on the log output without a logger
        // fixture; the goal here is just the decode behavior.
        UNKNOWN_BYTE_WARNED.store(false, Ordering::Relaxed);
        assert_eq!(decode(3), NavOverride::Default);
        assert_eq!(decode(42), NavOverride::Default);
        assert_eq!(decode(255), NavOverride::Default);
    }

    #[test]
    fn default_is_default_variant() {
        assert_eq!(NavOverride::default(), NavOverride::Default);
    }

    #[test]
    fn predicates_match_variants() {
        assert!(NavOverride::ForceBlocked.is_blocked());
        assert!(!NavOverride::ForceWalkable.is_blocked());
        assert!(!NavOverride::Default.is_blocked());

        assert!(NavOverride::ForceWalkable.is_walkable_override());
        assert!(!NavOverride::ForceBlocked.is_walkable_override());
        assert!(!NavOverride::Default.is_walkable_override());
    }
}
