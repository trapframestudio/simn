//! Content source — where the sim reads its content pack from.
//!
//! Historically every TOML / name list was baked into the binary via
//! `include_str!`, hard-wiring one game's factions/items/weapons into
//! the engine. To make the sim reusable, content now flows through a
//! [`ContentSource`]:
//!
//! - [`ContentSource::Embedded`] — the example pack compiled into the
//!   binary (the `content/` dir, embedded via [`include_dir`]). Lets
//!   the sim run standalone with zero external files; this is what the
//!   test suite and the default constructors use.
//! - [`ContentSource::Dir`] — a content directory supplied by a
//!   consuming game, overriding the embedded pack.
//!
//! Logical paths are always `/`-separated and relative to the pack
//! root (e.g. `"items/weapons.toml"`, `"names/first/slavic.txt"`),
//! regardless of variant.
//!
//! Determinism: never iterate the embedded `Dir`'s entries to drive
//! RNG. Loaders resolve explicit logical paths (or iterate a fixed
//! enum like `NationalityBucket::ALL`), so embed/filesystem ordering
//! can never leak into simulation state.

use std::path::PathBuf;

/// The example content pack compiled into the binary. Rooted at
/// `crates/simn-sim/content/`. Files resolve as `&'static` bytes.
static EMBEDDED_PACK: include_dir::Dir<'static> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/../../content");

/// Where the sim reads its content pack from. Cheap to clone (the
/// `Dir`/`Overlay` variants hold a `PathBuf`; `Embedded` is a unit).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ContentSource {
    /// The pack embedded into the binary at compile time.
    #[default]
    Embedded,
    /// A complete on-disk content directory. Logical paths resolve
    /// relative to this root; a missing file is an error (no
    /// fallback). Use for a self-contained full pack.
    Dir(PathBuf),
    /// On-disk files layered over the embedded base: a logical path
    /// resolves to the on-disk file if present, otherwise falls back
    /// to [`ContentSource::Embedded`]. Lets a game ship only the
    /// files it overrides (e.g. its own `factions.toml`, `names/**`,
    /// `chatter_lines.toml`) and inherit all mechanics/items from the
    /// embedded base. This is how a consuming game supplies its own
    /// proprietary creative content while SIMN ships a generic example base.
    Overlay(PathBuf),
}

/// Error resolving a content file. Hand-rolled to avoid a `thiserror`
/// dependency, matching `faction::registry::RegistryError`.
#[derive(Debug)]
pub enum ContentError {
    /// The logical path was not present in the pack.
    NotFound(String),
    /// The file existed but could not be read.
    Io {
        path: String,
        source: std::io::Error,
    },
    /// The file existed but was not valid UTF-8.
    Utf8(String),
}

impl std::fmt::Display for ContentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentError::NotFound(p) => write!(f, "content file not found: {p}"),
            ContentError::Io { path, source } => {
                write!(f, "content read failed for {path}: {source}")
            }
            ContentError::Utf8(p) => write!(f, "content file {p} is not valid UTF-8"),
        }
    }
}

impl std::error::Error for ContentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ContentError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl ContentSource {
    /// Resolve a logical path (e.g. `"items/weapons.toml"`) to its
    /// UTF-8 text.
    pub fn read_str(&self, logical: &str) -> Result<String, ContentError> {
        match self {
            ContentSource::Embedded => EMBEDDED_PACK
                .get_file(logical)
                .ok_or_else(|| ContentError::NotFound(logical.to_string()))?
                .contents_utf8()
                .map(str::to_owned)
                .ok_or_else(|| ContentError::Utf8(logical.to_string())),
            ContentSource::Dir(root) => {
                let p = root.join(logical);
                std::fs::read_to_string(&p).map_err(|e| match e.kind() {
                    std::io::ErrorKind::NotFound => ContentError::NotFound(logical.to_string()),
                    _ => ContentError::Io {
                        path: p.display().to_string(),
                        source: e,
                    },
                })
            }
            ContentSource::Overlay(dir) => {
                let p = dir.join(logical);
                match std::fs::read_to_string(&p) {
                    Ok(s) => Ok(s),
                    // Not in the overlay → inherit from the embedded base.
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        ContentSource::Embedded.read_str(logical)
                    }
                    Err(e) => Err(ContentError::Io {
                        path: p.display().to_string(),
                        source: e,
                    }),
                }
            }
        }
    }

    /// Like [`Self::read_str`] but returns `None` for a missing file
    /// (other errors still propagate). Used by loaders that treat a
    /// missing file as "empty registry" (loot pools / containers)
    /// rather than a hard failure.
    pub fn read_str_opt(&self, logical: &str) -> Result<Option<String>, ContentError> {
        match self.read_str(logical) {
            Ok(s) => Ok(Some(s)),
            Err(ContentError::NotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Test-only: extract the embedded pack to a directory so tests
    /// can exercise the `Dir` resolver against identical content.
    #[doc(hidden)]
    pub fn extract_embedded_to(dir: &std::path::Path) -> std::io::Result<()> {
        EMBEDDED_PACK.extract(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_overrides_present_and_falls_back_for_missing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "OVERLAY").unwrap();
        let src = ContentSource::Overlay(dir.path().to_path_buf());

        // Present in the overlay → overlay wins.
        assert_eq!(src.read_str("marker.txt").unwrap(), "OVERLAY");

        // Absent from the overlay → falls back to the embedded base.
        let embedded = ContentSource::Embedded
            .read_str("world/world_time.toml")
            .unwrap();
        assert_eq!(src.read_str("world/world_time.toml").unwrap(), embedded);

        // Absent from both → NotFound.
        assert!(matches!(
            src.read_str("does/not/exist.toml"),
            Err(ContentError::NotFound(_))
        ));
    }
}
