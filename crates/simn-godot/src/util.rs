//! Editor-side utilities exposed to GDScript.
//!
//! Currently hosts [`TerrainHash`], a tiny `RefCounted` that
//! recomputes BLAKE3 digests over canonical heightmap files. The
//! `Sync to Canonical` flow uses it to refresh `terrain.toml`'s
//! `blake3` field after writing a new `heightmap.r32` so the server-
//! side integrity check stays live (rather than relying on the
//! empty-string skip).

use godot::classes::ProjectSettings;
use godot::prelude::*;
use std::fs;

/// Editor-only helper class for computing BLAKE3 digests of asset
/// files. Lives as a `RefCounted` so GDScript can `.new()` it
/// without scene-tree ceremony.
///
/// Usage from GDScript:
/// ```gdscript
/// var hasher := TerrainHash.new()
/// var hex := hasher.blake3_file("res://assets/terrain/cascade_locks/heightmap.r32")
/// if hex.is_empty():
///     push_error("hash failed")
/// ```
///
/// Returns the lowercase hex digest, or an empty string on read
/// failure (with a `godot_error!` logged).
#[derive(GodotClass)]
#[class(tool, init, base = RefCounted)]
pub struct TerrainHash {
    base: Base<RefCounted>,
}

#[godot_api]
impl TerrainHash {
    /// Hash the given file's contents with BLAKE3 and return the
    /// lowercase hex digest. Accepts either a `res://` virtual path
    /// (resolved via `ProjectSettings::globalize_path`) or an
    /// absolute OS path.
    #[func]
    fn blake3_file(&self, path: GString) -> GString {
        let resolved = resolve_path(&path);
        match fs::read(&resolved) {
            Ok(bytes) => {
                let hex = blake3::hash(&bytes).to_hex().to_string();
                GString::from(&hex)
            }
            Err(e) => {
                godot_error!("TerrainHash: failed to read {resolved}: {e}");
                GString::new()
            }
        }
    }
}

/// Translate a `res://...` path to an absolute OS path; pass-through
/// for everything else (including absolute paths and `user://`,
/// which the editor flow doesn't need but we surface cleanly).
pub(crate) fn resolve_path(path: &GString) -> String {
    let s = path.to_string();
    if s.starts_with("res://") || s.starts_with("user://") {
        ProjectSettings::singleton()
            .globalize_path(path)
            .to_string()
    } else {
        s
    }
}
