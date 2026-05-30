//! Live dev-watcher for terrain assets.
//!
//! Run from the workspace root and leave running while you work:
//! ```text
//! cargo run --example terrain_watch -p simn-terrain --release
//! ```
//!
//! What it does:
//! - Recursively watches `godot/assets/terrain/` for modifications to
//!   any `heightmap.png` file.
//! - On every save (debounced ~300ms), runs the
//!   `png_to_canonical` sync for the affected map — rewrites
//!   `heightmap.r32` + refreshes `terrain.toml`'s `blake3`.
//! - Logs success + observed grid dims / spacing / vert range.
//!
//! Pair with Godot's extension hot-reload: save in Blender →
//! watcher updates `.r32` → Godot re-loads the heightmap on next
//! scene focus. You iterate on terrain detail without any manual
//! CLI invocations.
//!
//! Doesn't watch the other direction (`.r32` → `.png`). A re-bake
//! by `bake_corbett` or similar is rare + user-initiated; run
//! `canonical_to_png` by hand if you want Blender's view refreshed
//! from a new DEM import.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use notify::{event::EventKind, RecursiveMode, Watcher};
use simn_terrain::sync_png_to_canonical;

/// How long after the *last* event for a given map to wait before
/// actually running the sync. Blender and most editors write
/// temp-file-then-rename or a burst of small writes; grouping them
/// avoids running the pipeline mid-save on a truncated file.
const DEBOUNCE: Duration = Duration::from_millis(300);

fn terrain_root() -> Result<PathBuf> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = crate_dir.join("../../godot/assets/terrain");
    let canonical = path
        .canonicalize()
        .map_err(|e| anyhow!("terrain root {} is not accessible: {}", path.display(), e))?;
    Ok(canonical)
}

fn main() -> Result<()> {
    let root = terrain_root()?;

    let (tx, rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;
    watcher.watch(&root, RecursiveMode::Recursive)?;

    println!("terrain_watch: watching {}", root.display());
    println!("  on heightmap.png change → auto-run sync_png_to_canonical");
    println!("  Ctrl-C to stop.\n");

    // One pending timestamp per map directory. Latest save wins.
    let mut pending: HashMap<PathBuf, Instant> = HashMap::new();

    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok(event)) => {
                if !matches!(
                    event.kind,
                    EventKind::Modify(_) | EventKind::Create(_) | EventKind::Any
                ) {
                    continue;
                }
                for path in &event.paths {
                    if path.file_name().and_then(|n| n.to_str()) == Some("heightmap.png") {
                        if let Some(dir) = path.parent() {
                            pending.insert(dir.to_owned(), Instant::now());
                        }
                    }
                }
            }
            Ok(Err(e)) => eprintln!("watch error: {e}"),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Flush any pending maps whose debounce window closed.
                let now = Instant::now();
                let ready: Vec<PathBuf> = pending
                    .iter()
                    .filter(|(_, t)| now.duration_since(**t) >= DEBOUNCE)
                    .map(|(p, _)| p.clone())
                    .collect();
                for dir in ready {
                    pending.remove(&dir);
                    let map_id = map_id_of(&dir);
                    print!("[{map_id}] png → r32 … ");
                    match sync_png_to_canonical(&dir) {
                        Ok(report) => {
                            let resize_note = if report.resized {
                                format!(" (resized → {}×{})", report.width, report.height)
                            } else {
                                String::new()
                            };
                            println!(
                                "OK — {}×{} @ {:.2}m, vert [{:.1}, {:.1}]m{}",
                                report.width,
                                report.height,
                                report.spacing_m,
                                report.vert_min_m,
                                report.vert_max_m,
                                resize_note
                            );
                        }
                        Err(e) => println!("FAILED: {e}"),
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    Ok(())
}

fn map_id_of(dir: &Path) -> &str {
    dir.file_name().and_then(|n| n.to_str()).unwrap_or("?")
}
