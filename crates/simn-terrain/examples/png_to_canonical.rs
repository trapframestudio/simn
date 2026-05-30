//! Import a Blender-authored 16-bit grayscale PNG back into the
//! canonical `heightmap.r32` and refresh `terrain.toml`.
//!
//! Run from the workspace root:
//! ```text
//! cargo run --example png_to_canonical -p simn-terrain --release -- <map_id>
//! ```
//! `<map_id>` defaults to `corbett`.
//!
//! Lossy at the 16-bit boundary (PNG u16 → f32 meters via
//! `vert_min_m`/`vert_max_m` linear scaling). For continuous sync
//! while you work in Blender, use `terrain_watch`.

use std::path::PathBuf;

use anyhow::Result;
use simn_terrain::sync_png_to_canonical;

fn main() -> Result<()> {
    let map_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corbett".to_string());
    let dir = {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crate_dir.join(format!("../../godot/assets/terrain/{}", map_id))
    };

    let report = sync_png_to_canonical(&dir)?;
    if report.resized {
        println!(
            "  resolution change detected: now {}×{} @ {:.4}m spacing",
            report.width, report.height, report.spacing_m
        );
    }
    println!(
        "wrote heightmap.r32 ({}×{} @ {:.2}m), vert range [{:.1}, {:.1}] m",
        report.width, report.height, report.spacing_m, report.vert_min_m, report.vert_max_m
    );
    println!("  terrain.toml blake3 refreshed");
    Ok(())
}
