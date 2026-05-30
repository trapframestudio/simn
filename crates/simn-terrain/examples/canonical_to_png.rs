//! Export a canonical `heightmap.r32` to a 16-bit grayscale PNG so
//! it can be opened in Blender (or any image tool) for authoring.
//!
//! Run from the workspace root:
//! ```text
//! cargo run --example canonical_to_png -p simn-terrain --release -- <map_id>
//! ```
//! `<map_id>` defaults to `corbett`. Lossy at the 16-bit boundary
//! (canonical f32 meters quantize to u16 against `vert_min_m`/
//! `vert_max_m`). After editing, `png_to_canonical` pushes changes
//! back. For an always-on watcher, see the `terrain_watch` example.

use std::path::PathBuf;

use anyhow::Result;
use simn_terrain::sync_canonical_to_png;

fn main() -> Result<()> {
    let map_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corbett".to_string());
    let dir = {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crate_dir.join(format!("../../godot/assets/terrain/{}", map_id))
    };

    let report = sync_canonical_to_png(&dir)?;
    println!(
        "wrote {}/heightmap.png ({}×{}, 16-bit grayscale)",
        dir.display(),
        report.width,
        report.height
    );
    println!(
        "  elevation mapping: pixel 0 → {} m, pixel 65535 → {} m",
        report.vert_min_m, report.vert_max_m
    );
    Ok(())
}
