//! Generic map baker driven by spec files under `tools/bakes/`.
//!
//! Replaces the per-map `bake_corbett.rs` style of the earlier
//! iteration. Each map owns a single TOML file describing its UTM
//! bounds, grid spacing, and DEM source; this CLI reads it, fetches
//! source data (SRTM tile download + gunzip if needed), samples, and
//! writes the canonical `heightmap.r32` + `terrain.toml` + scene
//! skeleton.
//!
//! Usage:
//! ```text
//! cargo run --example terrain_bake -p simn-terrain --release -- <map_id>
//! ```
//!
//! Layout:
//! - Spec (committed):  `tools/bakes/<map_id>.toml`
//! - Asset output:      `godot/assets/terrain/<map_id>/{heightmap.r32, terrain.toml}`
//! - Scene output:      `godot/scenes/maps/<map_id>.tscn` (once)
//! - DEM cache:         `/tmp/noosphere-dem/` (auto-populated)

use std::path::PathBuf;

use anyhow::{Context, Result};
use simn_terrain::{bake_map, load_spec};

fn main() -> Result<()> {
    let map_id = std::env::args()
        .nth(1)
        .context("expected <map_id> argument (e.g. `corbett`)")?;

    let workspace = {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crate_dir
            .join("../..")
            .canonicalize()
            .context("resolving workspace root")?
    };

    let spec_path = workspace.join(format!("tools/bakes/{}.toml", map_id));
    let spec = load_spec(&spec_path)?;

    if spec.map_id != map_id {
        anyhow::bail!(
            "spec file's map_id ({:?}) doesn't match CLI argument ({:?})",
            spec.map_id,
            map_id
        );
    }

    let asset_dir = workspace.join(format!("godot/assets/terrain/{}", map_id));
    let scene_path = workspace.join(format!("godot/scenes/maps/{}.tscn", map_id));
    let dem_cache = PathBuf::from("/tmp/noosphere-dem");

    println!("baking {} from {}", spec.map_id, spec.source.label());
    println!(
        "  UTM {} NW: ({:.1} E, {:.1} N)",
        spec.bounds.utm_zone, spec.bounds.origin_east, spec.bounds.origin_north
    );
    let aligned_x = spec.bounds.aligned_extent_x();
    let aligned_z = spec.bounds.aligned_extent_z();
    if !spec.bounds.extent_x_was_aligned() || !spec.bounds.extent_z_was_aligned() {
        println!(
            "  extent {:.0} × {:.0} m → aligned {:.0} × {:.0} m \
             (region {} m) @ {} m spacing → {}×{} samples",
            spec.bounds.extent_x,
            spec.bounds.extent_z,
            aligned_x,
            aligned_z,
            spec.bounds.region_size_m(),
            spec.bounds.spacing,
            spec.bounds.width(),
            spec.bounds.height()
        );
    } else {
        println!(
            "  extent {:.0} m E-W × {:.0} m N-S @ {} m spacing → {}×{} samples",
            aligned_x,
            aligned_z,
            spec.bounds.spacing,
            spec.bounds.width(),
            spec.bounds.height()
        );
    }

    let report = bake_map(&spec, &asset_dir, &scene_path, &dem_cache)?;

    println!(
        "  observed elevation [{:.2}, {:.2}] m → stored [{:.1}, {:.1}] m",
        report.observed_y_min, report.observed_y_max, report.vert_min_m, report.vert_max_m
    );
    println!("  wrote {}", report.asset_dir.display());
    if report.scene_created {
        println!("  generated scene {}", report.scene_path.display());
    } else {
        println!(
            "  scene {} already exists — preserved",
            report.scene_path.display()
        );
    }

    Ok(())
}
