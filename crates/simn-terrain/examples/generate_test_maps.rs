//! Generate synthetic heightmaps for the four test maps.
//!
//! Run from the workspace root:
//! ```text
//! cargo run --example generate_test_maps -p simn-terrain
//! ```
//! Writes into `godot/assets/terrain/{test_map_1..4}/`.
//! Safe to re-run; overwrites existing files.
//!
//! Each map is 5 km × 5 km at 4 m spacing (1250² samples ≈ 6 MB/map
//! at f32). Iteration 5-14: each map keeps its identifiable **macro
//! shape** (ramp / hills / basin / ridge) but a multi-octave
//! OpenSimplex noise overlay adds organic variation on top with a
//! per-map seed + amplitude. The smooth macro plus the noise reads
//! as a real outdoor environment — something NPCs can navigate
//! around, something the player can orient by — rather than the
//! glossy ramps the v1 generator produced.

use std::f32::consts::TAU;
use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use noise::{NoiseFn, OpenSimplex};
use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
use simn_terrain::sampler::encode_r32;
use simn_terrain::TerrainMetadata;

const MAP_EXTENT_M: f32 = 5000.0;
const SPACING_M: f32 = 4.0;
const WIDTH: u32 = (MAP_EXTENT_M / SPACING_M) as u32; // 1250
const HEIGHT: u32 = WIDTH;
/// Baseline ceiling. The noise overlay can push individual samples
/// slightly above this — the actual max we write into `terrain.toml`
/// is computed from the data, not from this constant.
const VERT_MAX_BASELINE_M: f32 = 200.0;

/// Layered Simplex-noise overlay. Adds organic variation to a base
/// heightmap in-place. Multi-octave fBm (frequency doubles per octave,
/// amplitude scales by `persistence`); the summed value is normalized
/// to roughly [-1, 1] then multiplied by `amp_m`. Final samples are
/// clamped at 0 so the overlay can't push the surface below sea level
/// in basin regions.
///
/// `base_freq_per_m` is in cycles per meter — `1.0 / 300.0` means the
/// largest noise feature has a wavelength of ~300 m, which reads as a
/// "rolling hills" scale at the 5 km map size.
fn add_noise_overlay(
    samples: &mut [f32],
    seed: u32,
    octaves: u32,
    base_freq_per_m: f64,
    amp_m: f32,
    persistence: f64,
) {
    let noise = OpenSimplex::new(seed);

    // Per-octave amplitudes sum to determine the un-normalized peak.
    let mut peak_sum = 0.0f64;
    let mut a = 1.0f64;
    for _ in 0..octaves {
        peak_sum += a;
        a *= persistence;
    }
    let normalize = 1.0 / peak_sum;

    for z_idx in 0..HEIGHT {
        for x_idx in 0..WIDTH {
            let x_m = f64::from(x_idx) * f64::from(SPACING_M);
            let z_m = f64::from(z_idx) * f64::from(SPACING_M);
            let mut value = 0.0f64;
            let mut freq = base_freq_per_m;
            let mut amplitude = 1.0f64;
            for _ in 0..octaves {
                value += noise.get([x_m * freq, z_m * freq]) * amplitude;
                freq *= 2.0;
                amplitude *= persistence;
            }
            let overlay = (value * normalize) as f32 * amp_m;
            let idx = (z_idx * WIDTH + x_idx) as usize;
            samples[idx] = (samples[idx] + overlay).max(0.0);
        }
    }
}

/// map_1: gentle +X ramp (0 m at west edge, ~150 m at east edge) +
/// medium-scale noise. Reads as forested rolling hills with a gentle
/// eastward slope.
fn gen_ramp() -> Vec<f32> {
    let mut out = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    // Cap the macro at 150 m so the noise overlay (up to ~25 m) keeps
    // the final max comfortably under the legacy 200 m baseline.
    let macro_max_m = 150.0_f32;
    for _ in 0..HEIGHT {
        for x in 0..WIDTH {
            let t = x as f32 / (WIDTH - 1) as f32;
            out.push(t * macro_max_m);
        }
    }
    add_noise_overlay(&mut out, 0x5A_1A_CE_01, 5, 1.0 / 300.0, 25.0, 0.5);
    out
}

/// map_2: rolling hills — two overlapping sine waves on X and Z, plus
/// thick organic noise. Reads as classic rolling countryside.
fn gen_hills() -> Vec<f32> {
    let mut out = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    // Pull the macro down to ~120 m to leave headroom for 35 m of noise.
    let macro_max_m = 120.0_f32;
    for z_idx in 0..HEIGHT {
        for x_idx in 0..WIDTH {
            let x = x_idx as f32 * SPACING_M;
            let z = z_idx as f32 * SPACING_M;
            let a = (x * TAU / 1000.0).sin() * 0.5 + 0.5;
            let b = (z * TAU / 1700.0).sin() * 0.5 + 0.5;
            out.push((a + b) * 0.5 * macro_max_m);
        }
    }
    add_noise_overlay(&mut out, 0x5A_1A_CE_02, 6, 1.0 / 250.0, 35.0, 0.5);
    out
}

/// map_3: basin — low in the center, rises smoothly to the edges, with
/// subtle large-scale variation. Reads as a lake bed surrounded by
/// hills.
fn gen_basin() -> Vec<f32> {
    let cx = WIDTH as f32 / 2.0;
    let cz = HEIGHT as f32 / 2.0;
    let max_r = (cx * cx + cz * cz).sqrt();
    let mut out = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    // Macro tops out at 170 m on the edges; noise is gentler (20 m)
    // so the basin floor stays recognizable.
    let macro_max_m = 170.0_f32;
    for z_idx in 0..HEIGHT {
        for x_idx in 0..WIDTH {
            let dx = x_idx as f32 - cx;
            let dz = z_idx as f32 - cz;
            let r = (dx * dx + dz * dz).sqrt() / max_r;
            let t = (r * r).clamp(0.0, 1.0); // quadratic rise at edges
            out.push(t * macro_max_m);
        }
    }
    add_noise_overlay(&mut out, 0x5A_1A_CE_03, 4, 1.0 / 500.0, 20.0, 0.5);
    out
}

/// map_4: S-shaped ridge running roughly west-east, with side spurs
/// from fast-frequency noise. Reads as a dramatic ridge corridor.
fn gen_ridge() -> Vec<f32> {
    let mut out = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    // Ridge peaks at 130 m; noise (40 m) adds the spurs.
    let macro_max_m = 130.0_f32;
    let ridge_amp = HEIGHT as f32 * 0.15;
    let wavelength_samples = WIDTH as f32 * 0.6;
    let half_h = HEIGHT as f32 / 2.0;
    let falloff_scale = HEIGHT as f32 * 0.3;
    for z_idx in 0..HEIGHT {
        for x_idx in 0..WIDTH {
            let ridge_z = half_h + ridge_amp * (x_idx as f32 * TAU / wavelength_samples).sin();
            let d = (z_idx as f32 - ridge_z).abs();
            let falloff = (1.0 - (d / falloff_scale).clamp(0.0, 1.0)).powi(2);
            out.push(falloff * macro_max_m);
        }
    }
    add_noise_overlay(&mut out, 0x5A_1A_CE_04, 6, 1.0 / 200.0, 40.0, 0.5);
    out
}

fn write_map(dir: &std::path::Path, map_id: &str, samples: &[f32]) -> Result<()> {
    fs::create_dir_all(dir)?;
    let bytes = encode_r32(samples);
    let blake3_digest = blake3::hash(&bytes).to_hex().to_string();
    // Compute the actual height range from the data — the noise
    // overlay can push the peak above the macro baseline, and
    // downstream consumers (camera bounds, sky shader hints, the 16-
    // bit PNG inspection path) need the true range. `vert_min_m`
    // floors at 0 because `add_noise_overlay` clamps the surface
    // there.
    let (vert_min_m, vert_max_m) = samples
        .iter()
        .copied()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    // Guard against an all-NaN input (shouldn't happen but keeps the
    // toml writable).
    let vert_min_m = if vert_min_m.is_finite() {
        vert_min_m
    } else {
        0.0
    };
    let vert_max_m = if vert_max_m.is_finite() {
        vert_max_m
    } else {
        VERT_MAX_BASELINE_M
    };
    let metadata = TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: map_id.to_string(),
        width: WIDTH,
        height: HEIGHT,
        spacing_m: SPACING_M,
        vert_min_m,
        vert_max_m,
        origin_utm_zone: "10N".to_string(),
        origin_utm_easting: 0.0,
        origin_utm_northing: 0.0,
        blake3: blake3_digest,
        features_blake3: String::new(),
        region_size_m: 2048.0,
        playable_extent_x_m: 0.0,
        playable_extent_z_m: 0.0,
        nav_mask_format_version: 0,
        nav_mask_blake3: String::new(),
    };
    let toml_text = toml::to_string(&metadata)?;
    fs::write(dir.join("terrain.toml"), toml_text)?;
    fs::write(dir.join("heightmap.r32"), bytes)?;
    // Clean up a stale v1 .r16 if a previous generation left one alongside.
    let stale = dir.join("heightmap.r16");
    if stale.exists() {
        let _ = fs::remove_file(&stale);
    }
    println!(
        "wrote {:<12} ({}×{} @ {} m, range {:.1}–{:.1} m)",
        map_id, WIDTH, HEIGHT, SPACING_M, vert_min_m, vert_max_m,
    );
    Ok(())
}

fn main() -> Result<()> {
    let assets_root = {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crate_dir.join("../../godot/assets/terrain")
    };
    write_map(&assets_root.join("test_map_1"), "test_map_1", &gen_ramp())?;
    write_map(&assets_root.join("test_map_2"), "test_map_2", &gen_hills())?;
    write_map(&assets_root.join("test_map_3"), "test_map_3", &gen_basin())?;
    write_map(&assets_root.join("test_map_4"), "test_map_4", &gen_ridge())?;
    Ok(())
}
