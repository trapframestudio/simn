//! Conversion helpers for the canonical `heightmap.r32` ↔ 16-bit
//! grayscale PNG interchange. Used by the one-shot CLI examples
//! (`canonical_to_png`, `png_to_canonical`) and the live watcher
//! (`terrain_watch`).
//!
//! The `.r32` is the source of truth the game reads; the PNG is a
//! transient exchange file for Blender / image editors. The PNG path
//! is **lossy at the 16-bit boundary** — f32 meters are quantized to
//! u16 against `vert_min_m`/`vert_max_m` for export, then decoded
//! back to f32 on re-import. Round-trips through PNG cap at u16
//! precision; round-trips that stay in f32 (canonical → editor →
//! canonical) are bit-exact.
//!
//! Integrity model:
//! - `sync_canonical_to_png` is read-only on the `.r32` side.
//! - `sync_png_to_canonical` rewrites both `.r32` and `terrain.toml`
//!   (the TOML's `blake3` digest is refreshed from the new byte
//!   content). If the PNG's resolution differs, `width`/`height` on
//!   the TOML update and `spacing_m` recomputes so the map's
//!   *physical* extent stays the same.
//!
//! Vertical range (`vert_min_m` / `vert_max_m`) is never touched.
//! Callers who sculpt taller peaks must bump those by hand first
//! (otherwise the f32 → u16 quantize on export will clip).

use std::fs;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use png::{BitDepth, ColorType, Decoder, Encoder};

use crate::metadata::TerrainMetadata;
use crate::sampler::encode_r32;
use crate::Heightmap;

/// Per-call report returned by the sync helpers. Useful for logging.
#[derive(Debug, Clone, Copy)]
pub struct SyncReport {
    pub width: u32,
    pub height: u32,
    pub spacing_m: f32,
    pub vert_min_m: f32,
    pub vert_max_m: f32,
    /// True iff the sync changed `width` / `height` / `spacing_m`
    /// (i.e. Blender subdivided or decimated the grid).
    pub resized: bool,
}

/// Dump `<dir>/heightmap.r32` to `<dir>/heightmap.png` (16-bit
/// grayscale). Lossy at the 16-bit boundary; canonical f32 meters
/// are quantized via `vert_min_m`/`vert_max_m`. Does not modify
/// anything else.
pub fn sync_canonical_to_png(dir: &Path) -> Result<SyncReport> {
    let heightmap = Heightmap::load(dir)
        .with_context(|| format!("loading heightmap from {}", dir.display()))?;
    let md = heightmap.metadata().clone();

    let r32_path = dir.join("heightmap.r32");
    let r32_bytes =
        fs::read(&r32_path).with_context(|| format!("reading {}", r32_path.display()))?;
    let samples_f: Vec<f32> = r32_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // f32 meters → u16 PNG via vert_min/max linear scaling. PNG
    // 16-bit grayscale stores pixels big-endian on wire.
    let span = (md.vert_max_m - md.vert_min_m).max(1.0e-6);
    let mut be_bytes = Vec::with_capacity(samples_f.len() * 2);
    for &m in &samples_f {
        let t = ((m - md.vert_min_m) / span).clamp(0.0, 1.0);
        let s = (t * 65535.0).round() as u16;
        be_bytes.extend_from_slice(&s.to_be_bytes());
    }

    let png_path = dir.join("heightmap.png");
    let file =
        fs::File::create(&png_path).with_context(|| format!("creating {}", png_path.display()))?;
    let mut encoder = Encoder::new(BufWriter::new(file), md.width, md.height);
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(BitDepth::Sixteen);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&be_bytes)?;

    Ok(SyncReport {
        width: md.width,
        height: md.height,
        spacing_m: md.spacing_m,
        vert_min_m: md.vert_min_m,
        vert_max_m: md.vert_max_m,
        resized: false,
    })
}

/// Push `<dir>/heightmap.png` (16-bit grayscale) back into
/// `<dir>/heightmap.r32` and refresh `terrain.toml`. Decodes u16
/// samples to f32 meters via `vert_min_m`/`vert_max_m`.
///
/// Errors cleanly on: missing files, non-grayscale PNG, non-16-bit
/// PNG, or a resolution change where X and Z scale factors don't
/// match (caller either keeps resolution fixed or scales uniformly).
pub fn sync_png_to_canonical(dir: &Path) -> Result<SyncReport> {
    let png_path = dir.join("heightmap.png");
    let toml_path = dir.join("terrain.toml");
    let r32_path = dir.join("heightmap.r32");

    let file =
        fs::File::open(&png_path).with_context(|| format!("opening {}", png_path.display()))?;
    let decoder = Decoder::new(file);
    let mut reader = decoder.read_info()?;
    let info = reader.info().clone();

    if info.color_type != ColorType::Grayscale {
        return Err(anyhow!(
            "PNG must be grayscale (got {:?}). Flatten to a single luminance channel before importing.",
            info.color_type
        ));
    }
    if info.bit_depth != BitDepth::Sixteen {
        return Err(anyhow!(
            "PNG must be 16-bit (got {:?}). Blender: Image Editor → Save As → Color depth 16.",
            info.bit_depth
        ));
    }

    let mut buf = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut buf)?;
    let samples_u16: Vec<u16> = buf
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(
        samples_u16.len(),
        (info.width * info.height) as usize,
        "PNG decoder produced wrong sample count"
    );

    let mut md: TerrainMetadata = toml::from_str(
        &fs::read_to_string(&toml_path)
            .with_context(|| format!("reading {}", toml_path.display()))?,
    )?;

    let mut resized = false;
    if info.width != md.width || info.height != md.height {
        let old_extent_x = (md.width as f32 - 1.0) * md.spacing_m;
        let old_extent_z = (md.height as f32 - 1.0) * md.spacing_m;
        let new_spacing_x = old_extent_x / (info.width as f32 - 1.0);
        let new_spacing_z = old_extent_z / (info.height as f32 - 1.0);
        if (new_spacing_x - new_spacing_z).abs() > 1e-4 {
            return Err(anyhow!(
                "non-uniform resolution change: X {}→{} gives {}m spacing, Z {}→{} gives {}m. \
                 Edit PNG so both axes scale by the same factor, or update terrain.toml by hand.",
                md.width,
                info.width,
                new_spacing_x,
                md.height,
                info.height,
                new_spacing_z,
            ));
        }
        md.width = info.width;
        md.height = info.height;
        md.spacing_m = new_spacing_x;
        resized = true;
    }

    // u16 PNG → f32 meters via vert_min/max linear scaling.
    let span = md.vert_max_m - md.vert_min_m;
    let samples_f: Vec<f32> = samples_u16
        .iter()
        .map(|&s| {
            let t = f32::from(s) / 65535.0;
            md.vert_min_m + t * span
        })
        .collect();

    let bytes = encode_r32(&samples_f);
    fs::write(&r32_path, &bytes).with_context(|| format!("writing {}", r32_path.display()))?;
    md.blake3 = blake3::hash(&bytes).to_hex().to_string();
    fs::write(&toml_path, toml::to_string(&md)?)
        .with_context(|| format!("writing {}", toml_path.display()))?;

    // Re-validate through the public loader so any mismatch surfaces
    // now rather than on the next scene load.
    let hm = Heightmap::load(dir)?;

    Ok(SyncReport {
        width: hm.width(),
        height: hm.height(),
        spacing_m: hm.metadata().spacing_m,
        vert_min_m: hm.metadata().vert_min_m,
        vert_max_m: hm.metadata().vert_max_m,
        resized,
    })
}
