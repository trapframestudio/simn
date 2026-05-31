//! Mapzen Terrarium tile elevation source.
//!
//! Terrarium tiles are 256×256 PNG tiles served on AWS at arbitrary
//! zoom levels. Each pixel encodes elevation in the RGB channels:
//!
//! ```text
//! elevation_m = (R * 256 + G + B / 256) - 32768
//! ```
//!
//! The tiles are produced by compositing every public DEM source
//! Mapzen could get their hands on — SRTM globally, with USGS NED
//! / 3DEP 1-meter LIDAR merged in where available. That makes the
//! effective resolution *better than SRTM* across most of CONUS
//! without us having to do tile-by-tile survey discovery: we just
//! pick a zoom level and the pipeline pulls whatever's underneath.
//!
//! Zoom / effective resolution at lat 45°:
//!
//! | zoom | tile extent | sample spacing |
//! |------|-------------|----------------|
//! | 12   | ~9.8 km     | ~38 m          |
//! | 13   | ~4.9 km     | ~19 m          |
//! | 14   | ~2.4 km     | ~10 m          |
//! | 15   | ~1.2 km     | ~5 m           |
//!
//! For a 2 m output grid, zoom 15 (~5 m native) is a 2–3× improvement
//! over SRTM's 30 m native. **z15 is the ceiling** — the public
//! Mapzen Terrarium bucket does not serve z16+ tiles in PNW. For
//! sub-5 m detail we'd need the USGS 3DEP 1 m pipeline (task #49).
//! Tiles are ~50 KB each; a typical 5 km × 4 km SIMN map needs
//! ~15–25 tiles at zoom 15 (~1 MB download, < 10 s).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};

pub const TILE_SIZE: usize = 256;

/// One decoded Terrarium tile — 256 × 256 elevation samples in meters.
pub struct TerrariumTile {
    pub zoom: u8,
    pub x: u32,
    pub y: u32,
    /// Row-major elevations, north-up. Index 0 is the NW corner.
    pub data: Vec<f32>,
}

impl TerrariumTile {
    /// Sample the elevation at a fractional raster coordinate inside
    /// this tile. `u` / `v` are 0..TILE_SIZE with (0,0) at the NW
    /// corner. Bilinear interpolation.
    pub fn sample_uv(&self, u: f64, v: f64) -> f32 {
        let last = (TILE_SIZE - 1) as f64;
        let u = u.clamp(0.0, last);
        let v = v.clamp(0.0, last);
        let x0 = u.floor() as usize;
        let y0 = v.floor() as usize;
        let x1 = (x0 + 1).min(TILE_SIZE - 1);
        let y1 = (y0 + 1).min(TILE_SIZE - 1);
        let fx = (u - x0 as f64) as f32;
        let fy = (v - y0 as f64) as f32;
        let at = |r: usize, c: usize| self.data[r * TILE_SIZE + c];
        let h00 = at(y0, x0);
        let h01 = at(y0, x1);
        let h10 = at(y1, x0);
        let h11 = at(y1, x1);
        let top = h00 * (1.0 - fx) + h01 * fx;
        let bot = h10 * (1.0 - fx) + h11 * fx;
        top * (1.0 - fy) + bot * fy
    }
}

// ---------------------------------------------------------------------------
// TMS math — WGS84 ↔ zoom/x/y
// ---------------------------------------------------------------------------

/// Convert WGS84 (lat, lon) degrees to Terrarium tile coordinates
/// at a given zoom. Returns fractional `(tile_x, tile_y)` so the
/// caller can derive both the integer tile id and the sub-tile
/// sample position in one pass.
pub fn lonlat_to_tile_frac(lat_deg: f64, lon_deg: f64, zoom: u8) -> (f64, f64) {
    let n = (1u64 << zoom) as f64;
    let lat_rad = lat_deg.to_radians();
    let x = (lon_deg + 180.0) / 360.0 * n;
    let y = (1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n;
    (x, y)
}

// ---------------------------------------------------------------------------
// Tile fetch + decode
// ---------------------------------------------------------------------------

/// Ensure a single Terrarium tile exists on disk. Small — tiles are
/// ~50 KB each, so we eat the curl startup cost per-tile rather than
/// batch-requesting (simpler, and the S3 front-end handles hundreds
/// of parallel requests fine).
pub fn ensure_terrarium_tile(zoom: u8, x: u32, y: u32, cache_dir: &Path) -> Result<PathBuf> {
    let dir = cache_dir.join(format!("terrarium/{zoom}/{x}"));
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{y}.png"));
    if path.exists() {
        return Ok(path);
    }
    let url = format!("https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{zoom}/{x}/{y}.png");
    let status = Command::new("curl")
        .arg("-fsSL")
        .arg("--max-time")
        .arg("30")
        .arg("-o")
        .arg(&path)
        .arg(&url)
        .status()
        .context("running curl for Terrarium tile")?;
    if !status.success() {
        let _ = fs::remove_file(&path);
        return Err(anyhow!("curl failed fetching {url}"));
    }
    Ok(path)
}

/// Decode a Terrarium PNG into 256×256 floats (meters above sea
/// level). Uses the `png` crate already in the simn-terrain dep list
/// for the r16 round-trip path.
pub fn read_terrarium_tile(path: &Path, zoom: u8, x: u32, y: u32) -> Result<TerrariumTile> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder
        .read_info()
        .with_context(|| format!("reading PNG info from {}", path.display()))?;
    let info = reader.info();
    if info.width != TILE_SIZE as u32 || info.height != TILE_SIZE as u32 {
        return Err(anyhow!(
            "terrarium tile {} has unexpected size {}x{}",
            path.display(),
            info.width,
            info.height
        ));
    }
    let color_type = info.color_type;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    reader
        .next_frame(&mut buf)
        .with_context(|| format!("decoding PNG frame {}", path.display()))?;

    let pixels = TILE_SIZE * TILE_SIZE;
    let mut data = Vec::with_capacity(pixels);
    let bpp: usize = match color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => return Err(anyhow!("unexpected PNG color type {:?}", other)),
    };
    for i in 0..pixels {
        let r = buf[i * bpp] as f32;
        let g = buf[i * bpp + 1] as f32;
        let b = buf[i * bpp + 2] as f32;
        // Terrarium encoding: elevation = (R*256 + G + B/256) - 32768
        let elevation = (r * 256.0 + g + b / 256.0) - 32768.0;
        data.push(elevation);
    }
    Ok(TerrariumTile { zoom, x, y, data })
}

/// Sample the Terrarium mosaic at a WGS84 point. **Cross-tile
/// bilinear** — finds the 4 surrounding pixels in global pixel
/// space and bilinear-blends them, even when the 4 pixels span
/// tile boundaries.
///
/// This matters for visible seams. The previous in-tile-only
/// sampling clamped at tile edges, so a bilinear at fx ≈ 5290.999
/// would average 4 in-tile-5290 neighbors and skip tile 5291's
/// edge pixel entirely. Mapzen's encoding has ~0.1 m precision
/// rounding per pixel; without cross-tile interpolation, the
/// rounding bias on each tile produces a visible elevation
/// discontinuity at every 150 m tile boundary at zoom 15.
///
/// Returns `None` if any of the 4 surrounding tiles isn't loaded
/// — caller is responsible for prefetching with a 1-tile buffer
/// around the bbox.
pub fn sample_mosaic(tiles: &[TerrariumTile], lat: f64, lon: f64, zoom: u8) -> Option<f32> {
    let (fx, fy) = lonlat_to_tile_frac(lat, lon, zoom);
    let gx = fx * TILE_SIZE as f64;
    let gy = fy * TILE_SIZE as f64;
    let x0 = gx.floor();
    let y0 = gy.floor();
    let dx = (gx - x0) as f32;
    let dy = (gy - y0) as f32;
    let h00 = sample_pixel_global(tiles, x0 as i64, y0 as i64, zoom)?;
    let h10 = sample_pixel_global(tiles, x0 as i64 + 1, y0 as i64, zoom)?;
    let h01 = sample_pixel_global(tiles, x0 as i64, y0 as i64 + 1, zoom)?;
    let h11 = sample_pixel_global(tiles, x0 as i64 + 1, y0 as i64 + 1, zoom)?;
    let top = h00 * (1.0 - dx) + h10 * dx;
    let bot = h01 * (1.0 - dx) + h11 * dx;
    Some(top * (1.0 - dy) + bot * dy)
}

/// Look up a single pixel at global pixel coordinates `(gx, gy)`
/// across the entire zoom-level mosaic. The pixel's tile is
/// `(gx / TILE_SIZE, gy / TILE_SIZE)`; its in-tile offset is the
/// remainder. Returns `None` when the containing tile isn't loaded
/// or when the global coordinate is negative (off-mosaic).
fn sample_pixel_global(tiles: &[TerrariumTile], gx: i64, gy: i64, zoom: u8) -> Option<f32> {
    if gx < 0 || gy < 0 {
        return None;
    }
    let tile_size = TILE_SIZE as i64;
    let tx = (gx / tile_size) as u32;
    let ty = (gy / tile_size) as u32;
    let in_tile_x = (gx % tile_size) as usize;
    let in_tile_y = (gy % tile_size) as usize;
    for t in tiles {
        if t.zoom == zoom && t.x == tx && t.y == ty {
            return Some(t.data[in_tile_y * TILE_SIZE + in_tile_x]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_math_roundtrips_for_corbett() {
        // Corbett at ~(45.54, -122.41) should land inside a specific
        // tile at zoom 15. Sanity-check that lonlat_to_tile_frac gives
        // coordinates in the valid range.
        let (fx, fy) = lonlat_to_tile_frac(45.54, -122.41, 15);
        assert!(fx > 0.0 && fx < (1u64 << 15) as f64);
        assert!(fy > 0.0 && fy < (1u64 << 15) as f64);
    }

    #[test]
    fn prime_meridian_at_zoom_zero_is_half() {
        let (fx, _) = lonlat_to_tile_frac(0.0, 0.0, 0);
        assert!((fx - 0.5).abs() < 1e-9);
    }
}
