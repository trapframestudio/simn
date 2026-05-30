//! USGS 3DEP 1-meter LIDAR DEM source.
//!
//! Upgrades the elevation pipeline from SRTM 1-arcsec (~30 m native,
//! bilinearly upsampled to our 2 m grid) to USGS 3DEP 1-meter LIDAR
//! (native 1 m, bilinearly downsampled to 2 m). Visible improvements:
//!
//! - Road cut/fill benches render as sharp flat surfaces instead of
//!   getting smoothed into the surrounding slope.
//! - Cliff edges gain real definition — the bluff edges in Corbett
//!   now terminate in a crisp break rather than a gradient.
//! - Gullies, drainages, and narrow ridges that were blurred away at
//!   30 m native show up as distinct features.
//! - Buildings + earthworks (powerline corridors, farm terraces) are
//!   visible where LIDAR penetrated the canopy.
//!
//! # Discovery
//!
//! [`discover_3dep_tiles`] hits The National Map (TNM) Access API
//! with the map's WGS84 bbox and returns the list of `.tif` tile URLs
//! that cover it. TNM returns multiple candidates per location (newer
//! projects + older surveys); we prefer the most recent.
//!
//! # Tile format
//!
//! USGS 3DEP 1 m tiles are GeoTIFF, Float32 grayscale, 10 000 × 10 000
//! samples per tile, stored in UTM (NAD83) coordinates of the tile's
//! own zone. Tile naming convention:
//!
//! ```text
//! USGS_1M_<zone>_x<X>y<Y>_<project>.tif
//! USGS_one_meter_x<X>y<Y>_<project>.tif    // older naming
//! ```
//!
//! where `X = east/10 000` and `Y = north_upper/10 000` — i.e. the
//! tile's NW corner is at UTM (`X*10 000`, `Y*10 000`) and it extends
//! 10 km south + 10 km east. (The zone is implicit in older names;
//! we read it from the GeoTIFF's own GeoKeys.)

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::Tag;

/// A loaded USGS 3DEP 1 m tile, ready for UTM-coordinate sampling.
pub struct Tile {
    /// UTM easting of the tile's SW corner (meters).
    pub east_min: f64,
    /// UTM northing of the tile's SW corner (meters).
    pub north_min: f64,
    /// Cell size in meters (typically exactly 1.0 for 3DEP 1 m).
    pub cell_m: f64,
    /// Width in samples.
    pub width: usize,
    /// Height in samples.
    pub height: usize,
    /// Row-major elevations in meters, north-up — index 0 is the NW corner.
    pub data: Vec<f32>,
    /// NODATA sentinel from the TIFF's GDAL_NODATA tag. Defaults to
    /// `-1.0e38` per USGS convention; samples equal to this are
    /// treated as gaps and skipped.
    pub nodata: f32,
}

impl Tile {
    pub fn east_max(&self) -> f64 {
        self.east_min + self.cell_m * self.width as f64
    }
    pub fn north_max(&self) -> f64 {
        self.north_min + self.cell_m * self.height as f64
    }

    pub fn contains_utm(&self, east: f64, north: f64) -> bool {
        east >= self.east_min
            && east < self.east_max()
            && north >= self.north_min
            && north < self.north_max()
    }

    /// Bilinear sample at a UTM point inside the tile. Returns `None`
    /// if the query lands on a NODATA cell (or any neighbor is
    /// NODATA).
    pub fn sample(&self, east: f64, north: f64) -> Option<f32> {
        if !self.contains_utm(east, north) {
            return None;
        }
        let u = (east - self.east_min) / self.cell_m;
        // Tile row 0 is the NORTH edge (north_max); rows increase southward.
        let v = (self.north_max() - north) / self.cell_m;
        let x0 = u.floor() as usize;
        let y0 = v.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = (u - x0 as f64) as f32;
        let fy = (v - y0 as f64) as f32;
        let at = |row: usize, col: usize| -> Option<f32> {
            let v = self.data[row * self.width + col];
            if (v - self.nodata).abs() < 1e-3 {
                None
            } else {
                Some(v)
            }
        };
        let h00 = at(y0, x0)?;
        let h01 = at(y0, x1)?;
        let h10 = at(y1, x0)?;
        let h11 = at(y1, x1)?;
        let top = h00 * (1.0 - fx) + h01 * fx;
        let bot = h10 * (1.0 - fx) + h11 * fx;
        Some(top * (1.0 - fy) + bot * fy)
    }
}

// ---------------------------------------------------------------------------
// TNM API discovery
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct TnmResponse {
    items: Vec<TnmItem>,
}

#[derive(Debug, Deserialize)]
struct TnmItem {
    title: String,
    #[serde(rename = "publicationDate")]
    publication_date: Option<String>,
    #[serde(rename = "downloadURL")]
    download_url: String,
}

/// WGS84 lat/lon bbox. (south, west, north, east).
#[derive(Debug, Clone, Copy)]
pub struct LatLonBbox {
    pub south: f64,
    pub west: f64,
    pub north: f64,
    pub east: f64,
}

/// Discover USGS 3DEP 1-meter tiles that cover `bbox`. Returns the
/// download URLs for the best candidates, deduplicated by tile name
/// and preferring the most recent project.
pub fn discover_3dep_tiles(bbox: LatLonBbox, cache_dir: &Path) -> Result<Vec<String>> {
    fs::create_dir_all(cache_dir)?;
    let cache_key = format!(
        "tnm-3dep-{:.4}_{:.4}_{:.4}_{:.4}.json",
        bbox.south, bbox.west, bbox.north, bbox.east,
    );
    let cache_file = cache_dir.join(&cache_key);

    let body = if cache_file.exists() {
        fs::read_to_string(&cache_file)?
    } else {
        let bbox_arg = format!("{},{},{},{}", bbox.west, bbox.south, bbox.east, bbox.north);
        let output = Command::new("curl")
            .arg("-fsSL")
            .arg("--max-time")
            .arg("60")
            .arg("-G")
            .arg("https://tnmaccess.nationalmap.gov/api/v1/products")
            .arg("--data-urlencode")
            .arg(format!("bbox={bbox_arg}"))
            .arg("--data-urlencode")
            .arg("datasets=Digital Elevation Model (DEM) 1 meter")
            .arg("--data-urlencode")
            .arg("prodFormats=GeoTIFF")
            .arg("--data-urlencode")
            .arg("max=200")
            .output()
            .context("running curl for TNM Access API")?;
        if !output.status.success() {
            return Err(anyhow!(
                "TNM API request failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let text = String::from_utf8(output.stdout)?;
        fs::write(&cache_file, &text)?;
        text
    };

    let resp: TnmResponse = serde_json::from_str(&body)
        .with_context(|| format!("parsing TNM response from {}", cache_file.display()))?;

    // Group by the tile identifier (x##y###) and keep the most recent
    // project per tile.
    use std::collections::BTreeMap;
    let mut best_per_tile: BTreeMap<String, (String, String)> = BTreeMap::new();
    for item in resp.items {
        let Some(tile_id) = extract_tile_id(&item.title) else {
            continue;
        };
        let date = item.publication_date.unwrap_or_default();
        match best_per_tile.get(&tile_id) {
            Some((prev_date, _)) if prev_date.as_str() >= date.as_str() => {}
            _ => {
                best_per_tile.insert(tile_id, (date, item.download_url));
            }
        }
    }
    Ok(best_per_tile.into_values().map(|(_, url)| url).collect())
}

/// Extract the `x##y###` tile identifier from a TNM product title.
fn extract_tile_id(title: &str) -> Option<String> {
    // Titles look like "USGS 1 Meter 10 x54y505 <project>" or
    // "USGS one meter x54y505 <project>".
    for token in title.split_whitespace() {
        if token.starts_with('x') && token.contains('y') {
            return Some(token.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tile download + read
// ---------------------------------------------------------------------------

/// Download a USGS 3DEP tile to `cache_dir` if not already present.
/// Returns the local path.
///
/// Uses `curl -C -` (resume partial downloads) + a long timeout so
/// the ~150-500 MB tiles complete on slower links without manual
/// retry. Caching makes subsequent bakes instant.
pub fn ensure_3dep_tile(url: &str, cache_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(cache_dir)?;
    let fname = url.rsplit('/').next().ok_or_else(|| anyhow!("empty URL"))?;
    let path = cache_dir.join(fname);
    if path.exists() {
        return Ok(path);
    }
    println!("  downloading {fname}… (may take 5-15 min for 3DEP 1 m tiles)");
    let output = Command::new("curl")
        .arg("-fSL")
        .arg("-C")
        .arg("-")
        .arg("--max-time")
        .arg("1800")
        .arg("--retry")
        .arg("3")
        .arg("--retry-delay")
        .arg("5")
        .arg("-o")
        .arg(&path)
        .arg(url)
        .status()
        .context("running curl for 3DEP tile")?;
    if !output.success() {
        // Keep the partial download in place so the next run's
        // `-C -` can resume rather than starting from zero.
        return Err(anyhow!(
            "curl failed fetching {url} (partial file preserved for resume)"
        ));
    }
    Ok(path)
}

/// Read a USGS 3DEP 1 m GeoTIFF into memory. Handles both strip-based
/// and tile-based encodings with LZW/DEFLATE compression, and both
/// the classic-TIFF ModelTiepoint + ModelPixelScale georeferencing.
pub fn read_3dep_tile(path: &Path) -> Result<Tile> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut decoder =
        Decoder::new(reader).with_context(|| format!("decoding TIFF at {}", path.display()))?;
    decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());

    let (width_u32, height_u32) = decoder
        .dimensions()
        .with_context(|| format!("reading TIFF dimensions from {}", path.display()))?;
    let width = width_u32 as usize;
    let height = height_u32 as usize;

    // GeoTIFF georeferencing: ModelPixelScale (tag 33550) gives
    // (cell_x, cell_y, cell_z); ModelTiepoint (tag 33922) gives six
    // doubles (I, J, K, X, Y, Z) mapping raster (I, J) to UTM (X, Y).
    // For standard 3DEP tiles, (I, J) = (0, 0) pins the NW corner.
    let pixel_scale = decoder
        .get_tag_f64_vec(Tag::ModelPixelScaleTag)
        .context("reading ModelPixelScaleTag")?;
    let tiepoint = decoder
        .get_tag_f64_vec(Tag::ModelTiepointTag)
        .context("reading ModelTiepointTag")?;
    if pixel_scale.len() < 2 || tiepoint.len() < 6 {
        return Err(anyhow!(
            "unexpected GeoTIFF tag lengths: scale={} tiepoint={}",
            pixel_scale.len(),
            tiepoint.len()
        ));
    }
    let cell_x = pixel_scale[0];
    let cell_y = pixel_scale[1];
    // Use the X cell size as authoritative; tiles are square.
    let cell_m = cell_x;
    if (cell_x - cell_y).abs() > 1e-6 {
        tracing::warn!(
            cell_x,
            cell_y,
            "3DEP tile has non-square pixel; using X size"
        );
    }
    let raster_i = tiepoint[0];
    let raster_j = tiepoint[1];
    let world_x = tiepoint[3];
    let world_y = tiepoint[4];
    let east_min = world_x - raster_i * cell_x;
    let north_max = world_y + raster_j * cell_y;
    let north_min = north_max - height as f64 * cell_y;

    // NODATA sentinel from GDAL_NODATA tag (42113) — ASCII-stored float.
    let nodata = decoder
        .get_tag_ascii_string(Tag::Unknown(42113))
        .ok()
        .and_then(|s| s.trim().trim_end_matches('\0').parse::<f32>().ok())
        .unwrap_or(-1.0e38);

    // Read image chunks into a row-major f32 buffer.
    let chunk_type = decoder.get_chunk_type();
    let chunk_count = match chunk_type {
        ChunkType::Strip => decoder.strip_count()?,
        ChunkType::Tile => decoder.tile_count()?,
    };
    let tiles_per_row = match chunk_type {
        ChunkType::Strip => 1u32,
        ChunkType::Tile => {
            let tile_w = decoder.get_tag_u32(Tag::TileWidth)? as usize;
            width.div_ceil(tile_w) as u32
        }
    };
    let mut data = vec![0f32; width * height];
    for chunk_idx in 0..chunk_count {
        let chunk = decoder
            .read_chunk(chunk_idx)
            .with_context(|| format!("reading chunk {chunk_idx} of {}", path.display()))?;
        let bytes: Vec<f32> = match chunk {
            DecodingResult::F32(v) => v,
            DecodingResult::I16(v) => v.into_iter().map(|s| s as f32).collect(),
            DecodingResult::U16(v) => v.into_iter().map(|s| s as f32).collect(),
            other => {
                return Err(anyhow!(
                    "unexpected 3DEP sample type: {:?}",
                    std::mem::discriminant(&other)
                ));
            }
        };
        let (cw_u32, ch_u32) = decoder.chunk_data_dimensions(chunk_idx);
        let (cw, ch) = (cw_u32 as usize, ch_u32 as usize);
        let (tx, ty) = (
            (chunk_idx % tiles_per_row) as usize,
            (chunk_idx / tiles_per_row) as usize,
        );
        let (ox, oy) = match chunk_type {
            ChunkType::Strip => (0, ty * ch),
            ChunkType::Tile => {
                let (max_w, max_h) = decoder.chunk_dimensions();
                (tx * max_w as usize, ty * max_h as usize)
            }
        };
        for row in 0..ch {
            let dst_row = oy + row;
            if dst_row >= height {
                break;
            }
            let cols_to_copy = cw.min(width - ox);
            let src = row * cw;
            let dst = dst_row * width + ox;
            data[dst..dst + cols_to_copy].copy_from_slice(&bytes[src..src + cols_to_copy]);
        }
    }
    Ok(Tile {
        east_min,
        north_min,
        cell_m,
        width,
        height,
        data,
        nodata,
    })
}

// ---------------------------------------------------------------------------
// Sampling the mosaic
// ---------------------------------------------------------------------------

/// Sample a collection of loaded tiles at a UTM point. Checks tiles
/// in order; the first to contain the point + return a non-NODATA
/// value wins. Returns `None` if no tile covers the point or all
/// matching tiles are NODATA there.
pub fn sample_mosaic(tiles: &[Tile], east: f64, north: f64) -> Option<f32> {
    for tile in tiles {
        if let Some(v) = tile.sample(east, north) {
            return Some(v);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// USGS 3DEP 1/3 arc-second seamless (~10 m) — full-CONUS fallback
// ---------------------------------------------------------------------------

/// Ensure the 1/3 arc-second seamless tile exists locally; download
/// the most recent version from the 3DEP S3 bucket if missing.
/// Tile names are lowercase (`n46w123`), 1°×1° per tile.
pub fn ensure_3dep_13_tile(tile_id: &str, cache_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(cache_dir)?;
    // The canonical seamless URL is under `/current/<tile>/USGS_13_<tile>.tif`
    // and is kept up to date with the latest publication; no date-suffix
    // URL dance required.
    let fname = format!("USGS_13_{tile_id}.tif");
    let path = cache_dir.join(&fname);
    if path.exists() {
        return Ok(path);
    }
    let url = format!(
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/{tile_id}/{fname}"
    );
    println!("  downloading {fname} (3DEP 1/3 arc-sec seamless, ~500 MB)…");
    let status = Command::new("curl")
        .arg("-fSL")
        .arg("-C")
        .arg("-")
        .arg("--max-time")
        .arg("1800")
        .arg("--retry")
        .arg("3")
        .arg("--retry-delay")
        .arg("5")
        .arg("-o")
        .arg(&path)
        .arg(&url)
        .status()
        .context("running curl for 3DEP 1/3 arc-sec tile")?;
    if !status.success() {
        // Keep partial file for resume on next run.
        return Err(anyhow!(
            "curl failed fetching {url} (partial file preserved for resume)"
        ));
    }
    Ok(path)
}

/// Convert a WGS84 lat/lon to the 1°×1° seamless tile id it lies in.
/// Tiles are named by NW corner in lowercase, e.g. (45.54, -122.41)
/// falls inside `n46w123` (the tile whose NW corner is at 46°N 123°W).
pub fn tile_id_13_for(lat: f64, lon: f64) -> String {
    let lat_prefix = if lat >= 0.0 { "n" } else { "s" };
    let lon_prefix = if lon >= 0.0 { "e" } else { "w" };
    // NW corner: latitude rounds UP, longitude rounds DOWN (more west).
    let lat_n = lat.ceil() as i32;
    let lon_n = lon.floor() as i32;
    format!(
        "{}{:02}{}{:03}",
        lat_prefix,
        lat_n.unsigned_abs(),
        lon_prefix,
        lon_n.unsigned_abs()
    )
}

/// A loaded 1/3 arc-second seamless tile. Uses WGS84 lat/lon rather
/// than UTM, so sampling is done directly from the map's inverse-UTM
/// projection — same as the SRTM path.
pub struct Tile13 {
    /// NW corner latitude (degrees). Tile extends 1° south from here.
    pub lat_max: f64,
    /// NW corner longitude (degrees). Tile extends 1° east from here.
    pub lon_min: f64,
    pub width: usize,
    pub height: usize,
    /// Row-major floats, north-up. Index 0 is the NW corner.
    pub data: Vec<f32>,
    pub nodata: f32,
}

impl Tile13 {
    /// Bilinear sample at a WGS84 point. Returns `None` when out of
    /// bounds or when any bilinear neighbor is NODATA.
    pub fn sample(&self, lat: f64, lon: f64) -> Option<f32> {
        let lat_min = self.lat_max - 1.0;
        let lon_max = self.lon_min + 1.0;
        if lat < lat_min || lat > self.lat_max || lon < self.lon_min || lon > lon_max {
            return None;
        }
        let last_x = (self.width - 1) as f64;
        let last_y = (self.height - 1) as f64;
        let u = (lon - self.lon_min) * last_x;
        let v = (self.lat_max - lat) * last_y;
        let x0 = u.floor().clamp(0.0, last_x) as usize;
        let y0 = v.floor().clamp(0.0, last_y) as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = (u - x0 as f64) as f32;
        let fy = (v - y0 as f64) as f32;
        let at = |row: usize, col: usize| -> Option<f32> {
            let v = self.data[row * self.width + col];
            if (v - self.nodata).abs() < 1e-3 {
                None
            } else {
                Some(v)
            }
        };
        let h00 = at(y0, x0)?;
        let h01 = at(y0, x1)?;
        let h10 = at(y1, x0)?;
        let h11 = at(y1, x1)?;
        let top = h00 * (1.0 - fx) + h01 * fx;
        let bot = h10 * (1.0 - fx) + h11 * fx;
        Some(top * (1.0 - fy) + bot * fy)
    }
}

/// Decode a 1/3 arc-second seamless GeoTIFF into memory. Structurally
/// the same reader as [`read_3dep_tile`] but stays in WGS84 rather
/// than UTM since seamless tiles are lat/lon-aligned.
pub fn read_3dep_13_tile(path: &Path, tile_id: &str) -> Result<Tile13> {
    // Parse "n46w123" → NW corner.
    if tile_id.len() != 7 {
        return Err(anyhow!(
            "3DEP 1/3 arc-sec tile id must be 7 chars (got {tile_id:?})"
        ));
    }
    let lat_sign = match &tile_id[0..1] {
        "n" => 1.0,
        "s" => -1.0,
        other => return Err(anyhow!("unexpected lat prefix {other:?}")),
    };
    let lat_n: f64 = tile_id[1..3]
        .parse()
        .map_err(|_| anyhow!("bad lat digits in {tile_id:?}"))?;
    let lon_sign = match &tile_id[3..4] {
        "w" => -1.0,
        "e" => 1.0,
        other => return Err(anyhow!("unexpected lon prefix {other:?}")),
    };
    let lon_n: f64 = tile_id[4..]
        .parse()
        .map_err(|_| anyhow!("bad lon digits in {tile_id:?}"))?;
    let lat_max = lat_sign * lat_n;
    let lon_min = lon_sign * lon_n;

    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut decoder =
        Decoder::new(reader).with_context(|| format!("decoding TIFF at {}", path.display()))?;
    decoder = decoder.with_limits(tiff::decoder::Limits::unlimited());

    let (width_u32, height_u32) = decoder.dimensions()?;
    let width = width_u32 as usize;
    let height = height_u32 as usize;

    let nodata = decoder
        .get_tag_ascii_string(Tag::Unknown(42113))
        .ok()
        .and_then(|s| s.trim().trim_end_matches('\0').parse::<f32>().ok())
        .unwrap_or(-1.0e38);

    let chunk_type = decoder.get_chunk_type();
    let chunk_count = match chunk_type {
        ChunkType::Strip => decoder.strip_count()?,
        ChunkType::Tile => decoder.tile_count()?,
    };
    let tiles_per_row = match chunk_type {
        ChunkType::Strip => 1u32,
        ChunkType::Tile => {
            let tw = decoder.get_tag_u32(Tag::TileWidth)? as usize;
            width.div_ceil(tw) as u32
        }
    };
    let mut data = vec![0f32; width * height];
    for chunk_idx in 0..chunk_count {
        let chunk = decoder.read_chunk(chunk_idx)?;
        let bytes: Vec<f32> = match chunk {
            DecodingResult::F32(v) => v,
            DecodingResult::I16(v) => v.into_iter().map(|s| s as f32).collect(),
            DecodingResult::U16(v) => v.into_iter().map(|s| s as f32).collect(),
            other => {
                return Err(anyhow!(
                    "unexpected 3DEP 1/3 arc-sec sample type: {:?}",
                    std::mem::discriminant(&other)
                ))
            }
        };
        let (cw_u32, ch_u32) = decoder.chunk_data_dimensions(chunk_idx);
        let (cw, ch) = (cw_u32 as usize, ch_u32 as usize);
        let (tx, ty) = (
            (chunk_idx % tiles_per_row) as usize,
            (chunk_idx / tiles_per_row) as usize,
        );
        let (ox, oy) = match chunk_type {
            ChunkType::Strip => (0, ty * ch),
            ChunkType::Tile => {
                let (max_w, max_h) = decoder.chunk_dimensions();
                (tx * max_w as usize, ty * max_h as usize)
            }
        };
        for row in 0..ch {
            let dst_row = oy + row;
            if dst_row >= height {
                break;
            }
            let cols_to_copy = cw.min(width - ox);
            let src = row * cw;
            let dst = dst_row * width + ox;
            data[dst..dst + cols_to_copy].copy_from_slice(&bytes[src..src + cols_to_copy]);
        }
    }
    Ok(Tile13 {
        lat_max,
        lon_min,
        width,
        height,
        data,
        nodata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_tile_id_from_various_title_formats() {
        assert_eq!(
            extract_tile_id("USGS 1 Meter 10 x54y505 WA_FEMAHQ_2018_D18"),
            Some("x54y505".into())
        );
        assert_eq!(
            extract_tile_id("USGS one meter x55y505 WA_Western_South_2016"),
            Some("x55y505".into())
        );
        assert_eq!(extract_tile_id("USGS NED 1/3 arc-second"), None);
    }
}
