//! Feature classification layer — per-cell land cover enum baked
//! alongside the heightmap as `features.r8`.
//!
//! The `.r8` is a raw u8 grid with the same `W × H` layout as
//! `heightmap.r32`. Each byte is a [`FeatureClass`] discriminant
//! ([`FeatureClass::from_u8`] for safe conversion). Used by the
//! client to drive per-vertex coloring and, later, a proper
//! texture-blend shader.
//!
//! The canonical source is ESA WorldCover 2021 — a 10 m global
//! land cover raster, pre-projected to WGS84 (EPSG:4326), delivered
//! as 3° × 3° GeoTIFF tiles named by their SW corner (e.g.
//! `N45W123` covers lat 45–48, lon −123 to −120). Because the
//! projection already matches our SRTM pipeline, sampling works
//! exactly the same way: inverse-project the target grid to lat/lon
//! per vertex, look up the class byte at that lat/lon.
//!
//! Slope-derived overrides (e.g. `Cliff` on slopes > 40°) get
//! applied by the baker *after* WorldCover classification so
//! steep rock faces dominate regardless of vegetation label.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::Tag;

/// Per-cell land-cover classification. Values are stable byte
/// discriminants; keep additions additive so stored `.r8` assets
/// remain readable after the enum grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FeatureClass {
    /// Unclassified / no data.
    Unknown = 0,
    /// Permanent water (lake, river impoundment, wide river).
    Water = 1,
    /// Tree cover (forest, dense canopy).
    Forest = 2,
    /// Shrubland (sagebrush, chaparral, young forest regrowth).
    Shrubland = 3,
    /// Grassland / herbaceous.
    Grassland = 4,
    /// Cropland (cultivated fields).
    Cropland = 5,
    /// Built-up (roads, buildings, developed).
    BuiltUp = 6,
    /// Bare / sparse vegetation (gravel, sand, exposed soil).
    Bare = 7,
    /// Snow / permanent ice.
    Snow = 8,
    /// Wetland (seasonally inundated, marsh).
    Wetland = 9,
    /// Moss / lichen (alpine tundra, rare at PNW latitudes).
    Moss = 10,

    /// Slope-derived cliff / rock face. Override applied by baker
    /// after WorldCover classification when the heightmap's normal
    /// at that cell tilts past ~40° from vertical.
    Cliff = 20,

    /// Paved road (asphalt, concrete, or paver). From OSM
    /// `highway=motorway|trunk|primary|secondary|tertiary|residential|
    /// unclassified|living_street|road` absent an explicit unpaved
    /// surface tag, OR any `highway` tag with `surface=asphalt|paved|
    /// concrete|paving_stones`.
    PavedRoad = 21,

    /// Unpaved vehicular way (gravel, dirt, or hardpack). From OSM
    /// `highway=service|track` (default assumption), any way with
    /// `surface=dirt|gravel|unpaved|ground|earth|grass|sand|compacted`.
    UnpavedRoad = 22,

    /// Non-vehicular trail. From OSM `highway=path|footway|bridleway|
    /// cycleway|steps`. Narrower than road brushes — reads as a hint
    /// of stealth / back-country access.
    Trail = 23,
}

impl FeatureClass {
    /// Safe cast from a raw byte. Unknown values fall back to
    /// [`FeatureClass::Unknown`] so loading a `.r8` produced by a
    /// newer baker never panics the sampler.
    pub fn from_u8(b: u8) -> Self {
        match b {
            1 => Self::Water,
            2 => Self::Forest,
            3 => Self::Shrubland,
            4 => Self::Grassland,
            5 => Self::Cropland,
            6 => Self::BuiltUp,
            7 => Self::Bare,
            8 => Self::Snow,
            9 => Self::Wetland,
            10 => Self::Moss,
            20 => Self::Cliff,
            21 => Self::PavedRoad,
            22 => Self::UnpavedRoad,
            23 => Self::Trail,
            _ => Self::Unknown,
        }
    }
}

/// Map an ESA WorldCover class byte to our [`FeatureClass`].
///
/// ESA class values (from the v200 product specification):
/// - 10: Tree cover
/// - 20: Shrubland
/// - 30: Grassland
/// - 40: Cropland
/// - 50: Built-up
/// - 60: Bare / sparse vegetation
/// - 70: Snow and ice
/// - 80: Permanent water
/// - 90: Herbaceous wetland
/// - 95: Mangroves
/// - 100: Moss and lichen
pub fn map_esa_worldcover_class(esa: u8) -> FeatureClass {
    match esa {
        10 => FeatureClass::Forest,
        20 => FeatureClass::Shrubland,
        30 => FeatureClass::Grassland,
        40 => FeatureClass::Cropland,
        50 => FeatureClass::BuiltUp,
        60 => FeatureClass::Bare,
        70 => FeatureClass::Snow,
        80 => FeatureClass::Water,
        90 | 95 => FeatureClass::Wetland,
        100 => FeatureClass::Moss,
        _ => FeatureClass::Unknown,
    }
}

/// ESA WorldCover tile side length in samples. Each 3° × 3° tile
/// is 36 000 × 36 000 pixels at 10 m nominal resolution.
pub const WORLDCOVER_SIDE: usize = 36_000;

/// A loaded ESA WorldCover tile kept in memory for per-cell lookups
/// during a bake. Roughly 1.3 GB uncompressed — bake is an offline
/// tool so this is acceptable.
pub struct WorldCoverTile {
    /// SW corner latitude (degrees). Tile spans 3° north from here.
    pub lat_min: f64,
    /// SW corner longitude (degrees, negative for west). Spans 3° east.
    pub lon_min: f64,
    /// Side length of the (square) raster in samples.
    pub side: usize,
    /// Row-major grid, north-up — index 0 is the NW corner.
    pub data: Vec<u8>,
}

impl WorldCoverTile {
    /// Sample the ESA class byte at a lat/lon inside the tile.
    /// Nearest-neighbor; out-of-bounds queries clamp to the edge.
    pub fn sample_at(&self, lat: f64, lon: f64) -> u8 {
        let lat_max = self.lat_min + 3.0;
        let lon_max = self.lon_min + 3.0;
        let last = (self.side - 1) as f64;
        // Row 0 is the north edge (lat = lat_max).
        let row_f = (lat_max - lat) / 3.0 * last;
        let col_f = (lon - self.lon_min) / 3.0 * last;
        let row = row_f.round().clamp(0.0, last) as usize;
        let col = col_f.round().clamp(0.0, last) as usize;
        let max_idx = lat_max - 1e-9;
        if lat < self.lat_min || lat > max_idx || lon < self.lon_min || lon > lon_max - 1e-9 {
            // Treat far-out-of-bounds as no data; caller typically
            // won't hit this because the heightmap extents should
            // fit inside the chosen WorldCover tile.
        }
        self.data[row * self.side + col]
    }
}

/// Parse a WorldCover tile id (e.g. `"N45W123"`) → SW corner `(lat, lon)`.
///
/// Same 7-character format as SRTM tile ids — the only practical
/// difference is that ESA WorldCover spans 3° per tile while SRTM
/// spans 1°, so the SW corners are multiples of 3.
pub fn parse_worldcover_tile_sw(tile: &str) -> Result<(f64, f64)> {
    if tile.len() != 7 {
        return Err(anyhow!("WorldCover tile id must be 7 chars (got {tile:?})"));
    }
    let lat_sign = match &tile[0..1] {
        "N" => 1.0,
        "S" => -1.0,
        other => return Err(anyhow!("unexpected lat prefix {other:?}")),
    };
    let lat: f64 = tile[1..3]
        .parse()
        .map_err(|_| anyhow!("bad lat digits in {tile:?}"))?;
    let lon_sign = match &tile[3..4] {
        "E" => 1.0,
        "W" => -1.0,
        other => return Err(anyhow!("unexpected lon prefix {other:?}")),
    };
    let lon: f64 = tile[4..]
        .parse()
        .map_err(|_| anyhow!("bad lon digits in {tile:?}"))?;
    let (lat, lon) = (lat_sign * lat, lon_sign * lon);
    if (lat as i32) % 3 != 0 || (lon as i32) % 3 != 0 {
        return Err(anyhow!(
            "WorldCover SW corner {tile:?} must be multiples of 3°"
        ));
    }
    Ok((lat, lon))
}

/// Ensure the ESA WorldCover tile exists on disk; download from the
/// public S3 bucket if missing. Returns the path to the `.tif`.
///
/// After download, patches the TIFF's `PhotometricInterpretation`
/// tag from `RGBPalette` (3) to `BlackIsZero` (1) in place. The raw
/// class bytes are identical either way — the palette is purely a
/// display hint for GIS viewers — but the `tiff` crate we use for
/// decoding rejects `RGBPalette` photometric while happily reading
/// the same compressed strips as grayscale. Patching once at fetch
/// time keeps the cache self-contained and avoids a custom reader.
pub fn ensure_worldcover_tile(tile: &str, cache_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(cache_dir)?;
    let tif = cache_dir.join(format!("ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"));
    if tif.exists() {
        return Ok(tif);
    }
    // Validate the tile id before touching the network.
    let _ = parse_worldcover_tile_sw(tile)?;
    let url = format!(
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/\
         ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    );
    println!("fetching {tile} from ESA WorldCover S3…");
    let curl = Command::new("curl")
        .arg("-fsSL")
        .arg("-o")
        .arg(&tif)
        .arg(&url)
        .status()
        .context("running curl (needed for first-time WorldCover fetch)")?;
    if !curl.success() {
        // Clean up partial download so the next run retries cleanly.
        let _ = fs::remove_file(&tif);
        return Err(anyhow!("curl failed fetching {url}"));
    }
    patch_photometric_rgbpalette_to_grayscale(&tif)
        .with_context(|| format!("patching PhotometricInterpretation in {}", tif.display()))?;
    Ok(tif)
}

/// In-place patch of a TIFF's first-IFD `PhotometricInterpretation`
/// tag (id 262) from `RGBPalette` (3) to `BlackIsZero` (1).
///
/// Minimal parser — walks the first IFD only, understands classic
/// 32-bit TIFF offsets (ESA WorldCover tiles are not BigTIFF). Handles
/// both byte orders. Idempotent: patching an already-grayscale file
/// leaves it unchanged.
fn patch_photometric_rgbpalette_to_grayscale(path: &Path) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = fs::OpenOptions::new().read(true).write(true).open(path)?;
    let mut header = [0u8; 8];
    f.read_exact(&mut header)?;
    let le = match &header[..2] {
        b"II" => true,
        b"MM" => false,
        other => return Err(anyhow!("unexpected TIFF byte order {:?}", other)),
    };
    let read_u16 = |b: [u8; 2]| {
        if le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        }
    };
    let read_u32 = |b: [u8; 4]| {
        if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        }
    };
    let magic = read_u16([header[2], header[3]]);
    if magic != 42 {
        return Err(anyhow!("not a classic TIFF (magic = {magic})"));
    }
    let ifd_offset = read_u32([header[4], header[5], header[6], header[7]]);

    f.seek(SeekFrom::Start(ifd_offset as u64))?;
    let mut count_buf = [0u8; 2];
    f.read_exact(&mut count_buf)?;
    let entry_count = read_u16(count_buf);

    for i in 0..entry_count {
        let entry_offset = ifd_offset as u64 + 2 + (i as u64) * 12;
        f.seek(SeekFrom::Start(entry_offset))?;
        let mut entry = [0u8; 12];
        f.read_exact(&mut entry)?;
        let tag = read_u16([entry[0], entry[1]]);
        if tag != 262 {
            continue;
        }
        let typ = read_u16([entry[2], entry[3]]);
        let cnt = read_u32([entry[4], entry[5], entry[6], entry[7]]);
        // SHORT (type 3), count 1 — value lives in the first 2 bytes
        // of the 4-byte value field.
        if typ != 3 || cnt != 1 {
            return Err(anyhow!(
                "PhotometricInterpretation tag has unexpected type={typ} count={cnt}"
            ));
        }
        let current = read_u16([entry[8], entry[9]]);
        if current == 1 {
            return Ok(()); // already grayscale — nothing to do.
        }
        if current != 3 {
            return Err(anyhow!(
                "PhotometricInterpretation is {current}, not RGBPalette (3) — refusing to patch"
            ));
        }
        // Patch the two-byte value to 1 (BlackIsZero).
        let patched: [u8; 2] = if le {
            1u16.to_le_bytes()
        } else {
            1u16.to_be_bytes()
        };
        f.seek(SeekFrom::Start(entry_offset + 8))?;
        f.write_all(&patched)?;
        return Ok(());
    }
    Err(anyhow!(
        "PhotometricInterpretation tag (262) not found in first IFD"
    ))
}

/// Decode an ESA WorldCover GeoTIFF into memory. Returns the raw
/// class grid plus SW corner coordinates parsed from the tile id.
///
/// ESA WorldCover stores its class index as a single-band TIFF with
/// photometric interpretation `RGBPalette` — a palette lookup table
/// mapping each u8 class to an RGB display color. The `tiff` crate's
/// high-level `read_image()` rejects palette TIFFs, so we iterate
/// strip-by-strip via `read_chunk` instead, which returns the raw
/// class bytes untouched by the palette expansion.
pub fn read_worldcover_tile(path: &Path, tile: &str) -> Result<WorldCoverTile> {
    let (lat_min, lon_min) = parse_worldcover_tile_sw(tile)?;
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut decoder =
        Decoder::new(reader).with_context(|| format!("decoding TIFF at {}", path.display()))?;

    let (w, h) = decoder
        .dimensions()
        .with_context(|| format!("reading TIFF dimensions from {}", path.display()))?;
    if w as usize != WORLDCOVER_SIDE || h as usize != WORLDCOVER_SIDE {
        return Err(anyhow!(
            "WorldCover tile {tile} has unexpected size {w}x{h} (expected {s}x{s})",
            s = WORLDCOVER_SIDE
        ));
    }

    // Sanity check: single band, 8-bit per sample.
    if let Ok(n) = decoder.get_tag_u32(Tag::SamplesPerPixel) {
        if n != 1 {
            return Err(anyhow!(
                "WorldCover tile {tile} has SamplesPerPixel={n} (expected 1)"
            ));
        }
    }

    // WorldCover v200 tiles use TIFF tiles (not strips), with a
    // tile size tagged via TileWidth/TileLength. Read chunk-by-chunk
    // and splat each into the right sub-rect of the output grid.
    let (chunk_count, tiles_per_row) = match decoder.get_chunk_type() {
        ChunkType::Strip => {
            let count = decoder
                .strip_count()
                .with_context(|| format!("reading strip count from {}", path.display()))?;
            (count, 1)
        }
        ChunkType::Tile => {
            let count = decoder
                .tile_count()
                .with_context(|| format!("reading tile count from {}", path.display()))?;
            let tile_w = decoder
                .get_tag_u32(Tag::TileWidth)
                .with_context(|| format!("reading TileWidth from {}", path.display()))?
                as usize;
            let tiles_per_row = WORLDCOVER_SIDE.div_ceil(tile_w);
            (count, tiles_per_row as u32)
        }
    };

    let mut data = vec![0u8; WORLDCOVER_SIDE * WORLDCOVER_SIDE];
    for chunk_idx in 0..chunk_count {
        let chunk = decoder
            .read_chunk(chunk_idx)
            .with_context(|| format!("reading chunk {chunk_idx} of {}", path.display()))?;
        let bytes = match chunk {
            DecodingResult::U8(v) => v,
            other => {
                return Err(anyhow!(
                    "WorldCover chunk {chunk_idx} decoded to non-u8 type: {:?}",
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
        // Chunk origin within the full image. Strips span the full
        // image width; tiles use the full chunk-dimensions (not the
        // per-chunk data dims, which are smaller on edge chunks).
        let (ox, oy) = match decoder.get_chunk_type() {
            ChunkType::Strip => (0, ty * ch),
            ChunkType::Tile => {
                let (max_w, max_h) = decoder.chunk_dimensions();
                (tx * max_w as usize, ty * max_h as usize)
            }
        };
        for row in 0..ch {
            let dst_row = oy + row;
            if dst_row >= WORLDCOVER_SIDE {
                break;
            }
            let cols_to_copy = cw.min(WORLDCOVER_SIDE - ox);
            let src = row * cw;
            let dst = dst_row * WORLDCOVER_SIDE + ox;
            data[dst..dst + cols_to_copy].copy_from_slice(&bytes[src..src + cols_to_copy]);
        }
    }
    Ok(WorldCoverTile {
        lat_min,
        lon_min,
        side: WORLDCOVER_SIDE,
        data,
    })
}

/// Smooth classifier boundaries via per-class argmax Gaussian blur.
///
/// ESA WorldCover is a hard-quantized 10 m raster; sampled onto our
/// typical 2 m heightmap grid it yields 5×5 blocks of identical
/// class bytes with right-angle staircases at boundaries. That
/// "Minecraft outline" pattern survives every amount of texture-
/// side anti-aliasing because the underlying data really does step.
///
/// This pass fixes the *data* once at bake time. For every unique
/// class `c` present in the input, we build a binary mask (1.0
/// where `bytes[i] == c`, else 0.0), separable-Gaussian-blur it
/// with `sigma`, and then per-pixel assign the class whose blurred
/// response is highest. The zero level-set between two classes'
/// blurred masks becomes a smooth curve — so class boundaries read
/// as natural coastlines / treelines rather than rasterized steps.
///
/// **Thin OSM line classes** (`PavedRoad`, `UnpavedRoad`, `Trail`)
/// are restored from the input after smoothing. Their mass (1-3
/// cells wide) blurs to a near-zero response and would lose every
/// argmax against adjacent area classes; for these classes we want
/// the crisp rasterized line preserved. Area classes (water,
/// forest, grass, crop, bare, cliff, etc.) get the smoothing
/// treatment.
///
/// Cost: one pair of 1-D blurs per unique class. Typical real map
/// has ~10-15 classes in use, so for a 2500×2250 grid with `sigma
/// = 3.5` (radius ≈ 10) the pass is ~150 M float ops per class →
/// ~1-2 seconds in release. Runs once per bake; not on the hot
/// path.
pub fn smooth_class_boundaries(
    bytes: &[u8],
    width: usize,
    height: usize,
    sigma: f32,
    preserve_mask: Option<&[bool]>,
) -> Vec<u8> {
    assert_eq!(
        bytes.len(),
        width * height,
        "smooth_class_boundaries: byte count must equal width × height"
    );
    if let Some(mask) = preserve_mask {
        assert_eq!(
            mask.len(),
            bytes.len(),
            "preserve_mask must match bytes length"
        );
    }
    let n = bytes.len();
    if n == 0 || sigma <= 0.0 {
        return bytes.to_vec();
    }

    // Separable 1D Gaussian kernel, clipped at 3σ (covers >99.7%
    // of the Gaussian's mass — beyond this the weights are noise).
    let radius = (3.0 * sigma).ceil().max(1.0) as usize;
    let kernel: Vec<f32> = {
        let sigma2 = 2.0 * sigma * sigma;
        let size = 2 * radius + 1;
        let mut k = Vec::with_capacity(size);
        let mut sum = 0.0f32;
        for i in 0..size {
            let x = i as f32 - radius as f32;
            let w = (-x * x / sigma2).exp();
            k.push(w);
            sum += w;
        }
        for w in &mut k {
            *w /= sum;
        }
        k
    };

    // Unique classes present in the input — typically 5-15, so a
    // linear scan per pixel during argmax is cheap.
    let mut classes: Vec<u8> = bytes.to_vec();
    classes.sort_unstable();
    classes.dedup();

    let mut best_class = vec![0u8; n];
    let mut best_value = vec![f32::NEG_INFINITY; n];
    let mut mask = vec![0.0f32; n];
    let mut tmp = vec![0.0f32; n];
    let mut blurred = vec![0.0f32; n];

    for &c in &classes {
        // Binary mask for this class.
        for i in 0..n {
            mask[i] = if bytes[i] == c { 1.0 } else { 0.0 };
        }

        // Horizontal pass (mask → tmp). Clamp-to-edge at borders so
        // the map rim doesn't lose mass into the void.
        for y in 0..height {
            let row = y * width;
            for x in 0..width {
                let mut sum = 0.0f32;
                for (k, &kw) in kernel.iter().enumerate() {
                    let sx = (x as isize + k as isize - radius as isize)
                        .clamp(0, width as isize - 1) as usize;
                    sum += mask[row + sx] * kw;
                }
                tmp[row + x] = sum;
            }
        }

        // Vertical pass (tmp → blurred).
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0f32;
                for (k, &kw) in kernel.iter().enumerate() {
                    let sy = (y as isize + k as isize - radius as isize)
                        .clamp(0, height as isize - 1) as usize;
                    sum += tmp[sy * width + x] * kw;
                }
                blurred[y * width + x] = sum;
            }
        }

        // Argmax update.
        for i in 0..n {
            if blurred[i] > best_value[i] {
                best_value[i] = blurred[i];
                best_class[i] = c;
            }
        }
    }

    // Restore preserved cells from the input. Two layers:
    //
    // 1. **Caller-supplied mask** (typically: cells painted by an
    //    OSM polygon overlay). These are already at high resolution
    //    with clean edges, so smoothing would soften them right
    //    back into ~3σ-wide blobs and destroy the whole point of
    //    the polygon source. Mask their input class through the
    //    output unchanged.
    // 2. **Thin OSM line classes** (PavedRoad / UnpavedRoad /
    //    Trail). These are 1-3 cell-wide features; their argmax
    //    response is too small to win against any adjacent area
    //    class. Always restored regardless of σ.
    for i in 0..n {
        if let Some(mask) = preserve_mask {
            if mask[i] {
                best_class[i] = bytes[i];
                continue;
            }
        }
        let orig = bytes[i];
        if orig == FeatureClass::PavedRoad as u8
            || orig == FeatureClass::UnpavedRoad as u8
            || orig == FeatureClass::Trail as u8
        {
            best_class[i] = orig;
        }
    }

    best_class
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_class_roundtrips() {
        for c in [
            FeatureClass::Unknown,
            FeatureClass::Water,
            FeatureClass::Forest,
            FeatureClass::Shrubland,
            FeatureClass::Grassland,
            FeatureClass::Cropland,
            FeatureClass::BuiltUp,
            FeatureClass::Bare,
            FeatureClass::Snow,
            FeatureClass::Wetland,
            FeatureClass::Moss,
            FeatureClass::Cliff,
        ] {
            assert_eq!(FeatureClass::from_u8(c as u8), c);
        }
    }

    #[test]
    fn unknown_byte_maps_to_unknown() {
        assert_eq!(FeatureClass::from_u8(42), FeatureClass::Unknown);
        assert_eq!(FeatureClass::from_u8(255), FeatureClass::Unknown);
    }

    #[test]
    fn esa_mapping_covers_all_published_classes() {
        assert_eq!(map_esa_worldcover_class(10), FeatureClass::Forest);
        assert_eq!(map_esa_worldcover_class(20), FeatureClass::Shrubland);
        assert_eq!(map_esa_worldcover_class(30), FeatureClass::Grassland);
        assert_eq!(map_esa_worldcover_class(40), FeatureClass::Cropland);
        assert_eq!(map_esa_worldcover_class(50), FeatureClass::BuiltUp);
        assert_eq!(map_esa_worldcover_class(60), FeatureClass::Bare);
        assert_eq!(map_esa_worldcover_class(70), FeatureClass::Snow);
        assert_eq!(map_esa_worldcover_class(80), FeatureClass::Water);
        assert_eq!(map_esa_worldcover_class(90), FeatureClass::Wetland);
        assert_eq!(map_esa_worldcover_class(95), FeatureClass::Wetland);
        assert_eq!(map_esa_worldcover_class(100), FeatureClass::Moss);
    }

    #[test]
    fn parse_worldcover_tile_sw_accepts_3deg_grid() {
        assert_eq!(parse_worldcover_tile_sw("N45W123").unwrap(), (45.0, -123.0));
        assert_eq!(parse_worldcover_tile_sw("N45W120").unwrap(), (45.0, -120.0));
    }

    #[test]
    fn parse_worldcover_tile_rejects_non_3deg_grid() {
        assert!(parse_worldcover_tile_sw("N46W122").is_err());
        assert!(parse_worldcover_tile_sw("N45W122").is_err());
    }

    #[test]
    fn smooth_class_boundaries_is_identity_for_uniform_field() {
        // A constant classification must round-trip exactly — any
        // change would mean the blur+argmax pass drifts off a
        // majority class, which would bug real maps too.
        let w = 16;
        let h = 16;
        let bytes = vec![FeatureClass::Forest as u8; w * h];
        let out = smooth_class_boundaries(&bytes, w, h, 3.5, None);
        assert_eq!(out, bytes);
    }

    #[test]
    fn smooth_class_boundaries_preserves_thin_trails() {
        // A one-cell-wide Trail across a Grassland background would
        // be wiped out by argmax after blurring. Verify the OSM
        // line-class preservation path keeps it intact.
        let w = 32;
        let h = 32;
        let mut bytes = vec![FeatureClass::Grassland as u8; w * h];
        for x in 0..w {
            bytes[(h / 2) * w + x] = FeatureClass::Trail as u8;
        }
        let out = smooth_class_boundaries(&bytes, w, h, 3.5, None);
        for x in 0..w {
            assert_eq!(
                out[(h / 2) * w + x],
                FeatureClass::Trail as u8,
                "trail cell at x={x} was blurred away"
            );
        }
    }

    #[test]
    fn smooth_class_boundaries_preserve_mask_protects_marked_cells() {
        // OSM polygon use case: a small Forest island inside a
        // Grassland field. Without the mask the σ=4 blur would melt
        // it — the island is small and the surrounding grass
        // dominates. With the mask, every Forest cell of the island
        // stays Forest.
        let w = 32;
        let h = 32;
        let mut bytes = vec![FeatureClass::Grassland as u8; w * h];
        // 4×4 forest island at (10..14, 10..14)
        let mut mask = vec![false; w * h];
        for y in 10..14 {
            for x in 10..14 {
                bytes[y * w + x] = FeatureClass::Forest as u8;
                mask[y * w + x] = true;
            }
        }
        let out = smooth_class_boundaries(&bytes, w, h, 4.0, Some(&mask));
        // Without the mask this island melts away; with it, every
        // marked cell still reads as Forest.
        for y in 10..14 {
            for x in 10..14 {
                assert_eq!(
                    out[y * w + x],
                    FeatureClass::Forest as u8,
                    "preserved cell ({x},{y}) was overwritten",
                );
            }
        }
    }

    #[test]
    fn smooth_class_boundaries_curves_a_staircase_boundary() {
        // Build a classic ESA staircase: 10-cell-wide blocks of
        // Water abutting Grassland, where the boundary is a
        // rectangular zigzag. After smoothing, interior cells that
        // were Water but have Grassland on three sides should flip,
        // and vice versa — i.e. the boundary should relax toward
        // a straight line / curve rather than staying boxy.
        let w = 40;
        let h = 40;
        let mut bytes = vec![FeatureClass::Grassland as u8; w * h];
        // L-shaped water notch on the left half
        for y in 0..h {
            let boundary = if y < h / 2 { w / 4 } else { w / 2 };
            for x in 0..boundary {
                bytes[y * w + x] = FeatureClass::Water as u8;
            }
        }
        let out = smooth_class_boundaries(&bytes, w, h, 3.0, None);

        // The 90° inside corner at (w/2, h/2) should have relaxed:
        // cells just inside the corner (inside the grassland side)
        // should still be grassland, and the corner pixel in water
        // should still be water far from the step, but right at
        // the corner the argmax should pull one way or the other,
        // NOT retain the exact right angle. We verify by checking
        // that the mean class around a 3×3 patch at the corner is
        // neither pure water nor pure grassland (it bled).
        let cx = w / 4 + 1; // one cell into the grassland side of the short step
        let cy = h / 2 - 1; // one cell up from the corner
        let mut water_count = 0;
        let mut grass_count = 0;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let x = (cx as i32 + dx) as usize;
                let y = (cy as i32 + dy) as usize;
                match out[y * w + x] {
                    v if v == FeatureClass::Water as u8 => water_count += 1,
                    v if v == FeatureClass::Grassland as u8 => grass_count += 1,
                    _ => {}
                }
            }
        }
        assert!(
            water_count > 0 && grass_count > 0,
            "boundary didn't soften: water={water_count}, grass={grass_count}"
        );
    }
}
