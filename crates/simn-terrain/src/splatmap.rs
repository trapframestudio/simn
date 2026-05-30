//! Splatmap baking — convert the per-cell `FeatureClass` byte grid
//! (`features.r8`) into two 4-channel RGBA8 splatmaps that drive the
//! terrain shader's texture blending.
//!
//! Each splatmap channel is the per-pixel **blend weight** (0..255)
//! for one class group. The shader samples the splatmap with linear
//! filtering, gets a vec4 of weights, and blends 4 base diffuses
//! accordingly. Two splatmaps × 4 channels = 8 distinct class
//! identities preserved across the map. Per-pixel weights across
//! all 8 channels sum to ~255 (with rounding slop), so nothing
//! ever gets "no biome".
//!
//! ## Per-class sigma — variable smoothness
//!
//! Each channel has its own Gaussian σ before normalization. Wide
//! homogeneous classes (Forest, Grassland) get larger σ for soft
//! organic blends; narrow human-made classes (BuiltUp, Cliff,
//! Cropland) get smaller σ to keep their crisp edges. This is the
//! "increase resolution where needed, decrease where needed"
//! principle expressed at bake time:
//!
//! - Forest σ=3.0 cells: soft treelines bleed naturally into
//!   adjacent meadows
//! - Cliff σ=0.5 cells: rock face / non-rock boundary stays
//!   knife-edge sharp at the heightmap's facet resolution
//!
//! ## Lines stay on `features.r8`
//!
//! `PavedRoad` / `UnpavedRoad` / `Trail` are 1-3 cell-wide line
//! features. Folding them into a splatmap channel would either
//! eat them under blending (mass too small) or force σ=0 (no point
//! in a splatmap then). They're handled by the existing line-
//! overlay logic in the shader, sampled from the unchanged
//! `features.r8` grid.

use crate::features::FeatureClass;

/// One splatmap channel: which classes contribute to it, the
/// Gaussian sigma in grid cells used to soften its mask, and the
/// channel index within its RGBA8 splatmap.
#[derive(Debug, Clone, Copy)]
pub struct ChannelSpec {
    /// Classes whose presence contributes to this channel's mask.
    /// Multiple classes per channel = grouping (e.g. Forest +
    /// Shrubland + Moss → one "tree canopy" channel).
    pub classes: &'static [FeatureClass],

    /// Gaussian σ for this channel's mask blur, in grid cells.
    /// Smaller = crisper edges, larger = softer blends. Tuned per
    /// class based on typical real-world feature sharpness.
    pub sigma_cells: f32,

    /// 0..3 — which RGBA channel of the splatmap this writes to.
    pub channel_index: u8,
}

// **σ tuning rationale.** A splatmap is sampled with hardware
// `filter_linear`, which already gives sub-cell-precision smoothing
// at fetch time. The bake-time Gaussian is *additional* softening
// on top. Setting σ too high means small distinct features
// (BuiltUp town, single-cell Cliff strip) get crowded out under
// per-pixel normalization across all 8 channels — a wide-σ Forest
// neighbor with mass spilling into the town pixel can equal or
// outweigh the town's own narrow-σ peak, and the result is
// mostly-Forest at the town location. Lesson learned the hard way:
// **keep σ small, let the hardware filter do the work.**
//
// As a rule of thumb: σ ≤ 1 for everything; σ ≈ 0 for narrow
// features we want preserved exactly.

/// Splatmap A — biome-style classes that occupy wide regions.
pub const SPLATMAP_A_CHANNELS: [ChannelSpec; 4] = [
    // R: Forest group — broadleaf / coniferous / scrub / moss.
    // 1-cell σ gives a soft single-cell-wide treeline; the
    // hardware filter widens it to ~2-3 cells visually.
    ChannelSpec {
        classes: &[
            FeatureClass::Forest,
            FeatureClass::Shrubland,
            FeatureClass::Moss,
        ],
        sigma_cells: 1.0,
        channel_index: 0,
    },
    // G: Open-ground vegetation — grassland + wetland.
    ChannelSpec {
        classes: &[FeatureClass::Grassland, FeatureClass::Wetland],
        sigma_cells: 1.0,
        channel_index: 1,
    },
    // B: Water. Crisp shorelines.
    ChannelSpec {
        classes: &[FeatureClass::Water],
        sigma_cells: 0.5,
        channel_index: 2,
    },
    // A: Cropland. Crisp surveyed field edges.
    ChannelSpec {
        classes: &[FeatureClass::Cropland],
        sigma_cells: 0.5,
        channel_index: 3,
    },
];

/// Splatmap B — narrower / harder-edge classes. All very small σ
/// so towns / cliffs / bare patches survive normalization against
/// wider neighbors.
pub const SPLATMAP_B_CHANNELS: [ChannelSpec; 4] = [
    // R: Bare ground.
    ChannelSpec {
        classes: &[FeatureClass::Bare],
        sigma_cells: 0.5,
        channel_index: 0,
    },
    // G: BuiltUp — town footprints. σ=0 effectively (just enough
    // to anti-alias the per-cell rasterization).
    ChannelSpec {
        classes: &[FeatureClass::BuiltUp],
        sigma_cells: 0.25,
        channel_index: 1,
    },
    // B: Cliff — knife-edge rock face.
    ChannelSpec {
        classes: &[FeatureClass::Cliff],
        sigma_cells: 0.25,
        channel_index: 2,
    },
    // A: Snow. Snowlines mid-soft.
    ChannelSpec {
        classes: &[FeatureClass::Snow],
        sigma_cells: 0.5,
        channel_index: 3,
    },
];

/// Both splatmaps as an output bundle.
#[derive(Debug, Clone)]
pub struct SplatmapPair {
    /// Splatmap A — biome-style classes. Length = `4 * width * height` (RGBA8).
    pub map_a: Vec<u8>,
    /// Splatmap B — hard-surface classes. Length = `4 * width * height` (RGBA8).
    pub map_b: Vec<u8>,
}

/// Bake a [`SplatmapPair`] from the post-OSM per-cell `FeatureClass`
/// byte grid. For each channel:
///
/// 1. Build a binary mask: 1.0 where any of the channel's classes
///    is present, 0.0 elsewhere.
/// 2. Separable Gaussian blur with the channel's `sigma_cells`.
///
/// After all 8 channels are blurred, normalize per-pixel so the
/// sum of all 8 channel weights is ~1.0 (then encode as u8). This
/// guarantees the shader's weighted sum produces a valid color
/// even when all 8 raw channel responses are small (e.g. on a
/// `PavedRoad` cell — line classes contribute to no channel and
/// would otherwise produce a 0/black pixel).
pub fn bake_splatmap_pair(bytes: &[u8], width: usize, height: usize) -> SplatmapPair {
    assert_eq!(
        bytes.len(),
        width * height,
        "bake_splatmap_pair: byte count must equal width × height"
    );
    let n = bytes.len();

    // Per-channel f32 buffers, one per of 8 channels.
    // Chained iter so we don't have to know the static array sizes
    // at the call site.
    let channel_specs: Vec<&ChannelSpec> = SPLATMAP_A_CHANNELS
        .iter()
        .chain(SPLATMAP_B_CHANNELS.iter())
        .collect();
    let mut per_channel: Vec<Vec<f32>> = Vec::with_capacity(channel_specs.len());

    for spec in &channel_specs {
        let mut mask = vec![0.0f32; n];
        for (i, &b) in bytes.iter().enumerate() {
            if spec.classes.iter().any(|c| *c as u8 == b) {
                mask[i] = 1.0;
            }
        }
        let blurred = gaussian_blur_2d(&mask, width, height, spec.sigma_cells);
        per_channel.push(blurred);
    }

    // Per-pixel normalize across all 8 channels, so the shader's
    // weighted sum of textures stays in roughly the right range
    // even when only one or two classes have meaningful response
    // (e.g. deep forest interior → R near 1, others near 0; sum
    // already ≈ 1, normalization is a no-op there).
    //
    // **No-data fallback:** when every channel responds at
    // ≤ 1e-4 (typical for a Trail or PavedRoad cell which doesn't
    // belong to any channel's class group), default the pixel to
    // pure Forest (channel 0 in splatmap A). The shader's line
    // overlay layer will still paint road/trail textures on top
    // via the unchanged `features.r8` grid — the splatmap base is
    // just whatever the surrounding biome is, so Forest is a safe
    // PNW default.
    let mut map_a = vec![0u8; 4 * n];
    let mut map_b = vec![0u8; 4 * n];

    for i in 0..n {
        let mut sum = 0.0f32;
        for ch in &per_channel {
            sum += ch[i];
        }
        if sum < 1e-4 {
            // Line-class cell or completely-unmapped — fall back to
            // pure Forest so the shader has something to render.
            map_a[4 * i] = 255;
            continue;
        }
        let inv = 1.0 / sum;
        for (chan_idx, ch) in per_channel.iter().enumerate() {
            let normalized = (ch[i] * inv * 255.0).round().clamp(0.0, 255.0) as u8;
            if chan_idx < 4 {
                let spec = &channel_specs[chan_idx];
                map_a[4 * i + spec.channel_index as usize] = normalized;
            } else {
                let spec = &channel_specs[chan_idx];
                map_b[4 * i + spec.channel_index as usize] = normalized;
            }
        }
    }

    SplatmapPair { map_a, map_b }
}

/// Apply an N×N separable box blur to an RGBA8 splatmap in place,
/// matching the softening pass `terrain3d_loader.gd` runs before
/// encoding the Terrain3D control map. The two must agree exactly,
/// otherwise:
/// - The sim places foliage where Terrain3D won't show texture
///   (and AI claims concealment that doesn't exist visually), or
/// - Terrain3D shows biome texture where the sim has no foliage
///   (player sees grass but AI sees bare ground).
///
/// `radius` is the half-kernel size. The Godot loader uses 5
/// (an 11×11 kernel) — pass the same value here when feeding the
/// static-foliage bake.
///
/// Allocates two temporary RGBA8 buffers internally; output replaces
/// `bytes` on success. Returns Err if the byte count doesn't match
/// `width * height * 4`.
pub fn soften_rgba8(
    bytes: &mut [u8],
    width: usize,
    height: usize,
    radius: usize,
) -> anyhow::Result<()> {
    let expected = width * height * 4;
    if bytes.len() != expected {
        return Err(anyhow::anyhow!(
            "soften_rgba8: got {} bytes, expected {} ({}x{}x4)",
            bytes.len(),
            expected,
            width,
            height
        ));
    }
    let mut tmp = vec![0u8; expected];
    // Horizontal pass.
    for y in 0..height {
        let row_start = y * width * 4;
        for x in 0..width {
            let mut s = [0u32; 4];
            let mut count = 0u32;
            for dx in -(radius as isize)..=(radius as isize) {
                let nx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                let pi = row_start + nx * 4;
                for c in 0..4 {
                    s[c] += bytes[pi + c] as u32;
                }
                count += 1;
            }
            let oi = row_start + x * 4;
            for c in 0..4 {
                tmp[oi + c] = (s[c] / count) as u8;
            }
        }
    }
    // Vertical pass.
    for y in 0..height {
        for x in 0..width {
            let mut s = [0u32; 4];
            let mut count = 0u32;
            for dy in -(radius as isize)..=(radius as isize) {
                let ny = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                let pi = (ny * width + x) * 4;
                for c in 0..4 {
                    s[c] += tmp[pi + c] as u32;
                }
                count += 1;
            }
            let oi = (y * width + x) * 4;
            for c in 0..4 {
                bytes[oi + c] = (s[c] / count) as u8;
            }
        }
    }
    Ok(())
}

/// Radius matching `terrain3d_loader.gd._SOFTEN_RADIUS = 5`. Importing
/// crates that need to feed the same biome data as Terrain3D should
/// pass this radius to [`soften_rgba8`].
pub const TERRAIN3D_SOFTEN_RADIUS: usize = 5;

/// Separable 2D Gaussian blur, identical structure to the one in
/// `features::smooth_class_boundaries`. Clamps at 3σ; clamp-to-edge
/// at the borders.
fn gaussian_blur_2d(src: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let n = src.len();
    if sigma <= 0.0 {
        return src.to_vec();
    }
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

    let mut tmp = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];

    // Horizontal: src → tmp.
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let mut sum = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let sx = (x as isize + k as isize - radius as isize).clamp(0, width as isize - 1)
                    as usize;
                sum += src[row + sx] * kw;
            }
            tmp[row + x] = sum;
        }
    }

    // Vertical: tmp → out.
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for (k, &kw) in kernel.iter().enumerate() {
                let sy = (y as isize + k as isize - radius as isize).clamp(0, height as isize - 1)
                    as usize;
                sum += tmp[sy * width + x] * kw;
            }
            out[y * width + x] = sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_forest_splatmap_is_pure_red_in_map_a() {
        let w = 16;
        let h = 16;
        let bytes = vec![FeatureClass::Forest as u8; w * h];
        let pair = bake_splatmap_pair(&bytes, w, h);
        for px in 0..(w * h) {
            // Map A R should be ~255 (pure forest), other channels ~0.
            assert_eq!(pair.map_a[4 * px], 255, "map_a R at px {px}");
            assert_eq!(pair.map_a[4 * px + 1], 0, "map_a G at px {px}");
            assert_eq!(pair.map_a[4 * px + 2], 0, "map_a B at px {px}");
            assert_eq!(pair.map_a[4 * px + 3], 0, "map_a A at px {px}");
            // Map B all zero.
            for c in 0..4 {
                assert_eq!(pair.map_b[4 * px + c], 0, "map_b ch {c} at px {px}");
            }
        }
    }

    #[test]
    fn pure_cliff_splatmap_is_pure_b_channel_in_map_b() {
        let w = 16;
        let h = 16;
        let bytes = vec![FeatureClass::Cliff as u8; w * h];
        let pair = bake_splatmap_pair(&bytes, w, h);
        // Cliff is splatmap B channel 2 (B).
        for px in 0..(w * h) {
            assert_eq!(pair.map_b[4 * px + 2], 255, "map_b B at px {px}");
            for c in 0..4 {
                assert_eq!(pair.map_a[4 * px + c], 0, "map_a ch {c} at px {px}");
            }
            for c in [0, 1, 3] {
                assert_eq!(pair.map_b[4 * px + c], 0, "map_b ch {c} at px {px}");
            }
        }
    }

    #[test]
    fn line_class_cells_fall_back_to_forest() {
        // PavedRoad doesn't contribute to any splatmap channel —
        // the no-data fallback should paint forest so the shader
        // has something to blend.
        let w = 8;
        let h = 8;
        let bytes = vec![FeatureClass::PavedRoad as u8; w * h];
        let pair = bake_splatmap_pair(&bytes, w, h);
        for px in 0..(w * h) {
            assert_eq!(pair.map_a[4 * px], 255, "fallback to forest R at {px}");
            for c in 1..4 {
                assert_eq!(pair.map_a[4 * px + c], 0);
            }
            for c in 0..4 {
                assert_eq!(pair.map_b[4 * px + c], 0);
            }
        }
    }

    #[test]
    fn two_classes_blend_sums_to_unity() {
        // Half forest, half grassland — every pixel after blur+
        // normalize should have R + G ≈ 255 in splatmap A.
        let w = 32;
        let h = 32;
        let mut bytes = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                bytes[y * w + x] = if x < w / 2 {
                    FeatureClass::Forest as u8
                } else {
                    FeatureClass::Grassland as u8
                };
            }
        }
        let pair = bake_splatmap_pair(&bytes, w, h);
        for px in 0..(w * h) {
            let r = pair.map_a[4 * px] as u32;
            let g = pair.map_a[4 * px + 1] as u32;
            let b = pair.map_a[4 * px + 2] as u32;
            let a = pair.map_a[4 * px + 3] as u32;
            let total = r + g + b + a;
            // Within ±2 of 255 because of u8 rounding loss.
            assert!(
                (253..=257).contains(&total),
                "px {px} total={total}, R={r} G={g} B={b} A={a}"
            );
        }
    }

    #[test]
    fn cliff_keeps_sharp_edge_against_forest() {
        // Half forest (left), half cliff (right) — at the boundary
        // the cliff's σ=0.5 should not bleed forest more than ~1
        // cell into the cliff side. Verify the cell 2 columns into
        // the cliff is still essentially pure cliff.
        let w = 32;
        let h = 16;
        let mut bytes = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                bytes[y * w + x] = if x < w / 2 {
                    FeatureClass::Forest as u8
                } else {
                    FeatureClass::Cliff as u8
                };
            }
        }
        let pair = bake_splatmap_pair(&bytes, w, h);
        // Inspect a pixel deep on the cliff side: x = w/2 + 3, y = h/2.
        let px = (h / 2) * w + (w / 2 + 3);
        let cliff_b = pair.map_b[4 * px + 2];
        assert!(
            cliff_b > 200,
            "cliff B channel too bled at +3 cells into cliff: {cliff_b}"
        );
    }
}
