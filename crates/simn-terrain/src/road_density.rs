//! Road-density baking — convert the 1-3 cell-wide line classes
//! (`PavedRoad` / `UnpavedRoad` / `Trail`) from the per-cell
//! `features.r8` byte grid into a 4-channel RGBA8 density texture
//! sampled with `filter_linear` in the shader.
//!
//! ## Why this exists
//!
//! Roads in `features.r8` are categorical: a cell either is or
//! isn't a road. At our 2 m grid that produces visible stair-step
//! at every road edge, corner, and intersection — sharp 90° pixel
//! transitions on what should read as a smooth ribbon. Smoothing
//! at shader time (5×5 neighborhood Gaussian) widens the *edge*
//! falloff but doesn't fix the underlying *centerline* geometry —
//! a 4-cell-wide road still turns corners with 4-cell pixel jumps.
//!
//! Baking a separate Gaussian-blurred density texture gives:
//! - sub-cell precision via hardware `filter_linear` at sample time
//! - smoothed centerline corners (the blur happens on the data, not
//!   just on the falloff)
//! - per-class density preservation (paved/unpaved/trail kept on
//!   separate channels so the shader can pick the right surface
//!   texture per pixel)
//!
//! ## Channel layout (RGBA8, length 4 × W × H)
//!
//! - **R** — PavedRoad density (0..255), σ ≈ 1.0 cells
//! - **G** — UnpavedRoad density, σ ≈ 1.0
//! - **B** — Trail density, σ ≈ 0.75 (narrower brush, sharper)
//! - **A** — reserved (could carry a "road age" / "wear" channel
//!   later for visual variety)
//!
//! Channels are independent — a cell *can* have weight in two
//! channels if a paved road and a trail run nearly parallel within
//! the blur kernel. The shader resolves precedence (paved > unpaved
//! > trail).

use crate::features::FeatureClass;

/// Per-class σ in grid cells. Tuned to give roads a smooth fade
/// beyond their nominal brush width — wider σ = more visually
/// pronounced road, softer transition into the surrounding biome.
/// Bumped from the initial 1.0 / 0.75 round after user feedback
/// asking for more road treatment.
const PAVED_SIGMA: f32 = 1.5;
const UNPAVED_SIGMA: f32 = 1.5;
const TRAIL_SIGMA: f32 = 1.0;

/// Bake the 4-channel road-density texture. Length = `4 * width *
/// height` (RGBA8 row-major, NW origin). Cells where no line class
/// is within blur reach get all-zero density and the shader falls
/// back to the base biome diffuse.
pub fn bake_road_density(bytes: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_eq!(
        bytes.len(),
        width * height,
        "bake_road_density: byte count must equal width × height"
    );
    let n = bytes.len();

    let mut paved_mask = vec![0.0f32; n];
    let mut unpaved_mask = vec![0.0f32; n];
    let mut trail_mask = vec![0.0f32; n];
    for (i, &b) in bytes.iter().enumerate() {
        if b == FeatureClass::PavedRoad as u8 {
            paved_mask[i] = 1.0;
        } else if b == FeatureClass::UnpavedRoad as u8 {
            unpaved_mask[i] = 1.0;
        } else if b == FeatureClass::Trail as u8 {
            trail_mask[i] = 1.0;
        }
    }

    let paved = gaussian_blur_2d(&paved_mask, width, height, PAVED_SIGMA);
    let unpaved = gaussian_blur_2d(&unpaved_mask, width, height, UNPAVED_SIGMA);
    let trail = gaussian_blur_2d(&trail_mask, width, height, TRAIL_SIGMA);

    let mut out = vec![0u8; 4 * n];
    for i in 0..n {
        out[4 * i] = (paved[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        out[4 * i + 1] = (unpaved[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        out[4 * i + 2] = (trail[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        // A channel reserved.
    }
    out
}

/// Separable 2D Gaussian blur — same shape as the helper in
/// `splatmap.rs`. Clamps at 3σ; clamp-to-edge at borders.
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
    fn empty_grid_produces_zero_density() {
        let w = 16;
        let h = 16;
        let bytes = vec![FeatureClass::Forest as u8; w * h];
        let density = bake_road_density(&bytes, w, h);
        assert!(density.iter().all(|&b| b == 0));
    }

    #[test]
    fn single_paved_cell_centers_max_in_red_channel() {
        // A single PavedRoad cell at (8,8) in an otherwise-Forest
        // grid. Center of the blur kernel peaks at 1/(2πσ²);
        // for σ=1.5 that's ~0.07, so a single mass-1 cell renders
        // ~18 in the R byte. Verify it's positive and dominates
        // the unpaved/trail channels.
        let w = 17;
        let h = 17;
        let mut bytes = vec![FeatureClass::Forest as u8; w * h];
        bytes[8 * w + 8] = FeatureClass::PavedRoad as u8;
        let density = bake_road_density(&bytes, w, h);
        let r_at_center = density[4 * (8 * w + 8)];
        assert!(r_at_center > 10, "R channel at road center: {r_at_center}");
        let g = density[4 * (8 * w + 8) + 1];
        let b = density[4 * (8 * w + 8) + 2];
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn paved_unpaved_trail_route_to_distinct_channels() {
        // One cell of each line class; verify each lands in its
        // own channel.
        let w = 32;
        let h = 32;
        let mut bytes = vec![FeatureClass::Grassland as u8; w * h];
        bytes[5 * w + 5] = FeatureClass::PavedRoad as u8;
        bytes[5 * w + 15] = FeatureClass::UnpavedRoad as u8;
        bytes[5 * w + 25] = FeatureClass::Trail as u8;
        let density = bake_road_density(&bytes, w, h);
        // Paved at (5,5) → R only
        let i = 5 * w + 5;
        assert!(density[4 * i] > 10);
        assert_eq!(density[4 * i + 1], 0);
        assert_eq!(density[4 * i + 2], 0);
        // Unpaved at (5,15) → G only
        let i = 5 * w + 15;
        assert!(density[4 * i + 1] > 10);
        assert_eq!(density[4 * i], 0);
        assert_eq!(density[4 * i + 2], 0);
        // Trail at (5,25) → B only (TRAIL_SIGMA = 1.0 stays sharper)
        let i = 5 * w + 25;
        assert!(density[4 * i + 2] > 25);
        assert_eq!(density[4 * i], 0);
        assert_eq!(density[4 * i + 1], 0);
    }

    #[test]
    fn density_reserves_alpha_channel() {
        let w = 8;
        let h = 8;
        let mut bytes = vec![FeatureClass::Forest as u8; w * h];
        bytes[4 * w + 4] = FeatureClass::PavedRoad as u8;
        let density = bake_road_density(&bytes, w, h);
        // Every alpha byte is zero (reserved).
        for i in 0..(w * h) {
            assert_eq!(density[4 * i + 3], 0, "A channel non-zero at px {i}");
        }
    }
}
