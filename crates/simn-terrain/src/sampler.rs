//! Pure sampling math — free functions that operate on a raw f32 grid.
//!
//! Kept separate from [`crate::Heightmap`] so the math can be unit-tested
//! independently of I/O. Everything here is `#[inline]`-able and
//! branch-minimal; these functions are called once per AI query and
//! once per server-side player Y-snap, so they shouldn't bottleneck,
//! but staying simple keeps the parity contract clear.
//!
//! Format-version-2 canonical heightmaps store **literal meters** as
//! little-endian f32 (`heightmap.r32`). The legacy v1 u16 path
//! (`heightmap.r16`) lives in [`legacy_v1`] for the one-shot
//! migration binary.

/// Look up a sample at integer grid coordinates, clamping to edges.
#[inline]
pub fn get_clamped(samples: &[f32], width: u32, height: u32, x: i32, y: i32) -> f32 {
    let x = x.clamp(0, width as i32 - 1) as usize;
    let y = y.clamp(0, height as i32 - 1) as usize;
    samples[y * width as usize + x]
}

/// Bilinear interpolation of an f32 heightmap at fractional grid
/// coordinates `(u, v)` where `u` indexes columns (world-X / spacing)
/// and `v` indexes rows (world-Z / spacing).
///
/// Coordinates outside `[0, width-1] x [0, height-1]` clamp to the
/// nearest edge sample.
#[inline]
pub fn bilinear_f32(samples: &[f32], width: u32, height: u32, u: f32, v: f32) -> f32 {
    let u = u.clamp(0.0, (width - 1) as f32);
    let v = v.clamp(0.0, (height - 1) as f32);

    let iu = u.floor() as i32;
    let iv = v.floor() as i32;
    let fu = u - iu as f32;
    let fv = v - iv as f32;

    let h00 = get_clamped(samples, width, height, iu, iv);
    let h10 = get_clamped(samples, width, height, iu + 1, iv);
    let h01 = get_clamped(samples, width, height, iu, iv + 1);
    let h11 = get_clamped(samples, width, height, iu + 1, iv + 1);

    let top = h00 * (1.0 - fu) + h10 * fu;
    let bot = h01 * (1.0 - fu) + h11 * fu;
    top * (1.0 - fv) + bot * fv
}

/// Decode a raw `.r32` byte buffer (little-endian f32, row-major) into
/// a `Vec<f32>` of literal meters. Returns `None` if `bytes.len()` is
/// not a multiple of 4.
pub fn decode_r32(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// Encode an f32 grid (literal meters) to raw little-endian bytes,
/// suitable for writing to a `.r32` file. Inverse of [`decode_r32`].
pub fn encode_r32(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Legacy v1 helpers for reading the deprecated u16 `.r16` format.
///
/// Kept around for the one-shot migration binary
/// (`migrate_canonical_format`). The runtime loader path no longer
/// calls these — format-version-2 maps store f32 directly via
/// [`decode_r32`] / [`encode_r32`].
#[doc(hidden)]
pub mod legacy_v1 {
    /// Decode a raw `.r16` byte buffer (little-endian u16, row-major)
    /// into a `Vec<u16>`. Returns `None` if `bytes.len()` is not a
    /// multiple of 2.
    pub fn decode_r16(bytes: &[u8]) -> Option<Vec<u16>> {
        if !bytes.len().is_multiple_of(2) {
            return None;
        }
        Some(
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
        )
    }

    /// Convert a v1 u16 sample to meters, given the legacy
    /// `[vert_min_m, vert_max_m]` linear scaling. Used only by the
    /// migration binary.
    #[inline]
    pub fn u16_to_meters(sample: u16, vert_min_m: f32, vert_max_m: f32) -> f32 {
        let t = f32::from(sample) / 65535.0;
        vert_min_m + t * (vert_max_m - vert_min_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_midpoint_of_four_corners_is_average() {
        // 2x2 grid: 0.0, 100.0, 200.0, 300.0 (row-major)
        let samples = [0.0f32, 100.0, 200.0, 300.0];
        let v = bilinear_f32(&samples, 2, 2, 0.5, 0.5);
        // Expected: (0 + 100 + 200 + 300) / 4 = 150
        assert!((v - 150.0).abs() < 1e-4, "got {v}");
    }

    #[test]
    fn bilinear_on_grid_node_returns_exact_sample() {
        let samples = [10.0f32, 20.0, 30.0, 40.0];
        assert!((bilinear_f32(&samples, 2, 2, 0.0, 0.0) - 10.0).abs() < 1e-4);
        assert!((bilinear_f32(&samples, 2, 2, 1.0, 0.0) - 20.0).abs() < 1e-4);
        assert!((bilinear_f32(&samples, 2, 2, 0.0, 1.0) - 30.0).abs() < 1e-4);
        assert!((bilinear_f32(&samples, 2, 2, 1.0, 1.0) - 40.0).abs() < 1e-4);
    }

    #[test]
    fn bilinear_clamps_outside_bounds() {
        let samples = [7.0f32, 11.0, 13.0, 17.0];
        // Past the NW corner → clamps to (0, 0)
        assert!((bilinear_f32(&samples, 2, 2, -5.0, -5.0) - 7.0).abs() < 1e-4);
        // Past the SE corner → clamps to (1, 1)
        assert!((bilinear_f32(&samples, 2, 2, 100.0, 100.0) - 17.0).abs() < 1e-4);
    }

    #[test]
    fn bilinear_preserves_negative_meters() {
        // Below sea level — f32 storage handles it naturally; the
        // legacy u16 path could not.
        let samples = [-50.0f32, 0.0, 25.0, 100.0];
        let v = bilinear_f32(&samples, 2, 2, 0.5, 0.5);
        assert!((v - 18.75).abs() < 1e-4, "got {v}");
    }

    #[test]
    fn r32_roundtrip() {
        let samples = vec![0.0f32, 1.5, -100.25, 1234.5678, f32::MIN_POSITIVE];
        let bytes = encode_r32(&samples);
        assert_eq!(bytes.len(), samples.len() * 4);
        let decoded = decode_r32(&bytes).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn r32_decode_rejects_misaligned_length() {
        assert!(decode_r32(&[0u8, 1, 2]).is_none());
        assert!(decode_r32(&[0u8, 1, 2, 3, 4]).is_none());
    }

    #[test]
    fn legacy_v1_decode_r16() {
        let samples: Vec<u16> = vec![0, 1, 65535, 32768];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let decoded = legacy_v1::decode_r16(&bytes).unwrap();
        assert_eq!(decoded, samples);
        assert!(legacy_v1::decode_r16(&[0u8, 1, 2]).is_none());
    }

    #[test]
    fn legacy_v1_u16_to_meters_endpoints() {
        assert!((legacy_v1::u16_to_meters(0, 100.0, 500.0) - 100.0).abs() < 1e-4);
        assert!((legacy_v1::u16_to_meters(65535, 100.0, 500.0) - 500.0).abs() < 1e-4);
        assert!((legacy_v1::u16_to_meters(32768, 0.0, 100.0) - 50.000763).abs() < 1e-3);
    }
}
