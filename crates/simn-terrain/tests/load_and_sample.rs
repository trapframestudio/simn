//! End-to-end tests: write a deterministic fixture to disk, load it
//! through the public API, sample it, and assert against analytical
//! ground truth.
//!
//! Fixture is synthesized per-test into a `tempfile::TempDir` so the
//! repository stays free of binary blobs.

use std::fs;

use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
use simn_terrain::sampler::encode_r32;
use simn_terrain::{Heightmap, TerrainMetadata};
use tempfile::TempDir;

/// Write a canonical fixture (`terrain.toml` + `heightmap.r32`) into
/// `dir`. Caller supplies the f32 grid (literal meters) and its
/// declared metadata; the blake3 digest is computed and inserted so
/// integrity checking runs.
fn write_fixture(dir: &std::path::Path, mut metadata: TerrainMetadata, samples: &[f32]) {
    let bytes = encode_r32(samples);
    metadata.blake3 = blake3::hash(&bytes).to_hex().to_string();
    let toml_text = toml::to_string(&metadata).unwrap();
    fs::write(dir.join("terrain.toml"), toml_text).unwrap();
    fs::write(dir.join("heightmap.r32"), bytes).unwrap();
}

fn simple_metadata(width: u32, height: u32) -> TerrainMetadata {
    TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: "test_fixture".into(),
        width,
        height,
        spacing_m: 2.0,
        vert_min_m: 0.0,
        vert_max_m: 1000.0,
        origin_utm_zone: "10N".into(),
        origin_utm_easting: 0.0,
        origin_utm_northing: 0.0,
        blake3: String::new(),
        features_blake3: String::new(),
        region_size_m: 2048.0,
        playable_extent_x_m: 0.0,
        playable_extent_z_m: 0.0,
        nav_mask_format_version: 0,
        nav_mask_blake3: String::new(),
    }
}

#[test]
fn loads_and_samples_flat_fixture() {
    let tmp = TempDir::new().unwrap();
    // 4x4 grid, all samples = 500.0 m (literal meters, half of vert range).
    let samples = vec![500.0f32; 16];
    write_fixture(tmp.path(), simple_metadata(4, 4), &samples);

    let hm = Heightmap::load(tmp.path()).unwrap();
    // Sample anywhere — should be uniform.
    for (x, z) in [(0.0, 0.0), (3.5, 4.2), (5.9, 5.9)] {
        let y = hm.sample(x, z);
        assert!(
            (y - 500.0).abs() < 1e-4,
            "flat fixture: x={x} z={z} y={y} expected=500.0"
        );
    }
}

#[test]
fn sample_interpolates_linear_ramp_along_x() {
    // Height ramps in +X: row-major sample at (col, row) = col * K meters.
    let w = 5u32;
    let h = 2u32;
    let k = 250.0f32; // meters per column step
    let mut samples = Vec::with_capacity((w * h) as usize);
    for _row in 0..h {
        for col in 0..w {
            samples.push(col as f32 * k);
        }
    }
    let tmp = TempDir::new().unwrap();
    write_fixture(tmp.path(), simple_metadata(w, h), &samples);

    let hm = Heightmap::load(tmp.path()).unwrap();
    // spacing = 2 m, so world-X = col * 2. Midpoint col 0↔1 is world-X 1.
    // f32 at col 0 = 0, at col 1 = 250 → bilinear midpoint = 125.
    let y = hm.sample(1.0, 0.0);
    assert!(
        (y - 125.0).abs() < 1e-3,
        "ramp midpoint: got {y} expected 125"
    );
}

#[test]
fn sample_clamps_outside_map() {
    let tmp = TempDir::new().unwrap();
    // 2x2 grid with a clear gradient (literal meters).
    let samples = vec![0.0f32, 200.0, 600.0, 900.0];
    write_fixture(tmp.path(), simple_metadata(2, 2), &samples);

    let hm = Heightmap::load(tmp.path()).unwrap();
    // Far outside NW: should clamp to (0, 0) sample = 0
    assert!(hm.sample(-100.0, -100.0).abs() < 1e-3);
    // Far outside SE: should clamp to (1, 1) sample = 900
    let got = hm.sample(10_000.0, 10_000.0);
    assert!(
        (got - 900.0).abs() < 1e-3,
        "SE clamp: got {got} expected 900"
    );
}

#[test]
fn negative_meters_round_trip() {
    // f32 storage handles below-sea-level natively; v1 u16 could not.
    let tmp = TempDir::new().unwrap();
    let samples = vec![-50.0f32, -25.0, -10.0, 5.0];
    let mut md = simple_metadata(2, 2);
    md.vert_min_m = -100.0;
    md.vert_max_m = 100.0;
    write_fixture(tmp.path(), md, &samples);

    let hm = Heightmap::load(tmp.path()).unwrap();
    assert!((hm.sample(0.0, 0.0) - (-50.0)).abs() < 1e-4);
    assert!((hm.sample(2.0, 0.0) - (-25.0)).abs() < 1e-4);
}

#[test]
fn normal_is_up_on_flat_terrain() {
    let tmp = TempDir::new().unwrap();
    let samples = vec![123.45f32; 16];
    write_fixture(tmp.path(), simple_metadata(4, 4), &samples);
    let hm = Heightmap::load(tmp.path()).unwrap();
    let n = hm.sample_normal(3.0, 3.0);
    assert!(n[0].abs() < 1e-5, "nx expected 0, got {}", n[0]);
    assert!((n[1] - 1.0).abs() < 1e-5, "ny expected 1, got {}", n[1]);
    assert!(n[2].abs() < 1e-5, "nz expected 0, got {}", n[2]);
}

#[test]
fn load_rejects_hash_mismatch() {
    let tmp = TempDir::new().unwrap();
    let samples = vec![0.0f32; 16];
    let mut md = simple_metadata(4, 4);
    md.blake3 = "0".repeat(64); // wrong but correctly-shaped digest
    let toml_text = toml::to_string(&md).unwrap();
    fs::write(tmp.path().join("terrain.toml"), toml_text).unwrap();
    fs::write(tmp.path().join("heightmap.r32"), encode_r32(&samples)).unwrap();

    let err = Heightmap::load(tmp.path()).unwrap_err().to_string();
    assert!(err.contains("hash mismatch"), "unexpected error: {err}");
}

#[test]
fn load_rejects_sample_count_mismatch() {
    let tmp = TempDir::new().unwrap();
    // Declare 4x4 but write only 4 samples.
    let samples = vec![0.0f32; 4];
    write_fixture(tmp.path(), simple_metadata(4, 4), &samples);
    let err = Heightmap::load(tmp.path()).unwrap_err().to_string();
    assert!(err.contains("sample count mismatch"), "unexpected: {err}");
}

#[test]
fn load_rejects_unknown_format_version() {
    let tmp = TempDir::new().unwrap();
    let mut md = simple_metadata(2, 2);
    md.format_version = 999;
    write_fixture(tmp.path(), md, &[0.0f32; 4]);
    let err = Heightmap::load(tmp.path()).unwrap_err().to_string();
    assert!(err.contains("format version"), "unexpected: {err}");
}
