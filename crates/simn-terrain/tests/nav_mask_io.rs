//! Iteration 5-13 Phase A2 integration tests: write a canonical
//! fixture pair (`heightmap.r32` + `nav_mask.r8` + `terrain.toml`)
//! to disk, then verify `Heightmap::load` honors the integrity
//! checks on `nav_mask.r8` the same way it does on `features.r8`.

use std::fs;

use simn_terrain::metadata::CURRENT_FORMAT_VERSION;
use simn_terrain::sampler::encode_r32;
use simn_terrain::{Heightmap, NavOverride, TerrainMetadata, NAV_MASK_FORMAT_VERSION};
use tempfile::TempDir;

const W: u32 = 16;
const H: u32 = 16;

fn base_metadata() -> TerrainMetadata {
    TerrainMetadata {
        format_version: CURRENT_FORMAT_VERSION,
        map_id: "nav_mask_io_test".into(),
        width: W,
        height: H,
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

/// Write a heightmap + (optional) nav_mask fixture. Always populates
/// the heightmap's blake3 so the integrity check fires.
fn write_fixture(dir: &std::path::Path, nav_mask: Option<&[u8]>) -> TerrainMetadata {
    let mut metadata = base_metadata();
    let samples = vec![100.0_f32; (W * H) as usize];
    let r32_bytes = encode_r32(&samples);
    metadata.blake3 = blake3::hash(&r32_bytes).to_hex().to_string();
    if let Some(mask) = nav_mask {
        assert_eq!(mask.len(), (W * H) as usize);
        metadata.nav_mask_format_version = NAV_MASK_FORMAT_VERSION;
        metadata.nav_mask_blake3 = blake3::hash(mask).to_hex().to_string();
        fs::write(dir.join("nav_mask.r8"), mask).unwrap();
    }
    let toml_text = toml::to_string(&metadata).unwrap();
    fs::write(dir.join("terrain.toml"), toml_text).unwrap();
    fs::write(dir.join("heightmap.r32"), r32_bytes).unwrap();
    metadata
}

#[test]
fn loads_paired_nav_mask_when_declared() {
    let tmp = TempDir::new().unwrap();
    // Paint a single `ForceBlocked` cell at (col=4, row=5).
    let mut mask = vec![0u8; (W * H) as usize];
    mask[(5 * W + 4) as usize] = NavOverride::ForceBlocked as u8;
    write_fixture(tmp.path(), Some(&mask));

    let hm = Heightmap::load(tmp.path()).expect("load with nav_mask");
    assert_eq!(hm.nav_override_at(4, 5), NavOverride::ForceBlocked);
    // Neighbor cells should still report Default.
    assert_eq!(hm.nav_override_at(3, 5), NavOverride::Default);
    assert_eq!(hm.nav_override_at(4, 6), NavOverride::Default);
    // Raw bytes accessor returns the same data we wrote.
    let bytes = hm.nav_mask_bytes().expect("nav_mask bytes present");
    assert_eq!(bytes, mask.as_slice());
}

#[test]
fn missing_nav_mask_is_noop() {
    let tmp = TempDir::new().unwrap();
    write_fixture(tmp.path(), None);
    let hm = Heightmap::load(tmp.path()).expect("load without nav_mask");
    assert!(hm.nav_mask_bytes().is_none());
    // Every cell reports Default.
    assert_eq!(hm.nav_override_at(0, 0), NavOverride::Default);
    assert_eq!(hm.nav_override_at(7, 9), NavOverride::Default);
}

#[test]
fn corrupted_nav_mask_blake3_errors_on_load() {
    let tmp = TempDir::new().unwrap();
    let mask = vec![0u8; (W * H) as usize];
    write_fixture(tmp.path(), Some(&mask));
    // Mutate the on-disk file so its hash no longer matches what
    // `terrain.toml` says.
    let mut corrupted = mask.clone();
    corrupted[0] = NavOverride::ForceBlocked as u8;
    fs::write(tmp.path().join("nav_mask.r8"), &corrupted).unwrap();
    let err = Heightmap::load(tmp.path()).expect_err("hash mismatch should fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("nav_mask_blake3"),
        "error should name nav_mask_blake3; got: {msg}"
    );
}

#[test]
fn nav_mask_size_mismatch_errors_on_load() {
    let tmp = TempDir::new().unwrap();
    let mask = vec![0u8; (W * H) as usize];
    let _metadata = write_fixture(tmp.path(), Some(&mask));
    // Truncate the on-disk file but keep the toml's hash matching
    // the truncated bytes. Loader should reject on length check
    // even when the hash is consistent.
    let truncated = vec![0u8; ((W * H) as usize) / 2];
    let new_hash = blake3::hash(&truncated).to_hex().to_string();
    fs::write(tmp.path().join("nav_mask.r8"), &truncated).unwrap();
    // Patch the toml's nav_mask_blake3 to match the new (shorter) file.
    let toml_text = fs::read_to_string(tmp.path().join("terrain.toml")).unwrap();
    let patched = toml_text
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("nav_mask_blake3") {
                format!("nav_mask_blake3 = \"{new_hash}\"")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(tmp.path().join("terrain.toml"), patched).unwrap();
    let err = Heightmap::load(tmp.path()).expect_err("size mismatch should fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("nav_mask size mismatch"),
        "error should name nav_mask size mismatch; got: {msg}"
    );
}

#[test]
fn nav_mask_unknown_format_version_errors_on_load() {
    let tmp = TempDir::new().unwrap();
    let mask = vec![0u8; (W * H) as usize];
    let _metadata = write_fixture(tmp.path(), Some(&mask));
    // Patch format_version to an unknown future version.
    let toml_text = fs::read_to_string(tmp.path().join("terrain.toml")).unwrap();
    let patched = toml_text
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("nav_mask_format_version") {
                "nav_mask_format_version = 99".to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(tmp.path().join("terrain.toml"), patched).unwrap();
    let err = Heightmap::load(tmp.path()).expect_err("unknown version should fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("nav_mask format version"),
        "error should name nav_mask format version; got: {msg}"
    );
}
