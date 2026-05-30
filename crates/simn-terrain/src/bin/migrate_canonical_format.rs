//! One-shot migration: canonical heightmap format v1 (`.r16`) → v2 (`.r32`).
//!
//! Walks `godot/assets/terrain/<map_id>/` directories. For each map
//! whose `terrain.toml` declares `format_version = 1`:
//!
//! 1. Reads the legacy `heightmap.r16` and the toml's vertical range.
//! 2. Decodes each u16 sample to literal meters via the v1 linear scale.
//! 3. Writes `heightmap.r32` (LE f32 bytes).
//! 4. Rewrites `terrain.toml` line-by-line — bumping `format_version`
//!    to 2 and replacing `blake3` with the digest of the new `.r32`.
//!    Other lines (comments, field order, formatting) are preserved
//!    exactly so reviewer diffs stay tight.
//! 5. Deletes `heightmap.r16` so the old format doesn't co-exist.
//!
//! Idempotent: maps already at format_version >= 2 are skipped.
//!
//! Run from the workspace root:
//! ```text
//! cargo run -p simn-terrain --bin migrate_canonical_format
//! ```
//!
//! The conversion is bit-exact within u16 precision: `f32 = vert_min +
//! (u16/65535) * (vert_max - vert_min)` is exactly representable in
//! f32 for any input, so re-baking the same source DEM into v2 would
//! produce a heightmap that round-trip-quantizes back to the same u16
//! values this migration emits. No data is lost; gameplay-relevant
//! precision actually improves (no more u16 banding when displayed at
//! shallow gradients).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use simn_terrain::sampler::{encode_r32, legacy_v1};

/// Minimal subset of `TerrainMetadata` we need to read from a v1 toml.
/// Avoid using the full `TerrainMetadata` deserializer here because
/// it now expects `format_version = CURRENT_FORMAT_VERSION = 2` and
/// would reject v1 files outright (Heightmap::load enforces that).
#[derive(serde::Deserialize)]
struct V1MetaSubset {
    format_version: u32,
    map_id: String,
    width: u32,
    height: u32,
    vert_min_m: f32,
    vert_max_m: f32,
}

fn assets_root() -> Result<PathBuf> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = crate_dir.join("../../godot/assets/terrain");
    path.canonicalize()
        .map_err(|e| anyhow!("terrain root {} not accessible: {}", path.display(), e))
}

fn migrate_one(dir: &Path) -> Result<MigrateOutcome> {
    let toml_path = dir.join("terrain.toml");
    let r16_path = dir.join("heightmap.r16");
    let r32_path = dir.join("heightmap.r32");

    if !toml_path.exists() {
        return Ok(MigrateOutcome::SkippedNoToml);
    }

    let toml_text = fs::read_to_string(&toml_path)
        .with_context(|| format!("reading {}", toml_path.display()))?;
    let meta: V1MetaSubset =
        toml::from_str(&toml_text).with_context(|| format!("parsing {}", toml_path.display()))?;

    if meta.format_version >= 2 {
        return Ok(MigrateOutcome::AlreadyV2);
    }
    if meta.format_version != 1 {
        return Err(anyhow!(
            "{}: unexpected format_version {} (expected 1 or >=2)",
            dir.display(),
            meta.format_version
        ));
    }
    if !r16_path.exists() {
        return Err(anyhow!(
            "{}: format_version=1 but heightmap.r16 is missing",
            dir.display()
        ));
    }

    let r16_bytes =
        fs::read(&r16_path).with_context(|| format!("reading {}", r16_path.display()))?;
    let u16_samples = legacy_v1::decode_r16(&r16_bytes).ok_or_else(|| {
        anyhow!(
            "{}: heightmap.r16 has odd byte length {}",
            dir.display(),
            r16_bytes.len()
        )
    })?;
    let expected = meta.width as usize * meta.height as usize;
    if u16_samples.len() != expected {
        return Err(anyhow!(
            "{}: sample count {} != expected {} ({}x{})",
            dir.display(),
            u16_samples.len(),
            expected,
            meta.width,
            meta.height
        ));
    }

    // Convert u16 → meters via the v1 linear scale, then encode as f32 LE.
    let f32_samples: Vec<f32> = u16_samples
        .iter()
        .map(|&s| legacy_v1::u16_to_meters(s, meta.vert_min_m, meta.vert_max_m))
        .collect();
    let r32_bytes = encode_r32(&f32_samples);
    let new_blake3 = blake3::hash(&r32_bytes).to_hex().to_string();

    // Write canonical pair before mutating toml so a partial run leaves
    // a recoverable state (heightmap.r16 + heightmap.r32 + v1 toml).
    fs::write(&r32_path, &r32_bytes).with_context(|| format!("writing {}", r32_path.display()))?;

    // Rewrite toml line-by-line to preserve formatting / comments /
    // field order. Only `format_version` and `blake3` change.
    let new_toml_text = rewrite_toml(&toml_text, &new_blake3)?;
    fs::write(&toml_path, &new_toml_text)
        .with_context(|| format!("writing {}", toml_path.display()))?;

    // Delete the old .r16 only after the new files are committed to disk.
    fs::remove_file(&r16_path).with_context(|| format!("removing {}", r16_path.display()))?;

    Ok(MigrateOutcome::Migrated {
        map_id: meta.map_id,
        width: meta.width,
        height: meta.height,
        bytes: r32_bytes.len(),
    })
}

/// Replace `format_version = 1` (or whatever number) with `2`, and
/// `blake3 = "<old>"` with the new digest. Everything else stays
/// byte-identical. Errors if either field is missing — better to
/// surface that loudly than to silently emit a malformed toml.
fn rewrite_toml(text: &str, new_blake3: &str) -> Result<String> {
    let mut saw_format_version = false;
    let mut saw_blake3 = false;
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("format_version") && line.contains('=') {
            // Preserve leading whitespace; rewrite the value.
            let prefix_len = line.len() - trimmed.len();
            out.push_str(&line[..prefix_len]);
            out.push_str("format_version = 2");
            out.push('\n');
            saw_format_version = true;
            continue;
        }
        if trimmed.starts_with("blake3") && line.contains('=') && !trimmed.starts_with("blake3_") {
            // Match `blake3 = ...`, not `blake3_features = ...` or
            // `features_blake3 = ...`. The exact field name we care
            // about is `blake3`; guard against accidental hits on
            // related fields.
            let prefix_len = line.len() - trimmed.len();
            out.push_str(&line[..prefix_len]);
            out.push_str(&format!("blake3 = \"{}\"", new_blake3));
            out.push('\n');
            saw_blake3 = true;
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    // Preserve trailing-newline behavior of the source: if the
    // original didn't end in `\n`, drop the one we appended.
    if !text.ends_with('\n') {
        out.pop();
    }

    if !saw_format_version {
        return Err(anyhow!("toml is missing a `format_version =` line"));
    }
    if !saw_blake3 {
        return Err(anyhow!("toml is missing a `blake3 =` line"));
    }
    Ok(out)
}

#[derive(Debug)]
enum MigrateOutcome {
    Migrated {
        map_id: String,
        width: u32,
        height: u32,
        bytes: usize,
    },
    AlreadyV2,
    SkippedNoToml,
}

fn main() -> Result<()> {
    let root = assets_root()?;
    println!("migrate_canonical_format: walking {}", root.display());

    let mut migrated = 0u32;
    let mut already = 0u32;
    let mut skipped = 0u32;
    let mut failed: Vec<(PathBuf, String)> = Vec::new();

    let mut entries: Vec<PathBuf> = fs::read_dir(&root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();

    for dir in &entries {
        let map_label = dir.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        match migrate_one(dir) {
            Ok(MigrateOutcome::Migrated {
                map_id,
                width,
                height,
                bytes,
            }) => {
                println!(
                    "  [migrated]  {map_id} ({width}×{height}, {} MB)",
                    bytes as f32 / 1_048_576.0
                );
                migrated += 1;
            }
            Ok(MigrateOutcome::AlreadyV2) => {
                println!("  [skip:v2]   {map_label}");
                already += 1;
            }
            Ok(MigrateOutcome::SkippedNoToml) => {
                println!("  [skip:none] {map_label} — no terrain.toml");
                skipped += 1;
            }
            Err(e) => {
                println!("  [FAIL]      {map_label}: {e:#}");
                failed.push((dir.clone(), format!("{e:#}")));
            }
        }
    }

    println!(
        "\nmigrate_canonical_format: {migrated} migrated, {already} already v2, \
         {skipped} skipped, {} failed",
        failed.len()
    );
    if !failed.is_empty() {
        return Err(anyhow!(
            "migration failed for {} maps; see log above",
            failed.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_preserves_field_order_and_comments() {
        let src = "# header comment\n\
                   format_version = 1\n\
                   map_id = \"x\"\n\
                   blake3 = \"oldhash\"\n\
                   features_blake3 = \"abc\"\n\
                   # trailing comment\n";
        let out = rewrite_toml(src, "newhash").unwrap();
        assert!(out.contains("format_version = 2"));
        assert!(out.contains("blake3 = \"newhash\""));
        // `features_blake3` (different field) untouched.
        assert!(out.contains("features_blake3 = \"abc\""));
        assert!(out.starts_with("# header comment\n"));
        assert!(out.ends_with("# trailing comment\n"));
    }

    #[test]
    fn rewrite_preserves_no_trailing_newline() {
        let src = "format_version = 1\nblake3 = \"x\"";
        let out = rewrite_toml(src, "y").unwrap();
        assert!(!out.ends_with('\n'));
        assert!(out.contains("format_version = 2"));
        assert!(out.contains("blake3 = \"y\""));
    }

    #[test]
    fn rewrite_errors_on_missing_fields() {
        assert!(rewrite_toml("blake3 = \"x\"", "y").is_err());
        assert!(rewrite_toml("format_version = 1", "y").is_err());
    }

    #[test]
    fn rewrite_handles_indented_lines() {
        let src = "  format_version = 1\n\tblake3 = \"x\"\n";
        let out = rewrite_toml(src, "y").unwrap();
        assert!(out.contains("  format_version = 2"));
        assert!(out.contains("\tblake3 = \"y\""));
    }
}
