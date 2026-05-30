//! Determinism harness for the ticked simulation.
//!
//! Per `docs/book/src/planning/sim-hardening-plan.md` §2: prove the
//! sim produces identical outputs given identical inputs. Same seed
//! + same input stream → byte-for-byte identical snapshots, every
//! tick, every run, on this platform.
//!
//! Catches:
//! - `HashMap` iteration order leaking into observable state.
//! - Stray `rand::thread_rng()` / `SystemTime::now()` calls.
//! - Float NaN / order-dependent reductions.
//! - Bevy query iteration order regressions.
//!
//! Does NOT catch cross-platform drift (FPU mode, fmadd intrinsics);
//! that's a fixed-point arithmetic problem out of scope here.

use simn_sim::{ContentSource, RegionGraph, SavePaths, Sim};
use tempfile::TempDir;

fn paths(dir: &TempDir) -> SavePaths {
    SavePaths::in_dir(dir.path())
}

/// Two `Sim::new_with_seed(_, _, 42)` instances ticked side-by-side
/// for N ticks must produce byte-identical in-memory snapshots. The
/// `write_snapshot_to_vec` API includes the magic + version + tick +
/// body bytes + blake3 hash, so this catches even single-bit drift.
#[test]
fn ticked_sim_is_deterministic() {
    let dir_a = TempDir::new().unwrap();
    let dir_b = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim_a = Sim::new_with_seed(paths(&dir_a), graph.clone(), 42).unwrap();
    let mut sim_b = Sim::new_with_seed(paths(&dir_b), graph, 42).unwrap();
    // Scale population targets way down so the per-tick cost is
    // bearable in debug. The determinism contract is independent of
    // population size — fewer NPCs is just a faster check, not a
    // weaker one.
    sim_a.scale_all_population_targets(0.02);
    sim_b.scale_all_population_targets(0.02);
    // Phase 1A gate: enable tick-time spawning across every region
    // so two same-seed sims actually populate (and converge or
    // diverge — that's what we're testing).
    sim_a.activate_all_regions_for_test();
    sim_b.activate_all_regions_for_test();

    // Drive identical inputs - here, the simplest stream: 200 plain
    // ticks with no external input. Anything that diverges under
    // pure deterministic stepping must be a bug.
    for _ in 0..200 {
        sim_a.tick().unwrap();
        sim_b.tick().unwrap();
    }

    let snap_a = sim_a.write_snapshot_to_vec().expect("snapshot a");
    let snap_b = sim_b.write_snapshot_to_vec().expect("snapshot b");
    assert_eq!(
        snap_a.len(),
        snap_b.len(),
        "snapshot byte-length differs after 200 ticks ({} vs {})",
        snap_a.len(),
        snap_b.len(),
    );
    if snap_a != snap_b {
        // Find the first diverging byte for actionable diagnostics.
        let first_diff = snap_a
            .iter()
            .zip(snap_b.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(snap_a.len());
        panic!(
            "ticked sim snapshots diverged at byte {} (of {}); a={:?} b={:?}",
            first_diff,
            snap_a.len(),
            &snap_a[first_diff.saturating_sub(8)..(first_diff + 8).min(snap_a.len())],
            &snap_b[first_diff.saturating_sub(8)..(first_diff + 8).min(snap_b.len())],
        );
    }
}

/// Step a single sim forward and snapshot at multiple intervals;
/// then run another sim with the same seed and assert the *interim*
/// snapshots also match. Catches drift that only manifests after
/// ticking past some threshold (e.g. spawn-gated systems firing only
/// after population targets are reached).
#[test]
fn ticked_sim_snapshots_match_at_intervals() {
    let dir_a = TempDir::new().unwrap();
    let dir_b = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim_a = Sim::new_with_seed(paths(&dir_a), graph.clone(), 7).unwrap();
    let mut sim_b = Sim::new_with_seed(paths(&dir_b), graph, 7).unwrap();
    sim_a.scale_all_population_targets(0.02);
    sim_b.scale_all_population_targets(0.02);
    // Phase 1A gate: enable tick-time spawning across every region
    // so two same-seed sims actually populate (and converge or
    // diverge — that's what we're testing).
    sim_a.activate_all_regions_for_test();
    sim_b.activate_all_regions_for_test();

    // Three checkpoints: 50, 150, 400 ticks.
    let checkpoints = [50u32, 150, 400];
    let mut last = 0u32;
    for cp in checkpoints {
        let delta = cp - last;
        for _ in 0..delta {
            sim_a.tick().unwrap();
            sim_b.tick().unwrap();
        }
        let a = sim_a.write_snapshot_to_vec().unwrap();
        let b = sim_b.write_snapshot_to_vec().unwrap();
        assert_eq!(
            a, b,
            "snapshot divergence at tick {} (after population convergence + spawn / aggro / objective passes)",
            cp
        );
        last = cp;
    }
}

/// Content-source equivalence: a sim built from the embedded example
/// pack and one built from that same pack extracted to disk
/// (`ContentSource::Dir`) must tick byte-identically at the same seed.
/// Proves the two `ContentSource` resolvers yield identical content
/// and that the non-cached `Dir` parse path is itself deterministic —
/// the foundation of the content-externalization refactor.
#[test]
fn embedded_and_explicit_dir_match() {
    let pack_dir = TempDir::new().unwrap();
    ContentSource::extract_embedded_to(pack_dir.path()).expect("extract embedded pack");

    let dir_a = TempDir::new().unwrap();
    let dir_b = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim_a = Sim::new_with_seed(paths(&dir_a), graph.clone(), 42).unwrap();
    let mut sim_b = Sim::new_with_seed_and_content(
        paths(&dir_b),
        graph,
        42,
        ContentSource::Dir(pack_dir.path().to_path_buf()),
    )
    .unwrap();
    sim_a.scale_all_population_targets(0.02);
    sim_b.scale_all_population_targets(0.02);
    sim_a.activate_all_regions_for_test();
    sim_b.activate_all_regions_for_test();

    for _ in 0..200 {
        sim_a.tick().unwrap();
        sim_b.tick().unwrap();
    }

    let snap_a = sim_a.write_snapshot_to_vec().expect("snapshot a");
    let snap_b = sim_b.write_snapshot_to_vec().expect("snapshot b");
    assert_eq!(
        snap_a, snap_b,
        "embedded vs extracted-Dir content produced diverging sims at seed 42 — \
         the resolvers disagree or the Dir parse is non-deterministic"
    );
}

/// Cross-seed sanity: two sims with *different* seeds must NOT
/// produce identical snapshots. Catches a reverse failure mode -
/// if every test passes because the sim is oblivious to its seed,
/// the determinism guarantee is hollow.
#[test]
fn different_seeds_diverge() {
    let dir_a = TempDir::new().unwrap();
    let dir_b = TempDir::new().unwrap();
    let graph = RegionGraph::default_test_graph();
    let mut sim_a = Sim::new_with_seed(paths(&dir_a), graph.clone(), 1).unwrap();
    let mut sim_b = Sim::new_with_seed(paths(&dir_b), graph, 999_999).unwrap();
    sim_a.scale_all_population_targets(0.02);
    sim_b.scale_all_population_targets(0.02);
    // Phase 1A gate: enable tick-time spawning across every region
    // so two same-seed sims actually populate (and converge or
    // diverge — that's what we're testing).
    sim_a.activate_all_regions_for_test();
    sim_b.activate_all_regions_for_test();
    // ~150 ticks is enough for population spawning to kick in and
    // diverge between seeds (different RNG draws for spawn anchors,
    // squad sizes, etc.).
    for _ in 0..150 {
        sim_a.tick().unwrap();
        sim_b.tick().unwrap();
    }
    let snap_a = sim_a.write_snapshot_to_vec().unwrap();
    let snap_b = sim_b.write_snapshot_to_vec().unwrap();
    assert_ne!(
        snap_a, snap_b,
        "different seeds produced identical snapshots - either the seed isn't reaching the RNG, \
         or the world has no seed-driven content"
    );
}
