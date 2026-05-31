//! Headless NPC-behavior watcher.
//!
//! Spins up a fresh `Sim` in a temp dir, enables `BehaviorLog`, and
//! ticks at wall-clock 20Hz for `SIMN_WATCH_SECONDS` (default 30),
//! streaming human-readable NPC events to stdout. Prints a summary
//! line every `SIMN_WATCH_SUMMARY_TICKS` (default 100) ticks.
//!
//! Run:
//!
//! ```bash
//! cargo run --example watch -p simn-sim
//! SIMN_WATCH_SECONDS=120 cargo run --example watch -p simn-sim
//! RUST_LOG=npc.behavior=info cargo run --example watch -p simn-sim
//! ```
//!
//! The default `RUST_LOG` value is `npc.behavior=info` so only the
//! NPC events print; crank it up to see more sim internals.

use simn_sim::{RegionGraph, SavePaths, Sim};
use std::time::{Duration, Instant};
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("npc.behavior=info,simn_sim=warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_level(false)
        .without_time()
        .init();

    let seconds: u64 = std::env::var("SIMN_WATCH_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let summary_every: u64 = std::env::var("SIMN_WATCH_SUMMARY_TICKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let dir = tempfile::tempdir()?;
    let paths = SavePaths::in_dir(dir.path());
    let graph = RegionGraph::default_test_graph();
    let mut sim = Sim::new(paths, graph)?;
    sim.set_behavior_log(true);

    eprintln!(
        "# watching for {seconds}s @ 20Hz, summary every {summary_every} ticks, dir={}",
        dir.path().display()
    );

    let tick_period = Duration::from_millis(50);
    let start = Instant::now();
    let deadline = start + Duration::from_secs(seconds);
    let mut next_tick = Instant::now();

    while Instant::now() < deadline {
        if Instant::now() >= next_tick {
            sim.tick()?;
            next_tick += tick_period;
            let t = sim.current_tick();
            if summary_every > 0 && t > 0 && t.is_multiple_of(summary_every) {
                let c = sim.chronicle_summary();
                eprintln!(
                    "# tick={t} alive={} ever={}",
                    c.currently_alive, c.total_ever_spawned
                );
            }
        } else {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    eprintln!("# done; shutting down");
    sim.shutdown()?;
    Ok(())
}
