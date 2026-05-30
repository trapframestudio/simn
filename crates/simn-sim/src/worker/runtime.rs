//! `SimWorker` — the dedicated thread that owns `Sim` and runs
//! the tick loop. Step 3 of the threaded-sim PR C rollout.
//!
//! ## What this step lands
//!
//! - A real `std::thread`, named `"simn-sim"`, that owns
//!   the `Sim` value for its lifetime.
//! - A `crossbeam_channel::Sender<SimCommand>` for the main
//!   thread to enqueue mutations (drained at top of each tick).
//! - Two `Arc<ArcSwap<Option<T>>>` cells the worker writes
//!   at end-of-tick and the main thread reads lock-free:
//!   one for the `(prev, curr)` snapshot pair (renderer lerp),
//!   one for the `SimView` (HUD reads). See plan doc §4 / §5.
//! - A `Shutdown` command with a `oneshot` reply so the main
//!   thread can join the worker cleanly.
//!
//! ## What this step deliberately doesn't land
//!
//! - **Bridge rewires.** `SimHost` still owns `Sim` directly
//!   today; step 4 of the rollout is the mechanical flip from
//!   `&mut self.sim` to `self.worker.send(...)` / `self.worker.view()`.
//!   This step is exercised by a Rust smoke test only.
//! - **Load lifecycle.** The worker is constructed *with* a
//!   ready `Sim` value (`SimWorker::spawn(sim)`); the host
//!   builds the `Sim` synchronously and hands it off. Async
//!   `Load` lands when step 7 splits lifecycle off the host.
//! - **Panic propagation.** A worker panic today just unwinds
//!   the thread; the main thread doesn't observe it. Step 7
//!   adds the `Arc<AtomicBool>` flag + `sim_error` signal.
//! - **Delta / FX forwarding.** Step 6 adds the
//!   `crossbeam_channel<DeltaBatch>` from worker → main; this
//!   step's snapshot publish is enough for the lerp path to
//!   work but doesn't yet stream `projectile_spawned` etc.
//!
//! ## Tick clock
//!
//! The worker's main loop uses `recv_timeout` against the
//! command channel with the timeout set to the remaining time
//! until the next 50 ms deadline. `crossbeam-channel` returns
//! `Err(RecvTimeoutError::Timeout)` exactly when the deadline
//! is reached — that's the unified "wake to tick" path. No
//! `thread::sleep` needed.
//!
//! Catch-up: if the previous tick took longer than 50 ms (e.g.
//! a slow planner pass) the next deadline lands in the past;
//! the loop ticks immediately and rolls the deadline forward
//! one period at a time until it's back in the future. This
//! matches the doc's "drop to 20 Hz on overload, don't
//! double-tick" contract.

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::Result;
use arc_swap::ArcSwap;
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};

use crate::snapshot::SimSnapshot;
use crate::worker::command::{dispatch_command, SimCommand};
use crate::worker::view::{build_sim_view, SimView};
use crate::world::Sim;

/// Bounded capacity for the command channel. 256 = ~3 ticks
/// of worst-case input (12 players × 144 Hz Move = ~86
/// commands per tick). Sized so a one-tick spike doesn't drop
/// commands but a wedged worker is observable as `try_send`
/// failures rather than unbounded memory growth.
pub const COMMAND_CHANNEL_CAPACITY: usize = 256;

/// Fixed sim tick period — 20 Hz.
pub const TICK_PERIOD: Duration = Duration::from_millis(50);

/// The most recent snapshot pair the worker has published.
/// `Some` once the worker has completed at least two ticks
/// (so both `prev` and `curr` are populated). Read by the
/// renderer to lerp NPC poses; see [`crate::snapshot`].
#[derive(Clone, Debug)]
pub struct PublishedSnapshots {
    pub prev: SimSnapshot,
    pub curr: SimSnapshot,
}

/// Type-erased closure the main thread sends to the worker to
/// run against `&mut Sim`. The closure is responsible for
/// capturing whatever reply channel it needs and signalling
/// the caller (see [`SimWorker::inspect`] for the typed
/// wrapper). The worker drains these between command drain
/// and tick, so they observe a fully-applied command queue
/// but pre-tick state for the round they're served on.
type InspectFn = Box<dyn FnOnce(&mut Sim) + Send + 'static>;

/// Cross-thread payload bundled at end-of-tick. Both cells
/// are exposed independently so consumers that only care
/// about one (renderer = snapshots only) don't pay for the
/// other.
pub(crate) struct Channels {
    pub cmd_rx: Receiver<SimCommand>,
    pub inspect_rx: Receiver<InspectFn>,
    pub snapshots: Arc<ArcSwap<Option<PublishedSnapshots>>>,
    pub view: Arc<ArcSwap<Option<SimView>>>,
    pub shutdown_rx: Receiver<()>,
    /// Per-tick drained `WorldDelta` batches. The worker pushes
    /// `Sim::drain_tick_deltas()` here each tick; the main thread
    /// (`SimHost::process`) drains it to emit projectile-FX
    /// signals + future network broadcasts. Bounded so a
    /// pathological stall doesn't grow unbounded — drops oldest
    /// on full (FX is "missed shots show no tracer" worst case,
    /// not load-bearing for sim correctness).
    pub deltas_tx: Sender<Vec<crate::delta::WorldDelta>>,
}

/// Handle to the dedicated sim worker thread. Drop joins the
/// thread (after sending `Shutdown` first via [`shutdown`])
/// — leaks if shutdown wasn't called explicitly, since drop
/// can't wait synchronously on the thread's last tick.
pub struct SimWorker {
    cmd_tx: Sender<SimCommand>,
    inspect_tx: Sender<InspectFn>,
    snapshots: Arc<ArcSwap<Option<PublishedSnapshots>>>,
    view: Arc<ArcSwap<Option<SimView>>>,
    /// Region graph snapshot at spawn time. The graph is
    /// immutable for the session (topology, names, region IDs
    /// don't change after `Sim::load_or_new`), so a single Arc
    /// shared across both threads is cheaper than round-tripping
    /// `id_for_name` / `name_of` lookups through `inspect`.
    /// Hot-path callers (renderer's `snapshot_interp_npcs_near`,
    /// HUD region label) hit this without touching the worker.
    regions: Arc<crate::region::RegionGraph>,
    /// Item registry snapshot at spawn time. Also immutable for
    /// the session (TOML-loaded once at boot). HUD-side
    /// inventory / equipment dict conversions need item
    /// definitions for names + sizes + categories; sharing the
    /// registry via Arc keeps those conversions main-thread-only
    /// rather than round-tripping through inspect.
    items: Arc<crate::items::ItemRegistry>,
    /// Faction registry snapshot at spawn time. Immutable for
    /// the session (TOML-loaded once at boot). NPC/base view
    /// dict conversions need faction names + relations, so we
    /// share via Arc same as the other registries — bridge can
    /// build NPC/base dicts main-thread-side after fetching
    /// the view payload from the worker.
    factions: Arc<crate::faction::registry::FactionRegistry>,
    /// Per-tick `WorldDelta` batches drained from the worker.
    /// Main thread (`SimHost::process`) reads each batch to
    /// drive projectile-FX signal emission. See [`Channels::deltas_tx`].
    deltas_rx: Receiver<Vec<crate::delta::WorldDelta>>,
    shutdown_tx: Sender<()>,
    handle: Option<JoinHandle<()>>,
}

impl SimWorker {
    /// Spawn the worker thread with `sim` as its starting
    /// state. The thread runs until [`shutdown`](Self::shutdown)
    /// is called or the command channel is dropped.
    pub fn spawn(sim: Sim) -> Self {
        let (cmd_tx, cmd_rx) = bounded::<SimCommand>(COMMAND_CHANNEL_CAPACITY);
        let (inspect_tx, inspect_rx) = bounded::<InspectFn>(COMMAND_CHANNEL_CAPACITY);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
        // Bounded so a stalled consumer (e.g. mid-edit reload) can't
        // grow unbounded. 60 ticks = ~3 s of buffered deltas; that's
        // far more than any normal render-side hiccup needs, and on
        // overflow the worker drops the oldest batch (projectile FX
        // is not load-bearing for sim correctness — at worst a
        // tracer doesn't render).
        let (deltas_tx, deltas_rx) = bounded::<Vec<crate::delta::WorldDelta>>(60);
        let snapshots = Arc::new(ArcSwap::new(Arc::new(None)));
        let view = Arc::new(ArcSwap::new(Arc::new(None)));
        let regions = Arc::new(sim.regions().clone());
        let items = Arc::new(sim.item_registry().clone());
        let factions = Arc::new(sim.faction_registry().clone());
        let channels = Channels {
            cmd_rx,
            inspect_rx,
            snapshots: snapshots.clone(),
            view: view.clone(),
            shutdown_rx,
            deltas_tx,
        };
        let handle = thread::Builder::new()
            .name("simn-sim".to_string())
            .spawn(move || {
                if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_sim_loop(sim, channels);
                })) {
                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    eprintln!("[simn-sim WORKER PANIC] {msg}");
                    tracing::error!("sim worker thread panicked: {msg}");
                }
            })
            .expect("simn-sim worker thread failed to spawn");
        Self {
            cmd_tx,
            inspect_tx,
            snapshots,
            view,
            regions,
            items,
            factions,
            deltas_rx,
            shutdown_tx,
            handle: Some(handle),
        }
    }

    /// Drain all buffered `WorldDelta` batches the worker has
    /// produced since the last call. Returns a flat vec — main
    /// thread (`SimHost::process`) calls this once per render
    /// frame and feeds the result through
    /// `emit_projectile_fx_signals` so listeners (impact_fx.gd)
    /// see tracer + impact events.
    pub fn drain_tick_deltas(&self) -> Vec<crate::delta::WorldDelta> {
        let mut out = Vec::new();
        while let Ok(batch) = self.deltas_rx.try_recv() {
            out.extend(batch);
        }
        out
    }

    /// Item registry as cloned at spawn time. Immutable for the
    /// session; safe to share across threads via `Arc`. Used by
    /// the bridge's inventory / equipment / weapons dict
    /// conversions so they don't need a live `&Sim`.
    pub fn item_registry(&self) -> &Arc<crate::items::ItemRegistry> {
        &self.items
    }

    /// Faction registry as cloned at spawn time. Immutable for
    /// the session. Used by the bridge's NPC/base view dict
    /// conversions for faction names + relation lookups.
    pub fn faction_registry(&self) -> &Arc<crate::faction::registry::FactionRegistry> {
        &self.factions
    }

    /// Region graph as cloned at spawn time. Immutable for the
    /// session; safe to share across threads via `Arc`. Hot-path
    /// region-name lookups (renderer's `snapshot_interp_npcs_near`,
    /// HUD region label) call this without touching the worker.
    pub fn regions(&self) -> &Arc<crate::region::RegionGraph> {
        &self.regions
    }

    /// Hot-path render lerp wrapped for worker-mode callers.
    /// Reads the published snapshot pair lock-free and runs
    /// [`crate::snapshot::interp_npcs_near`] on it. Returns an
    /// empty `Vec` if the pair isn't ready yet (< 2 ticks
    /// published).
    ///
    /// Mirror of [`crate::world::Sim::snapshot_interp_npcs_near`]
    /// for the threaded path. Both share the same underlying
    /// math (`snapshot::interp_npcs_near`), so direct-mode and
    /// worker-mode output is identical for the same `(prev,
    /// curr, region, player_pos, max_dist, now)` tuple.
    pub fn snapshot_interp_npcs_near(
        &self,
        region: crate::region::RegionId,
        player_pos: [f32; 3],
        max_dist_m: f32,
        now: std::time::Instant,
    ) -> Vec<crate::snapshot::NpcInterpPose> {
        let guard = self.snapshots.load_full();
        let Some(pair) = guard.as_ref().as_ref() else {
            return Vec::new();
        };
        crate::snapshot::interp_npcs_near(
            &pair.prev, &pair.curr, region, player_pos, max_dist_m, now,
        )
    }

    /// Enqueue a command for the worker to process at the
    /// top of its next tick. Returns `Err` if the channel is
    /// full (~3 ticks backlogged — observable as input lag,
    /// and a real bug, not normal load). Sender drops the
    /// command and `tracing::warn!`s; caller can choose to
    /// retry or surface to the user.
    pub fn send(&self, cmd: SimCommand) -> Result<()> {
        self.cmd_tx
            .try_send(cmd)
            .map_err(|e| anyhow::anyhow!("sim command channel rejected: {e}"))
    }

    /// Run a closure against `&mut Sim` on the worker thread
    /// and block the caller until the result is back. The
    /// universal escape hatch for read paths that aren't yet
    /// covered by [`SimView`] — `SimHost` migrates from
    /// direct `&mut self.sim.foo()` to
    /// `self.worker.inspect(|sim| sim.foo())?`.
    ///
    /// **Latency.** The closure runs between command drain
    /// and tick on the worker, so worst-case wait is one
    /// full tick (~50 ms at 20 Hz) plus the closure cost.
    /// Suitable for cold-path reads (UI panels, inspector
    /// queries, save triggers). Hot-path per-frame reads
    /// should go through `view()` / `snapshots()` instead.
    ///
    /// **Mutation allowed.** The closure receives `&mut Sim`
    /// so it can also be used for migration of mutations
    /// that don't fit the [`SimCommand`] vocabulary yet
    /// (e.g. one-shot setup like installing the LOS provider
    /// at startup). Prefer the typed [`SimCommand`] path
    /// once a variant exists; `inspect` is the migration
    /// escape hatch, not the long-term home.
    pub fn inspect<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Sim) -> R + Send + 'static,
        R: Send + 'static,
    {
        let (reply_tx, reply_rx) = bounded::<R>(1);
        let boxed: InspectFn = Box::new(move |sim| {
            let result = f(sim);
            // Receiver may have been dropped if the caller
            // gave up; that's fine, the result is discarded.
            let _ = reply_tx.send(result);
        });
        self.inspect_tx
            .try_send(boxed)
            .map_err(|e| anyhow::anyhow!("sim inspect channel rejected: {e}"))?;
        reply_rx
            .recv()
            .map_err(|e| anyhow::anyhow!("sim inspect closure dropped without reply: {e}"))
    }

    /// Load the most recent published snapshot pair. `None`
    /// until the worker has completed two ticks. Cheap — one
    /// `Arc` clone, no lock.
    pub fn snapshots(&self) -> Option<Arc<PublishedSnapshots>> {
        let guard = self.snapshots.load_full();
        // ArcSwap stores `Arc<Option<T>>`; clone the inner T
        // into its own Arc for caller convenience. This is a
        // small clone (two SimSnapshots) but happens once per
        // frame at most, on a cold-cache line, so it's fine.
        // Future tuning could `ArcSwap<Option<Arc<...>>>` to
        // skip this; deferred.
        guard.as_ref().as_ref().map(|p| Arc::new(p.clone()))
    }

    /// Load the most recent `SimView`. Same semantics as
    /// [`snapshots`](Self::snapshots) — `None` before the
    /// first tick completes.
    pub fn view(&self) -> Option<Arc<SimView>> {
        let guard = self.view.load_full();
        guard.as_ref().as_ref().map(|v| Arc::new(v.clone()))
    }

    /// Signal shutdown and join the worker. Returns `Ok(())`
    /// once the worker has finished its current tick and
    /// exited cleanly. After this call the worker is gone and
    /// [`send`](Self::send) will fail.
    pub fn shutdown(mut self) -> Result<()> {
        // Drop the shutdown sender; the worker's select arm
        // observes the disconnect on its next channel poll.
        // We can't use a "send shutdown signal" semantic with
        // a 0-capacity channel here because the worker isn't
        // pulling from it inline; instead the worker checks
        // `try_recv` between ticks.
        let _ = self.shutdown_tx.try_send(());
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("sim worker thread panicked during shutdown"))?;
        }
        Ok(())
    }
}

fn run_sim_loop(mut sim: Sim, channels: Channels) {
    let Channels {
        cmd_rx,
        inspect_rx,
        snapshots,
        view,
        shutdown_rx,
        deltas_tx,
    } = channels;
    let mut next_deadline = Instant::now() + TICK_PERIOD;
    let mut prev_snapshot: Option<SimSnapshot> = None;
    loop {
        // 1. Drain commands up to the deadline. Inspects
        //    interleave at the top of each iteration so a
        //    waiting main-thread `inspect()` call doesn't
        //    block the full deadline wait — it observes a
        //    fully-applied command queue (any cmds enqueued
        //    before the inspect on the main thread will have
        //    been drained on previous iterations) but pre-tick
        //    state for the round being served.
        loop {
            // 1a. Run any pending inspect closures first.
            //     Bounded by however many were enqueued — no
            //     budget cap today, on the assumption that
            //     inspect is a cold path; revisit if profile
            //     shows main-thread query bursts starving the
            //     tick.
            while let Ok(inspect) = inspect_rx.try_recv() {
                inspect(&mut sim);
            }
            let now = Instant::now();
            if now >= next_deadline {
                break;
            }
            let timeout = next_deadline - now;
            match cmd_rx.recv_timeout(timeout) {
                Ok(cmd) => {
                    if let Err(e) = dispatch_command(&mut sim, cmd) {
                        tracing::warn!(target: "sim.worker", error = ?e, "command dispatch failed");
                    }
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    // Main thread dropped the sender without
                    // calling shutdown — treat as shutdown.
                    return;
                }
            }
        }
        // 2. Drain any inspects that arrived after the
        //    command loop exited but before the tick. Same
        //    rationale as 1a; serving them here means a
        //    caller blocked on `recv` doesn't wait an extra
        //    full tick when they just missed the iteration.
        while let Ok(inspect) = inspect_rx.try_recv() {
            inspect(&mut sim);
        }
        // 3. Check shutdown signal between command-drain and
        //    tick. Cheap try_recv; usually empty.
        if shutdown_rx.try_recv().is_ok() {
            return;
        }
        // 4. Tick.
        if let Err(e) = sim.tick() {
            tracing::error!(target: "sim.worker", error = ?e, "sim tick failed; worker exiting");
            return;
        }
        // 5. Drain `last_tick_deltas` and forward to the main
        //    thread for projectile-FX signal emission. Main
        //    thread's `SimWorker::drain_tick_deltas` pulls and
        //    feeds them through `SimHost::emit_projectile_fx_signals`
        //    so impact_fx.gd renders tracers + impacts. Empty
        //    batches are skipped to keep the channel quiet.
        //    `try_send` on full drops the oldest by replacing
        //    the channel head — FX is not load-bearing for sim
        //    correctness, a dropped tracer is invisible to
        //    gameplay.
        let deltas = sim.drain_tick_deltas();
        if !deltas.is_empty() && deltas_tx.try_send(deltas).is_err() {
            // Channel full — drop this batch. FX is not load-
            // bearing for sim correctness (a missed tracer is
            // invisible to gameplay). `try_send` doesn't expose a
            // pop-oldest path, so we just lose this one. (TODO:
            // switch to a ring-buffer if drops become observable
            // in practice.)
            tracing::debug!(
                target: "sim.worker",
                "deltas channel full; dropped a tick batch"
            );
        }
        // 6. Publish snapshot pair + view.
        let curr_snapshot = sim
            .current_snapshot()
            .cloned()
            .expect("Sim::tick should have published a snapshot");
        if let Some(prev) = prev_snapshot.take() {
            let pair = PublishedSnapshots {
                prev,
                curr: curr_snapshot.clone(),
            };
            snapshots.store(Arc::new(Some(pair)));
        }
        prev_snapshot = Some(curr_snapshot);
        let v = build_sim_view(&mut sim);
        view.store(Arc::new(Some(v)));
        // 6. Schedule next deadline. Catch-up: if we're behind,
        //    roll the deadline forward until it's in the future.
        //    This caps at 20 Hz on overload (no double-tick).
        next_deadline += TICK_PERIOD;
        let now = Instant::now();
        while next_deadline < now {
            next_deadline += TICK_PERIOD;
            // Don't tick again in this iteration — just realign.
            // Logging skipped ticks for ops visibility.
            tracing::trace!(target: "sim.worker", "tick deadline missed; realigning");
        }
    }
}
