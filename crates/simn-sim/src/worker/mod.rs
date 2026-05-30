//! Worker-thread support for moving the sim off the renderer
//! thread. See `docs/book/src/planning/threaded-sim-plan.md` §5
//! for the full design.
//!
//! Rollout is staged across 8 PRs; this module is introduced
//! incrementally:
//!
//! - **Step 1**: `SimView` — passive, denormalized read-only
//!   snapshot of `Sim` state. Built at end-of-tick; eventually
//!   published behind an `ArcSwap` so the main thread reads it
//!   without touching `Sim` directly.
//! - **Step 2**: `SimCommand` enum + same-thread
//!   `dispatch_command`. The vocabulary the main thread
//!   sends to the worker.
//! - **Step 3 (this commit)**: `SimWorker` — the dedicated
//!   thread that owns `Sim` and runs the tick loop. Drains
//!   `SimCommand`s, ticks at 20 Hz, publishes the snapshot
//!   pair + `SimView` via `Arc<ArcSwap<...>>` cells. The
//!   bridge still owns `Sim` directly today; step 4 is the
//!   mechanical flip of `SimHost` call sites onto the worker.
//! - **Step 4**: rewire every `SimHost` read to `SimView`.
//! - **Step 5+**: snapshot through `ArcSwap`, delta drain, lifecycle,
//!   cleanup.

pub mod command;
mod runtime;
pub mod view;

pub use command::{dispatch_command, SimCommand};
pub use runtime::{PublishedSnapshots, SimWorker, COMMAND_CHANNEL_CAPACITY, TICK_PERIOD};
pub use view::{build_sim_view, SimView};
