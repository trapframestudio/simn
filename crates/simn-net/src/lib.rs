//! Noosphere networking core.
//!
//! Listen-server peer-to-peer session over Steam Networking. Pure Rust,
//! engine-agnostic. The `simn-godot` crate wraps [`NetSession`] in a
//! gdext [`godot::classes::Node`] and translates [`NetEvent`] values
//! into Godot signals.
//!
//! **Slice 1 shape.** Role-based authority: one peer is the `Host`
//! (runs the authoritative sim), the rest are `Client`s (mirror sims
//! that consume host snapshots + deltas). Sim-type payloads
//! (`SnapshotBody`, `Vec<WorldDelta>`, `ActionKind`) are bincoded on
//! the `simn-sim` side and travel as opaque `Vec<u8>` through this
//! crate — the network layer stays ignorant of sim types.
//!
//! **Out of scope for slice 1:** client-side prediction +
//! reconciliation, 12-player cap (staying at 4 until profiling lands),
//! per-region delta subscription, input validation. See
//! `docs/book/src/architecture/networking.md`.

pub mod protocol;
mod session;

pub use protocol::{Msg, Reliability};
pub use session::{NetEvent, NetRole, NetSession, MAX_LOBBY_MEMBERS};
