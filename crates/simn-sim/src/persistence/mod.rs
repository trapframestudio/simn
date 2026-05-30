//! Persistence: snapshots + journal.
//!
//! See [`snapshot`] and [`journal`] for on-disk formats. Higher-level
//! orchestration lives in [`crate::world::Sim`], which owns both and
//! decides when to rotate.

pub mod format;
pub mod journal;
pub mod snapshot;
pub mod snapshot_writer;

pub use journal::{read_journal, JournalWriter};
pub use snapshot::{
    read_snapshot, write_snapshot, write_snapshot_bytes, write_snapshot_to_vec, SerializedEntity,
    SnapshotBody,
};
pub use snapshot_writer::SnapshotWriter;
