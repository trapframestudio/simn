//! Background thread that owns the periodic snapshot disk write.
//!
//! The sim worker serializes the ECS into a `Vec<u8>` on its own
//! thread (mandatory — `serialize_world` needs `&mut World`), then
//! hands the bytes to this writer over a channel. The writer does
//! the atomic-tmp-then-rename disk write off the worker tick path
//! so the periodic snapshot doesn't stall the 20 Hz schedule.
//!
//! Channel is unbounded — snapshot enqueues are rare (every 30 s of
//! sim time by default) and snapshot bytes are typically <10 MB,
//! so growth pressure is negligible. If the disk falls behind for
//! some reason the queue holds rather than dropping snapshots; the
//! shutdown path waits for the queue to drain.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

use anyhow::Result;

use super::snapshot::write_snapshot_bytes;

/// A single snapshot enqueued for disk write. The bytes are
/// already encoded (magic + version + tick + body + hash) — the
/// writer just does the atomic-disk-replace.
struct Job {
    path: PathBuf,
    bytes: Vec<u8>,
    tick: u64,
}

/// Handle to the background snapshot-writer thread. Spawned once
/// per `Sim` lifetime; dropped on shutdown (joins the thread after
/// draining the queue).
pub struct SnapshotWriter {
    tx: Option<Sender<Job>>,
    handle: Option<JoinHandle<()>>,
}

impl SnapshotWriter {
    /// Spawn the writer thread. Returns immediately; the thread
    /// loops blocking on the channel.
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let handle = thread::Builder::new()
            .name("simn-sim-snapshot-writer".into())
            .spawn(move || writer_loop(rx))
            .expect("spawn snapshot writer thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }

    /// Enqueue a snapshot to write. Returns immediately after
    /// pushing to the channel; the disk I/O happens on the writer
    /// thread. Errors only when the writer has shut down (channel
    /// closed) — at that point the caller is in shutdown anyway
    /// and can drop the bytes.
    pub fn enqueue(&self, path: PathBuf, bytes: Vec<u8>, tick: u64) -> Result<()> {
        let Some(tx) = self.tx.as_ref() else {
            anyhow::bail!("snapshot writer already shut down");
        };
        tx.send(Job { path, bytes, tick })
            .map_err(|e| anyhow::anyhow!("snapshot writer channel closed: {e}"))?;
        Ok(())
    }

    /// Flush any pending writes + join the thread. Called on
    /// graceful `Sim::shutdown`. Drops the sender to close the
    /// channel; writer drains remaining jobs and exits.
    pub fn shutdown(&mut self) {
        // Drop the sender — writer's `recv()` will return `Err`
        // once the queue is empty and exit the loop.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            if let Err(e) = h.join() {
                tracing::warn!("snapshot writer thread panicked: {e:?}");
            }
        }
    }
}

impl Drop for SnapshotWriter {
    fn drop(&mut self) {
        // Same as explicit shutdown — keeps the contract simple
        // for callers who don't manage Sim lifecycle directly.
        self.shutdown();
    }
}

fn writer_loop(rx: Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        if let Err(e) = write_snapshot_bytes(&job.path, &job.bytes) {
            tracing::warn!(
                "snapshot writer: failed to write tick={} to {}: {e:?}",
                job.tick,
                job.path.display(),
            );
        }
    }
}
