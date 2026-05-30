//! Journal file I/O.
//!
//! The journal is an append-only log of [`WorldDelta`] records. Each
//! tick, the sim appends zero or more deltas; reads are linear from
//! the start and stop at the first unreadable record (torn write,
//! truncated tail, crc mismatch).
//!
//! Writes are dispatched to a dedicated background thread so the sim
//! worker tick path never blocks on disk I/O. The producer (sim
//! worker) bincode-encodes the delta + crc on its own thread (cheap)
//! and ships the bytes over an mpsc channel; the writer thread
//! handles `write_all` and the periodic `fsync`. Ordering is
//! preserved by the channel (FIFO). A crash between fsyncs loses at
//! most the unflushed tail; a crash mid-write leaves a torn record
//! that [`read_journal`] skips.
//!
//! `rotate` and `flush_and_sync` are queued like any other op but
//! also wait for an ack from the writer thread, so callers can rely
//! on the disk-side state having settled before they continue.
//!
//! Format:
//! ```text
//! [8]     magic: b"NSPHJRNL"
//! [4]     version: u32 LE
//! [8]     snapshot_tick: u64 LE  (paired snapshot tick; 0 = no snapshot yet)
//! record*
//!
//! record:
//!   [4]   payload_len: u32 LE
//!   [N]   payload: bincode(WorldDelta)
//!   [4]   crc32_ieee(payload): u32 LE
//! ```

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::format::{FORMAT_VERSION, JOURNAL_MAGIC};
use crate::delta::WorldDelta;

const FSYNC_INTERVAL: Duration = Duration::from_secs(1);

/// One message to the writer thread. `Append` is fire-and-forget;
/// `Rotate` and `FlushAndSync` carry an ack channel so the caller
/// can block on completion when ordering matters (graceful shutdown,
/// snapshot rotation).
enum Op {
    /// Pre-encoded record bytes (len-prefix + payload + crc32).
    /// Encoding happens on the producer thread so serialization
    /// errors propagate synchronously and the writer is purely I/O.
    Append { bytes: Vec<u8> },
    /// Truncate the file + rewrite the header bound to the given
    /// snapshot tick.
    Rotate {
        snapshot_tick: u64,
        ack: Sender<Result<()>>,
    },
    /// Flush BufWriter + `sync_all`. Used at graceful shutdown.
    FlushAndSync { ack: Sender<Result<()>> },
}

/// Handle to the background journal-writer thread. Owned by `Sim`.
///
/// Dropping the handle (or explicit shutdown via `flush_and_sync` +
/// drop) closes the channel, lets the writer drain pending ops,
/// fsyncs once more, and joins the thread.
pub struct JournalWriter {
    tx: Option<Sender<Op>>,
    handle: Option<JoinHandle<()>>,
}

impl JournalWriter {
    /// Create or open the journal at `path`, paired with a snapshot
    /// taken at `snapshot_tick`. If the file already exists and its
    /// header already matches the snapshot tick, we append to it.
    /// Otherwise we truncate and rewrite the header so a stale
    /// journal from an older snapshot can't be replayed against a
    /// newer snapshot.
    ///
    /// File open + header write happen synchronously on the calling
    /// thread (so path / permission / disk-full errors propagate
    /// immediately); ongoing appends are then dispatched to the
    /// spawned writer thread.
    pub fn open(path: &Path, snapshot_tick: u64) -> Result<Self> {
        let existing_tick = if path.exists() {
            read_journal_snapshot_tick(path).ok()
        } else {
            None
        };

        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true);
        let needs_header = existing_tick != Some(snapshot_tick);
        if needs_header {
            opts.truncate(true);
        } else {
            opts.append(true);
        }

        let mut file = opts
            .open(path)
            .with_context(|| format!("open journal {}", path.display()))?;

        if needs_header {
            file.write_all(JOURNAL_MAGIC)?;
            file.write_all(&FORMAT_VERSION.to_le_bytes())?;
            file.write_all(&snapshot_tick.to_le_bytes())?;
            file.sync_all().context("fsync journal header")?;
        }

        let path_buf = path.to_path_buf();
        let (tx, rx) = unbounded::<Op>();
        let handle = thread::Builder::new()
            .name("simn-sim-journal-writer".into())
            .spawn(move || writer_loop(rx, BufWriter::new(file), path_buf))
            .context("spawn journal writer thread")?;

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    /// Encode `delta` and ship the bytes to the writer thread.
    /// Returns when the bytes are on the channel — disk I/O happens
    /// on the writer's own thread. Errors here are limited to
    /// bincode encoding + channel-closed (writer dead).
    pub fn append(&mut self, delta: &WorldDelta) -> Result<()> {
        let payload = bincode::serialize(delta).context("serialize delta")?;
        let len: u32 = payload
            .len()
            .try_into()
            .map_err(|_| anyhow!("delta too large (>{} bytes)", u32::MAX))?;
        let crc = crc32fast::hash(&payload);

        let mut bytes = Vec::with_capacity(4 + payload.len() + 4);
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes.extend_from_slice(&crc.to_le_bytes());

        let Some(tx) = self.tx.as_ref() else {
            anyhow::bail!("journal writer already shut down");
        };
        tx.send(Op::Append { bytes })
            .map_err(|e| anyhow!("journal channel closed: {e}"))?;
        Ok(())
    }

    /// Time-based fsync was the old responsibility of `maybe_fsync`
    /// called once per tick. The writer thread now owns that timer
    /// (see [`writer_loop`]), so the producer-side call is a no-op
    /// kept around for API compatibility / call-site clarity.
    pub fn maybe_fsync(&mut self) -> Result<()> {
        Ok(())
    }

    /// Block until the writer has drained all pending appends and
    /// fully fsync'd the file. Called on graceful shutdown.
    pub fn flush_and_sync(&mut self) -> Result<()> {
        let Some(tx) = self.tx.as_ref() else {
            return Ok(());
        };
        let (ack_tx, ack_rx) = bounded::<Result<()>>(1);
        tx.send(Op::FlushAndSync { ack: ack_tx })
            .map_err(|e| anyhow!("journal channel closed: {e}"))?;
        ack_rx
            .recv()
            .map_err(|e| anyhow!("journal flush ack: {e}"))?
    }

    /// Truncate the journal and rewrite a fresh header bound to
    /// `new_snapshot_tick`. Called after a snapshot is written so
    /// the journal only contains deltas since the latest snapshot.
    ///
    /// Blocks for the writer to acknowledge: all queued appends
    /// settle, the truncate happens, the new header is fsync'd —
    /// then we return and the next append lands in the fresh file.
    pub fn rotate(&mut self, new_snapshot_tick: u64) -> Result<()> {
        let Some(tx) = self.tx.as_ref() else {
            anyhow::bail!("journal writer already shut down");
        };
        let (ack_tx, ack_rx) = bounded::<Result<()>>(1);
        tx.send(Op::Rotate {
            snapshot_tick: new_snapshot_tick,
            ack: ack_tx,
        })
        .map_err(|e| anyhow!("journal channel closed: {e}"))?;
        ack_rx
            .recv()
            .map_err(|e| anyhow!("journal rotate ack: {e}"))?
    }

    /// Close the channel and join the writer thread. Pending
    /// appends drain + fsync before the worker exits.
    pub fn shutdown(&mut self) {
        self.tx.take();
        if let Some(h) = self.handle.take() {
            if let Err(e) = h.join() {
                tracing::warn!("journal writer thread panicked: {e:?}");
            }
        }
    }
}

impl Drop for JournalWriter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn writer_loop(rx: Receiver<Op>, mut file: BufWriter<File>, path: PathBuf) {
    let mut last_fsync = Instant::now();
    let mut pending_bytes: usize = 0;

    // `recv_timeout` lets us trigger a time-based fsync even when
    // the producer goes quiet for longer than `FSYNC_INTERVAL` —
    // matches the old per-tick `maybe_fsync` cadence.
    loop {
        match rx.recv_timeout(FSYNC_INTERVAL) {
            Ok(Op::Append { bytes }) => {
                if let Err(e) = file.write_all(&bytes) {
                    tracing::warn!("journal write failed at {}: {e:?}", path.display());
                    continue;
                }
                pending_bytes += bytes.len();
                if pending_bytes > 0 && last_fsync.elapsed() >= FSYNC_INTERVAL {
                    if let Err(e) = file.flush() {
                        tracing::warn!("journal flush failed: {e:?}");
                    } else if let Err(e) = file.get_mut().sync_data() {
                        tracing::warn!("journal fsync failed: {e:?}");
                    }
                    last_fsync = Instant::now();
                    pending_bytes = 0;
                }
            }
            Ok(Op::Rotate { snapshot_tick, ack }) => {
                let res = (|| -> Result<()> {
                    file.flush().context("flush before rotate")?;
                    file.get_mut().sync_all().context("fsync before rotate")?;
                    let new_file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(&path)
                        .with_context(|| format!("truncate journal {}", path.display()))?;
                    let mut nb = BufWriter::new(new_file);
                    nb.write_all(JOURNAL_MAGIC)?;
                    nb.write_all(&FORMAT_VERSION.to_le_bytes())?;
                    nb.write_all(&snapshot_tick.to_le_bytes())?;
                    nb.flush()?;
                    nb.get_mut().sync_all().context("fsync rotated journal")?;
                    file = nb;
                    last_fsync = Instant::now();
                    pending_bytes = 0;
                    Ok(())
                })();
                let _ = ack.send(res);
            }
            Ok(Op::FlushAndSync { ack }) => {
                let res = (|| -> Result<()> {
                    file.flush().context("flush journal")?;
                    file.get_mut()
                        .sync_all()
                        .context("fsync journal on shutdown")?;
                    last_fsync = Instant::now();
                    pending_bytes = 0;
                    Ok(())
                })();
                let _ = ack.send(res);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if pending_bytes > 0 {
                    if let Err(e) = file.flush() {
                        tracing::warn!("journal idle flush: {e:?}");
                    } else if let Err(e) = file.get_mut().sync_data() {
                        tracing::warn!("journal idle fsync: {e:?}");
                    }
                    last_fsync = Instant::now();
                    pending_bytes = 0;
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Channel closed — drain final flush so process exit doesn't
    // leave a buffered tail.
    let _ = file.flush();
    let _ = file.get_mut().sync_all();
}

/// Peek the snapshot_tick field of an existing journal file.
pub fn read_journal_snapshot_tick(path: &Path) -> Result<u64> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != JOURNAL_MAGIC {
        return Err(anyhow!("journal magic mismatch"));
    }
    let mut u32_buf = [0u8; 4];
    f.read_exact(&mut u32_buf)?;
    let version = u32::from_le_bytes(u32_buf);
    if version != FORMAT_VERSION {
        return Err(anyhow!(
            "journal version {version} unsupported (expected {FORMAT_VERSION})"
        ));
    }
    let mut u64_buf = [0u8; 8];
    f.read_exact(&mut u64_buf)?;
    Ok(u64::from_le_bytes(u64_buf))
}

/// Read every intact record from a journal.
///
/// Stops at the first torn / corrupted / short record and returns
/// everything up to that point. A missing file is treated as an empty
/// journal. This is the only load-path error policy: never fail the
/// caller because of a torn tail, since a crash mid-write is expected.
pub fn read_journal(path: &Path) -> Result<(u64, Vec<WorldDelta>)> {
    if !path.exists() {
        return Ok((0, Vec::new()));
    }
    let mut f = File::open(path).with_context(|| format!("open journal {}", path.display()))?;

    let mut magic = [0u8; 8];
    if f.read_exact(&mut magic).is_err() || &magic != JOURNAL_MAGIC {
        tracing::warn!("journal {} has no/bad magic, skipping", path.display());
        return Ok((0, Vec::new()));
    }
    let mut u32_buf = [0u8; 4];
    if f.read_exact(&mut u32_buf).is_err() {
        return Ok((0, Vec::new()));
    }
    let version = u32::from_le_bytes(u32_buf);
    if version != FORMAT_VERSION {
        tracing::warn!("journal {} has version {version}, skipping", path.display());
        return Ok((0, Vec::new()));
    }
    let mut u64_buf = [0u8; 8];
    if f.read_exact(&mut u64_buf).is_err() {
        return Ok((0, Vec::new()));
    }
    let snapshot_tick = u64::from_le_bytes(u64_buf);

    let mut out = Vec::new();
    loop {
        if f.read_exact(&mut u32_buf).is_err() {
            break; // Clean EOF or truncated length prefix: stop.
        }
        let payload_len = u32::from_le_bytes(u32_buf) as usize;
        // Cap payload len to something sane to avoid OOMing on garbage.
        if payload_len > 16 * 1024 * 1024 {
            tracing::warn!(
                "journal {} record claims {} bytes; treating as torn tail",
                path.display(),
                payload_len
            );
            break;
        }
        let mut payload = vec![0u8; payload_len];
        if f.read_exact(&mut payload).is_err() {
            tracing::warn!(
                "journal {} payload short read; treating as torn tail",
                path.display()
            );
            break;
        }
        if f.read_exact(&mut u32_buf).is_err() {
            tracing::warn!(
                "journal {} missing crc after payload; treating as torn tail",
                path.display()
            );
            break;
        }
        let got_crc = u32::from_le_bytes(u32_buf);
        let want_crc = crc32fast::hash(&payload);
        if got_crc != want_crc {
            tracing::warn!(
                "journal {} crc mismatch at record (got {:08x} want {:08x}); treating as torn tail",
                path.display(),
                got_crc,
                want_crc
            );
            break;
        }
        match bincode::deserialize::<WorldDelta>(&payload) {
            Ok(delta) => out.push(delta),
            Err(e) => {
                tracing::warn!(
                    "journal {} bincode decode failed ({e}); skipping record",
                    path.display()
                );
                // Keep reading — a single bad record shouldn't kill the tail,
                // but realistically bincode won't recover from this, so break.
                break;
            }
        }
    }

    Ok((snapshot_tick, out))
}
