//! On-disk format constants for snapshots and journals.

pub const SNAPSHOT_MAGIC: &[u8; 8] = b"SIMNSAVE";
pub const JOURNAL_MAGIC: &[u8; 8] = b"SIMNJRNL";
pub const FORMAT_VERSION: u32 = 25;
