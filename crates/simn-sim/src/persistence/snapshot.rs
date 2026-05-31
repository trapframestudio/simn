//! Snapshot file I/O.
//!
//! A snapshot is a full, self-contained dump of the simulation state
//! at a specific tick. Writes are atomic (write to `.tmp`, fsync,
//! rename over) so a crash mid-write leaves the previous snapshot
//! intact.
//!
//! Format:
//! ```text
//! [8]   magic: b"SIMNSAVE"
//! [4]   version: u32 LE
//! [8]   tick: u64 LE
//! [4]   body_len: u32 LE
//! [N]   body: bincode(SnapshotBody)
//! [32]  blake3(body) -- integrity check
//! ```
//!
//! blake3 rather than crc32 because truncation of the snapshot is a
//! bigger deal than truncation of a single journal record.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::format::{FORMAT_VERSION, SNAPSHOT_MAGIC};
use crate::chronicle::LifeChronicle;
use crate::components::{
    ActiveEffects, Actor, Aggression, Base, BodyParts, Contamination, CraftingQueue, DrugTolerance,
    Equipment, Group, Health, InRegion, Inventory, Lifespan, NearCampfire, NearWorkbench, Npc,
    NpcGoal, Pain, PlayerOwned, Position, Rotation, Stamina, SurvivalStats, WorldContainer, Wounds,
};
use crate::region::RegionGraph;
use crate::resources::{
    ContainerIdCounter, EffectIdCounter, JobIdCounter, NpcIdCounter, PopulationTargets,
    RegionControl, SimClock, WeatherState, WorldTime, WoundIdCounter,
};

/// Serialized view of one entity. Only the components we care about
/// persisting are included; "entity identity" is just the set of
/// components (entities get fresh IDs on load).
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SerializedEntity {
    pub player: Option<PlayerOwned>,
    pub actor: Option<Actor>,
    pub position: Option<Position>,
    pub rotation: Option<Rotation>,
    pub in_region: Option<InRegion>,
    #[serde(default)]
    pub health: Option<Health>,
    #[serde(default)]
    pub stamina: Option<Stamina>,
    #[serde(default)]
    pub body_parts: Option<BodyParts>,
    #[serde(default)]
    pub survival: Option<SurvivalStats>,
    #[serde(default)]
    pub wounds: Option<Wounds>,
    #[serde(default)]
    pub pain: Option<Pain>,
    #[serde(default)]
    pub contamination: Option<Contamination>,
    #[serde(default)]
    pub active_effects: Option<ActiveEffects>,
    #[serde(default)]
    pub drug_tolerance: Option<DrugTolerance>,
    #[serde(default)]
    pub inventory: Option<Inventory>,
    #[serde(default)]
    pub equipment: Option<Equipment>,
    #[serde(default)]
    pub near_campfire: Option<NearCampfire>,
    #[serde(default)]
    pub near_workbench: Option<NearWorkbench>,
    #[serde(default)]
    pub crafting_queue: Option<CraftingQueue>,
    #[serde(default)]
    pub world_container: Option<WorldContainer>,
    /// Faction allegiance, persisted as a registry **name string**
    /// (`"coalition"`, `"coalition_vanguard"`) so saves remain valid across registry
    /// edits. `None` for entities without a `Faction` (most
    /// containers, the player). Loaders resolve to a `FactionId`
    /// against the active registry.
    #[serde(default)]
    pub in_faction: Option<String>,
    #[serde(default)]
    pub base: Option<Base>,
    #[serde(default)]
    pub npc: Option<Npc>,
    #[serde(default)]
    pub npc_goal: Option<NpcGoal>,
    #[serde(default)]
    pub lifespan: Option<Lifespan>,
    #[serde(default)]
    pub group: Option<Group>,
    #[serde(default)]
    pub aggression: Option<Aggression>,
    /// Weapons phase 2: in-flight projectile. On authoritative
    /// sims, host restart preserves active shots; on mirror sims,
    /// projectile entities never exist (clients replay impact
    /// deltas as FX only).
    #[serde(default)]
    pub projectile: Option<crate::components::Projectile>,
}

/// What goes into a snapshot body (everything except the integrity
/// footer).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SnapshotBody {
    pub clock: SimClock,
    pub region_graph: RegionGraph,
    pub entities: Vec<SerializedEntity>,
    #[serde(default)]
    pub world_time: WorldTime,
    #[serde(default)]
    pub region_control: RegionControl,
    #[serde(default)]
    pub chronicle: LifeChronicle,
    #[serde(default)]
    pub npc_id_counter: NpcIdCounter,
    #[serde(default)]
    pub wound_id_counter: WoundIdCounter,
    #[serde(default)]
    pub effect_id_counter: EffectIdCounter,
    #[serde(default)]
    pub job_id_counter: JobIdCounter,
    #[serde(default)]
    pub projectile_id_counter: crate::resources::ProjectileIdCounter,
    #[serde(default)]
    pub container_id_counter: ContainerIdCounter,
    #[serde(default)]
    pub population_targets: PopulationTargets,
    #[serde(default)]
    pub weather: WeatherState,
    /// Runtime drift on the faction-vs-faction relation matrix.
    /// Empty for fresh worlds; accumulates as `WorldDelta::FactionRelationShift`
    /// records replay. Persisted so playthrough faction evolution
    /// survives save/load.
    #[serde(default)]
    pub relation_deltas: crate::faction::registry::RelationDeltas,
    /// Per-player faction reputation. Same persistence story as
    /// `relation_deltas`.
    #[serde(default)]
    pub player_reputation: crate::faction::registry::PlayerReputation,
    /// Offline-tier clock (Phase 1B). Persisted so the slow tier
    /// doesn't reset to tick 0 across save/load — matters once
    /// offline systems (Phase 1D/1E) start scheduling work against
    /// this counter. Fresh on snapshots from before this field
    /// existed via `#[serde(default)]`.
    #[serde(default)]
    pub offline_tier_clock: crate::offline_tier::OfflineTierClock,
}

/// Serialize a snapshot to an in-memory byte vector. Same byte
/// stream `write_snapshot` produces (magic + version + tick + body
/// length + body bytes + blake3 hash), just without the file I/O +
/// atomic rename. Used by the determinism harness and the
/// replication path (host serializes once, broadcasts the same
/// bytes to clients + writes them to disk).
pub fn write_snapshot_to_vec(tick: u64, body: &SnapshotBody) -> Result<Vec<u8>> {
    let body_bytes = bincode::serialize(body).context("serialize snapshot body")?;
    let body_hash = blake3::hash(&body_bytes);
    let len: u32 = body_bytes
        .len()
        .try_into()
        .map_err(|_| anyhow!("snapshot body too large (>{} bytes)", u32::MAX))?;
    let mut out = Vec::with_capacity(SNAPSHOT_MAGIC.len() + 4 + 8 + 4 + body_bytes.len() + 32);
    out.extend_from_slice(SNAPSHOT_MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&tick.to_le_bytes());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&body_bytes);
    out.extend_from_slice(body_hash.as_bytes());
    Ok(out)
}

pub fn write_snapshot(path: &Path, tick: u64, body: &SnapshotBody) -> Result<()> {
    let bytes = write_snapshot_to_vec(tick, body)?;
    write_snapshot_bytes(path, &bytes)
}

/// Atomic disk write of pre-serialized snapshot bytes. Split out
/// from [`write_snapshot`] so the background `SnapshotWriter` thread
/// can call this directly with bytes produced on the sim's worker
/// thread — keeps the (slow) sync-disk-write off the tick path.
pub fn write_snapshot_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("save.tmp");
    {
        let mut f =
            File::create(&tmp).with_context(|| format!("open tmp snapshot {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all().context("fsync snapshot tmp")?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

pub fn read_snapshot(path: &Path) -> Result<(u64, SnapshotBody)> {
    let mut f = File::open(path).with_context(|| format!("open snapshot {}", path.display()))?;

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).context("read magic")?;
    if &magic != SNAPSHOT_MAGIC {
        return Err(anyhow!("snapshot magic mismatch"));
    }
    let mut u32_buf = [0u8; 4];
    f.read_exact(&mut u32_buf).context("read version")?;
    let version = u32::from_le_bytes(u32_buf);
    if version != FORMAT_VERSION {
        return Err(anyhow!(
            "snapshot version {version} unsupported (expected {FORMAT_VERSION})"
        ));
    }
    let mut u64_buf = [0u8; 8];
    f.read_exact(&mut u64_buf).context("read tick")?;
    let tick = u64::from_le_bytes(u64_buf);

    f.read_exact(&mut u32_buf).context("read body_len")?;
    let body_len = u32::from_le_bytes(u32_buf) as usize;
    let mut body_bytes = vec![0u8; body_len];
    f.read_exact(&mut body_bytes).context("read body")?;

    let mut hash_bytes = [0u8; 32];
    f.read_exact(&mut hash_bytes).context("read hash")?;
    let actual = blake3::hash(&body_bytes);
    if actual.as_bytes() != &hash_bytes {
        return Err(anyhow!("snapshot hash mismatch (file is corrupted)"));
    }

    // Sanity: file shouldn't have trailing bytes beyond what we read.
    let pos = f.stream_position()?;
    let end = f.seek(SeekFrom::End(0))?;
    if pos != end {
        tracing::warn!(
            "snapshot {} has {} trailing bytes past the hash; ignoring",
            path.display(),
            end - pos
        );
    }

    let body: SnapshotBody = bincode::deserialize(&body_bytes).context("deserialize body")?;
    Ok((tick, body))
}
