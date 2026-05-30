//! `SimCommand` — the message vocabulary the main thread sends
//! to the sim worker. Step 2 of the threaded-sim PR C rollout.
//!
//! ## Why an outer enum around `ActionKind`
//!
//! `ActionKind` already covers ~30 client-originated mutations
//! (Move, FireWeapon, ApplyBandage, every inventory op, every
//! crafting op, …). The worker doesn't need a second enum
//! parallel to that — we just wrap it as
//! [`SimCommand::Action { steam_id, kind }`] and dispatch
//! through the existing [`Sim::apply_action`].
//!
//! What `ActionKind` doesn't cover is **non-player mutations**
//! — server-side operations that aren't initiated by a client
//! pressing a button: `upsert_player` (the player joining the
//! sim), `remove_player` (disconnect), `set_active_region`
//! (tier-filter focus shift), region terrain attach, world
//! event push, lifecycle (Load / Shutdown). Those get their
//! own variants here.
//!
//! ## Step 2 scope
//!
//! This commit lands the dispatcher pattern + a minimal set
//! of variants: `Action`, `UpsertPlayer`, `RemovePlayer`,
//! `SetActiveRegion`. Step 4 of the rollout extends the
//! variant list as it rewires call sites in
//! `crates/simn-godot/src/sim/mod.rs` — at each rewire the
//! corresponding `SimCommand` variant gets added here and
//! the dispatcher gains an arm. Keeping the variant set
//! minimal at step 2 means we don't carry dead enum cases
//! while the worker thread itself doesn't exist yet (step 3).
//!
//! ## Same-thread today, worker-thread soon
//!
//! [`dispatch_command`] runs the command on whatever thread
//! holds `&mut Sim`. Today that's the main thread; once step
//! 3 introduces `SimWorker`, the worker thread drains a
//! `crossbeam_channel::Receiver<SimCommand>` and calls this
//! dispatcher for each. The dispatcher itself is unchanged
//! — that's the point of getting it landed now.

use anyhow::Result;

use crate::action::ActionKind;
use crate::region::RegionId;
use crate::world::Sim;

/// Worker-bound command. Step 2 surface; extended in step 4
/// as call sites in `SimHost` get rewired off direct
/// `&mut self.sim` access.
#[derive(Clone, Debug)]
pub enum SimCommand {
    /// Client-originated action. Wraps the existing
    /// [`ActionKind`] vocabulary 1:1; the dispatcher delegates
    /// to [`Sim::apply_action`].
    Action { steam_id: u64, kind: ActionKind },
    /// Spawn or update the player entity for `steam_id`.
    /// Server-side op (e.g. on connect) — not a client action,
    /// so it doesn't go through `ActionKind`.
    UpsertPlayer {
        steam_id: u64,
        region: RegionId,
        pos: [f32; 3],
        yaw: f32,
    },
    /// Remove the player entity. Server-side; on disconnect.
    RemovePlayer { steam_id: u64 },
    /// Set the tier-filter focus region. The sim freezes NPCs
    /// in non-active regions, so this is the gate for which
    /// regions actually tick each frame. Today there's a
    /// single active region per server; eventually
    /// multi-region.
    SetActiveRegion { region: RegionId },
}

/// Run a command on `sim`. Same-thread dispatch today; the
/// future worker thread (step 3) will call this in a loop as
/// it drains its command channel. Errors propagate so the
/// caller (host wrapper or worker loop) can log them.
pub fn dispatch_command(sim: &mut Sim, cmd: SimCommand) -> Result<()> {
    match cmd {
        SimCommand::Action { steam_id, kind } => sim.apply_action(steam_id, kind),
        SimCommand::UpsertPlayer {
            steam_id,
            region,
            pos,
            yaw,
        } => sim.upsert_player(steam_id, region, pos, yaw),
        SimCommand::RemovePlayer { steam_id } => sim.remove_player(steam_id),
        SimCommand::SetActiveRegion { region } => {
            sim.set_active_region(region);
            Ok(())
        }
    }
}
