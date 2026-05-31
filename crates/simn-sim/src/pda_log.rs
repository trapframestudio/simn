//! PDA event log — player-visible feed of offline-tier happenings
//! (Phase 1F of `sim-iteration-5-12-plan.md`).
//!
//! Distinct from `WorldEventQueue`:
//! - The world event bus is an **AI** input channel (squad blackboards
//!   read it on the tick of emission; TTL is 1-4 ticks).
//! - The PDA log is a **player** notification channel. Events linger
//!   for ~60 s of sim time so the client can poll at any frame rate
//!   and reliably catch them, and so a player walking back from a
//!   bathroom break still sees what happened.
//!
//! Server-authoritative. The client polls
//! `Sim::recent_pda_events_since(last_seen_tick)` and renders new
//! entries as toast notifications. Events have a monotonic
//! `seq` field so the client can track "what was the highest seq I've
//! seen" without comparing wall-clock ticks.

use bevy_ecs::prelude::Resource;
use std::collections::VecDeque;

use crate::region::RegionId;

/// Player-visible event. Keep variants narrow — these become user-
/// facing strings, so every addition needs a translation/UX call.
/// String fields carry faction *names* (registry id strings like
/// `"coalition"`) because PDA-log entries persist across registry edits
/// (the same reason `LifeRecord::faction` is a string).
#[derive(Clone, Debug)]
pub enum PdaEvent {
    /// An NPC of `killed_faction` died offline at the hands of an NPC
    /// of `killer_faction`. Player UI: "Lost contact with [Faction]
    /// operator near [Region]."
    OfflineCombatDeath {
        killed_faction: String,
        killer_faction: String,
        region: RegionId,
    },
    /// Offline gunfire detected in `region`. Player UI: "Gunfire
    /// reported in [Region]." Coarsened from per-pair `Gunshot` bus
    /// events to one entry per region per offline tick to avoid
    /// flooding the toast queue.
    OfflineGunfire { region: RegionId },
    /// A base changed ownership. Phase 1F uses a simple "majority of
    /// hostile NPCs within radius after a kill" heuristic; full
    /// contestation lands in Phase 3.
    BaseFlip {
        new_owner: String,
        old_owner: Option<String>,
        region: RegionId,
    },
}

/// One queued PDA event with monotonic sequence + sim-tick stamp.
#[derive(Clone, Debug)]
pub struct PdaLogEntry {
    pub seq: u64,
    pub tick: u64,
    pub event: PdaEvent,
}

/// Bounded ring of recent player-visible events. Old entries drop
/// when either the cap is hit or their age exceeds `MAX_AGE_TICKS`.
#[derive(Resource, Debug)]
pub struct PdaEventLog {
    entries: VecDeque<PdaLogEntry>,
    next_seq: u64,
    cap: usize,
}

/// Cap on retained PDA events. ~60 s of sim time at the stock 20 Hz
/// tick rate plus headroom. Per-tick churn is low (a handful of
/// events at most), so this is plenty.
const PDA_LOG_CAP: usize = 256;

/// Max age before a PDA event is dropped from the log even if cap
/// isn't reached. 1200 sim ticks ≈ 60 s at 20 Hz.
const PDA_LOG_MAX_AGE_TICKS: u64 = 1200;

impl Default for PdaEventLog {
    fn default() -> Self {
        Self {
            entries: VecDeque::with_capacity(PDA_LOG_CAP),
            // Sequence ids start at 1 so callers can use `since(0)`
            // to mean "give me everything since boot" with the
            // exclusive-bookmark semantics (`seq > since_seq`).
            // Without this, the first event gets seq=0 and is
            // excluded from a `since(0)` poll.
            next_seq: 1,
            cap: PDA_LOG_CAP,
        }
    }
}

impl PdaEventLog {
    /// Push a new event. Returns the assigned `seq`. Trims the log
    /// to `cap` entries.
    pub fn push(&mut self, event: PdaEvent, tick: u64) -> u64 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.entries.push_back(PdaLogEntry { seq, tick, event });
        while self.entries.len() > self.cap {
            self.entries.pop_front();
        }
        seq
    }

    /// Drop entries older than `MAX_AGE_TICKS` relative to `now`.
    /// Called from the per-tick maintenance pass; cheap (head-of-
    /// deque check).
    pub fn evict_old(&mut self, now: u64) {
        while let Some(front) = self.entries.front() {
            if now.saturating_sub(front.tick) > PDA_LOG_MAX_AGE_TICKS {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    /// Iterate entries with `seq > since_seq`. Used by client polls.
    pub fn since(&self, since_seq: u64) -> impl Iterator<Item = &PdaLogEntry> {
        self.entries.iter().filter(move |e| e.seq > since_seq)
    }

    /// Iterate every retained entry, oldest first. `DoubleEndedIterator`
    /// so callers can `.rev()` to scan newest-first — the cooldown
    /// check in `offline_combat` does this.
    pub fn all(&self) -> impl DoubleEndedIterator<Item = &PdaLogEntry> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Highest `seq` ever assigned, or 0 if nothing has been pushed.
    /// Clients seed their `last_seen` bookmark from this after a
    /// snapshot load so they don't re-toast events that landed
    /// before the player joined. Combined with the exclusive
    /// `since(seq)` semantics, calling `since(high_water())`
    /// returns no events.
    pub fn high_water(&self) -> u64 {
        // `next_seq` always points at the seq the next push will use.
        // Subtract 1 to get the highest assigned seq, clamped at 0
        // for a fresh log (next_seq starts at 1 — see `Default`).
        self.next_seq.saturating_sub(1)
    }
}
