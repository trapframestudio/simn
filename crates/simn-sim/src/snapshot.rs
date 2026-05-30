//! Render-facing per-tick snapshots of NPC pose state.
//!
//! Threaded-sim PR A: build the data path. The sim still runs on
//! the main thread today, but every `Sim::tick` now publishes a
//! lightweight snapshot of authoritative NPC positions / yaw /
//! tick metadata. PR B will consume the most-recent **pair** of
//! snapshots from GDScript and lerp NPC visuals between them so
//! the renderer can run at any frame rate independent of the sim's
//! fixed 20 Hz cadence. PR C cuts the sim onto a dedicated worker
//! thread; the snapshot model defined here is what bridges the two
//! threads.
//!
//! See [`docs/book/src/planning/threaded-sim-plan.md`](../planning/threaded-sim-plan.md)
//! §4 for the design contract.
//!
//! ## What goes in a snapshot
//!
//! Render-facing only. **Not** the full ECS world — that would
//! defeat the point. Just what the main thread needs to draw and
//! interpolate. Offline-region NPCs are omitted entirely; they're
//! not rendered. Richer queries (full `NpcView` with wounds /
//! body parts / inventory) keep going through the existing
//! `Sim::npcs_in_region` / `npcs_near` API for inspector + label
//! consumers.
//!
//! ## Storage
//!
//! [`Sim`] keeps the most recent **two** snapshots: `prev` and
//! `curr`. Renderer reads the pair, computes
//! `alpha = (now - curr.published_at) / (curr.published_at - prev.published_at)`,
//! clamps to `[0, 1]`, and lerps per-NPC. Holding two slots means
//! a lagging renderer always has *something* to interpolate
//! between. Three or more would let us extrapolate; we explicitly
//! don't extrapolate (see plan doc §4.3 — produces glitches at
//! direction reversals).

use std::time::Instant;

use crate::components::NpcId;
use crate::region::RegionId;

/// One NPC's pose at a sim tick. Render-facing — what the
/// interpolating renderer needs and nothing else. Body parts /
/// wounds / inventory queries stay on the heavy `NpcView` path
/// (`Sim::npcs_near`), called on demand from inspector / label
/// code, not per-frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NpcSnapshot {
    pub id: NpcId,
    pub region: RegionId,
    pub pos: [f32; 3],
    pub yaw: f32,
}

/// Full per-tick snapshot of the active-region NPC state plus
/// metadata the renderer needs to interpolate. Built at the end
/// of [`crate::Sim::tick`] and stashed in a 2-slot ring on `Sim`
/// (`prev`, `curr`). PR B consumes via [`crate::Sim::snapshot_pair`].
///
/// **Active-region only.** Snapshots include NPCs whose
/// `InRegion` matches any region in [`crate::resources::ActiveRegions`].
/// Offline-region NPCs are frozen by the active-region tier
/// filter anyway, so omitting them from the snapshot saves
/// allocation + iteration cost on every tick without changing
/// what the renderer sees.
///
/// **Order is stable** — entries are sorted by `NpcId` so the
/// renderer can do a sorted-merge lookup against the previous
/// snapshot to identify (spawn / despawn / persist) sets without
/// an extra HashMap. Determinism harness already requires same-
/// seed sims to produce same-id NPCs in the same positions, so
/// the sort is data-dependent only.
#[derive(Clone, Debug)]
pub struct SimSnapshot {
    /// Sim tick this snapshot was published at.
    pub tick: u64,
    /// Wall-clock instant the snapshot was published. Used by the
    /// renderer to compute the `alpha` between consecutive
    /// snapshots in real seconds.
    pub published_at: Instant,
    /// All NPCs in any [`crate::resources::ActiveRegions`] region,
    /// sorted by `NpcId`.
    pub npcs: Vec<NpcSnapshot>,
}

impl SimSnapshot {
    /// Empty snapshot at the given tick. Used as the initial
    /// `prev` slot before any real ticks have published.
    pub fn empty(tick: u64) -> Self {
        Self {
            tick,
            published_at: Instant::now(),
            npcs: Vec::new(),
        }
    }

    /// Binary-search lookup by `NpcId`. O(log n). Returns `None`
    /// if the NPC isn't in the snapshot (either offline-region or
    /// despawned since the snapshot was built).
    pub fn find(&self, id: NpcId) -> Option<&NpcSnapshot> {
        match self.npcs.binary_search_by_key(&id.0, |s| s.id.0) {
            Ok(idx) => Some(&self.npcs[idx]),
            Err(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        self.npcs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.npcs.is_empty()
    }
}

/// Render-side interpolation alpha given the two most recent
/// snapshots' publish times and the renderer's current wall
/// clock. Clamped to `[0, 1]` — never extrapolates past the
/// latest snapshot (which produces visual glitches on direction
/// reversal, per plan doc §4.3).
pub fn snapshot_alpha(prev: &SimSnapshot, curr: &SimSnapshot, now: Instant) -> f32 {
    let span = curr
        .published_at
        .saturating_duration_since(prev.published_at);
    let span_s = span.as_secs_f32();
    if span_s <= 0.0 {
        return 1.0;
    }
    let since_prev = now
        .saturating_duration_since(prev.published_at)
        .as_secs_f32();
    (since_prev / span_s).clamp(0.0, 1.0)
}

/// Shortest-angle lerp between two yaw values (radians). Avoids
/// the "spin all the way around" bug when interpolating across
/// the ±π wrap. Standard formula: take `(b - a) mod 2π`,
/// remap to `[-π, π]`, multiply by `t`, add to `a`.
pub fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let diff = ((b - a) % two_pi + two_pi) % two_pi;
    let shortest = if diff > std::f32::consts::PI {
        diff - two_pi
    } else {
        diff
    };
    a + shortest * t
}

/// One NPC's interpolated render pose, computed from a snapshot
/// pair + alpha. Output of [`interp_npcs_near`] for the renderer
/// to consume directly. Mirrors the layout the gdext bridge
/// returns via parallel `PackedArray`s (id / position / yaw),
/// just packed into a single Rust-native row here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NpcInterpPose {
    pub id: NpcId,
    pub pos: [f32; 3],
    pub yaw: f32,
}

/// Compute interpolated poses for active-region NPCs within
/// `max_dist_m` of `player_pos`. Distance gate uses squared XZ
/// math. NPCs present in `curr` but not `prev` (just spawned) are
/// returned at their `curr` pose with no interp — they didn't
/// exist before, so there's nothing to lerp from. NPCs in `prev`
/// but not `curr` (despawned) are omitted from the output (correct
/// — they're gone). Output is in `curr` order (sorted by `NpcId`).
///
/// This is the hot path the renderer calls every frame. Per-NPC
/// cost is one binary-search into `prev.npcs` + one squared-
/// distance compare + one lerp + one yaw lerp. At 50 NPCs in
/// draw range × 144 FPS = 7,200 ops/sec — well under any budget.
pub fn interp_npcs_near(
    prev: &SimSnapshot,
    curr: &SimSnapshot,
    region: RegionId,
    player_pos: [f32; 3],
    max_dist_m: f32,
    now: Instant,
) -> Vec<NpcInterpPose> {
    let alpha = snapshot_alpha(prev, curr, now);
    let max_sq = max_dist_m * max_dist_m;
    let mut out = Vec::with_capacity(64);
    for n in &curr.npcs {
        if n.region != region {
            continue;
        }
        let dx = n.pos[0] - player_pos[0];
        let dz = n.pos[2] - player_pos[2];
        if dx * dx + dz * dz > max_sq {
            continue;
        }
        let (pos, yaw) = match prev.find(n.id) {
            Some(p) => {
                let lerp = |a: f32, b: f32| a + (b - a) * alpha;
                (
                    [
                        lerp(p.pos[0], n.pos[0]),
                        lerp(p.pos[1], n.pos[1]),
                        lerp(p.pos[2], n.pos[2]),
                    ],
                    lerp_angle(p.yaw, n.yaw, alpha),
                )
            }
            None => (n.pos, n.yaw),
        };
        out.push(NpcInterpPose { id: n.id, pos, yaw });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn snap(npc_ids: &[u64], tick: u64) -> SimSnapshot {
        SimSnapshot {
            tick,
            published_at: Instant::now(),
            npcs: npc_ids
                .iter()
                .map(|&id| NpcSnapshot {
                    id: NpcId(id),
                    region: 1,
                    pos: [id as f32, 0.0, 0.0],
                    yaw: 0.0,
                })
                .collect(),
        }
    }

    #[test]
    fn find_returns_matching_npc() {
        let s = snap(&[1, 3, 5, 7], 100);
        assert_eq!(s.find(NpcId(5)).unwrap().id, NpcId(5));
    }

    #[test]
    fn find_returns_none_for_missing_npc() {
        let s = snap(&[1, 3, 5, 7], 100);
        assert!(s.find(NpcId(4)).is_none());
        assert!(s.find(NpcId(8)).is_none());
    }

    #[test]
    fn empty_snapshot_has_no_npcs() {
        let s = SimSnapshot::empty(0);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(s.find(NpcId(1)).is_none());
    }

    #[test]
    fn alpha_at_midpoint_is_half() {
        let prev_at = Instant::now();
        let curr_at = prev_at + Duration::from_millis(50);
        let now = prev_at + Duration::from_millis(25);
        let prev = SimSnapshot {
            tick: 0,
            published_at: prev_at,
            npcs: vec![],
        };
        let curr = SimSnapshot {
            tick: 1,
            published_at: curr_at,
            npcs: vec![],
        };
        let a = snapshot_alpha(&prev, &curr, now);
        assert!((a - 0.5).abs() < 1e-3, "expected ~0.5, got {a}");
    }

    #[test]
    fn alpha_clamps_to_one_past_latest() {
        let prev_at = Instant::now();
        let curr_at = prev_at + Duration::from_millis(50);
        let now = curr_at + Duration::from_millis(200);
        let prev = SimSnapshot {
            tick: 0,
            published_at: prev_at,
            npcs: vec![],
        };
        let curr = SimSnapshot {
            tick: 1,
            published_at: curr_at,
            npcs: vec![],
        };
        assert_eq!(snapshot_alpha(&prev, &curr, now), 1.0);
    }

    #[test]
    fn alpha_clamps_to_zero_before_prev() {
        let prev_at = Instant::now() + Duration::from_millis(100);
        let curr_at = prev_at + Duration::from_millis(50);
        let now = prev_at - Duration::from_millis(10);
        let prev = SimSnapshot {
            tick: 0,
            published_at: prev_at,
            npcs: vec![],
        };
        let curr = SimSnapshot {
            tick: 1,
            published_at: curr_at,
            npcs: vec![],
        };
        assert_eq!(snapshot_alpha(&prev, &curr, now), 0.0);
    }

    #[test]
    fn lerp_angle_takes_shortest_path() {
        let pi = std::f32::consts::PI;
        // Lerp from 0 to π/2 at t=0.5 → π/4.
        let v = lerp_angle(0.0, pi / 2.0, 0.5);
        assert!((v - pi / 4.0).abs() < 1e-5);
        // Lerp from -π+0.1 to π-0.1 at t=0.5 should cross via 0,
        // not go the long way around. Expected: ≈ π (top of circle).
        let v = lerp_angle(-pi + 0.1, pi - 0.1, 0.5);
        // Short path goes from -π+0.1 → through ±π. At t=0.5 we should
        // be at the wrap point (±π).
        let dist_to_pi = (v.abs() - pi).abs();
        assert!(
            dist_to_pi < 0.05,
            "expected near ±π via short path, got {v}"
        );
    }

    fn npc_at(id: u64, pos: [f32; 3], yaw: f32) -> NpcSnapshot {
        NpcSnapshot {
            id: NpcId(id),
            region: 1,
            pos,
            yaw,
        }
    }

    #[test]
    fn interp_lerps_existing_npc_at_midpoint() {
        let prev_at = Instant::now();
        let curr_at = prev_at + Duration::from_millis(50);
        let now = prev_at + Duration::from_millis(25);
        let prev = SimSnapshot {
            tick: 0,
            published_at: prev_at,
            npcs: vec![npc_at(1, [0.0, 0.0, 0.0], 0.0)],
        };
        let curr = SimSnapshot {
            tick: 1,
            published_at: curr_at,
            npcs: vec![npc_at(1, [10.0, 0.0, 0.0], 0.0)],
        };
        let poses = interp_npcs_near(&prev, &curr, 1, [0.0, 0.0, 0.0], 100.0, now);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].id, NpcId(1));
        assert!(
            (poses[0].pos[0] - 5.0).abs() < 1e-3,
            "got {:?}",
            poses[0].pos
        );
    }

    #[test]
    fn interp_distance_filter_drops_far_npcs() {
        let prev_at = Instant::now();
        let curr_at = prev_at + Duration::from_millis(50);
        let now = curr_at;
        let prev = SimSnapshot {
            tick: 0,
            published_at: prev_at,
            npcs: vec![
                npc_at(1, [10.0, 0.0, 0.0], 0.0),
                npc_at(2, [500.0, 0.0, 0.0], 0.0),
            ],
        };
        let curr = SimSnapshot {
            tick: 1,
            published_at: curr_at,
            npcs: vec![
                npc_at(1, [10.0, 0.0, 0.0], 0.0),
                npc_at(2, [500.0, 0.0, 0.0], 0.0),
            ],
        };
        let poses = interp_npcs_near(&prev, &curr, 1, [0.0, 0.0, 0.0], 100.0, now);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].id, NpcId(1));
    }

    #[test]
    fn interp_just_spawned_npc_uses_curr_pose() {
        // NPC 2 appears in curr but not in prev (just spawned).
        // Interp should return its curr pose unchanged — there's
        // nothing to lerp from.
        let prev_at = Instant::now();
        let curr_at = prev_at + Duration::from_millis(50);
        let now = prev_at + Duration::from_millis(25);
        let prev = SimSnapshot {
            tick: 0,
            published_at: prev_at,
            npcs: vec![npc_at(1, [0.0, 0.0, 0.0], 0.0)],
        };
        let curr = SimSnapshot {
            tick: 1,
            published_at: curr_at,
            npcs: vec![
                npc_at(1, [10.0, 0.0, 0.0], 0.0),
                npc_at(2, [20.0, 0.0, 0.0], 0.5),
            ],
        };
        let poses = interp_npcs_near(&prev, &curr, 1, [0.0, 0.0, 0.0], 1000.0, now);
        assert_eq!(poses.len(), 2);
        // NPC 2 uses curr unchanged.
        let n2 = poses.iter().find(|p| p.id == NpcId(2)).unwrap();
        assert_eq!(n2.pos, [20.0, 0.0, 0.0]);
        assert_eq!(n2.yaw, 0.5);
    }

    #[test]
    fn interp_despawned_npc_is_omitted() {
        // NPC 2 was in prev but is not in curr (despawned).
        // Output should only include npcs that still exist.
        let prev_at = Instant::now();
        let curr_at = prev_at + Duration::from_millis(50);
        let now = curr_at;
        let prev = SimSnapshot {
            tick: 0,
            published_at: prev_at,
            npcs: vec![
                npc_at(1, [0.0, 0.0, 0.0], 0.0),
                npc_at(2, [5.0, 0.0, 0.0], 0.0),
            ],
        };
        let curr = SimSnapshot {
            tick: 1,
            published_at: curr_at,
            npcs: vec![npc_at(1, [10.0, 0.0, 0.0], 0.0)],
        };
        let poses = interp_npcs_near(&prev, &curr, 1, [0.0, 0.0, 0.0], 1000.0, now);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].id, NpcId(1));
    }

    #[test]
    fn interp_region_filter_drops_other_regions() {
        // NPCs in regions other than the target should be filtered.
        let prev_at = Instant::now();
        let curr_at = prev_at + Duration::from_millis(50);
        let now = curr_at;
        let mut a = npc_at(1, [0.0, 0.0, 0.0], 0.0);
        a.region = 1;
        let mut b = npc_at(2, [0.0, 0.0, 0.0], 0.0);
        b.region = 2;
        let prev = SimSnapshot {
            tick: 0,
            published_at: prev_at,
            npcs: vec![a, b],
        };
        let curr = SimSnapshot {
            tick: 1,
            published_at: curr_at,
            npcs: vec![a, b],
        };
        let poses = interp_npcs_near(&prev, &curr, 1, [0.0, 0.0, 0.0], 1000.0, now);
        assert_eq!(poses.len(), 1);
        assert_eq!(poses[0].id, NpcId(1));
    }
}
