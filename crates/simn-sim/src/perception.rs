//! NPC perception model — FOV, sight radius, and line-of-sight.
//!
//! The sim owns the canonical config (`PerceptionConfig`) and an
//! abstract `LosProvider` trait. `npc_aggro` consults both before
//! acquiring a target:
//!
//! 1. Distance ≤ `sight_radius_m`
//! 2. Target within forward FOV cone (`fov_deg`) of spotter's yaw
//! 3. `LosProvider::exposure(from, to, region) ≥ exposure_required`
//!
//! The provider returns a 0.0..=1.0 scalar where 1.0 = fully visible,
//! 0.0 = fully blocked. The default provider (`AlwaysVisibleLos`)
//! returns 1.0 — used by tests and the headless `watch` example,
//! where no physics world exists. The gdext crate installs a
//! `GodotLosProvider` on startup that does real multi-sample
//! raycasts against the Godot physics world.
//!
//! ## Exposure sampling
//!
//! A single raycast at the target's root (feet) fails on characters
//! behind short cover. We sample multiple heights on the target
//! (see `PerceptionConfig::sample_heights_m`, defaults to feet /
//! torso / head) and combine them. Each sample contributes its
//! weight to the final exposure. A fully occluded sample
//! contributes 0; a sample blocked only by *concealment* (see
//! below) contributes `concealment_visibility` (default 0.5).
//!
//! ## Concealment scaffolding (bushes, smoke, cloth)
//!
//! Solid occluders (walls, buildings, terrain) fully block a ray.
//! **Concealment** is a separate collision layer for foliage, smoke,
//! tarps — things that reduce visibility without fully blocking it.
//! The sim models concealment as a partial visibility scalar; the
//! Godot provider distinguishes the two layers by collision-mask and
//! applies the scalar when only concealment sat between sampler
//! and target. See `crates/simn-godot/src/los.rs` for the Godot-side
//! layer definitions and the physics query glue.

use std::sync::Arc;

use bevy_ecs::prelude::Resource;

use crate::region::RegionId;

/// Canonical perception tuning. Serialized-free — re-read each tick
/// so edits take effect without a snapshot bump.
#[derive(Resource, Clone, Debug)]
pub struct PerceptionConfig {
    /// Forward field-of-view in degrees (full cone width).
    pub fov_deg: f32,
    /// Maximum sight distance in meters.
    pub sight_radius_m: f32,
    /// Minimum exposure (0.0..=1.0) to treat a target as "seen".
    pub exposure_required: f32,
    /// Whether to consult `LosService` at all. `false` skips the
    /// raycast and treats everything within FOV+distance as visible
    /// (cheap offline-tier fallback).
    pub los_enabled: bool,
    /// Heights (meters above target's ground position) to sample.
    /// Each sample contributes equal weight to exposure. Defaults
    /// to feet / torso / head.
    pub sample_heights_m: Vec<f32>,
    /// Visibility scalar for a sample blocked only by concealment
    /// (bushes, smoke). `1.0` = no effect, `0.0` = treats concealment
    /// as solid. Default `0.5`.
    pub concealment_visibility: f32,
    /// Eye height of the spotter, used as the raycast origin.
    pub eye_height_m: f32,
}

impl Default for PerceptionConfig {
    fn default() -> Self {
        Self {
            fov_deg: 110.0,
            sight_radius_m: 80.0,
            exposure_required: 0.33,
            los_enabled: true,
            sample_heights_m: vec![0.2, 1.0, 1.7],
            concealment_visibility: 0.5,
            eye_height_m: 1.6,
        }
    }
}

/// Per-NPC sight radius scaled by the perception stat. `perception` is
/// `0..=100`; the multiplier is linear in `[0.6, 1.4]` centered on 1.0
/// at perception 50. With the default 80 m base, an unaware NPC at
/// perception 0 sees ~48 m and a sniper at perception 100 sees ~112 m.
/// The range is intentionally tight — too wide and a high-perception
/// NPC starts spotting from impossible distances and tactical play
/// breaks. Re-tunable via the constants here as we get playtest data.
pub fn sight_radius_for_perception(perception: u8, base_radius: f32) -> f32 {
    let mult = SIGHT_MULT_BIAS + SIGHT_MULT_SLOPE * f32::from(perception);
    base_radius * mult
}

const SIGHT_MULT_BIAS: f32 = 0.6;
const SIGHT_MULT_SLOPE: f32 = 0.008;

/// Pluggable line-of-sight oracle.
///
/// Implementations MUST be safe to call from the sim tick thread
/// (in practice: Godot's main thread, since that's where
/// `SimHost::process` runs the schedule). The sim treats the trait
/// as `Send + Sync` so it can live in a Bevy resource — gdext's
/// `Gd<T>` is technically main-thread-only, so the provider must
/// assert safety via `unsafe impl` with a clear invariant comment.
pub trait LosProvider: Send + Sync {
    /// Return exposure from `from` to `to` in `region`, in `0.0..=1.0`.
    /// `from` is already the spotter's eye position (sim adds
    /// `eye_height_m`). `to` is the target's ground position — the
    /// provider is responsible for sampling heights off the config.
    fn exposure(
        &self,
        from: [f32; 3],
        to: [f32; 3],
        region: RegionId,
        config: &PerceptionConfig,
    ) -> f32;

    /// Hint for parallel pair-scans: if `true`, the next
    /// `exposure(..)` call from the *current thread* will return
    /// `1.0` without doing real work or acquiring a contended lock.
    /// Callers (e.g. `npc_aggro`'s parallel cell loop) can skip the
    /// exposure call entirely in that case, dropping ~46k mutex
    /// acquisitions per tick at the rayon-worker boundary.
    ///
    /// Default returns `false` so existing providers stay correct;
    /// override to `true` only when the fast path is genuinely
    /// lock-free **and** behaviorally equivalent to a raycast that
    /// finds no occluder.
    fn is_pass_through_on_current_thread(&self) -> bool {
        false
    }
}

/// Passthrough provider — always fully visible. Used in tests,
/// headless runs, and as a fallback before the gdext provider is
/// installed.
pub struct AlwaysVisibleLos;

impl LosProvider for AlwaysVisibleLos {
    fn exposure(
        &self,
        _from: [f32; 3],
        _to: [f32; 3],
        _region: RegionId,
        _config: &PerceptionConfig,
    ) -> f32 {
        1.0
    }

    fn is_pass_through_on_current_thread(&self) -> bool {
        true
    }
}

/// Resource wrapping the currently installed provider. Swap via
/// `Sim::install_los_provider`.
#[derive(Resource, Clone)]
pub struct LosService {
    pub provider: Arc<dyn LosProvider>,
}

impl Default for LosService {
    fn default() -> Self {
        Self {
            provider: Arc::new(AlwaysVisibleLos),
        }
    }
}

/// Returns true if `target` lies within the forward FOV cone of
/// an actor at `from` facing `yaw` (radians, 0 = +X, rotating toward
/// +Z). Distance is assumed pre-checked.
pub fn in_fov(from: [f32; 3], yaw: f32, target: [f32; 3], fov_deg: f32) -> bool {
    let dx = target[0] - from[0];
    let dz = target[2] - from[2];
    let len_sq = dx * dx + dz * dz;
    // Coincident positions → treat as visible.
    if len_sq <= f32::EPSILON {
        return true;
    }
    let len = len_sq.sqrt();
    let fwd_x = yaw.cos();
    let fwd_z = yaw.sin();
    let dot = (dx * fwd_x + dz * fwd_z) / len;
    let half = (fov_deg * 0.5).to_radians();
    dot >= half.cos()
}
