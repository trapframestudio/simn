//! Godot-side line-of-sight provider.
//!
//! Implements `simn_sim::LosProvider` via
//! `PhysicsDirectSpaceState3D::intersect_ray` against the Godot
//! physics world. The sim filters candidates by distance + FOV first;
//! this layer only sees pairs that actually plausibly see each other,
//! so raycast volume stays manageable (low thousands per tick
//! typical).
//!
//! ## Collision layers
//!
//! The provider distinguishes **solid occluders** from **concealment**
//! via collision layers. Callers (scene authors, prop spawners, the
//! player scene, the humanoid dummy scene) must assign collider bits
//! accordingly:
//!
//! | Bit | Layer          | Meaning                                     |
//! |----:|----------------|---------------------------------------------|
//! | 0   | `SOLID`        | Walls, buildings, terrain, rock. Full stop. |
//! | 1   | `CONCEALMENT`  | Bushes, smoke, cloth, tarps. Partial vis.   |
//! | 2   | `NPC_HITBOX`   | All humanoid bodies — player and NPC dummies. Excluded from `LOS_QUERY_MASK` so humanoids never occlude sight lines to other humanoids; included in weapon-fire raycasts via `Layers.WEAPON_HIT_MASK` on the GDScript side. |
//!
//! A ray that hits a solid collider is fully blocked. A ray that
//! hits only concealment colliders is weighted by
//! `PerceptionConfig::concealment_visibility` (default 0.5) and
//! contributes partially to exposure. The final exposure is the
//! average over `sample_heights_m` (feet/torso/head by default).
//!
//! Keep the numeric bit assignment in sync with
//! `godot/scripts/layers.gd` (the `Layers` GDScript class). `.tscn`
//! files set `collision_layer` numerically; the GDScript constants
//! are the authoring source of truth at call sites in code.
//!
//! ## Thread-safety
//!
//! `Gd<T>` is main-thread-only in gdext. `LosProvider` requires
//! `Send + Sync` because it lives in a Bevy resource. In direct
//! mode the sim ticks on Godot's main thread (via
//! `SimHost::process`), so the raycast path is safe. In worker
//! mode (threaded-sim PR C) the sim ticks on the dedicated
//! `simn-sim` thread, and **Godot's physics space is not
//! thread-safe** — calling `intersect_ray` from the worker thread
//! produces `Condition "space->locked" is true` errors flooding
//! the console and silently returning "no hit" anyway.
//!
//! The provider captures the main thread's id at construction and
//! short-circuits to "fully visible" (returns `1.0`) from any
//! non-main-thread call. This means worker-mode sims currently
//! see full visibility — concealment + LOS-based perception are
//! disabled when the threaded path is in use. A future commit
//! will add a main-thread "LOS prefetch" pass that fills the
//! cache for expected NPC↔player pairs each frame, so the worker
//! reads cached values without touching Godot directly.

use std::sync::Mutex;

use godot::classes::{PhysicsDirectSpaceState3D, PhysicsRayQueryParameters3D};
use godot::prelude::*;
use simn_sim::{LosProvider, PerceptionConfig, RegionId};

/// Quantize positions to 2m cells for the LOS cache key. Moving
/// <2m / cache-lifetime keeps the same cached result — saves the
/// raycasts when NPCs are shuffling around each other inside a
/// squad cluster without meaningfully changing LOS.
const CACHE_CELL_M: f32 = 2.0;
/// Cache lifetime in tick-ish units. The provider doesn't see the
/// clock; we use an internal call counter as a monotonic proxy.
const CACHE_LIFETIME: u32 = 10;

/// Bit 0: terrain, walls, buildings, rock — anything opaque.
pub const LAYER_SOLID: u32 = 1 << 0;
/// Bit 1: foliage, smoke, cloth — reduces visibility without blocking.
pub const LAYER_CONCEALMENT: u32 = 1 << 1;
/// Bit 2: humanoid body colliders — player character + NPC dummies.
/// Populated by `godot/scenes/player.tscn` and
/// `godot/scenes/humanoid_dummy.tscn`. Mirrored on the GDScript side
/// as `Layers.NPC_HITBOX` (see `godot/scripts/layers.gd`); GDScript
/// is the only consumer, hence the `#[allow(dead_code)]`. Keep the
/// constant here so the numeric bit assignment lives next to the
/// `LOS_QUERY_MASK` that relies on it being excluded.
#[allow(dead_code)]
pub const LAYER_NPC_HITBOX: u32 = 1 << 2;

/// Mask we raycast against: solid + concealment. Humanoid hitboxes
/// (bit 2) are deliberately excluded so we don't "occlude" ourselves
/// or our target — LOS queries between two humanoids must never be
/// blocked by either humanoid's own body geometry.
pub const LOS_QUERY_MASK: u32 = LAYER_SOLID | LAYER_CONCEALMENT;

type CacheKey = (i32, i32, i32, i32, u32);

struct Inner {
    space: Option<Gd<PhysicsDirectSpaceState3D>>,
    /// (from_cell, to_cell, region) → (exposure, generation)
    cache: std::collections::HashMap<CacheKey, (f32, u32)>,
    /// Monotonic generation counter; bumps each `refresh()`, i.e.
    /// roughly per sim tick.
    generation: u32,
}

/// Godot-backed LOS provider. Holds the active space state behind a
/// mutex for the Send+Sync requirement; `refresh()` is called from
/// `SimHost::process` before the schedule runs so the state pointer
/// stays fresh across scene changes. Also owns a short-lived cache
/// keyed by quantized (from, to, region) so clusters of NPCs don't
/// re-raycast identical geometry every tick.
pub struct GodotLosProvider {
    /// Thread id of the main thread, captured at provider
    /// construction. Stored outside the mutex so
    /// `is_pass_through_on_current_thread` is lock-free — needed
    /// for parallel pair-scans in `npc_aggro` that spread across
    /// rayon worker threads.
    main_thread: std::thread::ThreadId,
    inner: Mutex<Inner>,
}

impl Default for GodotLosProvider {
    fn default() -> Self {
        // Captured at construction, which always happens on the
        // main thread (`SimHost::start` runs on the main thread,
        // and that's the only constructor caller). Compared per
        // `exposure` call so worker-thread sim ticks fall through
        // to the cache-only path instead of trying to raycast
        // against a locked physics space.
        Self {
            main_thread: std::thread::current().id(),
            inner: Mutex::new(Inner {
                space: None,
                cache: std::collections::HashMap::new(),
                generation: 0,
            }),
        }
    }
}

impl GodotLosProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the cached space state and bump the cache generation.
    /// Call from the main thread once per sim tick.
    pub fn refresh(&self, space: Option<Gd<PhysicsDirectSpaceState3D>>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.space = space;
            inner.generation = inner.generation.wrapping_add(1);
            // Periodically drop stale cache entries so the map
            // doesn't grow unbounded during long sessions.
            if inner.generation % 240 == 0 {
                let gen = inner.generation;
                inner
                    .cache
                    .retain(|_, (_, g)| gen.wrapping_sub(*g) < CACHE_LIFETIME);
            }
        }
    }
}

fn quantize(p: [f32; 3]) -> (i32, i32) {
    (
        (p[0] / CACHE_CELL_M).round() as i32,
        (p[2] / CACHE_CELL_M).round() as i32,
    )
}

impl LosProvider for GodotLosProvider {
    fn exposure(
        &self,
        from: [f32; 3],
        to: [f32; 3],
        region: RegionId,
        config: &PerceptionConfig,
    ) -> f32 {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return 1.0,
        };
        // No space state → no physics world loaded (menu, boot, or
        // offline tier region). Fall back to fully visible.
        if inner.space.is_none() {
            return 1.0;
        }
        let heights = &config.sample_heights_m;
        if heights.is_empty() {
            return 1.0;
        }
        let gen = inner.generation;
        let (fx, fz) = quantize(from);
        let (tx, tz) = quantize(to);
        let key = (fx, fz, tx, tz, region);
        if let Some((val, g)) = inner.cache.get(&key) {
            if gen.wrapping_sub(*g) < CACHE_LIFETIME {
                return *val;
            }
        }
        // Worker-thread guard. Godot's `PhysicsDirectSpaceState3D` is
        // main-thread-only; raycasting from the threaded-sim worker
        // produces `Condition "space->locked" is true` errors and
        // silently returns "no hit" anyway. Return fully-visible
        // here so behaviorally NPCs see each other / the player as
        // if there's no cover — a regression, but better than
        // flooding the console with errors. A future commit will
        // add a main-thread LOS prefetch that warms the cache so
        // worker reads can be cache hits instead of misses.
        if std::thread::current().id() != self.main_thread {
            return 1.0;
        }
        let mut space = inner.space.as_ref().unwrap().clone();
        let origin = Vector3::new(from[0], from[1], from[2]);
        let mut total = 0.0f32;
        for &h in heights {
            let sample = Vector3::new(to[0], to[1] + h, to[2]);
            total += sample_exposure(&mut space, origin, sample, config);
        }
        let exposure = total / heights.len() as f32;
        inner.cache.insert(key, (exposure, gen));
        exposure
    }

    /// Worker threads short-circuit to `1.0` without touching the
    /// physics space or the mutex, so any rayon-parallel pair scan
    /// that lands on a worker can skip the `exposure` call entirely.
    fn is_pass_through_on_current_thread(&self) -> bool {
        !self.is_main_thread()
    }
}

// SAFETY: `Gd<PhysicsDirectSpaceState3D>` is main-thread-only in
// gdext, but every call path into this provider runs from the sim
// tick, which in turn runs from `SimHost::process` on Godot's main
// thread. No other thread ever touches `inner`. The `Mutex` is still
// used so Bevy's `Send + Sync` bounds hold at the type level; it
// won't ever actually contend in practice.
impl GodotLosProvider {
    fn is_main_thread(&self) -> bool {
        std::thread::current().id() == self.main_thread
    }
}

unsafe impl Send for GodotLosProvider {}
unsafe impl Sync for GodotLosProvider {}

/// Single-ray exposure. Returns 1.0 if nothing in the way,
/// `concealment_visibility` if the first hit was concealment only,
/// 0.0 if the first hit was solid.
///
/// Simplification for this first pass: we take the *first* hit; if
/// it's concealment, we return the partial weight and don't keep
/// marching. Future iteration: march through concealment to the
/// solid or target, multiplying weights; that lets stacked foliage
/// compose correctly. Out of scope until real concealment props
/// exist in the scenes.
fn sample_exposure(
    space: &mut Gd<PhysicsDirectSpaceState3D>,
    from: Vector3,
    to: Vector3,
    config: &PerceptionConfig,
) -> f32 {
    let mut params = PhysicsRayQueryParameters3D::create(from, to)
        .unwrap_or_else(PhysicsRayQueryParameters3D::new_gd);
    params.set_collision_mask(LOS_QUERY_MASK);
    params.set_collide_with_bodies(true);
    params.set_collide_with_areas(false);
    let hit = space.intersect_ray(&params);
    if hit.is_empty() {
        return 1.0;
    }
    let layer: u32 = hit
        .get("collider")
        .and_then(|v| v.try_to::<Gd<godot::classes::CollisionObject3D>>().ok())
        .map(|c| c.get_collision_layer())
        .unwrap_or(LAYER_SOLID);
    if layer & LAYER_SOLID != 0 {
        0.0
    } else if layer & LAYER_CONCEALMENT != 0 {
        config.concealment_visibility
    } else {
        1.0
    }
}
