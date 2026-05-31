//! Godot bridge for [`simn_sim::Sim`].
//!
//! `SimHost` is a `Node` exposed to GDScript. On `start()` it opens
//! (or creates) the save files in the given directory and begins
//! ticking the simulation each frame. The public `#[func]` methods
//! are thin wrappers around the `Sim` API that translate between
//! Godot types (`Vector3`, `GString`) and the sim's plain-Rust types.
//!
//! All calls run on the Godot main thread. Errors from the sim are
//! logged with `godot_error!` and surfaced through the `sim_error`
//! signal; no `#[func]` method panics into Godot.

use godot::classes::{INode, Node, PhysicsDirectSpaceState3D};
use godot::prelude::*;
use simn_sim::{
    action::ActionKind, death_cause_to_str, moon_phase_name, relation_to_str, weather_from_str,
    weather_to_str, BaseView, NpcView, RegionGraph, SavePaths, Sim, SnapshotBody, WorldDelta,
    ALL_WEATHER,
};
use std::path::PathBuf;
use std::sync::Arc;

use crate::los::GodotLosProvider;
use crate::network::NetworkManager;

mod conversions;
use conversions::{
    base_view_to_dict, body_part_from_str, craft_job_to_dict, craftability_to_dict,
    drug_kind_from_str, equipment_to_dict, equipped_weapons_to_dict, fire_weapon_result,
    food_kind_from_str, inventory_to_array, item_category_to_str, npc_view_to_dict,
    projectile_impacted_to_dict, projectile_spawned_to_dict, recipe_to_dict, slot_def_to_dict,
    survival_stat_from_str, to_state_dict, tool_tier_from_str, tool_tier_to_str,
    water_kind_from_str,
};

/// Pull the current 3D world's direct-space-state from the scene
/// tree. Returns `None` while the node isn't in the tree (boot) or
/// when no 3D viewport is active (menu scene).
fn fetch_space_state(node: Gd<Node>) -> Option<Gd<PhysicsDirectSpaceState3D>> {
    let tree = node.get_tree();
    let root = tree.get_root()?;
    let world = root.get_world_3d()?;
    world.get_direct_space_state()
}

/// Threaded-sim PR C step 4b-iii: shared conversion shared
/// between direct-mode (`sim.world_time()`) and worker-mode
/// (`worker.view().world_time`) so both paths produce
/// byte-identical GDScript output. Pulling these out of the
/// `#[func]` bodies lets each branch read its source state and
/// hand off to the same builder.
fn world_time_to_dict(t: &simn_sim::WorldTime) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(&Variant::from("day"), &Variant::from(i64::from(t.day)));
    d.set(
        &Variant::from("seconds_of_day"),
        &Variant::from(t.seconds_of_day),
    );
    d.set(
        &Variant::from("day_length_seconds"),
        &Variant::from(t.day_length_seconds),
    );
    d.set(
        &Variant::from("sun_angle_rad"),
        &Variant::from(t.sun_angle_rad()),
    );
    d.set(&Variant::from("is_daytime"), &Variant::from(t.is_daytime()));
    let phase = t.moon_phase();
    d.set(&Variant::from("moon_phase"), &Variant::from(phase));
    d.set(
        &Variant::from("moon_illumination"),
        &Variant::from(t.moon_illumination()),
    );
    d.set(
        &Variant::from("moon_angle_rad"),
        &Variant::from(t.moon_angle_rad()),
    );
    d.set(
        &Variant::from("moon_phase_name"),
        &Variant::from(GString::from(moon_phase_name(phase))),
    );
    d
}

fn weather_state_to_dict(w: &simn_sim::WeatherState) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("current"),
        &Variant::from(GString::from(weather_to_str(w.current))),
    );
    d.set(
        &Variant::from("next"),
        &Variant::from(GString::from(weather_to_str(w.next))),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("transitions_at_tick"),
        &Variant::from(w.transitions_at_tick as i64),
    );
    d
}

fn chronicle_summary_to_dict(s: &simn_sim::ChronicleSummary) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("total_ever_spawned"),
        &Variant::from(s.total_ever_spawned as i64),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("currently_alive"),
        &Variant::from(s.currently_alive as i64),
    );
    let mut by_faction: Dictionary<Variant, Variant> = Dictionary::new();
    for (faction, stats) in &s.by_faction {
        let mut fac_d: Dictionary<Variant, Variant> = Dictionary::new();
        #[allow(clippy::cast_lossless)]
        fac_d.set(&Variant::from("alive"), &Variant::from(stats.alive as i64));
        #[allow(clippy::cast_lossless)]
        fac_d.set(&Variant::from("dead"), &Variant::from(stats.dead as i64));
        by_faction.set(
            &Variant::from(GString::from(faction.as_str())),
            &Variant::from(fac_d),
        );
    }
    d.set(&Variant::from("by_faction"), &Variant::from(by_faction));
    d
}

#[derive(GodotClass)]
#[class(init, base=Node)]
pub struct SimHost {
    /// Mirror-sim ownership (populated by `start_mirror`).
    /// Authoritative sims live behind `worker` instead; mirrors
    /// keep using direct-mode because they don't tick
    /// autonomously (they apply external snapshots + deltas from
    /// the host). Every read/write `#[func]` checks `worker`
    /// first and falls through to `sim` here for the mirror path.
    sim: Option<Sim>,
    /// Authoritative sim running on the dedicated worker thread.
    /// Populated by `start()`. Owns the `Sim` value for its
    /// lifetime; the main thread reads state via lock-free
    /// `ArcSwap` cells (snapshot pair, `SimView`) + cached
    /// `Arc<RegionGraph>` / `Arc<ItemRegistry>`, and pushes
    /// mutations via typed `SimCommand`s or generic `inspect`
    /// closures.
    worker: Option<simn_sim::worker::SimWorker>,
    los: Option<Arc<GodotLosProvider>>,
    /// Wall-clock seconds of real time accumulated but not yet
    /// consumed by a sim tick. Decoupled from render frame rate:
    /// the sim is designed for 20Hz (`SIM_DT_S`), but `process()`
    /// fires at display refresh rate (often 60–144Hz). Without an
    /// accumulator we'd over-tick 3–7× on a high-refresh display,
    /// which burned a huge slice of frame time for no gameplay gain.
    /// Only consulted in direct mode; the worker owns its own
    /// tick clock.
    accum_s: f64,
    /// Sibling `NetworkManager` node (optional; wired via
    /// [`SimHost::attach_network`]). When present, SimHost reads its
    /// role to decide whether to mutate locally (authoritative) or
    /// forward as a network action (client). Solo / no-network sessions
    /// leave this `None` and always mutate locally.
    network: Option<Gd<NetworkManager>>,
    /// Last tick we emitted `view_updated` for. Compared each frame
    /// against the current view tick (worker mode) or the sim tick
    /// (direct mode) so we fire exactly one signal per logical tick
    /// advance regardless of how many `_process` calls happened in
    /// between. Default `u64::MAX` means "haven't emitted anything
    /// yet" — first observed tick will trip the inequality.
    last_view_tick_emitted: u64,
    /// Optional content-pack root (resolved OS path), set by GDScript
    /// via [`SimHost::set_content_root`] before `start()`. When set,
    /// the sim loads content as a `ContentSource::Overlay` over the
    /// embedded base, so a game supplies its own proprietary content
    /// (factions / names / chatter) while inheriting mechanics from
    /// the embedded pack. Unset → embedded-only (the generic example).
    content_root: Option<String>,
    base: Base<Node>,
}

impl SimHost {
    /// Build the content source from `content_root`: an `Overlay` over
    /// the embedded base when a root is set, else `Embedded`.
    fn content_source(&self) -> simn_sim::ContentSource {
        match &self.content_root {
            Some(p) => simn_sim::ContentSource::Overlay(std::path::PathBuf::from(p)),
            None => simn_sim::ContentSource::Embedded,
        }
    }
}

const SIM_DT_S: f64 = 1.0 / 20.0;
/// Max ticks per `process()` call. Prevents the spiral-of-death if
/// a slow frame (GC, disk stall) leaves us with a fat accumulator.
const MAX_TICKS_PER_FRAME: u32 = 4;

#[godot_api]
impl INode for SimHost {
    fn process(&mut self, delta: f64) {
        if self.sim.is_none() && self.worker.is_none() {
            return;
        }
        // Refresh the LOS provider's cached space-state pointer so
        // raycasts land on the currently-loaded scene (scenes swap
        // when the player changes regions).
        if let Some(los) = self.los.as_ref() {
            los.refresh(fetch_space_state(self.to_gd().upcast()));
        }
        // Direct mode: drive ticks from the render frame via the
        // accumulator, then drain deltas. Worker mode: the worker
        // thread self-drives at 20 Hz; we just pull buffered FX
        // deltas off the worker channel and emit signals so
        // impact_fx.gd renders tracers + impacts.
        if self.worker.is_some() {
            // Pull whatever the worker has produced since the last
            // frame and run the same projectile-FX emit path the
            // direct-mode tick uses.
            let worker_deltas = self
                .worker
                .as_ref()
                .map(|w| w.drain_tick_deltas())
                .unwrap_or_default();
            if !worker_deltas.is_empty() {
                self.emit_projectile_fx_signals(&worker_deltas);
            }
            // Phase 2G: emit `view_updated` whenever the worker
            // publishes a new view. Cheap check (one ArcSwap load +
            // u64 compare) — UIs subscribe here instead of polling.
            self.maybe_emit_view_updated();
            return;
        }
        self.accum_s += delta;
        let mut ticks = 0u32;
        while self.accum_s >= SIM_DT_S && ticks < MAX_TICKS_PER_FRAME {
            if let Some(sim) = self.sim.as_mut() {
                if let Err(e) = sim.tick() {
                    godot_error!("sim tick failed: {e:?}");
                    let msg = GString::from(&format!("{e:?}"));
                    self.base_mut()
                        .emit_signal("sim_error", &[msg.to_variant()]);
                    self.accum_s = 0.0;
                    return;
                }
            }
            self.accum_s -= SIM_DT_S;
            ticks += 1;
        }
        // Drop backlog if we still couldn't catch up — better to run
        // a bit slow than freeze hunting ticks.
        if self.accum_s > SIM_DT_S * MAX_TICKS_PER_FRAME as f64 {
            self.accum_s = 0.0;
        }

        // Drain any deltas accumulated this frame so the authoritative
        // sim's `last_tick_deltas` buffer doesn't grow unbounded across
        // the session. Two cases:
        //
        // - **Host**: serialize the batch and emit `tick_completed` so
        //   the session layer forwards it to connected peers.
        // - **Solo / no network**: just discard — nothing listens, and
        //   if we skipped the drain the buffer would accumulate ~tens
        //   of KB per minute of gameplay (every mutation records a
        //   delta that nobody consumes).
        //
        // Mirror clients don't reach this branch — their mutation path
        // goes through `apply_external_delta`, which doesn't append to
        // the buffer.
        if ticks > 0 {
            let should_broadcast = self.is_host();
            // Collect the drained deltas + the current tick first,
            // then drop the sim borrow so we can emit signals on
            // `self.base_mut()`. Splitting the borrows this way
            // avoids the "two mutable self" compile error.
            let (deltas, current_tick) = if let Some(sim) = self.sim.as_mut() {
                (sim.drain_tick_deltas(), sim.current_tick())
            } else {
                return;
            };
            // Projectile FX signals — solo, host, or mirror. The
            // auth sim's own tick produced these; we fire them to
            // whatever local listener (impact_fx.gd) is hooked up.
            self.emit_projectile_fx_signals(&deltas);
            if should_broadcast && !deltas.is_empty() {
                let Ok(bytes) = bincode::serialize::<Vec<WorldDelta>>(&deltas) else {
                    return;
                };
                #[allow(clippy::cast_possible_wrap)]
                let tick = current_tick as i64;
                let payload = PackedByteArray::from(bytes.as_slice());
                self.base_mut()
                    .emit_signal("tick_completed", &[tick.to_variant(), payload.to_variant()]);
            }
            // Solo / mirror without broadcast: `deltas` drops here.

            // Phase 2G: emit `view_updated` regardless of host /
            // solo / client. Inventory + dev panel listen for this
            // to refresh after equip / drop / consume / etc. — the
            // `tick_completed` signal above only fires when
            // networked, leaving solo sessions with stale UI.
            if current_tick != self.last_view_tick_emitted {
                self.last_view_tick_emitted = current_tick;
                #[allow(clippy::cast_possible_wrap)]
                let t = current_tick as i64;
                self.base_mut()
                    .emit_signal("view_updated", &[t.to_variant()]);
            }
        }
    }
}

impl SimHost {
    /// Single dispatch path for every `#[func]` that maps to an
    /// [`ActionKind`] variant. Folds the three branches every
    /// mutation used to repeat — client / worker / direct —
    /// into one helper so each `#[func]` body collapses to one
    /// line of call-shape conversion + this call.
    ///
    /// Routing:
    /// - **Client** (`is_client() == true`): emits an
    ///   `action_requested` signal carrying the bincoded
    ///   `ActionKind`. `GameSession` forwards to
    ///   `NetworkManager.send_action`. Returns `true` (request
    ///   accepted; result reflects in the next host snapshot /
    ///   delta).
    /// - **Worker mode**: sends `SimCommand::Action { … }`
    ///   onto the worker's command channel. Returns `true` if
    ///   the channel accepted, `false` if it was full (queue
    ///   saturated — observable input lag, real bug).
    /// - **Direct mode**: calls `Sim::apply_action(sid, kind)`
    ///   on the main thread, exactly the same dispatcher the
    ///   worker thread uses. Returns `true` on `Ok`, `false`
    ///   on `Err` (with the error logged).
    ///
    /// Worker-or-direct helper for mutations that DON'T fit
    /// the `ActionKind` vocabulary — debug setters
    /// (damage_player, set_radiation, …), NPC-target
    /// treatments, and host-only ops. Skips the client branch
    /// entirely; these are host-side mutations that never get
    /// dispatched from a non-authoritative peer.
    ///
    /// Worker mode runs the closure via `inspect` (one-tick
    /// worst-case latency, fine for debug clicks and host-only
    /// flow). Direct mode runs it inline. No-op if neither
    /// backend is initialized.
    fn worker_or_direct_mut<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Sim) + Send + 'static,
    {
        if let Some(worker) = self.worker.as_ref() {
            if let Err(e) = worker.inspect(f) {
                godot_error!("worker inspect failed: {e:?}");
            }
            return;
        }
        if let Some(sim) = self.sim.as_mut() {
            f(sim);
        }
    }

    fn resolve_region(&self, name: &str) -> Option<simn_sim::region::RegionId> {
        if let Some(worker) = self.worker.as_ref() {
            worker.regions().id_for_name(name)
        } else {
            self.sim
                .as_ref()
                .and_then(|s| s.regions().id_for_name(name))
        }
    }

    fn resolve_region_faction(
        &self,
        region_name: &str,
        faction_name: &str,
    ) -> Option<(
        simn_sim::region::RegionId,
        Option<simn_sim::faction::registry::FactionId>,
    )> {
        let region_id = self.resolve_region(region_name);
        let Some(region_id) = region_id else {
            godot_error!("unknown region {region_name}");
            return None;
        };
        let faction_id = if faction_name.is_empty() {
            None
        } else {
            let registry = self.get_faction_registry();
            match registry.as_ref().and_then(|r| r.id_of(faction_name)) {
                Some(id) => Some(id),
                None => {
                    godot_warn!("unknown faction {faction_name:?}");
                    return None;
                }
            }
        };
        Some((region_id, faction_id))
    }

    fn get_faction_registry(&self) -> Option<&simn_sim::FactionRegistry> {
        if let Some(worker) = self.worker.as_ref() {
            Some(worker.faction_registry().as_ref())
        } else if let Some(sim) = self.sim.as_ref() {
            Some(sim.faction_registry())
        } else {
            None
        }
    }

    /// Return-value semantic shift in worker mode: `true`
    /// now means "request queued cleanly", versus direct
    /// mode's "operation succeeded". For user-driven
    /// inventory clicks the optimistic semantic is fine; the
    /// HUD reconciles on the next `SimView` (≤ 50 ms).
    /// Mutations that genuinely need synchronous typed
    /// results (`queue_craft` returns a job id,
    /// `fire_weapon` returns a hit dict) use `SimWorker::inspect`
    /// instead of going through this helper.
    fn dispatch_player_action(&mut self, steam_id: i64, kind: ActionKind) -> bool {
        if self.is_client() {
            self.emit_action(steam_id, kind);
            return true;
        }
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        if let Some(worker) = self.worker.as_ref() {
            return match worker.send(simn_sim::worker::SimCommand::Action {
                steam_id: sid,
                kind,
            }) {
                Ok(()) => true,
                Err(e) => {
                    godot_error!("worker action send failed: {e:?}");
                    false
                }
            };
        }
        let Some(sim) = self.sim.as_mut() else {
            return false;
        };
        match sim.apply_action(sid, kind) {
            Ok(()) => true,
            Err(e) => {
                godot_error!("action failed: {e:#}");
                false
            }
        }
    }
}

#[godot_api]
impl SimHost {
    #[signal]
    fn sim_ready();

    #[signal]
    fn sim_error(message: GString);

    /// Fired after each host frame's tick catch-up when the sim
    /// produced journaled deltas. Payload is a bincoded
    /// `Vec<WorldDelta>`; game_session.gd forwards to
    /// `NetworkManager.broadcast_delta(tick, payload)`.
    #[signal]
    fn tick_completed(tick: i64, payload: PackedByteArray);

    /// Fired whenever a new `SimView` is published (worker mode) or
    /// a tick completes (direct mode) — regardless of host / solo /
    /// client role. Unlike `tick_completed` (which only fires on
    /// the host in a lobby because it carries network deltas), this
    /// signal is the UI-refresh hook every panel can listen to and
    /// know the player-visible state may have changed.
    ///
    /// Phase 2G of `sim-iteration-5-12-plan.md`. Inventory + PDA
    /// toast + dev panel subscribe here instead of polling every
    /// frame.
    #[signal]
    fn view_updated(tick: i64);

    /// Fired when a mutating method is called but the local role is
    /// Client — the session layer forwards the action to the host via
    /// `NetworkManager.send_action(steam_id, payload)`. Payload is a
    /// bincoded `ActionKind`.
    #[signal]
    fn action_requested(steam_id: i64, payload: PackedByteArray);

    /// Fired when a snapshot has been applied to a mirror sim. The
    /// session layer can hide its "connecting..." spinner and
    /// instantiate the scene for the local player's region.
    #[signal]
    fn snapshot_applied(tick: i64);

    /// Phase 2 FX hook: a projectile was spawned this tick. Payload
    /// keys: `id` (int), `source_steam_id` (int), `round_id`
    /// (String), `origin` (Vector3), `velocity` (Vector3),
    /// `max_range_m` (float), `spawned_tick` (int). Emitted on
    /// auth + solo sims for local FX listeners (tracer spawn).
    /// Mirror clients receive the same events via the delta
    /// replay path, which re-emits this signal.
    #[signal]
    fn projectile_spawned(payload: Dictionary<Variant, Variant>);

    /// Phase 2 FX hook: a projectile impacted something. Payload
    /// keys: `id` (int), `pos` (Vector3), `npc_id` (int, `0` if
    /// terminal ground/out-of-range), `body_part` (String, `""`
    /// on null-target impacts), `damage_applied` (float),
    /// `penetrated` (bool).
    #[signal]
    fn projectile_impacted(payload: Dictionary<Variant, Variant>);

    /// Initialize the simulation with save files under `save_dir`.
    /// If a snapshot exists, we resume from it; otherwise we spin up
    /// with the default region graph and take an initial snapshot.
    /// Point the sim at a content-pack directory, overlaid on the
    /// embedded base (on-disk files override embedded; missing files
    /// fall back). Accepts a `res://` / `user://` / OS path. Call
    /// BEFORE `start()`. A consuming game uses this to supply its own
    /// factions / names / chatter while inheriting mechanics from the
    /// embedded example pack. Unset → embedded-only.
    #[func]
    fn set_content_root(&mut self, path: GString) {
        self.content_root = Some(crate::util::resolve_path(&path));
    }

    /// Safe to call multiple times; subsequent calls are no-ops.
    #[func]
    fn start(&mut self, save_dir: GString) {
        // Tear down any existing sim/worker first. Without this, a
        // user switching runs (menu → run A → leave → menu → run B)
        // silently kept run A's world because the worker was still
        // alive and the second `start` no-op'd. `wipe_world` /
        // `force_save` already shut down before re-starting, but
        // `_start_authoritative_sim` (the runs-screen path) did not
        // — so the new save_dir was ignored and the player saw the
        // old run. Idempotent: with nothing running, the take()s
        // return None and this is a cheap no-op.
        if let Some(mut sim) = self.sim.take() {
            if let Err(e) = sim.shutdown() {
                godot_error!("sim shutdown failed (during start re-entry): {e:?}");
            }
        }
        if let Some(worker) = self.worker.take() {
            match worker.inspect(|sim| sim.shutdown()) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    godot_error!("sim shutdown failed in worker (during start re-entry): {e:?}")
                }
                Err(e) => {
                    godot_error!("sim worker inspect channel failed (during start re-entry): {e:?}")
                }
            }
            if let Err(e) = worker.shutdown() {
                godot_error!("sim worker thread join failed (during start re-entry): {e:?}");
            }
        }
        let dir = PathBuf::from(save_dir.to_string());
        let save_paths = SavePaths::in_dir(&dir);
        let graph = RegionGraph::default_test_graph();
        match Sim::load_or_new_with_content(save_paths, graph, self.content_source()) {
            Ok(mut sim) => {
                let los = Arc::new(GodotLosProvider::new());
                sim.install_los_provider(los.clone());
                self.los = Some(los);
                // Phase 1A: bulk-seed populations across every region
                // so a fresh sim boots with its full target NPC count
                // and crossing into a new region doesn't trigger the
                // 8-squads/tick spawn flood. Idempotent — a
                // snapshot-loaded sim already has NPCs and the call
                // no-ops.
                sim.initial_bulk_seed_npcs();
                // Authoritative sims always run on the dedicated
                // worker thread now (PR C step 4b-vii). The
                // worker owns its own 20 Hz clock and self-drives;
                // `process()` is a no-op for tick driving in this
                // path. Mirror sims (via `start_mirror`) still
                // use direct-mode because they don't tick
                // autonomously — they apply external snapshots /
                // deltas instead.
                self.worker = Some(simn_sim::worker::SimWorker::spawn(sim));
                godot_print!("[sim] threaded worker started");
                self.base_mut().emit_signal("sim_ready", &[]);
            }
            Err(e) => {
                godot_error!("sim init failed: {e:?}");
                let msg = GString::from(&format!("{e:?}"));
                self.base_mut()
                    .emit_signal("sim_error", &[msg.to_variant()]);
            }
        }
    }

    /// Spawn or idempotently move the player entity for `steam_id` to
    /// the given region (by name) + transform.
    ///
    /// **Role gating:** authoritative sims (solo / host) mutate
    /// directly. Clients treat this as a no-op — the host owns player
    /// entity lifecycle and will have spawned them already (via the
    /// initial snapshot + a host-side spawn triggered by the Steam
    /// `PeerJoined` event). Clients do still want to call this on
    /// region-change scenarios, so we emit a `ChangeRegion` action
    /// so the host moves them.
    #[func]
    fn upsert_local_player(&mut self, steam_id: i64, region_name: GString, pos: Vector3, yaw: f32) {
        if self.is_client() {
            // Forward as a region change; host owns spawn. Position is
            // ignored in slice 1 (clients can't teleport themselves).
            self.emit_action(
                steam_id,
                ActionKind::ChangeRegion {
                    region_name: region_name.to_string(),
                },
            );
            // Also forward a Move so the local avatar's Godot
            // position is reflected in the host's sim.
            self.emit_action(
                steam_id,
                ActionKind::Move {
                    pos: [pos.x, pos.y, pos.z],
                    yaw,
                },
            );
            return;
        }
        let name = region_name.to_string();
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        if let Some(worker) = self.worker.as_ref() {
            // Worker-mode path. Resolve the region via the cached
            // `Arc<RegionGraph>` (no sim round-trip), then send
            // SetActiveRegion + UpsertPlayer commands. Errors
            // surface in the worker's tracing log; the bridge
            // doesn't get a synchronous Result back.
            let Some(region_id) = worker.regions().id_for_name(&name) else {
                godot_error!("unknown region: {name}");
                return;
            };
            if let Err(e) =
                worker.send(simn_sim::worker::SimCommand::SetActiveRegion { region: region_id })
            {
                godot_error!("set_active_region command failed: {e:?}");
            }
            if let Err(e) = worker.send(simn_sim::worker::SimCommand::UpsertPlayer {
                steam_id: sid,
                region: region_id,
                pos: [pos.x, pos.y, pos.z],
                yaw,
            }) {
                godot_error!("upsert_player command failed: {e:?}");
            }
            return;
        }
        let Some(sim) = self.sim.as_mut() else {
            return;
        };
        let Some(region_id) = sim.regions().id_for_name(&name) else {
            godot_error!("unknown region: {name}");
            return;
        };
        sim.set_active_region(region_id);
        if let Err(e) = sim.upsert_player(sid, region_id, [pos.x, pos.y, pos.z], yaw) {
            godot_error!("upsert_local_player failed: {e:?}");
        }
    }

    /// Move an existing player entity to the given transform. No-op
    /// if the player isn't known to the sim yet. On clients this
    /// packages an `ActionKind::Move` instead of mutating — the host
    /// is the source of truth for player positions.
    #[func]
    fn move_local_player(&mut self, steam_id: i64, pos: Vector3, yaw: f32) {
        if self.is_client() {
            self.emit_action(
                steam_id,
                ActionKind::Move {
                    pos: [pos.x, pos.y, pos.z],
                    yaw,
                },
            );
            return;
        }
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        if let Some(worker) = self.worker.as_ref() {
            // Worker-mode path. Move goes via the typed Action
            // command (which already routes to `Sim::move_player`
            // internally). The send is non-blocking — the actual
            // move applies at the top of the worker's next tick.
            // For a 144 Hz renderer driving moves at every frame,
            // the channel absorbs ~7 stale moves per sim tick;
            // the dispatcher only sees the freshest queue entry
            // applied per tick (older moves still get applied
            // but on the same tick, which is idempotent for
            // position).
            if let Err(e) = worker.send(simn_sim::worker::SimCommand::Action {
                steam_id: sid,
                kind: ActionKind::Move {
                    pos: [pos.x, pos.y, pos.z],
                    yaw,
                },
            }) {
                tracing::debug!("move_local_player worker.send: {e:?}");
            }
            return;
        }
        let Some(sim) = self.sim.as_mut() else {
            return;
        };
        if let Err(e) = sim.move_player(sid, [pos.x, pos.y, pos.z], yaw) {
            // Missing-player is common during boot; log at info, not error.
            tracing::debug!("move_local_player: {e:?}");
        }
    }

    /// Iteration 5-13 Phase B2: variant of [`Self::load_region_terrain`]
    /// that also accepts a list of `NavObstacleMarker3D`-derived
    /// AABBs to stamp into the sim's nav grid. Designers place
    /// markers in scenes (group `&"nav_obstacle_markers"`); the
    /// Godot caller walks the group, filters by region, and passes
    /// the array here. Idempotent per region — calling again with
    /// a new obstacle set replaces the previous set.
    ///
    /// Each dictionary in `obstacles` carries:
    /// - `pos: Vector3` — world-space center (Y ignored)
    /// - `extents: Vector3` — XZ half-size (Y ignored)
    /// - `kind: String` — `"block"` or `"walkable"`; unknown strings
    ///   default to `"block"`.
    #[func]
    fn load_region_terrain_with_obstacles(
        &mut self,
        region_name: GString,
        map_id: GString,
        obstacles: Array<Dictionary<Variant, Variant>>,
    ) {
        let obstacle_vec = parse_nav_obstacles(&obstacles);
        self.load_region_terrain_inner(region_name, map_id, obstacle_vec);
    }

    /// Load the canonical heightmap for a region and attach it to the
    /// sim, so bases get Y-snapped to ground and NPCs walk the surface
    /// instead of floating at Y=0.
    ///
    /// `map_id` resolves to `res://assets/terrain/<map_id>/`, where
    /// the loader expects `heightmap.r32` + `terrain.toml`. Idempotent
    /// per region — calling again with a new `map_id` replaces the
    /// previous attachment.
    #[func]
    fn load_region_terrain(&mut self, region_name: GString, map_id: GString) {
        self.load_region_terrain_inner(region_name, map_id, Vec::new());
    }

    /// Iteration 5-14 follow-up: attach a region heightmap built from
    /// **live, in-Godot data** — typically Terrain3D's per-pixel
    /// height grid sampled at runtime — instead of reading the
    /// canonical `.r32` from disk.
    ///
    /// Why this exists: `Terrain3DLoader::bake_into` is lossy on the
    /// canonical → Terrain3D direction (per-region storage drift
    /// produces 10-20 m Y deltas vs the canonical source). For
    /// production maps the workaround is to **Sync to Canonical**
    /// after Bake Now, so the canonical .r32 matches Terrain3D's
    /// data. But that workaround is fragile — anything that
    /// regenerates the canonical (e.g., re-running `bake_map` or
    /// `generate_test_maps`) silently drifts the sim's Y-snap away
    /// from Terrain3D's mesh until the next Sync.
    ///
    /// This bridge cuts the canonical out of the runtime path. The
    /// caller (GDScript-side, `test_map.gd` / `real_map.gd`) walks
    /// Terrain3D once at map-load and ships the live heights here.
    /// The sim builds an `simn_terrain::Heightmap` from those
    /// samples and attaches it to its `TerrainMaps` resource. From
    /// then on, every Y-snap (`TerrainMaps::ground_at`) matches
    /// what Terrain3D renders and what its per-region collision
    /// shape uses.
    ///
    /// `heights` is a flat `PackedFloat32Array` of `width * height`
    /// f32 meters, row-major, NW-up (same layout as `heightmap.r32`).
    /// `vert_min_m` / `vert_max_m` should be the actual observed
    /// range — used downstream for camera bounds + sky shader hints.
    /// `obstacles` is the same Phase B2 `Array<Dictionary>` shape as
    /// `load_region_terrain_with_obstacles`; pass an empty array
    /// when no obstacles need stamping.
    #[allow(clippy::too_many_arguments)]
    #[func]
    fn attach_region_terrain_from_packed_heights(
        &mut self,
        region_name: GString,
        width: u32,
        height: u32,
        spacing_m: f32,
        vert_min_m: f32,
        vert_max_m: f32,
        heights: godot::builtin::PackedFloat32Array,
        obstacles: Array<Dictionary<Variant, Variant>>,
    ) {
        let region_str = region_name.to_string();
        let region_id = if let Some(worker) = self.worker.as_ref() {
            worker.regions().id_for_name(&region_str)
        } else {
            self.sim
                .as_ref()
                .and_then(|s| s.regions().id_for_name(&region_str))
        };
        let Some(region_id) = region_id else {
            godot_error!("attach_region_terrain_from_packed_heights: unknown region {region_str}");
            return;
        };
        let expected = (width as usize) * (height as usize);
        if heights.len() != expected {
            godot_error!(
                "attach_region_terrain_from_packed_heights: heights len {} != width * height ({})",
                heights.len(),
                expected
            );
            return;
        }
        let samples: Vec<f32> = heights.to_vec();
        let metadata = simn_terrain::TerrainMetadata {
            format_version: simn_terrain::metadata::CURRENT_FORMAT_VERSION,
            map_id: region_str.clone(),
            width,
            height,
            spacing_m,
            vert_min_m,
            vert_max_m,
            origin_utm_zone: "10N".into(),
            origin_utm_easting: 0.0,
            origin_utm_northing: 0.0,
            blake3: String::new(),
            features_blake3: String::new(),
            region_size_m: 2048.0,
            playable_extent_x_m: 0.0,
            playable_extent_z_m: 0.0,
            nav_mask_format_version: 0,
            nav_mask_blake3: String::new(),
        };
        let heightmap = match simn_terrain::Heightmap::from_raw(metadata, samples) {
            Ok(h) => h,
            Err(e) => {
                godot_error!("attach_region_terrain_from_packed_heights {region_str}: {e}");
                return;
            }
        };
        let obstacle_vec = parse_nav_obstacles(&obstacles);
        let region_label = region_str.clone();
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) =
                sim.attach_region_terrain_with_obstacles(region_id, heightmap, &obstacle_vec)
            {
                tracing::error!("attach_region_terrain_from_packed_heights {region_label}: {e}");
            } else {
                tracing::info!(
                    "attached live terrain for region {region_label} ({}+{} obstacles)",
                    "Terrain3D",
                    obstacle_vec.len()
                );
            }
        });
    }

    /// Shared implementation for [`Self::load_region_terrain`] and
    /// [`Self::load_region_terrain_with_obstacles`]. The obstacle
    /// vec is empty in the back-compat path and populated in the
    /// Phase-B2 path; on the sim side both go through the same
    /// `attach_region_terrain_with_obstacles` method.
    fn load_region_terrain_inner(
        &mut self,
        region_name: GString,
        map_id: GString,
        obstacles: Vec<simn_sim::nav::NavObstacle>,
    ) {
        let region_str = region_name.to_string();
        let map_id_str = map_id.to_string();
        // Resolve the region ID. Worker mode uses the cached
        // `Arc<RegionGraph>`; direct mode goes through sim.
        let region_id = if let Some(worker) = self.worker.as_ref() {
            worker.regions().id_for_name(&region_str)
        } else {
            self.sim
                .as_ref()
                .and_then(|s| s.regions().id_for_name(&region_str))
        };
        let Some(region_id) = region_id else {
            godot_error!("load_region_terrain: unknown region {region_str}");
            return;
        };
        // Load the heightmap from disk (Godot main-thread IO is
        // fine here — terrain attach is a start-of-session op,
        // not hot path).
        let res_path = format!("res://assets/terrain/{map_id_str}");
        let global =
            godot::classes::ProjectSettings::singleton().globalize_path(&GString::from(&res_path));
        let dir = PathBuf::from(global.to_string());
        let heightmap = match simn_terrain::Heightmap::load(&dir) {
            Ok(h) => h,
            Err(e) => {
                godot_error!("load_region_terrain {map_id_str}: {e}");
                return;
            }
        };
        let region_label = region_str.clone();
        let map_label = map_id_str.clone();
        let obstacle_count = obstacles.len();
        self.worker_or_direct_mut(move |sim| {
            let result =
                sim.attach_region_terrain_with_obstacles(region_id, heightmap, &obstacles);
            match result {
                Err(e) => tracing::error!(
                    "attach_region_terrain {region_label}/{map_label}: {e}"
                ),
                Ok(()) => tracing::info!(
                    "loaded terrain {map_label} for region {region_label} (+{obstacle_count} obstacles)"
                ),
            }
        });
    }

    /// Iteration 5-13 Phase D2: replace `region`'s set of designer-
    /// placed interaction areas. Caller (Godot side, `game_session.gd`)
    /// walks `&"interaction_area_markers"` group on map load, filters
    /// by region, and ships the per-area dicts.
    ///
    /// Each dict carries:
    /// - `id: String` — stable area id (empty → auto-derived from
    ///   `pos`).
    /// - `kind: String` — free-form descriptor (`"rest"`, `"work"`,
    ///   `"socialize"`, …). Unknown kinds are accepted; the sim
    ///   treats them as a generic visit with low utility.
    /// - `pos: Vector3` — world-space center.
    /// - `extents: Vector3` — half-size (X + Z used; Y ignored).
    /// - `faction: String` — restriction; empty means any. Must
    ///   match a faction id from `factions.toml`; otherwise
    ///   silently dropped to "any" with a warn.
    /// - `capacity: int` — max concurrent occupants (≥ 1; clamped).
    /// - `tags: Dictionary` — free-form per-area metadata.
    ///
    /// Idempotent per region — calling again replaces the prior set.
    /// Bad entries (missing required fields, wrong types) are skipped
    /// with a single warn per call.
    #[func]
    fn attach_region_interaction_areas(
        &mut self,
        region_name: GString,
        areas: Array<Dictionary<Variant, Variant>>,
    ) {
        let region_str = region_name.to_string();
        let region_id = if let Some(worker) = self.worker.as_ref() {
            worker.regions().id_for_name(&region_str)
        } else {
            self.sim
                .as_ref()
                .and_then(|s| s.regions().id_for_name(&region_str))
        };
        let Some(region_id) = region_id else {
            godot_error!("attach_region_interaction_areas: unknown region {region_str}");
            return;
        };
        // Resolve faction names → ids up front (we hold &registry,
        // the closure below holds &mut sim — borrow checker won't
        // let us do both simultaneously).
        let registry: &simn_sim::FactionRegistry = if let Some(worker) = self.worker.as_ref() {
            worker.faction_registry().as_ref()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.faction_registry()
        } else {
            godot_error!("attach_region_interaction_areas: sim not started");
            return;
        };
        let parsed = parse_interaction_areas(&areas, registry, &region_str);
        let parsed_count = parsed.len();
        let region_label = region_str.clone();
        self.worker_or_direct_mut(move |sim| {
            match sim.attach_region_interaction_areas(region_id, parsed) {
                Err(e) => tracing::error!("attach_region_interaction_areas {region_label}: {e}"),
                Ok(()) => tracing::info!(
                    "loaded {parsed_count} interaction areas for region {region_label}"
                ),
            }
        });
    }

    /// Iteration 5-14 Phase B: spawn a scene-authored faction base.
    /// Mirrors the `attach_region_interaction_areas` enumeration
    /// pattern — the Godot caller (`base_spawner.gd` in Phase E)
    /// walks `&"poi_markers"` group, filters `BASE_*` kinds, and
    /// dispatches one of these per marker.
    ///
    /// - `region_name`: must resolve via `RegionGraph::id_for_name`.
    /// - `pos`: world-space center; Y is honored if no terrain is
    ///   attached and overridden by terrain Y otherwise (Y-snap).
    /// - `kind`: PascalCase `BaseKind` variant name — `"Checkpoint"`,
    ///   `"Outpost"`, `"Safehouse"`, `"Headquarters"`,
    ///   `"ResearchPost"`, or `"CampSite"`. Unknown strings warn +
    ///   return `false`.
    /// - `faction`: name from `factions.toml`. `"nomads"` for
    ///   neutral camp sites. Unknown strings warn + return `false`.
    ///
    /// Returns `true` on success. The sim's
    /// `register_authored_base` does the actual spawn and stamps the
    /// per-kind nav-obstacle footprint.
    #[func]
    fn register_authored_base(
        &mut self,
        region_name: GString,
        pos: Vector3,
        kind: GString,
        faction: GString,
    ) -> bool {
        let region_str = region_name.to_string();
        let region_id = if let Some(worker) = self.worker.as_ref() {
            worker.regions().id_for_name(&region_str)
        } else {
            self.sim
                .as_ref()
                .and_then(|s| s.regions().id_for_name(&region_str))
        };
        let Some(region_id) = region_id else {
            godot_error!("register_authored_base: unknown region {region_str}");
            return false;
        };
        let kind_str = kind.to_string();
        let kind = match kind_str.as_str() {
            "Checkpoint" => simn_sim::components::BaseKind::Checkpoint,
            "Outpost" => simn_sim::components::BaseKind::Outpost,
            "Safehouse" => simn_sim::components::BaseKind::Safehouse,
            "Headquarters" => simn_sim::components::BaseKind::Headquarters,
            "ResearchPost" => simn_sim::components::BaseKind::ResearchPost,
            "CampSite" => simn_sim::components::BaseKind::CampSite,
            other => {
                godot_warn!(
                    "register_authored_base: unknown BaseKind {other:?}; \
                     skipping. Valid: Checkpoint / Outpost / Safehouse / \
                     Headquarters / ResearchPost / CampSite."
                );
                return false;
            }
        };
        let faction_str = faction.to_string();
        let registry: &simn_sim::FactionRegistry = if let Some(worker) = self.worker.as_ref() {
            worker.faction_registry().as_ref()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.faction_registry()
        } else {
            godot_error!("register_authored_base: sim not started");
            return false;
        };
        let Some(faction_id) = registry.id_of(&faction_str) else {
            godot_warn!(
                "register_authored_base: unknown faction {faction_str:?} \
                 (check `factions.toml`); skipping."
            );
            return false;
        };
        let pos_arr = [pos.x, pos.y, pos.z];
        let region_label = region_str.clone();
        let kind_label = kind_str.clone();
        let faction_label = faction_str.clone();
        // Validation already passed; spawn dispatches through the
        // worker. Any sim-side failure is logged but can't be
        // returned from inside the closure (which must be
        // `FnOnce + Send + 'static`). For test_map usage where the
        // region was just attached + the faction exists, the spawn
        // is essentially infallible.
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.register_authored_base(region_id, pos_arr, kind, faction_id) {
                tracing::error!(
                    "register_authored_base {region_label}/{kind_label}/{faction_label}: {e}"
                );
            }
        });
        true
    }

    // ── Activity points / patrol routes / spawn points ────────────

    #[func]
    #[allow(clippy::too_many_arguments)]
    fn register_activity_point(
        &mut self,
        region_name: GString,
        kind: GString,
        pos: Vector3,
        facing_yaw_deg: f32,
        faction: GString,
        radius_m: f32,
        capacity: i32,
        priority: i32,
        loop_id: GString,
    ) -> bool {
        let (region_id, faction_id) =
            match self.resolve_region_faction(&region_name.to_string(), &faction.to_string()) {
                Some(r) => r,
                None => return false,
            };
        let kind_str = kind.to_string();
        let activity_kind = match kind_str.as_str() {
            "GuardStatic" => simn_sim::resources::ActivityKind::GuardStatic,
            "GuardPerimeter" => simn_sim::resources::ActivityKind::GuardPerimeter,
            "PatrolWaypoint" => simn_sim::resources::ActivityKind::PatrolWaypoint,
            "RestSpot" => simn_sim::resources::ActivityKind::RestSpot,
            "Lookout" => simn_sim::resources::ActivityKind::Lookout,
            "Campfire" => simn_sim::resources::ActivityKind::Campfire,
            "Workbench" => simn_sim::resources::ActivityKind::Workbench,
            "Stash" => simn_sim::resources::ActivityKind::Stash,
            "SniperNest" => simn_sim::resources::ActivityKind::SniperNest,
            "AmbushPoint" => simn_sim::resources::ActivityKind::AmbushPoint,
            other => {
                godot_warn!("register_activity_point: unknown kind {other:?}");
                return false;
            }
        };
        let pos_arr = [pos.x, pos.y, pos.z];
        let lid = if loop_id.to_string().is_empty() {
            None
        } else {
            Some(loop_id.to_string())
        };
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.register_activity_point(
                region_id,
                activity_kind,
                pos_arr,
                facing_yaw_deg,
                faction_id,
                radius_m,
                capacity as u8,
                priority as i8,
                lid,
            ) {
                tracing::error!("register_activity_point: {e}");
            }
        });
        true
    }

    #[func]
    fn register_patrol_route(
        &mut self,
        region_name: GString,
        route_id: GString,
        waypoints: PackedVector3Array,
        faction: GString,
        is_loop: bool,
        priority: i32,
    ) -> bool {
        let region_str = region_name.to_string();
        let region_id = self.resolve_region(&region_str);
        let Some(region_id) = region_id else {
            godot_error!("register_patrol_route: unknown region {region_str}");
            return false;
        };
        let faction_id = if faction.to_string().is_empty() {
            None
        } else {
            let fstr = faction.to_string();
            let registry = self.get_faction_registry();
            let Some(r) = registry.as_ref().and_then(|r| r.id_of(&fstr)) else {
                godot_warn!("register_patrol_route: unknown faction {fstr:?}");
                return false;
            };
            Some(r)
        };
        let rid = route_id.to_string();
        let wps: Vec<[f32; 3]> = waypoints.to_vec().iter().map(|v| [v.x, v.y, v.z]).collect();
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) =
                sim.register_patrol_route(region_id, rid, wps, faction_id, is_loop, priority as i8)
            {
                tracing::error!("register_patrol_route: {e}");
            }
        });
        true
    }

    #[func]
    #[allow(clippy::too_many_arguments)]
    fn register_spawn_point(
        &mut self,
        region_name: GString,
        pos: Vector3,
        faction: GString,
        spawn_rate: f32,
        max_concurrent: i32,
        squad_size_min: i32,
        squad_size_max: i32,
        spread_radius_m: f32,
        loadout_tier: i32,
        initial_delay_ticks: i32,
    ) -> bool {
        let (region_id, faction_id_opt) =
            match self.resolve_region_faction(&region_name.to_string(), &faction.to_string()) {
                Some(r) => r,
                None => return false,
            };
        let Some(faction_id) = faction_id_opt else {
            godot_error!("register_spawn_point: faction is required");
            return false;
        };
        let pos_arr = [pos.x, pos.y, pos.z];
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.register_spawn_point(
                region_id,
                pos_arr,
                faction_id,
                spawn_rate,
                max_concurrent as u8,
                (squad_size_min as u8, squad_size_max as u8),
                spread_radius_m,
                loadout_tier as u8,
                true,
                initial_delay_ticks as u64,
            ) {
                tracing::error!("register_spawn_point: {e}");
            }
        });
        true
    }

    #[func]
    #[allow(clippy::too_many_arguments)]
    fn register_cover_volume(
        &mut self,
        region_name: GString,
        pos: Vector3,
        half_extents: Vector3,
        rotation: Quaternion,
        material_name: GString,
        height: i32,
        thickness_mm: f32,
        destructible: bool,
        health: f32,
    ) -> bool {
        let region_str = region_name.to_string();
        let Some(region_id) = self.resolve_region(&region_str) else {
            godot_error!("register_cover_volume: unknown region {region_str}");
            return false;
        };
        let mat_str = material_name.to_string();
        let material_id = match mat_str.as_str() {
            "Concrete" => simn_sim::cover::CoverMaterialId::Concrete,
            "Brick" => simn_sim::cover::CoverMaterialId::Brick,
            "SteelThick" => simn_sim::cover::CoverMaterialId::SteelThick,
            "SteelThin" => simn_sim::cover::CoverMaterialId::SteelThin,
            "WoodThick" => simn_sim::cover::CoverMaterialId::WoodThick,
            "WoodThin" => simn_sim::cover::CoverMaterialId::WoodThin,
            "Sandbag" => simn_sim::cover::CoverMaterialId::Sandbag,
            "Earth" => simn_sim::cover::CoverMaterialId::Earth,
            "Glass" => simn_sim::cover::CoverMaterialId::Glass,
            "Vegetation" => simn_sim::cover::CoverMaterialId::Vegetation,
            "VehicleBody" => simn_sim::cover::CoverMaterialId::VehicleBody,
            other => {
                godot_warn!("register_cover_volume: unknown material {other:?}");
                return false;
            }
        };
        let cover_height = match height {
            0 => simn_sim::cover::CoverHeight::Low,
            1 => simn_sim::cover::CoverHeight::High,
            _ => simn_sim::cover::CoverHeight::Full,
        };
        let pos_arr = [pos.x, pos.y, pos.z];
        let he_arr = [half_extents.x, half_extents.y, half_extents.z];
        let rot_arr = [rotation.x, rotation.y, rotation.z, rotation.w];
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.register_cover_volume(
                region_id,
                pos_arr,
                he_arr,
                rot_arr,
                material_id,
                cover_height,
                thickness_mm,
                destructible,
                health,
            ) {
                tracing::error!("register_cover_volume: {e}");
            }
        });
        true
    }

    #[func]
    fn clear_activity_points(&mut self, region_name: GString) {
        let region_str = region_name.to_string();
        if let Some(region_id) = self.resolve_region(&region_str) {
            self.worker_or_direct_mut(move |sim| {
                sim.clear_activity_points_for_region(region_id);
            });
        }
    }

    #[func]
    fn clear_spawn_points(&mut self, region_name: GString) {
        let region_str = region_name.to_string();
        if let Some(region_id) = self.resolve_region(&region_str) {
            self.worker_or_direct_mut(move |sim| {
                sim.clear_spawn_points_for_region(region_id);
            });
        }
    }

    #[func]
    fn clear_cover_volumes(&mut self, region_name: GString) {
        let region_str = region_name.to_string();
        if let Some(region_id) = self.resolve_region(&region_str) {
            self.worker_or_direct_mut(move |sim| {
                sim.clear_cover_volumes_for_region(region_id);
            });
        }
    }

    /// Find a navigable path between `from` and `to` in `region_name`'s
    /// nav grid, weighted by `style`. Returns waypoints (start ... end)
    /// as a `PackedVector3Array`; an empty array means no nav data, no
    /// region by that name, or the goal is unreachable. Server-side
    /// pathfinding — the response is the same trajectory every client
    /// would see, since `simn_sim::Sim::path_in_region` is deterministic
    /// given the same heightmap + style.
    ///
    /// `style` is a string: `"road"` / `"road_hugger"` (military
    /// patrols, prefers paved roads strongly), `"mixed"` (default,
    /// mild road preference), or `"bush"` / `"bushwhacker"` (off-road
    /// faction culture, ignores roads). Unknown strings fall back to
    /// `"mixed"`.
    ///
    /// Phase 1 uses heightmap-derived traversability only (slope +
    /// feature-class gating); static obstacles like buildings + fences
    /// land in a phase-2 follow-up. See
    /// `docs/book/src/planning/npc-traversal-plan.md`.
    #[func]
    fn path_in_region(
        &self,
        region_name: GString,
        from: Vector3,
        to: Vector3,
        style: GString,
    ) -> PackedVector3Array {
        let region_str = region_name.to_string();
        let from_arr = [from.x, from.y, from.z];
        let to_arr = [to.x, to.y, to.z];
        let style_enum = simn_sim::travel_style_from_str(&style.to_string());
        let waypoints: Option<Vec<[f32; 3]>> = if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&region_str) else {
                godot_error!("path_in_region: unknown region {region_str}");
                return PackedVector3Array::new();
            };
            worker
                .inspect(move |sim| sim.path_in_region(region_id, from_arr, to_arr, style_enum))
                .ok()
                .flatten()
        } else if let Some(sim) = self.sim.as_ref() {
            let Some(region_id) = sim.regions().id_for_name(&region_str) else {
                godot_error!("path_in_region: unknown region {region_str}");
                return PackedVector3Array::new();
            };
            sim.path_in_region(region_id, from_arr, to_arr, style_enum)
        } else {
            godot_error!("path_in_region: sim not started");
            return PackedVector3Array::new();
        };
        match waypoints {
            Some(points) => points
                .into_iter()
                .map(|[x, y, z]| Vector3::new(x, y, z))
                .collect(),
            None => PackedVector3Array::new(),
        }
    }

    /// Cheap query: is `pos` on a traversable cell in `region_name`'s
    /// nav grid? Returns `false` for unknown regions or regions with
    /// no nav data.
    #[func]
    fn is_traversable(&self, region_name: GString, pos: Vector3) -> bool {
        let region_str = region_name.to_string();
        let pos_arr = [pos.x, pos.y, pos.z];
        if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&region_str) else {
                return false;
            };
            return worker
                .inspect(move |sim| sim.is_traversable(region_id, pos_arr))
                .unwrap_or(false);
        }
        let Some(sim) = self.sim.as_ref() else {
            return false;
        };
        let Some(region_id) = sim.regions().id_for_name(&region_str) else {
            return false;
        };
        sim.is_traversable(region_id, pos_arr)
    }

    /// Editor / debug helper: nav grid dimensions for `region_name`.
    /// Returns `Vector2i(width, height)`; `Vector2i::ZERO` when the
    /// region has no nav data. Used by the editor overlay to size the
    /// debug-viz texture.
    #[func]
    fn nav_grid_dims(&self, region_name: GString) -> Vector2i {
        let region_str = region_name.to_string();
        let dims: Option<(u32, u32)> = if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&region_str) else {
                return Vector2i::ZERO;
            };
            worker
                .inspect(move |sim| sim.nav_grid_dims(region_id))
                .ok()
                .flatten()
        } else if let Some(sim) = self.sim.as_ref() {
            let Some(region_id) = sim.regions().id_for_name(&region_str) else {
                return Vector2i::ZERO;
            };
            sim.nav_grid_dims(region_id)
        } else {
            None
        };
        match dims {
            Some((w, h)) => Vector2i::new(w as i32, h as i32),
            None => Vector2i::ZERO,
        }
    }

    /// Read a cached line-of-sight exposure value for the (observer,
    /// target) NPC pair, populated this tick by aggro perception. Returns
    /// `-1.0` when the pair wasn't evaluated this tick (out of FOV /
    /// perception range / sim not started); otherwise a value in
    /// `0.0..=1.0` where `0.0` is fully blocked and `1.0` is fully
    /// visible. Direction-keyed: `los_exposure(a, b)` and
    /// `los_exposure(b, a)` look up different entries.
    ///
    /// Future cover queries, tactical AI, and combat resolution will
    /// drive their decisions off this primitive without re-raycasting.
    #[func]
    fn los_exposure(&self, observer_npc_id: i64, target_npc_id: i64) -> f32 {
        let observer = simn_sim::components::NpcId(observer_npc_id as u64);
        let target = simn_sim::components::NpcId(target_npc_id as u64);
        if let Some(worker) = self.worker.as_ref() {
            return worker
                .inspect(move |sim| sim.los_exposure(observer, target).unwrap_or(-1.0))
                .unwrap_or(-1.0);
        }
        let Some(sim) = self.sim.as_ref() else {
            return -1.0;
        };
        sim.los_exposure(observer, target).unwrap_or(-1.0)
    }

    /// Editor / debug helper: per-cell traversability snapshot for
    /// `region_name`. Returns a `PackedByteArray` of length
    /// `width * height`; each byte is `1` (traversable) or `0` (not).
    /// Empty when the region has no nav data. NW-origin, row-major.
    /// Allocates — call sparingly (once per debug rebuild, not per
    /// frame).
    #[func]
    fn nav_traversability(&self, region_name: GString) -> PackedByteArray {
        let region_str = region_name.to_string();
        let cells: Vec<bool> = if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&region_str) else {
                return PackedByteArray::new();
            };
            match worker.inspect(move |sim| sim.nav_traversability(region_id)) {
                Ok(Some(c)) => c,
                _ => return PackedByteArray::new(),
            }
        } else if let Some(sim) = self.sim.as_ref() {
            let Some(region_id) = sim.regions().id_for_name(&region_str) else {
                return PackedByteArray::new();
            };
            match sim.nav_traversability(region_id) {
                Some(c) => c,
                None => return PackedByteArray::new(),
            }
        } else {
            return PackedByteArray::new();
        };
        cells.into_iter().map(u8::from).collect()
    }

    /// Change the player's current region by name.
    #[func]
    fn change_region(&mut self, steam_id: i64, region_name: GString) {
        if self.is_client() {
            self.emit_action(
                steam_id,
                ActionKind::ChangeRegion {
                    region_name: region_name.to_string(),
                },
            );
            return;
        }
        let name = region_name.to_string();
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&name) else {
                godot_error!("unknown region: {name}");
                return;
            };
            if let Err(e) =
                worker.send(simn_sim::worker::SimCommand::SetActiveRegion { region: region_id })
            {
                godot_error!("set_active_region command failed: {e:?}");
            }
            if let Err(e) = worker.send(simn_sim::worker::SimCommand::Action {
                steam_id: sid,
                kind: ActionKind::ChangeRegion { region_name: name },
            }) {
                godot_error!("change_region command failed: {e:?}");
            }
            return;
        }
        let Some(sim) = self.sim.as_mut() else {
            return;
        };
        let Some(region_id) = sim.regions().id_for_name(&name) else {
            godot_error!("unknown region: {name}");
            return;
        };
        sim.set_active_region(region_id);
        if let Err(e) = sim.change_player_region(sid, region_id) {
            godot_error!("change_region failed: {e:?}");
        }
    }

    /// Remove a player entity (disconnect). On client, no-op — the
    /// host observes the Steam lobby exit and despawns. On host /
    /// solo, runs normally.
    #[func]
    fn remove_player(&mut self, steam_id: i64) {
        if self.is_client() {
            return;
        }
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        if let Some(worker) = self.worker.as_ref() {
            if let Err(e) =
                worker.send(simn_sim::worker::SimCommand::RemovePlayer { steam_id: sid })
            {
                godot_error!("remove_player command failed: {e:?}");
            }
            return;
        }
        let Some(sim) = self.sim.as_mut() else {
            return;
        };
        let _ = sim.remove_player(sid);
    }

    /// Look up a player's current state. Returns an empty dictionary
    /// when the player is unknown.
    ///
    /// Worker-mode (PR C step 4b-iv): reads `PlayerView` from the
    /// published `SimView` (the vitals — HP / stamina / wounds /
    /// hunger / pain / effects / drug tolerance) so the HUD's
    /// health-bar / status panel survive the threading switch.
    /// Inventory, equipment, crafting-queue, near-station, and
    /// weapon-magazine details aren't on the view yet — those
    /// migrate in 4b-v alongside the SimView expansion. In worker
    /// mode the inventory-shaped keys come back empty / zero so
    /// the HUD's grid renders as 0×0 until then.
    #[func]
    fn player_state(&mut self, steam_id: i64) -> Dictionary<Variant, Variant> {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        if let Some(worker) = self.worker.as_ref() {
            let Some(view_arc) = worker.view() else {
                return Dictionary::new();
            };
            let Some(player_view) = view_arc.players.get(&sid) else {
                return Dictionary::new();
            };
            let mut d = to_state_dict(worker.regions(), player_view);
            // Pull the rest of the player_state payload from the
            // SimView's `player_extras` map (built end-of-tick on
            // the worker). Inventory / equipment / weapon dict
            // conversions take the cached `Arc<ItemRegistry>` so
            // they don't need a live `&Sim`. Missing extras (a
            // player joined this tick but the view rebuild
            // hasn't caught up) falls back to default empties.
            let extras_owned;
            let extras = match view_arc.player_extras.get(&sid) {
                Some(e) => e,
                None => {
                    extras_owned = simn_sim::worker::view::PlayerExtras::default();
                    &extras_owned
                }
            };
            let items_ref: &simn_sim::ItemRegistry = worker.item_registry();
            d.set(
                &Variant::from("inventory"),
                &Variant::from(inventory_to_array(items_ref, &extras.inventory)),
            );
            #[allow(clippy::cast_possible_wrap)]
            d.set(
                &Variant::from("inventory_width"),
                &Variant::from(extras.inventory.width as i64),
            );
            #[allow(clippy::cast_possible_wrap)]
            d.set(
                &Variant::from("inventory_height"),
                &Variant::from(extras.inventory.height as i64),
            );
            d.set(
                &Variant::from("inventory_weight"),
                &Variant::from(extras.weight),
            );
            d.set(
                &Variant::from("near_campfire"),
                &Variant::from(extras.near_campfire),
            );
            let near_wb = extras.near_workbench.map(tool_tier_to_str).unwrap_or("");
            d.set(
                &Variant::from("near_workbench"),
                &Variant::from(GString::from(near_wb)),
            );
            let mut queue_arr: Array<Variant> = Array::new();
            for job in &extras.crafting_queue {
                queue_arr.push(&Variant::from(craft_job_to_dict(job)));
            }
            d.set(&Variant::from("crafting_queue"), &Variant::from(queue_arr));
            d.set(
                &Variant::from("equipment"),
                &Variant::from(equipment_to_dict(items_ref, &extras.equipment)),
            );
            d.set(
                &Variant::from("equipped_weapons"),
                &Variant::from(equipped_weapons_to_dict(items_ref, &extras.equipment)),
            );
            return d;
        }
        let Some(sim) = self.sim.as_mut() else {
            return Dictionary::new();
        };
        let Some(view) = sim.player_view(sid) else {
            return Dictionary::new();
        };
        let mut d = to_state_dict(sim.regions(), &view);
        // Inventory section + near-campfire flag aren't on `PlayerView`;
        // fetch directly from sim and splice onto the state dict so
        // GDScript sees one bundle per call.
        let inv_grid = sim.inventory_view_grid(sid);
        let weight = sim.inventory_weight(sid);
        let near = sim.near_campfire(sid);
        d.set(
            &Variant::from("inventory"),
            &Variant::from(inventory_to_array(sim.item_registry(), &inv_grid)),
        );
        #[allow(clippy::cast_possible_wrap)]
        d.set(
            &Variant::from("inventory_width"),
            &Variant::from(inv_grid.width as i64),
        );
        #[allow(clippy::cast_possible_wrap)]
        d.set(
            &Variant::from("inventory_height"),
            &Variant::from(inv_grid.height as i64),
        );
        d.set(&Variant::from("inventory_weight"), &Variant::from(weight));
        d.set(&Variant::from("near_campfire"), &Variant::from(near));
        // Step 5 fields. `near_workbench` is "" / "basic" / "advanced" /
        // "expert" so GDScript can branch on a string instead of
        // matching a Variant nil. The crafting queue is the live job
        // list — empty array when the player isn't crafting.
        let near_wb = sim.near_workbench(sid).map(tool_tier_to_str).unwrap_or("");
        d.set(
            &Variant::from("near_workbench"),
            &Variant::from(GString::from(near_wb)),
        );
        let queue = sim.crafting_queue(sid);
        let mut queue_arr: Array<Variant> = Array::new();
        for job in &queue {
            queue_arr.push(&Variant::from(craft_job_to_dict(job)));
        }
        d.set(&Variant::from("crafting_queue"), &Variant::from(queue_arr));
        // Equipment loadout (PR-3). Keyed by slot id; values are
        // equipped-item dicts with an optional nested inner_grid for
        // containers (rigs / backpacks). Empty dict when nothing is
        // equipped.
        let eq = sim.equipment_view(sid);
        d.set(
            &Variant::from("equipment"),
            &Variant::from(equipment_to_dict(sim.item_registry(), &eq)),
        );
        // Weapons phase 1: equipped-weapon summary. Keys are the three
        // weapon slot ids; each value is either `null` or the weapon-
        // config + magazine-rounds dict so the HUD can render ammo
        // counters without a separate bridge call.
        let weapons = equipped_weapons_to_dict(sim.item_registry(), &eq);
        d.set(&Variant::from("equipped_weapons"), &Variant::from(weapons));
        d
    }

    /// Grant an item to a player (by item id string). Increments stack
    /// counts or adds new slots as needed. Returns `false` on unknown
    /// player / item or zero count. Client path packages a `GrantItem`
    /// action; host applies and the resulting `ItemPickedUp` delta
    /// broadcasts back.
    #[func]
    fn grant_item(&mut self, steam_id: i64, item_id: GString, count: i64) -> bool {
        if count <= 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let c = count as u32;
        self.dispatch_player_action(
            steam_id,
            ActionKind::GrantItem {
                item_id: item_id.to_string(),
                count: c,
            },
        )
    }

    /// Drop the slot at `slot_idx`. Step 4: the stack vanishes (no
    /// ground-item entity yet).
    #[func]
    fn drop_slot(&mut self, steam_id: i64, slot_idx: i64) -> bool {
        if slot_idx < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let slot = slot_idx as u32;
        self.dispatch_player_action(steam_id, ActionKind::DropSlot { slot_idx: slot })
    }

    /// Swap two slots.
    #[func]
    fn move_slot(&mut self, steam_id: i64, from_slot: i64, to_slot: i64) -> bool {
        if from_slot < 0 || to_slot < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let (f, t) = (from_slot as u32, to_slot as u32);
        self.dispatch_player_action(steam_id, ActionKind::MoveSlot { from: f, to: t })
    }

    /// Move the item at `(from_grid, from_idx)` into `to_grid` at the
    /// first free spot. Grid strings are `"pockets"` or
    /// `"equipped:<slot_id>"`. Returns `false` on failure (same grid,
    /// source empty, dest full, item unknown).
    #[func]
    fn move_between_grids(
        &mut self,
        steam_id: i64,
        from_grid: GString,
        from_idx: i64,
        to_grid: GString,
    ) -> bool {
        if from_idx < 0 {
            return false;
        }
        let from = from_grid.to_string();
        let to = to_grid.to_string();
        #[allow(clippy::cast_sign_loss)]
        let idx = from_idx as u32;
        self.dispatch_player_action(
            steam_id,
            ActionKind::MoveBetweenGrids {
                from_grid: from,
                from_idx: idx,
                to_grid: to,
            },
        )
    }

    /// Consume the item in slot `slot_idx`. `body_part` is only used for
    /// wound-treatment items (bandage/tourniquet/etc.); pass an empty
    /// string for food/drink/drugs/antibiotics.
    #[func]
    fn consume_slot(&mut self, steam_id: i64, slot_idx: i64, body_part: GString) -> bool {
        if slot_idx < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let slot = slot_idx as u32;
        let part_str = body_part.to_string();
        let part = if part_str.is_empty() {
            None
        } else {
            body_part_from_str(&part_str)
        };
        self.dispatch_player_action(
            steam_id,
            ActionKind::ConsumeSlot {
                slot_idx: slot,
                body_part: part,
            },
        )
    }

    /// Salvage the junk item in slot `slot_idx`. Requires the recipe's
    /// `tool_required` to be elsewhere in the inventory.
    #[func]
    fn salvage_slot(&mut self, steam_id: i64, slot_idx: i64) -> bool {
        if slot_idx < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let slot = slot_idx as u32;
        self.dispatch_player_action(steam_id, ActionKind::SalvageSlot { slot_idx: slot })
    }

    /// Craft a recipe by id.
    #[func]
    fn craft_recipe(&mut self, steam_id: i64, recipe_id: GString) -> bool {
        self.dispatch_player_action(
            steam_id,
            ActionKind::CraftRecipe {
                recipe_id: recipe_id.to_string(),
            },
        )
    }

    /// Toggle the debug "near campfire" flag (see
    /// [`simn_sim::Sim::set_player_near_campfire`]).
    #[func]
    fn set_near_campfire(&mut self, steam_id: i64, value: bool) -> bool {
        self.dispatch_player_action(steam_id, ActionKind::SetNearCampfire { value })
    }

    /// Set the workbench-tier proximity flag. `tier_str` is `""` /
    /// `"none"` (clear), `"basic"`, `"advanced"`, or `"expert"`. Step
    /// 5 Slice A's debug setter — the production proximity system
    /// driven by scene-placed workbench entities lands later. Returns
    /// `false` on unknown tier string or unknown player.
    #[func]
    fn set_near_workbench(&mut self, steam_id: i64, tier_str: GString) -> bool {
        let Some(tier) = tool_tier_from_str(&tier_str.to_string()) else {
            godot_error!("set_near_workbench: unknown tier {tier_str}");
            return false;
        };
        self.dispatch_player_action(steam_id, ActionKind::SetNearWorkbench { tier })
    }

    /// Every recipe the sim knows about. One dictionary per recipe with
    /// the schema documented in
    /// `docs/book/src/api/sim-host.md` — id, name, time, required tool /
    /// kit / context, inputs, outputs. Stable across the session;
    /// recipes are loaded once at sim init from the bundled TOML.
    #[func]
    fn recipe_catalog(&self) -> Array<Variant> {
        let mut out: Array<Variant> = Array::new();
        // The recipe list is immutable for the session (TOML-loaded
        // once). In worker mode we own the registry on the worker;
        // pull a cloned `Vec<Recipe>` over the thread boundary
        // via inspect, then build the Variants on the main thread.
        let recipes: Vec<simn_sim::Recipe> = if let Some(worker) = self.worker.as_ref() {
            worker
                .inspect(|sim| sim.recipes().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.recipes().cloned().collect()
        } else {
            return out;
        };
        for recipe in &recipes {
            out.push(&Variant::from(recipe_to_dict(recipe)));
        }
        out
    }

    /// Per-player craftability check. Returns the
    /// [`craftability_to_dict`] schema: `{ ok, inputs[{id,need,have}],
    /// missing_tool, missing_kit, wrong_station }`. Drives the
    /// recipe-browser "Requires:" lines and gates the Queue button.
    /// Returns the report's `ok=false` default for an unknown player
    /// or recipe — UIs should treat that as "can't craft" and show a
    /// generic error in their own copy.
    #[func]
    fn can_craft(&mut self, steam_id: i64, recipe_id: GString) -> Dictionary<Variant, Variant> {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        let recipe_str = recipe_id.to_string();
        let report: simn_sim::CraftabilityReport = if let Some(worker) = self.worker.as_ref() {
            let r = recipe_str.clone();
            worker
                .inspect(move |sim| sim.can_craft(sid, &r))
                .unwrap_or_default()
        } else if let Some(sim) = self.sim.as_mut() {
            sim.can_craft(sid, &recipe_str)
        } else {
            return Dictionary::new();
        };
        craftability_to_dict(&report)
    }

    /// Bulk craftability check. Returns a `Dictionary` keyed by
    /// recipe id with the same per-entry shape as [`Self::can_craft`].
    /// Single worker `inspect` round-trip covers every recipe, so the
    /// recipe-browser refresh is one tick of latency total instead of
    /// (one tick × recipe count). The inventory panel's
    /// `_refresh_recipes` calls this once per refresh and threads the
    /// per-recipe reports through to its row builders.
    #[func]
    fn can_craft_many(
        &mut self,
        steam_id: i64,
        recipe_ids: PackedStringArray,
    ) -> Dictionary<Variant, Variant> {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        let ids: Vec<String> = (0..recipe_ids.len())
            .map(|i| recipe_ids.get(i).unwrap_or_default().to_string())
            .collect();
        let reports: Vec<(String, simn_sim::CraftabilityReport)> =
            if let Some(worker) = self.worker.as_ref() {
                let ids_w = ids.clone();
                worker
                    .inspect(move |sim| {
                        ids_w
                            .iter()
                            .map(|r| (r.clone(), sim.can_craft(sid, r)))
                            .collect()
                    })
                    .unwrap_or_default()
            } else if let Some(sim) = self.sim.as_mut() {
                ids.iter()
                    .map(|r| (r.clone(), sim.can_craft(sid, r)))
                    .collect()
            } else {
                return Dictionary::new();
            };
        let mut out: Dictionary<Variant, Variant> = Dictionary::new();
        for (id, report) in reports {
            out.set(
                &Variant::from(GString::from(id.as_str())),
                &Variant::from(craftability_to_dict(&report)),
            );
        }
        out
    }

    /// Queue `count` units of `recipe_id` on the player. Materials lock
    /// up front; the queue ticks down deterministically. Returns the
    /// new job's id (positive `i64`), or `-1` on validation / unknown
    /// player / unknown recipe / count = 0. Client path packages a
    /// `QueueCraft` action and returns `0` (the host's actual job id
    /// arrives via the next snapshot / `CraftJobQueued` delta).
    #[func]
    fn queue_craft(&mut self, steam_id: i64, recipe_id: GString, count: i64) -> i64 {
        if count <= 0 {
            return -1;
        }
        #[allow(clippy::cast_sign_loss)]
        let c = count as u32;
        let id = recipe_id.to_string();
        if self.is_client() {
            self.emit_action(
                steam_id,
                ActionKind::QueueCraft {
                    recipe_id: id,
                    count: c,
                },
            );
            return 0;
        }
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        // Worker mode: route through `inspect` because we need
        // the typed job-id back synchronously — the ActionKind
        // path is fire-and-forget (no return channel today).
        // Costs at most one tick of latency.
        if let Some(worker) = self.worker.as_ref() {
            let id_w = id.clone();
            #[allow(clippy::cast_lossless)]
            return worker
                .inspect(move |sim| match sim.queue_craft(sid, &id_w, c) {
                    Ok(job_id) => job_id as i64,
                    Err(e) => {
                        tracing::error!("queue_craft (worker): {e:#}");
                        -1
                    }
                })
                .unwrap_or(-1);
        }
        let Some(sim) = self.sim.as_mut() else {
            return -1;
        };
        match sim.queue_craft(sid, &id, c) {
            #[allow(clippy::cast_lossless)]
            Ok(job_id) => job_id as i64,
            Err(e) => {
                godot_error!("queue_craft failed: {e:#}");
                -1
            }
        }
    }

    /// Cancel a queued craft job by id. Refunds materials for unstarted
    /// units; the in-progress unit (if any) is forfeit. Returns
    /// `false` on unknown player / unknown job id, or when called from
    /// a client (the action is dispatched to host instead — UI gets
    /// the canonical refund via the next `CraftJobCancelled` delta).
    #[func]
    fn cancel_craft(&mut self, steam_id: i64, job_id: i64) -> bool {
        if job_id < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let jid = job_id as u32;
        self.dispatch_player_action(steam_id, ActionKind::CancelCraft { job_id: jid })
    }

    /// Paper-doll slot catalog — one dict per slot defined in
    /// `equipment_slots.toml`. Stable across the session; the UI pulls
    /// this once when opening the inventory panel.
    ///
    /// Element schema: see `docs/book/src/api/sim-host.md`.
    #[func]
    fn equipment_slot_catalog(&self) -> Array<Variant> {
        let mut out: Array<Variant> = Array::new();
        let slots: Vec<simn_sim::EquipmentSlotDef> = if let Some(worker) = self.worker.as_ref() {
            worker
                .inspect(|sim| sim.equipment_slots().iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.equipment_slots().iter().cloned().collect()
        } else {
            return out;
        };
        for def in &slots {
            out.push(&Variant::from(slot_def_to_dict(def)));
        }
        out
    }

    /// Equip the item at `(source_grid, source_idx)` into `slot_id`.
    /// `source_grid` is `"pockets"` or `"equipped:<slot_id>"` for a
    /// nested container. Returns `false` on validation failure (wrong
    /// slot, item not found, slot already occupied, bad source grid).
    ///
    /// Client path dispatches a `WorldDelta::ItemEquipped` via
    /// `ActionKind::Equip`; the canonical move lands on the next tick.
    #[func]
    fn equip(
        &mut self,
        steam_id: i64,
        slot_id: GString,
        source_grid: GString,
        source_idx: i64,
    ) -> bool {
        if source_idx < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss)]
        let idx = source_idx as u32;
        self.dispatch_player_action(
            steam_id,
            ActionKind::Equip {
                slot_id: slot_id.to_string(),
                source_grid: source_grid.to_string(),
                source_idx: idx,
            },
        )
    }

    /// Pull the item at `slot_id` off the paper doll into `dest_grid`.
    /// `dest_grid` is `"pockets"` or `"equipped:<slot_id>"` for a
    /// nested container. Returns `false` on failure (slot empty, no
    /// room at destination).
    #[func]
    fn unequip(&mut self, steam_id: i64, slot_id: GString, dest_grid: GString) -> bool {
        self.dispatch_player_action(
            steam_id,
            ActionKind::Unequip {
                slot_id: slot_id.to_string(),
                dest_grid: dest_grid.to_string(),
            },
        )
    }

    // ---------- PR-4c looting bridges ----------

    /// World containers (ground drops, scene-placed crates, NPC
    /// corpses) within `radius_m` of the player, in the same region.
    /// Returns `Array[Dictionary]`; each entry is
    /// `{ id: int, pos: Vector3, is_public: bool }`. Empty array if
    /// the player isn't found or no containers are nearby. Used by
    /// the looting HUD to find the nearest interactable.
    #[func]
    fn containers_in_range(&mut self, steam_id: i64, radius_m: f32) -> Array<Variant> {
        let mut out: Array<Variant> = Array::new();
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        // Closure returns the Send `Vec<(ContainerId, [f32;3], bool)>`
        // payload; main thread builds Variants.
        let rows: Vec<(simn_sim::ContainerId, [f32; 3], bool)> =
            if let Some(worker) = self.worker.as_ref() {
                worker
                    .inspect(move |sim| sim.containers_in_range(sid, radius_m))
                    .unwrap_or_default()
            } else if let Some(sim) = self.sim.as_mut() {
                sim.containers_in_range(sid, radius_m)
            } else {
                return out;
            };
        for (id, pos, is_public) in rows {
            let mut d: Dictionary<Variant, Variant> = Dictionary::new();
            d.set(&Variant::from("id"), &Variant::from(i64::from(id.0)));
            d.set(
                &Variant::from("pos"),
                &Variant::from(Vector3::new(pos[0], pos[1], pos[2])),
            );
            d.set(&Variant::from("is_public"), &Variant::from(is_public));
            out.push(&Variant::from(d));
        }
        out
    }

    /// Snapshot a container's grid for rendering. Returns the grid
    /// dict (`{ width, height, items }`) using the same shape as
    /// `player_state.inventory`, so the looting panel can reuse the
    /// inventory-grid renderer. Returns an empty dict if the id is
    /// unknown.
    #[func]
    fn container_view(&mut self, container_id: i64) -> Dictionary<Variant, Variant> {
        let mut empty: Dictionary<Variant, Variant> = Dictionary::new();
        empty.set(&Variant::from("width"), &Variant::from(0_i64));
        empty.set(&Variant::from("height"), &Variant::from(0_i64));
        empty.set(
            &Variant::from("items"),
            &Variant::from(Array::<Variant>::new()),
        );
        if container_id < 0 {
            return empty;
        }
        let Some(sim) = self.sim.as_mut() else {
            return empty;
        };
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let id = simn_sim::ContainerId(container_id as u32);
        match sim.container_view(id) {
            Some(grid) => conversions::grid_to_dict(sim.item_registry(), &grid),
            None => empty,
        }
    }

    /// Take the item at `source_idx` out of `container_id` and grant
    /// it to the player's pockets. Routes through the action queue on
    /// clients (host validates). Returns `false` on validation
    /// failure (unknown container, idx out of range, pockets full).
    #[func]
    fn take_from_container(&mut self, steam_id: i64, container_id: i64, source_idx: i64) -> bool {
        if container_id < 0 || source_idx < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let cid = container_id as u32;
        #[allow(clippy::cast_sign_loss)]
        let idx = source_idx as u32;
        self.dispatch_player_action(
            steam_id,
            ActionKind::TakeFromContainer {
                container_id: cid,
                source_idx: idx,
            },
        )
    }

    /// Push the item at `(source_grid, source_idx)` from the player
    /// into `container_id`. `source_grid` is `"pockets"` or
    /// `"equipped:<slot_id>"`. Same client/host dispatch + return
    /// shape as `take_from_container`.
    #[func]
    fn put_in_container(
        &mut self,
        steam_id: i64,
        container_id: i64,
        source_grid: GString,
        source_idx: i64,
    ) -> bool {
        if container_id < 0 || source_idx < 0 {
            return false;
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let cid = container_id as u32;
        #[allow(clippy::cast_sign_loss)]
        let idx = source_idx as u32;
        self.dispatch_player_action(
            steam_id,
            ActionKind::PutInContainer {
                container_id: cid,
                source_grid: source_grid.to_string(),
                source_idx: idx,
            },
        )
    }

    /// Spawn a world container at `pos` in `region_name` with
    /// `(width × height)` cells. `is_public` controls whether its
    /// contents count toward the crafting kit-pool (parts bins ⇒
    /// `true`; player stashes / corpses ⇒ `false`). Returns the new
    /// container id, or `-1` on failure (unknown region, host-only
    /// call from a client). Used by region map scenes to place test
    /// crates and scripted loot drops.
    #[func]
    fn spawn_world_container(
        &mut self,
        region_name: GString,
        pos: Vector3,
        width: i64,
        height: i64,
        is_public: bool,
    ) -> i64 {
        if self.is_client() {
            // Spawning containers is host-authoritative; clients
            // shouldn't try (no journal write happens locally).
            godot_error!("spawn_world_container: client cannot spawn containers");
            return -1;
        }
        if width <= 0 || height <= 0 {
            return -1;
        }
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let (w, h) = (width as u32, height as u32);
        let region_str = region_name.to_string();
        let pos_arr = [pos.x, pos.y, pos.z];
        // Resolve region via the cached graph (worker) or directly (direct).
        let region_id = if let Some(worker) = self.worker.as_ref() {
            worker.regions().id_for_name(&region_str)
        } else {
            self.sim
                .as_ref()
                .and_then(|s| s.regions().id_for_name(&region_str))
        };
        let Some(region_id) = region_id else {
            godot_error!("spawn_world_container: unknown region {region_str}");
            return -1;
        };
        if let Some(worker) = self.worker.as_ref() {
            return worker
                .inspect(move |sim| {
                    match sim.spawn_world_container(pos_arr, region_id, w, h, is_public) {
                        Ok(id) => i64::from(id.0),
                        Err(e) => {
                            tracing::error!("spawn_world_container (worker): {e:#}");
                            -1
                        }
                    }
                })
                .unwrap_or(-1);
        }
        let Some(sim) = self.sim.as_mut() else {
            return -1;
        };
        match sim.spawn_world_container(pos_arr, region_id, w, h, is_public) {
            Ok(id) => i64::from(id.0),
            Err(e) => {
                godot_error!("spawn_world_container failed: {e:#}");
                -1
            }
        }
    }

    /// Phase 3D — register a hand-placed `LootContainerMarker3D`
    /// placement with the sim. Resolves the kind grid from
    /// `LootContainerRegistry`, rolls eager initial contents from
    /// `LootPoolRegistry`, and stamps `interaction_mode` onto the
    /// resulting `WorldContainer`.
    ///
    /// Args:
    /// - `region_name` — map id (e.g. `"map_a"`, `"corbett"`).
    /// - `pos` — world-space anchor.
    /// - `kind_id` — `"small_crate"` / `"medium_stash"` /
    ///   `"large_cache"`.
    /// - `is_public` — kit-pool participation; matches the
    ///   marker's export.
    /// - `container_id_str` — marker's stable id; hashed into the
    ///   roll RNG so same-marker same-roll within a save. Empty =
    ///   non-deterministic mix from `(tick, kind_id)`.
    /// - `faction` — owning faction for restock / loot flavor;
    ///   empty = `"nomads"` neutral fallback.
    /// - `depth_tier` — 1 / 2 / 3.
    /// - `interaction_mode` — `"openable"` (default) or
    ///   `"breakable"`. Breakable is data-only until the future
    ///   destruction system lands.
    ///
    /// Returns the new container id, or `-1` on failure (unknown
    /// kind / region, client call). Host-authoritative.
    #[func]
    #[allow(clippy::too_many_arguments)]
    fn register_authored_container(
        &mut self,
        region_name: GString,
        pos: Vector3,
        kind_id: GString,
        is_public: bool,
        container_id_str: GString,
        faction: GString,
        depth_tier: i64,
        interaction_mode: GString,
    ) -> i64 {
        if self.is_client() {
            godot_error!("register_authored_container: client cannot spawn containers");
            return -1;
        }
        let region_str = region_name.to_string();
        let kind_str = kind_id.to_string();
        let cid_str = container_id_str.to_string();
        let faction_str = faction.to_string();
        let mode_str = interaction_mode.to_string();
        let pos_arr = [pos.x, pos.y, pos.z];
        let tier = depth_tier.clamp(1, 255) as u8;

        // Resolve region.
        let region_id = if let Some(worker) = self.worker.as_ref() {
            worker.regions().id_for_name(&region_str)
        } else {
            self.sim
                .as_ref()
                .and_then(|s| s.regions().id_for_name(&region_str))
        };
        let Some(region_id) = region_id else {
            godot_error!("register_authored_container: unknown region {region_str}");
            return -1;
        };

        // Parse mode.
        let mode = match mode_str.as_str() {
            "breakable" => simn_sim::components::ContainerInteractionMode::Breakable,
            _ => simn_sim::components::ContainerInteractionMode::Openable,
        };
        let faction_opt = if faction_str.is_empty() {
            None
        } else {
            Some(faction_str)
        };

        // Hash the container_id_str into a seed so authored
        // placements roll deterministic-per-save contents.
        let seed: u64 = if cid_str.is_empty() {
            0
        } else {
            let mut h: u64 = 0xCBF2_9CE4_8422_2325;
            for b in cid_str.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x100_0000_01B3);
            }
            h
        };

        let do_register = move |sim: &mut simn_sim::Sim| -> i64 {
            match sim.register_authored_container(
                &kind_str,
                region_id,
                pos_arr,
                is_public,
                faction_opt.clone(),
                tier,
                mode,
                seed,
            ) {
                Ok(id) => i64::from(id.0),
                Err(e) => {
                    tracing::error!("register_authored_container: {e:#}");
                    -1
                }
            }
        };

        if let Some(worker) = self.worker.as_ref() {
            return worker.inspect(do_register).unwrap_or(-1);
        }
        let Some(sim) = self.sim.as_mut() else {
            return -1;
        };
        do_register(sim)
    }

    /// Trigger the belt-slot bound to `hotbar_idx` (1-based). Routes
    /// through the underlying `consume_action` (eat / drink /
    /// apply_drug / apply_bandage / …). `body_part` is `""` for
    /// non-treatment items; limb name for wound treatments (bandage,
    /// tourniquet, stitch, …).
    #[func]
    fn consume_hotbar(&mut self, steam_id: i64, hotbar_idx: i64, body_part: GString) -> bool {
        if !(1..=255).contains(&hotbar_idx) {
            return false;
        }
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let idx = hotbar_idx as u8;
        let body_part = body_part.to_string();
        let part = if body_part.is_empty() {
            None
        } else {
            body_part_from_str(&body_part)
        };
        if !body_part.is_empty() && part.is_none() {
            godot_error!("consume_hotbar: unknown body_part {body_part}");
            return false;
        }
        self.dispatch_player_action(
            steam_id,
            ActionKind::HotbarConsume {
                idx,
                body_part: part,
            },
        )
    }

    /// Reload the weapon at `slot_id` (expected: `"primary"`,
    /// `"secondary"`, or `"sidearm"`). Pulls the best-loaded matching-
    /// caliber magazine from the player's pockets, installs it, and
    /// returns any previously-loaded mag back to pockets. Returns
    /// `true` on success, `false` on any error (unknown player, slot
    /// empty, no matching magazine available, etc.). Client path
    /// emits a `ReloadWeapon` action; the host's journal broadcast
    /// will drive the actual state change locally.
    #[func]
    fn reload_weapon(&mut self, steam_id: i64, slot_id: GString) -> bool {
        self.dispatch_player_action(
            steam_id,
            ActionKind::ReloadWeapon {
                slot_id: slot_id.to_string(),
            },
        )
    }

    /// Eject the magazine from the weapon at `slot_id` back to the
    /// player's pockets. Preserves the mag's `loaded_rounds`. Returns
    /// `true` on success, `false` on any error.
    #[func]
    fn eject_magazine(&mut self, steam_id: i64, slot_id: GString) -> bool {
        self.dispatch_player_action(
            steam_id,
            ActionKind::EjectMagazine {
                slot_id: slot_id.to_string(),
            },
        )
    }

    /// Top up the magazine in `slot_id` from pocket ammo stacks
    /// of `round_id`. Returns the number of rounds loaded
    /// (possibly 0 if pockets were empty or the mag was full).
    /// `-1` on hard error (caliber mismatch, variant flip rejection,
    /// unknown slot, etc).
    #[func]
    fn load_rounds(&mut self, steam_id: i64, slot_id: GString, round_id: GString) -> i64 {
        let slot_str = slot_id.to_string();
        let round_str = round_id.to_string();
        if self.is_client() {
            self.emit_action(
                steam_id,
                ActionKind::LoadRoundsIntoMag {
                    slot_id: slot_str.clone(),
                    round_id: round_str.clone(),
                },
            );
            // Optimistic return; host's MagazineLoaded delta
            // rebroadcasts the true state to all clients.
            return 0;
        }
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        if let Some(worker) = self.worker.as_ref() {
            let slot_w = slot_str.clone();
            let round_w = round_str.clone();
            #[allow(clippy::cast_possible_wrap)]
            return worker
                .inspect(move |sim| {
                    match sim.load_rounds_into_mag(
                        sid,
                        &simn_sim::SlotId::from(slot_w.as_str()),
                        &simn_sim::ItemId::from(round_w.as_str()),
                    ) {
                        Ok(n) => n as i64,
                        Err(e) => {
                            tracing::error!("load_rounds (worker): {e:#}");
                            -1
                        }
                    }
                })
                .unwrap_or(-1);
        }
        let Some(sim) = self.sim.as_mut() else {
            return -1;
        };
        match sim.load_rounds_into_mag(
            sid,
            &simn_sim::SlotId::from(slot_str.as_str()),
            &simn_sim::ItemId::from(round_str.as_str()),
        ) {
            #[allow(clippy::cast_possible_wrap)]
            Ok(n) => n as i64,
            Err(e) => {
                godot_error!("load_rounds failed: {e:#}");
                -1
            }
        }
    }

    /// Top up a magazine sitting at `pocket_idx` in the player's
    /// pockets grid from matching-caliber pocket ammo stacks.
    /// Mirrors `load_rounds` semantics but targets a pre-reload
    /// spare — this is the entry point the inventory panel's
    /// right-click "Load rounds" action uses.
    ///
    /// Returns the number of rounds loaded (possibly `0` on
    /// zero-effect calls — empty pockets or full mag). `-1` on
    /// hard error: caliber mismatch, partial-mag variant flip,
    /// unknown slot/ammo, out-of-range index.
    #[func]
    fn load_rounds_into_pocket(
        &mut self,
        steam_id: i64,
        pocket_idx: i64,
        round_id: GString,
    ) -> i64 {
        if pocket_idx < 0 {
            return -1;
        }
        #[allow(clippy::cast_sign_loss)]
        let idx = pocket_idx as u32;
        let round_str = round_id.to_string();
        if self.is_client() {
            self.emit_action(
                steam_id,
                ActionKind::LoadRoundsIntoPocketMag {
                    pocket_idx: idx,
                    round_id: round_str.clone(),
                },
            );
            return 0;
        }
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        if let Some(worker) = self.worker.as_ref() {
            let round_w = round_str.clone();
            #[allow(clippy::cast_possible_wrap)]
            return worker
                .inspect(move |sim| {
                    match sim.load_rounds_into_pocket_mag(
                        sid,
                        idx,
                        &simn_sim::ItemId::from(round_w.as_str()),
                    ) {
                        Ok(n) => n as i64,
                        Err(e) => {
                            tracing::error!("load_rounds_into_pocket (worker): {e:#}");
                            -1
                        }
                    }
                })
                .unwrap_or(-1);
        }
        let Some(sim) = self.sim.as_mut() else {
            return -1;
        };
        match sim.load_rounds_into_pocket_mag(sid, idx, &simn_sim::ItemId::from(round_str.as_str()))
        {
            #[allow(clippy::cast_possible_wrap)]
            Ok(n) => n as i64,
            Err(e) => {
                godot_error!("load_rounds_into_pocket failed: {e:#}");
                -1
            }
        }
    }

    /// Fire the weapon at `slot_id`. Decrements one round from the
    /// loaded magazine and returns a dict the client raycast needs:
    ///
    /// - `ok: bool` — `false` on any error (dry-click, empty slot, …)
    /// - `error: String` — anyhow error text on failure; `""` on `ok`
    /// - `weapon_config: Dictionary` — `{ caliber, damage, range_m,
    ///   fire_interval_s, spread_deg }`, empty on failure
    /// - `remaining_rounds: int` — post-fire magazine count on success,
    ///   `0` on failure
    ///
    /// Hit resolution stays in GDScript for Phase 1 (the client
    /// raycasts with the returned stats). The client should still
    /// call this even when the raycast misses — the sim owns mag
    /// consumption. When called from a client, emits a `FireWeapon`
    /// action; the host's fire path broadcasts `WeaponFired` back.
    #[func]
    fn fire_weapon(
        &mut self,
        steam_id: i64,
        slot_id: GString,
        aim_yaw: f32,
        aim_pitch: f32,
    ) -> Dictionary<Variant, Variant> {
        let slot_str = slot_id.to_string();
        if self.is_client() {
            self.emit_action(
                steam_id,
                ActionKind::FireWeapon {
                    slot_id: slot_str.clone(),
                    aim_yaw,
                    aim_pitch,
                },
            );
            // Commit 4 drops the client tracer-init based on return
            // values — the client watches `ProjectileSpawned` deltas
            // for tracer FX instead. For now the dict still carries
            // `ok` + `remaining_rounds` for HUD-ready signalling.
            return fire_weapon_result(true, "", 0);
        }
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        // Worker mode: closure returns the raw (ok, err, rounds)
        // triple over the thread boundary; main thread builds the
        // Godot Dictionary since Variant types aren't Send.
        if let Some(worker) = self.worker.as_ref() {
            let slot_w = slot_str.clone();
            let triple: (bool, String, u32) = worker
                .inspect(move |sim| {
                    let slot = simn_sim::SlotId::from(slot_w.as_str());
                    match sim.fire_weapon(sid, &slot, aim_yaw, aim_pitch) {
                        Ok(()) => {
                            let remaining = sim
                                .equipment_view(sid)
                                .get(&slot)
                                .and_then(|eq| eq.weapon_state.as_ref())
                                .and_then(|ws| ws.loaded_magazine.as_ref())
                                .map(|mag| mag.loaded_rounds())
                                .unwrap_or(0);
                            (true, String::new(), remaining)
                        }
                        Err(e) => (false, format!("{e:#}"), 0),
                    }
                })
                .unwrap_or_else(|_| (false, "worker inspect channel failed".to_string(), 0));
            return fire_weapon_result(triple.0, triple.1.as_str(), triple.2);
        }
        let Some(sim) = self.sim.as_mut() else {
            return fire_weapon_result(false, "sim not started", 0);
        };
        let slot = simn_sim::SlotId::from(slot_str.as_str());
        match sim.fire_weapon(sid, &slot, aim_yaw, aim_pitch) {
            Ok(()) => {
                let remaining = sim
                    .equipment_view(sid)
                    .get(&slot)
                    .and_then(|eq| eq.weapon_state.as_ref())
                    .and_then(|ws| ws.loaded_magazine.as_ref())
                    .map(|mag| mag.loaded_rounds())
                    .unwrap_or(0);
                fire_weapon_result(true, "", remaining)
            }
            Err(e) => fire_weapon_result(false, &format!("{e:#}"), 0),
        }
    }

    /// List of every item the sim knows about. Each entry: `{ id, name,
    /// category, weight, stack_size, perishable_ticks }`. For debug
    /// overlays / future inventory UI.
    #[func]
    fn item_catalog(&self) -> Array<Variant> {
        let mut out: Array<Variant> = Array::new();
        // Worker mode reads from the cached `Arc<ItemRegistry>`;
        // direct mode reads from sim. Either way the same
        // immutable iterator drives the dict build below.
        let items_iter: Box<dyn Iterator<Item = &simn_sim::ItemDef>> =
            if let Some(worker) = self.worker.as_ref() {
                Box::new(worker.item_registry().iter())
            } else if let Some(sim) = self.sim.as_ref() {
                Box::new(sim.items())
            } else {
                return out;
            };
        for def in items_iter {
            let mut d: Dictionary<Variant, Variant> = Dictionary::new();
            d.set(
                &Variant::from("id"),
                &Variant::from(GString::from(def.id.0.as_str())),
            );
            d.set(
                &Variant::from("name"),
                &Variant::from(GString::from(def.name.as_str())),
            );
            d.set(
                &Variant::from("category"),
                &Variant::from(GString::from(item_category_to_str(def.category))),
            );
            d.set(&Variant::from("weight"), &Variant::from(def.weight));
            #[allow(clippy::cast_possible_wrap)]
            d.set(
                &Variant::from("stack_size"),
                &Variant::from(def.stack_size as i64),
            );
            #[allow(clippy::cast_possible_wrap)]
            d.set(
                &Variant::from("perishable_ticks"),
                &Variant::from(def.perishable_ticks.unwrap_or(0) as i64),
            );
            out.push(&Variant::from(d));
        }
        out
    }

    /// Region transition portals: `{ neighbor_name: Vector3, ... }`.
    /// Scenes use this at load time to spawn `TransitionCube`
    /// instances at the sim-authoritative positions, so the two
    /// sides stay in sync if the graph ever reshapes.
    #[func]
    fn region_transitions(&self, region_name: GString) -> Dictionary<Variant, Variant> {
        let mut out: Dictionary<Variant, Variant> = Dictionary::new();
        let graph: &simn_sim::RegionGraph = if let Some(worker) = self.worker.as_ref() {
            worker.regions().as_ref()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.regions()
        } else {
            return out;
        };
        let name = region_name.to_string();
        let Some(id) = graph.id_for_name(&name) else {
            return out;
        };
        let Some(region) = graph.get(id) else {
            return out;
        };
        for (neighbor_id, pos) in &region.transitions {
            let Some(neighbor) = graph.get(*neighbor_id) else {
                continue;
            };
            out.set(
                &Variant::from(GString::from(&neighbor.name)),
                &Variant::from(Vector3::new(pos[0], pos[1], pos[2])),
            );
        }
        out
    }

    /// Resolve a region name to its scene path (e.g. "map_a" →
    /// "res://scenes/test/test_map_1.tscn"). Returns an empty string
    /// if the region is unknown.
    #[func]
    fn region_map_scene(&self, region_name: GString) -> GString {
        let graph: &simn_sim::RegionGraph = if let Some(worker) = self.worker.as_ref() {
            worker.regions().as_ref()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.regions()
        } else {
            return GString::new();
        };
        let name = region_name.to_string();
        let Some(id) = graph.id_for_name(&name) else {
            return GString::new();
        };
        graph
            .get(id)
            .map(|r| GString::from(&r.map_scene))
            .unwrap_or_default()
    }

    /// Graceful shutdown: flush journal, roll a final snapshot. Call
    /// on quit so the next launch resumes cleanly.
    #[func]
    fn shutdown(&mut self) {
        if let Some(mut sim) = self.sim.take() {
            if let Err(e) = sim.shutdown() {
                godot_error!("sim shutdown failed: {e:?}");
            }
        }
        if let Some(worker) = self.worker.take() {
            // Trigger the worker's in-loop sim.shutdown() before
            // joining. Step 4b-i: do this via the inspect escape
            // hatch (a one-shot mutation). Step 7 promotes this to
            // a typed `SimCommand::Shutdown` with an explicit
            // oneshot reply that the worker handles between ticks.
            // The inner Result (from `Sim::shutdown`) and the outer
            // Result (from the inspect channel) are both logged but
            // shutdown proceeds either way — we still want to join
            // the thread.
            match worker.inspect(|sim| sim.shutdown()) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => godot_error!("sim shutdown failed (in worker): {e:?}"),
                Err(e) => godot_error!("sim worker inspect channel failed: {e:?}"),
            }
            if let Err(e) = worker.shutdown() {
                godot_error!("sim worker thread join failed: {e:?}");
            }
        }
    }

    /// Dev: enable or disable NPC behavior logging (tracing events
    /// under target `"npc.behavior"`). Off by default — chatty at
    /// high population. The Godot stdout/console shows these
    /// whenever `RUST_LOG` includes the target.
    #[func]
    fn set_behavior_log(&mut self, enabled: bool) {
        self.worker_or_direct_mut(move |sim| {
            sim.set_behavior_log(enabled);
        });
    }

    #[func]
    fn behavior_log_enabled(&self) -> bool {
        // Behavior log flag isn't on SimView; in worker mode the
        // flag flips through worker_or_direct_mut + the next view
        // rebuild reflects it. Conservative default for the
        // pre-first-tick window: false.
        if let Some(worker) = self.worker.as_ref() {
            return worker
                .inspect(|sim| sim.behavior_log_enabled())
                .unwrap_or(false);
        }
        self.sim
            .as_ref()
            .map(|s| s.behavior_log_enabled())
            .unwrap_or(false)
    }

    /// Scale every population target by `factor` (0.5 = halve, 2.0 =
    /// double). Used by the in-game density debug control (F10).
    #[func]
    fn scale_population(&mut self, factor: f32) {
        self.worker_or_direct_mut(move |sim| {
            sim.scale_all_population_targets(factor);
            tracing::info!("[population] scaled by {factor:.1}×");
        });
    }

    /// Set population target for a specific region + faction.
    #[func]
    fn set_population_target(&mut self, region_name: GString, faction_name: GString, count: i64) {
        let name = region_name.to_string();
        let faction_str = faction_name.to_string();
        #[allow(clippy::cast_sign_loss)]
        let c = count.max(0) as u32;
        // Resolve region via the cached graph (worker) or directly
        // (direct/mirror). Faction validation happens inside the
        // closure since `FactionRegistry::id_of` isn't on the cached
        // Arc yet.
        let region_id = if let Some(worker) = self.worker.as_ref() {
            worker.regions().id_for_name(&name)
        } else {
            self.sim
                .as_ref()
                .and_then(|s| s.regions().id_for_name(&name))
        };
        let Some(region_id) = region_id else {
            godot_error!("unknown region: {name}");
            return;
        };
        self.worker_or_direct_mut(move |sim| {
            if sim.faction_registry().id_of(&faction_str).is_none() {
                tracing::error!("unknown faction: {faction_str}");
                return;
            }
            sim.set_population_target(region_id, &faction_str, c);
        });
    }

    /// Current tick counter. Useful for HUD / debug overlays.
    /// Worker mode: read from the published `SimView` (lock-free).
    /// Direct mode: read `Sim::current_tick` directly.
    #[func]
    fn current_tick(&self) -> i64 {
        if let Some(worker) = self.worker.as_ref() {
            #[allow(clippy::cast_possible_wrap)]
            return worker.view().map(|v| v.tick as i64).unwrap_or(0);
        }
        #[allow(clippy::cast_possible_wrap)]
        self.sim
            .as_ref()
            .map(|s| s.current_tick() as i64)
            .unwrap_or(0)
    }

    /// Apply damage to a player. Clamped to `[0, max]`.
    #[func]
    fn damage_player(&mut self, steam_id: i64, amount: f32) {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_damage(sid, amount) {
                tracing::error!("damage_player failed: {e:?}");
            }
        });
    }

    /// Heal a player. Clamped to `[0, max]`.
    #[func]
    fn heal_player(&mut self, steam_id: i64, amount: f32) {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.heal(sid, amount) {
                tracing::error!("heal_player failed: {e:?}");
            }
        });
    }

    /// Set a player's stamina to a specific value. Clamped.
    #[func]
    fn set_player_stamina(&mut self, steam_id: i64, value: f32) {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.set_stamina(sid, value) {
                tracing::error!("set_player_stamina failed: {e:?}");
            }
        });
    }

    /// Apply damage to a specific body part. `part` is one of
    /// `"head" | "torso" | "left_arm" | "right_arm" | "left_leg" | "right_leg"`.
    /// Unknown names are ignored (logged). Aggregate `health` mirror
    /// updates to `min(head, torso)` automatically.
    #[func]
    fn damage_part(&mut self, steam_id: i64, part: GString, amount: f32) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("damage_part: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_damage_to_part(sid, p, amount) {
                tracing::error!("damage_part failed: {e:?}");
            }
        });
    }

    /// Heal a specific body part. See [`Self::damage_part`] for the
    /// `part` name mapping.
    #[func]
    fn heal_part(&mut self, steam_id: i64, part: GString, amount: f32) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("heal_part: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.heal_part(sid, p, amount) {
                tracing::error!("heal_part failed: {e:?}");
            }
        });
    }

    /// Apply damage to one of an NPC's body parts. Parallel to
    /// [`Self::damage_part`] for players. Aggregate `health` mirror
    /// updates to `min(head, torso)` automatically; when head or torso
    /// hit 0 the NPC dies on the next tick via `npc_death_check`.
    #[func]
    fn damage_npc_part(&mut self, npc_id: i64, part: GString, amount: f32) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("damage_npc_part: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_damage_to_npc_part(id, p, amount) {
                tracing::error!("damage_npc_part failed: {e:?}");
            }
        });
    }

    /// Heal one of an NPC's body parts. Mirror of
    /// [`Self::damage_npc_part`].
    #[func]
    fn heal_npc_part(&mut self, npc_id: i64, part: GString, amount: f32) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("heal_npc_part: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.heal_npc_part(id, p, amount) {
                tracing::error!("heal_npc_part failed: {e:?}");
            }
        });
    }

    /// Set a survival meter directly. `stat` is one of
    /// `"hunger" | "thirst" | "fatigue"`. Clamped to `[0, 100]`.
    #[func]
    fn set_survival_stat(&mut self, steam_id: i64, stat: GString, value: f32) {
        let Some(s) = survival_stat_from_str(&stat.to_string()) else {
            godot_error!("set_survival_stat: unknown stat {stat}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.set_survival_stat(sid, s, value) {
                tracing::error!("set_survival_stat failed: {e:?}");
            }
        });
    }

    /// Restore survival meters (food/drink/rest). Each delta adds
    /// then clamps to `[0, 100]`. Pass 0.0 to skip a meter.
    #[func]
    fn consume_food(
        &mut self,
        steam_id: i64,
        hunger_delta: f32,
        thirst_delta: f32,
        fatigue_delta: f32,
    ) {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.consume(sid, hunger_delta, thirst_delta, fatigue_delta) {
                tracing::error!("consume_food failed: {e:?}");
            }
        });
    }

    /// Apply a bandage to the most-severe untreated **light** Bleed
    /// (severity ≤ 3) on the given part. Errors logged on `godot_error!`
    /// — heavy bleed needs a tourniquet first.
    #[func]
    fn apply_bandage(&mut self, steam_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_bandage: unknown body part {part}");
            return;
        };
        let _ = self.dispatch_player_action(steam_id, ActionKind::ApplyBandage { part: p });
    }

    /// Apply a tourniquet to all untreated Bleed wounds on the given
    /// limb. Idempotent. Stops bleed of any severity.
    #[func]
    fn apply_tourniquet(&mut self, steam_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_tourniquet: unknown body part {part}");
            return;
        };
        let _ = self.dispatch_player_action(steam_id, ActionKind::ApplyTourniquet { part: p });
    }

    /// Remove a tourniquet from the given part. The wound resumes
    /// bleeding until properly treated (Step 6 introduces the
    /// stitch/disinfect chain that closes a tourniqueted wound).
    #[func]
    fn remove_tourniquet(&mut self, steam_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("remove_tourniquet: unknown body part {part}");
            return;
        };
        let _ = self.dispatch_player_action(steam_id, ActionKind::RemoveTourniquet { part: p });
    }

    /// Apply antiseptic to all `Untreated` Bleed wounds on the part —
    /// flips them to `Disinfected`, preventing infection.
    #[func]
    fn apply_disinfectant(&mut self, steam_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_disinfectant: unknown body part {part}");
            return;
        };
        let _ = self.dispatch_player_action(steam_id, ActionKind::ApplyDisinfectant { part: p });
    }

    /// Apply a stitch (suture kit) to bandaged/tourniqueted/wound-packed
    /// wounds on the part — closes the wound and halves the heal time.
    #[func]
    fn apply_stitch(&mut self, steam_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_stitch: unknown body part {part}");
            return;
        };
        let _ = self.dispatch_player_action(steam_id, ActionKind::ApplyStitch { part: p });
    }

    /// Apply a wound pack (pressure dressing) to untreated/disinfected
    /// Bleed wounds — stops bleed without the tourniquet's necrosis cost.
    #[func]
    fn apply_wound_pack(&mut self, steam_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_wound_pack: unknown body part {part}");
            return;
        };
        let _ = self.dispatch_player_action(steam_id, ActionKind::ApplyWoundPack { part: p });
    }

    /// Apply antibiotics — clears infection on every infected wound
    /// after the configured antibiotics window.
    #[func]
    fn apply_antibiotics(&mut self, steam_id: i64) {
        let _ = self.dispatch_player_action(steam_id, ActionKind::ApplyAntibiotics);
    }

    // ---------- NPC wound treatment API ----------
    //
    // These seven funcs mirror the player treatment entry points
    // above for NPCs. They're host-authoritative only (no
    // `emit_action` fallback for clients) because NPC state lives on
    // the authoritative sim and never round-trips through the client
    // action queue — a future medic-class player ability would
    // likely lean on a targeted `Sim::apply_bandage_npc` via a
    // dedicated `ActionKind`, not these bridge funcs.

    /// NPC twin of [`Self::apply_bandage`].
    #[func]
    fn apply_bandage_npc(&mut self, npc_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_bandage_npc: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_bandage_npc(id, p) {
                tracing::error!("apply_bandage_npc failed: {e:?}");
            }
        });
    }

    /// NPC twin of [`Self::apply_tourniquet`].
    #[func]
    fn apply_tourniquet_npc(&mut self, npc_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_tourniquet_npc: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_tourniquet_npc(id, p) {
                tracing::error!("apply_tourniquet_npc failed: {e:?}");
            }
        });
    }

    /// NPC twin of [`Self::remove_tourniquet`].
    #[func]
    fn remove_tourniquet_npc(&mut self, npc_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("remove_tourniquet_npc: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.remove_tourniquet_npc(id, p) {
                tracing::error!("remove_tourniquet_npc failed: {e:?}");
            }
        });
    }

    /// NPC twin of [`Self::apply_disinfectant`].
    #[func]
    fn apply_disinfectant_npc(&mut self, npc_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_disinfectant_npc: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_disinfectant_npc(id, p) {
                tracing::error!("apply_disinfectant_npc failed: {e:?}");
            }
        });
    }

    /// NPC twin of [`Self::apply_stitch`].
    #[func]
    fn apply_stitch_npc(&mut self, npc_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_stitch_npc: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_stitch_npc(id, p) {
                tracing::error!("apply_stitch_npc failed: {e:?}");
            }
        });
    }

    /// NPC twin of [`Self::apply_wound_pack`].
    #[func]
    fn apply_wound_pack_npc(&mut self, npc_id: i64, part: GString) {
        let Some(p) = body_part_from_str(&part.to_string()) else {
            godot_error!("apply_wound_pack_npc: unknown body part {part}");
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_wound_pack_npc(id, p) {
                tracing::error!("apply_wound_pack_npc failed: {e:?}");
            }
        });
    }

    /// NPC twin of [`Self::apply_antibiotics`].
    #[func]
    fn apply_antibiotics_npc(&mut self, npc_id: i64) {
        #[allow(clippy::cast_sign_loss)]
        let id = simn_sim::NpcId(npc_id as u64);
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.apply_antibiotics_npc(id) {
                tracing::error!("apply_antibiotics_npc failed: {e:?}");
            }
        });
    }

    /// Apply a drug. `drug` is one of `"painkiller" | "morphine" |
    /// "adrenaline" | "stim_cocktail" | "anti_rad" | "anti_tox"`.
    /// Returns `true` on a normal application, `false` on overdose.
    /// On overdose the player still gets the disorientation effect
    /// and tolerance bumps; the call doesn't error.
    #[func]
    fn apply_drug(&mut self, steam_id: i64, drug: GString) -> bool {
        let Some(d) = drug_kind_from_str(&drug.to_string()) else {
            godot_error!("apply_drug: unknown drug {drug}");
            return false;
        };
        if self.is_client() {
            self.emit_action(steam_id, ActionKind::ApplyDrug { drug: d });
            return true;
        }
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        // Worker mode: inspect for the typed Effect/Overdose
        // result so the HUD's "did the drug work?" signal stays
        // synchronous instead of waiting for the next view tick.
        if let Some(worker) = self.worker.as_ref() {
            return worker
                .inspect(move |sim| match sim.apply_drug(sid, d) {
                    Ok(simn_sim::DrugOutcome::Effect) => true,
                    Ok(simn_sim::DrugOutcome::Overdose) => false,
                    Err(e) => {
                        tracing::error!("apply_drug (worker): {e:?}");
                        false
                    }
                })
                .unwrap_or(false);
        }
        let Some(sim) = self.sim.as_mut() else {
            return false;
        };
        match sim.apply_drug(sid, d) {
            Ok(simn_sim::DrugOutcome::Effect) => true,
            Ok(simn_sim::DrugOutcome::Overdose) => false,
            Err(e) => {
                godot_error!("apply_drug failed: {e:?}");
                false
            }
        }
    }

    /// Eat a food item by `kind`. Names match `food_kind_from_str`:
    /// `preserved_ration | fresh_food | raw_meat | cooked_meat |
    /// contaminated_food | field_ration | energy_bar`.
    #[func]
    fn eat(&mut self, steam_id: i64, kind: GString) {
        let Some(k) = food_kind_from_str(&kind.to_string()) else {
            godot_error!("eat: unknown food kind {kind}");
            return;
        };
        let _ = self.dispatch_player_action(steam_id, ActionKind::Eat { kind: k });
    }

    /// Drink a beverage by `kind`. Names: `dirty_water | clean_water |
    /// energy_drink | vodka`.
    #[func]
    fn drink(&mut self, steam_id: i64, kind: GString) {
        let Some(k) = water_kind_from_str(&kind.to_string()) else {
            godot_error!("drink: unknown water kind {kind}");
            return;
        };
        let _ = self.dispatch_player_action(steam_id, ActionKind::Drink { kind: k });
    }

    /// Set radiation directly (debug). Clamped to `[0, 100]`.
    #[func]
    fn set_radiation(&mut self, steam_id: i64, value: f32) {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.set_radiation(sid, value) {
                tracing::error!("set_radiation failed: {e:?}");
            }
        });
    }

    /// Set toxicity directly (debug). Clamped to `[0, 100]`.
    #[func]
    fn set_toxicity(&mut self, steam_id: i64, value: f32) {
        #[allow(clippy::cast_sign_loss)]
        let sid = steam_id as u64;
        self.worker_or_direct_mut(move |sim| {
            if let Err(e) = sim.set_toxicity(sid, value) {
                tracing::error!("set_toxicity failed: {e:?}");
            }
        });
    }

    /// Per-region faction control:
    /// `{ "primary": String, "contested_by": Array[String], "tension": f32 }`.
    /// Empty dict if region or sim isn't ready.
    #[func]
    fn region_control(&mut self, region_name: GString) -> Dictionary<Variant, Variant> {
        let name = region_name.to_string();
        // Worker mode: read from the published `SimView` (lock-free
        // ArcSwap load + HashMap lookup). The debug overlay polls
        // this every frame while open, so going through `inspect`
        // here would burn ~50 ms per call and tank the renderer to
        // ~15 FPS while the overlay is up.
        let state: simn_sim::RegionControlState = if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&name) else {
                return Dictionary::new();
            };
            let Some(view_arc) = worker.view() else {
                return Dictionary::new();
            };
            match view_arc.region_control.get(&region_id) {
                Some(s) => s.clone(),
                None => return Dictionary::new(),
            }
        } else if let Some(sim) = self.sim.as_ref() {
            match sim.region_control_by_name(&name) {
                Some(s) => s.clone(),
                None => return Dictionary::new(),
            }
        } else {
            return Dictionary::new();
        };
        let mut d: Dictionary<Variant, Variant> = Dictionary::new();
        let primary = state
            .primary
            .as_deref()
            .map(GString::from)
            .unwrap_or_default();
        d.set(&Variant::from("primary"), &Variant::from(primary));
        let contested: VarArray = state
            .contested_by
            .iter()
            .map(|f| Variant::from(GString::from(f.as_str())))
            .collect();
        d.set(&Variant::from("contested_by"), &Variant::from(contested));
        d.set(&Variant::from("tension"), &Variant::from(state.tension));
        d
    }

    /// All bases in a region:
    /// `[ { "kind": String, "faction": String, "pos": Vector3,
    ///      "health": f32, "max_health": f32 }, … ]`.
    /// Empty array if region or sim isn't ready.
    #[func]
    fn bases_in_region(&mut self, region_name: GString) -> VarArray {
        let mut out: VarArray = VarArray::new();
        let name = region_name.to_string();
        // Same pattern as npcs_near: worker mode crosses the thread
        // boundary with the Send `Vec<BaseView>` payload via
        // inspect, main thread builds Variants.
        if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&name) else {
                return out;
            };
            let views: Vec<BaseView> = worker
                .inspect(move |sim| sim.bases_in_region(region_id))
                .unwrap_or_default();
            let registry = worker.faction_registry();
            for view in views {
                out.push(&Variant::from(base_view_to_dict(&view, registry)));
            }
            return out;
        }
        let Some(sim) = self.sim.as_mut() else {
            return out;
        };
        let Some(region_id) = sim.regions().id_for_name(&name) else {
            return out;
        };
        let views: Vec<_> = sim.bases_in_region(region_id);
        let registry = sim.faction_registry();
        for view in views {
            out.push(&Variant::from(base_view_to_dict(&view, registry)));
        }
        out
    }

    /// Lowercased relation between two factions, applying registry
    /// inheritance + runtime drift:
    /// `"hostile" | "cold" | "neutral" | "warm" | "friendly"`.
    /// Empty string if either name isn't in the registry.
    #[func]
    fn faction_relation(&self, a: GString, b: GString) -> GString {
        let a_str = a.to_string();
        let b_str = b.to_string();
        // Worker mode: resolve the faction IDs via the cached
        // registry, then inspect for the relation deltas (which
        // drift at runtime and aren't on the cached snapshot).
        // Inspect's one-tick latency is fine — this isn't hot path.
        if let Some(worker) = self.worker.as_ref() {
            let registry = worker.faction_registry();
            let Some(fa) = registry.id_of(&a_str) else {
                return GString::new();
            };
            let Some(fb) = registry.id_of(&b_str) else {
                return GString::new();
            };
            let registry_arc = registry.clone();
            let r = worker
                .inspect(move |sim| {
                    simn_sim::registry_faction_relation(
                        &registry_arc,
                        sim.relation_deltas(),
                        fa,
                        fb,
                    )
                })
                .unwrap_or(simn_sim::Relation::Neutral);
            return GString::from(relation_to_str(r));
        }
        let Some(sim) = self.sim.as_ref() else {
            return GString::new();
        };
        let registry = sim.faction_registry();
        let Some(fa) = registry.id_of(&a_str) else {
            return GString::new();
        };
        let Some(fb) = registry.id_of(&b_str) else {
            return GString::new();
        };
        let r = simn_sim::registry_faction_relation(registry, sim.relation_deltas(), fa, fb);
        GString::from(relation_to_str(r))
    }

    /// All live NPCs in a region, as
    /// `[ { id, faction, pos, yaw, health, max_health }, … ]`.
    /// Empty array if region or sim isn't ready.
    ///
    /// **Performance.** This call marshals every NPC in the region
    /// into a heavy `Dictionary` (~15 keys per NPC, plus nested
    /// `body_parts` + `wounds` arrays). At full population (~800
    /// NPCs per region) that's tens of thousands of `Variant`
    /// allocations per call — fine for one-off inspector queries
    /// but ruinous on a 20Hz dummy-sync poll. Use [`Self::npcs_near`]
    /// instead for the per-tick renderer path.
    #[func]
    fn npcs_in_region(&mut self, region_name: GString) -> VarArray {
        let mut out: VarArray = VarArray::new();
        let name = region_name.to_string();
        if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&name) else {
                return out;
            };
            let views: Vec<NpcView> = worker
                .inspect(move |sim| sim.npcs_in_region(region_id))
                .unwrap_or_default();
            let registry = worker.faction_registry();
            for view in views {
                out.push(&Variant::from(npc_view_to_dict(&view, registry)));
            }
            return out;
        }
        let Some(sim) = self.sim.as_mut() else {
            return out;
        };
        let Some(region_id) = sim.regions().id_for_name(&name) else {
            return out;
        };
        let views: Vec<_> = sim.npcs_in_region(region_id);
        let registry = sim.faction_registry();
        for view in views {
            out.push(&Variant::from(npc_view_to_dict(&view, registry)));
        }
        out
    }

    /// NPCs in `region_name` within `max_dist_m` of `player_pos`,
    /// same dict schema as [`Self::npcs_in_region`]. Same as the
    /// unfiltered call but skips the per-NPC clone / marshal for
    /// NPCs the player can't see — the typical case on a 20Hz
    /// dummy-sync poll where draw distance is well under sight
    /// radius. Distance gate uses squared XZ math (cheap).
    #[func]
    fn npcs_near(
        &mut self,
        region_name: GString,
        player_pos: godot::prelude::Vector3,
        max_dist_m: f32,
    ) -> VarArray {
        let mut out: VarArray = VarArray::new();
        let name = region_name.to_string();
        let pos = [player_pos.x, player_pos.y, player_pos.z];
        // Worker mode: read from the published `SimView` (lock-free
        // ArcSwap load + HashMap lookup), filter by distance on the
        // main thread. The `worker.inspect` path used to block ~25 ms
        // per call; `_sync_npc_dummies` running at 20 Hz was costing
        // half the main thread's time. The view caches every NPC in
        // every active region once per tick (see
        // `active_region_npc_views`).
        if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&name) else {
                return out;
            };
            let Some(view) = worker.view() else {
                return out;
            };
            let Some(views) = view.npcs_by_region.get(&region_id) else {
                return out;
            };
            let registry = worker.faction_registry();
            let r2 = max_dist_m * max_dist_m;
            for v in views {
                let dx = v.pos[0] - pos[0];
                let dz = v.pos[2] - pos[2];
                if dx * dx + dz * dz > r2 {
                    continue;
                }
                out.push(&Variant::from(npc_view_to_dict(v, registry)));
            }
            return out;
        }
        let Some(sim) = self.sim.as_mut() else {
            return out;
        };
        let Some(region_id) = sim.regions().id_for_name(&name) else {
            return out;
        };
        let views: Vec<_> = sim.npcs_near(region_id, pos, max_dist_m);
        let registry = sim.faction_registry();
        for view in views {
            out.push(&Variant::from(npc_view_to_dict(&view, registry)));
        }
        out
    }

    /// Threaded-sim PR A scaffold (2026-05-11): publish-only snapshot
    /// query. Returns `true` if both `prev` and `curr` snapshot slots
    /// are populated — i.e. enough ticks have run that the renderer
    /// can interpolate between consecutive snapshots. PR B will add
    /// the position-lerp API on top; this getter is the gate the
    /// renderer checks before switching from the `npcs_near` polling
    /// path to the snapshot-pair path.
    ///
    /// Returns `false` on fresh sims (one tick or fewer) and on mirror
    /// sims whose snapshot ring hasn't filled yet.
    #[func]
    fn has_snapshot_pair(&self) -> bool {
        if let Some(worker) = self.worker.as_ref() {
            return worker.snapshots().is_some();
        }
        self.sim.as_ref().and_then(|s| s.snapshot_pair()).is_some()
    }

    /// Current snapshot tick number. Useful for GDScript-side
    /// diagnostics — confirms snapshots are advancing at the
    /// sim's tick rate. Returns -1 if no snapshot has been
    /// published yet.
    #[func]
    fn snapshot_current_tick(&self) -> i64 {
        if let Some(worker) = self.worker.as_ref() {
            #[allow(clippy::cast_possible_wrap)]
            return worker.snapshots().map(|p| p.curr.tick as i64).unwrap_or(-1);
        }
        match self.sim.as_ref().and_then(|s| s.current_snapshot()) {
            #[allow(clippy::cast_possible_wrap)]
            Some(s) => s.tick as i64,
            None => -1,
        }
    }

    /// Rolling tick-perf snapshot for live diagnostics. Returns a
    /// `Dictionary` with `samples`, `avg_total_ms`, `p99_total_ms`,
    /// `avg_schedule_ms`, `p99_schedule_ms` over the last ~10 s of
    /// ticks (200 samples at 20 Hz). All zeros before the first tick
    /// completes. **Use case**: hook this to a debug HUD label or
    /// query via `execute_game_script` to chase a slow-tick
    /// regression without needing `SIMN_VERBOSE=1`.
    #[func]
    fn tick_perf(&self) -> Dictionary<Variant, Variant> {
        let perf = if let Some(worker) = self.worker.as_ref() {
            worker
                .inspect(|sim| sim.recent_tick_perf())
                .unwrap_or_default()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.recent_tick_perf()
        } else {
            simn_sim::TickPerfReport::default()
        };
        let mut out: Dictionary<Variant, Variant> = Dictionary::new();
        out.set(
            &Variant::from("samples"),
            &Variant::from(perf.samples as i64),
        );
        out.set(
            &Variant::from("avg_total_ms"),
            &Variant::from(perf.avg_total_ms),
        );
        out.set(
            &Variant::from("p99_total_ms"),
            &Variant::from(perf.p99_total_ms),
        );
        out.set(
            &Variant::from("avg_player_ms"),
            &Variant::from(perf.avg_player_ms),
        );
        out.set(
            &Variant::from("p99_player_ms"),
            &Variant::from(perf.p99_player_ms),
        );
        out.set(
            &Variant::from("avg_npc_index_ms"),
            &Variant::from(perf.avg_npc_index_ms),
        );
        out.set(
            &Variant::from("p99_npc_index_ms"),
            &Variant::from(perf.p99_npc_index_ms),
        );
        out.set(
            &Variant::from("avg_clear_los_ms"),
            &Variant::from(perf.avg_clear_los_ms),
        );
        out.set(
            &Variant::from("avg_sweep_bb_ms"),
            &Variant::from(perf.avg_sweep_bb_ms),
        );
        out.set(
            &Variant::from("avg_position_index_ms"),
            &Variant::from(perf.avg_position_index_ms),
        );
        out.set(
            &Variant::from("avg_drain_events_ms"),
            &Variant::from(perf.avg_drain_events_ms),
        );
        out.set(
            &Variant::from("avg_spatial_hash_ms"),
            &Variant::from(perf.avg_spatial_hash_ms),
        );
        out.set(
            &Variant::from("avg_event_count"),
            &Variant::from(perf.avg_event_count),
        );
        out.set(
            &Variant::from("max_event_count"),
            &Variant::from(perf.max_event_count as i64),
        );
        out.set(
            &Variant::from("avg_npc_threats_ms"),
            &Variant::from(perf.avg_npc_threats_ms),
        );
        out.set(
            &Variant::from("p99_npc_threats_ms"),
            &Variant::from(perf.p99_npc_threats_ms),
        );
        out.set(
            &Variant::from("avg_npc_aggro_ms"),
            &Variant::from(perf.avg_npc_aggro_ms),
        );
        out.set(
            &Variant::from("p99_npc_aggro_ms"),
            &Variant::from(perf.p99_npc_aggro_ms),
        );
        out.set(
            &Variant::from("avg_npc_perception_ms"),
            &Variant::from(perf.avg_npc_perception_ms),
        );
        out.set(
            &Variant::from("p99_npc_perception_ms"),
            &Variant::from(perf.p99_npc_perception_ms),
        );
        out.set(
            &Variant::from("avg_npc_planning_ms"),
            &Variant::from(perf.avg_npc_planning_ms),
        );
        out.set(
            &Variant::from("p99_npc_planning_ms"),
            &Variant::from(perf.p99_npc_planning_ms),
        );
        out.set(
            &Variant::from("avg_squad_planner_ms"),
            &Variant::from(perf.avg_squad_planner_ms),
        );
        out.set(
            &Variant::from("p99_squad_planner_ms"),
            &Variant::from(perf.p99_squad_planner_ms),
        );
        out.set(
            &Variant::from("avg_goal_arbitration_ms"),
            &Variant::from(perf.avg_goal_arbitration_ms),
        );
        out.set(
            &Variant::from("p99_goal_arbitration_ms"),
            &Variant::from(perf.p99_goal_arbitration_ms),
        );
        out.set(
            &Variant::from("avg_tick_npc_goals_ms"),
            &Variant::from(perf.avg_tick_npc_goals_ms),
        );
        out.set(
            &Variant::from("p99_tick_npc_goals_ms"),
            &Variant::from(perf.p99_tick_npc_goals_ms),
        );
        out.set(
            &Variant::from("avg_npc_combat_ms"),
            &Variant::from(perf.avg_npc_combat_ms),
        );
        out.set(
            &Variant::from("p99_npc_combat_ms"),
            &Variant::from(perf.p99_npc_combat_ms),
        );
        out.set(
            &Variant::from("avg_npc_lifecycle_ms"),
            &Variant::from(perf.avg_npc_lifecycle_ms),
        );
        out.set(
            &Variant::from("p99_npc_lifecycle_ms"),
            &Variant::from(perf.p99_npc_lifecycle_ms),
        );
        out.set(
            &Variant::from("avg_offline_loot_ms"),
            &Variant::from(perf.avg_offline_loot_ms),
        );
        out.set(
            &Variant::from("p99_offline_loot_ms"),
            &Variant::from(perf.p99_offline_loot_ms),
        );
        out
    }

    /// Threaded-sim PR B (2026-05-11): per-render-frame interpolated
    /// pose query. Returns parallel `PackedArrays` of `(ids,
    /// positions, yaws)` for active-region NPCs within `max_dist_m`
    /// of `player_pos`, interpolated between the two most recent
    /// snapshots based on wall-clock time since the latest publish.
    ///
    /// Schema (Dictionary):
    /// - `"ids"`: `PackedInt64Array` — NpcId per row
    /// - `"positions"`: `PackedVector3Array` — render position
    /// - `"yaws"`: `PackedFloat32Array` — render Y-rotation
    ///
    /// All three arrays have the same length and are aligned by
    /// index. Empty arrays when the snapshot pair isn't ready yet
    /// (fresh sim, < 2 ticks published) or the region isn't known.
    ///
    /// **Why parallel `PackedArrays`.** GDScript's `Dictionary` /
    /// `Array<Variant>` round-trip boxes every element. With ~50
    /// NPCs in draw range × 60+ FPS = thousands of elements/sec,
    /// that's Variant allocations dominating the per-frame budget.
    /// Packed arrays are tight `[T]` slices in shared memory — one
    /// allocation per array per frame, regardless of element count.
    ///
    /// Renderer use:
    /// ```gdscript
    /// var b: Dictionary = sim.snapshot_interp_npcs_near(region, player_pos, 300.0)
    /// var ids: PackedInt64Array = b.get("ids", PackedInt64Array())
    /// var positions: PackedVector3Array = b.get("positions", PackedVector3Array())
    /// for i in range(ids.size()):
    ///     var dummy = _npc_dummies.get(ids[i])
    ///     if dummy: dummy.global_position = positions[i]
    /// ```
    #[func]
    fn snapshot_interp_npcs_near(
        &self,
        region_name: GString,
        player_pos: godot::prelude::Vector3,
        max_dist_m: f32,
    ) -> Dictionary<Variant, Variant> {
        let mut out: Dictionary<Variant, Variant> = Dictionary::new();
        let name = region_name.to_string();
        let pos = [player_pos.x, player_pos.y, player_pos.z];
        let now = std::time::Instant::now();
        // Worker mode: lock-free path. Read the snapshot pair
        // and the region graph (both behind Arcs cached on the
        // worker) without touching the sim itself.
        let poses = if let Some(worker) = self.worker.as_ref() {
            let Some(region_id) = worker.regions().id_for_name(&name) else {
                return out;
            };
            worker.snapshot_interp_npcs_near(region_id, pos, max_dist_m, now)
        } else if let Some(sim) = self.sim.as_ref() {
            let Some(region_id) = sim.regions().id_for_name(&name) else {
                return out;
            };
            sim.snapshot_interp_npcs_near(region_id, pos, max_dist_m, now)
        } else {
            return out;
        };
        let mut ids = PackedInt64Array::new();
        let mut positions = PackedVector3Array::new();
        let mut yaws = PackedFloat32Array::new();
        for p in &poses {
            #[allow(clippy::cast_possible_wrap)]
            ids.push(p.id.0 as i64);
            positions.push(godot::prelude::Vector3::new(p.pos[0], p.pos[1], p.pos[2]));
            yaws.push(p.yaw);
        }
        out.set(&Variant::from("ids"), &Variant::from(ids));
        out.set(&Variant::from("positions"), &Variant::from(positions));
        out.set(&Variant::from("yaws"), &Variant::from(yaws));
        out
    }

    /// Chronicle aggregate. Fast — just an O(records) walk for now.
    #[func]
    fn chronicle_summary(&self) -> Dictionary<Variant, Variant> {
        if let Some(worker) = self.worker.as_ref() {
            return worker
                .view()
                .map(|v| chronicle_summary_to_dict(&v.chronicle_summary))
                .unwrap_or_default();
        }
        let Some(sim) = self.sim.as_ref() else {
            return Dictionary::new();
        };
        chronicle_summary_to_dict(&sim.chronicle_summary())
    }

    /// Most recent deaths, newest first, capped at `limit`.
    /// Each entry has id / faction / birth_tick / death_tick /
    /// birth_region / death_region (string names) / cause.
    #[func]
    fn recent_deaths(&self, limit: i64) -> VarArray {
        let mut out: VarArray = VarArray::new();
        #[allow(clippy::cast_sign_loss)]
        let cap = limit.max(0) as usize;
        let recs: Vec<simn_sim::LifeRecord> = if let Some(worker) = self.worker.as_ref() {
            worker
                .inspect(move |sim| {
                    sim.recent_deaths(cap)
                        .into_iter()
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.recent_deaths(cap).into_iter().cloned().collect()
        } else {
            return out;
        };
        for rec in &recs {
            let mut d: Dictionary<Variant, Variant> = Dictionary::new();
            #[allow(clippy::cast_possible_wrap)]
            d.set(&Variant::from("id"), &Variant::from(rec.id.0 as i64));
            d.set(
                &Variant::from("faction"),
                &Variant::from(GString::from(rec.faction.as_str())),
            );
            #[allow(clippy::cast_possible_wrap)]
            d.set(
                &Variant::from("birth_tick"),
                &Variant::from(rec.birth_tick as i64),
            );
            #[allow(clippy::cast_possible_wrap)]
            d.set(
                &Variant::from("death_tick"),
                &Variant::from(rec.death_tick.unwrap_or(0) as i64),
            );
            // Resolve region names via worker's cached graph in
            // worker mode (lock-free), or the mirror sim's graph
            // otherwise.
            let graph: Option<&simn_sim::RegionGraph> = if let Some(worker) = self.worker.as_ref() {
                Some(worker.regions().as_ref())
            } else {
                self.sim.as_ref().map(|s| s.regions())
            };
            d.set(
                &Variant::from("birth_region"),
                &Variant::from(GString::from(
                    graph
                        .and_then(|g| g.get(rec.birth_region))
                        .map(|r| r.name.as_str())
                        .unwrap_or(""),
                )),
            );
            d.set(
                &Variant::from("death_region"),
                &Variant::from(GString::from(
                    rec.death_region
                        .and_then(|r| graph.and_then(|g| g.get(r)))
                        .map(|r| r.name.as_str())
                        .unwrap_or(""),
                )),
            );
            d.set(
                &Variant::from("cause"),
                &Variant::from(GString::from(
                    rec.death_cause
                        .as_ref()
                        .map(death_cause_to_str)
                        .unwrap_or(""),
                )),
            );
            out.push(&Variant::from(d));
        }
        out
    }

    /// PDA event feed (Phase 1F). Returns offline-tier events with
    /// `seq > since_seq`, oldest first. Each entry is a Dictionary
    /// with `seq` (int), `tick` (int), `kind` (string —
    /// `"OfflineCombatDeath"` / `"OfflineGunfire"` / `"BaseFlip"`),
    /// and kind-specific extras (`killed_faction`, `killer_faction`,
    /// `new_owner`, `old_owner`, `region` as string name).
    ///
    /// Client UX contract: call once on `_ready` with `since_seq = 0`
    /// to initialize, then poll each frame (or on `view_updated`
    /// once that signal lands) passing the highest seq seen so far.
    #[func]
    fn recent_pda_events_since(&self, since_seq: i64) -> VarArray {
        let mut out: VarArray = VarArray::new();
        #[allow(clippy::cast_sign_loss)]
        let since = since_seq.max(0) as u64;
        // Worker mode: read from the published `SimView` (lock-free
        // ArcSwap load). The PDA toast script polls this at 4 Hz —
        // going through `worker.inspect` blocked the main thread up
        // to one full tick (50 ms) per poll, costing ~5-10 FPS even
        // when nothing changed. The view caches every PDA entry at
        // end-of-tick build (see `pda_log_view_snapshot`).
        let entries: Vec<simn_sim::PdaLogEntry> = if let Some(worker) = self.worker.as_ref() {
            let Some(view) = worker.view() else {
                return out;
            };
            view.pda_recent
                .iter()
                .filter(|e| e.seq > since)
                .cloned()
                .collect()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.recent_pda_events_since(since)
        } else {
            return out;
        };
        let graph: Option<&simn_sim::RegionGraph> = if let Some(worker) = self.worker.as_ref() {
            Some(worker.regions().as_ref())
        } else {
            self.sim.as_ref().map(|s| s.regions())
        };
        let region_name = |r: simn_sim::RegionId| -> String {
            graph
                .and_then(|g| g.get(r))
                .map(|reg| reg.name.clone())
                .unwrap_or_default()
        };
        for entry in entries {
            let mut d: Dictionary<Variant, Variant> = Dictionary::new();
            #[allow(clippy::cast_possible_wrap)]
            d.set(&Variant::from("seq"), &Variant::from(entry.seq as i64));
            #[allow(clippy::cast_possible_wrap)]
            d.set(&Variant::from("tick"), &Variant::from(entry.tick as i64));
            match entry.event {
                simn_sim::PdaEvent::OfflineCombatDeath {
                    killed_faction,
                    killer_faction,
                    region,
                } => {
                    d.set(
                        &Variant::from("kind"),
                        &Variant::from(GString::from("OfflineCombatDeath")),
                    );
                    d.set(
                        &Variant::from("killed_faction"),
                        &Variant::from(GString::from(killed_faction.as_str())),
                    );
                    d.set(
                        &Variant::from("killer_faction"),
                        &Variant::from(GString::from(killer_faction.as_str())),
                    );
                    d.set(
                        &Variant::from("region"),
                        &Variant::from(GString::from(region_name(region).as_str())),
                    );
                }
                simn_sim::PdaEvent::OfflineGunfire { region } => {
                    d.set(
                        &Variant::from("kind"),
                        &Variant::from(GString::from("OfflineGunfire")),
                    );
                    d.set(
                        &Variant::from("region"),
                        &Variant::from(GString::from(region_name(region).as_str())),
                    );
                }
                simn_sim::PdaEvent::BaseFlip {
                    new_owner,
                    old_owner,
                    region,
                } => {
                    d.set(
                        &Variant::from("kind"),
                        &Variant::from(GString::from("BaseFlip")),
                    );
                    d.set(
                        &Variant::from("new_owner"),
                        &Variant::from(GString::from(new_owner.as_str())),
                    );
                    d.set(
                        &Variant::from("old_owner"),
                        &Variant::from(GString::from(old_owner.unwrap_or_default().as_str())),
                    );
                    d.set(
                        &Variant::from("region"),
                        &Variant::from(GString::from(region_name(region).as_str())),
                    );
                }
            }
            out.push(&Variant::from(d));
        }
        out
    }

    /// Current highest seq in the PDA log. Clients call this on
    /// `_ready` and use the result as their initial bookmark so
    /// pre-join events don't get toasted.
    #[func]
    fn pda_log_high_water(&self) -> i64 {
        // View-based read, same rationale as
        // `recent_pda_events_since`. Avoids the inspect channel
        // block.
        let hw = if let Some(worker) = self.worker.as_ref() {
            worker.view().map(|v| v.pda_high_water).unwrap_or(0)
        } else if let Some(sim) = self.sim.as_ref() {
            sim.pda_log_high_water()
        } else {
            0
        };
        #[allow(clippy::cast_possible_wrap)]
        let out = hw as i64;
        out
    }

    /// In-world clock: `{ day, seconds_of_day, day_length_seconds }`.
    /// Empty dict if the sim hasn't started yet.
    #[func]
    fn world_time(&self) -> Dictionary<Variant, Variant> {
        if let Some(worker) = self.worker.as_ref() {
            return worker
                .view()
                .map(|v| world_time_to_dict(&v.world_time))
                .unwrap_or_default();
        }
        let Some(sim) = self.sim.as_ref() else {
            return Dictionary::new();
        };
        world_time_to_dict(&sim.world_time())
    }

    /// Force the current weather to `name` (e.g. `"heavy_rain"`).
    /// No-op if the name is unrecognized.
    #[func]
    fn set_weather(&mut self, name: GString) {
        let Some(w) = weather_from_str(&name.to_string()) else {
            godot_error!("unknown weather: {name}");
            return;
        };
        self.worker_or_direct_mut(move |sim| {
            sim.set_weather(w);
        });
    }

    /// Cycle to the next weather type (for debug hotkeys). Returns the
    /// new weather name.
    #[func]
    fn cycle_weather(&mut self) -> GString {
        // Inspect lets us read current + advance + return the new
        // tag in one round-trip. In worker mode this blocks main
        // for up to one tick; debug hotkey, not hot path.
        if let Some(worker) = self.worker.as_ref() {
            let name = worker
                .inspect(|sim| {
                    let current = sim.weather().current;
                    let idx = ALL_WEATHER.iter().position(|w| *w == current).unwrap_or(0);
                    let next = ALL_WEATHER[(idx + 1) % ALL_WEATHER.len()];
                    sim.set_weather(next);
                    weather_to_str(next).to_string()
                })
                .unwrap_or_default();
            return GString::from(name.as_str());
        }
        let Some(sim) = self.sim.as_mut() else {
            return GString::new();
        };
        let current = sim.weather().current;
        let idx = ALL_WEATHER.iter().position(|w| *w == current).unwrap_or(0);
        let next = ALL_WEATHER[(idx + 1) % ALL_WEATHER.len()];
        sim.set_weather(next);
        GString::from(weather_to_str(next))
    }

    /// Set the in-world time of day. `hour` is 0–23, `minute` is 0–59.
    #[func]
    fn set_time_of_day(&mut self, hour: i64, minute: i64) {
        #[allow(clippy::cast_sign_loss)]
        let (h, m) = (hour.clamp(0, 23) as u32, minute.clamp(0, 59) as u32);
        self.worker_or_direct_mut(move |sim| {
            sim.set_time_of_day(h, m);
        });
    }

    /// Advance in-world clock by `hours` hours. Handles day rollover.
    #[func]
    fn advance_time(&mut self, hours: f32) {
        self.worker_or_direct_mut(move |sim| {
            sim.advance_time(hours);
        });
    }

    /// All recognized weather type names, in order. See
    /// [`Self::set_weather`] for how the returned strings are used.
    ///
    /// **Returns:** `Array[String]` — snake_case weather tags, e.g.
    /// `["clear", "partly_cloudy", "overcast", "marine_layer", "fog",
    /// "drizzle", "light_rain", "heavy_rain", "windstorm",
    /// "thunderstorm", "smoke_haze"]`. Empty array if `start()` hasn't
    /// been called yet.
    #[func]
    fn all_weather_types(&self) -> VarArray {
        let mut out: VarArray = VarArray::new();
        for w in ALL_WEATHER {
            out.push(&Variant::from(GString::from(weather_to_str(*w))));
        }
        out
    }

    /// All recognized faction names. Use the returned strings with
    /// [`Self::faction_relation`], `set_population_target`, or any
    /// `#[func]` that accepts a faction tag.
    ///
    /// **Returns:** `Array[String]` — registry name strings in
    /// alphabetical order (the registry sorts by name at build).
    /// Empty array if `start()` hasn't been called yet.
    #[func]
    fn all_factions(&self) -> VarArray {
        let mut out: VarArray = VarArray::new();
        let registry: &simn_sim::FactionRegistry = if let Some(worker) = self.worker.as_ref() {
            worker.faction_registry().as_ref()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.faction_registry()
        } else {
            return out;
        };
        for def in registry.defs() {
            out.push(&Variant::from(GString::from(def.name.as_str())));
        }
        out
    }

    /// Per-faction debug color, sourced from `factions.toml` via the
    /// registry. Used by GDScript debug overlays (minimap dots,
    /// marker pills, dev-mode NPC tints) so the palette stays in
    /// sync with the registry without a hardcoded GDScript-side
    /// table. Returns magenta (`#ff00ff`) for unknown faction names
    /// so missing registry entries are visually obvious in dev.
    #[func]
    fn faction_debug_color(&self, name: GString) -> Color {
        const UNKNOWN: Color = Color {
            r: 1.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        };
        let registry: &simn_sim::FactionRegistry = if let Some(worker) = self.worker.as_ref() {
            worker.faction_registry().as_ref()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.faction_registry()
        } else {
            return UNKNOWN;
        };
        let name_str = name.to_string();
        let Some(rgb) = registry
            .id_of(&name_str)
            .map(|id| registry.def(id).debug_color)
        else {
            return UNKNOWN;
        };
        Color {
            r: f32::from(rgb[0]) / 255.0,
            g: f32::from(rgb[1]) / 255.0,
            b: f32::from(rgb[2]) / 255.0,
            a: 1.0,
        }
    }

    /// All region names the sim knows about. Use with
    /// [`Self::region_control`], `region_map_scene`, `bases_in_region`,
    /// `npcs_in_region`, or any `#[func]` that accepts a region tag.
    ///
    /// **Returns:** `Array[String]` — region names in the order the
    /// `RegionGraph` stores them. Empty array if `start()` hasn't been
    /// called yet.
    #[func]
    fn all_regions(&self) -> VarArray {
        let mut out: VarArray = VarArray::new();
        let graph: &simn_sim::RegionGraph = if let Some(worker) = self.worker.as_ref() {
            worker.regions().as_ref()
        } else if let Some(sim) = self.sim.as_ref() {
            sim.regions()
        } else {
            return out;
        };
        for region in graph.regions.values() {
            out.push(&Variant::from(GString::from(region.name.as_str())));
        }
        out
    }

    /// Global weather snapshot:
    /// `{ current: String, next: String, transitions_at_tick: i64 }`.
    /// Empty dict if the sim isn't started.
    #[func]
    fn weather_state(&self) -> Dictionary<Variant, Variant> {
        if let Some(worker) = self.worker.as_ref() {
            return worker
                .view()
                .map(|v| weather_state_to_dict(&v.weather))
                .unwrap_or_default();
        }
        let Some(sim) = self.sim.as_ref() else {
            return Dictionary::new();
        };
        weather_state_to_dict(&sim.weather())
    }

    // ---------- Sim/net slice-1 surface ----------

    /// Wire a sibling `NetworkManager` so mutation methods can check
    /// role + forward actions when this peer is a client. Passing
    /// `null` detaches; after detach, SimHost behaves like solo
    /// (always authoritative).
    #[func]
    fn attach_network(&mut self, nm: Gd<NetworkManager>) {
        self.network = Some(nm);
    }

    #[func]
    fn detach_network(&mut self) {
        self.network = None;
    }

    /// Create a **mirror** sim: no disk persistence, NPC-mutating
    /// systems disabled. The session layer should follow up with
    /// [`Self::apply_network_snapshot`] once the host's snapshot
    /// arrives. Safe to call multiple times; subsequent calls are
    /// no-ops (matches `start(save_dir)`).
    #[func]
    fn start_mirror(&mut self) {
        if self.sim.is_some() {
            return;
        }
        let graph = RegionGraph::default_test_graph();
        let mut sim = Sim::new_mirror(graph);
        let los = Arc::new(GodotLosProvider::new());
        sim.install_los_provider(los.clone());
        self.los = Some(los);
        self.sim = Some(sim);
        self.base_mut().emit_signal("sim_ready", &[]);
    }

    /// Apply a host-sent snapshot to the mirror sim. `payload` is
    /// bincoded `SnapshotBody`. Emits `snapshot_applied(tick)` so the
    /// session layer can transition out of its loading state.
    #[func]
    fn apply_network_snapshot(&mut self, tick: i64, payload: PackedByteArray) -> bool {
        let Some(sim) = self.sim.as_mut() else {
            return false;
        };
        let bytes: Vec<u8> = payload.to_vec();
        let body = match bincode::deserialize::<simn_sim::SnapshotBody>(&bytes) {
            Ok(b) => b,
            Err(e) => {
                godot_error!("snapshot decode failed: {e:?}");
                return false;
            }
        };
        #[allow(clippy::cast_sign_loss)]
        let t = tick as u64;
        sim.apply_external_snapshot(body, t);
        self.base_mut()
            .emit_signal("snapshot_applied", &[tick.to_variant()]);
        true
    }

    /// Apply a host-sent delta batch. `payload` is bincoded
    /// `Vec<WorldDelta>`; each is applied in order, then the mirror's
    /// clock is anchored to `tick`.
    #[func]
    fn apply_network_delta_batch(&mut self, tick: i64, payload: PackedByteArray) -> bool {
        let bytes: Vec<u8> = payload.to_vec();
        let deltas = match bincode::deserialize::<Vec<WorldDelta>>(&bytes) {
            Ok(d) => d,
            Err(e) => {
                godot_error!("delta batch decode failed: {e:?}");
                return false;
            }
        };
        {
            let Some(sim) = self.sim.as_mut() else {
                return false;
            };
            for d in &deltas {
                sim.apply_external_delta(d);
            }
            #[allow(clippy::cast_sign_loss)]
            sim.set_tick_for_mirror(tick as u64);
        }
        // Fire local FX signals so mirror clients see the same
        // `projectile_spawned` / `projectile_impacted` events auth
        // sims emit on their drain path.
        self.emit_projectile_fx_signals(&deltas);
        true
    }

    /// Host-side: decode a client-sent `ActionKind` and dispatch into
    /// the authoritative sim. Returns `false` on unknown action or
    /// failed dispatch (error is already logged).
    #[func]
    fn dispatch_network_action(&mut self, acting_steam_id: i64, payload: PackedByteArray) -> bool {
        let bytes: Vec<u8> = payload.to_vec();
        let action = match simn_sim::decode_action(&bytes) {
            Some(a) => a,
            None => {
                godot_error!("action decode failed");
                return false;
            }
        };
        #[allow(clippy::cast_sign_loss)]
        let sid = acting_steam_id as u64;
        // Worker mode: send through the typed command channel
        // so the action applies at the top of the worker's next
        // tick — same path client-originated actions take when
        // the host forwards them.
        if let Some(worker) = self.worker.as_ref() {
            return match worker.send(simn_sim::worker::SimCommand::Action {
                steam_id: sid,
                kind: action,
            }) {
                Ok(()) => true,
                Err(e) => {
                    godot_error!("action dispatch (worker): {e:?}");
                    false
                }
            };
        }
        let Some(sim) = self.sim.as_mut() else {
            return false;
        };
        match sim.apply_action(sid, action) {
            Ok(()) => true,
            Err(e) => {
                godot_error!("action dispatch failed: {e:#}");
                false
            }
        }
    }

    /// Host-side: serialize the current sim state to a snapshot blob
    /// ready for `NetworkManager.send_snapshot` / `broadcast_snapshot`.
    /// Returns an empty dict when the sim isn't initialized.
    #[func]
    fn serialize_snapshot_payload(&mut self) -> Dictionary<Variant, Variant> {
        let mut d: Dictionary<Variant, Variant> = Dictionary::new();
        let Some(sim) = self.sim.as_mut() else {
            return d;
        };
        let body: SnapshotBody = sim.serialize_snapshot_body();
        let Ok(bytes) = bincode::serialize::<SnapshotBody>(&body) else {
            return d;
        };
        let payload = PackedByteArray::from(bytes.as_slice());
        #[allow(clippy::cast_possible_wrap)]
        let tick = sim.current_tick() as i64;
        d.set(&Variant::from("tick"), &Variant::from(tick));
        d.set(&Variant::from("payload"), &Variant::from(payload));
        d
    }
}

impl SimHost {
    /// Fire the `projectile_spawned` and `projectile_impacted`
    /// signals for any matching deltas in the slice. Used by both
    /// the local tick-drain path (auth + solo) and the
    /// `apply_network_delta_batch` path (mirror clients), so FX
    /// listeners see a consistent event stream regardless of where
    /// the projectile physics ran.
    fn emit_projectile_fx_signals(&mut self, deltas: &[WorldDelta]) {
        for delta in deltas {
            match delta {
                WorldDelta::ProjectileSpawned {
                    id,
                    source_steam_id,
                    source_npc_id,
                    round_id,
                    variant,
                    origin,
                    velocity,
                    max_range_m,
                    spawned_tick,
                } => {
                    let d = projectile_spawned_to_dict(
                        *id,
                        *source_steam_id,
                        *source_npc_id,
                        round_id,
                        *variant,
                        *origin,
                        *velocity,
                        *max_range_m,
                        *spawned_tick,
                    );
                    self.base_mut()
                        .emit_signal("projectile_spawned", &[d.to_variant()]);
                }
                WorldDelta::ProjectileImpacted {
                    id,
                    pos,
                    hit_npc,
                    hit_player_steam_id,
                    body_part,
                    damage_applied,
                    penetrated,
                } => {
                    let d = projectile_impacted_to_dict(
                        *id,
                        *pos,
                        *hit_npc,
                        *hit_player_steam_id,
                        *body_part,
                        *damage_applied,
                        *penetrated,
                    );
                    self.base_mut()
                        .emit_signal("projectile_impacted", &[d.to_variant()]);
                }
                _ => {}
            }
        }
    }

    /// Whether the local sim is authoritative (solo or host). Reads
    /// the attached `NetworkManager` if present; defaults to `true`
    /// (solo) otherwise.
    #[allow(dead_code)]
    fn is_authoritative(&self) -> bool {
        match &self.network {
            Some(nm) => nm.bind().is_authoritative(),
            None => true,
        }
    }

    /// Phase 2G: emit `view_updated` if the worker has published a
    /// new view since the last frame we checked. Throttled to fire
    /// at most every `VIEW_UPDATED_TICK_INTERVAL` sim ticks (5 ticks
    /// = 4 Hz at the 20 Hz sim rate). UI listeners throttle their
    /// own refreshes to similar rates; firing on every tick (20 Hz)
    /// burned signal-cascade time for no perceptible benefit.
    fn maybe_emit_view_updated(&mut self) {
        const VIEW_UPDATED_TICK_INTERVAL: u64 = 5;
        let Some(worker) = self.worker.as_ref() else {
            return;
        };
        let Some(view) = worker.view() else {
            return;
        };
        let tick = view.tick;
        if tick == self.last_view_tick_emitted {
            return;
        }
        // Skip if we're not at least N ticks past the last emit. The
        // first emit (last==0) always fires so initial UIs get a
        // refresh as soon as the worker boots.
        if self.last_view_tick_emitted != 0
            && tick.saturating_sub(self.last_view_tick_emitted) < VIEW_UPDATED_TICK_INTERVAL
        {
            return;
        }
        self.last_view_tick_emitted = tick;
        #[allow(clippy::cast_possible_wrap)]
        let t = tick as i64;
        self.base_mut()
            .emit_signal("view_updated", &[t.to_variant()]);
    }

    /// Whether the local sim is specifically the host (authoritative
    /// AND there's at least one potential peer — i.e. a lobby exists).
    /// In slice 1 this is identical to "role == host"; used to gate
    /// the `tick_completed` broadcast so solo sessions don't pay the
    /// serialization cost.
    fn is_host(&self) -> bool {
        matches!(
            self.network.as_ref().map(|nm| nm.bind().current_role()),
            Some(simn_net::NetRole::Host)
        )
    }

    /// Whether the local sim is a client mirror (mutations forward to
    /// host via network actions).
    fn is_client(&self) -> bool {
        matches!(
            self.network.as_ref().map(|nm| nm.bind().current_role()),
            Some(simn_net::NetRole::Client { .. })
        )
    }

    /// Encode an action and emit the `action_requested` signal so the
    /// session layer can forward it to the host via `NetworkManager`.
    /// Used by every mutating `#[func]` on the client path.
    fn emit_action(&mut self, acting_steam_id: i64, action: ActionKind) {
        let bytes = match simn_sim::encode_action(&action) {
            Ok(b) => b,
            Err(e) => {
                godot_error!("action encode failed: {e:?}");
                return;
            }
        };
        let payload = PackedByteArray::from(bytes.as_slice());
        self.base_mut().emit_signal(
            "action_requested",
            &[acting_steam_id.to_variant(), payload.to_variant()],
        );
    }
}

/// Iteration 5-13 Phase B2: decode a Godot `Array<Dictionary>` of
/// nav-obstacle markers (`{ pos: Vector3, extents: Vector3, kind:
/// String }`) into a `Vec<simn_sim::nav::NavObstacle>` ready to
/// hand to `Sim::attach_region_terrain_with_obstacles`.
///
/// Unknown `kind` strings default to `"block"` with a single
/// `godot_warn!` per call. Dictionaries missing `pos` or `extents`
/// are skipped with a warn. Y on both vectors is ignored — the
/// sim's nav grid is 2D.
fn parse_nav_obstacles(
    obstacles: &Array<Dictionary<Variant, Variant>>,
) -> Vec<simn_sim::nav::NavObstacle> {
    let mut out = Vec::with_capacity(obstacles.len());
    let mut unknown_kind_warned = false;
    for entry in obstacles.iter_shared() {
        let Some(pos_v) = entry.get(&Variant::from("pos")) else {
            godot_warn!("parse_nav_obstacles: entry missing `pos`; skipped");
            continue;
        };
        let Some(extents_v) = entry.get(&Variant::from("extents")) else {
            godot_warn!("parse_nav_obstacles: entry missing `extents`; skipped");
            continue;
        };
        let kind_str: Option<String> = entry
            .get(&Variant::from("kind"))
            .and_then(|v| v.try_to::<GString>().ok())
            .map(|g| g.to_string());
        let Ok(pos) = pos_v.try_to::<Vector3>() else {
            godot_warn!("parse_nav_obstacles: `pos` is not a Vector3; skipped");
            continue;
        };
        let Ok(extents) = extents_v.try_to::<Vector3>() else {
            godot_warn!("parse_nav_obstacles: `extents` is not a Vector3; skipped");
            continue;
        };
        let kind = match kind_str.as_deref() {
            Some("block") | None => simn_sim::NavOverride::ForceBlocked,
            Some("walkable") => simn_sim::NavOverride::ForceWalkable,
            Some(other) => {
                if !unknown_kind_warned {
                    godot_warn!(
                        "parse_nav_obstacles: unknown kind {other:?}; treating as `block`. \
                         Further warnings suppressed."
                    );
                    unknown_kind_warned = true;
                }
                simn_sim::NavOverride::ForceBlocked
            }
        };
        out.push(simn_sim::nav::NavObstacle {
            center: [pos.x, pos.z],
            extents: [extents.x.abs(), extents.z.abs()],
            kind,
        });
    }
    out
}

/// Iteration 5-13 Phase D2: decode a Godot `Array<Dictionary>` of
/// interaction-area markers into a
/// `Vec<simn_sim::resources::InteractionArea>` ready to hand to
/// `Sim::attach_region_interaction_areas`.
///
/// Faction strings unknown to the registry resolve to `None`
/// (treated as "any faction") with a single warn per call.
/// Missing `id` / empty string → auto-derived `auto:<region>:<x>_<z>`
/// keyed on integer-rounded XZ so multiple unnamed markers on the
/// same tile don't collide silently.
fn parse_interaction_areas(
    areas: &Array<Dictionary<Variant, Variant>>,
    registry: &simn_sim::FactionRegistry,
    region_name: &str,
) -> Vec<simn_sim::resources::InteractionArea> {
    let mut out = Vec::with_capacity(areas.len());
    let mut unknown_faction_warned = false;
    let mut missing_field_warned = false;
    for entry in areas.iter_shared() {
        let Some(pos_v) = entry.get(&Variant::from("pos")) else {
            if !missing_field_warned {
                godot_warn!("parse_interaction_areas: entry missing `pos`; skipped");
                missing_field_warned = true;
            }
            continue;
        };
        let Some(extents_v) = entry.get(&Variant::from("extents")) else {
            if !missing_field_warned {
                godot_warn!("parse_interaction_areas: entry missing `extents`; skipped");
                missing_field_warned = true;
            }
            continue;
        };
        let Ok(pos) = pos_v.try_to::<Vector3>() else {
            godot_warn!("parse_interaction_areas: `pos` is not a Vector3; skipped");
            continue;
        };
        let Ok(extents) = extents_v.try_to::<Vector3>() else {
            godot_warn!("parse_interaction_areas: `extents` is not a Vector3; skipped");
            continue;
        };
        let kind = entry
            .get(&Variant::from("kind"))
            .and_then(|v| v.try_to::<GString>().ok())
            .map(|g| g.to_string())
            .unwrap_or_else(|| "rest".to_string());
        let id_authored = entry
            .get(&Variant::from("id"))
            .and_then(|v| v.try_to::<GString>().ok())
            .map(|g| g.to_string())
            .unwrap_or_default();
        let id = if id_authored.trim().is_empty() {
            format!(
                "auto:{}:{}_{}",
                region_name,
                pos.x.round() as i32,
                pos.z.round() as i32,
            )
        } else {
            id_authored
        };
        let capacity = entry
            .get(&Variant::from("capacity"))
            .and_then(|v| v.try_to::<i64>().ok())
            .map(|n| n.max(1) as u32)
            .unwrap_or(1);
        let faction_name = entry
            .get(&Variant::from("faction"))
            .and_then(|v| v.try_to::<GString>().ok())
            .map(|g| g.to_string())
            .unwrap_or_default();
        let faction = if faction_name.trim().is_empty() {
            None
        } else {
            let resolved = registry.id_of(faction_name.trim());
            if resolved.is_none() && !unknown_faction_warned {
                godot_warn!(
                    "parse_interaction_areas: unknown faction {:?} on area {:?}; \
                     treating as any-faction. Further warnings suppressed.",
                    faction_name,
                    id,
                );
                unknown_faction_warned = true;
            }
            resolved
        };
        // tags: convert each (StringName/String → Variant) entry to (String → String).
        let mut tags: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        if let Some(tags_v) = entry.get(&Variant::from("tags")) {
            if let Ok(tags_dict) = tags_v.try_to::<Dictionary<Variant, Variant>>() {
                for (k, v) in tags_dict.iter_shared() {
                    let key = k
                        .try_to::<GString>()
                        .map(|g| g.to_string())
                        .unwrap_or_else(|_| k.to_string());
                    let val = v
                        .try_to::<GString>()
                        .map(|g| g.to_string())
                        .unwrap_or_else(|_| v.to_string());
                    tags.insert(key, val);
                }
            }
        }
        out.push(simn_sim::resources::InteractionArea {
            id,
            kind,
            pos: [pos.x, pos.y, pos.z],
            extents: [extents.x.abs(), extents.z.abs()],
            faction,
            capacity,
            occupants: 0,
            tags,
        });
    }
    out
}
