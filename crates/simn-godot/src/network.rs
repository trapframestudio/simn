//! Godot bridge for [`simn_net::NetSession`].
//!
//! `NetworkManager` is a `Node` exposed to GDScript. It lazily initializes
//! Steam on the first `host_session` / `join_session` call, pumps the
//! session each frame, and translates [`simn_net::NetEvent`] values into
//! Godot signals that GDScript code can connect to.
//!
//! All Steam calls happen on Godot's main thread. `#[func]` methods are
//! panic-free: errors are logged via `godot_error!` and surfaced through
//! the `error` signal.

use godot::classes::{INode, Node};
use godot::prelude::*;
use simn_net::{protocol::Msg, NetEvent, NetRole, NetSession};

#[derive(GodotClass)]
#[class(init, base=Node)]
pub struct NetworkManager {
    session: Option<NetSession>,
    base: Base<Node>,
}

#[godot_api]
impl INode for NetworkManager {
    fn process(&mut self, _delta: f64) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        let events = session.tick();
        for ev in events {
            self.dispatch(ev);
        }
    }
}

#[godot_api]
impl NetworkManager {
    #[signal]
    fn peer_joined(steam_id: i64);

    #[signal]
    fn peer_left(steam_id: i64);

    #[signal]
    fn peer_state(steam_id: i64, map_id: GString, pos: Vector3, yaw: f32);

    #[signal]
    fn lobby_ready(lobby_id: i64);

    #[signal]
    fn join_requested(lobby_id: i64);

    #[signal]
    fn network_error(message: GString);

    /// Host received a `JoinRequest` from a newly-connected peer.
    /// GDScript handler should serialize the current snapshot and
    /// call `send_snapshot(peer_steam_id, ...)`.
    #[signal]
    fn snapshot_requested(peer_steam_id: i64);

    /// Client received a snapshot from host. Payload is bincoded
    /// `SnapshotBody`; GDScript forwards to `SimHost.apply_network_snapshot`.
    #[signal]
    fn snapshot_received(tick: i64, payload: PackedByteArray);

    /// Host received a delta batch or client received one from host.
    /// Payload is bincoded `Vec<WorldDelta>`.
    #[signal]
    fn delta_received(tick: i64, payload: PackedByteArray);

    /// Host received an action from a peer. Payload is bincoded
    /// `ActionKind`. Host GDScript forwards to
    /// `SimHost.dispatch_network_action`.
    #[signal]
    fn action_received(peer_steam_id: i64, steam_id: i64, payload: PackedByteArray);

    /// Create a Steam lobby and become host. Emits `lobby_ready` when
    /// the lobby handle is available; at that point the host can call
    /// `open_invite_overlay` to invite friends.
    #[func]
    fn host_session(&mut self) {
        if let Err(e) = self.ensure_session() {
            self.emit_error(format!("Steam init failed: {e}"));
            return;
        }
        self.session.as_mut().expect("session present").host();
    }

    /// Join an existing lobby by id. Typically called in response to a
    /// `join_requested` signal triggered by a Steam overlay invite.
    #[func]
    fn join_session(&mut self, lobby_id: i64) {
        if let Err(e) = self.ensure_session() {
            self.emit_error(format!("Steam init failed: {e}"));
            return;
        }
        #[allow(clippy::cast_sign_loss)]
        let id = lobby_id as u64;
        self.session.as_mut().expect("session present").join(id);
    }

    /// Open the Steam friends overlay so the host can invite someone to
    /// the current lobby. No-op if no lobby is active yet.
    #[func]
    fn open_invite_overlay(&mut self) {
        if let Some(session) = self.session.as_ref() {
            session.open_invite_overlay();
        }
    }

    /// Publish the local player's map + transform. The network layer
    /// rate-limits this to 20Hz; call it every physics tick.
    #[func]
    fn publish_state(&mut self, map_id: GString, pos: Vector3, yaw: f32) {
        if let Some(session) = self.session.as_mut() {
            session.publish_local_state(map_id.to_string(), [pos.x, pos.y, pos.z], yaw);
        }
    }

    /// The local user's Steam ID as i64. Returns 0 if Steam isn't initialized.
    #[func]
    fn local_steam_id(&self) -> i64 {
        #[allow(clippy::cast_possible_wrap)]
        self.session
            .as_ref()
            .map(|s| s.local_steam_id() as i64)
            .unwrap_or(0)
    }

    // ---------- Role + replication (slice-1) ----------

    /// Stable string tag for the local role: `"solo" | "host" | "client"`.
    /// GDScript uses this to gate UI (e.g. "join overlay" only on host)
    /// and SimHost uses it to gate mutation dispatch (clients send
    /// actions instead of mutating locally).
    #[func]
    pub fn role(&self) -> GString {
        GString::from(
            self.session
                .as_ref()
                .map(|s| s.role().as_str())
                .unwrap_or("solo"),
        )
    }

    /// Host's steam id when this peer is in `Client` role; 0 otherwise.
    #[func]
    pub fn host_steam_id(&self) -> i64 {
        #[allow(clippy::cast_possible_wrap)]
        self.session
            .as_ref()
            .map(|s| s.role().host_steam_id() as i64)
            .unwrap_or(0)
    }

    /// True when the local sim is authoritative (solo or host).
    #[func]
    pub fn is_authoritative(&self) -> bool {
        self.session
            .as_ref()
            .is_none_or(|s| s.role().is_authoritative())
    }

    /// Host broadcast: send a snapshot payload to every peer in the
    /// lobby. Reliable. Called on a `snapshot_requested` signal or
    /// immediately after a host tick. `tick` is stamped onto the
    /// message so clients can anchor their sim clock.
    #[func]
    fn broadcast_snapshot(&mut self, tick: i64, payload: PackedByteArray) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let msg = Msg::Snapshot {
            tick: tick as u64,
            payload: payload.to_vec(),
        };
        session.broadcast(&msg);
    }

    /// Host broadcast: per-tick `WorldDelta` batch. Reliable, ordered.
    #[func]
    fn broadcast_delta(&mut self, tick: i64, payload: PackedByteArray) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let msg = Msg::Delta {
            tick: tick as u64,
            payload: payload.to_vec(),
        };
        session.broadcast(&msg);
    }

    /// Host direct-send: snapshot to a specific peer. Used when a
    /// newly-joined peer sends `JoinRequest` and the host responds
    /// with one snapshot rather than broadcasting to everyone.
    #[func]
    fn send_snapshot(&mut self, peer_steam_id: i64, tick: i64, payload: PackedByteArray) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let msg = Msg::Snapshot {
            tick: tick as u64,
            payload: payload.to_vec(),
        };
        #[allow(clippy::cast_sign_loss)]
        session.send_to(peer_steam_id as u64, &msg);
    }

    /// Client → host: send an action payload (bincoded `ActionKind`).
    /// `acting_steam_id` is who the action applies to; in slice 1 this
    /// equals `local_steam_id`.
    #[func]
    fn send_action(&mut self, acting_steam_id: i64, payload: PackedByteArray) {
        let Some(session) = self.session.as_ref() else {
            return;
        };
        let NetRole::Client { host_steam_id } = session.role() else {
            // Not a client; mutation path takes the direct route instead.
            return;
        };
        #[allow(clippy::cast_sign_loss)]
        let msg = Msg::Action {
            steam_id: acting_steam_id as u64,
            payload: payload.to_vec(),
        };
        session.send_to(host_steam_id, &msg);
    }

    /// Client → host: request an initial snapshot after joining a lobby.
    /// No-op when not in `Client` role.
    #[func]
    fn send_join_request(&mut self) -> bool {
        let Some(session) = self.session.as_ref() else {
            return false;
        };
        session.send_join_request_to_host()
    }
}

impl NetworkManager {
    /// Rust-side role accessor (avoids the GString roundtrip the
    /// `#[func] role()` uses). Returns `NetRole::Solo` when no session
    /// has initialized yet. Used by `SimHost` for its mutation gate.
    pub fn current_role(&self) -> NetRole {
        self.session.as_ref().map_or(NetRole::Solo, |s| s.role())
    }

    fn ensure_session(&mut self) -> Result<(), String> {
        if self.session.is_some() {
            return Ok(());
        }
        match NetSession::init() {
            Ok(session) => {
                self.session = Some(session);
                Ok(())
            }
            Err(e) => Err(format!("{e:?}")),
        }
    }

    fn dispatch(&mut self, ev: NetEvent) {
        match ev {
            NetEvent::LobbyReady { lobby_id } => {
                if let Some(s) = self.session.as_mut() {
                    s.set_active_lobby(lobby_id);
                }
                #[allow(clippy::cast_possible_wrap)]
                self.base_mut()
                    .emit_signal("lobby_ready", &[(lobby_id as i64).to_variant()]);
            }
            NetEvent::JoinRequested { lobby_id } => {
                #[allow(clippy::cast_possible_wrap)]
                self.base_mut()
                    .emit_signal("join_requested", &[(lobby_id as i64).to_variant()]);
            }
            NetEvent::PeerJoined { steam_id } => {
                #[allow(clippy::cast_possible_wrap)]
                self.base_mut()
                    .emit_signal("peer_joined", &[(steam_id as i64).to_variant()]);
            }
            NetEvent::PeerLeft { steam_id } => {
                #[allow(clippy::cast_possible_wrap)]
                self.base_mut()
                    .emit_signal("peer_left", &[(steam_id as i64).to_variant()]);
            }
            NetEvent::PeerState {
                steam_id,
                map_id,
                pos,
                yaw,
            } => {
                #[allow(clippy::cast_possible_wrap)]
                let sid = steam_id as i64;
                let map = GString::from(&map_id);
                let p = Vector3::new(pos[0], pos[1], pos[2]);
                self.base_mut().emit_signal(
                    "peer_state",
                    &[
                        sid.to_variant(),
                        map.to_variant(),
                        p.to_variant(),
                        yaw.to_variant(),
                    ],
                );
            }
            NetEvent::SnapshotRequested { peer_steam_id } => {
                #[allow(clippy::cast_possible_wrap)]
                self.base_mut()
                    .emit_signal("snapshot_requested", &[(peer_steam_id as i64).to_variant()]);
            }
            NetEvent::SnapshotReceived { tick, payload } => {
                #[allow(clippy::cast_possible_wrap)]
                let t = tick as i64;
                let bytes = PackedByteArray::from(payload.as_slice());
                self.base_mut()
                    .emit_signal("snapshot_received", &[t.to_variant(), bytes.to_variant()]);
            }
            NetEvent::DeltaReceived { tick, payload } => {
                #[allow(clippy::cast_possible_wrap)]
                let t = tick as i64;
                let bytes = PackedByteArray::from(payload.as_slice());
                self.base_mut()
                    .emit_signal("delta_received", &[t.to_variant(), bytes.to_variant()]);
            }
            NetEvent::ActionReceived {
                peer_steam_id,
                steam_id,
                payload,
            } => {
                #[allow(clippy::cast_possible_wrap)]
                let peer = peer_steam_id as i64;
                #[allow(clippy::cast_possible_wrap)]
                let sid = steam_id as i64;
                let bytes = PackedByteArray::from(payload.as_slice());
                self.base_mut().emit_signal(
                    "action_received",
                    &[peer.to_variant(), sid.to_variant(), bytes.to_variant()],
                );
            }
            NetEvent::Error { msg } => {
                godot_error!("{msg}");
                self.emit_error(msg);
            }
        }
    }

    fn emit_error(&mut self, msg: String) {
        let g = GString::from(&msg);
        self.base_mut()
            .emit_signal("network_error", &[g.to_variant()]);
    }
}
