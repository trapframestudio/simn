//! Listen-server P2P session over Steam Networking.
//!
//! A [`NetSession`] owns the Steam client, an optional lobby handle, and
//! a table of remote peers. All Steam calls run on the thread that
//! constructed the session (the Godot main thread in production).
//! [`NetSession::tick`] must be called every frame; it pumps Steam
//! callbacks, drains incoming P2P packets, broadcasts local state, and
//! returns observable events for the engine layer to act on.
//!
//! **Role model.** Every session starts in [`NetRole::Solo`]. Calling
//! [`NetSession::host`] flips to [`NetRole::Host`]; [`NetSession::join`]
//! flips to [`NetRole::Client`]. The role determines whether the sim
//! layer above treats itself as authoritative (Host/Solo) or as a
//! mirror consuming host deltas (Client). See
//! `docs/book/src/architecture/networking.md`.

use crate::protocol::{decode, encode, Msg, Reliability};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};
use steamworks::{
    CallbackHandle, Client, ClientManager, LobbyId, LobbyType, SendType, SingleClient, SteamId,
};

/// Authority role for the local peer. See module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetRole {
    /// Single-player: no lobby, sim is trivially authoritative.
    Solo,
    /// Listen-server: this peer's sim is authoritative for all peers.
    Host,
    /// Joined someone else's lobby; mirror sim consumes host deltas.
    Client { host_steam_id: u64 },
}

impl NetRole {
    /// Whether this role runs the authoritative sim (solo or host).
    pub fn is_authoritative(&self) -> bool {
        matches!(self, NetRole::Solo | NetRole::Host)
    }

    /// Stable string tag for the role, for logging and the GDScript
    /// bridge.
    pub fn as_str(&self) -> &'static str {
        match self {
            NetRole::Solo => "solo",
            NetRole::Host => "host",
            NetRole::Client { .. } => "client",
        }
    }

    /// Host's steam id when this is a client role; 0 otherwise.
    pub fn host_steam_id(&self) -> u64 {
        match self {
            NetRole::Client { host_steam_id } => *host_steam_id,
            _ => 0,
        }
    }
}

/// Events raised by the session. The Godot layer translates these into
/// signals.
#[derive(Debug, Clone)]
pub enum NetEvent {
    /// Lobby was created (host path) or entered (join path).
    LobbyReady { lobby_id: u64 },
    /// Steam overlay requested that we join a lobby. The engine layer
    /// should call [`NetSession::join`] with this id.
    JoinRequested { lobby_id: u64 },
    /// A peer joined the lobby.
    PeerJoined { steam_id: u64 },
    /// A peer left the lobby.
    PeerLeft { steam_id: u64 },
    /// A peer published new state (legacy unreliable transform path).
    PeerState {
        steam_id: u64,
        map_id: String,
        pos: [f32; 3],
        yaw: f32,
    },
    /// Host received a `Msg::JoinRequest` from a newly-connected peer.
    /// Should respond with a snapshot send.
    SnapshotRequested { peer_steam_id: u64 },
    /// Client received a snapshot from the host.
    SnapshotReceived { tick: u64, payload: Vec<u8> },
    /// Client received a delta batch from the host.
    DeltaReceived { tick: u64, payload: Vec<u8> },
    /// Host received an action from a client.
    ActionReceived {
        peer_steam_id: u64,
        steam_id: u64,
        payload: Vec<u8>,
    },
    /// Non-fatal error that the engine layer should surface.
    Error { msg: String },
}

type EventSink = Arc<Mutex<Vec<NetEvent>>>;

#[derive(Clone, Default)]
struct LocalState {
    map_id: String,
    pos: [f32; 3],
    yaw: f32,
    dirty: bool,
}

/// Broadcast rate: 20Hz.
const TICK_INTERVAL: Duration = Duration::from_millis(50);

/// Maximum lobby size for slice 1. Raised to 12 in slice 2 after
/// profiling delta volume under load.
pub const MAX_LOBBY_MEMBERS: u32 = 4;

/// The top-level session. Owns the Steam client, the callback pump, the
/// current lobby handle, and the peer table.
pub struct NetSession {
    client: Client,
    single: SingleClient,
    events: EventSink,
    _callbacks: Vec<CallbackHandle>,

    role: NetRole,
    lobby: Option<LobbyId>,
    local_state: LocalState,
    last_broadcast: Instant,
    peers: Vec<SteamId>,
}

impl NetSession {
    /// Initialize Steam and return a new session. Reads `steam_appid.txt`
    /// from the current working directory to determine which app to
    /// connect as.
    pub fn init() -> anyhow::Result<Self> {
        let (client, single) =
            Client::init().map_err(|e| anyhow::anyhow!("Steam init failed: {e:?}"))?;
        let events: EventSink = Arc::new(Mutex::new(Vec::new()));
        let mut callbacks: Vec<CallbackHandle> = Vec::new();

        // Lobby chat update (member join/leave).
        {
            let events = events.clone();
            let h = client.register_callback(move |c: steamworks::LobbyChatUpdate| {
                let mut evs = events.lock();
                let state = c.member_state_change as u32;
                if state & 1 != 0 {
                    evs.push(NetEvent::PeerJoined {
                        steam_id: c.user_changed.raw(),
                    });
                } else if state & (2 | 4 | 8 | 16) != 0 {
                    evs.push(NetEvent::PeerLeft {
                        steam_id: c.user_changed.raw(),
                    });
                }
            });
            callbacks.push(h);
        }

        // P2P session request — auto-accept (trust lobby membership).
        {
            let client_inner = client.clone();
            let h = client.register_callback(move |c: steamworks::P2PSessionRequest| {
                client_inner.networking().accept_p2p_session(c.remote);
            });
            callbacks.push(h);
        }

        // Steam overlay / invite → join request.
        {
            let events = events.clone();
            let h = client.register_callback(move |c: steamworks::GameLobbyJoinRequested| {
                let mut evs = events.lock();
                evs.push(NetEvent::JoinRequested {
                    lobby_id: c.lobby_steam_id.raw(),
                });
            });
            callbacks.push(h);
        }

        Ok(Self {
            client,
            single,
            events,
            _callbacks: callbacks,
            role: NetRole::Solo,
            lobby: None,
            local_state: LocalState::default(),
            last_broadcast: Instant::now(),
            peers: Vec::new(),
        })
    }

    /// The local user's Steam ID.
    pub fn local_steam_id(&self) -> u64 {
        self.client.user().steam_id().raw()
    }

    /// Current [`NetRole`]. `Solo` until a host/join call lands.
    pub fn role(&self) -> NetRole {
        self.role
    }

    /// Create a new friends-only lobby. Emits [`NetEvent::LobbyReady`]
    /// when the async call completes. Flips role to [`NetRole::Host`]
    /// immediately so the sim layer can switch to authoritative tick
    /// before the lobby completes creation.
    pub fn host(&mut self) {
        self.role = NetRole::Host;
        let events = self.events.clone();
        self.client.matchmaking().create_lobby(
            LobbyType::FriendsOnly,
            MAX_LOBBY_MEMBERS,
            move |res| match res {
                Ok(id) => events
                    .lock()
                    .push(NetEvent::LobbyReady { lobby_id: id.raw() }),
                Err(e) => events.lock().push(NetEvent::Error {
                    msg: format!("create_lobby failed: {e:?}"),
                }),
            },
        );
    }

    /// Join an existing lobby by id. Flips role to [`NetRole::Client`]
    /// once the lobby membership resolves (host steam id is read from
    /// the lobby owner). Until then, stays in `Solo` so the sim layer
    /// can't mis-route actions to a non-existent host.
    pub fn join(&mut self, lobby_id: u64) {
        let events = self.events.clone();
        self.client
            .matchmaking()
            .join_lobby(LobbyId::from_raw(lobby_id), move |res| match res {
                Ok(id) => events
                    .lock()
                    .push(NetEvent::LobbyReady { lobby_id: id.raw() }),
                Err(()) => events.lock().push(NetEvent::Error {
                    msg: "join_lobby failed".into(),
                }),
            });
    }

    /// Remember the lobby we're in and refresh the peer list. Call after
    /// a `LobbyReady` event from host or join. For joiners, reads the
    /// lobby owner from Steam and promotes the session to
    /// [`NetRole::Client`] with that owner as the host.
    pub fn set_active_lobby(&mut self, lobby_id: u64) {
        let id = LobbyId::from_raw(lobby_id);
        self.lobby = Some(id);
        self.refresh_peers();

        // If we joined someone else's lobby, the owner is our host.
        // `host()` already flipped to Host locally, so only promote to
        // Client when the owner is a different peer.
        let owner = self.client.matchmaking().lobby_owner(id).raw();
        if owner != 0 && owner != self.local_steam_id() {
            self.role = NetRole::Client {
                host_steam_id: owner,
            };
        }
    }

    /// Open the Steam overlay invite dialog for the current lobby.
    pub fn open_invite_overlay(&self) {
        if let Some(lobby) = self.lobby {
            self.client.friends().activate_invite_dialog(lobby);
        }
    }

    /// Publish the local player's state; broadcast on the next tick.
    pub fn publish_local_state(&mut self, map_id: String, pos: [f32; 3], yaw: f32) {
        self.local_state = LocalState {
            map_id,
            pos,
            yaw,
            dirty: true,
        };
    }

    /// Send a message to a specific peer. Picks Steam reliability class
    /// from [`Msg::reliability`]. Returns `false` on encode failure or
    /// unknown peer.
    pub fn send_to(&self, steam_id: u64, msg: &Msg) -> bool {
        let Ok(bytes) = encode(msg) else {
            return false;
        };
        let networking = self.client.networking();
        let send_type = steam_send_type(msg.reliability());
        networking.send_p2p_packet(SteamId::from_raw(steam_id), send_type, &bytes)
    }

    /// Broadcast a message to every peer in the current lobby except
    /// the local user. Reliability class chosen per [`Msg::reliability`].
    pub fn broadcast(&self, msg: &Msg) {
        let Ok(bytes) = encode(msg) else {
            return;
        };
        let networking = self.client.networking();
        let me = self.client.user().steam_id();
        let reliability = msg.reliability();
        for peer in &self.peers {
            if *peer == me {
                continue;
            }
            networking.send_p2p_packet(*peer, steam_send_type(reliability), &bytes);
        }
    }

    /// Convenience: send a `JoinRequest` to the host. No-op unless the
    /// session is in `Client` role.
    pub fn send_join_request_to_host(&self) -> bool {
        let NetRole::Client { host_steam_id } = self.role else {
            return false;
        };
        self.send_to(host_steam_id, &Msg::JoinRequest)
    }

    /// Pump Steam callbacks, drain incoming P2P packets, maybe broadcast
    /// local state. Returns all events observed since the last call.
    pub fn tick(&mut self) -> Vec<NetEvent> {
        self.single.run_callbacks();

        // Drain incoming P2P packets. All Msg variants are decoded here
        // and translated into NetEvents; sim-layer types (SnapshotBody,
        // WorldDelta, ActionKind) stay opaque at the network layer.
        let networking = self.client.networking();
        while let Some(size) = networking.is_p2p_packet_available() {
            let mut buf = vec![0u8; size];
            let Some((sender, read)) = networking.read_p2p_packet(&mut buf) else {
                break;
            };
            buf.truncate(read);
            let sender_id = sender.raw();
            if sender_id == self.client.user().steam_id().raw() {
                continue;
            }
            let Some(msg) = decode(&buf) else {
                continue;
            };
            match msg {
                Msg::State { map_id, pos, yaw } => {
                    self.events.lock().push(NetEvent::PeerState {
                        steam_id: sender_id,
                        map_id,
                        pos,
                        yaw,
                    });
                }
                Msg::Snapshot { tick, payload } => {
                    self.events
                        .lock()
                        .push(NetEvent::SnapshotReceived { tick, payload });
                }
                Msg::Delta { tick, payload } => {
                    self.events
                        .lock()
                        .push(NetEvent::DeltaReceived { tick, payload });
                }
                Msg::JoinRequest => {
                    self.events.lock().push(NetEvent::SnapshotRequested {
                        peer_steam_id: sender_id,
                    });
                }
                Msg::Action { steam_id, payload } => {
                    self.events.lock().push(NetEvent::ActionReceived {
                        peer_steam_id: sender_id,
                        steam_id,
                        payload,
                    });
                }
            }
        }

        if self.lobby.is_some() {
            self.refresh_peers();
        }

        if self.should_broadcast() {
            self.broadcast_local_state();
            self.last_broadcast = Instant::now();
        }

        std::mem::take(&mut *self.events.lock())
    }

    fn should_broadcast(&self) -> bool {
        self.lobby.is_some()
            && self.local_state.dirty
            && self.last_broadcast.elapsed() >= TICK_INTERVAL
    }

    fn broadcast_local_state(&mut self) {
        let msg = Msg::State {
            map_id: self.local_state.map_id.clone(),
            pos: self.local_state.pos,
            yaw: self.local_state.yaw,
        };
        self.broadcast(&msg);
        self.local_state.dirty = false;
    }

    fn refresh_peers(&mut self) {
        if let Some(lobby) = self.lobby {
            self.peers = self.client.matchmaking().lobby_members(lobby);
        }
    }
}

fn steam_send_type(r: Reliability) -> SendType {
    match r {
        Reliability::Unreliable => SendType::Unreliable,
        Reliability::Reliable => SendType::Reliable,
    }
}

// Keep `ClientManager` import reachable; reserved for future bound-generic extension traits.
#[allow(dead_code)]
fn _keep_import(_: std::marker::PhantomData<ClientManager>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_role_helpers() {
        assert!(NetRole::Solo.is_authoritative());
        assert!(NetRole::Host.is_authoritative());
        let client = NetRole::Client { host_steam_id: 123 };
        assert!(!client.is_authoritative());
        assert_eq!(client.host_steam_id(), 123);
        assert_eq!(NetRole::Solo.as_str(), "solo");
        assert_eq!(NetRole::Host.as_str(), "host");
        assert_eq!(client.as_str(), "client");
    }
}
