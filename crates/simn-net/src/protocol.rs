//! Wire protocol for peer-to-peer messages.
//!
//! Serialized with `bincode` and sent over Steam Networking Sockets.
//! Keep this enum **additive** and version-tolerant: unknown variants
//! on the receiver side should be ignored rather than erroring.
//!
//! Two reliability classes:
//! - `State` is sent **unreliable** at 20Hz for pills / remote-player
//!   lerp. Loss is fine — the next packet supersedes the previous.
//! - `Snapshot`, `Delta`, `JoinRequest`, `Action` are sent **reliable
//!   ordered**. These drive the host-authoritative sim replication
//!   path, and missing/reordered deltas would desync the mirror.
//!
//! `Snapshot` / `Delta` / `Action` payloads are opaque byte blobs
//! owned by the sim layer (bincoded `SnapshotBody` / `Vec<WorldDelta>`
//! / `ActionKind`). `simn-net` is deliberately ignorant of sim types —
//! keeping the network crate free of ECS / game deps.
//!
//! Message classes are exposed via [`Msg::reliability`] so the session
//! layer can pick the right Steam send type without pattern-matching
//! on the whole enum.

use serde::{Deserialize, Serialize};

/// Reliability class for a message. See [`Msg::reliability`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    /// Unordered, lossy; newest packet wins. Transform broadcasts.
    Unreliable,
    /// Ordered, retransmitted on loss. Replication control messages.
    Reliable,
}

/// Wire-format message exchanged between peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Msg {
    /// Peer's current transform and which map they're on. Unreliable;
    /// sent every tick. Used by the legacy pill-lerp path; replaced by
    /// host-broadcast `Delta` once the sim-mirror path is live, but
    /// kept on the wire during transition.
    State {
        map_id: String,
        pos: [f32; 3],
        yaw: f32,
    },

    /// Host → joining client: full world state for initial sync or
    /// resync. `payload` is a bincoded `simn_sim::SnapshotBody`.
    /// Reliable; may be large (single-KB to hundreds-KB depending on
    /// entity count).
    Snapshot { tick: u64, payload: Vec<u8> },

    /// Host → all clients: the `WorldDelta`s produced at `tick`.
    /// `payload` is a bincoded `Vec<simn_sim::WorldDelta>`. Reliable
    /// **ordered** — clients apply in order and slave their
    /// `SimClock.tick` to `tick` after applying.
    Delta { tick: u64, payload: Vec<u8> },

    /// Client → host: "I just joined; send me a snapshot." Triggers a
    /// `Snapshot` response. Reliable; sent once per join.
    JoinRequest,

    /// Client → host: "I want to apply this mutation." `payload` is a
    /// bincoded `simn_sim::ActionKind`. The host validates, dispatches
    /// to its local sim, and the resulting deltas broadcast to
    /// everyone (including the sender) as `Delta`. Reliable.
    ///
    /// `steam_id` is the *acting player's* steam id (who is eating /
    /// bandaging / moving / etc.), not necessarily the sender — in
    /// slice 1 they're always the same, but future-proofed for
    /// host-initiated actions on a client's behalf.
    Action { steam_id: u64, payload: Vec<u8> },
}

impl Msg {
    /// Reliability class for this message. Lets the session layer
    /// pick between `SendType::Unreliable` and `SendType::Reliable`
    /// without pattern-matching on every variant.
    pub fn reliability(&self) -> Reliability {
        match self {
            Msg::State { .. } => Reliability::Unreliable,
            Msg::Snapshot { .. } | Msg::Delta { .. } | Msg::JoinRequest | Msg::Action { .. } => {
                Reliability::Reliable
            }
        }
    }
}

/// Serialize a message to a byte buffer.
pub fn encode(msg: &Msg) -> anyhow::Result<Vec<u8>> {
    Ok(bincode::serialize(msg)?)
}

/// Deserialize a byte buffer into a message. Returns `None` if the
/// buffer doesn't decode into a known variant (forward-compat).
pub fn decode(bytes: &[u8]) -> Option<Msg> {
    bincode::deserialize(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_roundtrip() {
        let m = Msg::State {
            map_id: "map_a".into(),
            pos: [1.0, 2.0, 3.0],
            yaw: 0.5,
        };
        let bytes = encode(&m).unwrap();
        let decoded = decode(&bytes).expect("decode");
        match decoded {
            Msg::State { map_id, pos, yaw } => {
                assert_eq!(map_id, "map_a");
                assert_eq!(pos, [1.0, 2.0, 3.0]);
                assert!((yaw - 0.5).abs() < f32::EPSILON);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn snapshot_roundtrip() {
        let m = Msg::Snapshot {
            tick: 1234,
            payload: vec![1, 2, 3, 4, 5],
        };
        let bytes = encode(&m).unwrap();
        let decoded = decode(&bytes).expect("decode");
        match decoded {
            Msg::Snapshot { tick, payload } => {
                assert_eq!(tick, 1234);
                assert_eq!(payload, vec![1, 2, 3, 4, 5]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn delta_roundtrip() {
        let m = Msg::Delta {
            tick: 42,
            payload: vec![0xde, 0xad, 0xbe, 0xef],
        };
        let bytes = encode(&m).unwrap();
        match decode(&bytes).expect("decode") {
            Msg::Delta { tick, payload } => {
                assert_eq!(tick, 42);
                assert_eq!(payload, vec![0xde, 0xad, 0xbe, 0xef]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn join_request_roundtrip() {
        let bytes = encode(&Msg::JoinRequest).unwrap();
        assert!(matches!(decode(&bytes), Some(Msg::JoinRequest)));
    }

    #[test]
    fn action_roundtrip() {
        let m = Msg::Action {
            steam_id: 76561_1234,
            payload: vec![7, 7, 7],
        };
        let bytes = encode(&m).unwrap();
        match decode(&bytes).expect("decode") {
            Msg::Action { steam_id, payload } => {
                assert_eq!(steam_id, 76561_1234);
                assert_eq!(payload, vec![7, 7, 7]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn reliability_classes() {
        assert_eq!(
            Msg::State {
                map_id: String::new(),
                pos: [0.0; 3],
                yaw: 0.0
            }
            .reliability(),
            Reliability::Unreliable
        );
        assert_eq!(
            Msg::Snapshot {
                tick: 0,
                payload: vec![]
            }
            .reliability(),
            Reliability::Reliable
        );
        assert_eq!(
            Msg::Delta {
                tick: 0,
                payload: vec![]
            }
            .reliability(),
            Reliability::Reliable
        );
        assert_eq!(Msg::JoinRequest.reliability(), Reliability::Reliable);
        assert_eq!(
            Msg::Action {
                steam_id: 0,
                payload: vec![]
            }
            .reliability(),
            Reliability::Reliable
        );
    }

    #[test]
    fn decode_garbage_returns_none() {
        assert!(decode(&[0xff, 0xff, 0xff]).is_none());
    }
}
