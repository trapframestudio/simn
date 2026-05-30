//! Deterministic-serialize helpers.
//!
//! `HashMap` iteration order in Rust is not stable across instances
//! (each map's `RandomState` produces a different hash seed). For
//! state that bincode-serializes into snapshots / journal entries —
//! which the determinism harness checks for byte-for-byte
//! equivalence between same-seed sims — that means random snapshot
//! drift even when world state is logically identical.
//!
//! Two options for fixing it:
//! 1. Replace `HashMap` with `BTreeMap` everywhere serialized.
//! 2. Keep `HashMap` for runtime ergonomics, but sort entries by
//!    key on serialize. We pick this — runtime hot paths benefit
//!    from `HashMap`'s O(1) lookups; the sort cost only applies at
//!    snapshot time (every N seconds, off the tick path).
//!
//! Apply via `#[serde(serialize_with = "crate::det_serde::sorted_map")]`
//! on `HashMap` fields whose containing type is reachable from
//! `SnapshotBody` or any journal delta payload. Round-trip with
//! `serde`'s default `HashMap` deserialize is preserved — it accepts
//! entries in any order.

use std::collections::HashMap;

use serde::ser::{SerializeMap, Serializer};
use serde::Serialize;

/// Serialize a `HashMap<K, V>` with entries sorted by key. Stable
/// across runs because the comparison is a real `Ord` and not the
/// per-instance hash seed.
pub fn sorted_map<K, V, S>(map: &HashMap<K, V>, ser: S) -> Result<S::Ok, S::Error>
where
    K: Serialize + Ord,
    V: Serialize,
    S: Serializer,
{
    let mut entries: Vec<(&K, &V)> = map.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut m = ser.serialize_map(Some(entries.len()))?;
    for (k, v) in entries {
        m.serialize_entry(k, v)?;
    }
    m.end()
}

/// Serialize a nested `HashMap<K, HashMap<InnerK, V>>` with both
/// outer and inner entries sorted by key. Used by
/// `PopulationTargets::by_region`.
pub fn sorted_nested_map<K, InnerK, V, S>(
    map: &HashMap<K, HashMap<InnerK, V>>,
    ser: S,
) -> Result<S::Ok, S::Error>
where
    K: Serialize + Ord,
    InnerK: Serialize + Ord,
    V: Serialize,
    S: Serializer,
{
    let mut outer: Vec<(&K, &HashMap<InnerK, V>)> = map.iter().collect();
    outer.sort_by(|a, b| a.0.cmp(b.0));
    let mut m = ser.serialize_map(Some(outer.len()))?;
    for (k, inner) in outer {
        // Sort the inner HashMap into a `Vec` and serialize via a
        // wrapper struct so the nested sort happens too. Bincode
        // doesn't care about wrapper-type names; the serialized
        // bytes match a normal HashMap because we emit a map
        // through `SerializeMap`.
        let inner_sorted = SortedHashMap(inner);
        m.serialize_entry(k, &inner_sorted)?;
    }
    m.end()
}

/// Wrapper for `HashMap` that serializes entries in key-sorted order.
/// Internal helper for [`sorted_nested_map`].
struct SortedHashMap<'a, K, V>(&'a HashMap<K, V>);

impl<K, V> Serialize for SortedHashMap<'_, K, V>
where
    K: Serialize + Ord,
    V: Serialize,
{
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        sorted_map(self.0, ser)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorted_map_emits_keys_in_order() {
        // Two HashMaps with the same content but inserted in
        // different orders should serialize to identical bytes via
        // `sorted_map`.
        let mut a: HashMap<u32, String> = HashMap::new();
        a.insert(3, "three".into());
        a.insert(1, "one".into());
        a.insert(2, "two".into());

        let mut b: HashMap<u32, String> = HashMap::new();
        b.insert(2, "two".into());
        b.insert(3, "three".into());
        b.insert(1, "one".into());

        #[derive(Serialize)]
        struct Wrap<'a> {
            #[serde(serialize_with = "sorted_map")]
            map: &'a HashMap<u32, String>,
        }
        let bytes_a = bincode::serialize(&Wrap { map: &a }).unwrap();
        let bytes_b = bincode::serialize(&Wrap { map: &b }).unwrap();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn sorted_nested_map_sorts_outer_and_inner() {
        let mut a: HashMap<u32, HashMap<u8, u16>> = HashMap::new();
        let mut a_inner_3: HashMap<u8, u16> = HashMap::new();
        a_inner_3.insert(20, 200);
        a_inner_3.insert(10, 100);
        a.insert(3, a_inner_3);
        let mut a_inner_1: HashMap<u8, u16> = HashMap::new();
        a_inner_1.insert(5, 50);
        a.insert(1, a_inner_1);

        // Same content, different insertion order at both levels.
        let mut b: HashMap<u32, HashMap<u8, u16>> = HashMap::new();
        let mut b_inner_1: HashMap<u8, u16> = HashMap::new();
        b_inner_1.insert(5, 50);
        b.insert(1, b_inner_1);
        let mut b_inner_3: HashMap<u8, u16> = HashMap::new();
        b_inner_3.insert(10, 100);
        b_inner_3.insert(20, 200);
        b.insert(3, b_inner_3);

        #[derive(Serialize)]
        struct Wrap<'a> {
            #[serde(serialize_with = "sorted_nested_map")]
            map: &'a HashMap<u32, HashMap<u8, u16>>,
        }
        let bytes_a = bincode::serialize(&Wrap { map: &a }).unwrap();
        let bytes_b = bincode::serialize(&Wrap { map: &b }).unwrap();
        assert_eq!(bytes_a, bytes_b);
    }
}
