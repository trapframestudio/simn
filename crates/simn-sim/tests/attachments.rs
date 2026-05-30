//! Phase 4C — attachment slot-tag graph (data only).
//!
//! Coverage:
//! - AKS-74 exposes its native slot tags via `WeaponConfig.slots`.
//! - Authored attachments parse with their consumes/provides edges.
//! - Validator accepts direct mounts (PSO-1 on the dovetail).
//! - Validator accepts a 2-stage chain (dovetail adapter → red dot).
//! - Validator rejects wrong-tag mounts with a clear error.
//! - Validator rejects duplicate consumes (two attachments racing
//!   for the same slot) with `TagAlreadyConsumed`.
//! - Validator's pool result tells the caller what tags remain
//!   after the chain (useful for future "still attachable" UX).

use simn_sim::{items::ItemRegistry, validate_attachment_chain, AttachmentError, ItemId};

fn registry() -> ItemRegistry {
    ItemRegistry::load()
}

fn id(s: &str) -> ItemId {
    ItemId::from(s)
}

#[test]
fn aks74_exposes_authored_slot_tags() {
    let reg = registry();
    let aks = reg
        .get(&id("rifle_aks74"))
        .expect("aks74 should be in the registry");
    let wc = aks.weapon_config.as_ref().expect("aks74 has weapon_config");
    assert!(
        !wc.slots.is_empty(),
        "aks74 should have authored slots — Phase 4C added them"
    );
    // Spot-check the five expected tags we set in
    // `weapons.toml`. Tag *positions* (slot_id) don't matter
    // for the validator; the tag value is what's load-bearing.
    let all_tags: Vec<&String> = wc.slots.iter().flat_map(|s| s.tags.iter()).collect();
    for needed in [
        "threaded_14x1_lh",
        "dovetail_side",
        "ak_handguard",
        "warsaw_stock",
        "ak_545_mag",
    ] {
        assert!(
            all_tags.iter().any(|t| t.as_str() == needed),
            "aks74 should expose `{needed}` slot tag; got {all_tags:?}",
        );
    }
}

#[test]
fn attachments_parse_with_consumes_and_provides_tags() {
    let reg = registry();
    let cases: &[(&str, &str, &[&str])] = &[
        ("att_pso1_scope", "dovetail_side", &[]),
        ("att_ak_dovetail_picatinny", "dovetail_side", &["picatinny"]),
        ("att_aimpoint_compm4", "picatinny", &[]),
        ("att_pbs1_suppressor", "threaded_14x1_lh", &[]),
        ("att_ultimak_rail", "ak_handguard", &["picatinny"]),
    ];
    for (att_id, consumes, provides) in cases {
        let def = reg
            .get(&id(att_id))
            .unwrap_or_else(|| panic!("{att_id} not in registry"));
        let cfg = def
            .attachment_config
            .as_ref()
            .unwrap_or_else(|| panic!("{att_id} has no attachment_config"));
        assert_eq!(cfg.consumes_tag, *consumes, "{att_id} consumes mismatch",);
        let got: Vec<&str> = cfg.provides_tags.iter().map(|s| s.as_str()).collect();
        assert_eq!(got.as_slice(), *provides, "{att_id} provides mismatch");
    }
}

#[test]
fn direct_dovetail_mount_succeeds() {
    let reg = registry();
    let result = validate_attachment_chain(&reg, &id("rifle_aks74"), &[id("att_pso1_scope")])
        .expect("PSO-1 on AKS-74 dovetail should validate");
    // PSO-1 provides no new tags; the remaining pool is the four
    // unclaimed slot tags.
    assert!(!result.iter().any(|t| t == "dovetail_side"));
    assert!(result.iter().any(|t| t == "threaded_14x1_lh"));
}

#[test]
fn two_stage_dovetail_to_picatinny_chain_succeeds() {
    let reg = registry();
    let result = validate_attachment_chain(
        &reg,
        &id("rifle_aks74"),
        &[
            id("att_ak_dovetail_picatinny"), // dovetail → picatinny
            id("att_aimpoint_compm4"),       // picatinny → ø
        ],
    )
    .expect("dovetail→pic adapter then red dot should chain");
    // Both consumes resolved; pool no longer has dovetail_side or
    // picatinny but still has the other three native tags.
    assert!(!result.iter().any(|t| t == "dovetail_side"));
    assert!(!result.iter().any(|t| t == "picatinny"));
    assert!(result.iter().any(|t| t == "threaded_14x1_lh"));
}

#[test]
fn red_dot_alone_rejects_missing_picatinny() {
    let reg = registry();
    let err = validate_attachment_chain(
        &reg,
        &id("rifle_aks74"),
        &[id("att_aimpoint_compm4")], // wants picatinny but AKS-74 has none natively
    )
    .expect_err("red dot should be rejected without an adapter");
    assert!(
        matches!(
            err,
            AttachmentError::NoMatchingSlot { ref needed_tag, .. } if needed_tag == "picatinny"
        ),
        "expected NoMatchingSlot for `picatinny`; got {err:?}",
    );
}

#[test]
fn duplicate_dovetail_consume_rejects_second_attachment() {
    let reg = registry();
    let err = validate_attachment_chain(
        &reg,
        &id("rifle_aks74"),
        &[
            id("att_pso1_scope"),            // takes dovetail_side
            id("att_ak_dovetail_picatinny"), // also wants dovetail_side
        ],
    )
    .expect_err("two attachments fighting over dovetail_side should fail");
    match err {
        AttachmentError::TagAlreadyConsumed { tag, attached, .. } => {
            assert_eq!(tag, "dovetail_side");
            assert_eq!(attached, id("att_pso1_scope"));
        }
        other => panic!("expected TagAlreadyConsumed; got {other:?}"),
    }
}

#[test]
fn ultimak_then_red_dot_chain_succeeds() {
    let reg = registry();
    // Ultimak takes the handguard slot and provides picatinny on
    // top of the gas tube — an alternative route to mounting a
    // red dot when the side dovetail is occupied.
    let result = validate_attachment_chain(
        &reg,
        &id("rifle_aks74"),
        &[id("att_ultimak_rail"), id("att_aimpoint_compm4")],
    )
    .expect("ultimak rail → red dot should chain");
    assert!(!result.iter().any(|t| t == "ak_handguard"));
    assert!(!result.iter().any(|t| t == "picatinny"));
}

#[test]
fn unknown_attachment_id_returns_unknown_item_error() {
    let reg = registry();
    let err = validate_attachment_chain(&reg, &id("rifle_aks74"), &[id("att_not_a_real_thing")])
        .expect_err("unknown item id should fail");
    assert!(
        matches!(err, AttachmentError::UnknownItem(_)),
        "got {err:?}",
    );
}

#[test]
fn weapon_id_pointing_at_non_weapon_returns_not_a_weapon_error() {
    let reg = registry();
    // `round_9x18` is an ammo entry, no weapon_config.
    let err = validate_attachment_chain(&reg, &id("round_9x18"), &[id("att_pso1_scope")])
        .expect_err("non-weapon as the chain's weapon should fail");
    assert!(matches!(err, AttachmentError::NotAWeapon(_)), "got {err:?}",);
}

#[test]
fn empty_attachment_chain_returns_full_native_tag_pool() {
    let reg = registry();
    let pool = validate_attachment_chain(&reg, &id("rifle_aks74"), &[])
        .expect("empty chain is always valid");
    // All 5 native AKS-74 slot tags should be in the pool.
    for needed in [
        "threaded_14x1_lh",
        "dovetail_side",
        "ak_handguard",
        "warsaw_stock",
        "ak_545_mag",
    ] {
        assert!(
            pool.iter().any(|t| t == needed),
            "empty chain should leave `{needed}` in the pool",
        );
    }
}
