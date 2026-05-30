//! Wound mechanics: bleed, infection, healing, tourniquet necrosis.
//!
//! All systems here are **pure**: outputs are a deterministic function
//! of component-and-clock state, so they don't journal. Persisted
//! `Wounds` + `SimClock` is enough for replay to re-derive everything.
//!
//! - [`apply_bleed_damage`] drains the wound's body part HP from each
//!   `Untreated` Bleed at `severity * 0.5 hp/in-world-sec`, plus a
//!   small `infected` bonus drain on Bandaged/Stitched wounds.
//! - [`age_and_heal_wounds`] flips Bandaged → Healed after
//!   `MedConfig::heal_ticks_bandaged` and Stitched → Healed after
//!   `MedConfig::heal_ticks_stitched` (≈ half), then drops Healed.
//!   Tourniquet, Disinfected, Untreated never auto-heal — they require
//!   explicit treatment.
//! - [`tick_infection`] flips untreated wounds to `infected` after
//!   `MedConfig::infection_trigger_ticks`; an active `AntibioticsActive`
//!   effect clears infection over time.
//! - [`tick_necrosis`] applies escalating limb HP drain to wounds that
//!   have been tourniqueted past `MedConfig::necrosis_warning_ticks`,
//!   doubling after `necrosis_severe_ticks` more. Removing the
//!   tourniquet stops it.
//!
//! Tuning lives in [`MedConfig`] (with sensible defaults). Pre-Step-2
//! callsites that import `WOUND_THRESHOLD_LIGHT` /
//! `WOUND_THRESHOLD_HEAVY` continue to work — those constants drive
//! `apply_damage_to_part`'s wound-spawning, which is journaled and
//! deterministic.

use bevy_ecs::prelude::{Entity, Query, Res, With};

use crate::components::{
    ActiveEffects, BodyPart, BodyParts, EffectKind, Group, Health, InRegion, Inventory, LimbStates,
    Npc, NpcCharacter, Position, Wound, WoundKind, WoundTreatment, Wounds,
};
use crate::items::ItemId;
use crate::resources::{MedConfig, SimClock};
use crate::systems::meds::has_active;

/// HP drain per in-world second per point of severity. A sev-4 wound
/// drains 2 HP/sec; sev-5 drains 2.5 HP/sec. Calibrated so a heavy
/// bleed is dangerous in the minute-or-two scale (per spec §3.3
/// "nothing should ever kill the player in under a minute from a
/// survival stat alone" — combat wounds are different but the lower
/// end of the curve still gives the player reaction time).
pub const BLEED_RATE_PER_SEVERITY_PER_SEC: f32 = 0.5;

/// HP drain per in-world second from an `infected` wound, regardless
/// of severity or treatment state (Bandaged/Stitched still drain).
/// Slow on purpose — antibiotics arrive with the next visit to a
/// trader, not by panicking.
pub const INFECTION_HP_DRAIN_PER_SEC: f32 = 0.05;

/// HP drain per in-world second on a tourniqueted limb after the
/// warning window elapses; doubles past the severe window.
pub const NECROSIS_HP_DRAIN_PER_SEC: f32 = 0.05;
pub const NECROSIS_SEVERE_HP_DRAIN_PER_SEC: f32 = 0.2;

/// Damage threshold (post-clamp) at or above which an
/// `apply_damage_to_part` call also spawns a light Bleed wound.
pub const WOUND_THRESHOLD_LIGHT: f32 = 10.0;
/// Damage threshold at or above which the spawned wound is a heavy
/// bleed (severity 4–5) instead of light (1–3).
pub const WOUND_THRESHOLD_HEAVY: f32 = 25.0;

/// Map a damage amount to a wound severity in `[1, 5]`. Returns `None`
/// for sub-threshold damage (a bruise — HP loss only, no persistent
/// wound). Light bleed: damage 10..25 → severity 1..3. Heavy bleed:
/// damage 25..55 → severity 4..5 (capped).
pub fn severity_from_damage(amount: f32) -> Option<u8> {
    if amount >= WOUND_THRESHOLD_HEAVY {
        let extra = ((amount - WOUND_THRESHOLD_HEAVY) / 15.0).floor() as i32;
        Some((4 + extra).clamp(4, 5) as u8)
    } else if amount >= WOUND_THRESHOLD_LIGHT {
        let extra = ((amount - WOUND_THRESHOLD_LIGHT) / 8.0).floor() as i32;
        Some((1 + extra).clamp(1, 3) as u8)
    } else {
        None
    }
}

/// Default ticks a `Bandaged` wound takes to heal and despawn. 6000
/// ticks = 300 real seconds = 5 real minutes ≈ 1 in-game hour at the
/// default 12× compression. Tunable in playtest. Tests override via
/// `Sim::set_heal_ticks_for_test` (the actual value is stored in the
/// [`MedConfig`] resource).
pub const DEFAULT_HEAL_TICKS_BANDAGED: u64 = 6000;

/// Per-NPC bleed-rate scaling from the `endurance` stat. Endurance is
/// `0..=100`; the multiplier is linear in `[0.7, 1.3]` *inverted* —
/// high endurance reduces the rate, low endurance amplifies it. A
/// frail conscript at endurance 0 bleeds 30% faster than baseline; a
/// tough veteran at endurance 100 bleeds 30% slower. Symmetric around
/// 1.0 at endurance 50 so changes at the tail don't dominate combat
/// pacing. Players (no `NpcCharacter`) collapse to the flat 1.0
/// baseline so existing wound tuning is unchanged.
pub fn bleed_rate_multiplier(endurance: u8) -> f32 {
    BLEED_MULT_BIAS - BLEED_MULT_SLOPE * f32::from(endurance)
}

const BLEED_MULT_BIAS: f32 = 1.3;
const BLEED_MULT_SLOPE: f32 = 0.006;

fn bleed_rate(w: &Wound) -> f32 {
    let from_bleed = match (w.kind, w.treatment) {
        (WoundKind::Bleed, WoundTreatment::Untreated) => {
            f32::from(w.severity) * BLEED_RATE_PER_SEVERITY_PER_SEC
        }
        _ => 0.0,
    };
    // Infected wounds add a small drain even when Bandaged/Stitched.
    // Healed wounds don't (they're about to despawn). Tourniquet
    // wounds also drain — infection doesn't care about pressure.
    let from_infection = if w.infected && !matches!(w.treatment, WoundTreatment::Healed) {
        INFECTION_HP_DRAIN_PER_SEC
    } else {
        0.0
    };
    from_bleed + from_infection
}

/// Per-tick HP drain on each wound's body part. Sums bleed (from
/// untreated wounds) and infection (from any infected wound except
/// Healed). Applies to every entity with `Wounds + BodyParts` —
/// players and NPCs share this pipeline. NPCs with `NpcCharacter`
/// scale the drain by `bleed_rate_multiplier(endurance)`; players
/// (no `NpcCharacter`) take the flat baseline.
pub fn apply_bleed_damage(
    mut q: Query<(
        &Wounds,
        &mut BodyParts,
        Option<&mut Health>,
        Option<&NpcCharacter>,
    )>,
    clock: Res<SimClock>,
) {
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    for (wounds, mut parts, health, character) in &mut q {
        let endurance_mult = character
            .map(|c| bleed_rate_multiplier(c.stats.endurance))
            .unwrap_or(1.0);
        let mut any_drain = false;
        for (_, w) in &wounds.0 {
            let rate = bleed_rate(w) * endurance_mult;
            if rate <= 0.0 {
                continue;
            }
            let slot = parts.get_mut(w.body_part);
            *slot = (*slot - rate * dt_s).max(0.0);
            any_drain = true;
        }
        if any_drain {
            let vital = parts.vital_min();
            if let Some(mut h) = health {
                h.current = vital.min(h.max);
            }
        }
    }
}

/// Age `Bandaged` and `Stitched` wounds toward `Healed`, then drop
/// terminal `Healed` wounds from the `Wounds` Vec. Stitched wounds
/// heal at `MedConfig::heal_ticks_stitched` (default half the bandage
/// timer) — the reward for bringing the kit. Untreated, Disinfected,
/// Tourniquet, and WoundPacked never auto-heal — they require
/// further treatment. Infected wounds also don't auto-heal — clear
/// infection (antibiotics) before the heal timer makes progress.
/// Pure: deterministic from `treatment_changed_tick + clock.tick`.
pub fn age_and_heal_wounds(
    mut q: Query<(&mut Wounds, Option<&mut LimbStates>)>,
    clock: Res<SimClock>,
    cfg: Res<MedConfig>,
) {
    let now = clock.tick;
    for (mut wounds, states) in &mut q {
        for (_, w) in wounds.0.iter_mut() {
            if w.infected {
                continue; // Heal timer paused while infected.
            }
            let elapsed = now.saturating_sub(w.treatment_changed_tick);
            let heal_due = match w.treatment {
                WoundTreatment::Bandaged => Some(cfg.heal_ticks_bandaged),
                WoundTreatment::Stitched => Some(cfg.heal_ticks_stitched),
                _ => None,
            };
            if let Some(deadline) = heal_due {
                if elapsed >= deadline {
                    w.treatment = WoundTreatment::Healed;
                    w.treatment_changed_tick = now;
                }
            }
        }
        wounds
            .0
            .retain(|(_, w)| !matches!(w.treatment, WoundTreatment::Healed));
        // Wounded → Intact: any limb whose last open wound just got
        // dropped flips back. Severed parts are left alone — sever is
        // permanent.
        if let Some(mut states) = states {
            states.recompute_from_wounds(&wounds);
        }
    }
}

/// Untreated wounds become infected after `MedConfig::infection_trigger_ticks`
/// elapsed since their last treatment change (== spawn tick for a
/// never-treated wound). Disinfecting / bandaging / tourniqueting
/// resets the timer because each bumps `treatment_changed_tick`.
/// Once cleared by antibiotics, the timer also resets — so a wound
/// that's still Untreated after clearing won't re-infect for another
/// full window.
///
/// Bandaged / Stitched / Tourniquet / WoundPacked / Disinfected /
/// Healed wounds can't trigger infection (intermediate-friendly:
/// covering or sterilising is enough; the spec's full
/// disinfect-then-bandage protocol is best practice but not strictly
/// required to avoid infection).
///
/// An active `AntibioticsActive` effect clears infection on any
/// infected wound after `MedConfig::antibiotics_clear_ticks` of
/// treatment.
pub fn tick_infection(
    mut q: Query<(&mut Wounds, &ActiveEffects)>,
    clock: Res<SimClock>,
    cfg: Res<MedConfig>,
) {
    let now = clock.tick;
    for (mut wounds, effects) in &mut q {
        let antibiotics = has_active(effects, EffectKind::AntibioticsActive, now);
        for (_, w) in wounds.0.iter_mut() {
            // Step 1: untreated wounds age toward infection.
            if !w.infected
                && matches!(w.treatment, WoundTreatment::Untreated)
                && now.saturating_sub(w.treatment_changed_tick) >= cfg.infection_trigger_ticks
            {
                w.infected = true;
                w.infection_started_tick = Some(now);
            }
            // Step 2: antibiotics clear infection over time.
            if w.infected && antibiotics {
                if let Some(started) = w.infection_started_tick {
                    if now.saturating_sub(started) >= cfg.antibiotics_clear_ticks {
                        w.infected = false;
                        w.infection_started_tick = None;
                        // Reset the trigger timer so the wound doesn't
                        // immediately re-infect on the next tick.
                        w.treatment_changed_tick = now;
                    }
                }
            }
        }
    }
}

/// Cadence (sim ticks) between self/squad-medic heal passes.
/// Heal logic walks every online NPC with active untreated bleeds;
/// 1 Hz is plenty for "NPC notices wound, applies bandage" and
/// keeps the cost off the hot tick path.
const NPC_HEAL_TICK_INTERVAL: u64 = 20;

/// Sim ticks between consecutive heal applications by the same
/// applicator (self or medic mate). 60 ticks ≈ 3 s at 20 Hz —
/// enough that a single bandage isn't followed instantly by
/// another, but a multi-wound NPC still gets stabilized within
/// ~10 s. Per-applicator throttle lives in `NpcLastHealTick`
/// (transient resource, rebuilt naturally from misses).
const NPC_HEAL_COOLDOWN_TICKS: u64 = 60;

/// Max distance a squad-mate medic will reach in to apply a
/// bandage on behalf of a wounded ally without items. Squads
/// already cluster within `formation_offset` ranges, so 10 m
/// catches realistic adjacency without forcing NPC navigation
/// to a downed teammate (that's a future
/// `goal_arbitration::HealAlly` feature).
const SQUAD_MEDIC_RADIUS_M: f32 = 10.0;
const SQUAD_MEDIC_RADIUS_SQ_M: f32 = SQUAD_MEDIC_RADIUS_M * SQUAD_MEDIC_RADIUS_M;

/// Transient per-NPC cooldown table for the heal system. Tracks
/// when each NPC last applied a treatment (to anyone, including
/// themselves) so we throttle the visual "everyone bandages
/// each other simultaneously" flurry. Cleared naturally because
/// it's `Default` and we never persist it; on cold start every
/// NPC can heal immediately.
#[derive(bevy_ecs::prelude::Resource, Default)]
pub struct NpcLastHealTick {
    pub by_npc: std::collections::HashMap<crate::components::NpcId, u64>,
}

/// Decide which medical item id matches the worst untreated bleed
/// on this body part. Light bleeds (severity ≤ 3) want bandages.
/// Heavy bleeds (severity ≥ 4) want a wound pack first, falling
/// back to a tourniquet as the emergency option (necrosis timer
/// kicks in, but that's better than bleeding out).
fn pick_item_for_bleed(sev: u8) -> &'static [&'static str] {
    if sev >= 4 {
        // Heavy: wound pack first (no necrosis cost), tourniquet as
        // last resort (stops bleed, starts necrosis timer).
        &["wound_pack", "combat_tourniquet"]
    } else {
        // Light: bandage. Future: prefer disinfectant first.
        &["bandage"]
    }
}

/// Apply a wound treatment in-place. Pure ECS mutation — mirrors
/// the per-item branches in `Sim::apply_*_npc` without going
/// through the command path (we're inside a system, not a Sim
/// command). Returns true on a real state change.
fn apply_treatment_in_place(
    wounds: &mut Wounds,
    part: BodyPart,
    sev: u8,
    item_id: &str,
    now: u64,
) -> bool {
    for (_, w) in wounds.0.iter_mut() {
        if w.body_part != part
            || !matches!(w.kind, WoundKind::Bleed)
            || !matches!(
                w.treatment,
                WoundTreatment::Untreated | WoundTreatment::Disinfected
            )
        {
            continue;
        }
        let matches_severity = match item_id {
            "bandage" => sev <= 3 && w.severity == sev,
            "wound_pack" => sev >= 4 && w.severity == sev,
            "combat_tourniquet" => sev >= 4 && w.severity == sev,
            _ => false,
        };
        if !matches_severity {
            continue;
        }
        w.treatment = match item_id {
            "bandage" => WoundTreatment::Bandaged,
            "wound_pack" => WoundTreatment::WoundPacked,
            "combat_tourniquet" => {
                w.tourniquet_started_tick = Some(now);
                WoundTreatment::Tourniquet
            }
            _ => return false,
        };
        w.treatment_changed_tick = now;
        return true;
    }
    false
}

/// Self-heal + squad-medic pass. Runs every
/// `NPC_HEAL_TICK_INTERVAL` ticks. For each online NPC with an
/// active untreated bleed, picks the worst wound and tries to
/// apply an appropriate item from:
///   1. Their own inventory (self-heal).
///   2. Any same-faction (group) squad-mate within
///      `SQUAD_MEDIC_RADIUS_M` who has the item (squad-medic).
///
/// Per-applicator cooldown via `NpcLastHealTick` prevents the
/// visual "everyone bandages everyone simultaneously" flurry.
/// Doesn't navigate medics to remote teammates — that's a
/// future goal-arbitration `HealAlly` candidate. Until then the
/// system catches the common case where a wounded squad-mate is
/// already standing next to a healthy one, which is most of the
/// time given how tight squad formations sit.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn npc_treat_wounds(
    clock: Res<SimClock>,
    active_regions: Res<crate::resources::ActiveRegions>,
    mut cooldown: bevy_ecs::system::ResMut<NpcLastHealTick>,
    // `ParamSet` groups the conflicting query views (one reads
    // Wounds + Inventory, the other writes both) into a single
    // parameter that Bevy's borrow check accepts. Only one view
    // is borrowed at a time.
    mut q: bevy_ecs::system::ParamSet<(
        // p0: scan view — read NPC pos, group, wounds, and
        // inventory for both wounded-detection and medic-discovery.
        Query<
            'static,
            'static,
            (
                Entity,
                &'static crate::components::NpcId,
                &'static InRegion,
                &'static Position,
                Option<&'static Group>,
                &'static Wounds,
                &'static Inventory,
            ),
            With<Npc>,
        >,
        // p1: apply view — mutate the wounded's Wounds and the
        // applicator's Inventory.
        Query<'static, 'static, (&'static mut Wounds, &'static mut Inventory)>,
    )>,
) {
    let now = clock.tick;
    if !now.is_multiple_of(NPC_HEAL_TICK_INTERVAL) {
        return;
    }
    struct HealPlan {
        wounded: Entity,
        applicator: Entity,
        wounded_id: crate::components::NpcId,
        applicator_id: crate::components::NpcId,
        part: BodyPart,
        sev: u8,
        item: ItemId,
    }
    let mut plans: Vec<HealPlan> = Vec::new();
    // Snapshot the scan inputs into Vecs so we don't hold the
    // ParamSet's read-view borrow across the planning loop (which
    // calls `q.p0()` again to iterate medic candidates).
    let scan: Vec<(
        Entity,
        crate::components::NpcId,
        crate::region::RegionId,
        [f32; 3],
        Option<u64>,
        // Worst untreated bleed (part, severity) — None means
        // not wounded.
        Option<(BodyPart, u8)>,
        // Per-item availability counts for the three medical
        // items this system cares about.
        u32, // bandage
        u32, // wound_pack
        u32, // combat_tourniquet
    )> = q
        .p0()
        .iter()
        .map(|(e, id, r, p, g, wounds, inv)| {
            let mut worst: Option<(BodyPart, u8)> = None;
            for (_, w) in wounds.0.iter() {
                if !matches!(w.kind, WoundKind::Bleed)
                    || !matches!(
                        w.treatment,
                        WoundTreatment::Untreated | WoundTreatment::Disinfected
                    )
                {
                    continue;
                }
                let cur_sev = worst.map(|(_, s)| s).unwrap_or(0);
                if w.severity > cur_sev {
                    worst = Some((w.body_part, w.severity));
                }
            }
            let bandages = crate::inventory_grid::count_of(&inv.0, &ItemId("bandage".to_string()));
            let packs = crate::inventory_grid::count_of(&inv.0, &ItemId("wound_pack".to_string()));
            let tourniquets =
                crate::inventory_grid::count_of(&inv.0, &ItemId("combat_tourniquet".to_string()));
            (
                e,
                *id,
                r.0,
                p.0,
                g.map(|g| g.id),
                worst,
                bandages,
                packs,
                tourniquets,
            )
        })
        .collect();
    for (i, (w_entity, w_id, w_region, w_pos, w_group, worst, _, _, _)) in scan.iter().enumerate() {
        if !active_regions.is_active(*w_region) {
            continue;
        }
        // Cooldown gate: don't double-treat within the window.
        if let Some(&last) = cooldown.by_npc.get(w_id) {
            if now.saturating_sub(last) < NPC_HEAL_COOLDOWN_TICKS {
                continue;
            }
        }
        let Some((part, sev)) = worst else {
            continue;
        };
        let preferred = pick_item_for_bleed(*sev);
        // Try self-heal first using the cached per-item counts on
        // this NPC's own scan row.
        let mut picked = false;
        for &item in preferred {
            let avail = match item {
                "bandage" => scan[i].6,
                "wound_pack" => scan[i].7,
                "combat_tourniquet" => scan[i].8,
                _ => 0,
            };
            if avail > 0 {
                plans.push(HealPlan {
                    wounded: *w_entity,
                    applicator: *w_entity,
                    wounded_id: *w_id,
                    applicator_id: *w_id,
                    part: *part,
                    sev: *sev,
                    item: ItemId(item.to_string()),
                });
                picked = true;
                break;
            }
        }
        if picked {
            continue;
        }
        // No self-heal possible — scan nearby same-group mates.
        let Some(group_id) = w_group else {
            continue;
        };
        for (j, (m_entity, m_id, m_region, m_pos, m_group, _, _, _, _)) in scan.iter().enumerate() {
            if j == i || *m_region != *w_region {
                continue;
            }
            if m_group.is_none_or(|g| g != *group_id) {
                continue;
            }
            let dx = m_pos[0] - w_pos[0];
            let dz = m_pos[2] - w_pos[2];
            if dx * dx + dz * dz > SQUAD_MEDIC_RADIUS_SQ_M {
                continue;
            }
            if let Some(&last) = cooldown.by_npc.get(m_id) {
                if now.saturating_sub(last) < NPC_HEAL_COOLDOWN_TICKS {
                    continue;
                }
            }
            for &item in preferred {
                let avail = match item {
                    "bandage" => scan[j].6,
                    "wound_pack" => scan[j].7,
                    "combat_tourniquet" => scan[j].8,
                    _ => 0,
                };
                if avail > 0 {
                    plans.push(HealPlan {
                        wounded: *w_entity,
                        applicator: *m_entity,
                        wounded_id: *w_id,
                        applicator_id: *m_id,
                        part: *part,
                        sev: *sev,
                        item: ItemId(item.to_string()),
                    });
                    picked = true;
                    break;
                }
            }
            if picked {
                break;
            }
        }
    }
    // Apply plans. Consume from the applicator's inventory, mutate
    // the wounded's `Wounds`. Order matters: consume first so a
    // failed mutation doesn't lose the item; then re-add if the
    // wound mutation no-op'd (shouldn't happen given our checks).
    let mut apply_q = q.p1();
    for plan in plans {
        // Consume one of the item from the applicator.
        let consumed = if let Ok((_, mut inv)) = apply_q.get_mut(plan.applicator) {
            crate::inventory_grid::consume_from_grid(&mut inv.0, &plan.item, 1)
        } else {
            0
        };
        if consumed == 0 {
            continue;
        }
        // Apply treatment on the wounded NPC.
        let applied = if let Ok((mut wounds, _)) = apply_q.get_mut(plan.wounded) {
            apply_treatment_in_place(&mut wounds, plan.part, plan.sev, &plan.item.0, now)
        } else {
            false
        };
        if !applied {
            // Shouldn't happen: the wounded-q scan above already
            // matched the wound to the item. If it does (e.g. wound
            // was treated by a parallel path in the same tick),
            // the item is lost — cheap and visible in profiling
            // rather than a silent infinite loop.
            tracing::warn!(
                "npc_treat_wounds: consumed {:?} but apply no-op'd on {:?}",
                plan.item,
                plan.wounded_id
            );
            continue;
        }
        cooldown.by_npc.insert(plan.wounded_id, now);
        cooldown.by_npc.insert(plan.applicator_id, now);
    }
}

/// A wound with `tourniquet_started_tick` past the warning window
/// drains its limb HP. After the severe window (additional ticks
/// beyond warning), the drain rate doubles. Removing the tourniquet
/// (Sim::remove_tourniquet clears `tourniquet_started_tick`) stops it.
pub fn tick_necrosis(
    mut q: Query<(&Wounds, &mut BodyParts, Option<&mut Health>)>,
    clock: Res<SimClock>,
    cfg: Res<MedConfig>,
) {
    let now = clock.tick;
    let dt_s = f32::from(u16::try_from(clock.fixed_dt_ms).unwrap_or(u16::MAX)) / 1000.0;
    for (wounds, mut parts, health) in &mut q {
        let mut any_drain = false;
        for (_, w) in &wounds.0 {
            let Some(started) = w.tourniquet_started_tick else {
                continue;
            };
            let on_for = now.saturating_sub(started);
            if on_for < cfg.necrosis_warning_ticks {
                continue;
            }
            let rate = if on_for >= cfg.necrosis_warning_ticks + cfg.necrosis_severe_ticks {
                NECROSIS_SEVERE_HP_DRAIN_PER_SEC
            } else {
                NECROSIS_HP_DRAIN_PER_SEC
            };
            let slot = parts.get_mut(w.body_part);
            *slot = (*slot - rate * dt_s).max(0.0);
            any_drain = true;
        }
        if any_drain {
            let vital = parts.vital_min();
            if let Some(mut h) = health {
                h.current = vital.min(h.max);
            }
        }
    }
}
