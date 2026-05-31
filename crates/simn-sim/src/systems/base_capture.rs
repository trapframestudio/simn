//! Base capture mechanic. Lets hostile-faction squads take over
//! enemy POIs in active regions when they've eliminated the
//! defenders and have boots on the ground at the base.
//!
//! Headquarters bases are immune by design — they're narrative
//! anchors, not mechanically flippable.
//!
//! The companion "try to capture" half lives in
//! `squad_planner::build_investigate`, which now seeds the
//! Investigate target as a nearby hostile-faction base when one
//! exists. Squads walking the Investigate path engage defenders
//! via `npc_aggro`; this system finishes the job by flipping
//! `InFaction` once the defenders are gone.

use bevy_ecs::prelude::{Entity, Query, Res, ResMut, With};
use std::collections::HashMap;

use crate::components::{Base, Health, InFaction, InRegion, Npc, Position};
use crate::faction::registry::{FactionId, FactionRegistry, RelationDeltas};
use crate::helpers::quantize_post_pos;
use crate::resources::{ActiveRegions, GuardPosts, SimClock};

/// Capture-check cadence (sim ticks). 3 s at 20 Hz — captures
/// happen on the scale of seconds-to-minutes, no need to scan
/// every tick.
const CAPTURE_CHECK_INTERVAL_TICKS: u64 = 60;
/// Radius (meters) around a base within which NPCs count toward
/// the capture tally. Tighter than `OFFLINE_ENGAGEMENT_RADIUS_M`
/// (150 m) because "at the base" is more specific than "fighting
/// nearby" — a passing patrol shouldn't capture a base from a
/// distance.
const CAPTURE_RADIUS_M: f32 = 40.0;
const CAPTURE_RADIUS_SQ_M: f32 = CAPTURE_RADIUS_M * CAPTURE_RADIUS_M;
/// Minimum hostile attackers required to flip a base. One NPC is
/// a passer-by; two or more is an occupation.
const MIN_ATTACKERS_FOR_CAPTURE: u32 = 2;

/// Online-tier base capture pass. Runs at `CAPTURE_CHECK_INTERVAL_TICKS`
/// cadence; scans every base in active regions, counts attackers
/// (hostile to owner) vs. defenders (owner faction) within
/// `CAPTURE_RADIUS_M`, and flips ownership when defenders are
/// gone and a single hostile faction has at least
/// `MIN_ATTACKERS_FOR_CAPTURE` NPCs on site. Headquarters bases
/// never flip.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn base_capture_check(
    clock: Res<SimClock>,
    active_regions: Res<ActiveRegions>,
    registry: Res<FactionRegistry>,
    deltas: Res<RelationDeltas>,
    mut event_bus: ResMut<crate::world_event_bus::WorldEventQueue>,
    mut pda_log: ResMut<crate::pda_log::PdaEventLog>,
    mut posts: ResMut<GuardPosts>,
    npcs: Query<(&InFaction, &InRegion, &Position, &Health), With<Npc>>,
    mut bases: Query<
        (Entity, &Base, &mut InFaction, &InRegion, &Position),
        bevy_ecs::query::Without<Npc>,
    >,
) {
    let now = clock.tick;
    if !now.is_multiple_of(CAPTURE_CHECK_INTERVAL_TICKS) {
        return;
    }

    // Snapshot live NPCs (with full health > 0) so the inner
    // base loop is a quick spatial filter rather than a re-query
    // per base.
    struct Snap {
        pos: [f32; 3],
        faction: FactionId,
        region: crate::region::RegionId,
    }
    let snaps: Vec<Snap> = npcs
        .iter()
        .filter(|(_, _, _, h)| h.current > 0.0)
        .map(|(f, r, p, _)| Snap {
            pos: p.0,
            faction: f.0,
            region: r.0,
        })
        .collect();

    for (_entity, base, mut base_faction, base_region, base_pos) in bases.iter_mut() {
        if !active_regions.is_active(base_region.0) {
            continue;
        }
        if crate::poi::base_is_victory_target(base.kind) {
            continue;
        }
        let owner = base_faction.0;
        let bp = base_pos.0;
        let region = base_region.0;
        let mut defenders: u32 = 0;
        let mut attacker_counts: HashMap<FactionId, u32> = HashMap::new();
        for s in &snaps {
            if s.region != region {
                continue;
            }
            let dx = s.pos[0] - bp[0];
            let dz = s.pos[2] - bp[2];
            if dx * dx + dz * dz > CAPTURE_RADIUS_SQ_M {
                continue;
            }
            if s.faction == owner {
                defenders += 1;
            } else {
                let rel = crate::faction::registry::faction_relation(
                    &registry, &deltas, s.faction, owner,
                );
                if matches!(rel, crate::faction::Relation::Hostile) {
                    *attacker_counts.entry(s.faction).or_default() += 1;
                }
            }
        }
        if defenders > 0 {
            continue;
        }
        // Sort by NpcId-less proxy (faction id raw) for
        // deterministic tiebreak when two attacker factions tie
        // on count.
        let mut sorted: Vec<(FactionId, u32)> = attacker_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let Some(&(new_owner, count)) = sorted.first() else {
            continue;
        };
        if count < MIN_ATTACKERS_FOR_CAPTURE {
            continue;
        }
        // Flip ownership.
        base_faction.0 = new_owner;
        // Vacate any guard post claim on this base — the old
        // holder's squad is dead or routed, and the new owner
        // hasn't claimed it through the planner yet.
        let key = (region, quantize_post_pos(bp));
        posts.by_key.remove(&key);
        // Bus event + PDA toast.
        event_bus.push(
            crate::world_event_bus::WorldEventKind::BaseFlip {
                new_owner,
                old_owner: Some(owner),
            },
            bp,
            region,
            now,
            /* ttl_ticks = */ 4,
        );
        pda_log.push(
            crate::pda_log::PdaEvent::BaseFlip {
                new_owner: registry.name_of(new_owner).to_string(),
                old_owner: Some(registry.name_of(owner).to_string()),
                region,
            },
            now,
        );
    }
}
