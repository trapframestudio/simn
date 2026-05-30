//! Objective builder functions and helpers for the squad planner.

use super::*;

// ── Activity-point-driven objective builders ─────────────────────

pub(crate) fn try_guard_from_activity_points(
    aps: &mut crate::resources::ActivityPoints,
    summary: &GroupSummary,
    group_id: u64,
    _now: u64,
) -> Option<SquadObjective> {
    let points = aps.by_region.get(&summary.region)?;
    let mut best: Option<(usize, f32)> = None;
    for (i, pt) in points.iter().enumerate() {
        if !pt.kind.is_guard() {
            continue;
        }
        if !pt.has_capacity() && !pt.is_claimed_by(group_id) {
            continue;
        }
        if let Some(fac) = pt.faction {
            if fac != summary.faction {
                continue;
            }
        }
        let dx = pt.pos[0] - summary.centroid[0];
        let dz = pt.pos[2] - summary.centroid[2];
        let dist = (dx * dx + dz * dz).sqrt();
        // Penalize crowded points so squads spread across APs.
        let crowding_penalty = pt.claimed_by_groups.len() as f32 * 200.0;
        let score = pt.priority as f32 * 100.0 - dist - crowding_penalty;
        if best.is_none() || score > best.unwrap().1 {
            best = Some((i, score));
        }
    }
    let (idx, _) = best?;
    let pt = &aps.by_region.get(&summary.region)?[idx];
    let pos = pt.pos;
    if let Some(points_mut) = aps.by_region.get_mut(&summary.region) {
        if !points_mut[idx].claimed_by_groups.contains(&group_id) {
            points_mut[idx].claimed_by_groups.push(group_id);
        }
    }
    Some(SquadObjective::Guard {
        base_pos: pos,
        expires_at: u64::MAX,
        post_key: None,
    })
}

pub(crate) fn try_patrol_from_activity_points(
    aps: &mut crate::resources::ActivityPoints,
    summary: &GroupSummary,
    group_id: u64,
    now: u64,
) -> Option<SquadObjective> {
    let routes = aps.routes_by_region.get(&summary.region)?;
    let mut best: Option<(usize, f32)> = None;
    for (i, route) in routes.iter().enumerate() {
        if route.claimed_by_group.is_some() && route.claimed_by_group != Some(group_id) {
            continue;
        }
        if let Some(fac) = route.faction {
            if fac != summary.faction {
                continue;
            }
        }
        if route.waypoints.len() < 2 {
            continue;
        }
        let first_wp = route.waypoints[0];
        let dx = first_wp[0] - summary.centroid[0];
        let dz = first_wp[2] - summary.centroid[2];
        let dist = (dx * dx + dz * dz).sqrt();
        let score = route.priority as f32 * 100.0 - dist;
        if best.is_none() || score > best.unwrap().1 {
            best = Some((i, score));
        }
    }
    let (idx, _) = best?;
    let route = &aps.routes_by_region.get(&summary.region)?[idx];
    let waypoints = route.waypoints.clone();
    if let Some(routes_mut) = aps.routes_by_region.get_mut(&summary.region) {
        routes_mut[idx].claimed_by_group = Some(group_id);
    }
    Some(SquadObjective::Patrol {
        route: waypoints,
        current_idx: 0,
        expires_at: now.wrapping_add(pcfg().patrol_duration_ticks),
    })
}

pub(crate) fn try_rest_from_activity_points(
    aps: &mut crate::resources::ActivityPoints,
    summary: &GroupSummary,
    group_id: u64,
    now: u64,
) -> Option<SquadObjective> {
    let points = aps.by_region.get(&summary.region)?;
    let mut best: Option<(usize, f32)> = None;
    for (i, pt) in points.iter().enumerate() {
        if !pt.kind.is_rest() {
            continue;
        }
        if !pt.has_capacity() && !pt.is_claimed_by(group_id) {
            continue;
        }
        if let Some(fac) = pt.faction {
            if fac != summary.faction {
                continue;
            }
        }
        let dx = pt.pos[0] - summary.centroid[0];
        let dz = pt.pos[2] - summary.centroid[2];
        let dist = (dx * dx + dz * dz).sqrt();
        let crowding_penalty = pt.claimed_by_groups.len() as f32 * 200.0;
        let score = pt.priority as f32 * 100.0 - dist - crowding_penalty;
        if best.is_none() || score > best.unwrap().1 {
            best = Some((i, score));
        }
    }
    let (idx, _) = best?;
    let pt = &aps.by_region.get(&summary.region)?[idx];
    let pos = pt.pos;
    if let Some(points_mut) = aps.by_region.get_mut(&summary.region) {
        if !points_mut[idx].claimed_by_groups.contains(&group_id) {
            points_mut[idx].claimed_by_groups.push(group_id);
        }
    }
    Some(SquadObjective::Rest {
        base_pos: pos,
        expires_at: now.wrapping_add(pcfg().rest_duration_ticks),
        area_id: None,
    })
}

pub(crate) fn build_patrol(
    rng: &mut ChaCha8Rng,
    summary: &GroupSummary,
    now: u64,
    recent: &VecDeque<[i32; 3]>,
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
) -> Option<SquadObjective> {
    let pool = same_faction_or_any_in_region(summary.faction, summary.region, bases);
    if pool.is_empty() {
        return None;
    }
    // Prefer bases not in recent. If all are recent, use the full pool.
    let mut filtered: Vec<[f32; 3]> = pool
        .iter()
        .copied()
        .filter(|p| !recent.contains(&quantize(*p)))
        .collect();
    if filtered.is_empty() {
        filtered = pool;
    }
    filtered.shuffle(rng);
    let route: Vec<[f32; 3]> = filtered.into_iter().take(PATROL_ROUTE_LEN).collect();
    Some(SquadObjective::Patrol {
        route,
        current_idx: 0,
        expires_at: now + pcfg().patrol_duration_ticks,
    })
}

/// Minimum spacing (meters) between active guard posts within a
/// region. Two same-faction bases placed close together on the
/// map (e.g. POI baker scatter) would otherwise each get their own
/// squad, and the squads' formation rings would overlap. 30 m is
/// just past the formation-ring outer edge (10 m base + ~20 m
/// per-squad jitter) so rings can sit at adjacent bases without
/// piling. Loose enough that dense base clusters still get
/// multiple guard squads — earlier 60 m gate was over-rejecting
/// and dropping new squads to Wander.
const MIN_GUARD_SPACING_M: f32 = 30.0;
const MIN_GUARD_SPACING_SQ_M: f32 = MIN_GUARD_SPACING_M * MIN_GUARD_SPACING_M;

pub(crate) fn build_guard(
    rng: &mut ChaCha8Rng,
    summary: &GroupSummary,
    _now: u64,
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
    posts: &GuardPosts,
    group_id: u64,
    recent: &VecDeque<[i32; 3]>,
) -> Option<SquadObjective> {
    // Materialize the set of held post positions in this region so
    // the per-candidate distance check is a small linear scan
    // (vs. re-walking `posts.by_key` per candidate).
    let mut held_positions: Vec<[f32; 3]> = Vec::new();
    for (key, info) in posts.by_key.iter() {
        if key.0 != summary.region {
            continue;
        }
        if info.group_id == group_id {
            continue;
        }
        // Reconstitute the world-space center from the 10 m
        // quantized key. Exact match with the original base
        // position is unnecessary — we only need it for the
        // distance check.
        held_positions.push([(key.1[0] as f32) * 10.0, 0.0, (key.1[2] as f32) * 10.0]);
    }
    // Only consider bases that are not currently posted by someone
    // else AND aren't sitting within `MIN_GUARD_SPACING_M` of an
    // existing post. If everything's taken, fall through and the
    // caller will try `build_relieve` instead.
    let mut unposted: Vec<[f32; 3]> = Vec::new();
    for (_, f, r, p) in bases.iter() {
        if r.0 != summary.region {
            continue;
        }
        if f.0 != summary.faction {
            continue;
        }
        let key = (summary.region, quantize_post_pos(p.0));
        let held = posts
            .by_key
            .get(&key)
            .is_some_and(|info| info.group_id != group_id);
        if held {
            continue;
        }
        let too_close = held_positions.iter().any(|h| {
            let dx = h[0] - p.0[0];
            let dz = h[2] - p.0[2];
            dx * dx + dz * dz < MIN_GUARD_SPACING_SQ_M
        });
        if too_close {
            continue;
        }
        unposted.push(p.0);
    }
    if unposted.is_empty() {
        return None;
    }
    // Prefer a post the squad hasn't visited recently — without
    // this filter a squad whose Guard objective expired (relief
    // or eviction) would often re-pick the same nearby post on
    // its next roll, contributing to the "stuck around one base"
    // pattern. If everything's recent we fall back to the
    // unfiltered list so the build doesn't fail.
    let filtered: Vec<[f32; 3]> = unposted
        .iter()
        .copied()
        .filter(|p| !recent.contains(&quantize(*p)))
        .collect();
    let pool = if filtered.is_empty() {
        unposted
    } else {
        filtered
    };
    let pos = pick_nearby(rng, summary.centroid, pool, 3);
    let key = (summary.region, quantize_post_pos(pos));
    Some(SquadObjective::Guard {
        base_pos: pos,
        // Posted guards ignore `expires_at` by design; we set
        // u64::MAX so the expires-check can never trip.
        expires_at: u64::MAX,
        post_key: Some(key),
    })
}

/// Find a post in the same region that's been held long enough to
/// be "relief-eligible" and dispatch this squad to take it over.
/// Prefers posts near the squad centroid so relief doesn't require
/// a cross-map march.
pub(crate) fn build_relieve(
    rng: &mut ChaCha8Rng,
    summary: &GroupSummary,
    now: u64,
    posts: &GuardPosts,
    group_id: u64,
    relief_targeted: &std::collections::HashSet<(RegionId, [i32; 3])>,
) -> Option<SquadObjective> {
    // Eligible posts: in same region, held by a same-faction squad
    // that isn't us, held long enough to be relief-eligible. The
    // same-faction gate is what stops a PWA squad from "relieving"
    // a federal guard post — cross-faction post takeover should
    // require combat, not a peaceful handoff. Allied factions
    // (NAP partners) don't relieve either; their reinforcement
    // shows up as a separate Guard objective in the same region.
    type ReliefCandidate = ((RegionId, [i32; 3]), [f32; 3]);
    let mut candidates: Vec<ReliefCandidate> = Vec::new();
    for (key, info) in &posts.by_key {
        if key.0 != summary.region {
            continue;
        }
        if info.faction != summary.faction {
            continue;
        }
        if info.group_id == group_id {
            continue;
        }
        if now.saturating_sub(info.since_tick) < POST_MIN_AGE_FOR_RELIEF_TICKS {
            continue;
        }
        // Skip posts already targeted by another Relieve objective —
        // otherwise multiple squads converge on the same dest_pos
        // and pile up while only one of them gets to be the new
        // post-holder.
        if relief_targeted.contains(key) {
            continue;
        }
        // Reconstitute the world-space position from the quantized key.
        let world_pos = [(key.1[0] as f32) * 10.0, 0.0, (key.1[2] as f32) * 10.0];
        candidates.push((*key, world_pos));
    }
    if candidates.is_empty() {
        return None;
    }
    // Sort by distance from our centroid, pick from the nearest 3.
    candidates.sort_by(|a, b| {
        let da = (a.1[0] - summary.centroid[0]).powi(2) + (a.1[2] - summary.centroid[2]).powi(2);
        let db = (b.1[0] - summary.centroid[0]).powi(2) + (b.1[2] - summary.centroid[2]).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let pick_idx = rng.gen_range(0..candidates.len().min(3));
    let (key, dest_pos) = candidates[pick_idx];
    Some(SquadObjective::Relieve {
        post_key: key,
        dest_pos,
        expires_at: now + pcfg().relieve_duration_ticks,
    })
}

pub(crate) fn build_investigate(
    rng: &mut ChaCha8Rng,
    summary: &GroupSummary,
    now: u64,
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
    registry: &crate::faction::registry::FactionRegistry,
    deltas: &crate::faction::registry::RelationDeltas,
) -> SquadObjective {
    // Prefer to scout/raid an enemy-held base in the squad's
    // current region — captures the "PWA squad walking through
    // a Federal region eyes a Federal outpost" intuition. Only
    // hostile-faction non-Headquarters bases are eligible
    // (HQs are flip-immune by design). With multiple candidates,
    // pick from the nearest 3 with squad-RNG jitter so squads
    // don't all converge on the same target.
    let mut hostile_bases: Vec<[f32; 3]> = Vec::new();
    for (b, f, r, p) in bases.iter() {
        if r.0 != summary.region {
            continue;
        }
        if matches!(b.kind, BaseKind::Headquarters) {
            continue;
        }
        if matches!(b.kind, BaseKind::CampSite) {
            continue;
        }
        if f.0 == summary.faction {
            continue;
        }
        let relation =
            crate::faction::registry::faction_relation(registry, deltas, summary.faction, f.0);
        if !matches!(relation, crate::faction::Relation::Hostile) {
            continue;
        }
        hostile_bases.push(p.0);
    }
    if !hostile_bases.is_empty() {
        let pos = pick_nearby(rng, summary.centroid, hostile_bases, 3);
        return SquadObjective::Investigate {
            target: pos,
            expires_at: now + pcfg().investigate_duration_ticks,
        };
    }
    // Fallback: random open-world point.
    SquadObjective::Investigate {
        target: [
            rng.gen_range(-2200.0..2200.0),
            0.0,
            rng.gen_range(-2200.0..2200.0),
        ],
        expires_at: now + pcfg().investigate_duration_ticks,
    }
}

pub(crate) fn build_explore(
    rng: &mut ChaCha8Rng,
    summary: &GroupSummary,
    now: u64,
    graph: &RegionGraph,
) -> Option<SquadObjective> {
    let region = graph.get(summary.region)?;
    if region.neighbors.is_empty() {
        return None;
    }
    let dest = *region.neighbors.choose(rng)?;
    let portal_pos = *region.transitions.get(&dest)?;
    Some(SquadObjective::Explore {
        dest_region: dest,
        portal_pos,
        // 2km walk at 3 m/s = ~11 min — Explore needs a long
        // leash or the objective expires before they arrive.
        expires_at: now + pcfg().explore_duration_ticks,
    })
}

/// Squads that pick `Rest` look here first: any designer-placed
/// `kind == "rest"` `InteractionArea` whose faction matches (or
/// is unrestricted) AND that sits within this radius of the
/// squad centroid wins over the generic base-pool fallback. Far
/// enough that camps clear of the squad are reachable; close
/// enough that squads don't trek across the region just for a
/// rest spot when a base is right next to them.
const REST_INTERACTION_AREA_PREFER_RADIUS_M: f32 = 150.0;

pub(crate) fn build_rest(
    rng: &mut ChaCha8Rng,
    summary: &GroupSummary,
    now: u64,
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
    interaction_areas: &mut crate::resources::InteractionAreas,
    taken_rest_anchors: &std::collections::HashSet<(RegionId, [i32; 3])>,
    recent: &VecDeque<[i32; 3]>,
) -> Option<SquadObjective> {
    // Iteration 5-13 Phase D3: prefer designer-placed
    // `kind == "rest"` interaction areas within
    // `REST_INTERACTION_AREA_PREFER_RADIUS_M` of the squad
    // centroid, before falling back to the base-pool path. The
    // area must (a) be in the squad's region, (b) accept the
    // squad's faction (None area faction = any), (c) have free
    // capacity. The reservation lifecycle is sim-side; the
    // matching release lives at the prior-objective swap in
    // `squad_planner`.
    if let Some((area_id, pos)) = pick_rest_area(rng, summary, interaction_areas) {
        interaction_areas.reserve_internal(&area_id);
        return Some(SquadObjective::Rest {
            base_pos: pos,
            expires_at: now + pcfg().rest_duration_ticks,
            area_id: Some(area_id),
        });
    }
    // Prefer neutral campsites, then same-faction Safehouse/Outpost.
    // Either way, any faction is welcome at a CampSite. Filter out
    // anchors already claimed by another squad's Rest (across
    // ticks — `taken_rest_anchors` is seeded from existing
    // objectives at the top of the planner). If every same-faction
    // option is already taken, return None so the caller falls
    // through to Wander — better than piling a third or fourth
    // squad onto the same outpost and forming a giant clump.
    let mut camp: Vec<[f32; 3]> = Vec::new();
    let mut same: Vec<[f32; 3]> = Vec::new();
    for (b, f, r, p) in bases.iter() {
        if r.0 != summary.region {
            continue;
        }
        if matches!(b.kind, BaseKind::CampSite) {
            camp.push(p.0);
        } else if f.0 == summary.faction
            && matches!(b.kind, BaseKind::Safehouse | BaseKind::Outpost)
        {
            same.push(p.0);
        }
    }
    let pool_unfiltered = if !camp.is_empty() { camp } else { same };
    if pool_unfiltered.is_empty() {
        return None;
    }
    let filtered: Vec<[f32; 3]> = pool_unfiltered
        .iter()
        .copied()
        .filter(|p| !taken_rest_anchors.contains(&(summary.region, quantize_anchor(*p))))
        .filter(|p| !recent.contains(&quantize(*p)))
        .collect();
    if filtered.is_empty() {
        return None;
    }
    // Nearest-N with jitter (see build_guard) so rest points are
    // reachable and actually observable — not across the map.
    let pos = pick_nearby(rng, summary.centroid, filtered, 3);
    Some(SquadObjective::Rest {
        base_pos: pos,
        expires_at: now + pcfg().rest_duration_ticks,
        area_id: None,
    })
}

/// Per-(group, tick, kind) deterministic noise term added to each
/// objective's utility score so two same-faction squads with
/// identical weights + personality don't always roll the same
/// objective on the same tick. Magnitude tuned so it can flip
/// "weak preference" picks but not override strong blackboard
/// signals (e.g. under-fire 5x multipliers).
///
/// Mixed via a 64-bit splitmix-style hash so adjacent group ids
/// and adjacent kinds get well-separated outputs. Deterministic:
/// same `(group_id, tick, kind)` always produces the same noise.
pub(crate) fn squad_objective_noise(group_id: u64, tick: u64, kind: ObjKind) -> f32 {
    let kind_idx: u64 = match kind {
        ObjKind::Patrol => 0,
        ObjKind::Guard => 1,
        ObjKind::Investigate => 2,
        ObjKind::Rest => 3,
        ObjKind::Explore => 4,
        ObjKind::Wander => 5,
    };
    let mut x = group_id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= tick.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= kind_idx.wrapping_mul(0x94D0_49BB_1331_11EB);
    // Splitmix64 finalizer.
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    // Map u64 to [0, 1.0).
    (x as f32 / u64::MAX as f32).clamp(0.0, 0.999)
}

/// Quantize a world-space anchor (base/rally position) to a stable
/// integer triple at 10 m granularity. Used as the key for the
/// per-tick `taken_rest_anchors` reservation set so jitter in the
/// stored `base_pos` across ticks doesn't fail the dedupe.
pub(crate) fn quantize_anchor(p: [f32; 3]) -> [i32; 3] {
    [
        (p[0] / 10.0).round() as i32,
        (p[1] / 10.0).round() as i32,
        (p[2] / 10.0).round() as i32,
    ]
}

/// Iteration 5-13 Phase D3. Scan the per-region interaction-area
/// set for the closest `kind == "rest"` candidate that
///
/// - matches the squad's faction (or is faction-unrestricted)
/// - has free capacity (`occupants < capacity`)
/// - sits within `REST_INTERACTION_AREA_PREFER_RADIUS_M` of the
///   squad centroid
///
/// Returns `(area_id, world-space pos)` when one is found.
/// Reservation is the caller's responsibility — `build_rest`
/// calls `reserve_internal` on the chosen area immediately.
pub(crate) fn pick_rest_area(
    rng: &mut ChaCha8Rng,
    summary: &GroupSummary,
    interaction_areas: &crate::resources::InteractionAreas,
) -> Option<(String, [f32; 3])> {
    let areas = interaction_areas.by_region.get(&summary.region)?;
    let mut candidates: Vec<(String, [f32; 3], f32)> = Vec::new();
    let max_d2 = REST_INTERACTION_AREA_PREFER_RADIUS_M * REST_INTERACTION_AREA_PREFER_RADIUS_M;
    for area in areas {
        if area.kind != "rest" {
            continue;
        }
        if let Some(required) = area.faction {
            if required != summary.faction {
                continue;
            }
        }
        if area.occupants >= area.capacity {
            continue;
        }
        let dx = area.pos[0] - summary.centroid[0];
        let dz = area.pos[2] - summary.centroid[2];
        let d2 = dx * dx + dz * dz;
        if d2 > max_d2 {
            continue;
        }
        candidates.push((area.id.clone(), area.pos, d2));
    }
    if candidates.is_empty() {
        return None;
    }
    // Sort ascending by distance, then random-jitter pick among
    // the top-3 like base-pool selection. Stable id sort as final
    // tiebreak so two same-distance areas pick deterministically.
    candidates.sort_by(|a, b| {
        a.2.partial_cmp(&b.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let k = candidates.len().min(3);
    let chosen = candidates.swap_remove(rng.gen_range(0..k));
    Some((chosen.0, chosen.1))
}

/// Pick from the `top_n` nearest positions in `pool` to `from`,
/// with uniform random tiebreak so squads don't all converge on
/// exactly one base. Falls back to `pool[0]` if the pool is tiny.
pub(crate) fn pick_nearby(
    rng: &mut ChaCha8Rng,
    from: [f32; 3],
    mut pool: Vec<[f32; 3]>,
    top_n: usize,
) -> [f32; 3] {
    if pool.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    pool.sort_by(|a, b| {
        let da = (a[0] - from[0]).powi(2) + (a[2] - from[2]).powi(2);
        let db = (b[0] - from[0]).powi(2) + (b[2] - from[2]).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = top_n.min(pool.len());
    pool[rng.gen_range(0..k)]
}

pub(crate) fn wander(now: u64) -> SquadObjective {
    SquadObjective::Wander {
        expires_at: now + pcfg().wander_duration_ticks,
    }
}

pub(crate) fn same_faction_or_any_in_region(
    faction: crate::faction::registry::FactionId,
    region: RegionId,
    bases: &Query<(&Base, &InFaction, &InRegion, &Position)>,
) -> Vec<[f32; 3]> {
    let mut same: Vec<[f32; 3]> = Vec::new();
    let mut any: Vec<[f32; 3]> = Vec::new();
    for (_, f, r, p) in bases.iter() {
        if r.0 != region {
            continue;
        }
        any.push(p.0);
        if f.0 == faction {
            same.push(p.0);
        }
    }
    if !same.is_empty() {
        same
    } else {
        any
    }
}

pub(crate) fn objkind_matches_tag(k: ObjKind, t: crate::resources::SquadObjectiveKindTag) -> bool {
    use crate::resources::SquadObjectiveKindTag as T;
    matches!(
        (k, t),
        (ObjKind::Patrol, T::Patrol)
            | (ObjKind::Guard, T::Guard)
            | (ObjKind::Investigate, T::Investigate)
            | (ObjKind::Rest, T::Rest)
            | (ObjKind::Explore, T::Explore)
            | (ObjKind::Wander, T::Wander)
    )
}

pub(crate) fn objective_kind_tag(obj: &SquadObjective) -> crate::resources::SquadObjectiveKindTag {
    use crate::resources::SquadObjectiveKindTag as T;
    match obj {
        SquadObjective::Patrol { .. } => T::Patrol,
        SquadObjective::Guard { .. } => T::Guard,
        SquadObjective::Rest { .. } => T::Rest,
        SquadObjective::Investigate { .. } => T::Investigate,
        SquadObjective::Explore { .. } => T::Explore,
        SquadObjective::Relieve { .. } => T::Relieve,
        SquadObjective::Wander { .. } => T::Wander,
        SquadObjective::Regroup { .. } => T::Regroup,
    }
}

pub(crate) fn position_of_objective(obj: &SquadObjective) -> Option<[f32; 3]> {
    match obj {
        SquadObjective::Patrol {
            route, current_idx, ..
        } => route.get(*current_idx).copied(),
        SquadObjective::Guard { base_pos, .. } => Some(*base_pos),
        SquadObjective::Rest { base_pos, .. } => Some(*base_pos),
        SquadObjective::Investigate { target, .. } => Some(*target),
        SquadObjective::Explore { portal_pos, .. } => Some(*portal_pos),
        SquadObjective::Relieve { dest_pos, .. } => Some(*dest_pos),
        _ => None,
    }
}

pub(crate) fn push_recent(recent: &mut VecDeque<[i32; 3]>, pos: [f32; 3]) {
    let q = quantize(pos);
    if !recent.contains(&q) {
        recent.push_back(q);
        while recent.len() > RECENT_VISITED_CAP {
            recent.pop_front();
        }
    }
}

pub(crate) fn quantize(p: [f32; 3]) -> [i32; 3] {
    [
        (p[0] / RECENT_QUANTIZE_M).round() as i32,
        (p[1] / RECENT_QUANTIZE_M).round() as i32,
        (p[2] / RECENT_QUANTIZE_M).round() as i32,
    ]
}

#[cfg(test)]
mod tests {
    //! Iteration 5-13 Phase D3 unit tests for the rest-area
    //! preference logic. Exercises `pick_rest_area` directly so
    //! we don't have to spin up the full ECS schedule.

    use super::*;
    use crate::resources::{InteractionArea, InteractionAreas};
    use rand::SeedableRng;
    use std::collections::HashMap;

    const REGION: crate::region::RegionId = 1;
    const FACTION_A: crate::faction::registry::FactionId = crate::faction::registry::FactionId(1);
    const FACTION_B: crate::faction::registry::FactionId = crate::faction::registry::FactionId(2);

    fn summary(centroid: [f32; 3], faction: crate::faction::registry::FactionId) -> GroupSummary {
        GroupSummary {
            faction,
            region: REGION,
            member_count: 3,
            any_aggroed: false,
            centroid,
        }
    }

    fn area(
        id: &str,
        pos: [f32; 3],
        faction: Option<crate::faction::registry::FactionId>,
    ) -> InteractionArea {
        InteractionArea {
            id: id.into(),
            kind: "rest".into(),
            pos,
            extents: [1.5, 1.5],
            faction,
            capacity: 1,
            occupants: 0,
            tags: HashMap::new(),
        }
    }

    fn areas_with(items: Vec<InteractionArea>) -> InteractionAreas {
        let mut store = InteractionAreas::default();
        for (idx, a) in items.iter().enumerate() {
            store.by_id.insert(a.id.clone(), (REGION, idx));
        }
        store.by_region.insert(REGION, items);
        store
    }

    #[test]
    fn picks_in_range_rest_area_over_far() {
        let store = areas_with(vec![
            area("near", [10.0, 0.0, 10.0], None),
            area("far", [500.0, 0.0, 500.0], None),
        ]);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let s = summary([0.0, 0.0, 0.0], FACTION_A);
        let chosen = pick_rest_area(&mut rng, &s, &store);
        let id = chosen.expect("in-range area must be picked").0;
        assert_eq!(id, "near", "far area is outside prefer radius");
    }

    #[test]
    fn rejects_full_capacity_area() {
        let mut a = area("full", [10.0, 0.0, 10.0], None);
        a.capacity = 1;
        a.occupants = 1;
        let store = areas_with(vec![a]);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let s = summary([0.0, 0.0, 0.0], FACTION_A);
        assert!(
            pick_rest_area(&mut rng, &s, &store).is_none(),
            "full area must be skipped",
        );
    }

    #[test]
    fn rejects_mismatched_faction() {
        let store = areas_with(vec![area("b_only", [10.0, 0.0, 10.0], Some(FACTION_B))]);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let s = summary([0.0, 0.0, 0.0], FACTION_A);
        assert!(
            pick_rest_area(&mut rng, &s, &store).is_none(),
            "faction filter must reject other-faction squad",
        );
    }

    #[test]
    fn accepts_unrestricted_faction_for_any_squad() {
        let store = areas_with(vec![area("open", [10.0, 0.0, 10.0], None)]);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        for faction in [FACTION_A, FACTION_B] {
            let s = summary([0.0, 0.0, 0.0], faction);
            assert!(
                pick_rest_area(&mut rng, &s, &store).is_some(),
                "unrestricted area must accept any squad",
            );
        }
    }

    #[test]
    fn skips_far_area_outside_radius() {
        let store = areas_with(vec![area(
            "too_far",
            [REST_INTERACTION_AREA_PREFER_RADIUS_M + 50.0, 0.0, 0.0],
            None,
        )]);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
        let s = summary([0.0, 0.0, 0.0], FACTION_A);
        assert!(
            pick_rest_area(&mut rng, &s, &store).is_none(),
            "areas past prefer radius are out of scope",
        );
    }
}
