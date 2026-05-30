//! Utility-scoring objective picker. Per `npc-ai-plan.md` umbrella
//! item #10 — squad_planner picks objectives by utility score
//! (faction base × personality fractions × blackboard signals)
//! instead of pure weighted-random.

use simn_sim::systems::{
    objective_utility, squad_planner, BlackboardSignals, ObjKind, SquadPersonality,
};
// Reference the system function so its public type signature is exercised
// by the dependency graph, even though we don't call it directly here.
const _: () = {
    let _ = squad_planner;
};

fn baseline_personality() -> SquadPersonality {
    SquadPersonality::default()
}

#[test]
fn empty_personality_no_signals_uses_base_weight() {
    // With all-zero personality + no blackboard signals, utility ≈
    // base × 1.0 × 1.0 ... so the order-by-utility matches order
    // by base weight.
    let p = baseline_personality();
    let bb = BlackboardSignals::default();
    let u_low = objective_utility(ObjKind::Patrol, 1, &p, &bb, true);
    let u_high = objective_utility(ObjKind::Patrol, 5, &p, &bb, true);
    assert!(u_high > u_low);
    // Different kinds with the same base should score the same with
    // empty personality (no signals).
    let a = objective_utility(ObjKind::Patrol, 3, &p, &bb, true);
    let b = objective_utility(ObjKind::Investigate, 3, &p, &bb, true);
    // Both should be near the base × 1.0 multiplier, but the patrol
    // path applies the loyal/disciplined zero-mults too. Empty
    // personality → 1.0 mult → both equal ish. Allow a 5% delta
    // for trait-table cross-talk.
    assert!((a - b).abs() / a.max(b) < 0.05);
}

#[test]
fn disciplined_squad_prefers_guard_over_wander() {
    let mut p = baseline_personality();
    p.disciplined = 1.0;
    p.loyal = 0.5;
    let bb = BlackboardSignals::default();
    let guard = objective_utility(ObjKind::Guard, 3, &p, &bb, true);
    let wander = objective_utility(ObjKind::Wander, 3, &p, &bb, true);
    assert!(
        guard > wander,
        "disciplined → guard should outrank wander: {} vs {}",
        guard,
        wander
    );
}

#[test]
fn curious_squad_prefers_investigate_over_rest() {
    let mut p = baseline_personality();
    p.curious = 1.0;
    let bb = BlackboardSignals::default();
    let inv = objective_utility(ObjKind::Investigate, 3, &p, &bb, true);
    let rest = objective_utility(ObjKind::Rest, 3, &p, &bb, true);
    assert!(inv > rest);
}

#[test]
fn heard_gunshot_strongly_boosts_investigate() {
    let p = baseline_personality();
    let bb_quiet = BlackboardSignals::default();
    let mut bb_loud = BlackboardSignals::default();
    bb_loud.heard_gunshot = true;
    let quiet = objective_utility(ObjKind::Investigate, 3, &p, &bb_quiet, true);
    let loud = objective_utility(ObjKind::Investigate, 3, &p, &bb_loud, true);
    assert!(
        loud > quiet * 1.4,
        "heard_gunshot should boost Investigate ≥ 40%: {} vs {}",
        loud,
        quiet
    );
}

#[test]
fn under_fire_dampens_rest_and_explore() {
    let p = baseline_personality();
    let bb_quiet = BlackboardSignals::default();
    let mut bb_war = BlackboardSignals::default();
    bb_war.under_fire = true;
    let rest_quiet = objective_utility(ObjKind::Rest, 3, &p, &bb_quiet, true);
    let rest_war = objective_utility(ObjKind::Rest, 3, &p, &bb_war, true);
    let explore_quiet = objective_utility(ObjKind::Explore, 3, &p, &bb_quiet, true);
    let explore_war = objective_utility(ObjKind::Explore, 3, &p, &bb_war, true);
    assert!(rest_war < rest_quiet * 0.5);
    assert!(explore_war < explore_quiet * 0.6);
}

#[test]
fn no_territorial_standing_dampens_guard() {
    let p = baseline_personality();
    let bb = BlackboardSignals::default();
    let with_standing = objective_utility(ObjKind::Guard, 3, &p, &bb, true);
    let without = objective_utility(ObjKind::Guard, 3, &p, &bb, false);
    assert!(without < with_standing * 0.5);
}

#[test]
fn aggressive_squad_avoids_rest() {
    let mut p = baseline_personality();
    p.aggressive = 1.0;
    let bb = BlackboardSignals::default();
    let aggro_rest = objective_utility(ObjKind::Rest, 3, &p, &bb, true);
    let baseline_rest =
        objective_utility(ObjKind::Rest, 3, &SquadPersonality::default(), &bb, true);
    assert!(aggro_rest < baseline_rest * 0.7);
}

#[test]
fn solitary_squad_prefers_wander_explore_over_patrol() {
    let mut p = baseline_personality();
    p.solitary = 1.0;
    let bb = BlackboardSignals::default();
    let wander = objective_utility(ObjKind::Wander, 3, &p, &bb, true);
    let patrol = objective_utility(ObjKind::Patrol, 3, &p, &bb, true);
    assert!(wander > patrol);
}

#[test]
fn zero_base_weight_yields_zero_utility() {
    let p = baseline_personality();
    let bb = BlackboardSignals::default();
    let u = objective_utility(ObjKind::Guard, 0, &p, &bb, true);
    assert_eq!(u, 0.0);
}

#[test]
fn amplified_curious_dominates_investigate() {
    // Phase A: with tripled personality multipliers, a 100% curious
    // squad on Investigate should score well above the same squad's
    // utility for Rest — even with equal base weights — by at least
    // 2x. The old tuning only produced ~1.5x; that's not visible in
    // playtest.
    let mut p = baseline_personality();
    p.curious = 1.0;
    let bb = BlackboardSignals::default();
    let inv = objective_utility(ObjKind::Investigate, 3, &p, &bb, true);
    let rest = objective_utility(ObjKind::Rest, 3, &p, &bb, true);
    assert!(
        inv > rest * 2.0,
        "curious should dominate investigate vs rest by 2x+: {inv} vs {rest}"
    );
}

#[test]
fn amplified_aggressive_zeros_rest() {
    // A fully aggressive squad should have effectively zero utility
    // on Rest — the damping factor (1 - 1.20 * 1.0) goes negative and
    // is clamped to zero by the final mult.max(0.0). Intentional: a
    // 100% aggressive squad never picks Rest.
    let mut p = baseline_personality();
    p.aggressive = 1.0;
    let bb = BlackboardSignals::default();
    let rest = objective_utility(ObjKind::Rest, 3, &p, &bb, true);
    assert!(
        rest <= 0.0001,
        "aggressive Rest should collapse to 0: {rest}"
    );
}

#[test]
fn amplified_solitary_strongly_avoids_patrol() {
    // Solitary squads should clearly prefer Wander/Explore over
    // Patrol — by a factor of at least 3x with the new tuning, since
    // solitary now drops Patrol mult by 60% and boosts Wander by 90%.
    let mut p = baseline_personality();
    p.solitary = 1.0;
    let bb = BlackboardSignals::default();
    let wander = objective_utility(ObjKind::Wander, 3, &p, &bb, true);
    let patrol = objective_utility(ObjKind::Patrol, 3, &p, &bb, true);
    assert!(
        wander > patrol * 3.0,
        "solitary Wander should dominate Patrol by 3x: {wander} vs {patrol}"
    );
}

#[test]
fn partial_personality_scales_continuously() {
    // Scoring should respond proportionally to the trait fraction
    // — 50% disciplined yields ~half the boost of 100%. Catches a
    // step-function regression where the math accidentally treats
    // the fraction as a bool threshold.
    let bb = BlackboardSignals::default();
    let mut p_full = SquadPersonality::default();
    p_full.disciplined = 1.0;
    let mut p_half = SquadPersonality::default();
    p_half.disciplined = 0.5;
    let p_zero = SquadPersonality::default();
    let full = objective_utility(ObjKind::Patrol, 3, &p_full, &bb, true);
    let half = objective_utility(ObjKind::Patrol, 3, &p_half, &bb, true);
    let zero = objective_utility(ObjKind::Patrol, 3, &p_zero, &bb, true);
    assert!(full > half);
    assert!(half > zero);
}
