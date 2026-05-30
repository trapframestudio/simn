//! Combat action library for the GOAP planner. Defines the concrete
//! actions NPCs can take in combat, their preconditions and effects,
//! and per-faction cost overrides that create distinct tactical
//! doctrines (military NPCs favor cover; bandits rush; cautious
//! factions retreat early).

use crate::components::CombatRole;
use crate::goap::*;

/// Build the full combat action set with faction-specific cost
/// modifiers applied.
pub fn combat_actions(faction_name: &str, role: CombatRole) -> Vec<Action> {
    let costs = faction_costs(faction_name);
    let role_mod = role_cost_modifier(role);

    vec![
        Action {
            name: "Shoot",
            preconditions: WorldState(HAS_TARGET | IN_RANGE | HAS_LOS | HAS_AMMO),
            positive_effects: WorldState(TARGET_DEAD),
            negative_effects: WorldState(0),
            cost: costs.shoot * role_mod.shoot,
        },
        Action {
            name: "Advance",
            preconditions: WorldState(HAS_TARGET),
            positive_effects: WorldState(IN_RANGE | HAS_LOS),
            negative_effects: WorldState(IN_COVER),
            cost: costs.advance * role_mod.advance,
        },
        Action {
            name: "MoveToCover",
            preconditions: WorldState(COVER_AVAILABLE),
            positive_effects: WorldState(IN_COVER),
            negative_effects: WorldState(0),
            cost: costs.move_to_cover * role_mod.move_to_cover,
        },
        Action {
            name: "PeekFromCover",
            preconditions: WorldState(IN_COVER | HAS_TARGET),
            positive_effects: WorldState(HAS_LOS | IN_RANGE),
            negative_effects: WorldState(0),
            cost: costs.peek * role_mod.peek,
        },
        Action {
            name: "Flank",
            preconditions: WorldState(HAS_TARGET),
            positive_effects: WorldState(IN_RANGE | HAS_LOS | IS_FLANKING),
            negative_effects: WorldState(IN_COVER),
            cost: costs.flank * role_mod.flank,
        },
        Action {
            name: "Suppress",
            preconditions: WorldState(HAS_TARGET | IN_RANGE | HAS_LOS | HAS_AMMO),
            positive_effects: WorldState(0),
            negative_effects: WorldState(0),
            cost: costs.suppress * role_mod.suppress,
        },
        Action {
            name: "Retreat",
            preconditions: WorldState(0),
            positive_effects: WorldState(AT_SAFE_POS),
            negative_effects: WorldState(IN_RANGE | IN_COVER | HAS_LOS),
            cost: costs.retreat * role_mod.retreat,
        },
        Action {
            name: "Reload",
            preconditions: WorldState(IN_COVER),
            positive_effects: WorldState(HAS_AMMO),
            negative_effects: WorldState(0),
            cost: costs.reload * role_mod.reload,
        },
        Action {
            name: "HealAlly",
            preconditions: WorldState(ALLY_DOWN | NEAR_ALLY),
            positive_effects: WorldState(0),
            negative_effects: WorldState(ALLY_DOWN),
            cost: costs.heal_ally * role_mod.heal_ally,
        },
    ]
}

/// Build the goal set for a given NPC's current situation.
pub fn combat_goals(
    health_frac: f32,
    taking_fire: bool,
    ally_down: bool,
    squad_retreating: bool,
) -> Vec<Goal> {
    let mut goals = Vec::with_capacity(6);

    // Squad-level retreat overrides everything.
    if squad_retreating {
        goals.push(Goal {
            name: "SquadRetreat",
            desired_state: WorldState(AT_SAFE_POS),
            priority: 35,
        });
    }

    // Critical health — flee.
    if health_frac < 0.3 {
        goals.push(Goal {
            name: "Retreat",
            desired_state: WorldState(AT_SAFE_POS),
            priority: 30,
        });
    }

    // Moderate damage — get behind cover and stay there.
    if health_frac < 0.6 {
        goals.push(Goal {
            name: "StayAlive",
            desired_state: WorldState(AT_SAFE_POS | IN_COVER),
            priority: 25,
        });
    }

    if ally_down {
        goals.push(Goal {
            name: "SaveAlly",
            desired_state: WorldState(NEAR_ALLY),
            priority: 22,
        });
    }

    // Taking any fire at all — get to cover NOW, before engaging.
    if taking_fire {
        goals.push(Goal {
            name: "GetToCover",
            desired_state: WorldState(IN_COVER),
            priority: 20,
        });
    }

    // Default combat: fight from cover if possible, direct if not.
    goals.push(Goal {
        name: "EngageFromCover",
        desired_state: WorldState(TARGET_DEAD | IN_COVER),
        priority: 15,
    });

    goals.push(Goal {
        name: "EngageDirect",
        desired_state: WorldState(TARGET_DEAD),
        priority: 8,
    });

    goals
}

struct FactionCosts {
    shoot: f32,
    advance: f32,
    move_to_cover: f32,
    peek: f32,
    flank: f32,
    suppress: f32,
    retreat: f32,
    reload: f32,
    heal_ally: f32,
}

fn faction_costs(faction: &str) -> FactionCosts {
    // Survival-first doctrine: cover is cheap, open-field advance
    // is expensive. Military factions are disciplined (use cover
    // well), bandits are aggressive (rush more), wanderers flee.
    match faction {
        "pwa" | "linemen" => FactionCosts {
            shoot: 1.5,
            advance: 4.5,
            move_to_cover: 0.8,
            peek: 0.8,
            flank: 2.5,
            suppress: 1.5,
            retreat: 5.0,
            reload: 1.5,
            heal_ally: 1.5,
        },
        "federal" | "ghost_teams" | "recovery_division" => FactionCosts {
            shoot: 1.5,
            advance: 4.0,
            move_to_cover: 0.8,
            peek: 0.8,
            flank: 2.0,
            suppress: 1.0,
            retreat: 4.0,
            reload: 1.5,
            heal_ally: 1.5,
        },
        "looters" | "bandits" => FactionCosts {
            shoot: 1.2,
            advance: 2.5,
            move_to_cover: 2.0,
            peek: 2.0,
            flank: 1.5,
            suppress: 3.0,
            retreat: 2.0,
            reload: 2.0,
            heal_ally: 4.0,
        },
        "wanderers" => FactionCosts {
            shoot: 2.0,
            advance: 5.0,
            move_to_cover: 1.0,
            peek: 1.5,
            flank: 4.0,
            suppress: 5.0,
            retreat: 0.8,
            reload: 2.0,
            heal_ally: 3.0,
        },
        "merged" => FactionCosts {
            shoot: 0.5,
            advance: 1.0,
            move_to_cover: 5.0,
            peek: 4.0,
            flank: 1.5,
            suppress: 3.0,
            retreat: 10.0,
            reload: 1.0,
            heal_ally: 5.0,
        },
        _ => FactionCosts {
            shoot: 1.0,
            advance: 3.0,
            move_to_cover: 2.0,
            peek: 1.5,
            flank: 3.0,
            suppress: 2.5,
            retreat: 4.0,
            reload: 2.0,
            heal_ally: 2.5,
        },
    }
}

struct RoleCostMod {
    shoot: f32,
    advance: f32,
    move_to_cover: f32,
    peek: f32,
    flank: f32,
    suppress: f32,
    retreat: f32,
    reload: f32,
    heal_ally: f32,
}

fn role_cost_modifier(role: CombatRole) -> RoleCostMod {
    match role {
        CombatRole::Pointman => RoleCostMod {
            shoot: 0.8,
            advance: 0.6,
            move_to_cover: 1.5,
            peek: 1.2,
            flank: 0.8,
            suppress: 1.0,
            retreat: 2.0,
            reload: 1.0,
            heal_ally: 2.0,
        },
        CombatRole::Support => RoleCostMod {
            shoot: 0.9,
            advance: 1.5,
            move_to_cover: 0.7,
            peek: 0.8,
            flank: 1.5,
            suppress: 0.6,
            retreat: 1.2,
            reload: 0.8,
            heal_ally: 1.5,
        },
        CombatRole::Flanker => RoleCostMod {
            shoot: 1.0,
            advance: 0.8,
            move_to_cover: 1.0,
            peek: 1.0,
            flank: 0.5,
            suppress: 1.5,
            retreat: 1.0,
            reload: 1.0,
            heal_ally: 2.0,
        },
        CombatRole::Medic => RoleCostMod {
            shoot: 1.5,
            advance: 1.5,
            move_to_cover: 0.8,
            peek: 1.2,
            flank: 2.0,
            suppress: 2.0,
            retreat: 0.7,
            reload: 1.0,
            heal_ally: 0.3,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn military_prefers_cover_over_advance() {
        let actions = combat_actions("pwa", CombatRole::Support);
        let cover_cost = actions
            .iter()
            .find(|a| a.name == "MoveToCover")
            .unwrap()
            .cost;
        let advance_cost = actions.iter().find(|a| a.name == "Advance").unwrap().cost;
        assert!(
            cover_cost < advance_cost,
            "Military support should prefer cover ({cover_cost}) over advance ({advance_cost})"
        );
    }

    #[test]
    fn bandits_prefer_advance_over_cover() {
        let actions = combat_actions("bandits", CombatRole::Pointman);
        let cover_cost = actions
            .iter()
            .find(|a| a.name == "MoveToCover")
            .unwrap()
            .cost;
        let advance_cost = actions.iter().find(|a| a.name == "Advance").unwrap().cost;
        assert!(
            advance_cost < cover_cost,
            "Bandit pointman should prefer advance ({advance_cost}) over cover ({cover_cost})"
        );
    }

    #[test]
    fn medic_prioritizes_healing() {
        let actions = combat_actions("federal", CombatRole::Medic);
        let heal_cost = actions.iter().find(|a| a.name == "HealAlly").unwrap().cost;
        let shoot_cost = actions.iter().find(|a| a.name == "Shoot").unwrap().cost;
        assert!(
            heal_cost < shoot_cost,
            "Medic should prefer heal ({heal_cost}) over shoot ({shoot_cost})"
        );
    }

    #[test]
    fn retreat_goal_outranks_kill_when_squad_retreating() {
        let goals = combat_goals(1.0, false, false, true);
        assert_eq!(goals[0].name, "SquadRetreat");
        assert!(goals[0].priority > goals.last().unwrap().priority);
    }

    #[test]
    fn full_plan_with_faction_costs() {
        let state = WorldState(HAS_TARGET | COVER_AVAILABLE | HAS_AMMO);
        let goals = combat_goals(1.0, false, false, false);
        let actions = combat_actions("pwa", CombatRole::Support);
        let result = crate::goap::plan(state, &goals, &actions, 8);
        assert!(result.is_some(), "PWA support should find a plan");
        let p = result.unwrap();
        assert!(!p.actions.is_empty());
    }
}
