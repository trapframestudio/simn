//! Goal-Oriented Action Planning (GOAP) — F.E.A.R.-style planner.
//!
//! World state is a compact bitfield of boolean conditions. Actions
//! have preconditions (required bits), effects (bits set/cleared),
//! and a cost. The planner searches backward from the goal state
//! via A* to find the cheapest sequence of actions that transforms
//! the current world state into the goal state.
//!
//! Engine-agnostic — no bevy dependency. The `npc_tactical` system
//! builds a `WorldState` per NPC per tick, runs the planner when
//! the current plan is empty or invalidated, and maps plan steps
//! to `CombatStance` transitions.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// World state as a 32-bit bitmask. Each bit represents a boolean
/// condition the planner reasons about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldState(pub u32);

impl WorldState {
    pub fn has(self, flag: u32) -> bool {
        self.0 & flag != 0
    }

    pub fn set(self, flag: u32) -> Self {
        Self(self.0 | flag)
    }

    pub fn clear(self, flag: u32) -> Self {
        Self(self.0 & !flag)
    }

    pub fn satisfies(self, required: WorldState) -> bool {
        self.0 & required.0 == required.0
    }

    pub fn distance(self, goal: WorldState) -> u32 {
        (self.0 ^ goal.0).count_ones()
    }
}

// World state flags — each is a single bit
pub const HAS_TARGET: u32 = 1 << 0;
pub const IN_RANGE: u32 = 1 << 1;
pub const IN_COVER: u32 = 1 << 2;
pub const HAS_LOS: u32 = 1 << 3;
pub const TARGET_DEAD: u32 = 1 << 4;
pub const IS_SUPPRESSED: u32 = 1 << 5;
pub const HEALTH_LOW: u32 = 1 << 6;
pub const AT_SAFE_POS: u32 = 1 << 7;
pub const IS_RELOADING: u32 = 1 << 8;
pub const HAS_AMMO: u32 = 1 << 9;
pub const ALLY_DOWN: u32 = 1 << 10;
pub const NEAR_ALLY: u32 = 1 << 11;
pub const SQUAD_RETREATING: u32 = 1 << 12;
pub const IS_FLANKING: u32 = 1 << 13;
pub const COVER_AVAILABLE: u32 = 1 << 14;
pub const TAKING_FIRE: u32 = 1 << 15;

/// A GOAP action with preconditions, effects, and cost.
#[derive(Clone, Debug)]
pub struct Action {
    pub name: &'static str,
    pub preconditions: WorldState,
    pub positive_effects: WorldState,
    pub negative_effects: WorldState,
    pub cost: f32,
}

impl Action {
    pub fn can_run(&self, state: WorldState) -> bool {
        state.satisfies(self.preconditions)
    }

    pub fn apply(&self, state: WorldState) -> WorldState {
        let with_pos = WorldState(state.0 | self.positive_effects.0);
        WorldState(with_pos.0 & !self.negative_effects.0)
    }
}

/// A goal the planner tries to achieve.
#[derive(Clone, Debug)]
pub struct Goal {
    pub name: &'static str,
    pub desired_state: WorldState,
    pub priority: u8,
}

/// A planned sequence of actions to execute.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub actions: Vec<&'static str>,
    pub total_cost: f32,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn current_action(&self) -> Option<&'static str> {
        self.actions.first().copied()
    }

    pub fn advance(&mut self) {
        if !self.actions.is_empty() {
            self.actions.remove(0);
        }
    }
}

#[derive(Clone, Debug)]
struct SearchNode {
    state: WorldState,
    cost: f32,
    heuristic: f32,
    actions: Vec<&'static str>,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.heuristic == other.heuristic
    }
}

impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        let self_f = self.cost + self.heuristic;
        let other_f = other.cost + other.heuristic;
        other_f.partial_cmp(&self_f).unwrap_or(Ordering::Equal)
    }
}

/// Run the GOAP planner. Searches backward from goals through
/// available actions to find the cheapest plan that transforms
/// `current_state` into a state satisfying the highest-priority
/// goal. Returns `None` if no plan can be found.
///
/// `max_depth` caps the search to prevent runaway on large action
/// sets. 8 is plenty for tactical combat (most plans are 2-4 steps).
pub fn plan(
    current_state: WorldState,
    goals: &[Goal],
    actions: &[Action],
    max_depth: usize,
) -> Option<Plan> {
    let mut sorted_goals: Vec<&Goal> = goals.iter().collect();
    sorted_goals.sort_by(|a, b| b.priority.cmp(&a.priority));

    for goal in sorted_goals {
        if current_state.satisfies(goal.desired_state) {
            continue;
        }
        if let Some(p) = search(current_state, goal.desired_state, actions, max_depth) {
            return Some(p);
        }
    }
    None
}

fn search(
    start: WorldState,
    goal: WorldState,
    actions: &[Action],
    max_depth: usize,
) -> Option<Plan> {
    let mut open = BinaryHeap::new();
    let mut visited = std::collections::HashSet::new();

    open.push(SearchNode {
        state: start,
        cost: 0.0,
        heuristic: start.distance(goal) as f32,
        actions: Vec::new(),
    });

    while let Some(node) = open.pop() {
        if node.state.satisfies(goal) {
            return Some(Plan {
                actions: node.actions,
                total_cost: node.cost,
            });
        }

        if node.actions.len() >= max_depth {
            continue;
        }

        if !visited.insert(node.state) {
            continue;
        }

        for action in actions {
            if !action.can_run(node.state) {
                continue;
            }
            let new_state = action.apply(node.state);
            if visited.contains(&new_state) {
                continue;
            }
            let new_cost = node.cost + action.cost;
            let mut new_actions = node.actions.clone();
            new_actions.push(action.name);
            open.push(SearchNode {
                state: new_state,
                cost: new_cost,
                heuristic: new_state.distance(goal) as f32,
                actions: new_actions,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_actions() -> Vec<Action> {
        vec![
            Action {
                name: "MoveToCover",
                preconditions: WorldState(COVER_AVAILABLE),
                positive_effects: WorldState(IN_COVER),
                negative_effects: WorldState(0),
                cost: 2.0,
            },
            Action {
                name: "Shoot",
                preconditions: WorldState(HAS_TARGET | IN_RANGE | HAS_LOS | HAS_AMMO),
                positive_effects: WorldState(TARGET_DEAD),
                negative_effects: WorldState(0),
                cost: 1.0,
            },
            Action {
                name: "Advance",
                preconditions: WorldState(HAS_TARGET),
                positive_effects: WorldState(IN_RANGE | HAS_LOS),
                negative_effects: WorldState(IN_COVER),
                cost: 3.0,
            },
            Action {
                name: "PeekFromCover",
                preconditions: WorldState(IN_COVER | HAS_TARGET),
                positive_effects: WorldState(HAS_LOS | IN_RANGE),
                negative_effects: WorldState(0),
                cost: 1.5,
            },
            Action {
                name: "Retreat",
                preconditions: WorldState(0),
                positive_effects: WorldState(AT_SAFE_POS),
                negative_effects: WorldState(IN_RANGE | IN_COVER | HAS_LOS),
                cost: 4.0,
            },
            Action {
                name: "Reload",
                preconditions: WorldState(IN_COVER),
                positive_effects: WorldState(HAS_AMMO),
                negative_effects: WorldState(IS_RELOADING),
                cost: 2.0,
            },
            Action {
                name: "Flank",
                preconditions: WorldState(HAS_TARGET | COVER_AVAILABLE),
                positive_effects: WorldState(IN_RANGE | HAS_LOS | IS_FLANKING),
                negative_effects: WorldState(IN_COVER),
                cost: 4.0,
            },
        ]
    }

    #[test]
    fn finds_simple_kill_plan() {
        let state = WorldState(HAS_TARGET | IN_RANGE | HAS_LOS | HAS_AMMO);
        let goals = vec![Goal {
            name: "KillTarget",
            desired_state: WorldState(TARGET_DEAD),
            priority: 10,
        }];
        let actions = test_actions();
        let result = plan(state, &goals, &actions, 8).unwrap();
        assert_eq!(result.actions, vec!["Shoot"]);
    }

    #[test]
    fn prefers_cover_route_when_cheaper() {
        let state = WorldState(HAS_TARGET | COVER_AVAILABLE | HAS_AMMO);
        let goals = vec![Goal {
            name: "KillFromCover",
            desired_state: WorldState(TARGET_DEAD | IN_COVER),
            priority: 10,
        }];
        let actions = test_actions();
        let result = plan(state, &goals, &actions, 8).unwrap();
        assert!(result.actions.contains(&"MoveToCover"));
        assert!(result.actions.contains(&"Shoot"));
    }

    #[test]
    fn retreat_when_health_low() {
        let state = WorldState(HAS_TARGET | HEALTH_LOW);
        let goals = vec![
            Goal {
                name: "StayAlive",
                desired_state: WorldState(AT_SAFE_POS),
                priority: 20,
            },
            Goal {
                name: "KillTarget",
                desired_state: WorldState(TARGET_DEAD),
                priority: 10,
            },
        ];
        let actions = test_actions();
        let result = plan(state, &goals, &actions, 8).unwrap();
        assert_eq!(result.actions, vec!["Retreat"]);
    }

    #[test]
    fn reload_behind_cover() {
        let state = WorldState(HAS_TARGET | IN_COVER);
        let goals = vec![Goal {
            name: "KillTarget",
            desired_state: WorldState(TARGET_DEAD),
            priority: 10,
        }];
        let actions = test_actions();
        let result = plan(state, &goals, &actions, 8).unwrap();
        assert!(result.actions.contains(&"Reload"));
        assert!(result.actions.contains(&"Shoot"));
    }

    #[test]
    fn no_plan_returns_none() {
        let state = WorldState(0);
        let goals = vec![Goal {
            name: "KillTarget",
            desired_state: WorldState(TARGET_DEAD),
            priority: 10,
        }];
        let actions = vec![];
        let result = plan(state, &goals, &actions, 8);
        assert!(result.is_none());
    }

    #[test]
    fn flank_when_available() {
        let state = WorldState(HAS_TARGET | COVER_AVAILABLE | HAS_AMMO);
        let goals = vec![Goal {
            name: "KillTarget",
            desired_state: WorldState(TARGET_DEAD | IS_FLANKING),
            priority: 10,
        }];
        let actions = test_actions();
        let result = plan(state, &goals, &actions, 8).unwrap();
        assert!(result.actions.contains(&"Flank"));
    }

    #[test]
    fn world_state_operations() {
        let s = WorldState(0);
        let s = s.set(HAS_TARGET).set(IN_RANGE);
        assert!(s.has(HAS_TARGET));
        assert!(s.has(IN_RANGE));
        assert!(!s.has(IN_COVER));
        let s = s.clear(IN_RANGE);
        assert!(!s.has(IN_RANGE));
        assert!(s.satisfies(WorldState(HAS_TARGET)));
        assert!(!s.satisfies(WorldState(HAS_TARGET | IN_RANGE)));
    }
}
