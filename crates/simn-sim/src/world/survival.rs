//! Survival meters, contamination, and consumption API on `Sim`.
//!
//! Three concerns bundled here because they share the
//! [`ConsumeProfile`] pathway — food/drink call `consume` (survival
//! deltas) and may also call `add_radiation` / `add_toxicity`:
//!
//! - **Survival meters** — `set_survival_stat`, `consume` (add + clamp).
//! - **Contamination** — `set_radiation`, `add_radiation`,
//!   `set_toxicity`, `add_toxicity`.
//! - **Consumables** — `eat`, `drink`, plus the `ConsumeProfile`
//!   struct and per-kind `food_profile` / `water_profile` tables.
//!
//! All mutations journal `SetSurvivalStat` / `RadiationChanged` /
//! `ToxicityChanged` deltas. Per-tick drain (`drain_survival_stats`)
//! and passive decay (`tick_contamination`) happen in the tick
//! schedule and are pure.

use anyhow::Result;

use crate::components::{
    Contamination, DrugKind, FoodKind, SurvivalStat, SurvivalStats, WaterKind,
};
use crate::delta::WorldDelta;

use super::Sim;

impl Sim {
    /// Eat a food item — applies the kind's profile (hunger/thirst/
    /// fatigue restore + rad/tox contamination + optional granted
    /// effect like an EnergyDrink's Stim). String-keyed via FoodKind;
    /// proper inventory items wrap this in Step 4.
    pub fn eat(&mut self, steam_id: u64, kind: FoodKind) -> Result<()> {
        let p = food_profile(kind);
        self.consume(steam_id, p.hunger_d, p.thirst_d, p.fatigue_d)?;
        if p.rad_add != 0.0 {
            self.add_radiation(steam_id, p.rad_add)?;
        }
        if p.tox_add != 0.0 {
            self.add_toxicity(steam_id, p.tox_add)?;
        }
        if let Some(d) = p.grants_drug {
            let _ = self.apply_drug(steam_id, d)?;
        }
        Ok(())
    }

    /// Drink a water/beverage item — same shape as `eat`.
    pub fn drink(&mut self, steam_id: u64, kind: WaterKind) -> Result<()> {
        let p = water_profile(kind);
        self.consume(steam_id, p.hunger_d, p.thirst_d, p.fatigue_d)?;
        if p.rad_add != 0.0 {
            self.add_radiation(steam_id, p.rad_add)?;
        }
        if p.tox_add != 0.0 {
            self.add_toxicity(steam_id, p.tox_add)?;
        }
        if let Some(d) = p.grants_drug {
            let _ = self.apply_drug(steam_id, d)?;
        }
        Ok(())
    }

    /// Set a player's radiation to a specific value, clamped to
    /// `[0, 100]`. Journaled. Use [`Self::add_radiation`] for delta
    /// updates.
    pub fn set_radiation(&mut self, steam_id: u64, value: f32) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let new = {
            let mut c = self
                .world
                .get_mut::<Contamination>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Contamination"))?;
            c.radiation = value.clamp(0.0, 100.0);
            c.radiation
        };
        self.record_delta(WorldDelta::RadiationChanged {
            steam_id,
            value: new,
        })?;
        Ok(())
    }

    /// Add (or subtract) radiation; clamps to `[0, 100]`. Journaled.
    pub fn add_radiation(&mut self, steam_id: u64, delta: f32) -> Result<()> {
        let cur = self
            .find_player_entity(steam_id)
            .and_then(|e| self.world.get::<Contamination>(e))
            .map(|c| c.radiation)
            .unwrap_or(0.0);
        self.set_radiation(steam_id, cur + delta)
    }

    /// Mirror of [`Self::set_radiation`] for toxicity.
    pub fn set_toxicity(&mut self, steam_id: u64, value: f32) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let new = {
            let mut c = self
                .world
                .get_mut::<Contamination>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no Contamination"))?;
            c.toxicity = value.clamp(0.0, 100.0);
            c.toxicity
        };
        self.record_delta(WorldDelta::ToxicityChanged {
            steam_id,
            value: new,
        })?;
        Ok(())
    }

    /// Add (or subtract) toxicity; clamps to `[0, 100]`. Journaled.
    pub fn add_toxicity(&mut self, steam_id: u64, delta: f32) -> Result<()> {
        let cur = self
            .find_player_entity(steam_id)
            .and_then(|e| self.world.get::<Contamination>(e))
            .map(|c| c.toxicity)
            .unwrap_or(0.0);
        self.set_toxicity(steam_id, cur + delta)
    }

    /// Set a survival meter (hunger / thirst / fatigue) to a specific
    /// value. Clamped to `[0, SurvivalStats::FULL]`. Journaled.
    pub fn set_survival_stat(
        &mut self,
        steam_id: u64,
        stat: SurvivalStat,
        value: f32,
    ) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let new = {
            let mut s = self
                .world
                .get_mut::<SurvivalStats>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no SurvivalStats"))?;
            let slot = s.get_mut(stat);
            *slot = value.clamp(0.0, SurvivalStats::FULL);
            *slot
        };
        self.record_delta(WorldDelta::SetSurvivalStat {
            steam_id,
            stat,
            current: new,
        })?;
        Ok(())
    }

    /// Restore survival meters (e.g. eating food, drinking water,
    /// resting). Each delta is added then clamped to
    /// `[0, SurvivalStats::FULL]`. Emits one `SetSurvivalStat` record
    /// per non-zero delta.
    pub fn consume(
        &mut self,
        steam_id: u64,
        hunger_delta: f32,
        thirst_delta: f32,
        fatigue_delta: f32,
    ) -> Result<()> {
        let Some(e) = self.find_player_entity(steam_id) else {
            return Err(anyhow::anyhow!("unknown player {steam_id}"));
        };
        let (new_hunger, new_thirst, new_fatigue) = {
            let mut s = self
                .world
                .get_mut::<SurvivalStats>(e)
                .ok_or_else(|| anyhow::anyhow!("player {steam_id} has no SurvivalStats"))?;
            s.hunger = (s.hunger + hunger_delta).clamp(0.0, SurvivalStats::FULL);
            s.thirst = (s.thirst + thirst_delta).clamp(0.0, SurvivalStats::FULL);
            s.fatigue = (s.fatigue + fatigue_delta).clamp(0.0, SurvivalStats::FULL);
            (s.hunger, s.thirst, s.fatigue)
        };
        if hunger_delta != 0.0 {
            self.record_delta(WorldDelta::SetSurvivalStat {
                steam_id,
                stat: SurvivalStat::Hunger,
                current: new_hunger,
            })?;
        }
        if thirst_delta != 0.0 {
            self.record_delta(WorldDelta::SetSurvivalStat {
                steam_id,
                stat: SurvivalStat::Thirst,
                current: new_thirst,
            })?;
        }
        if fatigue_delta != 0.0 {
            self.record_delta(WorldDelta::SetSurvivalStat {
                steam_id,
                stat: SurvivalStat::Fatigue,
                current: new_fatigue,
            })?;
        }
        Ok(())
    }
}

/// Per-consumable nutritional + contamination + effect profile.
/// Returned by [`food_profile`] and [`water_profile`]. Item
/// definitions (`crates/simn-sim/content/items.toml`) reference a
/// FoodKind/WaterKind and pull the same profile via `Sim::eat` /
/// `Sim::drink`.
#[derive(Debug, Clone, Copy)]
pub struct ConsumeProfile {
    pub hunger_d: f32,
    pub thirst_d: f32,
    pub fatigue_d: f32,
    pub rad_add: f32,
    pub tox_add: f32,
    /// Optional effect granted on consumption (e.g. EnergyDrink → Stim).
    pub grants_drug: Option<DrugKind>,
}

impl ConsumeProfile {
    const fn empty() -> Self {
        Self {
            hunger_d: 0.0,
            thirst_d: 0.0,
            fatigue_d: 0.0,
            rad_add: 0.0,
            tox_add: 0.0,
            grants_drug: None,
        }
    }
}

/// Profile for each [`FoodKind`]. Numbers are tuned for the Step 3
/// "intermediate-friendly" dial — see
/// `docs/book/src/mechanics/food-and-water.md`.
pub fn food_profile(kind: FoodKind) -> ConsumeProfile {
    match kind {
        FoodKind::PreservedRation => ConsumeProfile {
            hunger_d: 30.0,
            thirst_d: -5.0,
            ..ConsumeProfile::empty()
        },
        FoodKind::FreshFood => ConsumeProfile {
            hunger_d: 35.0,
            ..ConsumeProfile::empty()
        },
        FoodKind::RawMeat => ConsumeProfile {
            hunger_d: 20.0,
            tox_add: 20.0,
            ..ConsumeProfile::empty()
        },
        FoodKind::CookedMeat => ConsumeProfile {
            hunger_d: 40.0,
            ..ConsumeProfile::empty()
        },
        FoodKind::ContaminatedFood => ConsumeProfile {
            hunger_d: 30.0,
            rad_add: 30.0,
            tox_add: 20.0,
            ..ConsumeProfile::empty()
        },
        FoodKind::FieldRation => ConsumeProfile {
            hunger_d: 50.0,
            thirst_d: -5.0,
            ..ConsumeProfile::empty()
        },
        FoodKind::EnergyBar => ConsumeProfile {
            hunger_d: 15.0,
            ..ConsumeProfile::empty()
        },
    }
}

/// Profile for each [`WaterKind`].
pub fn water_profile(kind: WaterKind) -> ConsumeProfile {
    match kind {
        WaterKind::DirtyWater => ConsumeProfile {
            thirst_d: 50.0,
            rad_add: 15.0,
            tox_add: 10.0,
            ..ConsumeProfile::empty()
        },
        WaterKind::CleanWater => ConsumeProfile {
            thirst_d: 60.0,
            ..ConsumeProfile::empty()
        },
        WaterKind::EnergyDrink => ConsumeProfile {
            thirst_d: 40.0,
            grants_drug: Some(DrugKind::StimCocktail),
            ..ConsumeProfile::empty()
        },
        WaterKind::Vodka => ConsumeProfile {
            thirst_d: 20.0,
            rad_add: -5.0,
            tox_add: 5.0,
            ..ConsumeProfile::empty()
        },
    }
}
