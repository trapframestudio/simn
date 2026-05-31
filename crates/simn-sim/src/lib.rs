//! simn-sim — engine-agnostic survival-sim world simulation.
//!
//! Server-authoritative, engine-agnostic world simulation built on
//! `bevy_ecs`. The core mechanic (eventually) is two-tier fidelity:
//! entities near players run with full detail (online tier), entities
//! elsewhere in the world run as an abstract graph-based simulation
//! (offline tier). Both tiers run continuously on the server, and
//! entities transition between them as players move through the world.
//!
//! This crate currently provides the foundation:
//!
//! - An ECS-based data model for players + regions
//! - A fixed-rate tick loop with a `bevy_ecs::Schedule`
//! - Journal-then-snapshot persistence (atomic, crash-tolerant)
//!
//! NPCs spawn, wander between bases, occasionally migrate to a
//! neighbor region, and die of natural causes; every NPC the world
//! has ever produced lives on in the `LifeChronicle`. The
//! online/offline tier hand-off (fidelity split) is still ahead and
//! plugs into this same data model when it lands.

pub mod action;
pub mod behavior_config;
pub mod chronicle;
pub mod components;
pub mod content;
pub mod cover;
pub mod delta;
pub mod det_serde;
pub mod faction;
pub mod goap;
pub mod goap_actions;
pub mod helpers;
pub mod inventory_grid;
pub mod items;
pub mod loot_containers;
pub mod loot_pools;
pub mod los_cache;
pub mod names;
pub mod nav;
pub mod npc_loadouts;
pub mod offline_tier;
pub mod patrol_zone;
pub mod pda_log;
pub mod perception;
pub mod persistence;
pub mod poi;
pub mod region;
pub mod resources;
pub mod snapshot;
pub mod squad_blackboard;
pub mod systems;
pub mod worker;
mod world;
pub mod world_event_bus;
pub mod world_seed;

pub use action::{decode_action, encode_action, ActionKind};
pub use behavior_config::{BehaviorConfig, WorldTimeConfig};
pub use chronicle::{
    death_cause_to_str, ChronicleSummary, DeathCause, FactionStats, LifeChronicle, LifeRecord,
};
pub use components::{
    ActiveEffect, ActiveEffects, ActiveGoal, Actor, ActorKind, Aggression, Aggro, AttackerHit,
    Base, BaseKind, BodyPart, BodyParts, CharacterId, ContainerId, Contamination, CraftJob,
    CraftingQueue, DrugKind, DrugTolerance, EffectId, EffectKind, Equipment, EquippedItem,
    EquippedWeaponState, FoodKind, GoalKind, GoalSource, GridInventory, Group, Health, InFaction,
    InRegion, Inventory, ItemInstance, ItemRotation, JamState, LastDamager, Lifespan, LimbState,
    LimbStates, MagazineState, NearCampfire, NearWorkbench, Npc, NpcCharacter, NpcGoal, NpcId,
    NpcRank, NpcStats, Pain, Path, PersonalityArchetype, PersonalityTraits, PlacedItem,
    PlayerOwned, Position, Projectile, ProjectileId, RecentAttackers, Rotation, Stamina,
    SurvivalStat, SurvivalStats, WaterKind, WorldContainer, Wound, WoundId, WoundKind,
    WoundTreatment, Wounds,
};
pub use content::{ContentError, ContentSource};
pub use delta::WorldDelta;
pub use faction::registry::{
    faction_relation as registry_faction_relation,
    faction_relation_score as registry_faction_relation_score,
    load_default as load_default_faction_registry, load_from_path as load_faction_registry,
    load_from_str as load_faction_registry_from_str, player_relation as registry_player_relation,
    player_relation_score as registry_player_relation_score,
    shift_faction_relation as registry_shift_faction_relation,
    shift_player_rep as registry_shift_player_rep, CombatCosts, FactionDef, FactionId,
    FactionRegistry, PlayerReputation, RegistryError, RelationDeltas, SquadSize,
};
pub use faction::{anchor_score, band_from_score, relation_from_str, relation_to_str, Relation};
pub use helpers::{
    base_kind_to_str, moon_phase_name, quantize_post_pos, weather_from_str, weather_to_str,
};
pub use inventory_grid::{PlaceOutcome, PlacementError};
pub use items::{
    jam_chance_at_condition, validate_attachment_chain, AmmoConfig, AmmoVariant, ArmorConfig,
    AttachmentConfig, AttachmentError, Caliber, ConsumeAction, CraftStation, EquipmentSlotDef,
    EquipmentSlotRegistry, GridSize, ItemCategory, ItemDef, ItemId, ItemRegistry, ItemStack,
    KitRequirement, MagazineConfig, Recipe, RecipeContext, RecipeRegistry, SalvageOutput,
    SalvageRecipe, SlotId, SlotPosition, Specialty, ToolSpec, ToolTier, WeaponConfig, WeaponSlot,
};
pub use los_cache::{clear_los_cache, LosCache, LosEntry};

// Iteration 5-13: re-export the terrain crate's painter-override
// enum so consumers (sim's `GridNavQuery::apply_obstacles`, the
// Godot bridge, tests) can name it via the sim crate without
// having to also depend on `simn-terrain` directly.
pub use names::{NameRegistry, NationalityBucket};
pub use nav::{travel_style_from_str, GridNavQuery, NavQueries, NavQuery, TravelStyle};
pub use npc_loadouts::{Loadout, LoadoutRoll, NpcLoadoutRegistry};
pub use offline_tier::{
    body_parts_to_health_class, offline_combat, offline_movement, offline_tick_just_advanced,
    project_offline_to_online, project_online_to_offline, tick_offline_clock, HealthClass,
    LoadoutClass, OfflineCombatState, OfflineNpc, OfflineTierClock, OFFLINE_ENGAGEMENT_RADIUS_M,
    OFFLINE_TIER_TICK_DIVISOR, OFFLINE_WALK_SPEED_M_PER_S,
};
pub use pda_log::{PdaEvent, PdaEventLog, PdaLogEntry};
pub use perception::{
    sight_radius_for_perception, AlwaysVisibleLos, LosProvider, LosService, PerceptionConfig,
};
pub use persistence::snapshot::SnapshotBody;
pub use region::{Region, RegionGraph, RegionId};
pub use resources::{
    ActiveRegions, ActivityKind, BallisticsConfig, BehaviorLog, BodyPartMultipliers,
    ContainerIdCounter, EffectIdCounter, GuardPostInfo, GuardPosts, InventoryConfig, JobIdCounter,
    MedConfig, MirrorMode, NpcIdCounter, NpcSpatialHash, PendingKillCredits, PopulationTargets,
    ProjectileIdCounter, RegionControl, RegionControlState, SavePaths, SimClock, SpatialEntry,
    SpatialGrid, SquadObjective, SquadObjectives, TerrainMaps, Weather, WeatherState, WorldTime,
    WoundIdCounter, ALL_WEATHER, SPATIAL_CELL_SIZE_M,
};
pub use simn_terrain::NavOverride;
pub use squad_blackboard::{
    sweep_squad_blackboards, BlackboardEntry, BlackboardKey, BlackboardValue, GroupBlackboard,
    SquadBlackboards, ThreatEntry,
};
pub use world::weapons::default_npc_round_for_faction;
pub use world::{
    food_profile, water_profile, BaseView, ConsumeProfile, CraftabilityReport, DrugOutcome,
    InputStatus, NpcView, PlayerView, Sim, TickPerfReport, TickSegments,
};
pub use world_event_bus::{
    audible_radius_m, audience_for, drain_world_events, Audience, CaliberAudibleBand, CaliberClass,
    ChatterIntent, WorldEvent, WorldEventKind, WorldEventQueue,
};
