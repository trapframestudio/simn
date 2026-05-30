//! ECS components. Each one derives `Serialize` / `Deserialize` so the
//! snapshot path can round-trip them without reflection.

use bevy_ecs::prelude::Component;
use serde::{Deserialize, Serialize};

use crate::items::ItemId;
use crate::region::RegionId;

/// World-space position in meters. Matches Godot's Vector3 layout.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Position(pub [f32; 3]);

/// Yaw rotation in radians. Pitch/roll are rendered engine-side only,
/// not authoritative for gameplay yet.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Rotation(pub f32);

/// Which region this entity currently lives in. The region graph
/// defines legal values; the sim rejects writes that reference an
/// unknown region.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct InRegion(pub RegionId);

/// What kind of actor this entity represents.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Actor {
    pub kind: ActorKind,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActorKind {
    Player,
    Npc,
}

/// Marks an entity as owned by a specific player. Uniquely keyed by
/// Steam ID so the sim can look up "the entity for this player"
/// without iterating.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayerOwned {
    pub steam_id: u64,
}

/// Hit points. Applicable to any actor (player or NPC). Values are
/// clamped to `[0, max]` by the sim API; the component itself has no
/// invariant — if you stuff it with garbage you'll get garbage back.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Health {
    pub current: f32,
    pub max: f32,
}

impl Health {
    pub const DEFAULT_MAX: f32 = 100.0;
    pub fn new_full() -> Self {
        Self {
            current: Self::DEFAULT_MAX,
            max: Self::DEFAULT_MAX,
        }
    }
}

/// Stamina. Regenerates passively at `regen_per_sec` units per
/// in-game second (see the `regen_stamina` system). Journal records
/// only the discrete `SetStamina` events; per-tick regen is a pure
/// function of last-known value + elapsed ticks, so on a crash you
/// may lose up to one snapshot interval of regen drift.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Stamina {
    pub current: f32,
    pub max: f32,
    pub regen_per_sec: f32,
}

impl Stamina {
    pub const DEFAULT_MAX: f32 = 100.0;
    pub const DEFAULT_REGEN: f32 = 15.0;

    pub fn new_full() -> Self {
        Self {
            current: Self::DEFAULT_MAX,
            max: Self::DEFAULT_MAX,
            regen_per_sec: Self::DEFAULT_REGEN,
        }
    }
}

/// Stable per-NPC identity that survives respawn / re-instantiation.
/// Distinct from [`NpcId`] (which is the live ECS handle); a chronicle
/// entry references `CharacterId` so later-spawned NPCs never collide
/// with names / backstories of the dead. Currently derived from
/// `(npc_id, faction_id)` so the same identity re-rolls on snapshot
/// reload without inline persistence. See
/// `docs/book/src/planning/npc-character-authoring-plan.md` §3.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct CharacterId(pub u64);

/// Per-NPC stat block. Eight 0–100 attributes that downstream systems
/// consume: `accuracy` → npc_combat aim cone (when projectiles land);
/// `perception` → npc_aggro sight radius; `stealth` → world-event-bus
/// audible-radius reduction; `marksmanship` → ballistic compensation
/// threshold; `endurance` → wound healing rate / stamina regen;
/// `leadership` → squad cohesion bonus; `strength` → carry capacity /
/// melee damage; `luck` → offline-tier dice modifier.
///
/// Currently substrate-only — no system reads these yet. Each downstream
/// integration ships independently as the matching plan-step lands.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NpcStats {
    pub accuracy: u8,
    pub perception: u8,
    pub stealth: u8,
    pub strength: u8,
    pub endurance: u8,
    pub marksmanship: u8,
    pub leadership: u8,
    pub luck: u8,
}

impl NpcStats {
    /// Sum of the five combat-relevant stats (accuracy, perception,
    /// marksmanship, endurance, luck). Range: `0..=500`. Used as the
    /// input to [`NpcRank::from_stats`] — bigger sum, harder fight.
    /// Strength / leadership / stealth are excluded: they affect
    /// utility behaviors (carry, squad cohesion, audibility) rather
    /// than head-to-head combat survivability.
    pub fn combat_competence(&self) -> u32 {
        u32::from(self.accuracy)
            + u32::from(self.perception)
            + u32::from(self.marksmanship)
            + u32::from(self.endurance)
            + u32::from(self.luck)
    }

    /// Roll fresh stats from `rng`. Each stat is uniform in `30..=80`
    /// with a small bias toward `accuracy` / `marksmanship` from the
    /// faction's `base_aggression` (more-aggressive factions have
    /// better trigger-pullers on average). Personality-trait + per-
    /// faction archetype tuning lands in a follow-up.
    pub fn roll<R: rand::Rng>(rng: &mut R, base_aggression: f32) -> Self {
        // base_aggression ∈ [0, 1]; map to a 0..=20 nudge applied
        // to combat-flavored stats. Defensive stats (perception,
        // stealth, endurance, leadership, strength, luck) take the
        // base roll only.
        let combat_nudge = (base_aggression * 20.0).round() as i32;
        let roll = |rng: &mut R| rng.gen_range(30u8..=80u8);
        let nudged = |rng: &mut R| {
            (i32::from(rng.gen_range(30u8..=80u8)) + combat_nudge).clamp(0, 100) as u8
        };
        Self {
            accuracy: nudged(rng),
            perception: roll(rng),
            stealth: roll(rng),
            strength: roll(rng),
            endurance: roll(rng),
            marksmanship: nudged(rng),
            leadership: roll(rng),
            luck: roll(rng),
        }
    }
}

/// Universal NPC rank tier, S.T.A.L.K.E.R.-style. Every faction uses
/// the same five-tier ladder so the player can read enemy threat
/// at a glance regardless of who they're fighting. Today the rank
/// is a pure function of [`NpcStats::combat_competence`]; once
/// chronicle-driven experience tracking lands, lived combat
/// (kills, firefights survived) will buff the effective stat sum
/// and promote NPCs through the ranks over their lifetime.
///
/// Threshold calibration: with the current `30..=80` stat roll plus
/// a 0..=20 aggression nudge on accuracy/marksmanship, the typical
/// 5-stat sum sits in the `200..=440` range. The thresholds are
/// chosen so most fresh NPCs land at `Rookie` / `Experienced`,
/// `Veteran` is uncommon, `Master` rare, and `Legend` exceptional.
#[repr(u8)]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum NpcRank {
    Rookie = 0,
    Experienced = 1,
    Veteran = 2,
    Master = 3,
    Legend = 4,
}

impl NpcRank {
    /// Promotion thresholds against [`NpcStats::combat_competence`].
    /// Each entry is the floor for that rank — at-or-above is the
    /// rank, below falls through to the previous one.
    pub const ROOKIE_FLOOR: u32 = 0;
    pub const EXPERIENCED_FLOOR: u32 = 280;
    pub const VETERAN_FLOOR: u32 = 350;
    pub const MASTER_FLOOR: u32 = 410;
    pub const LEGEND_FLOOR: u32 = 460;

    /// Compute rank from a stat block. Pure function of
    /// `combat_competence` against the floor table. Equivalent to
    /// `from_competence(stats.combat_competence())`; kept as the
    /// shorthand for fresh-roll callers that don't track lived
    /// experience yet.
    pub fn from_stats(stats: &NpcStats) -> Self {
        Self::from_competence(stats.combat_competence())
    }

    /// Compute rank from a raw competence score. Used by
    /// [`NpcCharacter::record_kill`] which promotes from
    /// `effective_competence` (base stats + lived-experience buff).
    pub fn from_competence(score: u32) -> Self {
        if score >= Self::LEGEND_FLOOR {
            Self::Legend
        } else if score >= Self::MASTER_FLOOR {
            Self::Master
        } else if score >= Self::VETERAN_FLOOR {
            Self::Veteran
        } else if score >= Self::EXPERIENCED_FLOOR {
            Self::Experienced
        } else {
            Self::Rookie
        }
    }

    /// Player-facing display name. Static — no localization layer
    /// in the sim crate; gdext bridge can translate at the boundary.
    pub fn label(self) -> &'static str {
        match self {
            Self::Rookie => "Rookie",
            Self::Experienced => "Experienced",
            Self::Veteran => "Veteran",
            Self::Master => "Master",
            Self::Legend => "Legend",
        }
    }
}

/// Per-NPC personality traits. Ten boolean flags rolled at
/// character-derive time from a faction-archetype probability table.
/// Used by `goal_arbitration` to nudge candidate priorities so an
/// `aggressive` NPC engages where a `cautious` one disengages, etc.
/// Mutually-coexisting flags are intentional — an NPC can be both
/// `disciplined` and `loyal`, both `greedy` and `reckless`. The
/// archetype mapping seeds correlated flags but doesn't enforce
/// exclusivity; per-NPC variance is the point.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PersonalityTraits {
    pub aggressive: bool,
    pub cautious: bool,
    pub curious: bool,
    pub greedy: bool,
    pub loyal: bool,
    pub bloodthirsty: bool,
    pub social: bool,
    pub solitary: bool,
    pub disciplined: bool,
    pub reckless: bool,
}

impl PersonalityTraits {
    /// Goals this personality contributes to arbitration even when
    /// no other source nominates one. These are the "weak
    /// candidates" from `npc-character-authoring-plan.md` §5 — they
    /// score below squad objectives, so an NPC only acts on them
    /// when nothing more urgent is happening. Returns at most one
    /// goal per kind (deduplicated by the arbiter implicitly via
    /// the Vec contents).
    ///
    /// Trait → drive mapping (illustrative, tunable):
    ///
    /// - `curious` → `Hunt`
    /// - `greedy` → `Loot`
    /// - `bloodthirsty` → `Bloodsport`
    /// - `social` → `Socialize`
    pub fn introduces_drives(&self) -> Vec<PersonalityDrive> {
        let mut out = Vec::with_capacity(4);
        if self.curious {
            out.push(PersonalityDrive::Hunt);
        }
        if self.greedy {
            out.push(PersonalityDrive::Loot);
        }
        if self.bloodthirsty {
            out.push(PersonalityDrive::Bloodsport);
        }
        if self.social {
            out.push(PersonalityDrive::Socialize);
        }
        out
    }
}

/// Personality-driven candidate type that the arbiter resolves to a
/// fully-targeted [`GoalKind`] using contextual lookups (group
/// centroid, corpse position, activity point catalog). Decouples the
/// trait-introduction step (which knows nothing about the world)
/// from the world-aware targeting step (which lives in
/// `goal_arbitration`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersonalityDrive {
    Hunt,
    Socialize,
    Loot,
    Bloodsport,
}

/// Faction archetype that drives the personality-trait probability
/// roll. Each faction declares its archetype in `factions.toml`
/// (`archetype = "disciplined"` etc.). The
/// [`Self::from_faction_name`] table remains as a fallback for
/// factions whose TOML predates the field — it ships sensible
/// defaults for the canonical roster.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PersonalityArchetype {
    /// Drilled state actor. PWA, Federal, Aegis Pacific, Revere
    /// Guard, and their elite subfactions. High `disciplined` +
    /// `loyal`, moderate `aggressive`.
    Disciplined,
    /// Rapid violence as the first answer. Linemen, Cartel,
    /// Merged. High `aggressive` + `bloodthirsty`, moderate
    /// `reckless`, lower `cautious`.
    Aggressive,
    /// Profit-driven, opportunistic. Looters, Bandits, Gulf
    /// Compact, Registry. High `greedy` + `reckless`, low `loyal`.
    Greedy,
    /// Drifters. Wanderers. High `curious` + `solitary`,
    /// moderate `cautious`, low `aggressive`.
    Curious,
    /// Belief-driven. Attuned, Choir. High `loyal` + `social`,
    /// moderate `disciplined`.
    Reverent,
    /// Fallback when no archetype is defined for the faction.
    /// Uniform 50/50 across all traits.
    #[default]
    Default,
}

impl PersonalityArchetype {
    /// Map a faction name (registry key) to the archetype that
    /// drives the trait roll. Unknown / unmapped names fall back to
    /// `Default`.
    pub fn from_faction_name(name: &str) -> Self {
        match name {
            "pwa" | "federal" | "ghost_teams" | "aegis_pacific" | "recovery_division"
            | "revere_guard" => Self::Disciplined,
            "linemen" | "cartel" | "merged" => Self::Aggressive,
            "looters" | "bandits" | "gulf_compact" | "registry" => Self::Greedy,
            "wanderers" => Self::Curious,
            "attuned" | "choir" => Self::Reverent,
            _ => Self::Default,
        }
    }

    /// Per-trait probability of `true` on this archetype. Each value
    /// in `[0.0, 1.0]`; the rolling function compares each trait's
    /// probability against an independent uniform draw. Tuned by eye
    /// — playtest will surface the dials worth pulling on.
    fn trait_probabilities(self) -> [f32; 10] {
        // Order matches PersonalityTraits field declaration order:
        // aggressive, cautious, curious, greedy, loyal, bloodthirsty,
        // social, solitary, disciplined, reckless.
        match self {
            Self::Disciplined => [0.45, 0.55, 0.20, 0.10, 0.70, 0.15, 0.50, 0.20, 0.85, 0.10],
            Self::Aggressive => [0.85, 0.15, 0.20, 0.30, 0.45, 0.65, 0.40, 0.25, 0.60, 0.55],
            Self::Greedy => [0.55, 0.30, 0.30, 0.85, 0.20, 0.40, 0.45, 0.30, 0.20, 0.65],
            Self::Curious => [0.25, 0.55, 0.80, 0.20, 0.30, 0.10, 0.30, 0.70, 0.25, 0.20],
            Self::Reverent => [0.35, 0.45, 0.45, 0.10, 0.85, 0.20, 0.75, 0.15, 0.65, 0.10],
            Self::Default => [0.5; 10],
        }
    }

    /// Roll a `PersonalityTraits` bitmap from this archetype. Each
    /// trait's flag is set independently from a uniform draw vs.
    /// the archetype's per-trait probability.
    pub fn roll_traits<R: rand::Rng>(self, rng: &mut R) -> PersonalityTraits {
        let p = self.trait_probabilities();
        PersonalityTraits {
            aggressive: rng.gen::<f32>() < p[0],
            cautious: rng.gen::<f32>() < p[1],
            curious: rng.gen::<f32>() < p[2],
            greedy: rng.gen::<f32>() < p[3],
            loyal: rng.gen::<f32>() < p[4],
            bloodthirsty: rng.gen::<f32>() < p[5],
            social: rng.gen::<f32>() < p[6],
            solitary: rng.gen::<f32>() < p[7],
            disciplined: rng.gen::<f32>() < p[8],
            reckless: rng.gen::<f32>() < p[9],
        }
    }
}

/// Procedural per-NPC identity. Substrate plus `name + nationality`
/// for the multicultural roster, `personality + rank` for the
/// behavior + threat-tier wiring, `stats` for everything downstream.
/// Backstory templates land in a later slice of
/// `npc-character-authoring-plan.md`. Spawned for every NPC at
/// `npc_spawn` and re-rolled deterministically from
/// `(npc_id, faction_id, archetype)` on snapshot load.
///
/// `NpcCharacter` is `Clone` but **not** `Copy` because of the
/// `name: String` field. The component is mutated in place via
/// `World::get_mut`, so the loss of `Copy` only forces explicit
/// `.cloned()` on accessor returns — see `world::debug` test
/// helpers.
#[derive(Component, Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct NpcCharacter {
    pub character_id: CharacterId,
    pub stats: NpcStats,
    pub personality: PersonalityTraits,
    /// Universal STALKER-style threat tier derived from
    /// [`Self::effective_competence`]. Refreshed on `roll` and on
    /// every kill via [`Self::record_kill`].
    pub rank: NpcRank,
    /// Display name "First Last", rolled from the global
    /// multicultural pool. Stored inline so it survives
    /// re-derivation (the underlying RNG sequence depends on
    /// rng-call ordering, but the rolled string itself is what
    /// gameplay reads).
    pub name: String,
    /// Cultural / ethnic-origin bucket the name was rolled from.
    /// Drives downstream character-mesh selection.
    pub nationality: crate::names::NationalityBucket,
    /// Lifetime kills. Increments on the killing-blow hit attributed
    /// to this NPC. Buffs `effective_competence` (and therefore
    /// `rank`) so a long-surviving veteran outranks a fresh-rolled
    /// one with the same baseline stats. Capped at `u16::MAX`;
    /// chronicle-side kill records aren't persisted yet, so a
    /// despawn-and-respawn cycle resets this — that's fine until
    /// chronicle-driven identity persistence lands.
    pub kills: u16,
}

impl NpcCharacter {
    /// Derive the canonical `CharacterId` from `(npc_id, faction_id)`.
    /// A simple multiplicative hash keeps determinism without pulling
    /// in a hash-dependency crate. The `faction_id` term means an NPC
    /// of the same `npc_id` but a different faction (e.g., between
    /// reroster passes) gets a different identity, which is the right
    /// semantic. World-seed mixing folds in once `Sim` carries one.
    pub fn derive_id(
        npc_id: NpcId,
        faction_id: crate::faction::registry::FactionId,
    ) -> CharacterId {
        let mut h = npc_id.0;
        h = h.wrapping_mul(0xA5A5_A5A5_A5A5_A5A5);
        h ^= u64::from(faction_id.0).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h.rotate_left(17);
        CharacterId(h)
    }

    /// Roll a fresh character with deterministic identity from
    /// `(npc_id, faction_id, archetype)`. Stats, personality, name,
    /// and nationality are all rolled from a per-character
    /// `ChaCha8Rng` seeded by `character_id`, so re-rolling the
    /// same identity + archetype always produces byte-identical
    /// state. `rank` is derived from the rolled stats via
    /// [`NpcRank::from_stats`]. Names follow the faction's
    /// `nationality_weights` skew (uniform when the map is empty).
    pub fn roll(
        npc_id: NpcId,
        faction_id: crate::faction::registry::FactionId,
        archetype: PersonalityArchetype,
        base_aggression: f32,
        names: &crate::names::NameRegistry,
        faction_nationality_weights: &std::collections::HashMap<String, u32>,
        male_name_weight: Option<f32>,
    ) -> Self {
        use rand::SeedableRng;
        let character_id = Self::derive_id(npc_id, faction_id);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(character_id.0);
        let stats = NpcStats::roll(&mut rng, base_aggression);
        let personality = archetype.roll_traits(&mut rng);
        let rank = NpcRank::from_stats(&stats);
        let male_w = male_name_weight.unwrap_or(crate::names::DEFAULT_MALE_NAME_WEIGHT);
        let (nationality, name) =
            names.roll_for_faction_gendered(&mut rng, faction_nationality_weights, male_w);
        Self {
            character_id,
            stats,
            personality,
            rank,
            name,
            nationality,
            kills: 0,
        }
    }

    /// Effective combat competence after lived-experience buffs.
    /// `combat_competence + kills × KILL_COMPETENCE_BUFF`, capped at
    /// 500 (the natural ceiling for a 5-stat sum at 100 each). A
    /// 25-kill veteran gets +75 to their competence, typically
    /// promoting one rank tier; a 50-kill veteran reliably hits
    /// `Master`. Pure function of `(stats, kills)` — no RNG, no
    /// dependence on registry state — so re-derivation on snapshot
    /// reload returns the same value as long as the saved kill
    /// count survives.
    pub fn effective_competence(&self) -> u32 {
        let base = self.stats.combat_competence();
        let bonus = u32::from(self.kills).saturating_mul(KILL_COMPETENCE_BUFF);
        (base + bonus).min(500)
    }

    /// Increment kill count and refresh `rank` from the new
    /// effective competence. Caller is `npc_combat` on a hit that
    /// brings the target's `vital_min` to zero. Kept on
    /// `NpcCharacter` (rather than as a free function in
    /// `systems::npc_combat`) so the invariant
    /// "rank tracks effective_competence" lives next to the field
    /// it maintains.
    pub fn record_kill(&mut self) {
        self.kills = self.kills.saturating_add(1);
        let competence = self.effective_competence();
        self.rank = NpcRank::from_competence(competence);
    }
}

/// Per-kill bonus to `combat_competence` for lived-experience rank
/// promotion. Tunable; small enough that bumping a Rookie to
/// `Master` requires real combat history (~50 kills).
pub const KILL_COMPETENCE_BUFF: u32 = 3;

/// Per-body-part hit points. Six independent pools
/// (head/torso/arms/legs); head or torso at 0 ⇒ death; limbs at
/// 0 ⇒ disabled (clients gate sprint/aim on `limb_disabled`). Aggregate
/// [`Health`] on the same entity is maintained at `min(head, torso)` so
/// existing death-gate consumers keep working without iterating the
/// six values. Spawned for players AND NPCs; the wound tick pipeline in
/// `systems/wounds.rs` (bleed / heal / infection / necrosis) operates on
/// any entity carrying `Wounds + BodyParts`.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct BodyParts {
    pub head: f32,
    pub torso: f32,
    pub left_arm: f32,
    pub right_arm: f32,
    pub left_leg: f32,
    pub right_leg: f32,
}

impl BodyParts {
    pub const DEFAULT_MAX: f32 = 100.0;

    pub fn new_full() -> Self {
        Self {
            head: Self::DEFAULT_MAX,
            torso: Self::DEFAULT_MAX,
            left_arm: Self::DEFAULT_MAX,
            right_arm: Self::DEFAULT_MAX,
            left_leg: Self::DEFAULT_MAX,
            right_leg: Self::DEFAULT_MAX,
        }
    }

    pub fn get(&self, p: BodyPart) -> f32 {
        match p {
            BodyPart::Head => self.head,
            BodyPart::Torso => self.torso,
            BodyPart::LeftArm => self.left_arm,
            BodyPart::RightArm => self.right_arm,
            BodyPart::LeftLeg => self.left_leg,
            BodyPart::RightLeg => self.right_leg,
        }
    }

    pub fn get_mut(&mut self, p: BodyPart) -> &mut f32 {
        match p {
            BodyPart::Head => &mut self.head,
            BodyPart::Torso => &mut self.torso,
            BodyPart::LeftArm => &mut self.left_arm,
            BodyPart::RightArm => &mut self.right_arm,
            BodyPart::LeftLeg => &mut self.left_leg,
            BodyPart::RightLeg => &mut self.right_leg,
        }
    }

    pub fn is_alive(&self) -> bool {
        self.head > 0.0 && self.torso > 0.0
    }

    /// Aggregate health for the legacy [`Health`] mirror: `min(head, torso)`.
    /// Limb damage doesn't pull this down — limbs degrade function, not
    /// the death gate.
    pub fn vital_min(&self) -> f32 {
        self.head.min(self.torso)
    }

    /// True for limb parts at or below 0. Always false for head/torso —
    /// those are the death gate, not a "disabled" state.
    pub fn limb_disabled(&self, p: BodyPart) -> bool {
        match p {
            BodyPart::Head | BodyPart::Torso => false,
            _ => self.get(p) <= 0.0,
        }
    }
}

/// Addressable body parts. Used as the destination for damage/heal
/// calls and the key for [`BodyParts`] field access.
///
/// `#[serde(rename_all = "snake_case")]` so TOML-driven data
/// (`items.toml` armor coverage arrays, etc.) can reference parts
/// as `"head"` / `"left_arm"`. Bincode (the snapshot + journal
/// format) serializes enum variants by index, so this rename is a
/// no-op on persistence.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BodyPart {
    Head,
    Torso,
    LeftArm,
    RightArm,
    LeftLeg,
    RightLeg,
}

impl BodyPart {
    /// All six parts in canonical iteration order. Used by the
    /// limb-state transition pipeline and tests.
    pub const ALL: [BodyPart; 6] = [
        BodyPart::Head,
        BodyPart::Torso,
        BodyPart::LeftArm,
        BodyPart::RightArm,
        BodyPart::LeftLeg,
        BodyPart::RightLeg,
    ];
}

/// Coarse per-limb status that the wound pipeline + sever resolution
/// maintain alongside the numeric HP in [`BodyParts`]. The HP track
/// answers "how much is left in this pool"; the state track answers
/// "is this limb structurally there at all". Decoupling them lets a
/// limb at 100 HP be `Wounded` (open bleed in progress) while a limb
/// at 0 HP can stay `Intact` (functionally disabled but recoverable).
/// `Severed` is permanent: HP can't refill, the limb cannot be healed
/// back. See `docs/book/src/planning/dismemberment-plan.md` §3.
#[repr(u8)]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum LimbState {
    /// No active wounds, limb intact.
    #[default]
    Intact = 0,
    /// At least one open wound on this limb. Set when a wound spawns,
    /// cleared back to `Intact` when the last wound resolves.
    Wounded = 1,
    /// Limb permanently lost. Cannot be healed back; HP stays at 0.
    /// Severing head or torso also drives the death gate (HP → 0 →
    /// aggregate `Health` follows).
    Severed = 2,
}

/// Per-limb [`LimbState`] for an entity, sibling to [`BodyParts`]. Spawned
/// alongside `BodyParts` for both players and NPCs; default is
/// six `Intact` states.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LimbStates {
    pub head: LimbState,
    pub torso: LimbState,
    pub left_arm: LimbState,
    pub right_arm: LimbState,
    pub left_leg: LimbState,
    pub right_leg: LimbState,
}

impl LimbStates {
    pub fn get(&self, p: BodyPart) -> LimbState {
        match p {
            BodyPart::Head => self.head,
            BodyPart::Torso => self.torso,
            BodyPart::LeftArm => self.left_arm,
            BodyPart::RightArm => self.right_arm,
            BodyPart::LeftLeg => self.left_leg,
            BodyPart::RightLeg => self.right_leg,
        }
    }

    pub fn get_mut(&mut self, p: BodyPart) -> &mut LimbState {
        match p {
            BodyPart::Head => &mut self.head,
            BodyPart::Torso => &mut self.torso,
            BodyPart::LeftArm => &mut self.left_arm,
            BodyPart::RightArm => &mut self.right_arm,
            BodyPart::LeftLeg => &mut self.left_leg,
            BodyPart::RightLeg => &mut self.right_leg,
        }
    }

    /// True when the limb is `Severed`. Independent of HP — for the
    /// HP-zero "disabled but attached" case query [`BodyParts::limb_disabled`]
    /// instead. Most consumers want both: a limb is non-functional if
    /// EITHER returns true.
    pub fn is_severed(&self, p: BodyPart) -> bool {
        matches!(self.get(p), LimbState::Severed)
    }

    /// Flip the limb to `Wounded` if currently `Intact`. No-op for
    /// `Wounded` (idempotent for repeated wound spawns on the same
    /// part) and for `Severed` (a severed limb cannot regress to
    /// merely wounded). Call this from every wound-spawn site.
    pub fn mark_wounded(&mut self, p: BodyPart) {
        let slot = self.get_mut(p);
        if matches!(*slot, LimbState::Intact) {
            *slot = LimbState::Wounded;
        }
    }

    /// Flip the limb to `Severed`. HP zeroing is the caller's
    /// responsibility — this method only owns the state field.
    /// Severing is permanent; subsequent `mark_wounded` calls are no-ops.
    pub fn mark_severed(&mut self, p: BodyPart) {
        *self.get_mut(p) = LimbState::Severed;
    }

    /// Recompute `Wounded → Intact` transitions after the wound list
    /// changes. For each part, if the part is currently `Wounded` AND
    /// the supplied wound list contains no entries for it, flip back
    /// to `Intact`. `Severed` parts are left alone; `Intact` parts are
    /// left alone. Call after `age_and_heal_wounds` retains the wound
    /// list (i.e., after `Healed` wounds get dropped).
    pub fn recompute_from_wounds(&mut self, wounds: &Wounds) {
        for part in BodyPart::ALL {
            if !matches!(self.get(part), LimbState::Wounded) {
                continue;
            }
            let still_open = wounds.0.iter().any(|(_, w)| w.body_part == part);
            if !still_open {
                *self.get_mut(part) = LimbState::Intact;
            }
        }
    }
}

/// Player survival meters. All three are 0–100 where 100 = full and
/// 0 = depleted. Drained over in-world time by
/// `drain_survival_stats`; restored by `Sim::consume`. Below-threshold
/// values cause degraded function (regen halved, slow HP trickle) per
/// `docs/survival-and-crafting-plan.md` §3.3 — never instant death from
/// the survival layer alone.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct SurvivalStats {
    pub hunger: f32,
    pub thirst: f32,
    pub fatigue: f32,
}

impl SurvivalStats {
    pub const FULL: f32 = 100.0;

    pub fn new_full() -> Self {
        Self {
            hunger: Self::FULL,
            thirst: Self::FULL,
            fatigue: Self::FULL,
        }
    }

    pub fn get(&self, s: SurvivalStat) -> f32 {
        match s {
            SurvivalStat::Hunger => self.hunger,
            SurvivalStat::Thirst => self.thirst,
            SurvivalStat::Fatigue => self.fatigue,
        }
    }

    pub fn get_mut(&mut self, s: SurvivalStat) -> &mut f32 {
        match s {
            SurvivalStat::Hunger => &mut self.hunger,
            SurvivalStat::Thirst => &mut self.thirst,
            SurvivalStat::Fatigue => &mut self.fatigue,
        }
    }
}

/// Addressable survival meter. Used by [`crate::Sim::set_survival_stat`]
/// and the `SetSurvivalStat` journal record.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurvivalStat {
    Hunger,
    Thirst,
    Fatigue,
}

/// Stable identity for a wound, minted from
/// [`crate::resources::WoundIdCounter`] at spawn time. Persistent
/// across save/load so journal records can reference a specific wound
/// without ambiguity. Never reused.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct WoundId(pub u64);

/// One discrete wound on a body part. Multiple wounds can coexist on
/// the same part. Spawned for both players (medical/treatment loop) and
/// NPCs (combat damage); the wound tick pipeline applies to either.
/// Bleeding HP-drain is computed per tick by `apply_bleed_damage` from
/// the active untreated wounds; healing of `Bandaged` wounds is driven
/// by `age_and_heal_wounds` (terminal `Healed` state then despawns).
/// See `docs/survival-and-crafting-plan.md` §4.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Wound {
    pub body_part: BodyPart,
    pub kind: WoundKind,
    /// 1..=5 per spec §4.1. Drives bleed rate and treatment eligibility
    /// (severity ≤ 3 = light: bandageable; severity ≥ 4 = heavy:
    /// requires tourniquet or wound-pack first).
    pub severity: u8,
    pub spawned_tick: u64,
    pub treatment: WoundTreatment,
    /// Tick at which the current treatment was applied. Used by
    /// `age_and_heal_wounds` to gate the Bandaged → Healed transition.
    pub treatment_changed_tick: u64,
    /// True once `tick_infection` has flipped this wound. Set after the
    /// untreated wound has aged past `MedConfig::infection_trigger_ticks`
    /// without being disinfected. Cleared by antibiotics. Drains HP at
    /// a low rate while true (independent of bleed).
    #[serde(default)]
    pub infected: bool,
    /// Tick at which `infected` became `true` (or `None` if not
    /// infected). Lets `tick_infection` and antibiotics derive
    /// progress without a separate counter.
    #[serde(default)]
    pub infection_started_tick: Option<u64>,
    /// Tick at which a tourniquet was applied to this wound's part.
    /// Used by `tick_necrosis` for the warning + escalating-damage
    /// timer. Cleared on `remove_tourniquet`.
    #[serde(default)]
    pub tourniquet_started_tick: Option<u64>,
}

/// Wound categories. Step 2 implements `Bleed` only; the other variants
/// in `survival-and-crafting-plan.md` §4.1 (`Fracture`, `Burn`,
/// `Puncture`, `Laceration`) land with full medical depth in Step 6.
/// The enum is additive — existing journal records stay valid as
/// variants are added.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WoundKind {
    Bleed,
}

/// Treatment progression for a wound, full spec §4.1 / §4.2.
///
/// **Light bleed pipeline:** `Untreated → [Disinfected] → Bandaged →
/// [Stitched] → Healed`. Disinfecting before Bandaging prevents
/// infection (see `Wound::infected`); Stitching after Bandaging halves
/// the heal time. `apply_bandage` accepts both Untreated and
/// Disinfected, but skipping Disinfect leaves the wound at infection
/// risk.
///
/// **Heavy bleed pipeline:** `Untreated → Tourniquet | WoundPacked →
/// [Stitched] → Healed`. `Tourniquet` is the emergency option; it
/// stops bleed at the cost of the necrosis timer (`tourniquet_started_tick`).
/// `WoundPacked` (wound pack / pressure dressing) is the
/// no-cost alternative — but the spec leaves it as a craftable item
/// gated on later steps. `Stitch` closes either path.
///
/// Auto-transitions (driven by `age_and_heal_wounds`):
/// - `Bandaged` → `Healed` after `MedConfig::heal_ticks_bandaged`.
/// - `Stitched` → `Healed` after `MedConfig::heal_ticks_stitched`
///   (≈ half of bandaged).
///
/// `Tourniquet` and `Untreated` and `Disinfected` never auto-heal —
/// they require explicit treatment to proceed.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WoundTreatment {
    Untreated,
    Disinfected,
    Bandaged,
    Stitched,
    Tourniquet,
    WoundPacked,
    Healed,
}

/// All active wounds on one entity — players and NPCs both. Empty
/// (or absent) means uninjured. Order is unspecified; consumers
/// iterate. `Vec` rather than `HashMap` because typical N is 0–6 and
/// serdes trivially.
#[derive(Component, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Wounds(pub Vec<(WoundId, Wound)>);

/// Derived per-tick pain meter, in `[0, 100]`. Computed by `tick_pain`
/// from the player's active wounds (untreated > bandaged > stitched)
/// and reduced by Painkiller / Morphine effects. Player-only.
/// Above `MedConfig::pain_regen_threshold` (default 50), stamina
/// regen is halved (per spec §3.3 / §4.4 — pain affects function).
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
pub struct Pain(pub f32);

/// Radiation + Toxicity meters, both `[0, 100]`. Player-only. Decay
/// passively per `tick_contamination`; rise from contaminated food /
/// dirty water / future fault exposure. Above
/// `MedConfig::contamination_hp_threshold` (default 80), apply slow HP
/// drain (parallel to the §3.3 hunger/thirst HP gate).
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
pub struct Contamination {
    pub radiation: f32,
    pub toxicity: f32,
}

/// Stable identity for one active drug/effect instance. Minted by
/// `EffectIdCounter`. Persisted; never reused. Used as the journal
/// key in `EffectApplied` records.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct EffectId(pub u64);

/// One in-flight drug or status effect on a player.
///
/// Lifecycle is stateless: the effect contributes its modifier while
/// `current_tick - applied_tick < duration_ticks`, then `tick_active_effects`
/// retires it. Multi-phase effects (Stim's active+rebound, Adrenaline's
/// active+crash) are modeled as **two separate `ActiveEffect`s scheduled
/// in sequence** — the crash-phase effect uses
/// `applied_tick = active_phase_end_tick`, so it activates only when
/// its window starts. Avoids per-effect state machines and keeps
/// replay deterministic.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct ActiveEffect {
    pub id: EffectId,
    pub kind: EffectKind,
    pub applied_tick: u64,
    pub duration_ticks: u64,
    pub intensity: f32,
}

/// All in-flight effects on one entity — players and NPCs both (NPCs
/// gained `ActiveEffects` alongside `Wounds` so `apply_antibiotics_npc`
/// can clear infection the same way it does for players). Order is
/// unspecified.
#[derive(Component, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct ActiveEffects(pub Vec<ActiveEffect>);

/// Categories of `ActiveEffect`. Includes both player-applied drugs
/// (top group) and system-emitted status effects (Withdrawal,
/// OverdoseDisorientation, AdrenalineCrash, FatigueRebound,
/// AntibioticsActive). The enum is additive — older journals stay
/// valid as variants land.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectKind {
    Painkiller,
    Morphine,
    Adrenaline,
    StimCocktail,
    AntiRad,
    AntiTox,
    AntibioticsActive,
    Withdrawal,
    OverdoseDisorientation,
    AdrenalineCrash,
    FatigueRebound,
}

/// The subset of [`EffectKind`] that's an "addictive drug" — has a
/// `tolerance` counter, may overdose, may withdraw. Decoupled from
/// `EffectKind` so the system-emitted statuses (Withdrawal, etc.)
/// don't accidentally get a tolerance entry.
///
/// Serialized as snake_case (e.g. `"painkiller"`, `"anti_rad"`) so
/// `items.toml` can reference variants directly in `consume_action`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DrugKind {
    Painkiller,
    Morphine,
    Adrenaline,
    StimCocktail,
    AntiRad,
    AntiTox,
}

impl DrugKind {
    /// Map back to the matching `EffectKind` for the active phase.
    pub fn primary_effect(self) -> EffectKind {
        match self {
            DrugKind::Painkiller => EffectKind::Painkiller,
            DrugKind::Morphine => EffectKind::Morphine,
            DrugKind::Adrenaline => EffectKind::Adrenaline,
            DrugKind::StimCocktail => EffectKind::StimCocktail,
            DrugKind::AntiRad => EffectKind::AntiRad,
            DrugKind::AntiTox => EffectKind::AntiTox,
        }
    }
}

/// Per-drug tolerance counter, `[0, 100]`. Rises with each use,
/// decays passively. Used by `apply_drug` to gate overdose
/// (`tolerance > overdose_threshold` AND another dose) and by
/// `tick_active_effects` to gate withdrawal
/// (`tolerance > withdrawal_threshold` AND no active dose AND
/// elapsed > withdrawal_delay).
///
/// `Vec` rather than `HashMap` because there are ~6 drug kinds and
/// serdes trivially. Keys are unique by convention; helpers ensure
/// only one entry per drug kind.
#[derive(Component, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct DrugTolerance(pub Vec<(DrugKind, f32)>);

impl DrugTolerance {
    pub fn get(&self, k: DrugKind) -> f32 {
        self.0
            .iter()
            .find(|(kk, _)| *kk == k)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    }

    pub fn set(&mut self, k: DrugKind, v: f32) {
        if let Some((_, slot)) = self.0.iter_mut().find(|(kk, _)| *kk == k) {
            *slot = v;
        } else {
            self.0.push((k, v));
        }
    }

    pub fn add(&mut self, k: DrugKind, delta: f32) {
        let cur = self.get(k);
        self.set(k, (cur + delta).clamp(0.0, 100.0));
    }
}

/// Categories of food the player can `eat`. Each kind has a fixed
/// nutritional profile defined by `food_profile`; no per-instance
/// state. Serialized as snake_case so `items.toml` references like
/// `food_kind = "cooked_meat"` resolve directly.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FoodKind {
    PreservedRation,
    FreshFood,
    RawMeat,
    CookedMeat,
    ContaminatedFood,
    FieldRation,
    EnergyBar,
}

/// Categories of drink, mirror of [`FoodKind`]. Profile in
/// `water_profile`. Some drinks grant a temporary effect (e.g.,
/// `EnergyDrink` grants a short Stim) by routing through `apply_drug`.
/// Serialized as snake_case; referenced by `items.toml`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum WaterKind {
    DirtyWater,
    CleanWater,
    EnergyDrink,
    Vodka,
}

/// Faction allegiance keyed by registry id. Attaches to NPCs, bases,
/// and any other entity with a "side." Players are intentionally
/// faction-agnostic.
///
/// Not serialized via the auto-derived path — persistence emits the
/// faction's registry **name string** (`"pwa"`, `"linemen"`) so
/// saves stay valid across registry edits. Loaders rebuild the
/// `FactionId` by name lookup against the active registry.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct InFaction(pub crate::faction::registry::FactionId);

/// Marks an entity as a faction base (checkpoint, outpost, …). The
/// rest of a base's data lives in sibling components: `Position`,
/// `InRegion`, `InFaction`, `Health`.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Base {
    pub kind: BaseKind,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseKind {
    Checkpoint,
    Outpost,
    Safehouse,
    Headquarters,
    ResearchPost,
    /// Neutral, non-contestable rest spot. Anyone can use it at any
    /// time; no faction owns it (stored with `Faction::Wanderers` as
    /// the neutral placeholder, but territorial control ignores it).
    CampSite,
}

impl BaseKind {
    /// All variants in declaration order. The GDScript-side `Kind`
    /// enum in `godot/scripts/world/poi_marker.gd` mirrors this list
    /// (with a `BASE_` prefix); drift is caught at build time by
    /// `crates/simn-godot/tests/poi_enum_sync.rs`.
    pub const ALL: [BaseKind; 6] = [
        BaseKind::Checkpoint,
        BaseKind::Outpost,
        BaseKind::Safehouse,
        BaseKind::Headquarters,
        BaseKind::ResearchPost,
        BaseKind::CampSite,
    ];

    /// Iteration 5-13 follow-up to Phase B2. Conservative nav-
    /// blocking footprint each base kind stamps onto the per-
    /// region nav grid at `attach_region_terrain` time. Returns
    /// XZ half-extents in meters; `None` means "no structure to
    /// block around" — currently only `CampSite` (open camps
    /// where NPCs literally rest in the open).
    ///
    /// These are tuned small enough that NPCs still naturally
    /// path *around* the base center (a 4 m × 4 m footprint on
    /// a 2 m nav grid is two cells wide — pathfinding flows
    /// around it cleanly), but big enough that the grid records
    /// "structure exists here" so squads don't try to plant a
    /// guard post on top of an existing outpost. Real authored
    /// bases — buildings with walls, doors, gates — land in a
    /// follow-up iteration; these are the placeholder
    /// footprints for the procedurally-seeded set.
    pub fn nav_footprint_xz_m(self) -> Option<[f32; 2]> {
        match self {
            BaseKind::Checkpoint => Some([3.0, 3.0]),
            BaseKind::Outpost => Some([5.0, 5.0]),
            BaseKind::Safehouse => Some([4.0, 4.0]),
            BaseKind::Headquarters => Some([8.0, 8.0]),
            BaseKind::ResearchPost => Some([6.0, 6.0]),
            BaseKind::CampSite => None,
        }
    }
}

/// Stable identity for an NPC, kept in the [`crate::chronicle::LifeChronicle`]
/// after the entity itself despawns. Minted from
/// [`crate::resources::NpcIdCounter`] at spawn time so ids are unique
/// across the lifetime of a save.
#[derive(
    Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd,
)]
pub struct NpcId(pub u64);

/// Marks an entity as an NPC, with its stable id. Sibling components
/// hold the rest of the state: `InFaction`, `InRegion`, `Position`,
/// `Rotation`, `Health`, `NpcGoal`, `Lifespan`.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Npc {
    pub id: NpcId,
}

/// Current behavior state for an NPC. Tiny FSM:
/// `Idle` waits, `MoveTo` walks toward a target, `RestAt` pauses
/// somewhere. Driven by [`crate::systems::tick_npc_goals`].
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub enum NpcGoal {
    Idle { until_tick: u64 },
    MoveTo { target: [f32; 3] },
    RestAt { until_tick: u64 },
}

/// Visual pose for an NPC during a dwell. Communicates to the renderer
/// whether to play standing, sitting, or crouching idle animations.
/// Written by [`crate::systems::tick_npc_goals`] when entering a dwell
/// state; read by the Godot bridge.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DwellPose {
    /// Default idle stance.
    #[default]
    Standing,
    /// Sitting at a rest spot or campfire.
    Sitting,
    /// Crouching, e.g. at a lookout or in cover.
    Crouching,
}

/// Per-NPC dwell visual + position-shift state. Inserted when an NPC
/// enters a dwell at a squad objective; the executor uses
/// `last_shift_tick` to time periodic micro-position nudges at Guard
/// posts so guards visibly shift weight instead of standing perfectly
/// still for the full guard tenure.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
pub struct DwellState {
    pub pose: DwellPose,
    pub last_shift_tick: u64,
}

/// Marks a `WorldContainer` entity as having been spawned from a dead
/// NPC. Carries the dead NPC's id and faction so the loot arbiter
/// can pick targets by faction relations (a bandit NPC's corpse is a
/// more attractive target than a same-faction corpse). Not journaled
/// — corpse containers reload from `WorldContainerSpawned` deltas;
/// the marker is recreated on load by re-spawn / replay paths.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorpseMarker {
    pub dead_npc_id: NpcId,
    pub dead_faction: crate::faction::registry::FactionId,
}

/// A cached route an NPC is currently following. Inserted by
/// [`crate::systems::tick_npc_goals`] when an NPC commits to a
/// movement target, advanced along per tick, and dropped on
/// completion. Recomputed when the target moves more than
/// `PATH_RECOMPUTE_DIST_M` from `target` (e.g. a player pursuing
/// across a region). Not journaled - paths are derived from
/// goal/target/heightmap and rebuild from current state on replay.
///
/// **Determinism:** waypoints come from `simn_sim::nav::NavQuery::path`,
/// which is deterministic given identical inputs (heightmap + style +
/// endpoints). Two reloads of the same save reproduce the same paths.
#[derive(Component, Clone, Debug)]
pub struct Path {
    /// World-space waypoints, including start and end. Ordered.
    pub waypoints: Vec<[f32; 3]>,
    /// Index of the next waypoint the NPC is heading toward.
    pub current: u32,
    /// Tick when this path was computed; for staleness checks.
    pub computed_tick: u64,
    /// Original target the path was computed for. If the NPC's current
    /// target drifts more than `PATH_RECOMPUTE_DIST_M` from this, the
    /// path is recomputed.
    pub target: [f32; 3],
}

/// Built-in expiration. When `clock.tick >= die_at_tick`, the
/// `age_npcs` system kills the entity with `DeathCause::NaturalCauses`.
/// Real combat death (HP → 0) doesn't depend on this and lands later.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lifespan {
    pub spawned_tick: u64,
    pub die_at_tick: u64,
}

/// Marks an NPC as part of a coherent group (squad, gang, cult cell).
/// Group members share a deterministic RNG seed when picking new
/// patrol targets, so they walk toward the same destination instead
/// of dispersing. Lone NPCs (Wanderers, sometimes Looters/Compact
/// contractors) don't have this component.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub id: u64,
}

/// Transient perception state: which other NPC this one currently
/// considers a target, and when it was last seen. Set by
/// `npc_aggro`, cleared by aggro decay or target loss. **Not
/// serialized** — perception re-acquires after load. Drives the
/// patrol-vs-pursue branch in `tick_npc_goals` and the firing
/// decision in `npc_combat`.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Aggro {
    pub target: NpcId,
    pub last_seen_tick: u64,
}

/// Transient: identity, faction, and timing of whoever last
/// damaged this NPC. Used by `npc_death_check` for kill credit
/// (`faction`), by the threat-board sweep / arbitration for
/// recency gating (`tick`), and by future cross-region tactical
/// memory (`attacker_id`). Not serialized — combat events refill
/// it on the next damaging shot after load.
///
/// `attacker_id = None` means the damage source isn't an NPC the
/// sim can name — typically a player projectile (player damage
/// flows through `Sim::apply_damage_to_npc_part`, which doesn't
/// have an NpcId for the attacker today). When players gain
/// NPC-equivalent ids in the multiplayer A-Life pass, this field
/// will populate uniformly.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct LastDamager {
    pub attacker_id: Option<NpcId>,
    pub faction: crate::faction::registry::FactionId,
    pub tick: u64,
}

/// One recent damage event on an NPC. Pushed by combat systems
/// when damage lands; aggregated by the squad threat board into a
/// per-attacker score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttackerHit {
    pub attacker_id: NpcId,
    pub tick: u64,
    pub damage: f32,
}

/// Recent damage events on this NPC, oldest first. Capped at
/// `MAX_RECENT_ATTACKERS` (~8) entries; older entries get evicted
/// FIFO on push, and entries past `THREAT_TTL_TICKS` get swept at
/// tick start by `sweep_threats`.
///
/// Substrate for the squad threat board: `npc_combat` writes here
/// on damage, the sweep system aggregates into the squad
/// blackboard, `goal_arbitration` reads the aggregated scores to
/// pick a target. See
/// `docs/book/src/planning/threat-board-plan.md`.
///
/// Transient — same persistence policy as `Aggro` (rebuilt from
/// the next combat event after load; a few seconds of recency
/// drift across crash-recovery is acceptable).
#[derive(Component, Clone, Debug, Default)]
pub struct RecentAttackers {
    pub events: Vec<AttackerHit>,
}

impl RecentAttackers {
    /// Push a new hit, evicting the oldest entry if we'd exceed
    /// `cap`. If the same attacker has a recent entry, accumulate
    /// damage onto it instead of pushing a duplicate (so a stream
    /// of fire from one shooter doesn't crowd out other threats
    /// from the cap-bounded ring). `tick` of the merged entry is
    /// the latest hit so recency-decay reads from "most recent
    /// damage" not "first damage."
    pub fn record(&mut self, attacker_id: NpcId, tick: u64, damage: f32, cap: usize) {
        if let Some(existing) = self
            .events
            .iter_mut()
            .find(|h| h.attacker_id == attacker_id)
        {
            existing.tick = tick;
            existing.damage += damage;
            return;
        }
        if self.events.len() >= cap {
            self.events.remove(0);
        }
        self.events.push(AttackerHit {
            attacker_id,
            tick,
            damage,
        });
    }

    /// Drop entries older than `cutoff_tick`. Called per-tick by
    /// `sweep_threats`.
    pub fn sweep(&mut self, cutoff_tick: u64) {
        self.events.retain(|h| h.tick >= cutoff_tick);
    }
}

/// Squad combat role assigned when a squad enters combat. Influences
/// stance decisions in `npc_tactical` — Pointmen are more aggressive,
/// Support holds position, Flankers seek lateral cover, Medics
/// prioritize downed allies.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombatRole {
    Pointman,
    Support,
    Flanker,
    Medic,
}

impl CombatRole {
    pub fn assign(stats: &NpcStats, personality: &PersonalityTraits) -> Self {
        let aggro_score = u32::from(stats.accuracy)
            + u32::from(stats.marksmanship)
            + if personality.aggressive { 30 } else { 0 }
            + if personality.reckless { 20 } else { 0 };

        let support_score = u32::from(stats.leadership)
            + u32::from(stats.endurance)
            + if personality.disciplined { 30 } else { 0 };

        let flank_score = u32::from(stats.perception)
            + u32::from(stats.stealth)
            + if personality.curious { 30 } else { 0 }
            + if personality.reckless { 15 } else { 0 };

        let medic_score = u32::from(stats.endurance)
            + u32::from(stats.luck)
            + if personality.cautious { 30 } else { 0 }
            + if personality.loyal { 25 } else { 0 };

        let max = aggro_score
            .max(support_score)
            .max(flank_score)
            .max(medic_score);
        if max == aggro_score {
            Self::Pointman
        } else if max == flank_score {
            Self::Flanker
        } else if max == medic_score {
            Self::Medic
        } else {
            Self::Support
        }
    }
}

/// Per-NPC GOAP plan. The `npc_tactical` system runs the planner
/// when the plan empties (all steps executed) and executes one step
/// per stance transition.
#[derive(Component, Clone, Debug, Default)]
pub struct GoapPlanComp {
    pub actions: Vec<&'static str>,
    pub planned_at_tick: u64,
    pub last_world_state: u32,
}

impl GoapPlanComp {
    pub fn current_action(&self) -> Option<&'static str> {
        self.actions.first().copied()
    }

    pub fn advance(&mut self) {
        if !self.actions.is_empty() {
            self.actions.remove(0);
        }
    }

    #[allow(dead_code)]
    pub fn is_stale(&self, now: u64, max_age_ticks: u64) -> bool {
        now.saturating_sub(self.planned_at_tick) > max_age_ticks
    }
}

/// Tactical combat stance for an NPC in active combat. Set by the
/// `npc_tactical` system based on threat assessment and cover
/// availability. Drives movement target selection in `tick_npc_goals`
/// and fire-decision gating in `npc_combat`.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub enum CombatStance {
    /// Moving toward engagement range, no cover claim yet.
    Approaching,
    /// Behind a cover volume, alternating between peeking and hiding.
    InCover {
        volume_id: u64,
        peek_until_tick: u64,
        next_peek_tick: u64,
    },
    /// Exposed, actively firing. Brief window before seeking cover.
    Firing { since_tick: u64 },
    /// Pinned by concentrated incoming fire. Won't peek or fire.
    Suppressed { until_tick: u64 },
    /// Moving to a lateral position around the target.
    Flanking,
    /// Disengaging toward safety.
    Retreating,
}

impl CombatStance {
    pub fn is_peeking(&self, now: u64) -> bool {
        match self {
            Self::InCover {
                peek_until_tick, ..
            } => now < *peek_until_tick,
            Self::Firing { .. } => true,
            Self::Flanking => true,
            Self::Approaching => true,
            _ => false,
        }
    }

    pub fn can_fire(&self, now: u64) -> bool {
        match self {
            Self::Suppressed { until_tick } => now >= *until_tick,
            Self::InCover {
                peek_until_tick, ..
            } => now < *peek_until_tick,
            Self::Retreating => false,
            _ => true,
        }
    }
}

/// Per-NPC aggression attribute, `[0.0, 1.0]`. Drives how readily an
/// NPC fires on aggro targets — high-aggression NPCs hit more often
/// and engage more decisively. Set at spawn from a faction base
/// (see `faction_base_aggression`) plus a small per-NPC jitter, so
/// individuals within a faction differ. Persona-weighted goal
/// priority (the real version) lands with the Persona system; this
/// is the placeholder.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Aggression(pub f32);

/// Why an [`ActiveGoal`] was selected. Each variant carries the same
/// stable rank position in the priority table so a system inspecting
/// the active goal can ask "is this combat-driven?" / "is this
/// scripted?" without re-deriving from sources.
///
/// Ranks are illustrative; tune on playtest. The default table lives
/// in `systems::goal_arbitration` so the priorities and the resolver
/// stay close.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalSource {
    /// A scripted-quest layer claimed this NPC for a beat. Highest
    /// priority short of survival. Stage 4+ — placeholder source for
    /// now, no system emits it yet.
    ScriptedClaim,
    /// HP / wounds threshold tripped: flee, self-treat, retreat. Stage
    /// 4+ — placeholder.
    IndividualSurvival,
    /// Squad has agreed on a target — multiple members hold the same
    /// `Aggro.target`, or the spotter shared via blackboard.
    SquadAggro,
    /// This NPC alone perceives a target. May be promoted via flank
    /// bonus to outrank `SquadAggro`. Stage 2 will add the bonus; for
    /// now `SquadAggro` simply outranks.
    IndividualAggro,
    /// Driven by a non-aggro blackboard entry: heard a gunshot, saw a
    /// downed ally, etc. Stage 2.
    BlackboardUrgency,
    /// Squad-level objective from `SquadObjectives`: Patrol, Guard,
    /// Investigate, Rest, Wander, etc.
    SquadObjective,
    /// Per-NPC trait nudges. Stage 4 — placeholder.
    PersonalityBias,
    /// No other source produced a candidate. Falls into the legacy
    /// solo idle FSM (Idle → MoveTo → RestAt).
    Idle,
}

/// What the executor (currently `tick_npc_goals`) does to satisfy the
/// goal. Stage-2 personality-introduced kinds (`Hunt` / `Socialize` /
/// `Loot` / `Bloodsport`) currently fall through to the
/// `SoloIdleFsm` branch in the executor — they're substrate for the
/// arbitration layer until ecosystem (huntable fauna), corpse-loot
/// surfaces, and faction-cultural arena targeting land.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GoalKind {
    /// Pursue an NPC by id. Executor reads live position from
    /// `NpcPositionIndex` each tick (so the target can keep moving).
    /// Halts at engage range; combat does the work.
    PursueTarget { target: NpcId },
    /// Follow the squad's `SquadObjective` (formation offset, patrol
    /// leg advance, etc.). Executor re-derives the world-space
    /// movement target each tick from the resource.
    SquadFollowObjective,
    /// Legacy per-NPC FSM: Idle / MoveTo / RestAt with an occasional
    /// long-march. Used when no higher-priority source fires.
    SoloIdleFsm,
    /// Curious / aggressive trait drive to hunt prey or points of
    /// interest. `target_pos` is the location the NPC will move to;
    /// `activity_point_id` (when present) identifies an authored POI
    /// slot the goal was nominated against so the executor can clear
    /// the slot on completion.
    Hunt {
        target_pos: [f32; 3],
        activity_point_id: Option<u64>,
    },
    /// Social trait drive to congregate near squad-mates. `target_pos`
    /// is the gathering centroid the NPC walks to and faces. NPCs
    /// gather, face inward, and dwell. Distinct from a Rest objective
    /// (which keeps formation slot positions) — Socialize collapses
    /// the squad to a tight ring around the centroid.
    Socialize { target_pos: [f32; 3] },
    /// Greedy / reckless trait drive to rifle corpses + stashes.
    /// `target_pos` is the corpse / container position. The optional
    /// `target_container` carries the ContainerId so the executor can
    /// correlate dwell completion with inventory transfer logic.
    Loot {
        target_pos: [f32; 3],
        target_container: Option<u64>,
    },
    /// Bloodthirsty trait drive to fight for fun, faction-cultural
    /// (Wanderers / Looters specifically). Target arrives with the
    /// arena / spar concept.
    Bloodsport,
    /// React to a blackboard urgency (HeardGunshot, UnderFireAt).
    /// Move toward `pos` at urgent travel style. Distinct from
    /// `PursueTarget` (no target id; position is static) and from
    /// `SquadFollowObjective` (not derived from squad state). Real
    /// suppress / take-cover behavior lands with tactical AI.
    InvestigateAt { pos: [f32; 3] },
    /// React to a fallen squadmate (`DownedAlly`). Move to the
    /// ally's last position. `id` is preserved for the future
    /// revive / loot surface and so the executor can clear the
    /// goal when the body is reached. Substrate today; the
    /// regroup-and-mourn / revive flow lands later.
    RegroupOnAlly { id: NpcId, pos: [f32; 3] },
    /// Critically wounded NPC fleeing toward medical aid (nearest
    /// same-faction RestSpot / Campfire). Highest-priority non-combat
    /// goal at `PRIO_INDIVIDUAL_SURVIVAL`. On arrival the NPC dwells
    /// until wounds heal (handled by the existing
    /// [`crate::systems::wounds`] system) or it dies.
    SeekMedical { target_pos: [f32; 3] },
}

/// Resolved per-tick goal for an NPC. Written by `goal_arbitration`,
/// read by `tick_npc_goals` (and any future movement / combat
/// system). Not serialized — derived from the source components every
/// tick on load.
///
/// `priority` is the value the resolver picked at, including any
/// situational bonuses (flank, etc). Hysteresis: once an `ActiveGoal`
/// is set, a new candidate must beat its priority by at least
/// `HYSTERESIS_PRIO_DELTA` to preempt it. Re-derivation by the same
/// source updates `expires_at` instead.
///
/// `expires_at = None` means "as long as the source still holds."
/// Sources can opt into a hard expiry to force re-evaluation (used
/// for blackboard urgencies that should burn out).
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct ActiveGoal {
    pub source: GoalSource,
    pub kind: GoalKind,
    pub priority: u8,
    pub created_tick: u64,
    pub expires_at: Option<u64>,
    /// Pursue-progress tracking: position when progress was last
    /// recorded and the tick it was recorded at. If the NPC hasn't
    /// gotten `PURSUE_PROGRESS_M` closer to its target in
    /// `PURSUE_TIMEOUT_TICKS`, Aggro is cleared. Only meaningful
    /// when `kind == PursueTarget`.
    pub pursue_progress: Option<PursueProgress>,
    /// Commitment window: non-combat candidates are skipped until
    /// this tick. Set when a SquadObjective goal is first assigned
    /// so squads don't get yanked off-task by low-priority
    /// distractions (HeardGunshot, PersonalityBias) in the first
    /// 30 seconds. Combat sources (IndividualAggro, SquadAggro,
    /// DownedAlly, UnderFireAt) always bypass the window.
    pub committed_until_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PursueProgress {
    pub pos: [f32; 3],
    pub tick: u64,
}

impl Default for ActiveGoal {
    fn default() -> Self {
        Self {
            source: GoalSource::Idle,
            kind: GoalKind::SoloIdleFsm,
            priority: 0,
            created_tick: 0,
            expires_at: None,
            pursue_progress: None,
            committed_until_tick: 0,
        }
    }
}

/// One stack of an item. `spawned_tick` is minted at pickup / craft /
/// salvage time; perishable items use `clock.tick - spawned_tick` as
/// their age. Stacks of the same item id with different `spawned_tick`
/// stay separate so older instances expire first (no age mixing).
///
/// In the grid model (Tarkov/STALKER hybrid) this struct represents
/// the **stack data** independent of where it sits — see
/// [`PlacedItem`] for the position binding. Kept as its own struct
/// so callers don't have to drag `(x, y, rotation)` around when they
/// only care about "how many bandages does the player have."
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ItemInstance {
    pub id: ItemId,
    pub count: u32,
    pub spawned_tick: u64,
    /// Runtime state that only exists on magazine items. `None` on
    /// every non-magazine instance. The def's `magazine_config`
    /// tells the engine whether this field is meaningful; the
    /// field itself carries the dynamic per-mag state
    /// (loaded-round count today; loaded-variant later once
    /// caliber variants ship). `#[serde(default)]` so existing
    /// saves deserialize cleanly.
    #[serde(default)]
    pub magazine_state: Option<MagazineState>,
}

impl ItemInstance {
    /// Loaded-round count for magazine instances. Non-magazines and
    /// magazines missing their `magazine_state` return `0` — callers
    /// treat "no state" and "empty" as the same thing for the fire /
    /// HUD paths.
    pub fn loaded_rounds(&self) -> u32 {
        self.magazine_state
            .as_ref()
            .map(|m| m.loaded_rounds)
            .unwrap_or(0)
    }
}

/// Runtime per-magazine state. Populated on every magazine
/// [`ItemInstance`] — how many rounds are currently in it. A fresh
/// magazine grants from the crafting / loot path will land with
/// `loaded_rounds = 0`; reload eats rounds from inventory until
/// `def.magazine_config.capacity` is reached. The engine never
/// hardcodes capacity — it always reads `magazine_config.capacity`
/// from the item def.
/// The `variant` field is no longer `Copy` — it holds an `ItemId` —
/// so `MagazineState` drops the `Copy` derive; existing copy sites
/// use `.clone()` instead. Legacy saves (pre-phase-2) deserialize
/// with `variant: None` via `#[serde(default)]`.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MagazineState {
    pub loaded_rounds: u32,
    /// Ammo variant currently loaded in this magazine (e.g.
    /// `round_5_45x39_ap`). `None` on fresh or pre-phase-2 mags —
    /// the fire path treats `None` as "no round to fire" (dry-click)
    /// so the player must explicitly load a variant before shooting.
    #[serde(default)]
    pub variant: Option<ItemId>,
}

/// Cardinal-only rotation for grid-placed items. Tarkov / Resident
/// Evil 4 style: items are either upright (`Deg0`) or rotated 90°
/// counter-clockwise (`Deg90`). 180° / 270° aren't necessary because
/// item footprints are rectangular — the visible orientation is
/// purely cosmetic on the third 90.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ItemRotation {
    #[default]
    Deg0,
    Deg90,
}

/// One stack placed at a specific `(x, y)` in a [`GridInventory`]
/// with a specific rotation. Items occupy a rectangle whose width and
/// height come from the [`crate::items::ItemDef::size`] (swapped if
/// `rotation == Deg90`).
///
/// If the underlying item is a container (def has `inner_grid`
/// set), the placement carries its own [`GridInventory`] in
/// `inner_grid` — a loaded backpack sitting in your pockets keeps
/// its contents attached. Non-container items leave `inner_grid`
/// as `None`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PlacedItem {
    pub stack: ItemInstance,
    pub x: u32,
    pub y: u32,
    pub rotation: ItemRotation,
    #[serde(default)]
    pub inner_grid: Option<GridInventory>,
}

/// 2D grid inventory — Tarkov/STALKER hybrid model. Each
/// [`PlacedItem`] occupies a rectangle (`def.size`, possibly rotated)
/// anchored at `(x, y)` with the origin in the top-left. Two items
/// may not overlap.
///
/// The grid itself stores **only the placed-item list**; cell
/// occupancy is recomputed on demand by the placement engine
/// ([`crate::inventory_grid`]). That keeps serialization small and
/// avoids invariant-keeping across the structure.
///
/// Containers are GridInventories too (rigs, backpacks, world
/// crates, corpses). Players carry one as their `Inventory(...)`
/// "pockets" grid; equipped containers expose additional grids
/// through the equipment system in PR-2.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GridInventory {
    pub width: u32,
    pub height: u32,
    pub items: Vec<PlacedItem>,
}

impl GridInventory {
    /// Default player "pockets" grid — 4×4 (16 cells). Sized to hold
    /// a handful of small items + one or two medium ones; the rest
    /// of carry capacity comes from equipped containers.
    pub const DEFAULT_PLAYER_WIDTH: u32 = 4;
    pub const DEFAULT_PLAYER_HEIGHT: u32 = 4;

    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            items: Vec::new(),
        }
    }

    pub fn player_default() -> Self {
        Self::new(Self::DEFAULT_PLAYER_WIDTH, Self::DEFAULT_PLAYER_HEIGHT)
    }

    /// Total cell count.
    pub fn capacity_cells(&self) -> u32 {
        self.width.saturating_mul(self.height)
    }

    /// Sum of every placed stack's count. Doesn't account for cell
    /// area — this is the legacy "how many items am I carrying" view.
    pub fn total_count(&self) -> u32 {
        self.items.iter().map(|p| p.stack.count).sum()
    }
}

impl Default for GridInventory {
    fn default() -> Self {
        Self::player_default()
    }
}

/// Player inventory — wraps a [`GridInventory`] (the player's
/// "pockets"). Equipped containers (rig, backpack) bring their own
/// nested grids via the [`Equipment`] component. Player-only for
/// now; NPC inventories land alongside corpse loot.
///
/// The wrapper keeps the `Inventory` type name stable across the
/// flat-list → grid migration so call sites that don't care about
/// layout (e.g. "is the player's inventory empty") continue to read
/// naturally as `inv.0.items`.
#[derive(Component, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Inventory(pub GridInventory);

/// One item held in an equipment slot. `inner_grid` is populated iff
/// the item's [`crate::items::ItemDef::inner_grid`] is `Some` — when
/// you unequip a backpack or rig, you pull it off **with its
/// contents intact**, so the grid travels with the `EquippedItem`
/// rather than living separately. For non-container equippable
/// items (a helmet, a rifle today), `inner_grid` is `None`.
///
/// `weapon_state` is populated iff the item is a weapon (its
/// [`crate::items::ItemDef::weapon_config`] is `Some`). Non-weapon
/// slots leave it `None`. The weapon's loaded magazine, once
/// attachments + parts land, future per-weapon runtime state all
/// hang off this struct.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EquippedItem {
    pub stack: ItemInstance,
    pub inner_grid: Option<GridInventory>,
    #[serde(default)]
    pub weapon_state: Option<EquippedWeaponState>,
}

/// Per-equipped-weapon runtime state. Today carries only the
/// loaded magazine (if any); designed to extend to
/// `attachments_equipped: Vec<ItemInstance>` and
/// `parts_condition: Vec<PartCondition>` in the attachment /
/// wear phases without breaking serialization.
///
/// The loaded magazine is a full [`ItemInstance`] (with its own
/// `magazine_state.loaded_rounds` and everything) so ejecting a
/// half-empty mag back to inventory preserves the remaining
/// rounds exactly.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EquippedWeaponState {
    #[serde(default)]
    pub loaded_magazine: Option<ItemInstance>,
    /// Phase 4D: aggregate weapon condition (0.0 – 100.0). 100
    /// = pristine, 0 = catastrophic failure. Decremented per
    /// shot in `fire_weapon` by `WeaponConfig.wear_per_shot`.
    /// Drives `jam_chance_at_condition` and (future) accuracy
    /// / muzzle-velocity degradation curves. Single aggregate
    /// (no per-part breakdown) for v1; the §5.1 part roster
    /// from `weapons-plan.md` lands in a later iteration.
    /// `#[serde(default = "default_full_condition")]` so
    /// pre-4D snapshots round-trip with weapons at full
    /// condition. The manual `Default` impl also writes 100.0
    /// so freshly-constructed weapons start pristine, not
    /// dead-on-arrival.
    #[serde(default = "default_full_condition")]
    pub condition: f32,
    /// Phase 4D: whether the weapon is currently jammed. A
    /// fire attempt rolls jam *before* expending a round; on
    /// jam, the weapon transitions to `Jammed` and stays
    /// stuck until `clear_weapon_jam` runs. Default
    /// `Cleared` for both fresh weapons and pre-4D snapshots.
    #[serde(default)]
    pub jam_state: JamState,
}

impl Default for EquippedWeaponState {
    fn default() -> Self {
        Self {
            loaded_magazine: None,
            // Phase 4D: weapons spawn at full condition, not 0.
            // A derive(Default) here would zero condition and
            // make every fresh equip jam-prone — caught by the
            // `fresh_weapon_decrements_condition_on_fire_and_journals_delta`
            // and `fresh_weapon_never_jams_over_a_full_magazine`
            // tests when the field was first added.
            condition: default_full_condition(),
            jam_state: JamState::Cleared,
        }
    }
}

fn default_full_condition() -> f32 {
    100.0
}

/// Phase 4D: weapon jam state machine.
///
/// Held on [`EquippedWeaponState`]. Fire attempts on a jammed
/// weapon return a dry-click error; the player must run
/// `clear_weapon_jam` to recover. Kinds carry the *cause* so
/// the UI can pick a clear-jam animation / time (FTF → quick
/// rack, FTE → mortar, etc. — wired in the future UI slice).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum JamState {
    /// Weapon is firable (default).
    #[default]
    Cleared,
    /// Failure to feed — the spring / mag interface failed to
    /// chamber a round. Quick rack-bolt clears it.
    FailureToFeed,
    /// Failure to extract — the spent case stayed in the
    /// chamber. Requires manual extract / mortar.
    FailureToExtract,
    /// Stovepipe — case caught in the ejection port. Quick
    /// rack-bolt clears it.
    Stovepipe,
}

impl JamState {
    /// `true` if the weapon is in any non-cleared state.
    pub fn is_jammed(self) -> bool {
        !matches!(self, JamState::Cleared)
    }
}

/// Paper-doll equipment on the player. Keyed by
/// [`crate::items::SlotId`] (e.g. `"head"`, `"backpack"`,
/// `"belt_1"`). An absent key = nothing equipped at that slot. The
/// slot definitions themselves live in
/// [`crate::items::EquipmentSlotRegistry`]; this component only
/// tracks what's in them.
///
/// The layout is **data-driven**: every slot id matches a row in
/// `equipment_slots.toml`, and no engine code names a specific slot
/// id. Adding `offhand_shield` is a TOML row + optional new
/// [`crate::items::ItemCategory`] variant; no changes to this
/// struct. Player-only for now; NPC equipment lands alongside the
/// corpse-loot pass.
#[derive(Component, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct Equipment(pub std::collections::HashMap<crate::items::SlotId, EquippedItem>);

// Custom `Serialize` so the slot map emits in key-sorted order. The
// determinism harness (`tests/determinism.rs`) relies on snapshot
// bytes being identical across same-seed sims; default `HashMap`
// serialization uses iteration order, which differs per instance.
impl serde::Serialize for Equipment {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        crate::det_serde::sorted_map(&self.0, ser)
    }
}

/// Stable id for a world-placed container (ground drop, scene-placed
/// crate, corpse). Minted from
/// [`crate::resources::ContainerIdCounter`]; persisted across save /
/// load so journal records (`WorldContainerSpawned`,
/// `WorldContainerItemAdded`, …) keep referring to the same entity
/// after a reload.
#[derive(
    Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd,
)]
pub struct ContainerId(pub u32);

/// World-placed container — the entity holds an inventory grid plus
/// position / region. Spawned by:
///
/// - **Ground drop** — when a player calls `Sim::drop_item`, a small
///   personal container appears at their feet (`is_public = false`).
/// - **Scene-placed crate** — worldbuilding will instance these in
///   bench / safehouse / loot scenes (`is_public = true` for shared
///   work-bench parts bins; `false` for private stashes).
/// - **NPC corpse** — converted from a dead NPC entity in PR-4b.
///
/// **`is_public` controls crafting kit-pool inclusion**: public
/// containers within `CRAFTING_SHARE_RADIUS_M` of the crafter
/// contribute their kits + tools (a parts bin chained to a workbench);
/// private ones (a player's stash) don't, even if the player is
/// standing right next to them.
///
/// The entity also carries [`Position`] + [`InRegion`] sibling
/// components so the spatial-proximity / scene-spawn paths can find
/// it without a custom index. NPC corpses get a TTL via the
/// `Lifespan` component (PR-4b).
/// How the player accesses a `WorldContainer`'s contents.
/// Authored by `LootContainerMarker3D` in the editor;
/// procedural / corpse / drop containers default to `Openable`.
///
/// `Breakable` runtime semantics (damage routing, break VFX,
/// contents transfer to a ground pile) are not yet wired —
/// scaffold-only. The field stays on the entity through the
/// snapshot so the future destruction system has the data it
/// needs when it lands.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ContainerInteractionMode {
    /// `[F] open` — the standard looting flow. Existing PR-4c
    /// proximity prompt + Phase 3E unified inventory panel.
    #[default]
    Openable,
    /// Smash to loot — container must be destroyed first, then
    /// contents drop as a ground container. Runtime damage
    /// routing is a future slice.
    Breakable,
}

#[derive(Component, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct WorldContainer {
    pub id: ContainerId,
    pub grid: GridInventory,
    pub is_public: bool,
    /// Faction that owns / services this container. Drives loot-pool
    /// flavor on initial roll + restock. `None` for player-spawned
    /// containers (ground drops, corpses — corpses still have a
    /// faction conceptually, but they don't restock and their
    /// contents come from the NPC's inventory). Phase 3A scattered
    /// containers + Phase 3D authored markers both populate this.
    /// Serialized with `#[serde(default)]` so pre-3C snapshots
    /// load with `None`.
    #[serde(default)]
    pub faction: Option<String>,
    /// Depth tier (1-3) the container's loot rolls against. 1 =
    /// surface (default), 2 = interior, 3 = deep. Hand-set per
    /// region (or per authored marker) by the worldbuilder.
    /// `#[serde(default)]` floors pre-3C snapshots to tier 1.
    #[serde(default = "default_depth_tier")]
    pub depth_tier: u8,
    /// Last sim tick this container received a restock sweep
    /// (Phase 3C). `0` for never-restocked / pre-3C snapshots —
    /// the sweep treats those the same way as a stale container.
    #[serde(default)]
    pub last_restock_tick: u64,
    /// How the player accesses contents (Phase 3D authored
    /// marker scaffold). `#[serde(default)]` floors pre-3D
    /// snapshots to `Openable` — same behavior as before.
    #[serde(default)]
    pub interaction_mode: ContainerInteractionMode,
}

fn default_depth_tier() -> u8 {
    1
}

/// Debug-only flag marking the player as "standing near a campfire" for
/// the crafting-context check (Step 4). Step 5 replaces this with a
/// real `Campfire` entity + proximity test in the sim — the
/// [`crate::items::Recipe::required_context`] field survives that
/// transition unchanged.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NearCampfire(pub bool);

/// Tier of the nearest workbench the player is standing next to.
/// `None` = no workbench in range. Matches GAMMA progression:
/// Basic / Advanced / Expert. Used by `Sim::queue_craft` to gate
/// recipes whose `required_context` names a bench tier — higher
/// tiers satisfy lower-tier requirements (standing at an Advanced
/// bench lets you run Basic-tier recipes). Analogous in structure to
/// [`NearCampfire`] but carries a tier rather than a bool.
///
/// Step 5 ships the component + debug setter; wiring up a proximity
/// system based on world-placed workbench entities lands with the
/// bench-placement / scene work later in the slice.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct NearWorkbench(pub Option<crate::items::ToolTier>);

/// One queued crafting job. See [`CraftingQueue`] for the per-player
/// list; [`crate::world::Sim::queue_craft`] for the API that appends.
/// The active job is the one at index 0; when `ticks_remaining` hits
/// zero the job pops one unit's outputs into inventory, decrements
/// `count_remaining`, and either resets `ticks_remaining` to the
/// recipe's `time_ticks` for the next unit or drops from the queue
/// when the count is exhausted.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct CraftJob {
    pub id: u32,
    pub recipe_id: String,
    pub count_remaining: u32,
    pub ticks_remaining: u64,
    pub started_tick: u64,
}

/// Player-owned FIFO of [`CraftJob`]s. The front (index 0) is the
/// active job; the [`crate::systems::tick_crafting_queue`] system
/// ticks it down each frame, grants outputs, and advances. Empty /
/// absent on an entity means "not crafting".
///
/// Materials are consumed **up front** on `queue_craft` (so the
/// player can't dupe inputs by cancelling mid-craft and re-queueing
/// elsewhere) and refunded proportionally on cancel.
#[derive(Component, Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct CraftingQueue(pub Vec<CraftJob>);

/// Stable per-projectile identifier, minted on spawn by the
/// `ProjectileIdCounter` resource. Used to correlate the
/// `ProjectileSpawned` and `ProjectileImpacted` deltas client-side
/// so trace FX can terminate at the right impact point.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProjectileId(pub u64);
impl From<u64> for ProjectileId {
    fn from(v: u64) -> Self {
        Self(v)
    }
}

/// Weapons phase 2: an in-flight bullet. Ticked by the host's
/// `tick_projectiles` system with gravity + drag, resolved against
/// humanoid hitboxes (see `crate::world::hitbox`). Despawned on
/// impact or when `distance_traveled_m` passes `max_range_m`.
///
/// Snapshots persist this component so a host restart mid-flight
/// doesn't lose the shot. Mirrors never own `Projectile` entities —
/// the client lifts trace + impact FX from the broadcast deltas
/// (`ProjectileSpawned` / `ProjectileImpacted`).
///
/// All ballistic tuning (mass, drag, muzzle velocity, damage,
/// penetration) lives in the round's `AmmoConfig`; this component
/// pins down the per-entity dynamic state only.
#[derive(Component, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Projectile {
    pub id: ProjectileId,
    /// Steam ID of the shooter — excluded from hit candidates so
    /// the player can't self-hit. `0` for NPC-fired projectiles
    /// (see `source_npc_id`).
    pub source_steam_id: u64,
    /// Phase 4A v1: `Some(npc_id)` when the projectile was fired
    /// by an NPC. Used by the tick to skip damage application
    /// (NPC-vs-NPC damage still routes through `npc_combat`'s
    /// dice path in v1; 4A v2 will migrate damage to projectile
    /// impact resolution). `None` for player-fired projectiles —
    /// those keep their existing damage / hit pipeline.
    /// `#[serde(default)]` so pre-4A snapshots in flight load
    /// without the field and default-resolve to the player path.
    #[serde(default)]
    pub source_npc_id: Option<NpcId>,
    /// Ammo item id (e.g. `round_5_45x39_ap`). The tick reads
    /// `ItemDef::ammo_config` for mass / drag / pen / damage.
    pub round_id: ItemId,
    /// World-space position (m). Updated each tick.
    pub pos: [f32; 3],
    /// Velocity (m/s). Updated each tick by gravity + drag.
    pub vel: [f32; 3],
    /// Distance traveled since spawn (m). Despawn threshold read
    /// from the firing weapon's `weapon_config.range_m`; pinned on
    /// spawn so the projectile doesn't care if the weapon changes.
    pub distance_traveled_m: f32,
    /// Max range before despawn (from the firing weapon's
    /// `weapon_config.range_m`).
    pub max_range_m: f32,
    /// Tick the projectile was spawned. Used for life-limit fallback
    /// and debugging.
    pub spawned_tick: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> NpcId {
        NpcId(n)
    }

    #[test]
    fn recent_attackers_record_appends_new_entry() {
        let mut r = RecentAttackers::default();
        r.record(id(1), 100, 25.0, 8);
        assert_eq!(r.events.len(), 1);
        assert_eq!(r.events[0].attacker_id, id(1));
        assert_eq!(r.events[0].damage, 25.0);
        assert_eq!(r.events[0].tick, 100);
    }

    #[test]
    fn recent_attackers_record_merges_repeat_attacker() {
        // A stream of fire from one shooter accumulates into a
        // single entry rather than crowding out other threats.
        let mut r = RecentAttackers::default();
        r.record(id(1), 100, 25.0, 8);
        r.record(id(1), 110, 30.0, 8);
        assert_eq!(r.events.len(), 1, "same attacker merges");
        assert_eq!(r.events[0].damage, 55.0, "damage accumulates");
        assert_eq!(r.events[0].tick, 110, "tick is most-recent");
    }

    #[test]
    fn recent_attackers_record_evicts_oldest_at_cap() {
        let mut r = RecentAttackers::default();
        for i in 0..10u64 {
            r.record(id(i + 1), 100 + i, 5.0, 8);
        }
        assert_eq!(r.events.len(), 8, "FIFO cap holds");
        // Oldest two (id 1, 2) evicted; id 3 is now first.
        assert_eq!(r.events[0].attacker_id, id(3));
        assert_eq!(r.events[7].attacker_id, id(10));
    }

    #[test]
    fn recent_attackers_sweep_drops_expired() {
        let mut r = RecentAttackers::default();
        r.record(id(1), 100, 10.0, 8);
        r.record(id(2), 200, 10.0, 8);
        r.record(id(3), 300, 10.0, 8);
        // Cutoff = keep entries with tick >= 200.
        r.sweep(200);
        assert_eq!(r.events.len(), 2);
        assert_eq!(r.events[0].attacker_id, id(2));
        assert_eq!(r.events[1].attacker_id, id(3));
    }

    #[test]
    fn base_kind_nav_footprint_scales_with_kind() {
        // CampSite is the only open kind — no structure, no footprint.
        assert_eq!(BaseKind::CampSite.nav_footprint_xz_m(), None);
        // Every other kind blocks at least its own immediate cell.
        let structured = [
            BaseKind::Checkpoint,
            BaseKind::Outpost,
            BaseKind::Safehouse,
            BaseKind::Headquarters,
            BaseKind::ResearchPost,
        ];
        for k in structured {
            let f = k
                .nav_footprint_xz_m()
                .unwrap_or_else(|| panic!("{k:?} must have a nav footprint"));
            assert!(f[0] > 0.0 && f[1] > 0.0, "footprint must be positive");
            // 2 m nav cells — a 3 m half-extent is the minimum that
            // touches the surrounding cell ring on every side.
            assert!(f[0] >= 3.0 && f[1] >= 3.0, "{k:?} footprint too small");
        }
        // Sanity check the ordering — bigger / more important bases
        // should have bigger footprints. Drops a regression if someone
        // tunes one kind way out of band.
        let outpost = BaseKind::Outpost.nav_footprint_xz_m().unwrap();
        let hq = BaseKind::Headquarters.nav_footprint_xz_m().unwrap();
        assert!(hq[0] > outpost[0], "HQ should be larger than Outpost");
    }
}
