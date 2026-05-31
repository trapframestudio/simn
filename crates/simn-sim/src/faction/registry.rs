//! TOML-driven faction registry. Replaces the closed `Faction` enum
//! during the migration documented in
//! `docs/book/src/planning/faction-registry-plan.md`.
//!
//! The registry is built from a single TOML file at sim startup
//! (modders can layer more files via the mod manifest). Code never
//! names factions by enum variant — it asks the registry by string
//! id (`"coalition"`, `"coalition_vanguard"`) and gets back a `FactionId` (a small
//! interned `u32`) for hot-path equality / hashing.
//!
//! Three pieces of runtime state, all keyed by string `name` for
//! save resilience across registry edits:
//!
//! - [`FactionRegistry`] — the parsed config: faction defs + relation
//!   matrix overrides + default relation. Immutable after build;
//!   rebuilds on every sim startup from the TOML source of truth.
//! - [`RelationDeltas`] — mutable runtime drift on top of the matrix.
//!   `i16` scores added to the registry's base. Persisted in saves.
//! - [`PlayerReputation`] — per-player rep, isolated by SteamId so
//!   one player's actions don't penalize another. Same `i16` score
//!   space as faction-vs-faction. Persisted in saves.
//!
//! ## Lookup contracts
//!
//! - [`faction_relation`] — reads base from registry, applies drift,
//!   walks parent chains (subfaction → parent) for inheritance, snaps
//!   to band.
//! - [`player_relation`] — reads per-player rep; falls back to
//!   faction-vs-faction baseline against the configured
//!   `player_baseline` faction if the player has no entry yet (first
//!   contact uses the baseline's matrix row).
//!
//! ## Determinism
//!
//! `FactionId` values come from sorting `defs` by `name` at build
//! time. Same TOML in → same id assignment out, run after run. Saves
//! serialize the `name` string (not the id) so editing the TOML
//! between sessions doesn't corrupt save state.

use std::collections::HashMap;
use std::path::Path;

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use super::{anchor_score, band_from_score, relation_from_str, Relation, SCORE_MAX, SCORE_MIN};

/// Interned id for a faction. Stable for a given registry build, NOT
/// stable across registry edits. Save state uses
/// [`FactionDef::name`] (string) instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FactionId(pub u32);

/// Per-faction config: identity + tunable knobs. The full row of
/// per-faction "what kind of NPC is this" data the sim reads at
/// gameplay time. Add fields here when a new system needs faction-
/// scoped data; preserve backwards compatibility by giving them
/// `#[serde(default = "...")]` so old TOML files still parse.
#[derive(Clone, Debug, PartialEq)]
pub struct FactionDef {
    pub id: FactionId,
    /// Canonical lowercase ascii id, e.g. `"coalition"`, `"coalition_vanguard"`. The
    /// stable identity used in saves, journals, and TOML cross-refs.
    pub name: String,
    /// Display-cased name for UI, e.g. `"Coalition"`, `"Vanguard"`.
    pub display: String,
    /// `Some(parent)` if this faction is a subfaction (Vanguard → Coalition).
    /// Relation lookups walk the parent chain when no explicit
    /// override exists for a pair.
    pub parent: Option<FactionId>,
    /// Roads-feel-natural for this faction's NPCs. Drives travel
    /// style picks (`RoadHugger` / `Mixed` / `Bushwhacker`) in
    /// `tick_npc_goals`. Subfactions inherit unless explicitly set.
    pub road_friendly: bool,
    /// Baseline aggression `[0, 1]` jittered per-NPC at spawn.
    pub base_aggression: f32,
    /// Default loadout id (resolves against the loadout registry).
    pub default_loadout: String,
    /// RGB color for debug overlays — minimap dots, marker pills,
    /// dev-mode NPC tints. NOT the production rendering color (which
    /// comes from per-NPC outfit material). 0..=255 per channel.
    pub debug_color: [u8; 3],
    /// Personality archetype that seeds NPC trait probabilities at
    /// roll time. TOML-driven (`archetype = "disciplined"` etc.); a
    /// faction that omits it falls back to
    /// [`crate::components::PersonalityArchetype::Default`]. The
    /// archetype tag set is `disciplined / aggressive / greedy /
    /// curious / reverent / default` (see
    /// [`crate::components::PersonalityArchetype`]).
    pub archetype: crate::components::PersonalityArchetype,
    /// Optional per-faction skew on the multicultural name pool.
    /// Each entry maps a `NationalityBucket::name()` snake-case key
    /// to a non-negative weight. Empty (default) → uniform draw
    /// across all buckets. A faction with `{ "latin_american": 3,
    /// "american": 2, "western_european": 1 }` rolls Latin-American
    /// names 3× as often as Western-European, etc. Buckets not
    /// listed get weight 0 (excluded). Single-bucket entries are
    /// allowed when a faction is demographically narrow.
    pub nationality_weights: HashMap<String, u32>,
    /// Optional override for the male/female first-name split.
    /// Range `[0.0, 1.0]`; 0 = all-female, 1 = all-male. `None`
    /// → falls back to [`crate::names::DEFAULT_MALE_NAME_WEIGHT`]
    /// (heavily male, matches the setting's combatant population).
    /// Use this when a faction has a meaningfully different
    /// gender mix from the default — e.g. a medical / civilian
    /// faction might set `0.55`, a strict-doctrine military
    /// subfaction might set `0.98`.
    pub male_name_weight: Option<f32>,
    /// Squad size rolled at spawn (min..=max inclusive). `None` inherits
    /// the parent's value, then a global default. TOML-driven via
    /// `squad_size = { min = 3, max = 5 }`.
    pub squad_size: Option<SquadSize>,
    /// GOAP combat-doctrine action-cost weights. `None` uses the global
    /// default. TOML-driven via `combat = { shoot = 1.5, advance = 4.5,
    /// ... }`.
    pub combat: Option<CombatCosts>,
    /// Weighted base-kind preferences used when seeding this faction's
    /// bases. Empty inherits the parent's, then a global default.
    /// TOML-driven via `base_kinds = [{ kind = "Outpost", weight = 4 }]`.
    pub base_kinds: Vec<BaseKindWeight>,
}

/// One weighted base-kind preference entry (`{ kind = "Outpost",
/// weight = 4 }`). Data-driven; the engine no longer hardcodes which
/// base kinds a faction seeds.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct BaseKindWeight {
    pub kind: crate::components::BaseKind,
    pub weight: u32,
}

/// Squad-size range for a faction's spawns. Data-driven; the engine no
/// longer hardcodes squad sizes per faction.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub struct SquadSize {
    pub min: u32,
    pub max: u32,
}

impl Default for SquadSize {
    fn default() -> Self {
        Self { min: 2, max: 4 }
    }
}

/// GOAP combat action-cost weights (doctrine) for a faction. Data-driven;
/// the engine no longer hardcodes per-faction combat tuning. Higher cost
/// makes the planner less likely to pick that action.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct CombatCosts {
    pub shoot: f32,
    pub advance: f32,
    pub move_to_cover: f32,
    pub peek: f32,
    pub flank: f32,
    pub suppress: f32,
    pub retreat: f32,
    pub reload: f32,
    pub heal_ally: f32,
}

impl Default for CombatCosts {
    fn default() -> Self {
        Self {
            shoot: 1.0,
            advance: 3.0,
            move_to_cover: 2.0,
            peek: 1.5,
            flank: 3.0,
            suppress: 2.5,
            retreat: 4.0,
            reload: 2.0,
            heal_ally: 2.5,
        }
    }
}

#[derive(Clone, Debug, Resource)]
pub struct FactionRegistry {
    defs: Vec<FactionDef>,
    by_name: HashMap<String, FactionId>,
    /// `pair_overrides[(a, b)]` is the canonical-pair score override
    /// (always stored with `a.0 <= b.0`). Lookups symmetrize via
    /// [`canonical_pair`] before reading.
    pair_overrides: HashMap<(FactionId, FactionId), i16>,
    default_relation: i16,
    /// Score returned for `relation(a, a)`. Defaults to `Warm` (the
    /// legacy semantics — same-faction members are allies that
    /// support each other in combat). Modders can set
    /// `default_self_relation = "friendly"` if they want self-pair
    /// to register as the spectrum's top band, but most callsites
    /// using `== Warm` to test "is this a squadmate" will then
    /// silently fail-closed.
    self_relation: i16,
    /// Faction id used for first-contact player rep fallback. Players
    /// implicitly belong to this faction's row in the matrix until
    /// they accumulate per-player rep.
    player_baseline: Option<FactionId>,
}

/// Runtime-mutable drift on top of the base matrix. Keyed by the
/// canonical `(min_name, max_name)` pair so the table is symmetric
/// without duplicate storage. Persisted in saves (step 7 of the
/// migration; today the resource lives only for the runtime
/// session).
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Resource)]
pub struct RelationDeltas {
    pub by_pair: HashMap<(String, String), i16>,
}

/// Per-player faction rep, isolated by SteamId. One player's bad
/// blood with the Vanguard does not bleed onto squadmates. Persisted
/// in saves (step 7).
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Resource)]
pub struct PlayerReputation {
    pub by_player: HashMap<u64, HashMap<String, i16>>,
}

impl FactionRegistry {
    /// Iterator over all faction defs in id-order. Useful for the
    /// GDScript bridge to dump the whole table.
    pub fn defs(&self) -> impl Iterator<Item = &FactionDef> {
        self.defs.iter()
    }

    pub fn count(&self) -> usize {
        self.defs.len()
    }

    pub fn def(&self, id: FactionId) -> &FactionDef {
        &self.defs[id.0 as usize]
    }

    pub fn id_of(&self, name: &str) -> Option<FactionId> {
        self.by_name.get(name).copied()
    }

    pub fn name_of(&self, id: FactionId) -> &str {
        &self.defs[id.0 as usize].name
    }

    pub fn default_relation(&self) -> i16 {
        self.default_relation
    }

    pub fn player_baseline(&self) -> Option<FactionId> {
        self.player_baseline
    }

    /// Walk `id`'s parent chain and return the first name a caller's
    /// closure recognizes. Used by gameplay tables (`weights_for`,
    /// `squad_size_for`, `pick_base_kind`) so a subfaction (Devout,
    /// Vanguard, Smugglers) inherits its parent's tuning by default.
    /// Returns `None` if neither the id nor any ancestor matched.
    pub fn resolve_with_parent_walk<T, F>(&self, id: FactionId, mut pick: F) -> Option<T>
    where
        F: FnMut(&str) -> Option<T>,
    {
        let mut current = Some(id);
        while let Some(fid) = current {
            if let Some(t) = pick(self.name_of(fid)) {
                return Some(t);
            }
            current = self.def(fid).parent;
        }
        None
    }

    /// Iterator over only the top-level factions (those without a
    /// parent). Used by region seeders that want to assign primary
    /// control to a faction that *owns* territory rather than to a
    /// subfaction (Vanguard, Devout, Directorate Recon, Registry, Recovery
    /// Division, Smugglers, Looters all have parents and never spawn as
    /// region primaries — they spawn within their parent's territory
    /// when squad-size + objective rolls call for them).
    pub fn top_level(&self) -> impl Iterator<Item = &FactionDef> {
        self.defs.iter().filter(|d| d.parent.is_none())
    }

    /// Squad-size range for `id`, inheriting from the parent chain when a
    /// subfaction doesn't author its own, then falling back to the global
    /// default. Replaces the old hardcoded `squad_size_for_known` match.
    pub fn squad_size(&self, id: FactionId) -> SquadSize {
        self.resolve_with_parent_walk(id, |name| {
            self.id_of(name).and_then(|i| self.def(i).squad_size)
        })
        .unwrap_or_default()
    }

    /// Combat-doctrine cost weights for `id`. No parent walk (matches the
    /// legacy direct-name `faction_costs`): a subfaction that doesn't
    /// author `combat` uses the default, not the parent's. Replaces the
    /// old hardcoded `faction_costs` match.
    pub fn combat_costs(&self, id: FactionId) -> CombatCosts {
        self.def(id).combat.unwrap_or_default()
    }

    /// Weighted base-kind preferences for `id`, inheriting from the
    /// parent chain when a subfaction doesn't author its own, then
    /// falling back to a generic default. Replaces the old hardcoded
    /// `base_kind_weights_for_known` match.
    pub fn base_kind_weights(&self, id: FactionId) -> Vec<(crate::components::BaseKind, u32)> {
        self.resolve_with_parent_walk(id, |name| {
            let def = self.def(self.id_of(name)?);
            if def.base_kinds.is_empty() {
                None
            } else {
                Some(
                    def.base_kinds
                        .iter()
                        .map(|w| (w.kind, w.weight))
                        .collect::<Vec<_>>(),
                )
            }
        })
        .unwrap_or_else(|| {
            vec![
                (crate::components::BaseKind::Outpost, 3),
                (crate::components::BaseKind::Safehouse, 1),
            ]
        })
    }
}

/// Faction-to-faction relation, applying parent-chain inheritance +
/// runtime drift. Self-relation defaults to `Warm` (configurable
/// via TOML `default_self_relation`).
pub fn faction_relation(
    reg: &FactionRegistry,
    deltas: &RelationDeltas,
    a: FactionId,
    b: FactionId,
) -> Relation {
    band_from_score(faction_relation_score(reg, deltas, a, b))
}

/// Continuous score variant of [`faction_relation`]. Exposes the raw
/// number for systems that want finer-grained drift logic
/// (e.g. quest givers grading "how much does this faction like us").
pub fn faction_relation_score(
    reg: &FactionRegistry,
    deltas: &RelationDeltas,
    a: FactionId,
    b: FactionId,
) -> i16 {
    if a == b {
        return reg.self_relation;
    }
    let base = base_score_with_inheritance(reg, a, b).unwrap_or(reg.default_relation);
    let key = canonical_pair_names(reg, a, b);
    let delta = deltas.by_pair.get(&key).copied().unwrap_or(0);
    base.saturating_add(delta).clamp(SCORE_MIN, SCORE_MAX)
}

/// Look up a base score for `(a, b)`, walking parent chains for
/// subfaction inheritance. Returns `None` when nothing matches —
/// caller falls back to the registry default.
fn base_score_with_inheritance(reg: &FactionRegistry, a: FactionId, b: FactionId) -> Option<i16> {
    if let Some(s) = reg.pair_overrides.get(&canonical_pair(a, b)) {
        return Some(*s);
    }
    // Walk a's parents against b.
    let mut cur = reg.def(a).parent;
    while let Some(p) = cur {
        if let Some(s) = reg.pair_overrides.get(&canonical_pair(p, b)) {
            return Some(*s);
        }
        cur = reg.def(p).parent;
    }
    // Walk b's parents against a.
    let mut cur = reg.def(b).parent;
    while let Some(p) = cur {
        if let Some(s) = reg.pair_overrides.get(&canonical_pair(a, p)) {
            return Some(*s);
        }
        cur = reg.def(p).parent;
    }
    None
}

fn canonical_pair(a: FactionId, b: FactionId) -> (FactionId, FactionId) {
    if a.0 <= b.0 {
        (a, b)
    } else {
        (b, a)
    }
}

fn canonical_pair_names(reg: &FactionRegistry, a: FactionId, b: FactionId) -> (String, String) {
    let (lo, hi) = canonical_pair(a, b);
    (reg.name_of(lo).to_string(), reg.name_of(hi).to_string())
}

/// Per-player relation. Falls back to the faction-vs-faction
/// baseline (against the registry's `player_baseline` faction) when
/// the player has no rep entry for `f`.
pub fn player_relation(
    reg: &FactionRegistry,
    rep: &PlayerReputation,
    deltas: &RelationDeltas,
    player: u64,
    f: FactionId,
) -> Relation {
    band_from_score(player_relation_score(reg, rep, deltas, player, f))
}

pub fn player_relation_score(
    reg: &FactionRegistry,
    rep: &PlayerReputation,
    deltas: &RelationDeltas,
    player: u64,
    f: FactionId,
) -> i16 {
    let name = reg.name_of(f);
    if let Some(per_faction) = rep.by_player.get(&player) {
        if let Some(score) = per_faction.get(name) {
            return (*score).clamp(SCORE_MIN, SCORE_MAX);
        }
    }
    // First contact: fall back to baseline-vs-faction relation.
    if let Some(baseline) = reg.player_baseline {
        return faction_relation_score(reg, deltas, baseline, f);
    }
    reg.default_relation
}

/// Apply a runtime drift on the faction-vs-faction matrix. `delta`
/// accumulates without clamping — only the *net* score (base +
/// accumulated delta) is clamped to `[SCORE_MIN, SCORE_MAX]` on
/// read. So a +200 push against a base of -100 fully cancels and
/// saturates the net to +100; a subsequent -50 push moves the net
/// back to +50 (it doesn't accumulate dead weight). Caller is
/// responsible for journaling so the chronicle can attribute the
/// shift.
pub fn shift_faction_relation(
    reg: &FactionRegistry,
    deltas: &mut RelationDeltas,
    a: FactionId,
    b: FactionId,
    delta: i16,
) {
    let key = canonical_pair_names(reg, a, b);
    let entry = deltas.by_pair.entry(key).or_insert(0);
    *entry = (*entry).saturating_add(delta);
}

/// Apply a runtime drift on a player's per-faction rep. Same
/// accumulator semantics as [`shift_faction_relation`] — net score
/// clamps on read.
pub fn shift_player_rep(
    reg: &FactionRegistry,
    rep: &mut PlayerReputation,
    player: u64,
    f: FactionId,
    delta: i16,
) {
    let name = reg.name_of(f).to_string();
    let entry = rep
        .by_player
        .entry(player)
        .or_default()
        .entry(name)
        .or_insert(0);
    *entry = (*entry).saturating_add(delta);
}

// ─── TOML loader ────────────────────────────────────────────────────

/// On-disk grammar. See `docs/book/src/planning/faction-registry-plan.md`
/// §4 for the full reference. New optional fields land with
/// `#[serde(default)]` so older config files keep parsing.
#[derive(Deserialize, Debug)]
struct ConfigFile {
    #[serde(default = "default_relation_default")]
    default_relation: String,
    #[serde(default = "default_self_relation_default")]
    default_self_relation: String,
    #[serde(default)]
    player_baseline: Option<String>,
    #[serde(default, rename = "faction")]
    factions: Vec<FactionEntry>,
    #[serde(default, rename = "relation")]
    relations: Vec<RelationEntry>,
}

fn default_relation_default() -> String {
    "neutral".to_string()
}

fn default_self_relation_default() -> String {
    "warm".to_string()
}

#[derive(Deserialize, Debug)]
struct FactionEntry {
    name: String,
    display: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default)]
    road_friendly: bool,
    #[serde(default = "default_aggression")]
    base_aggression: f32,
    #[serde(default = "default_loadout")]
    default_loadout: String,
    /// `debug_color` is the new canonical name. `color` is accepted
    /// as an alias for backwards-compat with overlay TOML written
    /// against the older field name.
    #[serde(default = "default_color", alias = "color")]
    debug_color: [u8; 3],
    /// `None` falls back to a name-derived archetype (the legacy
    /// hardcoded mapping). Once every faction in `factions.toml`
    /// declares its own archetype this can become required.
    #[serde(default)]
    archetype: Option<crate::components::PersonalityArchetype>,
    #[serde(default)]
    nationality_weights: HashMap<String, u32>,
    /// Optional per-faction male-name-weight override; see
    /// [`FactionDef::male_name_weight`].
    #[serde(default)]
    male_name_weight: Option<f32>,
    #[serde(default)]
    squad_size: Option<SquadSize>,
    #[serde(default)]
    combat: Option<CombatCosts>,
    #[serde(default)]
    base_kinds: Vec<BaseKindWeight>,
}

fn default_aggression() -> f32 {
    0.5
}

fn default_loadout() -> String {
    "default".to_string()
}

fn default_color() -> [u8; 3] {
    [0x88, 0x88, 0x88]
}

#[derive(Deserialize, Debug)]
struct RelationEntry {
    a: String,
    b: String,
    value: String,
}

/// Failure modes for [`load_from_path`] / [`load_from_str`].
/// Hand-rolled (no `thiserror` in workspace deps); matches the
/// crate's `anyhow`-driven error style elsewhere via the
/// `Display`/`Error` impls.
#[derive(Debug)]
pub enum RegistryError {
    Io {
        path: String,
        source: std::io::Error,
    },
    Parse {
        path: String,
        source: toml::de::Error,
    },
    DuplicateFaction(String),
    UnknownParent {
        child: String,
        parent: String,
    },
    UnknownFactionInPair(String),
    UnknownRelationValue(String),
    UnknownPlayerBaseline(String),
    SelfRelation(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "read {path}: {source}"),
            Self::Parse { path, source } => write!(f, "parse {path}: {source}"),
            Self::DuplicateFaction(name) => write!(f, "duplicate faction name: {name}"),
            Self::UnknownParent { child, parent } => {
                write!(f, "unknown parent for {child}: {parent}")
            }
            Self::UnknownFactionInPair(name) => {
                write!(f, "unknown faction in relation pair: {name}")
            }
            Self::UnknownRelationValue(v) => write!(f, "unknown relation value: {v}"),
            Self::UnknownPlayerBaseline(name) => write!(f, "unknown player_baseline: {name}"),
            Self::SelfRelation(name) => write!(f, "self-pair relation declared: {name}"),
        }
    }
}

impl std::error::Error for RegistryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Build the default registry from the embedded example content pack
/// (`content/factions/factions.toml`). Used by `Sim::new` when no
/// override pack is supplied. Cached process-wide via `OnceLock` —
/// every `Sim::new` calls this and the parse cost was paying ~30+
/// tests' tax in the test suite.
pub fn load_default() -> FactionRegistry {
    static CACHE: std::sync::OnceLock<FactionRegistry> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            let toml = crate::ContentSource::Embedded
                .read_str("factions/factions.toml")
                .expect("embedded factions.toml present");
            load_from_str(&toml).expect("embedded factions.toml must parse")
        })
        .clone()
}

/// Build the registry from an explicit content source. `Embedded`
/// routes through the cached [`load_default`]; a `Dir` pack reads
/// `factions.toml` from disk. Stable `FactionId`s come from
/// `build_registry`'s sort-by-name, so the source never perturbs
/// determinism.
pub fn load_from(src: &crate::ContentSource) -> FactionRegistry {
    match src {
        crate::ContentSource::Embedded => load_default(),
        other => {
            let text = other
                .read_str("factions/factions.toml")
                .unwrap_or_else(|e| panic!("factions content load failed: {e}"));
            load_from_str(&text).expect("factions.toml pack must parse")
        }
    }
}

/// Map a legacy [`crate::faction::Faction`] enum variant to its
/// Load a registry from a single TOML file. The mod loader
/// (`load_with_overlays`) layers more files on top.
pub fn load_from_path(path: &Path) -> Result<FactionRegistry, RegistryError> {
    let bytes = std::fs::read_to_string(path).map_err(|e| RegistryError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let parsed: ConfigFile = toml::from_str(&bytes).map_err(|e| RegistryError::Parse {
        path: path.display().to_string(),
        source: e,
    })?;
    build_registry(parsed)
}

/// Load from an in-memory TOML string. Used by tests and by the
/// modding layer when a mod ships its config inline.
pub fn load_from_str(toml_src: &str) -> Result<FactionRegistry, RegistryError> {
    let parsed: ConfigFile = toml::from_str(toml_src).map_err(|e| RegistryError::Parse {
        path: "<inline>".to_string(),
        source: e,
    })?;
    build_registry(parsed)
}

fn build_registry(cfg: ConfigFile) -> Result<FactionRegistry, RegistryError> {
    let default_relation = parse_relation_value(&cfg.default_relation)?;
    let self_relation = parse_relation_value(&cfg.default_self_relation)?;

    // Sort entries by name so FactionId assignment is stable across
    // registry rebuilds on the same TOML.
    let mut entries = cfg.factions;
    entries.sort_by(|a, b| a.name.cmp(&b.name));

    // Pass 1: assign ids, populate by_name. Defer parent resolution
    // until pass 2 (parents may reference factions later in the list,
    // even though we sort — a forward reference to a same-prefix name
    // is still possible, e.g. `pwa_aux` comes after `coalition`).
    let mut defs: Vec<FactionDef> = Vec::with_capacity(entries.len());
    let mut by_name: HashMap<String, FactionId> = HashMap::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        if by_name.contains_key(&entry.name) {
            return Err(RegistryError::DuplicateFaction(entry.name.clone()));
        }
        let id = FactionId(i as u32);
        by_name.insert(entry.name.clone(), id);
        let archetype = entry.archetype.unwrap_or_default();
        defs.push(FactionDef {
            id,
            name: entry.name.clone(),
            display: entry.display.clone(),
            parent: None, // resolved in pass 2
            road_friendly: entry.road_friendly,
            base_aggression: entry.base_aggression,
            default_loadout: entry.default_loadout.clone(),
            debug_color: entry.debug_color,
            archetype,
            nationality_weights: entry.nationality_weights.clone(),
            male_name_weight: entry.male_name_weight,
            squad_size: entry.squad_size,
            combat: entry.combat,
            base_kinds: entry.base_kinds.clone(),
        });
    }

    // Pass 2: resolve parents.
    for (i, entry) in entries.iter().enumerate() {
        if let Some(parent_name) = &entry.parent {
            let parent_id =
                by_name
                    .get(parent_name)
                    .copied()
                    .ok_or_else(|| RegistryError::UnknownParent {
                        child: entry.name.clone(),
                        parent: parent_name.clone(),
                    })?;
            defs[i].parent = Some(parent_id);
        }
    }

    // Pass 3: relations.
    let mut pair_overrides: HashMap<(FactionId, FactionId), i16> =
        HashMap::with_capacity(cfg.relations.len());
    for r in &cfg.relations {
        if r.a == r.b {
            return Err(RegistryError::SelfRelation(r.a.clone()));
        }
        let id_a = by_name
            .get(&r.a)
            .copied()
            .ok_or_else(|| RegistryError::UnknownFactionInPair(r.a.clone()))?;
        let id_b = by_name
            .get(&r.b)
            .copied()
            .ok_or_else(|| RegistryError::UnknownFactionInPair(r.b.clone()))?;
        let score = parse_relation_value(&r.value)?;
        pair_overrides.insert(canonical_pair(id_a, id_b), score);
    }

    let player_baseline = match cfg.player_baseline {
        Some(name) => Some(
            by_name
                .get(&name)
                .copied()
                .ok_or(RegistryError::UnknownPlayerBaseline(name))?,
        ),
        None => None,
    };

    Ok(FactionRegistry {
        defs,
        by_name,
        pair_overrides,
        default_relation,
        self_relation,
        player_baseline,
    })
}

fn parse_relation_value(s: &str) -> Result<i16, RegistryError> {
    // Accept named bands ("hostile", "cold", ...) OR a literal i16.
    if let Some(r) = relation_from_str(s) {
        return Ok(anchor_score(r));
    }
    if let Ok(n) = s.parse::<i16>() {
        return Ok(n.clamp(SCORE_MIN, SCORE_MAX));
    }
    Err(RegistryError::UnknownRelationValue(s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal three-faction fixture covering the inheritance case.
    /// `child` is a subfaction of `parent`; `other` is a top-level
    /// faction with a base relation to `parent` only.
    const FIXTURE: &str = r#"
default_relation = "neutral"
player_baseline = "nomads"

[[faction]]
name = "parent"
display = "Parent"
road_friendly = true
base_aggression = 0.5
default_loadout = "parent_basic"
color = [10, 20, 30]

[[faction]]
name = "child"
display = "Child"
parent = "parent"
road_friendly = true
base_aggression = 0.7
default_loadout = "child_elite"
color = [40, 50, 60]

[[faction]]
name = "other"
display = "Other"
road_friendly = false
base_aggression = 0.4
default_loadout = "other_basic"
color = [70, 80, 90]

[[faction]]
name = "nomads"
display = "Nomads"
base_aggression = 0.2
default_loadout = "wanderer"
color = [200, 200, 200]

[[relation]]
a = "parent"
b = "other"
value = "hostile"

[[relation]]
a = "parent"
b = "child"
value = "friendly"
"#;

    fn fixture_registry() -> FactionRegistry {
        load_from_str(FIXTURE).expect("fixture should parse")
    }

    #[test]
    fn loads_and_assigns_stable_ids() {
        let reg = fixture_registry();
        assert_eq!(reg.count(), 4);
        // Sorted by name: child, nomads, other, parent.
        assert_eq!(reg.name_of(FactionId(0)), "child");
        assert_eq!(reg.name_of(FactionId(1)), "nomads");
        assert_eq!(reg.name_of(FactionId(2)), "other");
        assert_eq!(reg.name_of(FactionId(3)), "parent");
        assert_eq!(reg.id_of("parent"), Some(FactionId(3)));
    }

    #[test]
    fn parent_inheritance_falls_through() {
        let reg = fixture_registry();
        let deltas = RelationDeltas::default();
        let child = reg.id_of("child").unwrap();
        let other = reg.id_of("other").unwrap();
        // child has no explicit relation to other; falls back to
        // parent ↔ other = Hostile.
        assert_eq!(
            faction_relation(&reg, &deltas, child, other),
            Relation::Hostile
        );
    }

    #[test]
    fn explicit_pair_overrides_inheritance() {
        // If we add child↔other = "warm", child no longer inherits
        // parent's hostility.
        let cfg =
            format!("{FIXTURE}\n[[relation]]\na = \"child\"\nb = \"other\"\nvalue = \"warm\"\n");
        let reg = load_from_str(&cfg).unwrap();
        let deltas = RelationDeltas::default();
        let child = reg.id_of("child").unwrap();
        let other = reg.id_of("other").unwrap();
        assert_eq!(
            faction_relation(&reg, &deltas, child, other),
            Relation::Warm
        );
    }

    #[test]
    fn drift_clamps_to_score_range() {
        let reg = fixture_registry();
        let mut deltas = RelationDeltas::default();
        let parent = reg.id_of("parent").unwrap();
        let other = reg.id_of("other").unwrap();
        // Base parent↔other = Hostile (-100). Push +200 — should
        // saturate at +100 (Friendly).
        shift_faction_relation(&reg, &mut deltas, parent, other, 200);
        assert_eq!(
            faction_relation(&reg, &deltas, parent, other),
            Relation::Friendly
        );
        // Push back -50 → score 50 (Warm).
        shift_faction_relation(&reg, &mut deltas, parent, other, -50);
        assert_eq!(
            faction_relation(&reg, &deltas, parent, other),
            Relation::Warm
        );
    }

    #[test]
    fn self_relation_defaults_to_warm() {
        // Fixture omits `default_self_relation`; default is `warm`
        // (matches legacy semantics where same-faction is squadmate-tier
        // ally, not the player-tier `friendly`).
        let reg = fixture_registry();
        let deltas = RelationDeltas::default();
        let parent = reg.id_of("parent").unwrap();
        assert_eq!(
            faction_relation(&reg, &deltas, parent, parent),
            Relation::Warm
        );
    }

    #[test]
    fn self_relation_can_be_overridden() {
        let cfg = format!("default_self_relation = \"friendly\"\n{FIXTURE}");
        let reg = load_from_str(&cfg).unwrap();
        let deltas = RelationDeltas::default();
        let parent = reg.id_of("parent").unwrap();
        assert_eq!(
            faction_relation(&reg, &deltas, parent, parent),
            Relation::Friendly
        );
    }

    #[test]
    fn missing_pair_uses_default() {
        let reg = fixture_registry();
        let deltas = RelationDeltas::default();
        // nomads vs other has no override and no parent
        // inheritance possible (both top-level). Default = Neutral.
        let nomads = reg.id_of("nomads").unwrap();
        let other = reg.id_of("other").unwrap();
        assert_eq!(
            faction_relation(&reg, &deltas, nomads, other),
            Relation::Neutral
        );
    }

    #[test]
    fn player_rep_isolates_per_player() {
        let reg = fixture_registry();
        let deltas = RelationDeltas::default();
        let mut rep = PlayerReputation::default();
        let parent = reg.id_of("parent").unwrap();
        // Player A trashes their parent rep.
        shift_player_rep(&reg, &mut rep, 1, parent, -200);
        assert_eq!(
            player_relation(&reg, &rep, &deltas, 1, parent),
            Relation::Hostile
        );
        // Player B is untouched — falls back to baseline (nomads
        // has no override vs parent → Neutral).
        assert_eq!(
            player_relation(&reg, &rep, &deltas, 2, parent),
            Relation::Neutral
        );
    }

    #[test]
    fn player_rep_first_contact_uses_baseline() {
        let reg = fixture_registry();
        let deltas = RelationDeltas::default();
        let rep = PlayerReputation::default();
        let other = reg.id_of("other").unwrap();
        // No player rep for player 7. Baseline = nomads; nomads
        // has no explicit relation to other (default Neutral).
        assert_eq!(
            player_relation(&reg, &rep, &deltas, 7, other),
            Relation::Neutral
        );
    }

    #[test]
    fn rejects_self_pair() {
        let bad =
            format!("{FIXTURE}\n[[relation]]\na = \"parent\"\nb = \"parent\"\nvalue = \"warm\"\n");
        let err = load_from_str(&bad).unwrap_err();
        matches!(err, RegistryError::SelfRelation(_));
    }

    #[test]
    fn rejects_unknown_parent() {
        let bad = r#"
default_relation = "neutral"
[[faction]]
name = "kid"
display = "Kid"
parent = "ghost"
default_loadout = "x"
color = [0, 0, 0]
"#;
        let err = load_from_str(bad).unwrap_err();
        matches!(err, RegistryError::UnknownParent { .. });
    }

    #[test]
    fn relation_value_accepts_numeric_score() {
        // Non-anchor scores work too — useful for fine-tuned matrices.
        let cfg = r#"
default_relation = "neutral"
[[faction]]
name = "a"
display = "A"
default_loadout = "x"
color = [0, 0, 0]
[[faction]]
name = "b"
display = "B"
default_loadout = "x"
color = [0, 0, 0]
[[relation]]
a = "a"
b = "b"
value = "-30"
"#;
        let reg = load_from_str(cfg).unwrap();
        let deltas = RelationDeltas::default();
        let a = reg.id_of("a").unwrap();
        let b = reg.id_of("b").unwrap();
        // -30 falls in the Cold band (-75..=-25 inclusive lower).
        assert_eq!(faction_relation(&reg, &deltas, a, b), Relation::Cold);
        assert_eq!(faction_relation_score(&reg, &deltas, a, b), -30);
    }

    /// Canary: if the canonical TOML breaks, we want this to fire
    /// instead of an opaque panic at sim startup. Registry size +
    /// presence of the full canonical roster (9 top-level + 7
    /// subfactions = 16) per the lore docs.
    #[test]
    fn default_toml_loads_clean() {
        let reg = load_default();
        assert_eq!(reg.count(), 16, "canonical TOML has 16 factions");
        for name in [
            // top-level
            "coalition",
            "homesteaders",
            "directorate",
            "syndicate",
            "consortium",
            "the_order",
            "the_afflicted",
            "raiders",
            "nomads",
            // subfactions
            "coalition_vanguard",
            "directorate_recon",
            "syndicate_enforcers",
            "consortium_recovery",
            "order_devout",
            "looters",
            "smugglers",
        ] {
            assert!(reg.id_of(name).is_some(), "registry should contain {name}",);
        }
    }

    /// Subfactions inherit their parent's relation matrix unless a
    /// specific override exists. Spot-check the raiders umbrella:
    /// looters has no explicit coalition override → inherits raiders ↔ coalition
    /// = Hostile. Smugglers's syndicate override flips raiders ↔
    /// compact (Cold) into Neutral.
    #[test]
    fn subfaction_inherits_parent_relations() {
        let reg = load_default();
        let deltas = RelationDeltas::default();
        let coalition = reg.id_of("coalition").unwrap();
        let looters = reg.id_of("looters").unwrap();
        let smugglers = reg.id_of("smugglers").unwrap();
        let compact = reg.id_of("syndicate").unwrap();
        assert_eq!(
            faction_relation(&reg, &deltas, looters, coalition),
            Relation::Hostile,
            "looters inherits raiders ↔ coalition",
        );
        assert_eq!(
            faction_relation(&reg, &deltas, smugglers, compact),
            Relation::Neutral,
            "smugglers override flips raiders ↔ compact from Cold to Neutral",
        );
        assert_eq!(
            faction_relation(&reg, &deltas, smugglers, coalition),
            Relation::Hostile,
            "smugglers inherits parent's raiders ↔ coalition",
        );
    }

    #[test]
    fn band_thresholds() {
        // Verify the documented anchor-centered bands.
        assert_eq!(band_from_score(-100), Relation::Hostile);
        assert_eq!(band_from_score(-76), Relation::Hostile);
        assert_eq!(band_from_score(-75), Relation::Hostile);
        assert_eq!(band_from_score(-74), Relation::Cold);
        assert_eq!(band_from_score(-50), Relation::Cold);
        assert_eq!(band_from_score(-25), Relation::Cold);
        assert_eq!(band_from_score(-24), Relation::Neutral);
        assert_eq!(band_from_score(0), Relation::Neutral);
        assert_eq!(band_from_score(24), Relation::Neutral);
        assert_eq!(band_from_score(25), Relation::Warm);
        assert_eq!(band_from_score(50), Relation::Warm);
        assert_eq!(band_from_score(74), Relation::Warm);
        assert_eq!(band_from_score(75), Relation::Friendly);
        assert_eq!(band_from_score(100), Relation::Friendly);
    }
}
