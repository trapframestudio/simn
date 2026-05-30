//! Conversions between sim-side Rust types and GDScript-facing Godot
//! values.
//!
//! Two kinds of helpers live here:
//!
//! 1. **Dict / Array builders** — shape the `player_state`,
//!    `npcs_in_region`, `bases_in_region`, and inventory payloads into
//!    the exact key/type layouts that `docs/book/src/api/sim-host.md`
//!    documents. These are the single source of truth for those
//!    schemas; any change here is a public API change.
//!
//! 2. **String-enum codecs** — `snake_case` ↔ enum variant for every
//!    sim enum that crosses the Godot boundary. Kept consistent so new
//!    variants drift in one place instead of twelve.
//!
//! All helpers are `pub(super)` — visible to `sim/mod.rs` but not
//! exported outside the `sim` module.

use godot::prelude::*;
use simn_sim::{
    base_kind_to_str, BaseView, BodyPart, CraftJob, CraftStation, CraftabilityReport, DrugKind,
    EffectKind, EquipmentSlotDef, EquippedItem, FoodKind, GridInventory, InputStatus, ItemCategory,
    ItemRotation, KitRequirement, NpcView, PlayerView, ProjectileId, Recipe, SlotId, Specialty,
    SurvivalStat, ToolTier, WaterKind, WeaponConfig, Wound, WoundId, WoundKind, WoundTreatment,
};

/// Map a sim `BodyPart` back to the lowercase snake_case string
/// used in item TOML files + client-side FX listeners. Mirrors
/// the `#[serde(rename_all = "snake_case")]` on the enum — this
/// helper is the bridge-side equivalent since `serde_json` isn't
/// wired here.
pub(super) fn body_part_to_snake(part: BodyPart) -> &'static str {
    match part {
        BodyPart::Head => "head",
        BodyPart::Torso => "torso",
        BodyPart::LeftArm => "left_arm",
        BodyPart::RightArm => "right_arm",
        BodyPart::LeftLeg => "left_leg",
        BodyPart::RightLeg => "right_leg",
    }
}

/// Shape a `ProjectileSpawned` delta as a GDScript-facing dict
/// for the `SimHost::projectile_spawned` signal. See the signal
/// docs in `sim/mod.rs` for the schema.
///
/// `source_npc_id` is `Some(...)` for NPC-fired projectiles (Phase
/// 4A v2) and `None` for player-fired. GDScript can use this to
/// drive tracer color, hit-shake direction (incoming vs outgoing),
/// and AI bookkeeping.
///
/// `variant` carries the round's `AmmoVariant` family tag (Phase
/// 4B v2): `"fmj"` / `"hp"` / `"ap"` / `"tracer"` / `"overpressure"`.
/// Client uses this to pick tracer color, casing-eject SFX, and
/// (future) impact-FX selection without doing a registry lookup.
#[allow(clippy::too_many_arguments)]
pub(super) fn projectile_spawned_to_dict(
    id: ProjectileId,
    source_steam_id: u64,
    source_npc_id: Option<simn_sim::NpcId>,
    round_id: &simn_sim::ItemId,
    variant: simn_sim::AmmoVariant,
    origin: [f32; 3],
    velocity: [f32; 3],
    max_range_m: f32,
    spawned_tick: u64,
) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("id"), &Variant::from(id.0 as i64));
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("source_steam_id"),
        &Variant::from(source_steam_id as i64),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("source_npc_id"),
        &Variant::from(source_npc_id.map(|n| n.0 as i64).unwrap_or(0)),
    );
    d.set(
        &Variant::from("round_id"),
        &Variant::from(GString::from(round_id.0.as_str())),
    );
    d.set(
        &Variant::from("variant"),
        &Variant::from(GString::from(variant.as_str())),
    );
    d.set(
        &Variant::from("origin"),
        &Variant::from(Vector3::new(origin[0], origin[1], origin[2])),
    );
    d.set(
        &Variant::from("velocity"),
        &Variant::from(Vector3::new(velocity[0], velocity[1], velocity[2])),
    );
    d.set(&Variant::from("max_range_m"), &Variant::from(max_range_m));
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("spawned_tick"),
        &Variant::from(spawned_tick as i64),
    );
    d
}

/// Shape a `ProjectileImpacted` delta as a GDScript-facing dict.
///
/// `hit_player_steam_id` is the player struck by an NPC-fired
/// projectile (Phase 4A v2). Either `npc_id` or
/// `hit_player_steam_id` is non-zero on a hit; both are zero on a
/// miss / despawn. GDScript decides which side gets the local FX
/// (blood splatter on a player vs. NPC).
#[allow(clippy::too_many_arguments)]
pub(super) fn projectile_impacted_to_dict(
    id: ProjectileId,
    pos: [f32; 3],
    hit_npc: Option<simn_sim::NpcId>,
    hit_player_steam_id: Option<u64>,
    body_part: Option<BodyPart>,
    damage_applied: f32,
    penetrated: bool,
) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("id"), &Variant::from(id.0 as i64));
    d.set(
        &Variant::from("pos"),
        &Variant::from(Vector3::new(pos[0], pos[1], pos[2])),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("npc_id"),
        &Variant::from(hit_npc.map(|n| n.0 as i64).unwrap_or(0)),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("hit_player_steam_id"),
        &Variant::from(hit_player_steam_id.map(|s| s as i64).unwrap_or(0)),
    );
    d.set(
        &Variant::from("body_part"),
        &Variant::from(GString::from(
            body_part.map(body_part_to_snake).unwrap_or(""),
        )),
    );
    d.set(
        &Variant::from("damage_applied"),
        &Variant::from(damage_applied),
    );
    d.set(&Variant::from("penetrated"), &Variant::from(penetrated));
    d
}

/// Slot ids the weapons HUD + fire path cycles through. Matches the
/// `primary` / `secondary` / `sidearm` rows in `equipment_slots.toml`.
/// Kept here (rather than as Rust constants duplicating the TOML)
/// because the GDScript + HUD sides need the exact strings; the sim
/// side reads slot ids out of the registry at runtime.
pub(super) const WEAPON_SLOT_IDS: [&str; 3] = ["primary", "secondary", "sidearm"];

/// Build the return dict for `SimHost.fire_weapon`. Fixed shape so
/// GDScript never has to branch on missing keys:
///
/// | key | type | notes |
/// |---|---|---|
/// | `ok` | `bool` | `false` on any sim-side failure (dry-click, unknown slot, …). |
/// | `error` | `String` | anyhow text on failure; `""` on `ok`. |
/// | `remaining_rounds` | `int` | post-fire magazine count on success, `0` otherwise. |
///
/// Phase 2 dropped the `weapon_config` field — hit resolution runs
/// sim-side now. Trace + impact FX ride `ProjectileSpawned` /
/// `ProjectileImpacted` deltas instead of a synchronous return.
pub(super) fn fire_weapon_result(
    ok: bool,
    error: &str,
    remaining_rounds: u32,
) -> Dictionary<Variant, Variant> {
    let mut out: Dictionary<Variant, Variant> = Dictionary::new();
    out.set(&Variant::from("ok"), &Variant::from(ok));
    out.set(
        &Variant::from("error"),
        &Variant::from(GString::from(error)),
    );
    #[allow(clippy::cast_possible_wrap)]
    out.set(
        &Variant::from("remaining_rounds"),
        &Variant::from(remaining_rounds as i64),
    );
    out
}

/// Shape a [`WeaponConfig`] as the `{ caliber, damage, range_m,
/// fire_interval_s, spread_deg }` dict both `fire_weapon`'s return
/// value and `player_state.equipped_weapons[slot]` expose. Same
/// schema in both places so HUD/client code reads one shape.
pub(super) fn weapon_config_to_dict(wc: &WeaponConfig) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("caliber"),
        &Variant::from(GString::from(wc.caliber.0.as_str())),
    );
    d.set(&Variant::from("damage"), &Variant::from(wc.damage));
    d.set(&Variant::from("range_m"), &Variant::from(wc.range_m));
    d.set(
        &Variant::from("fire_interval_s"),
        &Variant::from(wc.fire_interval_s),
    );
    d.set(&Variant::from("spread_deg"), &Variant::from(wc.spread_deg));
    d
}

// ---------- Dict / Array builders ----------

/// Shape an [`NpcView`] as a GDScript-facing dict. Schema:
///
/// | key | type | notes |
/// |---|---|---|
/// | `id` | `int` | NpcId |
/// | `faction` | `String` | snake_case faction tag |
/// | `pos` | `Vector3` | world-space position |
/// | `yaw` | `float` | radians |
/// | `health` | `float` | current aggregate HP (`min(head, torso)`) |
/// | `max_health` | `float` | max HP |
/// | `body_parts` | `Dictionary` | per-part current HP, keys `head`/`torso`/`left_arm`/`right_arm`/`left_leg`/`right_leg`. Omitted only for NPCs loaded from pre-migration snapshots that haven't re-spawned. |
/// | `wounds` | `Array<Dictionary>` | active wounds on the NPC. Same per-element schema as `player_state["wounds"]` (id, body_part, kind, severity, treatment, spawned_tick, infected). Empty array for uninjured NPCs. |
/// | `goal` | `String` | goal tag — squad-objective / FSM (`idle` / `move` / `rest` / `pursue` / `patrol` / `guard` / `guard_post` / `explore` / `relieve` / `investigate` / `wander` / `regroup`) or ActiveGoal override (`hunt` / `socialize` / `loot` / `bloodsport` / `seek_medical` / `investigate_at` / `regroup_on_ally`). |
/// | `group_id` | `int` | 0 for solo NPCs |
/// | `aggro_target` | `int` | 0 if not aggroed, else target NpcId |
/// | `name` | `String` | procedural display name "First Last"; empty on legacy NPCs |
/// | `nationality` | `String` | snake_case bucket tag (e.g. `american` / `slavic`); empty on legacy NPCs |
/// | `rank` | `String` | rank tier label (`Rookie` / `Experienced` / `Veteran` / `Master` / `Legend`); empty on legacy NPCs |
/// | `combat_stance` | `String` | `approaching` / `in_cover` / `firing` / `suppressed` / `flanking` / `retreating`; empty when not in combat. |
/// | `combat_role` | `String` | squad-assigned role: `pointman` / `support` / `flanker` / `medic`; empty when not assigned. |
/// | `dwell_pose` | `String` | renderer animation hint during Rest/Guard dwell: `standing` / `sitting` / `crouching`; empty when not dwelling. |
/// | `goal_source` | `String` | arbiter source for current `ActiveGoal`: `scripted` / `survival` / `aggro_squad` / `aggro_solo` / `blackboard` / `squad_obj` / `personality` / `idle`. |
/// | `goal_priority` | `int` | numeric priority of current `ActiveGoal` (`0..=255`); cross-reference `PRIO_*` in `goal_arbitration.rs`. |
pub(super) fn npc_view_to_dict(
    view: &NpcView,
    registry: &simn_sim::FactionRegistry,
) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("id"), &Variant::from(view.id.0 as i64));
    d.set(
        &Variant::from("faction"),
        &Variant::from(GString::from(registry.name_of(view.faction))),
    );
    d.set(
        &Variant::from("pos"),
        &Variant::from(Vector3::new(view.pos[0], view.pos[1], view.pos[2])),
    );
    d.set(&Variant::from("yaw"), &Variant::from(view.yaw));
    d.set(
        &Variant::from("health"),
        &Variant::from(view.health.current),
    );
    d.set(
        &Variant::from("max_health"),
        &Variant::from(view.health.max),
    );
    if let Some(bp) = view.body_parts {
        let mut parts: Dictionary<Variant, Variant> = Dictionary::new();
        parts.set(&Variant::from("head"), &Variant::from(bp.head));
        parts.set(&Variant::from("torso"), &Variant::from(bp.torso));
        parts.set(&Variant::from("left_arm"), &Variant::from(bp.left_arm));
        parts.set(&Variant::from("right_arm"), &Variant::from(bp.right_arm));
        parts.set(&Variant::from("left_leg"), &Variant::from(bp.left_leg));
        parts.set(&Variant::from("right_leg"), &Variant::from(bp.right_leg));
        d.set(&Variant::from("body_parts"), &Variant::from(parts));
    }
    d.set(
        &Variant::from("wounds"),
        &Variant::from(wounds_to_array(&view.wounds)),
    );
    d.set(
        &Variant::from("goal"),
        &Variant::from(GString::from(view.goal)),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("group_id"),
        &Variant::from(view.group_id as i64),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("aggro_target"),
        &Variant::from(view.aggro_target as i64),
    );
    d.set(
        &Variant::from("name"),
        &Variant::from(GString::from(view.name.as_str())),
    );
    let nationality_tag = view.nationality.map(|n| n.name()).unwrap_or("");
    d.set(
        &Variant::from("nationality"),
        &Variant::from(GString::from(nationality_tag)),
    );
    let rank_tag = view.rank.map(|r| r.label()).unwrap_or("");
    d.set(
        &Variant::from("rank"),
        &Variant::from(GString::from(rank_tag)),
    );
    d.set(
        &Variant::from("combat_stance"),
        &Variant::from(GString::from(view.combat_stance.unwrap_or(""))),
    );
    d.set(
        &Variant::from("combat_role"),
        &Variant::from(GString::from(view.combat_role.unwrap_or(""))),
    );
    d.set(
        &Variant::from("dwell_pose"),
        &Variant::from(GString::from(view.dwell_pose.unwrap_or(""))),
    );
    d.set(
        &Variant::from("goal_source"),
        &Variant::from(GString::from(view.goal_source)),
    );
    d.set(
        &Variant::from("goal_priority"),
        &Variant::from(view.goal_priority as i64),
    );
    d
}

/// Shape a [`BaseView`] as a GDScript-facing dict. Schema:
///
/// | key | type | notes |
/// |---|---|---|
/// | `kind` | `String` | snake_case `BaseKind` |
/// | `faction` | `String` | snake_case faction tag |
/// | `pos` | `Vector3` | world-space position |
/// | `health` | `float` | current HP |
/// | `max_health` | `float` | max HP |
pub(super) fn base_view_to_dict(
    view: &BaseView,
    registry: &simn_sim::FactionRegistry,
) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("kind"),
        &Variant::from(GString::from(base_kind_to_str(view.kind))),
    );
    d.set(
        &Variant::from("faction"),
        &Variant::from(GString::from(registry.name_of(view.faction))),
    );
    d.set(
        &Variant::from("pos"),
        &Variant::from(Vector3::new(view.pos[0], view.pos[1], view.pos[2])),
    );
    d.set(
        &Variant::from("health"),
        &Variant::from(view.health.current),
    );
    d.set(
        &Variant::from("max_health"),
        &Variant::from(view.health.max),
    );
    d
}

/// Shape a [`PlayerView`] into the big `player_state` dict. Every key
/// exposed to GDScript is defined here; callers of `SimHost::player_state`
/// see exactly this layout, with inventory / weight / near_campfire
/// spliced on top by the `#[func]` wrapper (those aren't on `PlayerView`).
///
/// Schema: see `docs/book/src/api/sim-host.md#player_state` for the
/// full reader-facing table. The shape is:
///
/// - top-level: `region, pos, yaw, health, max_health, stamina,
///   max_stamina, hunger, thirst, fatigue, pain, radiation, toxicity`
/// - `body_parts` → nested dict, one float per limb
/// - `wounds` → array of wound dicts (`id / body_part / kind / severity
///   / treatment / spawned_tick / infected`)
/// - `active_effects` → array of effect dicts (`id / kind / applied_tick
///   / duration_ticks / intensity`)
/// - `drug_tolerance` → dict keyed by drug name → float
pub(super) fn to_state_dict(
    graph: &simn_sim::RegionGraph,
    view: &PlayerView,
) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    let region_name = graph
        .get(view.region)
        .map(|r| r.name.as_str())
        .unwrap_or("");
    d.set(
        &Variant::from("region"),
        &Variant::from(GString::from(region_name)),
    );
    d.set(
        &Variant::from("pos"),
        &Variant::from(Vector3::new(view.pos[0], view.pos[1], view.pos[2])),
    );
    d.set(&Variant::from("yaw"), &Variant::from(view.yaw));
    d.set(
        &Variant::from("health"),
        &Variant::from(view.health.current),
    );
    d.set(
        &Variant::from("max_health"),
        &Variant::from(view.health.max),
    );
    d.set(
        &Variant::from("stamina"),
        &Variant::from(view.stamina.current),
    );
    d.set(
        &Variant::from("max_stamina"),
        &Variant::from(view.stamina.max),
    );

    let mut parts: Dictionary<Variant, Variant> = Dictionary::new();
    parts.set(&Variant::from("head"), &Variant::from(view.body_parts.head));
    parts.set(
        &Variant::from("torso"),
        &Variant::from(view.body_parts.torso),
    );
    parts.set(
        &Variant::from("left_arm"),
        &Variant::from(view.body_parts.left_arm),
    );
    parts.set(
        &Variant::from("right_arm"),
        &Variant::from(view.body_parts.right_arm),
    );
    parts.set(
        &Variant::from("left_leg"),
        &Variant::from(view.body_parts.left_leg),
    );
    parts.set(
        &Variant::from("right_leg"),
        &Variant::from(view.body_parts.right_leg),
    );
    d.set(&Variant::from("body_parts"), &Variant::from(parts));

    d.set(
        &Variant::from("hunger"),
        &Variant::from(view.survival.hunger),
    );
    d.set(
        &Variant::from("thirst"),
        &Variant::from(view.survival.thirst),
    );
    d.set(
        &Variant::from("fatigue"),
        &Variant::from(view.survival.fatigue),
    );

    d.set(
        &Variant::from("wounds"),
        &Variant::from(wounds_to_array(&view.wounds)),
    );

    d.set(&Variant::from("pain"), &Variant::from(view.pain.0));
    d.set(
        &Variant::from("radiation"),
        &Variant::from(view.contamination.radiation),
    );
    d.set(
        &Variant::from("toxicity"),
        &Variant::from(view.contamination.toxicity),
    );

    let mut effect_arr: Array<Variant> = Array::new();
    for e in &view.active_effects {
        let mut ed: Dictionary<Variant, Variant> = Dictionary::new();
        #[allow(clippy::cast_possible_wrap)]
        ed.set(&Variant::from("id"), &Variant::from(e.id.0 as i64));
        ed.set(
            &Variant::from("kind"),
            &Variant::from(GString::from(effect_kind_to_str(e.kind))),
        );
        #[allow(clippy::cast_possible_wrap)]
        ed.set(
            &Variant::from("applied_tick"),
            &Variant::from(e.applied_tick as i64),
        );
        #[allow(clippy::cast_possible_wrap)]
        ed.set(
            &Variant::from("duration_ticks"),
            &Variant::from(e.duration_ticks as i64),
        );
        ed.set(&Variant::from("intensity"), &Variant::from(e.intensity));
        effect_arr.push(&Variant::from(ed));
    }
    d.set(&Variant::from("active_effects"), &Variant::from(effect_arr));

    let mut tol_dict: Dictionary<Variant, Variant> = Dictionary::new();
    for (drug, value) in &view.drug_tolerance {
        tol_dict.set(
            &Variant::from(GString::from(drug_kind_to_str(*drug))),
            &Variant::from(*value),
        );
    }
    d.set(&Variant::from("drug_tolerance"), &Variant::from(tol_dict));

    d
}

/// Shape a [`GridInventory`] as a GDScript array of grid-placed
/// stacks. Each element is a dict with keys `id, name, category,
/// count, spawned_tick, x, y, w, h, rotation`. `name` and `category`
/// come from the sim's `ItemRegistry`; `(w, h)` is the item's
/// **effective** footprint after applying rotation; unknown ids fall
/// back to the raw id string, `"misc"` category, and 1×1 footprint.
///
/// Magazine items additionally carry `caliber`, `magazine_capacity`,
/// `loaded_rounds`, and `loaded_variant` so the inventory panel can
/// render `AP 24/30` overlays and filter the "Load rounds" menu to
/// matching-caliber ammo. Non-magazine items set these to `""` /
/// `0` respectively.
pub(super) fn inventory_to_array(
    items: &simn_sim::ItemRegistry,
    grid: &GridInventory,
) -> Array<Variant> {
    let mut out: Array<Variant> = Array::new();
    for placed in &grid.items {
        let mut d: Dictionary<Variant, Variant> = Dictionary::new();
        d.set(
            &Variant::from("id"),
            &Variant::from(GString::from(placed.stack.id.0.as_str())),
        );
        let def_ref = items.get(&placed.stack.id);
        let (name, category, w, h) = match def_ref {
            Some(def) => {
                let footprint = match placed.rotation {
                    ItemRotation::Deg0 => def.size,
                    ItemRotation::Deg90 => def.size.rotated(),
                };
                (
                    def.name.as_str(),
                    item_category_to_str(def.category),
                    footprint.w,
                    footprint.h,
                )
            }
            None => (placed.stack.id.0.as_str(), "misc", 1u32, 1u32),
        };
        d.set(&Variant::from("name"), &Variant::from(GString::from(name)));
        d.set(
            &Variant::from("category"),
            &Variant::from(GString::from(category)),
        );
        #[allow(clippy::cast_possible_wrap)]
        d.set(
            &Variant::from("count"),
            &Variant::from(placed.stack.count as i64),
        );
        #[allow(clippy::cast_possible_wrap)]
        d.set(
            &Variant::from("spawned_tick"),
            &Variant::from(placed.stack.spawned_tick as i64),
        );
        #[allow(clippy::cast_possible_wrap)]
        d.set(&Variant::from("x"), &Variant::from(placed.x as i64));
        #[allow(clippy::cast_possible_wrap)]
        d.set(&Variant::from("y"), &Variant::from(placed.y as i64));
        #[allow(clippy::cast_possible_wrap)]
        d.set(&Variant::from("w"), &Variant::from(w as i64));
        #[allow(clippy::cast_possible_wrap)]
        d.set(&Variant::from("h"), &Variant::from(h as i64));
        d.set(
            &Variant::from("rotation"),
            &Variant::from(GString::from(item_rotation_to_str(placed.rotation))),
        );
        // Magazine-specific fields. Empty strings / zeros on
        // non-magazines so GDScript code can read unconditionally.
        let (caliber, capacity) = def_ref
            .and_then(|def| def.magazine_config.as_ref())
            .map(|mc| (mc.caliber.0.as_str(), mc.capacity))
            .unwrap_or(("", 0));
        d.set(
            &Variant::from("caliber"),
            &Variant::from(GString::from(caliber)),
        );
        #[allow(clippy::cast_possible_wrap)]
        d.set(
            &Variant::from("magazine_capacity"),
            &Variant::from(capacity as i64),
        );
        let loaded_rounds = placed.stack.loaded_rounds();
        #[allow(clippy::cast_possible_wrap)]
        d.set(
            &Variant::from("loaded_rounds"),
            &Variant::from(loaded_rounds as i64),
        );
        let loaded_variant = placed
            .stack
            .magazine_state
            .as_ref()
            .and_then(|ms| ms.variant.as_ref())
            .map(|v| v.0.as_str())
            .unwrap_or("");
        d.set(
            &Variant::from("loaded_variant"),
            &Variant::from(GString::from(loaded_variant)),
        );
        out.push(&Variant::from(d));
    }
    out
}

/// Build the `equipped_weapons` sub-dict for `player_state`. Keys are
/// slot ids (`"primary"` / `"secondary"` / `"sidearm"`); values are
/// either `null` (slot empty or non-weapon) or a per-slot dict:
///
/// | key | type | notes |
/// |---|---|---|
/// | `item_id` | `String` | e.g. `"rifle_aks74"` |
/// | `name` | `String` | display name from `items.toml` |
/// | `caliber` | `String` | e.g. `"5.45x39"` |
/// | `damage` | `float` | per-round damage |
/// | `range_m` | `float` | hard raycast cutoff |
/// | `fire_interval_s` | `float` | cooldown between shots |
/// | `spread_deg` | `float` | half-angle spread cone |
/// | `loaded_rounds` | `int` | `0` if no mag loaded |
/// | `magazine_capacity` | `int` | `0` if no mag loaded; from the
/// loaded mag's `magazine_config.capacity` |
/// | `has_magazine` | `bool` | `true` iff a mag is installed |
pub(super) fn equipped_weapons_to_dict(
    items: &simn_sim::ItemRegistry,
    equipment: &std::collections::HashMap<SlotId, EquippedItem>,
) -> Dictionary<Variant, Variant> {
    let mut out: Dictionary<Variant, Variant> = Dictionary::new();
    for slot in WEAPON_SLOT_IDS {
        let slot_id = simn_sim::SlotId::from(slot);
        let value = equipment
            .get(&slot_id)
            .and_then(|equipped| {
                let def = items.get(&equipped.stack.id)?;
                let wc = def.weapon_config.as_ref()?;
                let mut d = weapon_config_to_dict(wc);
                d.set(
                    &Variant::from("item_id"),
                    &Variant::from(GString::from(equipped.stack.id.0.as_str())),
                );
                d.set(
                    &Variant::from("name"),
                    &Variant::from(GString::from(def.name.as_str())),
                );
                let (loaded_rounds, mag_capacity, has_mag, loaded_variant) = equipped
                    .weapon_state
                    .as_ref()
                    .and_then(|ws| ws.loaded_magazine.as_ref())
                    .map(|mag| {
                        let cap = items
                            .get(&mag.id)
                            .and_then(|d| d.magazine_config.as_ref())
                            .map(|mc| mc.capacity)
                            .unwrap_or(0);
                        let variant_str: String = mag
                            .magazine_state
                            .as_ref()
                            .and_then(|ms| ms.variant.as_ref())
                            .map(|v| v.0.clone())
                            .unwrap_or_default();
                        (mag.loaded_rounds(), cap, true, variant_str)
                    })
                    .unwrap_or((0, 0, false, String::new()));
                #[allow(clippy::cast_possible_wrap)]
                d.set(
                    &Variant::from("loaded_rounds"),
                    &Variant::from(loaded_rounds as i64),
                );
                #[allow(clippy::cast_possible_wrap)]
                d.set(
                    &Variant::from("magazine_capacity"),
                    &Variant::from(mag_capacity as i64),
                );
                d.set(&Variant::from("has_magazine"), &Variant::from(has_mag));
                d.set(
                    &Variant::from("loaded_variant"),
                    &Variant::from(GString::from(loaded_variant.as_str())),
                );
                Some(Variant::from(d))
            })
            .unwrap_or_else(Variant::nil);
        out.set(&Variant::from(GString::from(slot)), &value);
    }
    out
}

pub(super) fn item_rotation_to_str(r: ItemRotation) -> &'static str {
    match r {
        ItemRotation::Deg0 => "0",
        ItemRotation::Deg90 => "90",
    }
}

// ---------- String-enum codecs ----------
//
// Conventions:
// - `*_to_str` is total (every variant maps to a snake_case tag).
// - `*_from_str` is partial — returns `None` for unknown strings.
//   Some codecs accept aliases (e.g. `"stim"` for `"stim_cocktail"`)
//   to keep the in-game debug overlay keybinds ergonomic.
// - Strings match the serde `#[serde(rename_all = "snake_case")]`
//   used throughout the sim, so the wire/save format and the GDScript
//   bridge stay aligned.

pub(super) fn item_category_to_str(c: ItemCategory) -> &'static str {
    match c {
        ItemCategory::Food => "food",
        ItemCategory::Drink => "drink",
        ItemCategory::Medical => "medical",
        ItemCategory::Drug => "drug",
        ItemCategory::Junk => "junk",
        ItemCategory::Component => "component",
        ItemCategory::Tool => "tool",
        ItemCategory::Misc => "misc",
        ItemCategory::HeadGear => "head_gear",
        ItemCategory::Eyes => "eyes",
        ItemCategory::ArmorVest => "armor_vest",
        ItemCategory::ChestRig => "chest_rig",
        ItemCategory::Backpack => "backpack",
        ItemCategory::WeaponPrimary => "weapon_primary",
        ItemCategory::WeaponSecondary => "weapon_secondary",
        ItemCategory::Sidearm => "sidearm",
        ItemCategory::Melee => "melee",
        ItemCategory::Magazine => "magazine",
        ItemCategory::Ammo => "ammo",
        ItemCategory::Attachment => "attachment",
    }
}

pub(super) fn body_part_from_str(s: &str) -> Option<BodyPart> {
    match s {
        "head" => Some(BodyPart::Head),
        "torso" => Some(BodyPart::Torso),
        "left_arm" => Some(BodyPart::LeftArm),
        "right_arm" => Some(BodyPart::RightArm),
        "left_leg" => Some(BodyPart::LeftLeg),
        "right_leg" => Some(BodyPart::RightLeg),
        _ => None,
    }
}

/// Shape a slice of `(WoundId, Wound)` entries as a GDScript-facing
/// `Array<Dictionary>`. Used by both `player_view_to_dict` and
/// `npc_view_to_dict` so the per-wound schema (id, body_part, kind,
/// severity, treatment, spawned_tick, infected) stays identical for
/// both actor kinds.
pub(super) fn wounds_to_array(wounds: &[(WoundId, Wound)]) -> Array<Variant> {
    let mut arr: Array<Variant> = Array::new();
    for (id, w) in wounds {
        let mut wd: Dictionary<Variant, Variant> = Dictionary::new();
        #[allow(clippy::cast_possible_wrap)]
        wd.set(&Variant::from("id"), &Variant::from(id.0 as i64));
        wd.set(
            &Variant::from("body_part"),
            &Variant::from(GString::from(body_part_to_str(w.body_part))),
        );
        wd.set(
            &Variant::from("kind"),
            &Variant::from(GString::from(wound_kind_to_str(w.kind))),
        );
        wd.set(
            &Variant::from("severity"),
            &Variant::from(i64::from(w.severity)),
        );
        wd.set(
            &Variant::from("treatment"),
            &Variant::from(GString::from(wound_treatment_to_str(w.treatment))),
        );
        #[allow(clippy::cast_possible_wrap)]
        wd.set(
            &Variant::from("spawned_tick"),
            &Variant::from(w.spawned_tick as i64),
        );
        wd.set(&Variant::from("infected"), &Variant::from(w.infected));
        arr.push(&Variant::from(wd));
    }
    arr
}

pub(super) fn body_part_to_str(p: BodyPart) -> &'static str {
    match p {
        BodyPart::Head => "head",
        BodyPart::Torso => "torso",
        BodyPart::LeftArm => "left_arm",
        BodyPart::RightArm => "right_arm",
        BodyPart::LeftLeg => "left_leg",
        BodyPart::RightLeg => "right_leg",
    }
}

pub(super) fn survival_stat_from_str(s: &str) -> Option<SurvivalStat> {
    match s {
        "hunger" => Some(SurvivalStat::Hunger),
        "thirst" => Some(SurvivalStat::Thirst),
        "fatigue" => Some(SurvivalStat::Fatigue),
        _ => None,
    }
}

pub(super) fn wound_kind_to_str(k: WoundKind) -> &'static str {
    match k {
        WoundKind::Bleed => "bleed",
    }
}

pub(super) fn wound_treatment_to_str(t: WoundTreatment) -> &'static str {
    match t {
        WoundTreatment::Untreated => "untreated",
        WoundTreatment::Disinfected => "disinfected",
        WoundTreatment::Bandaged => "bandaged",
        WoundTreatment::Stitched => "stitched",
        WoundTreatment::Tourniquet => "tourniquet",
        WoundTreatment::WoundPacked => "wound_packed",
        WoundTreatment::Healed => "healed",
    }
}

pub(super) fn drug_kind_from_str(s: &str) -> Option<DrugKind> {
    match s {
        "painkiller" => Some(DrugKind::Painkiller),
        "morphine" => Some(DrugKind::Morphine),
        "adrenaline" => Some(DrugKind::Adrenaline),
        "stim_cocktail" | "stim" => Some(DrugKind::StimCocktail),
        "anti_rad" | "antirad" => Some(DrugKind::AntiRad),
        "anti_tox" | "antitox" => Some(DrugKind::AntiTox),
        _ => None,
    }
}

pub(super) fn drug_kind_to_str(d: DrugKind) -> &'static str {
    match d {
        DrugKind::Painkiller => "painkiller",
        DrugKind::Morphine => "morphine",
        DrugKind::Adrenaline => "adrenaline",
        DrugKind::StimCocktail => "stim_cocktail",
        DrugKind::AntiRad => "anti_rad",
        DrugKind::AntiTox => "anti_tox",
    }
}

pub(super) fn effect_kind_to_str(k: EffectKind) -> &'static str {
    match k {
        EffectKind::Painkiller => "painkiller",
        EffectKind::Morphine => "morphine",
        EffectKind::Adrenaline => "adrenaline",
        EffectKind::StimCocktail => "stim_cocktail",
        EffectKind::AntiRad => "anti_rad",
        EffectKind::AntiTox => "anti_tox",
        EffectKind::AntibioticsActive => "antibiotics",
        EffectKind::Withdrawal => "withdrawal",
        EffectKind::OverdoseDisorientation => "overdose",
        EffectKind::AdrenalineCrash => "adrenaline_crash",
        EffectKind::FatigueRebound => "fatigue_rebound",
    }
}

pub(super) fn food_kind_from_str(s: &str) -> Option<FoodKind> {
    match s {
        "preserved_ration" => Some(FoodKind::PreservedRation),
        "fresh_food" => Some(FoodKind::FreshFood),
        "raw_meat" => Some(FoodKind::RawMeat),
        "cooked_meat" => Some(FoodKind::CookedMeat),
        "contaminated_food" => Some(FoodKind::ContaminatedFood),
        "field_ration" => Some(FoodKind::FieldRation),
        "energy_bar" => Some(FoodKind::EnergyBar),
        _ => None,
    }
}

pub(super) fn water_kind_from_str(s: &str) -> Option<WaterKind> {
    match s {
        "dirty_water" => Some(WaterKind::DirtyWater),
        "clean_water" => Some(WaterKind::CleanWater),
        "energy_drink" => Some(WaterKind::EnergyDrink),
        "vodka" => Some(WaterKind::Vodka),
        _ => None,
    }
}

pub(super) fn tool_tier_to_str(t: ToolTier) -> &'static str {
    match t {
        ToolTier::Basic => "basic",
        ToolTier::Advanced => "advanced",
        ToolTier::Expert => "expert",
    }
}

/// Empty string ⇒ `None`; any other unknown tag also returns `None`.
pub(super) fn tool_tier_from_str(s: &str) -> Option<Option<ToolTier>> {
    match s {
        "" | "none" => Some(None),
        "basic" => Some(Some(ToolTier::Basic)),
        "advanced" => Some(Some(ToolTier::Advanced)),
        "expert" => Some(Some(ToolTier::Expert)),
        _ => None,
    }
}

pub(super) fn specialty_to_str(s: Specialty) -> &'static str {
    match s {
        Specialty::General => "general",
        Specialty::Gunsmith => "gunsmith",
        Specialty::ArmorRepair => "armor_repair",
        Specialty::WeaponRepair => "weapon_repair",
        Specialty::DrugMaking => "drug_making",
        Specialty::Shards => "shards",
    }
}

pub(super) fn craft_station_to_str(s: CraftStation) -> &'static str {
    match s {
        CraftStation::Campfire => "campfire",
        CraftStation::BasicBench => "basic_bench",
        CraftStation::AdvancedBench => "advanced_bench",
        CraftStation::ExpertBench => "expert_bench",
    }
}

// ---------- Crafting dict builders ----------

fn kit_requirement_to_dict(kit: &KitRequirement) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("specialty"),
        &Variant::from(GString::from(specialty_to_str(kit.specialty))),
    );
    d.set(
        &Variant::from("min_tier"),
        &Variant::from(GString::from(tool_tier_to_str(kit.min_tier))),
    );
    d
}

fn item_stack_to_dict(stack: &simn_sim::ItemStack) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("id"),
        &Variant::from(GString::from(stack.id.0.as_str())),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("count"), &Variant::from(stack.count as i64));
    d
}

/// Shape one [`Recipe`] for the recipe browser. Schema:
///
/// | key | type | notes |
/// |---|---|---|
/// | `id` | `String` | recipe id |
/// | `name` | `String` | display name |
/// | `time_ticks` | `int` | duration per unit (20 ticks/sec) |
/// | `required_tool` | `String \| ""` | exact item id, or empty when none |
/// | `required_kit` | `Dictionary \| null` | `{ specialty, min_tier }` or null |
/// | `required_context` | `String \| ""` | station tag, empty when none |
/// | `inputs` | `Array[Dictionary]` | each `{ id, count }` |
/// | `outputs` | `Array[Dictionary]` | each `{ id, count }` |
pub(super) fn recipe_to_dict(recipe: &Recipe) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("id"),
        &Variant::from(GString::from(recipe.id.as_str())),
    );
    d.set(
        &Variant::from("name"),
        &Variant::from(GString::from(recipe.name.as_str())),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("time_ticks"),
        &Variant::from(recipe.time_ticks as i64),
    );
    d.set(
        &Variant::from("required_tool"),
        &Variant::from(GString::from(
            recipe
                .required_tool
                .as_ref()
                .map(|t| t.0.as_str())
                .unwrap_or(""),
        )),
    );
    match recipe.required_kit {
        Some(kit) => d.set(
            &Variant::from("required_kit"),
            &Variant::from(kit_requirement_to_dict(&kit)),
        ),
        None => d.set(&Variant::from("required_kit"), &Variant::nil()),
    }
    d.set(
        &Variant::from("required_context"),
        &Variant::from(GString::from(
            recipe
                .required_context
                .map(craft_station_to_str)
                .unwrap_or(""),
        )),
    );
    let mut inputs: Array<Variant> = Array::new();
    for s in &recipe.inputs {
        inputs.push(&Variant::from(item_stack_to_dict(s)));
    }
    d.set(&Variant::from("inputs"), &Variant::from(inputs));
    let mut outputs: Array<Variant> = Array::new();
    for s in &recipe.outputs {
        outputs.push(&Variant::from(item_stack_to_dict(s)));
    }
    d.set(&Variant::from("outputs"), &Variant::from(outputs));
    d
}

fn input_status_to_dict(s: &InputStatus) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("id"),
        &Variant::from(GString::from(s.id.0.as_str())),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("need"), &Variant::from(s.need as i64));
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("have"), &Variant::from(s.have as i64));
    d
}

/// Shape a [`CraftabilityReport`] for the recipe browser's "requires"
/// line. Schema:
///
/// | key | type | notes |
/// |---|---|---|
/// | `ok` | `bool` | true ⇔ every other field is satisfied |
/// | `inputs` | `Array[Dictionary]` | one `{id, need, have}` per recipe input |
/// | `missing_tool` | `String \| ""` | exact item id missing, or "" when satisfied |
/// | `missing_kit` | `Dictionary \| null` | `{specialty, min_tier}` of the unmet kit, or null |
/// | `wrong_station` | `String \| ""` | required station tag if not standing at it |
pub(super) fn craftability_to_dict(r: &CraftabilityReport) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(&Variant::from("ok"), &Variant::from(r.ok));
    let mut inputs: Array<Variant> = Array::new();
    for s in &r.inputs {
        inputs.push(&Variant::from(input_status_to_dict(s)));
    }
    d.set(&Variant::from("inputs"), &Variant::from(inputs));
    d.set(
        &Variant::from("missing_tool"),
        &Variant::from(GString::from(
            r.missing_tool.as_ref().map(|t| t.0.as_str()).unwrap_or(""),
        )),
    );
    match r.missing_kit {
        Some(kit) => d.set(
            &Variant::from("missing_kit"),
            &Variant::from(kit_requirement_to_dict(&kit)),
        ),
        None => d.set(&Variant::from("missing_kit"), &Variant::nil()),
    }
    d.set(
        &Variant::from("wrong_station"),
        &Variant::from(GString::from(
            r.wrong_station.map(craft_station_to_str).unwrap_or(""),
        )),
    );
    d
}

/// Shape one [`CraftJob`] for the queue widget. Schema:
///
/// | key | type | notes |
/// |---|---|---|
/// | `id` | `int` | stable job id (use with `cancel_craft`) |
/// | `recipe_id` | `String` | recipe id this job runs |
/// | `count_remaining` | `int` | units left (incl. the head/in-progress one) |
/// | `ticks_remaining` | `int` | ticks until next unit completes |
/// | `started_tick` | `int` | tick the job was queued |
pub(super) fn craft_job_to_dict(job: &CraftJob) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("id"), &Variant::from(job.id as i64));
    d.set(
        &Variant::from("recipe_id"),
        &Variant::from(GString::from(job.recipe_id.as_str())),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("count_remaining"),
        &Variant::from(job.count_remaining as i64),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("ticks_remaining"),
        &Variant::from(job.ticks_remaining as i64),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("started_tick"),
        &Variant::from(job.started_tick as i64),
    );
    d
}

/// Shape an [`EquipmentSlotDef`] as a GDScript-facing dict. The
/// Godot UI consumes this to lay out the paper doll.
///
/// | key | type | notes |
/// |---|---|---|
/// | `id` | `String` | Slot id (e.g. `"head"`, `"belt_1"`). |
/// | `label` | `String` | Display name. |
/// | `accepts` | `Array[String]` | Item-category whitelist. |
/// | `position` | `Vector2i` | Paper-doll grid coordinate. |
/// | `is_hotbar` | `bool` | |
/// | `hotbar_index` | `int` | 1-based; 0 when not hotbar. |
pub(super) fn slot_def_to_dict(def: &EquipmentSlotDef) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("id"),
        &Variant::from(GString::from(def.id.0.as_str())),
    );
    d.set(
        &Variant::from("label"),
        &Variant::from(GString::from(def.label.as_str())),
    );
    let mut accepts: Array<Variant> = Array::new();
    for cat in &def.accepts {
        accepts.push(&Variant::from(GString::from(item_category_to_str(*cat))));
    }
    d.set(&Variant::from("accepts"), &Variant::from(accepts));
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("position"),
        &Variant::from(Vector2i::new(def.position.x as i32, def.position.y as i32)),
    );
    // Paper-doll footprint in cells. Defaults to 1×1 in the TOML
    // for entries that don't override it — the UI layer multiplies
    // the doll-cell size by these to produce wide/tall slots
    // (rifles, armor vests, etc.).
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("size"),
        &Variant::from(Vector2i::new(def.size.w as i32, def.size.h as i32)),
    );
    d.set(&Variant::from("is_hotbar"), &Variant::from(def.is_hotbar));
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("hotbar_index"),
        &Variant::from(def.hotbar_index as i64),
    );
    d
}

/// Shape an [`EquippedItem`] as a GDScript-facing dict. The
/// `inner_grid` field, if present, is rendered via the same
/// [`inventory_to_array`] shape used for pockets — one entry per
/// placed stack inside the container.
///
/// | key | type | notes |
/// |---|---|---|
/// | `id` | `String` | Item id. |
/// | `name` | `String` | Display name. |
/// | `category` | `String` | See `inventory` dict table. |
/// | `count` | `int` | Stack size. |
/// | `spawned_tick` | `int` | |
/// | `inner_grid` | `Dictionary \| null` | `{ width, height, items: Array }` when the item is a container; `null` otherwise. |
pub(super) fn equipped_item_to_dict(
    items: &simn_sim::ItemRegistry,
    eq: &EquippedItem,
) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    d.set(
        &Variant::from("id"),
        &Variant::from(GString::from(eq.stack.id.0.as_str())),
    );
    let (name, category) = match items.get(&eq.stack.id) {
        Some(def) => (def.name.as_str(), item_category_to_str(def.category)),
        None => (eq.stack.id.0.as_str(), "misc"),
    };
    d.set(&Variant::from("name"), &Variant::from(GString::from(name)));
    d.set(
        &Variant::from("category"),
        &Variant::from(GString::from(category)),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("count"),
        &Variant::from(eq.stack.count as i64),
    );
    #[allow(clippy::cast_possible_wrap)]
    d.set(
        &Variant::from("spawned_tick"),
        &Variant::from(eq.stack.spawned_tick as i64),
    );
    match eq.inner_grid.as_ref() {
        Some(g) => d.set(
            &Variant::from("inner_grid"),
            &Variant::from(grid_to_dict(items, g)),
        ),
        None => d.set(&Variant::from("inner_grid"), &Variant::nil()),
    }
    d
}

/// Shape a [`GridInventory`] as `{ width, height, items }` — the
/// `items` array uses the same schema as the top-level
/// `inventory` dict ([`inventory_to_array`]).
pub(super) fn grid_to_dict(
    items: &simn_sim::ItemRegistry,
    grid: &GridInventory,
) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("width"), &Variant::from(grid.width as i64));
    #[allow(clippy::cast_possible_wrap)]
    d.set(&Variant::from("height"), &Variant::from(grid.height as i64));
    d.set(
        &Variant::from("items"),
        &Variant::from(inventory_to_array(items, grid)),
    );
    d
}

/// Shape the full [`simn_sim::Equipment`] map as a GDScript dict
/// keyed by slot id (strings). Each value is an
/// [`equipped_item_to_dict`] entry.
pub(super) fn equipment_to_dict(
    items: &simn_sim::ItemRegistry,
    eq: &std::collections::HashMap<SlotId, EquippedItem>,
) -> Dictionary<Variant, Variant> {
    let mut d: Dictionary<Variant, Variant> = Dictionary::new();
    for (slot_id, eq_item) in eq {
        d.set(
            &Variant::from(GString::from(slot_id.0.as_str())),
            &Variant::from(equipped_item_to_dict(items, eq_item)),
        );
    }
    d
}
