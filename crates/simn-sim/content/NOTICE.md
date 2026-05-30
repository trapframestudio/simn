# SIMN Example Content Pack

This is the generic example content pack baked into `simn-sim` (via
`include_dir!`). It's here so the engine runs and tests on its own with
zero external files, and so the content-pack schema has a working
reference.

Everything in this directory is open source, under
[MIT](../../../LICENSE-MIT) or [Apache 2.0](../../../LICENSE-APACHE).
There's no proprietary game content here. Faction `display` strings are
just derived from their keys, the names are generic placeholders, and the
chatter is a minimal default block.

A game supplies its own content with a `ContentSource::Overlay(dir)`.
Files in the overlay directory win, and anything the game doesn't provide
falls back to this base. So a game usually ships only `factions.toml`,
`names/`, and `chatter_lines.toml` (its identity) and inherits all the
mechanics and items from here.

| File(s) | Role |
|---|---|
| `ballistics.toml`, `behavior.toml`, `recipes.toml`, `equipment_slots.toml`, `cover_materials.toml`, `loot_containers.toml`, `world_time.toml` | Mechanics and tuning |
| `items/`, `loot_pools.toml` | Item roster and loot mechanics |
| `npc_loadouts.toml` | Faction-keyed item grants |
| `factions.toml` | Faction keys, relations, archetypes, weights (generic displays) |
| `names/`, `chatter_lines.toml` | Generic placeholders (override per game) |
