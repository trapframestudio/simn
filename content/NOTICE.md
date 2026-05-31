# SIMN Example Content Pack

This is the generic example content pack baked into `simn-sim` (via
`include_dir!`). It's here so the engine runs and tests on its own with
zero external files, and so the content-pack schema has a working
reference.

Everything in this directory is open source, under
[MIT](../LICENSE-MIT) or [Apache 2.0](../LICENSE-APACHE). There's no
proprietary game content here. Faction `display` strings are derived
from their keys, the names are generic placeholders, and the chatter is
a minimal default block.

A game supplies its own content with a `ContentSource::Overlay(dir)`.
Files in the overlay directory win, and anything the game doesn't provide
falls back to this base. So a game usually ships only its identity files
(`factions/factions.toml`, `names/`, `ai/chatter_lines.toml`) and
inherits all the mechanics and items from here.

The pack is organized by concern:

| Folder | Contents |
|---|---|
| `factions/` | Faction keys, relations, archetypes, weights, and per-faction tuning (squad size, combat doctrine, base-kind preferences). Displays are generic. |
| `items/` | Item roster (weapons, ammo, armor, attachments, magazines, food, medical, salvage, tools, containers). |
| `loot/` | Loot pools + loot-container kinds. |
| `crafting/` | Recipes + equipment-slot layout. |
| `combat/` | Ballistics + cover materials. |
| `poi/` | Base (POI) type behavior: nav footprint + victory flag. |
| `ai/` | NPC behavior tuning, faction loadouts, activity-point type behavior, and generic chatter. |
| `world/` | World-clock tuning. |
| `names/` | Generic placeholder first/last name pools per bucket. |
