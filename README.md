# SIMN

SIMN is the survival-sim engine behind [Noosphere](https://trapframe.studio).
We pulled it out of the game so it stands on its own and anyone can build
on it.

It runs a server-authoritative world. The sim is two-tier: entities near
players run in full detail, and the rest of the world keeps simulating as
an abstract graph in the background. On top of that it ships the systems a
survival or tactical game actually needs: inventory grids, ballistics,
wounds and medical, crafting, factions and relations, NPC squad AI, loot,
weather, and crash-tolerant save/load. It's written in Rust, engine-agnostic
at the core, with a Godot 4.x bridge.

## Status

Early days. The simulation core, persistence, content pipeline, and the
Godot bridge all work today. Still cooking: the standalone Godot addon
(reference scenes and scripts plus the `.gdextension` packaging) and the
multiplayer transport.

## Crates

| Crate | What it does |
|---|---|
| `simn-common` | Shared utilities. Engine-agnostic. |
| `simn-sim` | The world simulation and all the gameplay systems. Engine-agnostic. |
| `simn-terrain` | Canonical heightmap loader and sampler. Engine-agnostic. |
| `simn-net` | Session and transport layer (Steam P2P today). Engine-agnostic. |
| `simn-godot` | The one crate that depends on `godot`. Bridges the sim into Godot via gdext (`cdylib` + `rlib`). |

The core crates (`simn-common`, `simn-sim`, `simn-terrain`, `simn-net`)
build without Godot. That's a hard rule, not a nicety. Anything
engine-specific, like line-of-sight or terrain meshing, sits behind a
trait in the core and gets implemented in a bridge. So SIMN isn't married
to Godot. You could drive it from Bevy, a headless server, or wrap it for
another engine over FFI.

## Content packs

Your game's content doesn't get baked into the engine. It flows through
`simn_sim::ContentSource`, which has three modes:

* `Embedded` is the generic example pack at the repo-root `content/`
  folder (organized by concern: `factions/`, `items/`, `loot/`,
  `crafting/`, `combat/`, `ai/`, `world/`, `poi/`, `names/`), compiled in
  so the engine runs and tests on its own with zero external files. It's
  open source and carries no proprietary game content.
* `Dir(path)` is a complete content directory on disk.
* `Overlay(path)` layers on-disk files over the embedded base, and missing
  files fall back to embedded. So a game ships only the files it actually
  overrides (its `factions.toml`, `names/`, `chatter_lines.toml`) and
  inherits all the mechanics and items from the base.

## Build

```bash
cargo build --workspace
cargo test -p simn-sim          # fast suite
cargo build -p simn-godot       # the cdylib Godot loads (libsimn_godot.*)
```

## Development

SIMN is a normal Rust workspace, so you can develop it standalone here: edit,
`cargo test -p simn-sim`, `cargo clippy --workspace`, `cargo fmt`. (There's no
enforced commit gate yet, so run those by hand before you push.)

When you're building a game on SIMN at the same time, you don't want to push and
re-pin for every engine edit. Clone SIMN beside your game and **link** it in
with the shipped script, so the game builds your live engine edits directly:

```bash
# in your game project (after copying godot-addon/sync-simn.sh into scripts/):
./scripts/sync-simn.sh --link /path/to/this/clone
# then: edit here, cargo build / run the game, repeat. no push, no pin bump.
```

Push engine changes from this repo on their own cadence; bump your game's
`scripts/SIMN_VERSION` when it adopts a version. The full consumer workflow,
including the one-time setup and the copy-vs-link modes, is in
[`godot-addon/README.md`](godot-addon/README.md).

## License

The code is dual-licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), your pick. The bundled example content pack
is under the same permissive license.
