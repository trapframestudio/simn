# Consuming SIMN as a Godot addon

SIMN is a Rust engine that bridges into Godot through a single GDExtension
crate (`simn-godot`). There's no prebuilt binary in this repo: you vendor the
crate sources into your project, build the cdylib, and point a `.gdextension`
file at it. This folder holds the bits you copy to make that wiring clean.

## The shape

Vendor SIMN into your Godot project under `addons/simn/` (we suggest
gitignoring it and syncing from a pinned commit, so SIMN stays the single
source of truth):

```
your-game/
  godot/
    addons/simn/        # synced from this repo (gitignore it)
      crates/simn-*/     #   the five engine crates
      content/           #   the generic example content pack
    simn.gdextension     # your copy of the template here (committed)
  Cargo.toml             # workspace; members point at addons/simn/crates/*
```

## Steps

1. **Vendor the source.** Copy this repo's `crates/` and `content/` into
   `addons/simn/`. Do **not** copy this repo's root `Cargo.toml` — your game's
   root `Cargo.toml` is the one and only Cargo workspace, and a second
   `[workspace]` underneath it is an error.

2. **Make your workspace build the crates.** In your game's root `Cargo.toml`:

   ```toml
   [workspace]
   resolver = "2"
   members = [
     "godot/addons/simn/crates/simn-common",
     "godot/addons/simn/crates/simn-sim",
     "godot/addons/simn/crates/simn-terrain",
     "godot/addons/simn/crates/simn-net",
     "godot/addons/simn/crates/simn-godot",
   ]

   [workspace.package]
   version = "0.1.0"
   edition = "2021"
   license = "MIT OR Apache-2.0"

   [workspace.dependencies]
   godot = { git = "https://github.com/godot-rust/gdext", branch = "master", features = ["experimental-threads"] }
   tracing = "0.1"
   anyhow = "1"
   ```

   The crates inherit `version`/`edition`/`godot`/`tracing`/`anyhow` from the
   workspace, and their internal `path = "../simn-*"` deps resolve because they
   all moved together. No edits to the crate manifests are needed.

3. **Wire the extension.** Copy `simn.gdextension.template` to
   `godot/simn.gdextension` and set the library paths to your build output. With
   the layout above, a debug build lands in your repo-root `target/`, so the
   default `res://../target/debug/...` paths just work. If your bridge links a
   native dep (Steam, etc.), add your own `[dependencies]` block.

4. **Build, then open Godot.** `cargo build -p simn-godot` from the repo root,
   then launch the editor. Godot discovers `simn.gdextension` and loads the lib;
   the `SimnExtension` classes register.

## Content

The engine embeds the generic `content/` pack via `include_dir!`, so it runs
and tests with zero external files. Ship your own game content with
`ContentSource::Overlay(dir)`: on-disk files win, anything you don't provide
falls back to the embedded base. So a game usually ships only its identity
(`factions/factions.toml`, `names/`, `ai/chatter_lines.toml`) and inherits the
rest. See the top-level `README.md`.

## The sync script

This folder ships `sync-simn.sh` so you don't have to write it. Copy it into
your project's `scripts/` and add a `scripts/SIMN_VERSION` file holding the SIMN
commit you want to pin:

```bash
cp /path/to/simn/godot-addon/sync-simn.sh  scripts/sync-simn.sh
chmod +x scripts/sync-simn.sh
echo <simn-commit-sha> > scripts/SIMN_VERSION
echo '/godot/addons/simn/' >> .gitignore
./scripts/sync-simn.sh        # vendors crates/ + content/ + builds the bridge
```

It has two modes:

- **Copy (default):** `./scripts/sync-simn.sh` drops a frozen snapshot at the
  pin into `godot/addons/simn/`. Reproducible; what contributors and CI use.
- **Link (`--link <clone>`):** symlinks `godot/addons/simn/{crates,content}` at
  a local SIMN working clone, so the engine builds live without a copy. This is
  the setup for developing the engine and your game together (see below).
  Symlinks work on Linux/macOS/WSL; native Windows needs a junction or Dev Mode.

## Developing the engine and your game together

If you're actively changing the sim, don't round-trip through copy-mode for
every edit. Clone SIMN beside your project and link it in:

```bash
./scripts/sync-simn.sh --link /path/to/simn-clone
```

Then three loops:

1. **Inner (constant):** edit the SIMN clone; `cargo build` / run your game. It
   compiles the clone's working tree through the symlink, including uncommitted
   edits. No push, no pin bump, no sync to test.
2. **Publish (when a change is solid):** `cargo test` (+ clippy/fmt) in the
   clone, then commit and push to SIMN. The engine keeps its own history.
3. **Adopt (record what your project runs on):** bump `scripts/SIMN_VERSION` to
   the pushed commit and commit it in your project. Do this on every adopted
   engine change so your main always pins a real engine commit, and anyone on
   copy mode gets exactly what you built against.

Treat the vendored `godot/addons/simn/` as read-only in copy mode (the next sync
overwrites it). In link mode you're editing the clone, which is the point.
