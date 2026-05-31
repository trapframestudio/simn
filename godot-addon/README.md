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

## Keeping in sync

Treat the vendored copy as read-only and pin it to a SIMN commit. When you want
engine changes, make them here in the SIMN repo, push, then bump your pin and
re-sync. A tiny `sync-simn.sh` in your game that clones this repo at a pinned
ref and copies `crates/` + `content/` into `addons/simn/` is all it takes.
