# Contributing to SIMN

Thanks for wanting to work on the engine. SIMN is the reusable Rust sim that
games are built on, so the bar is "does this keep the engine general, correct,
and deterministic." Here's how contributions work.

## The flow

SIMN is maintainer-gated. Nothing lands on `main` without a reviewed PR.

1. **Fork** the repo (or branch, if you're a maintainer with write access).
2. **Branch** off `main` for your change.
3. **Open a PR** against `main`. Direct pushes to `main` are blocked.
4. **A maintainer reviews it.** PRs need an approving review, and you can't
   approve your own, so a maintainer signs off.
5. **A maintainer merges it.** Only maintainers merge to `main` for now. Keep
   your PR focused and your history clean so review is quick.

If you're planning something large, open an issue first so we can agree on the
shape before you build it.

## What we expect of a change

- **Keep the core engine-agnostic.** `simn-common`, `simn-sim`, `simn-terrain`,
  and `simn-net` must compile without `godot`. Only `simn-godot` may depend on
  Godot. Anything engine-specific goes behind a trait in the core and gets
  implemented in a bridge.
- **Stay deterministic.** The sim must produce identical results from the same
  seed. Sort by a stable key before consuming RNG (never iterate a `HashMap` in
  RNG-affecting code), and don't seed RNG from non-deterministic values. Run the
  determinism tests after touching tick code.
- **Don't hardcode game content.** Factions, items, POI/activity types, and
  their tuning are data, loaded through `ContentSource`. The example pack under
  `content/` stays generic; game-specific content belongs in a consumer's
  overlay, not here.
- **Tests with behavior changes.** Add or update tests so the change is pinned.

## Before you open a PR

There's no CI gate yet, so run these yourself and make sure they're clean:

```bash
cargo test -p simn-sim                  # fast suite
cargo test -p simn-sim --test determinism
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

For broad changes, run the full suite: `cargo test -p simn-sim -- --include-ignored`.

## Licensing of contributions

SIMN is dual-licensed [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE). By
submitting a contribution you agree to license it to the project under those
same terms (inbound = outbound). Don't submit code you don't have the right to
license this way, and don't paste in GPL or other copyleft code.

## Consuming SIMN in a game

If you're here because you're building on SIMN rather than changing it, see
[`godot-addon/README.md`](godot-addon/README.md) for the vendoring + dev
workflow (copy vs link modes, pinning, the inner/publish/adopt loops).
