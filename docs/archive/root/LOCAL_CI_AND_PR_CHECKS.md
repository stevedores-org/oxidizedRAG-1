# local-ci and Fixing Failing PR Checks (oxidizedRAG)

## What Was Done

### 1. Fixed `.local-ci.toml` schema for local-ci

The config used `cmd` / `fixcmd`, but [local-ci](https://github.com/stevedores-org/local-ci) expects **`command`** and **`fix_command`** in the TOML. That caused:

```text
stage "" has empty command
```

**Change:** All `[stages.*]` entries were updated to use `command` and `fix_command` instead of `cmd` and `fixcmd`. This matches the schema and PR #126 (“align local-ci config and docs with actual CLI schema”).

### 2. Running local-ci

After the config fix, running:

```bash
cd /path/to/oxidizedRAG
/path/to/local-ci/local-ci --no-cache
```

- **fmt** fails because `rustfmt.toml` uses **nightly-only options** (e.g. `imports_granularity`, `wrap_comments`). Stable `cargo fmt` ignores them and reports many formatting diffs. PR #127 fixes this in **Nix CI** by using nightly rustfmt there; locally you can run `cargo +nightly fmt --all` before local-ci, or rely on Nix.
- **clippy** (and then **test**) hit a **compile error** in `graphrag-core`:

  ```text
  error[E0599]: no method named `process` found for struct `Arc<(dyn Stage<I, O> + 'static)>`
     --> graphrag-core/src/pipeline/cached_stage.rs:105
  ```

  So the current `main` (or the branch you’re on) has a Rust API/trait issue that must be fixed before clippy/test can pass.

## How to Use local-ci to Help With PRs

1. **Clone and use the fixed config**  
   Ensure `.local-ci.toml` uses `command` / `fix_command` (as in this fix). Then local-ci will at least parse and run the pipeline instead of failing with “empty command”.

2. **Format (fmt)**  
   - **Option A:** Use nightly: `cargo +nightly fmt --all` then `local-ci` (or set `command = ["cargo", "+nightly", "fmt", ...]` in `.local-ci.toml` if everyone has nightly).  
   - **Option B:** Use Nix: `nix develop --command cargo fmt --all` and rely on CI (e.g. PR #127) for the canonical fmt check.

3. **Build / clippy / test**  
   Fix the `graphrag-core` compile error (trait in scope or `dyn Stage` API) so that `cargo clippy` and `cargo test` succeed. Then local-ci’s clippy and test stages will pass.

4. **CI on GitHub**  
   oxidizedRAG CI is Nix-based (`.github/workflows/ci.yml`: `nix flake check`, `nix develop --command cargo test`, etc.). So:
   - **Nix check** is the source of truth for “will this PR pass CI?”.
   - **local-ci** is a fast, local mirror of the **cargo** parts (fmt, clippy, test) once the config is fixed and the code compiles.

## Open PRs and Checks

- **#129** – docs: CLAUDE.md (1 task; mergeable: unstable).  
- **#128** – ci: Attic cache (2 tasks; self-review notes on derivation vs output paths, etc.).  
- **#127** – fix: nightly rustfmt in Nix CI (2 tasks; mergeable: dirty; addresses fmt failures).  
- **#126** – fix: align local-ci config with CLI schema (same schema fix as above).  
- **#124** – docs: AI agent instruction files (2 tasks).  
- **#122** – feat: content-addressed stage-level caching (3 tasks).

Using local-ci (with the fixed `.local-ci.toml`) helps catch fmt/clippy/test issues before pushing; for the full CI story (including Nix and nightly rustfmt), rely on `nix flake check` and the workflows.
