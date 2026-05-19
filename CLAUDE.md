# CLAUDE.md — Instructions for Claude Code

This file provides guidance for Claude Code (claude.ai CLI) when working on the oxidizedRAG codebase.

## Project Overview

oxidizedRAG is a high-performance Rust GraphRAG (Graph Retrieval-Augmented Generation) library. It's a Cargo workspace with 5 crates targeting native and WASM platforms.

## Workspace Structure

| Crate | Purpose |
|---|---|
| `graphrag-core` | Core library (rlib + cdylib). All graph, retrieval, entity, embedding, pipeline logic. Works on native AND WASM. |
| `graphrag-server` | REST API binary using Actix-web + Apistos (OpenAPI). |
| `graphrag-cli` | TUI/CLI binary using ratatui. Talks directly to graphrag-core (no HTTP). |
| `graphrag-wasm` | WASM bindings for browser-side knowledge graphs. Uses Trunk. |
| `graphrag-aivcs` | AIVCS integration: run tracking, content-addressed config, observability. |

## Build & Test Commands

```bash
# Standard workflow
just fmt-check     # cargo fmt --all -- --check
just clippy        # cargo clippy --workspace --all-targets -- -D warnings
just test          # cargo test --workspace
just ci            # all of the above + bench compile + doc build

# Nix (preferred — reproducible env)
nix develop        # enter dev shell with all tools
just flake-check   # nix flake check --print-build-logs

# local-ci (optional Go tool)
just local-ci      # runs local-ci if installed
just local-ci-fix  # auto-fix formatting
```

## Key Architecture Patterns

### Dual Sync/Async Traits
Every core abstraction has both sync and async variants:
- `Storage` / `AsyncStorage`
- `Embedder` / `AsyncEmbedder`
- `LanguageModel` / `AsyncLanguageModel`
- `Retriever` / `AsyncRetriever`

Async traits use `#[async_trait]` and require `Send + Sync`. Dynamic dispatch uses boxed type aliases:
```rust
pub type BoxedAsyncLanguageModel = Box<dyn AsyncLanguageModel<Error = GraphRAGError> + Send + Sync>;
```

### Pipeline DAG
The `pipeline` module implements typed, composable stages:
- `Stage<I, O>` trait with `execute()`, `name()`, `version()`, `metadata()`
- `CachedStage` wrapper using SHA-256 content hashing and moka in-memory cache (feature-gated behind `caching`)
- `PipelineBuilder` for DAG construction with cycle detection

### Feature Flags
graphrag-core uses extensive feature gating. Default features: `memory-storage`, `basic-retrieval`, `parallel-processing`, `async`, `ureq`, `async-traits`.

LLM integrations: `ollama` (Ollama local inference), `vllm` (vLLM/llm-d OpenAI-compatible API). Both are feature-gated. See `graphrag-core/src/ollama/` and `graphrag-core/src/vllm/`.

When adding new functionality, gate it behind a feature flag if it adds dependencies.

## Code Style Rules

- **`#![warn(missing_docs)]`** is active — all public items need doc comments
- **Clippy runs as `-D warnings`** — all warnings are errors in CI
- Max line width: 100 chars
- Cognitive complexity threshold: 30
- Max function args: 7
- Disallowed standalone variable names: `foo`, `bar`, `baz`, `quux`, `temp`, `tmp`, `thing`, `stuff`, `data` (field access like `json["data"]` is fine)
- `rustfmt.toml` uses nightly-only options; `cargo fmt` may warn on stable Rust but this is expected

## Important Gotchas

1. **Arrow version split**: `graphrag-core` uses `arrow = "52"` for Parquet, workspace root declares `arrow = "56"` for lancedb. They coexist intentionally.
2. **burn/burn-wgpu disabled**: GPU via Burn is commented out (Metal/objc build issues on Linux). Only Candle works for GPU.
3. **No rust-toolchain.toml**: Rust version managed through Nix only. No version pinning outside Nix.
4. **Server uses Actix-web, core has Axum API**: `graphrag-server` = Actix-web. `graphrag-core::api` = Axum (feature-gated). Different frameworks.
5. **Qwen3 thinking tags**: `remove_thinking_tags()` strips `<think>...</think>` from LLM output. New LLM integrations should handle reasoning model artifacts.
6. **UUID and getrandom need `js` feature for WASM**: Already configured in workspace Cargo.toml.
7. **Pre-existing CI failures**: `nix-check` may fail due to nightly rustfmt options in `.rustfmt.toml`. Use `--admin` flag when merging past CI.

## PR Workflow

- Branch off `develop`, PR back to `develop`
- `main` is the stable release branch — `develop` is merged into `main` periodically
- Atomic PRs: one concern per PR
- Run `just ci` locally before pushing
- Commit messages: conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)

## File Locations

- Core traits: `graphrag-core/src/core/traits.rs`
- Pipeline stages: `graphrag-core/src/pipeline/`
- Retrieval (hybrid, fusion, explain): `graphrag-core/src/retrieval/`
- Entity extraction: `graphrag-core/src/entity/`
- Config system: `graphrag-core/src/config/`
- Code Agent API: `graphrag-core/src/api/code_agent.rs`
- LLM integrations: `graphrag-core/src/ollama/`, `graphrag-core/src/vllm/`
- AIVCS integration: `graphrag-aivcs/src/`
- CI config: `.github/workflows/ci.yml`, `.local-ci.toml`
- Nix build: `flake.nix`
