# AGENTS.md — Instructions for AI Coding Agents

This file provides context for AI coding agents (Codex, Copilot, Windsurf, Devin, etc.) working on oxidizedRAG.

## What is this project?

oxidizedRAG is a high-performance Rust GraphRAG (Graph Retrieval-Augmented Generation) library. It combines knowledge graphs with vector search and LLM generation for intelligent document retrieval. Targets both native platforms and WebAssembly.

## Workspace Layout

```
oxidizedRAG-1/
  graphrag-core/     # Core library (rlib + cdylib) — native + WASM
  graphrag-server/   # REST API server (Actix-web + Apistos/OpenAPI)
  graphrag-cli/      # Terminal UI (ratatui)
  graphrag-wasm/     # Browser WASM bindings (Trunk)
  graphrag-aivcs/    # Agent Version Control System integration
  tests/             # Workspace-level integration tests
  benches/           # Criterion benchmarks
  docs/              # Architecture docs
  flake.nix          # Nix build + dev shell
```

## How to Build and Test

```bash
# Enter Nix dev shell (recommended — includes all tools)
nix develop

# Or use bare Rust (stable, no version pinning outside Nix)
cargo check --workspace

# Standard checks (run before every PR)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Shorthand via justfile
just ci              # fmt + clippy + test + bench-compile + doc
just local-ci        # unified local-ci tool (if installed)
```

## Architecture at a Glance

### Dual Sync/Async Pattern
All core abstractions exist in both sync and async forms:

| Sync | Async | Purpose |
|------|-------|---------|
| `Storage` | `AsyncStorage` | Persistence |
| `Embedder` | `AsyncEmbedder` | Text-to-vector |
| `VectorStore` | `AsyncVectorStore` | Similarity search |
| `LanguageModel` | `AsyncLanguageModel` | LLM generation |
| `Retriever` | `AsyncRetriever` | Query handling |
| `EntityExtractor` | `AsyncEntityExtractor` | Named entity recognition |
| `GraphStore` | `AsyncGraphStore` | Knowledge graph CRUD |

Async traits use `#[async_trait]` from the `async-trait` crate and require `Send + Sync`.

### Pipeline System
Typed, composable stages connected as a DAG:
- `Stage<I, O>` — generic async trait for pipeline steps
- `CachedStage<I, O>` — transparent caching wrapper (SHA-256 content hashing + moka in-memory cache, feature-gated behind `caching`)
- `PipelineBuilder` — DAG construction with cycle detection and topological sort
- Batch types: `ChunkBatch`, `EmbeddingBatch`, `EntityGraphDelta`, `RetrievalSet`

### Feature Flags
graphrag-core uses extensive compile-time feature gating. Key categories:
- **Storage**: `memory-storage` (default), `persistent-storage`, `surrealdb-storage`, `persistent-cache`
- **Retrieval**: `basic-retrieval` (default), `graph-retrieval`, `hybrid-retrieval`, `pagerank`, `lightrag`
- **LLM**: `ollama`, `vllm`, `function-calling`
- **Platform**: `wasm`, `cuda`, `metal`
- **Processing**: `parallel-processing` (default), `incremental`, `code-chunking`, `corpus-processing`

Always gate new dependencies behind feature flags.

## Code Conventions

### Must-follow rules
- **`#![warn(missing_docs)]`** — every public item needs a doc comment
- **`cargo clippy -- -D warnings`** — zero warnings tolerated in CI
- Max line width: **100** characters
- Cognitive complexity threshold: **30**
- Max function parameters: **7**
- Banned standalone variable names: `foo`, `bar`, `baz`, `quux`, `temp`, `tmp`, `thing`, `stuff`, `data` (field access like `json["data"]` is fine)
- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

### Preferred patterns
- Feature-gate new optional dependencies
- Use `thiserror` for error types
- Use `serde` derive for serializable types
- Prefer `Arc<dyn Trait>` for shared trait objects
- Use `async_trait` (not bare `async fn in trait`) for trait objects

## Things That Will Surprise You

1. **Two web frameworks**: `graphrag-server` uses Actix-web. `graphrag-core::api` (feature-gated) uses Axum. They are independent.
2. **Arrow version mismatch**: `graphrag-core` uses `arrow = "52"` for Parquet; workspace root declares `arrow = "56"` for lancedb. Intentional.
3. **GPU support is partial**: burn/burn-wgpu are disabled (build issues). Only Candle backends (cuda, metal) work.
4. **No rust-toolchain.toml**: Rust version is managed via Nix's `rust-overlay`. Outside Nix, you use whatever `rustup` provides.
5. **Branch workflow**: Branch from `develop`, PR back to `develop`. `main` is the stable release branch — `develop` is merged into `main` periodically.
6. **rustfmt uses nightly options**: `.rustfmt.toml` contains settings that only work with nightly rustfmt. Warnings on stable are expected and harmless.
7. **WASM needs special handling**: `uuid` and `getrandom` use the `js` feature. `tokio` is optional (not available in WASM). The `async` feature gates tokio.
8. **LLM output post-processing**: `remove_thinking_tags()` strips `<think>...</think>` blocks (for reasoning models like Qwen3).

## Key File Map

| What | Where |
|------|-------|
| Core traits (sync + async) | `graphrag-core/src/core/traits.rs` |
| Error types | `graphrag-core/src/core/error.rs` |
| Main entry point (`GraphRAG`) | `graphrag-core/src/lib.rs` |
| Pipeline stages | `graphrag-core/src/pipeline/` |
| Retrieval (hybrid, fusion, explain) | `graphrag-core/src/retrieval/` |
| Entity extraction | `graphrag-core/src/entity/` |
| Config system | `graphrag-core/src/config/` |
| Code Agent API | `graphrag-core/src/api/code_agent.rs` |
| Shared API contracts | `graphrag-core/src/api/contracts.rs` |
| LLM integrations (Ollama, vLLM) | `graphrag-core/src/ollama/`, `graphrag-core/src/vllm/` |
| AIVCS run tracking | `graphrag-aivcs/src/` |
| Server routes | `graphrag-server/src/main.rs` |
| CLI app | `graphrag-cli/src/main.rs` |
| CI workflow | `.github/workflows/ci.yml` |
| Nix build | `flake.nix` |
| Local CI config | `.local-ci.toml` |
| Code style | `rustfmt.toml`, `clippy.toml` |

## Testing

```bash
# All workspace tests
cargo test --workspace

# Specific crate
cargo test -p graphrag-core --lib

# Feature-gated tests
cargo test -p graphrag-core --features "incremental,async" --test incremental_correctness_test
cargo test -p graphrag-core --features "surrealdb-storage" surrealdb
cargo test -p graphrag-core --features "vllm" -- vllm

# Code agent tests
cargo test -p graphrag-core --test code_agent_tests

# Integration tests (workspace root)
cargo test --test pipeline_caching
cargo test --test shared_contracts_tests
```

Dev dependencies: `criterion` (benchmarks), `pretty_assertions`, `proptest`, `tempfile`.
