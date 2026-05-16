# oxidizedRAG

A fast, modular Rust implementation of **GraphRAG** — graph-based retrieval-augmented generation — that runs natively, in a server, on the CLI, and in the browser via WebAssembly.

oxidizedRAG turns unstructured text into a knowledge graph and uses it to answer questions with multi-hop reasoning, hybrid retrieval (vector + BM25 + PageRank), and pluggable LLM backends.

## Why GraphRAG?

| | Traditional RAG | GraphRAG |
|---|---|---|
| Knowledge storage | Flat vector chunks | Interconnected knowledge graph |
| Context | Semantic similarity only | Relationships + entities + hierarchy |
| Multi-hop reasoning | Limited | Natural via graph traversal |
| Token efficiency | Baseline | Up to ~6000× reduction (LightRAG-style) |
| Accuracy | Good | ~15% better on empirical benchmarks |

## Workspace

oxidizedRAG is a Cargo workspace with five crates:

| Crate | Purpose |
|---|---|
| [`graphrag-core`](graphrag-core/) | Core library — graph construction, retrieval, entity extraction, embeddings, pipeline. Native and WASM. |
| [`graphrag-server`](graphrag-server/) | REST API binary (Actix-web + Apistos / OpenAPI 3.0). |
| [`graphrag-cli`](graphrag-cli/) | Terminal UI + CLI (Ratatui). Talks directly to `graphrag-core`. |
| [`graphrag-wasm`](graphrag-wasm/) | Browser GraphRAG (Leptos + Trunk). Runs the full pipeline in WebAssembly. |
| [`graphrag-aivcs`](graphrag-aivcs/) | AIVCS integration: run tracking, content-addressed config, observability. |

## Features

- **Three pipeline modes**: `semantic` (LLM/neural), `algorithmic` (pattern-based, no LLM), `hybrid` (RRF fusion)
- **Pluggable embeddings**: HuggingFace, OpenAI, Voyage AI, Cohere, Jina, Mistral, Together AI, Ollama, ONNX
- **Pluggable LLMs**: Ollama, vLLM / llm-d (OpenAI-compatible), WebLLM (in-browser)
- **Hybrid retrieval**: vector similarity, BM25, PageRank, adaptive
- **Knowledge graph**: incremental updates, community detection (Leiden), HippoRAG, LightRAG
- **Storage**: in-memory, Qdrant, LanceDB, SurrealDB, RocksDB-backed persistent cache
- **Content-hash stage caching**: 5–10× speedup on repeated runs with unchanged corpus
- **WebGPU acceleration** in the browser build

## Quickstart

### CLI

```bash
cargo build --release --package graphrag_cli

# Process a document (uses an example config from config/templates/)
./target/release/graphrag_cli load my_document.txt --config my_config.toml

# Query interactively (TUI)
./target/release/graphrag_cli --config my_config.toml tui

# Or query directly
./target/release/graphrag_cli --config my_config.toml query "What are the main themes?"
```

See [`graphrag-cli/README.md`](graphrag-cli/README.md) and [`graphrag-cli/USER_GUIDE.md`](graphrag-cli/USER_GUIDE.md).

### Server

```bash
# With Qdrant
docker compose -f graphrag-server/docker-compose.yml up -d
cargo run --bin graphrag-server --features qdrant

# Without external deps (in-memory)
cargo run --bin graphrag-server --no-default-features
```

API: `http://localhost:8080` — OpenAPI spec at `/openapi.json`. See [`graphrag-server/README.md`](graphrag-server/README.md).

### Browser (WASM)

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk

cd graphrag-wasm
trunk serve  # http://localhost:8080
```

See [`graphrag-wasm/README.md`](graphrag-wasm/README.md) and [`graphrag-wasm/QUICK_START.md`](graphrag-wasm/QUICK_START.md).

### Library

```toml
[dependencies]
graphrag-core = { git = "https://github.com/stevedores-org/oxidizedRAG", branch = "develop", features = ["huggingface-hub"] }
```

```rust
use graphrag_core::embeddings::huggingface::HuggingFaceEmbeddings;
use graphrag_core::embeddings::EmbeddingProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut embeddings = HuggingFaceEmbeddings::new(
        "sentence-transformers/all-MiniLM-L6-v2",
        None,
    );
    embeddings.initialize().await?;
    let v = embeddings.embed("Your text here").await?;
    println!("dim = {}", v.len());
    Ok(())
}
```

## Configuration

oxidizedRAG is configuration-driven — the same binary can run as a fast pattern-based pipeline (<10 ms entity extraction, no LLM) or a high-accuracy LLM pipeline, controlled entirely by TOML. See [`config/JSON5_CONFIG_GUIDE.md`](config/JSON5_CONFIG_GUIDE.md) and the templates in [`config/templates/`](config/templates/).

## Documentation

- [Architecture deep-dive](docs/architecture.md) — the 7-stage pipeline, end to end
- [CI architecture](docs/ci.md) — pre-commit, Nix, local-ci, GitHub Actions
- [Agents guide (`AGENTS.md`)](AGENTS.md) — context for AI coding assistants
- [`CLAUDE.md`](CLAUDE.md) — Claude Code-specific notes for this repo

## Development

```bash
just fmt-check     # cargo fmt --all -- --check
just clippy        # cargo clippy --workspace --all-targets -- -D warnings
just test          # cargo test --workspace
just ci            # full local CI

# Reproducible Nix env (preferred)
nix develop
just flake-check
```

`rustfmt.toml` uses nightly-only options; run `cargo +nightly fmt --all` for the canonical format. The CI uses nightly rustfmt; stable `cargo fmt` may warn on unstable options.

## Status

Active development. PRs land on `develop`; `develop` → `main` for releases. See open [issues](https://github.com/stevedores-org/oxidizedRAG/issues) and [PRs](https://github.com/stevedores-org/oxidizedRAG/pulls).

## License

MIT — see individual crate `Cargo.toml` for details.
