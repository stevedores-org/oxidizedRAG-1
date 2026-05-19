# docs/archive

Historical, point-in-time documents — implementation summaries, merge/test completion notes, one-off fix write-ups, status snapshots, and analyses. They describe **past work**, not the current project, and aren't part of the user- or contributor-facing docs.

They live here (rather than being deleted) so the rationale and acceptance criteria from past PRs remain searchable. Don't link to anything in this directory from the main README or live documentation.

Layout mirrors the source tree at the time the docs were archived:

- `root/` — root-level summaries (`IMPLEMENTATION_SUMMARY.md`, `LOCAL_CI_AND_PR_CHECKS.md`)
- `graphrag-core/` — core crate implementation notes
- `graphrag-cli/` — CLI crate verification/analysis notes
- `graphrag-wasm/` — WASM crate progress, integration, build-fix, and screenshot notes

If you're updating a kept reference doc (e.g. `graphrag-wasm/ONNX_EMBEDDINGS.md`, `graphrag-core/LEIDEN_INTEGRATION.md`), edit it in place — don't add a new `*_COMPLETE.md` here.
