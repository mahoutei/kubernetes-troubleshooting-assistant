# Debugging report — 2026-11-04

> **Historical record:** this document preserves earlier plans, decisions, or observations. See [current status and next milestones](../STATUS.md) for the maintained project status.


## Scope and outcome

Reviewed both projects, ran baseline and expanded tests, exercised real dataset preparation and legacy index loading, built the RAG wheel, and staged deployment without pushing. Existing user edits and evaluation artifacts were preserved. No remote deployment, online model call, GPU training, or new full-corpus embedding run was performed.

## Fixed failures

| Area | Failure | Correction |
|---|---|---|
| Generation evaluation | Twenty context chunks evaluated an application serving five | Separate `--context-k` (default 5) from retrieval `--top-k` (default 20); record both |
| Judge context | Citation support was judged without matching excerpt numbers | Reuse the generator's numbered context formatting, restricted to generation context |
| Judge cache | Model/provider changes reused old scores | Cache includes rubric v2, prompt, provider, model, and inputs |
| Request metadata | Shared timing state could cross concurrent requests | Context-local timing storage plus a response-local snapshot |
| Provider fallback | Empty or malformed successful responses bypassed fallback | Validate responses within the fallback loop; fail loudly when all endpoints fail |
| Chunking | Introductions were dropped; masked fence whitespace could swallow code after a heading | Preserve preambles; use horizontal whitespace in heading matching |
| Markdown fences | Tilde, longer, and indented fences were parsed incorrectly | Track delimiter type and length |
| Duplicate chunks | All three saved indexes contain repeated IDs | Deduplicate identical parser output; reject duplicate build inputs; normalize legacy retrieval in memory with a warning |
| Index consistency | Model mismatch and inconsistent BM25/table artifacts were accepted | Validate model, IDs, counts, conflicting duplicate rows, and embedding matrix before mutation |
| Index memory | Loading metadata materialized every vector | Project metadata columns before materialization |
| Encoder limits | Inputs beyond 512 tokens were silently truncated | Warn and record encoder limit/truncated count in new index metadata |
| API startup | Importing the API loaded models and could hide configuration errors | Load resources in lifespan; invalid settings fail explicitly |
| API/UI | Nullable health index, missing assets, network errors, and citation gaps were mishandled | Safe health count, asset 404, fetch recovery, submit disabling, preserved citation numbers |
| Training examples | Clarification responses lost their earlier request | Preserve preceding user/assistant messages in serialization, prompts, and deduplication |
| Data integrity | Counts alone could accept substituted split members | Compare exact example multisets as well as schema ownership; validate split ratios |
| Training compatibility | Old or differently tokenized processed data could be trained silently | Version preprocessing and validate manifest/model/sequence length before GPU imports |
| Smoke validation | Two steps never reached validation scheduled at step 200 | Schedule validation within a bounded smoke run |
| GPU initialization | Unsloth imported after trainer libraries | Import Unsloth first |
| Fine-tuning CI | No workflow covered this project | Add independent CPU tests and lint workflow |
| Docker packaging | Dependency layer attempted project installation without source | Install dependencies separately, then install the project after copying source |
| Packaged UI | Wheel installation changes the source-relative static path | Explicit `RAG_STATIC_DIR` in the container |
| Deployment | Staging copied the entire working directory | Allowlist deployment inputs; exclude bytecode; add `--dry-run` |

## Verification

- Baseline: RAG 72 tests; fine-tuning 24 tests; both lint clean.
- Final: RAG 96 tests; fine-tuning 32 tests; both lint clean.
- RAG test runtime pins match the lockfile for pytest, ruff, FastAPI, Starlette, httpx, numpy, LanceDB, rank-bm25, pydantic, and pyarrow. Full model/GPU dependencies were not installed in this isolated test environment.
- Gold question validator: `ok: 60 questions`; this is structural validation, not human certification.
- Wheel: `uv build --wheel` succeeds. The installed wheel passes UI, health, and offline `/ask` smoke checks with an injected retriever and the configured asset path. This does not prove the Docker image builds, models initialize, or online generation succeeds.
- Deployment: both shell scripts pass `bash -n`; deployment `--dry-run` stages expected source/config/UI files without pushing. Both CI workflows parse successfully. Node was unavailable, so the UI script was not independently syntax-checked or browser-tested.
- Real raw dataset SHA-256 and row checks pass. Regenerated splits: 4,097 train / 512 validation / 513 test after length filtering. 1,051 retained examples have conversation history; no unavailable tool names or schema leaks were found.
- Real indexes load with duplicate normalization: default 11,710 unique / 11,744 rows; 300-token 16,628 / 16,664; 800-token 9,980 / 10,014. BM25 search verified on each. Original artifacts were not rewritten.
- One deprecation warning remains from the locked Starlette/httpx TestClient integration; test outcomes pass. Resolve during a deliberate dependency update.

## Result compatibility

Historical result JSON and Markdown remain unchanged. Their old chunk counts, duplicate ranking behavior, and generation context configuration must not be presented as post-fix measurements. New index builds record chunker version 2 and encoder truncation information; new evaluation payloads carry the index metadata and context size. New code can read legacy indexes with an explicit warning, but rebuilding and re-evaluating is required for current benchmark claims.

The cached encoder limit is 512 tokens. Merely choosing 800-token chunks does not make their entire text available to dense embedding. Warnings make this limitation visible; a longer-context encoder or independently evaluated segmentation strategy is still needed to resolve it.

## Completion plan

1. **Refresh retrieval evidence.** Rebuild into new index directories, inspect truncation counts, rerun dev comparisons, and review changed retrieval failures. Keep old artifacts for provenance. Select a serving mode based on measured latency/quality; current historical reranking is about 15 seconds at p95 on CPU.
2. **Validate generated answers.** Configure provider keys, run five-context dev generation, inspect refusals/citations, and collect at least 20 human labels for judge calibration. Preserve actual generator provider metadata because fallback can use the same family as the judge.
3. **Certify and evaluate the held-out RAG set.** The human owner must review and sign `evals/CERTIFICATION.md`. Keep test evaluation out of tuning.
4. **Verify deployment.** Build/run the full Docker image; check cold startup, health, missing-key behavior, `/ask`, and UI. Then deploy to the intended Space after inspecting the dry run. No deployment was performed in this debugging pass.
5. **Execute fine-tuning.** Run the two-step GPU smoke test on regenerated data, capture working dependency versions, then train. Add inference and structured-output metrics before claiming an improvement.
6. **Finish comparison and release evidence.** Compare base/tuned/API models, measure forgetting, document uncertainty, review licenses, and publish the model card and refreshed READMEs.

Further hardening candidates: crash-safe multi-file index writes and data downloads; model/tokenizer revision pins; heading-anchor fidelity; claim-level citation validation; and calibrated refusal behavior. These remain explicit limitations, not completed guarantees.
