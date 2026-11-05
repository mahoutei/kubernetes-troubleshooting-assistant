# RAG execution ledger (checkpointed)

> **Historical record:** this document preserves earlier plans, decisions, or observations. See [current status and next milestones](../../STATUS.md) for the maintained project status.


> Checkpoint of the SDD execution ledger that previously lived only in an untracked `.superpowers/` directory of the external build worktree. Kept in-repo so the rulings that explain plan-vs-code drift survive.

- Plan: `docs/superpowers/plans/2026-10-29-rag-k8s-docs.md`
- Spec: `docs/superpowers/specs/2026-10-29-llm-portfolio-design.md`
- Baseline commit: `a96be06`
- Checkpointed: 2026-10-29 (branch `feat/rag-eval`)

## Execution update — 2026-10-29 evening (`feat/rag-eval`)

- `3b6f5c9` integration + checkpointed ledger; `6440aee` retired-model fix + fallback + typed `Usage`; `ce0248b` judge/runner/calibration/ablation harness; `d209300` certificate-state-independent tests; `93b4297` batched embeddings + index-build logging; `a7a32e3` plan/spec reconciliation + audit memos.
- Verification at `a7a32e3`: 72 tests pass, `ruff check` clean, gold set validates.

## W4 evaluation results (2026-10-29 late)

Full 500-token index: 11,744 chunks from 1,460 files, 34 MB, built in 1,957.8 s on CPU (no GPU).

| run | n | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev dense | 30 | 0.860 | 0.920 | 0.920 | 0.738 | 0.766 | 49 |
| dev hybrid | 30 | 0.760 | 0.940 | 0.940 | 0.756 | 0.788 | 128 |
| dev hybrid+rerank | 30 | 0.900 | 0.940 | 0.940 | 0.795 | 0.824 | 15,063 |
| smoke hybrid (15 dev) | 15 | 0.833 | 0.900 | 0.900 | 0.800 | 0.810 | 129 |
| smoke hybrid+rerank (15 dev) | 15 | 0.867 | 0.900 | 0.900 | 0.741 | 0.783 | 15,409 |

Findings: on the 30-question dev set, hybrid raises recall@10 and nDCG over dense but lowers recall@5; the cross-encoder recovers recall@5 and improves MRR/nDCG, but costs ~15 s per query on this CPU (~118x the hybrid retrieval latency). The 15-question smoke is noisy enough that rerank's nDCG dips below hybrid there; the n=30 comparison is the one to report. No generation/judge numbers: no API key and the gold set is unsigned. 300/800 chunk-size ablations are deferred (each index build is ~33-50 min on this CPU); `scripts/run_ablations.sh` and the exact commands are ready. Results JSON is committed under `evals/results/`.

- The Task 7 deferred item "retrieval aggregates must filter unanswerable questions" is resolved: unanswerable slices now omit retrieval metrics instead of reporting 0.0.

## Shipping artifacts (2026-10-30)

- `1a081ad` adds `.github/workflows/rag.yml` (uv sync --frozen, ruff, pytest, validator), `Dockerfile`, `.dockerignore`, and `scripts/deploy_space.sh`. Docker and HF deploy are **static-verified only** (no Docker binary, no HF token, no remote).
- `5ed5b08` adds `README.md`, `ATTRIBUTIONS.md`, and the offline extractive baseline.
- Offline baseline (`--offline --mode hybrid`, n=30): citation P/R 0.600/0.500, fact coverage 0.190, refusal 0.000, p95 225 ms. The extractive answerer always answers, so this is a no-key baseline, not generation quality.
- Local checks: 72 tests, ruff, `ok: 60 questions`, CI YAML parses with PyYAML, `bash -n` passes.
- Chunk-size sweep: 300-token index built in 2,087 s — 16,664 chunks, 43 MB. Dev retrieval (n=30): dense R@5/R@10/R@20 0.840/0.900/0.920, MRR 0.729, nDCG 0.754, p95 84 ms; hybrid 0.740/0.820/0.920, MRR 0.725, nDCG 0.729, p95 254 ms; hybrid+rerank running. 800-token index built in 1,778 s — 10,014 chunks, 30 MB: dense 0.860/0.920/0.920, MRR 0.769, nDCG 0.786, p95 59 ms; hybrid 0.760/0.900/0.940, MRR 0.769, nDCG 0.783, p95 125 ms; hybrid+rerank 0.880/0.940/0.940, MRR 0.819, nDCG 0.843, p95 15,128 ms. At 300 tokens the smaller chunks hurt recall; at 800 the reranker improves MRR/nDCG over 500 but recall@5 is 0.02 lower. With n=30 these are directional. Results: `evals/results/ablation-{300,800}-*.json`.
- Human gates unchanged: API keys for generation/judge, signed certification, Docker/HF deploy, license.

## Task 8 (recorded after the original ledger ended)

- `evals/questions.jsonl`: 60 records, ids q001–q060, 30 dev / 30 test.
- Categories: 20 single-hop, 15 multi-hop, 15 troubleshooting, 10 unanswerable; per-split 10/10, 8/7, 7/8, 5/5.
- Validation: `uv run python evals/validate_questions.py evals/questions.jsonl` → `ok: 60 questions`.
- Gold-source URLs, headings, and required facts were checked against the pinned docs tree; 8 records corrected post-review (q004, q010, q022, q023, q031, q036, q049, q050, q058).
- `evals/CERTIFICATION.md` remains **unsigned**: Reviewer/Date/Notes blank. Human gate.

---

<!-- Original ledger follows verbatim. -->

# SDD ledger — plan: docs/superpowers/plans/2026-10-29-rag-k8s-docs.md

Spec: docs/superpowers/specs/2026-10-29-llm-portfolio-design.md
Started: 2026-10-29
Baseline commit: a96be06

## Pre-flight scan

### Cross-task interfaces

| Pair | Producer → consumer | Finding |
|---|---|---|
| 1 → 2 | `Chunk`, `Settings` → ingest/chunk | Signatures match; `parse_markdown` uses `chunk_tokens`/`overlap`/`count_tokens`. |
| 2 → 3 | `Chunk` list → index build | `build_index` accepts `list[Chunk]`; build script passes tokenizer counter. |
| 1,3 → 4 | `Settings`, `Index` → retriever | `Retriever` reads `top_k_*`/`rrf_k` via `getattr`; test fakes match. |
| 1,4 → 5 | `Chunk`, `Answer` → generation | `answer_question` accepts `list[Chunk]`; citations are 1-based. |
| 4,5 → 6 | `Retriever`, `answer_question` → API | API passes `retrieval.last_timings`; fake retriever updated to expose it. |
| 1 → 7 | `Settings` → eval schema | No shared state; eval schema independent. |
| 7,8 → 9 | schema/metrics/questions → runner | `Judge.score` signature `(question, candidate, reference, context)` matches runner. |
| 9 → 10 | results JSON → summarizer | Summarizer reads `metrics` and `questions` keys produced by runner. |
| 6,9,10 → 11 | API/eval → Docker/CI | CI uses validator only; Docker builds full index. |
| 10,11 → 12 | results/CI → README | README numbers must come from committed results. |

### Task self-consistency

| Task | Check | Result |
|---|---|---|
| 1 | 6 tests vs config/models code | Pass (extracted and run). |
| 2 | 7 tests vs ingest/chunk code | Found fixture-path bug; fixed. Pass after fix. |
| 3 | index roundtrip test | Not run locally (lancedb download stalled); code/API reviewed. Will run during Task 3. |
| 4 | 2 retrieval tests | Pass. |
| 5 | 3 generation tests | Pass. |
| 6 | 3 API tests | Pass; timings assertion added. |
| 7 | 9 metric/schema tests | Found missing-sources test bug; fixed. Pass after fix. |
| 8 | 60-question quotas vs validator | Consistent (20/15/15/10, 30/30). |
| 9 | 3 judge/runner tests | Pass; expected count corrected to 3. |
| 10 | summarize test + ablation script | Pass. |
| 11 | Docker/CI commands | Static review; Docker build deferred to CI/remote (no local Docker). |
| 12 | README/results/attributions | Static; publication gated on results. |

### Rulings recorded before execution

- Ruling: fix the `iter_markdown` fixture to create `docs/content/...` — the test contradicted the implementation's documented root. Cost if wrong: none; test now matches production layout.
- Ruling: change the missing-sources validator test to use `single-hop` questions — unanswerable questions are intentionally allowed no gold sources. Cost if wrong: test would misstate the validator contract.
- Ruling: add `last_timings` to the API response and fake retriever — the plan's own `Retriever` produced timings but the API omitted them. Cost if wrong: extra response field, harmless.
- Ruling: add `transformers` as a direct dependency because `scripts/build_index.py` imports `AutoTokenizer`. Cost if wrong: redundant dependency, harmless.
- Ruling: reconcile spec repo layouts with plan file names (`prompts.py`, `metrics.py`, `api_baseline.py`, `forgetting.py`; remove `offline.py`/`schema.py`/`evaluate.py`). Cost if wrong: docs drift only.
- Ruling: add a root README task to the fine-tune plan — the spec's root layout requires it and no task created it. Cost if wrong: one extra docs task.

## Task progress

Task 1: complete (commits a96be06..7fed914, review clean)
Task 1: minor (deferred): frozen-dataclass tests are weak — they assert defaults and a couple of validation errors but not immutability or every field.
Task 1: minor (deferred): direct `Settings(llm_provider="gemini")` keeps Groq model/base_url; `Settings.from_env()` is the supported provider-switch path.
Task 1: minor (deferred): `llm_api_key` appears in the dataclass repr; add `field(repr=False)` before any code logs settings.
Task 1: minor (deferred): three ruff style findings in brief-verbatim code.
Task 1: minor (deferred): `uv.lock` resolves the CUDA torch stack via sentence-transformers; evaluate a CPU-only torch index before the Docker/HF Spaces task (Task 11).
Task 1: minor (deferred): config.py:55 UP037, tests/test_config.py:1 I001, tests/test_models.py:1 F401 ruff findings.
- Ruling: use `uv sync` (dev group is installed by default) because uv 0.13.6 rejects `uv sync --dev`; `--group dev` where explicit. Cost if wrong: a documentation command needs editing.
- Ruling: override the Task 2 brief's fence-unaware `HEADING_RE` scan — the spec requires that chunking never split a code block, and the brief's own `_split_blocks` tracks fences. Fix at the root in `parse_markdown`. Cost if wrong: a few more lines in `chunk.py`, no behavior loss.

Task 2: fix round 1/5 (1 addressed, 0 open — headings inside fences; commits 0a2fbde..4d0e93c)
Task 2: complete (commits 7fed914..4d0e93c, review clean)
Task 2: minor (deferred): preamble text before the first heading is dropped.
Task 2: minor (deferred): block-level overlap can inflate a chunk to ~max_tokens + largest block; untested ceiling.
Task 2: minor (deferred): `tests/fixtures/mini_docs/*` are created but no test parses them end-to-end.
Task 2: minor (deferred): `download_and_extract` is not crash-safe across the rename; partial archive on failed stream.
Task 2: minor (deferred): `_frontmatter` only handles flat `key: value`; `draft: yes` is not recognized.
Task 2: minor (deferred): bare ````` toggle mishandles nested four-backtick fences and indented code blocks; inherited by `_split_blocks` and `_mask_fenced_lines`.

Task 3: complete (commits 4d0e93c..0a8f57f, review clean)
Task 3: minor (deferred): `dense_search` uses `row.get("_distance", 0.0)`, masking a missing field with the best possible distance.
Task 3: minor (deferred): `Index.load` materializes the vector column via `to_pylist()`, wasting memory at corpus scale.
Task 3: minor (deferred): `zip(chunks, vectors)` silently truncates on an embedder row-count mismatch.
Task 3: minor (deferred): `Index.close()` and `self.settings` are unused/half-wired; no context manager.
Task 3: minor (deferred): naive whitespace tokenizer keeps punctuation attached, costing BM25 recall.
Task 3: minor (deferred): `Index.load` does not cross-check `bm25.pkl` ids against the table or read `meta.json`.
Task 3: minor (deferred): `scripts/build_index.py` hardcodes `"release-1.34"` separately from the pinned commit.
Task 3: minor (deferred): mid-function `AutoTokenizer` import and lambda token counter.
Task 3: minor (deferred): dense-search test asserts only non-emptiness; no empty-chunk guard in `build_index`.

Task 4: complete (commits 0a8f57f..062a59b, review clean)
Task 4: minor (deferred): no committed tests for the two ablation modes or `last_timings`; ad hoc only.
Task 4: minor (deferred): `zip(candidates, scores)` silently truncates on reranker score-count mismatch.
Task 4: minor (deferred): `RetrievedChunk.score` semantics differ per mode (distance vs RRF vs rerank); document or normalize before eval/API consume it.
Task 4: minor (deferred): integer millisecond truncation can report 0 for fast paths.
Task 4: minor (deferred): injected index/embedder/reranker/settings are untyped.

- Ruling: override the Task 5 brief's hardcoded offline `[1]` citation — the task contract says citations use 1-based context numbers, and wrong-source attribution defeats the citation guarantee. Cost if wrong: offline answers cite a different number than the brief's sample, which is more correct.
Task 5: fix round 1/5 (1 addressed, 0 open — offline citation index; commits f3e7b98..118bb2d)
Task 5: complete (commits 062a59b..118bb2d, review clean)
Task 5: minor (deferred): tests do not pin the exact refusal sentence or the `[2]`/u2 online numbering.
Task 5: minor (deferred): `LLMClient.complete` usage-None/content-empty handling is untested offline.
Task 5: minor (deferred): offline selection uses raw token overlap with a hand-maintained stopword set and no stemming.

- Ruling: run the Task 6 smoke test with `RAG_INDEX_DIR=data/index-smoke` — the brief's command omitted it and would have looked for the unbuilt `data/index`. Cost if wrong: one documentation command.
Task 6: fix round 1/5 (1 addressed, 0 open — hermetic API import/index guard; commits 5768b60..eceeb43)
Task 6: complete (commits 118bb2d..eceeb43, review clean)
Task 6: minor (deferred): `/health` chunk-count expression can raise if `index` is None and is hard to read.
Task 6: minor (deferred): `last_timings` is retriever-global mutable state; concurrent `/ask` calls can report another request's timings.
Task 6: minor (deferred): `CitationOut(**c.__dict__)` is brittle to future fields/slots.
Task 6: minor (deferred): `/` serves `FileResponse` even when `STATIC_DIR` is absent.
Task 6: minor (deferred): `app = None` on bad `Settings.from_env()` gives an opaque uvicorn error.
Task 6: minor (deferred): UI lacks fetch error handling, submit debounce, an input label, and `[n]` markers on sources.
Task 6: minor (deferred): broad `except Exception` in `create_app` also masks programming errors.
Task 6: minor (deferred): eager import-time app construction still loads embedder/reranker when an index exists (~13 min cold).
Task 6: minor (deferred): `tests/conftest.py` `tempfile.mkdtemp` leaks one empty directory per pytest run.

- Ruling: accept document-level metric dedupe (multiple chunks from one gold page count once) as a deliberate deviation from the brief's entry-counting formulas, and pin it with a dedicated test. Cost if wrong: reported metrics are stricter than the literal plan formulas.
Task 7: fix round 1/5 (1 addressed, 0 open — dedupe regression test; commits 9781909..0c7e338)
Task 7: complete (commits eceeb43..0c7e338, review clean)
Task 7: minor (deferred): citation precision is per-citation while recall is per-document — document the semantics for Task 8/9.
Task 7: minor (deferred): empty-gold behavior is inconsistent across metrics (recall 1.0, mrr/ndcg 0.0); retrieval aggregates must filter unanswerable questions.
Task 7: minor (deferred): validator branches for required facts and category quotas have no focused tests.
Task 7: minor (deferred): validator default path resolves against cwd and raises bare tracebacks on missing/malformed input.
Task 7: minor (deferred): `cohen_kappa` does not validate binary labels; `fact_coverage` substring matches can hit mid-word.

