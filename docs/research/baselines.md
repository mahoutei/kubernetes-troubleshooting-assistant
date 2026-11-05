# Baseline teardown — RAG and function-calling projects

> **Historical record:** this document preserves earlier plans, decisions, or observations. See [current status and next milestones](../STATUS.md) for the maintained project status.


Date: 2026-10-29
Purpose: pick proven patterns, avoid reinventing them, and keep licensing clean. These repos are references, not dependencies. No code is copied from unlicensed repositories; adapted code gets an entry in the project's `ATTRIBUTIONS.md`.

## Summary

| Repo | License | Relevant idea | Verdict |
|---|---|---|---|
| [`kvsdileep/rag-for-engineers`](https://github.com/kvsdileep/rag-for-engineers) | none | K8s-docs RAG with hybrid retrieval, rerank, and a 30-question gold set | Read-only reference. Useful shape, no code reuse. |
| [`Raudaschl/rag-fusion`](https://github.com/Raudaschl/rag-fusion) | MIT | RRF formula, hybrid+rerank ablation, measured fusion trade-offs | Borrow the evaluation logic and the "hybrid first, rewrites later" conclusion. |
| [`Vikas9892/rag_evaluation`](https://github.com/Vikas9892/rag_evaluation) | none | Inspectable retrieval, Precision@K / Recall / MRR / hit-rate harness | Read-only reference for metric definitions and UI clarity. |
| [`deshpandetanmay/qlora-structured-extraction`](https://github.com/deshpandetanmay/qlora-structured-extraction) | Apache-2.0 | Artifact-backed stages, deterministic base-vs-tuned parse-and-compare | Borrow the harness design; write our own code. |
| [`aghasalim/lora-forgetting`](https://github.com/aghasalim/lora-forgetting) | MIT | Base-vs-tuned task gain plus retained-ability check, seed variance | Borrow the reporting discipline, not the ARC setup. |

## RAG references

### `kvsdileep/rag-for-engineers` (no license)

What it does: a 3-hour build over ~72 Kubernetes markdown files, moving from naive RAG to BM25 + RRF + reranker, contextual ingest, query rewriting, and an agentic loop. It ships a 30-question gold set and a results scoreboard.

What is useful:
- The module order matches our implementation order: naive baseline → hybrid → rerank → context improvements.
- Its failure-mode framing (terminology mismatch, chunk boundaries, missing context) is the right reviewer lens.
- The gold-set-first workflow validates our decision to certify questions before publishing numbers.

What we do differently:
- Corpus is the full pinned `release-1.34` docs tree, not 72 hand-picked files, with a committed commit hash and a reproducible tarball download.
- Eval set is 60 questions split 30 dev / 30 test with category quotas and a signed certification file.
- We add an LLM-judge calibration gate (Cohen's κ ≥ 0.75) and committed results JSON.
- No code reuse: the repo has no license.

### `Raudaschl/rag-fusion` (MIT)

What it does: multi-query generation + BM25/vector retrieval + RRF + reranker, with BEIR/NFCorpus experiments and confidence intervals. The README's measured conclusion is the important part: hybrid + rerank reliably helps; vector-only fusion is roughly a wash once a reranker is present; rewrites add ~1.5 s and cost money for a small, noisy gain.

What we borrow:
- RRF scoring `sum(1 / (60 + rank))` and the rank-fusion interface.
- The ablation matrix: dense vs hybrid vs hybrid+rerank, evaluated at the same k.
- The design conclusion: ship hybrid+rerank first; skip LLM query rewriting on free-tier latency and keep it as future work.

What we do differently:
- No LLM rewrite step in the headline pipeline, so there is no extra API call before retrieval.
- Eval is a custom harness over a certified domain gold set, not BEIR.
- MIT permits reuse, but the implementation here is written from the formula and standard library pieces; any adapted code will be attributed.

### `Vikas9892/rag_evaluation` (no license)

What it does: a from-scratch hybrid FAISS + BM25 pipeline with cross-encoder reranking, an evaluation lab (Precision@K, Recall, MRR, hit rate), a benchmark matrix across retrievers/top-K/reranking, and a UI that shows retrieved chunks.

What is useful:
- Metric definitions and the "show the chunks" UI principle: a reviewer should be able to see why an answer was produced.
- The benchmark matrix is the same shape as our ablation table.
- It confirms that a small, explicit harness beats a framework for interview storytelling.

What we do differently:
- LanceDB + local BGE embeddings instead of FAISS + provider embeddings, to keep the demo free and reproducible.
- Retrieval metrics are URL-normalized (fragments ignored) and paired with citation precision/recall and judge scores.
- No code reuse: the repo has no license.

## Fine-tune references

### `deshpandetanmay/qlora-structured-extraction` (Apache-2.0)

What it does: LoRA fine-tune of Qwen2.5-0.5B for strict JSON extraction, with frozen splits, an artifact-backed pipeline (`runs/<timestamp>/`), deterministic parse-and-compare scoring, and base-vs-tuned results.

What we borrow:
- The harness-is-the-point framing: the fine-tune is only credible because the base-vs-tuned comparison is deterministic.
- Stage boundaries: prepare data → train → evaluate → compare → publish, each with a durable artifact.
- "Fail loudly on the wrong device" instead of silently running a degraded configuration.

What we do differently:
- Function-calling tool schemas instead of closed-vocabulary extraction.
- Split by tool-schema hash so test tools never appear in training; this is stricter than a random row split.
- Three-way comparison: base 1.5B, LoRA adapter, prompted 70B API, plus latency and cost.
- Apache-2.0 permits adaptation; we write our own code and attribute the design influence.

### `aghasalim/lora-forgetting` (MIT)

What it does: LoRA fine-tune of Qwen2.5-1.5B-Instruct for expense extraction, reporting task gain and retained general ability side by side, including slice-level regressions and training-seed variance.

What we borrow:
- Report the gain and the cost together; a fine-tune result without a forgetting check is incomplete.
- Show per-slice and seed-level variance rather than only a headline mean.
- Note the measurement-method sensitivity: log-likelihood ranking and generative accuracy can disagree.

What we do differently:
- Forgetting check uses 200 MMLU test items, zero-shot letter matching, base vs tuned, reported as a delta.
- One seed for the headline run (free T4 budget), with seed variance called out as a limitation rather than overclaimed.
- MIT permits reuse; the implementation is independent and attributed.

## Rules applied

1. Unlicensed repos are read-only: no code, no copy-paste, no near-verbatim rewrites.
2. MIT/Apache-2.0 repos may inform design; any adapted code is marked in-source and listed in `ATTRIBUTIONS.md`.
3. Every published number comes from a committed results file, never from a README written by hand.
4. Baselines are compared on the same held-out split and the same parsing rules.
