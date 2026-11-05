# Current project status

Updated 2026-11-04. This page is the authoritative current status and completion plan. Project READMEs describe operation and limitations; dated plans, audits, ledgers, and debugging reports preserve historical decisions and evidence.

## Contributor verification

The shared CPU workflow passes 23 tooling tests, 131 RAG tests, and 54 fine-tuning tests. Lint passes for the tooling and both projects, and structural validation passes for all 60 RAG questions. Use the [root setup and commands](../README.md#contributor-checks) to reproduce these checks.

Routine checks require Python 3.12, uv 0.13.6, and the root lockfile dependencies. They do not require torch, transformers, sentence-transformers, Unsloth, model weights, credentials, CUDA, or prepared datasets. First-time dependency installation requires network access; checks then run locally. A Starlette/httpx TestClient deprecation warning remains; tests pass.

## Kubernetes Troubleshooting Assistant

**Local beta implemented:** the [application](../projects/01-rag-k8s-docs/README.md) accepts manifests/logs/events, UTF-8 text uploads with a combined 100,000-byte limit, optional Kubernetes version, and opt-in GitHub context. `/diagnose` returns validated observations, hypotheses, missing evidence, conflicts, read-only diagnostic suggestions, evidence references, tool trace, usage, and timing. `/ask` remains available. Nothing connects to clusters or executes commands.

**Verified locally:** fake-provider end-to-end API tests, strict input/reference/command validation, bounded GitHub requests, caching/rate-limit/timeouts, real public GitHub snapshot capture, all three curated demos, desktop/mobile browser interaction, upload error handling, fake-provider rendering and recovery, wheel packaging, and deployment staging. The [walkthrough](media/troubleshooting-demo.webm) and [screenshot](media/troubleshooting-demo.png) show the real recorded-demo UI. A real-index server also verified startup/readiness with 12,084 chunks; live-diagnosis readiness is separately false without a provider. The earlier recorded-mode local release report is under `projects/01-rag-k8s-docs/evals/results/beta-local-release.json`; it honestly reports degraded live readiness because that smoke server intentionally had no index or provider credentials.

**Evidence boundaries:** curated demos are authored examples, not generated model evaluations. GitHub snapshots retain real API source URLs and capture timestamps. New incident cases are synthetic drafts, not human-certified gold. Semantic metrics remain null until hash-bound manual review; schema validity and fake-provider results are not diagnosis accuracy. No base/adapter/hosted domain comparison has run, and adapter routing remains experimental.

**Retrieval refresh completed:** final index has 12,084 chunks from 1,460 files, chunker version 3 with an exact source hash, 500-token target, and 34 explicitly reported oversized/truncated chunks. [Fresh dev results](../projects/01-rag-k8s-docs/README.md#refreshed-retrieval-measurements) compare dense, hybrid, and reranked retrieval. Hybrid recall@10 is 0.94 with 164 ms observed p95 retrieval; reranking recall@5 is 0.90 with 23,154 ms p95. The service defaults to hybrid; `RAG_RETRIEVAL_MODE=hybrid_rerank` enables the expensive option. These are retrieval-only dev measurements under local shared load, not generation-quality claims. Opt-in exact-text reuse records legacy provenance; historical vectors do not pin encoder revisions. Thirty incident/mode integration outputs validate with a fake provider and real refreshed retrieval, while semantic quality remains null.

**Evaluation:** 20 incident drafts are split 10 development / 10 held-out. Direct prompting, RAG, and RAG with recorded GitHub evidence have a reproducible comparison harness. Human labels cover correctness, unsupported claims, citation support, and appropriate requests for missing evidence. Held-out runs require signing a separate incident certification file and are excluded from demos, prompt tuning, and training.

**Release gates still pending:**

1. Configure server-side provider credentials and exercise the three live incidents with `scripts/release_check.py --live`, followed by human semantic review.
2. Review/certify the gold sets before held-out evaluation and report model quality with limitations.
3. Build/run the full Docker image and verify startup/readiness and live incident behavior. Docker is not installed here. The served dependency graph now explicitly uses CPU PyTorch; the CUDA training environment remains independent.
4. Configure a Hugging Face write token and an intended Space repository, then deploy only after release checks. Neither is available here. There is no public deployed URL yet.

The reproducible commands, API contract, benchmark instructions, representative limitations, and optional browser recorder are documented in the project README and `docs/beta-release.md`.

## Function-calling fine-tuning

**Verified:** pinned raw-data checksums and row counts, conversation-preserving preprocessing, deduplication, schema-disjoint splits, prompt rendering, training configuration helpers, and compatibility checks. The real preprocessing run retained 4,097 train / 512 validation / 513 test examples; 1,051 retain conversation history. The generated manifest remains authoritative. See the [project README](../projects/02-finetune-function-calling/README.md) for preparation and training commands.

**Evidence limits:** CPU tests do not exercise CUDA, quantization, trainer integration, adapter quality, or inference. No trained-model benchmark is available. Old processed data must be regenerated with preprocessing version 2.

**Standalone Colab:** the [single-file notebook](../projects/02-finetune-function-calling/training/colab.ipynb) now embeds setup, preparation, bounded smoke training, automatic promotion to fresh full training, adapter-reload inference, strict validation metrics, optional Drive checkpoint backup/resume, and downloadable export. Generation/parity and CPU behavior are tested; GPU and Drive execution remain unverified. A smoke pass is operational readiness, not evidence of improved accuracy.

**Next milestone:** validate the standalone notebook on a fresh web-Colab NVIDIA GPU, including Drive recovery, or install the [separate locked CUDA environment](../environments/gpu/README.md), pass its real-kernel readiness check, then run the two-step GPU smoke test on regenerated data and verify validation and adapter saving. The GPU lock resolves and installation dry-runs successfully; GPU runtime validation remains pending because this development machine has no visible NVIDIA driver utility.

After that:

1. Run the notebook smoke-to-full workflow and inspect actual adapter/configuration/log artifacts.
2. Validate inference and strict structured-output metrics on hardware; malformed/missing/extra calls count as errors and duplicate calls retain their multiplicity.
3. Compare base, tuned, and prompted API models on held-out schemas, measure forgetting, and publish a model card with uncertainty, limitations, and license review.

| Activity | Additional requirements |
|---|---|
| Prepare real data | Tokenizer/data downloads and preparation dependencies |
| Smoke/full training | Accessible NVIDIA CUDA GPU; isolated locked GPU environment for Linux/WSL, or the existing Colab kernel installation path |
| API baseline | Configured provider key |
| Forgetting assessment and publication | Evaluation-data downloads; Hub credentials when publishing |

## Historical records

These record what was planned or observed at their dates. Their results and task statuses may be superseded by this page.

- [Debugging findings and verification](debugging/2026-11-04-debugging-report.md)
- [Original portfolio design](superpowers/specs/2026-10-29-llm-portfolio-design.md)
- [RAG plan](superpowers/plans/2026-10-29-rag-k8s-docs.md) and [execution ledger](superpowers/status/2026-10-29-rag-execution-ledger.md)
- [Fine-tuning plan](superpowers/plans/2026-10-29-finetune-function-calling.md) and [execution ledger](superpowers/status/2026-10-30-finetune-ledger.md)
- [RAG research audit](research/2026-10-29-plan-audit-rag.md), [fine-tuning research audit](research/2026-10-29-plan-audit-finetune.md), and [baseline research](research/baselines.md)
