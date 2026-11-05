# Troubleshooting beta — verification and release

## What is usable now

The incident UI and three labeled authored demonstrations run without provider credentials or a live index. Live diagnosis has a bounded, validated workflow with an explicit unavailable response rather than disguised recorded output. Runtime diagnostics are local; no cluster connection or command execution is implemented.

See the [application guide](../projects/01-rag-k8s-docs/README.md), [walkthrough](media/troubleshooting-demo.webm), and [current status](STATUS.md).

## Verified artifacts

- Shared CPU workflow: 223 tests pass (38 tooling, 131 RAG, 54 fine-tuning); lint and notebook source parity pass.
- Refreshed index: 12,084 chunks; 34 oversized chunks explicitly truncate. Chunker source and legacy vector reuse provenance are recorded.
- Real-index hybrid offline `/ask`: 98 ms for the recorded verification request, with citations; this is not online diagnosis quality.
- Real-index readiness, authored demos, wheel resources, browser desktop/mobile/upload behavior, and deployment dry-run pass.
- Incident integration: 30 validated fake-provider responses across 10 dev cases and 3 arms; zero human-reviewed scores and no diagnosis-quality claim.

## Local checks

```bash
uv sync --frozen
uv run --frozen python scripts/dev.py check
uv run --frozen python scripts/build_colab.py --check
```

The independent CPU checks use fake providers and transport-level GitHub mocks. They cover UTF-8 byte limits, input schema, evidence references, read-only suggestions, version mismatches, missing evidence, demo/live separation, GitHub host/repository restrictions, two-request budget, caching, rate-limit backoff, malformed/time-out responses, split integrity, held-out certification, and hash-bound review aggregation. The fine-tuning domain benchmark has fake-generation metric tests and remains outside serving.

## Reproduce the real-browser walkthrough

Start the app according to its README, then from the repository root:

```bash
uv run --no-project --with playwright python -m playwright install chromium
# On a Linux host missing browser libraries, install them using your system's authorized package workflow.
uv run --no-project --with playwright python scripts/record_troubleshooting.py \
  --base-url http://127.0.0.1:8000 --output-dir docs/media
```

The recording uses actual local recorded-demo endpoints, never mocked live-generation footage. Separate real-browser checks verified a 390px mobile layout, no horizontal overflow, invalid UTF-8 and oversized file rejection, successful text upload, fake-provider response rendering, unavailable-provider recovery, and zero JavaScript page errors. Screenshot inspection was performed. Browser/FFmpeg dependencies are optional and excluded from root CPU dependencies.

## Remaining release checks

1. The fresh index and dev retrieval comparison are complete; see the project README for artifacts. Rebuild when changing ingestion/chunking/encoder configuration; preserve old indexes/results. Inspect source hash, chunk count, encoder limit and truncation count. If using exact-text reuse, preserve provenance and acknowledge unpinned legacy model revisions.
2. Run fresh dense/hybrid/reranked development retrieval comparisons and the incident comparison arms. Keep semantic scores unset until manual review. The incident harness can aggregate reviews of saved predictions without repeating model calls.
3. Configure provider credentials in the server environment, then run the release checker with `--live` against all three incidents. Review the resulting diagnosis and citations manually; a validated schema is not a correctness measurement.
4. Build and run the Docker image, check cold startup, health, models/index readiness, `/ask`, `/diagnose`, `/demos`, missing-key behavior, and UI.
5. Inspect deployment staging using `scripts/deploy_space.sh <owner>/<space> --dry-run`. Configure the intended Space and server-side secrets before deployment. The deployment script force-pushes that named repository.
6. Publish the actual hosted URL only after the hosted health and workflow checks pass. No hosted URL or real-provider quality claim is currently available.

The Docker CLI is installed, but the current user receives permission denied on the engine socket. Provider API keys and a Hugging Face write token remain unavailable; full image, live-provider and deployment checks have not been performed. GPU adapter benchmarking also remains pending GPU/model availability.

## Representative limitations

- Status labels alone do not establish root cause; manifests may describe desired state rather than observed state.
- Logs/events can refer to different workloads or time ranges. The result must separate conflicts and missing evidence from hypotheses.
- GitHub matches are contextual reports, not proof, and network failures can remove this optional evidence.
- User-supplied Kubernetes versions can differ from the pinned documentation version; mismatches are surfaced.
- Oversized code blocks can exceed the embedding budget even when normal prose fits.
- Prompt/schema/reference validation cannot eliminate unsupported semantic claims; human-reviewed cases and honest per-case reports remain necessary.
