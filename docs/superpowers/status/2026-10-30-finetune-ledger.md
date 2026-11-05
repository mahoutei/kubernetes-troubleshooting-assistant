# Function-calling fine-tune ledger

> **Historical record:** this document preserves earlier plans, decisions, or observations. See [current status and next milestones](../../STATUS.md) for the maintained project status.


- Plan: `docs/superpowers/plans/2026-10-29-finetune-function-calling.md`
- Spec: `docs/superpowers/specs/2026-10-29-llm-portfolio-design.md`
- Branch: `master`
- Started: 2026-10-30

## Status

| Task | State | Evidence |
|---|---|---|
| 1. Scaffold + data ingestion | complete (`62ab3b9`) | 17 tests in `tests/test_data.py` |
| 2. Prompts + dataset preparation | complete (`62ab3b9`) | 4 tests; real manifest produced |
| 3. QLoRA training script/notebook | code complete; Colab smoke test pending | 24 tests; `training/train.py`, `config.yaml`, `training/colab.ipynb` |
| 4-6. Inference/metrics/API baseline | not started | next block |
| 7-9. Forgetting/comparison/model card | not started | gated on GPU/API |
| 10. Root README | not started | blocked on Project 1 certified results |

Test command in this session (project `uv sync` is heavy and the sandbox uv cache is read-only):

```text
<python> -m pytest -q  -> 24 passed
<python> -m ruff check . -> clean
```

In this session: `uv run --no-project --with pytest --with pyyaml python -m pytest -q` → 24 passed; `uv run --no-project --with ruff ruff check .` → clean.

## Real-data preparation (`scripts/prepare_data.py --max-seq-length 2048`)

Pinned revision `dae3e1d28cfbcf4b915c04ea1e072030529b4bda`; checksums and row counts verified at download.

| field | value |
|---|---:|
| raw rows | 7,102 |
| parsed tool-call examples | 5,100 |
| skipped or duplicate | 2,002 |
| split (train/val/test) | 4,081 / 509 / 510 |
| kept after length filter | 4,080 / 509 / 510 |

The manifest at `data/processed/manifest.json` is gitignored; it records the revision, per-file sha256/bytes/rows, split counts, length-filtered counts, seed, ratios, and `max_seq_length`.

## Decisions made during the first real-data run

1. **First actual tool-call turn, not first gpt turn.** Multi-turn Glaive rows often open with a clarification question. The parser now scans for the first assistant message containing a valid tool call and pairs it with the preceding human turn. This recovered ~1,058 rows (3,286 → 4,344 Glaive parses).
2. **Near-dedupe includes the tool schema.** The previous key (normalized prompt + calls) collapsed valid examples whose available tools differed; it cut 3,267 Glaive parses to 1,152. Keying on `schema_hash` too keeps them (final 5,100 parsed before splits).
3. **The spec's ~5.7k train estimate is superseded.** Raw 7,102 includes clarification/no-call rows; the manifest is authoritative. Spec/plan updated to state ~5.4k usable turns and manifest-sourced split counts.

## Residual limitations

- Near-dedupe is lexical: paraphrases and semantic duplicates still survive; the split is schema-disjoint, not contamination-proof.
- Length filtering drops rows over 2,048 tokens (only 1 row here).
- The test/val split is small (~509/510); per-slice variance will be wide.
- `training/colab.ipynb` keeps `<your-repo-url>` until the repo has a public remote; fill it before the smoke test.

## Task 3 decisions

1. `resume_from_checkpoint("true")` maps to `True`; passing the string through would make `Trainer` look for a directory named `true`.
2. `requirements-train.txt` adds `httpx` and `pyyaml`: the Colab flow runs `prepare_data.py` (httpx) and `train.py` (yaml) after installing only that file.

## Environment and gates

- Tests run on CPU with the existing Python 3.12 venv; no network, model weights, GPU, or API keys.
- Training (Task 3) needs Colab/Kaggle T4; API baseline (Task 6) needs `GROQ_API_KEY`; forgetting (Task 7) needs the HF Hub MMLU download.
- Raw data and processed splits are gitignored; only the pin, scripts, tests, and this ledger are committed.


## Debugging update — 2026-11-04

Conversation-preserving preprocessing v2 supersedes the earlier counts above. The real-data rerun produced 5,123 parsed/deduplicated examples, split 4,098 / 512 / 513; length filtering retained 4,097 / 512 / 513. Of those, 1,051 preserve prior conversation history. Manifest/model/length compatibility is checked before training; regenerate old processed data. CPU verification is now 32 passing tests with lint clean. GPU smoke execution remains pending. See `docs/debugging/2026-11-04-debugging-report.md`.
