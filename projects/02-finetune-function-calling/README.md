# Function-calling QLoRA

A training pipeline for Qwen2.5-1.5B-Instruct on a pinned subset of the Hermes function-calling dataset. The pipeline is implemented; GPU execution and model-quality evaluation remain pending. See [current status and next milestones](../../docs/STATUS.md#function-calling-fine-tuning).

## Standalone web Colab

Download [the self-contained notebook](training/colab.ipynb), open [web Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and select an NVIDIA GPU under **Runtime → Change runtime type**. Upload only the notebook; there is no repository clone, folder upload, or required API key. Public dependencies, data, tokenizer, and model weights still download over the internet.

The configuration cell controls the model, seed, sequence length, batch size, run name, output location, optional Drive backup, and resume. Run all installs an isolated Python 3.12 CUDA environment, verifies the real dataset, trains a two-step smoke adapter on at most 32 training / 16 validation examples, reloads it, and checks inference. Finite losses, complete artifacts, and nonempty inference responses trigger **automatic full training from the original base model**; early accuracy is diagnostic and is not a quality threshold. Set `AUTO_FULL=False` for smoke only.

Full training evaluates the validation split and exports adapters/tokenizers, processed public-data splits, configuration, provenance, losses, predictions, and metrics. Test-split evaluation, API comparisons, and forgetting benchmarks are outside this notebook. The final archive excludes base weights, caches, credentials, and optimizer checkpoints. Optional Drive backup preserves completed full-training checkpoints for resume; local `/content` storage can disappear on reset. Use a new run name for a fresh experiment.

**Verification:** notebook generation, cell compilation, embedded source parity, and CPU tests pass locally. Actual web-Colab installation, GPU training, and Drive recovery remain pending target-hardware validation. CPU and TPU runtimes stop with instructions to select GPU.

The notebook is generated from canonical project code and the locked GPU environment. After changing an embedded source, run from the repository root:

```bash
uv run --frozen python scripts/build_colab.py
uv run --frozen python scripts/build_colab.py --check
```

Shared CPU checks reject stale embedded source. Edit source modules and regenerate rather than editing generated source cells directly. Configuration cells remain editable in your personal Colab copy.

## Data preparation

The two source files are pinned to revision `dae3e1d28cfbcf4b915c04ea1e072030529b4bda` and verified by SHA-256 and row count. Tool-schema groups stay within a single split. Deduplication includes prior conversation history, so clarification answers retain their original requests.

After the 2026-11-04 preprocessing fix:

| Stage | Count |
|---|---:|
| Raw rows | 7,102 |
| Parsed and deduplicated examples | 5,123 |
| Train / validation / test before length filtering | 4,098 / 512 / 513 |
| Train / validation / test after 2,048-token filtering | 4,097 / 512 / 513 |
| Retained examples with prior conversation history | 1,051 |

The generated `data/processed/manifest.json` records preprocessing version, tokenizer model, sequence length, source checksums, and counts. Training refuses older manifests or incompatible preprocessing settings. Old processed splits must be regenerated; the previous ledger's counts describe the old parser.

## Training

The recommended Linux/WSL CLI path uses the [locked GPU environment](../../environments/gpu/README.md). From the repository root:

```bash
uv sync --project environments/gpu --frozen
uv run --project environments/gpu --frozen python scripts/gpu_check.py
cd projects/02-finetune-function-calling
uv run --project ../../environments/gpu --frozen python scripts/prepare_data.py --max-seq-length 2048
uv run --project ../../environments/gpu --frozen python training/train.py --config config.yaml --max-steps 2
uv run --project ../../environments/gpu --frozen python training/train.py --config config.yaml
# After a disconnect, if a checkpoint exists:
uv run --project ../../environments/gpu --frozen python training/train.py --config config.yaml --resume-from-checkpoint true
```

The GPU environment is separate from contributor checks and requires an accessible NVIDIA CUDA device. Its lock resolves, but CUDA execution and training integration still require verification on the target hardware. The existing notebook's `pip install -r requirements-train.txt` remains an alternative for installing directly into a Colab kernel; it is not locked by the uv GPU environment.

The two-step smoke run schedules validation at step 2 and saves the final adapter. The full run evaluates every 200 optimizer steps. The standalone notebook under `training/colab.ipynb` embeds its project files and dependencies; it does not need a repository URL. Local CPU tests do not validate CUDA, quantization, trainer integration, or training quality.

## Contributor checks (CPU only)

From the repository root, use the shared environment. It does not install the CUDA or training libraries:

```bash
uv sync --frozen
uv run --frozen python scripts/dev.py check --project finetune
```

The [current-status page](../../docs/STATUS.md#function-calling-fine-tuning) tracks the GPU smoke test, training, evaluation, and publication milestones.

## Limitations

Schema grouping reduces leakage but does not eliminate semantic contamination. The roughly 500-example held-out splits still require per-slice uncertainty reporting. No tuned-model benchmark is available yet. CPU tests do not validate trainer integration or model quality.

## Domain tool-selection benchmark

The troubleshooting beta uses a hosted generator. Adapter routing remains experimental until a domain comparison demonstrates a useful tradeoff. Generic Hermes validation accuracy is not evidence of Kubernetes tool-selection performance.

A small benchmark of documentation search, supplied-evidence inspection, and explicitly requested GitHub lookup lives at `../01-rag-k8s-docs/evals/tool_selection.jsonl`. It is benchmark-only, not training data, and excludes held-out incident labels. Results retain raw predictions, strict tool/argument metrics, latency, and model identity. No base/adapter/hosted comparison has been run yet.

From this directory after training an adapter, with the GPU environment installed and `GROQ_API_KEY` set for the hosted arm:

```bash
uv run --project ../../environments/gpu --frozen python training/tool_benchmark.py \
  --mode all --adapter outputs/qwen2.5-1.5b-lora --output results/kubernetes-tools.json
```

Use `--mode base`, `tuned`, or `hosted` to run individual arms. Missing hosted credentials are recorded as not run. Existing result files are never overwritten. A successful syntax result alone does not qualify the adapter for production use.
