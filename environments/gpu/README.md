# NVIDIA GPU training environment

An independent Python 3.12 uv project for the function-calling QLoRA pipeline. Its environment is `environments/gpu/.venv` and its dependencies are locked in the adjacent `uv.lock`. The root `.venv` remains the lightweight CPU check environment; neither project is a uv workspace member.

## Requirements and verification limits

Use Linux x86-64 or WSL2 with a visible NVIDIA GPU and a driver compatible with CUDA 12.8. This environment does not install a host driver or configure GPU passthrough. macOS, AMD GPUs, and native Windows are outside this environment's supported target.

Torch and torchvision are explicitly sourced from the official CUDA 12.8 index, following [uv's PyTorch configuration guide](https://docs.astral.sh/uv/guides/integration/pytorch/). TRL is constrained to the [0.24 API generation](https://huggingface.co/docs/trl/v0.24.0/sft_trainer) and Transformers to 4.x for the current training script. Unsloth, PEFT, datasets, accelerate, bitsandbytes, and preparation dependencies are included.

The lockfile resolves and frozen installation passes a dry run. GPU packages have not been installed or executed on the development machine because it has no visible NVIDIA driver utility. Dependency resolution does not prove CUDA kernels, Unsloth integration, or model training work; the readiness check and two-step smoke run below are still required on the target GPU.

## Install and verify

From the repository root, with uv 0.13.6:

```bash
# Optional: inspect the installation without creating the GPU environment.
uv sync --project environments/gpu --frozen --dry-run

# Downloads large CUDA/runtime packages into the separate GPU environment.
uv sync --project environments/gpu --frozen
uv run --project environments/gpu --frozen python scripts/gpu_check.py
```

The check rejects CPU-only torch and missing CUDA, executes a CUDA tensor calculation, synchronizes the device, and prints the actual torch/CUDA/GPU information. It does not download model weights. If it fails, fix the driver, GPU access, or environment before training; there is no CPU fallback.

## Prepare and train

After the GPU check passes, change into the fine-tuning directory. `--project` selects the GPU environment while the current directory stays with the training data and configuration:

```bash
cd projects/02-finetune-function-calling
uv run --project ../../environments/gpu --frozen python scripts/prepare_data.py --max-seq-length 2048
uv run --project ../../environments/gpu --frozen python training/train.py --config config.yaml --max-steps 2
uv run --project ../../environments/gpu --frozen python training/train.py --config config.yaml
# Resume only when an existing checkpoint is available:
uv run --project ../../environments/gpu --frozen python training/train.py --config config.yaml --resume-from-checkpoint true
```

Preparation downloads the pinned public dataset and tokenizer if missing. Training downloads model weights. GPU VRAM and disk requirements depend on the configured model, sequence length, and batch size; kernel readiness alone does not establish training capacity.

The standalone Colab notebook embeds this locked environment and uses isolated Python 3.12 subprocesses, without changing the notebook kernel. `requirements-train.txt` remains an alternative unpinned installation path for custom kernels. See [fine-tuning instructions](../../projects/02-finetune-function-calling/README.md#training) and [current status](../../docs/STATUS.md#function-calling-fine-tuning).

## CPU checks and maintenance

Use the root workflow for tests and lint. CPU CI checks the environment declaration and GPU guard behavior with fakes; it never installs the CUDA environment. GPU readiness requires real hardware and is run explicitly.

To deliberately refresh the GPU dependencies, use `uv lock --project environments/gpu --upgrade`, review the diff, sync, and repeat kernel and training smoke checks before describing the new lock as GPU-verified. Routine installation uses `--frozen`.
