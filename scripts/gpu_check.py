"""Verify the GPU environment can execute an NVIDIA CUDA kernel."""
from __future__ import annotations

import sys


def check_cuda(torch) -> dict[str, str]:
    if not torch.version.cuda:
        raise RuntimeError("CPU-only PyTorch detected; install the locked environments/gpu project")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check NVIDIA drivers and GPU access (including WSL passthrough)")
    # Allocation alone does not establish that kernels support the detected GPU.
    result = (torch.ones(2, device="cuda") + 1).sum().item()
    if result != 4:
        raise RuntimeError("CUDA kernel check produced an unexpected result")
    torch.cuda.synchronize()
    return {
        "PyTorch": str(torch.__version__),
        "CUDA build": str(torch.version.cuda),
        "GPU": torch.cuda.get_device_name(0),
    }


def main() -> int:
    try:
        import torch

        details = check_cuda(torch)
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"GPU check failed: {exc}", file=sys.stderr)
        print("Use: uv sync --project environments/gpu --frozen", file=sys.stderr)
        return 1
    for name, value in details.items():
        print(f"{name}: {value}")
    print("CUDA kernel check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
