import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.gpu_check import check_cuda

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_environment_is_separate_and_uses_explicit_cuda_sources():
    gpu = tomllib.loads((ROOT / "environments/gpu/pyproject.toml").read_text())
    cpu = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert gpu["tool"]["uv"]["package"] is False
    assert "workspace" not in cpu["tool"]["uv"]
    index = gpu["tool"]["uv"]["index"][0]
    assert index["explicit"] is True
    assert index["url"] == "https://download.pytorch.org/whl/cu128"
    for package in ("torch", "torchvision"):
        assert gpu["tool"]["uv"]["sources"][package]["index"] == index["name"]


def fake_torch(cuda_version="12.8", available=True, result=4):
    class Tensor:
        def __add__(self, value):
            assert value == 1
            return self

        def sum(self):
            return self

        def item(self):
            return result

    def ones(size, *, device):
        assert size == 2 and device == "cuda"
        return Tensor()

    return SimpleNamespace(
        __version__="test+cu128",
        version=SimpleNamespace(cuda=cuda_version),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda index: "Test GPU",
            synchronize=lambda: None,
        ),
        ones=ones,
    )


def test_gpu_check_executes_kernel_and_reports_device():
    assert check_cuda(fake_torch())["GPU"] == "Test GPU"


@pytest.mark.parametrize("cuda_version,available,match", [(None, True, "CPU-only"), ("12.8", False, "CUDA is unavailable")])
def test_gpu_check_refuses_cpu_fallback(cuda_version, available, match):
    with pytest.raises(RuntimeError, match=match):
        check_cuda(fake_torch(cuda_version, available))


def test_gpu_check_rejects_failed_kernel_result():
    with pytest.raises(RuntimeError, match="unexpected result"):
        check_cuda(fake_torch(result=0))


def test_gpu_lock_contains_cuda_builds_without_leaking_into_cpu_lock():
    gpu = tomllib.loads((ROOT / "environments/gpu/uv.lock").read_text())
    cpu = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = {p["name"]: p for p in gpu["package"]}
    for name in ("torch", "torchvision"):
        assert packages[name]["version"].endswith("+cu128")
        assert packages[name]["source"]["registry"] == "https://download.pytorch.org/whl/cu128"
    assert "unsloth" in packages
    assert {p["name"] for p in cpu["package"]}.isdisjoint({"torch", "torchvision", "unsloth", "transformers"})
