"""CPU-testable artifact, checkpoint, and promotion rules for Colab."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(config: dict, processed: Path, source_manifest: Path) -> str:
    payload = {"config": config, "sources": json.loads(source_manifest.read_text()),
               "data": {p.name: digest(p) for p in sorted(processed.glob("*.jsonl"))},
               "manifest": digest(processed / "manifest.json")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require_adapter(path: Path) -> None:
    if not (path / "adapter_config.json").is_file() or not any(path.glob("adapter_model.*")):
        raise ValueError(f"adapter is incomplete: {path}")
    if not (path / "tokenizer_config.json").is_file():
        raise ValueError(f"tokenizer is missing: {path}")


def smoke_gate(report: dict, evaluation: dict, adapter: Path) -> None:
    require_adapter(adapter)
    if report.get("global_step") != 2:
        raise ValueError("smoke run must complete exactly two optimizer steps")
    for key in ("training_loss", "eval_loss"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"smoke {key} must be finite")
    if evaluation.get("adapter_reloaded") is not True:
        raise ValueError("smoke evaluation must reload the saved adapter")
    rows = evaluation.get("rows", [])
    if not rows or any(not row.get("prediction", "").strip() for row in rows):
        raise ValueError("adapter reload/inference did not produce nonempty responses")
    # Syntax/accuracy metrics are diagnostic after two steps, not promotion thresholds.


def checkpoint_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.name != "COMPLETE.json")


def verify_checkpoint(path: Path, identity: str) -> None:
    marker = json.loads((path / "COMPLETE.json").read_text())
    if marker.get("mode") != "full" or marker.get("fingerprint") != identity:
        raise ValueError("checkpoint is not from a compatible full-training run")
    for name, checksum in marker["files"].items():
        file = path / name
        if not file.resolve().is_relative_to(path.resolve()) or not file.is_file() or digest(file) != checksum:
            raise ValueError("checkpoint checksum mismatch or incomplete backup")
    for required in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json"):
        if required not in marker["files"]:
            raise ValueError(f"checkpoint missing {required}")
    if not any(name.startswith("adapter_model.") for name in marker["files"]):
        raise ValueError("checkpoint missing adapter weights")
    if not any(name.startswith("rng_state") for name in marker["files"]):
        raise ValueError("checkpoint missing RNG state")


def complete_checkpoint(path: Path, identity: str, mode: str, backup: Path | None = None) -> None:
    if mode != "full":
        return
    marker = {"mode": mode, "fingerprint": identity,
              "files": {p.relative_to(path).as_posix(): digest(p) for p in checkpoint_files(path)}}
    (path / "COMPLETE.json").write_text(json.dumps(marker, indent=2))
    verify_checkpoint(path, identity)
    if backup is not None:
        backup.mkdir(parents=True, exist_ok=True)
        destination = backup / path.name
        if destination.exists():
            verify_checkpoint(destination, identity)
            return
        # Publish the final name only after every file has been copied and verified.
        with tempfile.TemporaryDirectory(prefix=".partial-", dir=backup) as temporary:
            staged = Path(temporary) / path.name
            shutil.copytree(path, staged)
            verify_checkpoint(staged, identity)
            staged.rename(destination)


def select_checkpoint(roots: list[Path], identity: str) -> Path:
    compatible = []
    for root in roots:
        for path in root.glob("checkpoint-*"):
            try:
                verify_checkpoint(path, identity)
                step = int(path.name.removeprefix("checkpoint-"))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            compatible.append((step, path))
    if not compatible:
        raise ValueError("no completed compatible full-training checkpoint found; disable resume for a new run")
    return max(compatible, key=lambda item: item[0])[1]


def export_run(run: Path, processed: Path, destination: Path) -> None:
    allowed = {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
               "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
               "vocab.json", "merges.txt", "tokenizer.model", "README.md", "train_log.csv",
               "training_report.json", "evaluation.json", "predictions.jsonl", "run_config.json",
               "run_metadata.json", "environment.txt", "source_manifest.json", "gpu_info.json", "example.json", "uv.lock", "pyproject.toml"}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(run.rglob("*")):
            if file.is_file() and file.name in allowed and not any(part.startswith("checkpoint-") for part in file.relative_to(run).parts):
                archive.write(file, file.relative_to(run).as_posix())
        for file in [processed / "manifest.json", *sorted(processed.glob("*.jsonl"))]:
            archive.write(file, "data/" + file.name)
