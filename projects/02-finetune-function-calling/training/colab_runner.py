"""Stage runner embedded in the standalone notebook; GPU imports stay in subprocesses."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.colab_support import (
    digest,
    export_run,
    fingerprint,
    select_checkpoint,
    smoke_gate,
)


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def training_command(config_path: Path, mode: str, metadata: Path, backup: Path | None, resume: Path | None) -> list[str]:
    command = ["training/train.py", "--config", str(config_path), "--run-metadata", str(metadata)]
    if mode == "smoke":
        command += ["--max-steps", "2", "--max-train-examples", "32", "--max-eval-examples", "16"]
    elif mode != "full":
        raise ValueError("unknown training mode")
    if resume is not None:
        if mode != "full":
            raise ValueError("smoke runs cannot resume")
        command += ["--resume-from-checkpoint", str(resume)]
    if backup is not None and mode == "full":
        command += ["--backup-dir", str(backup)]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "smoke", "full", "export"))
    parser.add_argument("--settings", default="notebook_settings.json")
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text())
    config = settings["training"]
    run_dir = Path(settings["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    processed = Path("data/processed")
    sources = Path("source_manifest.json")
    backup = Path(settings["backup_dir"]) if settings.get("backup_dir") else None
    if args.stage == "prepare":
        run("scripts/prepare_data.py", "--model", config["model"], "--max-seq-length", config["max_seq_length"], "--seed", config["seed"])
        print((processed / "manifest.json").read_text())
        print("Example:", (processed / "train.jsonl").read_text().splitlines()[0])
        return
    identity = fingerprint(config, processed, sources)
    if args.stage == "export":
        for name in ("source_manifest.json",):
            shutil.copy2(name, run_dir / name)
        for name in ("uv.lock", "pyproject.toml"):
            shutil.copy2(Path("gpu") / name, run_dir / name)
        archive = run_dir.parent / (run_dir.name + ".zip")
        export_run(run_dir, processed, archive)
        if backup:
            backup.mkdir(parents=True, exist_ok=True)
            temporary = backup / (archive.name + ".partial")
            shutil.copy2(archive, temporary)
            if digest(archive) != digest(temporary):
                raise RuntimeError("Drive export checksum mismatch")
            temporary.replace(backup / archive.name)
        print(f"Download archive: {archive}")
        return
    mode = args.stage
    if mode == "full":
        evaluation = json.loads((run_dir / "smoke/evaluation.json").read_text())
        report = json.loads((run_dir / "smoke/training_report.json").read_text())
        smoke_metadata = json.loads((run_dir / "smoke/run_metadata.json").read_text())
        if smoke_metadata["fingerprint"] != identity:
            raise ValueError("settings or data changed since smoke; rerun smoke with a new run name")
        smoke_gate(report, evaluation, run_dir / "smoke")
        if not settings["auto_full"]:
            print("Smoke passed; full training disabled by configuration")
            return
    target = run_dir / mode
    resume = None
    if mode == "full" and settings["resume"]:
        checkpoint = select_checkpoint([target, backup] if backup else [target], identity)
        local = target / checkpoint.name
        if checkpoint.resolve() != local.resolve():
            if local.exists():
                raise ValueError("local resume checkpoint already exists; use a fresh run directory")
            shutil.copytree(checkpoint, local)
        resume = local
    elif target.exists() and any(target.iterdir()):
        raise ValueError(f"{target} already contains artifacts; select a new run name (or resume full training)")
    target.mkdir(parents=True, exist_ok=True)
    current = {**config, "output_dir": str(target)}
    config_path = target / "run_config.json"
    config_path.write_text(json.dumps(current, indent=2))
    metadata = target / "run_metadata.json"
    metadata.write_text(json.dumps({"fingerprint": identity, "mode": mode, "config": config}, indent=2))
    versions = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (run_dir / "environment.txt").write_text("\n".join(versions) + "\n")
    run(*training_command(config_path, mode, metadata, backup, resume))
    evaluation_command = ["training/evaluate.py", "--adapter", target, "--max-new-tokens", settings["max_new_tokens"], "--example", "interactive_example.json"]
    if mode == "smoke":
        evaluation_command += ["--limit", "16"]
    run(*evaluation_command)
    if mode == "smoke":
        smoke_gate(json.loads((target / "training_report.json").read_text()),
                   json.loads((target / "evaluation.json").read_text()), target)
        print("SMOKE PASSED: pipeline readiness only; accuracy is not a promotion threshold")


if __name__ == "__main__":
    main()
