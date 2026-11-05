"""Generate the standalone web-Colab notebook from canonical source files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("projects/02-finetune-function-calling")
NOTEBOOK = PROJECT / "training/colab.ipynb"
SOURCES = {
    **{name: PROJECT / name for name in (
        "src/__init__.py", "src/data.py", "src/prompts.py", "src/metrics.py",
        "scripts/prepare_data.py", "training/__init__.py", "training/train.py",
        "training/evaluate.py", "training/colab_support.py", "training/colab_runner.py", "config.yaml",
    )},
    "gpu/pyproject.toml": Path("environments/gpu/pyproject.toml"),
    "gpu/uv.lock": Path("environments/gpu/uv.lock"),
    "gpu/.python-version": Path("environments/gpu/.python-version"),
    "gpu_check.py": Path("scripts/gpu_check.py"),
}


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(root=ROOT):
    cells = [cell("markdown", """# Standalone function-calling QLoRA — web Colab
Upload just this notebook to https://colab.research.google.com/ and select **Runtime → Change runtime type → GPU**. No repository clone, folder upload, or API key is needed. Public packages, data, tokenizer, and model weights download over the internet.

**Verification:** locally validated; real Colab GPU execution and Drive recovery are pending. TPU/CPU training is not supported by this CUDA backend.

**Run all** first performs two optimizer steps (32 train / 16 validation examples), checks finite losses, adapter reload and inference, then automatically starts a fresh full epoch when operational checks pass. Early accuracy is diagnostic, not proof of quality. Set AUTO_FULL=False to stop after smoke. Existing artifacts are never silently overwritten; change RUN_NAME for a fresh experiment.

The embedded GPU environment uses Python 3.12 independently of the Colab kernel. Code runs in subprocesses, releasing GPU memory between training and evaluation stages. Optional Drive backup requires explicit authorization only when enabled.
"""), cell("code", '''import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/content/function_calling_colab")
RUN_NAME = "run-001"
AUTO_FULL = True
USE_DRIVE = False
RESUME_FULL = False
DRIVE_ROOT = "/content/drive/MyDrive/function_calling_colab"
TRAINING = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct", "seed": 42,
    "max_seq_length": 2048, "epochs": 1, "learning_rate": 0.0002,
    "batch_size": 2, "grad_accum": 8, "lora_r": 16, "lora_alpha": 32,
}
MAX_NEW_TOKENS = 256
INTERACTIVE_EXAMPLE = {
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "user": "What is the weather in Singapore?",
}
assert re.fullmatch(r"[A-Za-z0-9_-]+", RUN_NAME), "Use a simple unique run name"
assert MAX_NEW_TOKENS > 0
if not shutil.which("nvidia-smi"):
    raise RuntimeError("Select an NVIDIA GPU in Runtime → Change runtime type. TPU/CPU is unsupported by this QLoRA backend.")
hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"], check=True, capture_output=True, text=True).stdout
print(hardware)
WORK.mkdir(parents=True, exist_ok=True)
FILES = {}
''')]
    hashes = {}
    for target, source in SOURCES.items():
        text = (root / source).read_text()
        hashes[target] = hashlib.sha256(text.encode()).hexdigest()
        literal = "r\'\'\'" + text + "\'\'\'" if "\'\'\'" not in text else repr(text)
        cells += [cell("markdown", f"### Embedded source: `{target}`\nSHA-256: `{hashes[target]}`"),
                  cell("code", f"FILES[{target!r}] = {literal}\n")]
    cells += [cell("markdown", "## Materialize and install the isolated GPU environment\nThis can take several minutes and downloads large CUDA packages. No ML packages are installed into the notebook kernel."),
              cell("code", '''import hashlib

for name, source in FILES.items():
    path = WORK / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
source_manifest = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in FILES.items()}
(WORK / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True))
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
run_dir = WORK / "runs" / RUN_NAME
backup_dir = str(Path(DRIVE_ROOT) / RUN_NAME) if USE_DRIVE else None
settings = {"training": TRAINING, "run_dir": str(run_dir), "backup_dir": backup_dir,
            "auto_full": AUTO_FULL, "resume": RESUME_FULL, "max_new_tokens": MAX_NEW_TOKENS}
(WORK / "notebook_settings.json").write_text(json.dumps(settings, indent=2))
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "gpu_info.json").write_text(json.dumps({"nvidia_smi": hardware}, indent=2))
(WORK / "interactive_example.json").write_text(json.dumps(INTERACTIVE_EXAMPLE, indent=2))
subprocess.run([sys.executable, "-m", "pip", "install", "uv==0.13.6"], check=True)
uv = [sys.executable, "-m", "uv"]
subprocess.run(uv + ["sync", "--project", str(WORK / "gpu"), "--frozen"], cwd=WORK, check=True)
def gpu_run(*args):
    subprocess.run(uv + ["run", "--project", str(WORK / "gpu"), "--frozen", "python", *map(str, args)], cwd=WORK, check=True)
gpu_run("gpu_check.py")
'''), cell("markdown", "## Embedded lightweight checks\nVerify parsing and metrics before preparing data or loading model weights."),
              cell("code", '''checks = """from src.data import extract_tagged_json
from src.metrics import score_output
call = {"name": "demo", "arguments": {"x": 1}}
import json
text = "<tool_call>" + json.dumps(call) + "</tool_call>"
assert extract_tagged_json(text, "tool_call") == [call]
assert score_output(text, [call])["whole_example_exact_match"] == 1
assert score_output("plain prose", [call])["valid_output"] == 0
print("Embedded parsing/metric checks passed")
"""
gpu_run("-c", checks)
'''), cell("markdown", "## Prepare the verified real dataset\nChecksums and row counts are verified on every run; splits are rebuilt for the selected model, length, and seed. The held-out test split is never evaluated."),
              cell("code", '''gpu_run("training/colab_runner.py", "prepare")'''),
              cell("markdown", "## Smoke training and adapter-reload evaluation\nOperational failures stop execution. Accuracy and tool-call validity after two optimizer steps are displayed for inspection, not used as quality thresholds. On resume in the same session, reuse only an existing smoke result; the full-stage gate rechecks its identity."),
              cell("code", '''if RESUME_FULL and (run_dir / "smoke/evaluation.json").exists():
    print("Reusing smoke artifacts; full stage will verify compatibility")
else:
    gpu_run("training/colab_runner.py", "smoke")
smoke = json.loads((run_dir / "smoke/evaluation.json").read_text())
print(json.dumps(smoke["metrics"], indent=2))
for row in smoke["rows"][:3]:
    print(json.dumps({"prediction": row["prediction"], "gold": row["gold"]}, indent=2))
'''), cell("markdown", "## Automatic full training and validation\nAfter the smoke gate passes, full training starts from the base model in a new process. RESUME_FULL=True instead restores a completed compatible full-training checkpoint. No smoke weights or optimizer state enter a fresh full run. Drive backups are published only after checksum verification."),
              cell("code", '''gpu_run("training/colab_runner.py", "full")
result_dir = run_dir / ("full" if AUTO_FULL else "smoke")
result = json.loads((result_dir / "evaluation.json").read_text())
print(json.dumps(result["metrics"], indent=2))
print("Interactive example:", (result_dir / "example.json").read_text())
'''), cell("markdown", "## Export and optional Drive backup\nArchive includes adapters/tokenizers, configuration, manifests, processed public-data splits, source hashes, dependency versions, losses, predictions, and metrics. Base weights, caches, optimizer checkpoints, and credentials are excluded. Drive checkpoint backups remain separate for resume. If training fails, completed checkpoints already backed up to Drive survive a disconnected runtime."),
              cell("code", '''gpu_run("training/colab_runner.py", "export")
archive = run_dir.parent / (RUN_NAME + ".zip")
from google.colab import files

files.download(str(archive))
'''), cell("markdown", "## Reproduce or recover\nFor a new experiment change RUN_NAME. To recover from a disconnect, enable Drive backup again, reuse the same RUN_NAME and training settings, and set RESUME_FULL=True. Preparation and smoke can rerun, then compatible completed full checkpoints are selected by step. Without Drive, `/content` artifacts can disappear when Colab resets.\n\nTo change the editable example after training, rewrite `interactive_example.json` with the configuration above and rerun the evaluation command below; this does not train again."),
              cell("code", '''# Optional: rerun this cell after changing INTERACTIVE_EXAMPLE.
(WORK / "interactive_example.json").write_text(json.dumps(INTERACTIVE_EXAMPLE, indent=2))
# Reuses the trained adapter, evaluates one validation example, and updates example.json.
# Uncomment to execute after training:
# gpu_run("training/evaluate.py", "--adapter", result_dir, "--limit", "1", "--example", "interactive_example.json")
''')]
    for index, entry in enumerate(cells):
        entry["id"] = f"cell-{index:03d}"
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "embedded_sources": hashes}, "nbformat": 4, "nbformat_minor": 5}


def render(root=ROOT):
    return json.dumps(build_notebook(root), indent=1, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    path = ROOT / NOTEBOOK
    if args.check:
        if not path.exists() or path.read_text() != text:
            raise SystemExit("Notebook is stale: run uv run --frozen python scripts/build_colab.py")
        print("Standalone Colab notebook matches canonical sources")
    else:
        path.write_text(text)
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
