import json
from pathlib import Path

import pytest

from src.metrics import score_output, strict_calls
from training.colab_runner import training_command
from training.colab_support import (
    complete_checkpoint,
    export_run,
    fingerprint,
    select_checkpoint,
    smoke_gate,
    verify_checkpoint,
)


def output(calls):
    return "\n".join("<tool_call>" + json.dumps(c) + "</tool_call>" for c in calls)


def call(value=1):
    return {"name": "f", "arguments": {"x": value}}


@pytest.mark.parametrize("text", ["prose", "", '<tool_call>{"name":"f","arguments":{}}</tool_call> extra', '<tool_call>{"name":"f","name":"g","arguments":{}}</tool_call>', '<tool_call>{"name":"f","arguments":{"x":NaN}}</tool_call>', '<tool_call>{"name":"f","arguments":{}}', '<tool_call>{"name":"f","arguments":{}}</tool_call><tool_call>{bad}</tool_call>'])
def test_strict_output_rejects_partial_malformed_and_extra_text(text):
    assert strict_calls(text) is None
    assert score_output(text, [call()])["whole_example_exact_match"] == 0


@pytest.mark.parametrize("different", [True, 1.0, "1", [1]])
def test_metrics_preserve_json_types(different):
    assert score_output(output([call(different)]), [call(1)])["argument_exact_match"] == 0


def test_metrics_compare_multisets_with_duplicate_counts_and_object_order():
    first = {"name": "a", "arguments": {"x": 1, "y": [1, 2]}}
    reordered = {"arguments": {"y": [1, 2], "x": 1}, "name": "a"}
    assert score_output(output([call(), reordered]), [first, call()])["whole_example_exact_match"] == 1
    assert score_output(output([call()]), [call(), call()])["argument_exact_match"] == 0.5
    assert score_output(output([call(), call()]), [call()])["whole_example_exact_match"] == 0
    assert score_output(output([call([2, 1])]), [call([1, 2])])["argument_exact_match"] == 0


def adapter(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        (path / name).write_text("{}")
    return path


def test_smoke_promotion_is_operational_not_accuracy_threshold(tmp_path):
    path = adapter(tmp_path / "smoke")
    evaluation = {"adapter_reloaded": True, "rows": [{"prediction": "malformed but nonempty", "whole_example_exact_match": 0}]}
    report = {"global_step": 2, "training_loss": 2.0, "eval_loss": 3.0}
    smoke_gate(report, evaluation, path)
    for key in ("training_loss", "eval_loss"):
        with pytest.raises(ValueError, match="finite"):
            smoke_gate({**report, key: float("nan")}, evaluation, path)
    with pytest.raises(ValueError, match="nonempty"):
        smoke_gate(report, {"adapter_reloaded": True, "rows": [{"prediction": ""}]}, path)
    with pytest.raises(ValueError, match="reload"):
        smoke_gate(report, {**evaluation, "adapter_reloaded": False}, path)


def test_fresh_full_training_never_uses_smoke_weights_or_limits():
    full = training_command(Path("full/config.json"), "full", Path("metadata.json"), None, None)
    assert "--resume-from-checkpoint" not in full
    assert "--max-train-examples" not in full and "--max-steps" not in full
    smoke = training_command(Path("smoke/config.json"), "smoke", Path("metadata.json"), None, None)
    assert smoke[smoke.index("--max-train-examples") + 1] == "32"
    assert smoke[smoke.index("--max-eval-examples") + 1] == "16"
    with pytest.raises(ValueError):
        training_command(Path("c"), "smoke", Path("m"), None, Path("full/checkpoint-1"))


def checkpoint(path):
    adapter(path)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_text("{}")
    return path


def test_completed_compatible_checkpoints_only_are_selected_and_verified(tmp_path):
    full = checkpoint(tmp_path / "local/checkpoint-2")
    complete_checkpoint(full, "identity", "full", tmp_path / "drive")
    copied = tmp_path / "drive/checkpoint-2"
    verify_checkpoint(copied, "identity")
    checkpoint(tmp_path / "local/checkpoint-999")  # Incomplete: no completion marker.
    wrong = checkpoint(tmp_path / "local/checkpoint-888")
    complete_checkpoint(wrong, "other", "full")
    smoke = checkpoint(tmp_path / "local/checkpoint-777")
    complete_checkpoint(smoke, "identity", "smoke")
    assert select_checkpoint([tmp_path / "local"], "identity") == full
    (copied / "optimizer.pt").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(copied, "identity")
    with pytest.raises(ValueError, match="compatible"):
        select_checkpoint([tmp_path / "drive"], "identity")


def test_fingerprint_changes_with_configuration_data_or_source(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "manifest.json").write_text("{}")
    (processed / "train.jsonl").write_text("{}")
    sources = tmp_path / "sources.json"
    sources.write_text("{}")
    old = fingerprint({"seed": 42}, processed, sources)
    assert fingerprint({"seed": 43}, processed, sources) != old
    (processed / "train.jsonl").write_text('{"changed":true}')
    assert fingerprint({"seed": 42}, processed, sources) != old


def test_export_excludes_secrets_base_weights_caches_and_checkpoints(tmp_path):
    import zipfile

    run = tmp_path / "run"
    full = adapter(run / "full")
    (full / "model.safetensors").write_text("base weights")
    (run / ".env").write_text("secret")
    checkpoint(full / "checkpoint-2")
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    (data / "val.jsonl").write_text("{}")
    archive = tmp_path / "export.zip"
    export_run(run, data, archive)
    with zipfile.ZipFile(archive) as contents:
        assert "full/adapter_model.safetensors" in contents.namelist()
        assert "data/manifest.json" in contents.namelist()
        assert not any("checkpoint" in name or name.endswith(".env") or name == "full/model.safetensors" for name in contents.namelist())


@pytest.mark.parametrize("auto_full", [True, False])
def test_stage_runner_propagates_config_and_promotes_only_after_smoke(tmp_path, monkeypatch, auto_full):
    import sys

    from training import colab_runner

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/processed"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text("{}")
    (data / "train.jsonl").write_text("{}")
    (tmp_path / "source_manifest.json").write_text("{}")
    settings = {"training": {"model": "original-base", "seed": 7, "max_seq_length": 2048},
                "run_dir": str(tmp_path / "run"), "backup_dir": None,
                "auto_full": auto_full, "resume": False, "max_new_tokens": 32}
    (tmp_path / "notebook_settings.json").write_text(json.dumps(settings))
    commands = []

    def fake_run(*args):
        commands.append(args)
        if args[0] == "training/train.py":
            config = json.loads(Path(args[args.index("--config") + 1]).read_text())
            assert config["model"] == "original-base" and config["seed"] == 7
            target = adapter(Path(config["output_dir"]))
            (target / "training_report.json").write_text(json.dumps({"global_step": 2, "training_loss": 2.0, "eval_loss": 3.0}))
        elif args[0] == "training/evaluate.py":
            target = Path(args[args.index("--adapter") + 1])
            (target / "evaluation.json").write_text(json.dumps({"adapter_reloaded": True, "rows": [{"prediction": "nonempty"}]}))

    monkeypatch.setattr(colab_runner, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["runner", "smoke"])
    colab_runner.main()
    monkeypatch.setattr(sys, "argv", ["runner", "full"])
    colab_runner.main()
    train_calls = [args for args in commands if args[0] == "training/train.py"]
    assert len(train_calls) == (2 if auto_full else 1)
    if auto_full:
        assert "--max-steps" not in train_calls[1]
        assert "--resume-from-checkpoint" not in train_calls[1]
        assert (tmp_path / "run/full/run_config.json").exists()
        assert not (tmp_path / "run/full/config.json").exists()


def test_full_stage_cannot_bypass_missing_smoke_artifacts(tmp_path, monkeypatch):
    import sys

    from training import colab_runner

    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/processed"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text("{}")
    (tmp_path / "source_manifest.json").write_text("{}")
    (tmp_path / "notebook_settings.json").write_text(json.dumps({"training": {}, "run_dir": str(tmp_path / "run"), "auto_full": True}))
    commands = []
    monkeypatch.setattr(colab_runner, "run", lambda *args: commands.append(args))
    monkeypatch.setattr(sys, "argv", ["runner", "full"])
    with pytest.raises(FileNotFoundError):
        colab_runner.main()
    assert not commands
