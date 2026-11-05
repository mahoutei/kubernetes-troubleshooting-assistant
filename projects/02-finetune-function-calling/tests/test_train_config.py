import pytest
import yaml

from training.train import load_config, resume_from_checkpoint


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({
        "model": "m", "max_seq_length": 1024, "epochs": 1, "learning_rate": 0.0002,
        "batch_size": 2, "grad_accum": 8, "lora_r": 16, "lora_alpha": 32,
        "seed": 42, "output_dir": "out",
    }))
    cfg = load_config(path)
    assert cfg["model"] == "m"
    assert cfg["lora_r"] == 16


def test_load_config_rejects_missing_key(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"model": "m"}))
    with pytest.raises(ValueError, match="missing config keys"):
        load_config(path)


def test_resume_true_becomes_bool_and_paths_pass_through():
    assert resume_from_checkpoint("true") is True
    assert resume_from_checkpoint("outputs/checkpoint-200") == "outputs/checkpoint-200"
    assert resume_from_checkpoint(None) is None


def test_smoke_training_evaluates_within_requested_steps():
    from training.train import evaluation_steps

    assert evaluation_steps(2) == 2
    assert evaluation_steps(300) == 200
    assert evaluation_steps(-1) == 200


def test_training_rejects_stale_or_incompatible_preprocessing(tmp_path):
    import json

    from src.data import PREPROCESSING_VERSION
    from training.train import validate_manifest

    path = tmp_path / "manifest.json"
    valid = {"preprocessing_version": PREPROCESSING_VERSION, "tokenizer_model": "m", "max_seq_length": 2048}
    config = {"model": "m", "max_seq_length": 2048}
    path.write_text(json.dumps(valid))
    validate_manifest(path, config)
    for key, value in [("preprocessing_version", 1), ("tokenizer_model", "other"), ("max_seq_length", 100)]:
        path.write_text(json.dumps({**valid, key: value}))
        with pytest.raises(ValueError, match="rerun"):
            validate_manifest(path, config)
