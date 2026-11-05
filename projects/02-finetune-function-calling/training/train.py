from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

REQUIRED = {
    "model", "max_seq_length", "epochs", "learning_rate", "batch_size",
    "grad_accum", "lora_r", "lora_alpha", "seed", "output_dir",
}


def load_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    missing = REQUIRED - set(config or {})
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    return config


def resume_from_checkpoint(value: str | None) -> bool | str | None:
    """'true' means the latest checkpoint; any other value is a path."""
    if value == "true":
        return True
    return value


def evaluation_steps(max_steps: int) -> int:
    return min(200, max_steps) if max_steps > 0 else 200


def validate_manifest(path: str | Path, config: dict) -> None:
    from src.data import PREPROCESSING_VERSION

    manifest = json.loads(Path(path).read_text())
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("outdated processed data; rerun scripts/prepare_data.py to preserve conversation history")
    if manifest.get("tokenizer_model") != config["model"]:
        raise ValueError("processed tokenizer model differs from training model; rerun scripts/prepare_data.py --model")
    if manifest.get("max_seq_length") != config["max_seq_length"]:
        raise ValueError("processed max_seq_length differs from training config; rerun scripts/prepare_data.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="checkpoint dir (or 'true' for the latest); survives a Colab disconnect",
    )
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--run-metadata")
    parser.add_argument("--backup-dir")
    args = parser.parse_args()
    for limit in (args.max_train_examples, args.max_eval_examples):
        if limit is not None and limit < 1:
            parser.error("example limits must be positive")
    config = load_config(args.config)
    validate_manifest("data/processed/manifest.json", config)

    from unsloth import FastLanguageModel  # isort: skip

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from src.data import Example
    from src.prompts import render_example
    from training.colab_support import complete_checkpoint

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"],
        max_seq_length=config["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )

    def load_split(name: str, limit: int | None):
        rows = [json.loads(line) for line in Path(f"data/processed/{name}.jsonl").read_text().splitlines() if line.strip()]
        examples = [Example(**r) for r in rows[:limit]]
        return Dataset.from_list([{"text": render_example(e, tokenizer)} for e in examples])

    train_dataset = load_split("train", args.max_train_examples)
    eval_dataset = load_split("val", args.max_eval_examples)

    metadata = json.loads(Path(args.run_metadata).read_text()) if args.run_metadata else None

    class BackupCallback(TrainerCallback):
        def on_save(self, training_args, state, control, **kwargs):
            if metadata:
                complete_checkpoint(
                    Path(training_args.output_dir) / f"checkpoint-{state.global_step}",
                    metadata["fingerprint"], metadata["mode"],
                    Path(args.backup_dir) if args.backup_dir else None,
                )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[BackupCallback()],
        args=SFTConfig(
            output_dir=config["output_dir"],
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["grad_accum"],
            learning_rate=config["learning_rate"],
            warmup_ratio=0.03,
            logging_steps=1 if args.max_steps > 0 else 10,
            save_steps=200,
            eval_steps=evaluation_steps(args.max_steps),
            # Without eval_strategy the eval_dataset above is silently ignored.
            eval_strategy="steps",
            # T4 supports fp16 but not bf16; explicitly disable bf16.
            fp16=True,
            bf16=False,
            save_total_limit=2,
            seed=config["seed"],
            max_length=config["max_seq_length"],
            dataset_text_field="text",
            report_to="none",
            max_steps=args.max_steps,
        ),
    )
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint(args.resume_from_checkpoint))
    evaluation = trainer.evaluate()
    report = {"global_step": trainer.state.global_step, "training_loss": result.training_loss,
              "eval_loss": evaluation.get("eval_loss")}
    if any(not isinstance(report[key], (int, float)) or not math.isfinite(report[key])
           for key in ("training_loss", "eval_loss")):
        raise RuntimeError("training/validation produced a nonfinite loss; stop before promotion")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    history = trainer.state.log_history
    fieldnames = sorted({key for row in history for key in row})
    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
