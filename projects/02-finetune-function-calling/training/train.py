from __future__ import annotations

import argparse
import csv
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="checkpoint dir (or 'true' for the latest); survives a Colab disconnect",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from src.data import Example
    from src.prompts import render_example

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

    def load_split(name: str):
        rows = [json.loads(line) for line in Path(f"data/processed/{name}.jsonl").read_text().splitlines() if line.strip()]
        examples = [Example(r["id"], r["tools"], r["user"], r["calls"], r["source"]) for r in rows]
        return Dataset.from_list([{"text": render_example(e, tokenizer)} for e in examples])

    train_dataset = load_split("train")
    eval_dataset = load_split("val")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=config["output_dir"],
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["grad_accum"],
            learning_rate=config["learning_rate"],
            warmup_ratio=0.03,
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
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
    trainer.train(resume_from_checkpoint=resume_from_checkpoint(args.resume_from_checkpoint))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
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
