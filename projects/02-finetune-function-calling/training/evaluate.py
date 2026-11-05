"""Reload an adapter and evaluate deterministic tool generation on validation only."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--example")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    from unsloth import FastLanguageModel  # isort: skip

    import torch

    from src.data import Example
    from src.metrics import score_output, summarize
    from src.prompts import build_messages
    from training.colab_support import require_adapter

    adapter = Path(args.adapter)
    require_adapter(adapter)
    config = json.loads((adapter / "run_config.json").read_text())
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter), max_seq_length=config["max_seq_length"], load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)

    def generate(messages):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        if inputs["input_ids"].shape[1] + args.max_new_tokens > config["max_seq_length"]:
            return "", 0.0, "prompt plus generation budget exceeds model context"
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        for token in {tokenizer.eos_token, tokenizer.pad_token}:
            if token:
                text = text.replace(token, "")
        text = text.strip()
        return text, (time.perf_counter() - started) * 1000, None

    examples = [Example(**json.loads(line)) for line in Path("data/processed/val.jsonl").read_text().splitlines() if line.strip()]
    rows = []
    for example in examples[:args.limit]:
        prediction, latency, error = generate(build_messages(example))
        rows.append({"id": example.id, "prediction": prediction, "gold": example.calls,
                     "latency_ms": latency, "error": error, **score_output(prediction, example.calls)})
    result = {"split": "val", "adapter_reloaded": True, "metrics": summarize(rows), "rows": rows}
    (adapter / "evaluation.json").write_text(json.dumps(result, indent=2))
    (adapter / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    if args.example:
        example = json.loads(Path(args.example).read_text())
        prediction, latency, error = generate(build_messages(Example("interactive", example["tools"], example["user"], [], "interactive", example.get("history", []))))
        (adapter / "example.json").write_text(json.dumps({"input": example, "prediction": prediction, "latency_ms": latency, "error": error}, indent=2))
    print(json.dumps(result["metrics"], indent=2))
    for row in rows[:3]:
        print(json.dumps({"id": row["id"], "prediction": row["prediction"], "gold": row["gold"]}, indent=2))


if __name__ == "__main__":
    main()
