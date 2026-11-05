"""Experimental domain tool-selection benchmark; never used for production routing."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import Example
from src.metrics import score_output, summarize
from src.prompts import build_messages


def evaluate_backend(cases, generate, identity):
    rows = []
    for case in cases:
        example = Example(case["id"], case["tools"], case["user"], case["calls"], "benchmark-only")
        started = time.perf_counter()
        try:
            prediction, usage = generate(build_messages(example))
            error = None
        except Exception as exc:  # noqa: BLE001 - preserve provider failures as failed benchmark rows
            prediction, usage, error = "", {}, type(exc).__name__
        rows.append({"id": case["id"], "prediction": prediction, "gold": case["calls"], "error": error,
                     "latency_ms": (time.perf_counter() - started) * 1000, "usage": usage,
                     **score_output(prediction, case["calls"])})
    return {"backend": identity, "status": "experimental domain benchmark; not production routing", "metrics": summarize(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "tuned", "hosted", "all"), default="all")
    parser.add_argument("--cases", default=str(Path(__file__).resolve().parents[2] / "01-rag-k8s-docs/evals/tool_selection.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("choose a new output file; historical results are preserved")
    modes = ("base", "tuned", "hosted") if args.mode == "all" else (args.mode,)
    if "tuned" in modes and not args.adapter:
        parser.error("--adapter is required for tuned/all comparisons")
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines()]
    results = {}
    for mode in modes:
        if mode == "hosted":
            from openai import OpenAI

            key = os.getenv("GROQ_API_KEY")
            if not key:
                results[mode] = {"status": "not run: GROQ_API_KEY missing"}
                continue
            client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1", timeout=30, max_retries=0)
            hosted_model = os.getenv("BENCHMARK_HOSTED_MODEL", "openai/gpt-oss-120b")

            def generate(messages, client=client, hosted_model=hosted_model):
                response = client.chat.completions.create(model=hosted_model, messages=messages, temperature=0, max_tokens=256)
                usage = response.usage
                return response.choices[0].message.content or "", {"provider": "groq", "model": hosted_model,
                    "prompt_tokens": usage.prompt_tokens if usage else 0, "completion_tokens": usage.completion_tokens if usage else 0}

            results[mode] = evaluate_backend(cases, generate, {"provider": "groq", "model": hosted_model})
        else:
            from unsloth import FastLanguageModel  # isort: skip

            import torch

            model_name = args.model if mode == "base" else args.adapter
            model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=2048, load_in_4bit=True)
            FastLanguageModel.for_inference(model)

            def generate(messages, model=model, tokenizer=tokenizer, model_name=model_name):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
                if inputs["input_ids"].shape[1] > 1792:
                    raise ValueError("tool prompt exceeds generation context budget")
                torch.cuda.synchronize()
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                result = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
                for special in {tokenizer.eos_token, tokenizer.pad_token}:
                    if special:
                        result = result.replace(special, "")
                return result.strip(), {"provider": "local", "model": model_name}

            results[mode] = evaluate_backend(cases, generate, {"provider": "local", "model": model_name})
            del generate, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"benchmark_only": True, "results": results}, indent=2))
    print(json.dumps({key: value.get("metrics", value) for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
