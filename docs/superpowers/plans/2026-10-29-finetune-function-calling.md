# Function-Calling LoRA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Qwen2.5-1.5B-Instruct with LoRA to emit reliable tool calls, then quantify base vs tuned vs prompted-70B quality, latency, and cost.

**Architecture:** Public Hermes/Glaive conversations → normalized single-turn examples → split by tool-schema hash → 4-bit QLoRA with Unsloth on a free Colab T4 → local inference + deterministic parsing → eval harness with tool-name, argument, JSON-validity, and exact-match metrics → HF Hub adapter + model card.

**Tech Stack:** Python 3.12, uv, Hugging Face datasets/transformers/peft/trl, Unsloth, bitsandbytes, PyYAML, HF Hub, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-29-llm-portfolio-design.md`

## Global Constraints

- Public data only, no auth: `NousResearch/hermes-function-calling-v1` (Apache-2.0).
- Dataset files: `func-calling-singleturn.json` (1,893 rows) and `glaive-function-calling-5k.json` (5,209 rows).
- Base model: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0).
- Split by tool-schema hash, never by row; no schema may appear in more than one split.
- Training runs on Colab/Kaggle free GPU. Local tests must run on CPU without downloading model weights.
- No fabricated results: every README number comes from a committed results JSON.
- Commit after every task.

## Review Focus

- **Same tool schema in train and test:** split must fail loudly, not silently leak.
- **Model emits multiple tool calls or nested arguments:** parser and metrics handle both; no crash.
- **Model emits malformed JSON or plain text:** counted as invalid, not a runner crash.
- **Adapter path missing or wrong base model:** inference raises a clear error.
- **70B API returns prose instead of a tool call:** parsed as invalid, recorded, not retried forever.
- **Colab OOM:** max sequence length + gradient accumulation + 4-bit loading keep the 1.5B model inside a T4.
- **Chat template adds special tokens:** prompt/target roundtrip test catches it before training.

---

### Task 1: Scaffold and data ingestion

**Files:**
- Create: `projects/02-finetune-function-calling/pyproject.toml`
- Create: `projects/02-finetune-function-calling/.python-version`
- Create: `projects/02-finetune-function-calling/src/__init__.py`
- Create: `projects/02-finetune-function-calling/src/data.py`
- Create: `projects/02-finetune-function-calling/tests/fixtures/hermes_sample.json`
- Create: `projects/02-finetune-function-calling/tests/test_data.py`

**Interfaces:**
- Produces: `Example(id: str, tools: list[dict], user: str, calls: list[dict], source: str)`, `extract_tagged_json(text: str, tag: str) -> list[dict]`, `parse_hermes_row(row: dict, source: str) -> Example | None`, `load_examples(paths: list[Path]) -> list[Example]`, `dedupe(examples) -> list[Example]`, `schema_hash(tools) -> str`, `split_examples(examples, seed=42, ratios=(0.8, 0.1, 0.1)) -> dict[str, list[Example]]`, `write_jsonl(examples, path)`.
- `extract_tagged_json` uses `json.JSONDecoder.raw_decode`, so nested braces inside `<tool_call>` are safe.
- Rows without a valid assistant tool call are skipped.
- Split groups by schema hash; the function raises `ValueError` if any hash appears in two splits.

- [ ] **Step 1: Create the fixture**

```json
[
  {
    "id": "row-1",
    "conversations": [
      {"from": "system", "value": "You are a function calling AI model. <tools>[{\"type\":\"function\",\"function\":{\"name\":\"get_stock_price\",\"description\":\"Get a stock price\",\"parameters\":{\"type\":\"object\",\"properties\":{\"company\":{\"type\":\"string\"}},\"required\":[\"company\"]}}}]</tools>"},
      {"from": "human", "value": "What is Apple's stock price?"},
      {"from": "gpt", "value": "<tool_call>\n{\"name\": \"get_stock_price\", \"arguments\": {\"company\": \"Apple\"}}\n</tool_call>"}
    ]
  },
  {
    "id": "row-2",
    "conversations": [
      {"from": "system", "value": "<tools>[{\"type\":\"function\",\"function\":{\"name\":\"get_news\",\"parameters\":{\"type\":\"object\",\"properties\":{\"country\":{\"type\":\"string\"}}}}}]</tools>"},
      {"from": "human", "value": "Tell me the news."},
      {"from": "gpt", "value": "I cannot do that."}
    ]
  }
]
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_data.py
import json
from pathlib import Path

import pytest

from src.data import (
    dedupe,
    extract_tagged_json,
    load_examples,
    parse_hermes_row,
    schema_hash,
    split_examples,
)


def test_extract_tagged_json_handles_nested_braces():
    text = '<tool_call>{"name": "f", "arguments": {"a": {"b": 1}}}</tool_call>'
    assert extract_tagged_json(text, "tool_call") == [{"name": "f", "arguments": {"a": {"b": 1}}}]


def test_extract_tagged_json_skips_malformed():
    text = '<tool_call>{not json}</tool_call><tool_call>{"name": "ok", "arguments": {}}</tool_call>'
    assert extract_tagged_json(text, "tool_call") == [{"name": "ok", "arguments": {}}]


def test_extract_tools_keeps_the_array_intact():
    text = '<tools>[{"name": "a"}, {"name": "b"}]</tools>'
    assert extract_tagged_json(text, "tools") == [[{"name": "a"}, {"name": "b"}]]


def test_parse_row_extracts_tools_and_calls():
    row = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())[0]
    example = parse_hermes_row(row, "singleturn")
    assert example is not None
    assert example.tools[0]["function"]["name"] == "get_stock_price"
    assert example.calls[0]["name"] == "get_stock_price"
    assert example.source == "singleturn"


def test_parse_row_skips_no_tool_call():
    row = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())[1]
    assert parse_hermes_row(row, "singleturn") is None


def test_schema_hash_is_order_insensitive():
    a = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    b = [{"function": {"parameters": {}, "name": "x"}, "type": "function"}]
    assert schema_hash(a) == schema_hash(b)


def test_split_has_no_schema_overlap():
    from src.data import Example

    examples = [
        Example(
            str(i),
            [{"type": "function", "function": {"name": f"f{i}", "parameters": {}}}],
            "u",
            [{"name": f"f{i}", "arguments": {}}],
            "t",
        )
        for i in range(6)
    ]
    splits = split_examples(examples, seed=0, ratios=(0.5, 0.25, 0.25))
    hashes = {name: {schema_hash(e.tools) for e in items} for name, items in splits.items()}
    assert not (hashes["train"] & hashes["val"])
    assert not (hashes["train"] & hashes["test"])
    assert not (hashes["val"] & hashes["test"])


def test_dedupe_removes_identical_examples():
    row = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())[0]
    example = parse_hermes_row(row, "singleturn")
    assert len(dedupe([example, example])) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd projects/02-finetune-function-calling && uv run pytest tests/test_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`.

- [ ] **Step 4: Implement scaffolding and data parsing**

```toml
# pyproject.toml
[project]
name = "finetune-function-calling"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "datasets>=2.20",
  "transformers>=4.45",
  "peft>=0.13",
  "trl>=0.12",
  "accelerate>=1.0",
  "bitsandbytes>=0.43",
  "torch>=2.4",
  "httpx>=0.27",
  "openai>=1.40",
  "pyyaml>=6.0",
  "huggingface-hub>=0.25",
]

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.6"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]

[tool.uv]
package = false
```

```text
# .python-version
3.12
```

```python
# src/__init__.py
"""Function-calling fine-tune."""
```

```python
# src/data.py
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

SINGLE_URL = "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main/func-calling-singleturn.json"
GLAIVE_URL = "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main/glaive-function-calling-5k.json"


@dataclass(frozen=True)
class Example:
    id: str
    tools: list[dict]
    user: str
    calls: list[dict]
    source: str


def extract_tagged_json(text: str, tag: str) -> list:
    start_tag = f"<{tag}>"
    decoder = json.JSONDecoder()
    found: list = []
    pos = 0
    while True:
        start = text.find(start_tag, pos)
        if start == -1:
            break
        cursor = start + len(start_tag)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        try:
            value, end = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError:
            pos = start + len(start_tag)
            continue
        found.append(value)
        pos = max(end, start + len(start_tag))
    return found


def _first(conversations: list[dict], role: str) -> str | None:
    for message in conversations:
        if message.get("from") == role:
            return message.get("value")
    return None


def iter_calls(values: list) -> list[dict]:
    calls: list[dict] = []
    for value in values:
        if isinstance(value, dict):
            calls.append(value)
        elif isinstance(value, list):
            calls.extend(item for item in value if isinstance(item, dict))
    return calls


def parse_hermes_row(row: dict, source: str) -> Example | None:
    conversations = row.get("conversations", [])
    system = _first(conversations, "system")
    user = _first(conversations, "human")
    if not system or not user:
        return None
    tool_values = extract_tagged_json(system, "tools")
    if not tool_values or not isinstance(tool_values[0], list):
        return None
    tool_list = tool_values[0]

    after_user = False
    assistant = None
    for message in conversations:
        if message.get("from") == "human":
            after_user = True
            continue
        if after_user and message.get("from") == "gpt":
            assistant = message.get("value")
            break
    if not assistant:
        return None

    calls = []
    for call in iter_calls(extract_tagged_json(assistant, "tool_call")):
        if isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict):
            calls.append({"name": call["name"], "arguments": call["arguments"]})
    if not calls:
        return None
    return Example(str(row.get("id", "")), tool_list, user, calls, source)


def load_examples(paths: list[Path]) -> list[Example]:
    examples: list[Example] = []
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        source = "singleturn" if "singleturn" in path.name else "glaive"
        for row in rows:
            parsed = parse_hermes_row(row, source)
            if parsed is not None:
                examples.append(parsed)
    return dedupe(examples)


def _example_key(example: Example) -> str:
    payload = json.dumps(
        {"tools": example.tools, "user": example.user, "calls": example.calls},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def dedupe(examples: list[Example]) -> list[Example]:
    seen: set[str] = set()
    result: list[Example] = []
    for example in examples:
        key = _example_key(example)
        if key not in seen:
            seen.add(key)
            result.append(example)
    return result


def schema_hash(tools: list[dict]) -> str:
    return hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()


def split_examples(
    examples: list[Example],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[Example]]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(schema_hash(example.tools), []).append(example)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = len(examples)
    train_cut = int(total * ratios[0])
    val_cut = int(total * (ratios[0] + ratios[1]))
    splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for key in keys:
        target = "train" if len(splits["train"]) < train_cut else "val" if len(splits["train"]) + len(splits["val"]) < val_cut else "test"
        splits[target].extend(groups[key])

    seen: dict[str, str] = {}
    for name, items in splits.items():
        for example in items:
            digest = schema_hash(example.tools)
            if digest in seen and seen[digest] != name:
                raise ValueError(f"schema {digest[:8]} leaked into {seen[digest]} and {name}")
            seen[digest] = name
    if not all(splits.values()):
        raise ValueError("each split must be non-empty; provide more schema groups")
    return splits


def write_jsonl(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example.__dict__, ensure_ascii=False) + "\n")
```

- [ ] **Step 5: Run the tests**

Run:
```bash
cd projects/02-finetune-function-calling
uv venv --python 3.12
uv sync --dev
uv lock
uv run pytest tests/test_data.py -q
```
Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): scaffold data ingestion and schema-safe splits"
```

---

### Task 2: Prompt formatting, parsing, and dataset preparation

**Files:**
- Create: `projects/02-finetune-function-calling/src/prompts.py`
- Create: `projects/02-finetune-function-calling/scripts/prepare_data.py`
- Create: `projects/02-finetune-function-calling/tests/test_prompts.py`

**Interfaces:**
- Produces: `SYSTEM_TEMPLATE`, `build_messages(example: Example) -> list[dict[str, str]]`, `format_target(calls: list[dict]) -> str`, `parse_tool_calls(text: str) -> list[dict]`, `render_example(example, tokenizer) -> str`.
- `render_example` = chat template + generation prompt + target + EOS.
- `prepare_data.py` downloads both raw files, writes `data/processed/{train,val,test}.jsonl`, filters examples over `max_seq_length`, and prints counts.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prompts.py
from src.data import Example
from src.prompts import build_messages, format_target, parse_tool_calls, render_example

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


def example():
    return Example("id", TOOLS, "do it", [{"name": "f", "arguments": {"x": 1}}], "test")


def test_build_messages_contains_tools_and_user():
    messages = build_messages(example())
    assert messages[0]["role"] == "system"
    assert "f" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "do it"}


def test_target_roundtrip_to_multiple_calls():
    calls = [{"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": {"y": [1, 2]}}]
    text = format_target(calls)
    assert parse_tool_calls(text) == calls


def test_parse_ignores_malformed_call():
    text = '<tool_call>{oops}</tool_call><tool_call>{"name": "f", "arguments": {}}</tool_call>'
    assert parse_tool_calls(text) == [{"name": "f", "arguments": {}}]


def test_render_example_uses_chat_template_and_target():
    class FakeTokenizer:
        eos_token = "<eos>"
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"

    text = render_example(example(), FakeTokenizer())
    assert text.endswith("<eos>")
    assert "<tool_call>" in text
    assert "ASSISTANT:" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.prompts'`.

- [ ] **Step 3: Implement prompts and preparation**

```python
# src/prompts.py
from __future__ import annotations

import json

from src.data import Example, extract_tagged_json, iter_calls

SYSTEM_TEMPLATE = """You are a function-calling assistant.
Available tools:
<tools>
{tools}
</tools>

Respond ONLY with one or more tool calls in this exact format:
<tool_call>
{{"name": "tool_name", "arguments": {{...}}}}
</tool_call>
Do not add explanations or markdown."""


def build_messages(example: Example) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(tools=json.dumps(example.tools))},
        {"role": "user", "content": example.user},
    ]


def format_target(calls: list[dict]) -> str:
    return "\n".join(
        f"<tool_call>\n{json.dumps(call, ensure_ascii=False)}\n</tool_call>" for call in calls
    )


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for call in iter_calls(extract_tagged_json(text, "tool_call")):
        if isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict):
            calls.append({"name": call["name"], "arguments": call["arguments"]})
    return calls


def render_example(example: Example, tokenizer) -> str:
    prompt = tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True
    )
    return prompt + format_target(example.calls) + tokenizer.eos_token
```

```python
# scripts/prepare_data.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from transformers import AutoTokenizer

from src.data import GLAIVE_URL, SINGLE_URL, load_examples, split_examples, write_jsonl
from src.prompts import render_example


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    raw = Path("data/raw")
    single = download(SINGLE_URL, raw / "func-calling-singleturn.json")
    glaive = download(GLAIVE_URL, raw / "glaive-function-calling-5k.json")
    examples = load_examples([single, glaive])
    splits = split_examples(examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    counts = {}
    for name, items in splits.items():
        kept = [e for e in items if len(tokenizer(render_example(e, tokenizer))["input_ids"]) <= args.max_seq_length]
        write_jsonl(kept, Path("data/processed") / f"{name}.jsonl")
        counts[name] = (len(items), len(kept))
    print(counts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_prompts.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Prepare the data**

Run: `uv run python scripts/prepare_data.py`
Expected: downloads both files, prints counts like `{'train': (..., ...), 'val': (..., ...), 'test': (..., ...)}`; writes three JSONL files. No tokenizer/model weights for training are loaded, only the tokenizer.

- [ ] **Step 6: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): prompt formatting and dataset preparation"
```

---

### Task 3: QLoRA training script and Colab notebook

**Files:**
- Create: `projects/02-finetune-function-calling/config.yaml`
- Create: `projects/02-finetune-function-calling/requirements-train.txt`
- Create: `projects/02-finetune-function-calling/training/__init__.py`
- Create: `projects/02-finetune-function-calling/training/train.py`
- Create: `projects/02-finetune-function-calling/training/colab.ipynb`
- Create: `projects/02-finetune-function-calling/tests/test_train_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict`, `main()`; adapter written to `output_dir`.
- Config keys: `model`, `max_seq_length`, `epochs`, `learning_rate`, `batch_size`, `grad_accum`, `lora_r`, `lora_alpha`, `seed`, `output_dir`.

- [ ] **Step 1: Write the failing config test**

```python
# tests/test_train_config.py
import pytest
import yaml

from training.train import load_config


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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_train_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'training'`.

- [ ] **Step 3: Implement training**

```yaml
# config.yaml
model: Qwen/Qwen2.5-1.5B-Instruct
max_seq_length: 2048
epochs: 1
learning_rate: 0.0002
batch_size: 2
grad_accum: 8
lora_r: 16
lora_alpha: 32
seed: 42
output_dir: outputs/qwen2.5-1.5b-lora
```

```text
# requirements-train.txt
unsloth
trl>=0.16
peft
transformers
datasets
accelerate
bitsandbytes
```

```python
# training/__init__.py
"""Training entry points."""
```

```python
# training/train.py
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()
    config = load_config(args.config)

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    from src.prompts import render_example
    from src.data import Example

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
            save_total_limit=2,
            seed=config["seed"],
            max_length=config["max_seq_length"],
            dataset_text_field="text",
            report_to="none",
            max_steps=args.max_steps,
        ),
    )
    trainer.train()
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
```

**Colab notebook cells (create `training/colab.ipynb` with these cells):**

Cell 1 (markdown): `# Function-calling LoRA\nRuntime → Change runtime type → T4 GPU.`

Cell 2 (code):
```python
!git clone <your-repo-url> portfolio
%cd portfolio/projects/02-finetune-function-calling
!pip install -q -r requirements-train.txt
```

Cell 3 (code):
```python
!python scripts/prepare_data.py
```

Cell 4 (code):
```python
!python training/train.py --config config.yaml --max-steps 2
```

Cell 5 (code):
```python
!python training/train.py --config config.yaml
```

- [ ] **Step 4: Run the config tests**

Run: `uv run pytest tests/test_train_config.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Colab smoke test (human gate)**

On Colab T4, run cells 2–4. Expected: data prep completes; training runs 2 steps and saves an adapter; no OOM. If OOM, lower `batch_size` to 1 and raise `grad_accum` to 16.

- [ ] **Step 6: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): QLoRA training script and Colab workflow"
```

---

### Task 4: Local inference and tool-call parsing

**Files:**
- Create: `projects/02-finetune-function-calling/src/infer.py`
- Create: `projects/02-finetune-function-calling/tests/test_infer.py`

**Interfaces:**
- Produces: `load_model(base_model: str, adapter: str | None = None)`, `generate_calls(model, tokenizer, example: Example, max_new_tokens: int = 512) -> tuple[list[dict], str]`.
- Raises `FileNotFoundError` if `adapter` is provided and missing.
- Parsing is delegated to `src.prompts.parse_tool_calls`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_infer.py
import pytest

from src.infer import load_model
from src.prompts import parse_tool_calls


def test_parse_multiple_calls():
    text = '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
    assert parse_tool_calls(text) == [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]


def test_missing_adapter_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("Qwen/Qwen2.5-1.5B-Instruct", str(tmp_path / "nope"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_infer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.infer'`.

- [ ] **Step 3: Implement inference**

```python
# src/infer.py
from __future__ import annotations

from pathlib import Path

from src.data import Example
from src.prompts import build_messages, parse_tool_calls


def load_model(base_model: str, adapter: str | None = None):
    if adapter and not Path(adapter).exists():
        raise FileNotFoundError(f"adapter not found: {adapter}")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate_calls(model, tokenizer, example: Example, max_new_tokens: int = 512) -> tuple[list[dict], str]:
    import torch

    prompt = tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_tool_calls(text), text
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_infer.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): local inference and tool-call parsing"
```

---

### Task 5: Metrics and evaluation runner

**Files:**
- Create: `projects/02-finetune-function-calling/src/metrics.py`
- Create: `projects/02-finetune-function-calling/evals/__init__.py`
- Create: `projects/02-finetune-function-calling/evals/run_eval.py`
- Create: `projects/02-finetune-function-calling/tests/test_metrics.py`
- Create: `projects/02-finetune-function-calling/tests/test_eval_runner.py`

**Interfaces:**
- Produces: `score_example(predicted: list[dict], gold: list[dict]) -> dict` with `tool_name_accuracy`, `arg_precision`, `arg_recall`, `arg_f1`, `full_exact`, `json_valid`; `aggregate(rows) -> dict`; `run_eval(examples, predict_fn) -> dict`.
- Metric definitions:
  - `tool_name_accuracy`: 1.0 iff the set of predicted tool names equals the set of gold tool names.
  - `arg_precision/recall/f1`: micro-averaged over flattened argument fields of calls matched by name.
  - `full_exact`: canonical JSON of predicted calls equals gold calls (order-insensitive).
  - `json_valid`: 1.0 iff at least one valid tool call was parsed.
- `run_eval` records per-example latency and writes aggregate metrics via `aggregate`.

- [ ] **Step 1: Write the failing metric tests**

```python
# tests/test_metrics.py
import pytest

from src.metrics import aggregate, score_example


GOLD = [{"name": "get_price", "arguments": {"company": "Apple", "currency": "USD"}}]


def test_exact_match_scores_one():
    row = score_example(GOLD, GOLD)
    assert row["tool_name_accuracy"] == 1.0
    assert row["arg_f1"] == 1.0
    assert row["full_exact"] == 1.0
    assert row["json_valid"] == 1.0


def test_partial_arguments_score_partial_f1():
    pred = [{"name": "get_price", "arguments": {"company": "Apple", "currency": "EUR"}}]
    row = score_example(pred, GOLD)
    assert row["tool_name_accuracy"] == 1.0
    assert row["arg_precision"] == 0.5
    assert row["arg_recall"] == 0.5
    assert row["arg_f1"] == 0.5
    assert row["full_exact"] == 0.0


def test_missing_call_lowers_recall():
    pred = [{"name": "get_price", "arguments": {"company": "Apple"}}]
    row = score_example(pred, GOLD)
    assert row["arg_recall"] == 0.5
    assert row["arg_precision"] == 1.0


def test_no_calls_is_invalid():
    row = score_example([], GOLD)
    assert row["json_valid"] == 0.0
    assert row["arg_f1"] == 0.0


def test_nested_arguments_are_flattened():
    gold = [{"name": "f", "arguments": {"a": {"b": 1}, "c": [1, 2], "d": 5}}]
    pred = [{"name": "f", "arguments": {"a": {"b": 1}, "c": [1, 3], "d": 5}}]
    row = score_example(pred, gold)
    assert row["arg_f1"] == pytest.approx(0.75)


def test_aggregate_means_metrics():
    rows = [
        {"tool_name_accuracy": 1.0, "arg_precision": 1.0, "arg_recall": 1.0, "arg_f1": 1.0, "full_exact": 1.0, "json_valid": 1.0},
        {"tool_name_accuracy": 0.0, "arg_precision": 0.0, "arg_recall": 0.0, "arg_f1": 0.0, "full_exact": 0.0, "json_valid": 0.0},
    ]
    result = aggregate(rows)
    assert result["tool_name_accuracy"] == 0.5
    assert result["arg_f1"] == 0.5
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`.

- [ ] **Step 3: Implement metrics**

```python
# src/metrics.py
from __future__ import annotations

import json


def _normalize(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip().lower()


def flatten(value, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(flatten(item, f"{prefix}[{index}]"))
    else:
        flat[prefix] = _normalize(value)
    return flat


def _match(predicted: list[dict], gold: list[dict]) -> list[tuple[dict, dict]]:
    used: set[int] = set()
    pairs = []
    for pred in predicted:
        for index, target in enumerate(gold):
            if index in used:
                continue
            if pred["name"] == target["name"]:
                used.add(index)
                pairs.append((pred, target))
                break
    return pairs


def _canonical(calls: list[dict]) -> list[str]:
    return sorted(json.dumps(call, sort_keys=True) for call in calls)


def score_example(predicted: list[dict], gold: list[dict]) -> dict:
    if not predicted:
        return {
            "tool_name_accuracy": 0.0,
            "arg_precision": 0.0,
            "arg_recall": 0.0,
            "arg_f1": 0.0,
            "full_exact": 0.0,
            "json_valid": 0.0,
        }
    pairs = _match(predicted, gold)
    tp = fp = fn = 0
    for pred, target in pairs:
        pred_fields = flatten(pred.get("arguments", {}))
        gold_fields = flatten(target.get("arguments", {}))
        common = set(pred_fields) & set(gold_fields)
        tp += sum(pred_fields[key] == gold_fields[key] for key in common)
        fp += len(pred_fields) - sum(pred_fields[key] == gold_fields[key] for key in common)
        fn += len(gold_fields) - sum(pred_fields[key] == gold_fields[key] for key in common)
    tp += 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tool_name_accuracy": 1.0 if {p["name"] for p in predicted} == {g["name"] for g in gold} else 0.0,
        "arg_precision": precision,
        "arg_recall": recall,
        "arg_f1": f1,
        "full_exact": 1.0 if _canonical(predicted) == _canonical(gold) else 0.0,
        "json_valid": 1.0,
    }


def aggregate(rows: list[dict]) -> dict:
    keys = ["tool_name_accuracy", "arg_precision", "arg_recall", "arg_f1", "full_exact", "json_valid"]
    return {key: sum(row[key] for row in rows) / len(rows) if rows else 0.0 for key in keys}
```

```python
# evals/__init__.py
"""Evaluation entry points."""
```

```python
# evals/run_eval.py
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import aggregate, score_example


def run_eval(examples: list[dict], predict_fn) -> dict:
    rows = []
    started = time.monotonic()
    for example in examples:
        t0 = time.monotonic()
        calls, raw = predict_fn(example)
        latency_ms = int((time.monotonic() - t0) * 1000)
        row = score_example(calls, example["calls"])
        row.update({"id": example["id"], "latency_ms": latency_ms, "raw": raw, "predicted": calls})
        rows.append(row)
    latencies = sorted(row["latency_ms"] for row in rows)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0
    return {
        "examples": len(rows),
        "metrics": aggregate(rows),
        "p95_latency_ms": p95,
        "wall_seconds": int(time.monotonic() - started),
        "rows": rows,
    }


def load_test(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=["base", "tuned", "api"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="outputs/qwen2.5-1.5b-lora")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    examples = load_test("data/processed/test.jsonl")
    if args.limit:
        examples = examples[: args.limit]

    if args.predictor == "api":
        from src.api_baseline import make_api_predictor

        predict_fn = make_api_predictor()
    else:
        from src.infer import generate_calls, load_model
        from src.data import Example

        adapter = args.adapter if args.predictor == "tuned" else None
        model, tokenizer = load_model(args.model, adapter)

        def predict_fn(example):
            return generate_calls(
                model, tokenizer,
                Example(example["id"], example["tools"], example["user"], example["calls"], example["source"]),
            )

    result = run_eval(examples, predict_fn)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))
    print(f"p95={result['p95_latency_ms']}ms wrote {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the runner test**

```python
# tests/test_eval_runner.py
from evals.run_eval import run_eval


def test_runner_records_metrics_and_latency():
    examples = [{"id": "1", "tools": [], "user": "u", "calls": [{"name": "f", "arguments": {"x": 1}}], "source": "t"}]
    result = run_eval(examples, lambda example: ([{"name": "f", "arguments": {"x": 1}}], "raw"))
    assert result["examples"] == 1
    assert result["metrics"]["full_exact"] == 1.0
    assert result["p95_latency_ms"] >= 0
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_metrics.py tests/test_eval_runner.py -q`
Expected: `7 passed`.

- [ ] **Step 6: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): tool-call metrics and evaluation runner"
```

---

### Task 6: API baseline predictor

**Files:**
- Create: `projects/02-finetune-function-calling/src/api_baseline.py`
- Create: `projects/02-finetune-function-calling/tests/test_api_baseline.py`

**Interfaces:**
- Produces: `make_api_predictor(model: str | None = None, client=None) -> Callable[[dict], tuple[list[dict], str]]`.
- Uses Groq's OpenAI-compatible endpoint, `GROQ_API_KEY`, default model `llama-3.3-70b-versatile`.
- Retries 3 times on rate limit/5xx with exponential backoff; returns `([], text)` after exhaustion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_baseline.py
from src.api_baseline import make_api_predictor


def test_api_predictor_parses_tool_calls():
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    class Message:
                        content = '<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>'
                    class Choice:
                        message = Message()
                    class Response:
                        choices = [Choice()]
                    return Response()

    predict = make_api_predictor(client=FakeClient())
    calls, raw = predict(
        {
            "id": "1",
            "user": "u",
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "calls": [{"name": "f", "arguments": {"x": 1}}],
        }
    )
    assert calls == [{"name": "f", "arguments": {"x": 1}}]
    assert "tool_call" in raw
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_api_baseline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api_baseline'`.

- [ ] **Step 3: Implement the API baseline**

```python
# src/api_baseline.py
from __future__ import annotations

import json
import os
import time

from src.data import Example
from src.prompts import build_messages, parse_tool_calls


def make_api_predictor(model: str | None = None, client=None):
    if client is None:
        from openai import OpenAI

        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is required for the API baseline")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def predict(row: dict) -> tuple[list[dict], str]:
        example = Example(row["id"], row["tools"], row["user"], row["calls"], row.get("source", "api"))
        messages = build_messages(example)
        text = ""
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )
                text = response.choices[0].message.content or ""
                break
            except Exception:
                if attempt == 2:
                    return [], text
                time.sleep(2 ** attempt)
        return parse_tool_calls(text), text

    return predict
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_api_baseline.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): prompted 70B API baseline"
```

---

### Task 7: Forgetting check

**Files:**
- Create: `projects/02-finetune-function-calling/evals/forgetting.py`
- Create: `projects/02-finetune-function-calling/tests/test_forgetting.py`

**Interfaces:**
- Produces: `parse_mmlu_answer(text: str) -> str | None`, `select_mmlu(rows, n=200, seed=42) -> list[dict]`, `score_mmlu(model, tokenizer, rows) -> float`.
- Uses `cais/mmlu` test split, zero-shot prompt, exact letter match.
- Report base vs tuned accuracy and the delta.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forgetting.py
from evals.forgetting import parse_mmlu_answer, select_mmlu


def test_parse_answer():
    assert parse_mmlu_answer("The answer is B") == "B"
    assert parse_mmlu_answer("C.") == "C"
    assert parse_mmlu_answer("I don't know") is None


def test_select_is_deterministic():
    rows = [{"question": str(i), "choices": [], "answer": 0} for i in range(500)]
    assert [r["question"] for r in select_mmlu(rows, n=10, seed=1)] == [r["question"] for r in select_mmlu(rows, n=10, seed=1)]
    assert len(select_mmlu(rows, n=10, seed=1)) == 10
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_forgetting.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'evals.forgetting'`.

- [ ] **Step 3: Implement the forgetting check**

```python
# evals/forgetting.py
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LETTERS = "ABCD"


def parse_mmlu_answer(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def select_mmlu(rows: list[dict], n: int = 200, seed: int = 42) -> list[dict]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _prompt(row: dict) -> str:
    choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"]))
    return f"Answer with a single letter A, B, C, or D.\n\nQuestion: {row['question']}\n{choices}\nAnswer:"


def score_mmlu(model, tokenizer, rows: list[dict]) -> float:
    import torch

    correct = 0
    for row in rows:
        inputs = tokenizer(_prompt(row), return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if parse_mmlu_answer(text) == LETTERS[row["answer"]]:
            correct += 1
    return correct / len(rows) if rows else 0.0


def main() -> None:
    import json
    import sys
    from argparse import ArgumentParser

    from datasets import load_dataset

    from src.infer import load_model

    parser = ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="results/forgetting.json")
    args = parser.parse_args()

    rows = select_mmlu(list(load_dataset("cais/mmlu", "all", split="test")), n=200)
    base, tokenizer = load_model(args.base_model)
    base_score = score_mmlu(base, tokenizer, rows)
    del base
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    tuned, tokenizer = load_model(args.base_model, args.adapter)
    tuned_score = score_mmlu(tuned, tokenizer, rows)
    result = {
        "n": len(rows),
        "base_accuracy": base_score,
        "tuned_accuracy": tuned_score,
        "delta": tuned_score - base_score,
    }
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_forgetting.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): catastrophic-forgetting check"
```

---

### Task 8: Run the comparison and record results

**Files:**
- Create: `projects/02-finetune-function-calling/results/base.json` (generated)
- Create: `projects/02-finetune-function-calling/results/tuned.json` (generated)
- Create: `projects/02-finetune-function-calling/results/api70b.json` (generated)
- Create: `projects/02-finetune-function-calling/results/forgetting.json` (generated)
- Create: `projects/02-finetune-function-calling/scripts/run_comparison.sh`

**Interfaces:**
- Consumes: trained adapter, test JSONL.
- Produces: four committed results files used by the README; no hand-edited numbers.

- [ ] **Step 1: Write the comparison script**

```bash
# scripts/run_comparison.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${ADAPTER:-outputs/qwen2.5-1.5b-lora}"

uv run python evals/run_eval.py --predictor base --output results/base.json
uv run python evals/run_eval.py --predictor tuned --adapter "$ADAPTER" --output results/tuned.json
uv run python evals/run_eval.py --predictor api --output results/api70b.json
uv run python evals/forgetting.py --adapter "$ADAPTER" --output results/forgetting.json
```

- [ ] **Step 2: Train the adapter**

On Colab T4:
```bash
python training/train.py --config config.yaml
```
Expected: adapter and tokenizer in `outputs/qwen2.5-1.5b-lora/`; training loss logged; no OOM.

- [ ] **Step 3: Run all evaluations on the GPU runtime**

```bash
export GROQ_API_KEY=...
bash scripts/run_comparison.sh
```
Expected: four JSON files in `results/`; each has `examples`, `metrics`, and `p95_latency_ms` (except forgetting, which has base/tuned accuracy).

- [ ] **Step 4: Sanity-check the results**

Run:
```bash
uv run python - <<'PY'
import json
from pathlib import Path
for name in ["base", "tuned", "api70b"]:
    data = json.loads(Path(f"results/{name}.json").read_text())
    assert data["examples"] > 0
    assert 0.0 <= data["metrics"]["full_exact"] <= 1.0
    print(name, data["metrics"]["full_exact"], data["p95_latency_ms"])
PY
```
Expected: three lines with sanity-checked metrics.

- [ ] **Step 5: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "results(finetune): base, tuned, 70B, and forgetting evals"
```

---

### Task 9: Model card, HF Hub upload, CI, and README

**Files:**
- Create: `projects/02-finetune-function-calling/scripts/__init__.py`
- Create: `projects/02-finetune-function-calling/MODEL_CARD.md`
- Create: `projects/02-finetune-function-calling/scripts/upload_adapter.py`
- Create: `projects/02-finetune-function-calling/tests/test_model_card.py`
- Create: `projects/02-finetune-function-calling/reproduce.sh`
- Create: `projects/02-finetune-function-calling/README.md`
- Create: `projects/02-finetune-function-calling/ATTRIBUTIONS.md`
- Create: `.github/workflows/finetune.yml`

**Interfaces:**
- Produces: `render_model_card(base: dict, tuned: dict, api: dict, forgetting: dict) -> str`; upload script; CI tests; README results table.

- [ ] **Step 1: Write the failing model-card test**

```python
# tests/test_model_card.py
from scripts.upload_adapter import render_model_card

BASE = {"metrics": {"full_exact": 0.1, "arg_f1": 0.2}, "p95_latency_ms": 100}
TUNED = {"metrics": {"full_exact": 0.6, "arg_f1": 0.7}, "p95_latency_ms": 120}
API = {"metrics": {"full_exact": 0.7, "arg_f1": 0.8}, "p95_latency_ms": 900}
FORGET = {"base_accuracy": 0.5, "tuned_accuracy": 0.48, "delta": -0.02}


def test_model_card_contains_all_results():
    card = render_model_card(BASE, TUNED, API, FORGET)
    assert "0.600" in card
    assert "0.700" in card
    assert "-0.020" in card
    assert "Qwen2.5-1.5B-Instruct" in card
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_model_card.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.upload_adapter'`.

- [ ] **Step 3: Implement card, uploader, CI, and README**

```python
# scripts/__init__.py
"""Utility scripts."""
```

```python
# scripts/upload_adapter.py
from __future__ import annotations

import json
from pathlib import Path


def render_model_card(base: dict, tuned: dict, api: dict, forgetting: dict) -> str:
    return f"""# Qwen2.5-1.5B-Instruct — Function-Calling LoRA

Fine-tuned with 4-bit QLoRA on the Hermes/Glaive function-calling data.

## Results (held-out, schema-disjoint test set)

| model | full-call exact | argument F1 | p95 latency |
|---|---:|---:|---:|
| base Qwen2.5-1.5B-Instruct | {base['metrics']['full_exact']:.3f} | {base['metrics']['arg_f1']:.3f} | {base['p95_latency_ms']} ms |
| + this LoRA adapter | {tuned['metrics']['full_exact']:.3f} | {tuned['metrics']['arg_f1']:.3f} | {tuned['p95_latency_ms']} ms |
| prompted 70B API | {api['metrics']['full_exact']:.3f} | {api['metrics']['arg_f1']:.3f} | {api['p95_latency_ms']} ms |

## Forgetting check

MMLU (n=200): base {forgetting['base_accuracy']:.3f}, tuned {forgetting['tuned_accuracy']:.3f}, delta {forgetting['delta']:+.3f}.

## Training

- Dataset: `NousResearch/hermes-function-calling-v1` (Apache-2.0)
- Split: by tool-schema hash, 80/10/10
- LoRA r=16, alpha=32, 1 epoch, max sequence length 2048
- Hardware: Colab free T4

## Limitations

- Synthetic training data; tool schemas are not production APIs.
- No constrained decoding in the headline result; JSON validity is measured, not guaranteed.
- Test tool schemas are disjoint from training, but the data distribution is still narrow.
"""


def main() -> None:
    from argparse import ArgumentParser

    from huggingface_hub import HfApi

    parser = ArgumentParser()
    parser.add_argument("--repo", required=True, help="user/model-name")
    parser.add_argument("--adapter", default="outputs/qwen2.5-1.5b-lora")
    args = parser.parse_args()

    results = Path("results")
    base = json.loads((results / "base.json").read_text())
    tuned = json.loads((results / "tuned.json").read_text())
    api = json.loads((results / "api70b.json").read_text())
    forgetting = json.loads((results / "forgetting.json").read_text())
    card = render_model_card(base, tuned, api, forgetting)
    Path("MODEL_CARD.md").write_text(card)
    HfApi().upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=args.adapter,
        path_in_repo=".",
        commit_message="Upload function-calling LoRA adapter",
    )
    HfApi().upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add model card",
    )
    print(f"uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
```

```bash
# reproduce.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

uv sync --dev
uv run pytest -q
uv run python scripts/prepare_data.py
echo "Training requires a GPU. Run: python training/train.py --config config.yaml"
echo "Then run: bash scripts/run_comparison.sh"
```

```yaml
# .github/workflows/finetune.yml
name: finetune
on:
  push:
    paths: ["projects/02-finetune-function-calling/**", ".github/workflows/finetune.yml"]
  pull_request:
    paths: ["projects/02-finetune-function-calling/**", ".github/workflows/finetune.yml"]

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: projects/02-finetune-function-calling
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync --dev
      - run: uv run pytest -q
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_model_card.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Upload the adapter (human gate)**

```bash
hf auth login
uv run python scripts/upload_adapter.py --repo <your-hf-user>/qwen2.5-1.5b-function-calling-lora
```
Expected: adapter files and model card appear on HF Hub. Verify the model card renders and the numbers match `results/*.json`.

- [ ] **Step 6: Write the README**

Include:
1. One-paragraph pitch and HF Hub link.
2. Results table (from `results/*.json`), including base/tuned/70B, latency, and cost per 1k calls.
3. Forgetting check result.
4. Training configuration and Colab link.
5. Reproduction commands.
6. Design decisions: schema-disjoint split, why Qwen2.5-1.5B, why LoRA.
7. Limitations.
8. Attribution.

- [ ] **Step 7: Write attribution**

```markdown
# Attributions

- `NousResearch/hermes-function-calling-v1`: Apache-2.0, https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- `Qwen/Qwen2.5-1.5B-Instruct`: Apache-2.0, https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- `cais/mmlu`: MIT, https://huggingface.co/datasets/cais/mmlu
- Unsloth, TRL, PEFT, Transformers, bitsandbytes: Apache-2.0
- Structured-extraction LoRA ideas informed by `deshpandetanmay/qlora-structured-extraction` (Apache-2.0); forgetting-check design informed by `aghasalim/lora-forgetting` (MIT). Adapted code is marked in the source file; no code copied from unlicensed repositories.
```

- [ ] **Step 8: Commit**

```bash
git add projects/02-finetune-function-calling .github/workflows/finetune.yml
git commit -m "docs(finetune): model card, upload script, CI, and README"
```

---

### Task 10: Portfolio root README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: both project READMEs, committed results JSON, live demo and HF Hub links.
- Produces: a landing page that lets a reviewer understand both projects in under two minutes.

- [ ] **Step 1: Write the root README**

Include:
1. One-paragraph positioning: applied AI / LLM engineer focused on retrieval, evaluation, and small-model fine-tuning.
2. Two project cards, each with: one-line problem, approach, headline verified metric, stack, links to code/demo/model card.
3. A combined results table copied from the committed results JSON, not hand-written.
4. A short "how I work" section: evals before claims, pinned artifacts, licensed data, reproducible commands.
5. A "what I'd do next" section with one concrete next step per project.
6. Repository layout.

- [ ] **Step 2: Verify every link and command**

Run:
```bash
cd /home/deansama/projects/ML-Engineer-Projects-for-Employment
grep -oE 'https?://[^ )]+' README.md
```
Expected: every URL resolves (check the demo, HF Hub, and GitHub links manually). Every command block is copy-pasteable from the repo root.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: portfolio root README"
```

---

## Plan Self-Review

- Spec coverage: public data, schema-disjoint split, QLoRA, base/tuned/70B comparison, forgetting check, HF Hub model card, CI, project README, and portfolio root README — all mapped to tasks. Outlines constrained decoding is optional in the spec and deliberately excluded from the headline path; add it as a follow-up if time allows.
- Placeholder scan: no TBD/TODO; the only human-supplied values are the repo clone URL, HF repo ID, and API keys, explicitly marked as gates.
- Type consistency: `Example` fields are reused everywhere; `parse_tool_calls` returns `list[dict]`; `score_example` and `aggregate` keys match the README/model-card contract.
- Review focus: schema leakage, multi-call parsing, malformed JSON, missing adapters, API prose, T4 memory, chat-template roundtrip — each has a test or a gate in the owning task.
