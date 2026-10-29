# Function-Calling LoRA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune Qwen2.5-1.5B-Instruct with LoRA to emit reliable tool calls, then quantify base vs tuned vs prompted free-tier API model (`openai/gpt-oss-120b`) quality, latency, and cost per 1k calls.

**Architecture:** Public Hermes/Glaive conversations → normalized single-turn examples → dedupe → split by normalized tool-schema hash → 4-bit QLoRA with Unsloth on a free Colab T4 → local inference + deterministic parsing → eval harness with tool-name, argument, JSON-validity, exact-match, latency, and token-cost metrics → HF Hub adapter + model card.

**Tech Stack:** Python 3.12, uv, Hugging Face datasets/transformers/peft/trl, Unsloth, bitsandbytes, PyYAML, HF Hub, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-29-llm-portfolio-design.md`

## Global Constraints

- Public data only, no auth: `NousResearch/hermes-function-calling-v1` (Apache-2.0).
- Dataset is pinned to revision `dae3e1d28cfbcf4b915c04ea1e072030529b4bda`. Files: `func-calling-singleturn.json` (1,893 rows, 17,793,686 bytes, sha256 `fb2df9fe3f295dfd65b223c2dbf04d6b90006357d737f8f8927749f0bea09527`) and `glaive-function-calling-5k.json` (5,209 rows, 20,416,419 bytes, sha256 `b98eb3f160359f27ad15018e974ce6db444f566eb5be4aa9e4aa690b34d50832`). The pin lives in `src/data.py`; `prepare_data.py` refuses to continue on a checksum/row-count mismatch and writes `data/processed/manifest.json`.
- Base model: `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0).
- Prompted baseline: Groq `openai/gpt-oss-120b` (free tier). This replaces `llama-3.3-70b-versatile`, which Groq shut down on 2026-08-16; the plan never calls it "70B". Gemini `gemini-2.5-flash` is the documented fallback if the Groq free tier is unavailable; results record which provider/model produced them.
- No model ID may be hardcoded inline: prompts/scripts read the provider config, and a retired model must fail loudly (`404` / `model_not_found`), never degrade to empty predictions.
- Split by normalized tool-schema hash, never by row; no schema may appear in more than one split. Normalization uses tool names + parameter paths/types only (descriptions, defaults, and examples are ignored).
- Dedupe near-identical examples (same normalized user prompt + same target calls) before splitting, and record the residual limitation in the README and model card.
- Training runs on Colab/Kaggle free GPU with fp16 on T4. Local tests must run on CPU without downloading model weights or API keys.
- No fabricated results: every README/model-card number comes from a committed results JSON, which records token usage and a cost basis.
- Artifact policy: raw data, processed data, adapters, checkpoints, and training logs are not committed; only the dataset pin, manifests, scripts, results JSON, README, and model card are. `.gitignore` is created with the project in Task 1.
- Commit after every task.

## Review Focus

- **Same tool schema in train and test:** normalized split must fail loudly, and an independent leak test must catch any overlap.
- **Near-duplicate prompt in two schemas:** dedupe is applied before splitting; residual risk is documented rather than hidden.
- **Model emits multiple tool calls or nested arguments:** parser and metrics handle both; no crash.
- **Model emits malformed JSON or plain text:** counted as invalid, not a runner crash.
- **Adapter path missing or wrong base model:** inference raises a clear error.
- **API returns prose instead of a tool call:** parsed as invalid and recorded, not retried forever.
- **Retired/unknown API model:** the predictor raises on `model_not_found`/`404`; it never swallows every exception into `([], "")`.
- **T4 is fp16-only:** training config sets `fp16=True` and never requests bf16; eval uses `eval_strategy="steps"` so the validation split is actually scored.
- **Colab OOM:** max sequence length + gradient accumulation + 4-bit loading keep the 1.5B model inside a T4; `resume_from_checkpoint` survives a disconnected session.
- **Chat template adds special tokens:** prompt/target roundtrip test catches it before training.

---

### Task 1: Scaffold and data ingestion

**Files:**
- Create: `projects/02-finetune-function-calling/pyproject.toml`
- Create: `projects/02-finetune-function-calling/.python-version`
- Create: `projects/02-finetune-function-calling/.gitignore`
- Create: `projects/02-finetune-function-calling/src/__init__.py`
- Create: `projects/02-finetune-function-calling/src/data.py`
- Create: `projects/02-finetune-function-calling/tests/fixtures/hermes_sample.json`
- Create: `projects/02-finetune-function-calling/tests/test_data.py`

**Interfaces:**
- Produces: `Example(id: str, tools: list[dict], user: str, calls: list[dict], source: str)`, `extract_tagged_json(text: str, tag: str) -> list[dict]`, `parse_hermes_row(row: dict, source: str) -> Example | None`, `load_examples(paths: list[Path]) -> list[Example]`, `dedupe(examples) -> list[Example]`, `dedupe_near(examples) -> list[Example]`, `schema_hash(tools) -> str`, `find_schema_leaks(splits) -> dict[str, set[str]]`, `assert_split_integrity(examples, splits) -> dict[str, int]`, `split_examples(examples, seed=42, ratios=(0.8, 0.1, 0.1)) -> dict[str, list[Example]]`, `write_jsonl(examples, path)`.
- `extract_tagged_json` uses `json.JSONDecoder.raw_decode`, so nested braces inside `<tool_call>` are safe.
- `parse_hermes_row` reads the top-level `tools` column first (a JSON string in the real dataset) and falls back to the `<tools>` tag in the system message. It skips clarification turns and uses the first assistant turn that actually contains a valid tool call, pairing it with the human turn immediately before it; rows with no such call are skipped.
- `schema_hash` normalizes each tool to its function name + sorted parameter paths/types. Descriptions, defaults, examples, required lists, and enum ordering do not affect the hash.
- `load_examples` applies exact dedupe and near-duplicate dedupe (same normalized tool schema + normalized user prompt + same target calls) before returning.
- Split groups by schema hash; `assert_split_integrity` re-derives counts and schema ownership from the examples and raises `ValueError` on empty splits, a count mismatch, or a cross-split schema. `find_schema_leaks` is exposed so an independent test can cross-check it.
- `.gitignore` excludes `data/raw/`, `data/processed/`, `outputs/`, `artifacts/`, and `**/checkpoint-*/` (artifact policy lives here, not in the root `.gitignore`).

> **Post-execution reconciliation (2026-10-30, commit `62ab3b9`):** the first real-data run found that multi-turn rows often start with a clarification turn and that global near-dedupe was collapsing valid examples whose only difference was the available tool set. The committed `src/data.py` scans to the first actual tool-call turn and keys near-dedupe on the schema as well; `tests/test_data.py` has two extra tests (17), so Task 1+2 is **21 tests**. The embedded Task 1 code/test blocks below are the pre-execution draft; the branch is authoritative. Real run: 7,102 raw → 5,100 parsed → 4,080/509/510 kept.

- [ ] **Step 1: Create the fixture**

The real dataset rows have a top-level `tools` column that is a JSON string; the system message repeats the schemas inside `<tools>` tags. The fixture mirrors that shape, plus one row that only has the tag form.

```json
[
  {
    "id": "row-1",
    "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"get_stock_price\",\"description\":\"Get a stock price\",\"parameters\":{\"type\":\"object\",\"properties\":{\"company\":{\"type\":\"string\"}},\"required\":[\"company\"]}}}]",
    "conversations": [
      {"from": "system", "value": "You are a function calling AI model. <tools>[{\"type\":\"function\",\"function\":{\"name\":\"get_stock_price\",\"parameters\":{\"type\":\"object\",\"properties\":{\"company\":{\"type\":\"string\"}}}}}]</tools>"},
      {"from": "human", "value": "What is Apple's stock price?"},
      {"from": "gpt", "value": "<tool_call>\n{\"name\": \"get_stock_price\", \"arguments\": {\"company\": \"Apple\"}}\n</tool_call>"}
    ]
  },
  {
    "id": "row-2",
    "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"get_news\",\"parameters\":{\"type\":\"object\",\"properties\":{\"country\":{\"type\":\"string\"}}}}}]",
    "conversations": [
      {"from": "system", "value": "You are a function calling AI model."},
      {"from": "human", "value": "Tell me the news."},
      {"from": "gpt", "value": "I cannot do that."}
    ]
  },
  {
    "id": "row-3",
    "conversations": [
      {"from": "system", "value": "You are a function calling AI model. <tools>[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}]</tools>"},
      {"from": "human", "value": "What is the weather in Paris?"},
      {"from": "gpt", "value": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"}
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
    Example,
    assert_split_integrity,
    dedupe,
    dedupe_near,
    extract_tagged_json,
    find_schema_leaks,
    load_examples,
    parse_hermes_row,
    schema_hash,
    split_examples,
)


def _examples(n: int = 6) -> list[Example]:
    return [
        Example(
            str(i),
            [{"type": "function", "function": {"name": f"f{i}", "parameters": {}}}],
            "u",
            [{"name": f"f{i}", "arguments": {}}],
            "t",
        )
        for i in range(n)
    ]


def test_extract_tagged_json_handles_nested_braces():
    text = '<tool_call>{"name": "f", "arguments": {"a": {"b": 1}}}</tool_call>'
    assert extract_tagged_json(text, "tool_call") == [{"name": "f", "arguments": {"a": {"b": 1}}}]


def test_extract_tagged_json_skips_malformed():
    text = '<tool_call>{not json}</tool_call><tool_call>{"name": "ok", "arguments": {}}</tool_call>'
    assert extract_tagged_json(text, "tool_call") == [{"name": "ok", "arguments": {}}]


def test_extract_tools_keeps_the_array_intact():
    text = '<tools>[{"name": "a"}, {"name": "b"}]</tools>'
    assert extract_tagged_json(text, "tools") == [[{"name": "a"}, {"name": "b"}]]


def test_parse_row_prefers_the_top_level_tools_column():
    row = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())[0]
    example = parse_hermes_row(row, "singleturn")
    assert example is not None
    assert example.tools[0]["function"]["name"] == "get_stock_price"
    # The top-level column carries the description; the in-message tag copy does not.
    assert example.tools[0]["function"]["description"] == "Get a stock price"
    assert example.calls[0]["name"] == "get_stock_price"
    assert example.source == "singleturn"


def test_parse_row_falls_back_to_the_system_tag():
    rows = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())
    example = parse_hermes_row(rows[2], "singleturn")
    assert example is not None
    assert example.tools[0]["function"]["name"] == "get_weather"


def test_parse_row_skips_no_tool_call():
    row = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())[1]
    assert parse_hermes_row(row, "singleturn") is None


def test_schema_hash_is_order_insensitive():
    a = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    b = [{"function": {"parameters": {}, "name": "x"}, "type": "function"}]
    assert schema_hash(a) == schema_hash(b)


def test_schema_hash_ignores_descriptions_defaults_and_examples():
    a = [{"type": "function", "function": {"name": "x", "description": "one", "parameters": {
        "type": "object", "properties": {"q": {"type": "string", "description": "query", "default": "y"}},
    }}}]
    b = [{"type": "function", "function": {"name": "x", "description": "two", "parameters": {
        "type": "object", "properties": {"q": {"type": "string", "description": "different", "examples": ["z"]}},
    }}}]
    assert schema_hash(a) == schema_hash(b)


def test_schema_hash_changes_with_parameter_paths():
    a = [{"type": "function", "function": {"name": "x", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}},
    }}}]
    b = [{"type": "function", "function": {"name": "x", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
    }}}]
    assert schema_hash(a) != schema_hash(b)


def _independent_signature(tools: list[dict]) -> tuple:
    """Re-derive schema identity without calling src.data.schema_hash."""
    signature = []
    for tool in tools:
        function = tool.get("function", tool)
        paths: list[str] = []

        def walk(schema: dict, prefix: str = "") -> None:
            for name, prop in (schema.get("properties") or {}).items():
                path = f"{prefix}.{name}" if prefix else name
                paths.append(f"{path}:{prop.get('type')}")
                walk(prop, path)

        walk(function.get("parameters") or {})
        signature.append((function["name"], tuple(sorted(paths))))
    return tuple(sorted(signature))


def test_split_leak_check_is_independent_of_the_splitter():
    splits = split_examples(_examples(), seed=0, ratios=(0.5, 0.25, 0.25))
    owners: dict[tuple, set[str]] = {}
    for name, items in splits.items():
        for example in items:
            owners.setdefault(_independent_signature(example.tools), set()).add(name)
    assert all(len(names) == 1 for names in owners.values())
    assert find_schema_leaks(splits) == {}


def test_split_counts_cover_every_example():
    examples = _examples()
    splits = split_examples(examples, seed=0, ratios=(0.5, 0.25, 0.25))
    counts = assert_split_integrity(examples, splits)
    assert set(counts) == {"train", "val", "test"}
    assert sum(counts.values()) == len(examples)


def test_split_integrity_rejects_dropped_rows():
    examples = _examples()
    splits = split_examples(examples, seed=0, ratios=(0.5, 0.25, 0.25))
    splits["test"] = splits["test"][:-1]
    with pytest.raises(ValueError, match="do not cover"):
        assert_split_integrity(examples, splits)


def test_dedupe_removes_identical_examples():
    row = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())[0]
    example = parse_hermes_row(row, "singleturn")
    assert len(dedupe([example, example])) == 1


def test_dedupe_near_normalizes_prompt_variants():
    base = Example("1", [{"type": "function", "function": {"name": "f"}}], "What is Apple's stock price?",
                   [{"name": "f", "arguments": {"company": "Apple"}}], "t")
    variant = Example("2", base.tools, "  what is apple's   stock price ", base.calls, "t")
    other = Example("3", base.tools, base.user, [{"name": "f", "arguments": {"company": "MSFT"}}], "t")
    assert len(dedupe_near([base, variant])) == 1
    assert len(dedupe_near([base, other])) == 2


def test_load_examples_dedupes_across_files(tmp_path):
    rows = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())
    first = tmp_path / "func-calling-singleturn.json"
    second = tmp_path / "glaive-function-calling-5k.json"
    first.write_text(json.dumps(rows[:1]))
    second.write_text(json.dumps(rows[:1]))
    assert len(load_examples([first, second])) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd projects/02-finetune-function-calling && uv run --with pytest pytest tests/test_data.py -q`
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

# CI and local tests never need CUDA wheels: resolve torch from the CPU index.
# Training on Colab uses requirements-train.txt with pip, so CUDA torch stays there.
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

```text
# .python-version
3.12
```

```text
# .gitignore
# Rebuilt by scripts; never commit data or model artifacts.
data/raw/
data/processed/
outputs/
artifacts/
**/checkpoint-*/
optimizer.pt
scheduler.pt
trainer_state.json
train_log.csv
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
import re
from dataclasses import dataclass
from pathlib import Path

DATASET_REPO = "NousResearch/hermes-function-calling-v1"
DATASET_REVISION = "dae3e1d28cfbcf4b915c04ea1e072030529b4bda"
DATASET_FILES = {
    "func-calling-singleturn.json": {
        "rows": 1_893,
        "bytes": 17_793_686,
        "sha256": "fb2df9fe3f295dfd65b223c2dbf04d6b90006357d737f8f8927749f0bea09527",
    },
    "glaive-function-calling-5k.json": {
        "rows": 5_209,
        "bytes": 20_416_419,
        "sha256": "b98eb3f160359f27ad15018e974ce6db444f566eb5be4aa9e4aa690b34d50832",
    },
}


def dataset_url(filename: str, revision: str = DATASET_REVISION) -> str:
    """Immutable file URL: the revision SHA, never `main`."""
    return (
        f"https://huggingface.co/datasets/{DATASET_REPO}"
        f"/resolve/{revision}/{filename}"
    )


def verify_checksum(path: Path, filename: str) -> str:
    expected = DATASET_FILES[filename]["sha256"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"{filename}: sha256 {digest} != pinned {expected}; "
            f"re-download from revision {DATASET_REVISION}"
        )
    return digest


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


def _parse_tools_column(raw) -> list[dict] | None:
    """The dataset stores tool schemas as a JSON string in a top-level column."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        parsed = raw
    if isinstance(parsed, list) and parsed and all(isinstance(tool, dict) for tool in parsed):
        return parsed
    return None


def parse_hermes_row(row: dict, source: str) -> Example | None:
    conversations = row.get("conversations", [])
    system = _first(conversations, "system")
    user = _first(conversations, "human")
    if not system or not user:
        return None

    tool_list = _parse_tools_column(row.get("tools"))
    if not tool_list:
        # Fallback for rows that only carry schemas inside the system message.
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
    return dedupe_near(dedupe(examples))


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


def _normalize_prompt(user: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", user.lower()).split())


def _calls_key(calls: list[dict]) -> tuple:
    return tuple(
        sorted((call["name"], json.dumps(call["arguments"], sort_keys=True)) for call in calls)
    )


def dedupe_near(examples: list[Example]) -> list[Example]:
    """Collapse examples with the same normalized user prompt and target calls."""
    seen: set[str] = set()
    result: list[Example] = []
    for example in examples:
        key = hashlib.sha256(
            json.dumps(
                {"user": _normalize_prompt(example.user), "calls": _calls_key(example.calls)},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            result.append(example)
    return result


def _parameter_paths(schema: dict, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if not isinstance(schema, dict):
        return paths
    for name, prop in sorted((schema.get("properties") or {}).items()):
        path = f"{prefix}.{name}" if prefix else name
        if not isinstance(prop, dict):
            paths.append(f"{path}:unknown")
            continue
        paths.append(f"{path}:{prop.get('type', 'unknown')}")
        paths.extend(_parameter_paths(prop, path))
        items = prop.get("items")
        if isinstance(items, dict) and items.get("type") == "object":
            paths.extend(_parameter_paths(items, f"{path}[]"))
    return paths


def _canonical_tool(tool: dict) -> tuple[str, tuple[str, ...]]:
    function = tool.get("function", tool)
    return (str(function.get("name", "")), tuple(_parameter_paths(function.get("parameters") or {})))


def schema_hash(tools: list[dict]) -> str:
    """Hash tool names + parameter paths only; descriptions/defaults are ignored."""
    canonical = sorted(_canonical_tool(tool) for tool in tools)
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False).encode()).hexdigest()


def find_schema_leaks(splits: dict[str, list[Example]]) -> dict[str, set[str]]:
    owners: dict[str, set[str]] = {}
    for name, items in splits.items():
        for example in items:
            owners.setdefault(schema_hash(example.tools), set()).add(name)
    return {digest: names for digest, names in owners.items() if len(names) > 1}


def assert_split_integrity(examples: list[Example], splits: dict[str, list[Example]]) -> dict[str, int]:
    counts = {name: len(items) for name, items in splits.items()}
    if set(counts) != {"train", "val", "test"}:
        raise ValueError(f"unexpected split names: {sorted(counts)}")
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"each split must be non-empty: {counts}")
    if sum(counts.values()) != len(examples):
        raise ValueError(f"split counts {counts} do not cover {len(examples)} examples")
    leaks = find_schema_leaks(splits)
    if leaks:
        digest, names = next(iter(leaks.items()))
        raise ValueError(f"schema {digest[:8]} leaked into {sorted(names)}")
    return counts


def split_examples(
    examples: list[Example],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[Example]]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    if not examples:
        raise ValueError("no examples to split")
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

    assert_split_integrity(examples, splits)
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
uv sync
uv run pytest tests/test_data.py -q
```
Expected: `17 passed`.

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
- `prepare_data.py` downloads both raw files from the pinned revision, verifies sha256 + row count, dedupes, asserts split integrity, filters examples over `max_seq_length`, writes `data/processed/{train,val,test}.jsonl`, and writes `data/processed/manifest.json` with the revision, checksums, row counts, dedupe drops, and split counts. It refuses to write an empty split or an empty dataset.
- Residual limitation: `dedupe_near` only catches case/whitespace/punctuation variants of the same prompt + target calls. Paraphrases and semantically equivalent prompts are not caught; because splits are schema-based, a paraphrase paired with a differently-worded schema can still cross splits. Say this in the README and model card instead of claiming full contamination control.

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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from transformers import AutoTokenizer

from src.data import (
    DATASET_FILES,
    DATASET_REPO,
    DATASET_REVISION,
    assert_split_integrity,
    dataset_url,
    load_examples,
    split_examples,
    verify_checksum,
    write_jsonl,
)
from src.prompts import render_example

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def download(filename: str) -> Path:
    dest = RAW / filename
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", dataset_url(filename), follow_redirects=True, timeout=180) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    return dest


def verify(path: Path, filename: str) -> dict:
    expected = DATASET_FILES[filename]
    digest = verify_checksum(path, filename)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected["rows"]:
        raise ValueError(f"{filename}: expected {expected['rows']} rows, got {len(rows)}")
    return {"sha256": digest, "bytes": path.stat().st_size, "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    files = [download(filename) for filename in DATASET_FILES]
    manifest = {"repo": DATASET_REPO, "revision": DATASET_REVISION, "files": {}}
    for path in files:
        manifest["files"][path.name] = verify(path, path.name)

    raw_rows = sum(item["rows"] for item in manifest["files"].values())
    examples = load_examples(files)
    if not examples:
        raise ValueError("no examples parsed; refusing to write an empty dataset")
    splits = split_examples(examples)
    split_counts = assert_split_integrity(examples, splits)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kept_counts: dict[str, int] = {}
    for name, items in splits.items():
        kept = [
            example
            for example in items
            if len(tokenizer(render_example(example, tokenizer))["input_ids"]) <= args.max_seq_length
        ]
        if not kept:
            raise ValueError(f"{name} split is empty after length filtering; raise --max-seq-length")
        write_jsonl(kept, PROCESSED / f"{name}.jsonl")
        kept_counts[name] = len(kept)

    manifest.update(
        raw_rows=raw_rows,
        parsed_examples=len(examples),
        skipped_or_duplicate=raw_rows - len(examples),
        split_counts=split_counts,
        length_filtered=kept_counts,
        seed=42,
        ratios=[0.8, 0.1, 0.1],
        max_seq_length=args.max_seq_length,
    )
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"raw_rows": raw_rows, "parsed": len(examples), "split": split_counts, "kept": kept_counts}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_prompts.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Prepare the data**

Run: `uv run python scripts/prepare_data.py`
Expected: downloads both pinned files, verifies sha256 and row counts, prints `raw_rows`/`parsed`/`split`/`kept` counts, and writes three JSONL files plus `data/processed/manifest.json`. Only the tokenizer is downloaded (no model weights). `data/raw/` and `data/processed/` are gitignored.

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
- Produces: `load_config(path: str | Path) -> dict`, `main()`; adapter written to `output_dir`; training history written to `train_log.csv`.
- Config keys: `model`, `max_seq_length`, `epochs`, `learning_rate`, `batch_size`, `grad_accum`, `lora_r`, `lora_alpha`, `seed`, `output_dir`.
- SFTConfig uses `fp16=True` / `bf16=False` (T4 is fp16-only) and `eval_strategy="steps"` so the validation split is scored every 200 steps.
- `--resume-from-checkpoint <dir|true>` resumes a disconnected Colab session; checkpoints are gitignored.

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
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="checkpoint dir (or 'true' for the latest); survives a Colab disconnect",
    )
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
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
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

On Colab T4, run cells 2–4. Expected: data prep completes; training runs 2 steps in fp16 and saves an adapter; the val split is evaluated; no OOM. If OOM, lower `batch_size` to 1 and raise `grad_accum` to 16. If the session disconnects during the full run, rerun with `--resume-from-checkpoint true`.

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
- Produces: `score_example(predicted: list[dict], gold: list[dict]) -> dict` with `tool_name_accuracy`, `arg_precision`, `arg_recall`, `arg_f1`, `full_exact`, `json_valid`; `aggregate(rows) -> dict`; `estimate_cost(usage, pricing=PRICING) -> float`; `run_eval(examples, predict_fn) -> dict`.
- Predictor contract: `predict_fn(row) -> (calls, raw, usage)` where `usage` is `{"provider", "model", "prompt_tokens", "completion_tokens"}` or `None` for a local predictor with no per-token spend.
- Metric definitions:
  - `tool_name_accuracy`: 1.0 iff the set of predicted tool names equals the set of gold tool names.
  - `arg_precision/recall/f1`: micro-averaged over flattened argument fields of calls matched by name.
  - `full_exact`: canonical JSON of predicted calls equals gold calls (order-insensitive).
  - `json_valid`: 1.0 iff at least one valid tool call was parsed.
- `run_eval` records per-example latency + usage and writes aggregate metrics, token totals, `p95_latency_ms`, and `cost` (`usd`, `per_1k_calls_usd`, `basis`, `pricing` snapshot). `PRICING` documents the free-tier token rates ($0 for Groq `openai/gpt-oss-120b` and Gemini 2.5 Flash; local inference has no per-token spend); a paid-tier run must replace the table with the provider's current price card and keep the source link.

- [ ] **Step 1: Write the failing metric tests**

```python
# tests/test_metrics.py
import pytest

from src.metrics import aggregate, estimate_cost, score_example


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


def test_estimate_cost_uses_the_price_table():
    usage = {"model": "x", "prompt_tokens": 1_000_000, "completion_tokens": 2_000_000}
    pricing = {
        "x": {"input_usd_per_1m": 1.0, "output_usd_per_1m": 2.0},
        "local": {"input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0},
    }
    assert estimate_cost(usage, pricing) == pytest.approx(5.0)
    assert estimate_cost(None) == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`.

- [ ] **Step 3: Implement metrics**

```python
# src/metrics.py
from __future__ import annotations

import json

# USD per 1M tokens. Free tier only: Groq openai/gpt-oss-120b and Gemini 2.5 Flash
# have $0 token charges, and local inference has no per-token spend. If a paid tier
# is used, replace this table with the provider's current price card + link and re-run.
PRICING = {
    "openai/gpt-oss-120b": {"input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0},
    "gemini-2.5-flash": {"input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0},
    "local": {"input_usd_per_1m": 0.0, "output_usd_per_1m": 0.0},
}
COST_BASIS = (
    "free tier: $0 token charges on Groq openai/gpt-oss-120b and Gemini 2.5 Flash; "
    "local inference has no per-token spend (runs on a free Colab T4)"
)


def estimate_cost(usage: dict | None, pricing: dict[str, dict[str, float]] = PRICING) -> float:
    """USD for one call, from recorded token usage. Never a hardcoded per-call cost."""
    if not usage:
        return 0.0
    rate = pricing.get(usage.get("model", "local"), pricing["local"])
    return (
        usage.get("prompt_tokens", 0) * rate["input_usd_per_1m"]
        + usage.get("completion_tokens", 0) * rate["output_usd_per_1m"]
    ) / 1_000_000


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

from src.metrics import COST_BASIS, PRICING, aggregate, estimate_cost, score_example


def run_eval(examples: list[dict], predict_fn) -> dict:
    rows = []
    started = time.monotonic()
    for example in examples:
        t0 = time.monotonic()
        calls, raw, usage = predict_fn(example)
        latency_ms = int((time.monotonic() - t0) * 1000)
        row = score_example(calls, example["calls"])
        row.update({"id": example["id"], "latency_ms": latency_ms, "raw": raw, "predicted": calls, "usage": usage})
        rows.append(row)
    latencies = sorted(row["latency_ms"] for row in rows)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0
    prompt_tokens = sum((row["usage"] or {}).get("prompt_tokens", 0) for row in rows)
    completion_tokens = sum((row["usage"] or {}).get("completion_tokens", 0) for row in rows)
    total_cost = sum(estimate_cost(row["usage"]) for row in rows)
    models = sorted({(row["usage"] or {}).get("model") for row in rows if (row["usage"] or {}).get("model")})
    providers = sorted({(row["usage"] or {}).get("provider") for row in rows if (row["usage"] or {}).get("provider")})
    return {
        "examples": len(rows),
        "model": models[0] if len(models) == 1 else None,
        "provider": providers[0] if len(providers) == 1 else None,
        "metrics": aggregate(rows),
        "p95_latency_ms": p95,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "cost": {
            "usd": round(total_cost, 6),
            "per_1k_calls_usd": round(total_cost / len(rows) * 1000, 6) if rows else 0.0,
            "basis": COST_BASIS,
            "pricing": PRICING,
        },
        "wall_seconds": int(time.monotonic() - started),
        "rows": rows,
    }


def load_test(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=["base", "tuned", "api"], required=True)
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq")
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

        predict_fn = make_api_predictor(provider=args.provider)
    else:
        from src.infer import generate_calls, load_model
        from src.data import Example

        adapter = args.adapter if args.predictor == "tuned" else None
        model, tokenizer = load_model(args.model, adapter)

        def predict_fn(example):
            calls, raw = generate_calls(
                model, tokenizer,
                Example(example["id"], example["tools"], example["user"], example["calls"], example["source"]),
            )
            # No per-token spend on local hardware; tokens stay null rather than invented.
            return calls, raw, {"provider": "local", "model": "local", "prompt_tokens": 0, "completion_tokens": 0}

    result = run_eval(examples, predict_fn)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))
    print(f"p95={result['p95_latency_ms']}ms tokens={result['tokens']['total']} "
          f"cost/1k=${result['cost']['per_1k_calls_usd']} wrote {output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the runner test**

```python
# tests/test_eval_runner.py
from evals.run_eval import run_eval


def test_runner_records_metrics_latency_and_cost():
    examples = [{"id": "1", "tools": [], "user": "u", "calls": [{"name": "f", "arguments": {"x": 1}}], "source": "t"}]
    usage = {"provider": "groq", "model": "openai/gpt-oss-120b", "prompt_tokens": 10, "completion_tokens": 5}
    result = run_eval(examples, lambda example: ([{"name": "f", "arguments": {"x": 1}}], "raw", usage))
    assert result["examples"] == 1
    assert result["metrics"]["full_exact"] == 1.0
    assert result["p95_latency_ms"] >= 0
    assert result["tokens"]["total"] == 15
    assert result["cost"]["per_1k_calls_usd"] == 0.0
    assert result["model"] == "openai/gpt-oss-120b"
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_metrics.py tests/test_eval_runner.py -q`
Expected: `8 passed`.

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
- Produces: `PROVIDERS` (`groq` -> `openai/gpt-oss-120b`, `gemini` -> `gemini-2.5-flash`), `make_api_predictor(provider: str = "groq", model: str | None = None, client=None) -> Callable[[dict], tuple[list[dict], str, dict]]`.
- Reads the provider config + `GROQ_API_KEY` (or `GEMINI_API_KEY`); no model ID is hardcoded at call sites.
- Returns `(calls, raw, usage)` with `usage = {"provider", "model", "prompt_tokens", "completion_tokens"}` from the API response.
- Retries 3 times with exponential backoff only for rate limits, timeouts, and 5xx. A `404` / `model_not_found` raises `RuntimeError` immediately — the predictor never swallows every exception into `([], "")`, and an exhausted retryable error also raises rather than fabricating empty rows.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_baseline.py
import pytest

from src.api_baseline import make_api_predictor


def test_api_predictor_parses_tool_calls_and_usage():
    class Usage:
        prompt_tokens = 12
        completion_tokens = 7

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
                        usage = Usage()
                    return Response()

    predict = make_api_predictor(client=FakeClient())
    calls, raw, usage = predict(
        {
            "id": "1",
            "user": "u",
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "calls": [{"name": "f", "arguments": {"x": 1}}],
        }
    )
    assert calls == [{"name": "f", "arguments": {"x": 1}}]
    assert "tool_call" in raw
    assert usage == {"provider": "groq", "model": "openai/gpt-oss-120b", "prompt_tokens": 12, "completion_tokens": 7}


def test_retired_model_fails_loudly():
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("model_not_found: llama-3.3-70b-versatile was retired")

    predict = make_api_predictor(client=FakeClient())
    with pytest.raises(RuntimeError, match="unavailable|model_not_found"):
        predict({"id": "1", "user": "u", "tools": [], "calls": []})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_api_baseline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api_baseline'`.

- [ ] **Step 3: Implement the API baseline**

```python
# src/api_baseline.py
from __future__ import annotations

import os
import time

from src.data import Example
from src.prompts import build_messages, parse_tool_calls

PROVIDERS = {
    "groq": {
        "model": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
}
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _status(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def _is_model_not_found(exc: Exception) -> bool:
    message = str(exc).lower()
    return _status(exc) == 404 or "model_not_found" in message or "does not exist" in message


def _is_retryable(exc: Exception) -> bool:
    return _status(exc) in RETRYABLE_STATUS


def make_api_predictor(provider: str = "groq", model: str | None = None, client=None):
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose one of {sorted(PROVIDERS)}")
    config = PROVIDERS[provider]
    model = model or config["model"]
    if client is None:
        from openai import OpenAI

        key = os.getenv(config["key_env"])
        if not key:
            raise ValueError(f"{config['key_env']} is required for the {provider} API baseline")
        client = OpenAI(base_url=config["base_url"], api_key=key)

    def predict(row: dict) -> tuple[list[dict], str, dict]:
        example = Example(row["id"], row["tools"], row["user"], row["calls"], row.get("source", "api"))
        messages = build_messages(example)
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )
                text = response.choices[0].message.content or ""
                usage = {
                    "provider": provider,
                    "model": model,
                    "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0)),
                    "completion_tokens": int(getattr(response.usage, "completion_tokens", 0)),
                }
                return parse_tool_calls(text), text, usage
            except Exception as exc:
                # A retired/unknown model will never recover: fail loudly instead of
                # turning every row into an empty prediction.
                if _is_model_not_found(exc):
                    raise RuntimeError(
                        f"{provider}/{model} is unavailable (404/model_not_found); "
                        "check PROVIDERS against the provider's current model list"
                    ) from exc
                if not _is_retryable(exc):
                    raise
                if attempt == 2:
                    raise RuntimeError(f"{provider}/{model} failed after 3 attempts") from exc
                time.sleep(2 ** attempt)
        raise RuntimeError("retry loop exhausted without a result")

    return predict
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_api_baseline.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): prompted free-tier API baseline (openai/gpt-oss-120b)"
```

---

### Task 7: Forgetting check

**Files:**
- Create: `projects/02-finetune-function-calling/evals/forgetting.py`
- Create: `projects/02-finetune-function-calling/tests/test_forgetting.py`

**Interfaces:**
- Produces: `parse_mmlu_answer(text: str) -> str | None`, `select_mmlu(rows, n=200, seed=42) -> list[dict]`, `wilson_interval(correct, n, z=1.96) -> tuple[float, float]`, `score_mmlu(model, tokenizer, rows) -> tuple[float, int]`.
- Uses `cais/mmlu` test split with the model's chat template (same prompt path as inference, not raw text), greedy decode, and an anchored letter regex.
- The parse is anchored at the start of the response: it accepts `B` / `The answer is B.` and rejects `I think C might be right`, so a rambling response cannot be scored by the first letter it mentions.
- Report base vs tuned accuracy, the delta, and 95% Wilson confidence intervals. This is a coarse forgetting signal on a 200-row subsample, not a full MMLU benchmark.
- MMLU downloads from the HF Hub, so this is a documented manual command, not part of CI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forgetting.py
from evals.forgetting import parse_mmlu_answer, select_mmlu, wilson_interval


def test_parse_answer_is_anchored_and_case_insensitive():
    assert parse_mmlu_answer("B") == "B"
    assert parse_mmlu_answer("The answer is B.") == "B"
    assert parse_mmlu_answer("c.") == "C"
    assert parse_mmlu_answer("I don't know") is None
    assert parse_mmlu_answer("I think C might be right") is None


def test_wilson_interval_stays_in_bounds():
    for correct, n in [(0, 10), (5, 10), (10, 10)]:
        low, high = wilson_interval(correct, n)
        assert 0.0 <= low <= high <= 1.0


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

import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LETTERS = "ABCD"
# Anchored at the start: a response that first mentions "C" in prose must not be
# scored as C. Accepts "B", "B.", and "The answer is B." only.
ANCHORED_ANSWER = re.compile(r"^\s*(?:the answer is\s*)?([ABCD])(?![A-Za-z])", re.IGNORECASE)


def parse_mmlu_answer(text: str) -> str | None:
    match = ANCHORED_ANSWER.match(text)
    return match.group(1).upper() if match else None


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = correct / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def select_mmlu(rows: list[dict], n: int = 200, seed: int = 42) -> list[dict]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _messages(row: dict) -> list[dict[str, str]]:
    choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"]))
    content = f"Answer with a single letter A, B, C, or D.\n\nQuestion: {row['question']}\n{choices}"
    return [{"role": "user", "content": content}]


def score_mmlu(model, tokenizer, rows: list[dict]) -> tuple[float, int]:
    import torch

    correct = 0
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            _messages(row), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if parse_mmlu_answer(text) == LETTERS[row["answer"]]:
            correct += 1
    accuracy = correct / len(rows) if rows else 0.0
    return accuracy, correct


def main() -> None:
    import json
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
    base_score, base_correct = score_mmlu(base, tokenizer, rows)
    del base
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    tuned, tokenizer = load_model(args.base_model, args.adapter)
    tuned_score, tuned_correct = score_mmlu(tuned, tokenizer, rows)
    base_ci = wilson_interval(base_correct, len(rows))
    tuned_ci = wilson_interval(tuned_correct, len(rows))
    result = {
        "n": len(rows),
        "base_accuracy": base_score,
        "tuned_accuracy": tuned_score,
        "delta": tuned_score - base_score,
        "base_ci95": list(base_ci),
        "tuned_ci95": list(tuned_ci),
        "limitation": "200-row subsample of cais/mmlu test with 8-token greedy decoding; a coarse forgetting signal, not a full MMLU score.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_forgetting.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "feat(finetune): catastrophic-forgetting check with chat-template MMLU"
```

Note: the MMLU download needs the HF Hub, so this eval is a manual pre-release command. CI runs the parsing/interval unit tests only.

---

### Task 8: Run the comparison and record results

**Files:**
- Create: `projects/02-finetune-function-calling/results/base.json` (generated)
- Create: `projects/02-finetune-function-calling/results/tuned.json` (generated)
- Create: `projects/02-finetune-function-calling/results/api_gptoss.json` (generated)
- Create: `projects/02-finetune-function-calling/results/forgetting.json` (generated)
- Create: `projects/02-finetune-function-calling/scripts/run_comparison.sh`

**Interfaces:**
- Consumes: trained adapter, test JSONL, `GROQ_API_KEY`.
- Produces: four committed results files used by the README; no hand-edited numbers. Each eval file records `examples`, `metrics`, `p95_latency_ms`, `tokens`, and `cost` (`usd`, `per_1k_calls_usd`, `basis`).

- [ ] **Step 1: Write the comparison script**

```bash
# scripts/run_comparison.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${ADAPTER:-outputs/qwen2.5-1.5b-lora}"
PROVIDER="${PROVIDER:-groq}"

uv run python evals/run_eval.py --predictor base --output results/base.json
uv run python evals/run_eval.py --predictor tuned --adapter "$ADAPTER" --output results/tuned.json
uv run python evals/run_eval.py --predictor api --provider "$PROVIDER" --output results/api_gptoss.json
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
Expected: four JSON files in `results/`; each eval has `examples`, `metrics`, `p95_latency_ms`, `tokens`, and `cost` (forgetting has base/tuned accuracy + CIs). If Groq's free tier is unavailable, run `PROVIDER=gemini bash scripts/run_comparison.sh` and label the baseline as Gemini in the README — mixed-provider numbers are not directly comparable.

- [ ] **Step 4: Sanity-check the results**

Run:
```bash
uv run python - <<'PY'
import json
from pathlib import Path

for name in ["base", "tuned", "api_gptoss"]:
    data = json.loads(Path(f"results/{name}.json").read_text())
    assert data["examples"] > 0, f"{name}: no examples evaluated"
    assert data["metrics"]["json_valid"] > 0, f"{name}: parse rate is zero; the predictor failed"
    assert 0.0 <= data["metrics"]["full_exact"] <= 1.0
    assert "per_1k_calls_usd" in data["cost"], f"{name}: missing cost accounting"
    print(name, data["metrics"]["full_exact"], data["p95_latency_ms"], data["tokens"]["total"], data["cost"]["per_1k_calls_usd"])

forgetting = json.loads(Path("results/forgetting.json").read_text())
assert forgetting["n"] > 0
print("forgetting", forgetting["delta"])
PY
```
Expected: three lines with metrics, latency, token totals, and cost/1k; then the forgetting delta. A zero row count or a zero parse rate fails the check.

- [ ] **Step 5: Commit**

```bash
git add projects/02-finetune-function-calling
git commit -m "results(finetune): base, tuned, free-tier API, and forgetting evals"
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
- Produces: `render_model_card(base: dict, tuned: dict, api: dict, forgetting: dict, config: dict) -> str`; upload script; CI tests; README results table.
- The card reads `model`, `lora_r`, `lora_alpha`, `epochs`, and `max_seq_length` from `config.yaml`; it never hardcodes training numbers or metrics. The only placeholders are human-supplied values (`<your-hf-user>/<model-name>` and the clone URL).
- Every table cell comes from `results/*.json`; the card also names the exact commands that regenerate those files.

- [ ] **Step 1: Write the failing model-card test**

```python
# tests/test_model_card.py
from scripts.upload_adapter import render_model_card

BASE = {"metrics": {"full_exact": 0.1, "arg_f1": 0.2}, "p95_latency_ms": 100, "tokens": {"total": 0}, "cost": {"per_1k_calls_usd": 0.0}}
TUNED = {"metrics": {"full_exact": 0.6, "arg_f1": 0.7}, "p95_latency_ms": 120, "tokens": {"total": 0}, "cost": {"per_1k_calls_usd": 0.0}}
API = {"metrics": {"full_exact": 0.7, "arg_f1": 0.8}, "p95_latency_ms": 900, "model": "openai/gpt-oss-120b", "tokens": {"total": 1500}, "cost": {"per_1k_calls_usd": 0.0, "basis": "free tier"}}
FORGET = {"n": 200, "base_accuracy": 0.5, "tuned_accuracy": 0.48, "delta": -0.02}
CONFIG = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "lora_r": 8, "lora_alpha": 16, "epochs": 2, "max_seq_length": 1024, "seed": 42, "output_dir": "outputs/test"}


def test_model_card_uses_results_and_config_not_hardcoded_numbers():
    card = render_model_card(BASE, TUNED, API, FORGET, CONFIG)
    assert "0.600" in card
    assert "0.700" in card
    assert "-0.020" in card
    assert "openai/gpt-oss-120b" in card
    assert "Qwen/Qwen2.5-1.5B-Instruct" in card
    # Training facts come from config.yaml, not the template.
    assert "r=8, alpha=16, epochs=2, max_seq_length=1024" in card
    assert "uv run python scripts/prepare_data.py" in card
    assert "data/processed/manifest.json" in card
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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data import DATASET_REPO, DATASET_REVISION

# Files that actually constitute the adapter; checkpoints/optimizer state are excluded.
ADAPTER_PATTERNS = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def _metric(result: dict, key: str) -> str:
    return f"{result['metrics'][key]:.3f}"


def render_model_card(base: dict, tuned: dict, api: dict, forgetting: dict, config: dict) -> str:
    api_model = api.get("model") or "configured free-tier API model"
    model = config["model"]
    return f"""# {model} — Function-Calling LoRA

Fine-tuned with 4-bit QLoRA on `{DATASET_REPO}` (revision `{DATASET_REVISION}`).

## Results (held-out, schema-disjoint test set)

| model | full-call exact | argument F1 | p95 latency | tokens | cost / 1k calls |
|---|---:|---:|---:|---:|---:|
| base {model} | {_metric(base, 'full_exact')} | {_metric(base, 'arg_f1')} | {base['p95_latency_ms']} ms | {base['tokens']['total']} | ${base['cost']['per_1k_calls_usd']:.4f} |
| + this LoRA adapter | {_metric(tuned, 'full_exact')} | {_metric(tuned, 'arg_f1')} | {tuned['p95_latency_ms']} ms | {tuned['tokens']['total']} | ${tuned['cost']['per_1k_calls_usd']:.4f} |
| prompted free-tier API ({api_model}) | {_metric(api, 'full_exact')} | {_metric(api, 'arg_f1')} | {api['p95_latency_ms']} ms | {api['tokens']['total']} | ${api['cost']['per_1k_calls_usd']:.4f} |

Cost basis: {api['cost']['basis']}.

## Forgetting check

MMLU n={forgetting['n']}: base {forgetting['base_accuracy']:.3f}, tuned {forgetting['tuned_accuracy']:.3f}, delta {forgetting['delta']:+.3f}. 95% Wilson intervals are in `results/forgetting.json`.

## Training

- Dataset: `{DATASET_REPO}` (Apache-2.0), pinned revision `{DATASET_REVISION}`
- Split: normalized tool-schema hash, 80/10/10; deduped; exact counts in `data/processed/manifest.json`
- LoRA r={config['lora_r']}, alpha={config['lora_alpha']}, epochs={config['epochs']}, max_seq_length={config['max_seq_length']}
- Hardware: Colab free T4 (fp16)

## Reproduce

Run from `projects/02-finetune-function-calling/`:

- `uv run python scripts/prepare_data.py`
- `uv run python training/train.py --config config.yaml`
- `bash scripts/run_comparison.sh`
- `uv run python scripts/upload_adapter.py --repo <your-hf-user>/<model-name>`

## Limitations

- Synthetic training data; tool schemas are not production APIs.
- No constrained decoding in the headline result; JSON validity is measured, not guaranteed.
- Test tool schemas are disjoint from training, but the data distribution is still narrow.
- Dedupe only catches case/whitespace/punctuation variants of the same prompt + target calls; paraphrased duplicates are a residual contamination risk.
- MMLU is a 200-row subsample with 8-token greedy decoding; it is a coarse forgetting signal, not a full benchmark.
"""


def main() -> None:
    from argparse import ArgumentParser

    from huggingface_hub import HfApi

    parser = ArgumentParser()
    parser.add_argument("--repo", required=True, help="<hf-user>/<model-name>")
    parser.add_argument("--adapter", default="outputs/qwen2.5-1.5b-lora")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    results = Path("results")
    base = json.loads((results / "base.json").read_text())
    tuned = json.loads((results / "tuned.json").read_text())
    api = json.loads((results / "api_gptoss.json").read_text())
    forgetting = json.loads((results / "forgetting.json").read_text())
    config = yaml.safe_load(Path(args.config).read_text())
    card = render_model_card(base, tuned, api, forgetting, config)
    Path("MODEL_CARD.md").write_text(card)

    hf = HfApi()
    hf.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)
    hf.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=args.adapter,
        path_in_repo=".",
        allow_patterns=ADAPTER_PATTERNS,
        ignore_patterns=["checkpoint-*", "optimizer*", "trainer_state.json"],
        commit_message="Upload function-calling LoRA adapter",
    )
    hf.upload_file(
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

uv sync
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
    env:
      # Unit tests must not touch the HF Hub or an API; there is no key in CI.
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      # torch resolves from the CPU-only pytorch-cpu index pinned in pyproject.toml.
      - run: uv sync
      - run: uv run python -c "import torch; assert not torch.cuda.is_available(), 'CI must not install CUDA torch'"
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
Expected: `create_repo` succeeds even if the repo already exists; only `adapter_config.json`, the adapter weights, and tokenizer files are uploaded (`checkpoint-*`/optimizer state excluded by `allow_patterns`); `MODEL_CARD.md` is uploaded as the repo README. Verify the card renders and every number matches `results/*.json`.

- [ ] **Step 6: Write the README**

Include:
1. One-paragraph pitch and HF Hub link.
2. Results table (from `results/*.json`), including base/tuned/prompted free-tier API (`openai/gpt-oss-120b`), p95 latency, token totals, and cost per 1k calls with the cost basis.
3. Forgetting check result plus the 95% interval and its limitation.
4. Training configuration (read from `config.yaml`) and Colab link.
5. Reproduction commands.
6. Design decisions: normalized schema-disjoint split, dedupe + residual paraphrase risk, why Qwen2.5-1.5B, why LoRA, why fp16 on T4.
7. Limitations: model churn (a retired API model must be replaced, not silently ignored), MMLU subsample, synthetic schemas.
8. Attribution.

- [ ] **Step 7: Write attribution**

```markdown
# Attributions

- `NousResearch/hermes-function-calling-v1`: Apache-2.0, https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- `Qwen/Qwen2.5-1.5B-Instruct`: Apache-2.0, https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- `cais/mmlu`: MIT, https://huggingface.co/datasets/cais/mmlu
- Groq free-tier API (`openai/gpt-oss-120b`) and Google Gemini API (`gemini-2.5-flash`): used as hosted inference only; no weights redistributed, subject to each provider's terms.
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
- Depends on Project 1 being complete: this plan does not build the RAG project, and the root README needs its published artifacts. Blocked inputs are listed below; if any is missing, stop and report the blocker instead of inventing a Project 1 number.
- Consumes:
  - `projects/01-rag-k8s-docs/README.md` and `projects/01-rag-k8s-docs/evals/CERTIFICATION.md`
  - `projects/01-rag-k8s-docs/evals/results/*.json` — the certified test-split eval and ablation results behind Project 1's numbers
  - the deployed demo URL and the HF Spaces link
  - `projects/02-finetune-function-calling/README.md`
  - `projects/02-finetune-function-calling/results/{base,tuned,api_gptoss,forgetting}.json`
  - the HF Hub model-card URL
- Produces: a landing page that lets a reviewer understand both projects in under two minutes.

- [ ] **Step 1: Write the root README**

Include:
1. One-paragraph positioning: applied AI / LLM engineer focused on retrieval, evaluation, and small-model fine-tuning.
2. Two project cards, each with: one-line problem, approach, headline verified metric, stack, links to code/demo/model card. The Project 1 card is conditional on the inputs above; if the RAG project is not published yet, write "in progress" instead of a metric.
3. A combined results table copied from the committed results JSON (`projects/01-rag-k8s-docs/evals/results/` and `projects/02-finetune-function-calling/results/`), never hand-written.
4. A short "how I work" section: evals before claims, pinned artifacts, licensed data, reproducible commands.
5. A "what I'd do next" section with one concrete next step per project.
6. Repository layout.

- [ ] **Step 2: Verify every link and command**

Run:
```bash
cd /home/deansama/projects/ML-Engineer-Projects-for-Employment
grep -oE 'https?://[^ )]+' README.md
```
Expected: every URL resolves (live demo, HF Spaces, HF Hub, GitHub). Every command block is copy-pasteable from the repo root. Cross-check each root-table number against the source results file and record the file path next to the table.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: portfolio root README"
```

---

## Plan Self-Review

- Spec coverage: public pinned data, normalized schema-disjoint split, near-dup dedupe, QLoRA with fp16, base/tuned/prompted free-tier API comparison, cost per 1k calls with token accounting, forgetting check, HF Hub model card, CPU-only CI, project README, and conditional portfolio root README — all mapped to tasks. Constrained decoding is optional in the spec and deliberately excluded from the headline path; add it as a follow-up if time allows.
- Placeholder scan: no TBD/TODO; the only human-supplied values are the repo clone URL, HF repo ID, API keys, and the root README's Project 1 inputs, all explicitly marked as gates.
- Retired-model scan: `llama-3.3-70b-versatile` appears only in the audit memo and in the negative test that asserts a retired model fails loudly; no actionable path uses it.
- Type consistency: `Example` fields are reused everywhere; `parse_tool_calls` returns `list[dict]`; predictors return `(calls, raw, usage)`; `score_example`, `aggregate`, and `estimate_cost` keys match the README/model-card contract.
- Review focus: schema leakage (with an independent test), near-duplicate prompts, multi-call parsing, malformed JSON, missing adapters, API prose, retired models (`404`/`model_not_found`), T4 fp16 memory, resumed training, and chat-template roundtrip — each has a test or a gate in the owning task.
