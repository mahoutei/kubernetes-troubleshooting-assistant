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
    if not system:
        return None

    tool_list = _parse_tools_column(row.get("tools"))
    if not tool_list:
        # Fallback for rows that only carry schemas inside the system message.
        tool_values = extract_tagged_json(system, "tools")
        if not tool_values or not isinstance(tool_values[0], list):
            return None
        tool_list = tool_values[0]

    # Use the first assistant turn that actually calls a tool. Multi-turn rows
    # often start with a clarification question before the tool call.
    user = None
    calls = None
    for message in conversations:
        role = message.get("from")
        if role == "human":
            user = message.get("value")
        elif role == "gpt" and user:
            candidate = [
                {"name": call["name"], "arguments": call["arguments"]}
                for call in iter_calls(extract_tagged_json(message.get("value") or "", "tool_call"))
                if isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict)
            ]
            if candidate:
                calls = candidate
                break
    if not user or not calls:
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
    """Collapse examples with the same tool schema, normalized prompt, and calls.

    The schema is part of the key: the same prompt and target with a different
    set of available tools is a different task, not a duplicate.
    """
    seen: set[str] = set()
    result: list[Example] = []
    for example in examples:
        key = hashlib.sha256(
            json.dumps(
                {
                    "schema": schema_hash(example.tools),
                    "user": _normalize_prompt(example.user),
                    "calls": _calls_key(example.calls),
                },
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
