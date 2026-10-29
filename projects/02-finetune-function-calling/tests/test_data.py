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


def test_parse_row_finds_call_after_clarification():
    row = {
        "id": "clarify",
        "tools": json.dumps(
            [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
        ),
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model."},
            {"from": "human", "value": "first question"},
            {"from": "gpt", "value": "Which one do you mean?"},
            {"from": "human", "value": "the second one"},
            {"from": "gpt", "value": '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'},
        ],
    }
    example = parse_hermes_row(row, "glaive")
    assert example is not None
    assert example.user == "the second one"
    assert example.calls == [{"name": "f", "arguments": {"x": 1}}]


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

        def walk(schema: dict, collected: list[str], prefix: str = "") -> None:
            for name, prop in (schema.get("properties") or {}).items():
                path = f"{prefix}.{name}" if prefix else name
                collected.append(f"{path}:{prop.get('type')}")
                walk(prop, collected, path)

        walk(function.get("parameters") or {}, paths)
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


def test_dedupe_near_keeps_distinct_tool_schemas():
    def example(identifier: str, tool_name: str) -> Example:
        tools = [{"type": "function", "function": {"name": tool_name, "parameters": {"type": "object"}}}]
        return Example(identifier, tools, "do it", [{"name": "f", "arguments": {}}], "t")

    first = example("1", "a")
    second = example("2", "b")
    assert len(dedupe_near([first, second])) == 2
    assert len(dedupe_near([first, example("3", "a")])) == 1


def test_load_examples_dedupes_across_files(tmp_path):
    rows = json.loads(Path("tests/fixtures/hermes_sample.json").read_text())
    first = tmp_path / "func-calling-singleturn.json"
    second = tmp_path / "glaive-function-calling-5k.json"
    first.write_text(json.dumps(rows[:1]))
    second.write_text(json.dumps(rows[:1]))
    assert len(load_examples([first, second])) == 1
