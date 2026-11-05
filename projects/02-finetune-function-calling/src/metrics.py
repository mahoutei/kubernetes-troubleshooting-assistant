"""Strict, type-preserving metrics for complete tool-call outputs."""
from __future__ import annotations

import json
from collections import Counter


def canonical(value):
    # JSON's bool/int and int/float distinctions must survive Python equality.
    if isinstance(value, dict):
        return ("object", tuple(sorted((k, canonical(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ("array", tuple(canonical(v) for v in value))
    return (type(value).__name__, value)


def strict_calls(text: str) -> list[dict] | None:
    decoder = json.JSONDecoder(
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        object_pairs_hook=unique_object,
    )
    remaining = text.strip()
    calls = []
    while remaining:
        if not remaining.startswith("<tool_call>"):
            return None
        remaining = remaining[len("<tool_call>"):].lstrip()
        try:
            call, end = decoder.raw_decode(remaining)
        except (ValueError, TypeError):
            return None
        remaining = remaining[end:].lstrip()
        if not remaining.startswith("</tool_call>"):
            return None
        remaining = remaining[len("</tool_call>"):].lstrip()
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or not call["name"].strip()
                or not isinstance(call["arguments"], dict)):
            return None
        calls.append(call)
    return calls or None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def score_output(text: str, gold: list[dict]) -> dict:
    predicted = strict_calls(text)
    calls = predicted or []
    names = Counter(c["name"] for c in calls)
    expected_names = Counter(c["name"] for c in gold)
    matched_names = sum((names & expected_names).values())
    matches = sum((Counter(canonical(c) for c in calls) & Counter(canonical(c) for c in gold)).values())
    exact = predicted is not None and Counter(canonical(c) for c in calls) == Counter(canonical(c) for c in gold)
    return {
        "valid_output": int(predicted is not None),
        "tool_name_precision": matched_names / len(calls) if calls else 0.0,
        "tool_name_recall": matched_names / len(gold) if gold else 0.0,
        # Correct (name, arguments) calls divided by the larger call count penalizes extras and omissions.
        "argument_exact_match": matches / max(len(calls), len(gold), 1),
        "whole_example_exact_match": int(exact),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty predictions")
    keys = ("valid_output", "tool_name_precision", "tool_name_recall", "argument_exact_match", "whole_example_exact_match", "latency_ms")
    return {"n": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in keys}}
