from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = {"single-hop", "multi-hop", "troubleshooting", "unanswerable"}
EXPECTED = {"single-hop": 20, "multi-hop": 15, "troubleshooting": 15, "unanswerable": 10}


@dataclass(frozen=True)
class GoldSource:
    url: str
    heading: str


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    split: str
    category: str
    question: str
    reference_answer: str
    gold_sources: list[GoldSource]
    required_facts: list[str]


def load_questions(path: str | Path) -> list[EvalQuestion]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        rows.append(
            EvalQuestion(
                id=raw["id"],
                split=raw["split"],
                category=raw["category"],
                question=raw["question"],
                reference_answer=raw["reference_answer"],
                gold_sources=[GoldSource(**s) for s in raw["gold_sources"]],
                required_facts=raw["required_facts"],
            )
        )
    return rows


def validate_questions(questions: list[EvalQuestion]) -> list[str]:
    errors = []
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")
    for split, count in (("dev", 30), ("test", 30)):
        actual = sum(q.split == split for q in questions)
        if actual != count:
            errors.append(f"{split}: expected {count}, got {actual}")
    for category, count in EXPECTED.items():
        actual = sum(q.category == category for q in questions)
        if actual != count:
            errors.append(f"{category}: expected {count}, got {actual}")
    for q in questions:
        if q.category not in CATEGORIES:
            errors.append(f"{q.id}: bad category {q.category}")
        if q.category != "unanswerable" and not q.gold_sources:
            errors.append(f"{q.id}: missing gold sources")
        if q.category != "unanswerable" and not q.required_facts:
            errors.append(f"{q.id}: missing required facts")
    return errors
