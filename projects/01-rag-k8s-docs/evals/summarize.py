from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = (
    ("config", "config"),
    ("n", "n"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("recall@20", "R@20"),
    ("mrr@10", "MRR@10"),
    ("ndcg@10", "nDCG@10"),
    ("citation_precision", "cit-P"),
    ("citation_recall", "cit-R"),
    ("fact_coverage", "fact-cov"),
    ("refusal_accuracy", "refusal"),
    ("p95_latency_ms", "p95 ms"),
    ("cost_usd_per_1k", "$/1k"),
)


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_table(rows: list[tuple[str, dict]]) -> str:
    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    separator = "|" + "|".join("---:" for _ in COLUMNS) + "|"
    lines = [header, separator]
    for name, payload in rows:
        metrics = payload.get("metrics", {})
        cells = [name, str(payload.get("questions", metrics.get("n", "")))]
        for key, _ in COLUMNS[2:]:
            cells.append(_fmt(metrics.get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def payload_to_markdown(name: str, payload: dict) -> str:
    return format_table([(name, payload)])


def summarize(paths: list[str | Path]) -> str:
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append((Path(path).stem, payload))
    return format_table(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render eval results JSON as a markdown table.")
    parser.add_argument("paths", nargs="+", help="result JSON files")
    args = parser.parse_args(argv)
    print(summarize(args.paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
