"""Prepare a dev-only client pilot and summarize human-entered observations offline."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "projects/01-rag-k8s-docs/evals/incidents.jsonl"


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def template(cases):
    selected = [case["id"] for case in cases if case["split"] == "dev"]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("need nonempty, unique development cases")
    rows = []
    # Different people see each case in different arms; nobody repeats a case.
    for participant, sequence in enumerate((selected, list(reversed(selected)))):
        for order, case in enumerate(sequence, 1):
            original = selected.index(case)
            condition = "manual" if (original + participant) % 2 == 0 else "assistant"
            rows.append({"participant_id": f"participant-{participant + 1}", "case_id": case,
                         "condition": condition, "order": order,
                         "evidence_mode": "manual" if condition == "manual" else "live",
                         "elapsed_seconds": None, "completed": None, "review": None,
                         "artifact_reference": "", "notes": ""})
    return rows


def validate(rows):
    if not rows:
        raise ValueError("no observations supplied")
    identities, orders = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("each observation must be a JSON object")
        for key in ("participant_id", "case_id", "artifact_reference"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{key} must identify the participant, case, or saved evidence")
        identity = row["participant_id"], row["case_id"]
        if identity in identities:
            raise ValueError("a participant must not repeat a case, including across conditions")
        identities.add(identity)
        order = row.get("order")
        if type(order) is not int or order < 1 or (row["participant_id"], order) in orders:
            raise ValueError("order must be a unique positive integer per participant")
        orders.add((row["participant_id"], order))
        condition, mode = row.get("condition"), row.get("evidence_mode")
        if condition not in {"manual", "assistant"}:
            raise ValueError("condition must be manual or assistant")
        if (condition == "manual" and mode != "manual") or (condition == "assistant" and mode not in {"live", "recorded_demo", "fake"}):
            raise ValueError("evidence_mode must match the condition")
        elapsed = row.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("record a finite positive elapsed_seconds, including failed attempts")
        if type(row.get("completed")) is not bool:
            raise ValueError("completed must be a recorded boolean")
        review = row.get("review")
        if review is not None:
            if not isinstance(review, dict) or any(type(review.get(k)) is not bool for k in ("diagnosis_correct", "citation_support")):
                raise ValueError("review correctness and citation support must be booleans")
            if type(review.get("unsupported_claims")) is not int or review["unsupported_claims"] < 0:
                raise ValueError("unsupported_claims must be a nonnegative integer")
            if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
                raise ValueError("a completed review needs a reviewer identity")


def summarize(rows):
    validate(rows)
    eligible = [r for r in rows if r["evidence_mode"] in {"manual", "live"}]
    summaries = {}
    for condition in ("manual", "assistant"):
        arm = [r for r in eligible if r["condition"] == condition]
        reviewed = [r for r in arm if r["review"] is not None]
        acceptable = [r for r in reviewed if r["completed"] and r["review"]["diagnosis_correct"]
                      and r["review"]["citation_support"] and r["review"]["unsupported_claims"] == 0]
        summaries[condition] = {
            "attempts": len(arm), "completed": sum(r["completed"] for r in arm),
            "reviewed": len(reviewed), "unreviewed": len(arm) - len(reviewed),
            "median_attempt_seconds": statistics.median(r["elapsed_seconds"] for r in arm) if arm else None,
            "reviewed_acceptable_outcomes": len(acceptable),
            "acceptable_rate_among_reviewed": len(acceptable) / len(reviewed) if reviewed else None,
        }
    common = []
    for case in sorted({r["case_id"] for r in eligible}):
        values = {arm: [r["elapsed_seconds"] for r in eligible if r["case_id"] == case and r["condition"] == arm]
                  for arm in ("manual", "assistant")}
        if all(values.values()):
            manual, assisted = (statistics.median(values[arm]) for arm in ("manual", "assistant"))
            common.append({"case_id": case, "manual_n": len(values["manual"]),
                           "assistant_n": len(values["assistant"]), "manual_median_seconds": manual,
                           "assistant_median_seconds": assisted, "manual_minus_assistant_seconds": manual - assisted})
    return {"evidence_status": "human-entered observations; not independently verified",
            "excluded_by_mode": dict(Counter(r["evidence_mode"] for r in rows if r not in eligible)),
            "conditions": summaries, "common_case_comparisons": common,
            "median_common_case_difference_seconds": statistics.median(r["manual_minus_assistant_seconds"] for r in common) if common else None,
            "limitations": ["Descriptive pilot only; no causal, salary, financial ROI, or production MTTR claim.",
                            "Attempt durations include incomplete and incorrect attempts; inspect reviewed outcomes alongside timing.",
                            "Recorded and fake assistant responses are excluded; unreviewed outcomes have no quality score.",
                            "Common-case medians compare different people; skill, order, and case mix can confound differences."]}


def write_new(path, content):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="make an unfilled, counterbalanced development-case worksheet")
    init.add_argument("--output", required=True)
    report = commands.add_parser("summarize", help="summarize completed observations, without model calls")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            content = "".join(json.dumps(row) + "\n" for row in template(load_rows(CASES)))
        else:
            content = json.dumps(summarize(load_rows(args.input)), indent=2, allow_nan=False) + "\n"
        write_new(args.output, content)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Pilot error: {exc}\n")
    print(f"Saved {args.output}; no model calls or cluster access")


if __name__ == "__main__":
    main()
