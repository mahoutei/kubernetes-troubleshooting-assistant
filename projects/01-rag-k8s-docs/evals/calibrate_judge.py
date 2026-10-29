from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.judge import RUBRIC_VERSION
from evals.metrics import cohen_kappa, mean

MIN_PAIRS = 20


def load_pairs(labels_path: str | Path, results_path: str | Path) -> tuple[list[int], list[int], list[str]]:
    labels: dict[str, int] = {}
    for line in Path(labels_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("human_correct")
        if value in (0, 1):
            labels[row["id"]] = int(value)

    results = {
        row["id"]: row
        for row in json.loads(Path(results_path).read_text(encoding="utf-8"))["rows"]
    }
    human: list[int] = []
    judged: list[int] = []
    missing: list[str] = []
    for question_id, human_value in labels.items():
        judgement = results.get(question_id, {}).get("judge")
        if not judgement or judgement.get("invalid") or judgement.get("correct") not in (0, 1):
            missing.append(question_id)
            continue
        human.append(human_value)
        judged.append(int(judgement["correct"]))
    return human, judged, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against human labels.")
    parser.add_argument("--labels", default=str(ROOT / "evals" / "calibration.jsonl"))
    parser.add_argument("--results", required=True, help="dev results JSON containing judge scores")
    parser.add_argument("--out", default=str(ROOT / "evals" / "calibration.json"))
    parser.add_argument("--min-kappa", type=float, default=0.75)
    args = parser.parse_args(argv)

    if not Path(args.labels).exists():
        print(f"labels file not found: {args.labels}")
        return 1
    human, judged, missing = load_pairs(args.labels, args.results)
    if len(human) < MIN_PAIRS:
        print(f"need at least {MIN_PAIRS} paired labels, got {len(human)}")
        return 1

    kappa = cohen_kappa(human, judged)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rubric_version": RUBRIC_VERSION,
        "n": len(human),
        "agreement": mean(int(a == b) for a, b in zip(human, judged)),
        "kappa": kappa,
        "gate_pass": kappa >= args.min_kappa,
        "min_kappa": args.min_kappa,
        "missing_judgements": missing,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"n={payload['n']} agreement={payload['agreement']:.3f} "
        f"kappa={kappa:.3f} gate={'pass' if payload['gate_pass'] else 'fail'}"
    )
    if missing:
        print(f"missing or invalid judgements: {', '.join(missing)}")
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
