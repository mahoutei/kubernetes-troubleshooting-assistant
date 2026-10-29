from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.schema import load_questions, validate_questions

CERTIFICATION_FIELDS = ("Reviewer", "Date", "Notes")


def certification_errors(path: str | Path) -> list[str]:
    """Return missing/blank certification fields for a CERTIFICATION.md file."""
    text = Path(path).read_text(encoding="utf-8")
    errors: list[str] = []
    for field in CERTIFICATION_FIELDS:
        match = re.search(rf"^- {field}:\s*(.*)$", text, re.MULTILINE)
        if match is None or not match.group(1).strip().strip("_- "):
            errors.append(f"certification: {field} is blank")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the gold set.")
    parser.add_argument("path", nargs="?", default=str(ROOT / "evals" / "questions.jsonl"))
    parser.add_argument(
        "--require-certified",
        action="store_true",
        help="also require a signed evals/CERTIFICATION.md (used before test-split runs)",
    )
    parser.add_argument(
        "--certification", default=str(ROOT / "evals" / "CERTIFICATION.md")
    )
    args = parser.parse_args(argv)

    questions = load_questions(args.path)
    errors = validate_questions(questions)
    if args.require_certified:
        cert = Path(args.certification)
        if not cert.exists():
            errors.append(f"certification file not found: {cert}")
        else:
            errors.extend(certification_errors(cert))
    if errors:
        print("\n".join(errors))
        return 1
    print(f"ok: {len(questions)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
