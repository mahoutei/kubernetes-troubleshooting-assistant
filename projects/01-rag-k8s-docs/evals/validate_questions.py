from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.schema import load_questions, validate_questions


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "evals/questions.jsonl")
    questions = load_questions(path)
    errors = validate_questions(questions)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print(f"ok: {len(questions)} questions")


if __name__ == "__main__":
    main()
