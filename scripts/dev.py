"""Run independent project checks using the root CPU-only environment."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "rag": "projects/01-rag-k8s-docs",
    "finetune": "projects/02-finetune-function-calling",
}


@dataclass(frozen=True)
class Step:
    label: str
    cwd: Path
    command: tuple[str, ...]


def build_steps(action: str, project: str, root: Path = ROOT) -> list[Step]:
    selected = tuple(PROJECTS) if project == "all" else (project,)
    steps = []
    if action in {"check", "lint"}:
        steps.append(Step("tooling: lint", root, (sys.executable, "-m", "ruff", "check", "scripts", "tests", "pyproject.toml")))
    if action in {"check", "test"}:
        steps.append(Step("tooling: tests", root, (sys.executable, "-m", "pytest", "-q", "tests")))
    for name in selected:
        cwd = root / PROJECTS[name]
        if action in {"check", "lint"}:
            steps.append(Step(f"{name}: lint", cwd, (sys.executable, "-m", "ruff", "check", ".")))
        if action in {"check", "test"}:
            steps.append(Step(f"{name}: tests", cwd, (sys.executable, "-m", "pytest", "-q")))
        if action == "check" and name == "rag":
            steps.append(Step("rag: gold questions", cwd, (sys.executable, "evals/validate_questions.py", "evals/questions.jsonl")))
    return steps


def run_steps(steps: list[Step]) -> int:
    results = []
    for step in steps:
        print(f"\n>>> {step.label}", flush=True)
        try:
            result = subprocess.run(list(step.command), cwd=step.cwd, check=False)
            code = result.returncode
        except KeyboardInterrupt:
            # Children inherit the terminal process group and receive Ctrl-C too.
            print(f"\nInterrupted: {step.label}", flush=True)
            return 130
        except OSError as exc:
            print(f"Cannot run {step.label}: {exc}", file=sys.stderr, flush=True)
            code = 127
        if code < 0 or code == 130:
            print(f"Interrupted: {step.label}", flush=True)
            return 128 - code if code < 0 else code
        results.append((step.label, code))
        print(f"{'PASS' if code == 0 else 'FAIL'}: {step.label}", flush=True)
    print("\nCheck results:", flush=True)
    for label, code in results:
        print(f"  {'PASS' if code == 0 else f'FAIL ({code})'}  {label}", flush=True)
    failed = sum(code != 0 for _, code in results)
    print(f"{len(results) - failed} passed; {failed} failed", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "test", "lint"))
    parser.add_argument("--project", choices=("all", *PROJECTS), default="all")
    args = parser.parse_args(argv)
    return run_steps(build_steps(args.action, args.project))


if __name__ == "__main__":
    raise SystemExit(main())
