"""Reproducible incident comparisons; semantic quality requires hash-bound human reviews."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evals.run_eval import build_retriever
from evals.validate_questions import certification_errors
from rag.config import Settings
from rag.incident import DiagnosisContent, IncidentRequest, diagnose
from rag.models import Usage


class RecordedGitHub:
    def __init__(self):
        self.records = json.loads((ROOT / "src/rag/resources/github_recordings.json").read_text())

    def lookup(self, topic, version=None):
        result = self.records.get(topic, {"items": [], "warnings": ["No GitHub recording for this topic"]})
        return {**result, "status": "recorded_example", "requests": 0}


class FakeProvider:
    def complete(self, messages, max_tokens):
        evidence = json.loads(messages[1]["content"])["evidence"]
        supplied = next(item for item in evidence if item["kind"] == "user")
        claim = {"text": "Supplied evidence needs review; no verified diagnosis is established.", "evidence_ids": [supplied["id"]]}
        response = DiagnosisContent(observations=[claim], hypotheses=[], conflicts=[],
                                    missing_evidence=["Collect exact Pod status, recent events and timestamped logs"],
                                    next_steps=[{**claim, "command": "kubectl describe pods"}])
        return response.model_dump_json(), Usage(provider="fake", model="integration-test")


def prediction_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def review_metrics(rows, reviews):
    paired = []
    for row in rows:
        if "response" not in row:
            continue
        review = reviews.get((row["id"], row["mode"], row["prediction_sha256"]))
        if review and all(type(review.get(key)) is int and review[key] in (0, 1)
                          for key in ("diagnosis_correct", "citation_support", "appropriate_missing_evidence")) and type(review.get("unsupported_claims")) is int and review["unsupported_claims"] >= 0:
            paired.append(review)
    return {"human_reviewed_n": len(paired),
            **{key: sum(review[key] for review in paired) / len(paired) if paired else None
               for key in ("diagnosis_correct", "citation_support", "appropriate_missing_evidence", "unsupported_claims")}}


def run_comparison(cases, retriever, llm, modes, github):
    rows = []
    for case in cases:
        for mode in modes:
            request = IncidentRequest.model_validate({**case["request"], "github_lookup": mode == "rag_github"})
            row = {"id": case["id"], "mode": mode, "split": case["split"], "reference": {key: case[key] for key in ("expected_observations", "acceptable_hypotheses", "relevant_documentation", "appropriate_checks", "expected_missing_evidence")}}
            try:
                response = diagnose(request, retriever, llm, github, use_retrieval=mode != "baseline")
                row["response"] = response.model_dump()
                row["prediction_sha256"] = prediction_hash(row["response"])
            except (RuntimeError, ValueError, OSError) as exc:
                row["error"] = type(exc).__name__
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--mode", choices=("all", "baseline", "rag", "rag_github"), default="all")
    parser.add_argument("--index-dir", default="data/index-beta-final")
    parser.add_argument("--fake-provider", action="store_true", help="integration verification only; not quality evidence")
    parser.add_argument("--reviews")
    parser.add_argument("--summarize-existing", help="aggregate hash-bound reviews without rerunning model calls")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.summarize_existing:
        if not args.reviews:
            parser.error("--summarize-existing requires --reviews")
        payload = json.loads(Path(args.summarize_existing).read_text())
        reviews = {}
        for line in Path(args.reviews).read_text().splitlines():
            review = json.loads(line)
            key = (review["id"], review["mode"], review["prediction_sha256"])
            if key in reviews:
                raise ValueError("duplicate review identity")
            reviews[key] = review
        for mode in payload["metrics"]:
            payload["metrics"][mode].update(review_metrics([row for row in payload["rows"] if row["mode"] == mode], reviews))
        destination = Path(args.output)
        if destination.exists():
            raise ValueError("choose a new summary output; predictions are preserved")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload["metrics"], indent=2))
        return
    if args.reviews:
        parser.error("use --summarize-existing with --reviews to avoid changing prediction hashes")
    cases = [json.loads(line) for line in (ROOT / "evals/incidents.jsonl").read_text().splitlines()]
    if args.split == "test" and certification_errors(ROOT / "evals/INCIDENT_CERTIFICATION.md"):
        parser.error("held-out incident evaluation requires signed human certification")
    selected = [case for case in cases if case["split"] == args.split]
    modes = ("baseline", "rag", "rag_github") if args.mode == "all" else (args.mode,)
    settings = Settings.from_env()
    from dataclasses import replace

    settings = replace(settings, index_dir=args.index_dir)
    if args.fake_provider:
        llm = FakeProvider()
    else:
        from rag.generate import LLMClient

        llm = LLMClient(settings)
    retriever = build_retriever(settings, "hybrid") if any(mode != "baseline" for mode in modes) else None
    rows = run_comparison(selected, retriever, llm, modes, RecordedGitHub())
    reviews = {}
    if args.reviews:
        for line in Path(args.reviews).read_text().splitlines():
            review = json.loads(line)
            key = (review["id"], review["mode"], review["prediction_sha256"])
            if key in reviews:
                raise ValueError("duplicate review identity")
            reviews[key] = review
    summaries = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        successes = [row for row in mode_rows if "response" in row]
        summaries[mode] = {"n": len(mode_rows), "validated_responses": len(successes),
            "mean_latency_ms": sum(row["response"]["latency_ms"] for row in successes) / len(successes) if successes else None,
            "prompt_tokens": sum(row["response"]["usage"].get("prompt_tokens", 0) for row in successes),
            "completion_tokens": sum(row["response"]["usage"].get("completion_tokens", 0) for row in successes),
            **review_metrics(mode_rows, reviews)}
    payload = {"generated_at": datetime.now(UTC).isoformat(), "split": args.split,
               "evidence_status": "fake-provider integration check; not model quality" if args.fake_provider else "human semantic review required",
               "github": "recorded public API snapshots", "service_sha256": hashlib.sha256((ROOT / "src/rag/incident.py").read_bytes()).hexdigest(), "github_snapshot_sha256": hashlib.sha256((ROOT / "src/rag/resources/github_recordings.json").read_bytes()).hexdigest(), "metrics": summaries, "rows": rows}
    output = Path(args.output)
    if output.exists():
        raise ValueError("refusing to overwrite an existing result artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    template = [{"id": row["id"], "mode": row["mode"], "prediction_sha256": row["prediction_sha256"],
                 "diagnosis_correct": None, "unsupported_claims": None, "citation_support": None,
                 "appropriate_missing_evidence": None, "reviewer": "", "notes": ""} for row in rows if "response" in row]
    output.with_suffix(".reviews.jsonl").write_text("".join(json.dumps(row) + "\n" for row in template))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
