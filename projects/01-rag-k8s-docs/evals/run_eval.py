from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.judge import JUDGE_MODEL, JUDGE_PROVIDER, Judge
from evals.metrics import (
    citation_precision_recall,
    fact_coverage,
    is_refusal,
    mean,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
)
from evals.schema import EvalQuestion, load_questions
from evals.summarize import payload_to_markdown
from evals.validate_questions import certification_errors
from rag.config import PROVIDER_DEFAULTS, Settings

RETRIEVAL_KEYS = ("recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10")
SMOKE_THRESHOLDS = {"recall@10": 0.5, "ndcg@10": 0.35}


def _retrieval_ms(timings: dict) -> int:
    return int(sum(int(timings.get(key, 0)) for key in ("dense_ms", "bm25_ms", "rerank_ms")))


def _tokens(usage) -> tuple[int, int, str, str]:
    if hasattr(usage, "prompt_tokens"):
        return (
            int(usage.prompt_tokens),
            int(usage.completion_tokens),
            str(usage.provider),
            str(usage.model),
        )
    if isinstance(usage, dict):
        return (
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            str(usage.get("provider", "")),
            str(usage.get("model", "")),
        )
    return 0, 0, "", ""


def evaluate_question(
    question: EvalQuestion,
    retrieved,
    retrieval_ms: int,
    answer=None,
    judge: Judge | None = None,
) -> dict:
    urls = [item.chunk.url for item in retrieved]
    gold = {source.url for source in question.gold_sources}
    row: dict = {
        "id": question.id,
        "split": question.split,
        "category": question.category,
        "retrieved_urls": urls,
        "retrieval_latency_ms": retrieval_ms,
        "recall@5": recall_at_k(urls, gold, 5),
        "recall@10": recall_at_k(urls, gold, 10),
        "recall@20": recall_at_k(urls, gold, 20),
        "mrr@10": mrr_at_k(urls, gold, 10),
        "ndcg@10": ndcg_at_k(urls, gold, 10),
    }
    if answer is None:
        return row

    cited = [citation.url for citation in answer.citations]
    precision, recall = citation_precision_recall(cited, gold)
    prompt_tokens, completion_tokens, provider, model = _tokens(answer.usage)
    row.update(
        {
            "answer": answer.text,
            "citations": cited,
            "generation_latency_ms": answer.latency_ms,
            "latency_ms": retrieval_ms + answer.latency_ms,
            "fact_coverage": fact_coverage(answer.text, question.required_facts),
            "citation_precision": precision,
            "citation_recall": recall,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "provider": provider,
            "model": model,
            "refused": is_refusal(answer.text),
        }
    )
    if question.category == "unanswerable":
        row["refusal_correct"] = 1 if row["refused"] else 0
    if judge is not None:
        context = "\n\n".join(item.chunk.text for item in retrieved)
        row["judge"] = judge.score(
            question.question, answer.text, question.reference_answer, context
        )
    return row


def _aggregate(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["category"] != "unanswerable"]
    unanswerable = [row for row in rows if row["category"] == "unanswerable"]
    metrics: dict = {"n": len(rows)}
    for key in RETRIEVAL_KEYS:
        metrics[key] = mean(row[key] for row in answerable)

    generated = [row for row in answerable if "fact_coverage" in row]
    if generated:
        for key in ("fact_coverage", "citation_precision", "citation_recall"):
            metrics[key] = mean(row[key] for row in generated)
        metrics["over_refusal_rate"] = mean(row["refused"] for row in generated)
        metrics["p95_latency_ms"] = percentile([row["latency_ms"] for row in rows], 95)
        metrics["prompt_tokens"] = sum(row["prompt_tokens"] for row in rows)
        metrics["completion_tokens"] = sum(row["completion_tokens"] for row in rows)
        metrics["billed_usd"] = 0.0
        metrics["cost_usd_per_1k"] = 0.0
        metrics["cost_note"] = "free-tier API; token usage reported, billed cost is $0.00"
    else:
        metrics["p95_latency_ms"] = percentile(
            [row["retrieval_latency_ms"] for row in rows], 95
        )

    scored_refusals = [row for row in unanswerable if "refusal_correct" in row]
    if scored_refusals:
        metrics["refusal_accuracy"] = mean(row["refusal_correct"] for row in scored_refusals)

    judged = [row for row in generated if "judge" in row]
    if judged:
        valid = [row for row in judged if not row["judge"].get("invalid")]
        metrics["judge_n"] = len(valid)
        metrics["judge_invalid"] = len(judged) - len(valid)
        for field in ("correct", "faithful", "citation_support"):
            metrics[f"judge_{field}"] = mean(row["judge"][field] for row in valid) if valid else 0.0
    return metrics


def run_eval(
    questions: list[EvalQuestion],
    retriever,
    answer_fn=None,
    judge: Judge | None = None,
    top_k: int = 20,
) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not questions:
        raise ValueError("no questions selected")
    rows = []
    for question in questions:
        retrieved = retriever.retrieve(question.question, top_k=top_k)
        timings = getattr(retriever, "last_timings", {}) or {}
        answer = None
        if answer_fn is not None:
            answer = answer_fn(question.question, [item.chunk for item in retrieved])
        rows.append(
            evaluate_question(
                question,
                retrieved,
                _retrieval_ms(timings),
                answer=answer,
                judge=judge,
            )
        )
    categories = sorted({row["category"] for row in rows})
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "top_k": top_k,
        "questions": len(rows),
        "metrics": _aggregate(rows),
        "slices": {category: _aggregate([r for r in rows if r["category"] == category]) for category in categories},
        "rows": rows,
    }


def build_retriever(settings: Settings, mode: str):
    from rag.embed import Embedder
    from rag.index import Index
    from rag.retrieve import CrossEncoderReranker, Retriever

    reranker = (
        CrossEncoderReranker(settings.rerank_model) if mode == "hybrid_rerank" else None
    )
    return Retriever(
        Index.load(settings),
        Embedder(settings.embed_model),
        reranker,
        settings,
        dense_only=mode == "dense",
        use_reranker=mode == "hybrid_rerank",
    )


def build_judge(settings: Settings, model: str, cache_dir: str) -> Judge:
    from rag.generate import LLMClient

    defaults = PROVIDER_DEFAULTS[JUDGE_PROVIDER]
    key = os.getenv(defaults["key_env"])
    if not key:
        raise ValueError(f"{defaults['key_env']} is required for --judge")
    judge_settings = replace(
        settings,
        offline=False,
        llm_provider=JUDGE_PROVIDER,
        llm_model=model,
        llm_base_url=defaults["base_url"],
        llm_api_key=key,
        llm_fallback_provider=None,
        llm_fallback_model=None,
        llm_fallback_base_url=None,
        llm_fallback_api_key=None,
    )
    return Judge(LLMClient(judge_settings), cache_dir=cache_dir)


def threshold_failures(metrics: dict, thresholds: dict | None = None) -> list[str]:
    thresholds = SMOKE_THRESHOLDS if thresholds is None else thresholds
    return [
        f"{key}={metrics.get(key, 0.0):.3f} < {minimum:.2f}"
        for key, minimum in thresholds.items()
        if metrics.get(key, 0.0) < minimum
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG eval over the certified gold set.")
    parser.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument(
        "--mode", choices=("dense", "hybrid", "hybrid_rerank"), default="hybrid_rerank"
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="first 15 dev questions")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--cache-dir", default=str(ROOT / "evals" / "cache"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--certification", default=str(ROOT / "evals" / "CERTIFICATION.md"))
    parser.add_argument("--allow-uncertified", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-thresholds", action="store_true")
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    if args.smoke:
        selected = [q for q in questions if q.split == "dev"][:15]
    elif args.split == "all":
        selected = list(questions)
    else:
        selected = [q for q in questions if q.split == args.split]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        print(f"no questions selected for split={args.split}")
        return 1

    if any(q.split == "test" for q in selected) and not args.allow_uncertified:
        certification = Path(args.certification)
        errors = (
            [f"certification file not found: {certification}"]
            if not certification.exists()
            else certification_errors(certification)
        )
        if errors:
            print("refusing the test split until the gold set is certified:")
            print("\n".join(errors))
            return 1

    if args.retrieval_only and args.judge:
        parser.error("--judge requires generation; drop --retrieval-only")

    settings = Settings.from_env()
    if args.index_dir:
        settings = replace(settings, index_dir=args.index_dir)
    if args.offline:
        settings = replace(settings, offline=True)

    try:
        retriever = build_retriever(settings, args.mode)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot load index: {exc}")
        return 1

    answer_fn = None
    judge = None
    if not args.retrieval_only:
        if args.offline:
            from rag.generate import extractive_answer

            answer_fn = lambda question, chunks: extractive_answer(question, chunks)
        else:
            from rag.generate import LLMClient, answer_question

            try:
                llm = LLMClient(settings)
            except ValueError as exc:
                print(f"cannot configure generation: {exc}")
                return 1
            answer_fn = lambda question, chunks: answer_question(
                question, chunks, llm, settings
            )
            if args.judge:
                try:
                    judge = build_judge(settings, args.judge_model, args.cache_dir)
                except ValueError as exc:
                    print(f"cannot configure judge: {exc}")
                    return 1

    payload = run_eval(
        selected, retriever, answer_fn=answer_fn, judge=judge, top_k=args.top_k
    )
    payload.update(
        {
            "split": args.split,
            "mode": args.mode,
            "retrieval_only": args.retrieval_only,
            "offline": args.offline,
            "index_dir": settings.index_dir,
        }
    )

    stem = f"{args.split}-{args.mode}{'-smoke' if args.smoke else ''}"
    output = Path(args.output) if args.output else ROOT / "evals" / "results" / f"{stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = payload_to_markdown(output.stem, payload)
    output.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)

    if args.smoke and not args.no_thresholds:
        failures = threshold_failures(payload["metrics"])
        if failures:
            print("smoke thresholds failed: " + ", ".join(failures))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
