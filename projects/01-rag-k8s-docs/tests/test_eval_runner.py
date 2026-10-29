from pathlib import Path

import pytest

from evals.run_eval import main, run_eval, threshold_failures
from evals.schema import EvalQuestion, GoldSource
from rag.models import Answer, Chunk, Citation, RetrievedChunk, Usage

QUESTIONS = Path("evals/questions.jsonl")


def _question(qid="q1", category="single-hop", split="dev", sources=("https://x/a",), facts=("fact",)):
    return EvalQuestion(
        id=qid,
        split=split,
        category=category,
        question=f"what is {qid} ({category})?",
        reference_answer="ref",
        gold_sources=[GoldSource(url, "H") for url in sources],
        required_facts=list(facts),
    )


def _chunks():
    return [
        Chunk("c1", "The fact is here.", "https://x/a", "H", "a.md", "v"),
        Chunk("c2", "The fact is also here.", "https://x/a", "H2", "a.md", "v"),
        Chunk("c3", "Unrelated.", "https://x/other", "H3", "b.md", "v"),
    ]


class FakeRetriever:
    def __init__(self):
        self.last_timings = {"dense_ms": 2, "bm25_ms": 3, "rerank_ms": 1}
        self.calls = []
        self.chunks = _chunks()

    def retrieve(self, question, top_k=None):
        self.calls.append(top_k)
        return [
            RetrievedChunk(chunk, 1.0, dense_rank=rank, bm25_rank=rank, rerank_score=1.0)
            for rank, chunk in enumerate(self.chunks, start=1)
        ]


def _answer(question, chunks):
    if "unanswerable" in question:
        return Answer(
            "I don't know based on the provided documentation.",
            [],
            Usage(provider="test", model="fake", prompt_tokens=2, completion_tokens=1),
            1,
        )
    return Answer(
        "The fact is here [1].",
        [Citation(1, chunks[0].url, chunks[0].heading, chunks[0].id)],
        Usage(provider="test", model="fake", prompt_tokens=10, completion_tokens=5),
        4,
    )


def test_run_eval_requests_top_k_and_aggregates():
    questions = [
        _question("q1"),
        _question("q2", category="unanswerable", sources=(), facts=()),
    ]
    retriever = FakeRetriever()
    payload = run_eval(questions, retriever, answer_fn=_answer, top_k=20)

    assert retriever.calls == [20, 20]
    assert payload["questions"] == 2
    metrics = payload["metrics"]
    assert metrics["recall@20"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["fact_coverage"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["over_refusal_rate"] == 0.0
    assert metrics["p95_latency_ms"] == 10.0
    assert metrics["cost_usd_per_1k"] == 0.0
    assert metrics["prompt_tokens"] == 12
    assert "unanswerable" in payload["slices"]
    assert payload["rows"][0]["latency_ms"] == 10


def test_run_eval_without_generation_is_retrieval_only():
    payload = run_eval([_question("q1")], FakeRetriever(), top_k=20)
    assert "fact_coverage" not in payload["metrics"]
    assert payload["metrics"]["p95_latency_ms"] == 6.0


def test_run_eval_rejects_empty_selection():
    with pytest.raises(ValueError, match="no questions"):
        run_eval([], FakeRetriever())


def test_threshold_failures():
    assert threshold_failures({"recall@10": 0.6, "ndcg@10": 0.4}) == []
    assert threshold_failures({"recall@10": 0.1, "ndcg@10": 0.4}) == ["recall@10=0.100 < 0.50"]


def test_test_split_requires_certification(capsys):
    assert main(["--questions", str(QUESTIONS), "--split", "test", "--retrieval-only"]) == 1
    out = capsys.readouterr().out
    assert "refusing the test split" in out
    assert "Reviewer is blank" in out
