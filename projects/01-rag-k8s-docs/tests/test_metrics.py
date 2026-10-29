import pytest

from evals.metrics import (
    citation_precision_recall,
    cohen_kappa,
    fact_coverage,
    is_refusal,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert recall_at_k(["x"], {"a"}, 1) == 0.0


def test_mrr():
    assert mrr_at_k(["x", "a", "b"], {"a"}, 10) == 0.5
    assert mrr_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_ndcg():
    assert ndcg_at_k(["a", "x"], {"a"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(1 / 1.584962500721156)


def test_citation_pr():
    p, r = citation_precision_recall(["u1", "u2"], {"u1"})
    assert p == 0.5 and r == 1.0


def test_fragments_are_ignored():
    assert recall_at_k(["https://x/pods/#lifecycle"], {"https://x/pods/"}, 1) == 1.0


def test_duplicate_chunks_are_document_level():
    retrieved = ["https://x/pods/#a", "https://x/pods/#b", "https://x/pods/#c"]
    gold = {"https://x/pods/"}
    assert recall_at_k(retrieved, gold, 3) == 1.0
    assert ndcg_at_k(retrieved, gold, 3) == pytest.approx(1.0)
    assert citation_precision_recall(retrieved, gold) == (1.0, 1.0)


def test_fact_coverage_normalizes_case_and_punctuation():
    assert fact_coverage("A Pod is the smallest unit.", ["smallest unit"]) == 1.0
    assert fact_coverage("A Pod is small.", ["smallest unit"]) == 0.0


def test_cohen_kappa_perfect_agreement():
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0


def test_refusal_detection_ignores_case_and_punctuation():
    assert is_refusal("I don't know based on the provided documentation.")
    assert not is_refusal("A Pod is the smallest deployable unit.")


def test_percentile_nearest_rank():
    values = list(range(1, 21))
    assert percentile(values, 95) == 19.0
    assert percentile(values, 100) == 20.0
    assert percentile([], 95) == 0.0
