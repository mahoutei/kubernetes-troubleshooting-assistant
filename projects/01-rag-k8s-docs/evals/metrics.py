from __future__ import annotations

import math
import re


def _url_key(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/")


def _keys(urls) -> set[str]:
    return {_url_key(url) for url in urls}


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    gold_keys = _keys(gold)
    if not gold_keys:
        return 1.0
    # Count distinct documents: several chunks of the same page are one hit.
    return len(_keys(retrieved[:k]) & gold_keys) / len(gold_keys)


def mrr_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    gold_keys = _keys(gold)
    for rank, url in enumerate(retrieved[:k], start=1):
        if _url_key(url) in gold_keys:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    gold_keys = _keys(gold)
    dcg = 0.0
    seen: set[str] = set()
    for i, url in enumerate(retrieved[:k]):
        key = _url_key(url)
        if key in gold_keys and key not in seen:
            seen.add(key)
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_keys), k)))
    return dcg / idcg if idcg else 0.0


def citation_precision_recall(cited: list[str], gold: set[str]) -> tuple[float, float]:
    if not cited:
        return 0.0, 0.0
    gold_keys = _keys(gold)
    hits = sum(_url_key(url) in gold_keys for url in cited)
    covered = len(_keys(cited) & gold_keys)
    return hits / len(cited), covered / len(gold_keys) if gold_keys else 0.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def fact_coverage(answer: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    normalized = _normalize(answer)
    return sum(_normalize(f) in normalized for f in facts) / len(facts)


def cohen_kappa(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be non-empty and equal length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)
