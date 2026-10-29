from __future__ import annotations

import time
from collections import defaultdict

from rag.models import RetrievedChunk


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class Retriever:
    def __init__(
        self,
        index,
        embedder,
        reranker,
        settings=None,
        top_k_context: int | None = None,
        dense_only: bool = False,
        use_reranker: bool = True,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.dense_only = dense_only
        self.use_reranker = use_reranker and not dense_only
        self.top_k_dense = getattr(settings, "top_k_dense", 50)
        self.top_k_bm25 = getattr(settings, "top_k_bm25", 50)
        self.top_k_rerank = getattr(settings, "top_k_rerank", 20)
        self.rrf_k = getattr(settings, "rrf_k", 60)
        self.top_k_context = (
            top_k_context
            if top_k_context is not None
            else getattr(settings, "top_k_context", 5)
        )
        self.last_timings: dict[str, int] = {}

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        limit = self.top_k_context if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be >= 1")
        dense_started = time.perf_counter()
        vector = self.embedder.encode([question])[0]
        dense = self.index.dense_search(vector, self.top_k_dense)
        dense_ms = int((time.perf_counter() - dense_started) * 1000)
        dense_rank = {doc_id: i for i, (doc_id, _) in enumerate(dense, start=1)}
        if self.dense_only:
            self.last_timings = {"dense_ms": dense_ms, "bm25_ms": 0, "rerank_ms": 0}
            return [
                RetrievedChunk(
                    chunk=self.index.get(doc_id),
                    score=score,
                    dense_rank=rank,
                )
                for rank, (doc_id, score) in enumerate(dense[:limit], start=1)
            ]

        bm25_started = time.perf_counter()
        bm25 = self.index.bm25_search(question, self.top_k_bm25)
        bm25_ms = int((time.perf_counter() - bm25_started) * 1000)
        bm25_rank = {doc_id: i for i, (doc_id, _) in enumerate(bm25, start=1)}
        fused = reciprocal_rank_fusion(
            [[doc_id for doc_id, _ in dense], [doc_id for doc_id, _ in bm25]],
            self.rrf_k,
        )
        if not self.use_reranker:
            self.last_timings = {"dense_ms": dense_ms, "bm25_ms": bm25_ms, "rerank_ms": 0}
            return [
                RetrievedChunk(
                    chunk=self.index.get(doc_id),
                    score=score,
                    dense_rank=dense_rank.get(doc_id),
                    bm25_rank=bm25_rank.get(doc_id),
                )
                for doc_id, score in fused[:limit]
            ]

        pool = fused[: max(limit, self.top_k_rerank)]
        candidates = [self.index.get(doc_id) for doc_id, _ in pool]
        rerank_started = time.perf_counter()
        scores = self.reranker.score(question, [c.text for c in candidates])
        rerank_ms = int((time.perf_counter() - rerank_started) * 1000)
        self.last_timings = {"dense_ms": dense_ms, "bm25_ms": bm25_ms, "rerank_ms": rerank_ms}
        ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])[:limit]
        return [
            RetrievedChunk(
                chunk=chunk,
                score=score,
                dense_rank=dense_rank.get(chunk.id),
                bm25_rank=bm25_rank.get(chunk.id),
                rerank_score=score,
            )
            for chunk, score in ranked
        ]
