import numpy as np

from rag.models import Chunk
from rag.retrieve import Retriever, reciprocal_rank_fusion


def test_rrf_prefers_consensus():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c"]], k=60)
    assert fused[0][0] == "b"


def test_retriever_fuses_and_reranks():
    chunks = {
        "a": Chunk("a", "alpha", "u1", "A", "a.md", "v"),
        "b": Chunk("b", "beta", "u2", "B", "b.md", "v"),
        "c": Chunk("c", "gamma", "u3", "C", "c.md", "v"),
    }

    class FakeIndex:
        count = 3

        def dense_search(self, vector, k):
            return [("a", 0.1), ("b", 0.2)]

        def bm25_search(self, query, k):
            return [("b", 9.0), ("c", 8.0)]

        def get(self, chunk_id):
            return chunks[chunk_id]

    class FakeEmbedder:
        def encode(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

    class FakeReranker:
        def score(self, query, texts):
            return [0.1 if t == "alpha" else 0.9 for t in texts]

    r = Retriever(FakeIndex(), FakeEmbedder(), FakeReranker(), top_k_context=2)
    results = r.retrieve("anything")
    assert [x.chunk.id for x in results] == ["b", "c"]
    assert results[0].dense_rank == 2
    assert results[0].bm25_rank == 1
    assert results[0].rerank_score == 0.9
