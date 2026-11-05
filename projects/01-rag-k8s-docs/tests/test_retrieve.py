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


def test_retrieve_honours_top_k_beyond_context_size():
    chunks = {
        f"c{i}": Chunk(f"c{i}", f"text {i}", f"https://x/{i}", "H", f"{i}.md", "v")
        for i in range(25)
    }

    class FakeIndex:
        count = 25

        def dense_search(self, vector, k):
            return [(f"c{i}", float(i)) for i in range(25)]

        def bm25_search(self, query, k):
            return [(f"c{i}", float(25 - i)) for i in range(25)]

        def get(self, chunk_id):
            return chunks[chunk_id]

    class FakeEmbedder:
        def encode(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

    class FakeReranker:
        def score(self, query, texts):
            return [float(len(text)) for text in texts]

    retriever = Retriever(FakeIndex(), FakeEmbedder(), FakeReranker(), top_k_context=5)
    assert len(retriever.retrieve("q")) == 5
    assert len(retriever.retrieve("q", top_k=20)) == 20

    dense = Retriever(FakeIndex(), FakeEmbedder(), FakeReranker(), top_k_context=5, dense_only=True)
    assert len(dense.retrieve("q", top_k=20)) == 20


def test_retrieve_rejects_non_positive_top_k():
    chunks = {"a": Chunk("a", "alpha", "u", "H", "a.md", "v")}

    class FakeIndex:
        count = 1

        def dense_search(self, vector, k):
            return [("a", 0.1)]

        def bm25_search(self, query, k):
            return [("a", 0.1)]

        def get(self, chunk_id):
            return chunks[chunk_id]

    class FakeEmbedder:
        def encode(self, texts):
            return np.zeros((len(texts), 4), dtype=np.float32)

    class FakeReranker:
        def score(self, query, texts):
            return [0.5 for _ in texts]

    retriever = Retriever(FakeIndex(), FakeEmbedder(), FakeReranker())
    try:
        retriever.retrieve("q", top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_timing_metadata_is_isolated_between_concurrent_requests():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    retriever = Retriever(None, None, None)
    barrier = Barrier(2)

    def request(value):
        retriever.last_timings = {"dense_ms": value}
        barrier.wait(timeout=5)
        return retriever.last_timings

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(request, 1)
        second = pool.submit(request, 2)
        assert first.result(timeout=5) == {"dense_ms": 1}
        assert second.result(timeout=5) == {"dense_ms": 2}
    assert retriever.last_timings == {}
