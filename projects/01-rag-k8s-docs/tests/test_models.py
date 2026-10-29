from rag.models import Chunk, Citation, RetrievedChunk


def test_chunk_is_frozen():
    c = Chunk("id", "text", "https://example.com", "Heading", "concepts/a.md", "release-1.34")
    assert c.id == "id"


def test_retrieved_chunk_carries_ranks():
    c = Chunk("id", "text", "https://example.com", "Heading", "concepts/a.md", "release-1.34")
    r = RetrievedChunk(c, 0.5, dense_rank=1, bm25_rank=3, rerank_score=0.9)
    assert r.dense_rank == 1
    assert r.rerank_score == 0.9
