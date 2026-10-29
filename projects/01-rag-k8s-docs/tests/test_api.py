from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import Settings
from rag.models import Chunk, RetrievedChunk


def fake_app():
    chunk = Chunk("a", "A Pod is small.", "u1", "Pods", "pods.md", "v")

    class FakeRetriever:
        index = type("FakeIndex", (), {"count": 1})()
        last_timings = {"dense_ms": 1, "bm25_ms": 2, "rerank_ms": 3}

        def retrieve(self, question):
            return [RetrievedChunk(chunk, 1.0, dense_rank=1, bm25_rank=1, rerank_score=1.0)]

    class FakeLLM:
        def complete(self, messages, max_tokens):
            return "A Pod is small [1].", {"prompt_tokens": 1, "completion_tokens": 1}

    return create_app(Settings(offline=False), retriever=FakeRetriever(), llm=FakeLLM())


def test_ask_returns_citations():
    client = TestClient(fake_app())
    r = client.post("/ask", json={"question": "What is a Pod?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["url"] == "u1"
    assert body["answer"].startswith("A Pod")
    assert body["retrieval"]["dense_ms"] == 1


def test_empty_question_is_422():
    client = TestClient(fake_app())
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_health_reports_ready():
    client = TestClient(fake_app())
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 1


def test_missing_index_is_degraded(tmp_path):
    client = TestClient(create_app(Settings(index_dir=str(tmp_path / "missing"), offline=True)))
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert "index not found" in body["error"]


def test_missing_index_ask_is_503(tmp_path):
    client = TestClient(create_app(Settings(index_dir=str(tmp_path / "missing"), offline=True)))
    assert client.post("/ask", json={"question": "What is a Pod?"}).status_code == 503
