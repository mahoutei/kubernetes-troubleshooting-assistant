from typing import ClassVar

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import Settings
from rag.generate import LLMError
from rag.models import Chunk, RetrievedChunk, Usage


def fake_app():
    chunk = Chunk("a", "A Pod is small.", "u1", "Pods", "pods.md", "v")

    class FakeRetriever:
        index: ClassVar = type("FakeIndex", (), {"count": 1})()
        last_timings: ClassVar = {"dense_ms": 1, "bm25_ms": 2, "rerank_ms": 3}

        def retrieve(self, question):
            return [RetrievedChunk(chunk, 1.0, dense_rank=1, bm25_rank=1, rerank_score=1.0)]

    class FakeLLM:
        def complete(self, messages, max_tokens):
            return "A Pod is small [1].", Usage(
                provider="groq",
                model="openai/gpt-oss-120b",
                prompt_tokens=1,
                completion_tokens=1,
            )

    return create_app(Settings(offline=False), retriever=FakeRetriever(), llm=FakeLLM())


def test_ask_returns_citations():
    client = TestClient(fake_app())
    r = client.post("/ask", json={"question": "What is a Pod?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"][0]["url"] == "u1"
    assert body["answer"].startswith("A Pod")
    assert body["retrieval"]["dense_ms"] == 1
    assert body["usage"]["provider"] == "groq"
    assert body["usage"]["model"] == "openai/gpt-oss-120b"


def test_provider_failure_is_503():
    class FailingLLM:
        def complete(self, messages, max_tokens):
            raise LLMError("all LLM providers failed: groq: retired")

    app = create_app(Settings(offline=False), retriever=fake_app().state.retriever, llm=FailingLLM())
    r = TestClient(app).post("/ask", json={"question": "What is a Pod?"})
    assert r.status_code == 503
    assert "generation unavailable" in r.json()["detail"]


def test_empty_question_is_422():
    client = TestClient(fake_app())
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_health_reports_ready():
    client = TestClient(fake_app())
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 1


def test_missing_index_is_degraded(tmp_path):
    with TestClient(create_app(Settings(index_dir=str(tmp_path / "missing"), offline=True))) as client:
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert "index not found" in body["error"]


def test_missing_index_ask_is_503(tmp_path):
    with TestClient(create_app(Settings(index_dir=str(tmp_path / "missing"), offline=True))) as client:
        assert client.post("/ask", json={"question": "What is a Pod?"}).status_code == 503


def test_health_handles_unavailable_index_on_injected_retriever():
    app = fake_app()
    app.state.retriever.index = None
    assert TestClient(app).get("/health").json()["chunks"] == 0


def test_api_snapshots_timings_before_generation():
    app = fake_app()
    retriever = app.state.retriever

    class MutatingLLM:
        def complete(self, messages, max_tokens):
            retriever.last_timings = {"dense_ms": 999}
            return "A Pod [1]", Usage()

    app.state.llm = MutatingLLM()
    response = TestClient(app).post("/ask", json={"question": "What is a Pod?"})
    assert response.json()["retrieval"]["dense_ms"] == 1


def test_models_load_at_startup_instead_of_app_construction(monkeypatch):
    from rag import api

    calls = []
    retriever = fake_app().state.retriever

    def load(settings):
        calls.append("loaded")
        return retriever

    monkeypatch.setattr(api, "_default_retriever", load)
    app = create_app(Settings(offline=True))
    assert calls == []
    with TestClient(app) as client:
        assert calls == ["loaded"]
        assert client.get("/health").json()["status"] == "ok"
        assert client.post("/ask", json={"question": "What is a Pod?"}).status_code == 200
