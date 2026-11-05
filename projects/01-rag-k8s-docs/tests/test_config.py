import pytest

from rag.config import Settings


def test_defaults_pin_docs_commit():
    s = Settings()
    assert "47132eb5367cfdcba0a3b33a0b3ffc81185bd7e8" in s.docs_url
    assert s.chunk_tokens == 500
    assert s.top_k_context == 5


def test_default_model_is_not_retired():
    s = Settings()
    assert s.llm_model == "openai/gpt-oss-120b"
    assert "llama-3.3" not in s.llm_model


def test_rejects_bad_chunk_size():
    with pytest.raises(ValueError, match="chunk_tokens"):
        Settings(chunk_tokens=50)


def test_rejects_bad_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_overlap=1.0)


def test_from_env_reads_groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    s = Settings.from_env()
    assert s.llm_provider == "groq"
    assert s.llm_api_key == "test-key"


def test_from_env_reads_fallback_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    s = Settings.from_env()
    assert s.llm_fallback_provider == "gemini"
    assert s.llm_fallback_model == "gemini-2.5-flash"
    assert s.llm_fallback_api_key == "gemini-key"


def test_from_env_can_disable_fallback(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "none")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = Settings.from_env()
    assert s.llm_fallback_provider is None


def test_rejects_unknown_fallback_provider(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "nope")
    with pytest.raises(ValueError, match="LLM_FALLBACK_PROVIDER"):
        Settings.from_env()


def test_rejects_same_fallback_and_primary():
    with pytest.raises(ValueError, match="differ"):
        Settings(llm_fallback_provider="groq")


def test_settings_reject_non_positive_retrieval_and_generation_limits():
    import pytest

    for name in ("top_k_dense", "top_k_bm25", "top_k_rerank", "top_k_context", "rrf_k", "max_answer_tokens"):
        with pytest.raises(ValueError, match=name):
            Settings(**{name: 0})


def test_cpu_serving_defaults_to_hybrid_and_reranking_is_opt_in(monkeypatch):
    assert Settings().retrieval_mode == "hybrid"
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "hybrid_rerank")
    assert Settings.from_env().retrieval_mode == "hybrid_rerank"
    with pytest.raises(ValueError, match="retrieval_mode"):
        Settings(retrieval_mode="anything")
