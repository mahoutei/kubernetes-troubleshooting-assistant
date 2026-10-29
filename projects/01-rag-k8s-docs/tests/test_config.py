import pytest
from rag.config import Settings


def test_defaults_pin_docs_commit():
    s = Settings()
    assert "47132eb5367cfdcba0a3b33a0b3ffc81185bd7e8" in s.docs_url
    assert s.chunk_tokens == 500
    assert s.top_k_context == 5


def test_rejects_bad_chunk_size():
    with pytest.raises(ValueError, match="chunk_tokens"):
        Settings(chunk_tokens=50)


def test_rejects_bad_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        Settings(chunk_overlap=1.0)


def test_from_env_reads_groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = Settings.from_env()
    assert s.llm_provider == "groq"
    assert s.llm_api_key == "test-key"
