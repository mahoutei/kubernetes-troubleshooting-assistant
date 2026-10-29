from __future__ import annotations

import os
from dataclasses import dataclass

DOCS_COMMIT = "47132eb5367cfdcba0a3b33a0b3ffc81185bd7e8"
DOCS_URL = f"https://github.com/kubernetes/website/archive/{DOCS_COMMIT}.tar.gz"
DOCS_PREFIX = f"website-{DOCS_COMMIT}/content/en/docs"

PROVIDER_DEFAULTS = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
}


@dataclass(frozen=True)
class Settings:
    docs_url: str = DOCS_URL
    docs_prefix: str = DOCS_PREFIX
    docs_dir: str = "data/docs"
    index_dir: str = "data/index"
    chunk_tokens: int = 500
    chunk_overlap: float = 0.15
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    top_k_dense: int = 50
    top_k_bm25: int = 50
    top_k_rerank: int = 20
    top_k_context: int = 5
    rrf_k: int = 60
    llm_provider: str = "groq"
    llm_model: str = PROVIDER_DEFAULTS["groq"]["model"]
    llm_base_url: str = PROVIDER_DEFAULTS["groq"]["base_url"]
    llm_api_key: str | None = None
    offline: bool = False
    max_answer_tokens: int = 600

    def __post_init__(self) -> None:
        if not 100 <= self.chunk_tokens <= 2000:
            raise ValueError("chunk_tokens must be between 100 and 2000")
        if not 0.0 <= self.chunk_overlap < 1.0:
            raise ValueError("chunk_overlap must be in [0, 1)")
        if self.top_k_context > self.top_k_rerank:
            raise ValueError("top_k_context cannot exceed top_k_rerank")

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDER_DEFAULTS)}")
        defaults = PROVIDER_DEFAULTS[provider]
        return cls(
            docs_dir=os.getenv("RAG_DOCS_DIR", "data/docs"),
            index_dir=os.getenv("RAG_INDEX_DIR", "data/index"),
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", defaults["model"]),
            llm_base_url=os.getenv("LLM_BASE_URL", defaults["base_url"]),
            llm_api_key=os.getenv(defaults["key_env"]),
            offline=os.getenv("RAG_OFFLINE", "0") == "1",
        )
