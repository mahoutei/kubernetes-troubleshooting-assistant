from __future__ import annotations

import os
from dataclasses import dataclass, field

DOCS_COMMIT = "47132eb5367cfdcba0a3b33a0b3ffc81185bd7e8"
DOCS_URL = f"https://github.com/kubernetes/website/archive/{DOCS_COMMIT}.tar.gz"
DOCS_PREFIX = f"website-{DOCS_COMMIT}/content/en/docs"

# `llama-3.3-70b-versatile` was retired by Groq on 2026-08-16; gpt-oss-120b is
# the documented replacement. Keep provider model IDs here, not inline.
PROVIDER_DEFAULTS = {
    "groq": {
        "model": "openai/gpt-oss-120b",
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
    llm_api_key: str | None = field(default=None, repr=False)
    llm_fallback_provider: str | None = None
    llm_fallback_model: str | None = None
    llm_fallback_base_url: str | None = None
    llm_fallback_api_key: str | None = field(default=None, repr=False)
    offline: bool = False
    max_answer_tokens: int = 600

    def __post_init__(self) -> None:
        if not 100 <= self.chunk_tokens <= 2000:
            raise ValueError("chunk_tokens must be between 100 and 2000")
        if not 0.0 <= self.chunk_overlap < 1.0:
            raise ValueError("chunk_overlap must be in [0, 1)")
        if self.top_k_context > self.top_k_rerank:
            raise ValueError("top_k_context cannot exceed top_k_rerank")
        if self.llm_provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"unknown llm_provider: {self.llm_provider}")
        if (
            self.llm_fallback_provider is not None
            and self.llm_fallback_provider not in PROVIDER_DEFAULTS
        ):
            raise ValueError(f"unknown llm_fallback_provider: {self.llm_fallback_provider}")
        if self.llm_fallback_provider == self.llm_provider:
            raise ValueError("llm_fallback_provider must differ from llm_provider")

    @classmethod
    def from_env(cls) -> Settings:
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(PROVIDER_DEFAULTS)}")
        defaults = PROVIDER_DEFAULTS[provider]

        fallback = os.getenv("LLM_FALLBACK_PROVIDER", "gemini").strip().lower()
        if fallback in {"", "none", "off"}:
            fallback = None
        if fallback is not None and fallback not in PROVIDER_DEFAULTS:
            raise ValueError(
                f"LLM_FALLBACK_PROVIDER must be one of {sorted(PROVIDER_DEFAULTS)} or 'none'"
            )
        fallback_defaults = PROVIDER_DEFAULTS[fallback] if fallback else {}
        if fallback == provider:
            fallback = None

        return cls(
            docs_dir=os.getenv("RAG_DOCS_DIR", "data/docs"),
            index_dir=os.getenv("RAG_INDEX_DIR", "data/index"),
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", defaults["model"]),
            llm_base_url=os.getenv("LLM_BASE_URL", defaults["base_url"]),
            llm_api_key=os.getenv(defaults["key_env"]),
            llm_fallback_provider=fallback,
            llm_fallback_model=(
                os.getenv("LLM_FALLBACK_MODEL", fallback_defaults.get("model"))
                if fallback
                else None
            ),
            llm_fallback_base_url=(
                os.getenv("LLM_FALLBACK_BASE_URL", fallback_defaults.get("base_url"))
                if fallback
                else None
            ),
            llm_fallback_api_key=(
                os.getenv(fallback_defaults["key_env"]) if fallback else None
            ),
            offline=os.getenv("RAG_OFFLINE", "0") == "1",
        )
