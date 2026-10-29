from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    url: str
    heading: str
    source_path: str
    version: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class Citation:
    id: int
    url: str
    heading: str
    chunk_id: str


@dataclass(frozen=True)
class Usage:
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    usage: Usage
    latency_ms: int
