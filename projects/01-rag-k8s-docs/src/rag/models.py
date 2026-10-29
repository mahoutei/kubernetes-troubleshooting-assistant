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
class Answer:
    text: str
    citations: list[Citation]
    usage: dict[str, int]
    latency_ms: int
