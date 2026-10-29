from __future__ import annotations

import json
import pickle
from pathlib import Path

import lancedb
import numpy as np
from rank_bm25 import BM25Okapi

from rag.config import Settings
from rag.models import Chunk


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


class Index:
    def __init__(self, table, bm25: BM25Okapi, ids: list[str], chunks: dict[str, Chunk], settings: Settings):
        self.table = table
        self.bm25 = bm25
        self.ids = ids
        self.chunks = chunks
        self.settings = settings

    @property
    def count(self) -> int:
        return len(self.ids)

    def get(self, chunk_id: str) -> Chunk:
        return self.chunks[chunk_id]

    def dense_search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        rows = self.table.search(vector.tolist()).limit(k).to_list()
        return [(row["id"], float(row.get("_distance", 0.0))) for row in rows]

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    def close(self) -> None:
        self.table = None

    @classmethod
    def load(cls, settings: Settings) -> Index:
        index_dir = Path(settings.index_dir)
        db = lancedb.connect(str(index_dir / "lancedb"))
        table = db.open_table("chunks")
        with (index_dir / "bm25.pkl").open("rb") as f:
            payload = pickle.load(f)
        rows = table.to_arrow().to_pylist()
        chunks = {
            row["id"]: Chunk(
                id=row["id"],
                text=row["text"],
                url=row["url"],
                heading=row["heading"],
                source_path=row["source_path"],
                version=row["version"],
            )
            for row in rows
        }
        return cls(table, payload["bm25"], payload["ids"], chunks, settings)


def build_index(chunks: list[Chunk], embedder, settings: Settings) -> Index:
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectors = embedder.encode([c.text for c in chunks])
    rows = [
        {
            "id": c.id,
            "text": c.text,
            "url": c.url,
            "heading": c.heading,
            "source_path": c.source_path,
            "version": c.version,
            "vector": vec.tolist(),
        }
        for c, vec in zip(chunks, vectors)
    ]
    db = lancedb.connect(str(index_dir / "lancedb"))
    table = db.create_table("chunks", data=rows, mode="overwrite")

    ids = [c.id for c in chunks]
    bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump({"ids": ids, "bm25": bm25}, f)

    meta = {
        "chunks": len(chunks),
        "embed_model": settings.embed_model,
        "docs_url": settings.docs_url,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return Index(table, bm25, ids, {c.id: c for c in chunks}, settings)
