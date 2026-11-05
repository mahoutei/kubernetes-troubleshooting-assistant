from __future__ import annotations

import json
import pickle
import warnings
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
        # Legacy indexes may contain repeated rows for identical chunks.
        extra = self.table.count_rows() - self.count
        rows = self.table.search(vector.tolist()).limit(k + extra).to_list()
        unique: dict[str, float] = {}
        for row in rows:
            unique.setdefault(row["id"], float(row["_distance"]))
        return list(unique.items())[:k]

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    def close(self) -> None:
        self.table = None

    @classmethod
    def load(cls, settings: Settings) -> Index:
        index_dir = Path(settings.index_dir)
        meta = json.loads((index_dir / "meta.json").read_text())
        if meta.get("embed_model") != settings.embed_model:
            raise ValueError("index embedding model differs from settings; rebuild the index")
        db = lancedb.connect(str(index_dir / "lancedb"))
        table = db.open_table("chunks")
        with (index_dir / "bm25.pkl").open("rb") as f:
            payload = pickle.load(f)
        rows = table.search().select(["id", "text", "url", "heading", "source_path", "version"]).limit(table.count_rows()).to_arrow().to_pylist()
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
        ids = payload["ids"]
        if (len(rows) != len(ids) or set(ids) != set(chunks)
                or payload["bm25"].corpus_size != len(ids) or meta.get("chunks") != len(ids)):
            raise ValueError("inconsistent index artifacts; rebuild the index")
        for row in rows:
            if any(getattr(chunks[row["id"]], key) != value for key, value in row.items()):
                raise ValueError("conflicting rows share a chunk id; rebuild the index")
        bm25 = payload["bm25"]
        if len(ids) != len(set(ids)):
            warnings.warn(
                "legacy index contains duplicate chunks; deduplicating retrieval in memory. "
                "Rebuild the index and rerun evaluations before reporting new results",
                UserWarning, stacklevel=2,
            )
            ids = list(dict.fromkeys(ids))
            bm25 = BM25Okapi([_tokenize(chunks[chunk_id].text) for chunk_id in ids])
        return cls(table, bm25, ids, chunks, settings)


def build_index(chunks: list[Chunk], embedder, settings: Settings) -> Index:
    if not chunks:
        raise ValueError("cannot build an empty index")
    if len({c.id for c in chunks}) != len(chunks):
        raise ValueError("chunk ids must be unique")
    vectors = np.asarray(embedder.encode([c.text for c in chunks]))
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
        raise ValueError("embedder must return one non-empty vector per chunk")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding vectors must be finite")
    index_dir = Path(settings.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
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
        for c, vec in zip(chunks, vectors, strict=True)
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
        "chunker_version": 3,
        "embedding_max_seq_length": getattr(getattr(embedder, "model", None), "max_seq_length", None),
        "truncated_chunks": getattr(embedder, "last_truncated_count", None),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return Index(table, bm25, ids, {c.id: c for c in chunks}, settings)
