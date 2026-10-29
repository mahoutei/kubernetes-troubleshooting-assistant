from __future__ import annotations

import hashlib

import numpy as np


class Embedder:
    def __init__(self, model_name: str, show_progress: bool = False) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.show_progress = show_progress

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress,
            batch_size=64,
        )
        return np.asarray(vectors, dtype=np.float32)


class FakeEmbedder:
    """Deterministic embedder for tests; hashes text into a fixed-size vector."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            rows.append([b / 255.0 for b in digest[: self.dim]])
        return np.asarray(rows, dtype=np.float32)
