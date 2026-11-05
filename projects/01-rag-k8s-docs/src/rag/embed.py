from __future__ import annotations

import hashlib
import warnings

import numpy as np


class Embedder:
    def __init__(self, model_name: str, show_progress: bool = False) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.show_progress = show_progress
        self.last_truncated_count = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        lengths = self.model.tokenizer(
            texts, add_special_tokens=True, truncation=False, padding=False, return_length=True,
            verbose=False,
        )["length"]
        self.last_truncated_count = sum(length > self.model.max_seq_length for length in lengths)
        if self.last_truncated_count:
            warnings.warn(
                f"{self.last_truncated_count}/{len(texts)} embedding inputs exceed "
                f"the encoder limit of {self.model.max_seq_length} tokens and will be truncated; "
                "reduce chunk size or use a longer-context encoder for complete coverage",
                UserWarning, stacklevel=2,
            )
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


class ReusingEmbedder:
    """Opt-in exact-text reuse for an unchanged named encoder; records its provenance."""
    def __init__(self, embedder, cache):
        self.embedder = embedder
        self.model = embedder.model
        self.cache = cache
        self.reused = 0
        self.last_truncated_count = 0

    def encode(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        self.reused = len(texts) - sum(text not in self.cache for text in texts)
        print(f"Reusing {self.reused}/{len(texts)} exact-text vectors; embedding {len(missing)} unique new texts", flush=True)
        if missing:
            for text, vector in zip(missing, self.embedder.encode(missing), strict=True):
                self.cache[text] = vector
        lengths = self.model.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False, return_length=True, verbose=False)["length"]
        self.last_truncated_count = sum(length > self.model.max_seq_length for length in lengths)
        return np.asarray([self.cache[text] for text in texts], dtype=np.float32)
