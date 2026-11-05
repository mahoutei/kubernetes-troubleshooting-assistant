from types import SimpleNamespace

import numpy as np
import pytest

from rag.embed import Embedder


def test_encoder_warns_and_records_truncated_inputs():
    embedder = Embedder.__new__(Embedder)
    embedder.model = SimpleNamespace(
        max_seq_length=512,
        tokenizer=lambda texts, **kwargs: {"length": [10, 800]},
        encode=lambda texts, **kwargs: np.ones((len(texts), 4)),
    )
    embedder.show_progress = False
    with pytest.warns(UserWarning, match="1/2 embedding inputs"):
        vectors = embedder.encode(["short", "long"])
    assert vectors.shape == (2, 4)
    assert embedder.last_truncated_count == 1


def test_encoder_does_not_warn_for_inputs_within_limit():
    import warnings

    embedder = Embedder.__new__(Embedder)
    embedder.model = SimpleNamespace(
        max_seq_length=512,
        tokenizer=lambda texts, **kwargs: {"length": [10]},
        encode=lambda texts, **kwargs: np.ones((len(texts), 4)),
    )
    embedder.show_progress = False
    with warnings.catch_warnings(record=True) as caught:
        embedder.encode(["short"])
    assert caught == []
    assert embedder.last_truncated_count == 0


def test_exact_text_reuse_computes_only_missing_vectors():
    from rag.embed import ReusingEmbedder

    calls = []
    model = SimpleNamespace(max_seq_length=512, tokenizer=lambda texts, **kwargs: {"length": [10] * len(texts)})
    def encode(texts):
        calls.append(texts)
        return np.array([[3.0, 4.0] for _ in texts])
    wrapped = ReusingEmbedder(SimpleNamespace(model=model, encode=encode), {"old": [1.0, 2.0]})
    vectors = wrapped.encode(["old", "new", "old"])
    assert calls == [["new"]]
    assert wrapped.reused == 2
    assert vectors.tolist() == [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]
