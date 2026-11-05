import numpy as np

from rag.config import Settings
from rag.embed import FakeEmbedder
from rag.index import Index, build_index
from rag.models import Chunk


def chunks():
    return [
        Chunk("a", "pods are ephemeral", "u1", "Pods", "concepts/pods.md", "v1.34"),
        Chunk("b", "services expose pods", "u2", "Services", "concepts/services.md", "v1.34"),
        Chunk("c", "volumes store data", "u3", "Volumes", "concepts/volumes.md", "v1.34"),
    ]


def test_build_and_load_roundtrip(tmp_path):
    settings = Settings(index_dir=str(tmp_path / "index"))
    build_index(chunks(), FakeEmbedder(dim=8), settings)
    index = Index.load(settings)
    assert index.count == 3
    assert index.get("a").url == "u1"
    assert index.dense_search(np.ones(8), k=2)
    assert index.bm25_search("pods ephemeral", k=2)[0][0] == "a"


def test_index_rejects_mismatched_embedding_model(tmp_path):
    from dataclasses import replace

    import pytest

    settings = Settings(index_dir=str(tmp_path / "index"))
    build_index(chunks(), FakeEmbedder(), settings)
    with pytest.raises(ValueError, match="embedding model"):
        Index.load(replace(settings, embed_model="different-model"))


def test_index_rejects_inconsistent_bm25_artifacts(tmp_path):
    import pickle

    import pytest

    settings = Settings(index_dir=str(tmp_path / "index"))
    build_index(chunks(), FakeEmbedder(), settings)
    path = tmp_path / "index" / "bm25.pkl"
    payload = pickle.loads(path.read_bytes())
    payload["ids"][0] = "nonexistent"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="inconsistent"):
        Index.load(settings)


def test_bad_embedding_output_does_not_overwrite_existing_index(tmp_path):
    import pytest

    settings = Settings(index_dir=str(tmp_path / "index"))
    build_index(chunks(), FakeEmbedder(), settings)

    class BadEmbedder:
        def encode(self, texts):
            return np.zeros((1, 8))

    with pytest.raises(ValueError, match="one non-empty vector"):
        build_index(chunks(), BadEmbedder(), settings)
    assert Index.load(settings).count == 3


def test_index_rejects_empty_and_duplicate_chunks(tmp_path):
    import pytest

    settings = Settings(index_dir=str(tmp_path / "index"))
    for invalid in ([], [chunks()[0], chunks()[0]]):
        with pytest.raises(ValueError):
            build_index(invalid, FakeEmbedder(), settings)
    assert not (tmp_path / "index").exists()


def test_legacy_duplicate_rows_are_normalized_without_rebuilding_embeddings(tmp_path):
    import json
    import pickle

    import pytest
    from rank_bm25 import BM25Okapi

    settings = Settings(index_dir=str(tmp_path / "index"))
    index = build_index(chunks(), FakeEmbedder(), settings)
    row = index.table.search().limit(1).to_list()[0]
    index.table.add([row])
    ids = [c.id for c in chunks()] + [row["id"]]
    payload = {"ids": ids, "bm25": BM25Okapi([index.get(i).text.split() for i in ids])}
    (tmp_path / "index" / "bm25.pkl").write_bytes(pickle.dumps(payload))
    meta_path = tmp_path / "index" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["chunks"] = 4
    meta_path.write_text(json.dumps(meta))
    with pytest.warns(UserWarning, match="duplicate chunks"):
        loaded = Index.load(settings)
    assert loaded.count == 3
    results = loaded.dense_search(np.ones(8), 3)
    assert len(results) == len({identifier for identifier, _ in results}) == 3
