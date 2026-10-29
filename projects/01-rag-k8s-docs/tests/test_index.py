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
