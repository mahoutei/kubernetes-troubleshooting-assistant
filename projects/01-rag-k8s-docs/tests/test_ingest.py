import tarfile

import pytest

from rag.ingest import extract_tarball, iter_markdown


def test_extract_rejects_path_traversal(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tf:
        info = tarfile.TarInfo("../../escape.txt")
        data = b"nope"
        info.size = len(data)
        import io
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="unsafe path"):
        extract_tarball(evil, tmp_path / "out")


def test_iter_markdown_filters_prefixes(tmp_path):
    root = tmp_path / "docs"
    content = root / "content"
    (content / "concepts").mkdir(parents=True)
    (content / "concepts" / "a.md").write_text("# A\n")
    (content / "contribute").mkdir()
    (content / "contribute" / "b.md").write_text("# B\n")
    (content / "reference" / "generated").mkdir(parents=True)
    (content / "reference" / "generated" / "c.md").write_text("# C\n")
    found = [p.name for p in iter_markdown(root)]
    assert found == ["a.md"]
