from __future__ import annotations

import tarfile
from collections.abc import Iterator
from pathlib import Path

import httpx

from rag.config import Settings

INCLUDE_PREFIXES = ("concepts/", "tasks/", "tutorials/", "reference/")
EXCLUDE_PREFIXES = ("reference/generated/",)


def _is_safe_member(name: str, dest: Path) -> bool:
    target = (dest / name).resolve()
    return target.is_relative_to(dest.resolve())


def extract_tarball(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not _is_safe_member(member.name, dest):
                raise ValueError(f"unsafe path in archive: {member.name}")
        tf.extractall(dest, filter="data")
    return dest


def download_and_extract(settings: Settings) -> Path:
    dest = Path(settings.docs_dir)
    marker = dest / ".complete"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "docs.tar.gz"
    with httpx.stream("GET", settings.docs_url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with archive.open("wb") as f:
            for block in r.iter_bytes():
                f.write(block)
    extract_tarball(archive, dest)
    (dest / settings.docs_prefix).rename(dest / "content")
    marker.write_text("ok")
    archive.unlink(missing_ok=True)
    return dest


def iter_markdown(docs_root: Path) -> Iterator[Path]:
    content = docs_root / "content"
    for path in sorted(content.rglob("*.md")):
        rel = path.relative_to(content).as_posix()
        if not rel.startswith(INCLUDE_PREFIXES):
            continue
        if rel.startswith(EXCLUDE_PREFIXES):
            continue
        yield path
