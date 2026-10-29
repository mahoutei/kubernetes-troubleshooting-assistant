from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from rag.models import Chunk

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[match.end():]


def _slug(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", heading.lower())
    return re.sub(r"\s+", "-", slug.strip())


def _url(source_path: str, anchor_heading: str) -> str:
    rel = source_path.removesuffix(".md")
    rel = rel.replace("/_index", "").strip("/")
    base = f"https://kubernetes.io/docs/{rel}/"
    return f"{base}#{_slug(anchor_heading)}" if anchor_heading else base


def _split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
        if not in_fence and not line.strip():
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def chunk_text(
    text: str,
    max_tokens: int,
    overlap: float,
    count_tokens: Callable[[str], int],
) -> list[str]:
    blocks = _split_blocks(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    overlap_tokens = int(max_tokens * overlap)

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current))
        tail: list[str] = []
        tail_tokens = 0
        for block in reversed(current):
            if tail_tokens >= overlap_tokens:
                break
            tail.insert(0, block)
            tail_tokens += count_tokens(block)
        current = tail
        current_tokens = tail_tokens

    for block in blocks:
        tokens = count_tokens(block)
        if current and current_tokens + tokens > max_tokens:
            flush()
        current.append(block)
        current_tokens += tokens

    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]


def parse_markdown(
    text: str,
    source_path: str,
    version: str,
    chunk_tokens: int = 500,
    overlap: float = 0.15,
    count_tokens: Callable[[str], int] | None = None,
) -> list[Chunk]:
    count_tokens = count_tokens or (lambda s: len(s.split()))
    meta, body = _frontmatter(text)
    if meta.get("draft", "false").lower() == "true":
        return []
    body = SHORTCODE_RE.sub("", body)

    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    positions = list(HEADING_RE.finditer(body))
    sections: list[tuple[str, str, str]] = []
    if not positions:
        sections.append(("", meta.get("title", ""), body))
    for i, match in enumerate(positions):
        level = len(match.group(1))
        title = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        start = match.end()
        end = positions[i + 1].start() if i + 1 < len(positions) else len(body)
        heading_path = " > ".join(t for _, t in heading_stack)
        sections.append((heading_path, title, body[start:end].strip()))

    for heading_path, leaf_title, section in sections:
        if not section:
            continue
        for piece in chunk_text(section, chunk_tokens, overlap, count_tokens):
            digest = hashlib.sha256(f"{source_path}:{heading_path}:{piece}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=digest,
                    text=piece,
                    url=_url(source_path, leaf_title),
                    heading=heading_path or meta.get("title", ""),
                    source_path=source_path,
                    version=version,
                )
            )
    return chunks
