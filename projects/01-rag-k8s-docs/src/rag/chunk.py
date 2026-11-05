from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from rag.models import Chunk

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


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


def _next_fence(line: str, fence: tuple[str, int] | None) -> tuple[str, int] | None:
    match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
    if not match:
        return fence
    marker, suffix = match.groups()
    if fence is None:
        return (marker[0], len(marker))
    if marker[0] == fence[0] and len(marker) >= fence[1] and not suffix.strip():
        return None
    return fence


def _split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    buf: list[str] = []
    fence = None
    for line in text.splitlines():
        fence = _next_fence(line, fence)
        if fence is None and not line.strip():
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def _mask_fenced_lines(text: str) -> str:
    """Blank out fenced lines so heading scans ignore `#` comments in code."""
    masked: list[str] = []
    fence = None
    for line in text.splitlines(keepends=True):
        previous = fence
        fence = _next_fence(line.rstrip("\r\n"), fence)
        masked.append(re.sub(r"[^\n]", " ", line) if previous is not None or fence is not None else line)
    return "".join(masked)


def chunk_text(
    text: str,
    max_tokens: int,
    overlap: float,
    count_tokens: Callable[[str], int],
) -> list[str]:
    # Keep fenced blocks intact; oversized prose can be split without losing text.
    blocks = []
    for block in _split_blocks(text):
        if count_tokens(block) <= max_tokens or any(_next_fence(line, None) for line in block.splitlines()):
            blocks.append(block)
            continue
        words = list(re.finditer(r"\S+", block))
        start = 0
        while start < len(words):
            low, high = start + 1, len(words)
            while low < high:
                mid = (low + high + 1) // 2
                piece = block[words[start].start():words[mid - 1].end()]
                if count_tokens(piece) <= max_tokens:
                    low = mid
                else:
                    high = mid - 1
            blocks.append(block[words[start].start():words[low - 1].end()])
            start = low
    chunks = []
    current = ""
    for block in blocks:
        candidate = current + "\n\n" + block if current else block
        if current and count_tokens(candidate) > max_tokens:
            chunks.append(current)
            # Never create partial code fences in overlap text.
            tail = ""
            if "```" not in current and "~~~" not in current:
                words = current.split()
                desired = int(max_tokens * overlap)
                low, high = 0, len(words)
                while low < high:
                    mid = (low + high + 1) // 2
                    candidate_tail = " ".join(words[-mid:])
                    if count_tokens(candidate_tail) <= desired and count_tokens(candidate_tail + "\n\n" + block) <= max_tokens:
                        low = mid
                    else:
                        high = mid - 1
                tail = " ".join(words[-low:]) if low else ""
            current = tail + "\n\n" + block if tail else block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


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
    positions = list(HEADING_RE.finditer(_mask_fenced_lines(body)))
    sections: list[tuple[str, str, str]] = []
    preamble = body[:positions[0].start()] if positions else body
    if preamble.strip():
        sections.append(("", meta.get("title", ""), preamble))
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
    return list({chunk.id: chunk for chunk in chunks}.values())
