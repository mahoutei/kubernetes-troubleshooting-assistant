from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunk import parse_markdown
from rag.config import Settings
from rag.embed import Embedder
from rag.index import build_index
from rag.ingest import download_and_extract, iter_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="max markdown files (0 = all)")
    parser.add_argument("--chunk-tokens", type=int, default=500)
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()

    started = time.monotonic()
    base = Settings(chunk_tokens=args.chunk_tokens, index_dir=args.index_dir)
    docs_root = download_and_extract(base)
    files = list(iter_markdown(docs_root))
    if args.limit:
        files = files[: args.limit]
    print(f"docs root: {docs_root}", flush=True)
    print(f"files: {len(files)}", flush=True)

    chunks = []
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base.embed_model)
    count_tokens = lambda s: len(tokenizer.encode(s, add_special_tokens=False))
    for path in files:
        rel = path.relative_to(docs_root / "content").as_posix()
        chunks.extend(
            parse_markdown(
                path.read_text(encoding="utf-8"),
                rel,
                "release-1.34",
                chunk_tokens=base.chunk_tokens,
                overlap=base.chunk_overlap,
                count_tokens=count_tokens,
            )
        )
    if not chunks:
        raise SystemExit("no chunks produced")
    print(f"chunks: {len(chunks)} (chunk_tokens={base.chunk_tokens})", flush=True)

    print("embedding and writing the index...", flush=True)
    embedder = Embedder(base.embed_model, show_progress=True)
    build_index(chunks, embedder, base)
    print(
        f"indexed {len(chunks)} chunks from {len(files)} files into {base.index_dir} "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
