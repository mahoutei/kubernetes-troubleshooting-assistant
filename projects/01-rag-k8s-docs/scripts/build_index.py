from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunk import parse_markdown
from rag.config import Settings
from rag.embed import Embedder, ReusingEmbedder
from rag.index import build_index
from rag.ingest import download_and_extract, iter_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="max markdown files (0 = all)")
    parser.add_argument("--chunk-tokens", type=int, default=500)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--reuse-index", action="append", default=[], help="opt-in reuse of exact-text vectors from the same named encoder; records legacy provenance")
    args = parser.parse_args()

    from rag import chunk as chunk_module

    chunker_sha256 = hashlib.sha256(Path(chunk_module.__file__).read_bytes()).hexdigest()
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
    encoder_budget = min(510, tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=False))
    if base.chunk_tokens > encoder_budget:
        print(f"Capping requested chunk size {base.chunk_tokens} to encoder budget {encoder_budget}", flush=True)
        base = replace(base, chunk_tokens=encoder_budget)
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
    reused_sources = []
    if args.reuse_index:
        import lancedb

        cache = {}
        for old_path in args.reuse_index:
            old_meta = json.loads((Path(old_path) / "meta.json").read_text())
            if old_meta["embed_model"] != base.embed_model:
                raise ValueError("cannot reuse vectors from a different encoder")
            table = lancedb.connect(str(Path(old_path) / "lancedb")).open_table("chunks")
            for row in table.search().select(["text", "vector"]).limit(table.count_rows()).to_list():
                cache.setdefault(row["text"], row["vector"])
            reused_sources.append({"path": old_path, "metadata": old_meta})
        embedder = ReusingEmbedder(embedder, cache)
    build_index(chunks, embedder, base)
    metadata_path = Path(base.index_dir) / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(chunker_sha256=chunker_sha256, requested_chunk_tokens=args.chunk_tokens, encoder_token_budget=encoder_budget, reused_vectors=getattr(embedder, "reused", 0), reuse_sources=reused_sources, reuse_note="Opt-in exact text/model-ID reuse; legacy indexes do not pin encoder revision" if reused_sources else None)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(
        f"indexed {len(chunks)} chunks from {len(files)} files into {base.index_dir} "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
