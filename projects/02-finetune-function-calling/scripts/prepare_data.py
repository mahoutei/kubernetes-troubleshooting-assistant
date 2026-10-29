from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from transformers import AutoTokenizer

from src.data import (
    DATASET_FILES,
    DATASET_REPO,
    DATASET_REVISION,
    assert_split_integrity,
    dataset_url,
    load_examples,
    split_examples,
    verify_checksum,
    write_jsonl,
)
from src.prompts import render_example

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def download(filename: str) -> Path:
    dest = RAW / filename
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", dataset_url(filename), follow_redirects=True, timeout=180) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    return dest


def verify(path: Path, filename: str) -> dict:
    expected = DATASET_FILES[filename]
    digest = verify_checksum(path, filename)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected["rows"]:
        raise ValueError(f"{filename}: expected {expected['rows']} rows, got {len(rows)}")
    return {"sha256": digest, "bytes": path.stat().st_size, "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    files = [download(filename) for filename in DATASET_FILES]
    manifest = {"repo": DATASET_REPO, "revision": DATASET_REVISION, "files": {}}
    for path in files:
        manifest["files"][path.name] = verify(path, path.name)

    raw_rows = sum(item["rows"] for item in manifest["files"].values())
    examples = load_examples(files)
    if not examples:
        raise ValueError("no examples parsed; refusing to write an empty dataset")
    splits = split_examples(examples)
    split_counts = assert_split_integrity(examples, splits)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kept_counts: dict[str, int] = {}
    for name, items in splits.items():
        kept = [
            example
            for example in items
            if len(tokenizer(render_example(example, tokenizer))["input_ids"]) <= args.max_seq_length
        ]
        if not kept:
            raise ValueError(f"{name} split is empty after length filtering; raise --max-seq-length")
        write_jsonl(kept, PROCESSED / f"{name}.jsonl")
        kept_counts[name] = len(kept)

    manifest.update(
        raw_rows=raw_rows,
        parsed_examples=len(examples),
        skipped_or_duplicate=raw_rows - len(examples),
        split_counts=split_counts,
        length_filtered=kept_counts,
        seed=42,
        ratios=[0.8, 0.1, 0.1],
        max_seq_length=args.max_seq_length,
    )
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"raw_rows": raw_rows, "parsed": len(examples), "split": split_counts, "kept": kept_counts}, indent=2))


if __name__ == "__main__":
    main()
