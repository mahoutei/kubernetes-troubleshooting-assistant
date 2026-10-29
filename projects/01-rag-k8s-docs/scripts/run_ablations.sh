#!/usr/bin/env bash
# Retrieval ablations: chunk size x retrieval mode, dev split, retrieval only.
# Results land in evals/results/ablation-<size>-<mode>.json (+ .md).
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES="${RAG_ABLATION_CHUNK_SIZES:-300 500 800}"
MODES="${RAG_ABLATION_MODES:-dense hybrid hybrid_rerank}"
LIMIT="${RAG_ABLATION_LIMIT:-0}"

limit_args=()
if [ "$LIMIT" != "0" ]; then
  limit_args=(--limit "$LIMIT")
fi

for size in $SIZES; do
  echo "== building index: chunk_tokens=$size =="
  uv run python scripts/build_index.py \
    --chunk-tokens "$size" \
    --index-dir "data/index-$size" \
    "${limit_args[@]}"
done

mkdir -p evals/results
for size in $SIZES; do
  for mode in $MODES; do
    echo "== eval: chunk_tokens=$size mode=$mode =="
    uv run python evals/run_eval.py \
      --split dev \
      --retrieval-only \
      --top-k 20 \
      --mode "$mode" \
      --index-dir "data/index-$size" \
      --output "evals/results/ablation-${size}-${mode}.json"
  done
done

echo "== summary =="
uv run python -m evals.summarize evals/results/ablation-*.json
