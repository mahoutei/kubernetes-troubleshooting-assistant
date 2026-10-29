#!/usr/bin/env bash
# Deploy the project to a Hugging Face Docker Space.
# Human gate: requires `hf auth login` (or a configured HF token) and a Space repo.
set -euo pipefail
SPACE="${1:?usage: deploy_space.sh <hf-user>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp -r "$ROOT"/. "$TMP"/
rm -rf "$TMP/.git" "$TMP/.venv" "$TMP/data" "$TMP/.hf-cache" "$TMP/.uv-cache" \
       "$TMP/.pytest_cache" "$TMP/.ruff_cache" "$TMP/evals/cache"
find "$TMP" -name __pycache__ -type d -prune -exec rm -rf {} +

# Keep the project README content; HF only needs the metadata frontmatter on top.
{
  cat <<'EOF'
---
title: Kubernetes Docs RAG
emoji: 📚
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

EOF
  cat "$ROOT/README.md"
} > "$TMP/README.md"

cd "$TMP"
git init -q
git add .
git commit -qm "deploy"
git remote add space "https://huggingface.co/spaces/$SPACE"
git push --force space HEAD:main
