#!/usr/bin/env bash
# Deploy the project to a Hugging Face Docker Space.
# Human gate: requires `hf auth login` (or a configured HF token) and a Space repo.
set -euo pipefail
SPACE="${1:?usage: deploy_space.sh <hf-user>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Stage only deployable files; local secrets and caches never enter the Space repo.
tar -C "$ROOT" --exclude=__pycache__ -cf - \
  Dockerfile .dockerignore pyproject.toml uv.lock ATTRIBUTIONS.md src app scripts \
  | tar -C "$TMP" -xf -

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

if [ "${2:-}" = "--dry-run" ]; then
  echo "Deployment staging verified (no push):"
  find "$TMP" -type f | sed "s|$TMP/||" | sort
  exit 0
fi

cd "$TMP"
git init -q
git add .
git commit -qm "deploy"
git remote add space "https://huggingface.co/spaces/$SPACE"
git push --force space HEAD:main
