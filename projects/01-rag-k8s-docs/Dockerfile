# Dockerfile
# Runtime image: pinned uv, runtime deps only, both encoders baked in, index
# built at image build. Docker is not installed on the dev machine; CI/the
# deploy host builds this.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_OFFLINE=0 \
    HF_HOME=/models \
    RAG_INDEX_DIR=/app/data/index

WORKDIR /app

# Pin uv 0.13.6 instead of :latest so image builds are reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# Dependency layer first: src/app changes must not bust `uv sync`.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# BOTH models are pre-downloaded here: the API loads the embedder and the
# reranker at import time and the Space has no shell to warm them later.
RUN uv run --no-sync python - <<'PY'
from sentence_transformers import SentenceTransformer

SentenceTransformer("BAAI/bge-small-en-v1.5")
SentenceTransformer("BAAI/bge-reranker-base")
PY

# Index layer: copy only the builder inputs. app/, evals/ and tests/ are copied
# after this layer so changes there never invalidate the expensive index build.
COPY scripts ./scripts
COPY src ./src
RUN uv run --no-sync python scripts/build_index.py

COPY app ./app

EXPOSE 7860
# --no-sync: never re-resolve/re-install (and never pull dev deps) at startup.
CMD ["uv", "run", "--no-sync", "uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "7860"]
