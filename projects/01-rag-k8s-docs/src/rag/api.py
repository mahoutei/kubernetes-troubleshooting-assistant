from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.config import Settings
from rag.generate import LLMClient, LLMError, answer_question
from rag.models import Answer
from rag.retrieve import CrossEncoderReranker, Retriever

STATIC_DIR = Path(__file__).resolve().parents[2] / "app" / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class CitationOut(BaseModel):
    id: int
    url: str
    heading: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    retrieval: dict
    usage: dict
    latency_ms: int


def _default_retriever(settings: Settings) -> Retriever:
    from rag.embed import Embedder
    from rag.index import Index

    if not Path(settings.index_dir).is_dir():
        raise FileNotFoundError(
            f"index not found: {settings.index_dir}; run scripts/build_index.py"
        )
    return Retriever(
        Index.load(settings),
        Embedder(settings.embed_model),
        CrossEncoderReranker(settings.rerank_model),
        settings,
        top_k_context=settings.top_k_context,
    )


def create_app(settings: Settings, retriever=None, llm=None) -> FastAPI:
    app = FastAPI(title="Kubernetes Docs RAG")
    app.state.settings = settings
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.error = None

    try:
        if app.state.retriever is None:
            app.state.retriever = _default_retriever(settings)
        if app.state.llm is None and not settings.offline:
            app.state.llm = LLMClient(settings)
    except Exception as exc:  # noqa: BLE001 - keep /health alive when artifacts/keys are missing
        app.state.error = str(exc)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return {
            "status": "ok" if app.state.error is None else "degraded",
            "chunks": getattr(app.state.retriever, "index", None).count
            if app.state.retriever is not None and hasattr(app.state.retriever, "index")
            else 0,
            "error": app.state.error,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        if app.state.retriever is None or (app.state.llm is None and not settings.offline):
            raise HTTPException(status_code=503, detail=f"service not ready: {app.state.error}")
        started = time.monotonic()
        retrieved = app.state.retriever.retrieve(request.question)
        if not retrieved:
            raise HTTPException(status_code=404, detail="no documents retrieved")
        chunks = [r.chunk for r in retrieved]
        try:
            answer: Answer = answer_question(request.question, chunks, app.state.llm, settings)
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=f"generation unavailable: {exc}") from exc
        return AskResponse(
            answer=answer.text,
            citations=[CitationOut(**c.__dict__) for c in answer.citations],
            retrieval={
                "chunks_used": len(chunks),
                "top_chunk_id": chunks[0].id,
                **getattr(app.state.retriever, "last_timings", {}),
            },
            usage=answer.usage.as_dict(),
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    return app


try:
    app = create_app(Settings.from_env())
except ValueError:
    app = None
