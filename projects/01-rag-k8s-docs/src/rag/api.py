from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.config import Settings
from rag.demos import demo_response, load_demos
from rag.generate import LLMClient, LLMError, answer_question
from rag.github import GitHubLookup
from rag.incident import DiagnosisResponse, IncidentRequest, diagnose
from rag.models import Answer
from rag.retrieve import CrossEncoderReranker, Retriever

STATIC_DIR = Path(os.getenv("RAG_STATIC_DIR", str(Path(__file__).resolve().parents[2] / "app" / "static")))


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
        CrossEncoderReranker(settings.rerank_model) if settings.retrieval_mode == "hybrid_rerank" else None,
        settings,
        top_k_context=settings.top_k_context,
        dense_only=settings.retrieval_mode == "dense",
        use_reranker=settings.retrieval_mode == "hybrid_rerank",
    )


def create_app(settings: Settings, retriever=None, llm=None, github=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        try:
            # Missing credentials should not trigger expensive model initialization.
            if app.state.llm is None and not settings.offline:
                app.state.llm = LLMClient(settings)
            if app.state.retriever is None:
                app.state.retriever = _default_retriever(settings)
        except Exception as exc:  # noqa: BLE001 - allow health diagnostics when startup fails
            app.state.error = str(exc)
        yield

    app = FastAPI(title="Kubernetes Troubleshooting Assistant", lifespan=lifespan)
    app.state.settings = settings
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.error = None
    app.state.github = github if github is not None else GitHubLookup()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def root():
        path = STATIC_DIR / "index.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="UI assets not installed")
        return FileResponse(path)

    @app.get("/health")
    def health():
        return {
            "status": "ok" if app.state.error is None else "degraded",
            "chunks": getattr(getattr(app.state.retriever, "index", None), "count", 0),
            "error": app.state.error,
            "live_diagnosis_ready": not settings.offline and app.state.retriever is not None and app.state.llm is not None,
            "recorded_demos_ready": True,
            "retrieval_mode": settings.retrieval_mode,
        }

    @app.get("/demos")
    def demos():
        return [{"id": d["id"], "title": d["title"], "request": d["request"]} for d in load_demos()]

    @app.get("/demos/{demo_id}", response_model=DiagnosisResponse)
    def recorded(demo_id: str, github_lookup: bool = False):
        try:
            return demo_response(demo_id, github_lookup)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown demo") from exc

    @app.post("/diagnose", response_model=DiagnosisResponse)
    def incident(request: IncidentRequest):
        if settings.offline or app.state.retriever is None or app.state.llm is None:
            raise HTTPException(status_code=503, detail="Live diagnosis unavailable. Use an explicitly labeled recorded demo.")
        try:
            return diagnose(request, app.state.retriever, app.state.llm, app.state.github)
        except LLMError as exc:
            raise HTTPException(status_code=503, detail="Generation unavailable or invalid. Recorded demos remain available.") from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Retrieval unavailable; retry or use a recorded demo.") from exc

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest):
        if app.state.retriever is None or (app.state.llm is None and not settings.offline):
            raise HTTPException(status_code=503, detail=f"service not ready: {app.state.error}")
        started = time.monotonic()
        retrieved = app.state.retriever.retrieve(request.question)
        if not retrieved:
            raise HTTPException(status_code=404, detail="no documents retrieved")
        timings = dict(getattr(app.state.retriever, "last_timings", {}))
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
                **timings,
            },
            usage=answer.usage.as_dict(),
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    return app


app = create_app(Settings.from_env())
