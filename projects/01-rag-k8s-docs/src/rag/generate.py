from __future__ import annotations

import re
import time
from dataclasses import dataclass

from rag.config import PROVIDER_DEFAULTS, Settings
from rag.models import Answer, Chunk, Citation, Usage

SYSTEM_PROMPT = """You answer questions about Kubernetes using ONLY the provided documentation excerpts.
Rules:
- Cite every factual claim with [n] matching the excerpt number.
- If the excerpts do not contain the answer, reply exactly: "I don't know based on the provided documentation."
- Do not use outside knowledge.
- Be concise: at most 180 words."""

CITATION_RE = re.compile(r"\[(\d+)\]")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "what", "how", "why", "when", "where", "which", "who",
    "of", "to", "in", "on", "for", "and", "or",
    "do", "does", "did", "can", "could", "should",
    "i", "you", "it", "this", "that",
}


def _terms(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[a-z0-9]+", text)} - STOPWORDS


def _refusal() -> Answer:
    return Answer(
        text="I don't know based on the provided documentation.",
        citations=[],
        usage=Usage(provider="offline", model="extractive"),
        latency_ms=0,
    )


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{i}] {c.url}\nHeading: {c.heading}\n{c.text}" for i, c in enumerate(chunks, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Documentation excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]


@dataclass(frozen=True)
class Endpoint:
    provider: str
    model: str
    base_url: str
    api_key: str | None


class LLMError(RuntimeError):
    """Raised when every configured provider fails."""


def default_endpoints(settings: Settings) -> list[Endpoint]:
    endpoints = [
        Endpoint(
            provider=settings.llm_provider,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    ]
    if settings.llm_fallback_provider:
        defaults = PROVIDER_DEFAULTS[settings.llm_fallback_provider]
        endpoints.append(
            Endpoint(
                provider=settings.llm_fallback_provider,
                model=settings.llm_fallback_model or defaults["model"],
                base_url=settings.llm_fallback_base_url or defaults["base_url"],
                api_key=settings.llm_fallback_api_key,
            )
        )
    return endpoints


def _coerce_usage(raw) -> Usage:
    """Accept a Usage or a legacy token dict from injected clients."""
    if isinstance(raw, Usage):
        return raw
    if isinstance(raw, dict):
        return Usage(
            provider=str(raw.get("provider", "")),
            model=str(raw.get("model", "")),
            prompt_tokens=int(raw.get("prompt_tokens", 0)),
            completion_tokens=int(raw.get("completion_tokens", 0)),
        )
    return Usage()


class LLMClient:
    def __init__(self, settings: Settings, endpoints=None, client_factory=None) -> None:
        self.settings = settings
        self.endpoints = list(endpoints) if endpoints is not None else default_endpoints(settings)
        if not any(endpoint.api_key for endpoint in self.endpoints):
            raise ValueError("no LLM API key is configured for any provider")
        self._factory = client_factory or self._openai_client
        self._clients: dict[str, object] = {}

    @staticmethod
    def _openai_client(endpoint: Endpoint):
        from openai import OpenAI

        return OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key)

    def _client(self, endpoint: Endpoint):
        if endpoint.provider not in self._clients:
            self._clients[endpoint.provider] = self._factory(endpoint)
        return self._clients[endpoint.provider]

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, Usage]:
        errors: list[str] = []
        for endpoint in self.endpoints:
            if not endpoint.api_key:
                errors.append(f"{endpoint.provider}: missing API key")
                continue
            try:
                response = self._client(endpoint).chat.completions.create(
                    model=endpoint.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - outage, retired model, rate limit, bad key
                errors.append(f"{endpoint.provider}/{endpoint.model}: {exc}")
                continue
            text = response.choices[0].message.content or ""
            usage = response.usage
            return text.strip(), Usage(
                provider=endpoint.provider,
                model=endpoint.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        raise LLMError("all LLM providers failed: " + "; ".join(errors))


def _citations(text: str, chunks: list[Chunk]) -> list[Citation]:
    seen: set[int] = set()
    result: list[Citation] = []
    for raw in CITATION_RE.findall(text):
        index = int(raw)
        if index in seen or not 1 <= index <= len(chunks):
            continue
        seen.add(index)
        chunk = chunks[index - 1]
        result.append(Citation(index, chunk.url, chunk.heading, chunk.id))
    return sorted(result, key=lambda c: c.id)


def answer_question(question: str, chunks: list[Chunk], llm, settings: Settings) -> Answer:
    if settings.offline:
        return extractive_answer(question, chunks)
    started = time.monotonic()
    text, usage = llm.complete(build_messages(question, chunks), settings.max_answer_tokens)
    return Answer(
        text=text,
        citations=_citations(text, chunks),
        usage=_coerce_usage(usage),
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def extractive_answer(question: str, chunks: list[Chunk]) -> Answer:
    if not chunks:
        return _refusal()
    query = _terms(question)
    best_idx, best = max(
        enumerate(chunks, start=1),
        key=lambda pair: len(query & _terms(pair[1].text)),
    )
    if not best.text.strip():
        return _refusal()
    sentences = re.split(r"(?<=[.!?])\s+", best.text.strip())
    text = " ".join(sentences[:2]).strip() + f" [{best_idx}]"
    return Answer(
        text=text,
        citations=[Citation(best_idx, best.url, best.heading, best.id)],
        usage=Usage(provider="offline", model="extractive"),
        latency_ms=0,
    )
