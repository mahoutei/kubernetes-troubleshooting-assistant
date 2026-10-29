from __future__ import annotations

import re
import time

from rag.config import Settings
from rag.models import Answer, Chunk, Citation

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
        usage={"prompt_tokens": 0, "completion_tokens": 0},
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


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError("LLM API key is not configured")
        from openai import OpenAI

        self.settings = settings
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, int]]:
        response = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return text.strip(), usage


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
        usage=usage,
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
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        latency_ms=0,
    )
