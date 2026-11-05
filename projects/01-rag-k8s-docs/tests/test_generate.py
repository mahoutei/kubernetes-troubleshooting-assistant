import pytest

from rag.config import Settings
from rag.generate import (
    Endpoint,
    LLMClient,
    LLMError,
    answer_question,
    build_messages,
    extractive_answer,
)
from rag.models import Chunk, Usage


def _fake_client(*, fail=False, text="A Pod is small [1].", prompt_tokens=3, completion_tokens=4):
    class Completions:
        def create(self, **kwargs):
            if fail:
                raise RuntimeError("provider down")
            message = type("Message", (), {"content": text})()
            choice = type("Choice", (), {"message": message})()
            usage = type(
                "Usage",
                (),
                {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            )()
            return type("Response", (), {"choices": [choice], "usage": usage})()

    chat = type("Chat", (), {"completions": Completions()})()
    return type("Client", (), {"chat": chat})()


def _endpoints():
    return [
        Endpoint("groq", "openai/gpt-oss-120b", "https://groq.test", "k1"),
        Endpoint("gemini", "gemini-2.5-flash", "https://gemini.test", "k2"),
    ]


def chunks():
    return [
        Chunk("a", "A Pod is the smallest deployable unit.", "u1", "Pods", "pods.md", "v"),
        Chunk("b", "A Service exposes a set of Pods.", "u2", "Services", "svc.md", "v"),
    ]


def test_prompt_contains_numbered_context_and_refusal_rule():
    messages = build_messages("What is a Pod?", chunks())
    assert "[1]" in messages[1]["content"]
    assert "ONLY" in messages[0]["content"]
    assert "don't know" in messages[0]["content"].lower()


def test_citations_are_parsed_and_out_of_range_dropped():
    class FakeLLM:
        def complete(self, messages, max_tokens):
            return "A Pod is the smallest unit [1]. Ignore [9].", {"prompt_tokens": 1, "completion_tokens": 2}

    answer = answer_question("What is a Pod?", chunks(), FakeLLM(), Settings())
    assert answer.text.startswith("A Pod")
    assert [c.id for c in answer.citations] == [1]
    assert answer.citations[0].url == "u1"
    assert answer.usage.prompt_tokens == 1


def test_llm_falls_back_to_second_provider():
    client = LLMClient(
        Settings(offline=False, llm_fallback_provider="gemini"),
        endpoints=_endpoints(),
        client_factory=lambda endpoint: _fake_client(fail=endpoint.provider == "groq"),
    )
    text, usage = client.complete([{"role": "user", "content": "q"}], 10)
    assert text.startswith("A Pod")
    assert usage == Usage(
        provider="gemini",
        model="gemini-2.5-flash",
        prompt_tokens=3,
        completion_tokens=4,
    )


def test_llm_raises_when_all_providers_fail():
    client = LLMClient(
        Settings(offline=False, llm_fallback_provider="gemini"),
        endpoints=_endpoints(),
        client_factory=lambda endpoint: _fake_client(fail=True),
    )
    with pytest.raises(LLMError, match="all LLM providers failed"):
        client.complete([{"role": "user", "content": "q"}], 10)


def test_llm_skips_provider_without_key():
    endpoints = [
        Endpoint("groq", "openai/gpt-oss-120b", "https://groq.test", None),
        Endpoint("gemini", "gemini-2.5-flash", "https://gemini.test", "k2"),
    ]
    client = LLMClient(
        Settings(offline=False),
        endpoints=endpoints,
        client_factory=lambda endpoint: _fake_client(fail=False),
    )
    _, usage = client.complete([{"role": "user", "content": "q"}], 10)
    assert usage.provider == "gemini"


def test_llm_requires_at_least_one_key():
    endpoints = [Endpoint("groq", "openai/gpt-oss-120b", "https://groq.test", None)]
    with pytest.raises(ValueError, match="no LLM API key"):
        LLMClient(Settings(offline=False), endpoints=endpoints)


def test_offline_mode_is_extractive_and_cited():
    answer = extractive_answer("What is a Pod?", chunks())
    assert "smallest deployable unit" in answer.text
    assert answer.citations[0].id == 1


def test_offline_citation_points_at_the_selected_chunk():
    answer = extractive_answer("What is a Service?", chunks())
    assert answer.text.endswith("[2]")
    assert answer.citations[0].id == 2
    assert answer.citations[0].url == "u2"


@pytest.mark.parametrize("empty", ["", "  ", None])
def test_empty_provider_answer_uses_fallback(empty):
    client = LLMClient(
        Settings(), endpoints=_endpoints(),
        client_factory=lambda endpoint: _fake_client(text=empty if endpoint.provider == "groq" else "valid [1]"),
    )
    answer, usage = client.complete([], 10)
    assert answer == "valid [1]"
    assert usage.provider == "gemini"


def test_all_empty_provider_answers_fail_loudly():
    client = LLMClient(Settings(), endpoints=_endpoints(), client_factory=lambda endpoint: _fake_client(text=""))
    with pytest.raises(LLMError, match="empty answer"):
        client.complete([], 10)
