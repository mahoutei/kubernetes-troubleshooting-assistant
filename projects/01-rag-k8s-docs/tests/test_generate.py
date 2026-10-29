from rag.config import Settings
from rag.generate import answer_question, build_messages, extractive_answer
from rag.models import Chunk


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
    assert answer.usage["prompt_tokens"] == 1


def test_offline_mode_is_extractive_and_cited():
    answer = extractive_answer("What is a Pod?", chunks())
    assert "smallest deployable unit" in answer.text
    assert answer.citations[0].id == 1
