from evals.judge import Judge, cache_key, parse_judgement

VALID = '{"correct": 1, "faithful": 1, "citation_support": 1, "refusal_correct": null, "rationale": "ok"}'


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages, max_tokens):
        self.calls += 1
        return self.replies.pop(0), {}


def test_cache_key_is_stable():
    assert cache_key("q", "a", "ref", "c") == cache_key("q", "a", "ref", "c")
    assert cache_key("q", "a", "ref", "c") != cache_key("q", "a", "ref", "d")


def test_judge_parses_json_and_caches(tmp_path):
    llm = FakeLLM([VALID])
    judge = Judge(llm, cache_dir=tmp_path)
    result = judge.score("q", "candidate", "reference", "ctx")
    assert result["correct"] == 1
    assert result["rationale"] == "ok"
    assert len(list(tmp_path.glob("*.json"))) == 1

    cached = judge.score("q", "candidate", "reference", "ctx")
    assert cached == result
    assert llm.calls == 1


def test_invalid_json_is_retried_then_marked_invalid(tmp_path):
    llm = FakeLLM(["not json", "still not json"])
    result = Judge(llm, cache_dir=tmp_path).score("q", "a", "ref", "ctx")
    assert result["invalid"] is True
    assert result["correct"] is None
    assert llm.calls == 2
    assert not list(tmp_path.glob("*.json"))


def test_retry_recovers_with_valid_json(tmp_path):
    llm = FakeLLM(["prose", VALID])
    result = Judge(llm, cache_dir=tmp_path).score("q", "a", "ref", "ctx")
    assert result["correct"] == 1
    assert llm.calls == 2


def test_parse_rejects_out_of_range_values():
    assert parse_judgement(VALID)["correct"] == 1
    try:
        parse_judgement('{"correct": 2, "faithful": 1, "citation_support": 1}')
    except ValueError as exc:
        assert "correct" in str(exc)
    else:
        raise AssertionError("expected ValueError")
