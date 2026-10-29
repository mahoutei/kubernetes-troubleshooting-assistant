from evals.schema import EvalQuestion, GoldSource, validate_questions


def q(qid, split, category, sources=True):
    return EvalQuestion(
        qid, split, category, "question", "answer",
        [GoldSource("u", "h")] if sources else [],
        ["fact"] if category != "unanswerable" else [],
    )


def test_validator_flags_duplicate_ids():
    questions = [q("x", "dev", "single-hop") for _ in range(30)]
    questions += [q("x", "test", "single-hop") for _ in range(30)]
    errors = validate_questions(questions)
    assert "duplicate question ids" in errors


def test_validator_flags_missing_sources():
    questions = [q(f"d{i}", "dev", "single-hop", sources=False) for i in range(30)]
    questions += [q(f"t{i}", "test", "single-hop", sources=False) for i in range(30)]
    errors = validate_questions(questions)
    assert any("missing gold sources" in e for e in errors)
