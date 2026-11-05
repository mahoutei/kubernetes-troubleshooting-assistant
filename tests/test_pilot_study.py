import copy
import json

import pytest

from scripts.pilot_study import main, summarize, template, validate, write_new


def observation(condition="manual", **changes):
    return {"participant_id": condition, "case_id": "case-a", "condition": condition,
            "order": 1, "evidence_mode": "manual" if condition == "manual" else "live",
            "elapsed_seconds": 120 if condition == "manual" else 90, "completed": True,
            "review": None, "artifact_reference": "saved-review.txt", **changes}


def test_template_only_dev_no_repeat_and_counterbalanced():
    rows = template([{"id": "a", "split": "dev"}, {"id": "b", "split": "dev"}, {"id": "secret", "split": "test"}])
    assert len(rows) == 4
    assert {r["case_id"] for r in rows} == {"a", "b"}
    assert len({(r["participant_id"], r["case_id"]) for r in rows}) == 4
    for case in ("a", "b"):
        assert {r["condition"] for r in rows if r["case_id"] == case} == {"manual", "assistant"}
    assert all(r["elapsed_seconds"] is None and r["review"] is None for r in rows)
    with pytest.raises(ValueError):
        validate(rows)


@pytest.mark.parametrize("bad", [None, True, 0, -1, float("nan"), float("inf"), "120"])
def test_rejects_invalid_timing(bad):
    with pytest.raises(ValueError):
        summarize([observation(elapsed_seconds=bad)])


def test_no_duplicate_participant_case_even_across_conditions():
    with pytest.raises(ValueError, match="repeat"):
        summarize([observation(), observation("assistant", participant_id="manual", order=2)])


def test_missing_reviews_remain_unknown_and_negative_differences_survive():
    result = summarize([observation(), observation("assistant", elapsed_seconds=180)])
    assert result["conditions"]["assistant"]["acceptable_rate_among_reviewed"] is None
    assert result["median_common_case_difference_seconds"] == -60


def test_fake_and_recorded_cannot_be_business_evidence():
    for mode in ("fake", "recorded_demo"):
        result = summarize([observation(), observation("assistant", evidence_mode=mode)])
        assert result["conditions"]["assistant"]["attempts"] == 0
        assert result["excluded_by_mode"] == {mode: 1}
        assert result["median_common_case_difference_seconds"] is None


def test_reviewed_outcome_requires_completion_correctness_support_and_no_unsupported_claims():
    good = {"diagnosis_correct": True, "citation_support": True, "unsupported_claims": 0, "reviewer": "reviewer-a"}
    row = observation(review=good)
    assert summarize([row])["conditions"]["manual"]["acceptable_rate_among_reviewed"] == 1
    for change in ({"diagnosis_correct": False}, {"citation_support": False}, {"unsupported_claims": 1}):
        modified = copy.deepcopy(row)
        modified["review"].update(change)
        assert summarize([modified])["conditions"]["manual"]["acceptable_rate_among_reviewed"] == 0
    assert summarize([observation(review=good, completed=False)])["conditions"]["manual"]["acceptable_rate_among_reviewed"] == 0
    with pytest.raises(ValueError):
        summarize([observation(review={**good, "unsupported_claims": True})])


def test_cli_template_and_no_overwrite(tmp_path):
    path = tmp_path / "worksheet.jsonl"
    main(["init", "--output", str(path)])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 20
    with pytest.raises(FileExistsError):
        write_new(path, "replacement")
    report = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        main(["summarize", "--input", str(path), "--output", str(report)])
    assert not report.exists()


def test_cli_summarizes_real_format_and_preserves_existing_report(tmp_path):
    source = tmp_path / "observations.jsonl"
    output = tmp_path / "report.json"
    source.write_text("\n".join(json.dumps(r) for r in [observation(), observation("assistant")]))
    main(["summarize", "--input", str(source), "--output", str(output)])
    assert json.loads(output.read_text())["median_common_case_difference_seconds"] == 30
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        main(["summarize", "--input", str(source), "--output", str(output)])
    assert output.read_bytes() == original


def test_unique_order_and_explicit_review_identity_required():
    with pytest.raises(ValueError, match="order"):
        summarize([observation(), observation(case_id="case-b")])
    with pytest.raises(ValueError, match="reviewer"):
        summarize([observation(review={"diagnosis_correct": True, "citation_support": True, "unsupported_claims": 0, "reviewer": ""})])
    with pytest.raises(TypeError):
        summarize([None])
