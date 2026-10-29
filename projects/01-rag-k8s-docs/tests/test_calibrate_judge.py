import json

from evals.calibrate_judge import load_pairs, main


def _labels(tmp_path, values):
    path = tmp_path / "calibration.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": f"q{i:03d}", "human_correct": value}) for i, value in enumerate(values))
    )
    return path


def _results(tmp_path, values):
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"id": f"q{i:03d}", "judge": {"correct": value, "invalid": False}}
                    for i, value in enumerate(values)
                ]
            }
        )
    )
    return path


def test_load_pairs_skips_null_and_invalid(tmp_path):
    labels = _labels(tmp_path, [1, None, 0])
    results = tmp_path / "dev.json"
    results.write_text(
        json.dumps(
            {
                "rows": [
                    {"id": "q000", "judge": {"correct": 1}},
                    {"id": "q001", "judge": {"correct": 1}},
                    {"id": "q002", "judge": {"invalid": True}},
                ]
            }
        )
    )
    human, judged, missing = load_pairs(labels, results)
    assert human == [1]
    assert judged == [1]
    assert missing == ["q002"]


def test_perfect_agreement_passes_gate(tmp_path, capsys):
    labels = _labels(tmp_path, [1] * 25)
    results = _results(tmp_path, [1] * 25)
    out = tmp_path / "calibration.json"
    assert main(["--labels", str(labels), "--results", str(results), "--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["n"] == 25
    assert payload["kappa"] == 1.0
    assert payload["gate_pass"] is True


def test_low_agreement_fails_gate(tmp_path):
    labels = _labels(tmp_path, [1] * 25)
    results = _results(tmp_path, [0] * 25)
    out = tmp_path / "calibration.json"
    assert main(["--labels", str(labels), "--results", str(results), "--out", str(out)]) == 1
    assert json.loads(out.read_text())["gate_pass"] is False


def test_too_few_pairs_is_rejected(tmp_path, capsys):
    labels = _labels(tmp_path, [1] * 5)
    results = _results(tmp_path, [1] * 5)
    assert main(["--labels", str(labels), "--results", str(results)]) == 1
    assert "at least 20" in capsys.readouterr().out
