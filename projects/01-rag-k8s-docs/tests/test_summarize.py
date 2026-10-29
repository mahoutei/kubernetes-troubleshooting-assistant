import json

from evals.summarize import main, payload_to_markdown, summarize


def _payload(recall):
    return {
        "questions": 15,
        "metrics": {
            "recall@5": recall,
            "recall@10": recall,
            "recall@20": recall,
            "mrr@10": 0.5,
            "ndcg@10": 0.4,
            "p95_latency_ms": 123.0,
            "cost_usd_per_1k": 0.0,
        },
    }


def test_payload_to_markdown_has_header_and_value():
    table = payload_to_markdown("dev-hybrid", _payload(0.75))
    assert table.splitlines()[0].startswith("| config |")
    assert "| dev-hybrid | 15 | 0.750 |" in table


def test_summarize_reads_multiple_results(tmp_path):
    first = tmp_path / "dense.json"
    second = tmp_path / "hybrid.json"
    first.write_text(json.dumps(_payload(0.5)))
    second.write_text(json.dumps(_payload(0.8)))
    table = summarize([first, second])
    assert "| dense | 15 | 0.500 |" in table
    assert "| hybrid | 15 | 0.800 |" in table


def test_missing_metric_renders_dash():
    table = payload_to_markdown("x", {"questions": 1, "metrics": {}})
    assert "—" in table


def test_cli_prints_table(tmp_path, capsys):
    path = tmp_path / "dev-dense.json"
    path.write_text(json.dumps(_payload(0.6)))
    assert main([str(path)]) == 0
    assert "dev-dense" in capsys.readouterr().out
