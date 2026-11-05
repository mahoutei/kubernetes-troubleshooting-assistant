from training.tool_benchmark import evaluate_backend


def test_domain_benchmark_scores_output_without_gpu_imports():
    cases = [{"id": "domain", "tools": [{"name": "inspect_evidence"}], "user": "Inspect logs", "calls": [{"name": "inspect_evidence", "arguments": {"field": "logs"}}]}]
    def generate(messages):
        assert "Inspect logs" in messages[-1]["content"]
        return '<tool_call>{"name":"inspect_evidence","arguments":{"field":"logs"}}</tool_call>', {"model": "fake"}
    result = evaluate_backend(cases, generate, {"model": "fake"})
    assert result["metrics"]["whole_example_exact_match"] == 1
    assert result["rows"][0]["usage"]["model"] == "fake"


def test_domain_benchmark_records_generation_failures():
    cases = [{"id": "domain", "tools": [], "user": "inspect", "calls": []}]
    def generate(messages):
        raise RuntimeError("test unavailable")
    result = evaluate_backend(cases, generate, {"model": "fake"})
    assert result["rows"][0]["error"] == "RuntimeError"
    assert result["metrics"]["valid_output"] == 0
