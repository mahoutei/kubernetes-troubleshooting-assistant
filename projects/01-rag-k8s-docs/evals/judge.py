from __future__ import annotations

import hashlib
import json
from pathlib import Path

RUBRIC_VERSION = "v1"
# The judge deliberately uses a different model family from the generator so a
# single model cannot grade its own output. Override with --judge-model.
JUDGE_PROVIDER = "gemini"
JUDGE_MODEL = "gemini-2.5-flash"
SCORE_FIELDS = ("correct", "faithful", "citation_support")

JUDGE_PROMPT = """You are grading a RAG answer about Kubernetes documentation.
Return ONLY a JSON object with integer 0/1 fields:
- correct: does the answer cover the reference facts?
- faithful: is every claim supported by the context?
- citation_support: do the cited context items support the claims?
- refusal_correct: 1 if the question is unanswerable and the answer refuses, 0 if it hallucinates, null otherwise.
Also include a one-sentence rationale.
Example: {"correct": 1, "faithful": 1, "citation_support": 1, "refusal_correct": null, "rationale": "Matches the reference."}"""

RETRY_PROMPT = "Your previous reply was not valid. Return only the JSON object, with integer 0/1 values."


def cache_key(question: str, candidate: str, reference: str, context: str) -> str:
    payload = f"{RUBRIC_VERSION}\n{question}\n{candidate}\n{reference}\n{context}"
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_judgement(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response contains no JSON object")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise TypeError("judge response is not a JSON object")
    result: dict = {}
    for field in SCORE_FIELDS:
        value = data.get(field)
        if value not in (0, 1):
            raise ValueError(f"judge field {field!r} must be 0 or 1")
        result[field] = int(value)
    refusal = data.get("refusal_correct")
    if refusal not in (0, 1, None):
        raise ValueError("judge field 'refusal_correct' must be 0, 1, or null")
    result["refusal_correct"] = refusal
    result["rationale"] = str(data.get("rationale", ""))[:500]
    return result


def invalid_result(error: str) -> dict:
    result: dict = {field: None for field in SCORE_FIELDS}
    result["refusal_correct"] = None
    result["rationale"] = ""
    result["invalid"] = True
    result["error"] = error[:500]
    return result


class Judge:
    def __init__(self, llm, cache_dir: str | Path = "evals/cache") -> None:
        self.llm = llm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def score(
        self, question: str, candidate: str, reference: str, context: str
    ) -> dict:
        key = cache_key(question, candidate, reference, context)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nReference answer:\n{reference}\n\n"
                    f"Candidate answer:\n{candidate}\n\nContext:\n{context}"
                ),
            },
        ]
        errors: list[str] = []
        for attempt in range(2):
            try:
                text, _ = self.llm.complete(messages, 200)
                result = parse_judgement(text)
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                return result
            except Exception as exc:  # noqa: BLE001 - invalid output or provider error
                errors.append(f"attempt {attempt + 1}: {exc}")
                messages = messages[:2] + [{"role": "user", "content": RETRY_PROMPT}]
        return invalid_result("; ".join(errors))
