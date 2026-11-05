"""Packaged, explicitly labeled authored examples; never silently substitute for generation."""
import json
from copy import deepcopy
from pathlib import Path

from rag.incident import DiagnosisResponse, IncidentRequest

RESOURCES = Path(__file__).parent / "resources"


def load_demos():
    return json.loads((RESOURCES / "demos.json").read_text())


def demo_response(demo_id: str, github_lookup: bool = False):
    demo = next((d for d in load_demos() if d["id"] == demo_id), None)
    if demo is None:
        raise KeyError(demo_id)
    payload = deepcopy(demo["response"])
    if github_lookup:
        path = RESOURCES / "github_recordings.json"
        recordings = json.loads(path.read_text()) if path.exists() else {}
        lookup = recordings.get(demo["topic"], {})
        for index, item in enumerate(lookup.get("items", []), 1):
            payload["evidence"].append({"id": f"github-{index}", "kind": item["kind"], "title": item["title"], "text": item["text"], "url": item["url"], "origin": "recorded_example"})
        payload["tool_trace"].append({"tool": "github_lookup", "status": "recorded_example", "requests": 0, "recorded_at": lookup.get("recorded_at")})
        payload["warnings"].append("GitHub evidence is a recorded snapshot, not a live lookup or proof of cause")
    IncidentRequest.model_validate(demo["request"])
    return DiagnosisResponse.model_validate(payload)
