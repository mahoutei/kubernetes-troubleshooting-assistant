"""Explicit release verification; --live requires configured server-side model credentials."""
import argparse
import json
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        parser.error("choose a new release report path")
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        health = client.get("/health")
        health.raise_for_status()
        demos = client.get("/demos")
        demos.raise_for_status()
        results = []
        for demo in demos.json():
            recorded = client.get(f"/demos/{demo['id']}?github_lookup=true")
            recorded.raise_for_status()
            assert recorded.json()["mode"] == "recorded_demo"
            row = {"id": demo["id"], "recorded_demo": "passed"}
            if args.live:
                live = client.post("/diagnose", json=demo["request"])
                live.raise_for_status()
                assert live.json()["mode"] == "live"
                row["live_response"] = live.json()
            results.append(row)
    payload = {"health": health.json(), "demos": results,
               "live_provider_check": args.live, "semantic_review": "pending human review"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
