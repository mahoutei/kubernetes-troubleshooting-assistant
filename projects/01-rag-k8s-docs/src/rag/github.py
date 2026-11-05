"""Bounded read-only GitHub evidence. No caller-controlled endpoints or qualifiers."""
from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict
from urllib.parse import urlparse

import httpx


class GitHubLookup:
    def __init__(self, transport=None, clock=time.time):
        self.transport = transport
        self.clock = clock
        self.cache = OrderedDict()
        self.blocked_until = {}
        self.lock = threading.Lock()

    def lookup(self, topic: str, version: str | None = None) -> dict:
        # Never transmit user manifests, logs, names, or secrets to GitHub.
        topic = topic if topic in {"OOMKilled", "Unhealthy", "FailedMount", "CrashLoopBackOff", "ImagePullBackOff", "FailedScheduling", "Kubernetes troubleshooting"} else "Kubernetes troubleshooting"
        if version is not None and not re.fullmatch(r"1\.\d{1,2}(?:\.\d{1,2})?", version):
            raise ValueError("invalid Kubernetes version")
        key = (topic, version)
        with self.lock:
            cached = self.cache.get(key)
            if cached and cached[0] > self.clock():
                return {**cached[1], "status": "cached", "requests": 0}
            items, warnings = [], []
            requests = 0
            headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
            if os.getenv("GITHUB_TOKEN"):
                headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
            endpoints = [
                ("search", "/search/issues", {"q": f"repo:kubernetes/kubernetes is:issue {topic}", "per_page": 3}),
                ("core", f"/repos/kubernetes/kubernetes/releases/tags/v{version}" if version else "/repos/kubernetes/kubernetes/releases/latest", {}),
            ]
            with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=httpx.Timeout(4.0), follow_redirects=False, transport=self.transport) as client:
                for resource, path, params in endpoints:
                    if self.blocked_until.get(resource, 0) > self.clock():
                        warnings.append(f"GitHub {resource} rate-limited; using documentation only for this lookup")
                        continue
                    requests += 1
                    try:
                        response = client.get(path, params=params)
                        if response.status_code in {403, 429} or response.headers.get("x-ratelimit-remaining") == "0":
                            try:
                                reset = float(response.headers.get("x-ratelimit-reset", self.clock() + 60))
                                retry = float(response.headers.get("retry-after", 60))
                            except ValueError:
                                reset, retry = self.clock() + 60, 60
                            self.blocked_until[resource] = max(self.clock() + retry, reset)
                        response.raise_for_status()
                        data = response.json()
                        if isinstance(data, dict) and data.get("incomplete_results"):
                            warnings.append("GitHub search returned incomplete results")
                        candidates = data.get("items", []) if resource == "search" and isinstance(data, dict) else [data]
                        for entry in candidates[:3]:
                            if not isinstance(entry, dict):
                                continue
                            url = str(entry.get("html_url", ""))
                            parsed = urlparse(url)
                            if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.startswith("/kubernetes/kubernetes/"):
                                continue
                            items.append({"title": str(entry.get("title") or entry.get("name") or entry.get("tag_name", "GitHub evidence"))[:200], "url": url,
                                          "text": str(entry.get("body") or "")[:1800], "kind": "github_issue" if resource == "search" else "github_release"})
                    except (httpx.HTTPError, ValueError, TypeError):
                        warnings.append(f"GitHub {resource} unavailable; diagnosis continues without that evidence")
            result = {"status": "live" if items else "unavailable", "items": items, "warnings": warnings, "requests": requests}
            self.cache[key] = (self.clock() + (600 if items else 30), result)
            while len(self.cache) > 128:
                self.cache.popitem(last=False)
            return result
