"""Providers that do not call a model: for tests and deterministic re-runs only.

replay     returns responses recorded in a run's events.jsonl, in call order per slot ("purpose").
           Lets a decision sequence be re-run bit-for-bit (truth-isolation and switch-matrix tests).
scripted   returns fixed JSON replies from the config (mock episodes). NEVER used for results.
"""

from __future__ import annotations

import collections
import json
from typing import Any, Mapping

from providers.base import ProviderRequest, ProviderResponse


class ReplayProvider:
    def __init__(self, name: str, cfg: Mapping[str, Any]):
        self.name = name
        self.model = "replay"
        self._queues: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        path = cfg.get("events_path")
        if path:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("kind") == "provider_call":
                        self._queues[rec.get("purpose", "")].append(rec)

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        q = self._queues.get(req.purpose)
        if not q:
            return ProviderResponse(text="", parsed=None, usage={}, latency_s=0.0, model="replay",
                                    provider=self.name, error="replay exhausted")
        rec = q.popleft()
        return ProviderResponse(text=rec.get("response_text") or "", parsed=rec.get("parsed"),
                                usage=rec.get("usage") or {}, latency_s=0.0, model=rec.get("model") or "replay",
                                provider=self.name, error=rec.get("error"))


class ScriptedProvider:
    """cfg: {replies: {purpose: [dict, dict, ...]}}; the last reply repeats when the list runs out."""

    def __init__(self, name: str, cfg: Mapping[str, Any]):
        self.name = name
        self.model = "scripted"
        self._replies = {k: list(v) for k, v in dict(cfg.get("replies") or {}).items()}
        self._idx: dict[str, int] = collections.defaultdict(int)
        self.requests: list[ProviderRequest] = []

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        self.requests.append(req)
        replies = self._replies.get(req.purpose) or []
        if not replies:
            return ProviderResponse(text="", parsed=None, usage={}, latency_s=0.0, model="scripted",
                                    provider=self.name, error=f"no scripted reply for {req.purpose}")
        i = min(self._idx[req.purpose], len(replies) - 1)
        self._idx[req.purpose] += 1
        reply = replies[i]
        return ProviderResponse(text=json.dumps(reply), parsed=dict(reply), usage={"input_tokens": 0, "output_tokens": 0},
                                latency_s=0.0, model="scripted", provider=self.name)
