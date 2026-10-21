"""Probe a configured provider's real capabilities before any module relies on it.

    python -m providers.probe --config configs/L2_plan_check.yaml --provider astra --out docs/provider_probe_astra.json

Checks (each logged with latency, tokens, error; the key is never printed):
  text          plain text round trip
  json_schema   structured output against a small schema
  vision        a generated image (red square on the left, blue on the right) -> "which side is red?"
  latency       N repeated small calls -> p50 / p90 (decides how often M-plan / M-check can run
                inside the 1 s/step budget)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from agent.config import get, load_config
from providers import ProviderRequest, build_providers


def probe(provider, n_latency: int = 5) -> dict:
    out: dict = {"provider": provider.label, "checks": {}}

    r = provider.complete(ProviderRequest(system="You are terse.", user_text="Reply with the single word: ready",
                                          max_tokens=20, purpose="probe.text"))
    out["checks"]["text"] = {"ok": r.error is None and "ready" in r.text.lower(), "error": r.error,
                             "latency_s": round(r.latency_s, 3), "model": r.model, "text": r.text[:80]}

    schema = {"type": "object", "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
              "required": ["answer", "confidence"], "additionalProperties": False}
    r = provider.complete(ProviderRequest(system="Answer in JSON.", user_text="What is 2+3? answer as a word.",
                                          json_schema=schema, max_tokens=200, purpose="probe.json"))
    out["checks"]["json_schema"] = {"ok": r.error is None and isinstance(r.parsed, dict) and "answer" in (r.parsed or {}),
                                    "error": r.error, "parsed": r.parsed, "latency_s": round(r.latency_s, 3)}

    img = np.zeros((224, 224, 3), np.uint8)
    img[60:160, 20:100] = (220, 20, 20)
    img[60:160, 124:204] = (20, 20, 220)
    schema_v = {"type": "object", "properties": {"red_side": {"type": "string", "enum": ["left", "right"]}},
                "required": ["red_side"], "additionalProperties": False}
    r = provider.complete(ProviderRequest(system="You look at images carefully.", user_text="Is the red square on the left or the right?",
                                          images=[("test", img)], json_schema=schema_v, max_tokens=200, purpose="probe.vision"))
    out["checks"]["vision"] = {"ok": r.error is None and (r.parsed or {}).get("red_side") == "left", "error": r.error,
                               "parsed": r.parsed, "latency_s": round(r.latency_s, 3)}

    lat = []
    errors = 0
    for i in range(n_latency):
        r = provider.complete(ProviderRequest(system="You are terse.", user_text=f"Reply with the number {i}.",
                                              max_tokens=10, purpose="probe.latency"))
        if r.error:
            errors += 1
        else:
            lat.append(r.latency_s)
    out["checks"]["latency"] = {
        "n": n_latency, "errors": errors,
        "p50_s": round(statistics.median(lat), 3) if lat else None,
        "p90_s": round(sorted(lat)[max(0, int(0.9 * len(lat)) - 1)], 3) if lat else None,
    }
    out["probed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--n-latency", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    providers = build_providers(get(cfg, "providers", {}) or {})
    if args.provider not in providers:
        raise SystemExit(f"provider {args.provider!r} not in config (have {sorted(providers)})")
    result = probe(providers[args.provider], args.n_latency)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
