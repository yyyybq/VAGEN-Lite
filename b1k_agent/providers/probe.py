"""Probe a configured provider's real capabilities before any module relies on it.

    python -m providers.probe --config configs/L2_plan_check.yaml --provider astra --out docs/provider_probe_astra.json

Checks (each logged with latency, tokens, error; the key is never printed):
  text          plain text round trip
  json_schema   structured output against a small schema
  vision        a generated image (red square on the left, blue on the right) -> "which side is red?"
  latency       N minimal calls -> p50 / p90 / max (a lower bound)
  latency_realistic  N module-sized calls (long system prompt + one 224 px image + JSON schema) ->
                p50 / p90 / max, stop reasons, token usage (decides how often M-plan / M-check can
                run inside the max_steps x 1 s budget)
Every check also records stop_reason, usage and the model id the endpoint reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from agent.config import get, load_config
from providers import ProviderRequest, build_providers


def _rec(r) -> dict:
    """What every check records: error, latency, stop reason, token usage, the model id the endpoint reported."""
    return {"error": r.error, "latency_s": round(r.latency_s, 3), "stop_reason": getattr(r, "stop_reason", None),
            "usage": dict(r.usage or {}), "model": r.model}


def _pcts(lat: list) -> dict:
    if not lat:
        return {"p50_s": None, "p90_s": None, "max_s": None}
    s = sorted(lat)
    return {"p50_s": round(statistics.median(s), 3), "p90_s": round(s[min(len(s) - 1, int(0.9 * len(s)))], 3),
            "max_s": round(s[-1], 3)}


def _scene_image(seed: int) -> np.ndarray:
    """A 224x224 RGB image with some structure (the size of the DefaultWrapper head camera)."""
    rng = np.random.default_rng(seed)
    img = (rng.random((224, 224, 3)) * 40 + 90).astype(np.uint8)
    for _ in range(6):
        y, x = rng.integers(0, 180, 2)
        img[y:y + 40, x:x + 40] = rng.integers(0, 255, 3)
    return img


# a module-sized request: roughly the length of the plan/check prompts plus one camera image
_REALISTIC_SYSTEM = ("You are the high-level checker of a household robot. " * 40).strip()
_REALISTIC_SCHEMA = {"type": "object", "properties": {
    "status": {"type": "string", "enum": ["in_progress", "done", "failed"]},
    "confidence": {"type": "number"}, "evidence": {"type": "string"}},
    "required": ["status", "confidence", "evidence"], "additionalProperties": False}


def probe(provider, n_latency: int = 5, max_tokens: int = 256, realistic_max_tokens: int = 1024) -> dict:
    """``max_tokens`` is generous on purpose: reasoning models spend hidden tokens before any text, so a
    10-token cap reports a working endpoint as broken (stop_reason "length"/"max_tokens")."""
    out: dict = {"provider": provider.label, "max_tokens": max_tokens,
                 "realistic_max_tokens": realistic_max_tokens, "checks": {}}

    r = provider.complete(ProviderRequest(system="You are terse.", user_text="Reply with the single word: ready",
                                          max_tokens=max_tokens, purpose="probe.text"))
    out["checks"]["text"] = {"ok": r.error is None and "ready" in r.text.lower(), "text": r.text[:80], **_rec(r)}

    schema = {"type": "object", "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
              "required": ["answer", "confidence"], "additionalProperties": False}
    r = provider.complete(ProviderRequest(system="Answer in JSON.", user_text="What is 2+3? answer as a word.",
                                          json_schema=schema, max_tokens=max_tokens, purpose="probe.json"))
    out["checks"]["json_schema"] = {"ok": r.error is None and isinstance(r.parsed, dict) and "answer" in (r.parsed or {}),
                                    "parsed": r.parsed, **_rec(r)}

    img = np.zeros((224, 224, 3), np.uint8)
    img[60:160, 20:100] = (220, 20, 20)
    img[60:160, 124:204] = (20, 20, 220)
    schema_v = {"type": "object", "properties": {"red_side": {"type": "string", "enum": ["left", "right"]}},
                "required": ["red_side"], "additionalProperties": False}
    r = provider.complete(ProviderRequest(system="You look at images carefully.", user_text="Is the red square on the left or the right?",
                                          images=[("test", img)], json_schema=schema_v, max_tokens=max_tokens, purpose="probe.vision"))
    out["checks"]["vision"] = {"ok": r.error is None and (r.parsed or {}).get("red_side") == "left",
                               "parsed": r.parsed, **_rec(r)}

    # minimal calls: a lower bound on latency
    lat, errors, stops = [], 0, {}
    for i in range(n_latency):
        r = provider.complete(ProviderRequest(system="You are terse.", user_text=f"Reply with the number {i}.",
                                              max_tokens=max_tokens, purpose="probe.latency"))
        stops[str(r.stop_reason)] = stops.get(str(r.stop_reason), 0) + 1
        if r.error:
            errors += 1
        else:
            lat.append(r.latency_s)
    out["checks"]["latency"] = {"n": n_latency, "errors": errors, "stop_reasons": stops, **_pcts(lat)}

    # module-sized calls (long system prompt + one image + schema): the number that sets how often
    # M-plan / M-check can run inside the max_steps x 1 s budget
    lat, errors, stops, usage = [], 0, {}, {"input_tokens": 0, "output_tokens": 0}
    for i in range(n_latency):
        r = provider.complete(ProviderRequest(
            system=_REALISTIC_SYSTEM, user_text="Subtask: press the power button of the radio. Has it succeeded? "
            "Judge only from the image.", images=[("head", _scene_image(i))], json_schema=_REALISTIC_SCHEMA,
            max_tokens=realistic_max_tokens, purpose="probe.realistic"))
        stops[str(r.stop_reason)] = stops.get(str(r.stop_reason), 0) + 1
        for k in usage:
            usage[k] += int((r.usage or {}).get(k, 0) or 0)
        if r.error:
            errors += 1
        else:
            lat.append(r.latency_s)
    out["checks"]["latency_realistic"] = {"n": n_latency, "errors": errors, "stop_reasons": stops,
                                          "usage_total": usage, **_pcts(lat)}
    out["models_reported"] = sorted({str(c.get("model")) for c in out["checks"].values() if c.get("model")})
    out["probed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--n-latency", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--realistic-max-tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    providers = build_providers(get(cfg, "providers", {}) or {})
    if args.provider not in providers:
        raise SystemExit(f"provider {args.provider!r} not in config (have {sorted(providers)})")
    result = probe(providers[args.provider], args.n_latency, args.max_tokens, args.realistic_max_tokens)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
