"""Claude via the official Anthropic Python SDK (lazy import; install into the agent venv only:
``pip install anthropic`` -- see docs/ENVIRONMENT.md). Runs with Claude are a SUBSTITUTE for the
Astra provider and must be labelled as such in result tables (the provider label is in every log).

Config:
    kind: anthropic
    model: claude-opus-5-5        # default; any id from the Models API
    api_key_env: ANTHROPIC_API_KEY  # the SDK also resolves other credential sources
    effort: low                   # output_config.effort; low keeps latency inside the 1 s/step budget
    refusal_fallback: true        # server-side fallbacks="default" (beta server-side-fallback-2026-07-01)
    max_retries: 2
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping

from providers.base import ProviderRequest, ProviderResponse, ProviderUnavailable, b64, encode_jpeg, parse_json_loose


def _strict_schema(schema: Any) -> Any:
    """Structured outputs need additionalProperties: false on every object."""
    if isinstance(schema, dict):
        out = {k: _strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
            out.setdefault("required", list((out.get("properties") or {}).keys()))
        return out
    if isinstance(schema, list):
        return [_strict_schema(x) for x in schema]
    return schema


class AnthropicProvider:
    def __init__(self, name: str, cfg: Mapping[str, Any]):
        self.name = name
        self.cfg = dict(cfg)
        self.model = str(cfg.get("model", "claude-opus-5-5"))
        self.effort = cfg.get("effort", "low")
        self.refusal_fallback = bool(cfg.get("refusal_fallback", True))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.image_max_side = int(cfg.get("image_max_side", 768))
        self._client = None

    def _client_or_raise(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise ProviderUnavailable("the 'anthropic' package is not installed in this environment") from e
        key_env = self.cfg.get("api_key_env")
        key = os.environ.get(key_env) if key_env else None
        kwargs = {"max_retries": self.max_retries}
        if key:
            kwargs["api_key"] = key
        try:
            self._client = anthropic.Anthropic(**kwargs)
        except Exception as e:  # no credentials resolvable
            raise ProviderUnavailable(f"anthropic client could not be created: {type(e).__name__}") from e
        return self._client

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        client = self._client_or_raise()
        content: list = []
        for label, img in req.images:
            content.append({"type": "text", "text": f"[image: {label}]"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": b64(encode_jpeg(img, self.image_max_side))}})
        content.append({"type": "text", "text": req.user_text})
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if req.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": _strict_schema(dict(req.json_schema))}
        kwargs = dict(
            model=self.model,
            max_tokens=req.max_tokens,
            system=req.system,
            messages=[{"role": "user", "content": content}],
            output_config=output_config,
        )
        t0 = time.monotonic()
        # req.timeout_s is a total deadline: one attempt, no SDK retries inside it
        c = client.with_options(timeout=req.timeout_s, max_retries=0)
        if self.refusal_fallback:
            resp = c.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = c.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        stop = getattr(resp, "stop_reason", None)
        parsed = None
        if req.json_schema is not None and stop != "refusal":
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = parse_json_loose(text)
        usage = getattr(resp, "usage", None)
        return ProviderResponse(
            text=text,
            parsed=parsed,
            usage={"input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                   "output_tokens": int(getattr(usage, "output_tokens", 0) or 0)},
            latency_s=time.monotonic() - t0,
            model=str(getattr(resp, "model", self.model)),
            provider=self.name,
            error="refusal" if stop == "refusal" else ("truncated (stop_reason=max_tokens)" if stop == "max_tokens" else None),
            stop_reason=stop,
        )
