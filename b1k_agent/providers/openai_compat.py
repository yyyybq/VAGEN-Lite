"""Generic OpenAI-compatible chat-completions provider (httpx), for endpoints such as the one the
project may use for "Astra". Nothing about Astra is assumed: base URL, model id and key all come
from environment variables named in the config, and capabilities (vision, JSON schema output) are
config flags that providers/probe.py checks against the real endpoint before use.

Config:
    kind: openai_compat
    base_url_env: ASTRA_BASE_URL      # e.g. https://host/v1 (".../chat/completions" is appended)
    api_key_env: ASTRA_API_KEY
    model_env: ASTRA_MODEL            # or model: <id>  (no default: never guess a model id)
    structured_output: json_schema    # json_schema | json_object | prompt_only
    vision: true
    max_tokens_field: max_tokens      # newer OpenAI models take "max_completion_tokens"
    max_retries: 2

Identification (2026-10-22, search snippets only, NOT verified against the API): the only API-served
model named "Astra" found is OpenAI GPT-6 Astra, model id ``gpt-6-astra`` (Chat Completions and
Responses API, text+image input, structured outputs listed). Confirm with ``GET /v1/models`` and
providers/probe.py before use; do not hard-code a snapshot id.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Mapping

from providers.base import (
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ProviderUnavailable,
    b64,
    encode_jpeg,
    parse_json_loose,
)


class OpenAICompatProvider:
    def __init__(self, name: str, cfg: Mapping[str, Any]):
        self.name = name
        self.cfg = dict(cfg)
        self.structured = str(cfg.get("structured_output", "json_schema"))
        self.vision = bool(cfg.get("vision", True))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.image_max_side = int(cfg.get("image_max_side", 768))
        self.extra_body = dict(cfg.get("extra_body") or {})
        self.max_tokens_field = str(cfg.get("max_tokens_field", "max_tokens"))
        self.model = cfg.get("model") or os.environ.get(str(cfg.get("model_env", "")), "")

    def _endpoint(self) -> tuple[str, str]:
        base = self.cfg.get("base_url") or os.environ.get(str(self.cfg.get("base_url_env", "")), "")
        key = os.environ.get(str(self.cfg.get("api_key_env", "")), "")
        missing = [n for n, v in (("base_url", base), ("api key", key), ("model", self.model)) if not v]
        if missing:
            raise ProviderUnavailable(
                f"provider '{self.name}' not configured: missing {', '.join(missing)} "
                f"(env vars {self.cfg.get('base_url_env')}, {self.cfg.get('api_key_env')}, {self.cfg.get('model_env')})"
            )
        return base.rstrip("/") + "/chat/completions", key

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        """One call. ``req.timeout_s`` is a TOTAL deadline across retries (the loop budgets on it)."""
        import httpx

        url, key = self._endpoint()
        content: list = [{"type": "text", "text": req.user_text}]
        if req.images:
            if not self.vision:
                raise ProviderError(f"provider '{self.name}' is configured without vision")
            for label, img in req.images:
                content.append({"type": "text", "text": f"[image: {label}]"})
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + b64(encode_jpeg(img, self.image_max_side))}})
        system = req.system
        if req.json_schema is not None and self.structured != "json_schema":
            # json_object / prompt_only modes need the format in the prompt (and the word "JSON")
            system += ("\n\nReply with a single JSON object (no prose) that conforms to this JSON schema:\n"
                       + json.dumps(req.json_schema))
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            self.max_tokens_field: req.max_tokens,
            **self.extra_body,
        }
        if req.json_schema is not None:
            if self.structured == "json_schema":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "reply", "schema": req.json_schema, "strict": True}}
            elif self.structured == "json_object":
                body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        t0 = time.monotonic()
        deadline = t0 + float(req.timeout_s)
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            left = deadline - time.monotonic()
            if left <= 0.5:
                break
            try:
                r = httpx.post(url, headers=headers, content=json.dumps(body), timeout=left)
                if r.status_code in (408, 409, 429) or r.status_code >= 500:
                    raise ProviderError(f"HTTP {r.status_code}")
                if r.status_code >= 400:
                    raise ProviderError(f"HTTP {r.status_code}: {_redact(r.text[:300], key)}")
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                text = msg.get("content") or ""
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                usage = data.get("usage") or {}
                error = None
                if msg.get("refusal"):
                    error = f"refusal: {str(msg['refusal'])[:200]}"
                elif choice.get("finish_reason") == "length":
                    error = "truncated (finish_reason=length)"
                return ProviderResponse(
                    text=text,
                    parsed=parse_json_loose(text) if (req.json_schema is not None and error is None) else None,
                    usage={"input_tokens": int(usage.get("prompt_tokens") or 0),
                           "output_tokens": int(usage.get("completion_tokens") or 0)},
                    latency_s=time.monotonic() - t0,
                    model=str(data.get("model") or self.model),
                    provider=self.name,
                    error=error,
                    stop_reason=choice.get("finish_reason"),
                )
            except (ProviderError, httpx.TransportError) as e:
                last = e
                retryable = isinstance(e, httpx.TransportError) or str(e).startswith(("HTTP 408", "HTTP 409", "HTTP 429", "HTTP 5"))
                backoff = min(8.0, 1.0 * 2**attempt)
                if attempt < self.max_retries and retryable and time.monotonic() + backoff < deadline - 1.0:
                    time.sleep(backoff)
                    continue
                raise
        raise ProviderError(f"deadline of {req.timeout_s:.0f}s exhausted ({last})")


def _redact(text: str, key: str) -> str:
    """Error bodies from gateways may echo credentials; never let them reach the logs."""
    if key:
        text = text.replace(key, "***")
    return re.sub(r"(sk-[A-Za-z0-9_\-]{6,}|Bearer\s+\S+)", "***", text)
