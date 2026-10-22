"""Strong-model provider interface, call logging and cost accounting.

Keys are read from environment variables named in the config (``api_key_env``); they are never
logged, written to manifests, or included in error messages.

Every call is logged through the run recorder (task spec 7.3, distillation data): slot, episode,
step, prompt hash + user text, image references, raw response, parsed JSON, usage, latency, cost,
error. Image pixels are saved once per call as JPEG under the episode's frames/ directory.
"""

from __future__ import annotations

import base64
import dataclasses
import itertools
import hashlib
import io
import json
import re
import time
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from agent import context as ctxvars


class ProviderUnavailable(RuntimeError):
    """The provider is not configured (missing env vars) or not reachable."""


class ProviderError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ProviderRequest:
    system: str
    user_text: str
    images: Sequence[tuple[str, np.ndarray]] = ()  # (label, HxWx3 uint8)
    json_schema: Optional[Mapping[str, Any]] = None
    max_tokens: int = 2048
    timeout_s: float = 60.0
    purpose: str = ""  # e.g. "plan.next_subtask"


@dataclasses.dataclass(frozen=True)
class ProviderResponse:
    text: str
    parsed: Optional[dict]
    usage: Mapping[str, int]
    latency_s: float
    model: str
    provider: str
    error: Optional[str] = None
    cost_usd: Optional[float] = None
    stop_reason: Optional[str] = None


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, req: ProviderRequest) -> ProviderResponse: ...


def encode_jpeg(img: np.ndarray, max_side: int = 768, quality: int = 85) -> bytes:
    from PIL import Image

    im = Image.fromarray(np.asarray(img, dtype=np.uint8)[..., :3])
    if max(im.size) > max_side:
        scale = max_side / float(max(im.size))
        im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_loose(text: str) -> Optional[dict]:
    """Parse the first JSON object in a model reply (handles ```json fences)."""
    if not text:
        return None
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except Exception:
        return None


# One counter for every provider in the process, so call ids (and the image files named after them)
# never collide when two providers are used in the same episode.
_CALL_COUNTER = itertools.count(1)


class LoggedProvider:
    """Wraps a provider: timing, cost, per-episode logging, deterministic error capture.

    ``complete`` never raises for transport/model errors; it returns a response with ``error`` set
    so modules can take their documented fallback path. ProviderUnavailable (configuration) is
    also turned into an error response, but logged with error="unavailable: ...".
    """

    def __init__(self, inner: Any, *, price_in_per_mtok: float = 0.0, price_out_per_mtok: float = 0.0,
                 recorder: Any = None, label: str = ""):
        self.inner = inner
        self.name = getattr(inner, "name", "provider")
        self.label = label or self.name
        self.price_in = float(price_in_per_mtok or 0.0)
        self.price_out = float(price_out_per_mtok or 0.0)
        self.recorder = recorder
        self._calls = 0

    def complete(self, req: ProviderRequest) -> ProviderResponse:
        self._calls += 1
        call_id = f"{ctxvars.current_episode.get() or 'noep'}-c{next(_CALL_COUNTER):06d}"
        t0 = time.monotonic()
        try:
            resp = self.inner.complete(req)
        except ProviderUnavailable as e:
            resp = ProviderResponse(text="", parsed=None, usage={}, latency_s=time.monotonic() - t0,
                                    model=getattr(self.inner, "model", ""), provider=self.label,
                                    error=f"unavailable: {e}")
        except Exception as e:  # network, HTTP, SDK errors
            resp = ProviderResponse(text="", parsed=None, usage={}, latency_s=time.monotonic() - t0,
                                    model=getattr(self.inner, "model", ""), provider=self.label,
                                    error=f"{type(e).__name__}: {str(e)[:300]}")
        cost = (resp.usage.get("input_tokens", 0) * self.price_in + resp.usage.get("output_tokens", 0) * self.price_out) / 1e6
        resp = dataclasses.replace(resp, cost_usd=cost, provider=self.label)
        self._log(call_id, req, resp)
        return resp

    def _log(self, call_id: str, req: ProviderRequest, resp: ProviderResponse) -> None:
        if self.recorder is None:
            return
        episode = ctxvars.current_episode.get()
        image_refs = []
        if episode and req.images:
            try:
                import os

                d = os.path.join(self.recorder.local_dir, "episodes", episode, "frames")
                os.makedirs(d, exist_ok=True)
                for label, img in req.images:
                    p = os.path.join(d, f"{call_id}_{label}.jpg")
                    with open(p, "wb") as f:
                        f.write(encode_jpeg(img, max_side=448))
                    image_refs.append(os.path.relpath(p, self.recorder.local_dir))
            except Exception as e:  # logging must never break the episode
                image_refs.append(f"save_failed: {e}")
        rec = {
            "call_id": call_id,
            "slot": ctxvars.current_slot.get(),
            "step": ctxvars.current_step.get(),
            "purpose": req.purpose,
            "provider": self.label,
            "model": resp.model,
            "system_sha256": hashlib.sha256(req.system.encode("utf-8")).hexdigest()[:16],
            "user_text": req.user_text,
            "images": image_refs,
            "response_text": resp.text,
            "parsed": resp.parsed,
            "usage": dict(resp.usage),
            "latency_s": round(resp.latency_s, 3),
            "cost_usd": resp.cost_usd,
            "error": resp.error,
            "stop_reason": resp.stop_reason,
        }
        self.recorder.provider_call(episode or "noep", rec)
