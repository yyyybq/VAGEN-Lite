from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image

from vagen.envs_remote.multipart_codec import encode_multipart

from .service_http.multipart import decode_multipart

LOGGER = logging.getLogger(__name__)


class InteriorGSHTTPRenderClient:
    """HTTP multipart client for the InteriorGS render service."""

    def __init__(
        self,
        base_urls: str | List[str],
        *,
        token: str = "",
        timeout: float | None = None,
        retries: int | None = None,
        backoff: float | None = None,
        max_backoff: float | None = None,
    ):
        if isinstance(base_urls, str):
            parts = base_urls.replace(";", "\n").split("\n")
            self.base_urls = [p.strip().rstrip("/") for p in parts if p.strip()]
        else:
            self.base_urls = [u.rstrip("/") for u in base_urls]
        if not self.base_urls:
            raise ValueError("At least one HTTP render URL is required")
        self.token = token
        self.timeout = float(timeout if timeout is not None else os.getenv("INTERIORGS_HTTP_TIMEOUT", "300"))
        self.retries = int(retries if retries is not None else os.getenv("INTERIORGS_HTTP_RETRIES", "3"))
        self.backoff = float(backoff if backoff is not None else os.getenv("INTERIORGS_HTTP_BACKOFF", "1.0"))
        self.max_backoff = float(
            max_backoff if max_backoff is not None else os.getenv("INTERIORGS_HTTP_MAX_BACKOFF", "60.0")
        )


    def _ranked_base_urls(self, scene_id: str) -> List[str]:
        if len(self.base_urls) <= 1:
            return self.base_urls
        scored = []
        for url in self.base_urls:
            digest = hashlib.blake2b(f"{scene_id}|{url}".encode("utf-8"), digest_size=8).digest()
            score = int.from_bytes(digest, "big")
            scored.append((score, url))
        scored.sort(reverse=True)
        return [url for _score, url in scored]

    async def render(self, scene_id: str, tasks: List[Dict[str, Any]]) -> List[Image.Image]:
        meta = {"scene_id": scene_id, "tasks": tasks}
        boundary, body = encode_multipart(meta, boundary_prefix="interiorgs_req_")
        headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
        if self.token:
            headers["X-API-Key"] = self.token
        last_exc: Optional[BaseException] = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout), limits=httpx.Limits(max_connections=32)) as client:
            ranked_urls = self._ranked_base_urls(scene_id)
            for attempt in range(self.retries + 1):
                base = ranked_urls[attempt % len(ranked_urls)]
                url = base if base.endswith("/render") else f"{base}/render"
                try:
                    resp = await client.post(url, content=body, headers=headers)
                    if resp.status_code == 503:
                        raise RuntimeError("render server busy")
                    resp.raise_for_status()
                    _meta, images = decode_multipart(resp.headers.get("content-type", ""), resp.content)
                    return [img.convert("RGB") for img in images]
                except Exception as exc:
                    last_exc = exc
                    if attempt >= self.retries:
                        raise
                    delay = min(self.backoff * (2**attempt), self.max_backoff)
                    await asyncio.sleep(delay * (0.7 + 0.6 * random.random()))
        raise last_exc or RuntimeError("render request failed")
