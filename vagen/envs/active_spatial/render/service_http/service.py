from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from .handler import InteriorGSRenderHandler
from .multipart import decode_multipart, encode_multipart

LOGGER = logging.getLogger(__name__)


def build_app(
    handler: InteriorGSRenderHandler,
    *,
    max_inflight: int = 0,
    admit_timeout: float = 5.0,
    api_key: str = "",
    image_format: str = "PNG",
    image_mime: str = "image/png",
) -> FastAPI:
    sem: Optional[asyncio.Semaphore] = asyncio.Semaphore(max_inflight) if max_inflight > 0 else None

    async def lifespan(app: FastAPI):
        yield
        await handler.aclose()

    app = FastAPI(lifespan=lifespan)

    def authenticate(request: Request) -> None:
        if not api_key:
            return
        token = request.query_params.get("token") or request.headers.get("x-api-key")
        if token != api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True, "service": "interiorgs-render-service", "max_inflight": max_inflight or "unlimited"}

    @app.post("/render")
    async def render(request: Request) -> Response:
        authenticate(request)
        acquired = False
        if sem is not None:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=admit_timeout)
                acquired = True
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="server busy")
        try:
            content_type = request.headers.get("content-type", "")
            body = await request.body()
            meta, images = decode_multipart(content_type, body)
            result = await handler.handle(meta, images)
            boundary, resp_body = encode_multipart(
                result.get("meta", {}),
                encoded_images=result.get("encoded_images", []),
                image_format=image_format,
                image_mime=image_mime,
            )
            return Response(resp_body, media_type=f'multipart/mixed; boundary="{boundary}"')
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.exception("Render request failed")
            raise HTTPException(status_code=500, detail=repr(exc)) from exc
        finally:
            if acquired and sem is not None:
                sem.release()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve InteriorGS renders over HTTP multipart.")
    parser.add_argument("--gs-root", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--max-inflight", type=int, default=0)
    parser.add_argument("--admit-timeout", type=float, default=5.0)
    parser.add_argument("--forced-render-size", default=None)
    parser.add_argument("--api-key", default=os.getenv("RENDER_API_KEY", ""))
    parser.add_argument("--image-format", default="PNG", choices=["PNG", "JPEG", "WEBP", "png", "jpeg", "webp"])
    parser.add_argument("--image-quality", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    handler = InteriorGSRenderHandler(
        gs_root=args.gs_root,
        max_workers=args.max_workers,
        gpu_ids=args.gpu_ids,
        forced_render_size=args.forced_render_size,
        image_format=args.image_format,
        image_quality=args.image_quality,
    )
    fmt = args.image_format.upper()
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
    app = build_app(handler, max_inflight=args.max_inflight, admit_timeout=args.admit_timeout, api_key=args.api_key, image_format=fmt, image_mime=mime)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level=args.log_level)


if __name__ == "__main__":
    main()
