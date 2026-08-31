"""Simple WebSocket Gaussian Splat render server.

This server implements the protocol expected by ``GSRenderClient``:
client sends a JSON frame, then a binary frame; server replies with a JSON
frame, then a length-prefixed PNG blob.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import websockets
from PIL import Image

from .binary_utils import pack_length_prefixed, pil_to_png_bytes
from .gs_render_local import GaussianRenderer

LOGGER = logging.getLogger("simple_gs_render_server")


class RenderServer:
    def __init__(self, gs_root: str, gpu_device: int = 0, max_cached_scenes: int = 1):
        self.gs_root = gs_root
        self.gpu_device = gpu_device
        self.max_cached_scenes = max(1, max_cached_scenes)
        self.renderers: OrderedDict[str, GaussianRenderer] = OrderedDict()
        self.lock = asyncio.Lock()

    def _ply_path(self, scene_id: str) -> str:
        candidates = [
            os.path.join(self.gs_root, scene_id, "3dgs_compressed.ply"),
            os.path.join(self.gs_root, f"{scene_id}.ply"),
            os.path.join(self.gs_root, scene_id, "gaussian.ply"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find PLY for scene {scene_id}; tried {candidates}")

    def _get_renderer(self, scene_id: str) -> GaussianRenderer:
        renderer = self.renderers.get(scene_id)
        if renderer is not None:
            self.renderers.move_to_end(scene_id)
            return renderer

        renderer = GaussianRenderer(self._ply_path(scene_id), gpu_device=self.gpu_device)
        self.renderers[scene_id] = renderer
        self.renderers.move_to_end(scene_id)

        while len(self.renderers) > self.max_cached_scenes:
            self.renderers.popitem(last=False)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        return renderer

    @staticmethod
    def _to_pil(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _render_sync(self, scene_id: str, tasks: list[dict[str, Any]]) -> list[Image.Image]:
        renderer = self._get_renderer(scene_id)
        images: list[Image.Image] = []
        for task in tasks:
            if task.get("mode", "cam_param") != "cam_param":
                raise ValueError(f"Unsupported render mode: {task.get('mode')}")
            width, height = task.get("size", [256, 256])
            image = renderer.render_image_from_cam_param(
                np.asarray(task["intrinsics"], dtype=np.float32),
                np.asarray(task["extrinsics"], dtype=np.float32),
                int(width),
                int(height),
            )
            images.append(self._to_pil(image))
        return images

    async def handle(self, websocket):
        while True:
            try:
                meta_raw = await websocket.recv()
                binary = await websocket.recv()
            except websockets.ConnectionClosed:
                return

            try:
                meta = json.loads(meta_raw)
                if meta.get("op") != "render":
                    raise ValueError(f"Unsupported op: {meta.get('op')}")
                payload = meta.get("payload") or {}
                scene_id = payload["scene_id"]
                tasks = payload.get("tasks") or []

                async with self.lock:
                    images = await asyncio.to_thread(self._render_sync, scene_id, tasks)
                blob = pack_length_prefixed([pil_to_png_bytes(img) for img in images])
                response = {"ok": True, "req_id": meta.get("req_id"), "num_images": len(images)}
                await websocket.send(json.dumps(response))
                await websocket.send(blob)
            except Exception as exc:
                LOGGER.exception("Render request failed")
                response = {"ok": False, "req_id": None, "error": repr(exc)}
                try:
                    await websocket.send(json.dumps(response))
                    await websocket.send(binary if isinstance(binary, (bytes, bytearray)) else b"")
                except Exception:
                    return


async def amain() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs-root", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--max-cached-scenes", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    server = RenderServer(args.gs_root, args.gpu_device, args.max_cached_scenes)
    LOGGER.info(
        "Starting render server on %s:%s gs_root=%s gpu=%s",
        args.host,
        args.port,
        args.gs_root,
        args.gpu_device,
    )

    async def handler(websocket, *_args):
        return await server.handle(websocket)

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        await asyncio.Future()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
