"""Active Spatial environment HTTP server.

Starts a FastAPI + uvicorn server that exposes the Active Spatial environment
over HTTP so that the VAGEN-Lite training loop can interact with it as a
``RemoteEnv``.

Each training worker connects to this server via ``GymImageEnvClient``.  The
server manages session lifecycle (create / reset / step / close) and serialises
PIL images over HTTP.

Usage
-----
    # Auto-detect all GPUs, default settings:
    python -m vagen.envs.active_spatial.serve

    # Specify GPUs and limits:
    python -m vagen.envs.active_spatial.serve \\
        --devices='[0,1]' \\
        --max_sessions=32 \\
        --port=8001

    # With API key authentication:
    python -m vagen.envs.active_spatial.serve --api_key=my_secret_key

Environment variables
---------------------
ACTIVE_SPATIAL_VAGEN_PATH
    Path to the old VAGEN repo root (auto-detected relative to this file if
    not set).  The server imports ``vagen.env.active_spatial.*`` from there.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import subprocess
from typing import List, Optional

import fire
import uvicorn

from vagen.envs_remote import GymService
from vagen.envs.active_spatial.handler import ActiveSpatialHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _detect_gpus() -> List[int]:
    """Auto-detect NVIDIA GPUs via CUDA_VISIBLE_DEVICES or nvidia-smi."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [int(d) for d in vis.split(",") if d.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        return [int(line.strip()) for line in out.strip().split("\n") if line.strip()]
    except Exception:
        return [0]


def main(
    host: str = "0.0.0.0",
    port: int = 8001,
    # GPU devices. None = auto-detect all visible GPUs.
    devices: Optional[List[int]] = None,
    # Max concurrent sessions (0 = unlimited).
    max_sessions: int = 0,
    # Max concurrent HTTP requests being processed (0 = unlimited).
    max_inflight: int = 256,
    # Thread pool size for asyncio.to_thread().
    # Set >= max_sessions so all envs can run blocking calls concurrently.
    thread_pool_size: int = 256,
    # Session idle timeout before auto-cleanup (seconds).
    session_timeout: float = 3600.0,
    # API key for authentication. Empty = no auth.
    api_key: str = "",
    # Uvicorn workers. Keep at 1 (handler state is in-process).
    workers: int = 1,
):
    """Start the Active Spatial environment server."""
    if workers > 1:
        raise ValueError(
            f"workers={workers} is not supported. Sessions are stored in-memory "
            f"per process, so multiple workers would lose sessions across processes. "
            f"Use workers=1 (the default)."
        )

    if devices is None:
        devices = _detect_gpus()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size)

    LOGGER.info(
        f"Starting Active Spatial server | "
        f"GPUs: {devices} | max_sessions: {max_sessions or 'unlimited'} | "
        f"threads: {thread_pool_size} | port: {port}"
    )

    handler = ActiveSpatialHandler(
        devices=devices,
        session_timeout=session_timeout,
        max_sessions=max_sessions,
    )
    service = GymService(handler, max_inflight=max_inflight, api_key=api_key)
    app = service.build(
        startup_callback=lambda: asyncio.get_running_loop().set_default_executor(executor),
        shutdown_callback=lambda: executor.shutdown(wait=True),
    )

    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    fire.Fire(main)
