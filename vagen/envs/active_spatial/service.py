from __future__ import annotations

import argparse
import logging
import os

from vagen.envs_remote import GymService

from .handler import ActiveSpatialHandler


def build_app(
    *,
    max_sessions: int = 0,
    session_timeout: float = 3600.0,
    max_inflight: int = 0,
    admit_timeout: float = 5.0,
    api_key: str = "",
):
    handler = ActiveSpatialHandler(
        session_timeout=session_timeout,
        max_sessions=max_sessions,
    )
    return GymService(
        handler,
        max_inflight=max_inflight,
        admit_timeout=admit_timeout,
        api_key=api_key,
    ).build()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve ActiveSpatialGymEnv via VAGEN's HTTP RemoteEnv protocol."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--session-timeout", type=float, default=3600.0)
    parser.add_argument("--max-inflight", type=int, default=0)
    parser.add_argument("--admit-timeout", type=float, default=5.0)
    parser.add_argument("--api-key", default=os.getenv("GYM_API_KEY", ""))
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for this service process.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    if args.cuda_visible_devices:
        # Must be set before the first ActiveSpatial env initializes local GS rendering.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = build_app(
        max_sessions=args.max_sessions,
        session_timeout=args.session_timeout,
        max_inflight=args.max_inflight,
        admit_timeout=args.admit_timeout,
        api_key=args.api_key,
    )

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
