"""Handler for the Active Spatial environment HTTP service.

Usage (imported by serve.py, not used directly):
    handler = ActiveSpatialHandler(devices=[0, 1], max_sessions=32)
    app = GymService(handler).build()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from vagen.envs_remote.handler import BaseGymHandler, HandlerResult
from .active_spatial_env import ActiveSpatialGymEnv

LOGGER = logging.getLogger(__name__)


class ActiveSpatialHandler(BaseGymHandler):
    """Handler for managing Active Spatial environment sessions.

    Resource model
    --------------
    Each session creates one ``ActiveSpatialGymEnv`` instance.  Envs are NOT
    cached between sessions (unlike the NavigationHandler) because:

    - Gaussian Splatting renderers hold significant GPU memory per scene.
    - With ``render_backend="client"`` each env just holds a WebSocket
      connection — cheap to recreate.
    - With ``render_backend="local"`` each env loads a .ply scene into GPU
      VRAM; eviction logic would add complexity with limited benefit.

    GPU assignment uses round-robin across the provided device list.  Set
    ``gpu_device`` in ``env_config`` to override on a per-session basis.

    Parameters
    ----------
    devices:
        CUDA device indices to use.  Defaults to auto-detecting all visible GPUs.
    session_timeout:
        Seconds of idle time before a session is cleaned up automatically.
    max_sessions:
        Hard cap on concurrent sessions (0 = unlimited).
    """

    def __init__(
        self,
        devices: Optional[List[int]] = None,
        session_timeout: float = 3600.0,
        max_sessions: int = 0,
    ):
        super().__init__(session_timeout=session_timeout, max_sessions=max_sessions)
        self.devices: List[int] = devices or [0]
        self._rr_idx: int = 0

    def _next_device(self) -> int:
        """Round-robin GPU selection."""
        device = self.devices[self._rr_idx % len(self.devices)]
        self._rr_idx += 1
        return device

    async def create_env(self, env_config: Dict[str, Any]) -> ActiveSpatialGymEnv:
        """Create a new ActiveSpatialGymEnv.

        ``gpu_device`` is injected from the round-robin selector unless the
        caller already specified it in ``env_config``.
        """
        if "gpu_device" not in env_config:
            env_config = {**env_config, "gpu_device": self._next_device()}
        LOGGER.info(f"[ActiveSpatialHandler] Creating env gpu_device={env_config.get('gpu_device')}")
        return ActiveSpatialGymEnv(env_config)
