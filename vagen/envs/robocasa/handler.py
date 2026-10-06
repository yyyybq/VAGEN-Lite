"""Handler for RoboCasa kitchen environments.

Two usage styles (GymService works with both):

1. Factory: ``handler.make_env(cfg)`` / ``handler(cfg)`` returns a
   ``RoboCasaEnv`` for in-process evaluation.
2. Session map: ``reset`` / ``step`` / ``close`` / ``system_prompt``
   keyed by session_id. ``GymService`` uses ``create_env`` + the
   ``BaseGymHandler`` connect/call loop.

Ctor matches PrimitiveSkillHandler: devices, session_timeout,
max_envs_per_gpu. RoboCasa uses in-process EGL rendering (like Active
Spatial), not ManiSkill worker processes.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from vagen.envs.robocasa.robocasa_env import RoboCasaEnv
from vagen.envs_remote.handler import BaseGymHandler, HandlerResult, SessionContext

LOGGER = logging.getLogger(__name__)


class RoboCasaHandler(BaseGymHandler):
    """Remote-service + factory handler for RoboCasaEnv."""

    def __init__(
        self,
        devices: Optional[List[int]] = None,
        session_timeout: float = 600.0,
        max_envs_per_gpu: int = 16,
        acquire_timeout: float = 300.0,
        max_sessions: int = 0,
    ):
        self.devices = devices or [0]
        self.max_envs_per_gpu = int(max_envs_per_gpu)
        self.acquire_timeout = float(acquire_timeout)
        if max_sessions <= 0:
            max_sessions = self.max_envs_per_gpu * max(1, len(self.devices))
        super().__init__(session_timeout=session_timeout, max_sessions=max_sessions)
        self._round_robin = 0

    # ------------------------------------------------------------------
    # Factory (in-process)
    # ------------------------------------------------------------------

    def make_env(self, env_config: Optional[Dict[str, Any]] = None) -> RoboCasaEnv:
        """Build a RoboCasaEnv, assigning a GPU device from ``devices``."""
        cfg = dict(env_config or {})
        if "gpu_device" not in cfg and self.devices:
            cfg["gpu_device"] = int(self.devices[self._round_robin % len(self.devices)])
            self._round_robin += 1
        return RoboCasaEnv(cfg)

    def create(self, env_config: Optional[Dict[str, Any]] = None) -> RoboCasaEnv:
        return self.make_env(env_config)

    def __call__(self, env_config: Optional[Dict[str, Any]] = None) -> RoboCasaEnv:
        return self.make_env(env_config)

    async def create_env(self, env_config: Dict[str, Any]) -> RoboCasaEnv:
        """GymService entry point."""
        return self.make_env(env_config)

    # ------------------------------------------------------------------
    # Session-map helpers (usable without going through GymService.call)
    # ------------------------------------------------------------------

    def _require(self, session_id: str) -> SessionContext:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        ctx = self._sessions[session_id]
        ctx.last_access = time.time()
        return ctx

    async def system_prompt(self, session_id: str) -> Dict[str, Any]:
        ctx = self._require(session_id)
        return await ctx.env.system_prompt()

    async def reset(self, session_id: str, seed: int = 0):
        ctx = self._require(session_id)
        return await ctx.env.reset(int(seed))

    async def step(self, session_id: str, action_str: str):
        ctx = self._require(session_id)
        return await ctx.env.step(action_str)

    async def close_session(self, session_id: str) -> None:
        ctx = self._sessions.get(session_id)
        if ctx is None:
            return
        try:
            await ctx.env.close()
        finally:
            self._sessions.pop(session_id, None)

    async def open_session(self, env_config: Optional[Dict[str, Any]] = None) -> str:
        """Create a session without going through GymService.connect."""
        session_id = uuid.uuid4().hex
        env = self.make_env(env_config)
        self._sessions[session_id] = SessionContext(
            session_id=session_id,
            env=env,
            created_at=time.time(),
            last_access=time.time(),
        )
        return session_id
