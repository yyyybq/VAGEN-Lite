"""Active Spatial Navigation environment (GymImageEnv async interface).

This module wraps the legacy ActiveSpatialEnv (from the old VAGEN codebase)
with VAGEN-Lite's GymImageEnv async interface.

Key changes from the old interface:
- All methods are async (sync blocking calls run via asyncio.to_thread)
- system_prompt() returns an obs dict {"obs_str": "..."} instead of a string
- Observation dicts use "multi_modal_input" key instead of "multi_modal_data"

Usage (local, in-process):
    env = ActiveSpatialGymEnv({
        "jsonl_path": "/path/to/data.jsonl",
        "render_backend": "local",
        "gs_root": "/path/to/InteriorGS",
        "gpu_device": 0,
    })
    obs_dict = await env.system_prompt()
    obs, info = await env.reset(seed=0)
    obs, reward, done, info = await env.step("<think>...</think><action>move_forward</action>")
    await env.close()

Usage (remote, via serve.py):
    # Start the server:
    #   python -m vagen.envs.active_spatial.serve --devices='[0,1]' --port=8001
    # Use RemoteEnv in training yaml pointing to the server URL.
"""

from __future__ import annotations

import asyncio
import sys
import os
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Optional, Tuple

from vagen.envs.gym_image_env import GymImageEnv

# ---------------------------------------------------------------------------
# Lazy import helpers — old VAGEN codebase may not be on sys.path by default.
# Add the old VAGEN source tree to PYTHONPATH via the environment variable
# ACTIVE_SPATIAL_VAGEN_PATH, or install it in the current environment.
# ---------------------------------------------------------------------------

_LEGACY_VAGEN_PATH = os.environ.get(
    "ACTIVE_SPATIAL_VAGEN_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "VAGEN"),
)


def _ensure_legacy_vagen_on_path():
    """Add old VAGEN to sys.path so we can import vagen.env.active_spatial.*"""
    if _LEGACY_VAGEN_PATH not in sys.path:
        sys.path.insert(0, _LEGACY_VAGEN_PATH)


# ---------------------------------------------------------------------------
# Obs format conversion
# ---------------------------------------------------------------------------

def _convert_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old-style obs to VAGEN-Lite format.

    Old: {"obs_str": "...", "multi_modal_data": {"<image>": [PIL, ...]}}
    New: {"obs_str": "...", "multi_modal_input": {"<image>": [PIL, ...]}}
    """
    result: Dict[str, Any] = {"obs_str": obs["obs_str"]}
    if "multi_modal_data" in obs:
        result["multi_modal_input"] = obs["multi_modal_data"]
    elif "multi_modal_input" in obs:
        result["multi_modal_input"] = obs["multi_modal_input"]
    return result


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class ActiveSpatialGymEnv(GymImageEnv):
    """Active Spatial Navigation env wrapped in the GymImageEnv async interface.

    Delegates all business logic to the legacy ``ActiveSpatialEnv``.  The
    only responsibilities of this class are:

    1. Adapt the sync API to async via ``asyncio.to_thread``.
    2. Rename ``"multi_modal_data"`` → ``"multi_modal_input"`` in obs dicts.
    3. Ensure ``system_prompt()`` returns an obs dict (not a plain string).
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        # Import from locally migrated modules (no old VAGEN path needed)
        from .env import ActiveSpatialEnv
        from .env_config import ActiveSpatialEnvConfig

        # Build config from dict, silently ignoring unknown keys (e.g. base_urls
        # passed by the remote client framework).
        valid_keys = {f.name for f in dataclass_fields(ActiveSpatialEnvConfig)}
        filtered = {k: v for k, v in env_config.items() if k in valid_keys}
        cfg = ActiveSpatialEnvConfig(**filtered)

        self._inner: "ActiveSpatialEnv" = ActiveSpatialEnv(cfg)

    # ------------------------------------------------------------------
    # Sync helpers (called in a worker thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _sync_reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, info = self._inner.reset(seed)
        return _convert_obs(obs), info

    def _sync_step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._inner.step(action_str)
        return _convert_obs(obs), reward, done, info

    def _sync_close(self) -> None:
        self._inner.close()

    # ------------------------------------------------------------------
    # GymImageEnv async interface
    # ------------------------------------------------------------------

    async def system_prompt(self) -> Dict[str, Any]:
        """Return system prompt as an obs dict (no images)."""
        sp = self._inner.system_prompt()
        if isinstance(sp, str):
            return {"obs_str": sp}
        # Already an obs dict from a future version — just convert format key
        return _convert_obs(sp)

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_reset, seed)

    async def step(
        self, action_str: str
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_step, action_str)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_close)
