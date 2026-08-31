from __future__ import annotations

from typing import Any, Dict

from vagen.envs_remote.handler import BaseGymHandler

from .active_spatial_env import ActiveSpatialGymEnv


class ActiveSpatialHandler(BaseGymHandler):
    """Remote-service handler for Active Spatial environments.

    This is intentionally thin: the upstream ``GymService`` owns session
    management, multipart image transport, retries, and admission control. The
    handler only instantiates the Active Spatial gym wrapper on the service
    machine, where ``render_backend="local"`` can keep GS rendering colocated
    with the environment.
    """

    async def create_env(self, env_config: Dict[str, Any]) -> ActiveSpatialGymEnv:
        return ActiveSpatialGymEnv(env_config)
