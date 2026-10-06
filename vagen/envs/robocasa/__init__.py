"""RoboCasa kitchen environment for VAGEN-Lite evaluation and SFT.

Heavy imports (RoboCasaEnv / handler / PIL) are lazy so action parsers
and SFT conversion can run without gymnasium, robocasa, or Pillow.
"""

from typing import Any

__all__ = ["RoboCasaEnv", "RoboCasaEnvConfig", "RoboCasaHandler"]


def __getattr__(name: str) -> Any:
    if name in {"RoboCasaEnv", "RoboCasaEnvConfig"}:
        from vagen.envs.robocasa.robocasa_env import RoboCasaEnv, RoboCasaEnvConfig

        return RoboCasaEnv if name == "RoboCasaEnv" else RoboCasaEnvConfig
    if name == "RoboCasaHandler":
        from vagen.envs.robocasa.handler import RoboCasaHandler

        return RoboCasaHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
