"""M-exec: executors behind one interface (agent.interfaces.Executor), plus skill catalogs.

impls
  openpi_ws   pi0.5 served by executor/openpi_server/serve_b1k_prompted.py (openpi venv)
  hold        holds the robot still (base 0, joints at their current positions). For pipeline smoke
              tests only, never for reported results.

A skill catalog (configs/skills/*.yaml) describes what the CURRENT executor can do. The planner may
only use skills marked available; "verified" records the Step-5 diagnosis result.
"""

from __future__ import annotations

import fnmatch
import os
from typing import Any, Callable, Mapping, Optional

import numpy as np
import yaml

from agent.actions import hold_action
from agent.interfaces import FULL_TASK_SKILL, ActionChunk, EpisodeContext, Observation, SkillSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HoldExecutor:
    name = "hold"

    def __init__(self, params: Mapping[str, Any], env_idx: int = 0):
        self._last = None

    def reset(self, ctx: EpisodeContext) -> None:
        self._last = None

    def act(self, obs: Observation, instruction: str, horizon: int) -> ActionChunk:
        a = hold_action(obs.proprio, self._last)
        self._last = a
        return ActionChunk(actions=a[None], instruction=instruction, source="exec.hold", open_loop_steps=1)

    def interrupt(self) -> None:
        pass

    def close(self) -> None:
        pass


def _openpi_ws(params, env_idx):
    from executor.openpi_ws import OpenPIWebsocketExecutor

    return OpenPIWebsocketExecutor(params, env_idx=env_idx)


IMPLS: dict[str, Callable[[Mapping[str, Any], int], Any]] = {
    "openpi_ws": _openpi_ws,
    "hold": lambda params, env_idx: HoldExecutor(params, env_idx),
}


def build_executor(cfg: Mapping[str, Any], env_idx: int = 0):
    impl = str((cfg or {}).get("impl", "openpi_ws"))
    if impl not in IMPLS:
        raise KeyError(f"unknown executor impl {impl!r}; available {sorted(IMPLS)}")
    params = dict((cfg or {}).get("params") or {})
    # per-env port offset when several executor servers are used (one per env)
    if params.get("port_per_env"):
        params["port"] = int(params.get("port", 8000)) + env_idx
    return IMPLS[impl](params, env_idx)


def load_skill_catalog(path: Optional[str]) -> Callable[[str], tuple]:
    """Return task_name -> tuple[SkillSpec]. Without a catalog only the full-task skill exists."""
    full_task = SkillSpec(
        name=FULL_TASK_SKILL,
        instruction_template="{task_prompt}",
        description="Run the executor on the whole task with its trained task prompt.",
        available=True,
        verified=False,
    )
    if not path:
        return lambda task: (full_task,)
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    entries = []
    for s in data.get("skills") or []:
        entries.append(
            (
                tuple(s.get("tasks") or ["*"]),
                SkillSpec(
                    name=s["name"],
                    instruction_template=s.get("instruction_template", ""),
                    description=s.get("description", ""),
                    inputs=tuple(s.get("inputs") or ()),
                    ends_when=s.get("ends_when", ""),
                    known_failures=tuple(s.get("known_failures") or ()),
                    available=bool(s.get("available", False)),
                    verified=bool(s.get("verified", False)),
                    evidence=s.get("evidence", ""),
                ),
            )
        )

    def for_task(task: str) -> tuple:
        out = [spec for patterns, spec in entries if any(fnmatch.fnmatch(task, p) for p in patterns)]
        if not any(s.name == FULL_TASK_SKILL for s in out):
            out.insert(0, full_task)
        return tuple(out)

    return for_task


__all__ = ["IMPLS", "HoldExecutor", "build_executor", "load_skill_catalog", "np"]
