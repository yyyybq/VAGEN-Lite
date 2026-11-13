"""Executor client for a pi0.5 policy served by executor/openpi_server/serve_b1k_prompted.py.

Modes
  passthrough (default)  forward the evaluator's observation dict unchanged and get ONE action per
                         step; the server applies the official B1KPolicyWrapper chunking rule
                         (--control_mode receding_horizon --action_horizon 16 in the baseline).
                         With no prompt override this is the official baseline action stream.
  chunk                  ask for the raw model chunk (H actions) and let the loop run H steps
                         open loop (equivalent to receding_horizon with action_horizon=H).

Language: when the instruction differs from the server's default prompt the request carries
"__prompt__"; the server swaps the prompt and drops its buffered actions. Whether the policy
actually RESPONDS to a non-task instruction is an empirical question (Step-5 skill diagnosis);
the skill catalog records the answer.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Optional

import numpy as np

from agent.interfaces import ACTION_DIM_R1PRO, ActionChunk, EpisodeContext, Observation
from agent.wire import Packer, unpackb

logger = logging.getLogger("b1k_agent.executor.openpi_ws")


class ExecutorError(RuntimeError):
    pass


class OpenPIWebsocketExecutor:
    name = "openpi_ws"

    def __init__(self, params: Mapping[str, Any], env_idx: int = 0):
        self.host = str(params.get("host", "127.0.0.1"))
        self.port = int(params.get("port", 8000))
        self.mode = str(params.get("mode", "passthrough"))
        if self.mode not in ("passthrough", "chunk"):
            raise ValueError(f"unknown executor mode {self.mode!r}")
        self.recv_timeout_s = float(params.get("recv_timeout_s", 120.0))
        self.connect_timeout_s = float(params.get("connect_timeout_s", 600.0))
        self.allow_prompt_override = bool(params.get("allow_prompt_override", True))
        self.env_idx = env_idx
        self._packer = Packer()
        self._ws = None
        self._meta: dict = {}
        self._flush_next = False
        self._last_prompt: Optional[str] = None
        self._task_prompt: Optional[str] = None

    # ------------------------------------------------------------------ connection

    def _connect(self) -> None:
        from websockets.sync.client import connect

        deadline = time.monotonic() + self.connect_timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                self._ws = connect(
                    f"ws://{self.host}:{self.port}",
                    compression=None,
                    max_size=None,
                    open_timeout=30,
                    ping_interval=60,
                    ping_timeout=300,
                )
                self._meta = unpackb(self._ws.recv(timeout=60))
                logger.info("executor env%d connected to %s:%d meta=%s", self.env_idx, self.host, self.port,
                            {k: v for k, v in self._meta.items() if k != "prompts"})
                return
            except Exception as e:  # server still loading the checkpoint
                last_err = e
                time.sleep(5)
        raise ExecutorError(f"could not connect to executor server {self.host}:{self.port}: {last_err}")

    def _ensure(self) -> None:
        if self._ws is None:
            self._connect()

    @property
    def metadata(self) -> dict:
        self._ensure()
        return dict(self._meta)

    def default_prompt(self, task_name: str) -> Optional[str]:
        """The prompt the executor server uses for THIS task, or None when it does not serve the task
        (a single-task server publishes only its own task; it must not lend its prompt to others)."""
        return (self.metadata.get("prompts") or {}).get(task_name)

    # ------------------------------------------------------------------ Executor protocol

    def reset(self, ctx: EpisodeContext) -> None:
        self._ensure()
        try:
            self._ws.send(self._packer.pack({"reset": True}))
        except Exception as e:  # connection dropped between episodes: reconnect (fresh server session)
            logger.warning("executor env%d reset failed (%s); reconnecting", self.env_idx, e)
            self.close()
            self._connect()
            self._ws.send(self._packer.pack({"reset": True}))
        self._flush_next = False
        self._last_prompt = None
        self._task_prompt = self.default_prompt(ctx.task_name)

    def interrupt(self) -> None:
        self._flush_next = True

    def act(self, obs: Observation, instruction: str, horizon: int) -> ActionChunk:
        self._ensure()
        req = dict(obs.policy_dict)
        override = instruction if (instruction and instruction != self._task_prompt) else None
        if override is not None and not self.allow_prompt_override:
            raise ExecutorError("instruction differs from the executor's prompt for this task and overrides are disabled")
        if self._meta.get("multi_task"):
            # a multi-task server needs the prompt on every request (the per-task prompt or the override)
            prompt = instruction or self._task_prompt
            if prompt:
                req["__prompt__"] = prompt
        elif override is not None:
            req["__prompt__"] = override
        if self._flush_next or override != self._last_prompt:
            req["__flush__"] = True
        if self.mode == "chunk":
            req["__return_chunk__"] = int(max(1, horizon))
        t0 = time.monotonic()
        reply = self._roundtrip(req)
        infer_ms = (time.monotonic() - t0) * 1000.0
        self._flush_next = False
        self._last_prompt = override
        if self.mode == "chunk":
            actions = np.asarray(reply["action_chunk"], np.float32).reshape(-1, ACTION_DIM_R1PRO)
            n = actions.shape[0]
        else:
            actions = np.asarray(reply["action"], np.float32).reshape(-1, ACTION_DIM_R1PRO)[:1]
            n = 1
        return ActionChunk(
            actions=actions,
            instruction=instruction,
            source=f"exec.openpi_ws@{self.mode}",
            infer_ms=infer_ms,
            open_loop_steps=n,
            meta={"prompt_override": override is not None, "server_timing": reply.get("server_timing")},
        )

    def _roundtrip(self, req: dict) -> dict:
        data = self._packer.pack(req)
        for attempt in range(2):
            try:
                self._ws.send(data)
                raw = self._ws.recv(timeout=self.recv_timeout_s)
                if isinstance(raw, str):
                    raise ExecutorError(f"executor server error:\n{raw}")
                reply = unpackb(raw)
                if "action" not in reply and "action_chunk" not in reply:
                    raise ExecutorError(f"executor reply without action: {list(reply)}")
                return reply
            except ExecutorError:
                raise
            except Exception as e:
                logger.warning("executor roundtrip failed (attempt %d): %s", attempt + 1, e)
                self.close()
                if attempt == 0:
                    self._connect()
                    self._ws.send(self._packer.pack({"reset": True}))
                    continue
                raise ExecutorError(f"executor roundtrip failed: {e}") from e
        raise ExecutorError("unreachable")

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws = None
