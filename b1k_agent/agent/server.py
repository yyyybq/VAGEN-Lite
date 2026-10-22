"""Websocket policy server that the OFFICIAL evaluator connects to.

    official eval.py  --ws-->  this server (agent loop + modules)  --ws-->  executor server (pi0.5)

The protocol mirrors OmniGibson/omnigibson/eval/utils/network_utils.py WebsocketPolicyServer:
  * GET /healthz -> 200 "OK" (the evaluator's client polls it before connecting)
  * on connect the server sends a metadata dict
  * {"reset": True} -> reset (the evaluator sends it when a batch of instances is loaded)
  * any other message is a batched observation dict (leading dim = num_envs); the reply is
    {"action": float32 (num_envs, 23)}
We never return "action_chunk": an evaluator started with --replay-action-chunk-size falls back to
one request per step (network_utils.py logs "does not support action chunks").

Each logical environment gets its own EpisodeLoop, module objects and executor connection; nothing
is shared across environments or episodes except stateless provider clients.

Run (from the repo root, in an env with numpy/pyyaml/websockets/msgpack, e.g. the openpi venv):
    python -m agent.server --config configs/L0_exec_only.yaml --port 8100 \
        --tag task=turning_on_radio --tag instance=301 --max-steps 1500
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import logging
import os
import time
import traceback
import uuid
from typing import Any, Callable, Optional

import numpy as np

from agent import context as ctxvars
from agent.actions import hold_action
from agent.config import build_manifest, config_hash, get, load_config
from agent.fields import DEFAULT_CAMERA_ROLES, build_observation
from agent.interfaces import ACTION_DIM_R1PRO, EpisodeContext
from agent.loop import EpisodeLoop, LoopConfig
from agent.recorder import RunRecorder
from agent.registry import build_modules
from agent.task_meta import TaskMetadata, load_task_metadata
from agent.wire import Packer, unpackb

logger = logging.getLogger("b1k_agent.server")


class AgentPolicy:
    """Maps batched evaluator observations onto per-environment EpisodeLoops."""

    def __init__(
        self,
        cfg: dict,
        *,
        make_executor: Callable[[int], Any],
        make_modules: Callable[[int], Any],
        task_meta: TaskMetadata,
        recorder: Optional[RunRecorder],
        run_id: str,
        max_steps: Optional[int],
        skills_for_task: Callable[[str], tuple] = lambda task: (),
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.make_executor = make_executor
        self.make_modules = make_modules
        self.task_meta = task_meta
        self.recorder = recorder
        self.run_id = run_id
        self.max_steps = max_steps
        self.skills_for_task = skills_for_task
        self.clock = clock
        self.loop_cfg = LoopConfig.from_dict(get(cfg, "loop", {}))
        self.camera_roles = dict(get(cfg, "server.camera_roles", None) or DEFAULT_CAMERA_ROLES)
        self.on_error = get(cfg, "server.on_error", "fallback")
        self.seconds_per_step = get(cfg, "budget.seconds_per_step", 1.0)
        self.enforce_budget = bool(get(cfg, "budget.enforce", True))
        self._loops: dict[int, EpisodeLoop] = {}
        self._executors: dict[int, Any] = {}
        self._active: dict[int, bool] = {}
        self._steps: dict[int, int] = {}
        self._episode_counter = 0
        self.errors = 0

    # ------------------------------------------------------------------ episode management

    def _loop(self, env_idx: int) -> EpisodeLoop:
        if env_idx not in self._loops:
            ex = self.make_executor(env_idx)
            self._executors[env_idx] = ex
            self._loops[env_idx] = EpisodeLoop(
                self.make_modules(env_idx), ex, self.loop_cfg, recorder=self.recorder, clock=self.clock
            )
        return self._loops[env_idx]

    def reset(self) -> None:
        for env_idx, loop in self._loops.items():
            if self._active.get(env_idx):
                loop.close()
            self._active[env_idx] = False
        self._episode_counter += 1

    def snapshot(self) -> None:
        if self.recorder is None:
            return
        for env_idx, loop in self._loops.items():
            if self._active.get(env_idx) and loop.state is not None:
                self.recorder.snapshot(loop.state)

    def close(self) -> None:
        self.reset()
        for ex in self._executors.values():
            with contextlib.suppress(Exception):
                ex.close()

    def _context(self, env_idx: int, task_id: int) -> EpisodeContext:
        name = self.task_meta.name(task_id)
        ex = self._executors.get(env_idx)
        executor_default = getattr(ex, "default_prompt", None)
        if callable(executor_default):
            executor_default = executor_default(name)
        max_steps = self.max_steps or self.task_meta.max_steps(name)
        if self.max_steps is None and max_steps is not None:
            logger.info("max_steps for %s from the task catalog snapshot: %d (pass --max-steps to override)", name, max_steps)
        budget = max_steps * self.seconds_per_step if (max_steps and self.enforce_budget) else None
        return EpisodeContext(
            episode_id=f"{self.run_id}-ep{self._episode_counter:03d}-env{env_idx}",
            env_index=env_idx,
            task_name=name,
            task_id=task_id,
            task_instruction=self.task_meta.prompt(name, executor_default),
            goal_spec=self.task_meta.goal_spec(name),
            skills=tuple(self.skills_for_task(name)),
            max_steps=max_steps,
            wallclock_budget_s=budget,
            seed=int(get(self.cfg, "run.seed", 0) or 0),
        )

    # ------------------------------------------------------------------ inference

    def act(self, obs: dict) -> np.ndarray:
        now = self.clock()
        proprio_keys = [k for k in obs if isinstance(k, str) and k.endswith("::proprio")]
        if len(proprio_keys) != 1:
            raise KeyError(f"expected exactly one '<robot>::proprio' key, got {proprio_keys}")
        if np.asarray(obs[proprio_keys[0]]).ndim == 1:
            # Unbatched dict (the official evaluator always stacks a leading num_envs dim; accept both).
            obs = {k: (v[None] if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        batch = int(np.asarray(obs[proprio_keys[0]]).shape[0])
        actions = np.zeros((batch, ACTION_DIM_R1PRO), dtype=np.float32)
        for i in range(batch):
            # Non-array entries (protocol fields such as "__action_chunk_size__") are not observations.
            env_obs = {
                k: v[i : i + 1]
                for k, v in obs.items()
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == batch
            }
            loop = self._loop(i)
            if not self._active.get(i):
                task_id = int(np.asarray(env_obs.get("task_id", np.array([[-1]]))).reshape(-1)[0])
                loop.reset(self._context(i, task_id))
                self._active[i] = True
                self._steps[i] = 0
            o = build_observation(env_obs, step=self._steps[i], received_at=now, camera_roles=self.camera_roles)
            tok = (
                ctxvars.current_episode.set(loop.state.ctx.episode_id),
                ctxvars.current_step.set(o.step),
            )
            try:
                actions[i] = loop.step(o)
            except Exception:
                self.errors += 1
                logger.error("agent step failed (env %d, step %d):\n%s", i, o.step, traceback.format_exc())
                if self.on_error == "raise":
                    raise
                actions[i] = self._fallback_action(i, o)
                if self.recorder is not None and loop.state is not None:
                    self.recorder.step(loop.state, o, actions[i], "fallback")  # keep steps.jsonl gap-free
            finally:
                ctxvars.current_episode.reset(tok[0])
                ctxvars.current_step.reset(tok[1])
            self._steps[i] += 1
        return actions

    def _fallback_action(self, env_idx: int, o) -> np.ndarray:
        loop = self._loops[env_idx]
        st = loop.state
        st.log("fallback", "agent error -> executor with full task prompt", "server")
        st.fallbacks += 1
        try:
            chunk = self._executors[env_idx].act(o, st.ctx.task_instruction, 1)
            return np.asarray(chunk.actions, np.float32).reshape(-1, ACTION_DIM_R1PRO)[0]
        except Exception:
            logger.error("executor fallback failed:\n%s", traceback.format_exc())
            return hold_action(o.proprio, st.last_action)


# ---------------------------------------------------------------------- websocket transport


def obs_fingerprint(msg: dict) -> Optional[str]:
    """Cheap identity of an observation message (proprio + task id + head image bytes)."""
    import hashlib

    h = hashlib.sha1()
    found = False
    for k in sorted(k for k in msg if isinstance(k, str)):
        if k.endswith("::proprio") or k == "task_id" or k.endswith("zed_link:Camera:0::rgb"):
            v = msg[k]
            if isinstance(v, np.ndarray):
                h.update(k.encode())
                h.update(np.ascontiguousarray(v).tobytes())
                found = True
    return h.hexdigest() if found else None


def _health_check(connection, request):
    if getattr(request, "path", None) == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def serve(policy: AgentPolicy, host: str, port: int, metadata: dict) -> None:
    import websockets
    import websockets.asyncio.server as ws_server

    lock = asyncio.Lock()
    last = {"fingerprint": None, "action": None}  # survives reconnections (episode state is global)

    async def handler(websocket):
        logger.info("connection from %s", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(metadata))
        prev_total = None
        first_message = True  # 2026/eval resends the current obs after a reconnect, without a reset
        while True:
            try:
                t0 = time.monotonic()
                msg = unpackb(await websocket.recv())
                if "reset" in msg:
                    async with lock:
                        await asyncio.to_thread(policy.reset)
                        last["fingerprint"], last["action"] = None, None
                    first_message = False
                    continue
                t_inf = time.monotonic()
                fp = obs_fingerprint(msg)
                async with lock:
                    if first_message and fp is not None and fp == last["fingerprint"] and last["action"] is not None:
                        logger.warning("reconnect resend detected: replying with the cached action, not stepping")
                        action = last["action"]
                    else:
                        action = await asyncio.to_thread(policy.act, msg)
                        last["fingerprint"], last["action"] = fp, action
                first_message = False
                infer_ms = (time.monotonic() - t_inf) * 1000
                reply = {"action": np.asarray(action, np.float32), "server_timing": {"infer_ms": infer_ms}}
                if prev_total is not None:
                    reply["server_timing"]["prev_total_ms"] = prev_total * 1000
                await websocket.send(packer.pack(reply))
                prev_total = time.monotonic() - t0
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                async with lock:
                    await asyncio.to_thread(policy.snapshot)
                break
            except Exception:
                logger.error("server error:\n%s", traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise

    async with ws_server.serve(handler, host, port, compression=None, max_size=None, process_request=_health_check) as srv:
        logger.info("agent server listening on %s:%d", host, port)
        await srv.serve_forever()


def build_policy(cfg: dict, run_id: str, max_steps: Optional[int], tags: dict) -> tuple[AgentPolicy, RunRecorder]:
    from executor import build_executor, load_skill_catalog
    from providers import build_providers

    recorder = RunRecorder(
        run_id,
        local_root=get(cfg, "run.local_log_root", "~/b1k_local_logs"),
        nas_root=get(cfg, "run.log_root"),
        frame_every=int(get(cfg, "run.frame_every", 300)),
        save_frames=bool(get(cfg, "run.save_frames", True)),
        tags=tags,
    )
    providers = build_providers(get(cfg, "providers", {}) or {}, recorder=recorder)
    task_meta = load_task_metadata(get(cfg, "task_metadata", {}) or {})
    catalog = load_skill_catalog(get(cfg, "executor.skill_catalog"))

    policy = AgentPolicy(
        cfg,
        make_executor=lambda env_idx: build_executor(get(cfg, "executor", {}), env_idx=env_idx),
        make_modules=lambda env_idx: build_modules(get(cfg, "modules", {}) or {}, providers, recorder=recorder),
        task_meta=task_meta,
        recorder=recorder,
        run_id=run_id,
        max_steps=max_steps,
        skills_for_task=catalog,
    )
    return policy, recorder


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", "-o", action="append", default=[], help="dotted key=value (YAML value)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="official step cap for this task (sets the 1 s/step budget)")
    ap.add_argument("--tag", action="append", default=[], help="key=value labels stored with every episode (task, instance, seed)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    tags = dict(t.split("=", 1) for t in args.tag)
    run_id = args.run_id or f"{get(cfg, 'run.name', 'run')}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    policy, recorder = build_policy(cfg, run_id, args.max_steps, tags)
    modules_desc = policy._loop(0).m.describe()
    manifest = build_manifest(cfg, modules_desc=modules_desc, extra={"run_id": run_id, "tags": tags, "max_steps": args.max_steps})
    path = recorder.write_run_file("run_manifest.json", manifest)
    logger.info("run %s config_hash=%s manifest=%s", run_id, manifest["config_hash"], path)
    metadata = {"agent": "b1k_agent", "run_id": run_id, "config_hash": config_hash(cfg)}
    host = args.host or get(cfg, "server.host", "127.0.0.1")
    port = args.port or int(get(cfg, "server.port", 8100))
    import signal

    def _terminate(signum, frame):  # the launcher stops us with SIGTERM: finish summaries first
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminate)
    try:
        asyncio.run(serve(policy, host, port, metadata))
    finally:
        policy.close()


if __name__ == "__main__":
    main()
