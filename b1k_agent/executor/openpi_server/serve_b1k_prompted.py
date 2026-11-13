"""pi0.5 B1K policy server with per-request prompt override. Runs in the openpi venv.

This is openpi's scripts/b1k/serve_b1k.py (wensi-ai/openpi, branch "behavior") with four additions;
openpi's own tree is imported, not modified:

  1. One B1KPolicyWrapper per websocket connection (the model is loaded once and shared), so each
     logical environment / agent episode has its own action buffer.
  2. Optional request keys (absent => byte-for-byte the serve_b1k behaviour):
       "__prompt__":       str   use this text instead of the default task prompt for this request
       "__flush__":        bool  drop buffered actions before acting (new subtask / after a scripted motion)
       "__return_chunk__": int   return the raw model chunk (first N actions) as "action_chunk"
     A prompt different from the previous request's prompt also flushes the buffer.
  3. Robot-name translation: the openpi R1Pro config reads keys "robot::...", the official evaluator
     (OmniGibson/omnigibson/eval/r1pro.yaml, name robot_r1) sends "robot_r1::robot_r1:...". With
     --rename-robot-from robot_r1 the incoming KEY NAMES are rewritten; values are untouched.
  4. The connect-time metadata adds default_prompt, prompts, control_mode, action_horizon,
     supports_prompt_override, so the agent knows the trained task prompt.

Example (baseline settings from docs/challenge/baselines.md):
    cd /nas/baiqiao/openpi && CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    .venv/bin/python /nas/baiqiao/b1k_agent/executor/openpi_server/serve_b1k_prompted.py \
        --robot b1k/R1Pro --task b1k/turning_on_radio --repo-id turning_on_radio \
        --policy.config pi05_b1k --policy.dir /nas/baiqiao/checkpoints/pi05_turn_on_the_radio \
        --control-mode receding_horizon --action-horizon 16 --port 8000 --rename-robot-from robot_r1
"""

from __future__ import annotations

import asyncio
import dataclasses
import http
import logging
import time
import traceback
from copy import deepcopy
from typing import Optional

import numpy as np
import tyro

from openpi.configs.tasks import TASK_REGISTRY
from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_b1k_server import Packer, unpackb
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config

try:  # run as a script from any cwd
    from executor.openpi_server.keys import rename_keys
except ImportError:  # pragma: no cover
    import os as _os, sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from keys import rename_keys

logger = logging.getLogger("serve_b1k_prompted")


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Args:
    robot: str
    task: str
    policy: Checkpoint
    repo_id: Optional[str] = None
    control_mode: str = "receding_horizon"
    action_horizon: int = 16
    port: int = 8000
    host: str = "0.0.0.0"
    # Rewrite incoming key prefixes "<from>::<from>:" / "<from>::" to the openpi robot name.
    rename_robot_from: Optional[str] = None
    # Expose every prompt of this task bucket (multi-task checkpoints); the default prompt stays --task's.
    multi_task: bool = False


class PromptedB1KSession:
    """Per-connection state: its own B1KPolicyWrapper around the shared policy."""

    def __init__(self, policy, args: Args, robot_cfg_name: str, default_prompt: str, max_len: int):
        self.default_prompt = default_prompt
        self.wrapper = B1KPolicyWrapper(
            policy=policy,
            robot=robot_cfg_name,
            text_prompt=default_prompt,
            control_mode=args.control_mode,
            action_horizon=args.action_horizon,
            max_len=max_len,
        )
        self.robot_name = self.wrapper.robot.name
        self.args = args
        self.last_prompt = default_prompt

    def reset(self) -> None:
        self.wrapper.reset()
        self.wrapper.text_prompt = self.default_prompt
        self.last_prompt = self.default_prompt

    def act(self, msg: dict) -> dict:
        prompt = msg.pop("__prompt__", None) or self.default_prompt
        flush = bool(msg.pop("__flush__", False))
        chunk_n = msg.pop("__return_chunk__", None)
        msg.pop("__action_chunk_size__", None)
        obs = rename_keys(msg, self.args.rename_robot_from, self.robot_name)
        if prompt != self.last_prompt:
            flush = True
        if flush:
            self.wrapper.reset()
        self.wrapper.text_prompt = prompt
        self.last_prompt = prompt
        if chunk_n:
            inputs = self.wrapper.process_input(obs)
            chunks = np.stack([self.wrapper.policy.infer(x)["actions"] for x in inputs])  # (B, T, 23)
            chunk = chunks[:, : int(chunk_n)]
            return {"action": chunk[:, 0], "action_chunk": chunk}
        action = self.wrapper.act(obs)
        return {"action": np.asarray(action.cpu().numpy() if hasattr(action, "cpu") else action)}


def _health_check(connection, request):
    if getattr(request, "path", None) == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def run(args: Args) -> None:
    import websockets
    import websockets.asyncio.server as ws_server

    task_bucket, task_name = args.task.split("/")
    default_prompt = TASK_REGISTRY[task_bucket][task_name]
    config = _config.get_config(args.policy.config)
    norm_stats_repo_id = args.repo_id or args.task
    config = dataclasses.replace(
        config, data=dataclasses.replace(config.data, repo_id=norm_stats_repo_id, robot_config_name=args.robot)
    )
    policy = _policy_config.create_trained_policy(config, args.policy.dir, default_prompt=default_prompt)
    metadata = dict(policy.metadata or {})
    metadata.update(
        {
            "default_prompt": default_prompt,
            "prompts": dict(TASK_REGISTRY[task_bucket]) if args.multi_task else {task_name: default_prompt},
            "multi_task": bool(args.multi_task),
            "control_mode": args.control_mode,
            "action_horizon": args.action_horizon,
            "model_action_horizon": config.model.action_horizon,
            "supports_prompt_override": True,
            "checkpoint_dir": args.policy.dir,
            "train_config": args.policy.config,
            "norm_stats_repo_id": norm_stats_repo_id,
        }
    )
    lock = asyncio.Lock()  # one model; serialize inference across connections

    async def handler(websocket):
        logger.info("connection from %s", websocket.remote_address)
        session = PromptedB1KSession(policy, args, args.robot, default_prompt, config.model.action_horizon)
        packer = Packer()
        await websocket.send(packer.pack(metadata))
        while True:
            try:
                msg = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in msg:
                    session.reset()
                    continue
                t0 = time.monotonic()
                async with lock:
                    reply = await asyncio.to_thread(session.act, deepcopy(msg))
                reply["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
                await websocket.send(packer.pack(reply))
            except websockets.ConnectionClosed:
                logger.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                logger.error("error:\n%s", traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise

    async with ws_server.serve(
        handler, args.host, args.port, compression=None, max_size=None, process_request=_health_check
    ) as server:
        logger.info("serving %s (%s) on %s:%d, prompt=%r", args.policy.dir, args.policy.config, args.host, args.port,
                    default_prompt)
        await server.serve_forever()


def main(args: Args) -> None:
    asyncio.run(run(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
