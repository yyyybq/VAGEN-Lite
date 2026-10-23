"""Wire protocol tests (MOCK peers, no simulator / no model).

1. Our agent server against a client that follows the official WebsocketClientPolicy
   (OmniGibson/omnigibson/eval/utils/network_utils.py, v3.9.3-post2): /healthz, metadata frame,
   {"reset": True} without reply, batched obs -> {"action": (B, 23)}; plus the 2026/eval reconnect
   behaviour (same observation resent on a new connection without a reset -> no double step).
2. Our executor client against a fake prompt-capable openpi server (serve_b1k_prompted protocol).
3. Key renaming robot_r1 -> robot.
"""

import asyncio
import socket
import threading
import time

import httpx
import numpy as np
import pytest
from websockets.sync.client import connect

from agent.server import serve
from agent.wire import Packer, unpackb
from executor.openpi_server.keys import rename_keys
from helpers import TASK_PROMPT, make_cfg, make_policy
from synth import synth_obs


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_in_thread(coro_factory):
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=lambda: loop.run_until_complete(coro_factory()), daemon=True)
    t.start()
    return loop


def wait_health(port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError


def test_agent_server_speaks_official_protocol():
    policy, execs, _ = make_policy(make_cfg("L0_exec_only"))
    port = free_port()
    start_in_thread(lambda: serve(policy, "127.0.0.1", port, {"agent": "test"}))
    wait_health(port)
    packer = Packer()

    ws = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
    assert unpackb(ws.recv(timeout=5)) == {"agent": "test"}
    ws.send(packer.pack({"reset": True}))  # no reply expected
    obs0 = synth_obs(0)
    obs0["__action_chunk_size__"] = 16  # protocol field must be ignored, no action_chunk returned
    ws.send(packer.pack(obs0))
    reply = unpackb(ws.recv(timeout=10))
    assert set(reply) >= {"action"} and "action_chunk" not in reply
    assert reply["action"].shape == (1, 23) and reply["action"].dtype == np.float32
    ws.send(packer.pack(synth_obs(1)))
    r1 = unpackb(ws.recv(timeout=10))
    assert policy._steps[0] == 2

    # reconnect: the 2026/eval client resends the CURRENT observation without a reset
    ws.close()
    ws = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
    unpackb(ws.recv(timeout=5))
    ws.send(packer.pack(synth_obs(1)))
    r1b = unpackb(ws.recv(timeout=10))
    np.testing.assert_array_equal(r1["action"], r1b["action"])
    assert policy._steps[0] == 2, "a resent observation must not advance the episode"
    ws.send(packer.pack(synth_obs(2)))
    unpackb(ws.recv(timeout=10))
    assert policy._steps[0] == 3

    # batched: two logical environments in one request
    ws.send(packer.pack({"reset": True}))
    ws.send(packer.pack(synth_obs(0, num_envs=2)))
    assert unpackb(ws.recv(timeout=10))["action"].shape == (2, 23)
    ws.close()


class FakePromptedOpenPI:
    """Mimics executor/openpi_server/serve_b1k_prompted.py's wire behaviour."""

    def __init__(self, multi_task=False):
        self.requests = []
        self.resets = 0
        self.multi_task = multi_task

    async def run(self, port):
        import websockets.asyncio.server as ws_server

        async def handler(ws):
            packer = Packer()
            await ws.send(packer.pack({"default_prompt": TASK_PROMPT, "prompts": {"turning_on_radio": TASK_PROMPT},
                                       "multi_task": self.multi_task, "supports_prompt_override": True}))
            async for raw in ws:
                msg = unpackb(raw)
                if "reset" in msg:
                    self.resets += 1
                    continue
                self.requests.append({k: v for k, v in msg.items() if isinstance(k, str) and k.startswith("__")})
                n = int(msg.get("__return_chunk__") or 0)
                a = np.full((1, 23), 0.5, np.float32)
                reply = {"action": a}
                if n:
                    reply["action_chunk"] = np.repeat(a[:, None], n, axis=1)
                await ws.send(packer.pack(reply))

        async with ws_server.serve(handler, "127.0.0.1", port, compression=None, max_size=None) as s:
            await s.serve_forever()


def _obs(t):
    from agent.fields import build_observation

    return build_observation({k: v[0:1] for k, v in synth_obs(t).items()}, step=t, received_at=0.0)


def _ctx():
    from agent.interfaces import EpisodeContext

    return EpisodeContext(episode_id="e", env_index=0, task_name="turning_on_radio", task_id=0,
                          task_instruction=TASK_PROMPT, goal_spec=None, skills=(), max_steps=None, wallclock_budget_s=None)


@pytest.mark.parametrize("mode", ["passthrough", "chunk"])
def test_executor_prompt_override_and_flush(mode):
    from executor.openpi_ws import OpenPIWebsocketExecutor

    fake = FakePromptedOpenPI()
    port = free_port()
    start_in_thread(lambda: fake.run(port))
    time.sleep(0.3)
    ex = OpenPIWebsocketExecutor({"port": port, "mode": mode, "connect_timeout_s": 10})
    assert ex.default_prompt("turning_on_radio") == TASK_PROMPT
    ex.reset(_ctx())
    c = ex.act(_obs(0), TASK_PROMPT, 16)
    assert fake.requests[-1].get("__prompt__") is None, "task prompt must not be sent as an override"
    assert c.actions.shape == ((16, 23) if mode == "chunk" else (1, 23))
    ex.act(_obs(1), TASK_PROMPT, 16)
    assert "__flush__" not in fake.requests[-1]
    ex.act(_obs(2), "Go to the radio.", 16)
    assert fake.requests[-1]["__prompt__"] == "Go to the radio." and fake.requests[-1]["__flush__"]
    ex.act(_obs(3), "Go to the radio.", 16)
    assert "__flush__" not in fake.requests[-1]
    ex.interrupt()
    ex.act(_obs(4), TASK_PROMPT, 16)
    assert fake.requests[-1]["__flush__"] and "__prompt__" not in fake.requests[-1]
    assert fake.resets == 1
    ex.close()


def test_executor_multitask_server_always_gets_prompt():
    from executor.openpi_ws import OpenPIWebsocketExecutor

    fake = FakePromptedOpenPI(multi_task=True)
    port = free_port()
    start_in_thread(lambda: fake.run(port))
    time.sleep(0.3)
    ex = OpenPIWebsocketExecutor({"port": port, "connect_timeout_s": 10})
    ex.reset(_ctx())
    ex.act(_obs(0), TASK_PROMPT, 1)
    assert fake.requests[-1]["__prompt__"] == TASK_PROMPT
    ex.close()


def test_rename_keys():
    obs = {"robot_r1::robot_r1:zed_link:Camera:0::rgb": 1, "robot_r1::proprio": 2, "task_id": 3, "robot_r1::cam_rel_poses": 4}
    out = rename_keys(obs, "robot_r1", "robot")
    assert out == {"robot::robot:zed_link:Camera:0::rgb": 1, "robot::proprio": 2, "task_id": 3, "robot::cam_rel_poses": 4}
    assert rename_keys(obs, None, "robot") is obs
