"""scripts/skill_probe.py against a FAKE prompt-capable executor server (serve_b1k_prompted wire format) and on
SYNTHETIC fixtures. It checks the protocol (fresh connection and reset per query, prompt override, chunk mode) and
the analysis; it says nothing about pi0.5."""

import asyncio
import socket
import threading
import time
import zlib

import numpy as np
import pytest

from agent.wire import Packer, unpackb
from scripts import record_obs_fixture as rof
from scripts import skill_probe as sp

TASK_PROMPT = "Turn on the radio receiver that's on the table in the living room."


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakeExecutorServer:
    """Chunk = f(proprio) + noise (+ a prompt-dependent arm offset when ``sensitive``)."""

    def __init__(self, sensitive: bool, noise: float = 0.01):
        self.sensitive, self.noise = sensitive, noise
        self.connections = 0
        self.resets = 0
        self.requests = []
        self.rng = np.random.default_rng(0)
        self.port = free_port()

    async def run(self):
        import websockets.asyncio.server as ws_server

        async def handler(ws):
            self.connections += 1
            packer = Packer()
            await ws.send(packer.pack({"default_prompt": TASK_PROMPT, "prompts": {"turning_on_radio": TASK_PROMPT},
                                       "multi_task": False, "supports_prompt_override": True,
                                       "checkpoint_dir": "/fake/ckpt", "train_config": "pi05_b1k"}))
            async for raw in ws:
                msg = unpackb(raw)
                if "reset" in msg:
                    self.resets += 1
                    continue
                prompt = msg.get("__prompt__") or TASK_PROMPT
                n = int(msg.get("__return_chunk__") or 1)
                self.requests.append({"prompt": prompt, "n": n, "keys": sorted(k for k in msg if not k.startswith("__"))})
                proprio = np.asarray(msg["robot_r1::proprio"], np.float32)
                base = np.zeros((1, n, 23), np.float32)
                base[0, :, 7:14] = proprio[0, 3:10]
                chunk = base + self.noise * self.rng.standard_normal(base.shape).astype(np.float32)
                if self.sensitive and prompt != TASK_PROMPT:
                    chunk[0, :, 7:14] += 0.3 + (zlib.crc32(prompt.encode()) % 7) / 10.0
                await ws.send(packer.pack({"action": chunk[:, 0], "action_chunk": chunk}))

        async with ws_server.serve(handler, "127.0.0.1", self.port, compression=None, max_size=None) as s:
            await s.serve_forever()

    def start(self):
        threading.Thread(target=lambda: asyncio.run(self.run()), daemon=True).start()
        time.sleep(0.3)
        return self


@pytest.fixture
def fixture_path(tmp_path):
    path = str(tmp_path / "synthetic_fixture")
    rof.write_fixture(path, rof.synthetic_messages(2, res=(16, 16)), synthetic=True, source="test",
                      extra={"tags": {"task": "turning_on_radio"}})
    return path


@pytest.mark.parametrize("sensitive", [True, False])
def test_probe_detects_instruction_sensitivity(fixture_path, sensitive):
    srv = FakeExecutorServer(sensitive=sensitive).start()
    instructions = ["Pick up the radio.", "Open the door."]
    res = sp.run_probe(fixture=fixture_path, host="127.0.0.1", port=srv.port, instructions=instructions,
                       repeats=3, horizon=8, max_obs=2, connect_timeout_s=10)
    # one metadata connection + one fresh connection per (instruction, repeat); a reset before every observation
    assert srv.connections == 1 + 3 * 3 and srv.resets == 3 * 3 * 2 and len(srv.requests) == 3 * 3 * 2
    assert {r["prompt"] for r in srv.requests} == {TASK_PROMPT, *instructions}
    assert all(r["n"] == 8 for r in srv.requests)
    assert "robot_r1::proprio" in srv.requests[0]["keys"]
    assert res["reference"] == TASK_PROMPT and res["chunk_len"] == 8 and res["fixture"]["synthetic"] is True
    verdicts = {i: v["verdict"] for i, v in res["chunk"]["instructions"].items()}
    if sensitive:
        assert set(verdicts.values()) == {"responds"} and res["overall"].startswith("instruction-sensitive: 2/2")
        assert res["chunk"]["instructions"]["Open the door."]["ratio"]["left_arm"] > 10
    else:
        assert set(verdicts.values()) == {"no response"} and res["overall"].startswith("NO instruction sensitivity")
        assert res["chunk"]["instructions"]["Open the door."]["ratio"]["base"] < 2
    text = sp.render(res)
    assert "SYNTHETIC fixture" in text and "noise floor (same instruction)" in text


def test_analyse_edge_cases():
    a = np.zeros((4, 23))
    b = a.copy()
    b[:, 22] = 1.0  # right gripper only
    det = sp.analyse({"ref": [[a], [a]], "x": [[b], [b]], "y": [[a], [a]]}, "ref", threshold=2.0)
    assert det["chunk"]["instructions"]["x"]["verdict"] == "responds"  # deterministic executor: any change counts
    assert det["chunk"]["instructions"]["x"]["ratio"]["right_gripper"] == float("inf")
    assert det["chunk"]["instructions"]["y"]["verdict"] == "no response"
    one = sp.analyse({"ref": [[a]], "x": [[b]]}, "ref", threshold=2.0)
    assert one["overall"].startswith("unmeasured") and one["chunk"]["noise_floor"]["base"] is None


@pytest.mark.parametrize("bad", [["--repeats", "0"], ["--max-obs", "0"], ["--horizon", "0"]])
def test_cli_rejects_empty_probes(fixture_path, bad):
    with pytest.raises(SystemExit) as e:
        sp.main(["--fixture", fixture_path, "--instruction", "Open the door.", *bad])
    assert e.value.code == 2
