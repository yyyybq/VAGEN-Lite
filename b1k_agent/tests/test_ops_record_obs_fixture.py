"""scripts/record_obs_fixture.py: official-protocol round trip with a SYNTHETIC websocket client (websockets sync
client, synthetic observations). Checks hold actions, the saved fixture and the manifest; says nothing about
real simulator observations."""

import asyncio
import os
import socket
import threading
import time

import httpx
import numpy as np
from websockets.sync.client import connect

from agent.actions import hold_action
from agent.wire import Packer, unpackb
from scripts import record_obs_fixture as rof


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_health(port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1).status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError("fixture recorder did not answer /healthz")


def test_round_trip_saves_fixture_and_answers_hold_actions(tmp_path):
    out = str(tmp_path / "fixtures" / "synthetic_obs")
    rec = rof.FixtureRecorder(out, n=2, synthetic=True, tags={"task": "turning_on_radio"}, exit_after_save=True)
    port = free_port()
    server = threading.Thread(target=lambda: asyncio.run(rof.serve(rec, "127.0.0.1", port)), daemon=True)
    server.start()
    wait_health(port)

    sent = rof.synthetic_messages(3, num_envs=2, res=(32, 32), depth=True)
    sent[0]["__action_chunk_size__"] = 16  # non-array protocol field: kept in the manifest, not in the npz
    packer = Packer()
    ws = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None)
    meta = unpackb(ws.recv(timeout=5))
    assert meta["moves_nothing"] is True
    ws.send(packer.pack({"reset": True}))  # no reply expected
    replies = []
    for msg in sent[:2]:
        ws.send(packer.pack(msg))
        replies.append(unpackb(ws.recv(timeout=10)))
    ws.close()
    server.join(timeout=10)
    assert not server.is_alive(), "--exit-after-save should stop the server once N observations are saved"

    for msg, reply in zip(sent, replies):
        a = reply["action"]
        assert a.shape == (2, 23) and a.dtype == np.float32
        for e in range(2):
            np.testing.assert_allclose(a[e], hold_action(msg["robot_r1::proprio"][e]))
        assert np.all(a[:, :3] == 0.0)  # base velocity zero: it moves nothing

    msgs, manifest = rof.load_fixture(out + ".npz")
    assert manifest["synthetic"] is True and manifest["complete"] is True and manifest["resets_seen"] == 1
    assert manifest["tags"] == {"task": "turning_on_radio"} and len(msgs) == 2
    assert manifest["messages"][0]["non_array"] == {"__action_chunk_size__": 16}
    for got, want in zip(msgs, sent):
        arrays = {k: v for k, v in want.items() if isinstance(v, np.ndarray)}
        assert set(got) == set(arrays)
        for k, v in arrays.items():
            np.testing.assert_array_equal(got[k], v)
    cats = manifest["messages"][0]["keys"]
    assert cats["robot_r1::proprio"]["category"] == "proprio" and cats["task_id"]["shape"] == [2, 1]
    assert cats["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"]["category"] == "camera"


def test_partial_flush_and_synthetic_writer(tmp_path):
    rec = rof.FixtureRecorder(str(tmp_path / "partial"), n=5)
    rec.on_message({"reset": True})
    rec.on_message(rof.synthetic_messages(1)[0])
    npz, _ = rec.flush()
    msgs, manifest = rof.load_fixture(npz)
    assert len(msgs) == 1 and manifest["complete"] is False and manifest["synthetic"] is False

    assert rof.main(["--write-synthetic", str(tmp_path / "syn"), "--n", "2", "--res", "16", "16"]) == 0
    msgs, manifest = rof.load_fixture(str(tmp_path / "syn"))
    assert manifest["synthetic"] is True and len(msgs) == 2
    assert msgs[0]["robot_r1::robot_r1:zed_link:Camera:0::rgb"].shape == (1, 16, 16, 4)
    assert os.path.isfile(str(tmp_path / "syn.json"))
