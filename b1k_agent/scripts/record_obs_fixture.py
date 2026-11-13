"""Fixture recorder: a websocket "policy" that answers the OFFICIAL evaluator with hold actions and saves the
first N observations it receives, to build real-observation test fixtures (and for pipeline smoke tests).

    # on 105 (any python with numpy / websockets / msgpack, e.g. the openpi venv), from the repo root:
    python scripts/record_obs_fixture.py --port 8100 --n 5 --out tests/fixtures/real_obs_turning_on_radio_311 \
        --tag task=turning_on_radio --tag instance=311 --tag wrapper=rgbd
    # then, in the behavior env, on a GPU that nvidia-smi shows free (< 2048 MiB, < 10 %), point the official
    # evaluator at it (a Dev instance; --max-steps keeps the hold-only episode short instead of the full timeout):
    CUDA_VISIBLE_DEVICES=<free> OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES \
    python -m omnigibson.eval.eval --task-name turning_on_radio --mode public_test --instance-indices 10 \
        --host 127.0.0.1 --port 8100 --output-dir /tmp/fixture_eval --max-steps 20 \
        --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper
    # without --exit-after-save the recorder keeps serving until the evaluator finishes; stop it with Ctrl-C

IT MOVES NOTHING. Every reply is agent.actions.hold_action(proprio) for each environment: base velocity 0, torso
and arm joints commanded to their current positions, gripper commands from the current finger widths. It is not a
policy and is never used for results: an episode served by it is a fixture capture or a smoke test only.

Protocol (as agent/server.py, which mirrors OmniGibson/omnigibson/eval/utils/network_utils.py
WebsocketPolicyServer): GET /healthz -> 200; a metadata dict on connect; {"reset": True} -> no reply; any other
message is a batched observation dict -> {"action": float32 (num_envs, 23)}.

Fixture format (load_fixture reads it; scripts/skill_probe.py uses it):
  <out>.npz   one array per (message, key), named "<message index:04d>/<evaluator key>", exactly as received:
              leading num_envs dim kept, ALL keys, including those agent/fields.py would drop
  <out>.json  manifest: format, synthetic flag, source, tags, and per message the receive time, each key's shape,
              dtype and whitelist category (agent/fields.py classify_key; "dropped" = not a policy observation),
              and the non-array fields
--write-synthetic PATH writes a SYNTHETIC fixture instead (official keys and shapes, random pixels,
"synthetic": true): for wiring checks only, never evidence about the robot or the executor.

Status: written and unit-tested with a synthetic websocket client (tests/test_ops_record_obs_fixture.py); not yet
run against the simulator on 105.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.actions import hold_action
from agent.fields import PROPRIO_DIM, classify_key
from agent.server import _health_check
from agent.wire import Packer, unpackb

logger = logging.getLogger("record_obs_fixture")

FORMAT = "b1k_obs_fixture/v1"
METADATA = {"policy": "record_obs_fixture", "moves_nothing": True,
            "note": "hold actions only; fixture capture / smoke test, never a result"}
ROBOT = "robot_r1"
CAMERA_LINKS = ("zed_link", "left_realsense_link", "right_realsense_link")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _name(key: Any) -> str:
    return key.decode() if isinstance(key, bytes) else str(key)


def hold_actions(msg: Mapping) -> np.ndarray:
    """(num_envs, 23) hold actions from the message's '<robot>::proprio' (batched or a single env)."""
    keys = [k for k in msg if _name(k).endswith("::proprio")]
    if len(keys) != 1:
        raise KeyError(f"expected exactly one '<robot>::proprio' key, got {[_name(k) for k in keys]}")
    proprio = np.asarray(msg[keys[0]], np.float32)
    if proprio.ndim == 1:
        proprio = proprio[None]
    return np.stack([hold_action(p) for p in proprio]).astype(np.float32)


def _base(path: str) -> str:
    return path[:-4] if path.endswith((".npz", ".json")) else path


def write_fixture(path: str, messages: Sequence[Mapping], *, synthetic: bool, source: str,
                  received_at: Optional[Sequence[str]] = None, extra: Optional[dict] = None) -> tuple[str, str]:
    base = _base(path)
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    arrays, entries = {}, []
    for i, msg in enumerate(messages):
        keys, other = {}, {}
        for k, v in msg.items():
            name = _name(k)
            if isinstance(v, np.ndarray):
                arrays[f"{i:04d}/{name}"] = v
                keys[name] = {"shape": list(v.shape), "dtype": str(v.dtype), "category": classify_key(name) or "dropped"}
            else:
                other[name] = v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)
        entries.append({"index": i, "received_at": received_at[i] if received_at else None, "keys": keys,
                        "non_array": other})
    np.savez_compressed(base + ".npz", **arrays)
    manifest = {"format": FORMAT, "synthetic": bool(synthetic), "source": source, "created_at": now_iso(),
                "n_messages": len(messages), **(extra or {}), "messages": entries}
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return base + ".npz", base + ".json"


def load_fixture(path: str) -> tuple[list[dict], dict]:
    """(messages in order, manifest). A missing manifest gives {"synthetic": None} (provenance unknown)."""
    base = _base(path)
    manifest = {"format": None, "synthetic": None, "source": "unknown (no manifest)"}
    if os.path.isfile(base + ".json"):
        with open(base + ".json", encoding="utf-8") as f:
            manifest = json.load(f)
    by_index: dict = collections.defaultdict(dict)
    with np.load(base + ".npz") as z:
        for name in z.files:
            idx, key = name.split("/", 1)
            by_index[int(idx)][key] = z[name]
    return [by_index[i] for i in sorted(by_index)], manifest


def synthetic_messages(n: int, num_envs: int = 1, res: tuple = (224, 224), depth: bool = False, task_id: int = 0,
                       seed: int = 0) -> list[dict]:
    """SYNTHETIC observation dicts with the official evaluator's keys, shapes and dtypes (as tests/synth.py)."""
    rng = np.random.default_rng(seed)
    out = []
    for step in range(n):
        obs = {}
        for link in CAMERA_LINKS:
            obs[f"{ROBOT}::{ROBOT}:{link}:Camera:0::rgb"] = rng.integers(0, 255, size=(num_envs, *res, 4), dtype=np.uint8)
            if depth:
                obs[f"{ROBOT}::{ROBOT}:{link}:Camera:0::depth_linear"] = rng.uniform(0.2, 5.0, size=(num_envs, *res)).astype(np.float32)
        proprio = np.zeros((num_envs, PROPRIO_DIM), np.float32)
        proprio[:, 3:10] = 0.1 * np.sin(step / 30.0)
        proprio[:, 24:26] = 0.05
        proprio[:, 49:51] = 0.05
        proprio[:, 53:57] = np.array([1.025, -1.45, -0.47, 0.0], np.float32)
        obs[f"{ROBOT}::proprio"] = proprio
        obs[f"{ROBOT}::cam_rel_poses"] = np.zeros((num_envs, 21), np.float32)
        obs["task_id"] = np.full((num_envs, 1), task_id, np.int64)
        out.append(obs)
    return out


class FixtureRecorder:
    """Keeps the first ``n`` observation messages and answers every message with hold actions."""

    def __init__(self, out: str, n: int, *, synthetic: bool = False, tags: Optional[dict] = None,
                 exit_after_save: bool = False):
        self.out, self.n = out, int(n)
        self.synthetic = synthetic
        self.tags = dict(tags or {})
        self.exit_after_save = exit_after_save
        self.saved: list[dict] = []
        self.times: list[str] = []
        self.messages = 0
        self.resets = 0
        self.written: Optional[tuple[str, str]] = None

    def on_message(self, msg: Mapping) -> Optional[dict]:
        if "reset" in msg:
            self.resets += 1
            return None
        self.messages += 1
        if len(self.saved) < self.n:
            self.saved.append({_name(k): (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in msg.items()})
            self.times.append(now_iso())
            if len(self.saved) == self.n:
                self.flush()
        return {"action": hold_actions(msg)}

    def flush(self) -> Optional[tuple[str, str]]:
        if not self.saved or (self.written is not None and len(self.saved) == self.n):
            return self.written
        source = "test client (synthetic)" if self.synthetic else "official evaluator via record_obs_fixture.py"
        self.written = write_fixture(
            self.out, self.saved, synthetic=self.synthetic, source=source, received_at=self.times,
            extra={"tags": self.tags, "complete": len(self.saved) == self.n, "requested": self.n,
                   "resets_seen": self.resets, "policy": METADATA})
        logger.info("saved %d observation(s) -> %s", len(self.saved), self.written[0])
        return self.written

    @property
    def done(self) -> bool:
        return self.written is not None and len(self.saved) == self.n


async def serve(rec: FixtureRecorder, host: str, port: int, stop: Optional[asyncio.Event] = None) -> None:
    import websockets
    import websockets.asyncio.server as ws_server

    stop = stop or asyncio.Event()

    async def handler(websocket):
        logger.info("connection from %s", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(METADATA))
        try:
            async for raw in websocket:
                reply = rec.on_message(unpackb(raw))
                if reply is not None:
                    await websocket.send(packer.pack(reply))
                if rec.exit_after_save and rec.done:
                    stop.set()
        except websockets.ConnectionClosed:
            pass
        logger.info("connection from %s closed", websocket.remote_address)

    async with ws_server.serve(handler, host, port, compression=None, max_size=None, process_request=_health_check):
        logger.info("fixture recorder on %s:%d (hold actions only), saving %d observation(s) to %s",
                    host, port, rec.n, rec.out)
        await stop.wait()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--n", type=int, default=5, help="observation messages to save")
    ap.add_argument("--out", default=None, help="fixture path without extension (.npz + .json are written)")
    ap.add_argument("--tag", action="append", default=[], help="key=value stored in the manifest (task, instance, ...)")
    ap.add_argument("--exit-after-save", action="store_true",
                    help="stop serving once N observations are saved (the evaluator then loses its connection)")
    ap.add_argument("--mark-synthetic", action="store_true", help="the client is a test client, not the evaluator")
    ap.add_argument("--write-synthetic", default=None, metavar="PATH", help="write a SYNTHETIC fixture and exit")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--res", type=int, nargs=2, default=(224, 224))
    ap.add_argument("--depth", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tags = dict(t.split("=", 1) for t in args.tag)

    if args.write_synthetic:
        msgs = synthetic_messages(args.n, num_envs=args.num_envs, res=tuple(args.res), depth=args.depth)
        paths = write_fixture(args.write_synthetic, msgs, synthetic=True, source="synthetic_messages()",
                              extra={"tags": tags, "complete": True, "requested": args.n})
        print(f"SYNTHETIC fixture: {paths[0]} {paths[1]}")
        return 0
    if not args.out:
        ap.error("--out is required (or --write-synthetic)")

    rec = FixtureRecorder(args.out, args.n, synthetic=args.mark_synthetic, tags=tags,
                          exit_after_save=args.exit_after_save)

    def _terminate(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _terminate)
    try:
        asyncio.run(serve(rec, args.host, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        paths = rec.flush()
        print(f"saved {len(rec.saved)}/{rec.n} observation(s): {paths}" if paths else "no observation received")
    return 0 if rec.saved else 1


if __name__ == "__main__":
    sys.exit(main())
