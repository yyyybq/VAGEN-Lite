"""Tests of package tvreval (CONTRACT_wave2 5.2, 9): vpd/tvr/{thor_worker, run_tvr, score_tvr, parity_tvr,
fake_server, render_parity}.py.

Hermetic: no ai2thor, no Unity, no GPU, no network beyond 127.0.0.1, no rendering.  TVRBench's real thor_env /
thor_view_env (read-only checkout REPO; those tests skip without it) run on the FAKE ai2thor of vpd/tests/tvr_fakes.py
(frames are a pure function of (scene, pose)); the model is vpd.tvr.fake_server.FakeServer.  Login node, 1 core:
    cd WT && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 $PY_Q3 -B -m pytest -q \
        -p no:cacheprovider vpd/tests/test_tvr_eval.py
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vpd import paths
from vpd.tests import tvr_fakes as F
from vpd.tvr import (fake_server, kin_env, lvp, parity_tvr, pilot_cfg, probe_lvp, render_parity, run_tvr, score_tvr,
                     thor_worker, tvr_text)

REPO = lvp.TVR_REPO
HAVE_REPO = (REPO / "tvrbench" / "online_rl" / "thor_view_env.py").is_file()
needs_repo = pytest.mark.skipif(not HAVE_REPO, reason="pinned TVRBench checkout not present")
HAVE_FONT = lvp.font_status()["mode"] == "pinned"
EVAL_VA_KEYS = {"task_id", "dataset", "scene", "difficulty", "result", "actions", "start", "target",
                "raw_first_response", "raw_last_response"}
RESULT_KEYS = {"success", "stopped", "num_steps", "n_invalid_actions", "final_pos_error", "final_rot_error",
               "final_hor_error"}
CALL_KEYS = {"task_id", "t", "arm", "n_images", "prompt_tokens", "completion_tokens", "finish_reason", "reply",
             "action", "regex_ok", "parse_ok", "valid", "action_in_valid", "action_success", "pose_before",
             "pose_after", "cur_sha256", "latency_vlm_s", "latency_env_s", "utc", "error"}
SERVED = "qwen3vl-8B"  # starts.json code of Q3B
ENV = {"LP_NUM_THREADS": "2", "VPD_LVP_PROFILE": "envlib", "CUDA_VISIBLE_DEVICES": "0"}

# --- a small synthetic world -------------------------------------------------------------------------------------
GRID = [round(-1.0 + 0.25 * i, 2) for i in range(9)]
SCENES = ("FloorPlan11", "FloorPlan13", "FloorPlan14", "procthor_00003", "procthor_09990")
BLOCKED = {"FloorPlan13": [(0.5, 0.5)], "procthor_00003": [(-0.5, -0.5)]}
Y = 0.9


def T(task_id, scene, dataset, difficulty, start, target):
    """A TVRBench task dict; start / target = (x, z, rotation_y, horizon)."""
    def pose(p):
        return {"position": {"x": float(p[0]), "y": Y, "z": float(p[1])}, "rotation_y": float(p[2]),
                "horizon": float(p[3])}
    return {"task_id": task_id, "dataset": dataset, "scene": scene, "difficulty": difficulty, "seg_count": 1,
            "bfs_distance": 1, "start": pose(start), "target": pose(target)}


EVAL_TASKS = [
    T("ithor_FloorPlan13_easy_t00_s00", "FloorPlan13", "ithor", "easy", (0.0, 0.0, 0, 0), (0.0, 0.0, 90, 0)),
    T("ithor_FloorPlan13_hard_t00_s00", "FloorPlan13", "ithor", "hard", (0.25, 0.0, 0, 0), (0.5, 0.25, 45, 0)),
    T("procthor_procthor_00003_easy_t00_s00", "procthor_00003", "procthor", "easy", (0.0, 0.0, 270, 0),
      (0.0, 0.0, 0, 0)),
    T("procthor_procthor_00003_hard_t00_s00", "procthor_00003", "procthor", "hard", (0.0, 0.0, 0, 30),
      (0.25, 0.25, 0, 0)),
    T("ithor_FloorPlan14_easy_t00_s00", "FloorPlan14", "ithor", "easy", (0.0, 0.0, 90, 0), (0.0, 0.0, 180, 0)),
]
OPT = {"ithor_FloorPlan13_easy_t00_s00": 2, "ithor_FloorPlan13_hard_t00_s00": 2,
       "procthor_procthor_00003_easy_t00_s00": 2, "procthor_procthor_00003_hard_t00_s00": 3,
       "ithor_FloorPlan14_easy_t00_s00": 2}
SUCCESS_STOP_AT_2 = {"ithor_FloorPlan13_easy_t00_s00", "procthor_procthor_00003_easy_t00_s00",
                     "ithor_FloorPlan14_easy_t00_s00"}
SFT_TASKS = [
    T(run_tvr.SFT2[0], "FloorPlan11", "ithor", "easy", (0.0, 0.0, 0, 0), (0.25, 0.25, 45, 30)),
    T(run_tvr.SFT2[1], "procthor_09990", "procthor", "hard", (0.5, 0.5, 90, 0), (0.75, 0.75, 45, 0)),
]
SFT_ACTS = {run_tvr.SFT2[0]: ["RotateRight", "MoveAhead", "LookDown", "Stop"],
            run_tvr.SFT2[1]: ["MoveAhead", "MoveLeft", "RotateLeft", "Stop"]}


def positions_of(scene):
    return F.grid_positions(GRID, GRID, blocked=BLOCKED.get(scene, ()))


def write_reach(root: Path, tasks, override=None):
    """reach/v1 in the 6.3 format (frozen = the given positions)."""
    override = override or {}
    scenes, index = {}, {"schema": "vpd.tvr.reach/1", "created_utc": "test", "tasks": {}, "scene_files": {},
                         "unstable_tasks": [], "task_sets": {}, "code_sha256": None}
    for t in tasks:
        pos = override.get(t["task_id"]) or positions_of(t["scene"])
        d = kin_env.ReachSet.from_positions(pos).digest
        sc = scenes.setdefault(t["scene"], {"schema": "vpd.tvr.reach/1", "scene": t["scene"], "kind":
                                            lvp.scene_kind(t["scene"]), "passes": [], "tasks": {}, "sets": {},
                                            "stable": True})
        sc["sets"][d] = pos
        sc["tasks"][t["task_id"]] = {"A": {"digest": d, "n": len(pos)}, "B": {"digest": d, "n": len(pos)},
                                     "equal_AB": True}
        index["tasks"][t["task_id"]] = {"scene": t["scene"], "digest": d, "n": len(pos), "stable": True}
    (root / "scenes").mkdir(parents=True, exist_ok=True)
    for s, doc in scenes.items():
        p = root / "scenes" / f"{s}.json"
        p.write_text(json.dumps(doc))
        index["scene_files"][s] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / "index.json").write_text(json.dumps(index))
    (root / "DONE").write_text("ok\n")
    return root


def _purge():
    for k in list(sys.modules):
        if k in ("tvrbench", "ai2thor") or k.startswith(("tvrbench.", "ai2thor.")):
            del sys.modules[k]


def make_world(tmp: Path, *, by_start=None, reach_override=None) -> types.SimpleNamespace:
    w = types.SimpleNamespace()
    scenes = {}
    for s in SCENES:
        scenes[s] = {"positions": positions_of(s),
                     "spawn": {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation_y": 0.0, "horizon": 0.0}}
    for s, bs in (by_start or {}).items():
        scenes[s]["by_start"] = bs
    w.fake = F.write_fake_ai2thor(tmp / "fake", scenes)
    w.tvr_out = tmp / "tvr"
    w.tasks_dir = tmp / "tasks"
    w.tasks_dir.mkdir()
    (w.tasks_dir / "eval.json").write_text(json.dumps(EVAL_TASKS))
    (w.tasks_dir / "sft.json").write_text(json.dumps(SFT_TASKS))
    (w.tasks_dir / "rl.json").write_text(json.dumps([]))
    w.houses = w.tvr_out / "houses"
    w.houses.mkdir(parents=True)
    for s in SCENES:
        if lvp.is_procthor(s):
            (w.houses / f"{s}.json").write_text(json.dumps({"id": s, "rooms": []}))
    w.pt = tmp / "pt"
    w.pt.mkdir()
    w.reach = write_reach(w.tvr_out / "reach" / "v1", EVAL_TASKS + SFT_TASKS, reach_override)
    w.bfs = w.tvr_out / "bfs" / "eval500_opt.json"
    w.bfs.parent.mkdir(parents=True)
    w.bfs.write_text(json.dumps({"schema": "vpd.tvr.bfs/1", "reach_index_sha256": None, "n_infeasible": 0,
                                 "tasks": {k: {"feasible": True, "opt": v, "start_in_reach": True,
                                               "target_in_reach": True} for k, v in OPT.items()}}))
    return w


@pytest.fixture
def world(tmp_path):
    if not HAVE_REPO:
        pytest.skip("pinned TVRBench checkout not present")
    w = make_world(tmp_path)
    saved = list(sys.path)
    _purge()
    _te, w.tv = F.import_real_tvrbench(w.fake)
    try:
        yield w
    finally:
        _purge()
        sys.path[:] = saved


def font_of(tv):
    return lvp.install_font_rule(tv, allow_missing=True)


def fake_frame(scene, p):
    """The fake ai2thor's frame at pose p = (x, z, rot, hor), through its own Controller (contract 5.1.8 API)."""
    c = sys.modules["ai2thor.controller"].Controller(width=640, height=360, scene=scene, rotateStepDegrees=45)
    c.step(action="Teleport", position={"x": p[0], "y": Y, "z": p[1]}, rotation={"x": 0, "y": p[2], "z": 0},
           horizon=p[3], standing=True, forceAction=True)
    return c.last_event.frame.copy()


def local_pool(w, worker_cls=None, **wkw):
    font = font_of(w.tv)
    cls = worker_cls or thor_worker.Worker

    def factory(n, L, prof, out_dir):
        def mk(wid):
            return cls(w.tv, wid=wid, houses_dir=w.houses, procthor_dir=w.pt, font=font,
                       environ=dict(ENV, LP_NUM_THREADS=str(L), VPD_LVP_PROFILE=prof), **wkw)
        return thor_worker.ThorPool(n, L, prof, out_dir=out_dir, houses_dir=w.houses, local_factory=mk)
    return factory


def eval_args(w, out, *extra, arms="c", tier="raw", task_set="eval500"):
    return run_tvr.build_parser().parse_args(
        ["eval", "--start", "Q3B", "--tier", tier, "--arms", arms, "--task-set", task_set, "--out-root", str(out),
         "--envs", "2", "--lp", "2", "--profile", "envlib", "--tvr-out", str(w.tvr_out), "--tasks-dir",
         str(w.tasks_dir), "--reach", str(w.reach), "--bfs", str(w.bfs), "--houses", str(w.houses), *extra])


def run_eval(w, out, *extra, policy="stop_at", stop_after=2, worker_cls=None, deadline=None, srv=None, arms="c",
             **kw):
    srv = srv or fake_server.FakeServer(policy=policy, stop_after=stop_after, model=SERVED)
    a = eval_args(w, out, *extra, arms=arms, **kw)
    rc = run_tvr.cmd_eval(a, ctl=run_tvr.Ctl(deadline), pool_factory=local_pool(w, worker_cls),
                          server_factory=lambda: srv)
    return rc, srv


def rd(out, arm, tier="raw", prefix="pilot"):
    return out / f"{prefix}__Q3B__{tier}__{arm}"


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def traj(run, tid):
    return json.loads((run / "tasks" / tid / "trajectory.json").read_text())


def png_bytes(pil):
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ================================================================================================================
# worker-side modules: py3.10 syntax
# ================================================================================================================
def test_worker_side_modules_parse_as_py310():
    for rel in ("vpd/tvr/thor_worker.py", "vpd/tvr/render_parity.py", "vpd/tvr/lvp.py", "vpd/tvr/probe_lvp.py",
                "vpd/paths.py"):
        ast.parse((paths.WORKTREE / rel).read_text(), filename=rel, feature_version=(3, 10))


@pytest.mark.skipif(not Path(lvp.TVR_PY).exists(), reason="TVR env python not present")
def test_worker_side_modules_import_under_tvr_py():
    """The THOR-worker side really imports on py3.10 (stdlib only at import time; no ai2thor, no Unity)."""
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), VPD_WORKTREE=str(paths.WORKTREE), OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([str(lvp.TVR_PY), "-B", "-s", "-c",
                        "import sys, vpd.tvr.thor_worker, vpd.tvr.render_parity; "
                        "assert 'ai2thor' not in sys.modules and 'numpy' not in sys.modules; "
                        "print(sys.version_info[:2])"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "(3, 10)" in r.stdout


def test_cli_help():
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), VPD_WORKTREE=str(paths.WORKTREE), OMP_NUM_THREADS="1")
    for mod in ("run_tvr", "score_tvr", "parity_tvr", "render_parity", "thor_worker", "fake_server"):
        r = subprocess.run([sys.executable, "-B", "-m", f"vpd.tvr.{mod}", "--help"], env=env, capture_output=True,
                           text=True, timeout=120)
        assert r.returncode == 0 and "usage" in r.stdout, (mod, r.stderr[-1000:])


def test_code_files_and_fingerprint():
    c = run_tvr.code_sha256()
    assert set(c["files"]) == set(run_tvr.CODE_FILES) and len(run_tvr.CODE_FILES) == 12
    lines = "\n".join(f"{k} {v}" for k, v in sorted(c["files"].items()))
    assert c["code_sha256"] == hashlib.sha256(lines.encode()).hexdigest()
    assert thor_worker.CODE_FILES_TVRBENCH == ("tvrbench/env/thor_env.py", "tvrbench/online_rl/thor_view_env.py",
                                              "tvrbench/evaluation/metrics.py", "scripts/eval_va.py")


# ================================================================================================================
# fake server, request bodies
# ================================================================================================================
def _post(url, body):
    import urllib.request
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_fake_server_policies_and_usage():
    msgs = [{"role": "system", "content": "abc"},
            {"role": "user", "content": [{"type": "text", "text": "xy"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
    assert fake_server.prompt_tokens(msgs) == 3 + 2 + 222
    with fake_server.FakeServer(policy="cycle", model="m") as s:
        r = _post(s.base_url, {"model": "m", "messages": msgs, "structured_outputs": {"regex": pilot_cfg.REGEX}})
        assert r["choices"][0]["message"]["content"] == "Action: MoveAhead"
        assert r["usage"]["prompt_tokens"] == 227
        m2 = msgs + [{"role": "assistant", "content": "x"}] * 8
        assert _post(s.base_url, {"messages": m2})["choices"][0]["message"]["content"] == "Action: Stop"
    with fake_server.FakeServer(policy=lambda body, n: "nonsense") as s:
        out = _post(s.base_url, {"messages": msgs, "structured_outputs": {"regex": pilot_cfg.REGEX}})
        assert out["choices"][0]["message"]["content"] == "Action: Stop"  # regex respected
        assert _post(s.base_url, {"messages": msgs})["choices"][0]["message"]["content"] == "nonsense"
    with fake_server.FakeServer(policy="junk") as s:
        txt = _post(s.base_url, {"messages": msgs, "structured_outputs": {"regex": pilot_cfg.REGEX}})
        assert tvr_text.regex_action(txt["choices"][0]["message"]["content"]) is None


def test_chat_client_retries(monkeypatch):
    monkeypatch.setattr(run_tvr, "HTTP_BACKOFF_S", (0.01, 0.01, 0.01))
    ctl = run_tvr.Ctl(None)
    with fake_server.FakeServer(policy="stop_at", fail_first=2) as s:
        c = run_tvr.ChatClient(s.base_url, ctl)
        resp, lat, retries, errs = c.chat(json.dumps({"messages": []}).encode())
        assert retries == 2 and len(errs) == 2 and "HTTP 500" in errs[0]
        assert resp["choices"][0]["message"]["content"] == "Action: RotateRight"
    with fake_server.FakeServer(policy="stop_at", fail_first=99) as s:
        c = run_tvr.ChatClient(s.base_url, ctl)
        with pytest.raises(run_tvr.TaskError) as ei:
            c.chat(b"{}")
        assert ei.value.kind == "api" and s.n_chat == 4  # 1 + 3 retries


def test_convo_body_fields():
    for arm, mt, rx in (("c", 16, pilot_cfg.REGEX), ("u", 64, None), ("blind", 16, pilot_cfg.REGEX)):
        cv = run_tvr.Convo(SERVED, mt, rx)
        cv.add_user_images(tvr_text.build_concat_user_text(["RotateRight", "Stop"]),
                           (run_tvr.image_part("QUJD"), {}), (run_tvr.image_part("REVG"), {}))
        cv.add_assistant("Action: RotateRight")
        body = json.loads(cv.body())
        keys = {"model", "messages", "temperature", "top_p", "max_tokens", "top_k"} | ({"structured_outputs"} if rx
                                                                                       else set())
        assert set(body) == keys, arm
        assert body["max_tokens"] == mt and body["temperature"] == 0.0 and body["top_k"] == -1
        assert body["messages"][0] == {"role": "system", "content": tvr_text.SYSTEM_PROMPT}
        parts = body["messages"][1]["content"]
        assert [p["type"] for p in parts] == ["text", "image_url", "text", "image_url", "text"]
        assert parts[1]["image_url"]["url"] == "data:image/png;base64,QUJD"
        assert parts[0]["text"] == "Your CURRENT observation:\n"


# ================================================================================================================
# the worker (LocalHandle) on the real TVRBench + fake ai2thor
# ================================================================================================================
@needs_repo
def test_worker_ops_local(world, tmp_path):
    w = world
    pool = local_pool(w)(1, 2, "envlib", tmp_path / "pool")
    ready = pool.start()
    assert ready[0]["event"] == "ready" and ready[0]["lp"] == "2" and ready[0]["profile"] == "envlib"
    assert ready[0]["system_prompt_sha256"] == hashlib.sha256(tvr_text.SYSTEM_PROMPT.encode()).hexdigest()
    assert set(ready[0]["tvrbench_files_sha256"]) == set(thor_worker.CODE_FILES_TVRBENCH)
    task = EVAL_TASKS[0]
    r = pool.call(0, {"op": "load", "scene": "FloorPlan13", "force": False}, 30)
    assert r["ok"] and r["loaded"] and r["scene"] == "FloorPlan13"
    r = pool.call(0, {"op": "load", "scene": "FloorPlan13", "force": False}, 30)
    assert r["ok"] and not r["loaded"]
    r = pool.call(0, {"op": "reach_task", "task": task}, 30)
    assert r["ok"] and r["n"] == len(positions_of("FloorPlan13")) - 0 and kin_env.pose_of(r["call_pose"]) == \
        kin_env.pose_of(task["start"])
    assert r["digest"] == kin_env.ReachSet.from_positions(positions_of("FloorPlan13")).digest
    cur_p, tgt_p = tmp_path / "img" / "cur.png", tmp_path / "img" / "tgt.png"
    pos = positions_of("FloorPlan13")
    r = pool.call(0, {"op": "reset", "task": task, "reach": pos,
                      "images": {"inline": True, "cur": str(cur_p), "tgt": str(tgt_p)}}, 30)
    assert r["ok"], r
    assert r["reach_used"]["source"] == "frozen" and r["reach_equal"] is True
    assert r["reach_used"]["digest"] == r["reach_live"]["digest"] == kin_env.ReachSet.from_positions(pos).digest
    want_cur = png_bytes(w.tv.burn_label(fake_frame("FloorPlan13", (0.0, 0.0, 0, 0)), "CURRENT"))
    want_tgt = png_bytes(w.tv.burn_label(fake_frame("FloorPlan13", (0.0, 0.0, 90, 0)), "TARGET"))
    assert base64.b64decode(r["cur"]["png_b64"]) == cur_p.read_bytes() == want_cur
    assert base64.b64decode(r["tgt"]["png_b64"]) == tgt_p.read_bytes() == want_tgt
    assert r["cur"]["sha256"] == hashlib.sha256(want_cur).hexdigest() and r["cur"]["bytes"] == len(want_cur)
    assert r["obs_str"] == tvr_text.build_concat_user_text(r["valid"]) and r["valid"][-1] == "Stop"
    r = pool.call(0, {"op": "step", "action": "RotateRight", "images": {"inline": False, "cur": None}}, 30)
    assert r["ok"] and r["info"]["action_success"] is True and r["state"]["rotation_y"] == 45.0
    assert "png_b64" not in r["cur"] and not r["done"]
    r = pool.call(0, {"op": "step", "action": "InvalidAction", "images": {"inline": True}}, 30)
    assert r["ok"] and r["info"]["action_executed"] is False and r["info"]["action_success"] is None
    assert r["state"]["rotation_y"] == 45.0
    r = pool.call(0, {"op": "step", "action": "Stop", "images": {"inline": True}}, 30)
    assert r["ok"] and r["cur"] is None and r["done"] and r["info"]["stopped"] and r["info"]["success"] is False
    r = pool.call(0, {"op": "nope"}, 30)
    assert not r["ok"] and "unknown op" in r["error"]
    r = pool.call(0, {"op": "reset", "task": task, "reach": None, "images": {"inline": False}}, 30)
    assert r["ok"] and r["reach_used"]["source"] == "live"
    pool.close()
    assert pool.call(0, {"op": "load", "scene": "x"}, 1)["died"]


@needs_repo
def test_worker_frozen_set_differs_from_live(tmp_path):
    """by_start fake (F2): the live set at the start lacks (0.25, 0); the frozen set has it -> the valid list
    follows the frozen set, reach_used comes from thor._get_reachable_set(), reach_equal false."""
    small = [p for p in positions_of("FloorPlan13") if (p["x"], p["z"]) != (0.25, 0.0)]
    bs = {k: small for k in ("0.00,0.00", "0.0,0.0", "0,0")}
    w = make_world(tmp_path, by_start={"FloorPlan13": bs})
    saved = list(sys.path)
    _purge()
    try:
        _te, w.tv = F.import_real_tvrbench(w.fake)
        pool = local_pool(w)(1, 2, "envlib", tmp_path / "pool")
        pool.start()
        task = EVAL_TASKS[0]
        r = pool.call(0, {"op": "reach_task", "task": task}, 30)
        assert r["digest"] == kin_env.ReachSet.from_positions(small).digest  # R(task): the start-pose call
        full = positions_of("FloorPlan13")
        r = pool.call(0, {"op": "reset", "task": task, "reach": full, "images": {"inline": False}}, 30)
        assert r["ok"] and r["reach_equal"] is False
        assert r["reach_live"]["digest"] == kin_env.ReachSet.from_positions(small).digest
        assert r["reach_used"]["digest"] == kin_env.ReachSet.from_positions(full).digest
        assert "MoveRight" in r["valid"]
        assert r["valid"] == kin_env.valid_actions(kin_env.pose_of(task["start"]),
                                                   kin_env.ReachSet.from_positions(full))
        r2 = pool.call(0, {"op": "reset", "task": task, "reach": None, "images": {"inline": False}}, 30)
        assert "MoveRight" not in r2["valid"]  # the live set would have hidden it
        # an injection that does not take -> injection_failed
        thor = pool.handles[0].worker.env._env
        thor._get_reachable_set = lambda: {(9.0, 9.0)}
        r3 = pool.call(0, {"op": "reset", "task": task, "reach": full, "images": {"inline": False}}, 30)
        assert not r3["ok"] and "injection_failed" in r3["error"]
        pool.close()
    finally:
        _purge()
        sys.path[:] = saved


@needs_repo
@pytest.mark.skipif(not HAVE_FONT, reason="the pinned badge font of the TVR env is not readable")
def test_worker_subprocess_protocol(world, tmp_path):
    w = world
    fake_env = tmp_path / "fake_tvr_env"
    (fake_env / "lib").mkdir(parents=True)
    rel = tmp_path / "releases"
    rel.mkdir()
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONDONTWRITEBYTECODE": "1",
            "VPD_WORKTREE": str(paths.WORKTREE), "LP_NUM_THREADS": "99"}
    pool = thor_worker.ThorPool(1, 2, "envlib", out_dir=tmp_path / "out", houses_dir=w.houses, repo=REPO,
                                py=sys.executable, code_root=paths.WORKTREE, base_env=base, startup_timeout=90,
                                extra_pythonpath=(str(w.fake),), tvr_env=fake_env, releases=rel, procthor_dir=w.pt)
    try:
        ready = pool.start()[0]
        assert ready["ok"] and ready["lp"] == "2" and ready["env"]["LP_NUM_THREADS"] == "2"
        assert ready["env"]["LD_LIBRARY_PATH"] == str(fake_env / "lib") and ready["profile"] == "envlib"
        assert ready["font"]["mode"] == "pinned" and ready["font"]["sha256"] == lvp.BADGE_FONT_SHA256
        assert ready["settings"]["fieldOfView"] == 60
        task = EVAL_TASKS[2]  # ProcTHOR through the houses dir
        r = pool.call(0, {"op": "reset", "task": task, "reach": positions_of(task["scene"]),
                          "images": {"inline": True, "cur": str(tmp_path / "c.png")}}, 60)
        assert r["ok"], r
        assert base64.b64decode(r["cur"]["png_b64"]) == (tmp_path / "c.png").read_bytes()
        assert r["scene"] == "procthor_00003" and r["reach_used"]["source"] == "frozen"
        r = pool.call(0, {"op": "step", "action": "RotateRight", "images": {"inline": True}}, 60)
        assert r["ok"] and r["info"]["action_success"] is True
        pid = pool.handles[0].proc.pid
        os.killpg(pid, signal.SIGKILL)
        r = pool.call(0, {"op": "step", "action": "RotateRight", "images": {"inline": True}}, 20)
        assert not r["ok"] and (r.get("died") or "send failed" in str(r.get("error")))
        rr = pool.restart(0)
        assert rr["ok"] and pool.handles[0].proc.pid != pid
        assert pool.call(0, {"op": "load", "scene": "FloorPlan11"}, 60)["ok"]
        t0 = time.monotonic()
        pool.close(fast=True)
        assert time.monotonic() - t0 < 2.0
        assert pool.handles[0].proc.poll() is not None
    finally:
        pool.close(fast=True)
    assert (tmp_path / "out" / "logs" / "worker_w00_L2.log").is_file()


def test_pool_start_retry_and_failure(tmp_path):
    calls = []

    class W:
        def __init__(self, wid):
            self.wid = wid

        def start(self):
            calls.append(self.wid)
            if len(calls) == 1 or self.wid == 2:
                raise RuntimeError("boom")
            return {"event": "ready", "wid": self.wid}

        def handle(self, cmd):
            return {"echo": cmd.get("op")}

        def close(self):
            pass

    pool = thor_worker.ThorPool(2, 1, "envlib", out_dir=tmp_path, houses_dir=None, local_factory=W)
    assert len(pool.start()) == 2 and calls == [0, 0, 1]  # the first start-up failed once and was retried
    assert pool.call(1, {"op": "x"}, 1)["echo"] == "x"
    pool3 = thor_worker.ThorPool(3, 1, "envlib", out_dir=tmp_path, houses_dir=None, local_factory=W)
    with pytest.raises(thor_worker.PoolError):
        pool3.start()
    with pytest.raises(ValueError):
        thor_worker.ThorPool(1, 1, "nope", out_dir=tmp_path, houses_dir=None)


def test_reap_unity(tmp_path):
    pat = f"/.ai2thor/releases/FAKEPAT{os.getpid()}/FAKEPAT{os.getpid()}"
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", pat])
    try:
        time.sleep(0.3)
        lines = []
        got = thor_worker.reap_unity(log=lines.append, pattern=pat, grace_s=3)
        assert got == [p.pid] and lines
        p.wait(timeout=5)
        assert thor_worker.find_unity(pat) == []
    finally:
        if p.poll() is None:
            p.kill()


# ================================================================================================================
# run_tvr eval with FakeServer + fake THOR
# ================================================================================================================
def mixed_policy(body, n):
    """c / blind (regex): RotateRight twice then Stop; u: a parseable variant, junk, then Stop."""
    if (body.get("structured_outputs") or {}).get("regex"):
        return "Action: Stop" if n >= 2 else "Action: RotateRight"
    return ["I think Action: rotateright", "hmm, no idea", "Action: Stop"][min(n, 2)]


@needs_repo
def test_eval_three_arms_one_session(world, tmp_path):
    w = world
    out = tmp_path / "evals"
    srv = fake_server.FakeServer(policy=mixed_policy, model=SERVED)
    rc, srv = run_eval(w, out, srv=srv, arms="c,u,blind")
    assert rc == 0
    for arm in ("c", "u", "blind"):
        res = json.loads((rd(out, arm) / "results.json").read_text())
        assert res["valid"] and res["n_tasks"] == 5 and res["tasks_missing"] == [] and res["arm"] == arm
        assert res["server"]["max_model_len"] == 32768 and set(pilot_cfg.FROZEN["protocol"]) <= set(res["settings"])
        assert set(res["perf"]) >= {"steps", "wall_s", "steps_per_s", "cgroup_cores_mean", "cgroup_cores_max",
                                    "vllm_ready_s", "max_images_per_request", "max_prompt_tokens",
                                    "mean_steps_per_task"}
        assert set(res["server"]["versions"]) >= {"vllm", "xgrammar", "transformers", "torch", "ai2thor"}
        # the server manifest flat (WB run_vpd convention: server.cfg / cmd / ...) and also under server.manifest
        assert res["server"]["source"] == res["server"]["manifest"]["source"] == "vpd.tvr.fake_server.FakeServer"
        assert res["server"]["cfg"] == res["server"]["manifest"]["cfg"]
        assert res["code_sha256"] == run_tvr.code_sha256()["code_sha256"] and res["code"] == SERVED
        assert res["perf"]["segments"] == 1 and res["perf"]["resumed"] is False
        for t in EVAL_TASKS:
            tr = traj(rd(out, arm), t["task_id"])
            assert set(tr) == EVAL_VA_KEYS | {"vpd"} and set(tr["result"]) == RESULT_KEYS
            v = tr["vpd"]
            assert len(v["states"]) == tr["result"]["num_steps"] + 1 == len(v["replies"]) + 1
            assert len(v["user_texts"]) == len(v["valid"]) == tr["result"]["num_steps"]
            want = kin_env.ReachSet.from_positions(positions_of(t["scene"]))
            assert v["reach_digest"] == want.digest
            if arm == "blind":
                assert v["reach_used"] is None
            else:  # the worker's own report of the set in force
                assert v["reach_used"] == {"source": "frozen", "n": want.n, "digest": want.digest}
        for c in jl(rd(out, arm) / "calls.jsonl"):
            assert CALL_KEYS <= set(c)
    c_res = json.loads((rd(out, "c") / "results.json").read_text())
    assert c_res["sr"]["k"] == 3 and c_res["sr"]["n"] == 5 and c_res["spl"] == 0.6
    assert c_res["invalid"]["regex_mismatch"] == 0 and c_res["kin"]["tasks_equal"] == 5
    assert c_res["kin"]["valid_list_mismatches"] == 0 and c_res["reach"]["live_equal"] == 5
    assert c_res["audit"]["image_tokens"]["tokens_per_image"] == [222]
    assert c_res["audit"]["image_tokens"]["grid_set"] == [[1, 22, 40]]
    assert c_res["audit"]["prompt_parity"]["equal"] is True and c_res["audit"]["prompt_parity"]["n"] >= 5
    assert c_res["renderer"]["profile"] == "envlib" and c_res["renderer"]["lp"] == 2
    assert c_res["renderer"]["workers_ok"] is HAVE_FONT and c_res["renderer"]["n_workers_live"] == 2
    assert all(wk["system_prompt_ok"] and wk["tvrbench_files_ok"] for wk in c_res["renderer"]["workers"])
    assert {r["task_id"] for r in jl(rd(out, "c") / "per_task.jsonl")} == {t["task_id"] for t in EVAL_TASKS}
    assert {t["task_id"] for t in EVAL_TASKS if traj(rd(out, "c"), t["task_id"])["result"]["success"]} == \
        SUCCESS_STOP_AT_2
    # u arm: the junk reply is InvalidAction (a step spent, no env step), the raw reply stays in the history
    u = traj(rd(out, "u"), EVAL_TASKS[0]["task_id"])
    assert u["actions"] == ["RotateRight", "InvalidAction", "Stop"] and u["result"]["n_invalid_actions"] == 1
    assert u["vpd"]["action_success"] == [True, None, None] and u["vpd"]["n_unparseable"] == 1
    assert u["vpd"]["states"][2]["rotation_y"] == 45.0 and u["result"]["success"] is False
    ures = json.loads((rd(out, "u") / "results.json").read_text())
    assert ures["invalid"]["unparseable_steps"] == 5 and ures["invalid"]["regex_mismatch"] == 0
    # blind: kin_env, no images, the same text without placeholders
    b = traj(rd(out, "blind"), EVAL_TASKS[0]["task_id"])
    assert b["vpd"]["images"] is None and b["result"]["success"] is True and b["vpd"]["kin_equal"] is None
    # request bodies
    reqs = srv.requests
    assert all("chat_template_kwargs" not in q and "seed" not in q and "extra_body" not in q for q in reqs)
    c_reqs = [q for q in reqs if q.get("structured_outputs") and isinstance(q["messages"][1]["content"], list)]
    u_reqs = [q for q in reqs if not q.get("structured_outputs")]
    b_reqs = [q for q in reqs if q.get("structured_outputs") and isinstance(q["messages"][1]["content"], str)]
    assert len(c_reqs) == 15 and len(u_reqs) == 15 and len(b_reqs) == 15
    assert {q["max_tokens"] for q in c_reqs} == {16} and {q["max_tokens"] for q in u_reqs} == {64}
    assert {q["max_tokens"] for q in b_reqs} == {16}
    assert all(q["structured_outputs"] == {"regex": pilot_cfg.REGEX} for q in c_reqs + b_reqs)
    assert all(q["model"] == SERVED and q["temperature"] == 0.0 and q["top_p"] == 1.0 and q["top_k"] == -1
               for q in reqs)
    for q in c_reqs + u_reqs:
        ms = q["messages"]
        assert ms[0] == {"role": "system", "content": tvr_text.SYSTEM_PROMPT}
        users = [m for m in ms if m["role"] == "user"]
        assert [m["role"] for m in ms[1:]] == ["user", "assistant"] * (len(users) - 1) + ["user"]  # full history
        tgt0 = [p for p in users[0]["content"] if p["type"] == "image_url"][1]["image_url"]["url"]
        for m in users:
            imgs = [p for p in m["content"] if p["type"] == "image_url"]
            assert len(imgs) == 2 and imgs[1]["image_url"]["url"] == tgt0  # TARGET re-sent every turn
            assert imgs[0]["image_url"]["url"].startswith("data:image/png;base64,")
            assert base64.b64decode(imgs[0]["image_url"]["url"].split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
    for q in b_reqs:
        users = [m for m in q["messages"] if m["role"] == "user"]
        assert all("<image>" not in m["content"] and m["content"].startswith("Your CURRENT observation:\n")
                   for m in users)
    # u: the junk reply is the assistant turn of the next request
    tid0_u = [q for q in u_reqs if len(q["messages"]) == 6]
    assert any(q["messages"][4] == {"role": "assistant", "content": "hmm, no idea"} for q in tid0_u)
    # prompt dump: kept tasks (2 per category -> all 5 here), calls 0, 1, 2 (2 is the last), images redacted
    dump = jl(rd(out, "c") / "prompt_dump.jsonl")
    assert sorted({d["t"] for d in dump}) == [0, 1, 2] and len(dump) == 15
    for d in dump:
        for m in d["messages"]:
            if isinstance(m["content"], list):
                for p in m["content"]:
                    if p["type"] == "image_url":
                        assert set(p) == {"type", "sha256", "path"} and p["path"].startswith("images/")
                        f = rd(out, "c") / p["path"]
                        assert hashlib.sha256(f.read_bytes()).hexdigest() == p["sha256"]
    # cur sha of the trajectory == the image sent in that turn
    tr = traj(rd(out, "c"), EVAL_TASKS[1]["task_id"])
    req = [q for q in c_reqs if len(q["messages"]) == 2 and
           hashlib.sha256(base64.b64decode(q["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[1])
                          ).hexdigest() == tr["vpd"]["images"][0]["cur_sha256"]]
    assert len(req) == 1


@needs_repo
def test_eval_caps_30_40(world, tmp_path):
    out = tmp_path / "evals"
    rc, srv = run_eval(world, out, policy="stop_at", stop_after=10 ** 6)
    assert rc == 0
    for t in EVAL_TASKS:
        tr = traj(rd(out, "c"), t["task_id"])
        cap = 30 if t["dataset"] == "ithor" else 40
        assert tr["result"]["num_steps"] == cap and not tr["result"]["stopped"] and not tr["result"]["success"]
        calls = jl(rd(out, "c") / "tasks" / t["task_id"] / "calls.jsonl")
        assert len(calls) == cap and calls[-1]["n_images"] == 2 * cap
    res = json.loads((rd(out, "c") / "results.json").read_text())
    assert res["steps"]["capped_tasks"] == 5 and res["perf"]["max_images_per_request"] == 80
    assert res["stop"]["n_stopped"] == 0 and res["stop"]["fstop_given_stop"] is None


@needs_repo
def test_eval_mechanism_only_withholds_success(world, tmp_path):
    out = tmp_path / "smoke"
    rc, _ = run_eval(world, out, "--mechanism-only", "--name-prefix", "smoke")
    assert rc == 0
    run = rd(out, "c", prefix="smoke")
    for t in EVAL_TASKS:
        tr = traj(run, t["task_id"])
        assert tr["result"]["success"] is None and tr["vpd"]["withheld"] == "mechanism smoke"
    res = json.loads((run / "results.json").read_text())
    for k in ("sr", "spl", "within1", "stop", "reached", "bootstrap_scene", "errors_final"):
        assert k not in res
    assert res["withheld"] == "mechanism smoke" and res["valid"] and res["bfs_path"] is None
    assert all(set(r) == set(score_tvr.MECH_COLUMNS) for r in jl(run / "per_task.jsonl"))
    assert "sr" not in res["by_category"]["SR-easy"]


@needs_repo
def test_eval_resume_rules(world, tmp_path):
    w = world
    out = tmp_path / "evals"
    rc, s1 = run_eval(w, out)
    assert rc == 0
    res_p = rd(out, "c") / "results.json"
    before = res_p.read_bytes()
    with pytest.raises(run_tvr.Refused):  # an existing run without --resume
        run_eval(w, out)
    rc, s2 = run_eval(w, out, "--resume")  # complete arm: reported, skipped, nothing rewritten
    assert rc == 0 and s2.n_chat == 0 and res_p.read_bytes() == before
    with pytest.raises(run_tvr.Refused):  # protocol mismatch
        run_eval(w, out, "--resume", "--mechanism-only")
    # an arm that holds results_invalid.json is resumed: only its missing task runs, the file is replaced
    out2 = tmp_path / "evals2"
    bad = EVAL_TASKS[1]["task_id"]

    class Crash(thor_worker.Worker):
        def op_reset(self, cmd):
            if cmd["task"]["task_id"] == bad:
                raise RuntimeError("fake worker crash")
            return super().op_reset(cmd)

    rc, s3 = run_eval(w, out2, worker_cls=Crash)
    assert rc == 1
    inv = json.loads((rd(out2, "c") / "results_invalid.json").read_text())
    assert inv["valid"] is False and inv["tasks_missing"] == [bad] and inv["task_failures"] == 1
    assert inv["failed_tasks"][0]["task_id"] == bad and not (rd(out2, "c") / "results.json").exists()
    mt = {t["task_id"]: (rd(out2, "c") / "tasks" / t["task_id"] / "trajectory.json").stat().st_mtime_ns
          for t in EVAL_TASKS if t["task_id"] != bad}
    rc, s4 = run_eval(w, out2, "--resume")
    assert rc == 0 and s4.n_chat == 3  # only the missing task (RotateRight, RotateRight, Stop)
    assert (rd(out2, "c") / "results.json").exists() and not (rd(out2, "c") / "results_invalid.json").exists()
    assert all((rd(out2, "c") / "tasks" / k / "trajectory.json").stat().st_mtime_ns == v for k, v in mt.items())


@needs_repo
def test_eval_worker_crash_once_is_retried(world, tmp_path):
    seen = {"n": 0}
    tid = EVAL_TASKS[3]["task_id"]

    class CrashOnce(thor_worker.Worker):
        def op_step(self, cmd):
            if self.env._task["task_id"] == tid and seen["n"] == 0:
                seen["n"] += 1
                raise RuntimeError("one-off crash")
            return super().op_step(cmd)

    out = tmp_path / "evals"
    rc, _ = run_eval(world, out, worker_cls=CrashOnce)
    assert rc == 0 and seen["n"] == 1
    tr = traj(rd(out, "c"), tid)
    assert tr["vpd"]["attempt"] == 2 and tr["result"]["num_steps"] == 3


@needs_repo
def test_eval_kin_shadow_typing(world, tmp_path):
    class Lie(thor_worker.Worker):
        def op_reset(self, cmd):
            out = super().op_reset(cmd)
            if cmd["task"]["task_id"] == EVAL_TASKS[0]["task_id"]:
                out["valid"] = [a for a in out["valid"] if a != "RotateLeft"]  # broken injection stand-in
            return out

        def op_step(self, cmd):
            out = super().op_step(cmd)
            if self.env._task["task_id"] == EVAL_TASKS[2]["task_id"] and cmd["action"] == "RotateRight":
                out["info"]["action_success"] = False  # a THOR action that reported failure
            return out

    out = tmp_path / "evals"
    rc, _ = run_eval(world, out, worker_cls=Lie)
    assert rc == 0
    res = json.loads((rd(out, "c") / "results.json").read_text())
    k = res["kin"]
    assert k["tasks_unequal"] == 2 and k["tasks_equal"] == 3
    assert k["valid_list_mismatches"] == 1 and k["pose_or_success_mismatches"] == 2
    t0 = traj(rd(out, "c"), EVAL_TASKS[0]["task_id"])["vpd"]
    t2 = traj(rd(out, "c"), EVAL_TASKS[2]["task_id"])["vpd"]
    assert t0["kin_first_mismatch"]["type"] == "valid_list" and t0["kin_equal"] is False
    assert t2["kin_first_mismatch"]["type"] == "success" and t2["kin_mismatch_counts"]["success"] == 2
    assert {f["task_id"] for f in k["first"]} == {EVAL_TASKS[0]["task_id"], EVAL_TASKS[2]["task_id"]}


@needs_repo
def test_eval_deadline_partial(world, tmp_path, monkeypatch):
    monkeypatch.setattr(run_tvr, "NEW_TASK_MIN_S", 0.0)
    out = tmp_path / "evals"
    srv = fake_server.FakeServer(policy="stop_at", stop_after=10 ** 6, model=SERVED, delay_s=0.05)
    t0 = time.monotonic()
    rc, _ = run_eval(world, out, srv=srv, deadline=2.0)
    assert rc == 4 and time.monotonic() - t0 < 30
    run = rd(out, "c")
    part = json.loads((run / "results_partial.json").read_text())
    assert not (run / "results.json").exists() and part["valid"] is False and part["partial"] is True
    assert part["audit"] == {"image_tokens": "skipped_deadline", "prompt_parity": "skipped_deadline"}
    assert part["deadline"]["hit"] is True and len(part["tasks_missing"]) >= 1
    # resume (no deadline) runs the missing tasks and replaces results_partial.json
    monkeypatch.setattr(run_tvr, "NEW_TASK_MIN_S", 150.0)
    rc, _ = run_eval(world, out, "--resume")
    assert rc == 0 and (run / "results.json").exists() and not (run / "results_partial.json").exists()
    # perf covers every process that ran the arm (perf_parts.jsonl), not only the resumed slice
    res = json.loads((run / "results.json").read_text())
    segs = jl(run / "perf_parts.jsonl")
    assert res["perf"]["segments"] == len(segs) >= 1 and res["perf"]["resumed"] is (len(segs) > 1)
    assert res["perf"]["steps"] == sum(s["steps"] for s in segs) >= len(jl(run / "calls.jsonl"))
    assert res["perf"]["steps"] > res["perf"]["this_process"]["steps"] or len(segs) == 1
    assert res["resumed_from"]["file"] == "results_partial.json" and res["resumed_from"]["deadline"]["hit"] is True


@needs_repo
def test_eval_no_new_task_near_deadline(world, tmp_path):
    out = tmp_path / "evals"
    rc, srv = run_eval(world, out, deadline=100.0)  # < 150 s left: no task starts
    assert rc == 4 and srv.n_chat == 0
    part = json.loads((rd(out, "c") / "results_partial.json").read_text())
    assert len(part["tasks_missing"]) == 5 and part["deadline"]["new_tasks_stopped"] is True


@needs_repo
def test_eval_refusals(world, tmp_path):
    w = world
    out = tmp_path / "e"
    cases = [(("--limit", "2"), {}), ((), {"arms": "blind,c"}), ((), {"tier": "ALL"}), (("--max-model-len", "65536"), {}),
             ((), {"task_set": "eval9"}), ((), {"arms": "c,c"})]
    for extra, kw in cases:
        with pytest.raises(run_tvr.Refused):
            run_eval(w, out, *extra, **kw)
    a = run_tvr.build_parser().parse_args(["eval", "--start", "Q3B", "--tier", "raw", "--arms", "c", "--task-set",
                                           "eval500", "--out-root", str(out), "--envs", "6", "--lp", "2",
                                           "--profile", "envlib", "--tvr-out", str(w.tvr_out), "--tasks-dir",
                                           str(w.tasks_dir), "--reach", str(w.reach), "--bfs", str(w.bfs)])
    with pytest.raises(run_tvr.Refused, match="E x L"):
        run_tvr.cmd_eval(a, ctl=run_tvr.Ctl(None))
    # the registered fallback and --limit with --allow-limit are accepted
    rc, srv = run_eval(w, tmp_path / "f", "--max-model-len", "128000", "--limit", "2", "--allow-limit")
    assert rc == 0
    res = json.loads((rd(tmp_path / "f", "c") / "results.json").read_text())
    assert res["n_tasks"] == 2 and res["server"]["max_model_len"] == 128000
    assert json.loads((rd(tmp_path / "f", "c") / "protocol.json").read_text())["max_model_len_used"] == 128000


@needs_repo
def test_eval_adapter_meta_tier_check(world, tmp_path):
    ad = tmp_path / "adapters" / "100"
    ad.mkdir(parents=True)
    (ad / "adapter.pt").write_bytes(b"weights")
    (ad / "meta.json").write_text(json.dumps({"start": "Q3B", "phase": "tier100", "seed": 1001}))
    md = tmp_path / "merged"
    md.mkdir()
    with pytest.raises(run_tvr.Refused, match="phase"):
        run_eval(world, tmp_path / "e", "--adapter-meta", str(ad / "meta.json"), "--model-dir", str(md), tier="ALL")
    with pytest.raises(run_tvr.Refused, match="model-dir"):
        run_eval(world, tmp_path / "e", "--adapter-meta", str(ad / "meta.json"), tier="100")
    rc, _ = run_eval(world, tmp_path / "e", "--adapter-meta", str(ad / "meta.json"), "--model-dir", str(md),
                     tier="100")
    assert rc == 0
    res = json.loads((rd(tmp_path / "e", "c", tier="100") / "results.json").read_text())
    assert res["adapter"]["adapter_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert res["adapter"]["tier"] == "tier100" and res["tier"] == "100"


@needs_repo
def test_eval_vllm_parity_inline(world, tmp_path):
    dry = make_parity_inputs(tmp_path)
    out = tmp_path / "e"
    rc, _ = run_eval(world, out, "--vllm-parity", str(dry))
    assert rc == 0
    rep = json.loads((out / "pilot__Q3B__raw__vllm_parity.json").read_text())
    assert rep["equal"] is True and rep["served"] == SERVED


# ================================================================================================================
# replay, render parity, official aggregate
# ================================================================================================================
def write_public(w, path):
    """sft_train.json records of the two SFT2 tasks (public format; texts as TVRBench builds them)."""
    data = []
    for t in SFT_TASKS:
        acts = SFT_ACTS[t["task_id"]]
        conv, imgs = [], []
        for k, a in enumerate(acts):
            conv += [{"from": "human", "value": "x"}, {"from": "gpt", "value": f"Action: {a}"}]
            imgs += [f"images/{t['task_id']}/step_{k:02d}.png", f"images/{t['task_id']}/target.png"]
        data.append({"system": tvr_text.SYSTEM_PROMPT, "conversations": conv, "images": imgs})
    path.write_text(json.dumps(data))
    return path


def run_replay(w, run, images, *extra, deadline=None):
    pub = write_public(w, w.tvr_out / "sft_train.json")
    a = run_tvr.build_parser().parse_args(
        ["replay", "--task-set", "sft2", "--public", str(pub), "--run-dir", str(run), "--images-dir", str(images),
         "--envs", "1", "--lp", "2", "--profile", "envlib", "--tvr-out", str(w.tvr_out), "--tasks-dir",
         str(w.tasks_dir), "--reach", str(w.reach), "--houses", str(w.houses), *extra])
    return run_tvr.cmd_replay(a, ctl=run_tvr.Ctl(deadline), pool_factory=local_pool(w))


@needs_repo
def test_replay_and_render_parity(world, tmp_path):
    w = world
    ref, ref_img = tmp_path / "replay", tmp_path / "images"
    assert run_replay(w, ref, ref_img) == 0
    res = json.loads((ref / "results.json").read_text())
    assert res["valid"] and res["sr"]["k"] == 2 and res["spl"] is None and res["kin"]["tasks_equal"] == 2
    tid = run_tvr.SFT2[0]
    tr = traj(ref, tid)
    assert tr["actions"] == SFT_ACTS[tid] and tr["vpd"]["action_success"] == [True, True, True, None]
    names = sorted(p.name for p in (ref_img / tid).glob("*.png"))
    assert names == ["step_00.png", "step_01.png", "step_02.png", "step_03.png", "target.png"]
    assert tr["vpd"]["images"][1]["cur_path"] == str((ref_img / tid / "step_01.png").resolve())
    assert hashlib.sha256((ref_img / tid / "step_01.png").read_bytes()).hexdigest() == \
        tr["vpd"]["images"][1]["cur_sha256"]
    assert run_replay(w, ref, ref_img, "--resume") == 0  # complete: skipped
    new, new_img = tmp_path / "replay2", tmp_path / "images2"
    assert run_replay(w, new, new_img) == 0
    out = tmp_path / "rp.json"
    rc = render_parity.main(["--replay", str(new), "--ref", str(ref), "--ref-images", str(ref_img), "--images",
                             str(new_img), "--task-set", "sft2", "--out", str(out)])
    doc = json.loads(out.read_text())
    assert rc == 0 and doc["pass"] and doc["n_frames"] == 10 and all(doc["exact"].values())
    assert doc["ssim_definition_sha256"] == hashlib.sha256(probe_lvp.SSIM_DEFINITION.encode()).hexdigest()
    assert doc["renderer"]["profile"] == "envlib" and doc["renderer"]["lp"] == 2
    assert doc["renderer"]["source"] == "results renderer (measured)" and doc["states_raw_equal"] is True


def blocky(seed, h=360, w=640):
    small = np.random.default_rng(seed).integers(0, 256, (h // 20, w // 20, 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, 20, axis=0), 20, axis=1)


def mini_replay(root: Path, tid: str, frames, *, states=None, texts=None):
    (root / "tasks" / tid).mkdir(parents=True, exist_ok=True)
    imgs = root / "imgs" / tid
    imgs.mkdir(parents=True, exist_ok=True)
    for k, f in enumerate(frames[:-1]):
        Image.fromarray(f).save(imgs / f"step_{k:02d}.png", format="PNG")
    Image.fromarray(frames[-1]).save(imgs / "target.png", format="PNG")
    n = len(frames) - 1
    st = states or [{"position": {"x": 0.0, "y": 0.9, "z": 0.25 * k}, "rotation_y": 0.0, "horizon": 0.0}
                    for k in range(n + 1)]
    v = {"states": st, "valid": [["RotateRight", "Stop"]] * n, "user_texts": texts or [f"t{k}" for k in range(n)],
         "action_success": [True] * (n - 1) + [None]}
    (root / "tasks" / tid / "trajectory.json").write_text(json.dumps({"task_id": tid, "vpd": v}))
    return root


def test_render_parity_rule_on_synthetic_frames(tmp_path):
    tid = "t1"
    frames = [blocky(i) for i in range(3)] + [blocky(9)]
    ref = mini_replay(tmp_path / "ref", tid, frames)

    def check(name, new_frames, **kw):
        new = mini_replay(tmp_path / name, tid, new_frames, **kw)
        out = tmp_path / f"{name}.json"
        rc = render_parity.main(["--replay", str(new), "--ref", str(ref), "--ref-images", str(ref / "imgs"),
                                 "--images", str(new / "imgs"), "--task-set", f"ids:{tid}", "--out", str(out)])
        return rc, json.loads(out.read_text())

    rc, d = check("same", frames)
    assert rc == 0 and d["pass"] and d["n_pass"] == 4 and all(f["identical"] for f in d["frames"])
    rng = np.random.default_rng(5)
    noisy = []
    for f in frames:
        g = f.astype(np.int16)
        idx = rng.integers(0, 360, 50), rng.integers(0, 640, 50)
        g[idx[0], idx[1], 0] += 1
        noisy.append(np.clip(g, 0, 255).astype(np.uint8))
    rc, d = check("noise", noisy)
    assert rc == 0 and d["pass"] and d["min_psnr_db"] > 40 and not any(f["identical"] for f in d["frames"])
    shifted = [np.roll(f, 40, axis=1) for f in frames]
    rc, d = check("shift", shifted)
    assert rc == 1 and not d["pass"] and d["n_pass"] == 0
    st = [{"position": {"x": 0.25, "y": 0.9, "z": 0.25 * k}, "rotation_y": 0.0, "horizon": 0.0} for k in range(4)]
    rc, d = check("states", frames, states=st)
    assert rc == 1 and d["exact"]["states"] is False and d["n_pass"] == 4
    rc, d = check("text", frames, texts=["t0", "t1", "other"])
    assert rc == 1 and d["exact"]["user_texts"] is False
    # y is ignored (4.2), rounding to 2 decimals
    st_y = [{"position": {"x": 0.0001, "y": 0.95, "z": 0.25 * k}, "rotation_y": 0.04, "horizon": 0.0}
            for k in range(4)]
    rc, d = check("y", frames, states=st_y)
    assert rc == 0 and d["exact"]["states"] is True
    (tmp_path / "same" / "imgs" / tid / "target.png").unlink()
    out = tmp_path / "missing.json"
    rc = render_parity.main(["--replay", str(tmp_path / "same"), "--ref", str(ref), "--ref-images",
                             str(ref / "imgs"), "--images", str(tmp_path / "same" / "imgs"), "--task-set",
                             f"ids:{tid}", "--out", str(out)])
    assert rc == 2 and json.loads(out.read_text())["pass"] is False


def test_render_parity_thresholds_at_the_boundary(tmp_path):
    """The frozen rule's numbers are really applied: PSNR 40 dB (a uniform +2 / +3 brightness shift on mid-range
    frames: about 42.1 / 38.6 dB, SSIM above 0.99 for both) and SSIM 0.99 (sigma-1.5 noise: PSNR about 44.6 dB but
    SSIM below 0.99 on the flat blocks)."""
    assert pilot_cfg.FROZEN["render"]["consistency_rule"]["psnr_min_db"] == 40.0
    assert pilot_cfg.FROZEN["render"]["consistency_rule"]["ssim_min"] == 0.99
    tid = "t1"
    frames = [(40 + blocky(i).astype(np.int32) * 175 // 255).astype(np.uint8) for i in range(2)]
    ref = mini_replay(tmp_path / "ref", tid, frames)

    def check(name, new_frames):
        new = mini_replay(tmp_path / name, tid, new_frames)
        out = tmp_path / f"{name}.json"
        rc = render_parity.main(["--replay", str(new), "--ref", str(ref), "--ref-images", str(ref / "imgs"),
                                 "--images", str(new / "imgs"), "--task-set", f"ids:{tid}", "--out", str(out)])
        return rc, json.loads(out.read_text())

    rc, d = check("plus2", [f + 2 for f in frames])
    assert 41.5 < d["min_psnr_db"] < 42.5 and d["min_ssim"] >= 0.99, d["frames"]
    assert rc == 0 and d["pass"]
    rc, d = check("plus3", [f + 3 for f in frames])
    assert 38.0 < d["min_psnr_db"] < 39.0 and d["min_ssim"] >= 0.99, d["frames"]
    assert rc == 1 and not d["pass"] and d["n_pass"] == 0
    rng = np.random.default_rng(7)
    noisy = [np.clip(f + np.rint(rng.normal(0.0, 1.5, f.shape)), 0, 255).astype(np.uint8) for f in frames]
    rc, d = check("ssim", noisy)
    assert min(probe_lvp.fnum(f["psnr_db"]) for f in d["frames"]) > 42.0 and d["min_ssim"] < 0.99, d["frames"]
    assert rc == 1 and not d["pass"] and d["n_pass"] == 0


@needs_repo
def test_score_official_cli(world, tmp_path, monkeypatch):
    out = tmp_path / "evals"
    rc, _ = run_eval(world, out)
    assert rc == 0
    run = rd(out, "c")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(world.fake), str(paths.WORKTREE)]))
    rc = score_tvr.main(["official", "--run-dir", str(run), "--served", SERVED, "--bfs", str(world.bfs), "--py",
                         sys.executable, "--repo", str(REPO)])
    chk = json.loads((run / "official_check.json").read_text())
    assert rc == 0 and chk["equal"] and chk["official_k"] == 3 and chk["official_n"] == 5, chk
    assert (run / "tasks" / "results.json").is_file() and (run / "tasks" / "summary.txt").is_file()
    assert json.loads((run / "official_opt.json").read_text())[EVAL_TASKS[3]["task_id"]] == \
        {"feasible": True, "optimal_steps": 3}
    assert chk.get("spl_equal") is True
    # a mechanism-only dir is refused
    out2 = tmp_path / "smoke"
    rc, _ = run_eval(world, out2, "--mechanism-only")
    assert score_tvr.main(["official", "--run-dir", str(rd(out2, "c")), "--served", SERVED, "--bfs",
                           str(world.bfs)]) == 2
    assert score_tvr.main(["official", "--run-dir", str(tmp_path / "nothing"), "--served", SERVED, "--bfs",
                           str(world.bfs)]) == 2


# ================================================================================================================
# score_tvr on hand-made trajectories
# ================================================================================================================
def mk_traj(run: Path, tid, scene, dataset, difficulty, poses, actions, stopped, target, succ=None, mech=False):
    states = [kin_env.pose_dict(kin_env.Pose(*p), 0.9) for p in poses]
    tgt = kin_env.pose_dict(kin_env.Pose(*target), 0.9)
    fin = states[-1]
    success = bool(stopped and run_tvr.is_exact_match(fin, tgt))
    tr = {"task_id": tid, "dataset": dataset, "scene": scene, "difficulty": difficulty,
          "result": {"success": None if mech else success, "stopped": stopped, "num_steps": len(actions),
                     "n_invalid_actions": actions.count("InvalidAction"),
                     "final_pos_error": run_tvr.position_error(fin, tgt),
                     "final_rot_error": run_tvr.rotation_error(fin, tgt),
                     "final_hor_error": run_tvr.horizon_error(fin, tgt)},
          "actions": actions, "start": states[0], "target": tgt, "raw_first_response": "", "raw_last_response": "",
          "vpd": {"states": states, "cap": 30 if dataset == "ithor" else 40,
                  "category": pilot_cfg.CATEGORY_OF[(dataset, difficulty)],
                  "n_blocked": sum(1 for a, s in zip(actions, succ or []) if s is False),
                  "n_unparseable": actions.count("InvalidAction"), "regex_mismatch": 0}}
    (run / "tasks" / tid).mkdir(parents=True, exist_ok=True)
    (run / "tasks" / tid / "trajectory.json").write_text(json.dumps(tr))
    return tr


class FakeReach:
    def __init__(self, cells):
        self.rs = kin_env.ReachSet.from_positions([{"x": x, "y": 0.9, "z": z} for x, z in cells])
        self.sha256 = "x"

    def reach(self, tid):
        return self.rs

    def digest(self, tid):
        return self.rs.digest


def test_score_run_metrics(tmp_path):
    run = tmp_path / "run"
    cells = [(x, z) for x in GRID for z in GRID if (x, z) != (0.0, 0.25)]
    # a: success after a blocked MoveAhead + 2 rotations + Stop: n_nonstop 3, opt 2 -> SPL 2/3
    mk_traj(run, "a", "S1", "ithor", "easy", [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 45, 0), (0, 0, 90, 0), (0, 0, 90, 0)],
            ["MoveAhead", "RotateRight", "RotateRight", "Stop"], True, (0, 0, 90, 0), succ=[False, True, True, None])
    # b: stopped one RotateRight short (within1 true), not a success
    mk_traj(run, "b", "S1", "ithor", "hard", [(0, 0, 0, 0), (0, 0, 0, 0)], ["Stop"], True, (0, 0, 45, 0))
    # c: stopped far away
    mk_traj(run, "c", "S2", "procthor", "easy", [(0, 0, 0, 0), (0, 0, 0, 0)], ["Stop"], True, (0.75, 0.75, 0, 0))
    # d: capped, never stopped, visited the target once (reached)
    poses = [(0, 0, 0, 0)] + [((k % 2) * 0.25, 0, 0, 0) for k in range(1, 41)]
    acts = ["MoveRight" if k % 2 == 0 else "MoveLeft" for k in range(40)]
    mk_traj(run, "d", "S3", "procthor", "hard", poses, acts, False, (0.25, 0.0, 0, 0))
    bfs = {"tasks": {"a": {"feasible": True, "opt": 2}, "b": {"feasible": True, "opt": 1},
                     "c": {"feasible": True, "opt": 6}, "d": {"feasible": False, "opt": None}}}
    res, valid = score_tvr.score_run(run, task_ids=["a", "b", "c", "d"], bfs_table=bfs, reach_index=FakeReach(cells),
                                     arm="c", meta={"server_alive_at_end": True, "api_errors": 0})
    assert valid and (run / "results.json").exists()
    assert res["sr"] == {"k": 1, "n": 4, "rate": 0.25, "wilson95": score_tvr.wilson(1, 4)}
    assert res["spl"] == round((2 / 3) / 4, 6)
    assert res["spl_official"] == round((2 / 4) / 4, 6)  # eval_va: Stop counted, min/max
    assert res["within1"]["k"] == 2 and res["reached"]["k"] == 2
    assert res["stop"] == {"n_stopped": 3, "stop_rate": 0.75, "fstop_given_stop": round(2 / 3, 6),
                           "fstop_over_all": 0.5}
    assert res["invalid"]["blocked_steps"] == 1 and res["steps"]["capped_tasks"] == 1 and res["steps"]["total"] == 46
    assert res["by_category"]["SR-easy"]["sr"]["k"] == 1 and res["by_category"]["LR-hard"]["n"] == 1
    rows = {r["task_id"]: r for r in jl(run / "per_task.jsonl")}
    assert rows["a"]["n_nonstop"] == 3 and rows["a"]["spl"] == pytest.approx(2 / 3) and rows["d"]["opt"] is None
    assert rows["d"]["capped"] and rows["d"]["reached"] and not rows["d"]["within1"]
    lb, ub = score_tvr.wilson(1, 4)
    assert lb == pytest.approx(0.0456, abs=1e-3) and ub == pytest.approx(0.6994, abs=1e-3)
    b1 = score_tvr.scene_bootstrap(jl(run / "per_task.jsonl"), 500, 0)
    assert b1 == score_tvr.scene_bootstrap(jl(run / "per_task.jsonl"), 500, 0) and b1["n_clusters"] == 3
    assert res["bootstrap_scene"]["resamples"] == 10000 and res["bootstrap_scene"]["n_clusters"] == 3
    pt_before = (run / "per_task.jsonl").read_bytes()
    with pytest.raises(RuntimeError):  # a valid results.json is never overwritten (nor its per_task.jsonl)
        score_tvr.score_run(run, task_ids=["a"], bfs_table=bfs, reach_index=None, arm="c", meta={})
    assert (run / "per_task.jsonl").read_bytes() == pt_before and len(jl(run / "per_task.jsonl")) == 4
    # invalid: a missing task / an API error / a task failure / a dead server; partial
    for meta, ids in (({}, ["a", "zz"]), ({"api_errors": 1}, ["a"]), ({"task_failures": [{"task_id": "x"}]}, ["a"]),
                      ({"server_alive_at_end": False}, ["a"])):
        r2 = tmp_path / f"inv{len(meta)}{len(ids)}{list(meta)}"
        for tid in ("a",):
            (r2 / "tasks" / tid).mkdir(parents=True)
            (r2 / "tasks" / tid / "trajectory.json").write_text((run / "tasks" / tid / "trajectory.json").read_text())
        _, v = score_tvr.score_run(r2, task_ids=ids, bfs_table=None, reach_index=None, arm="c", meta=meta)
        assert not v and (r2 / "results_invalid.json").exists()
    _, v = score_tvr.score_run(run, task_ids=["a"], bfs_table=None, reach_index=None, arm="c", meta={}, partial=True,
                               overwrite_valid=True)
    assert not v and (run / "results_partial.json").exists() and not (run / "results.json").exists()


def test_score_run_mechanism_only(tmp_path):
    run = tmp_path / "m"
    mk_traj(run, "a", "S1", "ithor", "easy", [(0, 0, 0, 0), (0, 0, 0, 0)], ["Stop"], True, (0, 0, 0, 0), mech=True)
    res, valid = score_tvr.score_run(run, task_ids=["a"], bfs_table=None, reach_index=None, arm="c",
                                     mechanism_only=True, meta={})
    assert valid and res["withheld"] == "mechanism smoke"
    assert not ({"sr", "spl", "within1", "stop", "reached", "bootstrap_scene", "errors_final"} & set(res))
    assert set(jl(run / "per_task.jsonl")[0]) == set(score_tvr.MECH_COLUMNS)


# ================================================================================================================
# parity_tvr against FakeServer
# ================================================================================================================
def make_parity_inputs(tmp: Path, bad_turn=None) -> Path:
    img_dir = tmp / "pimg"
    img_dir.mkdir(parents=True, exist_ok=True)
    recs, parity = [], []
    for d in range(3):
        tid = f"demo{d}"
        msgs = [{"role": "system", "content": tvr_text.SYSTEM_PROMPT}]
        imgs = []
        for k in range(3):
            for name, seed in ((f"{tid}_s{k}.png", 10 * d + k), (f"{tid}_t.png", 100 + d)):
                p = img_dir / name
                if not p.exists():
                    Image.fromarray(blocky(seed)).save(p, format="PNG")
                imgs.append(str(p))
            msgs.append({"role": "user", "content": tvr_text.build_concat_user_text(["RotateRight", "Stop"])})
            msgs.append({"role": "assistant", "content": "Action: RotateRight" if k < 2 else "Action: Stop"})
        rec = {"id": tid, "messages": msgs, "images": imgs}
        recs.append(rec)
        pref = []
        from vpd.train import n2_train_vpd as TT
        for turn, ctx, cimgs in TT.prefix_contexts(rec, 0, False):
            body = parity_tvr.oai_request(ctx, cimgs, SERVED)
            n = fake_server.prompt_tokens(body["messages"]) + (1 if (d, turn) == bad_turn else 0)
            pref.append({"turn": turn, "n_images": len(cimgs), "prompt_tokens": n})
        parity.append({"id": tid, "prefixes": pref})
    data = tmp / "parity3.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in recs))
    dry = tmp / "dry_run.json"
    dry.write_text(json.dumps({"start": "Q3B", "data": str(data), "history_k": 0, "blind": False, "parity": parity}))
    return dry


def test_parity_tvr_against_fake_server(tmp_path):
    dry = make_parity_inputs(tmp_path / "ok")
    with fake_server.FakeServer(policy="stop_at", model=SERVED) as s:
        rep = parity_tvr.vllm_parity("Q3B", dry, s.base_url, tmp_path / "p.json")
        assert rep["equal"] and rep["n"] == 9 and rep["demos"] == ["demo0", "demo1", "demo2"]
        body = s.requests[0]
        assert body["max_tokens"] == 1 and body["model"] == SERVED
        urls = [p["image_url"]["url"] for m in body["messages"] if isinstance(m["content"], list)
                for p in m["content"] if p["type"] == "image_url"]
        assert urls and all(u.startswith("data:image/png;base64,") for u in urls)
        bad = make_parity_inputs(tmp_path / "bad", bad_turn=(1, 4))
        rep = parity_tvr.vllm_parity("Q3B", bad, s.base_url, tmp_path / "p2.json")
        assert not rep["equal"] and sum(not r["equal"] for r in rep["rows"]) == 1
        assert parity_tvr.main(["--start", "Q3B", "--dry-run-json", str(dry), "--base-url", s.base_url, "--out",
                                str(tmp_path / "p3.json")]) == 0
    with pytest.raises(ValueError):
        parity_tvr.vllm_parity("Q3S", dry, "http://127.0.0.1:1/v1", tmp_path / "x.json")


# ================================================================================================================
# SIGTERM: exit 143 within 15 s, no child left (subprocess workers on the fake ai2thor + in-process FakeServer)
# ================================================================================================================
def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ", 1)[1][0] != "Z"
    except OSError:
        return False


@needs_repo
@pytest.mark.skipif(not HAVE_FONT, reason="the pinned badge font of the TVR env is not readable")
def test_sigterm_fast_exit_no_children(world, tmp_path):
    w = world
    out = tmp_path / "evals"
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), VPD_WORKTREE=str(paths.WORKTREE),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    cmd = [sys.executable, "-B", "-m", "vpd.tvr.run_tvr", "eval", "--start", "Q3B", "--tier", "raw", "--arms", "c",
           "--task-set", "eval500", "--out-root", str(out), "--envs", "2", "--lp", "1", "--profile", "envlib",
           "--tvr-out", str(w.tvr_out), "--tasks-dir", str(w.tasks_dir), "--reach", str(w.reach), "--bfs",
           str(w.bfs), "--houses", str(w.houses), "--fake-server", "--fake-policy", "stop_at", "--fake-stop-after",
           "1000000", "--fake-thor", str(w.fake), "--deadline-s", "600"]
    log = open(tmp_path / "run.log", "w")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(paths.WORKTREE))
    run = rd(out, "c")
    pids = []
    try:
        end = time.monotonic() + 120
        while time.monotonic() < end and p.poll() is None:
            m = run / "manifest.json"
            if m.exists():
                doc = json.loads(m.read_text())
                pids = [r["pid"] for r in (doc.get("pool") or {}).get("ready") or [] if r]
                if len(pids) == 2:
                    break
            time.sleep(0.2)
        assert p.poll() is None and len(pids) == 2, (tmp_path / "run.log").read_text()[-3000:]
        time.sleep(1.0)
        t0 = time.monotonic()
        p.send_signal(signal.SIGTERM)
        rc = p.wait(timeout=15)
        assert rc == 143 and time.monotonic() - t0 < 15
        assert not (run / "results.json").exists()
        time.sleep(0.5)
        assert not any(_alive(x) for x in pids), pids
    finally:
        if p.poll() is None:
            p.kill()
        for x in pids:
            try:
                os.killpg(x, signal.SIGKILL)
            except OSError:
                pass
        log.close()


# ================================================================================================================
# robustness: deadline outside the arm threads, dead server, dead slot, keep-alive, spawn failures, signals
# ================================================================================================================
class SlowStartServer(fake_server.FakeServer):
    def __enter__(self):
        time.sleep(1.5)
        return super().__enter__()


@needs_repo
def test_eval_deadline_during_server_start_is_partial(world, tmp_path):
    out = tmp_path / "evals"
    srv = SlowStartServer(policy="stop_at", model=SERVED)
    t0 = time.monotonic()
    rc, _ = run_eval(world, out, srv=srv, deadline=1.0, arms="c,blind")
    assert rc == 4 and srv.n_chat == 0 and time.monotonic() - t0 < 30
    for arm in ("c", "blind"):
        part = json.loads((rd(out, arm) / "results_partial.json").read_text())
        assert part["partial"] and part["deadline"]["hit"] is True and len(part["tasks_missing"]) == 5
        assert not (rd(out, arm) / "results.json").exists()


@needs_repo
def test_eval_deadline_between_arms_is_partial(world, tmp_path, monkeypatch):
    orig = run_tvr.Harness.finalize

    def finalize(self, ar, partial):
        out = orig(self, ar, partial)
        if ar.arm == "c":
            self.ctl.deadline_s = 0.0  # the soft deadline passes during arm c's finalize
        return out

    monkeypatch.setattr(run_tvr.Harness, "finalize", finalize)
    out = tmp_path / "evals"
    srv = fake_server.FakeServer(policy="stop_at", model=SERVED)
    a = eval_args(world, out, arms="c,u")
    rc = run_tvr.cmd_eval(a, ctl=run_tvr.Ctl(1000.0), pool_factory=local_pool(world), server_factory=lambda: srv)
    assert rc == 4
    assert json.loads((rd(out, "c") / "results.json").read_text())["valid"] is True
    part = json.loads((rd(out, "u") / "results_partial.json").read_text())
    assert len(part["tasks_missing"]) == 5 and part["deadline"]["hit"] is True
    assert not any(q.get("max_tokens") == 64 for q in srv.requests)  # arm u never ran


@needs_repo
def test_eval_server_death_ends_the_run_fast(world, tmp_path):
    out = tmp_path / "evals"
    srv = fake_server.FakeServer(model=SERVED)
    seen = {"n": 0}

    def policy(body, n):
        seen["n"] += 1
        if seen["n"] == 4:
            srv.kill()
        return "Action: RotateRight"

    srv.policy = policy
    t0 = time.monotonic()
    rc, _ = run_eval(world, out, srv=srv, arms="c,u,blind")  # default backoff (2, 4, 8 s) is never waited out
    assert rc == 1 and time.monotonic() - t0 < 30
    c = json.loads((rd(out, "c") / "results_invalid.json").read_text())
    assert c["server_alive_at_end"] is False and c["server_dead"] is True and c["task_failures"] >= 1
    for arm in ("u", "blind"):
        r = json.loads((rd(out, arm) / "results_invalid.json").read_text())
        assert len(r["tasks_missing"]) == 5 and r["server_dead"] is True
    assert srv.n_chat <= 10


@needs_repo
def test_eval_dead_slot_requeues_with_its_attempt_count(world, tmp_path):
    starts, crashed = {"n": 0}, {"n": 0}
    bad = EVAL_TASKS[0]["task_id"]  # first task of the FloorPlan13 group

    class W(thor_worker.Worker):
        def start(self):
            starts["n"] += 1
            if starts["n"] > 2:
                raise RuntimeError("no more workers")  # every restart fails: the slot dies
            return super().start()

        def op_reset(self, cmd):
            if cmd["task"]["task_id"] == bad and crashed["n"] == 0:
                crashed["n"] += 1
                raise RuntimeError("one-off crash")
            return super().op_reset(cmd)

    out = tmp_path / "evals"
    rc, _ = run_eval(world, out, worker_cls=W)
    assert rc == 0 and crashed["n"] == 1 and starts["n"] == 3
    res = json.loads((rd(out, "c") / "results.json").read_text())
    assert res["valid"] and res["tasks_missing"] == [] and res["renderer"]["n_workers_live"] == 1
    trs = {t["task_id"]: traj(rd(out, "c"), t["task_id"]) for t in EVAL_TASKS}
    assert trs[bad]["vpd"]["attempt"] == 2  # attempt 2 on the other slot, not a fresh attempt 1
    assert len({tr["vpd"]["worker"] for tr in trs.values()}) == 1
    assert [h["ok"] for h in res["renderer"]["start_history"]] == [True, True, False]


def test_slot_harness_error_requeues_the_group(tmp_path, monkeypatch):
    """An unexpected exception outside the task (here: task_with_retry itself) never drops a scene group."""
    tasks = [dict(EVAL_TASKS[0]), dict(EVAL_TASKS[1])]
    ar = run_tvr.ArmRun(arm="c", name="x", run_dir=tmp_path / "x", tasks=tasks, protocol={"p": 1})
    ar.todo = list(tasks)

    class Pool:
        n = 2

        def alive(self, w):
            return True

    h = run_tvr.Harness(ctl=run_tvr.Ctl(None), reach_index=None, served=SERVED, pool=Pool(),
                        server=types.SimpleNamespace(base_url="http://127.0.0.1:9/v1", alive=lambda: True))
    ran = []

    def twr(ar_, task, *, wid, client, first_attempt=1):
        if not ran:
            ran.append(("boom", wid, first_attempt))
            raise KeyError("harness bug")
        ran.append((task["task_id"], wid, first_attempt))
        return "ok"

    monkeypatch.setattr(h, "task_with_retry", twr)
    h.run_image_arm(ar)
    assert ran[0][0] == "boom"
    dead = ran[0][1]
    done = ran[1:]
    assert sorted(r[0] for r in done) == sorted(t["task_id"] for t in tasks)  # the whole group ran once more
    assert all(r[1] != dead for r in done)


def test_chat_client_keepalive_idle_close(monkeypatch):
    body = json.dumps({"messages": []}).encode()
    with fake_server.FakeServer(policy="stop_at", idle_close_s=0.3) as s:
        # the server closed the idle socket: retried at once on a fresh connection, no retry spent
        monkeypatch.setattr(run_tvr, "KEEPALIVE_IDLE_S", 100.0)
        c = run_tvr.ChatClient(s.base_url, run_tvr.Ctl(None))
        assert c.chat(body)[2] == 0
        time.sleep(0.8)
        _, _, retries, errs = c.chat(body)
        assert retries == 0 and errs == [] and c.reconnects == 1
        # idle longer than KEEPALIVE_IDLE_S: a new connection before the request
        monkeypatch.setattr(run_tvr, "KEEPALIVE_IDLE_S", 0.2)
        c2 = run_tvr.ChatClient(s.base_url, run_tvr.Ctl(None))
        n0 = s.n_connections
        c2.chat(body)
        time.sleep(0.5)
        _, _, retries, errs = c2.chat(body)
        assert retries == 0 and errs == [] and c2.reconnects == 0 and s.n_connections == n0 + 2
        c.close()
        c2.close()


def test_chat_client_dead_server_fails_fast():
    with fake_server.FakeServer(policy="stop_at") as s:
        c = run_tvr.ChatClient(s.base_url, run_tvr.Ctl(None), alive=s.alive)
        c.chat(b"{}")
        s.kill()
        time.sleep(0.5)
        t0 = time.monotonic()
        with pytest.raises(run_tvr.TaskError) as ei:
            c.chat(b"{}")
        assert ei.value.kind == "server_dead" and time.monotonic() - t0 < 5


def _sleeper_pool(tmp_path, **kw):
    """A subprocess ThorPool whose 'python' never answers the start-up (sh: exec sleep)."""
    sh = tmp_path / "fakepy.sh"
    sh.write_text("#!/bin/sh\nexec sleep 30\n")
    sh.chmod(0o755)
    (tmp_path / "fake_tvr_env" / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "releases").mkdir(exist_ok=True)
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    return thor_worker.ThorPool(1, 1, "envlib", out_dir=tmp_path / "out", houses_dir=None, py=str(sh),
                                code_root=paths.WORKTREE, base_env=base, tvr_env=tmp_path / "fake_tvr_env",
                                releases=tmp_path / "releases", **kw)


def test_pool_spawn_errors_never_raise(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError(11, "fake EAGAIN")

    monkeypatch.setattr(lvp, "prepare_home", boom)
    pool = _sleeper_pool(tmp_path, startup_timeout=5)
    with pytest.raises(thor_worker.PoolError):  # start(): a PoolError, never the OSError
        pool.start()
    assert [h["ok"] for h in pool.history] == [False, False] and "spawn failed" in pool.history[0]["error"]
    pool2 = _sleeper_pool(tmp_path, startup_timeout=5)
    rr = pool2.restart(0)  # restart(): the ready reply with ok False (contract 5.2.1)
    assert rr["ok"] is False and "fake EAGAIN" in rr["error"] and pool2.handles[0] is None


def test_pool_terminate_during_startup_kills_the_starting_worker(tmp_path, monkeypatch):
    made = []

    class Rec(probe_lvp.SubprocessHandle):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(probe_lvp, "SubprocessHandle", Rec)
    pool = _sleeper_pool(tmp_path, startup_timeout=60)

    class Term(BaseException):
        pass

    def raise_term(signum, frame):
        raise Term()

    old = signal.signal(signal.SIGALRM, raise_term)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        with pytest.raises(Term):  # run_tvr.Terminated arrives like this, in the main thread inside the wait
            pool.start()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    assert len(made) == 1 and pool._spawning is None
    assert made[0].proc.poll() is not None and not _alive(made[0].proc.pid)


def test_signal_handler_raises_once():
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    old_ctl = run_tvr._CTL
    try:
        run_tvr._CTL = run_tvr.Ctl(None)
        with pytest.raises(run_tvr.Terminated):
            run_tvr._on_signal(signal.SIGTERM, None)
        assert run_tvr._CTL.terminated and run_tvr._CTL.stop.is_set()
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        assert run_tvr._on_signal(signal.SIGTERM, None) is None  # a second TERM: ignored
    finally:
        run_tvr._CTL = old_ctl
        for s, hd in saved.items():
            signal.signal(s, hd)


def test_perf_sums_segments_over_resumes(tmp_path):
    ar = run_tvr.ArmRun(arm="c", name="x", run_dir=tmp_path / "x", tasks=[EVAL_TASKS[0]], protocol={"p": 1})
    ar.run_dir.mkdir()
    prev = {"protocol_sha256": ar.protocol_sha, "steps": 90, "wall_s": 45.0, "cores_mean": 4.0, "cores_max": 6.0}
    other = {"protocol_sha256": "other", "steps": 1000, "wall_s": 1.0, "cores_mean": 9.0, "cores_max": 9.0}
    (ar.run_dir / run_tvr.PERF_PARTS).write_text(json.dumps(prev) + "\n" + json.dumps(other) + "\n")
    ar.ran, ar.steps, ar.wall, ar.n_written, ar.cpu = True, 10, 5.0, 1, {"cores_mean": 2.0, "cores_max": 3.0}
    h = run_tvr.Harness(ctl=run_tvr.Ctl(None), reach_index=None, served=None)
    p = h.perf(ar)
    assert p["steps"] == 100 and p["wall_s"] == 50.0 and p["steps_per_s"] == 2.0 and p["segments"] == 2
    assert p["resumed"] is True and p["cgroup_cores_mean"] == round((4 * 45 + 2 * 5) / 50, 3)
    assert p["cgroup_cores_max"] == 6.0 and p["this_process"]["steps_per_s"] == 2.0
    ar2 = run_tvr.ArmRun(arm="c", name="y", run_dir=tmp_path / "x", tasks=[EVAL_TASKS[0]], protocol={"p": 1})
    assert h.perf(ar2)["segments"] == 2  # a process that ran nothing adds no segment


@needs_repo
def test_eval_empty_audits_never_pass(world, tmp_path, monkeypatch):
    """Real-server audit path with nothing to audit (no kept task): prompt parity is not vacuously equal."""
    class RealLike(fake_server.FakeServer):
        fake = False

    monkeypatch.setattr(run_tvr, "processor_dir_for", lambda start, md: tmp_path)
    monkeypatch.setattr(run_tvr, "image_token_audit", lambda run, pdir: {"n": 0, "tokens_per_image": [],
                                                                         "grid_set": [], "samples": []})
    monkeypatch.setattr(run_tvr, "hf_prompt_parity", lambda run, pdir: {"n": 0, "equal": True, "rows": []})
    out = tmp_path / "evals"
    rc, _ = run_eval(world, out, "--keep-images-per-category", "0", srv=RealLike(policy="stop_at", model=SERVED))
    res = json.loads((rd(out, "c") / "results.json").read_text())
    assert rc == 0 and res["audit"]["prompt_parity"]["equal"] is False and res["audit"]["prompt_parity"]["empty"]
    assert res["audit"]["image_tokens"]["empty"] is True
