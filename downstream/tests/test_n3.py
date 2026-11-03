"""N3 (the transfer ladder; the user's decision 2 of 2026-11-03): the camera-sphere collision test against the upstream
slab test, the robot-camera settings, one camera per evaluation run, the solvability re-check and its collection, and
the re-check job (downstream/n/n3_env.py, n3_gen.py, n3_episode.py, run_n3_recheck.sh)."""
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import n3_env as N3  # noqa: E402
import n3_episode as NE  # noqa: E402
import n3_gen as NG  # noqa: E402
import n_env as N  # noqa: E402


def upstream_blocked(lo, hi, p0, p1) -> bool:
    """CollisionDetector.check_collision's object part for one box, as the upstream code reads (point inside, or
    AABB.intersects_segment with t < 1)."""
    if np.all(p1 >= lo) and np.all(p1 <= hi):
        return True
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-10:
        return bool(np.all(p0 >= lo) and np.all(p0 <= hi))
    d = d / length
    t_min, t_max = 0.0, length
    for i in range(3):
        if abs(d[i]) < 1e-10:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
        else:
            t1, t2 = (lo[i] - p0[i]) / d[i], (hi[i] - p0[i]) / d[i]
            t1, t2 = min(t1, t2), max(t1, t2)
            t_min, t_max = max(t_min, t1), min(t_max, t2)
            if t_min > t_max:
                return False
    return t_min / length < 1.0


def test_segment_test_matches_the_upstream_one():
    rng = random.Random(3)
    lo = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 2)] for _ in range(300)])
    hi = lo + np.array([[rng.uniform(0.05, 0.8) for _ in range(3)] for _ in range(300)])
    for _ in range(200):
        p0 = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), 1.5])
        a = rng.uniform(0, 2 * math.pi)
        p1 = p0 + 0.3 * np.array([math.cos(a), math.sin(a), 0.0]) * rng.choice([0.0, 1.0])
        got = N3.segment_hits(lo, hi, p0, p1)
        assert [bool(x) for x in got] == [upstream_blocked(lo[i], hi[i], p0, p1) for i in range(len(lo))]


def test_robot_camera_settings_and_their_intrinsics(monkeypatch):
    for k in ("CAM_HEIGHT_M", "FOVY_DEG", "NATIVE_K", "make_item"):
        monkeypatch.setattr(N, k, getattr(N, k))
    k60 = N3.native_k(60.0)
    assert k60[0][0] == pytest.approx(240 / math.tan(math.radians(30))) and k60[0][2] == 320 and k60[1][2] == 240
    assert N3.native_k(N.FOVY_DEG)[0][0] == pytest.approx(320.0)  # camera A: fx = fy = 320 (N1's NATIVE_K)
    monkeypatch.setattr(N3, "_MAKE_ITEM", lambda *a: {"init_camera": {"intrinsics": "canonical"}})
    N3.use_camera(1.07, 60.0)
    assert N.CAM_HEIGHT_M == 1.07 and N.FOVY_DEG == 60.0 and N.NATIVE_K == k60
    assert N.make_item("equidistance", "s", [], {}, (0, 0, 0))["init_camera"]["intrinsics"] == k60
    assert N.c2w_of((0.0, 0.0, 0.0))[2, 3] == pytest.approx(1.07)  # the prompt's pose line and the judge follow


def test_one_camera_per_run(monkeypatch):
    monkeypatch.setattr(N, "Scene", N.Scene)
    for k in ("CAM_HEIGHT_M", "FOVY_DEG", "NATIVE_K", "make_item"):
        monkeypatch.setattr(N, k, getattr(N, k))
    assert NE.camera_of([{"instance_id": "a"}, {"instance_id": "b"}]) == ("A", None)
    cam = {"mode": "robot", "height_m": 1.1, "fovy_deg": 60.0, "native_k": "x"}
    assert NE.camera_of([{"camera": cam}, {"camera": dict(cam, native_k="y")}]) == ("robot", {"height_m": 1.1,
                                                                                              "fovy_deg": 60.0})
    with pytest.raises(SystemExit):
        NE.camera_of([{"camera": cam}, {}])
    made = {}
    monkeypatch.setattr(N3, "Scene", lambda *a, **k: made.update(args=a, kw=k) or "scene")
    assert NE.install("sphere", [{"camera": cam}]) == ("robot", {"height_m": 1.1, "fovy_deg": 60.0})
    assert N.CAM_HEIGHT_M == 1.1 and N.Scene("T", 3, hide_robot=True) == "scene"
    assert made == {"args": ("T", 3), "kw": {"hide_robot": True, "collision": "sphere", "camera": "robot",
                                             "camera_values": {"height_m": 1.1, "fovy_deg": 60.0}}}


class FakeScene:
    """Success from 0.6 m forward; render checks counted."""

    def __init__(self):
        self.judged = 0

    def judge(self, item, pose, render=True):
        self.judged += 1
        return {"success": pose[0] > 0.5, "visible": {"a": 0.01}}

    def render(self, pose=None):
        return np.zeros((4, 4, 3), np.uint8), None


def test_recheck_finds_the_shortest_success_and_caps_the_checks(monkeypatch):
    states = {0: ((0.0, 0.0, 0.0), 0, None, None), 1: ((0.3, 0.0, 0.0), 1, 0, "move_forward"),
              2: ((0.6, 0.0, 0.0), 2, 1, "move_forward"), 3: ((0.9, 0.0, 0.0), 3, 2, "move_forward")}
    monkeypatch.setattr(N, "box_judge", lambda item, pose: {"success": pose[0] > 0.2})
    sc = FakeScene()
    r = NG.recheck_instance(sc, states, [0, 1, 2, 3], {"item": {}})
    assert r["solvable"] and r["actions"] == 2 and r["path"] == ["move_forward"] * 2 and r["checks"] == 2
    assert r["chroma"] == 0.0
    monkeypatch.setattr(NG, "MAX_CHECKS", 1)
    r = NG.recheck_instance(FakeScene(), states, [0, 1, 2, 3], {"item": {}})
    assert not r["solvable"] and r["stopped"] == "checks" and r["checks"] == 1


def test_collect_marks_solvable_unsolvable_corrupted_and_missing(tmp_path):
    insts = [{"instance_id": f"T_s{s}_{t}", "scene_id": f"T_s{s}", "task": "T", "seed": s, "task_type": t}
             for s in (1, 2, 3) for t in ("equidistance", "occlusion_alignment")]
    (tmp_path / "inst.jsonl").write_text("".join(json.dumps(x) + "\n" for x in insts))
    (tmp_path / "out/scenes").mkdir(parents=True)
    (tmp_path / "out/scenes/T_s1.json").write_text(json.dumps({
        "scene_id": "T_s1", "complete": True, "collision": "sphere", "start_chroma": 0.4, "start_inside": None,
        "instances": {"T_s1_equidistance": {"solvable": True, "actions": 3, "chroma": 0.5},
                      "T_s1_occlusion_alignment": {"solvable": False, "checks": 4}}}))
    (tmp_path / "out/scenes/T_s2.json").write_text(json.dumps({
        "scene_id": "T_s2", "complete": True, "collision": "sphere", "start_chroma": 40.0,
        "instances": {"T_s2_equidistance": {"solvable": True, "chroma": 0.5}}}))
    s = NG.recheck_collect(tmp_path / "inst.jsonl", tmp_path / "out")
    assert s["status"] == {"solvable": 1, "unsolvable": 1, "corrupted": 2, "missing": 2}
    assert s["missing_scenes"] == ["T_s3"] and s["collision"] == ["sphere"]
    solv = [json.loads(x) for x in (tmp_path / "out/solvable.jsonl").read_text().splitlines()]
    assert [x["instance_id"] for x in solv] == ["T_s1_equidistance"]
    checked = [json.loads(x) for x in (tmp_path / "out/instances_checked.jsonl").read_text().splitlines()]
    assert len(checked) == 6 and checked[0]["recheck"]["status"] == "solvable"


STUB = r"""import json, os, sys, time
from pathlib import Path
a = sys.argv[1:]
def arg(k):
    return a[a.index(k) + 1]
if a[0] == "recheck-scene":
    out = Path(arg("--out"))
    with open(out / "calls.jsonl", "a") as f:
        f.write(json.dumps(a) + "\n")
    if arg("--task") == "Slow":
        time.sleep(60)
    (out / "scenes" / f"{arg('--task')}_s{arg('--seed')}.json").write_text(json.dumps({"complete": True}))
else:
    Path(arg("--out"), "collected").write_text(" ".join(a))
"""


def test_recheck_job_runs_each_scene_once_and_skips_a_slow_one(tmp_path):
    from test_run_probe_episode import SIM_ENV

    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/eval").mkdir(parents=True)
    (wt / "downstream/n/run_n3_recheck.sh").write_text((ROOT / "n/run_n3_recheck.sh").read_text())
    (wt / "downstream/n/n3_gen.py").write_text(STUB)
    (wt / "downstream/eval/sim_env.sh").write_text(SIM_ENV)
    rows = [{"task": t, "seed": s} for t, s in (("Fast", 1), ("Slow", 2), ("Fast", 1), ("Fast", 3))]
    (tmp_path / "inst.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    env = dict(os.environ, SCENE_TIMEOUT="3")
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n3_recheck.sh"), str(out), str(tmp_path / "inst.jsonl"),
                          "sphere", "2"], capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
    assert sorted(c[c.index("--task") + 1] + c[c.index("--seed") + 1] for c in calls) == ["Fast1", "Fast3", "Slow2"]
    assert all(c[c.index("--collision") + 1] == "sphere" for c in calls)
    assert json.loads((out / "scenes/Slow_s2.timeout.json").read_text())["rc"] == 137
    assert (out / "collected").read_text().startswith("recheck-collect --instances")
    again = subprocess.run(["bash", str(wt / "downstream/n/run_n3_recheck.sh"), str(out), str(tmp_path / "inst.jsonl"),
                            "sphere", "2"], capture_output=True, text=True, env=env, timeout=60)
    assert again.returncode == 0 and len((out / "calls.jsonl").read_text().splitlines()) == 3  # nothing redone


def test_gen_job_makes_tier_e_scenes_with_the_robot_camera(tmp_path):
    """run_n_gen.sh N_CAMERA=robot: scenes by n3_gen.py scene --camera robot; collect and page stay n_gen.py's."""
    from test_n_gen import STUB as GEN_STUB
    from test_run_probe_episode import SIM_ENV

    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/eval").mkdir(parents=True)
    (wt / "downstream/n/run_n_gen.sh").write_text((ROOT / "n/run_n_gen.sh").read_text())
    (wt / "downstream/n/n_gen.py").write_text(GEN_STUB)
    (wt / "downstream/n/n3_gen.py").write_text(GEN_STUB.replace('cmd = sys.argv[1]', 'cmd = sys.argv[1]\n'
                                                                'open(sys.argv[sys.argv.index("--out") + 1] + "/n3", "a").write(cmd + chr(10))'))
    (wt / "downstream/eval/sim_env.sh").write_text(SIM_ENV)
    out = tmp_path / "out"
    env = dict(os.environ, N_X0=str(tmp_path / "x0.jsonl"), SCENE_TIMEOUT="3", SOFT_LIMIT="2", N_CAMERA="robot")
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n_gen.sh"), str(out), "3", "0", "50"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 3 and all(c[:3] == ["scene", "--camera", "robot"] for c in calls)
    assert (out / "n3").read_text().split() == ["scene"] * 3 and (out / "collect.done").is_file()


def test_model_job_runs_its_tiers_with_their_rules(tmp_path):
    """run_n3.sh: tier c (sphere) only for the six SFT / RL models, tier e (N1's rule) for all; Q3 models' prompt."""
    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/n/run_n3.sh").write_text((ROOT / "n/run_n3.sh").read_text())
    (wt / "downstream/n/run_n3_episodes.sh").write_text(
        'mkdir -p "$1"; echo "$1|$2|$3|$4|$5|$N3_COLLISION|${N_VAL_ENV:-}" >> "$(dirname "$(dirname "$1")")/calls"\n'
        'echo \'{"success": true}\' >> "$1/episodes.jsonl"\n')
    env = {k: v for k, v in os.environ.items() if k != "N_VAL_ENV"}
    for model in ("B", "Q3RS"):
        res = subprocess.run(["bash", str(wt / "downstream/n/run_n3.sh"), str(tmp_path / "n3"), model, "/c.jsonl",
                              "/e.jsonl"], capture_output=True, text=True, env=dict(env, N_VAL_ENV_QWEN3="/q3.yaml"),
                             timeout=60)
        assert res.returncode == 0, res.stdout + res.stderr
    calls = (tmp_path / "n3/calls").read_text().split()
    n3 = str(tmp_path / "n3")
    assert calls == [f"{n3}/e/B|/e.jsonl|B|12|0|robot|", f"{n3}/c/Q3RS|/c.jsonl|Q3RS|12|0|sphere|/q3.yaml",
                     f"{n3}/e/Q3RS|/e.jsonl|Q3RS|12|0|robot|/q3.yaml"]
    assert "tier c: B is not in it" in (tmp_path / "n3/n3_B.log").read_text()


def test_report_tiers_pairings_and_the_c_against_d_gap(tmp_path):
    import n3_report as NR

    def write(d, model, wins, ids):
        (d / model).mkdir(parents=True, exist_ok=True)
        (d / model / "episodes.jsonl").write_text("".join(json.dumps({"instance_id": i, "success": i in wins}) + "\n"
                                                          for i in ids))

    ids = [f"I{k}" for k in range(10)]
    n3 = tmp_path / "n3"
    write(n3 / "c", "S", ids[:2], ids[:8])  # tier c: only 8 instances solvable under the sphere
    write(n3 / "c", "RS", ids[:5], ids[:8])
    write(n3 / "e", "B", [], ids)
    write(n3 / "e", "S", ids[:3], ids)
    d = tmp_path / "n1"
    write(d, "S", ids[:4], ids)
    write(d, "RS", ids[:4] + ["I9"], ids)
    a = {"S": {f"t{k}": k < 4 for k in range(10)}, "RS": {f"t{k}": k < 7 for k in range(10)}}
    rep = NR.build(n3, exclude={"I9"} - {"I9"}, a=a, d_dirs=[d])
    t = rep["tiers"]
    assert t["c"]["models"]["RS"]["k"] == 5 and t["c"]["pairs"]["RS-S"]["diff"] == pytest.approx(3 / 8)
    assert t["d"]["pairs"]["RS-S"]["diff"] == pytest.approx(0.1) and t["a"]["pairs"]["RS-S"]["diff"] == pytest.approx(0.3)
    assert t["d_on_c"]["models"]["RS"]["n"] == 8 and t["d_on_c"]["pairs"]["RS-S"]["diff"] == 0  # I9 is not in c
    g = rep["across"]["RS-S"]["c_vs_d_same_instances"]
    assert g["n"] == 8 and g["gap"] == pytest.approx(0.375) and g["low"] <= g["gap"] <= g["high"]
    assert "a_vs_c" in rep["across"]["RS-S"] and "S-B" in t["e"]["pairs"]
    md = NR.markdown(rep)
    assert "## (c) RoboCasa + 上游身体" in md and "| RS-S |" in md
    ex = NR.build(n3, exclude={"I0"}, a=a, d_dirs=[d])
    assert ex["tiers"]["c"]["models"]["S"]["n"] == 7 and ex["excluded"] == ["I0"]
