"""N series smoke (downstream/n, 2026-10-30): the upstream actions on base poses, camera A's upstream pose, the
upstream judge on a hand-made left/right scene (needs the main checkout), the segmentation shares, the breadth-first
search and the instance choice on a stand-in scene; 2026-10-31: the robot left out of camera A's renders, and the job
script's per-part time limit and comparison views."""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
sys.path.insert(0, str(ROOT / "tests"))
import n_env as N  # noqa: E402
import n_smoke as S  # noqa: E402
from test_n_gen import isolate_main_modules  # noqa: E402


@pytest.fixture(autouse=True)
def isolated():  # the main checkout's vagen (n_env) and the worktree's (other tests) kept apart
    yield from isolate_main_modules()


def test_actions_follow_the_heading():
    p = N.apply_action((0.0, 0.0, 0.0), "move_forward")
    assert p == pytest.approx((0.3, 0.0, 0.0))
    assert N.apply_action((0.0, 0.0, 0.0), "move_left") == pytest.approx((0.0, 0.3, 0.0))
    assert N.apply_action((0.0, 0.0, 0.0), "move_right") == pytest.approx((0.0, -0.3, 0.0))
    up = (1.0, 1.0, math.pi / 2)
    assert N.apply_action(up, "move_forward") == pytest.approx((1.0, 1.3, math.pi / 2))
    assert N.apply_action(up, "move_left") == pytest.approx((0.7, 1.0, math.pi / 2))
    assert N.apply_action(up, "turn_left")[2] == pytest.approx(math.pi / 2 + math.radians(20))
    with pytest.raises(ValueError):
        N.apply_action(up, "done")


def test_pose_keys_merge_nearby_poses_and_wrap_headings():
    assert N.pose_key((1.0, 2.0, 0.0)) == N.pose_key((1.01, 2.01, math.radians(1)))
    assert N.pose_key((0, 0, 0)) == N.pose_key((0, 0, 2 * math.pi))
    assert N.pose_key((0, 0, 0)) != N.pose_key((0, 0, math.radians(20)))


def test_camera_a_is_the_upstream_level_camera():
    c2w = N.c2w_of((2.0, 3.0, math.radians(30)))
    R = c2w[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    assert np.allclose(c2w[:3, 2], [math.cos(math.radians(30)), math.sin(math.radians(30)), 0])  # forward
    assert np.allclose(c2w[:3, 1], [0, 0, -1]) and np.allclose(c2w[:3, 3], [2, 3, 1.5])  # down, height 1.5 m
    assert N.FOVY_DEG == pytest.approx(73.7398, abs=1e-3)  # 90 degrees across 320 px at 320 x 240


def box(key, name, center, half=0.2):
    c = np.asarray(center, float)
    return {"key": key, "name": name, "kind": "object", "bbox_min": list(c - half), "bbox_max": list(c + half),
            "center": list(c)}


@pytest.mark.skipif(not (N.MAIN / "vagen/envs/active_spatial/canonical_task_metrics.py").is_file(),
                    reason="needs the main checkout")
def test_upstream_judge_reads_left_and_right_as_the_image_does():
    a, b = box("a", "mug", (3.0, 1.0, 1.2)), box("b", "kettle", (3.0, -1.0, 1.2))  # facing +x: +y is image left
    left = N.box_judge(N.projective_item("s", a, b, "left", (0, 0, 0)), (0.0, 0.0, 0.0))
    right = N.box_judge(N.projective_item("s", a, b, "right", (0, 0, 0)), (0.0, 0.0, 0.0))
    assert left["success"] and not right["success"]
    assert left["lateral"]["ok"] and left["metric"]["relation_margin_px"] >= 12
    behind = N.box_judge(N.projective_item("s", a, b, "left", (0, 0, 0)), (0.0, 0.0, math.pi))
    assert not behind["success"]  # turned around: neither is in front
    item = N.projective_item("s", a, b, "left", (0, 0, 0))
    assert item["task_description"] == "Move so that the mug appears to the left of the kettle in your view."
    assert item["init_camera"]["intrinsics"] == N.NATIVE_K and item["target_region"]["params"]["relation"] == "left"


def test_visible_shares_from_segmentation():
    seg = np.full((10, 10), -1)
    seg[:2, :] = 5
    seg[5, :5] = 7
    assert N.visible_fractions(seg, {"a": {5}, "b": {7, 8}, "c": {9}}) == {"a": 0.2, "b": 0.05, "c": 0.0}


class FakeScene:
    """Free floor for x < 0.65; a wall beyond (x >= 0.65 collides)."""

    def __init__(self):
        self.start, self.pose, self.task, self.seed = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "T", 1
        self.entities = [box("a", "mug", (1.0, 1.0, 1.0)), box("b", "kettle", (1.0, -1.0, 1.0))]

    def set_pose(self, pose):
        self.pose = pose

    def collision(self):
        return "wall" if self.pose[0] >= 0.65 else None

    def judge(self, item, pose, render=True):
        ok = pose[0] > 0.5 and abs(pose[2]) < 1e-6
        return {"success": ok, "metric": {"gates": {}}, "lateral": {}, "visible": {"a": 0.01, "b": 0.01}}


def test_breadth_first_search_stops_at_walls_and_depth():
    sc = FakeScene()
    r = S.reachable(sc, max_depth=3, max_states=10000)
    st = r["states"]
    assert all(p[0] < 0.65 for p, *_ in st.values()) and r["blocked"] > 0
    assert max(d for _, d, _, _ in st.values()) == 3
    k = N.pose_key((0.6, 0.0, 0.0))
    assert S.path_to(st, k) == ["move_forward", "move_forward"]
    assert S.reachable(sc, max_depth=3, max_states=5)["capped"]


def test_instance_needs_two_actions_and_a_start_that_fails(monkeypatch):
    sc = FakeScene()
    st = S.reachable(sc, max_depth=3)["states"]
    monkeypatch.setattr(N, "box_judge", lambda item, pose: {"success": pose[0] > 0.5 and abs(pose[2]) < 1e-6,
                                                            "metric": {"gates": {}}, "lateral": {}})
    inst = S.find_instance(sc, st, random.Random(0))
    assert inst is not None and inst["depth"] == 2 and inst["path"] == ["move_forward", "move_forward"]
    assert inst["success_pose"][0] == pytest.approx(0.6) and inst["start_judge"]["success"] is False


def test_robot_hidden_moves_the_robot_out_of_the_drawn_groups_and_back():
    """2026-10-31: the first smoke showed the arm in the middle of every camera A frame (camera and arm ride on the
    same base); hide_robot renders without the robot's geoms (they still collide)."""
    from types import SimpleNamespace

    class Sim:
        _render_context_offscreen = None

        def render(self, **kw):  # robosuite makes the offscreen context on the first render
            if self._render_context_offscreen is None:
                self._render_context_offscreen = SimpleNamespace(
                    vopt=SimpleNamespace(geomgroup=np.array([1, 1, 1, 0, 0, 1], dtype=np.uint8)))

    sim, model = Sim(), SimpleNamespace(geom_group=np.array([0, 1, 1, 0, 2]))
    ids = np.array([1, 3])
    with N.robot_hidden(sim, model, ids):
        assert model.geom_group.tolist() == [0, 5, 1, 5, 2]
        assert sim._render_context_offscreen.vopt.geomgroup[N.HIDE_GROUP] == 0
    assert model.geom_group.tolist() == [0, 1, 1, 0, 2]
    assert sim._render_context_offscreen.vopt.geomgroup[N.HIDE_GROUP] == 1
    with pytest.raises(RuntimeError):
        with N.robot_hidden(sim, model, ids):
            raise RuntimeError("render failed")
    assert model.geom_group.tolist() == [0, 1, 1, 0, 2]  # put back after an error too
    with N.robot_hidden(sim, model, ids, hide=False):
        assert model.geom_group.tolist() == [0, 1, 1, 0, 2]


STUB = r"""import json, sys, time
from pathlib import Path
cmd = sys.argv[1]
out = Path(sys.argv[sys.argv.index("--out") + 1])
with open(out.parent / "calls.jsonl" if cmd != "generate" else out / "calls.jsonl", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
if cmd == "generate":
    if sys.argv[sys.argv.index("--part") + 1].startswith("2/"):
        time.sleep(60)  # a hung part
    (out / "instances.jsonl").write_text(json.dumps({"task": "T", "seed": 1, "instance": {}}) + "\n")
else:
    out.write_text("<p>page</p>")
"""


def test_smoke_job_kills_a_hung_part_hides_the_robot_and_makes_the_views(tmp_path):
    import os
    import shutil
    import subprocess

    from test_run_probe_episode import SIM_ENV

    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/eval").mkdir(parents=True)
    shutil.copy(ROOT / "n/run_n_smoke.sh", wt / "downstream/n/run_n_smoke.sh")
    (wt / "downstream/n/n_smoke.py").write_text(STUB)
    (wt / "downstream/eval/sim_env.sh").write_text(SIM_ENV)
    (tmp_path / "x0.jsonl").write_text("")
    out = tmp_path / "out"
    env = dict(os.environ, N_X0=str(tmp_path / "x0.jsonl"), HIDE_ROBOT="1", PART_TIMEOUT="3",
               VIEWS_SCENES="CoffeeSetupMug:1012 PreSoakPan:1013")
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n_smoke.sh"), str(out), "3", "1"], capture_output=True,
                         text=True, env=env, timeout=50)
    assert res.returncode == 0, res.stdout + res.stderr
    log = (out / "n_smoke.log").read_text()
    assert "part 2 rc=137" in log and "part 1 rc=0" in log and "page rc=0" in log and "views rc=0" in log
    gen = json.loads((out / "gen/part1/calls.jsonl").read_text().splitlines()[0])
    assert "--hide-robot" in gen and gen[gen.index("--x0") + 1] == str(tmp_path / "x0.jsonl")
    calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
    views = next(c for c in calls if c[0] == "views")
    assert views[views.index("--scenes") + 1: views.index("--out")] == ["CoffeeSetupMug:1012", "PreSoakPan:1013"]
    assert (out / "page.html").is_file() and (out / "views.html").is_file()
