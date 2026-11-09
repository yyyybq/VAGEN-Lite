"""WP-E3VS-A tests: geometry vs the official camera_geometry, lattice BFS, floors, download script, docs.

Login node, 1 core:  OMP_NUM_THREADS=1 ... nice -n 19 PY_EVAL -B -m pytest -q -p no:cacheprovider
vpd/tests/test_e3vs_geometry.py

The official module is loaded read-only by file path from the E3VS checkout (no package __init__, which would
import open3d/gsplat; no bytecode written).  Checkout: $VPD_E3VS_REPO, else paths.E3VS_REPO, else the
earlier-fetched read-only copy plan/login_tmp/vpd_e3vs/repo (reading it is allowed, DECISIONS section D).
Test outputs go to eval_external_assets/vpd/scratch/e3vs-a/pytest_<rand>/ (VPD_E3VS_TEST_DIR overrides) and
are removed at teardown.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vpd import paths  # noqa: E402
import vpd.e3vs as E  # noqa: E402
from vpd.e3vs import bfs_demos as B  # noqa: E402
from vpd.e3vs import floors as F  # noqa: E402
from vpd.e3vs import geometry as G  # noqa: E402

E3VS_DIR = paths.VPD_PKG / "e3vs"
TOL = 1e-9


def _repo() -> Path | None:
    for p in (os.environ.get("VPD_E3VS_REPO"), paths.E3VS_REPO, E.LOGIN_TMP_REPO):
        if p and (Path(p) / "dataset").is_dir():
            return Path(p)
    return None


REPO = _repo()
needs_repo = pytest.mark.skipif(REPO is None, reason="no E3VS checkout on this machine")


@pytest.fixture(scope="module")
def official():
    """The official camera_geometry module (torch + scipy), loaded by path, read-only."""
    if REPO is None:
        pytest.skip("no E3VS checkout")
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    torch.set_num_threads(1)
    src = REPO / "e3vs" / "environments" / "camera_geometry.py"
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("_e3vs_official_camera_geometry", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def episodes():
    if REPO is None:
        pytest.skip("no E3VS checkout")
    return {sp: E.load_episodes(sp, REPO) for sp in E.SPLITS}


@pytest.fixture(scope="module")
def outdir():
    base = os.environ.get("VPD_E3VS_TEST_DIR")
    d = Path(base) if base else paths.scratch("e3vs-a") / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    if not base:
        shutil.rmtree(d, ignore_errors=True)


def _random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _random_poses(rng, episodes, n_ep=200, n_free=100) -> list[np.ndarray]:
    starts = [ep["start_c2w"] for sp in ("val", "test") for ep in episodes[sp]]
    out = [starts[i] for i in rng.choice(len(starts), size=n_ep, replace=False)]
    for _ in range(n_free):  # arbitrary orientations, including rolled cameras
        c = np.eye(4)
        c[:3, :3] = _random_rotation(rng)
        c[:3, 3] = rng.uniform(-5, 5, size=3)
        out.append(c)
    down = np.eye(4)  # straight down: planar_forward degenerates to 0
    down[:3, :3] = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    out.append(down)
    return out


# ------------------------------------------------------------------------------------------------------------
# repo and episodes
# ------------------------------------------------------------------------------------------------------------
@needs_repo
def test_repo_is_pinned():
    info = E.check_repo(REPO)
    assert info["commit"] in (None, E.E3VS_COMMIT)
    assert E.repo_commit(REPO) == E.E3VS_COMMIT  # the copy on the cluster is a git checkout
    assert info["episode_md5"] == E.EPISODE_MD5


def test_repo_dir_has_no_silent_fallback(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("VPD_E3VS_REPO", str(paths.scratch("e3vs-a") / "does_not_exist"))
    with pytest.raises(FileNotFoundError):
        E.repo_dir()


def test_action_tables():
    assert len(E.ACTIONS) == 11 and E.ACTIONS[-1] == "stop"
    assert set(E.SEARCH_ORDER) == set(E.ACTIONS) - {"stop"} and len(E.SEARCH_ORDER) == 10
    assert E.TRANSLATIONS == {a for a in E.ACTIONS if a.startswith("move_")}


@needs_repo
def test_episode_counts_and_poses(episodes):
    for sp in E.SPLITS:
        eps = episodes[sp]
        assert len(eps) == E.EPISODE_COUNTS[sp]
        assert len({e["scene_id"] for e in eps}) == E.SCENE_COUNTS[sp]
        for e in eps:
            assert e["start_viewpoint"] in e["unans"]
            assert np.allclose(e["unans"][e["start_viewpoint"]], e["start_c2w"], atol=1e-12)
            assert len(e["goals"]) >= 1
            for c in [e["start_c2w"], *e["goals"].values(), *e["unans"].values()]:
                assert G.roll_of(c) < 1e-9  # every E3VS pose is roll-free (survey frame_probe)
    scenes = {sp: {e["scene_id"] for e in episodes[sp]} for sp in E.SPLITS}
    assert not (scenes["train"] & scenes["val"]) and not (scenes["train"] & scenes["test"])
    assert not (scenes["val"] & scenes["test"])


# ------------------------------------------------------------------------------------------------------------
# geometry vs official
# ------------------------------------------------------------------------------------------------------------
def test_rotvec_matches_scipy():
    Rot = pytest.importorskip("scipy.spatial.transform").Rotation
    rng = np.random.default_rng(1)
    vs = list(rng.normal(size=(200, 3))) + list(rng.normal(size=(50, 3)) * 1e-4) + [np.zeros(3)]
    for v in vs:
        assert np.abs(G.rotvec_matrix(v) - Rot.from_rotvec(v).as_matrix()).max() < 1e-12


@needs_repo
def test_apply_action_matches_official(official, episodes):
    rng = np.random.default_rng(0)
    poses = _random_poses(rng, episodes)
    worst = 0.0
    n_steps = 0
    for i in range(1000):
        coord = "world" if i % 10 == 9 else "camera"
        a_ = poses[rng.integers(len(poses))].copy()
        b_ = a_.copy()
        for _ in range(int(rng.integers(1, 26))):
            act = E.ACTIONS[rng.integers(len(E.ACTIONS))]
            a_ = G.apply_action(a_, act, 0.25, 30.0, coord)
            b_ = official.apply_action(b_, act, action_step_size=0.25, angle_step_size=30.0, action_coordinate=coord)
            worst = max(worst, float(np.abs(a_ - b_).max()))
            n_steps += 1
    assert n_steps > 5000
    assert worst < TOL, worst
    # no-op cases mirror the official code; unknown names raise (official: silent no-op)
    c = poses[0]
    for a in (None, "", "stop"):
        assert np.array_equal(G.apply_action(c, a), official.apply_action(
            c, a, action_step_size=0.25, angle_step_size=30.0, action_coordinate="camera"))
    with pytest.raises(ValueError):
        G.apply_action(c, "jump")


@needs_repo
def test_directions_match_official(official, episodes):
    rng = np.random.default_rng(2)
    for c in _random_poses(rng, episodes, n_ep=100, n_free=100):
        assert np.abs(G.view_forward(c) - official.view_forward(c)).max() < TOL
        assert np.abs(G.planar_forward(c) - official.planar_forward(c)).max() < TOL
        assert np.abs(G.planar_right(c) - official.planar_right(c)).max() < TOL
        for coord in ("camera", "world"):
            f1, r1 = G.movement_directions(c, coord)
            f2, r2 = official.movement_directions(c, coord)
            assert np.abs(f1 - f2).max() < TOL and np.abs(r1 - r2).max() < TOL


@needs_repo
def test_viewmat_and_K_match_official(official, episodes):
    rng = np.random.default_rng(3)
    for c in _random_poses(rng, episodes, n_ep=50, n_free=50):
        for W, H in ((512, 512), (640, 480)):
            vm, K = G.viewmat_and_K(c, W, H)
            vm_o, K_o = official.viewmat_and_K(c, W, H)
            vm_o, K_o = vm_o.numpy().astype(np.float64), K_o.numpy().astype(np.float64)
            # official returns float32 tensors of inv(c2w): equal after the same cast
            assert np.abs(vm.astype(np.float32).astype(np.float64) - vm_o).max() < TOL
            assert np.abs(K - K_o).max() < TOL
            assert np.abs(vm @ c - np.eye(4)).max() < 1e-9
    vm, K = G.viewmat_and_K(np.eye(4))
    assert K[0, 0] == 256.0 and K[0, 2] == 256.0 and K[1, 2] == 256.0  # 90 deg FOV at 512


# ------------------------------------------------------------------------------------------------------------
# success
# ------------------------------------------------------------------------------------------------------------
def _ref_success(c, goals, official, pos=0.25, ang=20.0, margin=1e-7):
    """Reference on the official view_forward; returns None near a threshold (ambiguous in float)."""
    out = False
    for g in goals:
        d = float(np.linalg.norm(c[:3, 3] - g[:3, 3]))
        a = math.degrees(math.acos(float(np.clip(official.view_forward(c) @ official.view_forward(g), -1, 1))))
        if abs(d - pos) < margin or abs(a - ang) < margin:
            return None
        out = out or (d <= pos and a <= ang)
    return out


@needs_repo
def test_success_matches_reference(official, episodes):
    rng = np.random.default_rng(4)
    n_true = n_checked = 0
    eps = episodes["val"] + episodes["test"]
    for i in range(3000):
        ep = eps[rng.integers(len(eps))]
        goals = list(ep["goals"].values())
        c = goals[rng.integers(len(goals))].copy()
        c[:3, 3] += rng.normal(scale=0.17, size=3)
        c[:3, :3] = G.rotvec_matrix(rng.normal(scale=0.3, size=3)) @ c[:3, :3]
        ref = _ref_success(c, goals, official)
        if ref is None:
            continue
        n_checked += 1
        n_true += ref
        assert G.success(c, goals) == ref
        assert G.success(c, ep["goals"]) == ref  # dict input
    assert n_checked > 2900 and 300 < n_true < n_checked - 300  # both outcomes well represented


def test_success_boundaries():
    g = np.eye(4)
    g[:3, :3] = G.rotvec_matrix([0.0, 0.0, 0.7]) @ np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
    c = g.copy()
    c[0, 3] += 0.25  # exactly at the position tolerance: inclusive
    assert G.success(c, [g])
    c[0, 3] += 1e-4
    assert not G.success(c, [g])
    for deg, ok in ((19.9, True), (20.1, False)):
        c = g.copy()
        c[:3, :3] = G.rotvec_matrix(np.radians(deg) * np.array([0.0, 0.0, 1.0])) @ g[:3, :3]
        assert G.success(c, [g]) is ok
        assert abs(G.view_angle_deg(c, g) - deg) < 1e-9
    assert not G.success(g, np.zeros((0, 4, 4)))
    i, d, a = G.nearest_goal(g, [g, c])
    assert i == 0 and d == 0.0 and a == 0.0


@needs_repo
def test_no_start_is_a_success(episodes):
    for sp in E.SPLITS:
        assert sum(G.success(e["start_c2w"], e["goals"]) for e in episodes[sp]) == 0


# ------------------------------------------------------------------------------------------------------------
# lattice and BFS
# ------------------------------------------------------------------------------------------------------------
@needs_repo
def test_lattice_matches_geometry(episodes):
    rng = np.random.default_rng(5)
    eps = episodes["train"] + episodes["test"]
    worst = 0.0
    for _ in range(300):
        ep = eps[rng.integers(len(eps))]
        lat = B.Lattice(ep["start_c2w"])
        st, c = lat.start, ep["start_c2w"].copy()
        for _ in range(int(rng.integers(1, 17))):
            a = E.SEARCH_ORDER[rng.integers(10)]
            t = lat.step(st, a)
            if t is None:  # pitch limit of the lattice only
                continue
            st, c = t, G.apply_action(c, a)
            worst = max(worst, float(np.abs(np.array(st[:3]) - c[:3, 3]).max()),
                        float(np.abs(np.array(lat.forward(st)) - G.view_forward(c)).max()))
    assert worst < TOL, worst


@needs_repo
def test_lattice_never_reaches_vertical(episodes):
    """+30 deg starts are stored as 29.99999999999998 deg: two look_ups must not reach a vertical view."""
    ep = next(e for e in episodes["train"] if abs(math.degrees(G.yaw_pitch(e["start_c2w"])[1]) - 30) < 1e-6)
    lat = B.Lattice(ep["start_c2w"])
    st = lat.step(lat.start, "look_up")
    assert st is not None and lat.step(st, "look_up") is None
    ep = next(e for e in episodes["train"] if abs(math.degrees(G.yaw_pitch(e["start_c2w"])[1]) + 45) < 1e-6)
    lat = B.Lattice(ep["start_c2w"])
    st = lat.step(lat.start, "look_down")  # -75 is allowed, -105 is not
    assert st is not None and lat.step(st, "look_down") is None
    # official semantics at a vertical view: sideways moves are no-ops (planar_forward = 0)
    up = _level_pose()
    up = G.apply_action(G.apply_action(G.apply_action(up, "look_up"), "look_up"), "look_up")
    assert abs(G.view_forward(up)[2] - 1.0) < 1e-12
    assert np.array_equal(G.apply_action(up, "move_left")[:3, 3], up[:3, 3])


def test_state_key_is_injective_on_neighbours():
    seen = {}
    for x in (-0.004, 0.0, 0.006, 1.0):
        for y in (-3.0, 0.0, 2.5):
            for z in (0.0, 1.25):
                for yi in range(-13, 14):
                    for pj in range(-5, 6):
                        k = B.state_key((x, y, z, yi, pj))
                        tup = (round(x * 100), round(y * 100), round(z * 100), yi % 12, pj)
                        assert seen.setdefault(k, tup) == tup


def _level_pose(pos=(0.0, 0.0, 1.0), yaw_deg=0.0) -> np.ndarray:
    """Roll-free camera looking along +x rotated by yaw (image down = -Z)."""
    yaw = math.radians(yaw_deg)
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    r = np.cross(f, [0.0, 0.0, 1.0])
    d = np.cross(f, r)
    c = np.eye(4)
    c[:3, 0], c[:3, 1], c[:3, 2], c[:3, 3] = r, d, f, pos
    return c


def _wall(x0=0.35, half=0.3, zc=1.0):
    """Collider: a square wall at x = x0, |y| < half, |z - zc| < half."""
    def collides(p0, p1):
        if (p0[0] - x0) * (p1[0] - x0) > 0 or p0[0] == p1[0]:
            return False
        t = (x0 - p0[0]) / (p1[0] - p0[0])
        y, z = p0[1] + t * (p1[1] - p0[1]), p0[2] + t * (p1[2] - p0[2])
        return abs(y) < half and abs(z - zc) < half
    return collides


def test_bfs_synthetic_path_and_tie_order():
    s = _level_pose()
    goal = G.apply_action(G.apply_action(s, "look_down"), "move_forward")
    goal[:3, 3] = s[:3, 3] + [0.6, 0.0, 0.0]  # 0.6 m ahead, looking 30 deg down
    r = B.bfs(s, {0: goal})
    assert r["status"] == "reached" and r["opt_len"] == 3
    assert r["actions"] == ["move_forward", "move_forward", "look_down"]  # first in the fixed order
    ep = {"split": "val", "episode_index": 7, "scene_id": "synthetic", "question_type": "t", "question": "q",
          "instance_id": 1, "goals": {0: goal}, "unans": {3: s}, "start_viewpoint": 3, "start_c2w": s}
    rec = B.solve_start(ep, 3, s, None, False, 16, B.STATE_CAP)
    assert rec["actions"][-1] == "stop" and len(rec["poses"]) == len(rec["actions"]) + 1
    assert rec["goal_viewpoint"] == 0 and rec["replay_dev"] < TOL
    assert G.success(np.array(rec["poses"][-1]), [goal])


def test_bfs_collision_detour_unreachable_and_cap():
    s = _level_pose()
    goal = G.apply_action(s, "look_down")
    goal[:3, 3] = s[:3, 3] + [0.6, 0.0, 0.0]
    wall = _wall()
    r = B.bfs(s, {0: goal}, collides=wall)
    assert r["status"] == "reached" and r["opt_len"] > 3
    poses, hits = G.rollout(s, r["actions"] + ["stop"], collides=wall)
    assert not any(hits) and G.success(poses[-1], [goal])
    # the same actions without the detour check would collide: the straight path is blocked
    _, hits_straight = G.rollout(s, ["move_forward", "move_forward", "look_down"], collides=wall)
    assert hits_straight[1]
    blocked = B.bfs(s, {0: goal}, collides=lambda p0, p1: True)
    assert blocked["status"] == "unreachable" and blocked["n_states"] < 200  # rotations only
    capped = B.bfs(s, {0: goal}, collides=wall, cap=50)
    assert capped["status"] == "capped" and capped["actions"] is None


def test_rollout_skips_colliding_moves_only():
    s = _level_pose()
    poses, hits = G.rollout(s, ["move_forward", "move_forward", "turn_left", "stop"], collides=_wall())
    assert hits == [False, True, False, False]
    assert np.allclose(poses[2], poses[1]) and not np.allclose(poses[3], poses[2])
    calls = []
    G.rollout(s, ["turn_left", "look_up", "stop"], collides=lambda a, b: calls.append(1) or True)
    assert calls == []  # rotations and stop never ask the collider


@needs_repo
def test_bfs_run_outputs(outdir):
    out = outdir / "bfs_val3"
    s = B.run("val", out=out, limit=3, cap=200_000, repo=REPO, quiet=True)
    assert s["n_episodes"] == 3 and s["reached"] + s["capped"] + s["unreachable"] == 3
    recs = [json.loads(x) for x in (out / "records.jsonl").read_text().splitlines()]
    assert len(recs) == 3 and not any(r["extra"] for r in recs)
    for r in recs:
        if r["status"] == "reached":
            assert r["actions"][-1] == "stop" and len(r["actions"]) == r["opt_len"] + 1
            assert len(r["poses"]) == len(r["actions"]) + 1 and r["replay_dev"] < TOL
    opt = json.loads((out / "opt_len.json").read_text())
    assert set(opt) == {str(r["episode_index"]) for r in recs}
    capped = json.loads((out / "capped.json").read_text())
    assert len(capped) == s["capped"] + s["unreachable"]
    summ = json.loads((out / "summary.json").read_text())
    assert summ["inputs"]["episode_md5"] == {"val": E.EPISODE_MD5["val"]}
    assert not (out / "demo_pool.jsonl").exists()  # train only
    # the login_tmp copy must never be written to OUT/e3vs/bfs by default
    if REPO.resolve() == E.LOGIN_TMP_REPO.resolve():
        with pytest.raises(SystemExit):
            B.run("val", out=None, limit=1, repo=REPO, quiet=True)


@needs_repo
def test_bfs_extra_starts_and_procs(outdir, episodes):
    out1, out2 = outdir / "bfs_extra_p1", outdir / "bfs_extra_p2"
    kw = dict(limit=2, cap=20_000, repo=REPO, quiet=True, extra=True)
    B.run("test", out=out1, procs=1, **kw)
    B.run("test", out=out2, procs=2, **kw)
    r1 = [json.loads(x) for x in (out1 / "records.jsonl").read_text().splitlines()]
    r2 = [json.loads(x) for x in (out2 / "records.jsonl").read_text().splitlines()]
    assert r1 == r2  # process count does not change results or order
    n_extra = sum(len(e["unans"]) - 1 for e in episodes["test"][:2])
    assert sum(r["extra"] for r in r1) == n_extra and sum(not r["extra"] for r in r1) == 2


def test_collider_factory_spec():
    assert B.load_collider_factory(None) is None
    f = B.load_collider_factory("vpd.e3vs.geometry:success")
    assert f is G.success
    with pytest.raises(ValueError):
        B.load_collider_factory("vpd.e3vs.geometry")


# ------------------------------------------------------------------------------------------------------------
# floors
# ------------------------------------------------------------------------------------------------------------
def test_floor_sequences_and_choice():
    seqs = F.sequences()
    assert len(seqs) == 1 + 10 + 100 + 1000 and seqs[0] == ()
    idx = {s: i for i, s in enumerate(seqs)}
    for s in seqs[1:]:
        assert idx[s[:-1]] < idx[s]  # prefixes first (episode_hits relies on it)
    s = _level_pose()
    goal = G.apply_action(s, "look_down")
    goal[2, 3] += 0.3  # move_up lands 0.05 m away; staying put is 0.3 m away
    ep = {"start_c2w": s, "goals": {0: goal}}
    h = F.episode_hits(ep, seqs)
    assert h[idx[("move_up", "look_down")]] and h[idx[("look_down", "move_up")]] and not h[0]
    table = np.stack([h, h])
    assert seqs[F.choose(table)] == ("move_up", "look_down")  # tie -> earlier in the fixed order
    up_blocked = lambda p0, p1: p1[2] > p0[2] + 1e-9  # noqa: E731
    h2 = F.episode_hits(ep, seqs, up_blocked)
    assert not h2[idx[("move_up", "look_down")]]


def test_wilson():
    lo, hi = F.wilson(53, 377)
    assert lo < 53 / 377 < hi and abs(lo - 0.1094) < 5e-4 and abs(hi - 0.1790) < 5e-4
    assert F.wilson(0, 10)[0] == 0.0 and F.wilson(10, 10)[1] == pytest.approx(1.0, abs=1e-12)
    assert F.wilson(0, 0) == (0.0, 1.0)


@needs_repo
def test_floors_run_small(outdir):
    out = outdir / "floors_small.json"
    res = F.run(out=out, repo=REPO, limit=40, quiet=True)
    assert json.loads(out.read_text())["n_sequences"] == 1111
    assert res["start_success"] == {"train": 0, "val": 0, "test": 0}
    assert res["gate_floor"]["seq"] == res["chosen"]["seq"] and res["gate_floor"]["n"] == 40
    assert set(res["per_episode"]) == {"val", "test"}


# ------------------------------------------------------------------------------------------------------------
# download script and docs
# ------------------------------------------------------------------------------------------------------------
SCRIPT = E3VS_DIR / "download_splats.sh"


def test_download_script_syntax():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_download_script_dry_run(outdir):
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    env.update(DRY_RUN="1", E3VS_DATA_DIR=str(outdir / "dl_data"))
    p = subprocess.run(["bash", str(SCRIPT), "splats", "all"], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    plan = [ln.split() for ln in p.stdout.splitlines() if ln.startswith("PLAN ")]
    assert len(plan) == 99
    splits = [x[2] for x in plan]
    assert set(splits[:31]) == {"val", "test"} and set(splits[31:]) == {"train"}
    assert sum(int(x[4]) for x in plan) == pytest.approx(35.05e9, rel=2e-3)
    assert "hf-mirror.com" in p.stdout and "--max-workers 1" in p.stdout
    assert not (outdir / "dl_data").exists()  # dry run writes nothing
    p2 = subprocess.run(["bash", str(SCRIPT), "splats", "valtest"], env=env, capture_output=True, text=True,
                        timeout=60)
    assert p2.returncode == 0 and sum(ln.startswith("PLAN ") for ln in p2.stdout.splitlines()) == 31
    p3 = subprocess.run(["bash", str(SCRIPT), "bogus"], env=env, capture_output=True, text=True, timeout=60)
    assert p3.returncode != 0


@needs_repo
def test_download_table_matches_episodes(episodes):
    rows = re.findall(r"^(train|val|test) ([0-9a-f]{9,10}) (\d+)$", SCRIPT.read_text(), flags=re.M)
    table = {sc: sp for sp, sc, _ in rows}
    assert len(table) == 99
    for sp in E.SPLITS:
        assert {e["scene_id"] for e in episodes[sp]} == {s for s, x in table.items() if x == sp}


def test_download_loop_children_keep_off_stdin():
    """Diagnosis 2026-11-09 B6: the per-scene loop reads the scene list on stdin (`done < <(scenes_for ...)`), so
    the hf download and sha256sum calls inside it get </dev/null and can never swallow the remaining scenes."""
    t = SCRIPT.read_text()
    body = t.split('n=$(scenes_for "$set" | wc -l)', 1)[1].split('verify_all "$set"', 1)[0]
    assert body.count('"$hf" download') == 1 and body.count("sha256sum") == 1
    calls, cur = [], ""
    for ln in body.splitlines():  # join backslash continuations, keep the lines that run hf / sha256sum
        cur += ln.rstrip("\\").strip() + " "
        if not ln.rstrip().endswith("\\"):
            if '"$hf" download' in cur or "sha256sum" in cur:
                calls.append(cur)
            cur = ""
    assert len(calls) == 2 and all("< /dev/null" in c for c in calls), calls
    assert "done < <(scenes_for \"$set\")" in body


def test_download_script_token_hygiene():
    t = SCRIPT.read_text()
    assert "/mnt/umm/users/yinbaiqiao/.secrets/hf_token_e3vs" in t
    assert "set -x" not in t and "HF_TOKEN=" not in t.replace("unset HF_TOKEN", "")
    assert not re.search(r"(echo|printf)[^\n]*\$\(?\s*cat[^\n]*token", t)


def test_docs_present():
    us = (E3VS_DIR / "USER_STEPS.md").read_text()
    for s in ("https://huggingface.co/datasets/GaussianWorld/scene_splat_7k",
              "https://huggingface.co/datasets/GaussianWorld/scannetpp_v2_mcmc_3dgs",
              '(umask 077; read -rs T; printf %s "$T" > ~/.secrets/hf_token_e3vs; unset T)',
              "chmod 600 ~/.secrets/hf_token_e3vs", "hf-mirror"):
        assert s in us, s
    pr = (paths.VPD_PKG / "PREREG_E3VS_PILOT.md").read_text(encoding="utf-8")
    for s in ("bb7e603c", "1001", "Q3B", "Q3S", "E2", "0.25", "B8", "B12", "377"):
        assert s in pr, s
    # audit 2026-11-08: the registration names what summary.gate / the payloads enforce
    for s in ("第 16 节的偏差日志", '"size": {"longest_edge": 1024000', "[258]", "vpd.e3vs.occupancy:load_collider",
              "prep_state/floors", "prep_state/sft", "init_adapter.sha256", "adapter_sha256", "system 提示词 sha256",
              "--mode u,c", "--convention auto", "near_frac", "Grid.collides"):
        assert s in pr, s
    assert "第 15 节的偏差日志" not in pr and "## 16. 偏差日志" in pr
