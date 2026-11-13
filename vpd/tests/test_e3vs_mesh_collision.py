"""Tests of vpd.e3vs.mesh_collision: the official E3VS Collisions (mesh + Open3D ray casting) as the pilot's
collision model (PREREG_E3VS_PILOT section 16, user 2026-11-12).

open3d is not installed yet (pending the user's OK).  Two backends run the same synthetic scenes:
  - "stub": a minimal stand-in for the open3d API that metrics.py and mesh_collision use (io.read_triangle_mesh,
    t.geometry.TriangleMesh.from_legacy, t.geometry.RaycastingScene add_triangles / cast_rays /
    compute_distance, core.Tensor), ray-casting axis-aligned box surfaces in numpy.  It runs now and exercises
    the OFFICIAL metrics.py code (executed from the pinned checkout) and our wiring around it;
  - "open3d": the real library on the same boxes written as triangle meshes; skipped (pytest.importorskip) until
    open3d is importable (OUT/envs/e3vs-o3d on PYTHONPATH).

Login node, 1 core:  OMP_NUM_THREADS=1 ... nice -n 19 PY_Q3 -B -m pytest -q -p no:cacheprovider
vpd/tests/test_e3vs_mesh_collision.py   (pytest tmp dirs only; with real open3d, run it under `taskset -c <one
core>`: Embree's BVH build uses every CPU it sees)
"""

from __future__ import annotations

import ast
import json
import os
import re
import struct
import subprocess
import sys
import types
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vpd import paths  # noqa: E402
import vpd.e3vs as E  # noqa: E402
from vpd.e3vs import bfs_demos as B  # noqa: E402
from vpd.e3vs import floors as F  # noqa: E402
from vpd.e3vs import geometry as G  # noqa: E402
from vpd.e3vs import mesh_collision as MC  # noqa: E402
from vpd.e3vs import mesh_verify as MV  # noqa: E402
from vpd.e3vs import render as RD  # noqa: E402

JOBS = paths.VPD_PKG / "jobs"


def _repo() -> Path | None:
    for p in (os.environ.get("VPD_E3VS_REPO"), paths.E3VS_REPO):
        if p and (Path(p) / "dataset").is_dir() and E.repo_commit(Path(p)) == E.E3VS_COMMIT:
            return Path(p)
    return None


REPO = _repo()
needs_repo = pytest.mark.skipif(REPO is None, reason="no pinned E3VS git checkout on this machine")


def look_c2w(pos, fwd) -> np.ndarray:
    """Roll-free OpenCV c2w at pos looking along fwd (world Z up)."""
    f = np.asarray(fwd, float) / np.linalg.norm(fwd)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    c = np.eye(4)
    c[:3, 0], c[:3, 1], c[:3, 2], c[:3, 3] = x, np.cross(f, x), f, pos
    return c


# ------------------------------------------------------------------------------------------------------------
# synthetic scenes: box surfaces, written as triangle meshes (real open3d) and registered with the stub
# ------------------------------------------------------------------------------------------------------------
ROOM = ((0.0, 0.0, 0.0), (4.0, 4.0, 3.0))
PILLAR = ((1.5, 1.5, 0.0), (2.5, 2.5, 3.0))
_FACES = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6),
          (0, 6, 4), (1, 5, 7), (1, 7, 3)]


def write_box_mesh(path: Path, boxes) -> Path:
    """Binary little-endian PLY of the surfaces of axis-aligned boxes (12 triangles each)."""
    verts, faces = [], []
    for lo, hi in boxes:
        base = len(verts)
        for i in range(8):
            verts.append(tuple((hi if i >> (2 - k) & 1 else lo)[k] for k in range(3)))
        faces += [tuple(base + j for j in f) for f in _FACES]
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty float x\n"
           f"property float y\nproperty float z\nelement face {len(faces)}\n"
           "property list uchar int vertex_indices\nend_header\n").encode()
    body = b"".join(struct.pack("<fff", *v) for v in verts) + b"".join(struct.pack("<Biii", 3, *f) for f in faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hdr + body)
    MV.check_mesh(path)  # a valid mesh for mesh_verify
    return path


class _Tensor:
    def __init__(self, a, dtype=None):
        self.a = np.array(a, dtype=np.float32)

    def numpy(self):
        return self.a


def _slab(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (lo - o) * inv, (hi - o) * inv
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    return np.minimum(t1, t2).max(axis=-1), np.maximum(t1, t2).min(axis=-1)


class StubScene:
    """RaycastingScene stand-in over box surfaces (first hit along the ray; unsigned distance to the surface)."""

    REG: dict = {}

    def __init__(self, *a, **k):
        self.boxes = []

    def add_triangles(self, mesh):
        self.boxes = [(np.array(lo, float), np.array(hi, float)) for lo, hi in self.REG.get(mesh.path, ())]
        return 0

    def cast_rays(self, rays):
        r = np.asarray(rays.numpy(), dtype=np.float64)
        o, d = r[:, :3], r[:, 3:]
        t = np.full(len(r), np.inf)
        for lo, hi in self.boxes:
            tmin, tmax = _slab(o, d, lo, hi)
            th = np.where(tmin >= 0, tmin, np.where(tmax >= 0, tmax, np.inf))
            t = np.minimum(t, np.where(tmax >= tmin, th, np.inf))
        return {"t_hit": _Tensor(t)}

    def compute_distance(self, q):
        p = np.asarray(q.numpy(), dtype=np.float64)
        d = np.full(len(p), np.inf)
        for lo, hi in self.boxes:
            out = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)
            inside = np.minimum(p - lo, hi - p).min(axis=1)
            d = np.minimum(d, np.where(out > 0, out, np.maximum(inside, 0.0)))
        return _Tensor(d)


class _Legacy:
    def __init__(self, path):
        self.path = str(Path(path))

    def get_axis_aligned_bounding_box(self):
        bx = StubScene.REG.get(self.path, ())
        lo = np.min([b[0] for b in bx], axis=0) if bx else np.zeros(3)
        hi = np.max([b[1] for b in bx], axis=0) if bx else np.zeros(3)
        return types.SimpleNamespace(min_bound=lo, max_bound=hi)


def make_stub():
    m = types.ModuleType("open3d")
    m.__version__ = "stub"
    m.core = types.SimpleNamespace(Tensor=_Tensor, Dtype=types.SimpleNamespace(Float32="Float32"))
    m.io = types.SimpleNamespace(read_triangle_mesh=lambda p: _Legacy(p))
    m.t = types.SimpleNamespace(geometry=types.SimpleNamespace(
        TriangleMesh=types.SimpleNamespace(from_legacy=lambda legacy: legacy), RaycastingScene=StubScene))
    return m


@pytest.fixture(params=["stub", "open3d"])
def backend(request, monkeypatch):
    """Registers box scenes: backend.scene(path, boxes) writes the mesh (and tells the stub)."""
    monkeypatch.setattr(MC, "_OFFICIAL", {})
    monkeypatch.setattr(MC, "_MEMO", {})
    if request.param == "stub":
        monkeypatch.setitem(sys.modules, "open3d", make_stub())
        monkeypatch.setattr(StubScene, "REG", {})
    else:
        pytest.importorskip("open3d")

    def scene(path, boxes):
        write_box_mesh(Path(path), boxes)
        StubScene.REG[str(Path(path))] = [tuple(map(tuple, b)) for b in boxes]
        return Path(path)

    return types.SimpleNamespace(kind=request.param, scene=scene)


# ------------------------------------------------------------------------------------------------------------
# constants, the pinned official file, the open3d location
# ------------------------------------------------------------------------------------------------------------
def test_constants_and_job_paths():
    assert MC.TOLERANCE_M == MC.IN_OBSTACLE_M == 1e-4 and MC.NEAR_M == 0.05 and MC.VALID_MAX_FRAC == 0.01
    assert MC.MOVES == ("move_forward", "move_backward", "move_up", "move_down", "move_left", "move_right")
    assert MC.FACTORY == "vpd.e3vs.mesh_collision:load_collider" and MC.TAG == "mesh"
    assert B.load_collider_factory(MC.FACTORY) is MC.load_collider
    # open3d lives in OUT/envs/e3vs-o3d: a constant of this module (not of vpd/paths.py)
    assert MC.O3D_DIR == paths.OUT / "envs" / "e3vs-o3d"
    assert "e3vs-o3d" not in (paths.VPD_PKG / "paths.py").read_text()


def test_importing_the_code_does_not_import_open3d():
    code = ("import sys, vpd.e3vs.mesh_collision, vpd.e3vs.run, vpd.e3vs.summary, vpd.e3vs.bfs_demos, "
            "vpd.e3vs.floors; print('open3d' in sys.modules)")
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr


@needs_repo
def test_pinned_official_collision_code():
    """The constants match the pinned metrics.py: its sha256, the class, the loader calls, the default tolerance
    and the collision rule (read by ast, no import)."""
    src_path = REPO / MC.METRICS_REL
    assert E.sha256_file(src_path) == MC.METRICS_SHA256
    assert MC.official_facts(REPO)["commit"] == E.E3VS_COMMIT
    src = src_path.read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == MC.OFFICIAL_CLASS)
    fns = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    ctc = fns["check_trajectory_collision"]
    assert [a.arg for a in ctc.args.args] == ["self", "start_points", "end_points", "tolerance"]
    assert ast.literal_eval(ctc.args.defaults[-1]) == MC.TOLERANCE_M
    seg = ast.get_source_segment(src, ctc)
    assert "is_collided = (t_hit > tolerance) & (t_hit < (distances - tolerance))" in seg
    assert ".astype(np.float32)" in seg and "unit_directions = directions / (distances[:, np.newaxis] + 1e-8)" in seg
    load = ast.get_source_segment(src, fns["load_scene_mesh"])
    for s in ("o3d.io.read_triangle_mesh(str(self.mesh_path))", "o3d.t.geometry.TriangleMesh.from_legacy",
              "o3d.t.geometry.RaycastingScene()", "self.scene.add_triangles(mesh_t)",
              "self.mesh_path_template.format(scene_id=scene_id)"):
        assert s in load, s
    # the evaluator's restrict_collision rule: one check per move, a colliding move is skipped
    ev = (REPO / "e3vs" / "eval" / "evaluator.py").read_text()
    assert ("check_trajectory_collision(observation.c2w[:3, 3][np.newaxis, :], next_c2w[:3, 3][np.newaxis, :])[0]"
            in ev)
    cfg = (REPO / "e3vs" / "configs" / "qwen_3.0_vl_8B_instruct" / "single_frame.yaml").read_text()
    assert re.search(r"""action_setting:\s*["']?restrict_collision""", cfg)
    assert re.search(r"collision_mesh_path:.*\{scene_id\}/scans/mesh_aligned_0\.05\.ply", cfg)
    assert str(MV.MESH_REL) == "scans/mesh_aligned_0.05.ply"


def _fake_repo(tmp_path, commit: str, metrics: bytes | None) -> Path:
    r = tmp_path / "repo"
    (r / ".git").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text(commit + "\n")
    if metrics is not None:
        (r / MC.METRICS_REL).parent.mkdir(parents=True)
        (r / MC.METRICS_REL).write_bytes(metrics)
    return r


def test_official_facts_refuse_other_code(tmp_path):
    with pytest.raises(AssertionError, match="not the pinned"):
        MC.official_facts(_fake_repo(tmp_path / "a", "0" * 40, b"x"))
    with pytest.raises(AssertionError, match="sha256"):
        MC.official_facts(_fake_repo(tmp_path / "b", E.E3VS_COMMIT, b"class Collisions: pass\n"))
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(AssertionError, match="not the pinned"):
        MC.official_facts(plain)  # not a git checkout: the commit cannot be verified


@needs_repo
def test_open3d_missing_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)  # import open3d -> ImportError
    monkeypatch.setattr(MC, "_OFFICIAL", {})
    with pytest.raises(MC.Open3DMissing, match=re.escape(str(MC.O3D_DIR))):
        MC.open3d()
    p = write_box_mesh(tmp_path / "s" / MV.MESH_REL, [ROOM])
    with pytest.raises(MC.Open3DMissing, match="pip --target"):
        MC.MeshCollider("s", p, repo=REPO)
    with pytest.raises(FileNotFoundError):
        MC.MeshCollider("s", tmp_path / "missing.ply", repo=REPO)


# ------------------------------------------------------------------------------------------------------------
# the official check on a box room (stub and real open3d give the same answers)
# ------------------------------------------------------------------------------------------------------------
FREE = [((0.5, 0.5, 1.5), (0.75, 0.5, 1.5)),      # open floor area
        ((1.0, 2.0, 1.5), (1.25, 2.0, 1.5)),      # towards the pillar, ends 0.25 m before it
        ((3.7, 0.5, 1.5), (3.95, 0.5, 1.5)),      # ends 5 cm before the wall: no 0.05 m extension any more
        ((3.75, 0.5, 1.5), (4.0, 0.5, 1.5)),      # ends ON the wall: within the official end tolerance
        ((4.0, 0.5, 1.5), (3.75, 0.5, 1.5)),      # starts on the wall, moves away
        ((3.9, 0.5, 1.5), (3.65, 0.5, 1.5)),      # starts 0.1 m from the wall (old 1-voxel margin): moves away
        ((3.99995, 0.5, 1.5), (4.24995, 0.5, 1.5)),  # starts 0.05 mm before the wall: within the start tolerance
        ((2.0, 2.0, 1.5), (2.25, 2.0, 1.5)),      # inside the pillar's surface shell: a mesh is a surface
        ((2.0, 0.6, 1.4), (2.0, 0.6, 1.4))]       # zero length: never a collision
HIT = [((1.3, 2.0, 1.5), (1.55, 2.0, 1.5)),       # into the pillar face x = 1.5
       ((3.7505, 0.5, 1.5), (4.0005, 0.5, 1.5)),  # ends 0.5 mm past the wall
       ((3.9995, 0.5, 1.5), (4.2495, 0.5, 1.5)),  # starts 0.5 mm before the wall, moves through it
       ((0.6, 0.6, 0.2), (0.6, 0.6, -0.05)),      # through the floor
       ((0.3, 1.9, 1.5), (3.8, 1.9, 1.5))]        # a long segment through the pillar


@needs_repo
def test_official_check_on_a_box_room(backend, tmp_path):
    eval_dir = REPO / "e3vs" / "eval"
    before = sorted(p.name for p in eval_dir.iterdir())
    p = backend.scene(tmp_path / "box" / MV.MESH_REL, [ROOM, PILLAR])
    col = MC.MeshCollider("box", p, repo=REPO)
    assert sorted(p.name for p in eval_dir.iterdir()) == before  # executed by path, no bytecode written
    for p0, p1 in FREE:
        assert not col(np.array(p0), np.array(p1)), (p0, p1)
    for p0, p1 in HIT:
        assert col(np.array(p0), np.array(p1)), (p0, p1)
    # collides_many is the same official call, vectorised
    segs = FREE + HIT
    many = col.collides_many([s[0] for s in segs], [s[1] for s in segs])
    assert many.tolist() == [False] * len(FREE) + [True] * len(HIT)
    # ... and it IS the official method with its default tolerance: same answers on random segments
    rng = np.random.default_rng(0)
    a = rng.uniform([0.1, 0.1, 0.1], [3.9, 3.9, 2.9], size=(300, 3))
    b = a + rng.normal(size=(300, 3)) * 0.3
    direct = col.official.check_trajectory_collision(a, b)
    assert np.array_equal(col.collides_many(a, b), direct) and 0 < direct.sum() < 300
    assert [col(x, y) for x, y in zip(a[:50], b[:50])] == direct[:50].tolist()
    # distance to the nearest triangle and "in obstacle" (< 1e-4 m)
    d = col.distance([[0.5, 0.5, 1.5], [2.0, 1.5, 1.0], [2.0, 2.0, 1.5], [3.99995, 0.5, 1.5], [0.5, 2.0, 1.5]])
    assert np.allclose(d, [0.5, 0.0, 0.5, 5e-5, 0.5], atol=2e-6)
    assert col.in_obstacle([[2.0, 1.5, 1.0], [3.99995, 0.5, 1.5], [3.9995, 0.5, 1.5]]).tolist() == [True, True, False]
    lo, hi = col.aabb()
    assert np.allclose(lo, ROOM[0]) and np.allclose(hi, ROOM[1])
    # geometry.rollout with the collider: a move into the wall is skipped, the step is consumed
    start = look_c2w([3.8, 0.5, 1.5], [1, 0, 0])
    poses, flags = G.rollout(start, ["move_forward", "turn_left", "move_backward"], collides=col)
    assert flags == [True, False, False] and np.allclose(poses[1], start)
    assert col.meta["official"]["sha256"] == MC.METRICS_SHA256 and col.meta["official"]["commit"] == E.E3VS_COMMIT
    assert MC.loaded_official()[0]["open3d_version"] == ("stub" if backend.kind == "stub" else
                                                          sys.modules["open3d"].__version__)


@needs_repo
def test_a_mesh_without_triangles_is_refused(backend, tmp_path):
    p = tmp_path / "empty" / MV.MESH_REL
    p.parent.mkdir(parents=True)
    p.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                  b"element face 0\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n")
    with pytest.raises(Exception) as ei:  # our guard, or Open3D's own error on an empty scene
        MC.MeshCollider("empty", p, repo=REPO)
    if backend.kind == "stub":
        assert "no triangles" in str(ei.value)


# ------------------------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------------------------
class PlaneCollider:
    """A wall x = 1 (numpy, for the counting logic only)."""

    def collides_many(self, p0, p1):
        a, b = np.asarray(p0, float).reshape(-1, 3), np.asarray(p1, float).reshape(-1, 3)
        return (a[:, 0] - 1.0) * (b[:, 0] - 1.0) < 0

    def distance(self, pts):
        return np.abs(np.asarray(pts, float).reshape(-1, 3)[:, 0] - 1.0)

    def aabb(self):
        return np.array([-5.0, -5.0, -5.0]), np.array([1.0, 5.0, 5.0])


def test_validate_scene_and_pooled():
    col = PlaneCollider()
    traj = [look_c2w([0.0, 0.1 * i, 1.0], [1, 0, 0]) for i in range(11)]
    traj += [look_c2w([1.5, 1.0, 1.0], [1, 0, 0]), look_c2w([0.5, 1.0, 1.0], [1, 0, 0])]  # 2 crossings, 1 outside
    vps = [("free", look_c2w([0.0, 0.0, 1.0], [1, 0, 0])), ("on_wall", look_c2w([1.0, 0.0, 1.0], [1, 0, 0])),
           ("near", look_c2w([0.97, 0.0, 1.0], [1, 0, 0]))]
    starts = [("test", vps[0][1]), ("test", look_c2w([0.9, 0.0, 1.0], [1, 0, 0])), ("val", vps[1][1])]
    rec, raw = MC.validate_scene(col, np.array(traj), vps, starts)
    assert rec["iphone_frames"] == 13 and rec["iphone_segments"] == 12 and rec["iphone_colliding"] == 2
    assert rec["iphone_in_aabb"] == 12 and rec["aabb"] == [[-5.0, -5.0, -5.0], [1.0, 5.0, 5.0]]
    assert rec["viewpoints"] == 3 and rec["viewpoints_inside"] == 1 and rec["viewpoints_near"] == 2
    assert rec["inside_tags"] == ["on_wall"]
    # starts: from x = 0.9 facing +x, move_forward (0.25 m) crosses the wall; on the wall nothing crosses (the
    # plane test here has no tolerance; the official one is tested above)
    t, v = rec["starts"]["test"], rec["starts"]["val"]
    assert t == {"n": 2, "inside": 0, "near": 0, "moves": 12, "moves_colliding": 1, "all_blocked": 0}
    assert v["n"] == 1 and v["inside"] == 1 and v["near"] == 1
    pooled = MC.pooled_validation({"a": rec}, {"a": raw})
    assert pooled["iphone_colliding_frac"] == round(2 / 12, 6) and not pooled["pass_iphone"]
    assert pooled["viewpoints_inside_frac"] == round(1 / 3, 6) and not pooled["pass_viewpoints"]
    assert pooled["starts_by_split"]["test"]["moves_colliding_frac"] == round(1 / 12, 6)
    assert pooled["starts_by_split"]["val"]["inside_frac"] == 1.0 and pooled["iphone_frame"]["in_aabb_frac"] == \
        round(12 / 13, 6)
    assert pooled["viewpoint_distance_m"]["min"] == 0.0 and pooled["starts_by_split"]["test"]["distance_m"]["n"] == 2
    big = {**rec, "iphone_segments": 1000, "iphone_colliding": 9, "viewpoints": 200, "viewpoints_inside": 1}
    p2 = MC.pooled_validation({"a": big})
    assert p2["pass_iphone"] and p2["pass_viewpoints"]  # 0.9 % and 0.5 % are < 1 %
    p3 = MC.pooled_validation({"a": {**big, "iphone_colliding": 10}})
    assert not p3["pass_iphone"]  # 1.0 % is not < 1 %
    empty = MC.pooled_validation({"a": {**rec, "iphone_segments": 0, "iphone_colliding": 0}})
    assert empty["iphone_colliding_frac"] is None and not empty["pass_iphone"]  # no segment: cannot pass


def _mesh_tree(tmp_path, scenes: dict, write) -> tuple[Path, Path]:
    """mesh_dl/<scene>/scans/mesh_aligned_0.05.ply for {scene: boxes} + the mesh_verify manifest."""
    root = tmp_path / "mesh_dl"
    for s, boxes in scenes.items():
        write(MC.mesh_path(s, root), boxes)
    ids = tmp_path / "ids.txt"
    ids.write_text("".join(f"{s}\n" for s in scenes))
    res = MV.verify(root, ids, tmp_path, expected=list(scenes))
    assert res["ok"], res["problems"]
    return root, tmp_path / MV.MANIFEST


def test_validate_refuses_unverified_meshes(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)  # refusals come before any open3d use
    root, man = _mesh_tree(tmp_path, {"s1": [ROOM], "s2": [ROOM, PILLAR]}, write_box_mesh)
    shas = MC.verify_inputs(["s1", "s2"], man, root)[1]
    assert shas == {s: E.sha256_file(MC.mesh_path(s, root)) for s in ("s1", "s2")}
    out = tmp_path / "val"
    with pytest.raises(MC.MeshRefused, match="no mesh manifest"):
        MC.validate(["s1"], [], out, tmp_path / "nope.json", root)
    with pytest.raises(MC.MeshRefused, match="s3: not in"):
        MC.validate(["s1", "s3"], [], out, man, root)
    m = json.loads(man.read_text())
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps({**m, "ok": False}))
    with pytest.raises(MC.MeshRefused, match="ok is False"):
        MC.validate(["s1"], [], out, bad, root)
    bad.write_text(json.dumps({**m, "mesh_set_sha256": "0" * 64}))
    with pytest.raises(MC.MeshRefused, match="does not match its scene list"):
        MC.validate(["s1"], [], out, bad, root)
    with open(MC.mesh_path("s2", root), "ab") as f:
        f.write(b"\0")
    with pytest.raises(MC.MeshRefused, match="s2: .* sha256 .* != the manifest's"):
        MC.validate(["s1", "s2"], [], out, man, root)
    MC.mesh_path("s1", root).unlink()
    with pytest.raises(MC.MeshRefused, match="s1: .* missing"):
        MC.validate(["s1"], [], out, man, root)
    assert not out.exists()  # nothing written


def _episodes(scene: str, starts, goal_on_pillar: bool = False) -> list[dict]:
    eps = []
    for i, (pos, fwd) in enumerate(starts):
        goal = look_c2w([2.0, 1.5, 1.0] if goal_on_pillar and i == 0 else [1.0, 3.0 - 0.1 * i, 1.2], [0, 1, 0])
        eps.append({"split": ("test", "val", "train")[i % 3], "episode_index": 100 + i, "scene_id": scene,
                    "start_c2w": look_c2w(pos, fwd), "goals": {1: goal},
                    "unans": {0: look_c2w(pos, fwd), 2: look_c2w([3.0, 0.6 + 0.05 * i, 2.0], [-1, 0, 0])}})
    return eps


STARTS = [((0.5, 0.5, 1.5), (1, 0, 0)), ((0.03, 3.0, 1.5), (1, 0, 0)), ((3.0, 3.0, 1.0), (0, -1, 0))]


@needs_repo
def test_validate_check_and_factory(backend, tmp_path, monkeypatch):
    root, man = _mesh_tree(tmp_path, {"s1": [ROOM, PILLAR]}, backend.scene)
    eps = _episodes("s1", STARTS)
    ring = np.array([look_c2w([2.0 + np.cos(a), 2.0 + np.sin(a), 1.4], [1, 0, 0])
                     for a in np.linspace(0, 2 * np.pi, 120)])  # 0.29-0.5 m off the pillar
    out = tmp_path / "val"
    res = MC.validate(["s1"], eps, out, man, root, iphone={"s1": ring}, repo=REPO, convention="opencv")
    v = json.loads(MC.validation_path(out).read_text())
    assert v == json.loads(json.dumps(res)) and res["pass"] and not res["warnings"]
    p = res["pooled"]
    assert p["iphone_segments"] == 119 and p["iphone_colliding"] == 0 and p["iphone_frame"]["in_aabb_frac"] == 1.0
    assert p["viewpoints"] == len(RD.scene_viewpoints(eps, "s1")) == 9 and p["viewpoints_inside"] == 0
    dq = p["iphone_frame"]["distance_m"]
    assert 0.29 < dq["min"] < 0.3 and 0.49 < dq["max"] < 0.501 and dq["n"] == 120
    # the corner start 3 cm from the wall x = 0: near (descriptive); its move_backward collides, forward does not
    st = p["starts_by_split"]["val"]
    assert st["n"] == 1 and st["near"] == 1 and st["inside"] == 0 and 1 <= st["moves_colliding"] < 6
    corner = eps[1]["start_c2w"]
    col = MC.MeshCollider("s1", MC.mesh_path("s1", root), repo=REPO)
    assert col(corner[:3, 3], G.apply_action(corner, "move_backward")[:3, 3])
    assert not col(corner[:3, 3], G.apply_action(corner, "move_forward")[:3, 3])
    assert st["moves_colliding"] == sum(col(corner[:3, 3], G.apply_action(corner, m)[:3, 3]) for m in MC.MOVES)
    assert res["mesh_set_sha256"] == json.loads(man.read_text())["mesh_set_sha256"] and res["manifest"]["key_matches"]
    assert res["official"]["sha256"] == MC.METRICS_SHA256 and res["tolerance_m"] == 1e-4
    assert res["iphone_convention"] == {"convention": "opencv", "source": "given (--convention)",
                                        "note": res["iphone_convention"]["note"]}
    assert not MC.definition_problems(v)
    # check: passes and prints the key; a changed mesh, another manifest key or another definition fail it
    c = MC.check(["s1"], out, man, root)
    assert c["ok"] and c["mesh_set_sha256"] == res["mesh_set_sha256"]
    argv = ["check", "--scenes", "s1", "--out", str(out), "--manifest", str(man), "--mesh-root", str(root)]
    assert MC.main(argv) == 0
    assert not MC.check(["s1", "s9"], out, man, root)["ok"]
    vp = MC.validation_path(out)
    good = vp.read_text()
    vp.write_text(json.dumps({**v, "tolerance_m": 0.05}))
    assert "tolerance" in MC.check(["s1"], out, man, root)["problems"][0]
    vp.write_text(good)
    # the factory: env-located files; refuses without a passing validation or with a changed mesh
    monkeypatch.setenv("VPD_E3VS_MESH_ROOT", str(root))
    monkeypatch.setenv("VPD_E3VS_MESH_MANIFEST", str(man))
    monkeypatch.setenv("VPD_E3VS_MESH_VAL_DIR", str(out))
    f = B.load_collider_factory(MC.FACTORY)
    g = f("s1")
    assert isinstance(g, MC.MeshCollider) and g(np.array(HIT[0][0]), np.array(HIT[0][1]))
    with pytest.raises(AssertionError, match="not in"):
        f("s9")
    with open(MC.mesh_path("s1", root), "ab") as fh:
        fh.write(b"\0")
    assert not MC.check(["s1"], out, man, root)["ok"] and MC.main(argv) == 1
    with pytest.raises(AssertionError, match="sha256"):
        f("s1")
    vp.write_text(json.dumps({**v, "pass": False}))
    with pytest.raises(AssertionError, match="not a passing validation"):
        f("s1")
    vp.unlink()
    with pytest.raises(FileNotFoundError, match="no mesh validation"):
        f("s1")


@needs_repo
def test_validation_fails_on_a_viewpoint_in_obstacle(backend, tmp_path):
    """A template viewpoint on a mesh surface counts as in obstacle; 1 of a few viewpoints is >= 1 %: no pass."""
    root, man = _mesh_tree(tmp_path, {"s1": [ROOM, PILLAR]}, backend.scene)
    eps = _episodes("s1", STARTS, goal_on_pillar=True)
    ring = np.array([look_c2w([2.0 + np.cos(a), 2.0 + np.sin(a), 1.4], [1, 0, 0])
                     for a in np.linspace(0, 2 * np.pi, 120)])
    res = MC.validate(["s1"], eps, tmp_path / "val", man, root, iphone={"s1": ring}, repo=REPO, convention="opencv")
    assert not res["pass"] and res["pooled"]["pass_iphone"] and not res["pooled"]["pass_viewpoints"]
    assert res["scenes"]["s1"]["inside_tags"] == ["ep100_goal1"]
    assert not MC.check(["s1"], tmp_path / "val", man, root)["ok"]
    # the iPhone poses outside the mesh frame: the metric passes trivially, the validation warns (descriptive)
    far = ring.copy()
    far[:, :3, 3] += 100.0
    res2 = MC.validate(["s1"], _episodes("s1", STARTS), tmp_path / "val2", man, root, iphone={"s1": far},
                       repo=REPO, convention="opencv")
    assert res2["pass"] and res2["warnings"] and "AABB" in res2["warnings"][0]


def test_no_bytecode_written():
    assert not list((paths.VPD_PKG / "e3vs").glob("__pycache__"))
