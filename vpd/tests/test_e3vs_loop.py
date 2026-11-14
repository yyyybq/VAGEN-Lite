"""WP-E3VS-B tests: renderer helpers, the (superseded, kept) occupancy collision model, SFT builder, closed loop
(fake renderer + fake server), summary and gate, the two sco payloads.  No GPU, no gsplat rendering, no model
weights.  The official mesh collision model (mesh_collision.py) has its own file, test_e3vs_mesh_collision.py.

Login node, 1 core:  OMP_NUM_THREADS=1 ... nice -n 19 PY_Q3 -B -m pytest -q -p no:cacheprovider
vpd/tests/test_e3vs_loop.py

The E3VS checkout is $VPD_E3VS_REPO, else paths.E3VS_REPO, else the earlier-fetched read-only copy
plan/login_tmp/vpd_e3vs/repo (reading it is allowed, DECISIONS section D); official code is executed from it
read-only (by path or by ast extraction; no bytecode written).  Test outputs go to
eval_external_assets/vpd/scratch/e3vs-b/pytest_<rand>/ (VPD_E3VS_B_TEST_DIR overrides), removed at teardown.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import typing
import uuid
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from vpd import paths  # noqa: E402
import vpd.e3vs as E  # noqa: E402
from vpd.e3vs import bfs_demos as B  # noqa: E402
from vpd.e3vs import geometry as G  # noqa: E402
from vpd.e3vs import make_sft as M  # noqa: E402
from vpd.e3vs import mesh_collision as MC  # noqa: E402
from vpd.e3vs import mesh_verify as MV  # noqa: E402
from vpd.e3vs import occupancy as OC  # noqa: E402
from vpd.e3vs import render as RD  # noqa: E402
from vpd.e3vs import run as RU  # noqa: E402
from vpd.e3vs import summary as S  # noqa: E402

JOBS = paths.VPD_PKG / "jobs"


def _repo() -> Path | None:
    for p in (os.environ.get("VPD_E3VS_REPO"), paths.E3VS_REPO, E.LOGIN_TMP_REPO):
        if p and (Path(p) / "dataset").is_dir():
            return Path(p)
    return None


REPO = _repo()
needs_repo = pytest.mark.skipif(REPO is None, reason="no E3VS checkout on this machine")


@pytest.fixture(scope="module")
def scratch():
    base = Path(os.environ.get("VPD_E3VS_B_TEST_DIR") or paths.scratch("e3vs-b"))
    d = base / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def val_eps():
    if REPO is None:
        pytest.skip("no E3VS checkout")
    return E.load_episodes("val", REPO)


def _torch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    return torch


def look_c2w(pos, fwd) -> np.ndarray:
    """Roll-free OpenCV c2w at pos looking along fwd (world Z up)."""
    f = np.asarray(fwd, float) / np.linalg.norm(fwd)
    x = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    c = np.eye(4)
    c[:3, 0], c[:3, 1], c[:3, 2], c[:3, 3] = x, y, f, pos
    return c


# collider factories used through --collides MODULE:FUNC
def block_moves(scene_id):
    """Every translation collides."""
    return lambda p0, p1: True


# ============================================================================================================
# render.py
# ============================================================================================================
def test_look_c2w_is_opencv():
    c = look_c2w([0, 0, 0], [1, 0, 0])
    assert np.allclose(np.cross(c[:3, 0], c[:3, 1]), c[:3, 2])
    assert np.allclose(c[:3, 1], [0, 0, -1])  # image down = world down
    assert G.roll_of(c) < 1e-12


def test_fake_renderer_depth_and_rgb():
    r = RD.FakeRenderer(room=((0, 0, 0), (4, 4, 3)), boxes=[((1.5, 1.5, 0), (2.5, 2.5, 3))])
    c = look_c2w([0.5, 0.5, 1.5], [1, 0, 0])
    d, a = r.render_depth(c, 64, 64)
    assert d.shape == (64, 64) and a.min() == 1.0
    # the centre ray hits the wall x = 4 at z-depth 3.5 (the pillar spans y 1.5-2.5, off this ray)
    assert abs(float(d[32, 32]) - 3.5) < 0.1
    c2 = look_c2w([0.5, 2.0, 1.5], [1, 0, 0])  # straight at the pillar face x = 1.5
    d2, _ = r.render_depth(c2, 64, 64)
    assert abs(float(d2[31:33, 31:33].mean()) - 1.0) < 1e-6
    img = r.render_rgb(c, 64, 48)
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8 and img.std() > 1
    assert np.array_equal(img, r.render_rgb(c, 64, 48))  # deterministic
    assert not np.array_equal(img, r.render_rgb(look_c2w([0.5, 0.5, 1.5], [0, 1, 0]), 64, 48))


def test_png_psnr_ssim():
    from PIL import Image
    import io

    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    png = RD.to_png(a)
    back = np.asarray(Image.open(io.BytesIO(png)))
    assert np.array_equal(back, a)
    assert RD.psnr(a, a) == RD.PSNR_IDENTICAL and abs(RD.ssim(a, a) - 1.0) < 1e-9
    b = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)
    assert 30 < RD.psnr(a, b) < 45 and RD.ssim(a, b) < 1.0
    with pytest.raises(ValueError):
        RD.to_png(a.astype(np.float32))


def test_conventions_and_intrinsics():
    c = look_c2w([1, 2, 3], [0, 1, 0])
    gl = c @ RD.GL_TO_CV  # the same camera stored in OpenGL convention
    assert np.allclose(RD.convention_c2w(gl, "opengl"), c)
    assert np.allclose(RD.convention_c2w(c, "opencv"), c)
    K = RD.intrinsics(512, 512)
    assert K[0, 0] == 256 and K[0, 2] == 256 and K[1, 2] == 256
    with pytest.raises(ValueError):
        RD.convention_c2w(c, "blender")


def _write_ply(path: Path, n: int, sh_rest: int, seed: int = 0):
    from plyfile import PlyData, PlyElement

    rng = np.random.default_rng(seed)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + \
        [f"f_rest_{i}" for i in range(sh_rest)] + ["opacity", "scale_0", "scale_1", "scale_2"] + \
        ["rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(n, dtype=[(k, "f4") for k in names])
    for k in names:
        arr[k] = rng.normal(size=n).astype(np.float32)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


@needs_repo
@pytest.mark.parametrize("sh_rest", [45, 0])
def test_load_splat_equals_official_loader(scratch, sh_rest):
    """Our ply_gaussian_loader path gives the official load_ply_model tensors (WXYZ quats, SH layout, activations)."""
    torch = _torch()
    pytest.importorskip("plyfile")
    pytest.importorskip("ply_gaussian_loader")
    p = scratch / f"tiny_{sh_rest}.ply"
    _write_ply(p, 200, sh_rest)
    ours = RD.load_splat(p, "cpu")
    off = RD.official_loader(REPO)(p, torch.device("cpu"))
    names = ("means", "quats", "scales", "opacities", "colors")
    for nm, t in zip(names, off[:5]):
        assert tuple(ours[nm].shape) == tuple(t.shape), nm
        assert torch.allclose(ours[nm], t, atol=1e-6, rtol=1e-6), nm
    assert ours["sh_degree"] == off[5] == (3 if sh_rest else 0)
    # the xyzw reorder (get_rotations_xyzw) would NOT match
    from ply_gaussian_loader import PLYGaussianLoader

    xyzw = torch.from_numpy(PLYGaussianLoader().load_ply(str(p)).get_rotations_xyzw())
    assert not torch.allclose(xyzw, off[1], atol=1e-3)


def test_render_module_never_uses_xyzw():
    src = (paths.VPD_PKG / "e3vs" / "render.py").read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    calls = re.findall(r"\.get_rotations_xyzw\(", code)
    assert calls == [], "render.py must pass the loader's WXYZ rotations"
    assert "GaussianRenderer(" not in code


@needs_repo
def test_scene_viewpoints_and_check_inputs(val_eps, scratch):
    sid = val_eps[0]["scene_id"]
    vps = RD.scene_viewpoints(val_eps, sid)
    keys = {tuple(np.round(c[:3, :4], 4).reshape(-1)) for _, c in vps}
    assert len(keys) == len(vps) >= 27
    assert vps == RD.scene_viewpoints(list(reversed(val_eps)), sid)  # order does not depend on the input order
    with pytest.raises(AssertionError):
        RD.check_inputs(REPO, scratch / "no_data")


# ============================================================================================================
# occupancy.py
# ============================================================================================================
ROOM = ((0.0, 0.0, 0.0), (4.0, 4.0, 3.0))
PILLAR = ((1.5, 1.5, 0.0), (2.5, 2.5, 3.0))


@pytest.fixture(scope="module")
def box_grid():
    r = RD.FakeRenderer("box", room=ROOM, boxes=[PILLAR])
    K = OC.scaled_K(RD.intrinsics(512, 512), 0.25)
    views = []
    for i, (x, y) in enumerate([(0.6, 0.6), (3.4, 0.6), (3.4, 3.4), (0.6, 3.4), (2.0, 0.7), (0.7, 2.0),
                                (3.3, 2.0), (2.0, 3.3)]):
        for j, f in enumerate([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, -1.5), (-1, -1, -1.5),
                               (1, -1, 1.2), (-1, 1, 1.2)]):
            views.append((f"v{i}_{j}", look_c2w([x, y, 1.4], f), K, 128, 128))
    keys, counts, _ = OC.accumulate(views, r)
    occ = counts >= OC.MIN_HITS
    return OC.Grid(keys[occ], hits=counts[occ], meta={"scene_id": "box"})


def test_keys_roundtrip_and_dilate():
    idx = np.array([[0, 0, 0], [-5, 7, 123], [1000, -1000, 3]])
    assert np.array_equal(OC.unpack(OC.pack(idx)), idx)
    d = OC.dilate(OC.pack(idx[:1]), 1)
    assert d.size == 27 and np.array_equal(np.sort(d), d)
    assert set(map(tuple, OC.unpack(d))) == {(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)}


def test_backproject_lies_on_the_wall():
    r = RD.FakeRenderer("box", room=ROOM)
    c = look_c2w([1.0, 2.0, 1.5], [1, 0, 0])
    K = RD.intrinsics(32, 32)
    d, a = r.render_depth(c, 32, 32, K)
    pts = OC.backproject(d, a, c, K)
    on_wall = np.abs(pts[:, 0] - 4.0) < 1e-6
    on_other = (np.abs(pts[:, 2]) < 1e-6) | (np.abs(pts[:, 2] - 3.0) < 1e-6) | (np.abs(pts[:, 1]) < 1e-6) | \
        (np.abs(pts[:, 1] - 4.0) < 1e-6)
    assert len(pts) == 32 * 32 and (on_wall | on_other).all() and on_wall.mean() > 0.3


def test_collides_on_synthetic_box_room(box_grid):
    g = box_grid
    assert g.occ.size > 1000
    free = [((0.5, 0.5, 1.5), (0.75, 0.5, 1.5)),  # open floor area, far from every surface
            ((1.0, 2.0, 1.5), (1.25, 2.0, 1.5)),  # towards the pillar but ends 0.2 m before it (+0.05 m, 1 voxel)
            ((3.0, 3.0, 1.5), (3.0, 3.0, 1.75)),
            ((2.0, 0.6, 1.4), (2.0, 0.6, 1.4))]  # zero length: never a collision (official ray test)
    hit = [((3.7, 0.5, 1.5), (3.95, 0.5, 1.5)),  # the +0.05 m extension reaches the wall x = 4
           ((1.2, 2.0, 1.5), (1.45, 2.0, 1.5)),  # into the pillar face x = 1.5
           ((0.6, 0.6, 0.2), (0.6, 0.6, -0.05)),  # through the floor
           ((0.3, 1.9, 1.5), (3.8, 1.9, 1.5))]  # a long segment through the pillar
    for p0, p1 in free:
        assert not g(np.array(p0), np.array(p1)), (p0, p1)
    for p0, p1 in hit:
        assert g(np.array(p0), np.array(p1)), (p0, p1)
    assert g.occupied([[2.0, 1.5, 1.0]]).all() or g.occupied([[2.0, 1.49, 1.0]]).all()
    assert not g.occupied([[0.8, 0.8, 1.5]]).any() and not g.near_occupied([[0.8, 0.8, 1.5]]).any()
    # geometry.rollout with the grid: a move into the wall is skipped, the step is consumed
    start = look_c2w([3.8, 0.5, 1.5], [1, 0, 0])
    poses, flags = G.rollout(start, ["move_forward", "turn_left", "move_backward"], collides=g)
    assert flags == [True, False, False] and np.allclose(poses[1], start)


def test_grid_save_load_and_factory(box_grid, scratch, monkeypatch):
    d = scratch / "occ"
    sha = box_grid.save(OC.occ_path("box", d))
    assert sha == E.sha256_file(OC.occ_path("box", d))
    monkeypatch.setenv("VPD_E3VS_OCC_DIR", str(d))
    g2 = OC.load_collider("box")
    assert np.array_equal(g2.occ, box_grid.occ) and np.array_equal(g2.dil, box_grid.dil)
    assert g2.meta["scene_id"] == "box" and g2.voxel == OC.VOXEL_M
    rng = np.random.default_rng(1)
    for _ in range(200):
        p0 = rng.uniform([0.2, 0.2, 0.2], [3.8, 3.8, 2.8])
        p1 = p0 + rng.normal(size=3) * 0.25
        assert g2(p0, p1) == box_grid(p0, p1)
    with pytest.raises(FileNotFoundError):
        OC.load_collider("no_such_scene")
    # the bfs_demos --collides spec resolves to this factory
    assert B.load_collider_factory("vpd.e3vs.occupancy:load_collider") is OC.load_collider


def test_validate_scene_and_pooled(box_grid):
    t = np.linspace(0, 2 * np.pi, 120)
    ring = [look_c2w([2.0 + 1.0 * np.cos(a), 2.0 + 1.0 * np.sin(a), 1.4], [1, 0, 0]) for a in t]  # 0.5 m off
    # depth sees surfaces only: the grid is a shell, so an "inside" viewpoint is the centre of an occupied voxel
    shell = (OC.unpack(box_grid.occ[len(box_grid.occ) // 2: len(box_grid.occ) // 2 + 1])[0] + 0.5) * OC.VOXEL_M
    vps = [("free", look_c2w([0.8, 0.8, 1.5], [1, 0, 0])), ("inside", look_c2w(shell, [1, 0, 0]))]
    # a free voxel next to the wall x = 4 at mid height: within the 1-voxel margin, not inside
    idx = OC.unpack(box_grid.occ)
    cand = idx[(np.abs((idx[:, 2] + 0.5) * OC.VOXEL_M - 1.4) < 0.1) & (np.abs((idx[:, 1] + 0.5) * OC.VOXEL_M - 2.0) < 0.3)]
    s = cand[np.argmax(cand[:, 0])].copy()
    while box_grid.occupied([(s + 0.5) * OC.VOXEL_M]).any():
        s[0] -= 1
    near_wall = look_c2w((s + 0.5) * OC.VOXEL_M, [-1, 0, 0])
    assert 3.8 < near_wall[0, 3] < 4.0
    starts = [("test", vps[0][1]), ("test", near_wall), ("train", vps[0][1])]
    r = OC.validate_scene(box_grid, np.array(ring), vps, starts)
    assert r["iphone_segments"] == 119 and r["iphone_colliding"] == 0
    assert r["viewpoints"] == 2 and r["viewpoints_inside"] == 1 and r["inside_tags"] == ["inside"]
    assert r["starts"] == {"test": {"n": 2, "inside": 0, "near": 1}, "train": {"n": 1, "inside": 0, "near": 0}}
    # option (ii): from a start within the margin, moves away from the wall are free, moves into it collide
    hit = [box_grid(near_wall[:3, 3], G.apply_action(near_wall, a)[:3, 3]) for a in sorted(E.TRANSLATIONS)]
    assert any(hit) and not all(hit)
    # the literal rule (option (i), start_skip 0) is the rotation-only case: every translation collides
    literal = OC.Grid(box_grid.occ, voxel=box_grid.voxel, margin=box_grid.margin, extend=box_grid.extend,
                      start_skip=0.0)
    assert all(literal(near_wall[:3, 3], G.apply_action(near_wall, a)[:3, 3]) for a in sorted(E.TRANSLATIONS))
    ps = OC.pooled_validation({"a": r})["starts_by_split"]
    assert ps["test"]["near_frac"] == 0.5 and ps["train"]["near_frac"] == 0.0
    pooled = OC.pooled_validation({"a": r, "b": {**r, "viewpoints": 200, "viewpoints_inside": 0}})
    assert pooled["pass_iphone"] and pooled["viewpoints_inside_frac"] == round(1 / 202, 6)
    assert pooled["pass_viewpoints"]  # 1 / 202 < 1 %
    assert not OC.pooled_validation({"a": r})["pass_viewpoints"]  # 1 / 2


@needs_repo
def test_build_scene_and_validate_with_fake_renderer(val_eps, scratch):
    """End to end on one val scene with the fake renderer (no iPhone poses): files, meta, validation.json."""
    sid = val_eps[0]["scene_id"]
    out = scratch / "occ_scene"
    r = RD.FakeRenderer(sid)
    grid, st = OC.build_scene(sid, val_eps, r, out, use_iphone=False)
    assert Path(st["path"]).is_file() and st["n_views"] == len(RD.scene_viewpoints(val_eps, sid))
    assert grid.meta["params"]["min_hits"] == 3 and grid.meta["params"]["hit_unit"] == "view"
    res = OC.validate([sid], val_eps, out, iphone={sid: None})
    assert (out / "validation.json").is_file() and res["scenes"][sid]["viewpoints_inside"] == 0
    assert res["pooled"]["iphone_segments"] == 0 and res["pass"] is False  # no iPhone segment: cannot pass
    assert grid.meta["iphone_convention"] is None and grid.meta["iphone_convention_source"] is None
    assert res["iphone_convention"]["conventions"] == ["None"] and res["scenes"][sid]["iphone_convention"] is None
    # the grids' identity, and the episode starts inside / within the margin (descriptive)
    sha = E.sha256_file(OC.occ_path(sid, out))
    assert res["grids_sha256"] == OC.grids_key({sid: sha}) != OC.grids_key({sid: "0" * 64})
    n_val = sum(e["scene_id"] == sid for e in val_eps)
    sv = res["pooled"]["starts_by_split"]["val"]
    assert sv["n"] == n_val and sv["inside"] == 0 and 0 <= sv["near"] <= n_val and "near_frac" in sv
    # --resume keeps a grid built with the same settings only
    p = OC.occ_path(sid, out)
    assert OC.reusable(p, "fake", sid, None, OC.IPHONE_EVERY)
    assert not OC.reusable(p, "fake", sid, None, 7) and not OC.reusable(p, "fake", sid, "opencv", OC.IPHONE_EVERY)
    assert not OC.reusable(p, "gsplat", sid, None, OC.IPHONE_EVERY) and not OC.reusable(out / "x.npz", "fake", sid,
                                                                                          None, OC.IPHONE_EVERY)
    mtime = p.stat().st_mtime_ns
    assert OC.main(["build", "--scenes", sid, "--renderer", "fake", "--no-iphone", "--resume", "--out", str(out),
                    "--repo", str(REPO)]) == 0
    assert p.stat().st_mtime_ns == mtime  # kept, not rebuilt
    # check: validation must pass, cover the scene, and the npz must be unchanged
    assert not OC.check([sid], out)["ok"]  # pass is False
    v = json.loads((out / "validation.json").read_text())
    v["pass"] = True
    (out / "validation.json").write_text(json.dumps(v))
    c = OC.check([sid], out)
    assert c["ok"] and c["grids_sha256"] == res["grids_sha256"]
    assert not OC.check([sid, "other_scene"], out)["ok"]
    assert OC.main(["check", "--scenes", sid, "--out", str(out)]) == 0
    with open(p, "ab") as f:
        f.write(b"x")
    assert not OC.check([sid], out)["ok"] and OC.main(["check", "--scenes", sid, "--out", str(out)]) == 1


def test_convention_auto_refuses_without_an_iphone_parity_result(scratch, val_eps, monkeypatch):
    """--convention auto never silently defaults: no parity file with an iPhone 'best' refuses; the source is kept."""
    d = scratch / "conv_parity"
    d.mkdir()
    with pytest.raises(SystemExit, match="iPhone 'best'"):
        OC.resolve_convention("auto", d)  # no parity file at all
    (d / "parity_a.json").write_text(json.dumps({"pass": True, "iphone": {"error": "OSError: no frame"}}))
    with pytest.raises(SystemExit, match="iPhone 'best'"):
        OC.resolve_convention("auto", d)  # the iPhone part errored
    (d / "parity_b.json").write_text(json.dumps({"pass": True, "iphone": {"best": "opengl"}}))
    assert OC.resolve_convention("auto", d) == ("opengl", "parity")
    assert OC.resolve_convention("opencv", d) == ("opencv", "given (--convention)")
    # the build records the convention and its source in the npz meta, and validate reports them
    sid = val_eps[0]["scene_id"]
    orig = OC.scene_views
    monkeypatch.setattr(OC, "scene_views", lambda s, e, md, *a, **k: orig(s, e, None, *a, **k))  # no iPhone files
    out = scratch / "conv_occ"
    grid, _ = OC.build_scene(sid, val_eps, RD.FakeRenderer(sid), out, convention="opengl", md={"stub": True},
                             convention_source="parity")
    assert grid.meta["iphone_convention"] == "opengl" and grid.meta["iphone_convention_source"] == "parity"
    assert OC.Grid.load(OC.occ_path(sid, out)).meta["iphone_convention_source"] == "parity"
    res = OC.validate([sid], val_eps, out, iphone={sid: None})
    assert res["iphone_convention"]["conventions"] == ["opengl"] and res["iphone_convention"]["sources"] == ["parity"]
    assert res["scenes"][sid]["iphone_convention_source"] == "parity"


# ============================================================================================================
# prompts (make_sft.Prompts) and the parser
# ============================================================================================================
def _official_by_path(rel: str):
    spec = importlib.util.spec_from_file_location("_e3vs_" + rel.replace("/", "_").replace(".", "_"), REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _official_member(rel: str, cls: str, name: str, ns: dict):
    src = (REPO / rel).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for f in node.body:
                if getattr(f, "name", None) == name:
                    code = textwrap.dedent(ast.get_source_segment(src, f, padded=True))
                    exec(compile(code, str(REPO / rel), "exec"), ns)  # noqa: S102 (pinned repo file)
                    return ns[name]
    raise LookupError(name)


@pytest.fixture(scope="module")
def prompts():
    if REPO is None:
        pytest.skip("no E3VS checkout")
    pytest.importorskip("jinja2")
    return M.Prompts(REPO)


@needs_repo
def test_prompt_constants_match_the_repo():
    t = _official_by_path("e3vs/prompts/templates.py")
    assert t.CAMERA_MOVEMENT_DESC == M.CAMERA_MOVEMENT_DESC
    assert E.sha256_file(REPO / "e3vs/prompts/templates.py") == M.TEMPLATES_PY_SHA256
    build = _official_member("e3vs/prompts/navigation_builder.py", "_ObservationDescriptionBuilder", "build", {})
    assert build(type("C", (), {"config": M.OFFICIAL_CONFIG})()) == M.OBSERVATION_DESC
    import yaml

    y = yaml.safe_load((REPO / "e3vs/configs/qwen_3.0_vl_8B_instruct/single_frame.yaml").read_text())
    sm, env = y["SearchModel"], y["Environment"]
    for k in ("use_cot", "action_chunk", "uses_think_tags", "include_translation_info", "include_rotation_info",
              "num_past_observations", "num_past_actions", "num_past_reasoning", "available_actions",
              "action_setting", "mention_step0_target_centered", "max_output_tokens", "temperature"):
        assert sm[k] == M.OFFICIAL_CONFIG[k], k
    assert env["action_step_size"] == M.OFFICIAL_CONFIG["action_step_size"]
    assert env["rotation_step_size"] == M.OFFICIAL_CONFIG["rotation_step_size"]
    assert env["max_steps"] == E.MAX_DECISIONS and env["image_width"] == env["image_height"] == 512


def test_prompt_system_and_user(prompts):
    s = prompts.system()
    assert s.startswith("You are an agent controlling a camera in a 3D environment.\n")
    assert "**AVAILABLE ACTIONS (for this step):** " + ", ".join(E.ACTIONS) + "\n" in s
    assert "- Each 'move' action translates the camera by 0.25 meters\n" in s
    assert "- Each 'turn' or 'look' action rotates it by 30.0 degrees\n" in s
    assert M.CAMERA_MOVEMENT_DESC in s and "the target object is centered in the view" in s
    assert s.endswith(M.NEUTRAL_TAIL) and "action: move_forward" not in s
    assert prompts.official_system.endswith(M.OFFICIAL_TAIL)
    assert prompts.official_system[: -len(M.OFFICIAL_TAIL)] == s[: -len(M.NEUTRAL_TAIL)]
    u0 = prompts.user("What color is the mug?", 0, [1.234, -0.5, 1.0])
    assert u0 == "Question: 'What color is the mug?'\nCurrent step: 0\nYour current state:\n" \
                 "- Position (X, Y, Z): (1.23, -0.50, 1.00)\n"
    u3 = prompts.user("What color is the mug?", 3, np.array([1.234, -0.5, 1.0]), "move_forward", True)
    assert u3 == u0.replace("step: 0", "step: 3") + "- Last action: move_forward\n" \
                                                     "- Last action result: Collision occurred. You remained at the same position.\n"
    u4 = prompts.user("q", 4, [0, 0, 0], "turn_left", False)
    assert u4.endswith("- Last action: turn_left\n") and "Collision" not in u4
    m = prompts.meta()
    assert m["template_sha256"] == M.TEMPLATE_SHA256 and m["regex"] == M.REGEX
    assert m["system_sha256"] == hashlib.sha256(s.encode()).hexdigest()


def test_reply_and_regex():
    for a in E.ACTIONS:
        assert re.fullmatch(M.REGEX, M.reply(a)) and RU.regex_ok(M.reply(a))
        assert RU.parse_lenient(M.reply(a)) == {"action": a, "error": False, "empty": False}
    for bad in ("ACTION: stop", "THOUGHT: x\nACTION: stop", "THOUGHT: \nACTION: stop\n", "THOUGHT: \nACTION: jump",
                "thought: \naction: stop"):
        assert not RU.regex_ok(bad), bad
    with pytest.raises(ValueError):
        M.reply("jump")


PARSE_CASES = [
    "THOUGHT: \nACTION: move_up", "action: turn_left", "Action : look_down", "ACTION: Move_Forward",
    "I will go there.\nmove backward is best", "blah blah", "We should stop here.", "x\n" * 10 + "action: move_left",
    "action: move_left\n" + "y\n" * 10, "action: fly\naction: turn_right", "look up then go", "ACTION:look_up",
    "THOUGHT: the mug is left\nACTION: turn_left\n", "turn_right? no, action: stop", "a" * 300 + " move_down",
    "move_down " + "a" * 300, "Final answer -> action:   move_right   ",
]


@needs_repo
def test_parse_lenient_equals_official():
    ns = {"re": re, "logger": logging.getLogger("t"), **{k: getattr(typing, k) for k in ("Dict", "List",
                                                                                         "Optional", "Tuple", "Any")}}
    fn = _official_member("e3vs/agents/base_agent.py", "BaseAgent", "_parse_response_to_action", ns)
    me = type("A", (), {"available_actions": list(M.OFFICIAL_CONFIG["available_actions"]),
                        "_uses_think_tags": lambda self: False})()
    for text in PARSE_CASES:
        want = fn(me, text.strip())  # agents/qwen.py strips the decoded text before parsing
        got = RU.parse_lenient(text)
        assert (got["action"], got["error"]) == (want["action"], want["error"]), text
    # an empty reply is stop with error=True (agents/qwen.py, before the parser)
    assert RU.parse_lenient("  \n") == {"action": "stop", "error": True, "empty": True}


# ============================================================================================================
# make_sft.py
# ============================================================================================================
def test_stratified_order_prefixes_and_dedup():
    rng = np.random.default_rng(0)
    recs = []
    for i in range(120):
        qt = "A" if i < 60 else ("B" if i < 96 else "C")
        recs.append({"status": "reached", "extra": False, "scene_id": f"s{i % 7}", "question": f"q{i}",
                     "question_type": qt, "episode_index": i, "opt_len": int(rng.integers(1, 9))})
    dups = [dict(r, episode_index=1000 + r["episode_index"]) for r in recs[:10]]  # same (scene, question)
    junk = [dict(recs[0], episode_index=2000, question="other", status="capped"),
            dict(recs[0], episode_index=2001, question="other2", extra=True)]
    ded = M.dedup_pool(recs + dups + junk)
    assert len(ded) == 120
    for r in recs[:10]:
        keep = min((r, dict(r, episode_index=1000 + r["episode_index"])),
                   key=lambda x: M.md5key(x["scene_id"], x["episode_index"]))
        assert keep["episode_index"] in {x["episode_index"] for x in ded}
    order = M.stratified_order(ded, 1001)
    assert sorted(r["episode_index"] for r in order) == sorted(r["episode_index"] for r in ded)
    share = {"A": 0.5, "B": 0.3, "C": 0.2}
    for n in range(1, len(order) + 1):
        cnt = {q: sum(r["question_type"] == q for r in order[:n]) for q in share}
        for q in share:
            assert abs(cnt[q] - share[q] * n) <= 1.0 + 1e-9, (n, q, cnt)
    assert [r["episode_index"] for r in M.stratified_order(ded, 1001)] == [r["episode_index"] for r in order]
    assert [r["episode_index"] for r in M.stratified_order(ded, 1)] != [r["episode_index"] for r in order]


@needs_repo
def test_tier_records_all_and_nesting():
    """Round 4 tiers (OUT/plan/ROUND4.md B): a numeric tier is a prefix of the order; ALL = every reached, non-extra
    pool record, the order's episodes first, then the other starts of the same (scene, question); tiers nest."""
    recs = []
    for i in range(60):
        qt = "A" if i < 30 else ("B" if i < 48 else "C")
        recs.append({"status": "reached", "extra": False, "scene_id": f"s{i % 5}", "question": f"q{i}",
                     "question_type": qt, "episode_index": i, "opt_len": 1 + i % 4})
    dups = [dict(r, episode_index=500 + r["episode_index"]) for r in recs[::3]]  # other starts, same question
    junk = [dict(recs[0], episode_index=900, question="x", status="capped"),
            dict(recs[0], episode_index=901, question="y", extra=True)]
    pool = recs + dups + junk
    order = {"order": [{"episode_index": r["episode_index"]} for r in M.stratified_order(M.dedup_pool(pool), 1001)]}
    head = [o["episode_index"] for o in order["order"]]

    def ids(rs):
        return [r["episode_index"] for r in rs]

    t10, t30 = ids(M.tier_records(order, pool, 10, 1001)), ids(M.tier_records(order, pool, 30, 1001))
    t_all = ids(M.tier_records(order, pool, M.ALL_TIER, 1001))
    assert t10 == head[:10] and t30 == head[:30] and t_all[:30] == t30 and t_all[:len(head)] == head
    reached = {r["episode_index"] for r in pool if r["status"] == "reached" and not r["extra"]}
    assert len(head) == 60 and len(t_all) == len(set(t_all)) == len(reached) == 80 and set(t_all) == reached
    assert t_all == ids(M.tier_records(order, list(reversed(pool)), M.ALL_TIER, 1001))  # pool order irrelevant
    rest = M.stratified_order([r for r in pool if r["episode_index"] in set(t_all) - set(head)], 1001)
    assert t_all[len(head):] == ids(rest)
    for bad in (0, -1, len(head) + 1, "all", True):
        with pytest.raises(ValueError):
            M.tier_records(order, pool, bad, 1001)
    assert M.parse_tiers("100") == [100] and M.parse_tiers("300,ALL") == [300, "ALL"]
    assert M.parse_tiers("300 ALL") == [300, "ALL"]
    with pytest.raises(ValueError):
        M.parse_tiers("all")
    assert {"300", "ALL"} <= set(RU.TIERS)


def test_warmup_selection_rule():
    tr = E.load_episodes("train", REPO)
    sel = M.warmup_selection(tr)
    assert len(sel) == 20 and len({e["scene_id"] for e in sel}) == 20
    best = {}
    for e in tr:
        k = hashlib.md5(f"{e['scene_id']}:{e['episode_index']}".encode()).hexdigest()
        best[e["scene_id"]] = min(best.get(e["scene_id"], (k, e)), (k, e), key=lambda x: x[0])
    want = [e["episode_index"] for _, e in sorted(best.values(), key=lambda x: x[0])[:20]]
    assert [e["episode_index"] for e in sel] == want


@pytest.fixture(scope="module")
def val_pool(scratch, val_eps):
    """3 val episodes through the real BFS (no collision), as a pool file."""
    summ = B.run("val", out=scratch / "bfs_val", limit=3, repo=REPO, quiet=True)
    recs = [json.loads(x) for x in open(scratch / "bfs_val" / "records.jsonl")]
    assert summ["n_episodes"] == 3
    reached = [r for r in recs if r["status"] == "reached"]
    assert reached, "no reached val episode in the first 3"
    p = scratch / "bfs_val" / "pool.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in reached))
    return p, reached


@needs_repo
def test_make_sft_tier_on_val_bfs_episodes(scratch, prompts, val_pool):
    from PIL import Image

    pool, reached = val_pool
    out = scratch / "sft"
    ren = M.SceneRenderers("fake")
    o = M.build_order(pool, 1001, out)
    assert o["n_dedup"] == len(reached) and (out / "order_seed1001.json").is_file()
    man = M.build_tier(out, 1001, len(reached), prompts, ren, pool)
    rows = [json.loads(x) for x in open(out / "seed1001" / f"demos_{len(reached)}.jsonl")]
    assert len(rows) == sum(len(r["actions"]) for r in reached) == man["n_records"]
    by_ep = {r["episode_index"]: r for r in reached}
    sys0 = prompts.system()
    for row in rows:
        mm = row["meta"]
        rec = by_ep[mm["episode_index"]]
        t = mm["step"]
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"][0]["content"] == sys0
        user = row["messages"][1]["content"]
        assert user.count("<image>") == 1 and user.endswith("<image>") and len(row["images"]) == 1
        assert f"Current step: {t}\n" in user and f"Question: '{rec['question']}'" in user
        if t:
            assert f"- Last action: {rec['actions'][t - 1]}\n" in user
        else:
            assert "Last action" not in user
        assert "Collision" not in user
        assert row["messages"][2]["content"] == M.reply(rec["actions"][t]) and RU.regex_ok(row["messages"][2]["content"])
        pos = np.asarray(rec["poses"][t])[:3, 3]
        assert f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})" in user
        with Image.open(row["images"][0]) as im:
            assert im.size == (512, 512) and im.format == "PNG"
        assert E.sha256_file(Path(row["images"][0])) == mm["image_sha256"]
    assert rows[-1]["messages"][2]["content"].endswith("stop")
    # the trainer reads these records: one sample, one image part after the text
    from vpd.train import n2_train_vpd as T

    smp = T.samples(rows[0], 0)
    assert len(smp) == 1 and len(smp[0]["images"]) == 1
    parts = T.chat_messages(smp[0]["messages"])[1]["content"]
    assert [p["type"] for p in parts] == ["text", "image"]
    # images are reused (not re-rendered) by a second tier build
    mt = Path(rows[0]["images"][0]).stat().st_mtime_ns
    M.build_tier(out, 1001, 1, prompts, ren, pool)
    assert Path(rows[0]["images"][0]).stat().st_mtime_ns == mt
    with pytest.raises(ValueError):
        M.build_tier(out, 1001, len(reached) + 1, prompts, ren, pool)
    # ALL (round 4): this val pool has one start per (scene, question), so ALL == the whole order, same file
    m_all = M.build_tier(out, 1001, M.ALL_TIER, prompts, ren, pool)
    assert m_all["tier"] == "ALL" and m_all["n_episodes"] == m_all["n_from_order"] == len(reached)
    assert m_all["jsonl_sha256"] == man["jsonl_sha256"] and m_all["n_records"] == man["n_records"]
    assert Path(m_all["jsonl"]).name == "demos_ALL.jsonl"


@needs_repo
def test_make_sft_warmup(scratch, prompts):
    out = scratch / "sft_w"
    man = M.build_warmup(out, prompts, M.SceneRenderers("fake"), REPO)
    rows = [json.loads(x) for x in open(out / "warmup20.jsonl")]
    assert man["n"] == len(rows) == 20 and len({r["meta"]["scene_id"] for r in rows}) == 20
    for i, r in enumerate(rows):
        name = E.ACTIONS[i % 11]
        assert r["meta"]["action"] == name and r["messages"][2]["content"] == M.reply(name)
        u = r["messages"][1]["content"]
        assert f"Question: '{M.WARMUP_QUESTION.format(name=name)}'" in u and "Current step: 0\n" in u
        assert "Last action" not in u and Path(r["images"][0]).is_file()
    assert (out / "warmup_manifest.json").is_file()


def test_make_sft_refuses_default_out_on_login_tmp_copy():
    if not (E.LOGIN_TMP_REPO / "dataset").is_dir():
        pytest.skip("no login_tmp copy")
    with pytest.raises(SystemExit):
        M.main(["warmup", "--repo", str(E.LOGIN_TMP_REPO), "--renderer", "fake"])


# ============================================================================================================
# run.py (fake renderer + fake OpenAI server)
# ============================================================================================================
def _run(argv):
    return RU.main(argv)


@needs_repo
def test_closed_loop_constrained_collision_flag(scratch, prompts, val_eps):
    fake = RU.FakeServer()
    url = fake.start()
    rd = scratch / "run_c"
    try:
        rc = _run(["--start", "Q3S", "--tier", "raw", "--mode", "c", "--split", "val", "--limit", "3",
                   "--renderer", "fake", "--collides", "vpd.tests.test_e3vs_loop:block_moves", "--base-url", url,
                   "--run-dir", str(rd), "--audit-processor", "", "--concurrency", "2", "--repo", str(REPO)])
    finally:
        fake.stop()
    assert rc == 0 and not fake.problems
    reqs = fake.requests
    assert len(reqs) == 3 * 5
    assert all(r["regex"] == M.REGEX and r["n_images"] == 1 and r["max_tokens"] == 128 and r["temperature"] == 0.0
               and r["system"] == prompts.system() for r in reqs)
    s1 = [r["text"] for r in reqs if r["step"] == 1]
    assert len(s1) == 3 and all("- Last action: move_forward\n- Last action result: Collision occurred." in t
                                for t in s1)
    s2 = [r["text"] for r in reqs if r["step"] == 2]
    assert all("- Last action: turn_left\n" in t and "Collision" not in t for t in s2)
    s4 = [r["text"] for r in reqs if r["step"] == 4]
    assert all("Last action: move_up\n- Last action result: Collision" in t for t in s4)
    logs = {r["episode_index"]: r for r in map(json.loads, open(rd / "episodes.jsonl"))}
    eps = {e["episode_index"]: e for e in val_eps}
    assert len(logs) == 3
    for i, lg in logs.items():
        assert lg["status"] == "ok" and lg["stopped"] and lg["n_decisions"] == 5 and lg["stop_step"] == 4
        assert [s["action"] for s in lg["steps"]] == ["move_forward", "turn_left", "look_down", "move_up", "stop"]
        assert [s["collision"] for s in lg["steps"]] == [True, False, False, True, False]
        assert all(s["regex_ok"] and not s["error"] for s in lg["steps"])
        want, _ = G.rollout(eps[i]["start_c2w"], ["turn_left", "look_down"])
        assert np.allclose(lg["poses"][1], eps[i]["start_c2w"]) and np.allclose(lg["poses"][-1], want[-1])
        assert len(lg["poses"]) == 6
    res = json.loads((rd / "results.json").read_text())
    assert res["valid"] and res["n_expected"] == 3 and res["parse"]["regex_valid_rate"] == 1.0
    assert res["collision_rate"]["collisions"] == 6 and res["stop_rate"]["k"] == 3
    man = json.loads((rd / "manifest.json").read_text())
    assert man["decoding"]["regex"] == M.REGEX and man["mode"] == "c" and man["e3vs"]["pinned_commit"] == E.E3VS_COMMIT
    assert man["prompt"]["template_sha256"] == M.TEMPLATE_SHA256 and man["n_failed"] == 0
    assert man["collides"] == "vpd.tests.test_e3vs_loop:block_moves" and len(man["scenes"]) >= 1
    dump = [json.loads(x) for x in open(rd / "prompt_dump.jsonl")]
    assert [d["t"] for d in dump] == [0, 1, 2] and all((rd / d["messages"][1]["content"][1]["path"]).is_file()
                                                        for d in dump)
    assert len(list((rd / "images").glob("*/t*.png"))) == 15  # --save-images 3 (default) x 5 decisions
    with pytest.raises(SystemExit):  # an existing results.json is never overwritten
        _run(["--start", "Q3S", "--tier", "raw", "--mode", "c", "--split", "val", "--limit", "3", "--renderer",
              "fake", "--collides", "none", "--fake-server", "--run-dir", str(rd), "--repo", str(REPO)])


def _never_stop(step, text, regex):
    if regex is not None:
        raise AssertionError("unconstrained request carries a regex")
    return "I think I should keep searching" if step % 3 == 0 else "ACTION: turn_left"


@needs_repo
def test_closed_loop_unconstrained_cap_and_lenient(scratch, val_eps):
    fake = RU.FakeServer(_never_stop)
    url = fake.start()
    rd = scratch / "run_u"
    ids = f"{val_eps[5]['episode_index']},{val_eps[9]['episode_index']}"
    try:
        rc = _run(["--start", "Q3S", "--tier", "100", "--mode", "u", "--split", "val", "--episodes", ids,
                   "--renderer", "fake", "--collides", "none", "--base-url", url, "--run-dir", str(rd),
                   "--audit-processor", "", "--save-images", "0", "--repo", str(REPO)])
    finally:
        fake.stop()
    assert rc == 0 and not fake.problems and all(r["regex"] is None for r in fake.requests)
    logs = [json.loads(x) for x in open(rd / "episodes.jsonl")]
    assert len(logs) == 2
    for lg in logs:
        assert lg["n_decisions"] == E.MAX_DECISIONS == 25 and not lg["stopped"]
        errs = [s["t"] for s in lg["steps"] if s["error"]]
        assert errs == list(range(0, 25, 3))  # fallback move_forward with error=True
        assert all(s["action"] == ("move_forward" if s["t"] % 3 == 0 else "turn_left") for s in lg["steps"])
        assert not any(s["regex_ok"] for s in lg["steps"])
    res = json.loads((rd / "results.json").read_text())
    assert res["parse"]["decisions"] == 50 and res["parse"]["lenient_valid"] == 50 - 18
    assert res["parse"]["regex_valid_rate"] == 0.0 and res["stop_rate"]["k"] == 0 and res["decisions_mean"] == 25
    assert not (rd / "images").exists() or not list((rd / "images").glob("*/*.png"))


def test_run_refusals():
    with pytest.raises(SystemExit):
        RU.main(["--start", "Q3S", "--tier", "raw", "--mode", "u", "--collides", "none"])  # gsplat needs collisions
    with pytest.raises(SystemExit):
        RU.main(["--start", "Q3S", "--tier", "raw", "--mode", "u", "--adapter-meta", "x/meta.json"])
    for bad in ("x", "u,u", "", "u,c,x"):
        with pytest.raises(SystemExit):
            RU.parse_modes(bad)
    assert RU.parse_modes("u,c") == ["u", "c"] and RU.parse_modes("c") == ["c"]
    with pytest.raises(SystemExit):  # one run dir cannot hold two modes
        RU.main(["--start", "Q3S", "--tier", "raw", "--mode", "u,c", "--renderer", "fake", "--collides", "none",
                 "--fake-server", "--run-dir", "x"])


def test_chat_retries_truncated_responses(monkeypatch):
    """A truncated / malformed HTTP response (http.client.HTTPException) is retried like a connection error."""
    import http.client
    import io
    import urllib.error

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    body = json.dumps({"choices": [{"message": {"content": "<action>stop</action>"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()
    calls, errs = [], []

    def fake(req, timeout=None):
        calls.append(req.full_url)
        if errs:
            raise errs.pop(0)
        return Resp(body)

    monkeypatch.setattr(RU.urllib.request, "urlopen", fake)
    monkeypatch.setattr(RU.time, "sleep", lambda s: None)
    errs[:] = [http.client.IncompleteRead(b"{\"cho"), http.client.BadStatusLine("garbage")]
    out = RU.chat("http://x/v1", "m", [], 8)
    assert out["content"] == "<action>stop</action>" and out["prompt_tokens"] == 5 and len(calls) == 3
    calls.clear()
    errs[:] = [http.client.IncompleteRead(b"")] * 3
    with pytest.raises(RuntimeError, match="after 3 tries: IncompleteRead"):
        RU.chat("http://x/v1", "m", [], 8)
    assert len(calls) == 3
    calls.clear()  # a 4xx is not retried
    errs[:] = [urllib.error.HTTPError("http://x/v1/chat/completions", 400, "bad", {}, None)]
    with pytest.raises(RuntimeError, match="HTTPError"):
        RU.chat("http://x/v1", "m", [], 8)
    assert len(calls) == 1


def test_scene_cache_never_evicts_a_scene_in_flight():
    """Scene-major episodes, more workers than scenes in flight, variable lengths: every scene loads once."""
    import random
    import threading
    from concurrent.futures import ThreadPoolExecutor

    loads = []

    def factory(s):
        loads.append(s)
        return object()

    rng = random.Random(0)
    eps = [(f"s{k:02d}", rng.choice([1, 1, 2, 3, 25])) for k in range(12) for _ in range(rng.randint(2, 9))]
    seen, lock = {}, threading.Lock()

    def episode(scene, n):
        with cache.hold(scene) as obj:
            for _ in range(n):
                time.sleep(0.0005)
                with lock:
                    seen.setdefault(scene, set()).add(id(obj))
                assert cache.d.get(scene) is obj  # pinned: still resident

    for cap in (1, 4):
        loads.clear()
        seen.clear()
        cache = RU.SceneCache(factory, cap)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda e: episode(*e), eps))
        assert sorted(loads) == sorted({s for s, _ in eps}) and all(len(v) == 1 for v in seen.values())
        st = cache.stats()
        assert st["reloads"] == 0 and st["n_loads"] == 12 and len(cache.d) <= cap and not cache.refs
    # the counter does see thrash: with no idle slot, sequential episodes of one scene reload it every time
    loads.clear()
    cache = RU.SceneCache(factory, 0)
    for _ in range(3):
        episode("a", 1)
    assert loads == ["a", "a", "a"] and cache.stats()["reloads"] == 2
    # a failed load is not cached and raises in the caller
    calls = []

    def flaky(s):
        calls.append(s)
        if len(calls) == 1:
            raise OSError("boom")
        return s

    c2 = RU.SceneCache(flaky, 2)
    with pytest.raises(OSError):
        c2.acquire("z")
    with c2.hold("z") as v:
        assert v == "z"
    assert calls == ["z", "z"] and not c2.loading and not c2.refs


def _scene_major_pick(val_eps, n_scenes=3, per_scene=5):
    by = {}
    for e in val_eps:
        by.setdefault(e["scene_id"], []).append(e)
    scenes = sorted(by)[:n_scenes]
    return [e for s in scenes for e in sorted(by[s], key=lambda e: e["episode_index"])[:per_scene]], scenes


@needs_repo
def test_session_u_and_c_shares_server_and_scene_loads(scratch, val_eps):
    picked, scenes = _scene_major_pick(val_eps)
    length = {e["question"]: (25 if i % 3 == 0 else 1 + i % 4) for i, e in enumerate(picked)}

    def policy(step, text, regex):
        n = next(v for q, v in length.items() if q in text)
        return M.reply("turn_left" if step < n - 1 else "stop")

    fake = RU.FakeServer(policy)
    url = fake.start()
    root = scratch / "session"
    ids = ",".join(str(e["episode_index"]) for e in picked)
    try:
        rc = _run(["--start", "Q3S", "--tier", "100", "--mode", "u,c", "--split", "val", "--episodes", ids,
                   "--renderer", "fake", "--collides", "vpd.tests.test_e3vs_loop:block_moves", "--base-url", url,
                   "--out-root", str(root), "--audit-processor", "", "--save-images", "1", "--concurrency", "8",
                   "--scene-cache", "1", "--repo", str(REPO)])
    finally:
        fake.stop()
    assert rc == 0 and not fake.problems
    want = sum(min(length[e["question"]], 25) for e in picked)
    assert sum(r["regex"] == M.REGEX for r in fake.requests) == want
    assert sum(r["regex"] is None for r in fake.requests) == want
    for m in ("u", "c"):
        rd = root / RU.eval_name("Q3S", "100", m)
        res = json.loads((rd / "results.json").read_text())
        man = json.loads((rd / "manifest.json").read_text())
        assert res["valid"] and res["n_expected"] == len(picked) and res["mode"] == m
        assert man["session"] == {"modes": ["u", "c"], "evals": [RU.eval_name("Q3S", "100", x) for x in "uc"]}
        assert man["decoding"]["regex"] == (M.REGEX if m == "c" else None)
        sl = man["scene_loads"]
        # each scene loaded exactly once for BOTH modes, renderers and colliders alike
        assert sl["renderers"]["n_loads"] == len(scenes) and sl["renderers"]["reloads"] == 0
        assert sl["colliders"]["n_loads"] == len(scenes) and set(sl["renderers"]["load_seconds"]) == set(scenes)
        logs = [json.loads(x) for x in open(rd / "episodes.jsonl")]
        assert sorted(lg["n_decisions"] for lg in logs) == sorted(min(length[e["question"]], 25) for e in picked)
        assert len({p.parent.name for p in (rd / "images").glob("*/t*.png")}) == 1  # --save-images 1 per mode
        assert [d["t"] for d in map(json.loads, open(rd / "prompt_dump.jsonl"))] == \
            list(range(min(3, length[picked[0]["question"]])))
        assert man["gsplat"] is None  # fake renderer


def fake_mesh_validation(shas: dict, ok: bool = True) -> dict:
    """A validation.json that mesh_collision.check / load_collider / the gate accept (pass, definition, key)."""
    return {"pass": ok, "scenes": {s: {"sha256": h} for s, h in shas.items()},
            "mesh_set_sha256": MV.mesh_set_sha256(shas), "tolerance_m": MC.TOLERANCE_M,
            "in_obstacle_m": MC.IN_OBSTACLE_M, "pooled": {"threshold": MC.VALID_MAX_FRAC},
            "official": {"sha256": MC.METRICS_SHA256, "commit": E.E3VS_COMMIT, "open3d_version": MC.O3D_VERSION},
            "frame_ok": True, "frame_warnings": []}


@needs_repo
def test_run_preflight_refuses_before_any_server(scratch, monkeypatch):
    """Data-independent: a tmp data root without meshes, mesh validation or PLYs (the real ones may exist)."""
    empty = scratch / "preflight_data"
    monkeypatch.setenv("VPD_E3VS_MESH_ROOT", str(empty / "mesh"))
    monkeypatch.setenv("VPD_E3VS_MESH_MANIFEST", str(empty / "mesh" / "MANIFEST.json"))
    monkeypatch.setenv("VPD_E3VS_MESH_VAL_DIR", str(empty / "mesh"))
    monkeypatch.setattr(RD.paths, "E3VS_DATA", empty / "data")  # RD.ply_path / ply_sha256 read it at call time
    rd = scratch / "preflight"
    base = ["--start", "Q3S", "--tier", "raw", "--mode", "u", "--split", "val", "--limit", "2", "--fake-server",
            "--run-dir", str(rd), "--repo", str(REPO), "--audit-processor", ""]
    with pytest.raises(SystemExit, match="without a collision mesh") as ei:
        RU.main(base + ["--renderer", "fake"])  # default collider = the frozen mesh model
    assert "no passing mesh validation" in str(ei.value)
    with pytest.raises(SystemExit, match="PLY"):
        RU.main(base + ["--renderer", "gsplat"])
    assert not rd.exists()  # refused before anything was written or started
    # meshes present but not the validated ones: refused too (the mesh set key differs)
    eps = RU.select_episodes("val", REPO, 2, None)
    shas = {}
    for s in sorted({e["scene_id"] for e in eps}):
        p = MC.mesh_path(s)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"not the validated mesh " + s.encode())
        shas[s] = "0" * 64
    MC.validation_path().parent.mkdir(parents=True, exist_ok=True)  # the mesh root holds the validation too
    MC.validation_path().write_text(json.dumps(fake_mesh_validation(shas)))
    with pytest.raises(SystemExit, match="not the validated one"):
        RU.main(base + ["--renderer", "fake"])
    assert not rd.exists()


def test_gsplat_facts_copy_the_probe(scratch):
    p = scratch / "probe.json"
    p.write_text(json.dumps({"compiled_now": False, "extension_seconds": 1.5}))
    g = RU.gsplat_facts("gsplat", p)
    assert g["probe"] == {"compiled_now": False, "extension_seconds": 1.5} and g["probe_path"] == str(p)
    assert RU.gsplat_facts("gsplat", scratch / "missing.json")["probe"] is None
    assert RU.gsplat_facts("fake", p) is None


_SIGTERM_CHILD = r"""
import json, os, signal, sys, threading, time
from vpd.e3vs import make_sft as M, run as RU
rd, repo = sys.argv[1], sys.argv[2]
def slow(step, text, regex):
    time.sleep(0.3)
    return M.reply("turn_left")
fake = RU.FakeServer(slow)
url = fake.start()
threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
t0 = time.time()
code = None
try:
    RU.main(["--start", "Q3S", "--tier", "raw", "--mode", "u", "--split", "val", "--limit", "3", "--renderer", "fake",
             "--collides", "none", "--base-url", url, "--run-dir", rd, "--audit-processor", "", "--concurrency", "1",
             "--repo", repo])
except SystemExit as e:
    code = e.code
dt = time.time() - t0
time.sleep(1.0)
fake.stop()
print(json.dumps({"code": code, "seconds": dt, "requests": len(fake.requests)}))
"""


@needs_repo
def test_sigterm_aborts_the_run_promptly(scratch):
    rd = scratch / "sigterm"
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", _SIGTERM_CHILD, str(rd), str(REPO)], capture_output=True,
                       text=True, timeout=120, env=env)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    # 3 episodes x 25 decisions x 0.3 s = 22 s without the abort
    assert out["code"] == 143 and out["seconds"] < 6 and out["requests"] < 15, (out, r.stderr[-2000:])
    assert not (rd / "results.json").exists() and not (rd / "results_invalid.json").exists()
    logs = [json.loads(x) for x in open(rd / "episodes.jsonl")]
    assert len(logs) == 1 and logs[0]["status"] == "error" and "Aborted" in logs[0]["error"]


@needs_repo
def test_image_token_audit_q3s(scratch):
    """B4/E0: a 512x512 PNG is 258 tokens for the Q3S processor (processor files only)."""
    _torch()
    pytest.importorskip("transformers")
    from vpd import starts

    pdir = starts.processor_dir("Q3S")
    if not (Path(pdir) / "preprocessor_config.json").exists():
        pytest.skip("no Q3S processor files")
    rd = scratch / "audit"
    img = RD.FakeRenderer().render_rgb(look_c2w([1, 1, 1], [1, 0, 0]))
    RD.write_png(img, rd / "images" / "00000" / "t00.png")
    try:
        au = RU.image_token_audit(rd, Path(pdir))
    except OSError as e:  # e.g. root-only tokenizer blobs
        pytest.skip(f"processor not readable: {e}")
    assert au["tokens_per_image"] == [RU.EXPECTED_IMAGE_TOKENS] and au["samples"][0]["grid_thw"] == [1, 32, 32]


# ============================================================================================================
# summary.py
# ============================================================================================================
def _synthetic_run(rd: Path, eps: list[dict], cases: dict, expected="all"):
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "manifest.json").write_text(json.dumps({"eval": "pilot__X__100__u", "start": "X", "tier": "100",
                                                  "mode": "u", "split": "val", "episode_indices": expected}))
    with open(rd / "episodes.jsonl", "w") as f:
        for ep in eps:
            kind = cases.get(ep["episode_index"], "fail")
            if kind == "missing":
                continue
            goal = ep["goals"][sorted(ep["goals"])[0]]
            final = goal if kind in ("stop_ok", "cap_ok") else ep["start_c2w"]
            n = 3 if kind in ("stop_ok", "fail") else 25
            steps = [{"t": t, "action": "move_forward", "error": t == 0, "empty": False, "regex_ok": t != 0,
                      "collision": t == 1, "finish_reason": "stop", "reply": ""} for t in range(n)]
            stopped = kind in ("stop_ok", "fail")
            if stopped:
                steps[-1]["action"] = "stop"
            f.write(json.dumps({"episode_index": ep["episode_index"], "scene_id": ep["scene_id"],
                                "question_type": ep["question_type"], "status": "error" if kind == "error" else "ok",
                                "steps": steps, "poses": [ep["start_c2w"].tolist()] * n + [np.asarray(final).tolist()],
                                "stopped": stopped, "n_decisions": n}) + "\n")


@needs_repo
def test_score_synthetic(scratch, val_eps):
    eps = val_eps[:6]
    ids = [e["episode_index"] for e in eps]
    cases = {ids[0]: "stop_ok", ids[1]: "cap_ok", ids[2]: "fail", ids[3]: "fail", ids[4]: "stop_ok",
             ids[5]: "fail"}
    rd = scratch / "score_ok"
    _synthetic_run(rd, eps, cases, expected=ids)
    opt = scratch / "opt_len.json"
    opt.write_text(json.dumps({str(ids[0]): 2, str(ids[1]): 4, str(ids[2]): None, str(ids[4]): 5}))
    floors = scratch / "floors.json"
    floors.write_text(json.dumps({"collides": "x", "gate_floor": {"rate": 0.1, "k": 1, "n": 10, "seq": ["move_up"],
                                                                  "definition": "d"}}))
    res = S.score(rd, REPO, floors=floors, opt_len=opt)
    assert res["valid"] and (rd / "results.json").is_file() and not (rd / "results_invalid.json").exists()
    assert res["pooled"]["k"] == 3 and res["pooled"]["n"] == 6
    assert res["pooled"]["wilson95"] == [round(x, 6) for x in S.wilson(3, 6)]
    assert res["stop_required"]["k"] == 2 and res["stop_rate"]["k"] == 5
    dec = 3 + 25 + 3 + 3 + 3 + 3
    assert res["parse"]["decisions"] == dec and res["parse"]["lenient_valid"] == dec - 6
    assert res["collision_rate"]["collisions"] == 6
    # SPL: ep0 2 / max(2, 2) = 1; ep1 4 / max(4, 25); ep4 5 / max(5, 2) = 1; ep2 failure 0; ep3 / ep5 no optimum
    assert res["spl"]["n"] == 3 and abs(res["spl"]["mean"] - round((1 + 4 / 25 + 1) / 3, 6)) < 1e-6
    assert res["floor"]["rate"] == 0.1 and res["final_dist_m"]["median"] is not None
    b1 = res["bootstrap_scene"]
    assert b1 == S.score(rd, REPO, write=False)["bootstrap_scene"] and b1["ci95"][0] <= 0.5 <= b1["ci95"][1]
    assert sum(v["n"] for v in res["per_question_type"].values()) == 6


@needs_repo
def test_score_invalid_when_missing_or_failed(scratch, val_eps):
    eps = val_eps[:4]
    ids = [e["episode_index"] for e in eps]
    rd = scratch / "score_bad"
    _synthetic_run(rd, eps, {ids[0]: "stop_ok", ids[1]: "missing", ids[2]: "error"}, expected=ids)
    res = S.score(rd, REPO)
    assert not res["valid"] and (rd / "results_invalid.json").is_file() and not (rd / "results.json").exists()
    assert res["episodes_without_log"] == 1 and res["episodes_failed"] == 1 and res["pooled"]["n"] == 4


def test_scene_bootstrap():
    sc = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    hits = [True] * 10 + [False] * 10 + [True, False] * 5
    b = S.scene_bootstrap(sc, hits, n_boot=2000, seed=0)
    assert b["n_scenes"] == 3 and 0.0 <= b["ci95"][0] < 0.5 < b["ci95"][1] <= 1.0
    assert b == S.scene_bootstrap(sc, hits, n_boot=2000, seed=0)
    assert S.scene_bootstrap(["a"] * 4, [True] * 4, 100)["ci95"] == [1.0, 1.0]


MESHES = {"sc1": "a" * 64, "sc2": "b" * 64}


SYS_SHA = "p" * 64


def _gate_tree(root: Path, rate_u=0.3, regex_c=1.0, sha=("s", "s", "s"), n=377, lo=0.25, parity_equal=True):
    key = MV.mesh_set_sha256(MESHES)
    for s, cs in zip(S.PILOT_STARTS, sha):
        ad = root / "pilot" / s / "adapters"
        pts = {}
        for t in ("0W", "100"):
            (ad / t).mkdir(parents=True, exist_ok=True)
            (ad / t / "adapter.pt").write_bytes(f"{s}-{t}".encode())
            pts[t] = E.sha256_file(ad / t / "adapter.pt")
        (ad / "0W" / "meta.json").write_text(json.dumps({"code_sha256": cs}))
        (ad / "100" / "meta.json").write_text(json.dumps({"code_sha256": cs, "init_adapter": {
            "dir": str(ad / "0W"), "sha256": pts["0W"]}}))
        for tier, mode in S.REQUIRED_EVALS:
            d = S.eval_dir(root / "pilot", s, tier, mode)
            d.mkdir(parents=True, exist_ok=True)
            res = {"n_expected": n, "split": "test", "episodes_without_log": 0, "episodes_failed": 0,
                   "pooled": {"k": 1, "n": n, "rate": rate_u, "wilson95": [lo, 0.4]},
                   "parse": {"lenient_valid_rate": 0.97, "regex_valid_rate": regex_c if mode == "c" else 0.0}}
            (d / "results.json").write_text(json.dumps(res))
            (d / "audit.json").write_text(json.dumps({"tokens_per_image": [258],
                                                      "prompt_parity": {"n": 3, "equal": parity_equal}}))
            (d / "manifest.json").write_text(json.dumps({"collides": S.FROZEN_COLLIDES, "scenes": {
                sc: {"mesh_sha256": h} for sc, h in MESHES.items()}, "collision": {"mesh_set_sha256": key},
                "prompt": {"system_sha256": SYS_SHA},
                "adapter": None if tier == "raw" else {"meta": str(ad / tier / "meta.json"),
                                                       "adapter_sha256": pts[tier]}}))
    (root / "parity").mkdir(exist_ok=True)
    (root / "parity" / "parity_x.json").write_text(json.dumps({"pass": True, "official": {"min_psnr": 100.0}}))
    (root / "validation.json").write_text(json.dumps(fake_mesh_validation(MESHES)))
    (root / "floors.json").write_text(json.dumps({"collides": S.FROZEN_COLLIDES,
                                                  "gate_floor": {"rate": 0.141, "k": 53, "n": 377, "seq": []}}))
    (root / "prep_state").mkdir(exist_ok=True)
    (root / "prep_state" / "floors").write_text(key + "\n")
    (root / "prep_state" / "sft").write_text(key + "\n")
    (root / "sft" / "seed1001").mkdir(parents=True, exist_ok=True)
    for f in ("warmup_manifest.json", "seed1001/demos_100_manifest.json"):
        (root / "sft" / f).write_text(json.dumps({"prompt": {"system_sha256": SYS_SHA}}))


def _gate(root):
    return S.gate(root / "pilot", floors=root / "floors.json", parity_dir=root / "parity",
                  validation=root / "validation.json", prep_state=root / "prep_state", sft_dir=root / "sft")


def _e0(root) -> str:
    g = _gate(root)
    assert not g["E0"]["pass"] and g["decision"].startswith("E0")
    return " | ".join(g["E0"]["problems"])


def test_gate_rules(scratch):
    root = scratch / "gate_ok"
    _gate_tree(root)
    g = _gate(root)
    assert all(g[k]["pass"] for k in ("E0", "E1", "E2", "E3")) and g["decision"].startswith("E0-E3 pass")
    assert g["prompt_parity_all_equal"] is True and len(g["audit"]) == 12
    assert g["provenance"]["sft_system_sha256"] == [SYS_SHA]
    assert g["provenance"]["sft_stamp"] == MV.mesh_set_sha256(MESHES)
    assert S.FROZEN_COLLIDES == "vpd.e3vs.mesh_collision:load_collider"
    assert set(g["provenance"]["adapter_sha256"]) == set(S.PILOT_STARTS)
    r2 = scratch / "gate_e2"
    _gate_tree(r2, regex_c=0.999)
    g2 = _gate(r2)
    assert g2["E0"]["pass"] and g2["E1"]["pass"] and not g2["E2"]["pass"] and g2["decision"].startswith("E2")
    r3 = scratch / "gate_e3"
    _gate_tree(r3, rate_u=0.85)
    assert not _gate(r3)["E3"]["pass"]
    r1 = scratch / "gate_e1"
    _gate_tree(r1, lo=0.141)  # the lower bound must be strictly above the floor
    assert not _gate(r1)["E1"]["pass"] and _gate(r1)["decision"].startswith("E1")
    r0 = scratch / "gate_e0"
    _gate_tree(r0, sha=("a", "a", "b"))
    assert "code_sha256" in _e0(r0)
    r0b = scratch / "gate_e0b"
    _gate_tree(r0b, n=300)
    assert not _gate(r0b)["E0"]["pass"]
    # prompt-token parity is descriptive: reported, never blocking
    rp = scratch / "gate_pp"
    _gate_tree(rp, parity_equal=False)
    gp = _gate(rp)
    assert gp["E0"]["pass"] and gp["prompt_parity_all_equal"] is False


def _patch_manifest(d: Path, **kv):
    m = json.loads((d / "manifest.json").read_text())
    m.update(kv)
    (d / "manifest.json").write_text(json.dumps(m))


def test_gate_reports_the_frame_facts(scratch):
    """S2: the frame facts of the validation and of every eval manifest (frame_ok, warnings, suspect scenes, an
    ack-frame record: note, date, sha256) are in E0.facts, descriptively: never an E0 condition."""
    root = scratch / "gate_frame_ok"
    _gate_tree(root)
    fr = _gate(root)["E0"]["facts"]["collision_frame"]
    assert fr["validation"]["frame_ok"] is True and fr["validation"]["ack"] is None and fr["eval_ack_sha256"] == []
    assert len(fr["evals"]) == 12
    rf = scratch / "gate_frame_ack"
    _gate_tree(rf)
    _patch_validation(rf, frame_ok=False, frame_warnings=["pooled: only 50.0% of the iPhone camera centres ..."])
    MC.ack_frame("user 2026-11-13: the iPhone poses are in the mesh frame (test)", rf)  # rf/frame_ack.json
    facts = MC.frame_facts(json.loads((rf / "validation.json").read_text()), rf)
    assert facts["ack"]["covers"] and facts["ack"]["sha256"] == E.sha256_file(rf / "frame_ack.json")
    key = MV.mesh_set_sha256(MESHES)
    for s in S.PILOT_STARTS:
        for tier, mode in S.REQUIRED_EVALS:
            _patch_manifest(S.eval_dir(rf / "pilot", s, tier, mode), collision={"mesh_set_sha256": key, "frame": facts})
    g = _gate(rf)
    assert g["E0"]["pass"], g["E0"]["problems"]  # descriptive only
    fr = g["E0"]["facts"]["collision_frame"]
    assert fr["validation"]["frame_ok"] is False and fr["validation"]["ack"]["note"].startswith("user 2026-11-13")
    assert fr["eval_ack_sha256"] == [facts["ack"]["sha256"]] and all(f["ack"]["covers"] for f in fr["evals"].values())


def _patch_validation(root: Path, **kv):
    v = json.loads((root / "validation.json").read_text())
    v.update(kv)
    (root / "validation.json").write_text(json.dumps(v))


def test_gate_e0_audit_and_collision_provenance(scratch):
    """B4: a missing / failed / empty audit fails E0; B8: the floor and every eval must use the frozen collider on
    the validated meshes (the official mesh model since the PREREG section-16 row of 2026-11-12)."""
    d = lambda root: S.eval_dir(root / "pilot", "Q3S", "100", "u")  # noqa: E731
    cases = {
        "no_audit": (lambda r: (d(r) / "audit.json").unlink(), "no audit.json"),
        "audit_error": (lambda r: (d(r) / "audit.json").write_text(json.dumps({"error": "OSError: x"})),
                        "audit failed"),
        "audit_empty": (lambda r: (d(r) / "audit.json").write_text(json.dumps({"tokens_per_image": []})),
                        "no image token count"),
        "audit_wrong": (lambda r: (d(r) / "audit.json").write_text(json.dumps({"tokens_per_image": [258, 330]})),
                        "image tokens"),
        "floor_nocoll": (lambda r: (r / "floors.json").write_text(json.dumps(
            {"collides": None, "gate_floor": {"rate": 0.141, "k": 53, "n": 377, "seq": []}})), "not the frozen"),
        "floor_stamp": (lambda r: (r / "prep_state" / "floors").write_text("other\n"), "not stamped"),
        "floor_nostamp": (lambda r: (r / "prep_state" / "floors").unlink(), "not stamped"),
        "eval_mesh": (lambda r: (d(r) / "manifest.json").write_text(json.dumps(
            {"collides": S.FROZEN_COLLIDES, "scenes": {"sc1": {"mesh_sha256": "c" * 64},
                                                       "sc2": {"mesh_sha256": MESHES["sc2"]}}})),
                      "validated collision meshes"),
        "eval_mesh_set": (lambda r: _patch_manifest(d(r), collision={"mesh_set_sha256": "k" * 64}),
                          "eval mesh set"),
        "eval_old_occ": (lambda r: _patch_manifest(d(r), collides="vpd.e3vs.occupancy:load_collider"),
                         "not the frozen vpd.e3vs.mesh_collision:load_collider"),
        "eval_collider": (lambda r: (d(r) / "manifest.json").write_text(json.dumps(
            {"collides": None, "scenes": {}})), "collider None"),
        "val_failed": (lambda r: _patch_validation(r, **{"pass": False}), "validation missing or failed"),
        "val_missing": (lambda r: (r / "validation.json").unlink(), "validation missing or failed"),
        "val_other_code": (lambda r: _patch_validation(r, official={"sha256": "x", "commit": E.E3VS_COMMIT}),
                           "not a validation of the frozen collision model"),
        "val_tolerance": (lambda r: _patch_validation(r, tolerance_m=0.05), "not a validation of the frozen"),
        "eval_nomanifest": (lambda r: (d(r) / "manifest.json").unlink(), "no manifest.json"),
        # provenance of the weights and of the SFT data
        "adapter_changed": (lambda r: (r / "pilot" / "Q3S" / "adapters" / "100" / "adapter.pt").write_bytes(b"new"),
                            "Q3S 100/u: eval adapter_sha256"),
        "init_not_0W": (lambda r: (r / "pilot" / "E2" / "adapters" / "100" / "meta.json").write_text(json.dumps(
            {"code_sha256": "s", "init_adapter": {"sha256": "x" * 64}})), "E2: adapter 100 not initialised"),
        "init_missing": (lambda r: (r / "pilot" / "E2" / "adapters" / "100" / "meta.json").write_text(json.dumps(
            {"code_sha256": "s", "init_adapter": None})), "E2: adapter 100 not initialised"),
        "raw_adapter": (lambda r: _patch_manifest(S.eval_dir(r / "pilot", "Q3B", "raw", "u"),
                                                  adapter={"meta": "x", "adapter_sha256": "y"}),
                        "Q3B raw/u: raw eval carries an adapter"),
        "eval_no_adapter": (lambda r: _patch_manifest(S.eval_dir(r / "pilot", "Q3B", "0W", "u"), adapter=None),
                            "Q3B 0W/u: eval adapter_sha256 None"),
        "sft_stamp": (lambda r: (r / "prep_state" / "sft").write_text("other\n"), "SFT data not stamped"),
        "sft_nostamp": (lambda r: (r / "prep_state" / "sft").unlink(), "SFT data not stamped"),
        "sft_manifest": (lambda r: (r / "sft" / "warmup_manifest.json").unlink(), "missing, or no prompt"),
        "sft_prompt": (lambda r: (r / "sft" / "seed1001" / "demos_100_manifest.json").write_text(json.dumps(
            {"prompt": {"system_sha256": "q" * 64}})), "SFT manifests disagree on the system prompt"),
        "eval_prompt": (lambda r: _patch_manifest(d(r), prompt={"system_sha256": "q" * 64}),
                        "Q3S 100/u: eval system prompt sha256"),
    }
    for name, (break_it, msg) in cases.items():
        root = scratch / f"gate_{name}"
        _gate_tree(root)
        break_it(root)
        assert msg in _e0(root), name


# ============================================================================================================
# sco payloads
# ============================================================================================================
@pytest.mark.parametrize("name", ["e3vs_prep.sh", "e3vs_start.sh", "e3vs_cal.sh"])
def test_job_scripts_parse(name):
    r = subprocess.run(["bash", "-n", str(JOBS / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _dry(script: str, args: list[str], run: Path, **extra) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("RUN", "VPD_E3VS_REPO", "VPD_E3VS_OCC_DIR", "VPD_MERGE_DIR",
                                                            "VPD_FAKE_FREE_GIB", "VPD_DRY_FAIL",
                                                            "VPD_JOB_DEADLINE_S", "VPD_E3VS_MESH_ROOT",
                                                            "VPD_E3VS_MESH_MANIFEST", "VPD_E3VS_MESH_VAL_DIR")}
    env.update(VPD_DRYRUN="1", PYTHONDONTWRITEBYTECODE="1", **extra)
    return subprocess.run(["bash", str(JOBS / script), *args, str(run)], capture_output=True, text=True, env=env,
                          timeout=300)


def _helpers(name: str) -> str:
    t = (JOBS / name).read_text()
    return t[t.index("# >>> timed-run helpers"):t.index("# <<< timed-run helpers")]


def test_timed_run_helpers_deadline_and_sigterm(scratch):
    """Both payloads share the helpers: a step gets at most the time left before the job deadline; SIGTERM is
    forwarded to the running step's process group, the cut step gets a steps.jsonl line, the EXIT handlers run."""
    h = _helpers("e3vs_prep.sh")
    assert h == _helpers("e3vs_start.sh") == _helpers("e3vs_cal.sh") and "timeout -k 30" in h
    pre = "set -uo pipefail\nRESERVE=0\n"
    s1 = pre + "DEADLINE=$(( $(date +%s) + 2 ))\nMIN_STEP_S=0\n" + h + \
        "tx sleep 30; echo rc1=$?\nDEADLINE=$(date +%s)\nMIN_STEP_S=60\ntx true; echo rc2=$?\n"
    t0 = time.time()
    r = subprocess.run(["bash", "-c", s1], capture_output=True, text=True, timeout=60)
    assert "rc1=124" in r.stdout and "rc2=124" in r.stdout and "not started" in r.stderr, r.stdout + r.stderr
    assert time.time() - t0 < 10
    tag = f"37.{uuid.uuid4().int % 10 ** 6:06d}"
    s2 = pre + "DEADLINE=$(( $(date +%s) + 100 ))\nMIN_STEP_S=0\n" + h + \
        f"trap 'echo EXIT_RAN' EXIT\ntx bash -c 'sleep {tag}; echo NOT_KILLED'\necho AFTER\n"
    p = subprocess.Popen(["bash", "-c", s2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.5)
    t0 = time.time()
    p.terminate()
    out, err = p.communicate(timeout=60)
    assert p.returncode == 143 and "EXIT_RAN" in out and "AFTER" not in out and "NOT_KILLED" not in out, (out, err)
    assert time.time() - t0 < 10 and "SIGTERM received" in err
    time.sleep(0.5)
    left_over = subprocess.run(["pgrep", "-f", f"sleep {tag}"], capture_output=True, text=True)
    assert left_over.stdout.strip() == "", "the step's process survived the SIGTERM"
    # step + vpd_step (common_env.sh): the step a SIGTERM cuts is recorded with rc 143 and the signal
    run = scratch / "helpers_run"
    run.mkdir()
    tag2 = f"38.{uuid.uuid4().int % 10 ** 6:06d}"
    s3 = f"RUN={run}\nsource {JOBS / 'common_env.sh'}\nRESERVE=0\nDEADLINE=$(( $(date +%s) + 100 ))\n" \
         f"MIN_STEP_S=0\n" + h + f"step quick true\nstep cut_me tx bash -c 'sleep {tag2}'\necho AFTER\n"
    p = subprocess.Popen(["bash", "-c", s3], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.5)
    p.terminate()
    out, err = p.communicate(timeout=60)
    assert p.returncode == 143 and "AFTER" not in out and "stopping the running step cut_me" in err, (out, err)
    lines = [json.loads(x) for x in open(run / "steps.jsonl")]
    assert [(x["name"], x["rc"], x.get("signal")) for x in lines] == [("quick", 0, None), ("cut_me", 143, "TERM")]
    assert lines[1]["secs"] >= 1 and lines[1]["start"] <= lines[1]["end"]
    time.sleep(0.5)
    assert subprocess.run(["pgrep", "-f", f"sleep {tag2}"], capture_output=True, text=True).stdout.strip() == ""


def _dry_cmds(out: str) -> list[str]:
    return [line[6:] for line in out.splitlines() if line.startswith("[dry] ")]


def test_prep_payload_dry_run(scratch):
    """Data-independent: an empty E3VS root (nothing to resume), so every step's command is printed."""
    run = scratch / "prep_run"
    root = scratch / "prep_dry_e3vs"
    root.mkdir()
    r = _dry("e3vs_prep.sh", [], run, VPD_E3VS_ROOT=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _dry_cmds(r.stdout)
    want = ["-m vpd.e3vs.render --probe --out " + str(root / "parity" / "gsplat_probe_prep.json"),
            "-m vpd.e3vs.render --parity 00dd871005",
            "-m vpd.e3vs.mesh_collision validate --scenes all",
            "-m vpd.e3vs.bfs_demos --split train --procs 12 --collides vpd.e3vs.mesh_collision:load_collider "
            "--tag mesh",
            "-m vpd.e3vs.bfs_demos --split val --procs 12",
            "-m vpd.e3vs.bfs_demos --split test --procs 12",
            "-m vpd.e3vs.floors --collides vpd.e3vs.mesh_collision:load_collider --tag mesh",
            "-m vpd.e3vs.make_sft all --seed 1001 --tiers 100 --overwrite"]
    pos = []
    for w in want:
        hit = [i for i, c in enumerate(cmds) if w in c]
        assert hit, (w, cmds)
        pos.append(hit[0])
    assert pos == sorted(pos)
    assert all(c.startswith(str(paths.PY_Q3)) for c in cmds if " -m vpd.e3vs." in c)
    assert [c.split()[1].split("/")[-1] for c in cmds if c.startswith("stamp ")] == \
        ["bfs_train", "bfs_val", "bfs_test", "floors", "sft"]
    steps = [json.loads(x) for x in open(run / "steps.jsonl")]
    assert [s["name"] for s in steps] == ["o3d_check", "guard", "inputs", "gsplat_probe", "parity", "mesh_val",
                                          "bfs_train", "bfs_val", "bfs_test", "floors", "sft"]
    assert all(s["rc"] == 0 for s in steps) and not (paths.E3VS_OUT / "PREP_DONE").exists()
    assert not any("occupancy" in c for c in cmds) and not list(root.iterdir())  # a dry run writes nothing there
    o3d = [c for c in cmds if c.endswith("  # +o3d")]
    assert [c.split(" -m ")[1].split()[0] for c in o3d] == ["vpd.e3vs.mesh_collision", "vpd.e3vs.bfs_demos",
                                                             "vpd.e3vs.bfs_demos", "vpd.e3vs.bfs_demos",
                                                             "vpd.e3vs.floors"]
    assert not any("--probe" in c or "--parity" in c or "make_sft" in c for c in o3d)
    assert "o3d-check would stop" not in r.stdout  # O3D_DIR shadows nothing of PY_Q3
    text = (JOBS / "e3vs_prep.sh").read_text()
    assert "unset VPD_E3VS_MESH_ROOT VPD_E3VS_MESH_MANIFEST VPD_E3VS_MESH_VAL_DIR" in text  # production: OUT only
    # 2.0 GPU-h = 7200 s: outer 6900 + kill 120 ends 180 s early (container start / teardown headroom)
    assert "deadline=6720s" in r.stdout and "timeout -k 120 6900 bash" in (JOBS / "e3vs_prep.sh").read_text()
    assert "7080" not in (JOBS / "e3vs_prep.sh").read_text()


def _mesh_root(root: Path, sids) -> str:
    """Synthetic mesh files in root/mesh (copies of root/mesh_dl, the listing root/data) + the mesh_verify
    manifest root/mesh/MANIFEST.json + a passing root/mesh/validation.json (the job hooks' layout); returns the key."""
    from vpd.tests.test_e3vs_mesh_verify import make_tree

    mesh, ids, kw = make_tree(root, sids)
    assert mesh == root / "mesh" and MC.mesh_path(sids[0], mesh).is_file()
    man = MV.verify(mesh, ids, mesh, expected=list(sids), **kw)
    assert man["ok"], man["problems"]
    shas = {s: r["sha256"] for s, r in man["scenes"].items()}
    (mesh / "validation.json").write_text(json.dumps(fake_mesh_validation(shas)))
    return man["mesh_set_sha256"]


def test_prep_payload_resumes(scratch):
    """A resubmitted prep job (a NEW RUN) skips the steps whose outputs are current, gsplat_probe and parity
    included; another mesh set (a new KEY) reruns the mesh validation and everything after it."""
    root = scratch / "prep_resume_e3vs"
    sid = "00dd871005"
    key = _mesh_root(root, [sid])
    assert key == MV.mesh_set_sha256({sid: E.sha256_file(MC.mesh_path(sid, root / "mesh"))})
    (root / "parity").mkdir(parents=True)
    (root / "parity" / f"parity_{sid}.json").write_text(json.dumps({"pass": True}))
    (root / "parity" / "gsplat_probe_prep.json").write_text(json.dumps({"extension_loaded": True}))
    (root / "torch_ext" / "gsplat_cuda").mkdir(parents=True)
    (root / "torch_ext" / "gsplat_cuda" / "gsplat_cuda.so").write_bytes(b"\x7fELF")
    for sp in ("train", "val", "test"):
        (root / "bfs" / "mesh" / sp).mkdir(parents=True)
        for f in ("summary.json", "opt_len.json"):
            (root / "bfs" / "mesh" / sp / f).write_text("{}")
    (root / "floors").mkdir()
    (root / "floors" / "floors_mesh.json").write_text("{}")
    (root / "sft" / "seed1001").mkdir(parents=True)
    for f in ("warmup20.jsonl", "order_seed1001.json", "seed1001/demos_100.jsonl"):
        (root / "sft" / f).write_text("{}\n")
    (root / "prep_state").mkdir()
    for st in ("bfs_train", "bfs_val", "bfs_test", "floors", "sft"):
        (root / "prep_state" / st).write_text(key + "\n")
    env = {"VPD_E3VS_ROOT": str(root), "VPD_PREP_SCENES": sid}
    r = _dry("e3vs_prep.sh", [], scratch / "prep_resume_run1", **env)
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _dry_cmds(r.stdout)
    assert not any(w in c for c in cmds for w in ("--probe", "--parity", "mesh_collision validate", "bfs_demos",
                                                  "floors", "make_sft", "stamp "))
    assert all(f"{s}: outputs exist and are current, skipped" in r.stdout for s in
               ("gsplat_probe", "parity", "mesh_val", "bfs_train", "bfs_val", "bfs_test", "floors", "sft"))
    # the probe reruns when its build is gone, or when the probe file says the extension did not load
    (root / "torch_ext" / "gsplat_cuda" / "gsplat_cuda.so").unlink()
    r1b = _dry("e3vs_prep.sh", [], scratch / "prep_resume_run1b", **env)
    assert [c for c in _dry_cmds(r1b.stdout) if "--probe --out " + str(root / "parity") in c]
    (root / "torch_ext" / "gsplat_cuda" / "gsplat_cuda.so").write_bytes(b"\x7fELF")
    (root / "parity" / "gsplat_probe_prep.json").write_text(json.dumps({"extension_loaded": False}))
    r1c = _dry("e3vs_prep.sh", [], scratch / "prep_resume_run1c", **env)
    assert [c for c in _dry_cmds(r1c.stdout) if "--probe" in c]
    (root / "parity" / "gsplat_probe_prep.json").write_text(json.dumps({"extension_loaded": True}))
    # a floors stamp from other meshes: only floors reruns
    (root / "prep_state" / "floors").write_text("other\n")
    r2 = _dry("e3vs_prep.sh", [], scratch / "prep_resume_run2", **env)
    cmds2 = _dry_cmds(r2.stdout)
    assert [c for c in cmds2 if "-m vpd.e3vs.floors" in c] and not any("bfs_demos" in c or "make_sft" in c
                                                                       for c in cmds2)
    # a changed mesh: the mesh check fails, the validation runs, and everything after it reruns
    with open(MC.mesh_path(sid, root / "mesh"), "ab") as f:
        f.write(b"x")
    r3 = _dry("e3vs_prep.sh", [], scratch / "prep_resume_run3", **env)
    cmds3 = _dry_cmds(r3.stdout)
    for w in ("mesh_collision validate --scenes all", "bfs_demos --split train", "-m vpd.e3vs.floors",
              "make_sft all"):
        assert any(w in c for c in cmds3), (w, cmds3)
    assert not any("--parity" in c or "--probe" in c for c in cmds3)


def test_prep_payload_stops_on_a_failed_mesh_validation(scratch):
    """'验证不过就停下问我': every validate rc but 0 ends the job (nothing after it runs, no PREP_DONE), with the
    message of that rc: 1 gating failure, 2 refused inputs, 3 error without a result, 4 possible frame mismatch."""
    root = scratch / "prep_meshfail_e3vs"
    root.mkdir()
    text = (JOBS / "e3vs_prep.sh").read_text()
    body = text[text.index("mesh_step() {"):text.index("bfs_split() {")]
    assert 'PYTHONPATH="$O3D_PP" x "$PY" -m vpd.e3vs.mesh_collision validate --scenes all || rc=$?' in body
    want = {1: "FAILED a gating metric", 2: "mesh inputs REFUSED", 3: "ERROR, no validation result",
            4: "possible frame mismatch", 124: "exited 124"}
    assert (MC.RC_GATE, MC.RC_REFUSED, MC.RC_ERROR, MC.RC_FRAME) == (1, 2, 3, 4)
    for rc, msg in want.items():
        stub = scratch / f"x_rc{rc}.sh"  # the step's command fails like validate would
        stub.write_text(f"DRY=0\nKEY=\nE3VS={root}\nMESH_SCENES=all\nPY=false\nO3D_PP=\nskip() {{ :; }}\n"
                        f"x() {{ return {rc}; }}\nmesh_key() {{ :; }}\n" + body + "mesh_step; echo rc=$?\n")
        r = subprocess.run(["bash", str(stub)], capture_output=True, text=True, timeout=60)
        assert f"rc={rc}" in r.stdout and msg in r.stderr and "the job stops here" in r.stderr, (rc, r.stdout,
                                                                                               r.stderr)
    # the payload calls the step at top level under set -e: a failing step ends the job
    assert "\nstep mesh_val mesh_step\n" in text and "set -euo pipefail" in (JOBS / "common_env.sh").read_text()


def test_o3d_check_is_a_recorded_step_after_the_chown_trap(scratch):
    """S1 / N6: o3d_check is the first step of both payloads (same function text), in the start job after the chown
    trap (so nothing root-owned is left behind); a failing check is a steps.jsonl line and exits 2."""
    fn = {}
    for job in ("e3vs_prep.sh", "e3vs_start.sh"):
        t = (JOBS / job).read_text()
        i = t.index("o3d_check() {")
        fn[job] = t[i:t.index("\n}\n", i) + 3]
        assert "\nstep o3d_check o3d_check || exit 2" in t and t.index("step o3d_check o3d_check") < t.index(
            "\nstep guard "), job
    assert fn["e3vs_prep.sh"] == fn["e3vs_start.sh"]
    st = (JOBS / "e3vs_start.sh").read_text()
    assert st.index('mkdir -p "$A" "$E"') < st.index('vpd_chown_trap "$RUN" "$PILOT"') < st.index(
        "step o3d_check o3d_check")
    run = scratch / "o3dcheck_run"
    run.mkdir()
    s = (f"RUN={run}\nsource {JOBS / 'common_env.sh'}\nRESERVE=0\nDEADLINE=$(( $(date +%s) + 100 ))\nMIN_STEP_S=0\n"
         + _helpers("e3vs_prep.sh") + "DRY=0\nPY_Q3=false\n" + fn["e3vs_prep.sh"]
         + "step o3d_check o3d_check || exit 2\necho AFTER\n")
    r = subprocess.run(["bash", "-c", s], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "AFTER" not in r.stdout, (r.stdout, r.stderr)
    assert [(x["name"], x["rc"]) for x in map(json.loads, open(run / "steps.jsonl"))] == [("o3d_check", 1)]


ROOMY = {"VPD_FAKE_FREE_GIB": "/tmp=100,/dev/shm=100"}


def test_start_payload_dry_run(scratch):
    run = scratch / "start_run"
    r = _dry("e3vs_start.sh", ["Q3S"], run, **ROOMY)
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _dry_cmds(r.stdout)
    A = paths.E3VS_RUNS / "pilot" / "Q3S" / "adapters"
    sft = paths.E3VS_SFT
    want = ["-m vpd.e3vs.mesh_collision smoke --split test --imports torch,gsplat",  # before training
            f"vpd/train/n2_train_vpd.py --start Q3S --data {sft}/warmup20.jsonl --out {A}/0W --epochs auto --seed 1001 "
            "--phase 0W --threads 2",
            f"--data {sft}/seed1001/demos_100.jsonl --out {A}/100 --epochs auto --seed 1001 --phase tier100 "
            f"--threads 2 --init-adapter {A}/0W",
            "-m vpd.e3vs.run --start Q3S --tier raw --mode u --split test",
            f"vpd/train/merge.py --start Q3S --adapter {A}/0W --out /tmp/vpd_merged/Q3S__0W",
            f"-m vpd.e3vs.run --start Q3S --tier 0W --mode u --split test --out-root",
            f"--model-dir /tmp/vpd_merged/Q3S__0W --adapter-meta {A}/0W/meta.json",
            "rm -rf -- /tmp/vpd_merged/Q3S__0W",
            f"vpd/train/merge.py --start Q3S --adapter {A}/100 --out /tmp/vpd_merged/Q3S__100",
            "-m vpd.e3vs.run --start Q3S --tier 100 --mode u,c --split test",  # one session for both modes
            "rm -rf -- /tmp/vpd_merged/Q3S__100"]
    pos = []
    for w in want:
        hit = [i for i, c in enumerate(cmds) if w in c]
        assert hit, (w, cmds)
        pos.append(hit[-1] if w.startswith("--model-dir") else hit[0])
    assert pos[:6] == sorted(pos[:6]) and pos[8:] == sorted(pos[8:])
    o3d = [c for c in cmds if c.endswith("  # +o3d")]
    assert len(o3d) == 4 and "smoke" in o3d[0] and all("-m vpd.e3vs.run" in c for c in o3d[1:])
    assert not any(c.endswith("  # +o3d") for c in cmds if "n2_train_vpd.py" in c or "merge.py" in c)
    evals = [c for c in cmds if "-m vpd.e3vs.run" in c]
    assert len(evals) == 3 and all(f"--gsplat-probe {run}/gsplat_probe.json" in c for c in evals)
    assert not any("--tier raw --mode u,c" in c or "--tier 0W --mode u,c" in c for c in cmds)  # optional off
    assert all(c.startswith(str(paths.PY_TRAIN)) for c in cmds if "n2_train_vpd.py" in c)
    steps = [s["name"] for s in map(json.loads, open(run / "steps.jsonl"))]
    assert steps == ["o3d_check", "guard", "gsplat", "collider_smoke", "train_0W", "train_100", "eval_raw", "merge_0W",
                     "eval_0W", "unmerge_0W", "merge_100", "eval_100", "unmerge_100"]
    # 2.8 GPU-h = 10080 s: outer 9780 + kill 120 ends 180 s early (container start / teardown headroom)
    text = (JOBS / "e3vs_start.sh").read_text()
    assert "deadline=9600s" in r.stdout and "timeout -k 120 9780 bash" in text and "9960" not in text
    assert "merge base /tmp/vpd_merged" in r.stdout and "[dry] touch" in r.stdout
    opt = _dry("e3vs_start.sh", ["E2"], scratch / "start_opt", VPD_OPTIONAL="1", **ROOMY)
    oc = [c for c in _dry_cmds(opt.stdout) if "-m vpd.e3vs.run" in c]
    assert opt.returncode == 0 and len(oc) == 3 and all("--mode u,c" in c for c in oc)
    bad = _dry("e3vs_start.sh", ["QRL-s1"], scratch / "start_bad")
    assert bad.returncode == 2


def _start_onescene(scratch: Path, sid: str) -> Path:
    """e3vs_start.sh whose guard checks the mesh key of one scene instead of all 99 (a one-scene test tree)."""
    text = (JOBS / "e3vs_start.sh").read_text()
    old = 'key=$("$PY_Q3" -m vpd.e3vs.mesh_collision check --scenes all | tail -n 1) || key='
    assert text.count(old) == 1
    text = text.replace(old, old.replace("--scenes all", f"--scenes {sid}"))
    src = 'source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"'
    assert text.count(src) == 1
    job = scratch / f"e3vs_start_{sid}.sh"
    job.write_text(text.replace(src, f'source "{JOBS / "common_env.sh"}"'))
    return job


def test_start_payload_guard_compares_the_mesh_key(scratch):
    """S4: with VPD_E3VS_ROOT (dry run only) the guard checks a test tree: key == prep_state/floors == prep_state/sft
    passes; a mismatching stamp, or a changed mesh, stops the job before training."""
    root = scratch / "start_guard_e3vs"
    sid = "00dd871005"
    key = _mesh_root(root, [sid])
    (root / "parity").mkdir()
    (root / "parity" / f"parity_{sid}.json").write_text(json.dumps({"pass": True}))
    for f in ("PREP_DONE", "sft/warmup20.jsonl", "sft/seed1001/demos_100.jsonl", "floors/floors_mesh.json"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("x\n")
    (root / "prep_state").mkdir()
    for st in ("floors", "sft"):
        (root / "prep_state" / st).write_text(key + "\n")
    job = _start_onescene(scratch, sid)

    def guard(name, script=job):
        env = {k: v for k, v in os.environ.items() if k not in ("RUN", "VPD_E3VS_REPO", "VPD_E3VS_MESH_ROOT",
                                                                "VPD_E3VS_MESH_MANIFEST", "VPD_E3VS_MESH_VAL_DIR",
                                                                "VPD_DRY_FAIL", "VPD_MERGE_DIR")}
        env.update(VPD_DRYRUN="1", PYTHONDONTWRITEBYTECODE="1", VPD_WORKTREE=str(paths.WORKTREE),
                   VPD_E3VS_ROOT=str(root), **ROOMY)
        r = subprocess.run(["bash", str(script), "Q3S", str(scratch / name)], capture_output=True, text=True, env=env,
                           timeout=300)
        return r, "[dry] guard would stop the job here" in r.stdout

    r, stopped = guard("sg_ok")
    assert not stopped and f"[e3vs] mesh set {key} (validated; floor and SFT stamped with it)" in r.stdout, \
        r.stdout + r.stderr
    assert "mesh validation + parity pass: True" in r.stdout
    A = root / "runs" / "pilot" / "Q3S" / "adapters"
    assert any(f"--out {A}/0W" in c for c in _dry_cmds(r.stdout))  # the hook moves the outputs too
    for st in ("floors", "sft"):
        (root / "prep_state" / st).write_text("other\n")
        r, stopped = guard(f"sg_{st}")
        assert stopped and f"prep_state/{st} 'other' != the validated mesh set {key}: refusing" in r.stderr, r.stderr
        (root / "prep_state" / st).write_text(key + "\n")
    assert not guard("sg_ok2")[1]
    with open(MC.mesh_path(sid, root / "mesh"), "ab") as f:
        f.write(b"x")
    r, stopped = guard("sg_mesh")
    assert stopped and "mesh check failed" in r.stderr and "changed since validation" in r.stderr, r.stderr
    # the real script checks all 99 scenes: on this one-scene tree it stops too
    r, stopped = guard("sg_real", JOBS / "e3vs_start.sh")
    assert stopped and "mesh check failed" in r.stderr


def test_start_payload_merge_base(scratch):
    """B11 at the payload level: /tmp if it has >= 40 GiB, else /dev/shm, else the guard stops before training;
    VPD_MERGE_DIR can only force one of the two job-local bases."""
    r = _dry("e3vs_start.sh", ["E2"], scratch / "mb_shm", VPD_FAKE_FREE_GIB="/tmp=10,/dev/shm=60")
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = _dry_cmds(r.stdout)
    A = paths.E3VS_RUNS / "pilot" / "E2" / "adapters"
    for w in (f"merge.py --start E2 --adapter {A}/0W --out /dev/shm/vpd_merged/E2__0W",
              "--model-dir /dev/shm/vpd_merged/E2__0W", "rm -rf -- /dev/shm/vpd_merged/E2__0W",
              f"merge.py --start E2 --adapter {A}/100 --out /dev/shm/vpd_merged/E2__100",
              "--model-dir /dev/shm/vpd_merged/E2__100", "rm -rf -- /dev/shm/vpd_merged/E2__100"):
        assert any(w in c for c in cmds), (w, cmds)
    assert not any("/tmp/vpd_merged" in c for c in cmds) and "merge base /dev/shm/vpd_merged" in r.stdout
    # neither base has room: the guard would stop the job before training; the merges refuse, their evals do not run
    r2 = _dry("e3vs_start.sh", ["E2"], scratch / "mb_none", VPD_FAKE_FREE_GIB="/tmp=10,/dev/shm=39")
    assert r2.returncode == 1 and "no job-local disk with >= 40 GiB free" in r2.stderr, r2.stdout + r2.stderr
    assert "[dry] guard would stop the job here" in r2.stdout and "[dry] touch" not in r2.stdout
    cmds2 = _dry_cmds(r2.stdout)
    assert not any("merge.py" in c or "--model-dir" in c for c in cmds2)
    assert any("--tier raw" in c for c in cmds2)
    steps = {s["name"]: s["rc"] for s in map(json.loads, open(scratch / "mb_none" / "steps.jsonl"))}
    assert steps["merge_0W"] == 1 and steps["merge_100"] == 1 and "eval_0W" not in steps and "eval_100" not in steps
    assert steps["unmerge_0W"] == 0 and "failed steps: merge_0W merge_100" in r2.stderr
    # a forced base is still checked; a base off job-local disk is refused
    r3 = _dry("e3vs_start.sh", ["E2"], scratch / "mb_forced", VPD_FAKE_FREE_GIB="/tmp=10,/dev/shm=60",
              VPD_MERGE_DIR="/tmp/vpd_merged")
    assert r3.returncode == 1 and "no job-local disk" in r3.stderr
    r4 = _dry("e3vs_start.sh", ["E2"], scratch / "mb_forced2", VPD_FAKE_FREE_GIB="/tmp=10,/dev/shm=60",
              VPD_MERGE_DIR="/dev/shm/vpd_merged")
    assert r4.returncode == 0 and any("--out /dev/shm/vpd_merged/E2__0W" in c for c in _dry_cmds(r4.stdout))
    r5 = _dry("e3vs_start.sh", ["E2"], scratch / "mb_bad", VPD_MERGE_DIR="/mnt/umm/users/x/merged")
    assert r5.returncode == 2 and "job-local" in r5.stderr
    # the hook is ignored outside a dry run: the payload parses and reads df (static check of the text)
    text = (JOBS / "e3vs_start.sh").read_text()
    assert 'if [[ $DRY == 1 && -n ${VPD_FAKE_FREE_GIB:-} ]]' in text
    for d in ("/tmp/vpd_merged/${START}__0W", "/dev/shm/vpd_merged/${START}__100"):
        assert d in text  # the EXIT cleanup covers both bases


def test_start_payload_keeps_going_after_a_failed_step(scratch):
    """A failed train / merge / eval is recorded; the steps that do not depend on it still run; exit 1, no DONE."""
    def steps_of(run):
        return [(s["name"], s["rc"]) for s in map(json.loads, open(run / "steps.jsonl"))]

    r = _dry("e3vs_start.sh", ["Q3B"], scratch / "kg1", VPD_DRY_FAIL="eval_raw,train_100", **ROOMY)
    assert r.returncode == 1 and "failed steps: train_100 eval_raw" in r.stderr, r.stdout + r.stderr
    assert steps_of(scratch / "kg1") == [("o3d_check", 0), ("guard", 0), ("gsplat", 0), ("collider_smoke", 0),
                                         ("train_0W", 0), ("train_100", 3), ("eval_raw", 3), ("merge_0W", 0),
                                         ("eval_0W", 0), ("unmerge_0W", 0)]
    assert "[dry] touch" not in r.stdout
    r2 = _dry("e3vs_start.sh", ["Q3B"], scratch / "kg2", VPD_DRY_FAIL="train_0W", **ROOMY)
    assert r2.returncode == 1 and steps_of(scratch / "kg2") == [("o3d_check", 0), ("guard", 0), ("gsplat", 0),
                                                                  ("collider_smoke", 0), ("train_0W", 3),
                                                                  ("eval_raw", 0)]
    r3 = _dry("e3vs_start.sh", ["Q3B"], scratch / "kg3", VPD_DRY_FAIL="merge_0W,eval_100", **ROOMY)
    assert r3.returncode == 1 and "failed steps: merge_0W eval_100" in r3.stderr
    assert steps_of(scratch / "kg3")[6:] == [("eval_raw", 0), ("merge_0W", 3), ("unmerge_0W", 0), ("merge_100", 0),
                                             ("eval_100", 3), ("unmerge_100", 0)]


def test_no_bytecode_written():
    assert not list((paths.VPD_PKG / "e3vs").glob("__pycache__"))


def test_collides_skips_the_start_margin(tmp_path):
    """Option (ii): a start inside the dilated margin can move away, but still collides moving into the wall."""
    g = OC.Grid(OC.pack(np.array([[0, 0, 0]])), meta={"scene_id": "skip"})
    start = (0.07, 0.025, 0.025)  # voxel x-index 1: inside the 1-voxel margin of the occupied voxel 0
    assert g.near_occupied(np.array([start]))[0]
    assert not g.collides(start, (0.32, 0.025, 0.025))   # away from the wall
    assert g.collides(start, (-0.18, 0.025, 0.025))      # through the wall
    literal = OC.Grid(OC.pack(np.array([[0, 0, 0]])), start_skip=0.0)
    assert literal.collides(start, (0.32, 0.025, 0.025))  # option (i) behaviour, for comparison
    p = tmp_path / "g.npz"
    g.save(p)
    assert OC.Grid.load(p).start_skip == g.start_skip == OC.VOXEL_M * OC.START_SKIP_VOX
