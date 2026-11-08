"""E3VS collision model: a per-scene voxel occupancy grid built from rendered splat depth (PREREG section 5).

The official benchmark ray-casts the ScanNet++ mesh (eval/metrics.py: a move collides when the ray from the old
to the new position hits the mesh at tolerance < t < length - tolerance).  We have no mesh, so (report v2 section
3.2, plan 3.8, frozen in PREREG_E3VS_PILOT section 5):

  build   per scene, render expected depth and alpha (render.SplatRenderer.render_depth, 'RGB+ED') at every
          IPHONE_EVERY-th iPhone pose of InsScene's scene_iphone_metadata.npz plus every distinct E3VS viewpoint
          (start / answerable / unanswerable templates) of the scene's episodes; back-project the pixels with
          alpha > ALPHA_MIN (pixel centres, z-depth, depth < MAX_DEPTH_M) into VOXEL_M = 5 cm voxels; a voxel is
          occupied when at least MIN_HITS = 3 views hit it (a hit = one view, counted once per view however many
          of its pixels land there: a noise filter that per-pixel counts would not give).
  collide collides(p0, p1): any occupied voxel within MARGIN_VOX = 1 voxel (the 3x3x3 neighbourhood) of the
          segment p0 -> p1 + EXTEND_M * u (u = unit direction, EXTEND_M = 0.05 m).  The segment is sampled every
          VOXEL_M / 4.  A zero-length segment never collides (as the official ray test).  Rotations are never
          asked (geometry.rollout, bfs_demos, floors and run.py call it for move_* only).  Space outside every
          observed voxel is free.
  validate  (frozen thresholds, before any model eval) the fraction of consecutive iPhone-trajectory segments
          that collide < 1 %, and the fraction of E3VS viewpoints whose own voxel is occupied < 1 %, pooled over
          the scenes; written to OUT/e3vs/occ/validation.json.  Also reported (descriptive): viewpoints within the
          1-voxel margin (every move from there collides), per-scene maxima, and per split the episode STARTS
          inside / within the margin (such an episode is rotation-only under the frozen rule: every translation
          from a start within the margin collides, for the floor, the BFS and the models alike).
          validation.json carries grids_sha256 = grids_key(per-scene npz sha256): the identity of the validated
          grids, which the prep job stamps on everything built from them (BFS, floors, SFT) and the gate checks.
  check   validation.json passes, lists every asked scene, and every npz still has the validated sha256; prints
          grids_sha256 (exit 0) or the problems (exit 1).  The prep job's resume test.

Files: OUT/e3vs/occ/<scene>.npz (sorted int64 keys of the occupied voxels, their hit counts, the parameters, the
renderer meta and the view list) and OUT/e3vs/occ/validation.json.  ``load_collider(scene_id)`` is the factory
for ``bfs_demos --collides vpd.e3vs.occupancy:load_collider`` and ``floors --collides ...`` (VPD_E3VS_OCC_DIR
overrides the directory; a missing file raises, it never silently means "no collision").

CLI (sco job vpd-e3vs-prep: GPU, splats; --renderer fake runs anywhere):
    python -m vpd.e3vs.occupancy build --scenes all|train|val|test|ID,ID [--validate] [--resume] [--out DIR]
                                       [--renderer gsplat|fake] [--convention auto|opencv|opengl]
    python -m vpd.e3vs.occupancy validate --scenes all [--out DIR]
    python -m vpd.e3vs.occupancy check --scenes all|ID,ID [--out DIR]
--resume keeps a scene's existing npz when its build parameters, iPhone convention, renderer and PLY sha256 equal
the ones this build would use (a resubmitted prep job does not re-render finished scenes).
--convention auto takes the iPhone convention the parity files chose; with no parity file holding an iPhone 'best'
the build REFUSES (no silent default: give --convention explicitly, or --no-iphone).  The convention and where it
came from are stored in every npz meta and reported in validation.json (iphone_convention).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

from vpd import paths
from vpd.e3vs import IMAGE_H, IMAGE_W, SPLITS, code_meta, load_episodes, repo_dir, sha256_file
from vpd.e3vs import render as RD

VOXEL_M = 0.05
MIN_HITS = 3
MARGIN_VOX = 1
EXTEND_M = 0.05
ALPHA_MIN = 0.5
MAX_DEPTH_M = 20.0
IPHONE_EVERY = 5
VALID_MAX_FRAC = 0.01
# depth render sizes (5 cm voxels need far less than full resolution: at 3 m a 256 px / 90 deg view has 2.3 cm
# pixels, the quarter-size iPhone view 1.6 cm)
TEMPLATE_DEPTH_WH = 256
IPHONE_DEPTH_SCALE = 0.25

_OFF = 1 << 20
_MASK = (1 << 21) - 1


# ------------------------------------------------------------------------------------------------------------
# voxel keys
# ------------------------------------------------------------------------------------------------------------
def voxel_index(points, voxel: float = VOXEL_M) -> np.ndarray:
    return np.floor(np.asarray(points, dtype=np.float64) / voxel).astype(np.int64)


def pack(idx: np.ndarray) -> np.ndarray:
    """(N, 3) int voxel indices -> (N,) int64 keys (+-2^20 voxels per axis)."""
    idx = np.asarray(idx, dtype=np.int64).reshape(-1, 3) + _OFF
    if (idx < 0).any() or (idx > _MASK).any():
        raise ValueError("voxel index out of the key range")
    return (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]


def unpack(keys: np.ndarray) -> np.ndarray:
    k = np.asarray(keys, dtype=np.int64)
    return np.stack([(k >> 42) & _MASK, (k >> 21) & _MASK, k & _MASK], axis=-1) - _OFF


def _key1(ix: int, iy: int, iz: int) -> int:
    return ((ix + _OFF) << 42) | ((iy + _OFF) << 21) | (iz + _OFF)


def dilate(keys: np.ndarray, margin: int = MARGIN_VOX) -> np.ndarray:
    """Sorted unique keys of every voxel within `margin` voxels (Chebyshev) of a key."""
    keys = np.asarray(keys, dtype=np.int64)
    if margin <= 0 or keys.size == 0:
        return np.unique(keys)
    idx = unpack(keys)
    out = [pack(idx + np.array(o)) for o in product(range(-margin, margin + 1), repeat=3)]
    return np.unique(np.concatenate(out))


def _in_sorted(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.int64)
    if sorted_keys.size == 0:
        return np.zeros(keys.shape, dtype=bool)
    pos = np.searchsorted(sorted_keys, keys)
    pos = np.minimum(pos, sorted_keys.size - 1)
    return sorted_keys[pos] == keys


# ------------------------------------------------------------------------------------------------------------
# the grid / collider
# ------------------------------------------------------------------------------------------------------------
class Grid:
    """Occupied voxels of one scene; ``grid(p0, p1)`` is the collision test (module docstring)."""

    def __init__(self, occ_keys, voxel: float = VOXEL_M, margin: int = MARGIN_VOX, extend: float = EXTEND_M,
                 hits=None, meta: dict | None = None):
        order = np.argsort(np.asarray(occ_keys, dtype=np.int64), kind="stable")
        self.occ = np.asarray(occ_keys, dtype=np.int64)[order]
        if self.occ.size and (np.diff(self.occ) == 0).any():
            raise ValueError("duplicate occupied keys")
        self.hits = None if hits is None else np.asarray(hits)[order]
        self.voxel, self.margin, self.extend = float(voxel), int(margin), float(extend)
        self.meta = dict(meta or {})
        self.dil = dilate(self.occ, self.margin)
        self._dilset = None  # python set for the per-call test (built on first use)

    # vectorised queries ------------------------------------------------------------------------------------
    def occupied(self, points) -> np.ndarray:
        return _in_sorted(self.occ, pack(voxel_index(np.atleast_2d(points), self.voxel)))

    def near_occupied(self, points) -> np.ndarray:
        return _in_sorted(self.dil, pack(voxel_index(np.atleast_2d(points), self.voxel)))

    # the collision test ------------------------------------------------------------------------------------
    def collides(self, p0, p1) -> bool:
        x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2])
        dx, dy, dz = float(p1[0]) - x0, float(p1[1]) - y0, float(p1[2]) - z0
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-9:
            return False
        if self._dilset is None:
            self._dilset = set(self.dil.tolist())
        ux, uy, uz = dx / length, dy / length, dz / length
        total = length + self.extend
        n = max(1, int(math.ceil(total / (self.voxel / 4.0))))
        v, s, seen = self.voxel, self._dilset, set()
        for i in range(n + 1):
            t = total * i / n
            k = _key1(math.floor((x0 + ux * t) / v), math.floor((y0 + uy * t) / v), math.floor((z0 + uz * t) / v))
            if k in seen:
                continue
            seen.add(k)
            if k in s:
                return True
        return False

    __call__ = collides

    # io ----------------------------------------------------------------------------------------------------
    def save(self, path: Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, occ_keys=self.occ, hits=(np.zeros(0, np.uint16) if self.hits is None else
                                                           self.hits.astype(np.uint16)),
                            params=np.array(json.dumps({"voxel": self.voxel, "margin": self.margin,
                                                        "extend": self.extend})),
                            meta=np.array(json.dumps(self.meta)))
        os.replace(tmp, path)
        return sha256_file(path)

    @classmethod
    def load(cls, path: Path) -> "Grid":
        with np.load(path, allow_pickle=False) as z:
            prm = json.loads(str(z["params"]))
            hits = z["hits"]
            g = cls(z["occ_keys"], voxel=prm["voxel"], margin=prm["margin"], extend=prm["extend"],
                    hits=hits if hits.size else None, meta=json.loads(str(z["meta"])))
        g.meta["path"] = str(path)
        return g


def occ_dir() -> Path:
    return Path(os.environ.get("VPD_E3VS_OCC_DIR") or paths.E3VS_OCC)


def occ_path(scene_id: str, directory: Path | None = None) -> Path:
    return Path(directory or occ_dir()) / f"{scene_id}.npz"


def load_collider(scene_id: str) -> Grid:
    """Factory for --collides vpd.e3vs.occupancy:load_collider: the scene's frozen grid (a callable)."""
    p = occ_path(scene_id)
    if not p.is_file():
        raise FileNotFoundError(f"no occupancy grid for scene {scene_id} at {p} (python -m vpd.e3vs.occupancy build)")
    return Grid.load(p)


# ------------------------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------------------------
def backproject(depth: np.ndarray, alpha: np.ndarray, c2w, K, alpha_min: float = ALPHA_MIN,
                max_depth: float = MAX_DEPTH_M) -> np.ndarray:
    """World points of the pixels with alpha > alpha_min and 0 < depth < max_depth (z-depth, pixel centres)."""
    depth = np.asarray(depth, dtype=np.float64)
    H, W = depth.shape
    rays = RD.pixel_rays(W, H, K)
    m = (np.asarray(alpha) > alpha_min) & np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    pc = rays[m] * depth[m][:, None]
    c2w = np.asarray(c2w, dtype=np.float64)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def accumulate(views, renderer, voxel: float = VOXEL_M, alpha_min: float = ALPHA_MIN,
               max_depth: float = MAX_DEPTH_M) -> tuple[np.ndarray, np.ndarray, list]:
    """(sorted keys, number of views hitting each key, per-view stats) over views [(tag, c2w, K, W, H)]."""
    keys = np.zeros(0, np.int64)
    counts = np.zeros(0, np.int64)
    pending, stats = [], []

    def merge(keys, counts, pending):
        if not pending:
            return keys, counts
        allk = np.concatenate([keys] + pending)
        allc = np.concatenate([counts] + [np.ones(p.size, np.int64) for p in pending])
        uk, inv = np.unique(allk, return_inverse=True)
        return uk, np.bincount(inv, weights=allc).astype(np.int64)

    for tag, c2w, K, W, H in views:
        depth, alpha = renderer.render_depth(c2w, W, H, K)
        pts = backproject(depth, alpha, c2w, K, alpha_min, max_depth)
        k = np.unique(pack(voxel_index(pts, voxel))) if len(pts) else np.zeros(0, np.int64)
        pending.append(k)
        stats.append({"view": tag, "points": int(len(pts)), "voxels": int(k.size)})
        if len(pending) >= 32:
            keys, counts = merge(keys, counts, pending)
            pending = []
    keys, counts = merge(keys, counts, pending)
    return keys, counts, stats


def scaled_K(K, s: float) -> np.ndarray:
    K = np.array(K, dtype=np.float64, copy=True)
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] *= s
    K[1, 2] *= s
    return K


def iphone_image_size(md: dict) -> tuple[int, int]:
    from PIL import Image

    with Image.open(md["dir"] / "images" / md["images"][0]) as im:
        return im.size


def scene_views(scene_id: str, episodes: list[dict], md: dict | None, convention: str,
                iphone_every: int = IPHONE_EVERY, template_wh: int = TEMPLATE_DEPTH_WH,
                iphone_scale: float = IPHONE_DEPTH_SCALE) -> list[tuple]:
    """[(tag, c2w (OpenCV), K, W, H)]: every iphone_every-th iPhone pose, then the scene's E3VS viewpoints."""
    views = []
    if md is not None:
        W0, H0 = iphone_image_size(md)
        W, H = max(1, int(round(W0 * iphone_scale))), max(1, int(round(H0 * iphone_scale)))
        sx, sy = W / W0, H / H0
        for i in range(0, len(md["c2w"]), iphone_every):
            K = md["K"][i].copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            views.append((f"iphone{i}", RD.convention_c2w(md["c2w"][i], convention), K, W, H))
    s = template_wh / float(IMAGE_W)
    Kt = scaled_K(RD.intrinsics(IMAGE_W, IMAGE_H), s)
    for tag, c2w in RD.scene_viewpoints(episodes, scene_id):
        views.append((tag, c2w, Kt, template_wh, int(round(IMAGE_H * s))))
    return views


def resolve_convention(convention: str, parity_dir: Path | None = None) -> tuple[str, str]:
    """(convention, where it came from) for 'auto' | 'opencv' | 'opengl'.  'auto' without any parity file holding an
    iPhone 'best' refuses: a silent default could misplace every iPhone trajectory the validation tests."""
    if convention != "auto":
        return convention, "given (--convention)"
    best = RD.best_iphone_convention(parity_dir)
    if not best:
        d = Path(parity_dir) if parity_dir else paths.E3VS_PARITY
        raise SystemExit(f"[occupancy] --convention auto: no parity file under {d} has an iPhone 'best' (the parity "
                         "run's iPhone part errored or is missing); rerun parity, or give --convention explicitly "
                         "(only with the user's OK)")
    return best, "parity"


def build_params(iphone_every: int = IPHONE_EVERY, **kw) -> dict:
    """The parameters a build records in its npz meta (and --resume compares)."""
    return {"voxel": kw.get("voxel", VOXEL_M), "min_hits": kw.get("min_hits", MIN_HITS),
            "margin": kw.get("margin", MARGIN_VOX), "extend": kw.get("extend", EXTEND_M),
            "alpha_min": kw.get("alpha_min", ALPHA_MIN), "max_depth": kw.get("max_depth", MAX_DEPTH_M),
            "iphone_every": iphone_every, "template_wh": kw.get("template_wh", TEMPLATE_DEPTH_WH),
            "iphone_scale": kw.get("iphone_scale", IPHONE_DEPTH_SCALE), "hit_unit": "view"}


def reusable(path: Path, renderer_kind: str, scene_id: str, convention: str | None, iphone_every: int,
             **kw) -> bool:
    """--resume: the existing npz was built with exactly this build's parameters, convention, renderer and PLY."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        meta = Grid.load(path).meta
    except Exception:  # noqa: BLE001  (an unreadable / partial file is rebuilt)
        return False
    ren = (meta.get("renderer") or {}).get("renderer")
    want_ply = RD.ply_sha256(scene_id) if renderer_kind == "gsplat" else None
    return (meta.get("params") == build_params(iphone_every, **kw) and meta.get("iphone_convention") == convention
            and ren == renderer_kind and meta.get("ply_sha256") == want_ply and meta.get("scene_id") == scene_id)


def build_scene(scene_id: str, episodes: list[dict], renderer, out_dir: Path | None = None,
                convention: str = "opencv", iphone_every: int = IPHONE_EVERY, md: dict | None = None,
                use_iphone: bool = True, convention_source: str | None = None, **kw) -> tuple[Grid, dict]:
    """Build and save one scene's grid; returns (grid, stats)."""
    t0 = time.time()
    if md is None and use_iphone:
        md = RD.iphone_metadata(scene_id)
    views = scene_views(scene_id, episodes, md if use_iphone else None, convention, iphone_every,
                        kw.get("template_wh", TEMPLATE_DEPTH_WH), kw.get("iphone_scale", IPHONE_DEPTH_SCALE))
    voxel = kw.get("voxel", VOXEL_M)
    min_hits = kw.get("min_hits", MIN_HITS)
    keys, counts, vstats = accumulate(views, renderer, voxel, kw.get("alpha_min", ALPHA_MIN),
                                      kw.get("max_depth", MAX_DEPTH_M))
    occ = counts >= min_hits
    meta = {"scene_id": scene_id, "renderer": getattr(renderer, "meta", {"renderer": type(renderer).__name__}),
            "ply_sha256": RD.ply_sha256(scene_id) if getattr(renderer, "name", "") == "gsplat" else None,
            "params": build_params(iphone_every, **kw),
            "iphone_convention": convention if use_iphone else None,
            "iphone_convention_source": convention_source if use_iphone else None,
            "n_views": len(views), "n_iphone_views": sum(v[0].startswith("iphone") for v in views),
            "n_hit_voxels": int(keys.size), "n_occupied": int(occ.sum()),
            "views": [v["view"] for v in vstats], "seconds": None,
            "code": code_meta(Path(__file__), Path(RD.__file__)),
            "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    grid = Grid(keys[occ], voxel=voxel, margin=kw.get("margin", MARGIN_VOX), extend=kw.get("extend", EXTEND_M),
                hits=counts[occ], meta=meta)
    meta["seconds"] = round(time.time() - t0, 1)
    grid.meta = meta
    stats = {"scene_id": scene_id, "n_views": len(views), "n_occupied": meta["n_occupied"],
             "seconds": meta["seconds"]}
    if out_dir is not None:
        stats["sha256"] = grid.save(occ_path(scene_id, out_dir))
        stats["path"] = str(occ_path(scene_id, out_dir))
    return grid, stats


# ------------------------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------------------------
def validate_scene(grid: Grid, iphone_c2w, viewpoints, starts=None) -> dict:
    """Counts for one scene: colliding consecutive iPhone segments; viewpoints inside / near occupied voxels;
    episode starts [(split, c2w)] inside / near (within the margin) per split (descriptive)."""
    pos = np.asarray(iphone_c2w, dtype=np.float64)[:, :3, 3] if iphone_c2w is not None and len(iphone_c2w) else \
        np.zeros((0, 3))
    seg = [bool(grid(pos[i], pos[i + 1])) for i in range(len(pos) - 1)]
    vp = np.array([c[:3, 3] for _, c in viewpoints]) if viewpoints else np.zeros((0, 3))
    inside = grid.occupied(vp) if len(vp) else np.zeros(0, bool)
    near = grid.near_occupied(vp) if len(vp) else np.zeros(0, bool)
    st = {}
    for sp, c in starts or ():
        p = np.asarray(c, dtype=np.float64)[:3, 3]
        r = st.setdefault(sp, {"n": 0, "inside": 0, "near": 0})
        r["n"] += 1
        r["inside"] += int(grid.occupied(p)[0])
        r["near"] += int(grid.near_occupied(p)[0])
    return {"iphone_segments": len(seg), "iphone_colliding": int(sum(seg)),
            "viewpoints": int(len(vp)), "viewpoints_inside": int(inside.sum()), "viewpoints_near": int(near.sum()),
            "inside_tags": [viewpoints[i][0] for i in np.flatnonzero(inside)][:20], "starts": st}


def _frac(a, b):
    return None if b == 0 else round(a / b, 6)


def pooled_validation(per_scene: dict, max_frac: float = VALID_MAX_FRAC) -> dict:
    seg = sum(r["iphone_segments"] for r in per_scene.values())
    col = sum(r["iphone_colliding"] for r in per_scene.values())
    vp = sum(r["viewpoints"] for r in per_scene.values())
    ins = sum(r["viewpoints_inside"] for r in per_scene.values())
    near = sum(r["viewpoints_near"] for r in per_scene.values())
    f_seg, f_in = _frac(col, seg), _frac(ins, vp)
    per_seg = [r["iphone_colliding"] / r["iphone_segments"] for r in per_scene.values() if r["iphone_segments"]]
    per_in = [r["viewpoints_inside"] / r["viewpoints"] for r in per_scene.values() if r["viewpoints"]]
    starts = {}
    for r in per_scene.values():
        for sp, x in (r.get("starts") or {}).items():
            t = starts.setdefault(sp, {"n": 0, "inside": 0, "near": 0})
            for k in t:
                t[k] += x[k]
    for t in starts.values():
        t["inside_frac"], t["near_frac"] = _frac(t["inside"], t["n"]), _frac(t["near"], t["n"])
    return {"iphone_segments": seg, "iphone_colliding": col, "iphone_colliding_frac": f_seg,
            "viewpoints": vp, "viewpoints_inside": ins, "viewpoints_inside_frac": f_in,
            "viewpoints_near": near, "viewpoints_near_frac": _frac(near, vp),
            "max_scene_iphone_colliding_frac": round(max(per_seg), 6) if per_seg else None,
            "max_scene_viewpoints_inside_frac": round(max(per_in), 6) if per_in else None,
            "starts_by_split": starts,
            "starts_note": "descriptive: an episode whose start is within the 1-voxel margin can only rotate "
                           "(every translation collides under the frozen rule)",
            "threshold": max_frac,
            "pass_iphone": f_seg is not None and f_seg < max_frac,
            "pass_viewpoints": f_in is not None and f_in < max_frac}


def grids_key(shas: dict) -> str:
    """Identity of a set of grids: sha256 over the sorted 'scene sha256' lines."""
    text = "".join(f"{s} {shas[s]}\n" for s in sorted(shas))
    return hashlib.sha256(text.encode()).hexdigest()


def validate(scenes: list[str], episodes: list[dict], directory: Path | None = None, out: Path | None = None,
             iphone=None) -> dict:
    """Validate the saved grids of `scenes`; `iphone` maps scene -> (N,4,4) c2w (default: InsScene metadata)."""
    directory = Path(directory or occ_dir())
    per = {}
    for sid in scenes:
        grid = Grid.load(occ_path(sid, directory))
        conv = grid.meta.get("iphone_convention") or "opencv"
        if iphone is not None:
            traj = iphone.get(sid)
        else:
            traj = np.array([RD.convention_c2w(c, conv) for c in RD.iphone_metadata(sid)["c2w"]])
        starts = [(e.get("split"), e["start_c2w"]) for e in sorted(episodes, key=lambda e: e["episode_index"])
                  if e["scene_id"] == sid]
        per[sid] = validate_scene(grid, traj, RD.scene_viewpoints(episodes, sid), starts)
        per[sid]["sha256"] = sha256_file(occ_path(sid, directory))
        per[sid]["iphone_convention"] = grid.meta.get("iphone_convention")
        per[sid]["iphone_convention_source"] = grid.meta.get("iphone_convention_source")
    convs = sorted({str(r["iphone_convention"]) for r in per.values()})
    srcs = sorted({str(r["iphone_convention_source"]) for r in per.values()})
    res = {"scenes": per, "pooled": pooled_validation(per), "n_scenes": len(per),
           "grids_sha256": grids_key({s: r["sha256"] for s, r in per.items()}),
           "iphone_convention": {"conventions": convs, "sources": srcs, "uniform": len(convs) == 1,
                                 "note": "per npz meta; 'None' = built without iPhone poses (convention unused) or "
                                         "by an older build that did not record the source"},
           "definition": "PREREG_E3VS_PILOT section 5: pooled iPhone segment collision fraction < 1% and pooled "
                         "viewpoint-in-occupied-voxel fraction < 1%",
           "params": {"voxel": VOXEL_M, "min_hits": MIN_HITS, "margin": MARGIN_VOX, "extend": EXTEND_M},
           "code": code_meta(Path(__file__)), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    res["pass"] = bool(res["pooled"]["pass_iphone"] and res["pooled"]["pass_viewpoints"])
    out = Path(out) if out else directory / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, out)
    return res


def check(scenes: list[str], directory: Path | None = None, validation: Path | None = None) -> dict:
    """validation.json passes, covers `scenes`, its grids_sha256 matches its scene list, and every npz is unchanged."""
    directory = Path(directory or occ_dir())
    vpath = Path(validation) if validation else directory / "validation.json"
    if not vpath.is_file():
        return {"ok": False, "problems": [f"no {vpath}"], "grids_sha256": None}
    v = json.loads(vpath.read_text())
    probs = [] if v.get("pass") is True else [f"{vpath}: pass is {v.get('pass')!r}"]
    rec = v.get("scenes") or {}
    key = v.get("grids_sha256")
    if not rec or key != grids_key({s: r.get("sha256") for s, r in rec.items()}):
        probs.append("grids_sha256 missing or does not match the scene list")
    for s in scenes:
        p = occ_path(s, directory)
        if s not in rec:
            probs.append(f"{s}: not in {vpath.name}")
        elif not p.is_file():
            probs.append(f"{s}: {p} missing")
        elif sha256_file(p) != rec[s].get("sha256"):
            probs.append(f"{s}: {p.name} changed since validation")
    return {"ok": not probs, "problems": probs[:50], "n_problems": len(probs), "grids_sha256": key,
            "n_scenes": len(scenes)}


# ------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------
def scene_list(spec: str, episodes: list[dict]) -> list[str]:
    by_split = {sp: sorted({e["scene_id"] for e in episodes if e["split"] == sp}) for sp in SPLITS}
    if spec == "all":
        return by_split["val"] + by_split["test"] + by_split["train"]
    if spec in by_split:
        return by_split[spec]
    ids = [s for s in spec.split(",") if s]
    known = {e["scene_id"] for e in episodes}
    bad = [s for s in ids if s not in known]
    if bad:
        raise SystemExit(f"unknown scenes {bad}")
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "validate", "check"))
    ap.add_argument("--scenes", default="all", help="all | train | val | test | ID,ID,...")
    ap.add_argument("--resume", action="store_true", help="build: keep npz files built with the same settings")
    ap.add_argument("--out", type=Path, default=None, help="grid directory (default OUT/e3vs/occ)")
    ap.add_argument("--renderer", default="gsplat", choices=("gsplat", "fake"))
    ap.add_argument("--convention", default="auto", choices=("auto",) + RD.CONVENTIONS,
                    help="iPhone pose convention (auto = the parity files' choice)")
    ap.add_argument("--iphone-every", type=int, default=IPHONE_EVERY)
    ap.add_argument("--no-iphone", action="store_true",
                    help="build from the E3VS viewpoints only (a fallback if parity shows the iPhone poses are not "
                         "in the splat frame; it changes the frozen design, so only with the user's OK)")
    ap.add_argument("--validate", action="store_true", help="build: validate afterwards")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.iphone_every < 1:
        ap.error("--iphone-every must be >= 1 (use --no-iphone to leave the iPhone poses out)")
    out = Path(a.out) if a.out else occ_dir()
    if a.cmd == "check" and a.scenes not in ("all",) + SPLITS:
        scenes, episodes = [s for s in a.scenes.split(",") if s], None  # explicit ids: no repo needed
    else:
        repo = Path(a.repo) if a.repo else repo_dir()
        episodes = [e for sp in SPLITS for e in load_episodes(sp, repo)]
        scenes = scene_list(a.scenes, episodes)
    if a.cmd == "check":
        r = check(scenes, out)
        if r["ok"]:
            print(r["grids_sha256"])
            return 0
        print(json.dumps(r), file=sys.stderr)
        return 1
    if a.cmd == "build":
        if a.no_iphone:
            conv, src = "opencv", "unused (--no-iphone)"
        else:
            conv, src = resolve_convention(a.convention)
        print(f"[occupancy] {len(scenes)} scenes -> {out}; iPhone convention {conv} ({src})", flush=True)
        for i, sid in enumerate(scenes):
            if a.resume and reusable(occ_path(sid, out), a.renderer, sid, None if a.no_iphone else conv,
                                     a.iphone_every):
                print(f"[occupancy] {i + 1}/{len(scenes)} {sid}: kept (--resume, same settings)", flush=True)
                continue
            renderer = RD.make_renderer(a.renderer, sid, device=a.device)
            _, st = build_scene(sid, episodes, renderer, out, convention=conv, iphone_every=a.iphone_every,
                                use_iphone=not a.no_iphone, convention_source=src)
            del renderer
            print(f"[occupancy] {i + 1}/{len(scenes)} {sid}: {st['n_views']} views, {st['n_occupied']} occupied, "
                  f"{st['seconds']} s", flush=True)
        if not a.validate:
            return 0
    res = validate(scenes, episodes, out)
    print(json.dumps({"pass": res["pass"], **{k: res["pooled"][k] for k in (
        "iphone_colliding_frac", "viewpoints_inside_frac", "viewpoints_near_frac")},
        "starts_near_frac": {sp: t["near_frac"] for sp, t in res["pooled"]["starts_by_split"].items()},
        "iphone_convention": res["iphone_convention"], "grids_sha256": res["grids_sha256"]}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
