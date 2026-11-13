"""E3VS collision model: the official ScanNet++ mesh + Open3D ray casting of the pinned E3VS-Bench repo.

Replaces the splat occupancy grid of occupancy.py (kept in the tree for the record), whose validation failed
(DECISIONS G19b / G19c); approved by the user on 2026-11-12 (PREREG_E3VS_PILOT section 16).  One collision object
serves the BFS demos, the blind floor and the closed-loop eval: --collides vpd.e3vs.mesh_collision:load_collider.

No geometry is re-implemented here.  The official class ``Collisions`` of E3VS-BENCH @ bb7e603c,
``e3vs/eval/metrics.py`` (sha256 METRICS_SHA256; the file hash and the repo commit are asserted before use), is
executed from the pinned checkout by file path (no package __init__, no bytecode written):
  load_scene_mesh   o3d.io.read_triangle_mesh(<scene>/scans/mesh_aligned_0.05.ply, the official configs'
                    collision_mesh_path) -> o3d.t.geometry.TriangleMesh.from_legacy -> RaycastingScene.add_triangles;
  check_trajectory_collision(p0s, p1s, tolerance=1e-4)   a float32 ray from p0 along the unit direction to p1;
                    collided iff its first hit t_hit is in (tolerance, |p1 - p0| - tolerance).
The official evaluator (evaluator.py, action_setting "restrict_collision") asks it for every move and skips a
colliding move (the pose stays, the step is consumed, the agent is told).  Every move check here is one call of
check_trajectory_collision with its default tolerance (TOLERANCE_M = 1e-4 m, asserted against the signature).

Rule (ii) of PREREG section 5 under the official semantics: a move p0 -> p1 collides iff the official check says
so.  No voxel margin, no 0.05 m extension past p1, no start skip: the only allowance at either end is the official
tolerance.  A zero-length segment never collides (t_hit < -tolerance is impossible); rotations are never asked.

"In obstacle" for a viewpoint (template, start, goal): its unsigned distance to the nearest mesh triangle
(RaycastingScene.compute_distance of the official scene object, an Open3D call) < IN_OBSTACLE_M = 1e-4 m, the
quantity the official check uses.  Descriptive "near": distance < NEAR_M = 0.05 m (the old 1-voxel margin).

validate  (prep job vpd-e3vs-prep, step mesh_val; CPU only)
  inputs   OUT/e3vs/mesh_manifest.json (vpd.e3vs.mesh_verify) must exist, be ok, have a consistent
           mesh_set_sha256 and list every scene, and every mesh file must still have the manifest's sha256;
           otherwise the run is REFUSED and nothing is written.
  gating   (PREREG section 5 thresholds: each < 1 %, pooled over the scenes)
           1. the share of consecutive iPhone-frame segments that collide: every pair (i, i + 1) of InsScene's
              scene_iphone_metadata.npz trajectory, camera centres c2w[:3, 3] (the pairs of the occupancy
              validation; the centre does not depend on the OpenCV / OpenGL convention, which is recorded);
           2. the share of template viewpoints (render.scene_viewpoints: starts, answerable, unanswerable) in
              obstacle.
  descriptive (never gating)  viewpoints near; distance quantiles; per split the episode starts inside / near,
           the share of the 6 start translations that collide and of the starts with all 6 blocked; iPhone frame
           sanity: the share of camera centres inside the mesh AABB and their distance to the mesh.
  output   OUT/e3vs/mesh_val/validation.json; its mesh_set_sha256 (mesh_verify.mesh_set_sha256 of the validated
           per-scene sha256s, for the 99 scenes equal to the manifest's) is the key the prep job stamps on
           everything built with this model (BFS, floors, SFT) and that the start jobs and the gate check.
check     validation.json passes, was made with this definition (official file, commit, tolerance, threshold),
          its key matches its scene list and the manifest, it covers the asked scenes, and every asked mesh still
          has the validated sha256 (re-hashed); prints mesh_set_sha256 (exit 0) or the problems (exit 1).
          Needs no open3d (the job dry runs and the start-job guard use it).
load_collider(scene_id)  the --collides factory: refuses unless validation.json passes check's definition tests
          and the scene's mesh file has the validated sha256 (a missing mesh never silently means "no collision").

Open3D is not in PY_Q3.  It lives in O3D_DIR = OUT/envs/e3vs-o3d (pip --target, after the user's OK), which only
the E3VS jobs put on PYTHONPATH; it is imported lazily (validate, MeshCollider), never at module import, and a
missing open3d raises Open3DMissing with that path.  Env overrides for tests and job dry runs (the jobs unset them
in production): VPD_E3VS_MESH_ROOT (the mesh_dl dir), VPD_E3VS_MESH_MANIFEST, VPD_E3VS_MESH_VAL_DIR.

CLI (sco job vpd-e3vs-prep; 1 thread):
    python -m vpd.e3vs.mesh_collision validate --scenes all|train|val|test|ID,ID [--out DIR] [--convention auto]
    python -m vpd.e3vs.mesh_collision check --scenes all|ID,ID [--out DIR]
validate exits 0 when both gating metrics pass, 1 when one fails (stop and ask the user), 2 when refused.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from vpd import paths
from vpd.e3vs import (ACTIONS, E3VS_COMMIT, SPLITS, TRANSLATIONS, code_meta, load_episodes, repo_commit, repo_dir,
                      sha256_file)
from vpd.e3vs import geometry as G
from vpd.e3vs import mesh_verify as MV
from vpd.e3vs import render as RD

FACTORY = "vpd.e3vs.mesh_collision:load_collider"
TAG = "mesh"  # output tag of the prep job: OUT/e3vs/bfs/mesh/<split>/, OUT/e3vs/floors/floors_mesh.json
METRICS_REL = Path("e3vs") / "eval" / "metrics.py"
METRICS_SHA256 = "c69b2630415fd167c46d1a7a94daf4a758be99e5c86932639178af16623b0065"  # bb7e603c
OFFICIAL_CLASS = "Collisions"
TOLERANCE_M = 1e-4  # official check_trajectory_collision default
IN_OBSTACLE_M = TOLERANCE_M
NEAR_M = 0.05  # descriptive only: the occupancy grid's 1-voxel margin
VALID_MAX_FRAC = 0.01
MOVES = tuple(a for a in ACTIONS if a in TRANSLATIONS)  # the 6 translations, official order
O3D_DIR = paths.ENVS / "e3vs-o3d"  # pip --target dir of open3d; the E3VS jobs add it to PYTHONPATH
FRAME_WARN_IN_AABB = 0.9
DEFINITION = ("PREREG_E3VS_PILOT section 5 with the section-16 row of 2026-11-12/13: official E3VS Collisions "
              "(metrics.py) on the ScanNet++ mesh_aligned_0.05.ply, check_trajectory_collision with its default "
              "tolerance 1e-4 m; gating: pooled iPhone consecutive-frame segment collision share < 1% and pooled "
              "template-viewpoint in-obstacle share (unsigned distance to the mesh < 1e-4 m) < 1%")


class Open3DMissing(ImportError):
    """open3d cannot be imported (it is installed into O3D_DIR only after the user's OK)."""


class MeshRefused(RuntimeError):
    """The mesh inputs are not the verified ones (manifest missing / not ok / sha256 differs)."""


# ------------------------------------------------------------------------------------------------------------
# locations
# ------------------------------------------------------------------------------------------------------------
def mesh_root() -> Path:
    return Path(os.environ.get("VPD_E3VS_MESH_ROOT") or MV.MESH_DL)


def manifest_path() -> Path:
    return Path(os.environ.get("VPD_E3VS_MESH_MANIFEST") or paths.E3VS_OUT / MV.MANIFEST)


def val_dir() -> Path:
    return Path(os.environ.get("VPD_E3VS_MESH_VAL_DIR") or paths.E3VS_OUT / "mesh_val")


def validation_path(directory: Path | None = None) -> Path:
    return Path(directory or val_dir()) / "validation.json"


def mesh_path(scene_id: str, root: Path | None = None) -> Path:
    return Path(root or mesh_root()) / scene_id / MV.MESH_REL


# ------------------------------------------------------------------------------------------------------------
# open3d and the official class
# ------------------------------------------------------------------------------------------------------------
def open3d():
    """The open3d module, imported lazily; Open3DMissing (with O3D_DIR) when it is not importable."""
    try:
        import open3d as o3d
    except ImportError as e:
        raise Open3DMissing(
            f"open3d is not importable ({type(e).__name__}: {e}).  The E3VS collision model runs the official "
            f"Open3D ray casting: open3d is installed (after the user's OK) with pip --target into {O3D_DIR}, which "
            f"the E3VS jobs (vpd/jobs/e3vs_prep.sh, e3vs_start.sh) put on PYTHONPATH; by hand: "
            f"PYTHONPATH=<code tree>:{O3D_DIR}") from e
    import numpy

    npf = Path(numpy.__file__).resolve()
    if Path(O3D_DIR).resolve() in npf.parents:
        raise ImportError(f"numpy is imported from {npf}, inside {O3D_DIR}: that copy shadows the env's numpy "
                          "(install open3d there with --no-deps)")
    return o3d


_OFFICIAL: dict = {}
_OFFICIAL_LOCK = threading.Lock()


def official_facts(repo: Path | None = None) -> dict:
    """Assert the pinned commit and the metrics.py sha256 (no import); the facts recorded with every use."""
    repo = Path(repo) if repo else repo_dir()
    commit = repo_commit(repo)
    if commit != E3VS_COMMIT:
        raise AssertionError(f"{repo}: HEAD {commit} is not the pinned {E3VS_COMMIT} (official collision code)")
    p = repo / METRICS_REL
    sha = sha256_file(p)
    if sha != METRICS_SHA256:
        raise AssertionError(f"{p}: sha256 {sha} != the pinned {METRICS_SHA256}")
    return {"repo": str(repo), "commit": commit, "file": str(METRICS_REL), "sha256": sha, "class": OFFICIAL_CLASS,
            "method": "check_trajectory_collision", "tolerance_m": TOLERANCE_M}


def official_metrics(repo: Path | None = None):
    """(module, facts): the pinned metrics.py executed by path (once per process), after official_facts."""
    facts = official_facts(repo)
    path = Path(facts["repo"]) / METRICS_REL
    key = str(path.resolve())
    with _OFFICIAL_LOCK:
        if key not in _OFFICIAL:
            o3d = open3d()  # metrics.py imports open3d at module level: give the clear error first
            mod = RD._load_by_path("_e3vs_official_metrics", path)
            tol = inspect.signature(mod.Collisions.check_trajectory_collision).parameters["tolerance"].default
            if tol != TOLERANCE_M:
                raise AssertionError(f"{path}: check_trajectory_collision default tolerance {tol} != {TOLERANCE_M}")
            f = getattr(o3d, "__file__", None)
            _OFFICIAL[key] = (mod, {**facts, "open3d_version": getattr(o3d, "__version__", None),
                                    "open3d_path": str(Path(f).parent) if f else None})
        return _OFFICIAL[key]


def loaded_official() -> list[dict]:
    """Facts (commit, file sha256, open3d version and path) of the official code this process executed."""
    with _OFFICIAL_LOCK:
        return [dict(f) for _, f in _OFFICIAL.values()]


class MeshCollider:
    """One scene's official Collisions object.  ``collider(p0, p1)`` is the move check of the BFS, the floor and
    the closed loop; collides_many / distance / aabb serve the validation.  Calls are serialised by a lock (the
    closed loop asks from several threads)."""

    def __init__(self, scene_id: str, path: Path | None = None, repo: Path | None = None,
                 sha256: str | None = None):
        self.scene_id = scene_id
        self.path = Path(path) if path is not None else mesh_path(scene_id)
        if not self.path.is_file():
            raise FileNotFoundError(f"no collision mesh for scene {scene_id} at {self.path} (the user downloads the "
                                    "ScanNet++ meshes; python -m vpd.e3vs.mesh_verify)")
        mod, facts = official_metrics(repo)
        self.o3d = open3d()
        self._lock = threading.Lock()
        t0 = time.time()
        template = str(self.path).replace("{", "{{").replace("}", "}}")  # the official .format(scene_id=...)
        self.official = getattr(mod, OFFICIAL_CLASS)(mesh_path_template=template)
        self.official.load_scene_mesh(scene_id)
        if self.official.scene is None:  # the official loader only logs a warning for a missing file
            raise RuntimeError(f"official Collisions.load_scene_mesh built no scene from {self.path}")
        d = self.distance(np.zeros((1, 3)))
        if not np.isfinite(d).all():
            raise RuntimeError(f"{self.path}: the ray-casting scene has no triangles (distance {d.tolist()})")
        self.meta = {"scene_id": scene_id, "mesh_path": str(self.path), "mesh_sha256": sha256, "official": facts,
                     "load_seconds": round(time.time() - t0, 3)}

    def collides_many(self, p0, p1) -> np.ndarray:
        """Official check_trajectory_collision (default tolerance) of the segments p0[i] -> p1[i]."""
        a = np.asarray(p0, dtype=np.float64).reshape(-1, 3)
        b = np.asarray(p1, dtype=np.float64).reshape(-1, 3)
        if a.shape != b.shape:
            raise ValueError(f"start / end shapes differ: {a.shape} vs {b.shape}")
        if not len(a):
            return np.zeros(0, dtype=bool)
        with self._lock:
            out = self.official.check_trajectory_collision(a, b)
        return np.asarray(out, dtype=bool).reshape(-1)

    def collides(self, p0, p1) -> bool:
        return bool(self.collides_many(p0, p1)[0])

    __call__ = collides

    def distance(self, points) -> np.ndarray:
        """Unsigned distance (m) to the nearest triangle: RaycastingScene.compute_distance of the official scene."""
        q = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if not len(q):
            return np.zeros(0)
        with self._lock:
            d = self.official.scene.compute_distance(self.o3d.core.Tensor(q, dtype=self.o3d.core.Dtype.Float32))
        return np.asarray(d.numpy(), dtype=np.float64).reshape(-1)

    def in_obstacle(self, points) -> np.ndarray:
        return self.distance(points) < IN_OBSTACLE_M

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """The mesh's axis-aligned bounding box (Open3D, a second read of the file; validation only)."""
        bb = self.o3d.io.read_triangle_mesh(str(self.path)).get_axis_aligned_bounding_box()
        return np.asarray(bb.min_bound, dtype=np.float64), np.asarray(bb.max_bound, dtype=np.float64)


# ------------------------------------------------------------------------------------------------------------
# the frozen factory
# ------------------------------------------------------------------------------------------------------------
_MEMO: dict = {}


def mesh_file_sha256(p: Path) -> str:
    """sha256 of a mesh, memoised per process on (path, size, mtime) (repeated loads of one scene)."""
    st = p.stat()
    k = ("sha", str(p.resolve()), st.st_size, st.st_mtime_ns)
    if k not in _MEMO:
        _MEMO[k] = sha256_file(p)
    return _MEMO[k]


def load_validation(directory: Path | None = None) -> dict | None:
    """validation.json (read on every call: it is small), None if missing."""
    p = validation_path(directory)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def definition_problems(v: dict) -> list[str]:
    """Why a validation.json is not one of this collision model (empty list: it is)."""
    probs = []
    off = v.get("official") or {}
    if off.get("sha256") != METRICS_SHA256 or off.get("commit") != E3VS_COMMIT:
        probs.append(f"made with other official code (metrics.py {str(off.get('sha256'))[:12]}, commit "
                     f"{str(off.get('commit'))[:12]})")
    if v.get("tolerance_m") != TOLERANCE_M or v.get("in_obstacle_m") != IN_OBSTACLE_M:
        probs.append(f"tolerance {v.get('tolerance_m')!r} / in-obstacle {v.get('in_obstacle_m')!r} != {TOLERANCE_M}")
    if (v.get("pooled") or {}).get("threshold") != VALID_MAX_FRAC:
        probs.append(f"threshold {(v.get('pooled') or {}).get('threshold')!r} != {VALID_MAX_FRAC}")
    rec = v.get("scenes") or {}
    if not rec or v.get("mesh_set_sha256") != MV.mesh_set_sha256({s: str(r.get("sha256")) for s, r in rec.items()}):
        probs.append("mesh_set_sha256 missing or does not match the scene list")
    return probs


def load_collider(scene_id: str) -> MeshCollider:
    """Factory for --collides vpd.e3vs.mesh_collision:load_collider: the scene's validated mesh (a callable)."""
    vp = validation_path()
    v = load_validation()
    if v is None:
        raise FileNotFoundError(f"no mesh validation {vp} (e3vs-prep step mesh_val: python -m "
                                "vpd.e3vs.mesh_collision validate)")
    probs = definition_problems(v)
    if v.get("pass") is not True or probs:
        raise AssertionError(f"{vp} is not a passing validation of this collision model (pass {v.get('pass')!r}; "
                             f"{'; '.join(probs) or 'definition ok'})")
    rec = (v.get("scenes") or {}).get(scene_id)
    if rec is None:
        raise AssertionError(f"scene {scene_id} is not in {vp}")
    p = mesh_path(scene_id)
    if not p.is_file():
        raise FileNotFoundError(f"no collision mesh for scene {scene_id} at {p}")
    sha = mesh_file_sha256(p)
    if sha != rec.get("sha256"):
        raise AssertionError(f"{p}: sha256 {sha[:12]} != the validated {str(rec.get('sha256'))[:12]} ({vp})")
    return MeshCollider(scene_id, p, sha256=sha)


# ------------------------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------------------------
def _quant(x) -> dict | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not x.size:
        return None
    q = np.quantile(x, [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {"n": int(x.size), **{k: round(float(v), 6) for k, v in zip(
        ("min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "max"), q)}}


def _frac(a, b):
    return None if b == 0 else round(a / b, 6)


def validate_scene(col, iphone_c2w, viewpoints, starts=None) -> tuple[dict, dict]:
    """(counts, raw distances) of one scene; `col` has collides_many, distance and aabb (a MeshCollider).
    iphone_c2w (N, 4, 4): segments (i, i + 1) of the camera centres; viewpoints [(tag, c2w)]; starts
    [(split, c2w)] (descriptive)."""
    pos = np.asarray(iphone_c2w, dtype=np.float64)[:, :3, 3] if iphone_c2w is not None and len(iphone_c2w) else \
        np.zeros((0, 3))
    seg = col.collides_many(pos[:-1], pos[1:]) if len(pos) > 1 else np.zeros(0, dtype=bool)
    d_ip = col.distance(pos) if len(pos) else np.zeros(0)
    box = col.aabb() if len(pos) else None
    in_box = ((pos >= box[0]) & (pos <= box[1])).all(axis=1) if box is not None else np.zeros(0, dtype=bool)
    vp = np.array([np.asarray(c, dtype=np.float64)[:3, 3] for _, c in viewpoints]) if viewpoints else \
        np.zeros((0, 3))
    d_vp = col.distance(vp) if len(vp) else np.zeros(0)
    inside, near = d_vp < IN_OBSTACLE_M, d_vp < NEAR_M
    rows = [(sp, np.asarray(c, dtype=np.float64)) for sp, c in starts or ()]
    st, d_st = {}, {}
    if rows:
        p0 = np.array([c[:3, 3] for _, c in rows])
        ds = col.distance(p0)
        ends = np.array([G.apply_action(c, m)[:3, 3] for _, c in rows for m in MOVES])
        hit = col.collides_many(np.repeat(p0, len(MOVES), axis=0), ends).reshape(len(rows), len(MOVES))
        for i, (sp, _) in enumerate(rows):
            r = st.setdefault(sp, {"n": 0, "inside": 0, "near": 0, "moves": 0, "moves_colliding": 0,
                                   "all_blocked": 0})
            r["n"] += 1
            r["inside"] += int(ds[i] < IN_OBSTACLE_M)
            r["near"] += int(ds[i] < NEAR_M)
            r["moves"] += len(MOVES)
            r["moves_colliding"] += int(hit[i].sum())
            r["all_blocked"] += int(hit[i].all())
            d_st.setdefault(sp, []).append(float(ds[i]))
    rec = {"iphone_frames": int(len(pos)), "iphone_segments": int(len(seg)), "iphone_colliding": int(seg.sum()),
           "iphone_in_aabb": int(in_box.sum()),
           "aabb": None if box is None else [np.round(box[0], 4).tolist(), np.round(box[1], 4).tolist()],
           "viewpoints": int(len(vp)), "viewpoints_inside": int(inside.sum()), "viewpoints_near": int(near.sum()),
           "inside_tags": [viewpoints[i][0] for i in np.flatnonzero(inside)][:20],
           "distance_m": {"viewpoints": _quant(d_vp), "iphone": _quant(d_ip)}, "starts": st}
    return rec, {"viewpoints": d_vp, "iphone": d_ip, "starts": d_st}


def pooled_validation(per: dict, raw: dict | None = None, max_frac: float = VALID_MAX_FRAC) -> dict:
    tot = {k: sum(r[k] for r in per.values()) for k in ("iphone_frames", "iphone_segments", "iphone_colliding",
                                                         "iphone_in_aabb", "viewpoints", "viewpoints_inside",
                                                         "viewpoints_near")}
    f_seg = _frac(tot["iphone_colliding"], tot["iphone_segments"])
    f_in = _frac(tot["viewpoints_inside"], tot["viewpoints"])
    per_seg = [r["iphone_colliding"] / r["iphone_segments"] for r in per.values() if r["iphone_segments"]]
    per_in = [r["viewpoints_inside"] / r["viewpoints"] for r in per.values() if r["viewpoints"]]
    starts = {}
    for r in per.values():
        for sp, x in (r.get("starts") or {}).items():
            t = starts.setdefault(sp, {k: 0 for k in x})
            for k in t:
                t[k] += x[k]
    for sp, t in starts.items():
        t.update(inside_frac=_frac(t["inside"], t["n"]), near_frac=_frac(t["near"], t["n"]),
                 moves_colliding_frac=_frac(t["moves_colliding"], t["moves"]),
                 all_blocked_frac=_frac(t["all_blocked"], t["n"]))
        if raw:
            t["distance_m"] = _quant([d for x in raw.values() for d in x["starts"].get(sp, [])])
    out = {**tot, "iphone_colliding_frac": f_seg, "viewpoints_inside_frac": f_in,
           "viewpoints_near_frac": _frac(tot["viewpoints_near"], tot["viewpoints"]),
           "max_scene_iphone_colliding_frac": round(max(per_seg), 6) if per_seg else None,
           "max_scene_viewpoints_inside_frac": round(max(per_in), 6) if per_in else None,
           "starts_by_split": starts,
           "iphone_frame": {"frames": tot["iphone_frames"], "in_aabb": tot["iphone_in_aabb"],
                            "in_aabb_frac": _frac(tot["iphone_in_aabb"], tot["iphone_frames"])},
           "notes": {"inside": f"unsigned distance to the mesh < {IN_OBSTACLE_M} m (gating for the viewpoints)",
                     "near": f"distance < {NEAR_M} m (descriptive; the occupancy grid's 1-voxel margin)",
                     "starts": "descriptive: inside / near starts, and the official move check of the 6 "
                               "translations from each start (all_blocked = rotation-only start)",
                     "iphone_frame": "descriptive frame sanity: iPhone camera centres inside the mesh AABB"},
           "threshold": max_frac,
           "pass_iphone": f_seg is not None and f_seg < max_frac,
           "pass_viewpoints": f_in is not None and f_in < max_frac}
    if raw:
        out["viewpoint_distance_m"] = _quant(np.concatenate([x["viewpoints"] for x in raw.values()]))
        out["iphone_frame"]["distance_m"] = _quant(np.concatenate([x["iphone"] for x in raw.values()]))
    return out


def verify_inputs(scenes: list[str], manifest: Path | None = None, root: Path | None = None) -> tuple[dict, dict]:
    """(manifest, {scene: sha256}) when the manifest exists, is ok, has a consistent key, lists every scene and
    every mesh file still has its sha256 (re-hashed); else MeshRefused with every problem."""
    p = Path(manifest) if manifest else manifest_path()
    if not p.is_file():
        raise MeshRefused(f"no mesh manifest {p} (python -m vpd.e3vs.mesh_verify, after the user's download)")
    man = json.loads(p.read_text())
    if man.get("ok") is not True:
        raise MeshRefused(f"{p}: ok is {man.get('ok')!r} (mesh_verify problems: {(man.get('problems') or [])[:3]})")
    rec = man.get("scenes") or {}
    msha = {s: str(r.get("sha256")) for s, r in rec.items()}
    probs = []
    if not man.get("mesh_set_sha256") or man["mesh_set_sha256"] != MV.mesh_set_sha256(msha):
        probs.append(f"{p}: mesh_set_sha256 missing or does not match its scene list")
    shas = {}
    for s in scenes:
        f = mesh_path(s, root)
        if s not in rec:
            probs.append(f"{s}: not in {p.name}")
        elif not f.is_file():
            probs.append(f"{s}: {f} missing")
        else:
            h = sha256_file(f)
            if h != msha[s]:
                probs.append(f"{s}: {f} sha256 {h[:12]} != the manifest's {msha[s][:12]}")
            shas[s] = h
    if probs:
        raise MeshRefused(f"{len(probs)} problems: " + "; ".join(probs[:10]))
    return man, shas


def iphone_convention(spec: str = "auto", parity_dir: Path | None = None) -> tuple[str, str]:
    """(convention, source) of the iPhone poses, for provenance only (the camera centre is the same in both)."""
    if spec != "auto":
        return spec, "given (--convention)"
    best = RD.best_iphone_convention(parity_dir)
    if best:
        return best, "parity"
    return "opencv", "default: no parity iPhone result (c2w[:3, 3] is the same in both conventions)"


def validate(scenes: list[str], episodes: list[dict], out_dir: Path | None = None, manifest: Path | None = None,
             root: Path | None = None, iphone: dict | None = None, repo: Path | None = None,
             convention: str = "auto") -> dict:
    """Validate the meshes of `scenes` (module docstring); writes and returns validation.json.  `iphone` maps
    scene -> (N, 4, 4) c2w (default: InsScene's metadata, in the recorded convention)."""
    out = validation_path(out_dir)
    man_p = Path(manifest) if manifest else manifest_path()
    man, shas = verify_inputs(scenes, man_p, root)
    _, official = official_metrics(repo)
    conv, src = iphone_convention(convention)
    per, raw = {}, {}
    for sid in scenes:
        t0 = time.time()
        col = MeshCollider(sid, mesh_path(sid, root), repo=repo, sha256=shas[sid])
        if iphone is not None:
            traj = iphone.get(sid)
        else:
            traj = np.array([RD.convention_c2w(c, conv) for c in RD.iphone_metadata(sid)["c2w"]])
        starts = [(e["split"], e["start_c2w"]) for e in sorted(episodes, key=lambda e: e["episode_index"])
                  if e["scene_id"] == sid]
        per[sid], raw[sid] = validate_scene(col, traj, RD.scene_viewpoints(episodes, sid), starts)
        per[sid].update(sha256=shas[sid], path=str(col.path), load_seconds=col.meta["load_seconds"],
                        seconds=round(time.time() - t0, 2))
        del col
    pooled = pooled_validation(per, raw)
    key = MV.mesh_set_sha256({s: r["sha256"] for s, r in per.items()})
    warn = []
    fr = pooled["iphone_frame"]["in_aabb_frac"]
    if fr is not None and fr < FRAME_WARN_IN_AABB:
        warn.append(f"only {fr:.1%} of the iPhone camera centres are inside the mesh AABB: are the iPhone poses in "
                    "the mesh frame?  The iPhone metric passes trivially if they are not (descriptive check)")
    res = {"scenes": per, "pooled": pooled, "n_scenes": len(per), "mesh_set_sha256": key,
           "manifest": {"path": str(man_p), "mesh_set_sha256": man.get("mesh_set_sha256"),
                        "verified_at": man.get("verified_at"), "n_ok": man.get("n_ok"),
                        "key_matches": key == man.get("mesh_set_sha256")},
           "official": official, "tolerance_m": TOLERANCE_M, "in_obstacle_m": IN_OBSTACLE_M, "near_m": NEAR_M,
           "factory": FACTORY,
           "iphone_convention": {"convention": conv, "source": src,
                                 "note": "provenance only: the camera centre c2w[:3, 3] is the same in both"},
           "definition": DEFINITION, "warnings": warn,
           "code": code_meta(Path(__file__), Path(MV.__file__), Path(RD.__file__), Path(G.__file__)),
           "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    res["pass"] = bool(pooled["pass_iphone"] and pooled["pass_viewpoints"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, out)
    return res


def check(scenes: list[str], directory: Path | None = None, manifest: Path | None = None,
          root: Path | None = None) -> dict:
    """validation.json passes with this definition, its key matches its scenes and the manifest, it covers
    `scenes`, and every mesh of `scenes` is unchanged (re-hashed).  No open3d needed."""
    vp = validation_path(directory)
    if not vp.is_file():
        return {"ok": False, "problems": [f"no {vp}"], "n_problems": 1, "mesh_set_sha256": None}
    v = json.loads(vp.read_text())
    probs = [] if v.get("pass") is True else [f"{vp}: pass is {v.get('pass')!r}"]
    probs += definition_problems(v)
    key = v.get("mesh_set_sha256")
    man_p = Path(manifest) if manifest else manifest_path()
    man = json.loads(man_p.read_text()) if man_p.is_file() else None
    if man is None or man.get("ok") is not True:
        probs.append(f"mesh manifest {man_p} missing or not ok")
    elif man.get("mesh_set_sha256") != key:
        probs.append(f"mesh set key {str(key)[:12]} != the manifest's {str(man.get('mesh_set_sha256'))[:12]}")
    if not probs:  # hash only when the records agree
        rec = v.get("scenes") or {}
        for s in scenes:
            p = mesh_path(s, root)
            if s not in rec:
                probs.append(f"{s}: not in {vp.name}")
            elif not p.is_file():
                probs.append(f"{s}: {p} missing")
            elif sha256_file(p) != rec[s].get("sha256"):
                probs.append(f"{s}: {p.name} changed since validation")
    return {"ok": not probs, "problems": probs[:50], "n_problems": len(probs), "mesh_set_sha256": key,
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


def summary(res: dict) -> dict:
    p = res["pooled"]
    return {"pass": res["pass"], **{k: p[k] for k in ("iphone_colliding_frac", "viewpoints_inside_frac",
                                                      "viewpoints_near_frac", "max_scene_iphone_colliding_frac",
                                                      "max_scene_viewpoints_inside_frac")},
            "starts": {sp: {k: t[k] for k in ("n", "inside_frac", "near_frac", "moves_colliding_frac",
                                              "all_blocked_frac")} for sp, t in p["starts_by_split"].items()},
            "iphone_frame": p["iphone_frame"], "warnings": res["warnings"],
            "iphone_convention": res["iphone_convention"]["convention"], "mesh_set_sha256": res["mesh_set_sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("validate", "check"))
    ap.add_argument("--scenes", default="all", help="all | train | val | test | ID,ID,...")
    ap.add_argument("--out", type=Path, default=None, help="validation dir (default OUT/e3vs/mesh_val)")
    ap.add_argument("--manifest", type=Path, default=None, help="default OUT/e3vs/mesh_manifest.json")
    ap.add_argument("--mesh-root", type=Path, default=None, help="default OUT/e3vs/mesh_dl")
    ap.add_argument("--convention", default="auto", choices=("auto",) + RD.CONVENTIONS,
                    help="iPhone pose convention, recorded only (auto = the parity files' choice)")
    ap.add_argument("--repo", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "check" and a.scenes not in ("all",) + SPLITS:
        scenes, episodes = [s for s in a.scenes.split(",") if s], None  # explicit ids: no repo needed
    else:
        repo = Path(a.repo) if a.repo else repo_dir()
        episodes = [e for sp in SPLITS for e in load_episodes(sp, repo)]
        scenes = scene_list(a.scenes, episodes)
    if a.cmd == "check":
        r = check(scenes, a.out, a.manifest, a.mesh_root)
        if r["ok"]:
            print(r["mesh_set_sha256"])
            return 0
        print(json.dumps(r), file=sys.stderr)
        return 1
    try:
        res = validate(scenes, episodes, a.out, a.manifest, a.mesh_root, repo=a.repo, convention=a.convention)
    except MeshRefused as e:
        print(f"[mesh_collision] REFUSED, nothing validated: {e}", file=sys.stderr)
        return 2
    print(json.dumps(summary(res)))
    for w in res["warnings"]:
        print(f"[mesh_collision] WARNING {w}", file=sys.stderr)
    if not res["pass"]:
        p = res["pooled"]
        print(f"[mesh_collision] mesh validation FAILED (threshold {VALID_MAX_FRAC:.0%} each): iPhone segments "
              f"colliding {p['iphone_colliding_frac']}, template viewpoints in obstacle {p['viewpoints_inside_frac']}"
              f".  Stop and ask the user (PREREG_E3VS_PILOT section 5; user 2026-11-12: 验证不过就停下问我).  "
              f"{validation_path(a.out)}", file=sys.stderr)
        return 1
    print(f"[mesh_collision] mesh validation passed: {res['n_scenes']} scenes, mesh set {res['mesh_set_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
