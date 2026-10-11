"""Generator-side occlusion and identifiability checks (workstream E).

The frozen canonical metric (``canonical_spatial_task_h1_v1``) projects AABB
corners only; it has no notion of occlusion.  The generator therefore adds a
geometric occlusion gate on top of it.  This module is the single place where
that gate is defined so that endpoint sampling (E), initial states (D) and
admission (A) agree.

Public API (keep stable; D and A import these exact signatures):

    visible_fraction(scene, c2w, obj, occluders=None, n=3) -> float
    identifiable(scene, c2w, obj, *, min_inside=0.5, min_area_ratio=0.002,
                 min_visible=0.5) -> dict

plus vectorised helpers used by ``endpoints.py``:

    occluders_for(scene) -> OccluderSet         (cached per scene object)
    target_samples(bmin, bmax, n) -> (n^3, 3)
    rays_occluded(occ, origins, points, exclude_ids) -> (P, K) bool
    frustum_mask(R, C, points) -> (N, K) bool

Occlusion model
---------------
* Sample points: an inset n x n x n grid inside the target AABB at fractions
  (i + 0.5) / n per axis (n = 3 -> 1/6, 1/2, 5/6).  Inset samples avoid false
  hits from supports that touch the target (a TV on a cabinet, a chair on a
  rug).
* A sample counts only if it is inside the canonical H1 image frustum
  (in front of the camera and projecting into [0, W-1] x [0, H-1] with the
  same K_effective the runtime metric uses).
* A sample is occluded when the segment camera-centre -> sample hits
  - another object's AABB (target itself excluded; the *other* task target is
    an occluder), skipping labels in NON_OCCLUDING_LABELS /
    NON_OCCLUDING_HEAD_NOUNS (matched on the full label or its last word), or
  - a structure wall: ``scene.detector.wall_segments`` extruded over
    z in [0, WALL_HEIGHT_M] (2.8 m), or
  - a thin axis-aligned "wall" label box (min horizontal thickness <=
    WALL_BOX_MAX_THICKNESS_M).  Thick "wall" label AABBs come from diagonal
    walls and would cover whole rooms, so they are skipped; structure walls
    still cover them.  Note the runtime CollisionDetector drops an entire
    room edge when it overlaps a door, so wall label boxes are what blocks
    cross-room rays on such edges.
* A camera centre inside an occluder AABB counts as occluded (conservative).
* visible_fraction = unoccluded in-frustum samples / in-frustum samples
  (0.0 when no sample is in the frustum).
"""

from __future__ import annotations

import math
import weakref
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vagen.envs.active_spatial.canonical_camera import (
    CANONICAL_CAMERA_H1_RESIZE_V1,
    build_canonical_camera,
    scale_intrinsics_for_resize,
)
from vagen.envs.active_spatial.visual_bbox_metrics import _project_object

from .contract import NATIVE_K, NATIVE_SIZE, RENDER_SIZE, Obj

OCCLUSION_MODEL_VERSION = "interiorgs_v10_aabb_wall_rays_v1"
WALL_HEIGHT_M = 2.8
WALL_BOX_MAX_THICKNESS_M = 0.35
DEFAULT_SAMPLES_PER_AXIS = 3
DEFAULT_MIN_VISIBLE_FRACTION = 0.5

# Exact labels that never occlude: floor coverings, structural shells handled
# elsewhere (walls) or never between a 1.5 m camera and a target (ceiling,
# overhead lighting), see-through or soft items.
NON_OCCLUDING_LABELS = frozenset({
    "rug", "carpet", "mat", "doormat", "floor mat", "bath mat",
    "floor", "ceiling", "room", "niche", "floor mold", "skirting", "baseboard",
    "light", "lights", "downlight", "downlights", "spotlight", "strip light",
    "ceiling lamp", "ceiling light", "pendant", "pendant lamp", "pendant light", "chandelier",
    "curtain", "curtains", "window", "window curtain",
})
# Head nouns (last word of the label) that never occlude, e.g. "ceiling light",
# "living room rug", "sheer curtain".  "floor lamp" -> "lamp" still occludes.
NON_OCCLUDING_HEAD_NOUNS = frozenset({
    "rug", "carpet", "mat", "doormat", "floor", "ceiling",
    "light", "lights", "downlight", "downlights", "spotlight", "chandelier",
    "curtain", "curtains", "window",
})
WALL_LABEL_HEAD_NOUNS = frozenset({"wall"})


def _norm_label(label: Any) -> str:
    return " ".join(str(label or "").strip().lower().split())


def is_non_occluding(label: Any) -> bool:
    text = _norm_label(label)
    if not text:
        return False
    if text in NON_OCCLUDING_LABELS:
        return True
    return text.split(" ")[-1] in NON_OCCLUDING_HEAD_NOUNS


def is_wall_label(label: Any) -> bool:
    text = _norm_label(label)
    return bool(text) and (text in WALL_LABEL_HEAD_NOUNS or text.split(" ")[-1] in WALL_LABEL_HEAD_NOUNS)


# --------------------------------------------------------------------------
# Canonical camera constants (identical to score_observation's H1 camera)
# --------------------------------------------------------------------------
_K_EFF = scale_intrinsics_for_resize(NATIVE_K, NATIVE_SIZE, RENDER_SIZE)
FX = float(_K_EFF[0, 0])
FY = float(_K_EFF[1, 1])
CX = float(_K_EFF[0, 2])
CY = float(_K_EFF[1, 2])
IMG_W, IMG_H = int(RENDER_SIZE[0]), int(RENDER_SIZE[1])


def canonical_intrinsics() -> np.ndarray:
    return _K_EFF.copy()


# --------------------------------------------------------------------------
# Target boxes
# --------------------------------------------------------------------------
def box_of(obj: Any) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """(id, label, bmin, bmax) for an Obj or a target_object entry dict."""
    if isinstance(obj, Obj):
        return str(obj.id), obj.label, np.asarray(obj.aabb_min, float), np.asarray(obj.aabb_max, float)
    if isinstance(obj, dict):
        bmin = obj.get("bbox_min", obj.get("aabb_min"))
        bmax = obj.get("bbox_max", obj.get("aabb_max"))
        return (str(obj.get("id", obj.get("ins_id", ""))), str(obj.get("label", "")),
                np.asarray(bmin, float), np.asarray(bmax, float))
    raise TypeError(f"unsupported object type {type(obj)!r}")


def entry_of(obj: Any) -> Dict[str, Any]:
    """Schema consumed by visual_bbox_metrics._project_object."""
    if isinstance(obj, Obj):
        return obj.target_object_entry()
    oid, label, bmin, bmax = box_of(obj)
    entry = dict(obj)
    entry.setdefault("id", oid)
    entry.setdefault("label", label)
    entry["bbox_min"] = [float(v) for v in bmin]
    entry["bbox_max"] = [float(v) for v in bmax]
    entry.setdefault("center", [float(v) for v in (bmin + bmax) / 2.0])
    return entry


def target_samples(bmin: np.ndarray, bmax: np.ndarray, n: int = DEFAULT_SAMPLES_PER_AXIS) -> np.ndarray:
    n = max(1, int(n))
    f = (np.arange(n, dtype=float) + 0.5) / n
    bmin = np.asarray(bmin, float)
    ext = np.asarray(bmax, float) - bmin
    gx, gy, gz = np.meshgrid(bmin[0] + f * ext[0], bmin[1] + f * ext[1], bmin[2] + f * ext[2], indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


# --------------------------------------------------------------------------
# Occluder set
# --------------------------------------------------------------------------
class OccluderSet:
    """Arrays of occluding AABBs plus extruded wall segments for one scene."""

    def __init__(self, ids: Sequence[str], labels: Sequence[str], mins: np.ndarray, maxs: np.ndarray,
                 wall_a: np.ndarray, wall_b: np.ndarray, wall_height: float = WALL_HEIGHT_M) -> None:
        self.ids = np.asarray([str(i) for i in ids], dtype=object)
        self.labels = list(labels)
        self.mins = np.asarray(mins, float).reshape(-1, 3)
        self.maxs = np.asarray(maxs, float).reshape(-1, 3)
        self.wall_a = np.asarray(wall_a, float).reshape(-1, 2)
        self.wall_b = np.asarray(wall_b, float).reshape(-1, 2)
        self.wall_height = float(wall_height)

    @classmethod
    def from_objects(cls, objects: Iterable[Any], wall_segments: Sequence[Tuple[Any, Any]] = (),
                     include_wall_boxes: bool = True) -> "OccluderSet":
        ids: List[str] = []
        labels: List[str] = []
        mins: List[np.ndarray] = []
        maxs: List[np.ndarray] = []
        for obj in objects:
            oid, label, bmin, bmax = box_of(obj)
            if is_non_occluding(label):
                continue
            if is_wall_label(label):
                if not include_wall_boxes:
                    continue
                dims = bmax - bmin
                if min(float(dims[0]), float(dims[1])) > WALL_BOX_MAX_THICKNESS_M:
                    continue
            if not (np.all(np.isfinite(bmin)) and np.all(np.isfinite(bmax))):
                continue
            ids.append(oid)
            labels.append(label)
            mins.append(bmin)
            maxs.append(bmax)
        wa = [np.asarray(a, float)[:2] for a, _ in wall_segments]
        wb = [np.asarray(b, float)[:2] for _, b in wall_segments]
        return cls(ids, labels, np.asarray(mins, float).reshape(-1, 3), np.asarray(maxs, float).reshape(-1, 3),
                   np.asarray(wa, float).reshape(-1, 2), np.asarray(wb, float).reshape(-1, 2))

    @classmethod
    def from_scene(cls, scene: Any, include_wall_boxes: bool = True) -> "OccluderSet":
        detector = getattr(scene, "detector", None)
        walls = list(getattr(detector, "wall_segments", None) or []) if detector is not None else []
        return cls.from_objects(scene.objects, walls, include_wall_boxes=include_wall_boxes)

    def __len__(self) -> int:
        return int(self.mins.shape[0])


_OCC_CACHE: Dict[int, Tuple[Any, OccluderSet]] = {}


def occluders_for(scene: Any) -> OccluderSet:
    """Default OccluderSet for ``scene`` (cached on object identity)."""
    key = id(scene)
    hit = _OCC_CACHE.get(key)
    if hit is not None:
        ref, occ = hit
        if ref() is scene:
            return occ
    occ = OccluderSet.from_scene(scene)
    try:
        ref = weakref.ref(scene)
    except TypeError:  # pragma: no cover - objects without weakref support
        ref = (lambda s=scene: s)
    _OCC_CACHE[key] = (ref, occ)
    if len(_OCC_CACHE) > 64:
        for k in list(_OCC_CACHE)[:-32]:
            _OCC_CACHE.pop(k, None)
    return occ


def _resolve_occluders(scene: Any, occluders: Any) -> OccluderSet:
    if occluders is None:
        return occluders_for(scene)
    if isinstance(occluders, OccluderSet):
        return occluders
    # A custom object list: those boxes plus the scene's structure walls.
    detector = getattr(scene, "detector", None)
    walls = list(getattr(detector, "wall_segments", None) or []) if detector is not None else []
    return OccluderSet.from_objects(occluders, walls)


# --------------------------------------------------------------------------
# Vectorised ray tests
# --------------------------------------------------------------------------
_T_END_EPS = 1e-6
_CHUNK_ELEMS = 1_500_000


def rays_occluded(occ: OccluderSet, origins: np.ndarray, points: np.ndarray,
                  exclude_ids: Iterable[str] = ()) -> np.ndarray:
    """(P, K) bool: segment origins[p] -> points[(p,)k] is blocked.

    ``points`` is (K, 3) (shared by all origins) or (P, K, 3).
    """
    O = np.asarray(origins, float).reshape(-1, 3)
    P = O.shape[0]
    S = np.asarray(points, float)
    if S.ndim == 2:
        S = np.broadcast_to(S[None, :, :], (P, S.shape[0], 3))
    K = S.shape[1]
    out = np.zeros((P, K), dtype=bool)
    if P == 0 or K == 0:
        return out

    excl = set(str(e) for e in exclude_ids)
    keep = np.array([i not in excl for i in occ.ids], dtype=bool) if len(occ) else np.zeros(0, bool)
    mins_all = occ.mins[keep]
    maxs_all = occ.maxs[keep]

    M = max(1, mins_all.shape[0] + occ.wall_a.shape[0])
    step = max(1, int(_CHUNK_ELEMS // max(1, K * M * 3)))
    for s0 in range(0, P, step):
        o = O[s0:s0 + step]
        s = S[s0:s0 + step]
        # Cull boxes that cannot touch any segment of this chunk.
        lo = np.minimum(o.min(axis=0), s.reshape(-1, 3).min(axis=0))
        hi = np.maximum(o.max(axis=0), s.reshape(-1, 3).max(axis=0))
        if mins_all.shape[0]:
            sel = np.all(maxs_all >= lo - 1e-9, axis=1) & np.all(mins_all <= hi + 1e-9, axis=1)
            mins = mins_all[sel]
            maxs = maxs_all[sel]
        else:
            mins = maxs = np.zeros((0, 3))
        blocked = np.zeros(s.shape[:2], dtype=bool)
        d = s - o[:, None, :]  # (p, K, 3)
        if mins.shape[0]:
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / d[:, :, None, :]  # (p, K, 1, 3)
                t1 = (mins[None, None, :, :] - o[:, None, None, :]) * inv
                t2 = (maxs[None, None, :, :] - o[:, None, None, :]) * inv
                tnear = np.fmax.reduce(np.fmin(t1, t2), axis=3)
                tfar = np.fmin.reduce(np.fmax(t1, t2), axis=3)
            hit = (tfar >= tnear) & (tfar > 1e-9) & (tnear < 1.0 - _T_END_EPS)
            blocked |= hit.any(axis=2)
        if occ.wall_a.shape[0]:
            blocked |= _walls_block(o, s, occ.wall_a, occ.wall_b, occ.wall_height)
        out[s0:s0 + step] = blocked
    return out


def _walls_block(o: np.ndarray, s: np.ndarray, wa: np.ndarray, wb: np.ndarray, height: float) -> np.ndarray:
    d = (s - o[:, None, :])[:, :, None, :]  # (p, K, 1, 3)
    e = (wb - wa)[None, None, :, :]  # (1, 1, W, 2)
    diff = (wa[None, :, :] - o[:, None, :2])[:, None, :, :]  # (p, 1, W, 2)
    denom = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (diff[..., 0] * e[..., 1] - diff[..., 1] * e[..., 0]) / denom
        u = (diff[..., 0] * d[..., 1] - diff[..., 1] * d[..., 0]) / denom
    z = o[:, None, None, 2] + t * d[..., 2]
    hit = ((np.abs(denom) > 1e-12) & (t > 1e-9) & (t < 1.0 - _T_END_EPS)
           & (u >= 0.0) & (u <= 1.0) & (z >= 0.0) & (z <= height))
    return hit.any(axis=2)


def frustum_mask(R: np.ndarray, C: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N, K) bool: points project inside the canonical H1 image.

    R: (N, 3, 3) c2w rotations (columns right, down, forward); C: (N, 3);
    points: (K, 3) shared or (N, K, 3).
    """
    R = np.asarray(R, float).reshape(-1, 3, 3)
    C = np.asarray(C, float).reshape(-1, 3)
    pts = np.asarray(points, float)
    if pts.ndim == 2:
        rel = pts[None, :, :] - C[:, None, :]
    else:
        rel = pts - C[:, None, :]
    pc = np.matmul(rel, R)  # pc[n,k,j] = sum_i rel[n,k,i] R[n,i,j] = (R^T rel)_j
    z = pc[..., 2]
    front = z > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        u = FX * pc[..., 0] / z + CX
        v = FY * pc[..., 1] / z + CY
    return front & (u >= 0.0) & (u <= IMG_W - 1) & (v >= 0.0) & (v <= IMG_H - 1)


# --------------------------------------------------------------------------
# Public single-pose API
# --------------------------------------------------------------------------
def visible_fraction(scene: Any, c2w: Any, obj: Any, occluders: Any = None,
                     n: int = DEFAULT_SAMPLES_PER_AXIS) -> float:
    """Unoccluded fraction of the target's in-frustum AABB samples.

    ``occluders``: None (scene default), an OccluderSet, or a list of Obj /
    target entries (plus the scene's structure walls).  The target itself is
    always excluded by id.
    """
    occ = _resolve_occluders(scene, occluders)
    oid, _, bmin, bmax = box_of(obj)
    c2w = np.asarray(c2w, float)
    samples = target_samples(bmin, bmax, n)
    infr = frustum_mask(c2w[None, :3, :3], c2w[None, :3, 3], samples)[0]
    count = int(infr.sum())
    if count == 0:
        return 0.0
    blocked = rays_occluded(occ, c2w[None, :3, 3], samples[infr], exclude_ids=(oid,))[0]
    return float((~blocked).sum()) / float(count)


def _area(box: Optional[Sequence[float]]) -> float:
    if not box:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def identifiable(scene: Any, c2w: Any, obj: Any, *, min_inside: float = 0.5,
                 min_area_ratio: float = 0.002, min_visible: float = DEFAULT_MIN_VISIBLE_FRACTION) -> Dict[str, Any]:
    """Geometric identifiability of one object at one pose.

    Uses the canonical H1 projection (``build_canonical_camera`` with
    contract.NATIVE_K / RENDER_SIZE and ``_project_object``) so the numbers
    match the runtime metric, plus the occlusion model of this module.
    """
    c2w = np.asarray(c2w, float)
    camera = build_canonical_camera(
        K_native=NATIVE_K,
        native_size=NATIVE_SIZE,
        render_size=RENDER_SIZE,
        transform="resize",
        c2w=c2w,
        camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1,
    )
    proj = _project_object(entry_of(obj), np.asarray(camera.c2w), np.asarray(camera.K_effective),
                           *camera.render_resolution)
    raw = _area(proj.get("bbox_raw"))
    inside = _area(proj.get("bbox")) / raw if raw > 1e-8 else 0.0
    area_ratio = float(proj.get("area_ratio", 0.0) or 0.0)
    in_front = bool(proj.get("center_in_front"))
    projected_visible = bool(proj.get("visible"))
    vis = visible_fraction(scene, c2w, obj)
    ok = bool(in_front and projected_visible and inside >= min_inside and area_ratio >= min_area_ratio
              and vis >= min_visible)
    return {
        "inside_frame_fraction": float(inside),
        "area_ratio": area_ratio,
        "visible_fraction": float(vis),
        "identifiable": ok,
        "center_in_front": in_front,
        "projected_visible": projected_visible,
        "center_uv": proj.get("center_uv"),
        "thresholds": {"min_inside": float(min_inside), "min_area_ratio": float(min_area_ratio),
                       "min_visible": float(min_visible)},
        "occlusion_model_version": OCCLUSION_MODEL_VERSION,
    }
