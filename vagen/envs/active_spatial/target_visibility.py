"""Target visibility shared by the RL env and the InteriorGS v10 generator ("one face clearly visible").

User decision of 2026-10-13: in a success frame every target that has to be recognised shows at least one
face of its 3D box clearly.  A face is *clearly visible* when

* it faces the camera,
* enough of it is visible: the share of its sample points that lie in front of the camera, project
  inside the image and are not hidden is at least ``MIN_FACE_VISIBLE_FRACTION``, and
* the visible part is large enough on screen: at least ``MIN_FACE_AREA_FRACTION`` of the image.

The same definition decides "at least one target can be recognised" in a start frame.  It replaces the
generator's former occlusion gate (half of the whole box unhidden), which judged built-in objects such as a
fridge that only shows its front as hidden.

Occluders: every labels.json box except labels that never block a view (rugs, ceiling lights,
curtains, ...) and thick "wall" label boxes (diagonal walls; the structure walls cover them), plus the
structure walls extruded to ``WALL_HEIGHT_M``.  Boxes of built-in objects overlap the wall or a neighbouring
cabinet; an overlap must not hide them, so an occluder only counts if the sight line enters it before the
line comes within ``OVERLAP_MARGIN_M`` of the target's box.

The thresholds are initial values (cluster_runs/REPORT_*_20261013.md explains them); the final values follow
human labels of examples near them.  ``score_with_visibility`` is what the env's success check calls;
the generator calls the same functions (vectorised over many poses) so both always agree.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

VISIBILITY_RULE_VERSION = "face_visible_v1"
FACE_SAMPLES_PER_AXIS = 5          # 5 x 5 sample points per face
MIN_FACE_VISIBLE_FRACTION = 0.5    # initial value, to be calibrated on human labels
MIN_FACE_AREA_FRACTION = 0.003     # initial value (~197 px of a 256 x 256 image), to be calibrated
OVERLAP_MARGIN_M = 0.10            # box overlaps with walls / neighbours within this distance never hide
MIN_DEPTH_M = 0.05                 # sample points closer than this to the camera plane count as not visible
WALL_HEIGHT_M = 2.8
WALL_BOX_MAX_THICKNESS_M = 0.35
RENDER_SIZE = (256, 256)

# Targets that must be clearly visible in a success frame, by index in target_object.objects
# (occlusion_alignment: only the occluder; the hidden target is required to be hidden by its own judge).
REQUIRED_VISIBLE_SLOTS: Dict[str, Tuple[int, ...]] = {
    "projective_relations": (0, 1),
    "fov_inclusion": (0, 1),
    "equidistance": (0, 1),
    "size_distance_invariance": (0, 1),
    "absolute_positioning": (0,),
    "screen_occupancy": (0,),
    "occlusion_alignment": (1,),
    "centering": (0, 1, 2),
}
_SLOT_LETTERS = "abcd"

# Exact labels that never occlude: floor coverings, structural shells handled elsewhere (walls) or never
# between a 1.5 m camera and a target (ceiling, overhead lighting), see-through or soft items.
NON_OCCLUDING_LABELS = frozenset({
    "rug", "carpet", "mat", "doormat", "floor mat", "bath mat",
    "floor", "ceiling", "room", "niche", "floor mold", "skirting", "baseboard",
    "light", "lights", "downlight", "downlights", "spotlight", "strip light",
    "ceiling lamp", "ceiling light", "pendant", "pendant lamp", "pendant light", "chandelier",
    "curtain", "curtains", "window", "window curtain",
})
# Head nouns (last word of the label) that never occlude, e.g. "ceiling light", "living room rug",
# "sheer curtain".  "floor lamp" -> "lamp" still occludes.
NON_OCCLUDING_HEAD_NOUNS = frozenset({
    "rug", "carpet", "mat", "doormat", "floor", "ceiling",
    "light", "lights", "downlight", "downlights", "spotlight", "chandelier",
    "curtain", "curtains", "window",
})
WALL_LABEL_HEAD_NOUNS = frozenset({"wall"})

_T_EPS = 1e-9
_CHUNK_ELEMS = 1_500_000


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


# =============================================================================
# Occluders
# =============================================================================
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
    def from_boxes(cls, boxes: Iterable[Tuple[str, str, Any, Any]], wall_segments: Sequence[Tuple[Any, Any]] = (),
                   include_wall_boxes: bool = True) -> "OccluderSet":
        """``boxes``: (id, label, bmin, bmax) per object."""
        ids: List[str] = []
        labels: List[str] = []
        mins: List[np.ndarray] = []
        maxs: List[np.ndarray] = []
        for oid, label, bmin, bmax in boxes:
            bmin, bmax = np.asarray(bmin, float), np.asarray(bmax, float)
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
            ids.append(str(oid))
            labels.append(str(label))
            mins.append(bmin)
            maxs.append(bmax)
        wa = [np.asarray(a, float)[:2] for a, _ in wall_segments]
        wb = [np.asarray(b, float)[:2] for _, b in wall_segments]
        return cls(ids, labels, np.asarray(mins, float).reshape(-1, 3), np.asarray(maxs, float).reshape(-1, 3),
                   np.asarray(wa, float).reshape(-1, 2), np.asarray(wb, float).reshape(-1, 2))

    @classmethod
    def from_detector(cls, detector: Any) -> "OccluderSet":
        """The scene's occluders from a loaded CollisionDetector (labels.json boxes + structure walls).

        Both the env and the generator build their occluders this way, from the same loader."""
        return cls.from_boxes(getattr(detector, "label_boxes", None) or [],
                              list(getattr(detector, "wall_segments", None) or []))

    def __len__(self) -> int:
        return int(self.mins.shape[0])


def _slab(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Entry / exit parameters of segments o + t d against boxes (broadcast over the leading axes)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
        tnear = np.fmax.reduce(np.fmin(t1, t2), axis=-1)
        tfar = np.fmin.reduce(np.fmax(t1, t2), axis=-1)
    return tnear, tfar


def blocked_mask(occ: OccluderSet, origins: np.ndarray, points: np.ndarray, target_min: Any, target_max: Any,
                 exclude_ids: Iterable[str] = ()) -> np.ndarray:
    """(P, S) bool: the sight line origins[p] -> points[s] (points on the target's box) is hidden.

    An occluder box or wall counts only where the line meets it before entering the target box
    inflated by OVERLAP_MARGIN_M.  A camera inside an occluder box sees nothing (conservative)."""
    O = np.asarray(origins, float).reshape(-1, 3)
    S = np.asarray(points, float).reshape(-1, 3)
    P, K = O.shape[0], S.shape[0]
    out = np.zeros((P, K), dtype=bool)
    if P == 0 or K == 0:
        return out
    tlo = np.asarray(target_min, float) - OVERLAP_MARGIN_M
    thi = np.asarray(target_max, float) + OVERLAP_MARGIN_M
    excl = {str(e) for e in exclude_ids}
    keep = np.array([i not in excl for i in occ.ids], dtype=bool) if len(occ) else np.zeros(0, bool)
    mins_all, maxs_all = occ.mins[keep], occ.maxs[keep]
    M = max(1, mins_all.shape[0] + occ.wall_a.shape[0])
    step = max(1, int(_CHUNK_ELEMS // max(1, K * M * 3)))
    for s0 in range(0, P, step):
        o = O[s0:s0 + step]  # (p, 3)
        d = S[None, :, :] - o[:, None, :]  # (p, K, 3)
        tn_t, _ = _slab(o[:, None, :], d, tlo, thi)
        limit = np.clip(tn_t, 0.0, 1.0)  # (p, K): where the line reaches the (inflated) target
        blocked = np.zeros(d.shape[:2], dtype=bool)
        if mins_all.shape[0]:
            lo = np.minimum(o.min(axis=0), S.min(axis=0))
            hi = np.maximum(o.max(axis=0), S.max(axis=0))
            sel = np.all(maxs_all >= lo - 1e-9, axis=1) & np.all(mins_all <= hi + 1e-9, axis=1)
            mins, maxs = mins_all[sel], maxs_all[sel]
            if mins.shape[0]:
                tn, tf = _slab(o[:, None, None, :], d[:, :, None, :], mins[None, None], maxs[None, None])
                hit = (tf >= tn) & (tf > _T_EPS) & (tn < limit[:, :, None] - _T_EPS)
                blocked |= hit.any(axis=2)
        if occ.wall_a.shape[0]:
            wa, wb = occ.wall_a, occ.wall_b
            dd = d[:, :, None, :]
            e = (wb - wa)[None, None, :, :]
            diff = (wa[None, :, :] - o[:, None, :2])[:, None, :, :]
            denom = dd[..., 0] * e[..., 1] - dd[..., 1] * e[..., 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (diff[..., 0] * e[..., 1] - diff[..., 1] * e[..., 0]) / denom
                u = (diff[..., 0] * dd[..., 1] - diff[..., 1] * dd[..., 0]) / denom
            z = o[:, None, None, 2] + t * dd[..., 2]
            hit = ((np.abs(denom) > 1e-12) & (t > _T_EPS) & (t < limit[:, :, None] - _T_EPS)
                   & (u >= 0.0) & (u <= 1.0) & (z >= 0.0) & (z <= occ.wall_height))
            blocked |= hit.any(axis=2)
        out[s0:s0 + step] = blocked
    return out


# =============================================================================
# Faces of a target box
# =============================================================================
class FaceGeometry:
    """Sample points (n x n per face, inset at (i + 0.5) / n), normals, centres and areas of an AABB."""

    def __init__(self, bmin: Any, bmax: Any, n: int = FACE_SAMPLES_PER_AXIS) -> None:
        lo, hi = np.minimum(bmin, bmax).astype(float), np.maximum(bmin, bmax).astype(float)
        self.lo, self.hi, self.n = lo, hi, int(n)
        f = (np.arange(self.n, dtype=float) + 0.5) / self.n
        pts, normals, centers, areas = [], [], [], []
        ext = hi - lo
        for axis in range(3):
            a, b = [k for k in range(3) if k != axis]
            ga, gb = np.meshgrid(lo[a] + f * ext[a], lo[b] + f * ext[b], indexing="ij")
            for side, value in ((-1.0, lo[axis]), (1.0, hi[axis])):
                p = np.zeros((self.n * self.n, 3))
                p[:, axis] = value
                p[:, a] = ga.ravel()
                p[:, b] = gb.ravel()
                pts.append(p)
                nrm = np.zeros(3)
                nrm[axis] = side
                normals.append(nrm)
                c = (lo + hi) / 2.0
                c[axis] = value
                centers.append(c)
                areas.append(float(ext[a] * ext[b]))
        self.points = np.concatenate(pts, axis=0)            # (6 n^2, 3), face-major
        self.normals = np.asarray(normals)                     # (6, 3): -x, +x, -y, +y, -z, +z
        self.centers = np.asarray(centers)                     # (6, 3)
        self.areas = np.asarray(areas)                         # (6,) m^2
        self.face_index = np.repeat(np.arange(6), self.n * self.n)


def face_visibility(R: np.ndarray, C: np.ndarray, geom: FaceGeometry, blocked: np.ndarray, K: Any,
                    width: int, height: int) -> Dict[str, np.ndarray]:
    """Per pose and face: visible fraction, visible on-screen area (fraction of the image), clearly visible.

    R: (N, 3, 3) camera-to-world rotations (columns right, down, forward); C: (N, 3) camera centres;
    blocked: (N, 6 n^2) from ``blocked_mask`` for each pose's centre."""
    R = np.asarray(R, float).reshape(-1, 3, 3)
    C = np.asarray(C, float).reshape(-1, 3)
    K = np.asarray(K, float)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    n2 = geom.n * geom.n
    # Elementwise arithmetic only (no matmul / einsum), so one pose and a batch give identical bits.
    rel = geom.points[None, :, :] - C[:, None, :]          # (N, S, 3)
    rx, ry, rz = rel[..., 0], rel[..., 1], rel[..., 2]
    xc = rx * R[:, None, 0, 0] + ry * R[:, None, 1, 0] + rz * R[:, None, 2, 0]   # camera coordinates
    yc = rx * R[:, None, 0, 1] + ry * R[:, None, 1, 1] + rz * R[:, None, 2, 1]
    z = rx * R[:, None, 0, 2] + ry * R[:, None, 1, 2] + rz * R[:, None, 2, 2]
    front = z > MIN_DEPTH_M
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * xc / z + cx
        v = fy * yc / z + cy
    inside = front & (u >= 0.0) & (u <= width) & (v >= 0.0) & (v <= height)
    tc = C[:, None, :] - geom.centers[None, :, :]            # (N, 6, 3)
    nm = geom.normals
    facing = (tc[..., 0] * nm[None, :, 0] + tc[..., 1] * nm[None, :, 1] + tc[..., 2] * nm[None, :, 2]) > 0.0
    vis = inside & ~np.asarray(blocked, bool) & facing[:, geom.face_index]
    frac = np.ascontiguousarray(vis.reshape(-1, 6, n2)).sum(axis=2) / float(n2)
    # on-screen area of each sample's patch: fx fy dA |n . (C - p)| / z^3 (pinhole projection)
    ns = geom.normals[geom.face_index]                       # (S, 3)
    cos_term = np.abs(rx * ns[None, :, 0] + ry * ns[None, :, 1] + rz * ns[None, :, 2])
    cell = (geom.areas / n2)[geom.face_index]                # (S,)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.where(front, fx * fy * cell[None, :] * cos_term / np.maximum(z, MIN_DEPTH_M) ** 3, 0.0)
    area = np.ascontiguousarray(np.where(vis, px, 0.0).reshape(-1, 6, n2)).sum(axis=2) / float(width * height)
    ok = facing & (frac >= MIN_FACE_VISIBLE_FRACTION) & (area >= MIN_FACE_AREA_FRACTION)
    return {"visible_fraction": frac, "area_fraction": area, "facing": facing, "face_ok": ok,
            "visible": ok.any(axis=1)}


def best_face(stats: Dict[str, np.ndarray], i: int = 0) -> Dict[str, Any]:
    """Summary of pose ``i``: the face that comes closest to passing, and whether the target is visible."""
    frac, area = stats["visible_fraction"][i], stats["area_fraction"][i]
    quality = np.minimum(np.clip(frac / MIN_FACE_VISIBLE_FRACTION, 0, 1), np.clip(area / MIN_FACE_AREA_FRACTION, 0, 1))
    quality = np.where(stats["facing"][i], quality, 0.0)
    f = int(np.argmax(np.where(stats["face_ok"][i], 2.0 + area, quality)))
    return {"visible": bool(stats["visible"][i]), "face": f, "visible_fraction": float(frac[f]),
            "area_fraction": float(area[f]), "progress": 1.0 if stats["visible"][i] else float(quality.max())}


# =============================================================================
# One pose (the env) and the success rule
# =============================================================================
def item_camera(item: Dict[str, Any]) -> Tuple[np.ndarray, int, int]:
    """K_effective and image size of the canonical H1 camera the success metrics use."""
    from .canonical_camera import CANONICAL_CAMERA_H1_RESIZE_V1, build_canonical_camera

    camera = build_canonical_camera(K_native=item["init_camera"]["intrinsics"], render_size=RENDER_SIZE,
                                    transform="resize", c2w=np.eye(4), item=item,
                                    camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1)
    width, height = (int(v) for v in camera.render_resolution)
    return np.asarray(camera.K_effective, float), width, height


def required_targets(item: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    """(slot, object entry) of the targets that must be clearly visible in a success frame."""
    from .visual_bbox_metrics import _extract_objects, _match_object

    region = item.get("target_region") or {}
    params = dict(region.get("params") or {}) if isinstance(region, dict) else {}
    objects = _extract_objects({"_target_object": item.get("target_object")})
    out = []
    for slot in REQUIRED_VISIBLE_SLOTS.get(str(item.get("task_type") or ""), (0,)):
        letter = _SLOT_LETTERS[slot]
        obj = _match_object(objects, object_id=params.get(f"object_{letter}_id"),
                            center=params.get(f"object_{letter}_center"), fallback_index=slot)
        if obj is not None and "bbox_min" in obj and "bbox_max" in obj:
            out.append((slot, obj))
    return out


def object_visibility(c2w: Any, obj: Dict[str, Any], occ: OccluderSet, K: Any, width: int,
                      height: int) -> Dict[str, Any]:
    """Is ``obj`` (a target_object entry) clearly visible from ``c2w``?"""
    c2w = np.asarray(c2w, float)
    geom = FaceGeometry(obj["bbox_min"], obj["bbox_max"])
    blocked = blocked_mask(occ, c2w[None, :3, 3], geom.points, geom.lo, geom.hi,
                           exclude_ids=(str(obj.get("id", obj.get("ins_id", ""))),))
    stats = face_visibility(c2w[None, :3, :3], c2w[None, :3, 3], geom, blocked, K, width, height)
    return best_face(stats)


def visibility_gates(item: Dict[str, Any], c2w: Any, occ: OccluderSet) -> Dict[str, Any]:
    K, width, height = item_camera(item)
    required = required_targets(item)
    targets = {}
    for slot, obj in required:
        targets[_SLOT_LETTERS[slot]] = dict(object_visibility(c2w, obj, occ, K, width, height),
                                            id=str(obj.get("id", obj.get("ins_id", ""))))
    found = len(required) == len(REQUIRED_VISIBLE_SLOTS.get(str(item.get("task_type") or ""), (0,)))
    ok = found and all(t["visible"] for t in targets.values())
    progress = float(np.prod([t["progress"] for t in targets.values()])) if found else 0.0
    return {"rule": VISIBILITY_RULE_VERSION, "visible": bool(ok), "found": bool(found), "progress": progress,
            "targets": targets}


def apply_visibility(metric: Dict[str, Any], vis: Dict[str, Any]) -> Dict[str, Any]:
    """The canonical metric with the visibility gates added: success needs every required target clearly
    visible; the shaping score keeps its meaning (>= 0.5 exactly on success)."""
    from .canonical_task_metrics_v2 import FAILING_PROGRESS_CAP

    out = dict(metric)
    gates = dict(out.get("gates") or {})
    for letter, t in vis["targets"].items():
        gates[f"target_{letter}_clearly_visible"] = bool(t["visible"])
    gates["required_targets_found"] = bool(vis["found"])
    success = bool(metric.get("success")) and bool(vis["visible"])
    base = metric.get("continuous_gate_progress")
    if base is None:  # metrics without a gate-aligned progress (v1 FOV): use the shaping score
        base = float(metric.get("shaping_score", 0.0) or 0.0)
    progress = float(min(1.0, max(0.0, float(base)))) * float(vis["progress"])
    if not success:
        progress = min(progress, FAILING_PROGRESS_CAP)
    out.update(success=success, gates=gates, continuous_gate_progress=progress,
               shaping_score=0.5 * progress + 0.5 * float(success), visibility=vis)
    return out


def score_with_visibility(item: Dict[str, Any], c2w: Any, occ: Optional[OccluderSet]) -> Dict[str, Any]:
    """The env's success metric: the canonical metric plus, when ``occ`` is given, the visibility rule."""
    from .canonical_task_metrics import score_canonical_task

    metric = score_canonical_task(item, np.asarray(c2w, float))
    if occ is None:
        return metric
    return apply_visibility(metric, visibility_gates(item, c2w, occ))
