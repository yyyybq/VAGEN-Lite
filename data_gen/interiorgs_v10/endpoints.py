"""Endpoint (terminal success state) sampling for Projective and FOV tasks.

Workstream E.  An *endpoint* is a legal pitch-0 pose at z = 1.5 m where the
frozen canonical metric (``sampling.score``, canonical_spatial_task_h1_v1)
reports success AND the generator-side occlusion gate holds (every target has
``occlusion.visible_fraction >= min_visible``, default 0.5).

Public API
----------
    sample_endpoints(scene, item, *, mode="v10", max_endpoints=8, seed=0, **budget)
        -> (List[dict], Funnel)
    success_fn(item, scene=None, *, min_visible=0.5) -> callable(c2w) -> bool
    batch_evaluate(scene, item, xs, ys, yaws, ...) -> dict of arrays
    compare_modes(scene_dirs, ...) -> report dict      (also: python -m ...endpoints)

Modes
-----
* ``v9_like`` - replica of the v9 sampler used only for comparison: positions
  from the R1 half-plane sweep around the pair midpoint (13 distance scales x
  11 along-offsets at a 2.0 m base distance on the relation side, then the
  legal half-plane floor grid ordered by distance to the main anchor), the
  first 64 *legal* positions in the midpoint's room are kept, yaw = face the
  pair midpoint + {0, +-5, +-10} deg.
* ``v10`` - free-floor grid over the room(s) that contain the targets
  (fallback: neighbouring rooms and a finer yaw step), exact-safe analytic
  prefilters (relation half-plane from the centre bearings; a vertical
  inside-frame upper bound that defines the near edge of the distance band;
  the yaw interval in which both targets fit the 90 deg HFOV) and yaws sampled
  every ``yaw_step_deg`` inside that interval.  All poses are scored with a
  vectorised replica of the canonical projection; kept endpoints are chosen
  by farthest-point sampling in (x, y, yaw) and re-verified one by one with
  ``sampling.score`` + exact ``visible_fraction`` + ``scene.is_pose_legal``.
* ``audit`` - diagnostics only: every grid point of the target rooms (legal or
  not) x yaws every 10 deg, no prefilters, so the gate histogram is unbiased.

Instrumentation (Funnel stages, stable reason strings)
-----------------------------------------------------
    endpoint_position         legal / illegal_<reason> / prefilter_relation_side /
                              prefilter_distance_band / prefilter_yaw_interval /
                              v9_other_room / budget_dropped
    endpoint_pose_first_fail  first failing gate of each scored pose, or "success"
    endpoint_pose_fail_any    every failing gate of each scored pose
    endpoint_pose_fail_set    "+"-joined failing gates
    endpoint_pose_sole_fail   poses failing exactly one gate
    endpoint_verify           ok / mismatch_canonical / mismatch_occlusion / mismatch_legal
    endpoint_item             has_endpoint / no_legal_position / no_candidate_pose /
                              no_endpoint_blocker_<gate>
    endpoint_count            poses_scored / positions_scored / endpoints_kept (+ pass2_fallback)
Gate order: illegal_<reason>, two_objects, in_front, visible, min_area,
inside_frame | full_bbox_in_frame, relation, margin | center_margin, occlusion.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vagen.envs.active_spatial.canonical_task_metrics import (
    FOV_CENTER_MARGIN_FRACTION,
    FOV_MIN_AREA_RATIO,
    FOV_MIN_INSIDE_FRACTION,
    PROJECTIVE_MIN_INSIDE_FRACTION,
    PROJECTIVE_MIN_MARGIN_PX,
)
from vagen.envs.active_spatial.visual_bbox_metrics import _as_array, _match_object

from . import sampling
from .contract import (CAMERA_HEIGHT_M, Funnel, Obj, horizon_locked, motion_model, pose_from_xy_yaw, stable_hash,
                       state_key, yaw_of)
from .occlusion import (
    CX,
    CY,
    DEFAULT_MIN_VISIBLE_FRACTION,
    FX,
    FY,
    IMG_H,
    IMG_W,
    OCCLUSION_MODEL_VERSION,
    frustum_mask,
    is_non_occluding,
    is_wall_label,
    occluders_for,
    rays_occluded,
    target_samples,
    visible_fraction,
)

ENDPOINT_SAMPLER_VERSION = "interiorgs_v10_endpoints_v1"
SUPPORTED_TASK_TYPES = ("projective_relations", "fov_inclusion")
HALF_HFOV_RAD = math.atan2(CX, FX)  # 45 deg for H1 at 256x256

DEFAULT_BUDGET: Dict[str, Any] = {
    "grid_spacing": 0.25,          # v10 / audit free-floor grid (m)
    "yaw_step_deg": 5.0,           # v10 yaw sampling inside the feasible interval
    "yaw_slack_deg": 4.0,          # widen the analytic yaw interval (safety)
    "refine_yaw_step_deg": 2.0,    # fallback pass yaw step
    "neighbour_rooms": "auto",     # True / False / "auto" (fallback pass only)
    "prune_distance": True,        # vertical inside-frame upper bound prefilter
    "min_visible": DEFAULT_MIN_VISIBLE_FRACTION,
    "occlusion_n": 3,
    "max_poses": 60000,
    "yaw_weight_m": 1.0,           # FPS metric: metres per radian of yaw
    "v9_max_positions": 64,
    "v9_yaw_offsets_deg": (0.0, -5.0, 5.0, -10.0, 10.0),
    "v9_base_distance": 2.0,
    "v9_grid_spacing": 0.2,
    "audit_yaw_step_deg": 10.0,
}

PROJECTIVE_GATES = ("two_objects", "in_front", "visible", "min_area", "inside_frame", "relation", "margin")
FOV_GATES = ("two_objects", "in_front", "visible", "min_area", "full_bbox_in_frame", "center_margin")


# ==========================================================================
# Item targets
# ==========================================================================
class _Target:
    def __init__(self, entry: Dict[str, Any]) -> None:
        self.entry = entry
        self.id = str(entry.get("id", entry.get("ins_id", entry.get("label", ""))))
        self.label = str(entry.get("label", ""))
        self.bmin = np.asarray(entry["bbox_min"], float)
        self.bmax = np.asarray(entry["bbox_max"], float)
        # _project_object uses entry["center"] when present.
        self.center = _as_array(entry.get("center"), (self.bmin + self.bmax) / 2.0)[:3]
        self.corners = np.array([[x, y, z] for x in (self.bmin[0], self.bmax[0])
                                 for y in (self.bmin[1], self.bmax[1])
                                 for z in (self.bmin[2], self.bmax[2])], dtype=float)
        self.footprint = np.array([[self.bmin[0], self.bmin[1]], [self.bmax[0], self.bmin[1]],
                                   [self.bmax[0], self.bmax[1]], [self.bmin[0], self.bmax[1]]], dtype=float)


def _item_targets(item: Dict[str, Any]) -> List[_Target]:
    task_type = item.get("task_type")
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"endpoint sampling supports {SUPPORTED_TASK_TYPES}, got {task_type!r}")
    objects = [o for o in (item.get("target_object") or {}).get("objects") or [] if isinstance(o, dict)]
    params = (item.get("target_region") or {}).get("params") or {}
    a = _match_object(objects, center=params.get("object_a_center"), fallback_index=0)
    b = _match_object(objects, center=params.get("object_b_center"), fallback_index=1)
    if a is None or b is None:
        raise ValueError("item needs two target objects")
    return [_Target(a), _Target(b)]


# ==========================================================================
# Vectorised canonical projection (replica of visual_bbox_metrics._project_object)
# ==========================================================================
def rotations(yaws: np.ndarray, pitch_rad: float = 0.0) -> np.ndarray:
    """(N, 3, 3) c2w rotations [right, down, forward] = camera_pose_from_forward."""
    yaws = np.asarray(yaws, float).reshape(-1)
    c, s = np.cos(yaws), np.sin(yaws)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    R = np.zeros((yaws.size, 3, 3))
    R[:, 0, 0], R[:, 1, 0] = s, -c                      # right
    R[:, 0, 1], R[:, 1, 1], R[:, 2, 1] = sp * c, sp * s, -cp  # down
    R[:, 0, 2], R[:, 1, 2], R[:, 2, 2] = cp * c, cp * s, sp   # forward
    return R


_NEAR = 1e-6
_OCCL_MEMO_MAX = 200000  # cached ray-blocked rows per (scene, target) before the memo is reset
# Horizon-locked experiments with pitch != 0: a camera looking down sees low, near objects at
# horizontal bearings beyond the 45 deg half-HFOV, so the pitch-0 yaw window is widened.
PITCHED_EXTRA_YAW_SLACK_DEG = 20.0


def _pitched() -> bool:
    return horizon_locked() and motion_model()[1] != 0.0


def _project_batch(t: _Target, R: np.ndarray, C: np.ndarray) -> Dict[str, np.ndarray]:
    W, H = IMG_W, IMG_H
    rel = t.corners[None, :, :] - C[:, None, :]
    pc = np.matmul(rel, R)  # (R^T (p - C)) per pose
    z = pc[..., 2]
    front = z > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        u = FX * pc[..., 0] / z + CX
        v = FY * pc[..., 1] / z + CY
    any_front = front.any(axis=1)
    x1 = np.where(front, u, np.inf).min(axis=1)
    x2 = np.where(front, u, -np.inf).max(axis=1)
    y1 = np.where(front, v, np.inf).min(axis=1)
    y2 = np.where(front, v, -np.inf).max(axis=1)
    cx1, cy1 = np.maximum(0.0, x1), np.maximum(0.0, y1)
    cx2, cy2 = np.minimum(float(W - 1), x2), np.minimum(float(H - 1), y2)
    cw = np.where(any_front, np.maximum(0.0, cx2 - cx1), 0.0)
    ch = np.where(any_front, np.maximum(0.0, cy2 - cy1), 0.0)
    area = cw * ch
    raw = np.where(any_front, np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = np.where(raw > 1e-8, area / np.where(raw > 1e-8, raw, 1.0), 0.0)
    cc = np.matmul((t.center[None, :] - C)[:, None, :], R)[:, 0, :]
    c_front = cc[:, 2] > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        cu = np.where(c_front, FX * cc[:, 0] / cc[:, 2] + CX, np.nan)
        cv = np.where(c_front, FY * cc[:, 1] / cc[:, 2] + CY, np.nan)
    c_in_frame = c_front & (cu >= 0.0) & (cu <= W - 1) & (cv >= 0.0) & (cv <= H - 1)
    visible = any_front & (area > 1.0) & (c_in_frame | ((cw > 2.0) & (ch > 2.0)))
    ratio = area / float(max(W * H, 1))
    # Poses whose quantities sit within _NEAR of a decision threshold may flip
    # between this replica and the exact scorer (w2c via inverse vs R^T).
    tol = _NEAR
    near = (np.abs(z - 1e-6) < tol).any(axis=1) | (np.abs(cc[:, 2] - 1e-6) < tol)
    near |= any_front & ((np.abs(area - 1.0) < tol) | (np.abs(cw - 2.0) < tol) | (np.abs(ch - 2.0) < tol)
                         | (np.abs(raw - 1e-8) < 1e-12) | (np.abs(ratio - FOV_MIN_AREA_RATIO) < 1e-12))
    near |= c_front & ((np.abs(cu) < tol) | (np.abs(cu - (W - 1)) < tol) | (np.abs(cv) < tol)
                       | (np.abs(cv - (H - 1)) < tol))
    return {"in_front": c_front, "cu": cu, "cv": cv, "visible": visible,
            "area_ratio": ratio, "inside": inside, "near": near}


def _canonical_batch(task_type: str, relation: str, pa: Dict[str, np.ndarray],
                     pb: Dict[str, np.ndarray]) -> Tuple[Tuple[str, ...], np.ndarray, Dict[str, np.ndarray]]:
    n = pa["inside"].shape[0]
    two = np.ones(n, dtype=bool)
    in_front = pa["in_front"] & pb["in_front"]
    visible = pa["visible"] & pb["visible"]
    min_area = (pa["area_ratio"] >= FOV_MIN_AREA_RATIO) & (pb["area_ratio"] >= FOV_MIN_AREA_RATIO)
    min_inside = np.minimum(pa["inside"], pb["inside"])
    min_ar = np.minimum(pa["area_ratio"], pb["area_ratio"])
    if task_type == "projective_relations":
        defined = visible & in_front & np.isfinite(pa["cu"]) & np.isfinite(pb["cu"])
        margin = np.where(defined, (pb["cu"] - pa["cu"]) if relation == "left" else (pa["cu"] - pb["cu"]), np.nan)
        relation_ok = defined & (margin > 0)
        margin_ok = defined & (margin >= PROJECTIVE_MIN_MARGIN_PX)
        inside_ok = (pa["inside"] >= PROJECTIVE_MIN_INSIDE_FRACTION) & (pb["inside"] >= PROJECTIVE_MIN_INSIDE_FRACTION)
        gates = np.stack([two, in_front, visible, min_area, inside_ok, relation_ok, margin_ok], axis=1)
        success = gates.all(axis=1)
        x = np.clip((np.nan_to_num(margin, nan=-1e9) - PROJECTIVE_MIN_MARGIN_PX) / 6.0, -60.0, 60.0)
        margin_comp = np.where(defined, 1.0 / (1.0 + np.exp(-x)), 0.0)
        prog = (in_front.astype(float) * visible.astype(float)
                * np.clip(min_ar / FOV_MIN_AREA_RATIO, 0.0, 1.0) * np.clip(min_inside, 0.0, 1.0) * margin_comp)
        shaping = 0.5 * prog + 0.5 * success.astype(float)
        near = pa["near"] | pb["near"]
        near |= (np.abs(pa["inside"] - PROJECTIVE_MIN_INSIDE_FRACTION) < _NEAR)
        near |= (np.abs(pb["inside"] - PROJECTIVE_MIN_INSIDE_FRACTION) < _NEAR)
        near |= defined & ((np.abs(margin - PROJECTIVE_MIN_MARGIN_PX) < _NEAR) | (np.abs(margin) < _NEAR))
        return PROJECTIVE_GATES, gates, {"margin": margin, "shaping": shaping, "min_inside": min_inside,
                                         "near": near}
    boundary = min(IMG_W, IMG_H) * FOV_CENTER_MARGIN_FRACTION
    both_c = pa["in_front"] & pb["in_front"]

    def _cm(p: Dict[str, np.ndarray]) -> np.ndarray:
        return np.minimum(np.minimum(p["cu"], IMG_W - p["cu"]), np.minimum(p["cv"], IMG_H - p["cv"]))

    cmargin = np.where(both_c, np.minimum(_cm(pa), _cm(pb)), np.nan)
    full_ok = (pa["inside"] >= FOV_MIN_INSIDE_FRACTION) & (pb["inside"] >= FOV_MIN_INSIDE_FRACTION)
    cm_ok = both_c & (np.nan_to_num(cmargin, nan=-1e9) >= boundary)
    gates = np.stack([two, in_front, visible, min_area, full_ok, cm_ok], axis=1)
    success = gates.all(axis=1)
    norm = np.where(both_c, np.clip(np.nan_to_num(cmargin, nan=0.0) / max(boundary, 1.0), 0.0, 1.0), 0.0)
    shaping = min_inside * norm * min_area.astype(float)
    near = pa["near"] | pb["near"]
    near |= (np.abs(pa["inside"] - FOV_MIN_INSIDE_FRACTION) < _NEAR) | (np.abs(pb["inside"] - FOV_MIN_INSIDE_FRACTION) < _NEAR)
    near |= both_c & (np.abs(np.nan_to_num(cmargin, nan=-1e9) - boundary) < _NEAR)
    return FOV_GATES, gates, {"margin": cmargin, "shaping": shaping, "min_inside": min_inside, "near": near}


def _blocked_rows(scene: Any, t: _Target, origins: np.ndarray, n: int, samples: np.ndarray) -> np.ndarray:
    """(P, K) ray-blocked flags for target ``t`` from exact 3D ``origins`` (cached per scene)."""
    tkey = (t.id, tuple(t.bmin.tolist()), tuple(t.bmax.tolist()), int(n))
    memo: Dict[Tuple[float, float, float], np.ndarray] = _cache(scene)["occl"].setdefault(tkey, {})
    pkeys = [(float(x), float(y), float(zz)) for x, y, zz in origins]
    missing = [i for i, k in enumerate(pkeys) if k not in memo]
    if missing:
        rows = rays_occluded(occluders_for(scene), origins[missing], samples, exclude_ids=(t.id,))
        for r, i in zip(rows, missing):
            memo[pkeys[i]] = r
    if not pkeys:
        return np.zeros((0, samples.shape[0]), dtype=bool)
    out = np.stack([memo[k] for k in pkeys], axis=0)
    # bound memory only after this call's rows are read (clearing first raised KeyError on 0463_840754)
    if len(memo) > _OCCL_MEMO_MAX:
        memo.clear()
    return out


def batch_evaluate(scene: Any, item: Dict[str, Any], xs: Sequence[float], ys: Sequence[float],
                   yaws: Sequence[float], *, pitch_deg: Optional[float] = None,
                   min_visible: float = DEFAULT_MIN_VISIBLE_FRACTION, occlusion_n: int = 3,
                   z: float = CAMERA_HEIGHT_M, legal_reasons: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Vectorised canonical gates + occlusion for poses (x, y, yaw[rad], pitch).

    ``pitch_deg`` defaults to the active motion model's pitch (0 under r1).
    Returns gate names, an (N, G) pass matrix (first column "legal" when
    ``legal_reasons`` is given, last column "occlusion"), canonical success,
    occlusion success, combined success, per-target inside / visible fractions,
    margin, shaping score and ``near`` (poses within 1e-6 of a canonical
    threshold).  Matches ``sampling.score`` up to float rounding; callers that
    keep a pose must still re-verify it exactly.
    """
    if pitch_deg is None:
        pitch_deg = motion_model()[1]
    xs = np.asarray(xs, float).reshape(-1)
    ys = np.asarray(ys, float).reshape(-1)
    yaws = np.asarray(yaws, float).reshape(-1)
    n = xs.size
    R = rotations(yaws, math.radians(pitch_deg))
    C = np.stack([xs, ys, np.full(n, float(z))], axis=1)
    return evaluate_poses(scene, item, R, C, min_visible=min_visible, occlusion_n=occlusion_n,
                          legal_reasons=legal_reasons)


def evaluate_c2ws(scene: Any, item: Dict[str, Any], c2ws: Any, **kw: Any) -> Dict[str, Any]:
    """``batch_evaluate`` for an (N, 4, 4) stack of c2w matrices."""
    c2ws = np.asarray(c2ws, float).reshape(-1, 4, 4)
    return evaluate_poses(scene, item, c2ws[:, :3, :3], c2ws[:, :3, 3], **kw)


def evaluate_poses(scene: Any, item: Dict[str, Any], R: np.ndarray, C: np.ndarray, *,
                   min_visible: float = DEFAULT_MIN_VISIBLE_FRACTION, occlusion_n: int = 3,
                   legal_reasons: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    targets = _item_targets(item)
    R = np.asarray(R, float).reshape(-1, 3, 3)
    C = np.asarray(C, float).reshape(-1, 3)
    n = C.shape[0]
    pa, pb = (_project_batch(t, R, C) for t in targets)
    relation = str(((item.get("target_region") or {}).get("params") or {}).get("relation", "left"))
    names, gates, extra = _canonical_batch(item["task_type"], relation, pa, pb)
    canonical_success = gates.all(axis=1)

    # Occlusion: rays are yaw/pitch independent -> computed once per (target,
    # position) and cached per scene; only the frustum test is per pose.
    vis = np.zeros((n, 2))
    if n:
        uniq, inv = np.unique(C, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        for j, t in enumerate(targets):
            samples = target_samples(t.bmin, t.bmax, occlusion_n)
            blocked = _blocked_rows(scene, t, uniq, occlusion_n, samples)
            infr = frustum_mask(R, C, samples)
            cnt = infr.sum(axis=1)
            free = (infr & ~blocked[inv]).sum(axis=1)
            vis[:, j] = np.where(cnt > 0, free / np.maximum(cnt, 1), 0.0)
    occl_ok = (vis[:, 0] >= min_visible) & (vis[:, 1] >= min_visible)

    all_names: List[str] = list(names) + ["occlusion"]
    cols = [gates, occl_ok[:, None]]
    legal = np.ones(n, dtype=bool)
    reasons = None
    if legal_reasons is not None:
        reasons = np.asarray(list(legal_reasons), dtype=object)
        legal = reasons == "ok"
        all_names = ["legal"] + all_names
        cols = [legal[:, None]] + cols
    G = np.concatenate(cols, axis=1)
    return {
        "gate_names": tuple(all_names),
        "gates": G,
        "legal_reasons": reasons,
        "canonical_success": canonical_success,
        "occlusion_ok": occl_ok,
        "success": canonical_success & occl_ok & legal,
        "visible_fraction": vis,
        "inside": np.stack([pa["inside"], pb["inside"]], axis=1),
        "area_ratio": np.stack([pa["area_ratio"], pb["area_ratio"]], axis=1),
        "margin": extra["margin"],
        "shaping_score": extra["shaping"],
        "near": extra["near"],
        "target_ids": [t.id for t in targets],
    }


# ==========================================================================
# Positions and legality
# ==========================================================================
_SCENE_CACHE: Dict[int, Tuple[Any, Dict[str, Any]]] = {}
_SCENE_REGISTRY: Dict[str, Any] = {}


def _cache(scene: Any) -> Dict[str, Any]:
    key = id(scene)
    hit = _SCENE_CACHE.get(key)
    if hit is not None and hit[0]() is scene:
        return hit[1]
    data: Dict[str, Any] = {"legal": {}, "grids": {}, "occl": {}}
    _SCENE_CACHE[key] = (weakref.ref(scene), data)
    if len(_SCENE_CACHE) > 64:
        for k in list(_SCENE_CACHE)[:-32]:
            _SCENE_CACHE.pop(k, None)
    return data


def register_scene(scene: Any) -> None:
    """Make ``success_fn(item)`` resolve this scene by item["scene_id"]."""
    _SCENE_REGISTRY[str(scene.scene_id)] = scene
    if len(_SCENE_REGISTRY) > 32:
        for k in list(_SCENE_REGISTRY)[:-16]:
            _SCENE_REGISTRY.pop(k, None)


def _legality(scene: Any, xy: np.ndarray) -> List[str]:
    """Per-point legality reason ("ok" when legal), cached per scene."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    memo: Dict[Tuple[float, float], str] = _cache(scene)["legal"]
    keys = [(round(float(x), 5), round(float(y), 5)) for x, y in xy]
    missing = [i for i, k in enumerate(keys) if k not in memo]
    if missing:
        done = False
        mask_fn = getattr(scene, "legal_mask", None)
        if callable(mask_fn):  # optional fast path from workstream G
            try:
                res = mask_fn(xy[missing])
                reasons_in = None
                if isinstance(res, tuple) and len(res) == 2:
                    res, reasons_in = res
                mask = np.asarray(res, dtype=bool).reshape(-1)
                if mask.shape[0] == len(missing):
                    n_illegal = int((~mask).sum())
                    for j, i in enumerate(missing):
                        if mask[j]:
                            memo[keys[i]] = "ok"
                        elif reasons_in is not None:
                            r = str(reasons_in[j])
                            memo[keys[i]] = r if r != "ok" else "legal_mask"
                        elif n_illegal <= 5000:  # reason strings for the funnel (reference check)
                            ok, r = scene.is_pose_legal(pose_from_xy_yaw(float(xy[i, 0]), float(xy[i, 1]), 0.0))
                            memo[keys[i]] = "legal_mask" if ok else str(r)
                        else:
                            memo[keys[i]] = "legal_mask"
                    done = True
            except Exception:  # fall back to the reference per-pose check
                done = False
        if not done:
            for i in missing:
                ok, reason = scene.is_pose_legal(pose_from_xy_yaw(float(xy[i, 0]), float(xy[i, 1]), 0.0))
                memo[keys[i]] = "ok" if ok else str(reason)
    return [memo[k] for k in keys]


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def _room_grid(scene: Any, room_indices: Sequence[int], spacing: float,
               offset: Tuple[float, float]) -> Tuple[np.ndarray, List[str]]:
    key = (tuple(sorted(int(r) for r in room_indices)), round(float(spacing), 6),
           round(float(offset[0]), 6), round(float(offset[1]), 6))
    grids = _cache(scene)["grids"]
    if key in grids:
        return grids[key]
    pts: List[np.ndarray] = []
    wanted = set(key[0])
    for room in scene.rooms:
        if room.index not in wanted:
            continue
        lo = room.polygon.min(axis=0)
        hi = room.polygon.max(axis=0)
        xs = np.arange(lo[0] + spacing / 2 + offset[0], hi[0], spacing)
        ys = np.arange(lo[1] + spacing / 2 + offset[1], hi[1], spacing)
        if xs.size == 0 or ys.size == 0:
            continue
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        cand = np.stack([gx.ravel(), gy.ravel()], axis=1)
        cand = cand[_points_in_polygon(cand, room.polygon)]
        # attribute each point to the first room that contains it (scene.room_of order)
        for other in scene.rooms:
            if other.index == room.index:
                break
            cand = cand[~_points_in_polygon(cand, other.polygon)]
        pts.append(cand)
    xy = np.concatenate(pts, axis=0) if pts else np.zeros((0, 2))
    reasons = _legality(scene, xy)
    grids[key] = (xy, reasons)
    return xy, reasons


def _poly_distance(p: np.ndarray, poly: np.ndarray) -> float:
    best = math.inf
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        den = float(ab @ ab)
        t = 0.0 if den < 1e-12 else float(np.clip((p - a) @ ab / den, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _target_rooms(scene: Any, targets: Sequence[_Target]) -> List[int]:
    by_id = {str(o.id): o for o in scene.objects}
    rooms: List[int] = []
    for t in targets:
        r = None
        obj = by_id.get(t.id)
        if obj is not None and obj.room_index is not None:
            r = int(obj.room_index)
        if r is None:
            r = scene.room_of(float(t.center[0]), float(t.center[1]))
        if r is None and scene.rooms:
            d = [(_poly_distance(t.center[:2], room.polygon), room.index) for room in scene.rooms]
            dist, idx = min(d)
            r = idx if dist <= 1.0 else None
        if r is not None and r not in rooms:
            rooms.append(int(r))
    if not rooms:
        rooms = [room.index for room in scene.rooms]
    return rooms


def _neighbour_rooms(scene: Any, rooms: Sequence[int], tol: float = 0.3) -> List[int]:
    base = [r for r in scene.rooms if r.index in set(rooms)]
    out = []
    for other in scene.rooms:
        if other.index in set(rooms):
            continue
        close = any(
            min(min(_poly_distance(v, b.polygon) for v in other.polygon),
                min(_poly_distance(v, other.polygon) for v in b.polygon)) <= tol
            for b in base
        )
        if close:
            out.append(other.index)
    return out


# ==========================================================================
# Candidate generation per mode
# ==========================================================================
def _wrap(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _v10_poses(scene: Any, item: Dict[str, Any], targets: Sequence[_Target], rooms: Sequence[int],
               cfg: Dict[str, Any], yaw_step_deg: float, offset: Tuple[float, float],
               funnel: Funnel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy, reasons = _room_grid(scene, rooms, float(cfg["grid_spacing"]), offset)
    reasons_arr = np.asarray(reasons, dtype=object)
    legal = reasons_arr == "ok"
    for r, c in zip(*np.unique(reasons_arr[~legal], return_counts=True)) if (~legal).any() else []:
        funnel.add("endpoint_position", f"illegal_{r}", int(c))
    funnel.add("endpoint_position", "legal", int(legal.sum()))
    P = xy[legal]
    if P.shape[0] == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    task = item["task_type"]
    a, b = targets
    vA = a.center[None, :2] - P
    vB = b.center[None, :2] - P
    keep = np.ones(P.shape[0], dtype=bool)
    # Both prefilters below are exact only at pitch 0 (image x order == bearing order, horizon row CY).
    pitched = _pitched()

    if task == "projective_relations" and not pitched:
        relation = str(item["target_region"]["params"].get("relation", "left"))
        cross = vA[:, 0] * vB[:, 1] - vA[:, 1] * vB[:, 0]
        side_ok = cross < 0 if relation == "left" else cross > 0
        funnel.add("endpoint_position", "prefilter_relation_side", int((keep & ~side_ok).sum()))
        keep &= side_ok

    if cfg.get("prune_distance", True) and not pitched:
        q = PROJECTIVE_MIN_INSIDE_FRACTION if task == "projective_relations" else FOV_MIN_INSIDE_FRACTION
        band_ok = np.ones(P.shape[0], dtype=bool)
        zc = CAMERA_HEIGHT_M
        for t in targets:
            zmin, zmax = float(t.bmin[2]), float(t.bmax[2])
            if not (zmax < zc and zmin < zmax):
                continue
            D = np.sqrt(((t.footprint[None, :, :] - P[:, None, :]) ** 2).sum(axis=2)).max(axis=1)
            D = np.maximum(D, 1e-6)
            t0 = CY + FY * (zc - zmax) / D
            b0 = CY + FY * (zc - zmin) / D
            with np.errstate(divide="ignore", invalid="ignore"):
                f_up = np.where(b0 > IMG_H - 1, (IMG_H - 1 - t0) / (b0 - t0), 1.0)
            band_ok &= f_up >= q - 1e-9
        funnel.add("endpoint_position", "prefilter_distance_band", int((keep & ~band_ok).sum()))
        keep &= band_ok

    # Feasible yaw interval (relative to the bearing of the pair midpoint).
    mid = (a.center[:2] + b.center[:2]) / 2.0
    ref = np.arctan2(mid[1] - P[:, 1], mid[0] - P[:, 0])
    h = HALF_HFOV_RAD + math.radians(float(cfg["yaw_slack_deg"]))
    if pitched:
        h += math.radians(PITCHED_EXTRA_YAW_SLACK_DEG)
    if task == "projective_relations":
        phA = _wrap(np.arctan2(vA[:, 1], vA[:, 0]) - ref)
        phB = _wrap(np.arctan2(vB[:, 1], vB[:, 0]) - ref)
        lo = np.maximum(phA, phB) - h
        hi = np.minimum(phA, phB) + h
    else:
        lo = np.full(P.shape[0], -np.inf)
        hi = np.full(P.shape[0], np.inf)
        for t in targets:
            v = t.footprint[None, :, :] - P[:, None, :]
            ph = _wrap(np.arctan2(v[..., 1], v[..., 0]) - ref[:, None])
            inside_fp = ((P[:, 0] >= t.bmin[0]) & (P[:, 0] <= t.bmax[0])
                         & (P[:, 1] >= t.bmin[1]) & (P[:, 1] <= t.bmax[1]))
            spread = ph.max(axis=1) - ph.min(axis=1)
            bad = inside_fp | (spread >= np.pi)
            lo = np.maximum(lo, np.where(bad, np.inf, ph.max(axis=1) - h))
            hi = np.minimum(hi, np.where(bad, -np.inf, ph.min(axis=1) + h))
    yaw_ok = hi >= lo
    funnel.add("endpoint_position", "prefilter_yaw_interval", int((keep & ~yaw_ok).sum()))
    keep &= yaw_ok
    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    step = math.radians(float(yaw_step_deg))
    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []
    YAW: List[np.ndarray] = []
    for i in idx:
        a0, a1 = ref[i] + lo[i], ref[i] + hi[i]
        k0, k1 = math.ceil(a0 / step - 1e-9), math.floor(a1 / step + 1e-9)
        ys_ = [k * step for k in range(k0, k1 + 1)]
        midyaw = 0.5 * (a0 + a1)
        if all(abs(midyaw - y) > math.radians(1.0) for y in ys_):
            ys_.append(midyaw)
        arr = np.asarray(ys_, float)
        X.append(np.full(arr.size, P[i, 0]))
        Y.append(np.full(arr.size, P[i, 1]))
        YAW.append(arr)
    xs, ys, yaws = np.concatenate(X), np.concatenate(Y), _wrap(np.concatenate(YAW))
    max_poses = int(cfg.get("max_poses") or 0)
    if max_poses and xs.size > max_poses:
        sel = np.linspace(0, xs.size - 1, max_poses).astype(int)
        funnel.add("endpoint_position", "budget_dropped_poses", int(xs.size - sel.size))
        xs, ys, yaws = xs[sel], ys[sel], yaws[sel]
    return xs, ys, yaws


def _v9_poses(scene: Any, item: Dict[str, Any], targets: Sequence[_Target], cfg: Dict[str, Any],
              funnel: Funnel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    a, b = targets
    mid = (a.center[:2] + b.center[:2]) / 2.0
    ab = b.center[:2] - a.center[:2]
    norm = float(np.linalg.norm(ab))
    direction = ab / norm if norm > 1e-9 else np.array([1.0, 0.0])
    n_left = np.array([direction[1], -direction[0]])
    if item["task_type"] == "projective_relations":
        relation = str(item["target_region"]["params"].get("relation", "left"))
        normals = [n_left if relation == "left" else -n_left]
    else:
        normals = [n_left, -n_left]
    dist = max(float(cfg["v9_base_distance"]), 0.75)
    ordered: List[np.ndarray] = []
    for scale in (1.0, 0.85, 1.15, 0.70, 1.30, 1.55, 1.85, 2.20, 0.55, 0.40, 0.30, 0.20, 0.15):
        for delta in (0.0, -0.25, 0.25, -0.75, 0.75, -1.5, 1.5, -2.5, 2.5, -4.0, 4.0):
            for nrm in normals:
                ordered.append(mid + delta * direction + dist * scale * nrm)
    mid_room = scene.room_of(float(mid[0]), float(mid[1]))
    if mid_room is None:
        mid_room = _target_rooms(scene, targets)[0] if scene.rooms else None
    # Then the legal half-plane floor grid ordered by distance to the main anchor.
    if mid_room is not None:
        gxy, greasons = _room_grid(scene, [mid_room], float(cfg["v9_grid_spacing"]), (0.0, 0.0))
        gxy = gxy[np.asarray(greasons, dtype=object) == "ok"]
        if gxy.shape[0]:
            if len(normals) == 1:
                gxy = gxy[(gxy - mid) @ normals[0] > 0.0]
            anchor = mid + dist * normals[0]
            order = np.argsort(np.linalg.norm(gxy - anchor, axis=1), kind="stable")
            ordered.extend(list(gxy[order]))
    seen = set()
    chosen: List[np.ndarray] = []
    cap = int(cfg["v9_max_positions"])
    tried = 0
    reasons_count: Dict[str, int] = {}
    for p in ordered:
        k = (round(float(p[0]), 6), round(float(p[1]), 6))
        if k in seen:
            continue
        seen.add(k)
        tried += 1
        r = _legality(scene, p[None, :])[0]
        if r == "ok" and mid_room is not None and scene.room_of(float(p[0]), float(p[1])) != mid_room:
            r = "v9_other_room"
        if r != "ok":
            reasons_count[r] = reasons_count.get(r, 0) + 1
            continue
        chosen.append(p)
        if len(chosen) >= cap:
            break
    for r, c in reasons_count.items():
        funnel.add("endpoint_position", r if r.startswith("v9_") else f"illegal_{r}", c)
    funnel.add("endpoint_position", "legal", len(chosen))
    offsets = [math.radians(float(o)) for o in cfg["v9_yaw_offsets_deg"]]
    X, Y, YAW = [], [], []
    for p in chosen:
        base = math.atan2(mid[1] - p[1], mid[0] - p[0])
        for o in offsets:
            X.append(p[0])
            Y.append(p[1])
            YAW.append(base + o)
    return np.asarray(X, float), np.asarray(Y, float), _wrap(np.asarray(YAW, float)), ["ok"] * len(X)


def _audit_poses(scene: Any, targets: Sequence[_Target], rooms: Sequence[int], cfg: Dict[str, Any],
                 offset: Tuple[float, float], funnel: Funnel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    xy, reasons = _room_grid(scene, rooms, float(cfg["grid_spacing"]), offset)
    reasons_arr = np.asarray(reasons, dtype=object)
    funnel.add("endpoint_position", "legal", int((reasons_arr == "ok").sum()))
    for r, c in zip(*np.unique(reasons_arr[reasons_arr != "ok"], return_counts=True)) if (reasons_arr != "ok").any() else []:
        funnel.add("endpoint_position", f"illegal_{r}", int(c))
    yaws = np.radians(np.arange(0.0, 360.0, float(cfg["audit_yaw_step_deg"])))
    k = yaws.size
    xs = np.repeat(xy[:, 0], k)
    ys = np.repeat(xy[:, 1], k)
    yw = _wrap(np.tile(yaws, xy.shape[0]))
    return xs, ys, yw, list(np.repeat(reasons_arr, k))


# ==========================================================================
# Instrumentation, selection, verification
# ==========================================================================
def _gate_label(name: str, reason: Optional[str]) -> str:
    if name == "legal":
        return f"illegal_{reason}"
    return name


def _record_gates(ev: Dict[str, Any], funnel: Funnel) -> Optional[str]:
    """Aggregate per-pose gate failures into the funnel; return the item blocker."""
    G = ev["gates"]
    names = ev["gate_names"]
    reasons = ev["legal_reasons"]
    n = G.shape[0]
    if n == 0:
        return None
    funnel.add("endpoint_count", "poses_scored", n)
    fails = ~G
    nfail = fails.sum(axis=1)
    funnel.add("endpoint_pose_first_fail", "success", int((nfail == 0).sum()))
    first = np.argmax(fails, axis=1)
    # Per-pose labels only for failing poses; vectorise by unique bit patterns.
    weights = (1 << np.arange(G.shape[1])).astype(np.int64)
    if reasons is not None and "legal" in names:
        illegal_reasons = np.where(G[:, 0], "", reasons).astype(str)
    else:
        illegal_reasons = np.full(n, "", dtype=object).astype(str)
    patt = fails.astype(np.int64) @ weights
    combo = np.char.add(patt.astype(str), np.char.add("|", illegal_reasons))
    uniq, inv, counts = np.unique(combo, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    for u_idx, (key, cnt) in enumerate(zip(uniq, counts)):
        bits_s, reason = str(key).split("|", 1)
        bits = int(bits_s)
        if bits == 0:
            continue
        failing = [_gate_label(names[g], reason) for g in range(len(names)) if bits >> g & 1]
        funnel.add("endpoint_pose_first_fail", failing[0], int(cnt))
        for f in failing:
            funnel.add("endpoint_pose_fail_any", f, int(cnt))
        funnel.add("endpoint_pose_fail_set", "+".join(failing), int(cnt))
        if len(failing) == 1:
            funnel.add("endpoint_pose_sole_fail", failing[0], int(cnt))
    if (nfail == 0).any():
        return None
    # Item blocker: most common first-failing gate among the poses with the
    # fewest failing gates (the closest misses).
    best = nfail == nfail.min()
    labels = [_gate_label(names[g], (reasons[i] if reasons is not None else None))
              for i, g in zip(np.nonzero(best)[0], first[best])]
    vals, cnts = np.unique(np.asarray(labels, dtype=object).astype(str), return_counts=True)
    return str(vals[int(np.argmax(cnts))])


def _fps_order(xs: np.ndarray, ys: np.ndarray, yaws: np.ndarray, quality: np.ndarray, w: float) -> Iterable[int]:
    """Farthest-point order in (x, y, w * yaw); starts at the best-quality pose."""
    n = xs.size
    if n == 0:
        return
    alive = np.ones(n, dtype=bool)
    dmin = np.full(n, np.inf)
    cur = int(np.lexsort((np.arange(n), -quality))[0])
    while True:
        yield cur
        alive[cur] = False
        if not alive.any():
            return
        d2 = (xs - xs[cur]) ** 2 + (ys - ys[cur]) ** 2 + (w * _wrap(yaws - yaws[cur])) ** 2
        dmin = np.minimum(dmin, d2)
        cand = np.where(alive, dmin, -np.inf)
        best = cand.max()
        ties = np.nonzero(alive & (cand >= best - 1e-12))[0]
        cur = int(ties[np.argmax(quality[ties])])


def _verify_and_pack(scene: Any, item: Dict[str, Any], targets: Sequence[_Target], x: float, y: float,
                     yaw: float, ev: Dict[str, Any], i: int, cfg: Dict[str, Any], mode: str,
                     funnel: Funnel) -> Optional[Dict[str, Any]]:
    c2w = pose_from_xy_yaw(float(x), float(y), float(yaw))
    ok, reason = scene.is_pose_legal(c2w)
    if not ok:
        funnel.add("endpoint_verify", "mismatch_legal")
        return None
    exact = sampling.score(item, c2w)
    if not exact.get("success"):
        funnel.add("endpoint_verify", "mismatch_canonical")
        return None
    vfs = {t.id: visible_fraction(scene, c2w, t.entry, n=int(cfg["occlusion_n"])) for t in targets}
    if min(vfs.values()) < float(cfg["min_visible"]):
        funnel.add("endpoint_verify", "mismatch_occlusion")
        return None
    funnel.add("endpoint_verify", "ok")
    margin = exact.get("relation_margin_px") if item["task_type"] == "projective_relations" else exact.get("center_margin_px")
    return {
        "c2w": c2w.tolist(),
        "position": [float(x), float(y), float(CAMERA_HEIGHT_M)],
        "yaw_deg": float(math.degrees(yaw_of(c2w))),
        "pitch_deg": motion_model()[1],
        "state_key": list(state_key(c2w)),
        "room_index": scene.room_of(float(x), float(y)),
        "score": {
            "success": bool(exact["success"]),
            "shaping_score": float(exact.get("shaping_score", 0.0)),
            "gates": dict(exact.get("gates", {})),
            "margin": (float(margin) if margin is not None else None),
            "inside_frame_fraction_min": float(exact.get("inside_frame_fraction_min", 0.0)),
            "inside_fractions": [float(v) for v in ev["inside"][i]],
            "metric_version": exact.get("metric_version"),
        },
        "visible_fractions": {k: float(v) for k, v in vfs.items()},
        "visible_fraction_min": float(min(vfs.values())),
        "sampler": {"mode": mode, "version": ENDPOINT_SAMPLER_VERSION,
                    "occlusion_model_version": OCCLUSION_MODEL_VERSION,
                    "min_visible": float(cfg["min_visible"])},
    }


def _select(scene: Any, item: Dict[str, Any], targets: Sequence[_Target], xs: np.ndarray, ys: np.ndarray,
            yaws: np.ndarray, ev: Dict[str, Any], max_endpoints: int, cfg: Dict[str, Any], mode: str,
            funnel: Funnel) -> List[Dict[str, Any]]:
    idx = np.nonzero(ev["success"])[0]
    if idx.size == 0 or max_endpoints <= 0:
        return []
    quality = ev["shaping_score"][idx] + 0.1 * ev["visible_fraction"][idx].min(axis=1)
    out: List[Dict[str, Any]] = []
    for j in _fps_order(xs[idx], ys[idx], yaws[idx], quality, float(cfg["yaw_weight_m"])):
        i = int(idx[j])
        packed = _verify_and_pack(scene, item, targets, xs[i], ys[i], yaws[i], ev, i, cfg, mode, funnel)
        if packed is not None:
            out.append(packed)
            if len(out) >= max_endpoints:
                break
    return out


# ==========================================================================
# Public entry points
# ==========================================================================
def sample_endpoints(scene: Any, item: Dict[str, Any], *, mode: str = "v10", max_endpoints: int = 8,
                     seed: int = 0, **budget: Any) -> Tuple[List[Dict[str, Any]], Funnel]:
    """Sample up to ``max_endpoints`` diverse, verified success endpoints.

    ``seed`` shifts the free-floor grid origin (seed 0 = canonical grid).
    Budget keys: see DEFAULT_BUDGET.  Every returned endpoint re-scores as
    success with ``sampling.score``, passes the occlusion gate and
    ``scene.is_pose_legal``.
    """
    unknown = set(budget) - set(DEFAULT_BUDGET)
    if unknown:
        raise TypeError(f"unknown budget keys: {sorted(unknown)}")
    if mode not in ("v10", "v9_like", "audit"):
        raise ValueError(f"unknown mode {mode!r}")
    cfg = dict(DEFAULT_BUDGET)
    cfg.update(budget)
    funnel = Funnel()
    register_scene(scene)
    targets = _item_targets(item)
    offset = (0.0, 0.0)
    if seed:
        rng = np.random.default_rng(int(seed))
        offset = tuple(float(v) for v in rng.uniform(0.0, float(cfg["grid_spacing"]), size=2))  # type: ignore[assignment]
    common = dict(min_visible=float(cfg["min_visible"]), occlusion_n=int(cfg["occlusion_n"]))

    endpoints: List[Dict[str, Any]] = []
    blocker: Optional[str] = None
    n_scored = 0
    n_legal_positions = 0
    if mode == "v9_like":
        xs, ys, yaws, reasons = _v9_poses(scene, item, targets, cfg, funnel)
        n_legal_positions = funnel.counts.get("endpoint_position", {}).get("legal", 0)
        ev = batch_evaluate(scene, item, xs, ys, yaws, **common)
        blocker = _record_gates(ev, funnel)
        n_scored += xs.size
        endpoints = _select(scene, item, targets, xs, ys, yaws, ev, max_endpoints, cfg, mode, funnel)
    elif mode == "audit":
        rooms = _target_rooms(scene, targets)
        xs, ys, yaws, reasons = _audit_poses(scene, targets, rooms, cfg, offset, funnel)
        n_legal_positions = funnel.counts.get("endpoint_position", {}).get("legal", 0)
        ev = batch_evaluate(scene, item, xs, ys, yaws, legal_reasons=reasons, **common)
        blocker = _record_gates(ev, funnel)
        n_scored += xs.size
        endpoints = _select(scene, item, targets, xs, ys, yaws, ev, max_endpoints, cfg, mode, funnel)
    else:
        rooms = _target_rooms(scene, targets)
        neigh = cfg.get("neighbour_rooms", "auto")
        passes: List[Tuple[List[int], float, str]] = []
        if neigh is True:
            passes.append((sorted(set(rooms) | set(_neighbour_rooms(scene, rooms))), float(cfg["yaw_step_deg"]), "pass1"))
        else:
            passes.append((list(rooms), float(cfg["yaw_step_deg"]), "pass1"))
            if neigh == "auto":
                passes.append((sorted(set(rooms) | set(_neighbour_rooms(scene, rooms))),
                               float(cfg["refine_yaw_step_deg"]), "pass2_fallback"))
        for prooms, step, tag in passes:
            if tag != "pass1":
                funnel.add("endpoint_count", tag)
            xs, ys, yaws = _v10_poses(scene, item, targets, prooms, cfg, step, offset, funnel)
            n_legal_positions = funnel.counts.get("endpoint_position", {}).get("legal", 0)
            ev = batch_evaluate(scene, item, xs, ys, yaws, **common)
            blocker = _record_gates(ev, funnel)
            n_scored += xs.size
            endpoints = _select(scene, item, targets, xs, ys, yaws, ev, max_endpoints, cfg, mode, funnel)
            if endpoints:
                break

    funnel.add("endpoint_count", "endpoints_kept", len(endpoints))
    if endpoints:
        funnel.add("endpoint_item", "has_endpoint")
    elif n_legal_positions == 0:
        funnel.add("endpoint_item", "no_legal_position")
    elif n_scored == 0:
        funnel.add("endpoint_item", "no_candidate_pose")
    else:
        funnel.add("endpoint_item", f"no_endpoint_blocker_{blocker or 'verify'}")
    return endpoints, funnel


_SUCCESS_FN_CACHE: Dict[Tuple[int, str, float, int], Callable[[Any], bool]] = {}


def success_fn(item: Dict[str, Any], scene: Any = None, *, min_visible: float = DEFAULT_MIN_VISIBLE_FRACTION,
               occlusion_n: int = 3) -> Callable[[Any], bool]:
    """callable(c2w) -> bool: canonical success AND occlusion gate.

    Results are cached by ``contract.state_key`` for canonical (pitch 0,
    z = 1.5 m) poses.  The callable also carries ``.exact(c2w)`` (uncached)
    and ``.batch(c2ws[N,4,4]) -> bool[N]`` (vectorised; near-threshold poses
    re-scored exactly), which ``difficulty.as_oracle`` picks up.  ``scene`` defaults to the scene last registered for
    ``item["scene_id"]`` (``sample_endpoints`` registers it; or call
    ``register_scene``).  Legality is NOT part of this predicate.
    """
    if scene is None:
        scene = _SCENE_REGISTRY.get(str(item.get("scene_id")))
        if scene is None:
            raise ValueError("success_fn needs a scene: pass scene= or call register_scene(scene) first")
    targets = _item_targets(item)
    ikey = stable_hash({"scene_id": item.get("scene_id"), "task_type": item.get("task_type"),
                        "target_region": item.get("target_region"),
                        "targets": [[t.id, t.bmin, t.bmax] for t in targets]})
    ckey = (id(scene), ikey, float(min_visible), int(occlusion_n))
    fn = _SUCCESS_FN_CACHE.get(ckey)
    if fn is not None and getattr(fn, "_scene_ref")() is scene:
        return fn
    occ = occluders_for(scene)
    cache: Dict[Tuple[int, int, int], bool] = {}

    def _exact(c2w: Any) -> bool:
        c2w = np.asarray(c2w, float)
        ok = bool(sampling.score(item, c2w).get("success"))
        if ok:
            ok = all(visible_fraction(scene, c2w, t.entry, occluders=occ, n=occlusion_n) >= min_visible
                     for t in targets)
        return ok

    def _fn(c2w: Any) -> bool:
        c2w = np.asarray(c2w, float)
        canonical = abs(float(c2w[2, 2])) < 1e-9 and abs(float(c2w[2, 3]) - CAMERA_HEIGHT_M) < 1e-6
        key = state_key(c2w) if canonical else None
        if key is not None and key in cache:
            return cache[key]
        ok = _exact(c2w)
        if key is not None:
            cache[key] = ok
        return ok

    def _batch(c2ws: Any) -> np.ndarray:
        """Vectorised; poses within 1e-6 of a canonical threshold are re-scored exactly."""
        c2ws = np.asarray(c2ws, float).reshape(-1, 4, 4)
        ev = evaluate_c2ws(scene, item, c2ws, min_visible=min_visible, occlusion_n=occlusion_n)
        ok = np.asarray(ev["success"], dtype=bool).copy()
        for i in np.flatnonzero(ev["near"]):
            ok[i] = _exact(c2ws[i])
        return ok

    _fn.exact = _exact  # type: ignore[attr-defined]
    _fn.batch = _batch  # type: ignore[attr-defined]
    _fn.cache = cache  # type: ignore[attr-defined]
    _fn._scene_ref = weakref.ref(scene)  # type: ignore[attr-defined]
    _SUCCESS_FN_CACHE[ckey] = _fn
    if len(_SUCCESS_FN_CACHE) > 4096:
        for k in list(_SUCCESS_FN_CACHE)[:-2048]:
            _SUCCESS_FN_CACHE.pop(k, None)
    return _fn


# ==========================================================================
# Comparison harness
# ==========================================================================
EXCLUDED_TARGET_LABELS = frozenset({"door", "window", "wall", "floor", "ceiling", "room"})


def candidate_objects(scene: Any) -> List[Obj]:
    """Non-structural objects with a room (comparison harness only)."""
    out = []
    for o in scene.objects:
        if o.room_index is None:
            continue
        if is_non_occluding(o.label) or is_wall_label(o.label) or o.label in EXCLUDED_TARGET_LABELS:
            continue
        out.append(o)
    return out


def comparison_items(scene: Any) -> List[Dict[str, Any]]:
    objs = candidate_objects(scene)
    items = []
    for a, b in itertools.combinations(objs, 2):
        if a.room_index != b.room_index:
            continue
        items.append(sampling.projective_item(scene.scene_id, a, b, "left"))
        items.append(sampling.projective_item(scene.scene_id, a, b, "right"))
        items.append(sampling.fov_item(scene.scene_id, a, b))
    return items


def _pitch_hypothetical(scene: Any, item: Dict[str, Any], pitches: Sequence[float], cfg: Dict[str, Any]) -> Dict[float, bool]:
    """HYPOTHETICAL: does any legal-position pose at fixed pitch succeed?

    Poses with pitch != 0 are NOT reachable under the v1 motion model
    (translations follow the camera axis); this only sizes the potential gain.
    """
    targets = _item_targets(item)
    rooms = _target_rooms(scene, targets)
    xy, reasons = _room_grid(scene, rooms, float(cfg["grid_spacing"]), (0.0, 0.0))
    P = xy[np.asarray(reasons, dtype=object) == "ok"]
    yaws = np.radians(np.arange(0.0, 360.0, float(cfg["audit_yaw_step_deg"])))
    xs = np.repeat(P[:, 0], yaws.size)
    ys = np.repeat(P[:, 1], yaws.size)
    yw = _wrap(np.tile(yaws, P.shape[0]))
    out = {}
    for p in pitches:
        ev = batch_evaluate(scene, item, xs, ys, yw, pitch_deg=float(p), min_visible=float(cfg["min_visible"]))
        out[float(p)] = bool(ev["success"].any())
    return out


def compare_modes(scene_dirs: Sequence[Any], *, modes: Sequence[str] = ("v9_like", "v10", "audit"),
                  pitches: Sequence[float] = (0.0, -10.0, -20.0, -30.0), max_endpoints: int = 8,
                  frozen_only: bool = False, **budget: Any) -> Dict[str, Any]:
    from .scene import load_scene

    cfg = dict(DEFAULT_BUDGET)
    cfg.update(budget)
    report: Dict[str, Any] = {"scenes": [], "modes": {}, "pitch_hypothetical": {}, "per_item": []}
    per_mode: Dict[str, Dict[str, Any]] = {
        m: {"funnel": Funnel(), "items": 0, "with_endpoint": 0, "endpoints": 0, "times": [],
            "by_task": {}} for m in modes}
    pitch_hits = {float(p): 0 for p in pitches}
    pitch_union = 0
    pitch_items = 0
    for scene_dir in scene_dirs:
        scene = load_scene(Path(scene_dir))
        status = scene.convention.get("status")
        items = comparison_items(scene)
        report["scenes"].append({"scene_id": scene.scene_id, "convention_status": status,
                                 "candidate_objects": len(candidate_objects(scene)), "items": len(items)})
        if frozen_only and status != "frozen":
            continue
        for item in items:
            ids = [o["id"] for o in item["target_object"]["objects"]]
            rel = item["target_region"]["params"].get("relation", "fov")
            row: Dict[str, Any] = {"scene_id": scene.scene_id, "pair": ids, "task_type": item["task_type"],
                                   "relation": rel}
            for m in modes:
                t0 = time.perf_counter()
                eps, fun = sample_endpoints(scene, item, mode=m, max_endpoints=max_endpoints, **budget)
                dt = time.perf_counter() - t0
                pm = per_mode[m]
                pm["funnel"].merge(fun)
                pm["items"] += 1
                pm["with_endpoint"] += int(bool(eps))
                pm["endpoints"] += len(eps)
                pm["times"].append(dt)
                bt = pm["by_task"].setdefault(item["task_type"], [0, 0])
                bt[0] += 1
                bt[1] += int(bool(eps))
                row[m] = {"n_endpoints": len(eps), "seconds": round(dt, 4),
                          "status": next(iter(fun.counts.get("endpoint_item", {"?": 1})))}
            if pitches:
                hits = _pitch_hypothetical(scene, item, pitches, cfg)
                pitch_items += 1
                for p, h in hits.items():
                    pitch_hits[p] += int(h)
                pitch_union += int(any(hits.values()))
                row["pitch_hypothetical"] = {str(k): v for k, v in hits.items()}
            report["per_item"].append(row)
    for m, pm in per_mode.items():
        times = np.asarray(pm["times"], float) if pm["times"] else np.zeros(1)
        f = pm["funnel"].to_dict()
        report["modes"][m] = {
            "items": pm["items"],
            "items_with_endpoint": pm["with_endpoint"],
            "endpoints_total": pm["endpoints"],
            "by_task_type": {k: {"items": v[0], "with_endpoint": v[1]} for k, v in pm["by_task"].items()},
            "seconds_per_item_mean": float(times.mean()),
            "seconds_per_item_p95": float(np.percentile(times, 95)),
            "seconds_total": float(times.sum()),
            "first_fail_hist": f.get("endpoint_pose_first_fail", {}),
            "fail_any_hist": f.get("endpoint_pose_fail_any", {}),
            "sole_fail_hist": f.get("endpoint_pose_sole_fail", {}),
            "item_status": f.get("endpoint_item", {}),
            "position_funnel": f.get("endpoint_position", {}),
            "verify": f.get("endpoint_verify", {}),
            "counts": f.get("endpoint_count", {}),
        }
    report["pitch_hypothetical"] = {
        "label": "HYPOTHETICAL - pitch != 0 is not reachable under the v1 motion model",
        "items": pitch_items,
        "items_with_endpoint_by_pitch_deg": {str(k): v for k, v in pitch_hits.items()},
        "items_with_endpoint_any_listed_pitch": pitch_union,
        "positions": "legal pitch-0 grid of the target rooms, yaw every audit_yaw_step_deg (10 deg)",
    }
    return report


def _main(argv: Optional[Sequence[str]] = None) -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Compare v9_like / v10 endpoint samplers")
    ap.add_argument("scene_dirs", nargs="*")
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic test scenes")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-pitch", action="store_true")
    ap.add_argument("--per-item", action="store_true")
    args = ap.parse_args(argv)
    dirs: List[Any] = list(args.scene_dirs)
    tmp = None
    if args.synthetic:
        import tempfile

        from .testing.synthetic_scene import write_synthetic_scenes

        tmp = tempfile.TemporaryDirectory()
        dirs += list(write_synthetic_scenes(Path(tmp.name)).values())
    rep = compare_modes(dirs, pitches=() if args.no_pitch else (0.0, -10.0, -20.0, -30.0))
    if not args.per_item:
        rep.pop("per_item", None)
    text = json.dumps(rep, indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v))
    if args.out:
        args.out.write_text(text)
    print(text)
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":  # pragma: no cover
    _main()
