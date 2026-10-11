"""occlusion_alignment: X (target) hidden behind Y (occluder) from your view."""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contract import Funnel, Obj
from . import common as C
from . import judges_v2 as J

TASK_TYPE = "occlusion_alignment"
REGION_TYPE = "occlusion_shadow"
N_OBJECTS = 2
ROLE_NAMES = ("occluded_target", "occluder")
# The target is hidden in the terminal frame by definition: only the occluder
# must be identifiable there.  The initial frame should show the target.
TERMINAL_IDENTIFIABLE_ROLES = (1,)
INITIAL_PREFERRED_ROLES = (0,)
SYMMETRIC = False
PARAM_MENU: Dict[str, Tuple[Any, ...]] = {}
MIN_TARGET_HEIGHT_M = 0.1
MIN_OCCLUDER_HEIGHT_M = 0.4
MIN_FOOTPRINT_GAP_M = 0.1
# AABBs of these are mostly empty (legs, open frames, foliage, glass) or flat
# against a wall, so "hidden behind its AABB" would not match the RGB image.
SEE_THROUGH_HEADS = frozenset({
    "table", "desk", "chair", "stool", "lamp", "plant", "plants", "flower", "flowers", "vase",
    "shelf", "shelves", "bookshelf", "rack", "stand", "railing", "fence", "glass", "mirror",
    "frame", "painting", "picture", "poster", "door", "window", "curtain", "curtains", "blind",
    "blinds", "light", "chandelier", "fan", "rug", "carpet", "mat", "hanger", "bicycle", "trolley",
    "partition", "screen",
})

GATES_DOC = f"""
occlusion_alignment (judge {J.CANONICAL_TASK_METRIC_VERSION_V2} / occlusion_alignment_h1)
  roles: a = occluded target X (objects[0]), b = occluder Y (objects[1]); env guidance uses this order.
  objects_present             both AABBs present and distinct
  occluder_in_front/visible/min_area/inside_frame   v1 visibility gates for Y
  target_in_front             all 8 corners of X at camera depth > {J.MIN_CORNER_DEPTH_M} m
  target_in_view_direction    >= {J.OCC_TARGET_IN_FRAME_MIN:.0%} of X's silhouette projects inside the frame
                              (so "outside the frame" is never "hidden")
  target_hidden               >= {J.OCC_HIDDEN_MIN:.0%} of X's in-frame silhouette is behind Y: camera-facing AABB
                              faces of X sampled {J.OCC_FACE_SAMPLES}x{J.OCC_FACE_SAMPLES} per face, weighted by projected face
                              area, segment camera->sample hits Y's AABB (exact slab test, depth-ordered)
  pairs: same room, footprint gap >= {MIN_FOOTPRINT_GAP_M} m, occluder height >= {MIN_OCCLUDER_HEIGHT_M} m and
  not see-through ({', '.join(sorted(list(SEE_THROUGH_HEADS))[:8])}, ...).
""".strip()


def _non_occluding(label: str) -> bool:
    if C.label_head(label) in SEE_THROUGH_HEADS:
        return True
    occ = C._occlusion
    if occ is not None and hasattr(occ, "is_non_occluding"):
        try:
            return bool(occ.is_non_occluding(label))
        except Exception:  # pragma: no cover
            return False
    return False


def tuple_ok(objs: Sequence[Obj]) -> Tuple[bool, str]:
    if len(objs) != 2:
        return False, "wrong_arity"
    target, occluder = objs
    if target.id == occluder.id:
        return False, "same_object"
    if C.is_structural(target) or C.is_structural(occluder):
        return False, "structural_label"
    if float(target.dims[2]) < MIN_TARGET_HEIGHT_M:
        return False, "target_too_flat"
    if float(occluder.dims[2]) < MIN_OCCLUDER_HEIGHT_M:
        return False, "occluder_too_low"
    if _non_occluding(occluder.label):
        return False, "occluder_see_through"
    if not C.same_room_ok(objs):
        return False, "different_rooms"
    if C.footprint_gap(target, occluder) < MIN_FOOTPRINT_GAP_M:
        return False, "footprints_touch"
    return True, "ok"


def describe(item: Dict[str, Any]) -> str:
    x, y = (C.ref_of(o) for o in C.item_objects(item)[:2])
    return f"Move so that {x} is hidden behind {y} from your viewpoint, while {y} stays in view."


def build_item(scene_id: str, objs: Sequence[Obj], params: Optional[Dict[str, Any]] = None,
               init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ok, reason = tuple_ok(objs)
    if not ok:
        raise ValueError(f"{TASK_TYPE}: {reason}")
    return C.make_item(scene_id, TASK_TYPE, objs, REGION_TYPE, {}, describe, init_c2w,
                       role_aliases={"occluded": 0, "occluder": 1})


def judge(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    return J.judge_occlusion_alignment(item, c2w)


def anchor_xy(item: Dict[str, Any]) -> Tuple[float, float]:
    c = C.entry_center(C.role_entry(item, 1))
    return float(c[0]), float(c[1])


def _corners_xy(entry: Dict[str, Any]) -> np.ndarray:
    lo = np.asarray(entry["bbox_min"], float)
    hi = np.asarray(entry["bbox_max"], float)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _angular_interval(corners: np.ndarray, xy: np.ndarray, ref_angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """[lo, hi] bearing interval of a rectangle seen from each xy, relative to ref_angle."""
    centre = corners.mean(axis=0)
    own = np.arctan2(centre[1] - xy[:, 1], centre[0] - xy[:, 0])
    rel = np.stack([_wrap(np.arctan2(c[1] - xy[:, 1], c[0] - xy[:, 0]) - own) for c in corners], axis=1)
    shift = _wrap(own - ref_angle)
    return rel.min(axis=1) + shift, rel.max(axis=1) + shift


def necessary_mask(item: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    """Exact necessary condition for target_hidden > 0 at any yaw.

    Some camera->X segment must cross Y's footprint in plan view (bearing
    intervals overlap) and Y must be nearer than X's farthest corner.  Poses
    inside X's footprint cannot have X in front; poses inside Y's footprint
    (above a low occluder) are kept.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    t_entry, o_entry = C.role_entry(item, 0), C.role_entry(item, 1)
    d_t = C.footprint_distances(t_entry, xy)
    d_o = C.footprint_distances(o_entry, xy)
    tc, oc = _corners_xy(t_entry), _corners_xy(o_entry)
    ref = np.arctan2(tc.mean(axis=0)[1] - xy[:, 1], tc.mean(axis=0)[0] - xy[:, 0])
    t_lo, t_hi = _angular_interval(tc, xy, ref)
    o_lo, o_hi = _angular_interval(oc, xy, ref)
    overlap = np.zeros(len(xy), bool)
    for shift in (-2.0 * np.pi, 0.0, 2.0 * np.pi):
        overlap |= np.maximum(t_lo, o_lo + shift) <= np.minimum(t_hi, o_hi + shift)
    far_t = np.max(np.hypot(tc[None, :, 0] - xy[:, None, 0], tc[None, :, 1] - xy[:, None, 1]), axis=1)
    nearer = d_o <= far_t
    return (d_t > 0.0) & ((d_o <= 0.0) | (overlap & nearer))


def param_menu(objs: Sequence[Obj], funnel: Optional[Funnel] = None) -> List[Dict[str, Any]]:
    fn = funnel if funnel is not None else Funnel()
    item = build_item("probe", objs)
    if C.generic_probe_feasible(item, sys.modules[__name__], radius=6.0, spacing=0.25):
        fn.add(f"family_{TASK_TYPE}_params", "ok")
        return [{}]
    fn.add(f"family_{TASK_TYPE}_params", "infeasible_at_pitch0")
    return []


def success_pose_candidates(scene: Any, item: Dict[str, Any], seed: int, **kwargs: Any) -> List[Dict[str, Any]]:
    return C.sample_success_poses(scene, item, seed, sys.modules[__name__], **kwargs)


sampler = success_pose_candidates
