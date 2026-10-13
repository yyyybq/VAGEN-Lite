"""fov_inclusion (v2): both X and Y fully in view (user decision 2026-10-13: a formal family).

"Can both be seen clearly" is the env's visibility rule (target_visibility.py: each has a clearly visible
face), not R1's FOV image gate; the geometric judge (env canonical_task_metrics_v2.judge_fov_inclusion)
needs both projected boxes at least 95% inside the frame.  Not bit-identical to R1's v1 FOV.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contract import Funnel, Obj
from . import common as C
from . import judges_v2 as J

TASK_TYPE = "fov_inclusion"
REGION_TYPE = "fov"
N_OBJECTS = 2
ROLE_NAMES = ("object_a", "object_b")
TERMINAL_IDENTIFIABLE_ROLES = (0, 1)
SYMMETRIC = True
PARAM_MENU: Dict[str, Tuple[Any, ...]] = {}
MIN_OBJECT_HEIGHT_M = 0.1
# Objects this close are one view target in practice (and trivially framed together).
MIN_FOOTPRINT_GAP_M = 0.5

GATES_DOC = f"""
fov_inclusion (judge {J.CANONICAL_TASK_METRIC_VERSION_V2} / fov_inclusion_h1_v2)
  objects_present  both AABBs present and distinct
  in_front/visible/min_area/inside_frame   v1-projective visibility gates for BOTH objects
  fully_in_frame   both projected AABBs >= {J.FOV_FULL_INSIDE_MIN:.0%} inside the frame
  plus the env's visibility rule: each object has a clearly visible face.
  pairs: distinct, same room, footprint gap >= {MIN_FOOTPRINT_GAP_M} m.
""".strip()


def tuple_ok(objs: Sequence[Obj]) -> Tuple[bool, str]:
    if len(objs) != 2:
        return False, "wrong_arity"
    a, b = objs
    if a.id == b.id:
        return False, "same_object"
    if C.is_structural(a) or C.is_structural(b):
        return False, "structural_label"
    if min(float(a.dims[2]), float(b.dims[2])) < MIN_OBJECT_HEIGHT_M:
        return False, "too_flat"
    if not C.same_room_ok(objs):
        return False, "different_rooms"
    if C.footprint_gap(a, b) < MIN_FOOTPRINT_GAP_M:
        return False, "pair_too_close"
    return True, "ok"


def describe(item: Dict[str, Any]) -> str:
    a, b = (C.ref_of(o) for o in C.item_objects(item)[:2])
    return f"Move so that both {a} and {b} are fully in your view."


def build_item(scene_id: str, objs: Sequence[Obj], params: Optional[Dict[str, Any]] = None,
               init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ok, reason = tuple_ok(objs)
    if not ok:
        raise ValueError(f"{TASK_TYPE}: {reason}")
    return C.make_item(scene_id, TASK_TYPE, objs, REGION_TYPE, {}, describe, init_c2w)


def judge(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    return J.judge_fov_inclusion(item, c2w)


def anchor_xy(item: Dict[str, Any]) -> Tuple[float, float]:
    c = 0.5 * (C.entry_center(C.role_entry(item, 0)) + C.entry_center(C.role_entry(item, 1)))
    return float(c[0]), float(c[1])


def necessary_mask(item: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    """The camera must stand outside both footprints (all corners in front)."""
    d_a = C.footprint_distances(C.role_entry(item, 0), xy)
    d_b = C.footprint_distances(C.role_entry(item, 1), xy)
    return (d_a > 0.0) & (d_b > 0.0)


def param_menu(objs: Sequence[Obj], funnel: Optional[Funnel] = None) -> List[Dict[str, Any]]:
    fn = funnel if funnel is not None else Funnel()
    item = build_item("probe", objs)
    if C.generic_probe_feasible(item, sys.modules[__name__]):
        fn.add(f"family_{TASK_TYPE}_params", "ok")
        return [{}]
    fn.add(f"family_{TASK_TYPE}_params", "infeasible_at_pitch0")
    return []


def success_pose_candidates(scene: Any, item: Dict[str, Any], seed: int, **kwargs: Any) -> List[Dict[str, Any]]:
    return C.sample_success_poses(scene, item, seed, sys.modules[__name__], **kwargs)


sampler = success_pose_candidates
