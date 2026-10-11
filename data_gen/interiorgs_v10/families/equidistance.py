"""equidistance: be equally far from X and Y while both are in view."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contract import Funnel, Obj
from . import common as C
from . import judges_v2 as J

TASK_TYPE = "equidistance"
REGION_TYPE = "equidistant_band"
N_OBJECTS = 2
ROLE_NAMES = ("object_a", "object_b")
TERMINAL_IDENTIFIABLE_ROLES = (0, 1)
SYMMETRIC = True
PARAM_MENU: Dict[str, Tuple[Any, ...]] = {}
MIN_OBJECT_HEIGHT_M = 0.1
# Footprints closer than this make |d_A - d_B| small almost everywhere
# (it is bounded by their Hausdorff distance), i.e. the task is trivial.
MIN_FOOTPRINT_GAP_M = 1.0

GATES_DOC = f"""
equidistance (judge {J.CANONICAL_TASK_METRIC_VERSION_V2} / equidistance_h1)
  objects_present  both AABBs present and distinct
  in_front/visible/min_area/inside_frame   v1-projective visibility gates for BOTH objects
                   (all corners depth > {J.MIN_CORNER_DEPTH_M} m, area >= {J.MIN_AREA_RATIO:g}, inside >= {J.MIN_INSIDE_FRACTION})
  equidistant      |d_A - d_B| <= max({J.EQ_TOL_FLOOR_M} m, {J.EQ_TOL_REL:.0%} * mean(d_A, d_B)), d = horizontal
                   camera -> AABB-footprint distance.  The band around the bisector curve is
                   >= 2*0.25/(2 sin(phi/2)) >= 0.35 m wide for phi <= 90 deg (both objects in view).
  pairs: distinct, same room, footprint gap >= {MIN_FOOTPRINT_GAP_M} m (else trivial).
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
    return (f"Move to a spot that is equally far from {a} and {b} "
            f"(measured to their nearest sides), with both of them in view.")


def build_item(scene_id: str, objs: Sequence[Obj], params: Optional[Dict[str, Any]] = None,
               init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ok, reason = tuple_ok(objs)
    if not ok:
        raise ValueError(f"{TASK_TYPE}: {reason}")
    return C.make_item(scene_id, TASK_TYPE, objs, REGION_TYPE,
                       {"distance_definition": "horizontal_to_aabb_footprint"}, describe, init_c2w)


def judge(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    return J.judge_equidistance(item, c2w)


def anchor_xy(item: Dict[str, Any]) -> Tuple[float, float]:
    c = 0.5 * (C.entry_center(C.role_entry(item, 0)) + C.entry_center(C.role_entry(item, 1)))
    return float(c[0]), float(c[1])


def necessary_mask(item: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    d_a = C.footprint_distances(C.role_entry(item, 0), xy)
    d_b = C.footprint_distances(C.role_entry(item, 1), xy)
    tol = np.maximum(J.EQ_TOL_FLOOR_M, J.EQ_TOL_REL * 0.5 * (d_a + d_b))
    return np.abs(d_a - d_b) <= tol + 1e-9


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
