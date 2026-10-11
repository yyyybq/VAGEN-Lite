"""size_distance_invariance: X and Y look equally tall in your view."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contract import Funnel, Obj
from . import common as C
from . import judges_v2 as J

TASK_TYPE = "size_distance_invariance"
REGION_TYPE = "equal_apparent_height"
N_OBJECTS = 2
ROLE_NAMES = ("object_a", "object_b")
TERMINAL_IDENTIFIABLE_ROLES = (0, 1)
SYMMETRIC = True
PARAM_MENU: Dict[str, Tuple[Any, ...]] = {}
MIN_OBJECT_HEIGHT_M = 0.3
# Projected AABB height of flat objects is dominated by their foreshortened
# top face, which does not read as "height" in RGB; keep upright-ish objects.
MIN_UPRIGHTNESS = 0.4          # height / max(footprint dims)
MIN_CENTRE_SEPARATION_M = 1.5

GATES_DOC = f"""
size_distance_invariance (judge {J.CANONICAL_TASK_METRIC_VERSION_V2} / apparent_height_equal_h1)
  objects_present / in_front / visible / min_area / inside_frame   v1 visibility gates for both
  vertical_contained     both objects clipped/raw height >= {J.SDI_VERTICAL_INSIDE_MIN}
  equal_apparent_height  |ln(h_A / h_B)| <= ln({J.SDI_RATIO_TOL}) on clipped projected bbox heights (px)
  pairs: same room, centre separation >= {MIN_CENTRE_SEPARATION_M} m, heights >= {MIN_OBJECT_HEIGHT_M} m,
  height >= {MIN_UPRIGHTNESS} x max footprint side (upright objects).
""".strip()


def tuple_ok(objs: Sequence[Obj]) -> Tuple[bool, str]:
    if len(objs) != 2:
        return False, "wrong_arity"
    a, b = objs
    if a.id == b.id:
        return False, "same_object"
    if C.is_structural(a) or C.is_structural(b):
        return False, "structural_label"
    for o in objs:
        h = float(o.dims[2])
        if h < MIN_OBJECT_HEIGHT_M:
            return False, "too_short"
        if h < MIN_UPRIGHTNESS * float(max(o.dims[0], o.dims[1])):
            return False, "not_upright"
    if not C.same_room_ok(objs):
        return False, "different_rooms"
    if C.centre_separation_xy(a, b) < MIN_CENTRE_SEPARATION_M:
        return False, "pair_too_close"
    return True, "ok"


def describe(item: Dict[str, Any]) -> str:
    a, b = (C.ref_of(o) for o in C.item_objects(item)[:2])
    return (f"Move so that {a} and {b} look equally tall in your view, "
            f"with both fully visible from top to bottom.")


def build_item(scene_id: str, objs: Sequence[Obj], params: Optional[Dict[str, Any]] = None,
               init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ok, reason = tuple_ok(objs)
    if not ok:
        raise ValueError(f"{TASK_TYPE}: {reason}")
    return C.make_item(scene_id, TASK_TYPE, objs, REGION_TYPE,
                       {"size_definition": "clipped_bbox_height_px"}, describe, init_c2w)


def judge(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    return J.judge_size_distance_invariance(item, c2w)


def anchor_xy(item: Dict[str, Any]) -> Tuple[float, float]:
    c = 0.5 * (C.entry_center(C.role_entry(item, 0)) + C.entry_center(C.role_entry(item, 1)))
    return float(c[0]), float(c[1])


def necessary_mask(item: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    a, b = C.role_entry(item, 0), C.role_entry(item, 1)
    return (C.footprint_distances(a, xy) > 0.0) & (C.footprint_distances(b, xy) > 0.0)


def param_menu(objs: Sequence[Obj], funnel: Optional[Funnel] = None) -> List[Dict[str, Any]]:
    fn = funnel if funnel is not None else Funnel()
    item = build_item("probe", objs)
    if C.generic_probe_feasible(item, sys.modules[__name__], radius=6.0, spacing=0.4):
        fn.add(f"family_{TASK_TYPE}_params", "ok")
        return [{}]
    fn.add(f"family_{TASK_TYPE}_params", "infeasible_at_pitch0")
    return []


def success_pose_candidates(scene: Any, item: Dict[str, Any], seed: int, **kwargs: Any) -> List[Dict[str, Any]]:
    return C.sample_success_poses(scene, item, seed, sys.modules[__name__], **kwargs)


sampler = success_pose_candidates
