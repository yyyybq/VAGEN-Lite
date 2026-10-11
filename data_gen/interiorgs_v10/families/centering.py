"""centering: X appears halfway between Y and Z (left-right) in your view."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contract import Funnel, Obj
from . import common as C
from . import judges_v2 as J

TASK_TYPE = "centering"
REGION_TYPE = "image_midpoint"
N_OBJECTS = 3
ROLE_NAMES = ("centred_object", "flank_b", "flank_c")
TERMINAL_IDENTIFIABLE_ROLES = (0, 1, 2)
SYMMETRIC = True   # flanks b/c are interchangeable
PARAM_MENU: Dict[str, Tuple[Any, ...]] = {}
MIN_OBJECT_HEIGHT_M = 0.1
MIN_FLANK_SEPARATION_M = 1.0
MIN_CENTRE_GAP_M = 0.5   # centred object vs each flank (plan-view centre distance)

GATES_DOC = f"""
centering (judge {J.CANONICAL_TASK_METRIC_VERSION_V2} / image_centering_h1)
  roles: a = object to centre (objects[0]), b, c = flanks
  objects_present / in_front / visible / min_area / inside_frame   v1 visibility gates for all three
  flank_separation   |u_B - u_C| >= {J.CEN_MIN_SEPARATION_FRAC:.0%} of image width (projected AABB centres)
  centered           |u_A - (u_B + u_C)/2| <= max({J.CEN_TOL_FLOOR_PX:g} px, {J.CEN_TOL_REL:.0%} |u_B - u_C|)
  triples: same room, flank centres >= {MIN_FLANK_SEPARATION_M} m apart, centred object >= {MIN_CENTRE_GAP_M} m from each.
""".strip()


def tuple_ok(objs: Sequence[Obj]) -> Tuple[bool, str]:
    if len(objs) != 3:
        return False, "wrong_arity"
    a, b, c = objs
    if len({a.id, b.id, c.id}) != 3:
        return False, "same_object"
    if any(C.is_structural(o) for o in objs):
        return False, "structural_label"
    if min(float(o.dims[2]) for o in objs) < MIN_OBJECT_HEIGHT_M:
        return False, "too_flat"
    if not C.same_room_ok(objs):
        return False, "different_rooms"
    if C.centre_separation_xy(b, c) < MIN_FLANK_SEPARATION_M:
        return False, "flanks_too_close"
    if min(C.centre_separation_xy(a, b), C.centre_separation_xy(a, c)) < MIN_CENTRE_GAP_M:
        return False, "centre_object_too_close_to_flank"
    return True, "ok"


def describe(item: Dict[str, Any]) -> str:
    a, b, c = (C.ref_of(o) for o in C.item_objects(item)[:3])
    return (f"Move so that {a} appears exactly halfway between {b} and {c} "
            f"(left to right) in your view, with all three in view.")


def build_item(scene_id: str, objs: Sequence[Obj], params: Optional[Dict[str, Any]] = None,
               init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ok, reason = tuple_ok(objs)
    if not ok:
        raise ValueError(f"{TASK_TYPE}: {reason}")
    return C.make_item(scene_id, TASK_TYPE, objs, REGION_TYPE,
                       {"center_definition": "projected_aabb_centre_u"}, describe, init_c2w)


def judge(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    return J.judge_centering(item, c2w)


def anchor_xy(item: Dict[str, Any]) -> Tuple[float, float]:
    c = sum(C.entry_center(C.role_entry(item, i)) for i in range(3)) / 3.0
    return float(c[0]), float(c[1])


def necessary_mask(item: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    mask = np.ones(len(np.asarray(xy).reshape(-1, 2)), bool)
    for i in range(3):
        mask &= C.footprint_distances(C.role_entry(item, i), xy) > 0.0
    return mask


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
