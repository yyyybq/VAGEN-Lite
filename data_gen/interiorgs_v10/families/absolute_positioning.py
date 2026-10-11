"""absolute_positioning: be about d metres from X while X is in view."""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contract import Funnel, Obj
from . import common as C
from . import judges_v2 as J

TASK_TYPE = "absolute_positioning"
REGION_TYPE = "footprint_distance_band"
N_OBJECTS = 1
ROLE_NAMES = ("target",)
TERMINAL_IDENTIFIABLE_ROLES = (0,)
SYMMETRIC = False
DISTANCES_M = (1.0, 1.5, 2.0, 2.5, 3.0)
PARAM_MENU = {"distance_m": DISTANCES_M}
MIN_OBJECT_HEIGHT_M = 0.1
PROBE_AZIMUTHS = 12

GATES_DOC = f"""
absolute_positioning (judge {J.CANONICAL_TASK_METRIC_VERSION_V2} / absolute_distance_h1)
  objects_present  target AABB present in target_object
  in_front         all 8 AABB corners at camera depth > {J.MIN_CORNER_DEPTH_M} m
  visible          _project_object visible
  min_area         clipped bbox area ratio >= {J.MIN_AREA_RATIO:g}
  inside_frame     clipped/raw bbox area >= {J.MIN_INSIDE_FRACTION}   (same as v1 projective)
  distance         |dist - d| <= max({J.ABS_TOL_FLOOR_M} m, {J.ABS_TOL_REL:.0%} d), dist = horizontal
                   camera-centre -> AABB-footprint distance (nearest side; camera height is fixed so
                   only horizontal distance is controllable).  Band width >= 0.5 m > one 0.3 m step.
  menu d in {DISTANCES_M}; a value is offered only if a facing pose at that footprint
  distance passes every gate for at least one of {PROBE_AZIMUTHS} azimuths (pitch-0 feasibility:
  low objects lose their lower part closer than ~2 m).
""".strip()


def tuple_ok(objs: Sequence[Obj]) -> Tuple[bool, str]:
    if len(objs) != 1:
        return False, "wrong_arity"
    (obj,) = objs
    if C.is_structural(obj):
        return False, "structural_label"
    if float(obj.dims[2]) < MIN_OBJECT_HEIGHT_M:
        return False, "too_flat"
    return True, "ok"


def describe(item: Dict[str, Any]) -> str:
    ref = C.ref_of(C.item_objects(item)[0])
    d = float(item["target_region"]["params"]["distance_m"])
    return (f"Move to a spot about {C.format_metres(d)} meters away from {ref} "
            f"(measured to its nearest side), keeping {ref} in view.")


def build_item(scene_id: str, objs: Sequence[Obj], params: Dict[str, Any],
               init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ok, reason = tuple_ok(objs)
    if not ok:
        raise ValueError(f"{TASK_TYPE}: {reason}")
    d = float(params["distance_m"])
    if d not in DISTANCES_M:
        raise ValueError(f"{TASK_TYPE}: distance {d} not in menu {DISTANCES_M}")
    return C.make_item(scene_id, TASK_TYPE, objs, REGION_TYPE,
                       {"distance_m": d, "distance_definition": "horizontal_to_aabb_footprint"},
                       describe, init_c2w)


def judge(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    return J.judge_absolute_positioning(item, c2w)


def anchor_xy(item: Dict[str, Any]) -> Tuple[float, float]:
    c = C.entry_center(C.role_entry(item, 0))
    return float(c[0]), float(c[1])


def necessary_mask(item: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    d = float(item["target_region"]["params"]["distance_m"])
    tol = max(J.ABS_TOL_FLOOR_M, J.ABS_TOL_REL * d)
    dist = C.footprint_distances(C.role_entry(item, 0), xy)
    return np.abs(dist - d) <= tol + 1e-9


def probe_poses(item: Dict[str, Any], d: float) -> List[np.ndarray]:
    entry = C.role_entry(item, 0)
    centre = C.entry_center(entry)
    poses = []
    for k in range(PROBE_AZIMUTHS):
        az = 2.0 * math.pi * k / PROBE_AZIMUTHS
        x, y = C.point_at_footprint_distance(entry, az, d)
        poses.append(C.facing_pose(x, y, centre[:2]))
    return poses


def param_menu(objs: Sequence[Obj], funnel: Optional[Funnel] = None) -> List[Dict[str, Any]]:
    fn = funnel if funnel is not None else Funnel()
    out = []
    for d in DISTANCES_M:
        item = build_item("probe", objs, {"distance_m": d})
        if C.probe_item_feasible(item, judge, probe_poses(item, d)):
            out.append({"distance_m": d})
            fn.add(f"family_{TASK_TYPE}_params", "ok")
        else:
            fn.add(f"family_{TASK_TYPE}_params", "infeasible_at_pitch0")
    return out


def success_pose_candidates(scene: Any, item: Dict[str, Any], seed: int, **kwargs: Any) -> List[Dict[str, Any]]:
    return C.sample_success_poses(scene, item, seed, sys.modules[__name__], **kwargs)


sampler = success_pose_candidates
