"""screen_occupancy: X fills about k of the vertical view (fully visible top to bottom)."""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contract import Funnel, Obj
from . import common as C
from . import judges_v2 as J

TASK_TYPE = "screen_occupancy"
REGION_TYPE = "vertical_occupancy_band"
N_OBJECTS = 1
ROLE_NAMES = ("target",)
TERMINAL_IDENTIFIABLE_ROLES = (0,)
SYMMETRIC = False
OCCUPANCY_RATIOS = (0.15, 0.25, 0.4, 0.6)
PARAM_MENU = {"occupancy_ratio": OCCUPANCY_RATIOS}
MIN_OBJECT_HEIGHT_M = 0.2
PROBE_AZIMUTHS = 8
PROBE_DISTANCES_M = tuple(np.round(np.arange(0.5, 7.01, 0.1), 2))
# A ratio is offered when some probe pose passes every non-occupancy gate with
# |occ - k| <= PROBE_MARGIN * tolerance (robustly inside the band).
PROBE_MARGIN = 0.5

GATES_DOC = f"""
screen_occupancy (judge {J.CANONICAL_TASK_METRIC_VERSION_V2} / screen_occupancy_h1)
  objects_present / in_front / visible / min_area / inside_frame   as absolute_positioning
  vertical_contained  clipped/raw projected bbox height >= {J.SO_VERTICAL_INSIDE_MIN} (top and bottom in view,
                      so the measured height is the whole object)
  occupancy           occ = clipped bbox height / image height (legacy definition);
                      |occ - k| <= max({J.SO_TOL_FLOOR}, {J.SO_TOL_REL:.0%} k); k in {OCCUPANCY_RATIOS}
                      (bands 0.12-0.18 / 0.21-0.29 / 0.34-0.46 / 0.51-0.69 do not overlap; occ ~ 1/depth
                      so the band is ~ +-15% of the viewing distance: >= 0.3 m beyond 2 m)
  pitch 0 at 1.5 m: a floor object's bottom enters the frame only beyond ~2 m, which caps
  achievable occupancy (~0.25 for 0.5 m tall objects, ~0.7 for 2.2 m wardrobes); the menu offers k
  only if a probe pose ({PROBE_AZIMUTHS} azimuths x 0.5-7 m) is robustly inside the band.
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
    k = float(item["target_region"]["params"]["occupancy_ratio"])
    return (f"Move so that {ref} is fully visible from top to bottom and its height fills "
            f"about {int(round(100 * k))}% of the image height.")


def build_item(scene_id: str, objs: Sequence[Obj], params: Dict[str, Any],
               init_c2w: Optional[np.ndarray] = None) -> Dict[str, Any]:
    ok, reason = tuple_ok(objs)
    if not ok:
        raise ValueError(f"{TASK_TYPE}: {reason}")
    k = float(params["occupancy_ratio"])
    if k not in OCCUPANCY_RATIOS:
        raise ValueError(f"{TASK_TYPE}: occupancy {k} not in menu {OCCUPANCY_RATIOS}")
    return C.make_item(scene_id, TASK_TYPE, objs, REGION_TYPE,
                       {"occupancy_ratio": k, "occupancy_definition": "clipped_bbox_height_over_image_height"},
                       describe, init_c2w)


def judge(item: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Any]:
    return J.judge_screen_occupancy(item, c2w)


def anchor_xy(item: Dict[str, Any]) -> Tuple[float, float]:
    c = C.entry_center(C.role_entry(item, 0))
    return float(c[0]), float(c[1])


def necessary_mask(item: Dict[str, Any], xy: np.ndarray) -> np.ndarray:
    # Occupancy depends on yaw (off-axis objects look taller); no cheap exact bound.
    entry = C.role_entry(item, 0)
    return C.footprint_distances(entry, xy) > 0.0


def occupancy_profile(objs: Sequence[Obj]) -> List[Tuple[float, float]]:
    """(footprint distance, occupancy) for probe poses passing every non-occupancy gate."""
    item = build_item("probe", objs, {"occupancy_ratio": OCCUPANCY_RATIOS[0]})
    entry = C.role_entry(item, 0)
    centre = C.entry_center(entry)
    out = []
    for k in range(PROBE_AZIMUTHS):
        az = 2.0 * math.pi * k / PROBE_AZIMUTHS
        for d in PROBE_DISTANCES_M:
            x, y = C.point_at_footprint_distance(entry, az, float(d))
            m = judge(item, C.facing_pose(x, y, centre[:2]))
            if all(v for g, v in m["gates"].items() if g != "occupancy"):
                out.append((float(d), float(m["occupancy"])))
    return out


def param_menu(objs: Sequence[Obj], funnel: Optional[Funnel] = None) -> List[Dict[str, Any]]:
    fn = funnel if funnel is not None else Funnel()
    profile = occupancy_profile(objs)
    out = []
    for k in OCCUPANCY_RATIOS:
        tol = J.screen_occupancy_tolerance(k)
        if any(abs(occ - k) <= PROBE_MARGIN * tol for _, occ in profile):
            out.append({"occupancy_ratio": k})
            fn.add(f"family_{TASK_TYPE}_params", "ok")
        else:
            fn.add(f"family_{TASK_TYPE}_params", "infeasible_at_pitch0")
    return out


def success_pose_candidates(scene: Any, item: Dict[str, Any], seed: int, **kwargs: Any) -> List[Dict[str, Any]]:
    return C.sample_success_poses(scene, item, seed, sys.modules[__name__], **kwargs)


sampler = success_pose_candidates
