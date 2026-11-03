"""Reward line, phase 2 direction 3 (perception statements; docs/reward/reports/PLAN_DIR3_20261103.md, the user's
decisions 1-5 甲 of 2026-11-03): what a statement about one frame should say, from the frame's pose alone (geometry,
no rendering), with the env's own success metric (target_visibility.score_with_visibility), so the labels and the
statement reward agree with the env's success.

Before each action the model writes

    <observe>where=A:left,B:in_view; face=A:no,B:yes; dist=far; missing=frame,clear</observe>

* where: per object of the task (A = object a, the first one the task names; B = b; C = c), where its centre is:
  "in_view" (in front of the camera and its projection inside the image), else by its horizontal bearing in the
  camera frame: "behind" beyond BEHIND_DEG from straight ahead, else "left" / "right".  Occlusion tasks: only the
  hidden target (a); the occluder is not stated.
* face: per object the success rule needs clearly visible (target_visibility.REQUIRED_VISIBLE_SLOTS; occlusion: the
  occluder b), whether it is (the env's face rule): "yes" / "no".
* dist: the horizontal distance from the camera to object a's footprint (canonical_task_metrics_v2.footprint_distance):
  absolute-distance tasks against the requested distance and the success tolerance ("near" closer than the band,
  "mid" inside it, "far" beyond); other tasks by DIST_EDGES_M ("near" below the first edge, "far" above the second).
* missing: which groups of success conditions are not met, in the order frame, clear, relation ("none" on success):
  "frame" the objects in front, visible, large enough, inside the frame (target_visibility.FRAME_CONDITIONS),
  "clear" the clearly-visible rule (every *_clearly_visible condition and required_targets_found), "relation" every
  other condition (distance, equidistant, relation, margin, occupancy, the 45-degree left/right rule, ...).

Pure functions; the env, the SFT data builder (reward/build_statement_sft.py) and the checks import them.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vagen.envs.active_spatial.canonical_task_metrics_v2 import (
    ABS_TOL_FLOOR_M,
    ABS_TOL_REL,
    _Context,
    footprint_distance,
)
from vagen.envs.active_spatial.target_visibility import (
    FRAME_CONDITIONS,
    REQUIRED_VISIBLE_SLOTS,
    _SLOT_LETTERS,
    score_with_visibility,
)

FIELDS = ("where", "face", "dist", "missing")
WHERE_VALUES = ("in_view", "left", "right", "behind")
DIST_VALUES = ("near", "mid", "far")
GROUPS = ("frame", "clear", "relation")
BEHIND_DEG = 135.0
MIN_DEPTH_M = 0.05
# the role letters stated in "where", per family (canonical_task_metrics_v2.ROLES; projective: a, b)
WHERE_ROLES = {
    "absolute_positioning": ("a",),
    "screen_occupancy": ("a",),
    "equidistance": ("a", "b"),
    "occlusion_alignment": ("a",),
    "size_distance_invariance": ("a", "b"),
    "centering": ("a", "b", "c"),
    "fov_inclusion": ("a", "b"),
    "projective_relations": ("a", "b"),
}
# the "dist" edges of the families without a requested distance, metres: the terciles of the footprint distance to
# object a over the C0 SFT data's 28,654 observed frames (reward/statement_check.py sft, 2026-11-03: 1.59 and 2.60),
# rounded; the "observe" prompt (vagen/envs/active_spatial/prompt.py) states them
DIST_EDGES_M: Optional[Tuple[float, float]] = (1.6, 2.6)
STATEMENT = re.compile(r"<observe>(.*?)</observe>", re.S)


def gate_group(gate: str) -> str:
    """The group of a success condition (a gates key of score_with_visibility)."""
    if gate.endswith("_clearly_visible") or gate == "required_targets_found":
        return "clear"
    return "frame" if gate in FRAME_CONDITIONS else "relation"


def missing_groups(failed: Sequence[str]) -> List[str]:
    """The groups of the failed conditions, in GROUPS order (empty: none failed)."""
    groups = {gate_group(g) for g in failed}
    return [g for g in GROUPS if g in groups]


def where_of(ctx: _Context, obj: Dict[str, Any]) -> str:
    """in_view / left / right / behind for an object's AABB centre."""
    lo = np.minimum(np.asarray(obj["bbox_min"], float)[:3], np.asarray(obj["bbox_max"], float)[:3])
    hi = np.maximum(np.asarray(obj["bbox_min"], float)[:3], np.asarray(obj["bbox_max"], float)[:3])
    pc = ctx.to_camera(((lo + hi) / 2.0)[None, :])[0]  # camera frame: x right, y down, z forward
    if pc[2] > MIN_DEPTH_M:
        u, v = ctx.uv(pc)
        if 0.0 <= u <= ctx.width and 0.0 <= v <= ctx.height:
            return "in_view"
    bearing = math.degrees(math.atan2(float(pc[0]), float(pc[2])))
    if abs(bearing) > BEHIND_DEG:
        return "behind"
    return "left" if bearing < 0 else "right"


def dist_of(item: Dict[str, Any], ctx: _Context, obj: Dict[str, Any],
            edges: Optional[Tuple[float, float]] = None) -> Tuple[str, float]:
    """(near / mid / far, the footprint distance in metres) to object a."""
    d = footprint_distance(float(ctx.position[0]), float(ctx.position[1]), obj)
    if str(item.get("task_type")) == "absolute_positioning":
        want = float(ctx.params["distance_m"])
        tol = max(ABS_TOL_FLOOR_M, ABS_TOL_REL * want)
        return ("near" if d < want - tol else "far" if d > want + tol else "mid"), d
    edges = edges if edges is not None else DIST_EDGES_M
    if edges is None:
        raise ValueError("statement_truth.DIST_EDGES_M is not set (pass edges)")
    return ("near" if d < edges[0] else "far" if d > edges[1] else "mid"), d


def truth(item: Dict[str, Any], c2w: Any, occ: Any, edges: Optional[Tuple[float, float]] = None,
          metric: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The four fields at one pose, and the metric's success and failed conditions.

    ``occ``: the scene's OccluderSet (the env's _visibility_occluders); ``metric``: the env's metric at this pose when
    the caller has it (score_with_visibility(item, c2w, occ)), so it is not computed twice."""
    task = str(item.get("task_type") or "")
    if metric is None:
        metric = score_with_visibility(item, np.asarray(c2w, float), occ)
    ctx = _Context(item, np.asarray(c2w, float))
    letters = WHERE_ROLES.get(task, ("a",))
    objs = ctx.roles(letters)
    if objs is None:
        raise ValueError(f"task {item.get('task_id')}: the objects {letters} are not found")
    where = {x.upper(): where_of(ctx, o) for x, o in zip(letters, objs)}
    vis = (metric.get("visibility") or {}).get("targets") or {}
    face = {}
    for slot in REQUIRED_VISIBLE_SLOTS.get(task, (0,)):
        x = _SLOT_LETTERS[slot]
        face[x.upper()] = bool((vis.get(x) or {}).get("visible", False))
    dist, meters = dist_of(item, ctx, objs[0], edges)  # every family's first role is a
    failed = sorted(k for k, v in (metric.get("gates") or {}).items() if not v)
    return {"where": where, "face": face, "dist": dist, "dist_m": round(meters, 4),
            "missing": missing_groups(failed), "success": bool(metric.get("success")), "failed_gates": failed}


def format_statement(t: Dict[str, Any]) -> str:
    """The statement text for a truth (the SFT label)."""
    where = ",".join(f"{k}:{v}" for k, v in t["where"].items())
    face = ",".join(f"{k}:{'yes' if v else 'no'}" for k, v in t["face"].items())
    missing = ",".join(t["missing"]) or "none"
    return f"<observe>where={where}; face={face}; dist={t['dist']}; missing={missing}</observe>"


def parse_statement(text: str) -> Optional[Dict[str, Any]]:
    """The fields of the first <observe> block of an answer, None when there is none or it cannot be read.  A field
    that is missing or malformed is None."""
    m = STATEMENT.search(text or "")
    if not m:
        return None
    out: Dict[str, Any] = {k: None for k in FIELDS}
    for part in m.group(1).split(";"):
        if "=" not in part:
            continue
        key, value = (s.strip().lower() for s in part.split("=", 1))
        if key in ("where", "face"):
            pairs = {}
            ok = True
            for item in value.split(","):
                if ":" not in item:
                    ok = False
                    break
                k, v = (s.strip() for s in item.split(":", 1))
                pairs[k.upper()] = v
            if not ok:
                continue
            if key == "where" and all(v in WHERE_VALUES for v in pairs.values()):
                out["where"] = pairs
            elif key == "face" and all(v in ("yes", "no") for v in pairs.values()):
                out["face"] = {k: v == "yes" for k, v in pairs.items()}
        elif key == "dist" and value in DIST_VALUES:
            out["dist"] = value
        elif key == "missing":
            names = [s.strip() for s in value.split(",") if s.strip()]
            if names == ["none"]:
                out["missing"] = []
            elif names and all(n in GROUPS for n in names):
                out["missing"] = [g for g in GROUPS if g in names]
    return out


def statement_token_count(tokenizer: Any, token_ids: Sequence[int]) -> int:
    """How many leading tokens of a generated answer hold its statement: the fewest whose decoded text contains
    "</observe>" (a token that also starts the action tag counts with the statement); 0 without one."""
    ids = list(token_ids)
    if "</observe>" not in tokenizer.decode(ids, skip_special_tokens=True):
        return 0
    lo, hi = 1, len(ids)
    while lo < hi:  # the decoded prefix only grows with more tokens
        mid = (lo + hi) // 2
        if "</observe>" in tokenizer.decode(ids[:mid], skip_special_tokens=True):
            hi = mid
        else:
            lo = mid + 1
    return lo


def score_statement(parsed: Optional[Dict[str, Any]], t: Dict[str, Any]) -> Dict[str, Any]:
    """Per field whether the statement is right (a field missing or unreadable is wrong) and the accuracy (right
    fields / 4); readable is False without an <observe> block."""
    if parsed is None:
        return {"readable": False, "accuracy": 0.0, **{k: False for k in FIELDS}}
    right = {"where": parsed["where"] == t["where"], "face": parsed["face"] == t["face"],
             "dist": parsed["dist"] == t["dist"], "missing": parsed["missing"] == t["missing"]}
    return {"readable": True, "accuracy": sum(right.values()) / len(FIELDS), **right}
