"""Grounding probe (the user's supplement of 2026-11-12, item 5): an evaluation variant in which the model, every
turn, first states where the task's first object is and how far it is, then gives its action; the statement is
scored against the environment's ground truth at the pose the turn's frame was rendered from.  Training, rewards,
stop conditions and the action reading are unchanged.

    python -m data_gen.interiorgs_v10.evaluation run ... --grounding-probe    # results rows carry "grounding_turns"
    python -m data_gen.interiorgs_v10.grounding_probe summary --result EVAL/NAME [--result ...] --out FILE.json

The statement (one line before the action, the instruction appended to every user turn):

    <state>bearing=B; distance=D</state>

* **Object:** object A, the first object of ``target_object.objects`` (the one the task names first; the role
  letter "a" of canonical_task_metrics_v2), named in the instruction by its reference text (``ref``).
* **Bearing B** (front / left / right / back): the horizontal angle from the camera's facing direction (the c2w
  +Z column, the direction move_forward moves; projected onto the floor) to the object's centre, positive to the
  camera's right (+X column).  Four 90-degree sectors: front |a| <= 45 (the camera's horizontal field of view is
  90 degrees), right 45 < a <= 135, left -135 <= a < -45, back otherwise.  Within ``BEARING_TOL_DEG`` of a sector
  boundary either neighbouring sector counts as right.  "behind" is read as back, "ahead" as front.
* **Distance D** (metres): the horizontal distance from the camera to the object's AABB footprint (0 inside it),
  the definition the tasks use ("measured to its nearest side", canonical_task_metrics_v2.footprint_distance).
  Right when |D - truth| <= max(``DIST_TOL_FLOOR_M``, ``DIST_TOL_REL`` x truth).
* **Distance condition** (absolute_positioning only, the one family whose success has a distance band): the band
  of the stated D and of the true distance relative to the task's distance and the success gate's tolerance
  (max(0.25 m, 10%), ABS_TOL_*): too_close / ok / too_far; right when the two bands are the same.
* A missing or unreadable statement scores every field as wrong; the action is read as before (``answer_part``
  takes the first ``<state>...</state>`` out of the answer before the env reads it).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

VERSION = "grounding_probe_v1"
BEARINGS = ("front", "right", "back", "left")
BEARING_TOL_DEG = 10.0
DIST_TOL_FLOOR_M = 0.3
DIST_TOL_REL = 0.25
SYNONYMS = {"front": "front", "ahead": "front", "back": "back", "behind": "back", "left": "left", "right": "right"}

_STATE = re.compile(r"<state>(.*?)</state>", re.S | re.I)
_BEARING = re.compile(r"bearing\s*[=:]\s*([a-z][a-z_-]*)", re.I)  # "front-left" etc. read as unreadable
_DISTANCE = re.compile(r"distance\s*[=:]\s*([0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:m|meters?|metres?)?\b", re.I)


# ---------------------------------------------------------------------------
# the object, the instruction, the statement
# ---------------------------------------------------------------------------
def object_a(item: Dict[str, Any]) -> Dict[str, Any]:
    """Object A: the target_region's object_a_id among target_object.objects, else the first object."""
    objects = list(((item.get("target_object") or {}).get("objects")) or [])
    if not objects:
        raise ValueError(f"task {item.get('task_id')}: no target_object.objects")
    params = ((item.get("target_region") or {}).get("params")) or {}
    want = params.get("object_a_id")
    if want is not None:
        for obj in objects:
            if str(obj.get("id")) == str(want):
                return obj
    return objects[0]


def object_ref(item: Dict[str, Any]) -> str:
    obj = object_a(item)
    ref = obj.get("ref")
    return str(ref) if ref else f"the {obj.get('label', 'target object')}"


def instruction(item: Dict[str, Any]) -> str:
    """The text appended to every user turn of a probe run."""
    return ("\n\nBefore the action, first write one line in this form:\n"
            "<state>bearing=B; distance=D</state>\n"
            f"B is where {object_ref(item)} is now, relative to the direction you face: front, left, right or back. "
            "D is its horizontal distance from you in meters, to its nearest side (for example 1.5). "
            "Then write the action as above.")


def parse_statement(answer: str) -> Dict[str, Any]:
    """{"present": a <state> block was found, "bearing": front/right/back/left or None, "distance": float or None}
    from the first <state>...</state> of the answer."""
    m = _STATE.search(answer or "")
    if not m:
        return {"present": False, "bearing": None, "distance": None}
    body = m.group(1)
    b = _BEARING.search(body)
    d = _DISTANCE.search(body)
    bearing = SYNONYMS.get(b.group(1).lower()) if b else None
    return {"present": True, "bearing": bearing, "distance": float(d.group(1)) if d else None}


def answer_part(answer: str) -> str:
    """The answer without its first <state>...</state> block: what the env reads in a probe run."""
    return _STATE.sub("", answer or "", count=1)


# ---------------------------------------------------------------------------
# the ground truth
# ---------------------------------------------------------------------------
def footprint_distance(x: float, y: float, obj: Dict[str, Any]) -> float:
    from vagen.envs.active_spatial.canonical_task_metrics_v2 import footprint_distance as fd

    return float(fd(x, y, obj))


def bearing_angle(c2w: Any, point: Sequence[float]) -> Optional[float]:
    """Degrees from the camera's floor-projected facing direction to the point, positive to the right; None when the
    camera looks straight up or down or the point is at the camera."""
    pose = np.asarray(c2w, dtype=np.float64)
    forward, right = pose[:3, 2].copy(), pose[:3, 0].copy()
    forward[2] = right[2] = 0.0
    to = np.asarray(point, dtype=np.float64)[:3] - pose[:3, 3]
    to[2] = 0.0
    if min(np.linalg.norm(forward), np.linalg.norm(right), np.linalg.norm(to)) < 1e-9:
        return None
    forward, right = forward / np.linalg.norm(forward), right / np.linalg.norm(right)
    return float(math.degrees(math.atan2(float(to @ right), float(to @ forward))))


def sector(angle: float) -> str:
    a = (angle + 180.0) % 360.0 - 180.0
    if abs(a) <= 45.0:
        return "front"
    if 45.0 < a <= 135.0:
        return "right"
    if -135.0 <= a < -45.0:
        return "left"
    return "back"


def accepted_bearings(angle: Optional[float]) -> List[str]:
    """The sector of the angle, and its neighbour within BEARING_TOL_DEG of a boundary."""
    if angle is None:
        return []
    out = {sector(angle), sector(angle - BEARING_TOL_DEG), sector(angle + BEARING_TOL_DEG)}
    return [b for b in BEARINGS if b in out]


def distance_band(d: float, required: float, tol: float) -> str:
    if d < required - tol:
        return "too_close"
    if d > required + tol:
        return "too_far"
    return "ok"


def abs_tolerance(required: float) -> float:
    from vagen.envs.active_spatial.canonical_task_metrics_v2 import ABS_TOL_FLOOR_M, ABS_TOL_REL

    return float(max(ABS_TOL_FLOOR_M, ABS_TOL_REL * required))


def truth(item: Dict[str, Any], c2w: Any) -> Dict[str, Any]:
    """The ground truth at a pose (c2w, the env's view_engine.get_pose())."""
    obj = object_a(item)
    pose = np.asarray(c2w, dtype=np.float64)
    x, y = float(pose[0, 3]), float(pose[1, 3])
    center = obj.get("center")
    if center is None:
        lo, hi = np.asarray(obj["bbox_min"], dtype=np.float64), np.asarray(obj["bbox_max"], dtype=np.float64)
        center = ((lo + hi) / 2.0).tolist()
    angle = bearing_angle(pose, center)
    dist = footprint_distance(x, y, obj)
    out: Dict[str, Any] = {"bearing_deg": None if angle is None else round(angle, 2),
                           "bearing": None if angle is None else sector(angle),
                           "bearing_accepted": accepted_bearings(angle), "distance_m": round(dist, 4)}
    if item.get("task_type") == "absolute_positioning":
        params = ((item.get("target_region") or {}).get("params")) or {}
        required = params.get("distance_m")
        if required is not None and float(required) > 0:
            tol = abs_tolerance(float(required))
            out.update(required_m=float(required), tolerance_m=round(tol, 4),
                       condition=distance_band(dist, float(required), tol))
    return out


def score(answer: str, gt: Dict[str, Any], turn: int) -> Dict[str, Any]:
    """One turn's record: what was stated, the truth, and which fields are right (missing = wrong)."""
    st = parse_statement(answer)
    rec: Dict[str, Any] = {"turn": int(turn), "present": st["present"], "stated_bearing": st["bearing"],
                           "stated_distance": st["distance"], "truth": gt,
                           "bearing_ok": bool(st["bearing"] is not None and st["bearing"] in gt["bearing_accepted"])}
    tol = max(DIST_TOL_FLOOR_M, DIST_TOL_REL * float(gt["distance_m"]))
    rec["distance_ok"] = bool(st["distance"] is not None and abs(st["distance"] - float(gt["distance_m"])) <= tol)
    if "condition" in gt:
        stated = (None if st["distance"] is None else
                  distance_band(st["distance"], float(gt["required_m"]), float(gt["tolerance_m"])))
        rec["stated_condition"] = stated
        rec["condition_ok"] = bool(stated is not None and stated == gt["condition"])
    rec["both_ok"] = rec["bearing_ok"] and rec["distance_ok"]
    return rec


def episode_summary(turns: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(turns)
    share = lambda k: (round(sum(bool(t.get(k)) for t in turns) / n, 4) if n else None)  # noqa: E731
    out = {"version": VERSION, "turns": n, "present": share("present"), "bearing_ok": share("bearing_ok"),
           "distance_ok": share("distance_ok"), "both_ok": share("both_ok")}
    if any("condition_ok" in t for t in turns):
        out["condition_ok"] = share("condition_ok")
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
FIELDS = ("present", "bearing_ok", "distance_ok", "both_ok", "condition_ok")


def _rate(values: Sequence[bool]) -> Optional[float]:
    return round(sum(bool(v) for v in values) / len(values), 4) if values else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation (point-biserial when ys are 0/1); None when either side has no spread."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-12 or sy < 1e-12:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 4)


def _two_by_two(first_ok: Sequence[bool], success: Sequence[bool]) -> Dict[str, Any]:
    """Success when the first turn's statement is right / wrong, and the phi coefficient."""
    a = sum(1 for f, s in zip(first_ok, success) if f and s)
    b = sum(1 for f, s in zip(first_ok, success) if f and not s)
    c = sum(1 for f, s in zip(first_ok, success) if not f and s)
    d = sum(1 for f, s in zip(first_ok, success) if not f and not s)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return {"right_n": a + b, "right_success": round(a / (a + b), 4) if a + b else None,
            "wrong_n": c + d, "wrong_success": round(c / (c + d), 4) if c + d else None,
            "phi": round((a * d - b * c) / den, 4) if den else None}


def baselines(truths: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What a constant answer would score on the same turns: bearing always "front" (the start pose has the object in
    front in about 3 of 4 tasks), distance always the turns' median true distance (an oracle constant), the
    absolute_positioning condition always its most common band."""
    if not truths:
        return {}
    dists = sorted(float(t["distance_m"]) for t in truths)
    med = dists[len(dists) // 2] if len(dists) % 2 else 0.5 * (dists[len(dists) // 2 - 1] + dists[len(dists) // 2])
    out: Dict[str, Any] = {
        "bearing_front": _rate(["front" in t["bearing_accepted"] for t in truths]),
        "distance_median_m": round(med, 4),
        "distance_median": _rate([abs(med - d) <= max(DIST_TOL_FLOOR_M, DIST_TOL_REL * d) for d in dists])}
    conds = [t["condition"] for t in truths if "condition" in t]
    if conds:
        top = max(sorted(set(conds)), key=conds.count)
        out.update(condition_majority=top, condition_majority_share=_rate([c == top for c in conds]))
    return out


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy per field (over turns), per turn index, per family, split by episode success; the correlation of each
    episode's share of right statements with its success (point-biserial), and success by the first turn's
    statement (2 x 2, phi)."""
    eps = [r for r in rows if "error" not in r and "grounding_turns" in r]
    turns = [t for r in eps for t in r["grounding_turns"]]
    out: Dict[str, Any] = {"version": VERSION, "episodes": len(eps), "turns": len(turns),
                           "success": _rate([r["success"] for r in eps])}
    out["accuracy"] = {k: _rate([t[k] for t in turns if k in t]) for k in FIELDS}
    out["baselines"] = baselines([t["truth"] for t in turns if "truth" in t])
    by_turn: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in eps:
        for t in r["grounding_turns"]:
            by_turn[int(t["turn"])].append(t)
            by_family[str(r.get("task_type"))].append(t)
    out["by_turn"] = {str(i): {"n": len(ts), **{k: _rate([t[k] for t in ts if k in t]) for k in FIELDS[:4]}}
                      for i, ts in sorted(by_turn.items())}
    out["by_family"] = {f: {"n": len(ts), **{k: _rate([t[k] for t in ts if k in t]) for k in FIELDS}}
                        for f, ts in sorted(by_family.items())}
    split = {}
    for name, keep in (("success", True), ("failure", False)):
        ts = [t for r in eps if bool(r["success"]) is keep for t in r["grounding_turns"]]
        split[name] = {"episodes": sum(1 for r in eps if bool(r["success"]) is keep), "turns": len(ts),
                       **{k: _rate([t[k] for t in ts if k in t]) for k in FIELDS}}
    out["by_outcome"] = split
    ys = [1.0 if r["success"] else 0.0 for r in eps if r["grounding_turns"]]
    corr = {}
    for k in ("bearing_ok", "distance_ok", "both_ok"):
        xs = [sum(bool(t[k]) for t in r["grounding_turns"]) / len(r["grounding_turns"]) for r in eps
              if r["grounding_turns"]]
        corr[k] = _pearson(xs, ys)
    out["episode_share_vs_success"] = corr
    firsts = [(r["grounding_turns"][0], bool(r["success"])) for r in eps if r["grounding_turns"]]
    out["first_turn_vs_success"] = {k: _two_by_two([bool(f[k]) for f, _ in firsts], [s for _, s in firsts])
                                    for k in ("bearing_ok", "distance_ok", "both_ok")}
    return out


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    rows = []
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if k != len(lines) - 1:  # only the last line of a run still writing may be partial
                raise
    return rows


def cmd_summary(args: argparse.Namespace) -> None:
    out = {}
    for result in args.result:
        res = Path(result)
        out[res.name] = summarize(_jsonl(res / "results.jsonl"))
        out[res.name]["result"] = str(res)
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="statement accuracy and its relation to success, per result folder")
    s.add_argument("--result", required=True, action="append", help="a probe run's result folder (repeatable)")
    s.add_argument("--out", default=None, help="JSON file")
    args = p.parse_args(argv)
    {"summary": cmd_summary}[args.cmd](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
