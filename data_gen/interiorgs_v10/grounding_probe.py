"""Grounding probe (the user's supplement of 2026-11-12, item 5): an evaluation variant in which the model, every
turn, first states where the task's first object is and how far it is, then gives its action; the statement is
scored against the environment's ground truth at the pose the turn's frame was rendered from.  Training, rewards,
stop conditions and the action reading are unchanged.

    python -m data_gen.interiorgs_v10.evaluation run ... --grounding-probe    # results rows carry "grounding_turns"
    python -m data_gen.interiorgs_v10.grounding_probe summary --result EVAL/NAME [--result ...] --out FILE.json

The statement (one line before the action; the instruction, ``instruction``, is appended to every user turn):

    <state>bearing=<B>; distance=<D></state>

* **Object:** object A, the first object of ``target_object.objects`` (the one the task names first; the role
  letter "a" of canonical_task_metrics_v2, matched by ``object_a_id`` when the task has one), named in the
  instruction by its reference text (``ref``).
* **Bearing B** (front / left / right / back): the horizontal angle from the camera's facing direction (the c2w
  +Z column, the direction move_forward moves; projected onto the floor) to the object's centre, positive to the
  camera's right (+X column).  Four sectors, as the instruction defines them: front |a| <= 45 (the camera's
  horizontal field of view is 90 degrees), right 45 < a <= 135, left -135 <= a < -45, back otherwise.  Within
  ``BEARING_TOL_DEG`` of a sector boundary either neighbouring sector counts as right; with the camera inside the
  object's footprint (distance 0) any bearing does.  Read as front: "front", "in front", "forward", "ahead",
  "straight ahead"; as back: "back", "behind".
* **Distance D** (metres; "cm" is divided by 100, a comma decimal is read): the horizontal distance from the camera
  to the object's AABB footprint (0 inside it), the definition the tasks use ("measured to its nearest side",
  canonical_task_metrics_v2.footprint_distance).  Right when |D - truth| <= max(``DIST_TOL_FLOOR_M``,
  ``DIST_TOL_REL`` x truth).
* **Distance condition** (absolute_positioning only, the one family whose success has a distance band): the band
  of the stated D and of the true distance relative to the task's distance and the success gate's tolerance
  (max(0.25 m, 10%), ABS_TOL_*): too_close / ok / too_far; right when the two bands are the same.
* A missing or unreadable statement scores every field as wrong; the action is read as before (``answer_part``
  takes the first ``<state>...</state>`` out of the answer before the env reads it).

v2, the side call (``run --grounding-side``, VERSION_SIDE; the user's choice of 2026-11-12 after v1, whose
statements no trained model wrote readably): the action call is the plain evaluation's call, unchanged (no probe text;
the action stream is the plain run's).  After it, on the same frame and before the env moves the camera, a separate
greedy call gets the same system prompt and the same earlier turns (plain texts and answers; side calls never enter
the history), the plain user text plus ``side_question``, and an assistant reply prefilled with ``SIDE_PREFIX``
(``prefixed_prompt``: the prefix's tokens after the chat template's generation prompt); it stops at ``SIDE_STOP``
or after ``SIDE_MAX_TOKENS``.  ``side_text`` (prefix + completion, closed when it was not) is scored as above.  Each
turn also records the raw completion, how it ended, the side prompt's tokens and the layout check (``layout_check``:
the side prompt is the action prompt's generation prompt followed by the prefix).

The summary (review of 2026-11-12): successful episodes are short and end with the object in view and near, so even
a constant answer correlates with success.  Every reading is therefore reported next to the same reading of constant
answers on the same turns (``BASELINE_*``: each bearing; distances of 0.5-3 m, the turns' median true distance, and
for absolute_positioning the task's required distance; for the condition also each band), with the best of them and
the model's excess over it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "grounding_probe_v1"
# v2 (the user's choice of 2026-11-12, after v1 got no readable statement: the trained models write only actions): a
# side call per turn whose assistant reply is prefilled with SIDE_PREFIX; the action call is the plain evaluation's
VERSION_SIDE = "grounding_probe_v2_side"
SIDE_PREFIX = "<state>bearing="
SIDE_STOP = "</state>"
SIDE_MAX_TOKENS = 24  # "straight ahead; distance=12.75 m</state>" is 14 tokens
BEARINGS = ("front", "right", "back", "left")
BEARING_TOL_DEG = 10.0
DIST_TOL_FLOOR_M = 0.3
DIST_TOL_REL = 0.25
SYNONYMS = {"front": "front", "in front": "front", "forward": "front", "ahead": "front", "straight ahead": "front",
            "back": "back", "behind": "back", "left": "left", "right": "right"}
BANDS = ("too_close", "ok", "too_far")
BASELINE_DISTANCES_M = (0.5, 1.0, 1.5, 2.0, 3.0)
FIELDS = ("bearing_ok", "distance_ok", "condition_ok", "both_ok")

_STATE = re.compile(r"<\s*state\s*>(.*?)<\s*/\s*state\s*>", re.S | re.I)
_BEARING = re.compile(r"bearing\s*[=:]\s*([a-z][a-z _-]*?)\s*(?=[;,|]|distance\b|$)", re.S | re.I)
_DISTANCE = re.compile(r"distance\s*[=:]\s*(\d+(?:[.,]\d+)?|[.,]\d+)\s*(cm|centimet(?:er|re)s?|m|met(?:er|re)s?)?"
                       r"(?![a-z])", re.I)


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
            "<state>bearing=<B>; distance=<D></state>\n"
            f"<B> is where {object_ref(item)} is now: front (within 45 degrees of straight ahead, inside your view), "
            "left or right (45 to 135 degrees to that side), or back (otherwise).\n"
            "<D> is its horizontal distance from you in meters, to its nearest side, as a number.\n"
            "Then write the action as above.")


def side_question(item: Dict[str, Any]) -> str:
    """v2: the text appended to the plain user turn in the side call (the action call never sees it)."""
    return ("\n\nQuestion (answer it, no action this time): where is " + object_ref(item) + " now?\n"
            "Answer in one line: <state>bearing=<B>; distance=<D></state>\n"
            "<B> is front (within 45 degrees of straight ahead, inside your view), left or right (45 to 135 degrees "
            "to that side), or back (otherwise).\n"
            "<D> is its horizontal distance from you in meters, to its nearest side, as a number.")


_CLOSE = re.compile(r"<\s*/\s*state\s*>", re.I)


def side_text(completion: str) -> str:
    """v2: the statement as it is scored: the prefix, the completion, and the closing tag when the completion ended
    without it (the model's end of turn or the token limit)."""
    text = SIDE_PREFIX + (completion or "")
    return text if _CLOSE.search(completion or "") else text + SIDE_STOP


def prefixed_prompt(prompt: Dict[str, Any], tokenizer: Any, prefix: str) -> Tuple[Dict[str, Any], List[int]]:
    """v2: a chat prompt (vLLM's LLM.preprocess_chat, the code LLM.chat runs: the chat template with the generation
    prompt, then token ids) with the prefix's tokens appended after the generation prompt; no end-of-turn token
    follows the prefix.  Returns the new prompt and the prefix's ids."""
    ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    return dict(prompt, prompt_token_ids=list(prompt["prompt_token_ids"]) + ids), ids


def layout_check(action_ids: Sequence[int], side_ids: Sequence[int], prefix_ids: Sequence[int],
                 end_id: int) -> Dict[str, Any]:
    """v2, recorded per turn: the side prompt ends with the action prompt's generation prompt (its tokens from the
    user turn's end-of-turn token on) and then the prefix; and how many leading tokens the two prompts share."""
    action_ids, side_ids, prefix_ids = list(action_ids), list(side_ids), list(prefix_ids)
    last = max((i for i, t in enumerate(action_ids) if t == end_id), default=None)
    tail = action_ids[last:] if last is not None else []
    body = side_ids[: len(side_ids) - len(prefix_ids)]
    ok = bool(tail) and side_ids[len(body):] == prefix_ids and body[len(body) - len(tail):] == tail
    shared = 0
    for a, b in zip(action_ids, side_ids):
        if a != b:
            break
        shared += 1
    return {"layout_ok": ok, "shared_tokens": shared}


def _bearing_word(text: str) -> Optional[str]:
    return SYNONYMS.get(re.sub(r"[\s_-]+", " ", text.strip().lower()))


def parse_statement(answer: str) -> Dict[str, Any]:
    """{"present": a <state> block was found, "bearing": front/right/back/left or None, "distance": metres or None}
    from the first <state>...</state> of the answer."""
    m = _STATE.search(answer or "")
    if not m:
        return {"present": False, "bearing": None, "distance": None}
    body = m.group(1)
    b = _BEARING.search(body)
    d = _DISTANCE.search(body)
    distance = None
    if d:
        distance = float(d.group(1).replace(",", "."))
        if d.group(2) and d.group(2).lower().startswith("c"):
            distance /= 100.0
    return {"present": True, "bearing": _bearing_word(b.group(1)) if b else None, "distance": distance}


def answer_part(answer: str) -> str:
    """The answer without its first <state>...</state> block: what the env reads in a probe run."""
    return _STATE.sub("", answer or "", count=1)


# ---------------------------------------------------------------------------
# the ground truth and the scoring
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
    """The sector of the angle, and its neighbour within BEARING_TOL_DEG of a boundary; every bearing when there is
    no angle (the object's centre straight above or below the camera)."""
    if angle is None:
        return list(BEARINGS)
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


def distance_right(stated: Optional[float], true: float) -> bool:
    return stated is not None and abs(float(stated) - float(true)) <= max(DIST_TOL_FLOOR_M, DIST_TOL_REL * float(true))


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
                           # inside the footprint the object is all around: any bearing is right
                           "bearing_accepted": list(BEARINGS) if dist <= 0.0 else accepted_bearings(angle),
                           "distance_m": round(dist, 4)}
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
                           "bearing_ok": bool(st["bearing"] is not None and st["bearing"] in gt["bearing_accepted"]),
                           "distance_ok": distance_right(st["distance"], gt["distance_m"])}
    if "condition" in gt:
        stated = (None if st["distance"] is None else
                  distance_band(st["distance"], float(gt["required_m"]), float(gt["tolerance_m"])))
        rec["stated_condition"] = stated
        rec["condition_ok"] = bool(stated is not None and stated == gt["condition"])
    rec["both_ok"] = rec["bearing_ok"] and rec["distance_ok"]
    return rec


def episode_summary(turns: Sequence[Dict[str, Any]], version: str = VERSION) -> Dict[str, Any]:
    n = len(turns)
    share = lambda k: (round(sum(bool(t.get(k)) for t in turns) / n, 4) if n else None)  # noqa: E731
    out = {"version": version, "turns": n, "present": share("present"), "bearing_ok": share("bearing_ok"),
           "distance_ok": share("distance_ok"), "both_ok": share("both_ok")}
    if any("condition_ok" in t for t in turns):
        out["condition_ok"] = share("condition_ok")
    return out


# ---------------------------------------------------------------------------
# the constant-answer baselines
# ---------------------------------------------------------------------------
def baseline_marks(gt: Dict[str, Any], median_m: float) -> Dict[str, Dict[str, bool]]:
    """Per field, whether each constant answer would be right at this truth: each bearing; the distances
    BASELINE_DISTANCES_M, the turns' median true distance ("median") and the task's required distance ("echo
    required", absolute_positioning only: elsewhere it has no answer and is wrong); for the condition also each band
    ("band ...").  "both" pairs every bearing with every distance answer."""
    d = float(gt["distance_m"])
    answers: Dict[str, Optional[float]] = {f"{c:g} m": c for c in BASELINE_DISTANCES_M}
    answers["median"] = float(median_m)
    answers["echo required"] = gt.get("required_m")
    bearing = {b: b in gt["bearing_accepted"] for b in BEARINGS}
    distance = {n: distance_right(v, d) for n, v in answers.items()}
    marks = {"bearing_ok": bearing, "distance_ok": distance,
             "both_ok": {f"{b} / {n}": bearing[b] and distance[n] for b in BEARINGS for n in answers}}
    if "condition" in gt:
        req, tol = float(gt["required_m"]), float(gt["tolerance_m"])
        cond = {n: v is not None and distance_band(float(v), req, tol) == gt["condition"] for n, v in answers.items()}
        cond.update({f"band {b}": b == gt["condition"] for b in BANDS})
        marks["condition_ok"] = cond
    return marks


Pair = Tuple[Dict[str, Any], Dict[str, Dict[str, bool]]]  # (a turn's record, its baseline marks)

READABLE: Dict[str, Callable[[Dict[str, Any]], bool]] = {
    "bearing_ok": lambda t: t.get("stated_bearing") is not None,
    "distance_ok": lambda t: t.get("stated_distance") is not None,
    "condition_ok": lambda t: t.get("stated_distance") is not None,
    "both_ok": lambda t: t.get("stated_bearing") is not None and t.get("stated_distance") is not None,
}


def _rate(values: Sequence[bool]) -> Optional[float]:
    return round(sum(bool(v) for v in values) / len(values), 4) if values else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _best(model: Optional[float], base: Dict[str, Optional[float]], key: str = "best_baseline") -> Dict[str, Any]:
    """The model's value, the best baseline (highest; ties: the first name in sorted order) and the excess."""
    known = {n: v for n, v in base.items() if v is not None}
    if not known:
        return {"value": model, key: None, key + "_name": None, "excess": None}
    name = max(sorted(known), key=lambda n: known[n])
    excess = None if model is None else round(model - known[name], 4)
    return {"value": model, key: known[name], key + "_name": name, "excess": excess}


def _reading(pairs: Sequence[Pair], field: str) -> Dict[str, Any]:
    """Accuracy of the field over the turns that have it, and the constant answers' accuracy on the same turns."""
    sel = [(t, m) for t, m in pairs if field in t]
    if not sel:
        return {"n": 0, "value": None}
    names = sorted(sel[0][1][field])
    return {"n": len(sel), **_best(_rate([t[field] for t, _ in sel]),
                                   {n: _rate([m[field][n] for _, m in sel]) for n in names})}


def _readings(pairs: Sequence[Pair]) -> Dict[str, Any]:
    return {"n": len(pairs), "present": _rate([t["present"] for t, _ in pairs]),
            **{f: _reading(pairs, f) for f in FIELDS}}


def _episode_mean(eps: Sequence[Tuple[Dict[str, Any], List[Pair]]], field: str) -> Dict[str, Any]:
    """The mean over episodes of each episode's share of right turns, model and constant answers."""
    shares, base = [], defaultdict(list)
    for _, pairs in eps:
        sel = [(t, m) for t, m in pairs if field in t]
        if not sel:
            continue
        shares.append(sum(bool(t[field]) for t, _ in sel) / len(sel))
        for n in sel[0][1][field]:
            base[n].append(sum(bool(m[field][n]) for _, m in sel) / len(sel))
    return {"episodes": len(shares), **_best(_mean(shares), {n: _mean(v) for n, v in base.items()})}


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


def _correlation(eps: Sequence[Tuple[Dict[str, Any], List[Pair]]], field: str) -> Dict[str, Any]:
    """Point-biserial correlation of each episode's share of right turns with its success; the constant answers'
    correlations on the same episodes (they are not 0: successful episodes are short and end near the object)."""
    xs, ys, base = [], [], defaultdict(list)
    for row, pairs in eps:
        sel = [(t, m) for t, m in pairs if field in t]
        if not sel:
            continue
        ys.append(1.0 if row["success"] else 0.0)
        xs.append(sum(bool(t[field]) for t, _ in sel) / len(sel))
        for n in sel[0][1][field]:
            base[n].append(sum(bool(m[field][n]) for _, m in sel) / len(sel))
    return {"episodes": len(xs), **_best(_pearson(xs, ys), {n: _pearson(v, ys) for n, v in base.items()})}


def _two_by_two(first_ok: Sequence[bool], success: Sequence[bool]) -> Dict[str, Any]:
    """Success when the first turn's statement is right / wrong, and the phi coefficient (None without spread)."""
    a = sum(1 for f, s in zip(first_ok, success) if f and s)
    b = sum(1 for f, s in zip(first_ok, success) if f and not s)
    c = sum(1 for f, s in zip(first_ok, success) if not f and s)
    d = sum(1 for f, s in zip(first_ok, success) if not f and not s)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return {"right_n": a + b, "right_success": round(a / (a + b), 4) if a + b else None,
            "wrong_n": c + d, "wrong_success": round(c / (c + d), 4) if c + d else None,
            "phi": round((a * d - b * c) / den, 4) if den else None}


def _first_turn(eps: Sequence[Tuple[Dict[str, Any], List[Pair]]], field: str) -> Dict[str, Any]:
    firsts = [(pairs[0], bool(row["success"])) for row, pairs in eps if pairs and field in pairs[0][0]]
    if not firsts:
        return {"episodes": 0}
    succ = [s for _, s in firsts]
    model = _two_by_two([bool(t[field]) for (t, _), _ in firsts], succ)
    base = {n: _two_by_two([bool(m[field][n]) for (_, m), _ in firsts], succ)["phi"] for n in firsts[0][0][1][field]}
    best = _best(model["phi"], base, key="best_baseline_phi")
    return {"episodes": len(firsts), **model, "best_baseline_phi": best["best_baseline_phi"],
            "best_baseline_phi_name": best["best_baseline_phi_name"], "excess_phi": best["excess"]}


def _median(values: Sequence[float]) -> float:
    v = sorted(values)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Every reading of the statements next to the constant answers' reading on the same turns or episodes:
    accuracy over all turns (an unreadable statement is wrong), over the turns whose statement could be read, the
    per-episode mean; per turn index, per family, per outcome; each episode's share of right turns against its
    success (point-biserial) and success by the first turn's statement (2 x 2, phi), overall and per family."""
    eps_rows = [r for r in rows if "error" not in r and r.get("grounding_turns")]
    truths = [t["truth"] for r in eps_rows for t in r["grounding_turns"]]
    median_m = round(_median([float(t["distance_m"]) for t in truths]), 4) if truths else 0.0
    eps = [(r, [(t, baseline_marks(t["truth"], median_m)) for t in r["grounding_turns"]]) for r in eps_rows]
    pairs = [p for _, ps in eps for p in ps]
    out: Dict[str, Any] = {
        "versions": sorted({str((r.get("grounding") or {}).get("version", VERSION)) for r in eps_rows}),
        "episodes": len(eps), "turns": len(pairs),
        "success": _rate([r["success"] for r, _ in eps]),
        "baseline_answers": {"bearing": list(BEARINGS), "distance_m": list(BASELINE_DISTANCES_M),
                             "median_m": median_m, "echo_required": "absolute_positioning: the task's distance",
                             "condition_bands": list(BANDS)},
        "present": _rate([t["present"] for t, _ in pairs]),
        "readable": {f: _rate([READABLE[f](t) for t, _ in pairs]) for f in ("bearing_ok", "distance_ok")},
        "accuracy": {f: _reading(pairs, f) for f in FIELDS},
        "accuracy_readable": {f: _reading([(t, m) for t, m in pairs if READABLE[f](t)], f) for f in FIELDS},
        "accuracy_episode_mean": {f: _episode_mean(eps, f) for f in FIELDS},
        "episode_share_vs_success": {f: _correlation(eps, f) for f in FIELDS},
        "first_turn_vs_success": {f: _first_turn(eps, f) for f in FIELDS},
    }
    side = [t for t, _ in pairs if "side_output" in t]
    if side:  # v2: how the side calls ended and how many of them could be read
        out["side"] = {
            "turns": len(side),
            "parsed": _rate([READABLE["both_ok"](t) for t in side]),
            "bearing_readable": _rate([READABLE["bearing_ok"](t) for t in side]),
            "distance_readable": _rate([READABLE["distance_ok"](t) for t in side]),
            "closed": _rate([bool(_CLOSE.search(t["side_output"] or "")) for t in side]),
            "finish": dict(sorted(Counter(str(t.get("side_finish")) for t in side).items())),
            "layout_ok": _rate([t["side_layout_ok"] for t in side if "side_layout_ok" in t]),
            "prompt_tokens_mean": _mean([float(t["side_prompt_tokens"]) for t in side
                                         if t.get("side_prompt_tokens") is not None])}
    by_turn: Dict[int, List[Pair]] = defaultdict(list)
    for t, m in pairs:
        by_turn[int(t["turn"])].append((t, m))
    out["by_turn"] = {str(i): _readings(ps) for i, ps in sorted(by_turn.items())}
    fams: Dict[str, list] = defaultdict(list)
    for r, ps in eps:
        fams[str(r.get("task_type"))].append((r, ps))
    out["by_family"] = {f: {"episodes": len(fe), **_readings([p for _, ps in fe for p in ps]),
                            "first_turn_vs_success": {k: _first_turn(fe, k) for k in FIELDS}}
                        for f, fe in sorted(fams.items())}
    out["by_outcome"] = {name: {"episodes": sum(1 for r, _ in eps if bool(r["success"]) is keep),
                                **_readings([p for r, ps in eps if bool(r["success"]) is keep for p in ps])}
                         for name, keep in (("success", True), ("failure", False))}
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


def result_key(result: Path, merge_name: bool) -> str:
    """The summary's key of a result folder: "SET/NAME" (its evaluation set's folder and its own; the same model on
    val and test are two keys), or with --merge-name only NAME (the folders of one name are summarised together)."""
    return result.name if merge_name else f"{result.parent.name}/{result.name}"


def cmd_summary(args: argparse.Namespace) -> None:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for result in args.result:
        res = Path(result)
        key = result_key(res, bool(args.merge_name))
        if not args.merge_name and groups.get(key):
            raise SystemExit(f"two result folders give the key {key}: {groups[key][0]} and {res}")
        groups[key].append(res)
    out = {}
    for key, folders in groups.items():
        out[key] = summarize([r for res in folders for r in _jsonl(res / "results.jsonl")])
        out[key]["results"] = [str(res) for res in folders]
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="statement accuracy and its relation to success, next to constant answers")
    s.add_argument("--result", required=True, action="append", help="a probe run's result folder (repeatable)")
    s.add_argument("--out", default=None, help="JSON file")
    s.add_argument("--merge-name", action="store_true",
                   help="summarise the folders of one name together (e.g. a model's val and test); default: one key "
                        "per SET/NAME")
    args = p.parse_args(argv)
    {"summary": cmd_summary}[args.cmd](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
