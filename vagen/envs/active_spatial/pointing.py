"""Group G of matrix v3 ("point first, then act"; the user's instruction of 2026-11-06, part 4): the targets segment
an answer gives before its action, the truth it is judged against, and the judgment.

    <targets>A: (120,88); B: none</targets><action>move_forward|turn_left|</action>

A, B, ... are the objects the task names (``target_object.objects``, in the order the task text names them; the
format block of each user turn says which object each letter is, prompt.targets_format_block).  (x,y) is a point of
the object in the frame the answer was given on: pixels of the 256 x 256 image ("targets_pixel") or 0 to 1000 across
the image ("targets_norm1000", Qwen-VL's grounding convention); x from the left edge, y from the top edge.  "none":
the object is not clearly visible in that frame.

Truth: an object is in view when it is clearly visible by the success metrics' own rule
(target_visibility.object_visibility, the face rule with the scene's occluders); its box is its 3D box projected
into the frame and clipped to it (visual_bbox_metrics._project_object).  A point inside the box of an object in view
scores 1, "none" for an object not in view scores 1, anything else 0.  A turn scores the mean over the task's
objects, and 0 unless the whole answer is exactly the targets segment for those objects, in order, followed by the
action segment: short structure only, no free text (the lesson of group E).  The same functions give the SFT labels
(``label``: the centre of each box) and the RL reward (env pointing_reward times the turn's score)."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .prompt import SLOT_LETTERS, TARGET_FORMATS

FRAME_SIZE = (256, 256)  # target_visibility.RENDER_SIZE: the frame the success metrics and the boxes use
Point = Optional[Tuple[float, float]]  # None: "none"

_SPECIAL = re.compile(r"<\|[^>]*\|>")
_STRUCTURE = re.compile(r"\s*<targets>([^<]*)</targets>\s*<action>[^<]*</action>\s*", re.IGNORECASE)
_ACTION_ONLY = re.compile(r"\s*<action>[^<]*</action>\s*", re.IGNORECASE)
_NUM = r"(-?\d+(?:\.\d+)?)"
_ENTRY = re.compile(rf"\s*([A-Za-z])\s*:\s*(?:\(\s*{_NUM}\s*,\s*{_NUM}\s*\)|(none))\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# the task's objects
# ---------------------------------------------------------------------------
def slot_objects(item: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The objects the task names, in the order its text names them (target_object.objects; A, B, ...)."""
    objs = ((item or {}).get("target_object") or {}).get("objects") or []
    return [o for o in objs if isinstance(o, dict)][: len(SLOT_LETTERS)]


def target_refs(item: Optional[Dict[str, Any]]) -> List[str]:
    """The task text's words for each object (its ``ref``), as the format block lists them."""
    return [str(o.get("ref") or ("the " + str(o.get("label") or "object"))) for o in slot_objects(item)]


# ---------------------------------------------------------------------------
# reading an answer
# ---------------------------------------------------------------------------
def parse_targets(text: Any, n_slots: int) -> Dict[str, Any]:
    """The answer's targets segment.  {"present": the answer has a <targets> tag, "action_only": the whole answer is
    exactly one action segment (the old format), "ok": the whole answer is exactly a targets segment for the
    ``n_slots`` objects (letters A, B, ... in order, each "(x,y)" or "none") followed by an action segment, "entries":
    one point or None per object when ok, "error": "" or why not ok (no_targets, structure, entry, letters)}.
    Special tokens (<|...|>) are ignored."""
    t = _SPECIAL.sub("", str(text or ""))
    out = {"present": "<targets" in t.lower(), "action_only": _ACTION_ONLY.fullmatch(t) is not None, "ok": False,
           "entries": None, "error": ""}
    if not out["present"]:
        return dict(out, error="no_targets")
    m = _STRUCTURE.fullmatch(t)
    if m is None:
        return dict(out, error="structure")
    letters: List[str] = []
    entries: List[Point] = []
    for part in m.group(1).split(";"):
        e = _ENTRY.fullmatch(part)
        if e is None:
            return dict(out, error="entry")
        letters.append(e.group(1).upper())
        entries.append(None if e.group(4) else (float(e.group(2)), float(e.group(3))))
    if n_slots < 1 or letters != list(SLOT_LETTERS[:n_slots]):
        return dict(out, error="letters")
    return dict(out, ok=True, entries=entries)


def to_pixels(point: Point, format_example: str, size: Tuple[int, int] = FRAME_SIZE) -> Point:
    """A point of the answer in pixels of the frame."""
    if point is None:
        return None
    x, y = point
    if TARGET_FORMATS[format_example] == "norm1000":
        return (x * size[0] / 1000.0, y * size[1] / 1000.0)
    return (float(x), float(y))


# ---------------------------------------------------------------------------
# the truth on a frame
# ---------------------------------------------------------------------------
def truth(item: Dict[str, Any], c2w: Any, occ: Any) -> List[Dict[str, Any]]:
    """Per object of the task: {"letter", "in_view": clearly visible from ``c2w`` (the success metrics' face rule
    with the occluders ``occ``), "box": [x1, y1, x2, y2] in pixels (the projected 3D box, clipped) or None}."""
    import numpy as np

    from .target_visibility import item_camera, object_visibility
    from .visual_bbox_metrics import _project_object

    K, width, height = item_camera(item)
    if (width, height) != FRAME_SIZE:
        raise ValueError(f"the boxes are judged on a {FRAME_SIZE} frame, the item's camera gives {(width, height)}")
    c2w = np.asarray(c2w, float)
    out = []
    for letter, obj in zip(SLOT_LETTERS, slot_objects(item)):
        seen, box = False, None
        if "bbox_min" in obj and "bbox_max" in obj:
            proj = _project_object(obj, c2w, K, width, height)
            if proj.get("visible") and proj.get("bbox") is not None:
                seen = bool(object_visibility(c2w, obj, occ, K, width, height)["visible"])
                box = [round(float(v), 3) for v in proj["bbox"]] if seen else None
        out.append({"letter": letter, "in_view": seen, "box": box})
    return out


def scene_occluders(gs_root: str, scene_id: str, wall_model: str = "thick") -> Any:
    """The occluders of a scene as the RL env builds them with its collision detector (env._visibility_occluders):
    labels.json boxes and the structure walls of ``wall_model`` (the v10 env templates': "thick")."""
    from .collision_detector import create_collision_detector
    from .target_visibility import OccluderSet

    detector = create_collision_detector({"wall_model": wall_model})
    detector.load_scene_from_gs_root(gs_root, scene_id)
    return OccluderSet.from_detector(detector)


# ---------------------------------------------------------------------------
# judging, labelling
# ---------------------------------------------------------------------------
def hit(point: Point, t: Dict[str, Any]) -> bool:
    """A point (pixels) inside the box of an object in view, or None for an object not in view."""
    if not t["in_view"]:
        return point is None
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = t["box"]
    return x1 <= x <= x2 and y1 <= y <= y2


def score(entries: Optional[Sequence[Point]], truth_rows: Sequence[Dict[str, Any]], format_example: str) -> float:
    """The share of the task's objects the entries place right (0 without entries or truth)."""
    if not entries or not truth_rows or len(entries) != len(truth_rows):
        return 0.0
    return sum(hit(to_pixels(e, format_example), t) for e, t in zip(entries, truth_rows)) / len(truth_rows)


def judge_turn(answer: Any, item: Dict[str, Any], c2w: Any, occ: Any, format_example: str,
               readable: bool = True) -> Dict[str, Any]:
    """One turn: {"score", "ok": the short structure for the task's objects and a readable action, "old": exactly one
    readable action segment, nothing else (the old format: read, not counted as a format failure), "bad": neither
    (a format failure: a malformed segment, free text, no readable action), "error", "entries", "truth"}.  ``occ``
    None: no truth (a row without the canonical camera), never ok."""
    n = len(slot_objects(item))
    parsed = parse_targets(answer, n)
    ok = bool(parsed["ok"] and readable)
    error = parsed["error"] or ("" if readable else "action")
    rows: List[Dict[str, Any]] = []
    if ok:
        if occ is None:
            ok, error = False, "no_truth"
        else:
            rows = truth(item, c2w, occ)
    old = bool(parsed["action_only"] and readable)
    return {"score": float(score(parsed["entries"], rows, format_example)) if ok else 0.0, "ok": ok, "old": old,
            "bad": not ok and not old, "error": "" if ok else error, "entries": parsed["entries"], "truth": rows}


def box_point(box: Sequence[float]) -> Tuple[int, int]:
    """The label point of a box: its centre, rounded to whole units inside the box."""
    def one(lo: float, hi: float) -> int:
        c = int(round((lo + hi) / 2.0))
        a, b = math.ceil(lo), math.floor(hi)
        return min(max(c, a), b) if a <= b else c

    return one(box[0], box[2]), one(box[1], box[3])


def label(truth_rows: Sequence[Dict[str, Any]], format_example: str, size: Tuple[int, int] = FRAME_SIZE) -> str:
    """The SFT label of a turn's targets segment: the centre of each box in the format's coordinates (whole units,
    inside the box), "none" for an object not in view."""
    sx, sy = ((1000.0 / size[0], 1000.0 / size[1]) if TARGET_FORMATS[format_example] == "norm1000" else (1.0, 1.0))

    def one(t: Dict[str, Any]) -> str:
        if not t["in_view"]:
            return f"{t['letter']}: none"
        x1, y1, x2, y2 = t["box"]
        x, y = box_point([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
        return f"{t['letter']}: ({x},{y})"

    return "<targets>" + "; ".join(one(t) for t in truth_rows) + "</targets>"
