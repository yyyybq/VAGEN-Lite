"""Readers (measurement -> answer) and question texts of the transition questions.

A reader takes the measurement dict of ``geometry.TaskGeometry.measure`` and returns ``Reading``: the answer
value and the quantity behind it, or the reason why no question is asked at this pose (object not visible,
too close to a decision threshold, ...).  ``pre=True`` readings are of the image the model sees, so the asked
objects must be clearly visible there; ``pre=False`` readings are of the pose after an action, which the
model does not see.

Thresholds (DEFAULT_CONFIG) are the judges' gates or sit outside them with a buffer
(docs/transition_qa/PROGRESS.md, table 1.3).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data_gen.interiorgs_v10.contract import INVERSE_ACTION, STEP_ROTATION_DEG, STEP_TRANSLATION_M
from data_gen.interiorgs_v10.qa_states import ref_text

DEFAULT_CONFIG: Dict[str, Any] = {
    "states_per_trajectory": 3,          # anchor states per trajectory, the terminal one included
    "goal_perturb_m": 0.05,              # goal labels must survive these pose changes
    "goal_perturb_deg": 3.0,
    "lr_min_sep_px": 20.0,               # projective margin gate 12 px + 8 px buffer (256-wide image)
    "lr_post_min_inside": 0.5,
    "closer_min_rel": 0.05,              # |dA - dB| >= 5% of the larger distance
    "dist_step_m": 0.5,                  # distance options: 4 neighbouring multiples of 0.5 m
    "dist_max_offset_m": 0.15,
    "dist_min_option_m": 0.5,
    "height_step": 0.1,                  # screen-height options: 4 neighbouring multiples of 10%
    "height_max_offset": 0.03,
    "height_min_option": 0.1,
    "height_max_option": 0.9,
    "height_vertical_min": 0.95,         # = the screen_occupancy judge's vertical containment gate
    "hidden_full": 0.85,                 # = the occlusion judge's hidden gate
    "hidden_none": 0.05,
    "hidden_partial": (0.25, 0.70),
    "oa_in_frame_min": 0.80,             # = the occlusion judge's in-frame gate
    "fov_in_min": 0.95,                  # = the FOV judge's fully-in-frame gate
    "fov_out_max": 0.30,
    "forward_k": (1, 2, 3),
    "forward_per_qtype": 3,              # candidates kept per anchor state and forward question type
    "goal_after_no_checked": 2,          # robustness-checked "no" candidates of goal_after per anchor state
    "changed": {"left_right": 10.0, "closer": 0.10, "distance_bin": 0.15, "height_bin": 0.03,
                "occluded": 0.10, "both_in_view": 0.10, "goal": 0.02},
    "inverse_k_max": 5,
}

FAMILY_QTYPE = {
    "projective_relations": "left_right",
    "equidistance": "closer",
    "absolute_positioning": "distance_bin",
    "occlusion_alignment": "occluded",
    "fov_inclusion": "both_in_view",
    "screen_occupancy": "height_bin",
}
QCLASS = {"now": "state", "after": "forward", "between": "inverse"}


@dataclasses.dataclass
class Reading:
    value: Any = None
    quantity: Any = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


def _skip(reason: str) -> Reading:
    return Reading(None, None, reason)


# =============================================================================
# readers
# =============================================================================
def read_left_right(m: Dict[str, Any], cfg: Dict[str, Any], pre: bool) -> Reading:
    a, b = m["objects"]["a"], m["objects"]["b"]
    if pre and not (a["clearly_visible"] and b["clearly_visible"]):
        return _skip("not_visible")
    if not (a["center_in_front"] and b["center_in_front"]) or a["u"] is None or b["u"] is None:
        return _skip("behind_camera")
    if not pre and min(a["inside_fraction"], b["inside_fraction"]) < cfg["lr_post_min_inside"]:
        return _skip("out_of_frame")
    if not m.get("lateral_ok", False):
        return _skip("beyond_45deg")
    du = float(a["u"]) - float(b["u"])
    if abs(du) < cfg["lr_min_sep_px"]:
        return _skip("near_boundary")
    return Reading("left" if du < 0 else "right", du)


def read_closer(m: Dict[str, Any], cfg: Dict[str, Any], pre: bool) -> Reading:
    a, b = m["objects"]["a"], m["objects"]["b"]
    if pre and not (a["clearly_visible"] and b["clearly_visible"]):
        return _skip("not_visible")
    da, db = float(a["distance_m"]), float(b["distance_m"])
    if abs(da - db) < cfg["closer_min_rel"] * max(da, db, 1e-6):
        return _skip("near_boundary")
    return Reading("a" if da < db else "b", da - db)


def _grid_value(x: float, step: float, max_offset: float, lo: float, hi: float) -> Tuple[Optional[float], str]:
    g = round(x / step) * step
    if g < lo - 1e-9:
        return None, "below_options"
    if g > hi + 1e-9:
        return None, "above_options"
    if abs(x - g) > max_offset:
        return None, "near_boundary"
    return round(g, 2), ""


def read_distance_bin(m: Dict[str, Any], cfg: Dict[str, Any], pre: bool) -> Reading:
    t = m["objects"]["a"]
    if pre and not t["clearly_visible"]:
        return _skip("not_visible")
    d = float(t["distance_m"])
    g, why = _grid_value(d, cfg["dist_step_m"], cfg["dist_max_offset_m"], cfg["dist_min_option_m"], math.inf)
    return Reading(g, d) if g is not None else _skip(why)


def read_height_bin(m: Dict[str, Any], cfg: Dict[str, Any], pre: bool) -> Reading:
    t = m["objects"]["a"]
    if pre and not t["clearly_visible"]:
        return _skip("not_visible")
    if not (t["all_corners_in_front"] and t["projected_visible"]):
        return _skip("behind_camera")
    if t["vertical_inside_fraction"] < cfg["height_vertical_min"]:
        return _skip("cut_off_vertically")
    occ = float(t["occupancy"])
    g, why = _grid_value(occ, cfg["height_step"], cfg["height_max_offset"], cfg["height_min_option"],
                         cfg["height_max_option"])
    return Reading(g, occ) if g is not None else _skip(why)


def read_occluded(m: Dict[str, Any], cfg: Dict[str, Any], pre: bool) -> Reading:
    a, b = m["objects"]["a"], m["objects"]["b"]
    if pre and not b["clearly_visible"]:
        return _skip("occluder_not_visible")
    if not pre and not (b["all_corners_in_front"] and b["projected_visible"]):
        return _skip("occluder_out_of_view")
    if not a["all_corners_in_front"]:
        return _skip("target_behind_camera")
    if float(m["in_frame_fraction"]) < cfg["oa_in_frame_min"]:
        return _skip("target_out_of_frame")
    h = float(m["hidden_fraction"])
    lo, hi = cfg["hidden_partial"]
    if h >= cfg["hidden_full"]:
        return Reading("full", h)
    if not a["clearly_visible"]:
        return _skip("target_not_visible")
    if h <= cfg["hidden_none"]:
        return Reading("none", h)
    if lo <= h <= hi:
        return Reading("partial", h)
    return _skip("near_boundary")


def _in_view_status(o: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[bool]:
    if o["all_corners_in_front"] and o["inside_fraction"] >= cfg["fov_in_min"] and o["clearly_visible"]:
        return True
    if (not o["center_in_front"]) or (not o["projected_visible"]) or \
            (o["all_corners_in_front"] and o["inside_fraction"] <= cfg["fov_out_max"]):
        return False
    return None


def read_both_in_view(m: Dict[str, Any], cfg: Dict[str, Any], pre: bool) -> Reading:
    sa = _in_view_status(m["objects"]["a"], cfg)
    sb = _in_view_status(m["objects"]["b"], cfg)
    if sa is None or sb is None:
        return _skip("near_boundary")
    value = {(True, True): "both", (True, False): "only_a", (False, True): "only_b", (False, False): "neither"}[(sa, sb)]
    q = (m["objects"]["a"]["inside_fraction"], m["objects"]["b"]["inside_fraction"])
    return Reading(value, q)


READERS = {
    "left_right": read_left_right,
    "closer": read_closer,
    "distance_bin": read_distance_bin,
    "height_bin": read_height_bin,
    "occluded": read_occluded,
    "both_in_view": read_both_in_view,
}


def changed(kind: str, pre: Reading, post: Reading, m_pre: Dict[str, Any], m_post: Dict[str, Any],
            cfg: Dict[str, Any]) -> bool:
    """Did the asked quantity change by a meaningful amount (or did the answer change)?"""
    thr = cfg["changed"][kind]
    if pre.ok and post.ok and pre.value != post.value:
        return True
    if kind == "left_right":
        ua = [m["objects"][s]["u"] for m in (m_pre, m_post) for s in ("a", "b")]
        if any(u is None for u in ua):
            return True
        return abs((ua[2] - ua[3]) - (ua[0] - ua[1])) >= thr
    if kind == "closer":
        d = [m["objects"]["a"]["distance_m"] - m["objects"]["b"]["distance_m"] for m in (m_pre, m_post)]
        return abs(d[1] - d[0]) >= thr
    if kind == "distance_bin":
        return abs(m_post["objects"]["a"]["distance_m"] - m_pre["objects"]["a"]["distance_m"]) >= thr
    if kind == "height_bin":
        return abs(m_post["objects"]["a"]["occupancy"] - m_pre["objects"]["a"]["occupancy"]) >= thr
    if kind == "occluded":
        return abs(m_post["hidden_fraction"] - m_pre["hidden_fraction"]) >= thr or \
            abs(m_post["in_frame_fraction"] - m_pre["in_frame_fraction"]) >= thr
    if kind == "both_in_view":
        return any(abs(m_post["objects"][s]["inside_fraction"] - m_pre["objects"][s]["inside_fraction"]) >= thr
                   for s in ("a", "b"))
    if kind == "goal":
        return bool(m_pre["success"]) != bool(m_post["success"]) or \
            abs(float(m_post["potential"]) - float(m_pre["potential"])) >= thr
    raise ValueError(kind)


# =============================================================================
# texts
# =============================================================================
LETTERS = "ABCD"
ANSWER_LETTER = "Answer with the option's letter."
ANSWER_YN = "Answer Yes or No."


def action_text(action: str, k: int, past: bool = False) -> str:
    k = int(k)
    if action.startswith("turn"):
        side = action.split("_", 1)[1]
        deg = int(round(k * STEP_ROTATION_DEG))
        return f"{'turned' if past else 'turns'} {side} by {deg} degrees"
    dist = f"{k * STEP_TRANSLATION_M:.1f} meters"
    verb = "moved" if past else "moves"
    return {
        "move_forward": f"{verb} forward by {dist}",
        "move_backward": f"{verb} backward by {dist}",
        "move_left": f"{verb} sideways to the left by {dist}",
        "move_right": f"{verb} sideways to the right by {dist}",
    }[action]


def options_block(options: Sequence[str]) -> str:
    return "\n".join(f"({LETTERS[i]}) {text}" for i, text in enumerate(options))


def stable_int(*parts: Any) -> int:
    return int(hashlib.sha1(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def grid_options(value: float, step: float, lo: float, hi: float, key: str) -> Tuple[List[float], int]:
    """4 neighbouring grid values containing ``value``; its position is drawn (by ``key``) among the feasible ones."""
    feasible = []
    for pos in range(4):
        first = value - pos * step
        last = first + 3 * step
        if first >= lo - 1e-9 and last <= hi + 1e-9:
            feasible.append(pos)
    if not feasible:
        raise ValueError(f"no 4 options around {value} in [{lo}, {hi}]")
    pos = feasible[stable_int(key, "grid") % len(feasible)]
    first = value - pos * step
    return [round(first + i * step, 2) for i in range(4)], pos


def _prefix(action: Optional[Tuple[str, int]]) -> str:
    """Condition of a state question (None) or of a forward question."""
    if action is None:
        return "In the current view, "
    return f"If the camera {action_text(*action)} from the current view, "


def build_text(kind: str, item: Dict[str, Any], roles: Dict[str, Dict[str, Any]], value: Any,
               action: Optional[Tuple[str, int]], key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Question text, options and the answer for one reading.  ``action`` None = state question."""
    prefix = _prefix(action)
    refs = {s: ref_text(e) for s, e in roles.items()}
    after = action is not None
    if kind == "goal":
        desc = str(item.get("task_description") or "").strip()
        if after:
            q = f"Task: {desc}\nIf the camera {action_text(*action)} from the current view, will the task's " \
                f"requirement be met? {ANSWER_YN}"
        else:
            q = f"Task: {desc}\nIn the current view, is the task's requirement already met? {ANSWER_YN}"
        return {"question": q, "options": [], "answer": "Yes" if value else "No", "answer_value": "yes" if value else "no"}
    if kind == "left_right":
        verb = "will" if after else "is"
        be = " be" if after else ""
        opts = ["to the left", "to the right"]
        q = f"{prefix}{verb} {refs['a']}{be} to the left or to the right of {refs['b']}?\n{options_block(opts)}\n{ANSWER_LETTER}"
        return {"question": q, "options": opts, "answer": LETTERS[0 if value == "left" else 1], "answer_value": value}
    if kind == "closer":
        order = ("a", "b") if stable_int(key, "order") % 2 == 0 else ("b", "a")
        opts = [refs[s] for s in order]
        verb = "will be" if after else "is"
        q = f"{prefix}which {verb} closer to you (horizontal distance to its nearest side): " \
            f"{opts[0]} or {opts[1]}?\n{options_block(opts)}\n{ANSWER_LETTER}"
        return {"question": q, "options": opts, "answer": LETTERS[order.index(value)], "answer_value": value}
    if kind == "distance_bin":
        vals, pos = grid_options(float(value), cfg["dist_step_m"], cfg["dist_min_option_m"], math.inf, key)
        opts = [f"about {v:.1f} meters" for v in vals]
        verb = "will you be" if after else "are you"
        q = f"{prefix}about how far {verb} from {refs['a']}, measured horizontally to its nearest side?\n" \
            f"{options_block(opts)}\n{ANSWER_LETTER}"
        return {"question": q, "options": opts, "answer": LETTERS[pos], "answer_value": f"{float(value):.1f}"}
    if kind == "height_bin":
        vals, pos = grid_options(float(value), cfg["height_step"], cfg["height_min_option"], cfg["height_max_option"], key)
        opts = [f"about {int(round(v * 100))}%" for v in vals]
        verb = "will" if after else "does"
        q = f"{prefix}about what fraction of the image height {verb} {refs['a']} fill, from its top to its bottom?\n" \
            f"{options_block(opts)}\n{ANSWER_LETTER}"
        return {"question": q, "options": opts, "answer": LETTERS[pos], "answer_value": f"{float(value):.1f}"}
    if kind == "occluded":
        opts = [f"fully hidden behind {refs['b']}", f"partly hidden behind {refs['b']}", f"not hidden behind {refs['b']}"]
        verb = "will" if after else "is"
        be = " be" if after else ""
        q = f"{prefix}{verb} {refs['a']}{be} hidden behind {refs['b']}?\n{options_block(opts)}\n{ANSWER_LETTER}"
        return {"question": q, "options": opts, "answer": LETTERS[["full", "partial", "none"].index(value)],
                "answer_value": value}
    if kind == "both_in_view":
        opts = ["both are", f"only {refs['a']}", f"only {refs['b']}", "neither"]
        if after:
            q = f"{prefix}will {refs['a']} and {refs['b']} both be inside the view?"
        else:
            q = f"{prefix}are {refs['a']} and {refs['b']} both inside the view?"
        q = f"{q}\n{options_block(opts)}\n{ANSWER_LETTER}"
        return {"question": q, "options": opts,
                "answer": LETTERS[["both", "only_a", "only_b", "neither"].index(value)], "answer_value": value}
    raise ValueError(kind)


# =============================================================================
# inverse questions
# =============================================================================
SAME_SIDE_OTHER_KIND = {"turn_left": "move_left", "turn_right": "move_right",
                        "move_left": "turn_left", "move_right": "turn_right"}


def inverse_options(action: str, k: int, key: str, k_max: int = 5) -> Tuple[List[Tuple[str, int]], int]:
    """Correct (action, k) plus 3 distractors from the action space: the same action another number of times,
    the opposite action the same number of times, the other kind of action on the same side (turn <-> strafe;
    a sideways move for forward / backward).  The correct option's position is drawn by ``key``."""
    k = int(k)
    others = [j for j in range(1, k_max + 1) if j != k]
    near = [j for j in others if abs(j - k) == 1] or others
    k2 = near[stable_int(key, "k") % len(near)]
    if action in SAME_SIDE_OTHER_KIND:
        third = (SAME_SIDE_OTHER_KIND[action], k)
    else:
        third = (("move_left", "move_right")[stable_int(key, "side") % 2], k)
    distractors = [(action, k2), (INVERSE_ACTION[action], k), third]
    pos = stable_int(key, "pos") % 4
    opts = list(distractors)
    opts.insert(pos, (action, k))
    if len(set(opts)) != 4:
        raise AssertionError(f"inverse options not distinct: {opts}")
    return opts, pos


def build_inverse_text(action: str, k: int, key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    opts, pos = inverse_options(action, k, key, int(cfg["inverse_k_max"]))
    texts = [f"the camera {action_text(a, j, past=True)}" for a, j in opts]
    q = ("The first image was taken right before a camera action and the second image right after it. "
         f"What did the camera do?\n{options_block(texts)}\n{ANSWER_LETTER}")
    return {"question": q, "options": texts, "answer": LETTERS[pos], "answer_value": f"{action}x{int(k)}"}
