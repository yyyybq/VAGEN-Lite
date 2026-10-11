"""Per-modality admission for InteriorGS v10 (workstream A).

Binding user decisions (2026-10-10, SPEC.md):
  * RL / SFT episodes: the INITIAL frame needs at least one identifiable
    target (the other may be cropped, occluded or out of view); the TERMINAL
    frame needs every target identifiable.  Intermediate frames are not
    required to show the targets.
  * Reject only genuinely invalid tasks: label/box misaligned, wrong object,
    referring expression ambiguous in the image, render artefact, task
    impossible.  Everything recoverable is HOLD (see configs/admission_v2.json).
  * RL, SFT and QA are admitted by their own information needs.  QA states are
    admitted per state, independently of the parent RL decision, but never
    from a task marked task_invalid.

Pipeline position::

    candidates --precheck_task--> plausible --qa_states.generate_qa--> QA items
      --review_requests--> [cluster: render + overlay + VLM/human review]
      --admit(candidates, reviews, policy, qa_items=..., renders=...)--> decisions
      --release.build_release--> RL rows / SFT / QA / registry / split manifest

Review schema (one JSON object per task and reviewer)::

    {"review_schema_version": "interiorgs_v10_review_v2",
     "task_id": "...",
     "reviewer": {"kind": "vlm" | "human", "name": "...", "prompt_version": "..."},
     "task_level": {"label_matches": {"A": bool|null, "B": ...},
                    "referring_unambiguous": {"A": bool|null, ...},
                    "task_possible": bool|null, "notes": ""},
     "frames": {"s0": {"render_ok": bool|null,
                       "targets_identifiable": {"A": bool|null, ...},
                       "box_alignment_ok": {"A": bool|null, ...},
                       "notes": ""}, ...}}

Frame ids: ``s<k>`` is the state after k certificate actions (``s0`` initial,
``s<d>`` terminal); ``n<k>_<action>`` are off-path QA neighbours.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .contract import ACTIONS, MAX_PRIMITIVE_STEPS, RENDER_SIZE, TASK_ROW_REQUIRED_KEYS, Funnel
from .qa_states import (
    certificate,
    certificate_poses,
    geometric_identifiable,
    project_entries,
    ref_text,
    score_backend,
    task_slots,
)

POLICY_PATH = Path(__file__).resolve().parent / "configs" / "admission_v2.json"
REVIEW_SCHEMA_VERSION = "interiorgs_v10_review_v2"
ADMIT, HOLD, REJECT = "ADMIT", "HOLD", "REJECT"
FRAME_BOOL_FIELDS = ("render_ok",)
FRAME_SLOT_FIELDS = ("targets_identifiable", "box_alignment_ok")
TASK_BOOL_FIELDS = ("task_possible",)
TASK_SLOT_FIELDS = ("label_matches", "referring_unambiguous")
OVERLAY_COLORS = {"A": "red", "B": "blue", "C": "green", "D": "yellow"}


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
def load_policy(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    policy = json.loads(Path(path or POLICY_PATH).read_text(encoding="utf-8"))
    if policy.get("review_schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("admission policy targets a different review schema")
    return policy


def _need(value: Any, n_slots: int) -> int:
    return n_slots if value == "all" else int(value)


# --------------------------------------------------------------------------
# Certificate -> SFT turns (shared with release.py)
# --------------------------------------------------------------------------
def chunk_certificate(actions: Sequence[str], strategy: str = "action_runs",
                      max_per_turn: int = 5) -> List[List[str]]:
    """Split certificate actions into assistant turns of <= max_per_turn actions.

    greedy: fill turns to the maximum; balanced: equal-size turns;
    action_runs: one turn per run of identical actions (e.g. rotate, then
    translate), split further only when a run exceeds the maximum;
    single_step: one action per turn.
    """
    actions = list(actions)
    if not actions:
        return []
    m = max(1, int(max_per_turn))
    if strategy == "greedy":
        return [actions[i:i + m] for i in range(0, len(actions), m)]
    if strategy == "balanced":
        n = int(math.ceil(len(actions) / m))
        sizes = [len(actions) // n + (1 if i < len(actions) % n else 0) for i in range(n)]
        out, i = [], 0
        for s in sizes:
            out.append(actions[i:i + s])
            i += s
        return out
    if strategy == "single_step":
        return [[a] for a in actions]
    if strategy == "action_runs":
        out: List[List[str]] = []
        for a in actions:
            if out and out[-1][-1] == a and len(out[-1]) < m:
                out[-1].append(a)
            else:
                out.append([a])
        return out
    raise ValueError(f"unknown chunking strategy {strategy!r}")


def turn_boundaries(task_row: Dict[str, Any], policy: Dict[str, Any]) -> List[int]:
    """State indices at which each SFT user turn shows a frame, plus the terminal index."""
    exp = policy["sft"]["export"]
    chunks = chunk_certificate(certificate(task_row), exp["chunking"], exp["max_actions_per_turn"])
    bounds = [0]
    for c in chunks:
        bounds.append(bounds[-1] + len(c))
    return bounds


def required_review_frames(task_row: Dict[str, Any], policy: Dict[str, Any],
                           qa_items: Iterable[Dict[str, Any]] = ()) -> Dict[str, List[str]]:
    d = len(certificate(task_row))
    roles: Dict[str, set] = {"s0": {"initial"}}
    roles.setdefault(f"s{d}", set()).add("terminal")
    for b in turn_boundaries(task_row, policy)[1:-1]:
        roles.setdefault(f"s{b}", set()).add("turn")
    if policy["rl"].get("intermediate_frames_required"):
        for k in range(1, d):
            roles.setdefault(f"s{k}", set()).add("intermediate")
    for q in qa_items:
        if q.get("parent_task_id") == task_row.get("task_id"):
            roles.setdefault(str(q["state_id"]), set()).add("qa")
    return {sid: sorted(r) for sid, r in sorted(roles.items(), key=lambda kv: _state_order(kv[0]))}


def _state_order(state_id: str) -> Tuple[int, str]:
    if state_id.startswith("s") and state_id[1:].isdigit():
        return int(state_id[1:]), ""
    head = state_id[1:].split("_", 1)[0]
    return (int(head) if head.isdigit() else 999), state_id


# --------------------------------------------------------------------------
# Review schema, prompt, validation, merge
# --------------------------------------------------------------------------
def review_template(task_row: Dict[str, Any], state_ids: Sequence[str]) -> Dict[str, Any]:
    slots = [s for s, _ in task_slots(task_row)]
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "task_id": task_row.get("task_id"),
        "reviewer": {"kind": "vlm", "name": None, "prompt_version": None},
        "task_level": {"label_matches": {s: None for s in slots},
                       "referring_unambiguous": {s: None for s in slots},
                       "task_possible": None, "notes": ""},
        "frames": {sid: {"render_ok": None, "targets_identifiable": {s: None for s in slots},
                         "box_alignment_ok": {s: None for s in slots}, "notes": ""} for sid in state_ids},
    }


def reviewer_prompt(task_row: Dict[str, Any], state_ids: Sequence[str], policy: Optional[Dict[str, Any]] = None) -> str:
    policy = policy or load_policy()
    slots = task_slots(task_row)
    targets = "\n".join(f"Target {s} ({OVERLAY_COLORS.get(s, 'white').upper()} outline): \"{ref_text(e)}\""
                        f" (category: {e.get('label')})" for s, e in slots)
    cats = ", ".join(f"{s}: {e.get('label')}" for s, e in slots)
    frames = "Frames, in order: " + ", ".join(state_ids)
    text = "\n".join(policy["reviewer_rubric"])
    schema = json.dumps(review_template(task_row, state_ids), sort_keys=False)
    for key, val in (("{targets}", targets), ("{goal}", str(task_row.get("task_description", ""))),
                     ("{categories}", cats), ("{schema}", schema)):
        text = text.replace(key, val)
    return text + "\n" + frames


def validate_review(review: Dict[str, Any], task_row: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    slots = {s for s, _ in task_slots(task_row)}
    if not isinstance(review, dict):
        return ["not_a_dict"]
    if review.get("review_schema_version") != REVIEW_SCHEMA_VERSION:
        problems.append("schema_version")
    if review.get("task_id") != task_row.get("task_id"):
        problems.append("task_id_mismatch")

    def ok_bool(v: Any) -> bool:
        return v is None or isinstance(v, bool)

    def ok_slots(d: Any) -> bool:
        return isinstance(d, dict) and set(d) <= slots and all(ok_bool(v) for v in d.values())

    tl = review.get("task_level")
    if tl is not None:
        if not isinstance(tl, dict):
            problems.append("task_level_type")
        else:
            for f in TASK_SLOT_FIELDS:
                if f in tl and not ok_slots(tl[f]):
                    problems.append(f"task_level.{f}")
            for f in TASK_BOOL_FIELDS:
                if not ok_bool(tl.get(f)):
                    problems.append(f"task_level.{f}")
    frames = review.get("frames", {})
    if not isinstance(frames, dict):
        return problems + ["frames_type"]
    for sid, fr in frames.items():
        if not isinstance(fr, dict):
            problems.append(f"frames.{sid}")
            continue
        for f in FRAME_BOOL_FIELDS:
            if not ok_bool(fr.get(f)):
                problems.append(f"frames.{sid}.{f}")
        for f in FRAME_SLOT_FIELDS:
            if f in fr and not ok_slots(fr[f]):
                problems.append(f"frames.{sid}.{f}")
    return problems


def _vote(values: Sequence[Optional[bool]], mode: str, tie: bool) -> Optional[bool]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if mode == "all":
        return all(vals)
    t, f = sum(vals), len(vals) - sum(vals)
    return True if t > f else False if f > t else bool(tie)


def merge_reviews(reviews: Sequence[Dict[str, Any]], policy: Dict[str, Any]) -> Dict[str, Any]:
    """Merge several reviews of one task: human overrides VLM; VLMs vote per field."""
    reviews = [r for r in reviews if isinstance(r, dict)]
    if len(reviews) == 1:
        return reviews[0]
    cfg = policy.get("review_merge", {})
    use_humans = bool(cfg.get("human_overrides_vlm", True))
    humans = [r for r in reviews if use_humans and (r.get("reviewer") or {}).get("kind") == "human"]
    machines = [r for r in reviews if not any(r is h for h in humans)]
    mode, tie = cfg.get("vlm_merge", "majority"), bool(cfg.get("tie", False))

    def combine(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {"review_schema_version": REVIEW_SCHEMA_VERSION, "task_id": group[0].get("task_id"),
               "reviewer": {"kind": "merged", "members": [r.get("reviewer") for r in group]},
               "task_level": {}, "frames": {}}
        tls = [r.get("task_level") or {} for r in group]
        if any(r.get("task_level") is not None for r in group):
            for f in TASK_BOOL_FIELDS:
                out["task_level"][f] = _vote([t.get(f) for t in tls], mode, tie)
            for f in TASK_SLOT_FIELDS:
                keys = sorted({k for t in tls for k in (t.get(f) or {})})
                out["task_level"][f] = {k: _vote([(t.get(f) or {}).get(k) for t in tls], mode, tie) for k in keys}
        else:
            out["task_level"] = None
        sids = sorted({s for r in group for s in (r.get("frames") or {})}, key=_state_order)
        for sid in sids:
            frs = [(r.get("frames") or {}).get(sid) for r in group]
            frs = [f for f in frs if isinstance(f, dict)]
            fr: Dict[str, Any] = {}
            for f in FRAME_BOOL_FIELDS:
                fr[f] = _vote([x.get(f) for x in frs], mode, tie)
            for f in FRAME_SLOT_FIELDS:
                keys = sorted({k for x in frs for k in (x.get(f) or {})})
                fr[f] = {k: _vote([(x.get(f) or {}).get(k) for x in frs], mode, tie) for k in keys}
            fr["notes"] = " | ".join(str(x.get("notes")) for x in frs if x.get("notes"))
            out["frames"][sid] = fr
        return out

    base = combine(machines) if machines else None
    if not humans:
        return base
    merged = copy.deepcopy(base) if base else {"review_schema_version": REVIEW_SCHEMA_VERSION,
                                                "task_id": humans[0].get("task_id"), "task_level": None,
                                                "frames": {}}
    merged["reviewer"] = {"kind": "human_over_vlm", "members": [r.get("reviewer") for r in reviews]}
    for h in humans:  # later human reviews win
        if h.get("task_level") is not None:
            tl = merged.get("task_level") or {}
            for f in TASK_BOOL_FIELDS:
                if h["task_level"].get(f) is not None:
                    tl[f] = h["task_level"][f]
            for f in TASK_SLOT_FIELDS:
                for k, v in (h["task_level"].get(f) or {}).items():
                    if v is not None:
                        tl.setdefault(f, {})[k] = v
            merged["task_level"] = tl
        for sid, fr in (h.get("frames") or {}).items():
            tgt = merged["frames"].setdefault(sid, {})
            for f in FRAME_BOOL_FIELDS:
                if fr.get(f) is not None:
                    tgt[f] = fr[f]
            for f in FRAME_SLOT_FIELDS:
                for k, v in (fr.get(f) or {}).items():
                    if v is not None:
                        tgt.setdefault(f, {})[k] = v
    return merged


def normalize_reviews(reviews: Any) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if reviews is None:
        return out
    if isinstance(reviews, dict):
        items: List[Any] = []
        for tid, val in reviews.items():
            items.extend(val if isinstance(val, list) else [val])
    else:
        items = list(reviews)
    for r in items:
        if isinstance(r, dict) and r.get("task_id") is not None:
            out.setdefault(str(r["task_id"]), []).append(r)
    return out


def normalize_renders(renders: Any) -> Optional[Dict[str, str]]:
    """``{"<task_id>:<state_id>": image_path}`` from a dict or a list of render records."""
    if renders is None:
        return None
    out: Dict[str, str] = {}
    items = renders.items() if isinstance(renders, dict) else (
        (f"{r['task_id']}:{r['state_id']}", r) for r in renders)
    for key, val in items:
        path = val.get("image_path") if isinstance(val, dict) else val
        if path:
            out[str(key)] = str(path)
    return out


# --------------------------------------------------------------------------
# Task validity
# --------------------------------------------------------------------------
def task_invalid_detail(review: Optional[Dict[str, Any]], task_row: Dict[str, Any],
                        policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reasons that invalidate the task definition (block every modality)."""
    if not review:
        return []
    slots = dict(task_slots(task_row))
    out: List[Dict[str, Any]] = []
    tl = review.get("task_level") or {}
    for slot, entry in slots.items():
        if (tl.get("label_matches") or {}).get(slot) is False:
            out.append({"reason": "task_invalid:wrong_object", "slot": slot, "object_id": entry.get("id")})
        if (tl.get("referring_unambiguous") or {}).get(slot) is False:
            out.append({"reason": "task_invalid:referring_ambiguous", "slot": slot, "object_id": entry.get("id")})
    if tl.get("task_possible") is False:
        out.append({"reason": "task_invalid:task_impossible", "slot": None, "object_id": None})
    rule = policy.get("box_misaligned_rule", {})
    terminal = f"s{len(certificate(task_row))}"
    frames = review.get("frames") or {}
    for slot, entry in slots.items():
        bad = [sid for sid, fr in frames.items() if ((fr or {}).get("box_alignment_ok") or {}).get(slot) is False]
        if (rule.get("terminal_frame", True) and terminal in bad) or len(bad) >= int(rule.get("min_frames", 2)):
            out.append({"reason": "task_invalid:box_misaligned", "slot": slot, "object_id": entry.get("id"),
                        "frames": sorted(bad, key=_state_order)})
    return out


# --------------------------------------------------------------------------
# Geometric pre-check (before any rendering)
# --------------------------------------------------------------------------
def precheck_task(task_row: Dict[str, Any], scene: Any = None, policy: Optional[Dict[str, Any]] = None,
                  funnel: Optional[Funnel] = None) -> Dict[str, Any]:
    """Cheap geometric plausibility check so that only plausible candidates are rendered.

    Identifiability uses ``occlusion.identifiable`` (workstream E) when it is
    importable and a scene is given; otherwise a projection-only check
    (cropping + size), recorded as ``backend``.
    """
    policy = policy or load_policy()
    cfg = policy["precheck"]
    funnel = funnel if funnel is not None else Funnel()

    def fail(reason: str, **details: Any) -> Dict[str, Any]:
        funnel.add("precheck", reason)
        return {"ok": False, "reason": reason, "details": details}

    missing = [k for k in TASK_ROW_REQUIRED_KEYS if k not in task_row]
    if missing:
        return fail("precheck:missing_keys", missing=missing)
    cert = certificate(task_row)
    if not cert:
        return fail("precheck:certificate_missing")
    if any(a not in ACTIONS for a in cert):
        return fail("precheck:certificate_unknown_action")
    if len(cert) > MAX_PRIMITIVE_STEPS:
        return fail("precheck:certificate_too_long")
    d_star = (task_row.get("difficulty") or {}).get("d_star")
    if d_star is not None and int(d_star) > len(cert):
        return fail("precheck:d_star_exceeds_certificate")
    if len(task_slots(task_row)) == 0:
        return fail("precheck:no_targets")
    scorer = score_backend(task_row)
    if scorer is None:
        return fail("precheck:scorer_unavailable")
    poses = certificate_poses(task_row)
    if scene is not None and cfg.get("check_pose_legality", True):
        ok, why = scene.is_pose_legal(poses[0])
        if not ok:
            return fail("precheck:initial_pose_illegal", why=why)
    if scene is not None and cfg.get("check_transitions", True):
        for i in range(len(cert)):
            ok, why = scene.transition_legal(poses[i], poses[i + 1])
            if not ok:
                return fail("precheck:certificate_illegal_transition", step=i, why=why)
    first = scorer(task_row, poses[0])
    if cfg.get("require_initial_not_successful", True) and bool(first.get("success")):
        return fail("precheck:initial_already_successful")
    last = scorer(task_row, poses[-1])
    if not bool(last.get("success")):
        return fail("precheck:certificate_not_successful")
    if cfg.get("reject_success_before_end", True):
        for i in range(1, len(poses) - 1):
            if bool(scorer(task_row, poses[i]).get("success")):
                return fail("precheck:success_before_end", step=i)
    n_slots = len(task_slots(task_row))
    kw = dict(min_inside=cfg["min_inside"], min_area_ratio=cfg["min_area_ratio"], min_visible=cfg["min_visible"])
    init_id = geometric_identifiable(task_row, poses[0], scene, **kw)
    term_id = geometric_identifiable(task_row, poses[-1], scene, **kw)
    backend = sorted({v["backend"] for v in list(init_id.values()) + list(term_id.values())})
    summary = {"initial_identifiable": {s: v["identifiable"] for s, v in init_id.items()},
               "terminal_identifiable": {s: v["identifiable"] for s, v in term_id.items()},
               "backend": backend}
    if sum(summary["initial_identifiable"].values()) < _need(cfg["initial_min_identifiable"], n_slots):
        return fail("precheck:initial_no_identifiable_target", **summary)
    if sum(summary["terminal_identifiable"].values()) < _need(cfg["terminal_min_identifiable"], n_slots):
        return fail("precheck:terminal_target_not_identifiable", **summary)
    funnel.add("precheck", "pass")
    return {"ok": True, "reason": "pass", "details": summary}


# --------------------------------------------------------------------------
# Render / review requests for the cluster GPU stage
# --------------------------------------------------------------------------
def overlay_boxes(task_row: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Dict[str, Any]]:
    proj = project_entries(task_row, c2w)
    out = {}
    for slot, entry in task_slots(task_row):
        p = proj[slot]
        out[slot] = {"bbox": p.get("bbox") if p.get("visible") else None, "color": OVERLAY_COLORS.get(slot, "white"),
                     "ref": ref_text(entry), "label": entry.get("label"), "image_size": list(RENDER_SIZE)}
    return out


def review_requests(task_rows: Iterable[Dict[str, Any]], qa_items: Iterable[Dict[str, Any]] = (),
                    policy: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Frames to render (render_paired_bank-compatible rows) and per-task review prompts."""
    policy = policy or load_policy()
    qa_items = list(qa_items)
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for q in qa_items:
        by_task.setdefault(str(q.get("parent_task_id")), []).append(q)
    frames, tasks = [], []
    for row in task_rows:
        tid = str(row["task_id"])
        poses = {f"s{k}": p for k, p in enumerate(certificate_poses(row))}
        for q in by_task.get(tid, []):
            poses.setdefault(str(q["state_id"]), np.asarray(q["state_pose_c2w"], dtype=float))
        roles = required_review_frames(row, policy, by_task.get(tid, []))
        for sid, rl in roles.items():
            pose = np.asarray(poses[sid], dtype=float)
            frames.append({
                "sample_id": f"{tid}:{sid}", "request_id": f"{tid}:{sid}", "task_id": tid,
                "scene_id": row.get("scene_id"), "state_id": sid, "roles": rl,
                "state_pose_c2w": pose.tolist(), "task_type": row.get("task_type"),
                "source_task_metric_version": row.get("canonical_task_metric_version"),
                "source_camera_model_version": row.get("camera_model_version"),
                "source_init_camera": {"intrinsics": (row.get("init_camera") or {}).get("intrinsics")},
                "camera": {"intrinsics": (row.get("init_camera") or {}).get("intrinsics"),
                           "resolution": list(RENDER_SIZE)},
                "private_answer": None,
                "overlay": overlay_boxes(row, pose),
            })
        tasks.append({"task_id": tid, "scene_id": row.get("scene_id"), "frames": list(roles),
                      "prompt": reviewer_prompt(row, list(roles), policy),
                      "review_template": review_template(row, list(roles))})
    return {"frames": frames, "tasks": tasks}


def draw_review_overlay(image_path: Union[str, Path], overlay: Dict[str, Dict[str, Any]],
                        out_path: Union[str, Path]) -> Path:
    """Draw the target outlines on a COPY of a rendered frame (review only, never training)."""
    from PIL import Image, ImageDraw

    with Image.open(image_path) as im:
        img = im.convert("RGB")
    draw = ImageDraw.Draw(img)
    for slot, spec in sorted(overlay.items()):
        box = spec.get("bbox")
        if not box:
            continue
        sw, sh = spec.get("image_size") or list(RENDER_SIZE)
        fx, fy = img.size[0] / float(sw), img.size[1] / float(sh)
        draw.rectangle([box[0] * fx, box[1] * fy, box[2] * fx, box[3] * fy], outline=spec.get("color", "white"),
                       width=max(1, int(round(2 * fx))))
        draw.text((box[0] * fx + 2, box[1] * fy + 2), slot, fill=spec.get("color", "white"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


# --------------------------------------------------------------------------
# Admission
# --------------------------------------------------------------------------
def _count_ident(fr: Dict[str, Any], slots: Sequence[str]) -> Tuple[int, int]:
    ids = fr.get("targets_identifiable") or {}
    return sum(ids.get(s) is True for s in slots), sum(ids.get(s) is None for s in slots)


def _decide(rejects: List[str], holds: List[str]) -> Tuple[str, List[str]]:
    rejects = list(dict.fromkeys(rejects))
    holds = list(dict.fromkeys(holds))
    if rejects:
        return REJECT, rejects + holds
    if holds:
        return HOLD, holds
    return ADMIT, []


def _episode_decision(row: Dict[str, Any], review: Optional[Dict[str, Any]], invalid: List[Dict[str, Any]],
                      policy: Dict[str, Any], modality: str, renders: Optional[Dict[str, str]],
                      replay: Optional[Dict[str, Any]], precheck: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    cfg = policy[modality]
    if invalid:
        return REJECT, sorted({d["reason"] for d in invalid})
    if precheck is not None and not precheck.get("ok", False):
        return REJECT, [str(precheck.get("reason"))]
    if review is None:
        return HOLD, ["review_missing"]
    if validate_review(review, row):
        return HOLD, ["review_invalid"]
    rejects: List[str] = []
    holds: List[str] = []
    if review.get("task_level") is None:
        holds.append("review_missing")
    slots = [s for s, _ in task_slots(row)]
    d = len(certificate(row))
    tid = str(row["task_id"])
    frames = review.get("frames") or {}
    checks = [("initial", "s0", _need(cfg["initial_min_identifiable"], len(slots)), "initial_no_identifiable_target"),
              ("terminal", f"s{d}", _need(cfg["terminal_min_identifiable"], len(slots)), "terminal_target_not_identifiable")]
    if modality == "rl" and cfg.get("intermediate_frames_required"):
        checks += [("intermediate", f"s{k}", _need(cfg["initial_min_identifiable"], len(slots)),
                    "intermediate_no_identifiable_target") for k in range(1, d)]
    if modality == "sft":
        checks += [("turn", f"s{b}", int(cfg.get("turn_frame_min_identifiable", 0)),
                    "sft_turn_frame_no_identifiable_target") for b in turn_boundaries(row, policy)[1:-1]]
    for role, sid, need, why in checks:
        fr = frames.get(sid)
        if fr is None:
            holds.append("sft_turn_frame_unreviewed" if role == "turn" else "review_missing")
            continue
        if role in cfg.get("render_ok_required", []):
            if fr.get("render_ok") is False:
                rejects.append("sft_render_artifact_turn_frame" if role == "turn" else f"render_artifact_{role}")
            elif fr.get("render_ok") is None:
                holds.append("review_uncertain")
        n_true, n_none = _count_ident(fr, slots)
        if n_true < need:
            holds.append("review_uncertain" if n_true + n_none >= need else why)
    if modality == "sft" and renders is not None:
        needed = [f"s{b}" for b in turn_boundaries(row, policy)]
        if any(f"{tid}:{sid}" not in renders for sid in needed):
            holds.append("sft_frame_missing_image")
    rep = (replay or {}).get(tid)
    if rep is not None and rep.get("success") is False:
        rejects.append("runtime_replay_failed")
    elif rep is None and cfg.get("require_runtime_replay"):
        holds.append("replay_missing")
    return _decide(rejects, holds)


def _qa_decision(q: Dict[str, Any], row: Optional[Dict[str, Any]], review: Optional[Dict[str, Any]],
                 invalid: List[Dict[str, Any]], policy: Dict[str, Any],
                 renders: Optional[Dict[str, str]]) -> Tuple[str, List[str]]:
    cfg = policy["qa"]
    if row is None:
        return REJECT, ["qa_parent_unknown"]
    if invalid and cfg.get("blocked_by_task_invalid", True):
        return REJECT, ["qa_parent_task_invalid"]
    if q.get("label_validity") != "valid" or q.get("private_answer") in (None, ""):
        return REJECT, ["qa_invalid_label"]
    if review is None:
        return HOLD, ["qa_review_missing"]
    if validate_review(review, row):
        return HOLD, ["review_invalid"]
    holds: List[str] = []
    rejects: List[str] = []
    if review.get("task_level") is None:
        holds.append("qa_parent_review_missing")
    fr = (review.get("frames") or {}).get(str(q["state_id"]))
    if fr is None:
        return _decide(rejects, holds + ["qa_review_missing"])
    if renders is not None and f"{q['parent_task_id']}:{q['state_id']}" not in renders:
        holds.append("qa_missing_image")
    if cfg.get("render_ok_required", True):
        if fr.get("render_ok") is False:
            rejects.append("qa_render_artifact")
        elif fr.get("render_ok") is None:
            holds.append("review_uncertain")
    required = list(q.get("requires_identifiable") or [])
    if cfg.get("referenced_targets_must_be_identifiable", True) and required:
        ids = fr.get("targets_identifiable") or {}
        vals = [ids.get(s) for s in required]
        if q.get("requires_identifiable_mode", "all") == "any":
            if not any(v is True for v in vals):
                if any(v is None for v in vals):
                    holds.append("review_uncertain")
                else:
                    rejects.append("qa_target_not_identifiable")
        else:
            if any(v is False for v in vals):
                rejects.append("qa_target_not_identifiable")
            elif any(v is None for v in vals):
                holds.append("review_uncertain")
    if cfg.get("box_alignment_required_for_referenced_targets", True):
        ids = fr.get("targets_identifiable") or {}
        box = fr.get("box_alignment_ok") or {}
        if any(box.get(s) is False and ids.get(s) is True for s in required):
            rejects.append("qa_box_misaligned_in_frame")
    return _decide(rejects, holds)


def admit(candidates: Iterable[Dict[str, Any]], reviews: Any, policy: Optional[Dict[str, Any]] = None, *,
          qa_items: Optional[Iterable[Dict[str, Any]]] = None, renders: Any = None,
          replay: Optional[Dict[str, Dict[str, Any]]] = None,
          prechecks: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Admit RL tasks, SFT trajectories and QA items, each by its own rule.

    Returns ``{"rl"|"sft"|"qa": {"admitted", "decisions", "status_counts",
    "reason_counts", "primary_reason_counts"}, "task_invalid": {...},
    "funnel": {...}}``.  ``renders`` maps ``"<task_id>:<state_id>"`` to an
    image path; when given, missing images HOLD the SFT/QA item.
    """
    policy = policy or load_policy()
    rows = {str(r["task_id"]): r for r in candidates}
    merged = {tid: merge_reviews(lst, policy) for tid, lst in normalize_reviews(reviews).items()}
    render_map = normalize_renders(renders)
    prechecks = prechecks or {}
    funnel = Funnel()
    result: Dict[str, Any] = {"policy_version": policy["policy_version"], "task_invalid": {}}
    invalid_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for tid, row in rows.items():
        inv = task_invalid_detail(merged.get(tid), row, policy)
        if inv:
            invalid_by_task[tid] = inv
            result["task_invalid"][tid] = inv
    for modality in ("rl", "sft"):
        decisions = {}
        for tid in sorted(rows):
            status, reasons = _episode_decision(rows[tid], merged.get(tid), invalid_by_task.get(tid, []), policy,
                                                modality, render_map, replay, prechecks.get(tid))
            decisions[tid] = {"status": status, "reasons": reasons, "scene_id": rows[tid].get("scene_id")}
            funnel.add(modality, status if status == ADMIT else f"{status}:{reasons[0]}")
        result[modality] = _summarize(decisions)
    qa_decisions = {}
    for q in qa_items or []:
        tid = str(q.get("parent_task_id"))
        status, reasons = _qa_decision(q, rows.get(tid), merged.get(tid), invalid_by_task.get(tid, []), policy,
                                       render_map)
        qa_decisions[str(q["sample_id"])] = {"status": status, "reasons": reasons, "parent_task_id": tid,
                                             "qa_kind": q.get("qa_kind"), "scene_id": q.get("scene_id")}
        funnel.add("qa", status if status == ADMIT else f"{status}:{reasons[0]}")
    result["qa"] = _summarize(qa_decisions)
    result["funnel"] = funnel.to_dict()
    return result


def _summarize(decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    reasons, primary, status = Counter(), Counter(), Counter()
    for d in decisions.values():
        status[d["status"]] += 1
        reasons.update(d["reasons"])
        if d["reasons"]:
            primary[d["reasons"][0]] += 1
    return {"admitted": sorted(k for k, d in decisions.items() if d["status"] == ADMIT),
            "decisions": decisions, "status_counts": dict(sorted(status.items())),
            "reason_counts": dict(sorted(reasons.items())), "primary_reason_counts": dict(sorted(primary.items()))}
