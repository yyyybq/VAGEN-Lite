"""QA states and geometric QA labels for InteriorGS v10 episodes (workstream A).

From one episode (initial pose + certificate actions + task row) this module
produces the camera states along the certificate path (optionally plus legal
off-path neighbours) and single-image QA items of three families:

  * Read    - what the current image shows (left/right order, frame cropping,
              which object is closer, is the goal already satisfied).
  * Predict - what the image will show after one primitive action
              (``contract.apply_action`` + re-projection).
  * Act     - which primitive actions bring the view closer to the goal
              (set of actions that strictly decrease d*, from a short BFS).

Every label is computed geometrically, never by a model:
  * object projections use the frozen H1 camera (``canonical_camera``) and
    ``visual_bbox_metrics._project_object`` - the same projection the
    canonical metric uses;
  * goal success uses ``sampling.score`` (v1 Projective / FOV) or, when the
    task row asks for ``canonical_spatial_task_h1_v2``, the feature-detected
    ``families.canonical_v2.score_canonical_task_v2``.
States whose quantity lies inside a numeric buffer around a decision threshold
are excluded and counted (``Funnel`` stage ``qa_gen``).

QA records are compatible with ``scripts/prepare_active_spatial_sft.py``
(``qa_record``/``bind_derived``) and ``data_gen/active_spatial_qa/render_paired_bank.py``.
They are emitted fail-closed (``qa_sft_allowed`` False, no image); release.py
stamps lineage, images and the review result after admission.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .contract import (
    ACTIONS,
    CANONICAL_TASK_METRIC_V1,
    CANONICAL_TASK_METRIC_V2,
    INVERSE_ACTION,
    RENDER_SIZE,
    STEP_ROTATION_DEG,
    STEP_TRANSLATION_M,
    Funnel,
    Obj,
    apply_action,
    state_key,
)

QA_SCHEMA_VERSION = "interiorgs_v10_qa_v1"
SLOT_NAMES = ("A", "B", "C", "D")
QA_FAMILIES = {
    "read_left_of": "read",
    "read_inside_frame": "read",
    "read_closer": "read",
    "read_goal": "read",
    "predict_visible_after": "predict",
    "predict_relation_after": "predict",
    "predict_shift": "predict",
    "act_yesno": "act",
    "act_choice": "act",
}

DEFAULT_QA_CONFIG: Dict[str, Any] = {
    "kinds": list(QA_FAMILIES),
    "identifiable_min_inside": 0.5,
    "identifiable_min_area_ratio": 0.002,
    "identifiable_min_visible": 0.5,
    "left_of_min_sep_px": 20.0,
    "inside_yes_min": 0.99,
    "inside_no_max": 0.85,
    "closer_min_rel_diff": 0.15,
    "closer_min_abs_diff_m": 0.3,
    "goal_inside_buffer": 0.08,
    "goal_margin_buffer_px": 6.0,
    "goal_center_margin_buffer_px": 4.0,
    "predict_visible_threshold": 0.5,
    "predict_visible_buffer": 0.1,
    "predict_shift_actions": ["move_forward", "move_backward", "turn_left", "turn_right"],
    "predict_shift_min_px": 3.0,
    "act_bfs_max_depth": 3,
    "include_offpath_neighbors": False,
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha(text: str, n: int = 20) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:n]


def merged_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_QA_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if not str(k).startswith("_")})
    return cfg


def task_slots(task_row: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    objects = ((task_row.get("target_object") or {}).get("objects")) or []
    return [(SLOT_NAMES[i], obj) for i, obj in enumerate(objects[: len(SLOT_NAMES)]) if isinstance(obj, dict)]


def ref_text(entry: Dict[str, Any]) -> str:
    """Referring expression for a target (``Obj.ref`` filled by referring.py)."""
    ref = entry.get("ref")
    if isinstance(ref, str) and ref.strip():
        return ref.strip()
    return f"the {str(entry.get('label') or 'object').strip()}"


def action_phrase(action: str) -> str:
    t = f"{STEP_TRANSLATION_M:g} meters"
    r = f"{STEP_ROTATION_DEG:g} degrees"
    return {
        "move_forward": f"moves forward by {t}",
        "move_backward": f"moves backward by {t}",
        "move_left": f"moves sideways to the left by {t}",
        "move_right": f"moves sideways to the right by {t}",
        "turn_left": f"turns left by {r}",
        "turn_right": f"turns right by {r}",
    }[action]


def lineage_fields(task_row: Dict[str, Any]) -> Dict[str, Any]:
    """Parent identity shared by RL, SFT and QA (split/hash final only after release)."""
    from vagen.envs.active_spatial.dataset_contract import row_digest

    split = task_row.get("split")
    return {
        "parent_task_id": task_row.get("task_id"),
        "scene_id": task_row.get("scene_id"),
        "split": split,
        "source_task_sha256": row_digest(task_row) if split else None,
        "source_versions": {k: task_row.get(k) for k in (
            "camera_model_version", "canonical_task_metric_version", "action_protocol_version")},
    }


# --------------------------------------------------------------------------
# Projection with the frozen H1 camera
# --------------------------------------------------------------------------
_K_CACHE: Dict[str, np.ndarray] = {}


def canonical_K(task_row: Dict[str, Any]) -> np.ndarray:
    from vagen.envs.active_spatial.canonical_camera import (
        CANONICAL_CAMERA_H1_RESIZE_V1,
        build_canonical_camera,
    )

    K_native = (task_row.get("init_camera") or {}).get("intrinsics")
    meta = {k: task_row[k] for k in ("native_resolution", "source_resolution", "camera_native_resolution",
                                     "native_width", "native_height") if k in task_row}
    key = json.dumps([K_native, meta], sort_keys=True, default=str)
    if key not in _K_CACHE:
        cam = build_canonical_camera(K_native=K_native, render_size=RENDER_SIZE, transform="resize",
                                     c2w=np.eye(4), item=meta,
                                     camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1)
        _K_CACHE[key] = np.asarray(cam.K_effective, dtype=float)
    return _K_CACHE[key]


def _box_area(box: Optional[Sequence[float]]) -> float:
    if not box:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def project_entries(task_row: Dict[str, Any], c2w: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Per-slot projection (canonical H1 camera, 256x256) plus derived quantities."""
    from vagen.envs.active_spatial.visual_bbox_metrics import _project_object

    c2w = np.asarray(c2w, dtype=float)
    K = canonical_K(task_row)
    W, H = RENDER_SIZE
    cam = c2w[:3, 3]
    out: Dict[str, Dict[str, Any]] = {}
    for slot, entry in task_slots(task_row):
        p = dict(_project_object(entry, c2w, K, W, H))
        raw = _box_area(p.get("bbox_raw"))
        p["inside_fraction"] = (_box_area(p.get("bbox")) / raw) if raw > 1e-8 else 0.0
        bmin = np.asarray(entry["bbox_min"], dtype=float)
        bmax = np.asarray(entry["bbox_max"], dtype=float)
        center = np.asarray(entry.get("center") or ((bmin + bmax) / 2.0), dtype=float)
        p["dist_center"] = float(np.linalg.norm(center - cam))
        p["dist_nearest"] = float(np.linalg.norm(np.clip(cam, bmin, bmax) - cam))
        bb = p.get("bbox")
        p["bbox_center_u"] = (float(bb[0]) + float(bb[2])) / 2.0 if bb else None
        out[slot] = p
    return out


# --------------------------------------------------------------------------
# Canonical success (v1 frozen backend, v2 feature-detected)
# --------------------------------------------------------------------------
def _load_v2_scorer() -> Optional[Callable[..., Dict[str, Any]]]:
    try:
        from .families.canonical_v2 import score_canonical_task_v2  # type: ignore
    except Exception:
        return None
    return score_canonical_task_v2


def score_backend(task_row: Dict[str, Any]) -> Optional[Callable[[Dict[str, Any], np.ndarray], Dict[str, Any]]]:
    version = task_row.get("canonical_task_metric_version")
    if version == CANONICAL_TASK_METRIC_V1:
        from . import sampling

        return sampling.score
    if version == CANONICAL_TASK_METRIC_V2:
        return _load_v2_scorer()
    return None


def score_task(task_row: Dict[str, Any], c2w: np.ndarray) -> Optional[Dict[str, Any]]:
    fn = score_backend(task_row)
    return None if fn is None else fn(task_row, np.asarray(c2w, dtype=float))


def goal_label_robust(task_row: Dict[str, Any], result: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (robust, how): is the success decision outside the numeric buffer?

    Success is robust when every numeric gate passes by at least its buffer;
    failure is robust when some gate fails by at least its buffer (or a
    boolean gate fails).  Everything else is a near-boundary state.
    """
    success = bool(result.get("success"))
    gates = result.get("gates") or {}
    task_type = task_row.get("task_type")
    numeric: List[Tuple[Optional[float], float, float]] = []  # (value, threshold, buffer)
    if task_type == "projective_relations" and "inside_frame_fraction_min" in result:
        numeric.append((result.get("inside_frame_fraction_min"), float(result.get("required_inside_frame_fraction", 0.5)),
                        float(cfg["goal_inside_buffer"])))
        numeric.append((result.get("relation_margin_px"), float(result.get("required_relation_margin_px", 12.0)),
                        float(cfg["goal_margin_buffer_px"])))
        boolean = [k for k in ("two_objects", "in_front", "visible", "min_area") if k in gates]
    elif task_type == "fov_inclusion" and "inside_frame_fraction_min" in result:
        numeric.append((result.get("inside_frame_fraction_min"), 0.95, min(float(cfg["goal_inside_buffer"]), 0.03)))
        numeric.append((result.get("center_margin_px"), float(result.get("required_center_margin_px") or 0.0),
                        float(cfg["goal_center_margin_buffer_px"])))
        boolean = [k for k in ("two_objects", "in_front", "visible", "min_area") if k in gates]
    else:
        # canonical_v2 families: honour an explicit margin report when present.
        if isinstance(result.get("near_boundary"), bool):
            return (not result["near_boundary"]), "v2_near_boundary_flag"
        margins = result.get("gate_margins")
        if isinstance(margins, dict) and margins:
            vals = [float(v) for v in margins.values() if v is not None]
            if success:
                return all(v >= 1.0 for v in vals), "v2_gate_margins"
            return any(v <= -1.0 for v in vals), "v2_gate_margins"
        return True, "unchecked_no_margin_report"
    if success:
        ok = all(v is not None and float(v) >= thr + buf for v, thr, buf in numeric)
        return ok, "all_gates_pass_with_buffer"
    if any(gates.get(k) is False for k in boolean):
        return True, "boolean_gate_fails"
    ok = any(v is None or float(v) < thr - buf for v, thr, buf in numeric)
    return ok, "a_gate_fails_with_buffer"


# --------------------------------------------------------------------------
# Identifiability (workstream E occlusion when importable)
# --------------------------------------------------------------------------
def _load_occlusion() -> Optional[Callable[..., Dict[str, Any]]]:
    try:
        from .occlusion import identifiable  # type: ignore
    except Exception:
        return None
    return identifiable


def _obj_for_entry(scene: Any, entry: Dict[str, Any]) -> Obj:
    oid = str(entry.get("id"))
    for obj in getattr(scene, "objects", []) or []:
        if str(obj.id) == oid:
            return obj
    return Obj(id=oid, label=str(entry.get("label", "")), aabb_min=np.asarray(entry["bbox_min"], dtype=float),
               aabb_max=np.asarray(entry["bbox_max"], dtype=float), ref=entry.get("ref"))


def geometric_identifiable(task_row: Dict[str, Any], c2w: np.ndarray, scene: Any = None, *,
                           min_inside: float = 0.5, min_area_ratio: float = 0.002, min_visible: float = 0.5,
                           projections: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """Per-slot geometric identifiability.

    Uses ``occlusion.identifiable`` (workstream E) when importable and a scene
    is given; otherwise a projection-only check (frame cropping + size, no
    occlusion), which is an upper bound of the occlusion-aware check.
    """
    fn = _load_occlusion() if scene is not None else None
    out: Dict[str, Dict[str, Any]] = {}
    proj = projections if projections is not None else project_entries(task_row, c2w)
    for slot, entry in task_slots(task_row):
        fallback_reason = "occlusion_unavailable" if fn is None else ""
        if fn is not None:
            try:
                res = fn(scene, np.asarray(c2w, dtype=float), _obj_for_entry(scene, entry), min_inside=min_inside,
                         min_area_ratio=min_area_ratio, min_visible=min_visible)
            except Exception as exc:  # fall back to projection, but record why
                fallback_reason = f"occlusion_error:{type(exc).__name__}"
            else:
                out[slot] = {"identifiable": bool(res.get("identifiable")), "backend": "occlusion",
                             "details": _jsonable({k: v for k, v in dict(res).items() if k != "identifiable"})}
                continue
        p = proj[slot]
        ok = bool(p.get("visible")) and float(p.get("inside_fraction", 0.0)) >= min_inside \
            and float(p.get("area_ratio", 0.0) or 0.0) >= min_area_ratio
        out[slot] = {"identifiable": ok, "backend": "projection_only",
                     "details": {"inside_fraction": float(p.get("inside_fraction", 0.0)),
                                 "area_ratio": float(p.get("area_ratio", 0.0) or 0.0),
                                 "fallback_reason": fallback_reason}}
    return out


# --------------------------------------------------------------------------
# States and BFS
# --------------------------------------------------------------------------
def certificate(task_row: Dict[str, Any]) -> List[str]:
    return list(((task_row.get("difficulty") or {}).get("certificate")) or [])


def certificate_poses(task_row: Dict[str, Any]) -> List[np.ndarray]:
    pose = np.asarray(task_row["init_camera"]["extrinsics"], dtype=float)
    poses = [pose]
    for a in certificate(task_row):
        pose = apply_action(pose, a)
        poses.append(pose)
    return poses


def scene_legal_fn(scene: Any) -> Optional[Callable[[np.ndarray, np.ndarray], bool]]:
    """Runtime transition rule (CollisionDetector with previous position)."""
    if scene is None:
        return None
    return lambda a, b: bool(scene.transition_legal(a, b)[0])


def build_states(task_row: Dict[str, Any], *, include_neighbors: bool = False,
                 legal_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None) -> List[Dict[str, Any]]:
    cert = certificate(task_row)
    poses = certificate_poses(task_row)
    states = [{"state_id": f"s{k}", "state_index": k, "source": "certificate_path",
               "prev_action": cert[k - 1] if k else None, "c2w": pose} for k, pose in enumerate(poses)]
    if include_neighbors:
        seen = {state_key(p) for p in poses}
        for k, pose in enumerate(poses):
            for a in ACTIONS:
                if (k < len(cert) and a == cert[k]) or (k > 0 and a == INVERSE_ACTION[cert[k - 1]]):
                    continue
                nb = apply_action(pose, a)
                key = state_key(nb)
                if key in seen or (legal_fn is not None and not legal_fn(pose, nb)):
                    continue
                seen.add(key)
                states.append({"state_id": f"n{k}_{a}", "state_index": k, "source": "offpath_neighbor",
                               "prev_action": a, "c2w": nb})
    return states


def optimal_first_actions(task_row: Dict[str, Any], c2w: np.ndarray, *, max_depth: int,
                          legal_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
                          success_fn: Optional[Callable[[np.ndarray], bool]] = None) -> Tuple[Optional[int], Set[str]]:
    """Exact d*(s) (if <= max_depth) and the set of first actions of shortest paths.

    An action strictly decreases d* iff it starts a shortest path, so the set
    returned is exactly the Act label.  Illegal moves leave the camera in
    place (runtime ``collision_invalidate_action``) and are never optimal.
    Returns (None, set()) when no success state is within ``max_depth``.
    """
    if success_fn is None:
        fn = score_backend(task_row)
        if fn is None:
            raise ValueError("no canonical scorer for this task row")
        success_fn = lambda pose: bool(fn(task_row, pose)["success"])  # noqa: E731
    start = np.asarray(c2w, dtype=float)
    if success_fn(start):
        return 0, set()
    visited = {state_key(start)}
    frontier: Dict[Tuple[int, int, int], Tuple[np.ndarray, Set[str]]] = {}
    for a in ACTIONS:
        nxt = apply_action(start, a)
        if legal_fn is not None and not legal_fn(start, nxt):
            continue
        key = state_key(nxt)
        if key in visited:
            continue
        if key in frontier:
            frontier[key][1].add(a)
        else:
            frontier[key] = (nxt, {a})
    for depth in range(1, max_depth + 1):
        winners: Set[str] = set()
        for key, (pose, firsts) in frontier.items():
            if success_fn(pose):
                winners |= firsts
        if winners:
            return depth, winners
        if depth == max_depth:
            break
        visited |= set(frontier)
        nxt_frontier: Dict[Tuple[int, int, int], Tuple[np.ndarray, Set[str]]] = {}
        for key, (pose, firsts) in frontier.items():
            for a in ACTIONS:
                nxt = apply_action(pose, a)
                if legal_fn is not None and not legal_fn(pose, nxt):
                    continue
                k2 = state_key(nxt)
                if k2 in visited:
                    continue
                if k2 in nxt_frontier:
                    nxt_frontier[k2][1].update(firsts)
                else:
                    nxt_frontier[k2] = (nxt, set(firsts))
        frontier = nxt_frontier
    return None, set()


# --------------------------------------------------------------------------
# QA generation
# --------------------------------------------------------------------------
class _Alternator:
    """Deterministic Yes/No target alternation per (task, kind)."""

    def __init__(self, seed: str) -> None:
        self.seed = seed
        self.count: Dict[str, int] = {}

    def want_yes(self, kind: str) -> bool:
        n = self.count.get(kind, 0)
        self.count[kind] = n + 1
        start = int(hashlib.sha1(f"{self.seed}:{kind}".encode()).hexdigest()[0], 16) & 1
        return (n + start) % 2 == 0


def _yn(flag: bool) -> str:
    return "Yes" if flag else "No"


def generate_qa(task_row: Dict[str, Any], *, scene: Any = None, config: Optional[Dict[str, Any]] = None,
                funnel: Optional[Funnel] = None,
                legal_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
                states: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """QA items (Read / Predict / Act) for every state of one episode."""
    cfg = merged_config(config)
    funnel = funnel if funnel is not None else Funnel()
    legal = legal_fn if legal_fn is not None else scene_legal_fn(scene)
    slots = task_slots(task_row)
    refs = {slot: ref_text(entry) for slot, entry in slots}
    if not slots:
        funnel.add("qa_gen", "no_target_objects")
        return []
    if states is None:
        states = build_states(task_row, include_neighbors=bool(cfg["include_offpath_neighbors"]), legal_fn=legal)
    scorer = score_backend(task_row)
    if scorer is None:
        funnel.add("qa_gen", "scorer_unavailable_goal_and_act_skipped")
    kinds = [k for k in cfg["kinds"] if k in QA_FAMILIES]
    alt = _Alternator(str(task_row.get("task_id")))
    desc = str(task_row.get("task_description") or "").strip()
    d_star = (task_row.get("difficulty") or {}).get("d_star")
    lower_complete = bool((task_row.get("difficulty") or {}).get("lower_bound_complete"))
    base = _base_record(task_row)
    out: List[Dict[str, Any]] = []

    for st in states:
        c2w = np.asarray(st["c2w"], dtype=float)
        proj = project_entries(task_row, c2w)
        ident = {s: v["identifiable"] for s, v in geometric_identifiable(
            task_row, c2w, scene, min_inside=cfg["identifiable_min_inside"],
            min_area_ratio=cfg["identifiable_min_area_ratio"], min_visible=cfg["identifiable_min_visible"],
            projections=proj).items()}
        ctx = {"st": st, "c2w": c2w, "proj": proj, "ident": ident}
        result = scorer(task_row, c2w) if scorer is not None and ("read_goal" in kinds or "act_yesno" in kinds
                                                                   or "act_choice" in kinds) else None
        post_cache: Dict[str, Optional[Dict[str, Dict[str, Any]]]] = {}

        def post(action: str) -> Optional[Dict[str, Dict[str, Any]]]:
            if action not in post_cache:
                nxt = apply_action(c2w, action)
                post_cache[action] = None if (legal is not None and not legal(c2w, nxt)) else project_entries(task_row, nxt)
            return post_cache[action]

        for kind in kinds:
            item = None
            if kind == "read_left_of":
                item = _read_left_of(ctx, refs, cfg, alt, funnel)
            elif kind == "read_inside_frame":
                item = _read_inside_frame(ctx, refs, cfg, alt, funnel)
            elif kind == "read_closer":
                item = _read_closer(ctx, refs, cfg, alt, funnel)
            elif kind == "read_goal":
                item = _read_goal(ctx, task_row, desc, result, cfg, funnel)
            elif kind == "predict_visible_after":
                item = _predict_visible(ctx, refs, cfg, alt, funnel, post)
            elif kind == "predict_relation_after":
                item = _predict_relation(ctx, refs, cfg, alt, funnel, post)
            elif kind == "predict_shift":
                item = _predict_shift(ctx, refs, cfg, alt, funnel, post)
            elif kind in ("act_yesno", "act_choice"):
                item = _act(kind, ctx, task_row, desc, result, cfg, alt, funnel, legal, d_star, lower_complete)
            if item is None:
                continue
            out.append(_finish(base, task_row, st, kind, item))
            funnel.add("qa_gen", f"emitted:{kind}")
    return out


def _base_record(task_row: Dict[str, Any]) -> Dict[str, Any]:
    rec = {"qa_schema_version": QA_SCHEMA_VERSION}
    rec.update(lineage_fields(task_row))
    rec.update({
        "task_type": task_row.get("task_type"),
        "source_task_metric_version": task_row.get("canonical_task_metric_version"),
        "source_camera_model_version": task_row.get("camera_model_version"),
        "source_init_camera": {"intrinsics": (task_row.get("init_camera") or {}).get("intrinsics")},
        "camera": {"intrinsics": (task_row.get("init_camera") or {}).get("intrinsics"),
                   "resolution": list(RENDER_SIZE)},
        "target_slots": {slot: {"id": e.get("id"), "label": e.get("label"), "ref": ref_text(e)}
                         for slot, e in task_slots(task_row)},
    })
    return rec


def _finish(base: Dict[str, Any], task_row: Dict[str, Any], st: Dict[str, Any], kind: str,
            item: Dict[str, Any]) -> Dict[str, Any]:
    rec = copy.deepcopy(base)
    variant = item.pop("variant")
    rec.update({
        "sample_id": _sha(f"{task_row.get('task_id')}:{st['state_id']}:{kind}:{variant}"),
        "qa_family": QA_FAMILIES[kind],
        "qa_kind": kind,
        "state_id": st["state_id"],
        "state_index": int(st["state_index"]),
        "state_source": st["source"],
        "state_pose_c2w": np.asarray(st["c2w"], dtype=float).tolist(),
        "public_observation": {"image_path": None, "state_id": st["state_id"], "observation_config": "single_image",
                               "history": [], "camera_pose_available": False},
        "label_validity": "valid",
        "observability_validity": "pending_render",
        "render_status": "not_requested",
        "qa_sft_allowed": False,  # set only by release.py after admission
    })
    rec["answer_format"] = item.pop("answer_format", "yes_no")
    rec.update(item)
    rec["_audit"] = _jsonable(rec.get("_audit", {}))
    return rec


def _yn_question(text: str) -> str:
    return text + " Answer Yes or No."


def _pair(ident: Dict[str, bool]) -> Optional[Tuple[str, str]]:
    if "A" in ident and "B" in ident:
        return "A", "B"
    return None


def _read_left_of(ctx, refs, cfg, alt, funnel):
    pair = _pair(ctx["ident"])
    if pair is None:
        funnel.add("qa_gen", "read_left_of:single_target_task")
        return None
    if not all(ctx["ident"][s] for s in pair):
        funnel.add("qa_gen", "read_left_of:not_identifiable_geometric")
        return None
    pa, pb = ctx["proj"]["A"], ctx["proj"]["B"]
    if pa.get("center_uv") is None or pb.get("center_uv") is None:
        funnel.add("qa_gen", "read_left_of:center_behind_camera")
        return None
    du = float(pa["center_uv"][0]) - float(pb["center_uv"][0])
    if abs(du) < float(cfg["left_of_min_sep_px"]):
        funnel.add("qa_gen", "near_boundary_separation")
        return None
    dbox = (pa.get("bbox_center_u") or 0.0) - (pb.get("bbox_center_u") or 0.0)
    if dbox * du <= 0:
        funnel.add("qa_gen", "read_left_of:box_and_center_order_disagree")
        return None
    a_left = du < 0
    want = alt.want_yes("read_left_of")
    x, y = ("A", "B") if (a_left == want) else ("B", "A")
    answer = (x == "A") == a_left
    return {"variant": f"{x}{y}", "question": _yn_question(f"In this image, is {refs[x]} to the left of {refs[y]}?"),
            "private_answer": _yn(answer), "requires_identifiable": [x, y], "requires_identifiable_mode": "all",
            "label_source": "canonical_h1_projection", "_audit": {"du_px": du}}


def _read_inside_frame(ctx, refs, cfg, alt, funnel):
    options = []
    for slot, ok in ctx["ident"].items():
        if not ok:
            continue
        f = float(ctx["proj"][slot].get("inside_fraction", 0.0))
        if f >= float(cfg["inside_yes_min"]):
            options.append((slot, True, f))
        elif f <= float(cfg["inside_no_max"]):
            options.append((slot, False, f))
        else:
            funnel.add("qa_gen", "near_boundary_inside")
    if not options:
        funnel.add("qa_gen", "read_inside_frame:no_option")
        return None
    want = alt.want_yes("read_inside_frame")
    match = [o for o in options if o[1] == want] or options
    slot, label, f = match[0]
    return {"variant": slot,
            "question": _yn_question(f"Is {refs[slot]} entirely inside the image, without being cut off by the image border?"),
            "private_answer": _yn(label), "requires_identifiable": [slot], "requires_identifiable_mode": "all",
            "label_source": "canonical_h1_projection", "_audit": {"inside_fraction": f}}


def _read_closer(ctx, refs, cfg, alt, funnel):
    pair = _pair(ctx["ident"])
    if pair is None:
        funnel.add("qa_gen", "read_closer:single_target_task")
        return None
    if not all(ctx["ident"][s] for s in pair):
        funnel.add("qa_gen", "read_closer:not_identifiable_geometric")
        return None
    pa, pb = ctx["proj"]["A"], ctx["proj"]["B"]
    diffs = []
    for key in ("dist_center", "dist_nearest"):
        da, db = float(pa[key]), float(pb[key])
        diffs.append((da - db, abs(da - db) / max(da, db, 1e-6)))
    signs = {d > 0 for d, _ in diffs}
    if len(signs) != 1 or any(abs(d) < float(cfg["closer_min_abs_diff_m"]) or r < float(cfg["closer_min_rel_diff"])
                              for d, r in diffs):
        funnel.add("qa_gen", "near_boundary_depth")
        return None
    a_closer = diffs[0][0] < 0
    want = alt.want_yes("read_closer")
    x, y = ("A", "B") if (a_closer == want) else ("B", "A")
    answer = (x == "A") == a_closer
    return {"variant": f"{x}{y}", "question": _yn_question(f"Is {refs[x]} closer to the camera than {refs[y]}?"),
            "private_answer": _yn(answer), "requires_identifiable": [x, y], "requires_identifiable_mode": "all",
            "label_source": "geometric_distance", "_audit": {"diffs_m": [d for d, _ in diffs]}}


def _read_goal(ctx, task_row, desc, result, cfg, funnel):
    if result is None:
        funnel.add("qa_gen", "read_goal:scorer_unavailable")
        return None
    if not any(ctx["ident"].values()):
        funnel.add("qa_gen", "read_goal:no_identifiable_target")
        return None
    robust, how = goal_label_robust(task_row, result, cfg)
    if not robust:
        funnel.add("qa_gen", "near_boundary_goal")
        return None
    return {"variant": "goal", "question": _yn_question(f"Goal: {desc} Is this goal already satisfied in the current view?"),
            "private_answer": _yn(bool(result.get("success"))),
            "requires_identifiable": sorted(ctx["ident"]), "requires_identifiable_mode": "any",
            "label_source": str(task_row.get("canonical_task_metric_version")),
            "_audit": {"robust": how, "gates": result.get("gates")}}


def _visible_fraction(p: Dict[str, Any]) -> float:
    if not p.get("visible") or not p.get("center_in_front", True):
        return 0.0
    return float(p.get("inside_fraction", 0.0))


def _predict_visible(ctx, refs, cfg, alt, funnel, post):
    thr, buf = float(cfg["predict_visible_threshold"]), float(cfg["predict_visible_buffer"])
    options = []
    order = sorted(ACTIONS, key=lambda a: (a.startswith("move_f") or a.startswith("move_b"), a))
    for slot, ok in ctx["ident"].items():
        if not ok:
            continue
        for a in order:
            pp = post(a)
            if pp is None:
                continue
            f = _visible_fraction(pp[slot])
            if f >= thr + buf:
                options.append((slot, a, True, f))
            elif f <= thr - buf:
                options.append((slot, a, False, f))
    if not options:
        funnel.add("qa_gen", "predict_visible_after:no_option_or_near_boundary")
        return None
    want = alt.want_yes("predict_visible_after")
    match = [o for o in options if o[2] == want] or options
    slot, a, label, f = match[ctx["st"]["state_index"] % len(match)]
    return {"variant": f"{slot}:{a}",
            "question": _yn_question(f"If the camera {action_phrase(a)}, will at least half of {refs[slot]} be inside the image?"),
            "private_answer": _yn(label), "requires_identifiable": [slot], "requires_identifiable_mode": "all",
            "label_source": "apply_action_reprojection", "_audit": {"action": a, "inside_after": f}}


def _predict_relation(ctx, refs, cfg, alt, funnel, post):
    pair = _pair(ctx["ident"])
    if pair is None:
        funnel.add("qa_gen", "predict_relation_after:single_target_task")
        return None
    if not all(ctx["ident"][s] for s in pair):
        funnel.add("qa_gen", "predict_relation_after:not_identifiable_geometric")
        return None
    sep = float(cfg["left_of_min_sep_px"])
    pa, pb = ctx["proj"]["A"], ctx["proj"]["B"]
    du_pre = None
    if pa.get("center_uv") is not None and pb.get("center_uv") is not None:
        du_pre = float(pa["center_uv"][0]) - float(pb["center_uv"][0])
    flips, keeps = [], []
    for a in ACTIONS:
        pp = post(a)
        if pp is None:
            continue
        qa_, qb_ = pp["A"], pp["B"]
        if _visible_fraction(qa_) < 0.5 or _visible_fraction(qb_) < 0.5 or qa_.get("center_uv") is None \
                or qb_.get("center_uv") is None:
            continue
        du = float(qa_["center_uv"][0]) - float(qb_["center_uv"][0])
        if abs(du) < sep:
            continue
        flip = du_pre is not None and abs(du_pre) >= sep and (du_pre < 0) != (du < 0)
        (flips if flip else keeps).append((a, du, flip))
    options = flips or keeps
    if not options:
        funnel.add("qa_gen", "predict_relation_after:no_option_or_near_boundary")
        return None
    a, du, flip = options[ctx["st"]["state_index"] % len(options)]
    a_left = du < 0
    want = alt.want_yes("predict_relation_after")
    x, y = ("A", "B") if (a_left == want) else ("B", "A")
    answer = (x == "A") == a_left
    return {"variant": f"{x}{y}:{a}",
            "question": _yn_question(f"If the camera {action_phrase(a)}, will {refs[x]} appear to the left of {refs[y]}?"),
            "private_answer": _yn(answer), "requires_identifiable": [x, y], "requires_identifiable_mode": "all",
            "label_source": "apply_action_reprojection", "_audit": {"action": a, "du_after_px": du, "relation_flips": flip}}


def _predict_shift(ctx, refs, cfg, alt, funnel, post):
    actions = [a for a in cfg["predict_shift_actions"] if a in ACTIONS]
    if not actions:
        return None
    k = ctx["st"]["state_index"]
    actions = actions[k % len(actions):] + actions[: k % len(actions)]
    want = alt.want_yes("predict_shift")
    for slot, ok in sorted(ctx["ident"].items()):
        if not ok or ctx["proj"][slot].get("center_uv") is None:
            continue
        u0 = float(ctx["proj"][slot]["center_uv"][0])
        for a in actions:
            pp = post(a)
            if pp is None or pp[slot].get("center_uv") is None:
                continue
            du = float(pp[slot]["center_uv"][0]) - u0
            if abs(du) < float(cfg["predict_shift_min_px"]):
                funnel.add("qa_gen", "near_boundary_shift")
                continue
            toward_left = du < 0
            edge = "left" if toward_left == want else "right"
            answer = (edge == "left") == toward_left
            return {"variant": f"{slot}:{a}:{edge}",
                    "question": _yn_question(f"If the camera {action_phrase(a)}, will {refs[slot]} move toward the {edge} edge of the image?"),
                    "private_answer": _yn(answer), "requires_identifiable": [slot], "requires_identifiable_mode": "all",
                    "label_source": "apply_action_reprojection",
                    "_audit": {"action": a, "du_px": du,
                               "image_dependent": a in ("move_forward", "move_backward")}}
    funnel.add("qa_gen", "predict_shift:no_option")
    return None


def _act(kind, ctx, task_row, desc, result, cfg, alt, funnel, legal, d_star, lower_complete):
    if result is None:
        funnel.add("qa_gen", f"{kind}:scorer_unavailable")
        return None
    if not any(ctx["ident"].values()):
        funnel.add("qa_gen", f"{kind}:no_identifiable_target")
        return None
    if bool(result.get("success")):
        funnel.add("qa_gen", f"{kind}:already_successful")
        return None
    max_depth = int(cfg["act_bfs_max_depth"])
    st = ctx["st"]
    if st["source"] == "certificate_path" and lower_complete and d_star is not None \
            and int(d_star) - int(st["state_index"]) > max_depth:
        funnel.add("qa_gen", "act_bfs_depth_exceeded")
        return None
    cache = ctx.setdefault("_act_cache", {})
    if "bfs" not in cache:
        cache["bfs"] = optimal_first_actions(task_row, ctx["c2w"], max_depth=max_depth, legal_fn=legal)
    depth, best = cache["bfs"]
    if depth is None or not best:
        funnel.add("qa_gen", "act_bfs_depth_exceeded")
        return None
    goal = f"Goal: {desc}"
    if kind == "act_choice":
        answer = ", ".join(a for a in ACTIONS if a in best)
        return {"variant": "choice", "answer_format": "action_set", "multi_answer": True,
                "question": (f"{goal} Which of these single actions would bring the view closer to satisfying the goal: "
                             f"{', '.join(ACTIONS)}? List every action that helps, separated by commas."),
                "private_answer": answer, "accepted_answers": [a for a in ACTIONS if a in best],
                "requires_identifiable": sorted(ctx["ident"]), "requires_identifiable_mode": "any",
                "label_source": "bfs_canonical_success", "_audit": {"d_star_state": depth}}
    legal_actions = [a for a in ACTIONS if legal is None or legal(ctx["c2w"], apply_action(ctx["c2w"], a))]
    yes = [a for a in legal_actions if a in best]
    no = [a for a in legal_actions if a not in best]
    want = alt.want_yes("act_yesno")
    pool = (yes if want else no) or (no if want else yes)
    if not pool:
        funnel.add("qa_gen", "act_yesno:no_option")
        return None
    a = pool[st["state_index"] % len(pool)]
    return {"variant": a, "question": _yn_question(
                f"{goal} Would the action {a} (the camera {action_phrase(a)}) bring the view closer to satisfying the goal?"),
            "private_answer": _yn(a in best), "requires_identifiable": sorted(ctx["ident"]),
            "requires_identifiable_mode": "any", "label_source": "bfs_canonical_success",
            "_audit": {"d_star_state": depth, "optimal_actions": sorted(best)}}


# --------------------------------------------------------------------------
# Balance
# --------------------------------------------------------------------------
def balance_yes_no(records: Iterable[Dict[str, Any]], *, seed: str = "interiorgs_v10",
                   funnel: Optional[Funnel] = None, max_majority_ratio: float = 1.0) -> List[Dict[str, Any]]:
    """Down-sample the majority label per (qa_kind, split); action-set items pass through."""
    records = list(records)
    groups: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
    passthrough = []
    for r in records:
        if r.get("answer_format", "yes_no") != "yes_no":
            passthrough.append(r)
            continue
        groups.setdefault((str(r.get("qa_kind")), str(r.get("split"))), {"Yes": [], "No": []})[r["private_answer"]].append(r)
    kept = list(passthrough)
    for (kind, split), by in sorted(groups.items()):
        n_min = min(len(by["Yes"]), len(by["No"]))
        cap = max(n_min, int(math.floor(n_min * float(max_majority_ratio))))
        for label in ("Yes", "No"):
            rows = sorted(by[label], key=lambda r: hashlib.sha1(f"{seed}:{r['sample_id']}".encode()).hexdigest())
            limit = min(len(rows), cap)
            kept.extend(rows[:limit])
            if funnel is not None and len(rows) > limit:
                funnel.add("qa_balance", f"downsampled:{kind}:{label}", len(rows) - limit)
    order = {id(r): i for i, r in enumerate(records)}
    kept.sort(key=lambda r: order[id(r)])
    return kept
