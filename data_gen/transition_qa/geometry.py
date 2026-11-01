"""Per-pose measurements of a task's objects, from the env's and the generator's own functions.

Nothing here re-implements geometry (table in docs/transition_qa/PROGRESS.md, section 1.3):

* success           - the body of ``target_visibility.score_with_visibility`` (canonical metric + visibility
                      rule + 45-degree rule), split so that the targets' visibility is computed once;
* clearly visible   - ``target_visibility.object_visibility`` (rule face_visible_v3);
* projection        - ``canonical_task_metrics_v2._Context.project`` (the judges' 256 x 256 camera);
* distance          - ``canonical_task_metrics_v2.footprint_distance`` (the absolute / equidistance judges');
* left / right      - image u of the projected centres (the projective judge's margin) and
                      ``target_visibility.lateral_gate``;
* hidden fraction   - ``canonical_task_metrics_v2.occlusion_silhouette`` (the occlusion judge's);
* poses / legality  - ``contract.apply_action`` and ``scene.Scene.transition_legal`` (the env's collision rule).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vagen.envs.active_spatial import canonical_task_metrics_v2 as M2
from vagen.envs.active_spatial import target_visibility as TV
from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task

from data_gen.interiorgs_v10.contract import ACTIONS, apply_action

ROLE_LETTERS = ("a", "b")
N_ROLES = {
    "projective_relations": 2,
    "equidistance": 2,
    "occlusion_alignment": 2,   # a = the target to hide, b = the occluder
    "fov_inclusion": 2,
    "absolute_positioning": 1,
    "screen_occupancy": 1,
}
FAMILIES = tuple(N_ROLES)


class ConsistencyError(RuntimeError):
    """The success decision and the potential disagree (the user asked to stop on this)."""


class TaskGeometry:
    """One task row in one loaded scene: measurements at any camera pose."""

    def __init__(self, item: Dict[str, Any], scene: Any, occ: TV.OccluderSet) -> None:
        self.item = item
        self.scene = scene
        self.occ = occ
        self.family = str(item["task_type"])
        if self.family not in N_ROLES:
            raise ValueError(f"unsupported task_type {self.family!r}")
        self.K, self.W, self.H = TV.item_camera(item)
        roles = M2._Context(item, np.eye(4)).roles(ROLE_LETTERS[: N_ROLES[self.family]])
        if roles is None:
            raise ValueError(f"task {item.get('task_id')}: target objects not found")
        self.roles: Dict[str, Dict[str, Any]] = dict(zip(ROLE_LETTERS, roles))

    # ------------------------------------------------------------------ success
    def success(self, c2w: np.ndarray, vis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """``TV.score_with_visibility(item, c2w, occ)``, reusing a ``TV.visibility_gates`` result if given."""
        c2w = np.asarray(c2w, float)
        if vis is None:
            vis = TV.visibility_gates(self.item, c2w, self.occ)
        metric = TV.apply_visibility(score_canonical_task(self.item, c2w), vis, TV.lateral_gate(self.item, c2w))
        if bool(metric["success"]) != (float(metric["shaping_score"]) >= 0.5):
            raise ConsistencyError(f"task {self.item.get('task_id')}: success {metric['success']} but potential "
                                   f"{metric['shaping_score']:.6f}")
        return metric

    def success_robust(self, c2w: np.ndarray, perturb_m: float, perturb_deg: float) -> Tuple[bool, bool]:
        """(success, robust): robust when the decision is the same after moving ``perturb_m`` along each of the
        camera's four horizontal directions and turning ``perturb_deg`` either way."""
        c2w = np.asarray(c2w, float)
        base = bool(self.success(c2w)["success"])
        for action in ACTIONS:
            if action.startswith("turn"):
                pose = apply_action(c2w, action, step_r_deg=perturb_deg)
            else:
                pose = apply_action(c2w, action, step_t=perturb_m)
            if bool(self.success(pose)["success"]) != base:
                return base, False
        return base, True

    # ------------------------------------------------------------- measurements
    def measure(self, c2w: np.ndarray, with_success: bool = True) -> Dict[str, Any]:
        """Every quantity the questions ask about, at one pose."""
        c2w = np.asarray(c2w, float)
        ctx = M2._Context(self.item, c2w)
        vis = TV.visibility_gates(self.item, c2w, self.occ)
        x, y = float(c2w[0, 3]), float(c2w[1, 3])
        objects: Dict[str, Dict[str, Any]] = {}
        for letter, obj in self.roles.items():
            p = ctx.project(obj)
            if letter in vis["targets"]:
                face = vis["targets"][letter]
            else:  # not a required target (the hidden target of occlusion_alignment): same function
                face = TV.object_visibility(c2w, obj, self.occ, self.K, self.W, self.H)
            centre = p.get("center_uv")
            objects[letter] = {
                "center_in_front": bool(p.get("center_in_front")),
                "u": float(centre[0]) if centre is not None else None,
                "all_corners_in_front": bool(p.get("all_corners_in_front")),
                "projected_visible": bool(p.get("visible")),
                "inside_fraction": float(p.get("inside_fraction", 0.0)),
                "vertical_inside_fraction": float(p.get("vertical_inside_fraction", 0.0)),
                "occupancy": float(p.get("height_px", 0.0) or 0.0) / float(self.H),
                "distance_m": float(M2.footprint_distance(x, y, obj)),
                "clearly_visible": bool(face["visible"]),
                "face_visible_fraction": float(face["visible_fraction"]),
                "face_area_fraction": float(face["area_fraction"]),
            }
        out: Dict[str, Any] = {"objects": objects}
        if self.family == "projective_relations":
            lat = TV.lateral_gate(self.item, c2w)
            out["lateral_ok"] = bool(lat["ok"])
            out["lateral_angle_deg"] = lat.get("angle_deg")
        if self.family == "occlusion_alignment":
            sil = M2.occlusion_silhouette(ctx, self.roles["a"], self.roles["b"])
            out["hidden_fraction"] = float(sil["hidden_fraction"])
            out["in_frame_fraction"] = float(sil["in_frame_fraction"])
        if with_success:
            metric = self.success(c2w, vis)
            out["success"] = bool(metric["success"])
            out["potential"] = float(metric["shaping_score"])
            out["failed_gates"] = sorted(k for k, v in (metric.get("gates") or {}).items() if not v)
        return out

    # ------------------------------------------------------------------ actions
    def action_poses(self, c2w: np.ndarray, action: str, k: int) -> Optional[List[np.ndarray]]:
        """Poses after 1..k repeats of ``action``, or None if a move is blocked (the env's collision rule)."""
        poses = []
        pose = np.asarray(c2w, float)
        for _ in range(int(k)):
            nxt = apply_action(pose, action)
            if action.startswith("move"):
                ok, _ = self.scene.transition_legal(pose, nxt)
                if not ok:
                    return None
            poses.append(nxt)
            pose = nxt
        return poses


def poses_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))


def repeated_action_pose(c2w: np.ndarray, action: str, k: int) -> np.ndarray:
    pose = np.asarray(c2w, float)
    for _ in range(int(k)):
        pose = apply_action(pose, action)
    return pose


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return round(v, 6) if np.isfinite(v) else None
    return value


def roles_for(family: str) -> Sequence[str]:
    return ROLE_LETTERS[: N_ROLES[family]]
