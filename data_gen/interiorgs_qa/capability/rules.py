"""Environment measurements and conservative capability-question drafts.

This module never certifies RGB visibility, rendered depth, reachability or a
human review. Drafts need the production pipeline's additional evidence.
"""
from __future__ import annotations

import json
import math
import hashlib
import random
from pathlib import Path
from typing import Any

import numpy as np

from data_gen.interiorgs_v10.families.common import ref_of
from vagen.envs.active_spatial import canonical_task_metrics_v2 as env_metrics
from vagen.envs.active_spatial import target_visibility as env_visibility


VERSION = "interiorgs_capability_rules_20261016_v1"
TASK_KINDS = {
    "absolute_positioning": "absolute_distance",
    "equidistance": "equidistance_closer",
    "projective_relations": "projective_direction",
    "occlusion_alignment": "occlusion_closer",
    "fov_inclusion": "fov_complete",
}


def load_config() -> dict:
    return json.loads(Path(__file__).with_name("config.json").read_text())


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def measure(task: dict, c2w: Any, occ: Any) -> dict:
    """Call the actual v10 environment judge and its geometry/visibility helpers.

    ``occ`` must be an OccluderSet loaded using the current environment's thick
    wall setting. A missing scene cannot silently disable visibility checks.
    The caller separately proves legal positions and action reachability.
    """
    if occ is None:
        raise ValueError("scene occluders are required")
    pose = np.asarray(c2w, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(pose[:3, :3])), 1.0, abs_tol=1e-6)):
        raise ValueError("invalid camera pose")
    if not np.allclose(pose[2], [0, -1, 0, 1.5], atol=1e-6):
        raise ValueError("capability states must use the horizontal 1.5 metre camera")
    kind = TASK_KINDS.get(task.get("task_type"))
    if kind is None:
        raise ValueError("task family is not enabled in this capability version")
    ctx = env_metrics._Context(task, pose)
    roles = ("a",) if task["task_type"] == "absolute_positioning" else ("a", "b")
    selected = ctx.roles(roles)
    if selected is None:
        raise ValueError("the environment could not resolve distinct target objects")
    metric = env_visibility.score_with_visibility(task, pose, occ)
    objects = []
    for role, obj in zip(roles, selected):
        # The same centre and projection functions as the environment are used.
        centre = 0.5 * (np.asarray(obj["bbox_min"], float) + np.asarray(obj["bbox_max"], float))
        visibility = env_visibility.object_visibility(pose, obj, occ, ctx.K, ctx.width, ctx.height)
        references = []
        anchor = obj.get("ref_anchor")
        if isinstance(anchor, dict):
            if "bbox_min" not in anchor or "bbox_max" not in anchor:
                raise ValueError("reference object has no environment bounds")
            references.append({"id": str(anchor.get("id", "")), "visibility":
                env_visibility.object_visibility(pose, anchor, occ, ctx.K, ctx.width, ctx.height)})
        objects.append({
            "role": role, "id": str(obj.get("id", "")), "label": str(obj.get("label", "")),
            "ref": ref_of(obj), "projection": ctx.project(obj), "visibility": visibility,
            "references": references,
            "distance_m": env_metrics.footprint_distance(float(ctx.position[0]), float(ctx.position[1]), obj),
            "center_distance_m": float(np.linalg.norm(centre[:2] - ctx.position[:2])),
            "center_camera": ctx.to_camera(centre[None, :])[0], "center_world": centre,
        })
    comparison = env_metrics.judge_equidistance(task, pose) if len(objects) == 2 else None
    return _plain({
        "version": VERSION, "source_task_id": task.get("task_id"), "task_type": task["task_type"],
        "pose_c2w": pose, "metric": metric, "objects": objects,
        "comparison_metric": comparison,
        "lateral": env_visibility.lateral_gate(task, pose),
        "camera": {"K_effective": ctx.K, "width": ctx.width, "height": ctx.height},
        "camera_valid": True, "rgb_verified": False, "rendered_depth_verified": False,
    })


def _distance_options(distance: float, cfg: dict, identity: dict) -> list[float]:
    menu = [float(v) for v in cfg["distance_options_m"]]
    if len(menu) != 4 or menu != sorted(set(menu)) or menu[0] <= 0:
        raise ValueError("distance options must contain four increasing positive distances")
    step = menu[1] - menu[0]
    if any(not math.isclose(menu[i+1]-menu[i], step) for i in range(3)):
        raise ValueError("the current distance menu must use one regular interval")
    if not math.isclose(step, float(cfg["distance_extension_step_m"])):
        raise ValueError("distance extension must use the environment target interval")
    if menu[0] <= distance <= menu[-1]:
        # Prefer the actual formal-task menu whenever it covers this state.
        # Balance answer ranks by choosing states, never by changing this menu.
        return menu
    # Every window contains the nearest positive grid point. The window is
    # used only outside the formal menu's range and sampled using state identity
    # only, with no desired answer rank supplied.
    index = max(1, math.floor(distance / step + 0.5))
    material = json.dumps({"seed": cfg["seed"], "identity": identity, "purpose": "distance_window"},
                          sort_keys=True, allow_nan=False)
    rng = random.Random(int.from_bytes(hashlib.sha256(material.encode()).digest(), "big"))
    starts = list(range(max(1, index-3), index+1))
    start = rng.choice(starts) * step
    return [round(start + i * step, 8) for i in range(4)]


def _nearest(distance: float, options: list[float], ratio: float) -> int | None:
    errors = sorted((abs(distance - v), i) for i, v in enumerate(options))
    if errors[1][0] <= 1e-10 or errors[0][0] > ratio * errors[1][0] + 1e-10:
        return None
    return errors[0][1]


def _comparison(a: float, b: float, tol: float, cfg: dict) -> str | None:
    diff = abs(a - b)
    if diff <= max(0.0, tol - float(cfg["comparison_same_inset_m"])):
        return "same"
    if diff >= max(tol + float(cfg["comparison_tolerance_buffer_m"]),
                   float(cfg["comparison_far_relative_gap"]) * max(a, b)):
        return "a_closer" if a < b else "b_closer"
    return None


def _draft(task: dict, measured: dict, kind: str, key: str, keys: list[str],
           fields: dict, evidence: list[str], extra: dict | None = None) -> dict:
    return {"kind": kind, "answer_key": key, "option_keys": keys, "fields": fields,
            "evidence_zh": evidence,
            "extra": {"rules_version": VERSION, "source_task_id": task.get("task_id"),
                      "geometry_only": True, "final_eligible": False,
                      "needs_rgb_evidence": True, "needs_depth_confirmation": True,
                      "environment_success": measured["metric"].get("success"), **(extra or {})}}


def propose(task: dict, measured: dict, config: dict | None = None) -> tuple[list[dict], list[str]]:
    """Return geometry drafts and explicit exclusion reasons, never final labels."""
    cfg = load_config()
    if config is not None:
        cfg.update(config)
    kind = TASK_KINDS.get(task.get("task_type"))
    if kind not in cfg["enabled_kinds"]:
        return [], ["family_not_enabled"]
    if (measured.get("source_task_id") != task.get("task_id")
            or measured.get("task_type") != task.get("task_type")):
        return [], ["measurement_task_mismatch"]
    if measured.get("camera_valid") is not True:
        return [], ["camera_not_current_horizontal_model"]
    objs = measured.get("objects", [])
    if len(objs) != (1 if kind == "absolute_distance" else 2):
        return [], ["missing_targets"]
    refs = [str(o.get("ref") or "").strip() for o in objs]
    if any(not r for r in refs) or len(set(r.casefold() for r in refs)) != len(refs):
        return [], ["ambiguous_target_references"]
    # For the nearer-object question both targets must be seen, even though the
    # original hide-A-behind-B task requires only B to be seen at success.
    for obj in objs:
        if obj.get("visibility", {}).get("visible") is not True:
            return [], ["target_not_identifiable"]
        if any(r["visibility"].get("visible") is not True for r in obj.get("references", [])):
            return [], ["reference_not_identifiable"]
        p = obj["projection"]
        if (not p.get("all_corners_in_front") or not p.get("visible")
                or float(p.get("area_ratio", 0)) < env_metrics.MIN_AREA_RATIO):
            return [], ["target_projection_insufficient"]
        if kind != "fov_complete" and p["inside_fraction"] < env_metrics.MIN_INSIDE_FRACTION:
            return [], ["target_too_far_outside_image"]
    fields = {"a": refs[0]}
    if len(refs) == 2:
        fields["b"] = refs[1]
    evidence = [f"{o['ref']}：相机到标注几何盒子最近侧面的水平距离为{o['distance_m']:.3f}米，"
                f"到盒子中心的水平距离为{o['center_distance_m']:.3f}米。" for o in objs]
    if kind == "absolute_distance":
        d, centre = objs[0]["distance_m"], objs[0]["center_distance_m"]
        identity = {"task_id": task.get("task_id"), "pose_c2w": measured["pose_c2w"]}
        values = _distance_options(d, cfg, identity)
        ratio = float(cfg["distance_error_to_runner_up_ratio"])
        step = values[1] - values[0]
        nearest_grid = max(1, math.floor(d / step + 0.5))
        neighbors = [i*step for i in range(max(1, nearest_grid-1), nearest_grid+2)]
        if _nearest(d, neighbors, ratio) is None:
            return [], ["distance_grid_ambiguous"]
        if _nearest(centre, neighbors, ratio) != _nearest(d, neighbors, ratio):
            return [], ["distance_side_center_disagree_or_ambiguous"]
        idx, centre_idx = _nearest(d, values, ratio), _nearest(centre, values, ratio)
        if idx is None:
            return [], ["distance_option_ambiguous"]
        if centre_idx != idx:
            return [], ["distance_side_center_disagree_or_ambiguous"]
        keys = [f"distance_{i}" for i in range(4)]
        fields["distance_options_m"] = values
        evidence.append(f"四个递增距离选项为{values}米；最近侧面距离与中心距离均明确对应第{idx+1}个选项。")
        return [_draft(task, measured, kind, keys[idx], keys, fields, evidence)], []
    if kind in ("equidistance_closer", "occlusion_closer"):
        tol = float(measured["comparison_metric"]["equidistance_tolerance_m"])
        a, b = objs
        key = _comparison(a["distance_m"], b["distance_m"], tol, cfg)
        center_key = _comparison(a["center_distance_m"], b["center_distance_m"], tol, cfg)
        if key is None or (kind == "occlusion_closer" and key == "same"):
            return [], ["comparison_ambiguous"]
        if center_key != key:
            return [], ["comparison_side_center_disagree_or_ambiguous"]
        if kind == "occlusion_closer":
            depth_key = "a_closer" if a["center_camera"][2] < b["center_camera"][2] else "b_closer"
            if depth_key != key:
                return [], ["depth_side_distance_order_disagree"]
        evidence.append(f"环境等距容差为{tol:.3f}米；两目标最近侧面水平距离差为{abs(a['distance_m']-b['distance_m']):.3f}米。")
        keys = ["a_closer", "b_closer"] + (["same"] if kind == "equidistance_closer" else [])
        return [_draft(task, measured, kind, key, keys, fields, evidence)], []
    if kind == "projective_direction":
        a, b = objs
        angle = measured["lateral"].get("angle_deg")
        if angle is None:
            return [], ["direction_unmeasured"]
        boundary = float(measured["lateral"].get("max_angle_deg", cfg["direction_boundary_deg"]))
        halfwidth = float(cfg["direction_ambiguity_halfwidth_deg"])
        if boundary-halfwidth <= angle <= boundary+halfwidth:
            return [], ["direction_angular_ambiguity"]
        du = a["projection"]["center_uv"][0] - b["projection"]["center_uv"][0]
        dx = a["center_camera"][0] - b["center_camera"][0]
        dz = a["center_camera"][2] - b["center_camera"][2]
        if angle < boundary-halfwidth:
            if (abs(du) < cfg["direction_min_center_gap_px"] or abs(dx) < cfg["direction_min_lateral_m"]
                    or du * dx <= 0):
                return [], ["direction_screen_order_unclear"]
            key = "left" if du < 0 else "right"
        else:
            tol = float(measured["comparison_metric"]["equidistance_tolerance_m"])
            nearer = _comparison(a["distance_m"], b["distance_m"], tol, cfg)
            center_nearer = _comparison(a["center_distance_m"], b["center_distance_m"], tol, cfg)
            depth_nearer = "a_closer" if dz < 0 else "b_closer"
            if nearer not in ("a_closer", "b_closer") or nearer != depth_nearer or center_nearer != nearer:
                return [], ["direction_depth_distance_ambiguous_or_disagree"]
            key = "front" if dz < 0 else "behind"
        evidence.append(f"两物中心连线在地面上与相机左右轴的夹角为{angle:.2f}度；"
                        f"A减B的图像中心横向差为{du:.2f}像素，相机纵深差为{dz:.3f}米。")
        return [_draft(task, measured, kind, key, ["left", "right", "front", "behind"], fields, evidence,
                       {"direction_is_capability_partition_not_full_task_success": True})], []
    if kind == "fov_complete":
        framed = []
        width, height = measured["camera"]["width"], measured["camera"]["height"]
        for obj in objs:
            p = obj["projection"]
            fraction, box = float(p["inside_fraction"]), p["bbox_raw"]
            margin = float(cfg["framing_full_border_margin_px"])
            if (fraction >= cfg["framing_full_min_fraction"] and box[0] >= margin and box[1] >= margin
                    and box[2] <= width-1-margin and box[3] <= height-1-margin):
                framed.append(True)
            elif cfg["framing_cropped_min_fraction"] <= fraction <= cfg["framing_cropped_max_fraction"]:
                framed.append(False)
            else:
                return [], ["framing_geometry_ambiguous"]
            evidence.append(f"{obj['ref']}的三维几何盒子投影框有{fraction*100:.2f}%面积在画面内；这不是实际轮廓入框比例。")
        key = {(True, True): "both", (True, False): "a_only", (False, True): "b_only", (False, False): "neither"}[tuple(framed)]
        return [_draft(task, measured, kind, key, ["both", "a_only", "b_only", "neither"], fields, evidence,
                       {"projected_box_full": framed, "full_flags": framed,
                        "requires_actual_contour_agreement": True})], []
    return [], ["kind_not_implemented"]
