"""Versioned canonical spatial metrics used only by the R1 data repair chain.

This adapter deliberately leaves the historical environment scorer untouched.  It
shares camera construction with ``canonical_camera`` and makes every metric gate
explicit in the generated manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vagen.envs.active_spatial.canonical_camera import (
    CANONICAL_CAMERA_H1_RESIZE_V1,
    build_canonical_camera,
    camera_params_for_visual_metrics,
    camera_pose_from_forward,
)
from vagen.envs.active_spatial.visual_bbox_metrics import (
    _extract_objects,
    _match_object,
    _project_object,
    compute_visual_bbox_metrics,
)

CANONICAL_TASK_METRIC_VERSION = "canonical_spatial_task_h1_v1"
FOV_MIN_INSIDE_FRACTION = 0.95
FOV_CENTER_MARGIN_FRACTION = 0.05
FOV_MIN_AREA_RATIO = 1e-4
PROJECTIVE_MIN_MARGIN_PX = 12.0
PROJECTIVE_MIN_INSIDE_FRACTION = 0.50


def area(box: Any) -> float:
    if not box:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def inside_fraction(obj: dict[str, Any]) -> float:
    raw = area(obj.get("bbox_raw"))
    return area(obj.get("bbox")) / raw if raw > 1e-8 else 0.0


def score_observation(item: dict[str, Any], c2w: np.ndarray, render_size: tuple[int, int] = (256, 256)) -> dict[str, Any]:
    """Score one pose under the shared H1 camera and attach detailed projections."""
    camera = build_canonical_camera(
        K_native=item["init_camera"]["intrinsics"],
        render_size=render_size,
        transform="resize",
        c2w=c2w,
        item=item,
        camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1,
    )
    vm = compute_visual_bbox_metrics(
        np.asarray(c2w[:3, 3], dtype=float),
        np.asarray(c2w[:3, 2], dtype=float),
        item["task_type"],
        camera_params_for_visual_metrics(item, camera),
        item["target_region"],
    )
    params = item.get("target_region", {}).get("params", {})
    objects = _extract_objects({"_target_object": item.get("target_object")})
    selected = (
        _match_object(objects, center=params.get("object_a_center"), fallback_index=0),
        _match_object(objects, center=params.get("object_b_center"), fallback_index=1),
    )
    projections = [
        _project_object(obj, np.asarray(camera.c2w), np.asarray(camera.K_effective), *camera.render_resolution)
        for obj in selected if obj is not None
    ]
    vm = dict(vm)
    vm["objects"] = projections
    return {"camera": camera.to_dict(), "visual_metrics": vm}


def canonical_projective(result: dict[str, Any]) -> dict[str, Any]:
    vm = result["visual_metrics"]
    objs = vm.get("objects") or []
    margin = vm.get("visual_relation_margin_px")
    gates = {
        "two_objects": len(objs) == 2,
        "in_front": len(objs) == 2 and all(bool(o.get("center_in_front")) for o in objs),
        "visible": len(objs) == 2 and all(bool(o.get("visible")) for o in objs),
        "min_area": len(objs) == 2 and all(float(o.get("area_ratio", 0.0) or 0.0) >= FOV_MIN_AREA_RATIO for o in objs),
        "inside_frame": len(objs) == 2 and all(inside_fraction(o) >= PROJECTIVE_MIN_INSIDE_FRACTION for o in objs),
        "relation": bool(vm.get("visual_relation_satisfied")),
        "margin": margin is not None and float(margin) >= PROJECTIVE_MIN_MARGIN_PX,
    }
    return {
        "metric_version": CANONICAL_TASK_METRIC_VERSION,
        "task_metric": "projective_relation_h1",
        "success": all(gates.values()),
        "score": float(vm.get("visual_score", 0.0) or 0.0),
        "relation_margin_px": float(margin) if margin is not None else None,
        "gates": gates,
    }


def canonical_fov(result: dict[str, Any]) -> dict[str, Any]:
    """Strict R1 FOV task: both full 3D bboxes must actually fit the image.

    The historical score has a 0.7 floor whenever objects are merely visible;
    this metric has no such floor.  It is versioned data/audit semantics only.
    """
    vm = result["visual_metrics"]
    objs = vm.get("objects") or []
    width, height = result["camera"]["render_resolution"]
    boundary = min(width, height) * FOV_CENTER_MARGIN_FRACTION
    centers = [o.get("center_uv") for o in objs if o.get("center_uv") is not None]
    if len(centers) == 2:
        center_margin = min(
            min(float(u), float(width) - float(u), float(v), float(height) - float(v))
            for u, v in centers
        )
    else:
        # Invalid projections are semantic failures, not an exception path.
        center_margin = -float("inf")
    fractions = [inside_fraction(o) for o in objs]
    gates = {
        "two_objects": len(objs) == 2,
        "in_front": len(objs) == 2 and all(bool(o.get("center_in_front")) for o in objs),
        "visible": len(objs) == 2 and all(bool(o.get("visible")) for o in objs),
        "min_area": len(objs) == 2 and all(float(o.get("area_ratio", 0.0) or 0.0) >= FOV_MIN_AREA_RATIO for o in objs),
        "full_bbox_in_frame": len(objs) == 2 and all(x >= FOV_MIN_INSIDE_FRACTION for x in fractions),
        "center_margin": center_margin >= boundary,
    }
    # A continuous diagnostic only; success always comes from the above gates.
    inclusion = min(fractions) if fractions else 0.0
    normalized_margin = max(0.0, min(1.0, center_margin / max(boundary, 1.0)))
    return {
        "metric_version": CANONICAL_TASK_METRIC_VERSION,
        "task_metric": "fov_full_bbox_h1",
        "success": all(gates.values()),
        "score": float(inclusion * normalized_margin * (1.0 if gates["min_area"] else 0.0)),
        "inside_frame_fraction_min": min(fractions) if fractions else 0.0,
        "center_margin_px": center_margin,
        "required_center_margin_px": boundary,
        "gates": gates,
    }


def pose_from_item_target(item: dict[str, Any]) -> np.ndarray:
    return camera_pose_from_forward(item["sample_target"], item["camera_params"]["forward"])
