"""Shared versioned canonical task metrics for R1 data, runtime, and eval."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .canonical_camera import (
    CANONICAL_CAMERA_H1_RESIZE_V1,
    build_canonical_camera,
    camera_params_for_visual_metrics,
)
from .visual_bbox_metrics import (
    _extract_objects,
    _match_object,
    _project_object,
    compute_visual_bbox_metrics,
)

CANONICAL_TASK_METRIC_VERSION = "canonical_spatial_task_h1_v1"
GATE_ALIGNED_SHAPING_SCORE_VERSION = "canonical_gate_aligned_shaping_v1"
FOV_MIN_INSIDE_FRACTION = 0.95
FOV_CENTER_MARGIN_FRACTION = 0.05
FOV_MIN_AREA_RATIO = 1e-4
PROJECTIVE_MIN_MARGIN_PX = 12.0
PROJECTIVE_MIN_INSIDE_FRACTION = 0.50
SUPPORTED_TASK_TYPES = frozenset(("projective_relations", "fov_inclusion"))


def _area(box: Any) -> float:
    if not box:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _inside_fraction(obj: dict[str, Any]) -> float:
    raw = _area(obj.get("bbox_raw"))
    return _area(obj.get("bbox")) / raw if raw > 1e-8 else 0.0


def _projective_margin_progress(margin: Any) -> float:
    """Smooth relation progress with the canonical 12 px gate at its centre."""
    if margin is None:
        return 0.0
    value = float(margin)
    if not math.isfinite(value):
        return 0.0
    x = max(-60.0, min(60.0, (value - PROJECTIVE_MIN_MARGIN_PX) / 6.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def score_observation(
    item: dict[str, Any],
    c2w: np.ndarray,
    render_size: tuple[int, int] = (256, 256),
) -> dict[str, Any]:
    """Project one runtime pose through the single H1 camera construction."""
    camera = build_canonical_camera(
        K_native=item["init_camera"]["intrinsics"],
        render_size=render_size,
        transform="resize",
        c2w=c2w,
        item=item,
        camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1,
    )
    metrics = compute_visual_bbox_metrics(
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
        _project_object(
            obj,
            np.asarray(camera.c2w),
            np.asarray(camera.K_effective),
            *camera.render_resolution,
        )
        for obj in selected
        if obj is not None
    ]
    metrics = dict(metrics)
    metrics["objects"] = projections
    return {"camera": camera.to_dict(), "visual_metrics": metrics}


def canonical_projective(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["visual_metrics"]
    objects = metrics.get("objects") or []
    margin = metrics.get("visual_relation_margin_px")
    fractions = [_inside_fraction(obj) for obj in objects]
    gates = {
        "two_objects": len(objects) == 2,
        "in_front": len(objects) == 2
        and all(bool(obj.get("center_in_front")) for obj in objects),
        "visible": len(objects) == 2 and all(bool(obj.get("visible")) for obj in objects),
        "min_area": len(objects) == 2
        and all(
            float(obj.get("area_ratio", 0.0) or 0.0) >= FOV_MIN_AREA_RATIO
            for obj in objects
        ),
        "inside_frame": len(objects) == 2
        and all(value >= PROJECTIVE_MIN_INSIDE_FRACTION for value in fractions),
        "relation": bool(metrics.get("visual_relation_satisfied")),
        "margin": margin is not None and float(margin) >= PROJECTIVE_MIN_MARGIN_PX,
    }
    min_inside = min(fractions) if len(fractions) == 2 else 0.0
    min_area_ratio = (
        min(float(obj.get("area_ratio", 0.0) or 0.0) for obj in objects)
        if len(objects) == 2
        else 0.0
    )
    shaping_components = {
        "two_objects": 1.0 if gates["two_objects"] else 0.0,
        "in_front": 1.0 if gates["in_front"] else 0.0,
        "visible": 1.0 if gates["visible"] else 0.0,
        "min_area": float(np.clip(min_area_ratio / FOV_MIN_AREA_RATIO, 0.0, 1.0)),
        # Keep the actual clipped/raw bbox fraction: a very large separation
        # therefore cannot hide that an object has left the image.
        "inside_frame": float(np.clip(min_inside, 0.0, 1.0)),
        "margin": _projective_margin_progress(margin),
    }
    continuous_gate_progress = float(np.prod(list(shaping_components.values())))
    # Reserve the upper half of the score range for states that actually pass
    # every canonical gate. This guarantees that a near miss cannot outrank a
    # true success while retaining smooth progress inside each band.
    canonical_success = all(gates.values())
    shaping_score = 0.5 * continuous_gate_progress + 0.5 * float(canonical_success)
    return {
        "metric_version": CANONICAL_TASK_METRIC_VERSION,
        "task_metric": "projective_relation_h1",
        "success": canonical_success,
        # Historical reports keep `score`; rewards consume `shaping_score`.
        "score": float(metrics.get("visual_score", 0.0) or 0.0),
        "shaping_score_version": GATE_ALIGNED_SHAPING_SCORE_VERSION,
        "shaping_score": shaping_score,
        "continuous_gate_progress": continuous_gate_progress,
        "shaping_components": shaping_components,
        "inside_frame_fraction_min": min_inside,
        "required_inside_frame_fraction": PROJECTIVE_MIN_INSIDE_FRACTION,
        "relation_margin_px": float(margin) if margin is not None else None,
        "required_relation_margin_px": PROJECTIVE_MIN_MARGIN_PX,
        "gates": gates,
    }


def canonical_fov(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["visual_metrics"]
    objects = metrics.get("objects") or []
    width, height = result["camera"]["render_resolution"]
    boundary = min(width, height) * FOV_CENTER_MARGIN_FRACTION
    centers = [obj.get("center_uv") for obj in objects if obj.get("center_uv") is not None]
    center_margin = None
    if len(centers) == 2:
        center_margin = min(
            min(float(u), float(width) - float(u), float(v), float(height) - float(v))
            for u, v in centers
        )
    fractions = [_inside_fraction(obj) for obj in objects]
    gates = {
        "two_objects": len(objects) == 2,
        "in_front": len(objects) == 2
        and all(bool(obj.get("center_in_front")) for obj in objects),
        "visible": len(objects) == 2 and all(bool(obj.get("visible")) for obj in objects),
        "min_area": len(objects) == 2
        and all(
            float(obj.get("area_ratio", 0.0) or 0.0) >= FOV_MIN_AREA_RATIO
            for obj in objects
        ),
        "full_bbox_in_frame": len(objects) == 2
        and all(value >= FOV_MIN_INSIDE_FRACTION for value in fractions),
        "center_margin": center_margin is not None and center_margin >= boundary,
    }
    inclusion = min(fractions) if fractions else 0.0
    normalized_margin = (
        max(0.0, min(1.0, center_margin / max(boundary, 1.0)))
        if center_margin is not None
        else 0.0
    )
    score = float(inclusion * normalized_margin * (1.0 if gates["min_area"] else 0.0))
    return {
        "metric_version": CANONICAL_TASK_METRIC_VERSION,
        "task_metric": "fov_full_bbox_h1",
        "success": all(gates.values()),
        "score": score,
        "shaping_score_version": GATE_ALIGNED_SHAPING_SCORE_VERSION,
        "shaping_score": score,
        "inside_frame_fraction_min": min(fractions) if fractions else 0.0,
        "center_margin_px": center_margin,
        "required_center_margin_px": boundary,
        "gates": gates,
    }


def score_canonical_task(item: dict[str, Any], c2w: np.ndarray) -> dict[str, Any]:
    if item.get("canonical_task_metric_version") != CANONICAL_TASK_METRIC_VERSION:
        raise ValueError("item does not request the R1 canonical metric backend")
    task_type = str(item.get("task_type") or "")
    result = score_observation(item, np.asarray(c2w, dtype=float))
    if task_type == "projective_relations":
        return canonical_projective(result)
    if task_type == "fov_inclusion":
        return canonical_fov(result)
    raise ValueError(f"canonical metric is not defined for task_type={task_type!r}")


def uses_canonical_backend(item: dict[str, Any] | None) -> bool:
    if item and item.get("canonical_task_metric_version"):
        if item["canonical_task_metric_version"] != CANONICAL_TASK_METRIC_VERSION:
            raise ValueError("unknown canonical_task_metric_version; refusing legacy fallback")
        if item.get("task_type") not in SUPPORTED_TASK_TYPES:
            raise ValueError("canonical metric version does not support this task type")
    return bool(
        item
        and item.get("canonical_task_metric_version") == CANONICAL_TASK_METRIC_VERSION
        and item.get("task_type") in SUPPORTED_TASK_TYPES
    )
