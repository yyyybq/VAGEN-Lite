"""Shared Active -> Yes/No contract.

The module intentionally delegates scoring to the existing Active Spatial
``SpatialPotentialField`` and the versioned canonical metric backend.  It does
not create a second task evaluator.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

import numpy as np

from vagen.envs.active_spatial.canonical_task_metrics import (
    CANONICAL_TASK_METRIC_VERSION,
    score_canonical_task,
    uses_canonical_backend,
)
from vagen.envs.active_spatial.spatial_potential_field import create_potential_field

QA_SCHEMA_VERSION = "active_spatial_paired_qa_v1"
SUCCESS_PREDICATE_VERSION = "active_spatial_env_success_v1"
ACTIVE_TASK_TYPES = (
    "absolute_positioning",
    "delta_control",
    "equidistance",
    "projective_relations",
    "centering",
    "occlusion_alignment",
    "fov_inclusion",
    "size_distance_invariance",
    "apparent_size_ordering",
    "screen_occupancy",
)


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _task_params(item: Dict[str, Any], config: Any = None, pose: np.ndarray | None = None) -> Dict[str, Any]:
    params = dict(item.get("task_params") or {})
    params["_target_object"] = item.get("target_object")
    camera = item.get("init_camera") or {}
    if camera.get("intrinsics") is not None:
        params["_camera_intrinsics"] = camera["intrinsics"]
    params["_image_width"] = int(_config_value(config, "image_width", 512))
    params["_image_height"] = int(_config_value(config, "image_height", 512))
    params["_fov_horizontal"] = float(_config_value(config, "fov_horizontal", 90.0))
    params["_fov_vertical"] = float(_config_value(config, "fov_vertical", 90.0))
    if pose is not None:
        params["_camera_pose_c2w"] = np.asarray(pose, dtype=float).tolist()
    return params


def _field(config: Any = None):
    return create_potential_field({
        "position_weight": float(_config_value(config, "potential_field_position_weight", 0.7)),
        "orientation_weight": float(_config_value(config, "potential_field_orientation_weight", 0.3)),
        "max_distance": float(_config_value(config, "max_distance", 5.0)),
        "fov_horizontal": float(_config_value(config, "fov_horizontal", 90.0)),
        "fov_vertical": float(_config_value(config, "fov_vertical", 90.0)),
        "use_visual_bbox_scoring": bool(_config_value(config, "use_visual_bbox_scoring", True)),
    })


def evaluate_state(item: Dict[str, Any], c2w: Sequence[Sequence[float]], config: Any = None) -> Dict[str, Any]:
    """Return the exact Active success decision and diagnostics for one pose."""
    pose = np.asarray(c2w, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"c2w must be 4x4, got {pose.shape}")
    if uses_canonical_backend(item):
        metric = score_canonical_task(item, pose)
        return {
            "success": bool(metric["success"]),
            "score": float(metric.get("score", 0.0)),
            "predicate_version": metric.get("metric_version", CANONICAL_TASK_METRIC_VERSION),
            "metric": metric,
        }

    field = _field(config)
    result = field.compute_score(
        camera_position=pose[:3, 3],
        camera_forward=pose[:3, 2],
        task_type=str(item.get("task_type") or ""),
        task_params=_task_params(item, config, pose),
        target_region=item.get("target_region") or {},
    )
    threshold = float(_config_value(config, "success_score_threshold", 0.95))
    return {
        "success": bool(float(result.total_score) >= threshold),
        "score": float(result.total_score),
        "predicate_version": SUCCESS_PREDICATE_VERSION,
        "threshold": threshold,
        "position_score": float(result.position_score),
        "orientation_score": float(result.orientation_score),
        "details": result.details,
    }


def task_contract_rows() -> list[dict[str, Any]]:
    """Human-readable contract; params remain data-driven from task_region."""
    templates = {
        "absolute_positioning": "Is the camera at the requested distance from {object_label} and facing it with the object visible?",
        "delta_control": "Given the permitted reference observation, is the camera at the requested delta-control target and is the target visible?",
        "equidistance": "Are the camera distances to {object_label} equal, with both objects visible and the camera facing their midpoint?",
        "projective_relations": "Does the image satisfy the requested projective relation for the target objects, with both visible?",
        "centering": "Is the designated object centered in the requested relation, with all required objects visible?",
        "occlusion_alignment": "Does the view satisfy the requested occlusion alignment and visibility conditions?",
        "fov_inclusion": "Are all required target objects inside the field of view with the required safe margins?",
        "size_distance_invariance": "Do the required objects have equal apparent size under the Active scoring conditions?",
        "apparent_size_ordering": "Does the designated larger object appear at least the required ratio larger, with required visibility?",
        "screen_occupancy": "Does the target object occupy the requested screen fraction while visible and faced?",
    }
    params = {
        "absolute_positioning": "target_region.params.object_center, radius/requested_distance, target_object, visibility/facing gates",
        "delta_control": "target_region.params.start_position + delta/target point, target_object, reference state",
        "equidistance": "target_region.params.object_a_center/object_b_center, midpoint/line tolerance, visibility/facing gates",
        "projective_relations": "target_region.params relation, object_a/object_b centers, canonical camera + margin gates",
        "centering": "target_region.params ray/origin/direction, target_object, visibility/facing gates",
        "occlusion_alignment": "target_region.params occluder/target relation, visibility/alignment gates",
        "fov_inclusion": "target_region.params required objects, canonical full-bbox/center-margin gates",
        "size_distance_invariance": "target_region.params object pair + apparent-size tolerance, visibility gates",
        "apparent_size_ordering": "target_region.params ordered pair + ratio, visibility gates",
        "screen_occupancy": "target_region.params.object_center, occupancy_ratio/radius, visibility/facing gates",
    }
    return [{
        "task_type": task,
        "complete_goal_parameters": params[task],
        "success_entry": "data_gen.active_spatial_qa.contract.evaluate_state -> Active Spatial potential field/canonical backend",
        "observation": "single_image when sufficient; matched_history for delta_control or any history-dependent row",
        "qa_template": templates[task],
    } for task in ACTIVE_TASK_TYPES]


def build_task_contract() -> Dict[str, Any]:
    return {
        "version": QA_SCHEMA_VERSION,
        "success_predicate_version": SUCCESS_PREDICATE_VERSION,
        "task_types": task_contract_rows(),
        "environment_supported_tasks": list(ACTIVE_TASK_TYPES),
        "checkpoint_coverage_note": "Checkpoint coverage is recorded from training manifest metadata; environment support is not model coverage.",
    }
