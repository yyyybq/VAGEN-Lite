"""Region-aware geometric diagnostics for Active Spatial tasks.

These metrics are intentionally diagnostic: they do not change the environment
success gate by themselves.  They make it explicit whether a pose is close to
the task's valid region instead of only close to one arbitrary sample point.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _as_array(value: Any, default: Any) -> np.ndarray:
    try:
        return np.array(value if value is not None else default, dtype=np.float64)
    except Exception:
        return np.array(default, dtype=np.float64)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-10:
        return vec
    return vec / norm


def _distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _exp_score(distance: Optional[float], scale: float = 1.0) -> Optional[float]:
    if distance is None:
        return None
    return float(math.exp(-max(distance, 0.0) / max(scale, 1e-6)))


def _sample_point(target_region: Dict[str, Any]) -> Optional[np.ndarray]:
    point = target_region.get("sample_point")
    if point is None:
        return None
    try:
        return np.array(point, dtype=np.float64)
    except Exception:
        return None


def _with_height(point_2d: np.ndarray, z: float) -> list:
    return [float(point_2d[0]), float(point_2d[1]), float(z)]


def _point_to_line_distance(
    point: np.ndarray,
    line_point: np.ndarray,
    direction: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    d = _normalize(direction[:2])
    rel = point[:2] - line_point[:2]
    progress = float(np.dot(rel, d))
    nearest = line_point[:2] + progress * d
    return float(np.linalg.norm(point[:2] - nearest)), nearest, progress


def _point_to_ray_segment_distance(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    min_distance: float = 0.0,
    max_distance: Optional[float] = None,
) -> Tuple[float, np.ndarray, float, float]:
    d = _normalize(direction[:2])
    rel = point[:2] - origin[:2]
    progress = float(np.dot(rel, d))
    clamped = max(progress, float(min_distance))
    if max_distance is not None and np.isfinite(float(max_distance)):
        clamped = min(clamped, float(max_distance))
    nearest = origin[:2] + clamped * d
    return float(np.linalg.norm(point[:2] - nearest)), nearest, progress, clamped


def _circle_region(
    camera_position: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> Tuple[float, np.ndarray, float]:
    current = _distance_2d(camera_position, center)
    deviation = abs(current - float(radius))
    direction = camera_position[:2] - center[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        nearest = center[:2] + np.array([float(radius), 0.0])
    else:
        nearest = center[:2] + direction / norm * float(radius)
    return deviation, nearest, current


def compute_region_metrics(
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    task_type: str,
    task_params: Dict[str, Any],
    target_region: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute task-region diagnostics for a camera pose.

    Returns standard keys:
      - distance_to_region: 0 when inside/on the valid region when meaningful
      - region_score: smooth [0, 1] score derived from the region distance
      - region_target_point: nearest or representative point on the region
      - sample_target_distance: distance to sample_point, kept as auxiliary
    """
    del camera_forward, task_params  # reserved for future view-dependent metrics

    params = target_region.get("params", {}) if isinstance(target_region, dict) else {}
    region_type = target_region.get("type") if isinstance(target_region, dict) else None
    pos = np.array(camera_position, dtype=np.float64)
    sample = _sample_point(target_region if isinstance(target_region, dict) else {})
    sample_distance = _distance_2d(pos, sample) if sample is not None else None
    z = float(pos[2]) if pos.shape[0] >= 3 else 0.0

    out: Dict[str, Any] = {
        "task_type": task_type,
        "region_type": region_type,
        "distance_to_region": None,
        "region_score": None,
        "region_target_point": sample.tolist() if sample is not None else None,
        "sample_target_distance": sample_distance,
        "sample_target_is_auxiliary": task_type not in {"delta_control"},
    }

    try:
        if task_type in {"absolute_positioning", "screen_occupancy"}:
            center = _as_array(params.get("object_center", params.get("center")), [0, 0, 0])
            radius = float(params.get("radius", params.get("sample_distance", 2.0)))
            dist, nearest, current = _circle_region(pos, center, radius)
            out.update({
                "distance_to_region": dist,
                "region_score": _exp_score(dist, scale=max(radius * 0.3, 0.5)),
                "region_target_point": _with_height(nearest, z),
                "current_radius": current,
                "target_radius": radius,
            })

        elif task_type == "delta_control":
            if sample is not None:
                out.update({
                    "distance_to_region": sample_distance,
                    "region_score": _exp_score(sample_distance, scale=1.0),
                    "region_target_point": sample.tolist(),
                    "sample_target_is_auxiliary": False,
                })

        elif task_type == "equidistance":
            center_a = _as_array(params.get("object_a_center"), [0, 0, 0])
            center_b = _as_array(params.get("object_b_center"), [1, 0, 0])
            dist_a = _distance_2d(pos, center_a)
            dist_b = _distance_2d(pos, center_b)
            distance_diff = abs(dist_a - dist_b)
            if "direction" in params and ("midpoint" in params or "start" in params):
                line_point = _as_array(params.get("midpoint", params.get("start")), [0, 0])
                direction = _as_array(params.get("direction"), [1, 0])
                line_dist, nearest, progress = _point_to_line_distance(pos, line_point, direction)
            else:
                midpoint = (center_a[:2] + center_b[:2]) / 2.0
                ab = center_b[:2] - center_a[:2]
                direction = np.array([-ab[1], ab[0]], dtype=np.float64)
                line_dist, nearest, progress = _point_to_line_distance(pos, midpoint, direction)
            out.update({
                "distance_to_region": line_dist,
                "region_score": _exp_score(line_dist, scale=0.75),
                "region_target_point": _with_height(nearest, z),
                "distance_to_a": dist_a,
                "distance_to_b": dist_b,
                "distance_difference": distance_diff,
                "line_progress": progress,
            })

        elif task_type == "projective_relations":
            boundary = _as_array(params.get("boundary_point"), [0, 0])
            normal = _normalize(_as_array(params.get("normal"), [1, 0])[:2])
            signed = float(np.dot(pos[:2] - boundary[:2], normal))
            violation = max(0.0, -signed)
            nearest = pos[:2] if signed >= 0 else pos[:2] + violation * normal
            out.update({
                "distance_to_region": violation,
                "region_score": 1.0 if signed >= 0 else _exp_score(violation, scale=1.0),
                "region_target_point": _with_height(nearest, z),
                "signed_distance_to_region": signed,
                "inside_half_plane": signed >= 0,
                "relation": params.get("relation"),
            })

        elif task_type in {"centering", "occlusion_alignment"}:
            origin = _as_array(params.get("origin"), [0, 0])
            direction = _as_array(params.get("direction"), [1, 0])
            min_dist = float(params.get("min_distance", 0.0) or 0.0)
            max_dist_raw = params.get("max_distance", None)
            max_dist = float(max_dist_raw) if max_dist_raw is not None else None
            ray_dist, nearest, progress, clamped = _point_to_ray_segment_distance(
                pos, origin, direction, min_dist, max_dist
            )
            out.update({
                "distance_to_region": ray_dist,
                "region_score": _exp_score(ray_dist, scale=0.75),
                "region_target_point": _with_height(nearest, z),
                "ray_progress": progress,
                "ray_clamped_progress": clamped,
                "ray_min_distance": min_dist,
                "ray_max_distance": max_dist,
                "inside_ray_interval": progress >= min_dist and (max_dist is None or progress <= max_dist),
            })

        elif task_type == "fov_inclusion":
            center = _as_array(params.get("center"), [0, 0])
            min_radius = float(params.get("min_radius", 0.0))
            max_radius = float(params.get("max_radius", max(min_radius, 1.0)))
            dist_mid = _distance_2d(pos, center)
            if dist_mid < min_radius:
                violation = min_radius - dist_mid
                target_radius = min_radius
            elif dist_mid > max_radius:
                violation = dist_mid - max_radius
                target_radius = max_radius
            else:
                violation = 0.0
                target_radius = dist_mid
            _, nearest, _ = _circle_region(pos, center, target_radius)
            out.update({
                "distance_to_region": violation,
                "region_score": 1.0 if violation == 0.0 else _exp_score(violation, scale=1.0),
                "region_target_point": _with_height(nearest, z),
                "distance_to_midpoint": dist_mid,
                "min_radius": min_radius,
                "max_radius": max_radius,
                "inside_annulus": violation == 0.0,
            })

        elif task_type == "size_distance_invariance":
            center_a = _as_array(params.get("object_a_center"), [0, 0, 0])
            center_b = _as_array(params.get("object_b_center"), [1, 0, 0])
            size_a = max(float(params.get("object_a_size", 1.0)), 0.05)
            size_b = max(float(params.get("object_b_size", 1.0)), 0.05)
            dist_a = max(_distance_2d(pos, center_a), 0.1)
            dist_b = max(_distance_2d(pos, center_b), 0.1)
            ratio = (size_a / dist_a) / max(size_b / dist_b, 1e-6)
            log_deviation = abs(math.log(max(ratio, 1e-6)))
            if "center" in params and "radius" in params:
                center = _as_array(params.get("center"), [0, 0])
                radius = float(params.get("radius"))
                circle_dist, nearest, _ = _circle_region(pos, center, radius)
                distance_to_region = circle_dist
            else:
                nearest = sample[:2] if sample is not None else pos[:2]
                distance_to_region = log_deviation
            out.update({
                "distance_to_region": distance_to_region,
                "region_score": _exp_score(distance_to_region, scale=0.75),
                "region_target_point": _with_height(nearest, z),
                "size_ratio": ratio,
                "log_size_ratio_deviation": log_deviation,
            })

        elif task_type == "apparent_size_ordering":
            large_center = _as_array(params.get("larger_object_center", params.get("object_a_center")), [0, 0, 0])
            small_center = _as_array(params.get("smaller_object_center", params.get("object_b_center")), [1, 0, 0])
            large_size = max(float(params.get("larger_object_size", 1.0)), 0.05)
            small_size = max(float(params.get("smaller_object_size", 1.0)), 0.05)
            target_ratio = max(float(params.get("target_ratio", 1.25)), 1.01)
            dist_large = max(_distance_2d(pos, large_center), 0.1)
            dist_small = max(_distance_2d(pos, small_center), 0.1)
            apparent_ratio = (large_size / dist_large) / max(small_size / dist_small, 1e-6)
            violation = max(0.0, math.log(target_ratio) - math.log(max(apparent_ratio, 1e-6)))
            out.update({
                "distance_to_region": violation,
                "region_score": 1.0 if violation == 0.0 else _exp_score(violation, scale=0.35),
                "region_target_point": sample.tolist() if sample is not None else _with_height(pos[:2], z),
                "apparent_ratio": apparent_ratio,
                "target_ratio": target_ratio,
                "ordering_satisfied": apparent_ratio >= target_ratio,
            })

        else:
            if sample is not None:
                out.update({
                    "distance_to_region": sample_distance,
                    "region_score": _exp_score(sample_distance, scale=1.0),
                    "region_target_point": sample.tolist(),
                })
    except Exception as exc:
        out["region_metric_error"] = str(exc)

    return out


def region_heuristic(metrics: Dict[str, Any]) -> float:
    """Planner/search heuristic derived from region metrics."""
    score = metrics.get("region_score")
    if score is not None:
        try:
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            pass
    distance = metrics.get("distance_to_region")
    if distance is None:
        return 0.0
    try:
        return float(math.exp(-max(float(distance), 0.0) / 3.0))
    except Exception:
        return 0.0
