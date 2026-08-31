"""Layout-aware target sampling and audit helpers for Active Spatial data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point2D = Tuple[float, float]
Segment2D = Tuple[Point2D, Point2D]


def as_xy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[0] < 2:
        return None
    return arr[:2]


def point_in_poly(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def distance_point_to_segment(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    p = np.asarray(point, dtype=np.float64)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    ab = bb - aa
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - aa))
    t = float(np.clip(np.dot(p - aa, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (aa + t * ab)))


def normalize_xy(value: Any) -> Optional[np.ndarray]:
    xy = as_xy(value)
    if xy is None:
        return None
    norm = float(np.linalg.norm(xy))
    if norm <= 1e-8:
        return None
    return xy / norm


@dataclass
class LayoutGeometry:
    room_polys: List[List[Point2D]]
    wall_segments: List[Segment2D]

    @classmethod
    def from_scene_path(cls, scene_path: Path) -> "LayoutGeometry":
        structure_path = scene_path / "structure.json"
        if not structure_path.exists():
            return cls(room_polys=[], wall_segments=[])
        data = json.loads(structure_path.read_text(encoding="utf-8"))
        room_polys: List[List[Point2D]] = []
        wall_segments: List[Segment2D] = []
        for room in data.get("rooms", []):
            poly = [
                (float(p[0]), float(p[1]))
                for p in room.get("profile", [])
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            if len(poly) >= 3:
                room_polys.append(poly)
                for i in range(len(poly)):
                    wall_segments.append((poly[i], poly[(i + 1) % len(poly)]))
        for wall in data.get("walls", []):
            loc = wall.get("location")
            if isinstance(loc, list) and len(loc) >= 2 and len(loc[0]) >= 2 and len(loc[1]) >= 2:
                wall_segments.append(((float(loc[0][0]), float(loc[0][1])), (float(loc[1][0]), float(loc[1][1]))))
        return cls(room_polys=room_polys, wall_segments=wall_segments)

    def contains_xy(self, xy: Sequence[float]) -> bool:
        if not self.room_polys:
            return True
        x, y = float(xy[0]), float(xy[1])
        return any(point_in_poly(x, y, poly) for poly in self.room_polys)

    def wall_distance(self, xy: Sequence[float]) -> float:
        if not self.wall_segments:
            return float("inf")
        point = (float(xy[0]), float(xy[1]))
        return min(distance_point_to_segment(point, a, b) for a, b in self.wall_segments)

    def is_safe_xy(self, xy: Sequence[float], min_wall_clearance: float) -> bool:
        return self.contains_xy(xy) and self.wall_distance(xy) >= min_wall_clearance

    def candidate_points(self, grid_spacing: float = 0.25) -> Iterable[np.ndarray]:
        for poly in self.room_polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            for x in np.arange(min(xs), max(xs) + 1e-6, grid_spacing):
                for y in np.arange(min(ys), max(ys) + 1e-6, grid_spacing):
                    if point_in_poly(float(x), float(y), poly):
                        yield np.array([float(x), float(y)], dtype=np.float64)


def load_room_polys(scene_path: Path) -> List[List[Point2D]]:
    return LayoutGeometry.from_scene_path(scene_path).room_polys


def extract_target_xy(item: Dict[str, Any]) -> Optional[np.ndarray]:
    if item.get("sample_target") is not None:
        return as_xy(item.get("sample_target"))
    region = item.get("target_region")
    if isinstance(region, dict):
        return as_xy(region.get("sample_point"))
    return None


def extract_init_xy(item: Dict[str, Any]) -> Optional[np.ndarray]:
    extrinsics = item.get("init_camera", {}).get("extrinsics")
    if extrinsics is None:
        return None
    mat = np.asarray(extrinsics, dtype=np.float64)
    if mat.shape[0] < 3 or mat.shape[1] < 4:
        return None
    return mat[:2, 3]


def image_std(image: Any) -> Optional[float]:
    if image is None:
        return None
    try:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        return float(arr.std())
    except Exception:
        return None


def region_to_dict(region: Any) -> Dict[str, Any]:
    if hasattr(region, "to_dict"):
        return region.to_dict()
    if isinstance(region, dict):
        return region
    return {}


def region_type(region: Any) -> str:
    if hasattr(region, "region_type"):
        return str(getattr(region.region_type, "value", region.region_type)).lower()
    return str(region_to_dict(region).get("type", "")).lower()


def region_params(region: Any) -> Dict[str, Any]:
    if hasattr(region, "params"):
        return region.params
    return region_to_dict(region).get("params", {}) or {}


def target_height(region: Any, default: float = 1.5) -> float:
    if hasattr(region, "height"):
        return float(region.height)
    return float(region_to_dict(region).get("height", default))


def look_at_for_region(region: Any) -> Optional[np.ndarray]:
    params = region_params(region)
    for key in ("object_center", "object_b_center", "midpoint", "center", "object_a_center", "boundary_point"):
        value = params.get(key)
        if value is not None:
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape[0] == 2:
                return np.array([arr[0], arr[1], target_height(region)], dtype=np.float64)
            if arr.shape[0] >= 3:
                return arr[:3]
    return None


def compute_forward(point3: np.ndarray, look_at: Optional[np.ndarray]) -> np.ndarray:
    if look_at is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    direction = np.asarray(look_at, dtype=np.float64)[:3] - point3[:3]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def distance_to_region_objects(xy: np.ndarray, params: Dict[str, Any]) -> float:
    centers = []
    for key in ("object_center", "object_a_center", "object_b_center", "center"):
        center = as_xy(params.get(key))
        if center is not None:
            centers.append(center)
    if not centers:
        return float("inf")
    return min(float(np.linalg.norm(xy - center)) for center in centers)


def region_candidate_error(xy: np.ndarray, region: Any) -> Optional[float]:
    rtype = region_type(region)
    params = region_params(region)
    min_distance = float(params.get("min_distance", 0.0) or 0.0)
    if distance_to_region_objects(xy, params) < min_distance:
        return None

    if rtype == "circle":
        center = as_xy(params.get("center"))
        if center is None:
            center = as_xy(params.get("object_center"))
        radius = params.get("radius", params.get("sample_distance"))
        if center is None or radius is None:
            return None
        return abs(float(np.linalg.norm(xy - center)) - float(radius))

    if rtype == "annulus":
        center = as_xy(params.get("center"))
        if center is None:
            return None
        radius = float(np.linalg.norm(xy - center))
        min_radius = float(params.get("min_radius", 0.0))
        max_radius = float(params.get("max_radius", float("inf")))
        return max(min_radius - radius, radius - max_radius, 0.0)

    if rtype == "line":
        midpoint = as_xy(params.get("midpoint"))
        start = as_xy(params.get("start"))
        end = as_xy(params.get("end"))
        direction = normalize_xy(params.get("direction"))
        if direction is None and start is not None and end is not None:
            direction = normalize_xy(end - start)
        if direction is None:
            return None
        anchor = midpoint if midpoint is not None else (start if start is not None else np.zeros(2))
        delta = xy - anchor
        lateral = abs(float(direction[0] * delta[1] - direction[1] * delta[0]))
        min_offset = float(params.get("min_offset_from_midpoint", 0.0) or 0.0)
        offset_error = 0.0
        if midpoint is not None and min_offset > 0.0:
            offset = abs(float(np.dot(xy - midpoint, direction)))
            offset_error = max(min_offset - offset, 0.0)
        return lateral + offset_error

    if rtype == "ray":
        origin = as_xy(params.get("origin"))
        direction = normalize_xy(params.get("direction"))
        if origin is None or direction is None:
            return None
        rel = xy - origin
        proj = float(np.dot(rel, direction))
        lateral = float(np.linalg.norm(rel - proj * direction))
        error = lateral + max(float(params.get("min_distance", 0.0) or 0.0) - proj, 0.0)
        max_proj = params.get("max_distance")
        if max_proj is not None:
            error += max(proj - float(max_proj), 0.0)
        return error

    if rtype == "half_plane":
        boundary = as_xy(params.get("boundary_point"))
        normal = normalize_xy(params.get("normal"))
        if boundary is None or normal is None:
            return None
        signed = float(np.dot(xy - boundary, normal))
        return max(-signed, 0.0)

    if rtype == "curve":
        points = [as_xy(pt) for pt in params.get("points", [])]
        points = [pt for pt in points if pt is not None]
        if not points:
            return None
        return min(float(np.linalg.norm(xy - pt)) for pt in points)

    if rtype == "point":
        target = as_xy(params.get("point"))
        if target is None:
            target = as_xy(region_to_dict(region).get("sample_point"))
        return float(np.linalg.norm(xy - target)) if target is not None else 0.0

    return None


def repair_region_sample_to_layout(
    region: Any,
    layout: LayoutGeometry,
    *,
    min_wall_clearance: float = 0.5,
    grid_spacing: float = 0.25,
    max_error: float = 0.35,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
    if not layout.room_polys:
        return None

    current = as_xy(region_to_dict(region).get("sample_point"))
    if current is not None and layout.is_safe_xy(current, min_wall_clearance):
        point3 = np.array([current[0], current[1], target_height(region)], dtype=np.float64)
        return point3, compute_forward(point3, look_at_for_region(region)), {
            "layout_repaired": 0.0,
            "target_wall_distance": layout.wall_distance(current),
            "region_error": 0.0,
        }

    best = None
    for xy in layout.candidate_points(grid_spacing=grid_spacing):
        wall_dist = layout.wall_distance(xy)
        if wall_dist < min_wall_clearance:
            continue
        err = region_candidate_error(xy, region)
        if err is None:
            continue
        rank = (float(err), -float(wall_dist))
        if best is None or rank < (best[0], best[1]):
            best = (rank[0], rank[1], xy)

    if best is None or best[0] > max_error:
        return None
    xy = best[2]
    point3 = np.array([xy[0], xy[1], target_height(region)], dtype=np.float64)
    return point3, compute_forward(point3, look_at_for_region(region)), {
        "layout_repaired": 1.0,
        "target_wall_distance": layout.wall_distance(xy),
        "region_error": float(best[0]),
    }
