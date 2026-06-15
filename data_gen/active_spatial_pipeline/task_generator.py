"""
Task Generator Module

This module generates task-specific target REGIONS based on 
the 9 types of spatial navigation tasks defined in task_design.py.

Key Insight: Most tasks have multiple valid target positions (not just one point).
This module returns target_region representing the valid solution space:

1. Absolute Positioning: CIRCLE (radius around object)
2. Delta Control: POINT (single position)
3. Equidistance: LINE (perpendicular bisector)
4. Projective Relations: HALF_PLANE (A left/right of B)
5. Centering: RAY (from A through midpoint of BC)
6. Occlusion Alignment: RAY (from A through B, extending beyond)
7. FoV Inclusion: ANNULUS/REGION (distance constraints for visibility)
8. Size-Distance Invariance: CURVE (iso-size positions)
9. Screen Occupancy: CIRCLE (specific distance for occupancy ratio)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    from .config import TaskConfig
    from .camera_sampler import CameraPose
except ImportError:
    from config import TaskConfig
    from camera_sampler import CameraPose


class RegionType(Enum):
    """Types of target regions."""
    POINT = "point"           # Single point
    CIRCLE = "circle"         # Circle in XY plane (height fixed)
    LINE = "line"             # Line segment
    RAY = "ray"               # Ray (infinite in one direction)
    HALF_PLANE = "half_plane" # Half of 2D plane
    ANNULUS = "annulus"       # Ring region (min/max distance)
    CURVE = "curve"           # Parametric curve (sampled points)


@dataclass
class TargetRegion:
    """
    Represents a target region (set of valid target positions).
    
    Attributes:
        region_type: Type of region (point, circle, line, ray, etc.)
        params: Region-specific parameters
        sample_point: One sampled valid point (for training/visualization)
        sample_forward: Forward direction at sample_point
        height: Z coordinate (agent height)
    """
    region_type: RegionType
    params: Dict[str, Any]
    sample_point: np.ndarray
    sample_forward: np.ndarray
    height: float = 1.5
    
    @staticmethod
    def _convert_to_native(value: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.bool_,)):
            return bool(value)
        elif isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, dict):
            return {k: TargetRegion._convert_to_native(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [TargetRegion._convert_to_native(v) for v in value]
        else:
            return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "type": self.region_type.value,
            "params": {},
            "sample_point": self.sample_point.tolist(),
            "sample_forward": self.sample_forward.tolist(),
            "height": float(self.height),
        }
        
        # Convert all numpy types in params to native Python types
        for key, value in self.params.items():
            result["params"][key] = self._convert_to_native(value)
        
        return result
    
    def sample_points(self, n: int = 10) -> List[np.ndarray]:
        """Sample n points from this region."""
        points = []
        
        if self.region_type == RegionType.POINT:
            return [self.sample_point.copy()]
        
        elif self.region_type == RegionType.CIRCLE:
            center = np.array(self.params["center"])
            radius = self.params["radius"]
            for i in range(n):
                angle = 2 * np.pi * i / n
                pt = np.array([
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    self.height
                ])
                points.append(pt)
        
        elif self.region_type == RegionType.LINE:
            start = np.array(self.params["start"])
            end = np.array(self.params["end"])
            for i in range(n):
                t = i / (n - 1) if n > 1 else 0.5
                pt = start + t * (end - start)
                pt[2] = self.height
                points.append(pt)
        
        elif self.region_type == RegionType.RAY:
            origin = np.array(self.params["origin"])
            direction = normalize(np.array(self.params["direction"]))
            min_dist = self.params.get("min_distance", 1.0)
            max_dist = self.params.get("max_distance", 10.0)
            for i in range(n):
                t = min_dist + (max_dist - min_dist) * i / (n - 1)
                pt = origin + direction * t
                pt[2] = self.height
                points.append(pt)
        
        elif self.region_type == RegionType.HALF_PLANE:
            # Sample points in the valid half-plane
            boundary_point = np.array(self.params["boundary_point"])
            boundary_dir = normalize(np.array(self.params["boundary_direction"]))
            normal = np.array(self.params["normal"])  # Points into valid region
            
            for i in range(n):
                # Along boundary + offset into valid region
                t_along = (i - n/2) * 1.0
                t_normal = np.random.uniform(1, 5)
                pt = boundary_point + t_along * boundary_dir + t_normal * normal
                pt = np.array([pt[0], pt[1], self.height])
                points.append(pt)
        
        elif self.region_type == RegionType.ANNULUS:
            center = np.array(self.params["center"])
            min_radius = self.params["min_radius"]
            max_radius = self.params["max_radius"]
            for i in range(n):
                angle = 2 * np.pi * i / n
                radius = (min_radius + max_radius) / 2
                pt = np.array([
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    self.height
                ])
                points.append(pt)
        
        elif self.region_type == RegionType.CURVE:
            # Return pre-computed curve points
            if "points" in self.params:
                for pt in self.params["points"][:n]:
                    points.append(np.array(pt))
        
        return points if points else [self.sample_point.copy()]


@dataclass
class BoundingBox3D:
    """3D Bounding Box representation."""
    vertices: np.ndarray  # Shape: (8, 3)
    label: str
    ins_id: str
    
    @property
    def center(self) -> np.ndarray:
        return np.mean(self.vertices, axis=0)
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        return tuple(max_coords - min_coords)
    
    @property
    def min_coords(self) -> np.ndarray:
        return np.min(self.vertices, axis=0)
    
    @property
    def max_coords(self) -> np.ndarray:
        return np.max(self.vertices, axis=0)
    
    @property
    def height(self) -> float:
        return self.dimensions[2]
    
    @property
    def width(self) -> float:
        return self.dimensions[0]
    
    @property
    def depth(self) -> float:
        return self.dimensions[1]
    
    @property
    def max_dimension(self) -> float:
        return max(self.dimensions)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """2D Euclidean distance (ignoring z)."""
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def compute_min_distance_to_object(obj: 'BoundingBox3D', absolute_min: float = 0.5) -> float:
    """
    Compute minimum safe distance to an object.
    
    The minimum distance is the maximum of:
    - The object's largest dimension (so camera can see the whole object)
    - An absolute minimum (default 0.5m, to prevent camera from being too close)
    
    Args:
        obj: The target object
        absolute_min: Absolute minimum distance in meters
        
    Returns:
        Minimum safe distance in meters
    """
    return max(obj.max_dimension, absolute_min)


def compute_min_distance_to_objects(objects: List['BoundingBox3D'], absolute_min: float = 0.5) -> float:
    """
    Compute minimum safe distance for multiple objects.
    
    For object pairs/triples, uses the maximum dimension among all objects.
    
    Args:
        objects: List of target objects
        absolute_min: Absolute minimum distance in meters
        
    Returns:
        Minimum safe distance in meters
    """
    if not objects:
        return absolute_min
    max_dim = max(obj.max_dimension for obj in objects)
    return max(max_dim, absolute_min)


def compute_forward_direction(from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
    """Compute forward direction from camera position to look-at point."""
    direction = to_pos - from_pos
    return normalize(direction)


def angle_to_preset(angle: float) -> str:
    """Convert angle (radians) to preset name (front/right/back/left)."""
    angle = angle % (2 * np.pi)
    if angle < np.pi / 4 or angle >= 7 * np.pi / 4:
        return "front"
    elif angle < 3 * np.pi / 4:
        return "right"
    elif angle < 5 * np.pi / 4:
        return "back"
    else:
        return "left"


@dataclass
class TaskResult:
    """Result of a task generation."""
    task_type: str
    task_params: Dict[str, Any]
    target_region: TargetRegion
    preset: str
    is_valid: bool
    description: str
    object_label: str = ""
    
    @property
    def sample_point(self) -> np.ndarray:
        return self.target_region.sample_point
    
    @property
    def sample_forward(self) -> np.ndarray:
        return self.target_region.sample_forward
    
    @property
    def distance(self) -> float:
        """Distance from sample point to primary object."""
        return self.target_region.params.get("sample_distance", 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_type': self.task_type,
            'task_params': TargetRegion._convert_to_native(self.task_params),
            'target_region': self.target_region.to_dict(),
            'preset': self.preset,
            'is_valid': bool(self.is_valid),
            'description': self.description,
            'object_label': self.object_label,
        }


class TaskGenerator:
    """Generates spatial navigation tasks and computes target regions."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
    
    def load_labels(self, scene_path: Path) -> Dict[str, BoundingBox3D]:
        """Load objects from labels.json as BoundingBox3D."""
        labels_path = scene_path / 'labels.json'
        if not labels_path.exists():
            return {}
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        objects = {}
        for item in data:
            if 'bounding_box' in item and (item.get('ins_id') or item.get('id')):
                bbox_points = item['bounding_box']
                vertices = np.array([[p['x'], p['y'], p['z']] for p in bbox_points])
                ins_id = str(item.get('ins_id') or item.get('id'))
                bbox = BoundingBox3D(
                    vertices=vertices,
                    label=item.get('label', ''),
                    ins_id=ins_id
                )
                objects[ins_id] = bbox
        
        return objects
    
    # =========================================================================
    # Task 1.1: Absolute Positioning
    # Region Type: CIRCLE (all points at distance d from object)
    # =========================================================================
    def generate_absolute_positioning(self, target: BoundingBox3D, 
                                       target_distance: float,
                                       agent_height: float) -> TaskResult:
        """
        Generate positions at specified distance from target object.
        
        Solution space: Circle of radius target_distance centered at object (in XY plane).
        The distance is clamped to be at least max(object's max dimension, 0.5m).
        """
        center = target.center.copy()
        center_2d = center[:2]
        
        # Compute minimum safe distance based on object size
        min_dist = compute_min_distance_to_object(target, absolute_min=0.5)
        
        # Ensure target_distance is at least min_dist
        effective_distance = max(target_distance, min_dist)
        
        # Sample one point on the circle
        angle = np.random.uniform(0, 2 * np.pi)
        sample_pt = np.array([
            center_2d[0] + effective_distance * np.cos(angle),
            center_2d[1] + effective_distance * np.sin(angle),
            agent_height
        ])
        
        # Forward direction points toward object
        forward = compute_forward_direction(sample_pt, center)
        
        # Create CIRCLE region
        region = TargetRegion(
            region_type=RegionType.CIRCLE,
            params={
                "center": center_2d,
                "radius": effective_distance,
                "object_center": center,
                "sample_distance": effective_distance,
                "min_distance": min_dist,
                "requested_distance": target_distance,
            },
            sample_point=sample_pt,
            sample_forward=forward,
            height=agent_height
        )
        
        preset = angle_to_preset(angle)
        
        return TaskResult(
            task_type='absolute_positioning',
            task_params={'target_distance': effective_distance, 'target_object': target.ins_id},
            target_region=region,
            preset=preset,
            is_valid=True,
            description=f"Move to any position {effective_distance:.1f}m from {target.label}",
            object_label=target.label
        )
    
    # =========================================================================
    # Task 1.2: Delta Control
    # Region Type: POINT (single specific position)
    # =========================================================================
    def generate_delta_control(self, target: BoundingBox3D,
                                start_pos: np.ndarray,
                                delta: float) -> TaskResult:
        """
        Generate target position by moving delta meters toward target.
        
        Solution space: Single point (move exactly delta along view direction).
        
        Note: If moving closer would result in a position too close to the object,
        the task is marked as invalid. Minimum distance = max(object's max dimension, 0.5m).
        """
        target_center = target.center.copy()
        
        # Direction from start to target (2D)
        direction_2d = normalize(target_center[:2] - start_pos[:2])
        
        # New position
        new_pos = start_pos.copy()
        new_pos[:2] = start_pos[:2] + direction_2d * delta
        
        # Forward direction points toward object
        forward = compute_forward_direction(new_pos, target_center)
        
        # Compute distance to object
        dist_to_obj = distance_2d(new_pos, target_center)
        
        # Compute minimum distance based on object size
        # min_dist = max(object's max dimension, 0.5m)
        min_dist = compute_min_distance_to_object(target, absolute_min=0.5)
        
        # For "closer" tasks (delta > 0), ensure we don't get too close
        if delta > 0 and dist_to_obj < min_dist:
            # Calculate the maximum delta that keeps us at min_dist
            current_dist = distance_2d(start_pos, target_center)
            max_delta = current_dist - min_dist
            
            if max_delta <= 0:
                # Already too close, invalid task
                return TaskResult(
                    task_type='delta_control',
                    task_params={'delta': delta, 'target_object': target.ins_id, 'start_pos': start_pos.tolist()},
                    target_region=TargetRegion(
                        region_type=RegionType.POINT,
                        params={"delta": delta, "start_position": start_pos, "sample_distance": dist_to_obj},
                        sample_point=new_pos,
                        sample_forward=forward,
                        height=new_pos[2]
                    ),
                    preset="closer",
                    is_valid=False,
                    description=f"Invalid: Would be too close ({dist_to_obj:.2f}m < {min_dist:.2f}m) to {target.label}",
                    object_label=target.label
                )
            
            # Adjust position to maintain minimum distance
            new_pos[:2] = start_pos[:2] + direction_2d * max_delta
            dist_to_obj = distance_2d(new_pos, target_center)
            forward = compute_forward_direction(new_pos, target_center)
        
        # Create POINT region
        region = TargetRegion(
            region_type=RegionType.POINT,
            params={
                "delta": delta,
                "start_position": start_pos,
                "sample_distance": dist_to_obj,
                "min_distance": min_dist,  # Store the computed min distance
                "min_distance_enforced": dist_to_obj <= min_dist + 0.1,  # Flag if min distance was applied
            },
            sample_point=new_pos,
            sample_forward=forward,
            height=new_pos[2]
        )
        
        preset = "closer" if delta > 0 else "farther"
        
        return TaskResult(
            task_type='delta_control',
            task_params={'delta': delta, 'target_object': target.ins_id, 'start_pos': start_pos.tolist()},
            target_region=region,
            preset=preset,
            is_valid=True,
            description=f"Move {abs(delta):.1f}m {'toward' if delta > 0 else 'away from'} {target.label}",
            object_label=target.label
        )
    
    # =========================================================================
    # Task 1.3: Equidistance
    # Region Type: LINE (perpendicular bisector of AB, intersected with floor)
    # =========================================================================
    def generate_equidistance(self, object_a: BoundingBox3D,
                               object_b: BoundingBox3D,
                               agent_height: float,
                               line_extent: float = 10.0) -> TaskResult:
        """
        Generate positions equidistant from both objects.
        
        Solution space: Line (perpendicular bisector of A-B in XY plane).
        Ensures sampled point is at least min_distance from objects.
        """
        center_a = object_a.center
        center_b = object_b.center
        
        # Compute minimum distance based on object sizes
        min_dist = compute_min_distance_to_objects([object_a, object_b], absolute_min=0.5)
        
        # Midpoint of A and B
        midpoint = (center_a + center_b) / 2
        midpoint_2d = midpoint[:2]
        
        # Distance from midpoint to either object
        half_dist_ab = distance_2d(center_a, midpoint) 
        
        # Minimum offset from midpoint to ensure min_dist from objects
        # Using Pythagorean theorem: offset^2 + half_dist^2 >= min_dist^2
        min_offset = np.sqrt(max(0, min_dist**2 - half_dist_ab**2))
        
        # Direction perpendicular to AB line (in XY plane)
        ab_dir = center_b[:2] - center_a[:2]
        perp_dir = normalize(np.array([-ab_dir[1], ab_dir[0]]))
        
        # Line endpoints (adjusted to ensure minimum distance)
        effective_extent = max(line_extent, min_offset + 1.0)
        line_start = np.array([
            midpoint_2d[0] - effective_extent * perp_dir[0],
            midpoint_2d[1] - effective_extent * perp_dir[1],
            agent_height
        ])
        line_end = np.array([
            midpoint_2d[0] + effective_extent * perp_dir[0],
            midpoint_2d[1] + effective_extent * perp_dir[1],
            agent_height
        ])
        
        # Sample one point on the line, ensuring it's at least min_offset from midpoint
        # This guarantees the sample is at least min_dist from both objects
        if min_offset > 0:
            # Sample from [min_offset, effective_extent] or [-effective_extent, -min_offset]
            if np.random.random() > 0.5:
                t_abs = np.random.uniform(min_offset, effective_extent)
            else:
                t_abs = np.random.uniform(-effective_extent, -min_offset)
        else:
            t_abs = np.random.uniform(-effective_extent, effective_extent)
        
        sample_pt = np.array([
            midpoint_2d[0] + t_abs * perp_dir[0],
            midpoint_2d[1] + t_abs * perp_dir[1],
            agent_height
        ])
        
        # Forward direction points toward midpoint of objects
        look_at = np.array([midpoint_2d[0], midpoint_2d[1], (center_a[2] + center_b[2]) / 2])
        forward = compute_forward_direction(sample_pt, look_at)
        
        dist_to_a = distance(sample_pt, center_a)
        
        # Create LINE region
        region = TargetRegion(
            region_type=RegionType.LINE,
            params={
                "start": line_start[:2],
                "end": line_end[:2],
                "midpoint": midpoint_2d,
                "direction": perp_dir,
                "object_a_center": center_a,
                "object_b_center": center_b,
                "sample_distance": dist_to_a,
                "min_distance": min_dist,
                "min_offset_from_midpoint": min_offset,
            },
            sample_point=sample_pt,
            sample_forward=forward,
            height=agent_height
        )
        
        return TaskResult(
            task_type='equidistance',
            task_params={'object_a': object_a.ins_id, 'object_b': object_b.ins_id},
            target_region=region,
            preset='equidistant',
            is_valid=True,
            description=f"Move to any position equidistant from {object_a.label} and {object_b.label}",
            object_label=f"{object_a.label}+{object_b.label}"
        )
    
    # =========================================================================
    # Task 2.1: Projective Relations (Left/Right)
    # Region Type: HALF_PLANE (positions where A appears left/right of B)
    # =========================================================================
    def generate_projective_relations(self, object_a: BoundingBox3D,
                                       object_b: BoundingBox3D,
                                       relation: str,
                                       agent_height: float) -> TaskResult:
        """
        Generate positions where object_a appears left/right of object_b.
        
        Solution space: Half-plane defined by the line through A and B.
        Ensures sampled point is at least min_distance from objects.
        """
        center_a = object_a.center
        center_b = object_b.center
        
        # Compute minimum distance based on object sizes
        min_dist = compute_min_distance_to_objects([object_a, object_b], absolute_min=0.5)
        
        # Line from A to B defines the boundary
        ab_dir = normalize(center_b[:2] - center_a[:2])
        
        # Normal perpendicular to AB line
        # For "left": camera should be on the left side of the AB vector
        # For "right": camera should be on the right side
        if relation == 'left':
            normal = np.array([-ab_dir[1], ab_dir[0]])  # 90° counterclockwise
        else:  # right
            normal = np.array([ab_dir[1], -ab_dir[0]])  # 90° clockwise
        
        # Boundary point is midpoint
        midpoint_2d = (center_a[:2] + center_b[:2]) / 2
        
        # Sample a valid point in the half-plane
        # Ensure offset_dist is at least min_dist to keep safe distance from objects
        offset_dist = np.random.uniform(max(min_dist, 2), max(min_dist + 4, 6))
        offset_along = np.random.uniform(-3, 3)
        
        sample_pt = np.array([
            midpoint_2d[0] + offset_dist * normal[0] + offset_along * ab_dir[0],
            midpoint_2d[1] + offset_dist * normal[1] + offset_along * ab_dir[1],
            agent_height
        ])
        
        # Forward direction points toward midpoint
        look_at = np.array([midpoint_2d[0], midpoint_2d[1], (center_a[2] + center_b[2]) / 2])
        forward = compute_forward_direction(sample_pt, look_at)
        
        dist_to_mid = distance_2d(sample_pt, midpoint_2d)
        
        # Create HALF_PLANE region
        region = TargetRegion(
            region_type=RegionType.HALF_PLANE,
            params={
                "boundary_point": midpoint_2d,
                "boundary_direction": ab_dir,
                "normal": normal,  # Points into valid region
                "object_a_center": center_a,
                "object_b_center": center_b,
                "relation": relation,
                "sample_distance": dist_to_mid,
                "min_distance": min_dist,
            },
            sample_point=sample_pt,
            sample_forward=forward,
            height=agent_height
        )
        
        return TaskResult(
            task_type='projective_relations',
            task_params={'object_a': object_a.ins_id, 'object_b': object_b.ins_id, 'relation': relation},
            target_region=region,
            preset=f"{relation}_of",
            is_valid=True,
            description=f"Position where {object_a.label} appears to the {relation} of {object_b.label}",
            object_label=f"{object_a.label}+{object_b.label}"
        )
    
    # =========================================================================
    # Task 2.2: Centering (requires 3 objects)
    # Region Type: RAY (from A through midpoint of BC, extending outward)
    # =========================================================================
    def generate_centering(self, object_a: BoundingBox3D,
                            object_b: BoundingBox3D,
                            object_c: BoundingBox3D,
                            agent_height: float) -> TaskResult:
        """
        Generate positions where object_a appears between B and C.
        
        Solution space: Ray starting from beyond the midpoint of BC,
        pointing away from A, such that A is centered.
        """
        center_a = object_a.center
        center_b = object_b.center
        center_c = object_c.center
        
        # Midpoint of B and C
        midpoint_bc = (center_b + center_c) / 2
        
        # Direction from A toward midpoint of BC
        dir_a_to_bc = normalize(midpoint_bc[:2] - center_a[:2])
        
        # Ray origin: start from beyond midpoint_bc
        ray_origin = midpoint_bc[:2] + dir_a_to_bc * 2.0
        
        # Ray direction: away from the objects (looking back at them)
        ray_direction = dir_a_to_bc
        
        # Sample a point on the ray
        t = np.random.uniform(2, 8)
        sample_pt = np.array([
            ray_origin[0] + t * ray_direction[0],
            ray_origin[1] + t * ray_direction[1],
            agent_height
        ])
        
        # Forward direction points toward A (which should be centered)
        forward = compute_forward_direction(sample_pt, center_a)
        
        dist_to_a = distance(sample_pt, center_a)
        
        # Create RAY region
        region = TargetRegion(
            region_type=RegionType.RAY,
            params={
                "origin": ray_origin,
                "direction": ray_direction,
                "min_distance": 2.0,
                "max_distance": 15.0,
                "object_a_center": center_a,
                "midpoint_bc": midpoint_bc,
                "sample_distance": dist_to_a,
            },
            sample_point=sample_pt,
            sample_forward=forward,
            height=agent_height
        )
        
        return TaskResult(
            task_type='centering',
            task_params={
                'object_a': object_a.ins_id, 
                'object_b': object_b.ins_id, 
                'object_c': object_c.ins_id
            },
            target_region=region,
            preset='center',
            is_valid=True,
            description=f"Position where {object_a.label} is centered between {object_b.label} and {object_c.label}",
            object_label=f"{object_a.label}+{object_b.label}+{object_c.label}"
        )
    
    # =========================================================================
    # Task 2.3: Occlusion Alignment
    # Region Type: RAY (from A through B, extending beyond B)
    # =========================================================================
    def generate_occlusion_alignment(self, object_a: BoundingBox3D,
                                      object_b: BoundingBox3D,
                                      min_distance: float,
                                      agent_height: float) -> TaskResult:
        """
        Generate positions where object_a is hidden behind object_b.
        
        Solution space: Ray from A through B, starting beyond B.
        """
        center_a = object_a.center
        center_b = object_b.center
        
        # Direction from A to B
        dir_a_to_b = normalize(center_b[:2] - center_a[:2])
        
        # Ray origin: just beyond B
        dist_ab = distance_2d(center_a, center_b)
        ray_origin = center_b[:2] + dir_a_to_b * (object_b.max_dimension / 2 + min_distance)
        
        # Ray direction: continuing away from A
        ray_direction = dir_a_to_b
        
        # Sample a point on the ray
        t = np.random.uniform(0, 5)
        sample_pt = np.array([
            ray_origin[0] + t * ray_direction[0],
            ray_origin[1] + t * ray_direction[1],
            agent_height
        ])
        
        # Forward direction points toward B (which occludes A)
        forward = compute_forward_direction(sample_pt, center_b)
        
        dist_to_b = distance(sample_pt, center_b)
        
        # Create RAY region
        region = TargetRegion(
            region_type=RegionType.RAY,
            params={
                "origin": ray_origin,
                "direction": ray_direction,
                "min_distance": 0.0,  # Can start right at origin
                "max_distance": 10.0,
                "object_a_center": center_a,
                "object_b_center": center_b,
                "occluder": object_b.ins_id,
                "occluded": object_a.ins_id,
                "sample_distance": dist_to_b,
            },
            sample_point=sample_pt,
            sample_forward=forward,
            height=agent_height
        )
        
        return TaskResult(
            task_type='occlusion_alignment',
            task_params={
                'object_a': object_a.ins_id, 
                'object_b': object_b.ins_id, 
                'min_distance': min_distance
            },
            target_region=region,
            preset='occluded',
            is_valid=True,
            description=f"Position where {object_a.label} is hidden behind {object_b.label}",
            object_label=f"{object_a.label}+{object_b.label}"
        )
    
    # =========================================================================
    # Task 3.1: FoV Inclusion
    # Region Type: ANNULUS (positions far enough to see both objects)
    # =========================================================================
    def generate_fov_inclusion(self, object_a: BoundingBox3D,
                                object_b: BoundingBox3D,
                                fov_horizontal: float,
                                margin: float,
                                agent_height: float) -> TaskResult:
        """
        Generate positions where both objects are visible in the frame.
        
        Solution space: Annulus around midpoint of objects 
        (must be far enough to include both in FOV).
        """
        center_a = object_a.center
        center_b = object_b.center
        
        # Calculate minimum distance to see both objects
        dist_ab = distance_2d(center_a, center_b)
        effective_span = dist_ab + object_a.max_dimension/2 + object_b.max_dimension/2
        
        fov_rad = np.radians(fov_horizontal) * (1 - 2 * margin)
        min_radius = (effective_span / 2) / np.tan(fov_rad / 2)
        max_radius = min_radius * 2.5  # Allow some flexibility
        
        midpoint_2d = (center_a[:2] + center_b[:2]) / 2
        
        # Sample a point in the annulus
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(min_radius, (min_radius + max_radius) / 2)
        
        sample_pt = np.array([
            midpoint_2d[0] + radius * np.cos(angle),
            midpoint_2d[1] + radius * np.sin(angle),
            agent_height
        ])
        
        # Forward direction points toward midpoint
        look_at = np.array([midpoint_2d[0], midpoint_2d[1], (center_a[2] + center_b[2]) / 2])
        forward = compute_forward_direction(sample_pt, look_at)
        
        # Create ANNULUS region
        region = TargetRegion(
            region_type=RegionType.ANNULUS,
            params={
                "center": midpoint_2d,
                "min_radius": min_radius,
                "max_radius": max_radius,
                "object_a_center": center_a,
                "object_b_center": center_b,
                "fov_horizontal": fov_horizontal,
                "sample_distance": radius,
            },
            sample_point=sample_pt,
            sample_forward=forward,
            height=agent_height
        )
        
        preset = angle_to_preset(angle)
        
        return TaskResult(
            task_type='fov_inclusion',
            task_params={
                'object_a': object_a.ins_id, 
                'object_b': object_b.ins_id, 
                'fov_horizontal': fov_horizontal
            },
            target_region=region,
            preset=preset,
            is_valid=True,
            description=f"Position where both {object_a.label} and {object_b.label} are visible",
            object_label=f"{object_a.label}+{object_b.label}"
        )
    
    # =========================================================================
    # Task 3.2: Size-Distance Invariance
    # Region Type: CURVE (positions where size_A/dist_A = size_B/dist_B)
    # =========================================================================
    def generate_size_distance_invariance(self, object_a: BoundingBox3D,
                                           object_b: BoundingBox3D,
                                           agent_height: float,
                                           num_curve_points: int = 20) -> TaskResult:
        """
        Generate positions where both objects appear same size on screen.
        
        For objects with different sizes, this forms a curve (Apollonius circle).
        For same-size objects, it's the perpendicular bisector (degenerate case).
        """
        center_a = object_a.center
        center_b = object_b.center
        size_a = object_a.height  # Use height as "size"
        size_b = object_b.height
        
        # The locus of points where size_a/dist_a = size_b/dist_b
        # is equivalent to dist_a/dist_b = size_a/size_b = k
        
        if abs(size_a - size_b) < 0.05:  # Nearly equal sizes
            # Degenerate case: perpendicular bisector (same as equidistance)
            return self.generate_equidistance(object_a, object_b, agent_height)
        
        k = size_a / size_b  # ratio of distances d_a/d_b
        
        # Apollonius circle: center and radius
        # Given A, B and ratio k, the locus is a circle
        # Center: (A - k^2 * B) / (1 - k^2)
        # Radius: k * |A - B| / |1 - k^2|
        
        A = center_a[:2]
        B = center_b[:2]
        dist_AB = np.linalg.norm(A - B)
        
        if abs(k - 1.0) < 0.01:
            # k ≈ 1: perpendicular bisector
            return self.generate_equidistance(object_a, object_b, agent_height)
        
        k_sq = k * k
        apollonius_center = (A - k_sq * B) / (1 - k_sq)
        apollonius_radius = k * dist_AB / abs(1 - k_sq)
        
        # Sample points on the Apollonius circle
        curve_points = []
        for i in range(num_curve_points):
            theta = 2 * np.pi * i / num_curve_points
            pt = np.array([
                apollonius_center[0] + apollonius_radius * np.cos(theta),
                apollonius_center[1] + apollonius_radius * np.sin(theta),
                agent_height
            ])
            curve_points.append(pt.tolist())
        
        # FEASIBILITY CHECK: At least some curve points must have both objects
        # within the vertical FoV. The FoV half-angle determines the max
        # downward angle from the forward direction. If objects are at floor
        # level and the camera is at agent_height on a small circle, the
        # vertical angle can exceed the FoV.
        # The env uses fov_vertical=60° (half=30°). The scorer marks objects
        # "completely invisible" when >15° beyond the edge (excess > 15°),
        # applying a severe penalty. So the effective max tolerable angle is:
        #   fov_half + completely_invisible_threshold = 30° + 15° = 45°
        # We use 90° total (45° half) to match this threshold.
        feasibility_fov = 90.0  # degrees — matches scorer's "completely invisible" boundary
        fov_v_half = np.radians(feasibility_fov / 2)
        feasible_count = 0
        for pt in curve_points:
            pt_arr = np.array(pt)
            # Forward direction is horizontal (toward midpoint of objects)
            # This matches the scorer which uses a horizontal forward vector
            mid = (center_a + center_b) / 2
            fwd_2d = mid[:2] - pt_arr[:2]
            fwd_2d_norm = np.linalg.norm(fwd_2d)
            if fwd_2d_norm < 1e-6:
                continue
            fwd_2d = fwd_2d / fwd_2d_norm
            
            # Check vertical angle to each object center from the optical axis
            # The scorer uses horizontal forward, so vertical_angle = arctan2(dz, forward_dot)
            all_in_fov = True
            for obj_center in [center_a, center_b]:
                to_obj = obj_center - pt_arr
                # Project onto forward direction (horizontal)
                forward_dot = to_obj[0] * fwd_2d[0] + to_obj[1] * fwd_2d[1]
                if forward_dot <= 0:
                    # Object is behind camera
                    all_in_fov = False
                    break
                vertical_angle = abs(np.arctan2(to_obj[2], forward_dot))
                if vertical_angle > fov_v_half:
                    all_in_fov = False
                    break
            if all_in_fov:
                feasible_count += 1
        
        if feasible_count < max(1, num_curve_points * 0.1):
            # No curve point can see both objects → task infeasible
            return TaskResult(
                task_type='size_distance_invariance',
                task_params={'object_a': object_a.ins_id, 'object_b': object_b.ins_id},
                target_region=TargetRegion(
                    region_type=RegionType.CURVE,
                    params={"center": apollonius_center, "radius": apollonius_radius},
                    sample_point=np.array(curve_points[0]),
                    sample_forward=np.array([1, 0, 0]),
                    height=agent_height
                ),
                preset='equal_size',
                is_valid=False,
                description=f"INFEASIBLE: Only {feasible_count}/{num_curve_points} curve points have both objects in FoV",
                object_label=f"{object_a.label}+{object_b.label}"
            )
        
        # Sample one feasible point
        sample_idx = np.random.randint(num_curve_points)
        sample_pt = np.array(curve_points[sample_idx])
        
        # Forward direction points toward midpoint of objects
        midpoint = (center_a + center_b) / 2
        forward = compute_forward_direction(sample_pt, midpoint)
        
        dist_to_a = distance(sample_pt, center_a)
        
        # Create CURVE region (approximated by sampled points)
        region = TargetRegion(
            region_type=RegionType.CURVE,
            params={
                "curve_type": "apollonius_circle",
                "center": apollonius_center,
                "radius": apollonius_radius,
                "points": curve_points,
                "size_ratio": k,
                "object_a_center": center_a,
                "object_b_center": center_b,
                "object_a_size": size_a,
                "object_b_size": size_b,
                "sample_distance": dist_to_a,
            },
            sample_point=sample_pt,
            sample_forward=forward,
            height=agent_height
        )
        
        return TaskResult(
            task_type='size_distance_invariance',
            task_params={'object_a': object_a.ins_id, 'object_b': object_b.ins_id},
            target_region=region,
            preset='equal_size',
            is_valid=True,
            description=f"Position where {object_a.label} and {object_b.label} appear same size",
            object_label=f"{object_a.label}+{object_b.label}"
        )
    
    # =========================================================================
    # Task 3.3: Screen Occupancy
    # Region Type: CIRCLE (positions at distance where object occupies k% of FOV)
    # =========================================================================
    def generate_screen_occupancy(self, target: BoundingBox3D,
                                   occupancy_ratio: float,
                                   fov_vertical: float,
                                   agent_height: float) -> TaskResult:
        """
        Generate positions where object occupies specified percentage of vertical FOV.
        
        Solution space: Circle at specific distance from object.
        d = (h / 2) / tan(occupancy_ratio * fov_v / 2)
        
        Distance is clamped to be at least max(object's max dimension, 0.5m).
        If the required distance < min_distance (i.e., can't get close enough),
        the task is marked as INVALID.
        """
        center = target.center.copy()
        obj_height = target.height
        
        # Compute minimum safe distance based on object size
        min_dist = compute_min_distance_to_object(target, absolute_min=0.5)
        
        # Calculate required distance for target occupancy
        fov_rad = np.radians(fov_vertical)
        target_angular_size = occupancy_ratio * fov_rad
        
        # Distance calculation: tan(angle/2) = (h/2) / d
        if target_angular_size < 0.01:
            target_angular_size = 0.01  # Minimum to avoid division by zero
        
        required_distance = (obj_height / 2) / np.tan(target_angular_size / 2)
        
        # FEASIBILITY CHECK: If the theoretical distance is less than min_dist,
        # the camera can't get close enough to achieve the target occupancy.
        # The max achievable occupancy at min_dist is:
        #   max_occ = (2 * arctan(obj_height / (2 * min_dist))) / fov_rad
        # If this is significantly below target, mark task as invalid.
        theoretical_distance = required_distance  # before clamping
        max_occupancy_at_min_dist = (2 * np.arctan(obj_height / (2 * min_dist))) / fov_rad
        
        if theoretical_distance < min_dist and max_occupancy_at_min_dist < occupancy_ratio * 0.85:
            # Task is infeasible - can't achieve target occupancy even at closest distance
            occupancy_pct = int(occupancy_ratio * 100)
            return TaskResult(
                task_type='screen_occupancy',
                task_params={
                    'target_object': target.ins_id,
                    'occupancy_ratio': occupancy_ratio,
                    'fov_vertical': fov_vertical
                },
                target_region=TargetRegion(
                    region_type=RegionType.CIRCLE,
                    params={},
                    sample_point=center,
                    sample_forward=np.array([1, 0, 0]),
                    height=agent_height
                ),
                preset=f"occupancy_{occupancy_pct}",
                is_valid=False,
                description=f"INFEASIBLE: {target.label} occupancy {occupancy_pct}% requires d={theoretical_distance:.2f}m but min_dist={min_dist:.2f}m",
                object_label=target.label
            )
        
        # Clamp to reasonable range, respecting minimum distance
        required_distance = np.clip(required_distance, min_dist, 20.0)
        
        # Sample a point on the circle
        angle = np.random.uniform(0, 2 * np.pi)
        sample_pt = np.array([
            center[0] + required_distance * np.cos(angle),
            center[1] + required_distance * np.sin(angle),
            agent_height
        ])
        
        # Forward direction points toward object
        forward = compute_forward_direction(sample_pt, center)
        
        # Create CIRCLE region
        region = TargetRegion(
            region_type=RegionType.CIRCLE,
            params={
                "center": center[:2],
                "radius": required_distance,
                "object_center": center,
                "object_height": obj_height,
                "occupancy_ratio": occupancy_ratio,
                "fov_vertical": fov_vertical,
                "sample_distance": required_distance,
                "min_distance": min_dist,
            },
            sample_point=sample_pt,
            sample_forward=forward,
            height=agent_height
        )
        
        occupancy_pct = int(occupancy_ratio * 100)
        preset = f"occupancy_{occupancy_pct}"
        
        return TaskResult(
            task_type='screen_occupancy',
            task_params={
                'target_object': target.ins_id, 
                'occupancy_ratio': occupancy_ratio,
                'fov_vertical': fov_vertical
            },
            target_region=region,
            preset=preset,
            is_valid=True,
            description=f"Position where {target.label} occupies {occupancy_pct}% of vertical FOV",
            object_label=target.label
        )
    
    # =========================================================================
    # Generate all tasks for a given set of objects
    # =========================================================================
    def generate_all_tasks(self, objects: List[BoundingBox3D],
                           camera_pose: CameraPose,
                           agent_height: float = 1.5) -> List[TaskResult]:
        """Generate all enabled tasks for the given objects and camera pose."""
        results = []
        
        if not objects:
            return results
        
        primary_obj = objects[0]
        secondary_obj = objects[1] if len(objects) > 1 else None
        tertiary_obj = objects[2] if len(objects) > 2 else None
        
        enabled = self.config.enabled_tasks
        
        # Single object tasks
        if 'absolute_positioning' in enabled:
            for dist in self.config.absolute_positioning_distances:
                result = self.generate_absolute_positioning(primary_obj, dist, agent_height)
                results.append(result)
        
        if 'delta_control' in enabled:
            start_pos = camera_pose.position
            for delta in self.config.delta_control_deltas:
                result = self.generate_delta_control(primary_obj, start_pos, delta)
                results.append(result)
        
        if 'screen_occupancy' in enabled:
            for ratio in self.config.screen_occupancy_ratios:
                result = self.generate_screen_occupancy(
                    primary_obj, ratio, self.config.fov_vertical, agent_height
                )
                results.append(result)
        
        # Two object tasks
        if secondary_obj:
            if 'equidistance' in enabled:
                result = self.generate_equidistance(primary_obj, secondary_obj, agent_height)
                results.append(result)
            
            if 'projective_relations' in enabled:
                for relation in self.config.projective_relations:
                    result = self.generate_projective_relations(
                        primary_obj, secondary_obj, relation, agent_height
                    )
                    results.append(result)
            
            if 'occlusion_alignment' in enabled:
                result = self.generate_occlusion_alignment(
                    primary_obj, secondary_obj, 
                    self.config.occlusion_min_distance, agent_height
                )
                results.append(result)
            
            if 'fov_inclusion' in enabled:
                result = self.generate_fov_inclusion(
                    primary_obj, secondary_obj,
                    self.config.fov_horizontal,
                    self.config.fov_margin,
                    agent_height
                )
                results.append(result)
            
            if 'size_distance_invariance' in enabled:
                result = self.generate_size_distance_invariance(
                    primary_obj, secondary_obj, agent_height
                )
                results.append(result)
        
        # Three object tasks
        if tertiary_obj:
            if 'centering' in enabled:
                result = self.generate_centering(
                    primary_obj, secondary_obj, tertiary_obj, agent_height
                )
                results.append(result)
        
        return results
    
    # =========================================================================
    # Task generation methods grouped by number of objects required
    # =========================================================================
    def generate_single_object_tasks(self, obj: Any, camera_pose: CameraPose,
                                      task_types: List[str],
                                      agent_height: float = 1.5) -> List[TaskResult]:
        """
        Generate tasks that require only one object.
        
        Supported tasks: absolute_positioning, delta_control, screen_occupancy
        
        Args:
            obj: SceneObject or dict with object info
            camera_pose: Current camera pose
            task_types: List of task types to generate
            agent_height: Agent eye height in meters
        
        Returns:
            List of TaskResult objects
        """
        results = []
        
        # Convert to BoundingBox3D if needed
        if hasattr(obj, 'bbox_points'):
            # SceneObject
            import numpy as np
            vertices = np.array([[p['x'], p['y'], p['z']] for p in obj.bbox_points])
            bbox = BoundingBox3D(vertices=vertices, label=obj.label, ins_id=obj.id)
        elif isinstance(obj, dict):
            # Already a dict, need to load full bbox from scene
            # For now, create a simple bbox from center
            center = np.array(obj.get('center', [0, 0, 0]))
            # Create a unit bbox around center (placeholder)
            vertices = np.array([
                center + np.array([d1*0.5, d2*0.5, d3*0.5])
                for d1 in [-1, 1] for d2 in [-1, 1] for d3 in [-1, 1]
            ])
            bbox = BoundingBox3D(vertices=vertices, label=obj.get('label', ''), ins_id=obj.get('id', ''))
        else:
            bbox = obj
        
        if 'absolute_positioning' in task_types:
            for dist in self.config.absolute_positioning_distances:
                result = self.generate_absolute_positioning(bbox, dist, agent_height)
                results.append(result)
        
        if 'delta_control' in task_types:
            start_pos = camera_pose.position
            for delta in self.config.delta_control_deltas:
                result = self.generate_delta_control(bbox, start_pos, delta)
                results.append(result)
        
        if 'screen_occupancy' in task_types:
            for ratio in self.config.screen_occupancy_ratios:
                result = self.generate_screen_occupancy(
                    bbox, ratio, self.config.fov_vertical, agent_height
                )
                results.append(result)
        
        return results
    
    def generate_two_object_tasks(self, obj_a: Any, obj_b: Any, 
                                   camera_pose: CameraPose,
                                   task_types: List[str],
                                   agent_height: float = 1.5) -> List[TaskResult]:
        """
        Generate tasks that require two objects.
        
        Supported tasks: equidistance, projective_relations, occlusion_alignment,
                        fov_inclusion, size_distance_invariance
        
        Args:
            obj_a, obj_b: SceneObject or dict with object info
            camera_pose: Current camera pose
            task_types: List of task types to generate
            agent_height: Agent eye height in meters
        
        Returns:
            List of TaskResult objects
        """
        results = []
        
        # Convert to BoundingBox3D
        def to_bbox(obj):
            if hasattr(obj, 'bbox_points'):
                vertices = np.array([[p['x'], p['y'], p['z']] for p in obj.bbox_points])
                return BoundingBox3D(vertices=vertices, label=obj.label, ins_id=obj.id)
            elif isinstance(obj, dict):
                center = np.array(obj.get('center', [0, 0, 0]))
                vertices = np.array([
                    center + np.array([d1*0.5, d2*0.5, d3*0.5])
                    for d1 in [-1, 1] for d2 in [-1, 1] for d3 in [-1, 1]
                ])
                return BoundingBox3D(vertices=vertices, label=obj.get('label', ''), ins_id=obj.get('id', ''))
            return obj
        
        bbox_a = to_bbox(obj_a)
        bbox_b = to_bbox(obj_b)
        
        if 'equidistance' in task_types:
            result = self.generate_equidistance(bbox_a, bbox_b, agent_height)
            results.append(result)
        
        if 'projective_relations' in task_types:
            for relation in self.config.projective_relations:
                result = self.generate_projective_relations(
                    bbox_a, bbox_b, relation, agent_height
                )
                results.append(result)
        
        if 'occlusion_alignment' in task_types:
            result = self.generate_occlusion_alignment(
                bbox_a, bbox_b,
                self.config.occlusion_min_distance, agent_height
            )
            results.append(result)
        
        if 'fov_inclusion' in task_types:
            result = self.generate_fov_inclusion(
                bbox_a, bbox_b,
                self.config.fov_horizontal,
                self.config.fov_margin,
                agent_height
            )
            results.append(result)
        
        if 'size_distance_invariance' in task_types:
            result = self.generate_size_distance_invariance(
                bbox_a, bbox_b, agent_height
            )
            results.append(result)
        
        return results
    
    def generate_three_object_tasks(self, obj_a: Any, obj_b: Any, obj_c: Any,
                                     camera_pose: CameraPose,
                                     task_types: List[str],
                                     agent_height: float = 1.5) -> List[TaskResult]:
        """
        Generate tasks that require three objects.
        
        Supported tasks: centering
        
        Args:
            obj_a, obj_b, obj_c: SceneObject or dict with object info
            camera_pose: Current camera pose
            task_types: List of task types to generate
            agent_height: Agent eye height in meters
        
        Returns:
            List of TaskResult objects
        """
        results = []
        
        # Convert to BoundingBox3D
        def to_bbox(obj):
            if hasattr(obj, 'bbox_points'):
                vertices = np.array([[p['x'], p['y'], p['z']] for p in obj.bbox_points])
                return BoundingBox3D(vertices=vertices, label=obj.label, ins_id=obj.id)
            elif isinstance(obj, dict):
                center = np.array(obj.get('center', [0, 0, 0]))
                vertices = np.array([
                    center + np.array([d1*0.5, d2*0.5, d3*0.5])
                    for d1 in [-1, 1] for d2 in [-1, 1] for d3 in [-1, 1]
                ])
                return BoundingBox3D(vertices=vertices, label=obj.get('label', ''), ins_id=obj.get('id', ''))
            return obj
        
        bbox_a = to_bbox(obj_a)
        bbox_b = to_bbox(obj_b)
        bbox_c = to_bbox(obj_c)
        
        if 'centering' in task_types:
            result = self.generate_centering(bbox_a, bbox_b, bbox_c, agent_height)
            results.append(result)
        
        return results
