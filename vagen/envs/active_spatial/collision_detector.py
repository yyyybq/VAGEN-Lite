"""
Collision Detection Module for Active Spatial Environment

This module provides collision detection to prevent the camera from:
1. Entering inside objects (furniture, walls, etc.)
2. Going outside room boundaries
3. Passing through walls

Design Philosophy:
==================
We use a hybrid approach:
1. Action Invalidation: If collision detected, camera stays in place (physically correct)
2. Moderate Penalty: Small negative reward (-0.1 to -0.2) to discourage collision attempts
3. Explicit Feedback: Observation includes "Action blocked by collision" message

This is better than just negative rewards because:
- Physically correct (no "ghost" walking through walls)
- Model learns to avoid collisions through penalty
- Clear feedback helps model understand what happened
"""

import json
from . import gs_io
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


COLLISION_CONVENTION_VERSION = "interiorgs_structure_label_alignment_v1"
COLLISION_CONVENTION_MIN_EVIDENCE = 20
COLLISION_CONVENTION_MIN_RELATIVE_MARGIN = 0.10


@dataclass
class CollisionResult:
    """Result of collision check."""
    has_collision: bool
    collision_type: str  # "object", "boundary", "floor", "ceiling", "none"
    collision_object: Optional[str] = None  # Label of collided object
    distance_to_collision: float = float('inf')  # Distance to nearest collision
    safe_position: Optional[np.ndarray] = None  # Suggested safe position
    details: Optional[Dict[str, Any]] = None


class AABB:
    """Axis-Aligned Bounding Box."""
    
    def __init__(self, min_point: np.ndarray, max_point: np.ndarray, label: str = "", ins_id: str = ""):
        self.min_point = np.array(min_point, dtype=np.float64)
        self.max_point = np.array(max_point, dtype=np.float64)
        self.label = label
        self.ins_id = ins_id
        
        # Ensure min < max for all dimensions
        for i in range(3):
            if self.min_point[i] > self.max_point[i]:
                self.min_point[i], self.max_point[i] = self.max_point[i], self.min_point[i]
    
    @classmethod
    def from_corners(cls, corners: List[Dict[str, float]], label: str = "", ins_id: str = "") -> 'AABB':
        """Create AABB from 8 corner points."""
        points = np.array([[c['x'], c['y'], c['z']] for c in corners])
        return cls(points.min(axis=0), points.max(axis=0), label, ins_id)
    
    def contains(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if point is inside the AABB (with optional margin)."""
        return np.all(point >= self.min_point - margin) and np.all(point <= self.max_point + margin)
    
    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate distance from point to nearest surface of AABB."""
        # Clamp point to box
        clamped = np.clip(point, self.min_point, self.max_point)
        return np.linalg.norm(point - clamped)
    
    def intersects_segment(self, start: np.ndarray, end: np.ndarray) -> Tuple[bool, float]:
        """
        Check if line segment intersects AABB.
        Returns (intersects, t) where t is the parameter at intersection (0-1).
        """
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-10:
            return self.contains(start), 0.0
        
        direction = direction / length
        
        # Ray-box intersection using slab method
        t_min = 0.0
        t_max = length
        
        for i in range(3):
            if abs(direction[i]) < 1e-10:
                # Ray parallel to slab
                if start[i] < self.min_point[i] or start[i] > self.max_point[i]:
                    return False, float('inf')
            else:
                t1 = (self.min_point[i] - start[i]) / direction[i]
                t2 = (self.max_point[i] - start[i]) / direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return False, float('inf')
        
        return True, t_min / length if length > 0 else 0.0
    
    def expand(self, margin: float) -> 'AABB':
        """Return expanded AABB by margin."""
        return AABB(
            self.min_point - margin,
            self.max_point + margin,
            self.label,
            self.ins_id
        )


class CollisionDetector:
    """
    Collision detector for active spatial environment.
    
    Handles collision detection between camera and:
    - Scene objects (from labels.json)
    - Room boundaries
    - Floor and ceiling
    """
    
    # Labels that are typically "walkable" or don't need collision (ceiling, floor, lights)
    IGNORED_LABELS = {
        # Structural/environmental
        'ceiling', 'floor', 'room', 'wall', 'niche', 'floor mold',
        # Floor coverings (can walk on)
        'rug', 'carpet', 'mat', 'doormat',
        # Overhead items (typically above camera)
        'light', 'lamp', 'chandelier', 'pendant', 'ceiling lamp', 'downlights', 
        'strip light', 'floor lamp',  # floor lamps are thin
        # Small/thin items that shouldn't block
        'curtain', 'towel', 'yarn', 'decorative painting', 'mirror',
        # Ambiguous labels
        'other',  # Too generic, often misclassified
        # Mounted items (on walls, not blocking floor space)
        'wall cabinet', 'wall design combination', 'tuyere',
    }
    
    # Labels that are solid furniture (should have collision)
    SOLID_LABELS = {
        'table', 'chair', 'sofa', 'bed', 'cabinet', 'wardrobe', 'dresser',
        'refrigerator', 'washing_machine', 'tv', 'bookshelf', 'cupboard',
        'cooking bench', 'basin cabinet', 'shoe cabinet', 'storage rack',
        'teatable', 'bedside table', 'stool', 'toilet', 'shower room partition',
    }
    
    # Labels that are walls or room boundaries
    WALL_LABELS = {'wall', 'door', 'window', 'partition', 'shower room partition'}
    
    def __init__(
        self,
        camera_radius: float = 0.15,  # Camera collision sphere radius (meters)
        floor_height: float = 0.0,    # Minimum camera height
        ceiling_height: float = 3.0,  # Maximum camera height
        safety_margin: float = 0.05,  # Additional safety margin
        enable_object_collision: bool = True,
        enable_boundary_collision: bool = True,
    ):
        """
        Initialize collision detector.
        
        Args:
            camera_radius: Radius of camera collision sphere
            floor_height: Minimum allowed camera height
            ceiling_height: Maximum allowed camera height
            safety_margin: Additional safety margin around objects
            enable_object_collision: Whether to check object collisions
            enable_boundary_collision: Whether to check floor/ceiling
        """
        self.camera_radius = camera_radius
        self.floor_height = floor_height
        self.ceiling_height = ceiling_height
        self._configured_floor_height = float(floor_height)
        self._configured_ceiling_height = float(ceiling_height)
        self.safety_margin = safety_margin
        self.enable_object_collision = enable_object_collision
        self.enable_boundary_collision = enable_boundary_collision
        
        # Scene data
        self.object_boxes: List[AABB] = []
        self.room_boundary: Optional[AABB] = None
        self.room_profiles: List[np.ndarray] = []  # Room polygons from structure.json
        self.wall_segments: List[Tuple[np.ndarray, np.ndarray]] = []  # Wall line segments
        self.structure_y_sign: float = -1.0
        self.structure_alignment_scores: Dict[str, int] = {}
        self.structure_convention_status: str = "unverified"
        self.structure_convention_version: str = COLLISION_CONVENTION_VERSION
        self.structure_alignment_evidence_count: int = 0
        self.structure_alignment_relative_margin: float = 0.0
        self._label_xy_centers: List[np.ndarray] = []
        self.scene_loaded = False
        self._current_scene_id: Optional[str] = None  # Track loaded scene to avoid reloading
    
    def load_scene(self, scene_path: Path, scene_id: Optional[str] = None) -> bool:
        """
        Load scene collision data from labels.json AND structure.json.
        
        Args:
            scene_path: Path to scene directory containing labels.json and structure.json
            scene_id: Optional scene identifier for caching (skip reload if same)
            
        Returns:
            True if loaded successfully
        """
        # Skip reload if same scene is already loaded
        if scene_id is not None and scene_id == self._current_scene_id and self.scene_loaded:
            return True
        self.floor_height = self._configured_floor_height
        self.ceiling_height = self._configured_ceiling_height
        
        labels_path = scene_path / 'labels.json'
        structure_path = scene_path / 'structure.json'
        
        success = False
        
        # Load objects from labels.json
        if labels_path.exists():
            try:
                with open(labels_path, 'r') as f:
                    labels_data = json.load(f)
                
                self.object_boxes = []
                self._label_xy_centers = []
                all_points = []
                
                for obj in labels_data:
                    label = obj.get('label', '').lower()
                    ins_id = obj.get('ins_id', '')
                    bbox = obj.get('bounding_box', [])
                    
                    if len(bbox) != 8:
                        continue
                    
                    # Create AABB from corners
                    aabb = AABB.from_corners(bbox, label, ins_id)
                    self._label_xy_centers.append((aabb.min_point[:2] + aabb.max_point[:2]) / 2.0)
                    
                    # Skip ignored labels (floor, ceiling, lights)
                    if any(ignored in label for ignored in self.IGNORED_LABELS):
                        continue
                    
                    # Expand box by camera radius + safety margin
                    expanded = aabb.expand(self.camera_radius + self.safety_margin)
                    self.object_boxes.append(expanded)
                    
                    # Collect all points for room boundary estimation
                    for corner in bbox:
                        all_points.append([corner['x'], corner['y'], corner['z']])
                
                # Estimate room boundary from all object points
                if all_points:
                    all_points = np.array(all_points)
                    room_min = all_points.min(axis=0) - 0.5
                    room_max = all_points.max(axis=0) + 0.5
                    self.room_boundary = AABB(room_min, room_max, "room_boundary", "room")
                    
                    self.floor_height = max(self.floor_height, room_min[2] + self.camera_radius)
                    self.ceiling_height = min(self.ceiling_height, room_max[2] - self.camera_radius)
                
                print(f"[CollisionDetector] Loaded {len(self.object_boxes)} collision objects from labels.json")
                success = True
                
            except Exception as e:
                print(f"[CollisionDetector] Error loading labels.json: {e}")
        
        # Load walls from structure.json (CRITICAL for wall collision!)
        if structure_path.exists():
            try:
                with open(structure_path, 'r') as f:
                    structure_data = json.load(f)
                
                self.room_profiles = []
                self.wall_segments = []
                door_segments = []
                
                y_sign = self._select_structure_y_sign(structure_data)

                # Extract room profiles using the convention best aligned with
                # this scene's label coordinates.
                for room in structure_data.get('rooms', []):
                    profile = room.get('profile', [])
                    if profile and len(profile) >= 3:
                        points = np.array([[p[0], y_sign * p[1]] for p in profile])
                        self.room_profiles.append(points)
                
                # Extract doors (openings in walls - camera can pass through)
                for hole in structure_data.get('holes', []):
                    hole_profile = hole.get('profile', [])
                    hole_type = hole.get('type', '')
                    
                    if hole_type == 'DOOR' and len(hole_profile) >= 2:
                        xs = [p[0] for p in hole_profile]
                        ys = [y_sign * p[1] for p in hole_profile]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        
                        if (max_x - min_x) > (max_y - min_y):
                            center_y = (min_y + max_y) / 2
                            door_segments.append((
                                np.array([min_x, center_y]),
                                np.array([max_x, center_y])
                            ))
                        else:
                            center_x = (min_x + max_x) / 2
                            door_segments.append((
                                np.array([center_x, min_y]),
                                np.array([center_x, max_y])
                            ))
                
                # Extract wall segments from room profiles (excluding doors)
                for room_profile in self.room_profiles:
                    n = len(room_profile)
                    for i in range(n):
                        p1 = room_profile[i].copy()
                        p2 = room_profile[(i + 1) % n].copy()
                        
                        is_door = self._segment_overlaps_door(p1, p2, door_segments)
                        if not is_door:
                            self.wall_segments.append((p1, p2))
                
                print(
                    f"[CollisionDetector] Loaded {len(self.room_profiles)} rooms, "
                    f"{len(self.wall_segments)} wall segments from structure.json "
                    f"(structure_y_sign={y_sign:+.0f}, status={self.structure_convention_status}, "
                    f"alignment={self.structure_alignment_scores})"
                )
                success = True
                
            except Exception as e:
                print(f"[CollisionDetector] Error loading structure.json: {e}")
        
        self.scene_loaded = success
        if success and scene_id is not None:
            self._current_scene_id = scene_id
        return success
    
    def _segment_overlaps_door(self, p1: np.ndarray, p2: np.ndarray,
                               door_segments: List[Tuple[np.ndarray, np.ndarray]],
                               tolerance: float = 0.5) -> bool:
        """Check if a wall segment overlaps with any door opening."""
        for door_start, door_end in door_segments:
            wall_mid = (p1 + p2) / 2
            door_mid = (door_start + door_end) / 2
            dist = np.linalg.norm(wall_mid - door_mid)
            wall_len = np.linalg.norm(p2 - p1)
            door_len = np.linalg.norm(door_end - door_start)
            
            if dist < (wall_len + door_len) / 2:
                wall_dir = (p2 - p1) / (wall_len + 1e-6)
                door_dir = (door_end - door_start) / (door_len + 1e-6)
                dot = abs(np.dot(wall_dir, door_dir))
                if dot > 0.8:
                    return True
        return False

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """Dependency-free 2D point-in-polygon used for convention inference."""
        x, y = float(point[0]), float(point[1])
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = float(polygon[i][0]), float(polygon[i][1])
            xj, yj = float(polygon[j][0]), float(polygon[j][1])
            if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
            j = i
        return inside

    def _select_structure_y_sign(self, structure_data: Dict[str, Any]) -> float:
        """Infer whether structure Y already shares label/world coordinates.

        InteriorGS exports in circulation contain both conventions.  The old
        unconditional ``-y`` transform puts valid cameras against mirrored
        walls for raw-coordinate scenes.  We choose the convention containing
        the most label-box centers; ties retain the historical ``-y`` behavior.
        """
        scores: Dict[str, int] = {}
        for sign in (1.0, -1.0):
            polygons = []
            for room in structure_data.get("rooms", []):
                profile = room.get("profile", [])
                if len(profile) >= 3:
                    polygons.append(np.asarray([[p[0], sign * p[1]] for p in profile], dtype=float))
            score = sum(
                any(self._point_in_polygon(center, polygon) for polygon in polygons)
                for center in self._label_xy_centers
            )
            scores[f"{sign:+.0f}"] = int(score)
        self.structure_alignment_scores = scores
        positive = scores.get("+1", 0)
        negative = scores.get("-1", 0)
        best = max(positive, negative)
        self.structure_alignment_evidence_count = len(self._label_xy_centers)
        self.structure_alignment_relative_margin = (
            abs(positive - negative) / max(best, 1)
        )
        sufficiently_supported = (
            self.structure_alignment_evidence_count >= COLLISION_CONVENTION_MIN_EVIDENCE
            and positive != negative
            and self.structure_alignment_relative_margin >= COLLISION_CONVENTION_MIN_RELATIVE_MARGIN
        )
        self.structure_convention_status = "frozen" if sufficiently_supported else "ambiguous"
        # A deterministic operational fallback is retained for historical rows;
        # R1 callers must reject `ambiguous` rather than treating this sign as
        # validated evidence.
        self.structure_y_sign = 1.0 if positive > negative else -1.0
        return self.structure_y_sign

    def convention_record(self) -> Dict[str, Any]:
        return {
            "version": self.structure_convention_version,
            "status": self.structure_convention_status,
            "structure_y_sign": self.structure_y_sign,
            "alignment_scores": dict(self.structure_alignment_scores),
            "evidence_label_centers": self.structure_alignment_evidence_count,
            "relative_margin": self.structure_alignment_relative_margin,
            "min_evidence": COLLISION_CONVENTION_MIN_EVIDENCE,
            "min_relative_margin": COLLISION_CONVENTION_MIN_RELATIVE_MARGIN,
        }
    
    def _point_to_segment_distance_2d(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        """Calculate the distance from a 2D point to a line segment."""
        v = seg_end - seg_start
        w = point - seg_start
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(point - seg_start)
        
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(point - seg_end)
        
        b = c1 / c2
        proj = seg_start + b * v
        return np.linalg.norm(point - proj)
    
    def _segment_intersects_wall_2d(self, p1: np.ndarray, p2: np.ndarray,
                                    wall_start: np.ndarray, wall_end: np.ndarray) -> Tuple[bool, float]:
        """
        Check if movement segment (p1 -> p2) intersects with a wall segment.
        Returns (intersects, t) where t is the parameter along p1->p2 where intersection occurs.
        """
        d1 = p2 - p1  # Direction of movement
        d2 = wall_end - wall_start  # Direction of wall
        
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        
        if abs(cross) < 1e-10:
            return False, float('inf')
        
        diff = wall_start - p1
        t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
        u = (diff[0] * d1[1] - diff[1] * d1[0]) / cross
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True, t
        
        return False, float('inf')
    
    def load_scene_from_gs_root(self, gs_root: str, scene_id: str) -> bool:
        """Load scene from GS root directory (local path or fj:s3://...)."""
        # Skip reload if same scene is already loaded
        if scene_id == self._current_scene_id and self.scene_loaded:
            return True

        if gs_io.is_remote(gs_root):
            return self._load_scene_remote(gs_root, scene_id)

        scene_path = Path(gs_root) / scene_id
        return self.load_scene(scene_path, scene_id=scene_id)

    def _load_scene_remote(self, gs_root: str, scene_id: str) -> bool:
        """Load labels/structure JSON from AOSS without requiring a local scene dir."""
        self.floor_height = self._configured_floor_height
        self.ceiling_height = self._configured_ceiling_height
        success = False
        labels_data = gs_io.load_scene_json(gs_root, scene_id, "labels.json")
        if labels_data is not None:
            try:
                self.object_boxes = []
                self._label_xy_centers = []
                all_points = []
                for obj in labels_data:
                    label = obj.get("label", "").lower()
                    ins_id = obj.get("ins_id", "")
                    bbox = obj.get("bounding_box", [])
                    if len(bbox) != 8:
                        continue
                    aabb = AABB.from_corners(bbox, label, ins_id)
                    self._label_xy_centers.append((aabb.min_point[:2] + aabb.max_point[:2]) / 2.0)
                    if any(ignored in label for ignored in self.IGNORED_LABELS):
                        continue
                    expanded = aabb.expand(self.camera_radius + self.safety_margin)
                    self.object_boxes.append(expanded)
                    for corner in bbox:
                        all_points.append([corner["x"], corner["y"], corner["z"]])
                if all_points:
                    all_points = np.array(all_points)
                    room_min = all_points.min(axis=0) - 0.5
                    room_max = all_points.max(axis=0) + 0.5
                    self.room_boundary = AABB(room_min, room_max, "room_boundary", "room")
                    self.floor_height = max(self.floor_height, room_min[2] + self.camera_radius)
                    self.ceiling_height = min(self.ceiling_height, room_max[2] - self.camera_radius)
                print(f"[CollisionDetector] Loaded {len(self.object_boxes)} collision objects from AOSS labels.json")
                success = True
            except Exception as e:
                print(f"[CollisionDetector] Error loading AOSS labels.json: {e}")

        structure_data = gs_io.load_scene_json(gs_root, scene_id, "structure.json")
        if structure_data is not None:
            try:
                self.room_profiles = []
                self.wall_segments = []
                door_segments = []
                y_sign = self._select_structure_y_sign(structure_data)
                for room in structure_data.get("rooms", []):
                    profile = room.get("profile", [])
                    if profile and len(profile) >= 3:
                        points = np.array([[p[0], y_sign * p[1]] for p in profile])
                        self.room_profiles.append(points)
                for hole in structure_data.get("holes", []):
                    hole_profile = hole.get("profile", [])
                    hole_type = hole.get("type", "")
                    if hole_type == "DOOR" and len(hole_profile) >= 2:
                        xs = [p[0] for p in hole_profile]
                        ys = [y_sign * p[1] for p in hole_profile]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        if (max_x - min_x) > (max_y - min_y):
                            center_y = (min_y + max_y) / 2
                            door_segments.append((np.array([min_x, center_y]), np.array([max_x, center_y])))
                        else:
                            center_x = (min_x + max_x) / 2
                            door_segments.append((np.array([center_x, min_y]), np.array([center_x, max_y])))
                for room_profile in self.room_profiles:
                    n = len(room_profile)
                    for i in range(n):
                        p1 = room_profile[i].copy()
                        p2 = room_profile[(i + 1) % n].copy()
                        if not self._segment_overlaps_door(p1, p2, door_segments):
                            self.wall_segments.append((p1, p2))
                print(
                    f"[CollisionDetector] Loaded {len(self.room_profiles)} rooms, "
                    f"{len(self.wall_segments)} wall segments from AOSS structure.json "
                    f"(structure_y_sign={y_sign:+.0f}, status={self.structure_convention_status}, "
                    f"alignment={self.structure_alignment_scores})"
                )
                success = True
            except Exception as e:
                print(f"[CollisionDetector] Error loading AOSS structure.json: {e}")

        self.scene_loaded = success
        if success:
            self._current_scene_id = scene_id
        return success
    
    def check_collision(
        self,
        position: np.ndarray,
        previous_position: Optional[np.ndarray] = None
    ) -> CollisionResult:
        """
        Check if camera position collides with scene.
        
        Args:
            position: Current/target camera position [x, y, z]
            previous_position: Previous position (for segment check)
            
        Returns:
            CollisionResult with collision details
        """
        position = np.array(position, dtype=np.float64)
        
        # 1. Check floor/ceiling boundaries
        if self.enable_boundary_collision:
            if position[2] < self.floor_height:
                return CollisionResult(
                    has_collision=True,
                    collision_type="floor",
                    distance_to_collision=self.floor_height - position[2],
                    safe_position=np.array([position[0], position[1], self.floor_height]),
                    details={"floor_height": self.floor_height}
                )
            
            if position[2] > self.ceiling_height:
                return CollisionResult(
                    has_collision=True,
                    collision_type="ceiling",
                    distance_to_collision=position[2] - self.ceiling_height,
                    safe_position=np.array([position[0], position[1], self.ceiling_height]),
                    details={"ceiling_height": self.ceiling_height}
                )
        
        # 2. Check object collisions
        if self.enable_object_collision and self.scene_loaded:
            for aabb in self.object_boxes:
                # Point collision check
                if aabb.contains(position):
                    return CollisionResult(
                        has_collision=True,
                        collision_type="object",
                        collision_object=aabb.label,
                        distance_to_collision=0.0,
                        details={"object_id": aabb.ins_id, "object_label": aabb.label}
                    )
                
                # Segment collision check (if previous position provided)
                if previous_position is not None:
                    prev = np.array(previous_position, dtype=np.float64)
                    intersects, t = aabb.intersects_segment(prev, position)
                    if intersects and t < 1.0:
                        # Calculate safe position (just before collision)
                        safe_t = max(0.0, t - 0.01)
                        safe_pos = prev + safe_t * (position - prev)
                        return CollisionResult(
                            has_collision=True,
                            collision_type="object",
                            collision_object=aabb.label,
                            distance_to_collision=np.linalg.norm(position - prev) * t,
                            safe_position=safe_pos,
                            details={"object_id": aabb.ins_id, "object_label": aabb.label, "intersection_t": t}
                        )
        
        # 3. Check wall collisions (from structure.json room profiles)
        if self.enable_object_collision and self.wall_segments:
            pos_2d = position[:2]
            
            # Check if position is too close to any wall
            for wall_start, wall_end in self.wall_segments:
                dist = self._point_to_segment_distance_2d(pos_2d, wall_start, wall_end)
                if dist < self.camera_radius:
                    return CollisionResult(
                        has_collision=True,
                        collision_type="wall",
                        collision_object="wall",
                        distance_to_collision=dist,
                        details={"wall_start": wall_start.tolist(), "wall_end": wall_end.tolist()}
                    )
            
            # Segment collision check for walls (if previous position provided)
            if previous_position is not None:
                prev = np.array(previous_position, dtype=np.float64)
                prev_2d = prev[:2]
                
                for wall_start, wall_end in self.wall_segments:
                    intersects, t = self._segment_intersects_wall_2d(
                        prev_2d, pos_2d, wall_start, wall_end
                    )
                    if intersects and 0 < t < 1.0:
                        safe_t = max(0.0, t - 0.05)
                        safe_pos = prev + safe_t * (position - prev)
                        return CollisionResult(
                            has_collision=True,
                            collision_type="wall",
                            collision_object="wall",
                            distance_to_collision=np.linalg.norm(position - prev) * t,
                            safe_position=safe_pos,
                            details={"intersection_t": t, "wall_start": wall_start.tolist(), "wall_end": wall_end.tolist()}
                        )
        
        # No collision
        return CollisionResult(
            has_collision=False,
            collision_type="none",
            distance_to_collision=float('inf')
        )
    
    def get_collision_penalty(self, collision_result: CollisionResult) -> float:
        """
        Calculate collision penalty based on collision type.
        
        Args:
            collision_result: Result from check_collision
            
        Returns:
            Negative penalty value (or 0 if no collision)
        """
        if not collision_result.has_collision:
            return 0.0
        
        # Different penalties for different collision types
        penalties = {
            "object": -0.15,    # Hit furniture
            "wall": -0.2,       # Hit wall (more severe)
            "boundary": -0.1,   # Hit room boundary
            "floor": -0.1,      # Too low
            "ceiling": -0.1,    # Too high
        }
        
        return penalties.get(collision_result.collision_type, -0.1)
    
    def get_nearest_object_distance(self, position: np.ndarray) -> Tuple[float, Optional[str]]:
        """
        Get distance to nearest object from position.
        
        Returns:
            (distance, object_label) tuple
        """
        if not self.scene_loaded or not self.object_boxes:
            return float('inf'), None
        
        position = np.array(position, dtype=np.float64)
        min_dist = float('inf')
        nearest_label = None
        
        for aabb in self.object_boxes:
            dist = aabb.distance_to_point(position)
            if dist < min_dist:
                min_dist = dist
                nearest_label = aabb.label
        
        return min_dist, nearest_label


def create_collision_detector(config: Dict[str, Any]) -> CollisionDetector:
    """Factory function to create CollisionDetector from config dict."""
    return CollisionDetector(
        camera_radius=config.get("camera_radius", 0.15),
        floor_height=config.get("floor_height", 0.0),
        ceiling_height=config.get("ceiling_height", 3.0),
        safety_margin=config.get("safety_margin", 0.05),
        enable_object_collision=config.get("enable_object_collision", True),
        enable_boundary_collision=config.get("enable_boundary_collision", True),
    )
