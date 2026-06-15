"""
Visibility and Occlusion Checking Module

This module provides visibility analysis for active spatial perception:
1. Field of View (FoV) Check - Is the target within camera's view frustum?
2. Occlusion Check - Is the target blocked by other objects?
3. Target Size Check - Is the target large enough to be recognizable?

These checks are crucial because reaching the "correct position" is meaningless
if the agent can't actually SEE the target object.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class VisibilityResult:
    """Result of visibility analysis."""
    is_visible: bool  # Overall visibility (in FoV AND not occluded AND large enough)
    in_fov: bool  # Target center is within camera frustum
    occlusion_ratio: float  # 0.0 = fully visible, 1.0 = fully occluded
    screen_coverage: float  # Fraction of screen covered by target (0-1)
    target_distance: float  # Distance from camera to target
    details: Dict[str, Any] = None
    
    @property
    def visibility_score(self) -> float:
        """
        Calculate overall visibility score in [0, 1].
        Combines FoV, occlusion, and size factors.
        """
        if not self.in_fov:
            return 0.0
        
        # Occlusion penalty: 1.0 when fully visible, 0.0 when fully occluded
        occlusion_score = 1.0 - self.occlusion_ratio
        
        # Size score: penalize if target is too small on screen
        # Ideal coverage is around 5-30% of screen
        if self.screen_coverage < 0.01:  # Too small to see
            size_score = 0.0
        elif self.screen_coverage < 0.05:  # Small but visible
            size_score = self.screen_coverage / 0.05
        elif self.screen_coverage > 0.8:  # Too close, filling screen
            size_score = 0.5
        else:
            size_score = 1.0
        
        return occlusion_score * size_score


class VisibilityChecker:
    """
    Checks visibility of target objects from camera viewpoint.
    
    Uses geometric calculations for:
    - Frustum culling (FoV check)
    - Ray-box intersection (occlusion check)
    - Screen projection (size check)
    """
    
    def __init__(
        self,
        fov_horizontal: float = 60.0,  # Horizontal FoV in degrees
        fov_vertical: float = 60.0,    # Vertical FoV in degrees
        near_plane: float = 0.1,       # Near clipping plane
        far_plane: float = 20.0,       # Far clipping plane
        min_visible_size: float = 0.02,  # Minimum screen coverage to be "visible"
    ):
        self.fov_horizontal = np.radians(fov_horizontal)
        self.fov_vertical = np.radians(fov_vertical)
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.min_visible_size = min_visible_size
        
        # Scene data for occlusion checking
        self.occluder_boxes: List[Dict] = []
        self.wall_segments: List[Tuple[np.ndarray, np.ndarray]] = []  # Wall line segments from structure.json
        self.room_profiles: List[np.ndarray] = []  # Room polygons for containment check
        self.scene_loaded = False
        self._current_scene_id: Optional[str] = None  # Track loaded scene to avoid reloading
    
    def load_scene(self, scene_path: Path, scene_id: Optional[str] = None) -> bool:
        """Load scene objects and walls for occlusion checking.
        
        Args:
            scene_path: Path to scene directory
            scene_id: Optional scene identifier for caching (skip reload if same)
        """
        # Skip reload if same scene is already loaded
        if scene_id is not None and scene_id == self._current_scene_id and self.scene_loaded:
            return True
        
        labels_path = scene_path / 'labels.json'
        structure_path = scene_path / 'structure.json'
        
        success = False
        
        # Load objects from labels.json
        if labels_path.exists():
            try:
                with open(labels_path, 'r') as f:
                    labels_data = json.load(f)
                
                self.occluder_boxes = []
                for obj in labels_data:
                    label = obj.get('label', '').lower()
                    bbox = obj.get('bounding_box', [])
                    
                    if len(bbox) != 8:
                        continue
                    
                    # Skip non-occluding objects
                    if label in {'ceiling', 'floor', 'room', 'other', 'light', 'lamp'}:
                        continue
                    
                    # Convert corners to min/max
                    points = np.array([[c['x'], c['y'], c['z']] for c in bbox])
                    self.occluder_boxes.append({
                        'label': label,
                        'min': points.min(axis=0),
                        'max': points.max(axis=0),
                        'center': points.mean(axis=0),
                    })
                success = True
            except Exception as e:
                print(f"[VisibilityChecker] Error loading labels.json: {e}")
        
        # Load walls from structure.json (CRITICAL for wall occlusion!)
        if structure_path.exists():
            try:
                with open(structure_path, 'r') as f:
                    structure_data = json.load(f)
                
                self.room_profiles = []
                self.wall_segments = []
                door_segments = []
                
                # Extract room profiles
                for room in structure_data.get('rooms', []):
                    profile = room.get('profile', [])
                    if profile and len(profile) >= 3:
                        # structure.json uses (x, -y) coordinate system
                        points = np.array([[p[0], -p[1]] for p in profile])
                        self.room_profiles.append(points)
                
                # Extract doors (openings in walls)
                for hole in structure_data.get('holes', []):
                    hole_profile = hole.get('profile', [])
                    hole_type = hole.get('type', '')
                    
                    if hole_type == 'DOOR' and len(hole_profile) >= 2:
                        xs = [p[0] for p in hole_profile]
                        ys = [-p[1] for p in hole_profile]
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
                        
                        # Check if this segment overlaps with a door
                        is_door = self._segment_overlaps_door(p1, p2, door_segments)
                        if not is_door:
                            self.wall_segments.append((p1, p2))
                
                print(f"[VisibilityChecker] Loaded {len(self.room_profiles)} rooms, {len(self.wall_segments)} wall segments")
                success = True
            except Exception as e:
                print(f"[VisibilityChecker] Error loading structure.json: {e}")
        
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
    
    def check_in_fov(
        self,
        camera_position: np.ndarray,
        camera_forward: np.ndarray,
        camera_up: np.ndarray,
        target_center: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """
        Check if target center is within camera's field of view.
        
        Returns:
            (in_fov, horizontal_angle, vertical_angle) 
            Angles are in radians, 0 = center of view
        """
        # Vector from camera to target
        to_target = target_center - camera_position
        distance = np.linalg.norm(to_target)
        
        if distance < 1e-6:
            return True, 0.0, 0.0  # Target at camera position
        
        to_target = to_target / distance
        
        # Calculate camera coordinate system
        forward = camera_forward / np.linalg.norm(camera_forward)
        right = np.cross(forward, camera_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Project to_target onto camera axes
        forward_dot = np.dot(to_target, forward)
        right_dot = np.dot(to_target, right)
        up_dot = np.dot(to_target, up)
        
        # Check if target is behind camera
        if forward_dot <= 0:
            return False, np.pi, np.pi
        
        # Calculate angles
        horizontal_angle = np.arctan2(right_dot, forward_dot)
        vertical_angle = np.arctan2(up_dot, forward_dot)
        
        # Check against FoV
        in_fov = (
            abs(horizontal_angle) <= self.fov_horizontal / 2 and
            abs(vertical_angle) <= self.fov_vertical / 2 and
            self.near_plane <= distance <= self.far_plane
        )
        
        return in_fov, horizontal_angle, vertical_angle
    
    def estimate_screen_coverage(
        self,
        camera_position: np.ndarray,
        target_bbox_min: np.ndarray,
        target_bbox_max: np.ndarray,
        image_width: int = 512,
        image_height: int = 512,
    ) -> float:
        """
        Estimate how much of the screen the target occupies.
        
        Returns:
            Screen coverage ratio in [0, 1]
        """
        # Target size
        target_size = target_bbox_max - target_bbox_min
        target_center = (target_bbox_min + target_bbox_max) / 2
        
        # Distance to target
        distance = np.linalg.norm(target_center - camera_position)
        if distance < 0.1:
            return 1.0  # Too close
        
        # Approximate angular size
        max_dimension = np.max(target_size)
        angular_size = 2 * np.arctan(max_dimension / (2 * distance))
        
        # Convert to screen fraction
        fov_avg = (self.fov_horizontal + self.fov_vertical) / 2
        coverage = (angular_size / fov_avg) ** 2  # Approximate area coverage
        
        return min(1.0, coverage)
    
    def check_occlusion(
        self,
        camera_position: np.ndarray,
        target_center: np.ndarray,
        target_label: str = "",
        target_id: str = "",
        target_bbox_min: Optional[np.ndarray] = None,
        target_bbox_max: Optional[np.ndarray] = None,
        num_samples: int = 9,
        use_2d_object_occlusion: bool = True,
        check_wall_occlusion: bool = True,    # Whether to check wall occlusion
        check_object_occlusion: bool = False, # Whether to check object occlusion
        skip_same_room: bool = True,          # Skip wall check if camera and target in same room
    ) -> Tuple[float, str]:
        """
        Check if target is occluded by walls OR other objects.
        
        Key insight: For object visibility, we care about whether we can see
        ANY face of the object. Using 2D occlusion (XY plane projection) is more
        appropriate because:
        1. If camera-to-target line is clear in 2D, we can see at least one face
        2. Objects beside the target (like cabinets beside a range hood) won't
           falsely occlude the front view
        3. This matches human intuition: "Can I see around the obstacle?"
        
        Wall occlusion still uses 2D line intersection (walls are vertical).
        
        Args:
            camera_position: Camera position in 3D
            target_center: Target center position in 3D
            target_label: Label of target object (to avoid self-occlusion)
            target_id: ID of target object
            target_bbox_min: Minimum corner of target bounding box (for sampling)
            target_bbox_max: Maximum corner of target bounding box (for sampling)
            num_samples: Number of sample points
            use_2d_object_occlusion: If True, use 2D projection for object occlusion
            check_wall_occlusion: If True, check wall occlusion
            check_object_occlusion: If True, check object occlusion
            skip_same_room: If True, skip wall check when camera and target are in same room
        
        Returns:
            (occlusion_ratio, occlusion_type): ratio in [0, 1], type is "wall", "object", or "none"
        """
        if not self.scene_loaded:
            return 0.0, "none"  # Assume visible if no scene data
        
        ray_origin = np.array(camera_position, dtype=np.float64)
        target_center = np.array(target_center, dtype=np.float64)
        
        # Generate sample points on target (only use center for 2D check)
        # For 2D occlusion, we only need to check if the line from camera to target
        # center is blocked in the XY plane
        if use_2d_object_occlusion:
            # Simplified 2D occlusion check - more robust for side-by-side objects
            return self._check_occlusion_2d(
                ray_origin, target_center, target_label,
                check_wall_occlusion=check_wall_occlusion,
                check_object_occlusion=check_object_occlusion,
                skip_same_room=skip_same_room,
            )
        
        # Original 3D multi-ray sampling (kept for backward compatibility)
        sample_points = self._generate_sample_points(
            target_center, target_bbox_min, target_bbox_max, num_samples
        )
        
        # Check occlusion for each sample point
        blocked_count = 0
        for sample_point in sample_points:
            if self._is_point_occluded_3d(ray_origin, sample_point, target_label):
                blocked_count += 1
        
        # Return occlusion ratio (3D check doesn't distinguish wall vs object)
        ratio = blocked_count / len(sample_points)
        return ratio, "wall_or_object" if ratio > 0.5 else "none"
    
    def _check_occlusion_2d(
        self,
        camera_position: np.ndarray,
        target_center: np.ndarray,
        target_label: str = "",
        check_wall_occlusion: bool = True,
        check_object_occlusion: bool = False,
        skip_same_room: bool = True,
    ) -> Tuple[float, str]:
        """
        Check occlusion using 2D projection (XY plane).
        
        This is more appropriate for object visibility because:
        1. Objects beside the target (in 3D) won't falsely occlude it
        2. Only objects truly "in front of" the target (in 2D view) cause occlusion
        3. Matches human intuition: "Can I see around the obstacle?"
        
        Args:
            camera_position: Camera position in 3D
            target_center: Target center position in 3D
            target_label: Label of target object (to avoid self-occlusion)
            check_wall_occlusion: If True, check wall occlusion
            check_object_occlusion: If True, check object occlusion
            skip_same_room: If True, skip wall check when camera and target are in same room
        
        Returns:
            Occlusion ratio in [0, 1], and occlusion_type ("wall", "object", or "none")
        """
        camera_2d = camera_position[:2]
        target_2d = target_center[:2]
        target_distance_2d = np.linalg.norm(target_2d - camera_2d)
        
        if target_distance_2d < 1e-6:
            return 0.0, "none"
        
        # Check if camera and target are in the same room (for skip_same_room option)
        same_room = False
        if skip_same_room:
            camera_room = self._find_containing_room(camera_2d)
            target_room = self._find_containing_room(target_2d)
            same_room = (camera_room is not None and target_room is not None and 
                        camera_room == target_room)
        
        # Check 1: Wall occlusion (walls are opaque vertical surfaces)
        # Skip if: (1) wall check disabled, or (2) same room and skip_same_room enabled
        if check_wall_occlusion and not same_room and self.wall_segments:
            for wall_start, wall_end in self.wall_segments:
                if self._line_segments_intersect_2d(camera_2d, target_2d, wall_start, wall_end):
                    return 1.0, "wall"  # Wall completely blocks view
        
        # Check 2: Object occlusion in 2D (only if enabled)
        if not check_object_occlusion:
            return 0.0, "none"  # No wall occlusion, and object occlusion disabled
        
        # An object occludes the target only if:
        # - Its 2D projection (XY bounding box) intersects the camera-target line
        # - It is closer to camera than the target (in 2D)
        # - It is "substantial" enough to block view (not a thin object)
        
        occluding_objects = 0
        total_checks = 0
        
        for box in self.occluder_boxes:
            # Skip the target object itself
            if box['label'].lower() == target_label.lower():
                continue
            
            # Get 2D bounding box of occluder
            box_min_2d = box['min'][:2]
            box_max_2d = box['max'][:2]
            box_center_2d = box['center'][:2]
            
            # Check if this object's 2D bbox intersects the camera-target line
            if self._line_intersects_rect_2d(camera_2d, target_2d, box_min_2d, box_max_2d):
                # Check if object is between camera and target
                dist_to_box = np.linalg.norm(box_center_2d - camera_2d)
                if dist_to_box < target_distance_2d - 0.1:
                    # Check if the object is substantial enough to block view
                    # (not a thin object that we can see past)
                    box_width = np.linalg.norm(box_max_2d - box_min_2d)
                    if box_width > 0.3:  # At least 30cm wide
                        occluding_objects += 1
                total_checks += 1
        
        # Return occlusion ratio based on how many objects block the view
        if total_checks == 0:
            return 0.0, "none"
        
        # If any substantial object blocks the 2D line, consider it occluded
        # But we're more lenient than before
        occlusion_ratio = min(1.0, occluding_objects * 0.5)  # Each object adds 50% occlusion
        if occlusion_ratio > 0.5:
            return occlusion_ratio, "object"
        return occlusion_ratio, "none"
    
    def _line_intersects_rect_2d(
        self,
        line_start: np.ndarray,
        line_end: np.ndarray,
        rect_min: np.ndarray,
        rect_max: np.ndarray,
    ) -> bool:
        """Check if a 2D line segment intersects with a 2D axis-aligned rectangle."""
        # Check all 4 edges of the rectangle
        edges = [
            (np.array([rect_min[0], rect_min[1]]), np.array([rect_max[0], rect_min[1]])),  # bottom
            (np.array([rect_max[0], rect_min[1]]), np.array([rect_max[0], rect_max[1]])),  # right
            (np.array([rect_max[0], rect_max[1]]), np.array([rect_min[0], rect_max[1]])),  # top
            (np.array([rect_min[0], rect_max[1]]), np.array([rect_min[0], rect_min[1]])),  # left
        ]
        
        for edge_start, edge_end in edges:
            if self._line_segments_intersect_2d(line_start, line_end, edge_start, edge_end):
                return True
        
        # Also check if line is entirely inside the rectangle
        if (rect_min[0] <= line_start[0] <= rect_max[0] and 
            rect_min[1] <= line_start[1] <= rect_max[1]):
            return True
        
        return False
    
    def _generate_sample_points(
        self,
        target_center: np.ndarray,
        target_bbox_min: Optional[np.ndarray],
        target_bbox_max: Optional[np.ndarray],
        num_samples: int,
    ) -> List[np.ndarray]:
        """
        Generate sample points on target for multi-ray occlusion checking.
        
        Args:
            target_center: Center of target
            target_bbox_min: Min corner of bounding box
            target_bbox_max: Max corner of bounding box
            num_samples: Number of samples (1, 9, or 27)
            
        Returns:
            List of 3D sample points
        """
        # If no bounding box provided, only use center
        if target_bbox_min is None or target_bbox_max is None:
            return [target_center]
        
        bbox_min = np.array(target_bbox_min, dtype=np.float64)
        bbox_max = np.array(target_bbox_max, dtype=np.float64)
        
        if num_samples == 1:
            # Center only
            return [target_center]
        
        elif num_samples <= 9:
            # Center + 8 corners of bounding box
            points = [target_center]
            
            # 8 corners
            for x in [bbox_min[0], bbox_max[0]]:
                for y in [bbox_min[1], bbox_max[1]]:
                    for z in [bbox_min[2], bbox_max[2]]:
                        points.append(np.array([x, y, z]))
            
            return points[:num_samples]
        
        else:
            # Center + corners + edge midpoints + face centers (up to 27 points)
            points = [target_center]
            
            # 8 corners
            for x in [bbox_min[0], bbox_max[0]]:
                for y in [bbox_min[1], bbox_max[1]]:
                    for z in [bbox_min[2], bbox_max[2]]:
                        points.append(np.array([x, y, z]))
            
            # 6 face centers
            mid = (bbox_min + bbox_max) / 2
            points.extend([
                np.array([bbox_min[0], mid[1], mid[2]]),  # -X face
                np.array([bbox_max[0], mid[1], mid[2]]),  # +X face
                np.array([mid[0], bbox_min[1], mid[2]]),  # -Y face
                np.array([mid[0], bbox_max[1], mid[2]]),  # +Y face
                np.array([mid[0], mid[1], bbox_min[2]]),  # -Z face
                np.array([mid[0], mid[1], bbox_max[2]]),  # +Z face
            ])
            
            # 12 edge midpoints
            for x in [bbox_min[0], bbox_max[0]]:
                for y in [bbox_min[1], bbox_max[1]]:
                    points.append(np.array([x, y, mid[2]]))
            for x in [bbox_min[0], bbox_max[0]]:
                for z in [bbox_min[2], bbox_max[2]]:
                    points.append(np.array([x, mid[1], z]))
            for y in [bbox_min[1], bbox_max[1]]:
                for z in [bbox_min[2], bbox_max[2]]:
                    points.append(np.array([mid[0], y, z]))
            
            return points[:num_samples]
    
    def _is_point_occluded_3d(
        self,
        ray_origin: np.ndarray,
        target_point: np.ndarray,
        target_label: str = "",
    ) -> bool:
        """
        Check if a single point is occluded from camera using 3D ray casting.
        
        Note: This method can produce false positives for objects beside the target.
        Consider using _check_occlusion_2d for more robust results.
        
        Args:
            ray_origin: Camera position
            target_point: Point to check visibility
            target_label: Label of target (to skip self-occlusion)
            
        Returns:
            True if point is occluded, False if visible
        """
        ray_direction = target_point - ray_origin
        target_distance = np.linalg.norm(ray_direction)
        
        if target_distance < 1e-6:
            return False
        
        ray_direction = ray_direction / target_distance
        
        # 2D positions for wall check
        camera_2d = ray_origin[:2]
        target_2d = target_point[:2]
        
        # Check 1: Wall occlusion (CRITICAL!)
        if self.wall_segments:
            for wall_start, wall_end in self.wall_segments:
                if self._line_segments_intersect_2d(camera_2d, target_2d, wall_start, wall_end):
                    return True  # Wall blocks view
        
        # Check 2: Object occlusion
        for box in self.occluder_boxes:
            # Skip the target object itself
            if box['label'].lower() == target_label.lower():
                continue
            
            # Ray-AABB intersection
            t_hit = self._ray_aabb_intersection(
                ray_origin, ray_direction,
                box['min'], box['max']
            )
            
            if t_hit is not None and t_hit < target_distance - 0.1:
                # Something is blocking the view
                return True
        
        return False
    
    def _line_segments_intersect_2d(self, p1: np.ndarray, p2: np.ndarray,
                                     p3: np.ndarray, p4: np.ndarray) -> bool:
        """Check if 2D line segment p1-p2 intersects with p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    def is_inside_any_room(self, position_2d: np.ndarray) -> bool:
        """Check if a 2D position is inside any room polygon."""
        for room in self.room_profiles:
            if self._point_in_polygon(position_2d, room):
                return True
        return False
    
    def _find_containing_room(self, position_2d: np.ndarray) -> Optional[int]:
        """Find which room contains this 2D position. Returns room index or None."""
        for i, room in enumerate(self.room_profiles):
            if self._point_in_polygon(position_2d, room):
                return i
        return None
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        if n < 3:
            return False
        inside = False
        x, y = point[0], point[1]
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def _ray_aabb_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> Optional[float]:
        """
        Ray-AABB intersection using slab method.
        
        Returns:
            Distance to intersection, or None if no intersection
        """
        t_min = 0.0
        t_max = float('inf')
        
        for i in range(3):
            if abs(ray_direction[i]) < 1e-10:
                if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                    return None
            else:
                t1 = (box_min[i] - ray_origin[i]) / ray_direction[i]
                t2 = (box_max[i] - ray_origin[i]) / ray_direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None
        
        return t_min if t_min >= 0 else None
    
    def check_visibility(
        self,
        camera_position: np.ndarray,
        camera_forward: np.ndarray,
        camera_up: np.ndarray,
        target_center: np.ndarray,
        target_bbox_min: Optional[np.ndarray] = None,
        target_bbox_max: Optional[np.ndarray] = None,
        target_label: str = "",
    ) -> VisibilityResult:
        """
        Comprehensive visibility check.
        
        Combines FoV, occlusion, and size checks.
        """
        camera_position = np.array(camera_position, dtype=np.float64)
        camera_forward = np.array(camera_forward, dtype=np.float64)
        camera_up = np.array(camera_up, dtype=np.float64)
        target_center = np.array(target_center, dtype=np.float64)
        
        # 1. FoV check
        in_fov, h_angle, v_angle = self.check_in_fov(
            camera_position, camera_forward, camera_up, target_center
        )
        
        # 2. Distance
        target_distance = np.linalg.norm(target_center - camera_position)
        
        # 3. Occlusion check (with multi-ray sampling if bbox available)
        occlusion_ratio, _occlusion_type = self.check_occlusion(
            camera_position, 
            target_center, 
            target_label,
            target_id="",
            target_bbox_min=target_bbox_min,
            target_bbox_max=target_bbox_max,
            num_samples=9,  # Use 9-point sampling for better accuracy
        )
        
        # 4. Screen coverage
        if target_bbox_min is not None and target_bbox_max is not None:
            screen_coverage = self.estimate_screen_coverage(
                camera_position,
                np.array(target_bbox_min),
                np.array(target_bbox_max),
            )
        else:
            # Estimate from distance (assume 0.5m object size)
            screen_coverage = 0.25 / max(target_distance, 0.5) ** 2
        
        # Overall visibility
        is_visible = (
            in_fov and
            occlusion_ratio < 0.5 and
            screen_coverage >= self.min_visible_size
        )
        
        return VisibilityResult(
            is_visible=is_visible,
            in_fov=in_fov,
            occlusion_ratio=occlusion_ratio,
            screen_coverage=screen_coverage,
            target_distance=target_distance,
            details={
                "horizontal_angle_deg": np.degrees(h_angle),
                "vertical_angle_deg": np.degrees(v_angle),
            }
        )


def compute_visibility_reward(
    visibility_result: VisibilityResult,
    prev_visibility: Optional[VisibilityResult] = None,
    reward_scale: float = 0.3,
) -> float:
    """
    Compute visibility-based reward.
    
    Rewards:
    - Entering FoV: +0.1
    - Reducing occlusion: proportional reward
    - Good screen coverage: bonus
    
    Args:
        visibility_result: Current visibility
        prev_visibility: Previous visibility (for delta rewards)
        reward_scale: Scale factor
        
    Returns:
        Visibility reward
    """
    current_score = visibility_result.visibility_score
    
    if prev_visibility is None:
        # First step, return absolute score scaled down
        return current_score * reward_scale * 0.5
    
    prev_score = prev_visibility.visibility_score
    
    # Delta reward
    delta = current_score - prev_score
    
    # Bonus for entering FoV
    fov_bonus = 0.0
    if visibility_result.in_fov and not prev_visibility.in_fov:
        fov_bonus = 0.1
    
    return (delta + fov_bonus) * reward_scale


def create_visibility_checker(config: Dict[str, Any]) -> VisibilityChecker:
    """Factory function to create VisibilityChecker from config."""
    return VisibilityChecker(
        fov_horizontal=config.get("fov_horizontal", 60.0),
        fov_vertical=config.get("fov_vertical", 60.0),
        near_plane=config.get("near_plane", 0.1),
        far_plane=config.get("far_plane", 20.0),
        min_visible_size=config.get("min_visible_size", 0.02),
    )
