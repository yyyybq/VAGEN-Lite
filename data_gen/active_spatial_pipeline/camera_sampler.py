"""
Camera Sampler Module

This module handles sampling valid camera poses around objects or object pairs.

Enhanced with:
- Scene boundary validation from occupancy.json
- Multi-room polygon validation
- Stricter camera position filtering to avoid placing cameras outside the scene
- Occlusion detection using ray-AABB intersection
- Camera collision detection (avoid placing camera inside furniture)
- Wall collision detection (avoid placing camera inside or too close to walls)

Note: Utility functions (projection, FOV checking, occlusion detection, etc.)
have been moved to camera_utils.py for better modularity.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import Counter

try:
    from .config import CameraSamplingConfig
    from .object_selector import SceneObject
    from .camera_utils import (
        # Data classes
        SceneBounds,
        AABB,
        CameraPose,
        # Ray-AABB intersection
        intersects_ray_aabb,
        # Camera projection
        camtoworld_from_pos_target,
        world_to_camera,
        project_point_to_image,
        point_in_image_bounds,
        aabb_corners,
        # FOV checking
        is_target_in_fov,
        check_multiple_targets_in_fov,
        check_pair_centers_in_fov,
        # Occlusion detection
        is_point_occluded_by_single_aabb,
        is_point_occluded_by_aabb_list,
        is_target_occluded,
        # Geometry utilities
        point_to_segment_distance_2d,
        distance_to_polygon_boundary,
        # Enhanced visibility
        count_visible_corners,
        calculate_projected_area_ratio,
        calculate_occlusion_area_2d,
        # Wall occlusion checking
        check_camera_forward_wall_occlusion,
        check_camera_fov_wall_occlusion,
    )
except ImportError:
    from config import CameraSamplingConfig
    from object_selector import SceneObject
    from camera_utils import (
        # Data classes
        SceneBounds,
        AABB,
        CameraPose,
        # Ray-AABB intersection
        intersects_ray_aabb,
        # Camera projection
        camtoworld_from_pos_target,
        world_to_camera,
        project_point_to_image,
        point_in_image_bounds,
        aabb_corners,
        # FOV checking
        is_target_in_fov,
        check_multiple_targets_in_fov,
        check_pair_centers_in_fov,
        # Occlusion detection
        is_point_occluded_by_single_aabb,
        is_point_occluded_by_aabb_list,
        is_target_occluded,
        # Geometry utilities
        point_to_segment_distance_2d,
        distance_to_polygon_boundary,
        # Enhanced visibility
        count_visible_corners,
        calculate_projected_area_ratio,
        calculate_occlusion_area_2d,
        # Wall occlusion checking
        check_camera_forward_wall_occlusion,
        check_camera_fov_wall_occlusion,
    )

# Import cv2 for use in CameraSampler class methods
try:
    import cv2
except ImportError:
    cv2 = None


# Note: All standalone utility functions (intersects_ray_aabb, camtoworld_from_pos_target,
# world_to_camera, project_point_to_image, aabb_corners, is_target_in_fov,
# check_multiple_targets_in_fov, is_target_occluded, point_to_segment_distance_2d,
# distance_to_polygon_boundary, count_visible_corners, calculate_projected_area_ratio,
# calculate_occlusion_area_2d, etc.) have been moved to camera_utils.py


class CameraSampler:
    """Samples valid camera poses around objects."""
    
    def __init__(self, config: CameraSamplingConfig):
        self.config = config
        self._intrinsics = None
    
    @property
    def max_camera_height(self) -> float:
        """Maximum camera height above ground (meters)."""
        return getattr(self.config, 'max_camera_height', 1.6)
    
    @property
    def min_camera_height(self) -> float:
        """Minimum camera height above ground (meters)."""
        return getattr(self.config, 'min_camera_height', 0.8)
    
    @property
    def camera_height_offset(self) -> float:
        """Height offset above object top (meters)."""
        return getattr(self.config, 'camera_height_offset', 0.1)
    
    @property
    def max_height_ratio(self) -> float:
        """Maximum allowed ratio of tallest to shortest object height."""
        return getattr(self.config, 'max_height_ratio', 2.5)
    
    @property
    def intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix K."""
        if self._intrinsics is None:
            fx = self.config.focal_length
            fy = self.config.focal_length
            cx = self.config.image_width / 2
            cy = self.config.image_height / 2
            self._intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=float)
        return self._intrinsics
    
    def compute_camera_heights_for_objects(self, obj_tops: List[float], 
                                            num_heights: int = 1) -> List[float]:
        """
        Compute camera height based on object heights.
        
        Camera height = object top height + 0.1m, capped at max_camera_height (1.6m).
        Only returns a single height value.
        
        Args:
            obj_tops: List of object top heights (z_max values)
            num_heights: Ignored, always returns single height
            
        Returns:
            List containing single camera height
        """
        if not obj_tops:
            # Fallback to default height
            return [self.max_camera_height]
        
        # Get the maximum object top height
        max_obj_top = max(obj_tops)
        
        # Camera height = object top + offset, capped at max_camera_height
        camera_height = min(max_obj_top + self.camera_height_offset, self.max_camera_height)
        
        return [camera_height]
    
    def validate_object_heights(self, obj_bbox_list: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[bool, str]:
        """
        Validate that objects in a pair/triple have compatible heights.
        
        Very relaxed constraints - mostly disabled for maximum data generation.
        The camera height will be set to see the tallest object.
        
        Args:
            obj_bbox_list: List of (bmin, bmax) tuples for each object
            
        Returns:
            (is_valid, reason) - True if objects have compatible heights
        """
        if len(obj_bbox_list) < 2:
            return True, 'ok'
        
        # Calculate object heights (z dimension)
        obj_heights = []
        for bmin, bmax in obj_bbox_list:
            height = bmax[2] - bmin[2]
            obj_heights.append(height)
        
        min_height = min(obj_heights)
        max_height = max(obj_heights)
        
        if min_height < 0.01:
            # Avoid division by zero for very flat objects
            return False, 'object_too_flat'
        
        # Almost no restriction - allow very different heights
        # The camera will be positioned to see all objects
        height_ratio = max_height / min_height
        if height_ratio > 10.0:  # Only reject extreme cases
            return False, f'height_ratio_too_large_{height_ratio:.2f}'
        
        return True, 'ok'
    
    def get_object_top_height(self, bbox: List[Dict[str, float]]) -> float:
        """
        Get the top (maximum z) height of an object from its bounding box.
        
        Args:
            bbox: List of 8 bounding box corner points
            
        Returns:
            Maximum z coordinate (top of object)
        """
        zs = [float(p.get('z', 0.0)) for p in bbox]
        return max(zs)
        return self._intrinsics
    
    def load_labels(self, scene_path: Path) -> List[Dict[str, Any]]:
        """Load labels.json from scene."""
        labels_path = scene_path / 'labels.json'
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json not found: {labels_path}")
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_scene_bounds(self, scene_path: Path) -> Optional[SceneBounds]:
        """Load scene bounds from occupancy.json."""
        occupancy_path = scene_path / 'occupancy.json'
        if not occupancy_path.exists():
            print(f"[CameraSampler] Warning: occupancy.json not found: {occupancy_path}")
            return None
        
        try:
            with open(occupancy_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return SceneBounds.from_occupancy(data)
        except Exception as e:
            print(f"[CameraSampler] Warning: Failed to load occupancy.json: {e}")
            return None
    
    def load_room_polys(self, scene_path: Path) -> List[List[List[float]]]:
        """Load room polygons from structure.json."""
        structure_path = scene_path / 'structure.json'
        if not structure_path.exists():
            return []
        
        try:
            with open(structure_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            room_polys = []
            if isinstance(data, dict):
                rooms = data.get('rooms', [])
                for room in rooms:
                    # Try different possible keys for room polygon
                    poly = None
                    for key in ['profile', 'polygon', 'floor', 'boundary']:
                        if key in room:
                            poly = room[key]
                            break
                    
                    if poly is not None and isinstance(poly, list) and len(poly) >= 3:
                        room_polys.append(poly)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key in ['profile', 'polygon', 'floor', 'boundary']:
                            if key in item:
                                room_polys.append(item[key])
                                break
            
            return room_polys
        except Exception as e:
            print(f"[CameraSampler] Warning: Failed to parse structure.json: {e}")
            return []
    
    def load_object_aabbs(self, scene_path: Path, 
                          exclude_labels: Optional[List[str]] = None) -> List[AABB]:
        """
        Load all object AABBs from labels.json for collision/occlusion detection.
        
        Args:
            scene_path: Path to scene folder
            exclude_labels: Labels to exclude (default: wall, floor, ceiling, room)
            
        Returns:
            List of AABB objects representing scene furniture/objects
        """
        if exclude_labels is None:
            exclude_labels = ['wall', 'floor', 'ceiling', 'room', 'unknown']
        
        aabbs = []
        try:
            labels = self.load_labels(scene_path)
            for obj in labels:
                if not isinstance(obj, dict):
                    continue
                
                label = str(obj.get('label', '')).lower()
                if label in exclude_labels:
                    continue
                
                bbox = obj.get('bounding_box')
                ins_id = str(obj.get('ins_id', obj.get('id', '')))
                
                if not bbox or len(bbox) < 8:
                    continue
                
                xs = [float(p.get('x', 0.0)) for p in bbox]
                ys = [float(p.get('y', 0.0)) for p in bbox]
                zs = [float(p.get('z', 0.0)) for p in bbox]
                
                bmin = np.array([min(xs), min(ys), min(zs)], dtype=float)
                bmax = np.array([max(xs), max(ys), max(zs)], dtype=float)
                
                aabbs.append(AABB(id=ins_id, label=label, bmin=bmin, bmax=bmax))
        except Exception as e:
            print(f"[CameraSampler] Warning: Failed to load object AABBs: {e}")
        
        return aabbs
    
    def load_wall_aabbs(self, scene_path: Path) -> List[AABB]:
        """
        Load wall AABBs from structure.json for wall collision detection.
        
        Each wall is modeled as a vertical rectangular prism based on:
        - location: [[x1,y1], [x2,y2]] - wall endpoints
        - thickness: wall thickness in meters
        - height: wall height in meters
        
        Returns:
            List of AABB objects representing walls
        """
        structure_path = scene_path / 'structure.json'
        walls = []
        
        if not structure_path.exists():
            return walls
        
        try:
            with open(structure_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            wall_items = data.get('walls', [])
            
            for idx, w in enumerate(wall_items):
                try:
                    loc = w.get('location', None)
                    thickness = float(w.get('thickness', 0.2) or 0.2)
                    height = float(w.get('height', 2.8) or 2.8)
                    
                    if not loc or len(loc) != 2:
                        continue
                    
                    x1, y1 = float(loc[0][0]), float(loc[0][1])
                    x2, y2 = float(loc[1][0]), float(loc[1][1])
                    
                    p1 = np.array([x1, y1], dtype=float)
                    p2 = np.array([x2, y2], dtype=float)
                    seg = p2 - p1
                    seg_len = float(np.linalg.norm(seg))
                    
                    half_thick = thickness * 0.5
                    
                    if seg_len < 1e-6:
                        # Degenerate: use a square of thickness around point
                        xs = [x1 - half_thick, x1 + half_thick]
                        ys = [y1 - half_thick, y1 + half_thick]
                    else:
                        # Calculate perpendicular direction in XY plane
                        dir_xy = seg / seg_len
                        n = np.array([-dir_xy[1], dir_xy[0]], dtype=float)
                        
                        # Rectangle corners in XY
                        q1 = p1 + n * half_thick
                        q2 = p1 - n * half_thick
                        q3 = p2 + n * half_thick
                        q4 = p2 - n * half_thick
                        
                        xs = [q1[0], q2[0], q3[0], q4[0]]
                        ys = [q1[1], q2[1], q3[1], q4[1]]
                    
                    bmin = np.array([min(xs), min(ys), 0.0], dtype=float)
                    bmax = np.array([max(xs), max(ys), height], dtype=float)
                    
                    walls.append(AABB(id=f"wall_{idx}", label='wall', bmin=bmin, bmax=bmax))
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"[CameraSampler] Warning: Failed to load wall AABBs: {e}")
        
        return walls
    
    def check_camera_collision(self, cam_pos: np.ndarray, 
                                object_aabbs: List[AABB],
                                wall_aabbs: List[AABB],
                                margin: float = 0.20) -> Tuple[bool, Optional[str]]:
        """
        Check if camera position collides with any object or wall.
        
        NOTE: margin=0.20 matches the ENV collision detector which uses
        camera_radius(0.15) + safety_margin(0.05) = 0.20m
        This ensures consistency between data generation and runtime.
        
        Args:
            cam_pos: Camera position (x, y, z)
            object_aabbs: List of furniture/object AABBs
            wall_aabbs: List of wall AABBs
            margin: Safety margin around objects (default 20cm, matches ENV)
            
        Returns:
            (has_collision, colliding_label) - True if collision detected
        """
        # Check collision with objects (with margin)
        # Using 0.20m = camera_radius(0.15) + safety_margin(0.05) to match ENV
        for aabb in object_aabbs:
            if aabb.contains_point(cam_pos, margin=margin):
                return True, f"inside_object_{aabb.label}"
        
        # Check collision with walls (with larger margin)
        wall_margin = 0.25  # Stay 25cm from walls (slightly more than objects)
        for aabb in wall_aabbs:
            if aabb.contains_point(cam_pos, margin=wall_margin):
                return True, f"inside_wall_{aabb.id}"
        
        return False, None
    
    def check_wall_distance(self, cam_pos: np.ndarray,
                            room_polys: List[List[List[float]]],
                            min_wall_distance: float = 0.3) -> Tuple[bool, float]:
        """
        Check if camera is too close to room walls (2D polygon edges).
        
        Args:
            cam_pos: Camera position
            room_polys: Room boundary polygons
            min_wall_distance: Minimum allowed distance to wall (default 30cm)
            
        Returns:
            (is_too_close, actual_distance) - True if camera is too close to wall
        """
        if not room_polys:
            return False, float('inf')
        
        x, y = float(cam_pos[0]), float(cam_pos[1])
        min_dist = float('inf')
        
        for poly in room_polys:
            if self.point_in_poly(x, y, poly):
                dist = distance_to_polygon_boundary(x, y, poly)
                min_dist = min(min_dist, dist)
        
        return min_dist < min_wall_distance, min_dist
    
    def check_occlusion(self, cam_pos: np.ndarray,
                        target_bmin: np.ndarray, target_bmax: np.ndarray,
                        occluders: List[AABB],
                        target_id: Optional[str] = None,
                        occlusion_threshold: float = 0.5,
                        min_occluder_distance: float = 0.3,
                        max_overlap_ratio: float = 0.3) -> Tuple[bool, Optional[str]]:
        """
        Check if target object is occluded by any other object.
        
        Automatically filters out occluders that are too close to or overlapping
        with the target (e.g., wall cabinets above a range hood).
        
        Args:
            cam_pos: Camera position
            target_bmin: Target object AABB minimum corner
            target_bmax: Target object AABB maximum corner
            occluders: List of potential occluding AABBs
            target_id: Target object ID to exclude from occluder list
            occlusion_threshold: Fraction of sample points that must be occluded (default 0.5)
            min_occluder_distance: Ignore occluders closer than this to target (meters)
            max_overlap_ratio: Ignore occluders with higher overlap ratio with target
            
        Returns:
            (is_occluded, occluder_label) - True if target is occluded
        """
        return is_target_occluded(cam_pos, target_bmin, target_bmax, 
                                  occluders, target_id, 
                                  sample_corners=True,
                                  occlusion_threshold=occlusion_threshold,
                                  min_occluder_distance=min_occluder_distance,
                                  max_overlap_ratio=max_overlap_ratio)
    
    def check_target_in_fov(self, cam_pos: np.ndarray, cam_target: np.ndarray,
                            target_bmin: np.ndarray, target_bmax: np.ndarray,
                            require_center: bool = True,
                            border: int = 5) -> Tuple[bool, str]:
        """
        Check if target object is within camera's field of view using full projection.
        
        This uses the camera intrinsic matrix K to accurately project the target
        object's bounding box to the image plane and check if it falls within
        the image boundaries.
        
        Args:
            cam_pos: Camera position in world coordinates
            cam_target: Camera look-at target in world coordinates  
            target_bmin: Target object AABB minimum corner
            target_bmax: Target object AABB maximum corner
            require_center: If True, object center must project to image
            border: Margin from image edges in pixels
            
        Returns:
            (is_in_fov, reason) - True if target is in FOV
        """
        return is_target_in_fov(
            self.intrinsics, cam_pos, cam_target,
            target_bmin, target_bmax,
            self.config.image_width, self.config.image_height,
            require_center=require_center,
            border=border
        )
    
    def check_multiple_targets_in_fov(self, cam_pos: np.ndarray, cam_target: np.ndarray,
                                       targets: List[Tuple[np.ndarray, np.ndarray, str]],
                                       require_all_centers: bool = True,
                                       border: int = 5) -> Tuple[bool, List[str]]:
        """
        Check if multiple target objects are all within camera's field of view.
        
        Args:
            cam_pos: Camera position in world coordinates
            cam_target: Camera look-at target in world coordinates
            targets: List of (bmin, bmax, id) tuples for each target object
            require_all_centers: If True, all object centers must project to image
            border: Margin from image edges in pixels
            
        Returns:
            (all_in_fov, failure_reasons) - True if all targets in FOV
        """
        return check_multiple_targets_in_fov(
            self.intrinsics, cam_pos, cam_target, targets,
            self.config.image_width, self.config.image_height,
            require_all_centers=require_all_centers,
            border=border
        )

    def check_visible_corners_count(self, cam_pos: np.ndarray, cam_target: np.ndarray,
                                     target_bmin: np.ndarray, target_bmax: np.ndarray,
                                     min_corners: int = 1,
                                     check_occlusion: bool = False,
                                     occluders: Optional[List[AABB]] = None,
                                     target_id: Optional[str] = None) -> Tuple[bool, int]:
        """
        Check if enough corners of the target object are visible.
        
        This provides more precise visibility control than just checking if "any" corner
        is visible. For example, you might want at least 2-3 corners visible to ensure
        good object coverage.
        
        Args:
            cam_pos: Camera position in world coordinates
            cam_target: Camera look-at target in world coordinates
            target_bmin: Target object AABB minimum corner
            target_bmax: Target object AABB maximum corner
            min_corners: Minimum number of corners that must be visible (default: 1)
            check_occlusion: If True, also verify corners are not occluded (default: False)
            occluders: List of potential occluding AABBs (used if check_occlusion=True)
            target_id: ID of target to exclude from occlusion check
            
        Returns:
            (has_enough_corners, visible_count) where:
                - has_enough_corners: True if visible_count >= min_corners
                - visible_count: Actual number of visible corners (0-8)
        """
        visible_count = count_visible_corners(
            self.intrinsics, cam_pos, cam_target,
            target_bmin, target_bmax,
            self.config.image_width, self.config.image_height,
            border=2,
            check_occlusion=check_occlusion,
            occluders=occluders,
            target_id=target_id
        )
        
        return visible_count >= min_corners, visible_count
    
    def check_projected_area(self, cam_pos: np.ndarray, cam_target: np.ndarray,
                             target_bmin: np.ndarray, target_bmax: np.ndarray,
                             min_area_ratio: float = 0.05) -> Tuple[bool, float, float]:
        """
        Check if target object's projected area is large enough in the image.
        
        This ensures the object is not too small (e.g., too far away or tiny object).
        A typical threshold is 5% (0.05) of the image area.
        
        Args:
            cam_pos: Camera position in world coordinates
            cam_target: Camera look-at target in world coordinates
            target_bmin: Target object AABB minimum corner
            target_bmax: Target object AABB maximum corner
            min_area_ratio: Minimum projected area / image area ratio (default: 0.05 = 5%)
            
        Returns:
            (is_large_enough, area_ratio, projected_pixels) where:
                - is_large_enough: True if area_ratio >= min_area_ratio
                - area_ratio: Actual projected area ratio (0.0 to 1.0)
                - projected_pixels: Approximate number of pixels covered by object
        """
        area_ratio, projected_pixels = calculate_projected_area_ratio(
            self.intrinsics, cam_pos, cam_target,
            target_bmin, target_bmax,
            self.config.image_width, self.config.image_height
        )
        
        return area_ratio >= min_area_ratio, area_ratio, projected_pixels
    
    def check_occlusion_2d(self, cam_pos: np.ndarray, cam_target: np.ndarray,
                           target_bmin: np.ndarray, target_bmax: np.ndarray,
                           occluders: List[AABB],
                           target_id: Optional[str] = None,
                           max_occlusion_ratio: float = 0.7,
                           depth_mode: str = "min") -> Tuple[bool, Dict[str, float]]:
        """
        Check occlusion using 2D image-space calculation (more accurate than 3D ray casting).
        
        This projects both target and occluders to the image plane and calculates
        pixel-level overlap, which accurately reflects what will be visible in the
        final rendered image.
        
        Requires OpenCV (cv2). Falls back to 3D ray casting if cv2 is not available.
        
        Args:
            cam_pos: Camera position in world coordinates
            cam_target: Camera look-at target in world coordinates
            target_bmin: Target object AABB minimum corner
            target_bmax: Target object AABB maximum corner
            occluders: List of potential occluding AABBs
            target_id: ID of target to exclude from occlusion check
            max_occlusion_ratio: Maximum allowed occlusion ratio (default: 0.7 = 70%)
            depth_mode: "min" or "mean" - how to calculate depth for ordering (default: "min")
            
        Returns:
            (is_acceptable, occlusion_info) where:
                - is_acceptable: True if occlusion_ratio_target <= max_occlusion_ratio
                - occlusion_info: Dictionary with detailed occlusion statistics:
                    - target_area_px: Target's projected area in pixels
                    - occluded_area_px: Occluded portion in pixels
                    - visible_area_px: Visible portion in pixels
                    - occlusion_ratio_target: Ratio of occluded to total target area
                    - occlusion_ratio_image: Ratio of occluded area to total image area
        """
        occlusion_info = calculate_occlusion_area_2d(
            self.intrinsics, cam_pos, cam_target,
            target_bmin, target_bmax,
            occluders,
            self.config.image_width, self.config.image_height,
            target_id=target_id,
            depth_mode=depth_mode
        )
        
        occlusion_ratio = occlusion_info.get('occlusion_ratio_target', 0.0)
        is_acceptable = occlusion_ratio <= max_occlusion_ratio
        
        return is_acceptable, occlusion_info

    def point_in_poly(self, x: float, y: float, poly: List[List[float]]) -> bool:
        """Check if point is inside polygon."""
        if poly is None or len(poly) < 3:
            return False
        
        px = [p[0] for p in poly]
        py = [p[1] for p in poly]
        inside = False
        n = len(poly)
        j = n - 1
        
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        
        return inside
    
    def point_in_any_room(self, x: float, y: float, room_polys: List[List[List[float]]]) -> bool:
        """Check if point is inside ANY room polygon."""
        if not room_polys:
            return True  # No room info, cannot filter
        for poly in room_polys:
            if self.point_in_poly(x, y, poly):
                return True
        return False
    
    def validate_camera_position(self, cam_pos: np.ndarray, 
                                  room_polys: List[List[List[float]]],
                                  scene_bounds: Optional[SceneBounds],
                                  target_room_idx: Optional[int] = None,
                                  require_in_target_room: bool = False) -> Tuple[bool, str]:
        """
        Validate that a camera position is valid within the scene.
        
        This performs multiple validation checks:
        1. Scene bounds check (if available) - camera must be within the 3D scene
        2. Height check - camera must be at a reasonable height
        3. Room polygon check - camera should be inside a valid room
        
        Args:
            cam_pos: Camera position (x, y, z)
            room_polys: List of room polygon coordinates
            scene_bounds: Scene boundary information from occupancy.json
            target_room_idx: Index of the target room (where objects are)
            require_in_target_room: If True, camera must be in target room specifically
            
        Returns:
            (is_valid, rejection_reason)
        """
        x, y, z = cam_pos[0], cam_pos[1], cam_pos[2]
        
        # Check 1: Scene bounds (most important for 3DGS rendering)
        if scene_bounds is not None:
            # Add margin to stay safely within bounds (shrink inward by 20cm)
            margin = 0.2  # 20cm margin - ensures camera stays well inside scene
            if not scene_bounds.contains_point_2d(x, y, margin=margin):
                return False, 'outside_scene_bounds_xy'
            
            # Check z is reasonable (not below floor or above ceiling)
            if z < scene_bounds.min_point[2] or z > scene_bounds.max_point[2] + 1.0:
                return False, 'outside_scene_bounds_z'
        
        # Check 2: Height sanity check
        if z < 0.5 or z > 3.0:  # Camera should be between 0.5m and 3m
            return False, 'invalid_height'
        
        # Check 3: Room polygon check
        if room_polys:
            if require_in_target_room and target_room_idx is not None:
                # Must be in the specific target room
                if not self.point_in_poly(x, y, room_polys[target_room_idx]):
                    return False, 'not_in_target_room'
            else:
                # Must be in at least one room
                if not self.point_in_any_room(x, y, room_polys):
                    return False, 'not_in_any_room'
        # If no room_polys and no scene_bounds, allow position (useful for triples)
        # Height check above still applies
        
        return True, 'ok'
    
    def validate_camera_position_full(self, cam_pos: np.ndarray,
                                       room_polys: List[List[List[float]]],
                                       scene_bounds: Optional[SceneBounds],
                                       object_aabbs: List[AABB],
                                       wall_aabbs: List[AABB],
                                       target_room_idx: Optional[int] = None,
                                       require_in_target_room: bool = False,
                                       check_collision: bool = True,
                                       check_wall_dist: bool = True,
                                       min_wall_distance: float = 0.25) -> Tuple[bool, str]:
        """
        Full validation of camera position including collision and wall distance checks.
        
        This performs all validation checks:
        1. Scene bounds check - camera must be within the 3D scene
        2. Height check - camera must be at a reasonable height
        3. Room polygon check - camera should be inside a valid room
        4. Object collision check - camera must not be inside any furniture
        5. Wall collision check - camera must not be inside any wall
        6. Wall distance check - camera must be at least min_wall_distance from walls
        
        Args:
            cam_pos: Camera position (x, y, z)
            room_polys: List of room polygon coordinates
            scene_bounds: Scene boundary information from occupancy.json
            object_aabbs: List of furniture/object AABBs for collision check
            wall_aabbs: List of wall AABBs for collision check
            target_room_idx: Index of the target room (where objects are)
            require_in_target_room: If True, camera must be in target room specifically
            check_collision: If True, check for collisions with objects/walls
            check_wall_dist: If True, check minimum distance to walls
            min_wall_distance: Minimum allowed distance to wall edges
            
        Returns:
            (is_valid, rejection_reason)
        """
        # First do basic validation
        is_valid, reason = self.validate_camera_position(
            cam_pos, room_polys, scene_bounds, target_room_idx, require_in_target_room
        )
        if not is_valid:
            return False, reason
        
        # Check 4 & 5: Object and wall collision
        if check_collision:
            has_collision, collision_label = self.check_camera_collision(
                cam_pos, object_aabbs, wall_aabbs
            )
            if has_collision:
                return False, collision_label
        
        # Check 6: Wall distance (using room polygon edges)
        if check_wall_dist and room_polys:
            is_too_close, actual_dist = self.check_wall_distance(
                cam_pos, room_polys, min_wall_distance
            )
            if is_too_close:
                return False, f'too_close_to_wall_{actual_dist:.2f}m'
        
        return True, 'ok'
    
    def validate_camera_complete(self, cam_pos: np.ndarray,
                                  cam_target: np.ndarray,
                                  room_polys: List[List[List[float]]],
                                  scene_bounds: Optional[SceneBounds],
                                  object_aabbs: List[AABB],
                                  wall_aabbs: List[AABB],
                                  target_objects: List[Tuple[np.ndarray, np.ndarray, str]],
                                  target_room_idx: Optional[int] = None,
                                  require_in_target_room: bool = False,
                                  check_collision: bool = True,
                                  check_wall_dist: bool = True,
                                  check_fov: bool = True,
                                  check_occlusion: bool = True,
                                  min_wall_distance: float = 0.25,
                                  occlusion_threshold: float = 0.5,
                                  fov_border: int = 5) -> Tuple[bool, str]:
        """
        Complete validation of camera position including FOV and occlusion checks.
        
        This performs ALL validation checks:
        1. Scene bounds check - camera must be within the 3D scene
        2. Height check - camera must be at a reasonable height
        3. Room polygon check - camera should be inside a valid room
        4. Object collision check - camera must not be inside any furniture
        5. Wall collision check - camera must not be inside any wall
        6. Wall distance check - camera must be at least min_wall_distance from walls
        7. FOV check - all target objects must project to image bounds
        8. Occlusion check - target objects must not be occluded by other objects
        
        Args:
            cam_pos: Camera position (x, y, z)
            cam_target: Camera look-at target point
            room_polys: List of room polygon coordinates
            scene_bounds: Scene boundary information from occupancy.json
            object_aabbs: List of furniture/object AABBs for collision check
            wall_aabbs: List of wall AABBs for collision check
            target_objects: List of (bmin, bmax, id) for each target object
            target_room_idx: Index of the target room (where objects are)
            require_in_target_room: If True, camera must be in target room specifically
            check_collision: If True, check for collisions with objects/walls
            check_wall_dist: If True, check minimum distance to walls
            check_fov: If True, check that all targets are in camera FOV
            check_occlusion: If True, check that targets are not occluded
            min_wall_distance: Minimum allowed distance to wall edges
            occlusion_threshold: Fraction of sample points for occlusion check
            fov_border: Margin from image edges in pixels for FOV check
            
        Returns:
            (is_valid, rejection_reason)
        """
        # First do position validation (bounds, collision, wall distance)
        is_valid, reason = self.validate_camera_position_full(
            cam_pos, room_polys, scene_bounds, object_aabbs, wall_aabbs,
            target_room_idx, require_in_target_room,
            check_collision, check_wall_dist, min_wall_distance
        )
        if not is_valid:
            return False, reason
        
        # Check 7: FOV - all target objects must be visible in image
        if check_fov and target_objects:
            all_in_fov, failures = self.check_multiple_targets_in_fov(
                cam_pos, cam_target, target_objects,
                require_all_centers=True, border=fov_border
            )
            if not all_in_fov:
                return False, f'not_in_fov:{failures[0]}'
        
        # Check 8: Occlusion - targets should not be occluded by other objects
        # Note: Automatically filters out occluders that overlap with or are very
        # close to the target (e.g., wall cabinet above range hood)
        if check_occlusion and target_objects:
            all_occluders = object_aabbs + wall_aabbs
            for bmin, bmax, obj_id in target_objects:
                is_occluded, occluder = self.check_occlusion(
                    cam_pos, bmin, bmax, all_occluders, obj_id, occlusion_threshold,
                    min_occluder_distance=0.3, max_overlap_ratio=0.3
                )
                if is_occluded:
                    return False, f'occluded_by_{occluder}'
        
        return True, 'ok'

    def find_obj_in_labels(self, labels: List[Dict[str, Any]], ins_id: str) -> Optional[Dict[str, Any]]:
        """Find object by instance ID."""
        for item in labels:
            if not isinstance(item, dict):
                continue
            iid = str(item.get('ins_id') or item.get('id') or '')
            if iid == str(ins_id):
                return item
        return None
    
    def get_bbox_minmax(self, bbox_points: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Get AABB min/max from bbox points."""
        xs = [float(p['x']) for p in bbox_points]
        ys = [float(p['y']) for p in bbox_points]
        zs = [float(p['z']) for p in bbox_points]
        return (
            np.array([min(xs), min(ys), min(zs)], dtype=float),
            np.array([max(xs), max(ys), max(zs)], dtype=float)
        )
    
    def get_bbox_center(self, bbox_points: List[Dict[str, Any]]) -> np.ndarray:
        """Get center of bounding box."""
        xs = [float(p['x']) for p in bbox_points]
        ys = [float(p['y']) for p in bbox_points]
        zs = [float(p['z']) for p in bbox_points]
        return np.array([np.mean(xs), np.mean(ys), np.mean(zs)], dtype=float)
    
    def calculate_auto_radii(self, bbox_a: List[Dict], bbox_b: List[Dict]) -> Tuple[List[float], float]:
        """Calculate candidate camera distances based on object size and FOV."""
        min_a, max_a = self.get_bbox_minmax(bbox_a)
        min_b, max_b = self.get_bbox_minmax(bbox_b)
        
        union_min = np.minimum(min_a, min_b)
        union_max = np.maximum(max_a, max_b)
        
        world_size = float(np.linalg.norm(union_max - union_min))
        focal_length = self.intrinsics[0, 0]
        fov_factor = focal_length / self.config.image_width
        
        safety_margin = 1.2
        base_dist = max(world_size * fov_factor * safety_margin, 0.5)
        
        radii = [
            base_dist * 0.8,
            base_dist,
            base_dist * 1.3,
            base_dist * 1.7,
            base_dist * 2.2,
            base_dist * 2.8,
        ]
        
        return radii, base_dist
    
    def point_to_aabb_distance(self, point: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
        """Distance from point to AABB."""
        closest = np.minimum(np.maximum(point, bmin), bmax)
        return float(np.linalg.norm(point - closest))
    
    def check_visibility(self, cam_pos: np.ndarray, cam_target: np.ndarray,
                         bbox_min: np.ndarray, bbox_max: np.ndarray) -> Tuple[bool, str]:
        """
        Check if object is visible from camera position.
        
        This is a simplified visibility check. For full occlusion checking,
        you would need to load scene geometry and perform ray casting.
        """
        # Basic checks
        to_target = cam_target - cam_pos
        dist = np.linalg.norm(to_target)
        
        if dist < 0.1:
            return False, 'too_close'
        
        # Check if bbox center is in front of camera
        bbox_center = (bbox_min + bbox_max) / 2
        to_bbox = bbox_center - cam_pos
        forward = to_target / dist
        
        dot = np.dot(to_bbox, forward)
        if dot <= 0:
            return False, 'behind_camera'
        
        # Check approximate field of view (simplified)
        angle = np.arccos(np.clip(dot / (np.linalg.norm(to_bbox) + 1e-9), -1, 1))
        half_fov = np.radians(45)  # Approximate horizontal half-FOV
        
        if angle > half_fov:
            return False, 'outside_fov'
        
        return True, 'ok'
    
    def sample_camera_for_single(self, scene_path: Path, obj: Dict[str, Any],
                                  num_samples: int = 5,
                                  enable_occlusion_check: bool = True,
                                  enable_collision_check: bool = True) -> List[CameraPose]:
        """
        Sample camera poses for a single object with enhanced validation.
        
        Args:
            scene_path: Path to scene folder
            obj: Object dict with 'id' field
            num_samples: Number of camera poses to sample
            enable_occlusion_check: If True, check for occlusion by other objects
            enable_collision_check: If True, check for camera collision with objects/walls
            
        Returns:
            List of valid CameraPose objects
        """
        labels = self.load_labels(scene_path)
        room_polys = self.load_room_polys(scene_path)
        scene_bounds = self.load_scene_bounds(scene_path)
        
        # Load AABBs for collision and occlusion checks
        object_aabbs = []
        wall_aabbs = []
        if enable_collision_check or enable_occlusion_check:
            object_aabbs = self.load_object_aabbs(scene_path)
            wall_aabbs = self.load_wall_aabbs(scene_path)
        
        obj_data = self.find_obj_in_labels(labels, obj['id'])
        if obj_data is None:
            print(f"[CameraSampler] Warning: Object {obj['id']} not found in labels")
            return []
        
        bbox = obj_data['bounding_box']
        obj_min, obj_max = self.get_bbox_minmax(bbox)
        obj_center = self.get_bbox_center(bbox)
        obj_id = str(obj['id'])
        
        # Get object top height for camera height calculation
        obj_top = self.get_object_top_height(bbox)
        
        # Compute camera height: object top + 0.1m, capped at 1.6m
        camera_heights = self.compute_camera_heights_for_objects([obj_top])
        
        # Calculate appropriate distance
        obj_size = np.linalg.norm(obj_max - obj_min)
        focal_length = self.intrinsics[0, 0]
        fov_factor = focal_length / self.config.image_width
        base_dist = max(obj_size * fov_factor * 1.5, 1.0)
        
        # Limit radii based on scene bounds to avoid going outside
        max_radius = base_dist * 3.0  # Increased from 2.0 to allow farther init views
        if scene_bounds is not None:
            # Calculate max radius that keeps camera within scene bounds
            scene_size = np.linalg.norm(scene_bounds.max_point[:2] - scene_bounds.min_point[:2])
            # Use a larger fraction of scene size as max radius
            max_radius = min(max_radius, scene_size * 0.5)  # Increased from 0.4
        
        # More diverse radii: include farther distances for harder initial positions
        radii = [
            base_dist * 0.8,
            base_dist,
            base_dist * 1.3,
            base_dist * 1.7,
            min(base_dist * 2.2, max_radius),
            min(base_dist * 2.8, max_radius),
        ]
        radii = sorted(set(r for r in radii if r <= max_radius))  # Deduplicate and filter
        
        # Find room for object
        target_room_idx = None
        for i, poly in enumerate(room_polys):
            if self.point_in_poly(obj_center[0], obj_center[1], poly):
                target_room_idx = i
                break
        
        valid_poses = []
        rejection_stats = {}
        cfg = self.config
        
        # Combine object and wall AABBs for occlusion checking
        all_occluders = object_aabbs + wall_aabbs
        
        for r in radii:
            for z in camera_heights:  # Use dynamic camera heights instead of cfg.camera_heights
                for yaw in np.linspace(0, 2 * np.pi, cfg.per_angle, endpoint=False):
                    cam_pos = np.array([
                        obj_center[0] + r * math.cos(yaw),
                        obj_center[1] + r * math.sin(yaw),
                        z
                    ], dtype=float)
                    
                    cam_target = np.array([obj_center[0], obj_center[1], obj_center[2]], dtype=float)
                    
                    # Full validation: position + collision + wall distance
                    is_valid_pos, reject_reason = self.validate_camera_position_full(
                        cam_pos, room_polys, scene_bounds, 
                        object_aabbs if enable_collision_check else [],
                        wall_aabbs if enable_collision_check else [],
                        target_room_idx,
                        require_in_target_room=True,
                        check_collision=enable_collision_check,
                        check_wall_dist=enable_collision_check
                    )
                    
                    if not is_valid_pos:
                        rejection_stats[reject_reason] = rejection_stats.get(reject_reason, 0) + 1
                        continue
                    
                    # Check precise FOV using camera projection
                    is_in_fov, fov_reason = self.check_target_in_fov(
                        cam_pos, cam_target, obj_min, obj_max, 
                        require_center=True, border=5
                    )
                    if not is_in_fov:
                        rejection_stats[f'fov_{fov_reason}'] = \
                            rejection_stats.get(f'fov_{fov_reason}', 0) + 1
                        continue
                    
                    # ========== NEW: Enhanced Visibility Checks ==========
                    # Check 1: Visible corners count
                    has_enough_corners, corner_count = self.check_visible_corners_count(
                        cam_pos, cam_target, obj_min, obj_max, min_corners=1,
                        check_occlusion=False
                    )
                    
                    if not has_enough_corners:
                        rejection_stats[f'insufficient_corners_{corner_count}'] = \
                            rejection_stats.get(f'insufficient_corners_{corner_count}', 0) + 1
                        continue
                    
                    # Check 2: Projected area ratio
                    is_large_enough, area_ratio, _ = self.check_projected_area(
                        cam_pos, cam_target, obj_min, obj_max, min_area_ratio=0.05
                    )
                    
                    if not is_large_enough:
                        rejection_stats[f'area_too_small_{area_ratio:.3f}'] = \
                            rejection_stats.get(f'area_too_small_{area_ratio:.3f}', 0) + 1
                        continue
                    
                    # Check 3a: 2D image-space occlusion (preferred if cv2 available)
                    if enable_occlusion_check and all_occluders and cv2 is not None:
                        is_acceptable, occ_info = self.check_occlusion_2d(
                            cam_pos, cam_target, obj_min, obj_max, all_occluders,
                            target_id=obj_id, max_occlusion_ratio=0.7
                        )
                        
                        if not is_acceptable:
                            occ_ratio = occ_info.get('occlusion_ratio_target', 0.0)
                            rejection_stats[f'occluded_2d_{occ_ratio:.2f}'] = \
                                rejection_stats.get(f'occluded_2d_{occ_ratio:.2f}', 0) + 1
                            continue
                    
                    # Check 3b: Fallback to 3D occlusion check
                    elif enable_occlusion_check and all_occluders:
                        is_occluded, occluder_label = self.check_occlusion(
                            cam_pos, obj_min, obj_max, all_occluders, obj_id
                        )
                        if is_occluded:
                            rejection_stats[f'occluded_3d_by_{occluder_label}'] = \
                                rejection_stats.get(f'occluded_3d_by_{occluder_label}', 0) + 1
                            continue
                    # ========== END: Enhanced Visibility Checks ==========
                    
                    # ========== NEW: Check camera forward direction wall occlusion ==========
                    # Calculate camera forward direction based on look-at target
                    cam_forward = cam_target - cam_pos
                    cam_forward = cam_forward / (np.linalg.norm(cam_forward) + 1e-8)
                    
                    # Check if camera's forward view is blocked by a wall within min distance
                    is_view_blocked, nearest_wall_dist, blocking_wall = check_camera_forward_wall_occlusion(
                        cam_pos=cam_pos,
                        cam_forward=cam_forward,
                        wall_aabbs=wall_aabbs,
                        min_clear_distance=0.5,  # Must be able to see at least 0.5m ahead
                        max_check_distance=3.0
                    )
                    
                    if is_view_blocked:
                        rejection_stats[f'forward_wall_blocked_{nearest_wall_dist:.2f}m'] = \
                            rejection_stats.get(f'forward_wall_blocked_{nearest_wall_dist:.2f}m', 0) + 1
                        continue
                    # ========== END: Camera forward wall occlusion check ==========
                    
                    pose = CameraPose(
                        position=cam_pos,
                        target=cam_target,
                        yaw=float(yaw),
                        radius=float(r)
                    )
                    valid_poses.append(pose)
                    
                    if len(valid_poses) >= num_samples * 3:
                        break
                if len(valid_poses) >= num_samples * 3:
                    break
            if len(valid_poses) >= num_samples * 3:
                break
        
        # Log rejection stats if no valid poses found
        if len(valid_poses) == 0 and rejection_stats:
            print(f"[CameraSampler] Warning: No valid camera poses for object {obj.get('label', obj['id'])}. "
                  f"Rejection reasons: {rejection_stats}")
        
        # Randomly select requested number of samples
        if len(valid_poses) > num_samples:
            indices = np.random.choice(len(valid_poses), num_samples, replace=False)
            valid_poses = [valid_poses[i] for i in indices]
        
        return valid_poses
    
    def sample_camera_for_pair(self, scene_path: Path, obj_a: Dict[str, Any], obj_b: Dict[str, Any],
                                num_samples: int = 5,
                                enable_occlusion_check: bool = True,
                                enable_collision_check: bool = True) -> List[CameraPose]:
        """
        Sample camera poses for an object pair with enhanced validation.
        
        Args:
            scene_path: Path to scene folder
            obj_a: First object dict with 'id' field
            obj_b: Second object dict with 'id' field
            num_samples: Number of camera poses to sample
            enable_occlusion_check: If True, check for occlusion by other objects
            enable_collision_check: If True, check for camera collision with objects/walls
            
        Returns:
            List of valid CameraPose objects
        """
        labels = self.load_labels(scene_path)
        room_polys = self.load_room_polys(scene_path)
        scene_bounds = self.load_scene_bounds(scene_path)
        
        # Load AABBs for collision and occlusion checks
        object_aabbs = []
        wall_aabbs = []
        if enable_collision_check or enable_occlusion_check:
            object_aabbs = self.load_object_aabbs(scene_path)
            wall_aabbs = self.load_wall_aabbs(scene_path)
        
        la = self.find_obj_in_labels(labels, obj_a['id'])
        lb = self.find_obj_in_labels(labels, obj_b['id'])
        
        if la is None or lb is None:
            print(f"[CameraSampler] Warning: One or both objects not found in labels: "
                  f"{obj_a['id']}, {obj_b['id']}")
            return []
        
        a_min, a_max = self.get_bbox_minmax(la['bounding_box'])
        b_min, b_max = self.get_bbox_minmax(lb['bounding_box'])
        a_center = self.get_bbox_center(la['bounding_box'])
        b_center = self.get_bbox_center(lb['bounding_box'])
        obj_a_id = str(obj_a['id'])
        obj_b_id = str(obj_b['id'])
        
        # Calculate object heights for logging
        height_a = a_max[2] - a_min[2]
        height_b = b_max[2] - b_min[2]
        
        # Validate object height compatibility
        is_height_valid, height_reason = self.validate_object_heights(
            [(a_min, a_max), (b_min, b_max)]
        )
        if not is_height_valid:
            label_a = la.get('label', obj_a.get('label', obj_a_id))
            label_b = lb.get('label', obj_b.get('label', obj_b_id))
            print(f"[CameraSampler] Skipping pair due to height incompatibility: {height_reason}")
            print(f"    Object A: '{label_a}' (height={height_a:.3f}m)")
            print(f"    Object B: '{label_b}' (height={height_b:.3f}m)")
            return []
        
        # Get object top heights for camera height calculation
        a_top = self.get_object_top_height(la['bounding_box'])
        b_top = self.get_object_top_height(lb['bounding_box'])
        
        # Compute camera height: max object top + 0.1m, capped at 1.6m
        camera_heights = self.compute_camera_heights_for_objects([a_top, b_top])
        
        mid = (a_center + b_center) / 2.0
        
        radii, ideal_dist = self.calculate_auto_radii(la['bounding_box'], lb['bounding_box'])
        
        # Limit radii based on scene bounds
        max_radius = max(radii) if radii else 3.0
        if scene_bounds is not None:
            scene_size = np.linalg.norm(scene_bounds.max_point[:2] - scene_bounds.min_point[:2])
            max_radius = min(max_radius, scene_size * 0.5)  # Increased from 0.4
            radii = [r for r in radii if r <= max_radius]
            if not radii:
                radii = [max_radius * 0.6, max_radius * 0.8]  # Fallback radii
        
        # Find room containing both objects
        target_room_idx = None
        for i, poly in enumerate(room_polys):
            if (self.point_in_poly(a_center[0], a_center[1], poly) and 
                self.point_in_poly(b_center[0], b_center[1], poly)):
                target_room_idx = i
                break
        
        valid_poses = []
        rejection_stats = {}
        cfg = self.config
        
        # Combine object and wall AABBs for occlusion checking
        all_occluders = object_aabbs + wall_aabbs
        
        for r in radii:
            for z in camera_heights:  # Use dynamic camera heights instead of cfg.camera_heights
                for yaw in np.linspace(0, 2 * np.pi, cfg.per_angle, endpoint=False):
                    cam_pos = np.array([
                        mid[0] + r * math.cos(yaw),
                        mid[1] + r * math.sin(yaw),
                        z
                    ], dtype=float)
                    
                    target_z = (a_center[2] + b_center[2]) / 2.0
                    cam_target = np.array([mid[0], mid[1], target_z], dtype=float)
                    
                    # Full validation: position + collision + wall distance
                    is_valid_pos, reject_reason = self.validate_camera_position_full(
                        cam_pos, room_polys, scene_bounds,
                        object_aabbs if enable_collision_check else [],
                        wall_aabbs if enable_collision_check else [],
                        target_room_idx,
                        require_in_target_room=True,
                        check_collision=enable_collision_check,
                        check_wall_dist=enable_collision_check
                    )
                    
                    if not is_valid_pos:
                        rejection_stats[reject_reason] = rejection_stats.get(reject_reason, 0) + 1
                        continue
                    
                    # Check visibility for both objects using precise FOV projection
                    fov_a, reason_a = self.check_target_in_fov(
                        cam_pos, cam_target, a_min, a_max, require_center=True, border=5
                    )
                    fov_b, reason_b = self.check_target_in_fov(
                        cam_pos, cam_target, b_min, b_max, require_center=True, border=5
                    )
                    
                    if not fov_a:
                        rejection_stats[f'obj_a_fov_{reason_a}'] = rejection_stats.get(f'obj_a_fov_{reason_a}', 0) + 1
                    if not fov_b:
                        rejection_stats[f'obj_b_fov_{reason_b}'] = rejection_stats.get(f'obj_b_fov_{reason_b}', 0) + 1
                    
                    if not (fov_a and fov_b):
                        continue
                    
                    # ========== NEW: Check that both object centers project to valid image region ==========
                    # Ensure both objects are well-positioned in the image (not at extreme edges)
                    # This is critical for pair tasks where both objects need to be clearly visible
                    is_pair_centered, pair_reason, pair_info = check_pair_centers_in_fov(
                        self.intrinsics, cam_pos, cam_target,
                        a_center, b_center,
                        self.config.image_width, self.config.image_height,
                        inner_margin_ratio=0.12  # 12% margin from edges
                    )
                    
                    if not is_pair_centered:
                        rejection_stats[f'pair_center_{pair_reason}'] = \
                            rejection_stats.get(f'pair_center_{pair_reason}', 0) + 1
                        continue
                    # ========== END: Pair center check ==========
                    
                    # ========== NEW: Enhanced Visibility Checks ==========
                    # Check 1: Visible corners count (ensure adequate corner visibility)
                    has_enough_corners_a, corner_count_a = self.check_visible_corners_count(
                        cam_pos, cam_target, a_min, a_max, min_corners=1,
                        check_occlusion=False  # Fast check, no occlusion yet
                    )
                    has_enough_corners_b, corner_count_b = self.check_visible_corners_count(
                        cam_pos, cam_target, b_min, b_max, min_corners=1,
                        check_occlusion=False
                    )
                    
                    if not has_enough_corners_a:
                        rejection_stats[f'obj_a_insufficient_corners_{corner_count_a}'] = \
                            rejection_stats.get(f'obj_a_insufficient_corners_{corner_count_a}', 0) + 1
                    if not has_enough_corners_b:
                        rejection_stats[f'obj_b_insufficient_corners_{corner_count_b}'] = \
                            rejection_stats.get(f'obj_b_insufficient_corners_{corner_count_b}', 0) + 1
                    
                    if not (has_enough_corners_a and has_enough_corners_b):
                        continue
                    
                    # Check 2: Projected area ratio (ensure objects are not too small)
                    is_large_enough_a, area_ratio_a, _ = self.check_projected_area(
                        cam_pos, cam_target, a_min, a_max, min_area_ratio=0.05
                    )
                    is_large_enough_b, area_ratio_b, _ = self.check_projected_area(
                        cam_pos, cam_target, b_min, b_max, min_area_ratio=0.05
                    )
                    
                    if not is_large_enough_a:
                        rejection_stats[f'obj_a_area_too_small_{area_ratio_a:.3f}'] = \
                            rejection_stats.get(f'obj_a_area_too_small_{area_ratio_a:.3f}', 0) + 1
                    if not is_large_enough_b:
                        rejection_stats[f'obj_b_area_too_small_{area_ratio_b:.3f}'] = \
                            rejection_stats.get(f'obj_b_area_too_small_{area_ratio_b:.3f}', 0) + 1
                    
                    if not (is_large_enough_a and is_large_enough_b):
                        continue
                    
                    # Check 3a: 2D image-space occlusion (more accurate than 3D ray casting)
                    # Try 2D occlusion first, fall back to 3D if cv2 not available
                    if enable_occlusion_check and all_occluders and cv2 is not None:
                        is_acceptable_a, occ_info_a = self.check_occlusion_2d(
                            cam_pos, cam_target, a_min, a_max, all_occluders,
                            target_id=obj_a_id, max_occlusion_ratio=0.7
                        )
                        is_acceptable_b, occ_info_b = self.check_occlusion_2d(
                            cam_pos, cam_target, b_min, b_max, all_occluders,
                            target_id=obj_b_id, max_occlusion_ratio=0.7
                        )
                        
                        if not is_acceptable_a:
                            occ_ratio_a = occ_info_a.get('occlusion_ratio_target', 0.0)
                            rejection_stats[f'obj_a_occluded_2d_{occ_ratio_a:.2f}'] = \
                                rejection_stats.get(f'obj_a_occluded_2d_{occ_ratio_a:.2f}', 0) + 1
                        if not is_acceptable_b:
                            occ_ratio_b = occ_info_b.get('occlusion_ratio_target', 0.0)
                            rejection_stats[f'obj_b_occluded_2d_{occ_ratio_b:.2f}'] = \
                                rejection_stats.get(f'obj_b_occluded_2d_{occ_ratio_b:.2f}', 0) + 1
                        
                        if not (is_acceptable_a and is_acceptable_b):
                            continue
                    
                    # Check 3b: Fallback to original 3D occlusion check (for compatibility)
                    elif enable_occlusion_check and all_occluders:
                        is_a_occluded, occ_a_label = self.check_occlusion(
                            cam_pos, a_min, a_max, all_occluders, obj_a_id
                        )
                        is_b_occluded, occ_b_label = self.check_occlusion(
                            cam_pos, b_min, b_max, all_occluders, obj_b_id
                        )
                        if is_a_occluded:
                            rejection_stats[f'obj_a_occluded_3d_by_{occ_a_label}'] = \
                                rejection_stats.get(f'obj_a_occluded_3d_by_{occ_a_label}', 0) + 1
                        if is_b_occluded:
                            rejection_stats[f'obj_b_occluded_3d_by_{occ_b_label}'] = \
                                rejection_stats.get(f'obj_b_occluded_3d_by_{occ_b_label}', 0) + 1
                        if is_a_occluded or is_b_occluded:
                            continue
                    # ========== END: Enhanced Visibility Checks ==========
                    
                    # ========== NEW: Check camera forward direction wall occlusion ==========
                    # Calculate camera forward direction based on look-at target
                    cam_forward = cam_target - cam_pos
                    cam_forward = cam_forward / (np.linalg.norm(cam_forward) + 1e-8)
                    
                    # Check if camera's forward view is blocked by a wall within min distance
                    is_view_blocked, nearest_wall_dist, blocking_wall = check_camera_forward_wall_occlusion(
                        cam_pos=cam_pos,
                        cam_forward=cam_forward,
                        wall_aabbs=wall_aabbs,
                        min_clear_distance=0.5,  # Must be able to see at least 0.5m ahead
                        max_check_distance=3.0
                    )
                    
                    if is_view_blocked:
                        rejection_stats[f'forward_wall_blocked_{nearest_wall_dist:.2f}m'] = \
                            rejection_stats.get(f'forward_wall_blocked_{nearest_wall_dist:.2f}m', 0) + 1
                        continue
                    # ========== END: Camera forward wall occlusion check ==========
                    
                    pose = CameraPose(
                        position=cam_pos,
                        target=cam_target,
                        yaw=float(yaw),
                        radius=float(r)
                    )
                    valid_poses.append(pose)
                    
                    if len(valid_poses) >= num_samples * 3:
                        break
                if len(valid_poses) >= num_samples * 3:
                    break
            if len(valid_poses) >= num_samples * 3:
                break
        
        # Log rejection stats if no valid poses found
        if len(valid_poses) == 0 and rejection_stats:
            print(f"[CameraSampler] Warning: No valid camera poses for object pair "
                  f"{obj_a.get('label', obj_a['id'])}, {obj_b.get('label', obj_b['id'])}. "
                  f"Rejection reasons: {rejection_stats}")
        
        # Randomly select requested number of samples
        if len(valid_poses) > num_samples:
            indices = np.random.choice(len(valid_poses), num_samples, replace=False)
            valid_poses = [valid_poses[i] for i in indices]
        
        return valid_poses
    
    def sample_camera_for_triple(self, scene_path: Path, 
                                  obj_a: Dict[str, Any], obj_b: Dict[str, Any], obj_c: Dict[str, Any],
                                  num_samples: int = 5,
                                  enable_occlusion_check: bool = True,
                                  enable_collision_check: bool = True) -> List[CameraPose]:
        """
        Sample camera poses for three objects.
        
        For three objects, the camera should look at the SECOND object (obj_b)
        to place it in the center of the view, while all three objects should
        be visible in the frame.
        
        Args:
            scene_path: Path to scene folder
            obj_a: First object dict
            obj_b: Second object dict (target - should be in view center)
            obj_c: Third object dict
            num_samples: Number of samples to generate
            enable_occlusion_check: If True, check for occlusion by other objects
            enable_collision_check: If True, check for camera collision with objects/walls
            
        Returns:
            List of CameraPose objects
        """
        labels = self.load_labels(scene_path)
        room_polys = self.load_room_polys(scene_path)
        scene_bounds = self.load_scene_bounds(scene_path)
        
        # Load AABBs for collision and occlusion checks
        object_aabbs = []
        wall_aabbs = []
        if enable_collision_check or enable_occlusion_check:
            object_aabbs = self.load_object_aabbs(scene_path)
            wall_aabbs = self.load_wall_aabbs(scene_path)
        
        la = self.find_obj_in_labels(labels, obj_a['id'])
        lb = self.find_obj_in_labels(labels, obj_b['id'])
        lc = self.find_obj_in_labels(labels, obj_c['id'])
        
        if la is None or lb is None or lc is None:
            print(f"[CameraSampler] Warning: One or more objects not found in labels: "
                  f"{obj_a['id']}, {obj_b['id']}, {obj_c['id']}")
            return []
        
        a_min, a_max = self.get_bbox_minmax(la['bounding_box'])
        b_min, b_max = self.get_bbox_minmax(lb['bounding_box'])
        c_min, c_max = self.get_bbox_minmax(lc['bounding_box'])
        a_center = self.get_bbox_center(la['bounding_box'])
        b_center = self.get_bbox_center(lb['bounding_box'])
        c_center = self.get_bbox_center(lc['bounding_box'])
        obj_a_id = str(obj_a['id'])
        obj_b_id = str(obj_b['id'])
        obj_c_id = str(obj_c['id'])
        
        # Calculate object heights for logging
        height_a = a_max[2] - a_min[2]
        height_b = b_max[2] - b_min[2]
        height_c = c_max[2] - c_min[2]
        
        # Validate object height compatibility for all three objects
        is_height_valid, height_reason = self.validate_object_heights(
            [(a_min, a_max), (b_min, b_max), (c_min, c_max)]
        )
        if not is_height_valid:
            label_a = la.get('label', obj_a.get('label', obj_a_id))
            label_b = lb.get('label', obj_b.get('label', obj_b_id))
            label_c = lc.get('label', obj_c.get('label', obj_c_id))
            print(f"[CameraSampler] Skipping triple due to height incompatibility: {height_reason}")
            print(f"    Object A: '{label_a}' (height={height_a:.3f}m)")
            print(f"    Object B: '{label_b}' (height={height_b:.3f}m)")
            print(f"    Object C: '{label_c}' (height={height_c:.3f}m)")
            return []
        
        # Get object top heights for camera height calculation
        a_top = self.get_object_top_height(la['bounding_box'])
        b_top = self.get_object_top_height(lb['bounding_box'])
        c_top = self.get_object_top_height(lc['bounding_box'])
        
        # Compute camera height: max object top + 0.1m, capped at 1.6m
        camera_heights = self.compute_camera_heights_for_objects([a_top, b_top, c_top])
        
        # For triple objects, look at the SECOND object (b_center)
        # This places the second object in the center of the view
        look_target = b_center.copy()
        
        # Camera orbit center: use centroid of all three for positioning
        orbit_center = (a_center + b_center + c_center) / 3.0
        
        # Calculate radii based on the span of all three objects
        all_centers = [a_center, b_center, c_center]
        max_dist = max(np.linalg.norm(c - orbit_center) for c in all_centers)
        max_obj_size = max(
            np.linalg.norm(a_max - a_min),
            np.linalg.norm(b_max - b_min),
            np.linalg.norm(c_max - c_min)
        )
        
        focal_length = self.intrinsics[0, 0]
        fov_factor = focal_length / self.config.image_width
        base_dist = max((max_dist + max_obj_size) * fov_factor * 2.5, 2.0)
        
        # Limit radii based on scene bounds
        max_radius = base_dist * 2.0
        if scene_bounds is not None:
            scene_size = np.linalg.norm(scene_bounds.max_point[:2] - scene_bounds.min_point[:2])
            max_radius = min(max_radius, scene_size * 0.4)
        
        radii = [base_dist * 0.9, base_dist * 1.1, base_dist * 1.4]
        radii = [r for r in radii if r <= max_radius]
        if not radii:
            radii = [max_radius * 0.7, max_radius * 0.9]
        
        # Find room containing all objects
        target_room_idx = None
        for i, poly in enumerate(room_polys):
            if (self.point_in_poly(a_center[0], a_center[1], poly) and 
                self.point_in_poly(b_center[0], b_center[1], poly) and
                self.point_in_poly(c_center[0], c_center[1], poly)):
                target_room_idx = i
                break
        
        valid_poses = []
        rejection_stats = {}
        cfg = self.config
        
        # Combine object and wall AABBs for occlusion checking
        all_occluders = object_aabbs + wall_aabbs
        
        for r in radii:
            for z in camera_heights:  # Use dynamic camera heights instead of cfg.camera_heights
                for yaw in np.linspace(0, 2 * np.pi, cfg.per_angle, endpoint=False):
                    cam_pos = np.array([
                        orbit_center[0] + r * math.cos(yaw),
                        orbit_center[1] + r * math.sin(yaw),
                        z
                    ], dtype=float)
                    
                    # Camera looks at the SECOND object center
                    cam_target = np.array([look_target[0], look_target[1], look_target[2]], dtype=float)
                    
                    # For triples, require camera to be inside SOME room.
                    # If all three objects share a room, prefer that room.
                    # Otherwise, just require the camera to be in any room polygon.
                    is_valid_pos, reject_reason = self.validate_camera_position_full(
                        cam_pos, room_polys,
                        scene_bounds,
                        object_aabbs if enable_collision_check else [],
                        wall_aabbs if enable_collision_check else [],
                        target_room_idx,  # May be None if objects span rooms
                        require_in_target_room=(target_room_idx is not None),
                        check_collision=enable_collision_check,
                        check_wall_dist=enable_collision_check
                    )
                    
                    if not is_valid_pos:
                        rejection_stats[reject_reason] = rejection_stats.get(reject_reason, 0) + 1
                        continue
                    
                    # Check visibility for all three objects using precise FOV projection
                    fov_a, reason_a = self.check_target_in_fov(
                        cam_pos, cam_target, a_min, a_max, require_center=True, border=5
                    )
                    fov_b, reason_b = self.check_target_in_fov(
                        cam_pos, cam_target, b_min, b_max, require_center=True, border=5
                    )
                    fov_c, reason_c = self.check_target_in_fov(
                        cam_pos, cam_target, c_min, c_max, require_center=True, border=5
                    )
                    
                    if not fov_a:
                        rejection_stats[f'obj_a_fov_{reason_a}'] = rejection_stats.get(f'obj_a_fov_{reason_a}', 0) + 1
                    if not fov_b:
                        rejection_stats[f'obj_b_fov_{reason_b}'] = rejection_stats.get(f'obj_b_fov_{reason_b}', 0) + 1
                    if not fov_c:
                        rejection_stats[f'obj_c_fov_{reason_c}'] = rejection_stats.get(f'obj_c_fov_{reason_c}', 0) + 1
                    
                    # Require all three objects to be visible in FOV
                    if not (fov_a and fov_b and fov_c):
                        continue
                    
                    # ========== NEW: Enhanced Visibility Checks ==========
                    # Check 1: Visible corners count for all three objects
                    has_corners_a, corner_count_a = self.check_visible_corners_count(
                        cam_pos, cam_target, a_min, a_max, min_corners=1, check_occlusion=False
                    )
                    has_corners_b, corner_count_b = self.check_visible_corners_count(
                        cam_pos, cam_target, b_min, b_max, min_corners=1, check_occlusion=False
                    )
                    has_corners_c, corner_count_c = self.check_visible_corners_count(
                        cam_pos, cam_target, c_min, c_max, min_corners=1, check_occlusion=False
                    )
                    
                    if not has_corners_a:
                        rejection_stats[f'obj_a_insufficient_corners_{corner_count_a}'] = \
                            rejection_stats.get(f'obj_a_insufficient_corners_{corner_count_a}', 0) + 1
                    if not has_corners_b:
                        rejection_stats[f'obj_b_insufficient_corners_{corner_count_b}'] = \
                            rejection_stats.get(f'obj_b_insufficient_corners_{corner_count_b}', 0) + 1
                    if not has_corners_c:
                        rejection_stats[f'obj_c_insufficient_corners_{corner_count_c}'] = \
                            rejection_stats.get(f'obj_c_insufficient_corners_{corner_count_c}', 0) + 1
                    
                    if not (has_corners_a and has_corners_b and has_corners_c):
                        continue
                    
                    # Check 2: Projected area ratio for all three objects
                    # Use very low threshold (0.005 = 0.5%) for triples since objects are often far apart
                    is_large_a, area_ratio_a, _ = self.check_projected_area(
                        cam_pos, cam_target, a_min, a_max, min_area_ratio=0.005
                    )
                    is_large_b, area_ratio_b, _ = self.check_projected_area(
                        cam_pos, cam_target, b_min, b_max, min_area_ratio=0.005
                    )
                    is_large_c, area_ratio_c, _ = self.check_projected_area(
                        cam_pos, cam_target, c_min, c_max, min_area_ratio=0.005
                    )
                    
                    if not is_large_a:
                        rejection_stats[f'obj_a_area_too_small_{area_ratio_a:.3f}'] = \
                            rejection_stats.get(f'obj_a_area_too_small_{area_ratio_a:.3f}', 0) + 1
                    if not is_large_b:
                        rejection_stats[f'obj_b_area_too_small_{area_ratio_b:.3f}'] = \
                            rejection_stats.get(f'obj_b_area_too_small_{area_ratio_b:.3f}', 0) + 1
                    if not is_large_c:
                        rejection_stats[f'obj_c_area_too_small_{area_ratio_c:.3f}'] = \
                            rejection_stats.get(f'obj_c_area_too_small_{area_ratio_c:.3f}', 0) + 1
                    
                    if not (is_large_a and is_large_b and is_large_c):
                        continue
                    
                    # Check 3a: 2D image-space occlusion (preferred if cv2 available)
                    # For triples, use very relaxed threshold (0.95) since objects often separated by rooms
                    if enable_occlusion_check and all_occluders and cv2 is not None:
                        is_acceptable_a, occ_info_a = self.check_occlusion_2d(
                            cam_pos, cam_target, a_min, a_max, all_occluders,
                            target_id=obj_a_id, max_occlusion_ratio=0.95
                        )
                        is_acceptable_b, occ_info_b = self.check_occlusion_2d(
                            cam_pos, cam_target, b_min, b_max, all_occluders,
                            target_id=obj_b_id, max_occlusion_ratio=0.95
                        )
                        is_acceptable_c, occ_info_c = self.check_occlusion_2d(
                            cam_pos, cam_target, c_min, c_max, all_occluders,
                            target_id=obj_c_id, max_occlusion_ratio=0.95
                        )
                        
                        if not is_acceptable_a:
                            occ_ratio_a = occ_info_a.get('occlusion_ratio_target', 0.0)
                            rejection_stats[f'obj_a_occluded_2d_{occ_ratio_a:.2f}'] = \
                                rejection_stats.get(f'obj_a_occluded_2d_{occ_ratio_a:.2f}', 0) + 1
                        if not is_acceptable_b:
                            occ_ratio_b = occ_info_b.get('occlusion_ratio_target', 0.0)
                            rejection_stats[f'obj_b_occluded_2d_{occ_ratio_b:.2f}'] = \
                                rejection_stats.get(f'obj_b_occluded_2d_{occ_ratio_b:.2f}', 0) + 1
                        if not is_acceptable_c:
                            occ_ratio_c = occ_info_c.get('occlusion_ratio_target', 0.0)
                            rejection_stats[f'obj_c_occluded_2d_{occ_ratio_c:.2f}'] = \
                                rejection_stats.get(f'obj_c_occluded_2d_{occ_ratio_c:.2f}', 0) + 1
                        
                        if not (is_acceptable_a and is_acceptable_b and is_acceptable_c):
                            continue
                    
                    # Check 3b: Fallback to 3D occlusion check
                    elif enable_occlusion_check and all_occluders:
                        is_a_occluded, occ_a_label = self.check_occlusion(
                            cam_pos, a_min, a_max, all_occluders, obj_a_id
                        )
                        is_b_occluded, occ_b_label = self.check_occlusion(
                            cam_pos, b_min, b_max, all_occluders, obj_b_id
                        )
                        is_c_occluded, occ_c_label = self.check_occlusion(
                            cam_pos, c_min, c_max, all_occluders, obj_c_id
                        )
                        if is_a_occluded:
                            rejection_stats[f'obj_a_occluded_3d_by_{occ_a_label}'] = \
                                rejection_stats.get(f'obj_a_occluded_3d_by_{occ_a_label}', 0) + 1
                        if is_b_occluded:
                            rejection_stats[f'obj_b_occluded_3d_by_{occ_b_label}'] = \
                                rejection_stats.get(f'obj_b_occluded_3d_by_{occ_b_label}', 0) + 1
                        if is_c_occluded:
                            rejection_stats[f'obj_c_occluded_3d_by_{occ_c_label}'] = \
                                rejection_stats.get(f'obj_c_occluded_3d_by_{occ_c_label}', 0) + 1
                        if is_a_occluded or is_b_occluded or is_c_occluded:
                            continue
                    # ========== END: Enhanced Visibility Checks ==========
                    
                    # ========== NEW: Check camera forward direction wall occlusion ==========
                    # Calculate camera forward direction based on look-at target
                    cam_forward = cam_target - cam_pos
                    cam_forward = cam_forward / (np.linalg.norm(cam_forward) + 1e-8)
                    
                    # Check if camera's forward view is blocked by a wall within min distance
                    is_view_blocked, nearest_wall_dist, blocking_wall = check_camera_forward_wall_occlusion(
                        cam_pos=cam_pos,
                        cam_forward=cam_forward,
                        wall_aabbs=wall_aabbs,
                        min_clear_distance=0.5,  # Must be able to see at least 0.5m ahead
                        max_check_distance=3.0
                    )
                    
                    if is_view_blocked:
                        rejection_stats[f'forward_wall_blocked_{nearest_wall_dist:.2f}m'] = \
                            rejection_stats.get(f'forward_wall_blocked_{nearest_wall_dist:.2f}m', 0) + 1
                        continue
                    # ========== END: Camera forward wall occlusion check ==========
                    
                    pose = CameraPose(
                        position=cam_pos,
                        target=cam_target,  # Looking at second object
                        yaw=float(yaw),
                        radius=float(r)
                    )
                    valid_poses.append(pose)
                    
                    if len(valid_poses) >= num_samples * 3:
                        break
                if len(valid_poses) >= num_samples * 3:
                    break
            if len(valid_poses) >= num_samples * 3:
                break
        
        # Log rejection stats if no valid poses found
        if len(valid_poses) == 0 and rejection_stats:
            print(f"[CameraSampler] Warning: No valid camera poses for object triple "
                  f"{obj_a.get('label', obj_a['id'])}, {obj_b.get('label', obj_b['id'])}, "
                  f"{obj_c.get('label', obj_c['id'])}. Rejection reasons: {rejection_stats}")
        
        # Randomly select requested number of samples
        if len(valid_poses) > num_samples:
            indices = np.random.choice(len(valid_poses), num_samples, replace=False)
            valid_poses = [valid_poses[i] for i in indices]
        
        return valid_poses
    
    def _to_dict(self, obj: Union[SceneObject, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert SceneObject to dict if needed."""
        if isinstance(obj, SceneObject):
            return obj.to_dict()
        return obj
    
    # =========================================================================
    # ROTATION PATTERN METHODS
    # =========================================================================
    
    def compute_room_centers(self, scene_path: Path) -> List[Dict[str, Any]]:
        """
        Compute the center of each room from structure.json.
        
        Returns:
            List of dictionaries with room info:
            [{'room_idx': 0, 'center': np.array([x, y]), 'polygon': [...]}]
        """
        room_polys = self.load_room_polys(scene_path)
        room_centers = []
        
        for room_idx, poly in enumerate(room_polys):
            if not poly or len(poly) < 3:
                continue
            
            # Compute centroid of polygon
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            
            room_centers.append({
                'room_idx': room_idx,
                'center': np.array([center_x, center_y], dtype=float),
                'polygon': poly,
            })
        
        return room_centers
    
    def generate_rotation_poses(self, scene_path: Path, 
                                 all_scene_objects: Optional[List[Dict[str, Any]]] = None
                                 ) -> List[Tuple[CameraPose, int]]:
        """
        Generate camera poses for rotation pattern.
        
        For each room, stand at the center and rotate 360 degrees.
        
        Args:
            scene_path: Path to scene folder
            all_scene_objects: All objects in the scene for collision checking
            
        Returns:
            List of (CameraPose, room_idx) tuples
        """
        room_centers = self.compute_room_centers(scene_path)
        if not room_centers:
            print("  [Warning] No room polygons found, cannot generate rotation poses")
            return []
        
        interval = getattr(self.config, 'rotation_interval', 5.0)
        camera_height = getattr(self.config, 'rotation_camera_height', 1.5)
        num_angles = int(360 / interval)
        
        # Load AABBs for collision check
        object_aabbs = self.load_object_aabbs(scene_path)
        
        all_poses = []
        
        for room_info in room_centers:
            room_idx = room_info['room_idx']
            center_2d = room_info['center']
            
            # Camera position: room center at fixed height
            cam_x, cam_y = center_2d[0], center_2d[1]
            cam_z = camera_height
            position = np.array([cam_x, cam_y, cam_z])
            
            # Check if camera position collides with any object
            collision = False
            for aabb in object_aabbs:
                if aabb.contains_point(position, margin=0.1):
                    collision = True
                    break
            
            if collision:
                print(f"  [Warning] Room {room_idx} center collides with object, skipping")
                continue
            
            # Generate poses for each angle
            for angle_idx in range(num_angles):
                yaw = angle_idx * interval
                yaw_rad = np.radians(yaw)
                
                # Look direction: 1 meter in front of camera at same height
                look_distance = 1.0
                target_x = cam_x + look_distance * np.cos(yaw_rad)
                target_y = cam_y + look_distance * np.sin(yaw_rad)
                target_z = cam_z  # Look horizontally
                target = np.array([target_x, target_y, target_z])
                
                pose = CameraPose(
                    position=position.copy(),
                    target=target,
                    yaw=yaw,
                    radius=0.0,  # Not applicable for rotation
                )
                
                all_poses.append((pose, room_idx))
        
        return all_poses
    
    # =========================================================================
    # LINEAR PATTERN METHODS (PASS_BY ONLY)
    # =========================================================================
    
    def _validate_visibility_linear(self, pose: CameraPose, 
                                    target_objects: List[Tuple[np.ndarray, np.ndarray, str]],
                                    all_occluders: List[AABB]) -> bool:
        """
        Validate visibility for linear pattern (relaxed version).
        
        For linear pattern, camera orientation is FIXED, so object may not be at image center.
        Only check that object is within FOV (anywhere in frame), not necessarily centered.
        
        Args:
            pose: Camera pose to validate
            target_objects: List of (bmin, bmax, obj_id) for target objects
            all_occluders: All AABBs in scene for occlusion checking
            
        Returns:
            True if objects are visible somewhere in frame
        """
        K = self.intrinsics
        width = self.config.image_width
        height = self.config.image_height
        
        for bmin, bmax, obj_id in target_objects:
            # Relaxed check: Object just needs to be in FOV (don't require center in image)
            in_fov, reason = is_target_in_fov(
                K=K,
                cam_pos=pose.position,
                cam_target=pose.target,
                target_bmin=bmin,
                target_bmax=bmax,
                width=width,
                height=height,
                require_center=False,  # Don't require center - just any part visible
                border=0  # No margin requirement
            )
            
            if not in_fov:
                return False
            
            # Check occlusion (use existing method)
            is_occluded, _ = self.check_occlusion(
                pose.position, bmin, bmax, all_occluders, obj_id,
                occlusion_threshold=0.7  # More lenient - only reject if 70%+ blocked
            )
            
            if is_occluded:
                return False
        
        return True
    
    def generate_linear_poses(self, scene_path: Path,
                              target_objects: List[Dict[str, Any]],
                              num_trajectories: int = 1) -> List[List[CameraPose]]:
        """
        Generate sequences of camera poses for linear movement pattern (pass_by).
        
        For each trajectory, generates multiple poses along a STRAIGHT LINE path.
        Camera orientation (yaw/pitch) remains FIXED throughout the trajectory.
        Only camera POSITION changes along a linear path.
        
        Pass_by pattern: Camera walks in a straight line past the object.
        At the middle of the trajectory, object is roughly at center of view.
        
        Args:
            scene_path: Path to scene folder
            target_objects: Target objects (list of dicts with 'id' field)
            num_trajectories: Number of trajectories to generate
            
        Returns:
            List of trajectories, each trajectory is a List[CameraPose]
        """
        if not target_objects:
            return []
        
        labels = self.load_labels(scene_path)
        room_polys = self.load_room_polys(scene_path)
        scene_bounds = self.load_scene_bounds(scene_path)
        object_aabbs = self.load_object_aabbs(scene_path)
        wall_aabbs = self.load_wall_aabbs(scene_path)
        all_occluders = object_aabbs + wall_aabbs
        
        # Get target object info
        target_info = []
        for obj in target_objects:
            obj_data = self.find_obj_in_labels(labels, obj['id'])
            if obj_data is None:
                continue
            bmin, bmax = self.get_bbox_minmax(obj_data['bounding_box'])
            center = self.get_bbox_center(obj_data['bounding_box'])
            target_info.append((bmin, bmax, str(obj['id']), center))
        
        if not target_info:
            return []
        
        # Compute combined target center
        target_center = np.mean([info[3] for info in target_info], axis=0)
        
        # Parameters
        num_steps = getattr(self.config, 'linear_num_steps', 5)
        move_distance = getattr(self.config, 'linear_move_distance', 0.5)
        max_tries = self.config.max_tries
        
        # Compute camera height based on object heights
        obj_tops = [float(info[1][2]) for info in target_info]
        camera_heights = self.compute_camera_heights_for_objects(obj_tops)
        camera_height = camera_heights[0] if camera_heights else 1.5
        
        # Calculate base distance
        max_obj_size = max(np.linalg.norm(info[1] - info[0]) for info in target_info)
        focal_length = self.intrinsics[0, 0]
        fov_factor = focal_length / self.config.image_width
        base_dist = max(max_obj_size * fov_factor * 1.5, 1.5)
        
        trajectories = []
        
        for _ in range(max_tries):
            if len(trajectories) >= num_trajectories:
                break
            
            # Random initial direction (from target to camera base position)
            initial_yaw = np.random.uniform(0, 360)
            initial_yaw_rad = np.radians(initial_yaw)
            
            # Direction vector from target to camera base position (in XY plane)
            dir_from_target = np.array([np.cos(initial_yaw_rad), np.sin(initial_yaw_rad), 0.0])
            
            # Random distance from object
            base_radius = np.random.uniform(base_dist * 0.8, base_dist * 1.5)
            
            # =============================================================
            # PASS_BY: Camera walks in a straight line past the object
            # =============================================================
            # Direction perpendicular to dir_from_target (in XY plane)
            perp_dir = np.array([-dir_from_target[1], dir_from_target[0], 0.0])
            
            # Base position: at base_radius from object
            base_pos = np.array([
                target_center[0] + base_radius * dir_from_target[0],
                target_center[1] + base_radius * dir_from_target[1],
                camera_height
            ])
            
            # Start and end positions (along perpendicular line, centered at base_pos)
            start_pos = base_pos - (move_distance / 2) * perp_dir
            end_pos = base_pos + (move_distance / 2) * perp_dir
            
            # FIXED camera orientation: looking toward the target direction
            look_dir = -dir_from_target
            fixed_yaw = np.degrees(np.arctan2(look_dir[1], look_dir[0]))
            
            # Compute fixed pitch (looking slightly down toward target)
            dz = target_center[2] - camera_height
            fixed_pitch = np.degrees(np.arctan2(-dz, base_radius))
            
            # FIXED look target
            look_distance = 10.0
            fixed_look_target = np.array([
                base_pos[0] + look_distance * look_dir[0],
                base_pos[1] + look_distance * look_dir[1],
                target_center[2]
            ])
            
            trajectory_poses = []
            trajectory_valid = True
            
            for step in range(num_steps):
                t = step / max(1, num_steps - 1)
                
                # Linear interpolation: position moves along straight line
                position = start_pos + t * (end_pos - start_pos)
                current_dist = np.linalg.norm(position[:2] - target_center[:2])
                
                # Validate position
                is_valid, reason = self.validate_camera_position_full(
                    position, room_polys, scene_bounds,
                    object_aabbs, wall_aabbs,
                    target_room_idx=None,
                    require_in_target_room=False,
                    check_collision=True,
                    check_wall_dist=True
                )
                
                if not is_valid:
                    trajectory_valid = False
                    break
                
                pose = CameraPose(
                    position=position.copy(),
                    target=fixed_look_target.copy(),  # FIXED look target
                    yaw=fixed_yaw,                    # FIXED yaw
                    radius=current_dist,
                )
                
                # Use relaxed visibility check for linear pattern
                target_tuples = [(info[0], info[1], info[2]) for info in target_info]
                if not self._validate_visibility_linear(pose, target_tuples, all_occluders):
                    trajectory_valid = False
                    break
                
                trajectory_poses.append(pose)
            
            if trajectory_valid and len(trajectory_poses) == num_steps:
                trajectories.append(trajectory_poses)
        
        return trajectories
    
    def sample_cameras(self, scene_path: Path, objects: Any, 
                       num_samples: int = 5) -> List[CameraPose]:
        """
        Sample cameras for objects (single, pair, or triple).
        
        Supports multiple move patterns:
        - 'around': Horizontal circle around object (default)
        - 'rotation': Stand at room center, rotate 360°
        - 'linear': Walk past object in straight line (passing motion)
        
        Camera orientation strategy (for 'around' pattern):
        - Single object: Camera looks directly at the object center
        - Two objects: Camera looks at the midpoint between two object centers
        - Three objects: Camera looks at the SECOND object (places it in view center)
        
        Args:
            scene_path: Path to scene folder
            objects: Single object (dict/SceneObject) or list of objects
            num_samples: Number of camera poses to sample
        
        Returns:
            List of CameraPose objects
        """
        move_pattern = getattr(self.config, 'move_pattern', 'around')
        
        # For linear pattern, use generate_linear_poses to get trajectories
        if move_pattern == 'linear':
            # Convert objects to list of dicts
            if isinstance(objects, (SceneObject, dict)):
                obj_list = [self._to_dict(objects) if isinstance(objects, SceneObject) else objects]
            elif isinstance(objects, (list, tuple)):
                obj_list = [self._to_dict(o) for o in objects]
            else:
                return []
            
            trajectories = self.generate_linear_poses(scene_path, obj_list, num_trajectories=num_samples)
            # Flatten trajectories to list of poses
            poses = []
            for traj in trajectories:
                poses.extend(traj)
            return poses
        
        # For rotation pattern, use generate_rotation_poses
        # Note: rotation is room-centric, so objects parameter is not used for positioning
        # It's still passed for potential visibility analysis
        if move_pattern == 'rotation':
            rotation_poses = self.generate_rotation_poses(scene_path)
            # Return just the poses (without room indices)
            return [pose for pose, room_idx in rotation_poses]
        
        # Default 'around' pattern: use existing sampling methods
        # Handle SceneObject type
        if isinstance(objects, SceneObject):
            # Single object
            return self.sample_camera_for_single(scene_path, self._to_dict(objects), num_samples)
        elif isinstance(objects, dict):
            # Single object dict
            return self.sample_camera_for_single(scene_path, objects, num_samples)
        elif isinstance(objects, (list, tuple)):
            # Convert all to dicts
            obj_dicts = [self._to_dict(o) for o in objects]
            if len(obj_dicts) == 1:
                return self.sample_camera_for_single(scene_path, obj_dicts[0], num_samples)
            elif len(obj_dicts) == 2:
                # Pair: look at midpoint of the two objects
                return self.sample_camera_for_pair(scene_path, obj_dicts[0], obj_dicts[1], num_samples)
            elif len(obj_dicts) >= 3:
                # Triple or more: look at the SECOND object (place it in view center)
                return self.sample_camera_for_triple(
                    scene_path, obj_dicts[0], obj_dicts[1], obj_dicts[2], num_samples
                )
        
        return []
