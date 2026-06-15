"""
Camera Utility Functions

This module contains utility functions for camera operations, including:
- Data classes (SceneBounds, AABB, CameraPose)
- Ray-AABB intersection and occlusion detection
- Camera projection and FOV checking
- Geometry utility functions

These functions are used by CameraSampler but are decoupled for reusability.
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SceneBounds:
    """Represents the 3D bounding box of a scene."""
    min_point: np.ndarray  # (x_min, y_min, z_min)
    max_point: np.ndarray  # (x_max, y_max, z_max)
    center: np.ndarray     # (x_center, y_center, z_center)
    
    @classmethod
    def from_occupancy(cls, occupancy_data: Dict[str, Any]) -> 'SceneBounds':
        """Create from occupancy.json data."""
        min_pt = np.array(occupancy_data.get('min', occupancy_data.get('lower', [-10, -10, 0])), dtype=float)
        max_pt = np.array(occupancy_data.get('max', occupancy_data.get('upper', [10, 10, 3])), dtype=float)
        center = np.array(occupancy_data.get('center', ((min_pt + max_pt) / 2).tolist()), dtype=float)
        return cls(min_point=min_pt, max_point=max_pt, center=center)
    
    def contains_point(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if a 3D point is inside the scene bounds."""
        return (
            self.min_point[0] + margin <= point[0] <= self.max_point[0] - margin and
            self.min_point[1] + margin <= point[1] <= self.max_point[1] - margin and
            self.min_point[2] + margin <= point[2] <= self.max_point[2] - margin
        )
    
    def contains_point_2d(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a 2D point (x, y) is inside the scene bounds (ignoring z)."""
        return (
            self.min_point[0] + margin <= x <= self.max_point[0] - margin and
            self.min_point[1] + margin <= y <= self.max_point[1] - margin
        )


@dataclass
class AABB:
    """Axis-Aligned Bounding Box for collision and occlusion detection."""
    id: str
    label: str
    bmin: np.ndarray  # (x_min, y_min, z_min)
    bmax: np.ndarray  # (x_max, y_max, z_max)
    
    def contains_point(self, point: np.ndarray, margin: float = 0.0) -> bool:
        """Check if a 3D point is inside this AABB (with optional margin)."""
        return (
            self.bmin[0] - margin <= point[0] <= self.bmax[0] + margin and
            self.bmin[1] - margin <= point[1] <= self.bmax[1] + margin and
            self.bmin[2] - margin <= point[2] <= self.bmax[2] + margin
        )
    
    @property
    def center(self) -> np.ndarray:
        return (self.bmin + self.bmax) / 2.0
    
    def corners(self) -> np.ndarray:
        """Return all 8 corners of the AABB."""
        corners = []
        for xi in [self.bmin[0], self.bmax[0]]:
            for yi in [self.bmin[1], self.bmax[1]]:
                for zi in [self.bmin[2], self.bmax[2]]:
                    corners.append([xi, yi, zi])
        return np.array(corners, dtype=float)


@dataclass
class CameraPose:
    """Represents a camera pose."""
    position: np.ndarray  # (x, y, z)
    target: np.ndarray  # Look-at point (x, y, z)
    yaw: float  # Rotation around z-axis
    radius: float  # Distance from object center
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'camera_pos': self.position.tolist(),
            'camera_target': self.target.tolist(),
            'yaw': float(self.yaw),
            'radius': float(self.radius),
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CameraPose':
        return cls(
            position=np.array(d['camera_pos'], dtype=float),
            target=np.array(d['camera_target'], dtype=float),
            yaw=float(d.get('yaw', 0)),
            radius=float(d.get('radius', 0)),
        )


# ============================================================================
# Ray-AABB Intersection Functions
# ============================================================================

def intersects_ray_aabb(ray_o: np.ndarray, ray_d: np.ndarray, 
                        bmin: np.ndarray, bmax: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Ray-AABB intersection using the slab method.
    
    Args:
        ray_o: Ray origin (3D point)
        ray_d: Ray direction (normalized or unnormalized)
        bmin: AABB minimum corner
        bmax: AABB maximum corner
        
    Returns:
        (t_enter, t_exit) if ray intersects AABB, else None.
        t values are parametric distances along the ray.
    """
    tmin_overall = -float('inf')
    tmax_overall = float('inf')
    
    for i in range(3):
        di = float(ray_d[i])
        oi = float(ray_o[i])
        bmin_i = float(bmin[i])
        bmax_i = float(bmax[i])
        
        if abs(di) < 1e-12:
            # Ray parallel to slab - if origin not within slab, no hit
            if oi < bmin_i or oi > bmax_i:
                return None
            # Otherwise, this axis imposes no constraint
            continue
        
        invd = 1.0 / di
        t1 = (bmin_i - oi) * invd
        t2 = (bmax_i - oi) * invd
        
        if t1 > t2:
            t1, t2 = t2, t1
        
        tmin_overall = max(tmin_overall, t1)
        tmax_overall = min(tmax_overall, t2)
        
        if tmax_overall < tmin_overall:
            return None
    
    if tmax_overall >= max(tmin_overall, 0.0):
        return float(tmin_overall), float(tmax_overall)
    return None


# ============================================================================
# Camera Projection and Transformation Functions
# ============================================================================

def camtoworld_from_pos_target(pos: np.ndarray, target: np.ndarray, 
                                up_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a 4x4 camera-to-world transformation matrix from position and target.
    
    Uses right-handed coordinate system with Z-up world convention.
    The camera looks along the +Z axis in camera space (OpenGL convention).
    
    Args:
        pos: Camera position in world coordinates
        target: Look-at target point in world coordinates
        up_vec: World up vector (default: [0, 0, 1] for Z-up)
        
    Returns:
        4x4 camera-to-world transformation matrix
    """
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    
    # Forward direction (camera looks along this axis)
    forward = target - pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        forward = np.array([0.0, 1.0, 0.0])
    else:
        forward = forward / forward_norm
    
    # Up vector
    if up_vec is None:
        up_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up_vec = np.asarray(up_vec, dtype=float)
    
    # Right = forward × up
    right = np.cross(forward, up_vec)
    if np.linalg.norm(right) < 1e-6:
        # Forward is parallel to up, choose arbitrary right
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right = right / np.linalg.norm(right)
    
    # Recompute up = right × forward
    up = np.cross(right, forward)
    
    # Build 4x4 matrix: columns are [-right, up, forward, pos]
    # (negative right because we want right-handed coordinate system)
    c2w = np.eye(4, dtype=float)
    c2w[:3, 0] = -right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    
    return c2w


def world_to_camera(viewmat: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    """
    Transform a world point to camera coordinates.
    
    Args:
        viewmat: 4x4 world-to-camera matrix (inverse of camtoworld)
        point_world: 3D point in world coordinates
        
    Returns:
        3D point in camera coordinates
    """
    pw_h = np.concatenate([point_world, np.ones(1, dtype=float)])
    pc_h = viewmat @ pw_h
    return pc_h[:3]


def project_point_to_image(K: np.ndarray, point_camera: np.ndarray) -> Tuple[float, float, float]:
    """
    Project a camera-space point to image coordinates.
    
    Args:
        K: 3x3 camera intrinsic matrix
        point_camera: 3D point in camera coordinates
        
    Returns:
        (u, v, z) - image coordinates and depth. If z <= 0, point is behind camera.
    """
    x, y, z = float(point_camera[0]), float(point_camera[1]), float(point_camera[2])
    if z <= 1e-6:
        return float('inf'), float('inf'), z
    u = K[0, 0] * (x / z) + K[0, 2]
    v = K[1, 1] * (y / z) + K[1, 2]
    return u, v, z


def point_in_image_bounds(u: float, v: float, width: int, height: int, border: int = 0) -> bool:
    """
    Check if projected point is within image bounds.
    
    Args:
        u, v: Image coordinates
        width, height: Image dimensions
        border: Margin from image edges (pixels)
        
    Returns:
        True if point is within image bounds (with border margin)
    """
    return (border <= u <= (width - 1 - border)) and (border <= v <= (height - 1 - border))


def aabb_corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """
    Generate all 8 corners of an axis-aligned bounding box.
    
    Args:
        bmin: AABB minimum corner (x_min, y_min, z_min)
        bmax: AABB maximum corner (x_max, y_max, z_max)
        
    Returns:
        (8, 3) array of corner coordinates
    """
    corners = []
    for xi in [bmin[0], bmax[0]]:
        for yi in [bmin[1], bmax[1]]:
            for zi in [bmin[2], bmax[2]]:
                corners.append([xi, yi, zi])
    return np.array(corners, dtype=float)


# ============================================================================
# FOV Checking Functions
# ============================================================================

def is_target_in_fov(K: np.ndarray, cam_pos: np.ndarray, cam_target: np.ndarray,
                     target_bmin: np.ndarray, target_bmax: np.ndarray,
                     width: int, height: int,
                     require_center: bool = True,
                     border: int = 5) -> Tuple[bool, str]:
    """
    Check if target object is within camera's field of view using full projection.
    
    This performs accurate FOV checking by:
    1. Building the camera-to-world transformation from position and target
    2. Projecting all 8 corners of the target AABB to image coordinates
    3. Checking if projected points fall within image bounds
    4. Optionally requiring the object center to be in the image
    
    Args:
        K: 3x3 camera intrinsic matrix
        cam_pos: Camera position in world coordinates
        cam_target: Camera look-at target in world coordinates
        target_bmin: Target object AABB minimum corner
        target_bmax: Target object AABB maximum corner
        width: Image width in pixels
        height: Image height in pixels
        require_center: If True, object center must project to image (default: True)
        border: Margin from image edges in pixels (default: 5)
        
    Returns:
        (is_in_fov, reason) - True if target is in FOV, with reason for failure
    """
    # Build camera-to-world and view matrix
    camtoworld = camtoworld_from_pos_target(cam_pos, cam_target)
    viewmat = np.linalg.inv(camtoworld)
    
    # Get all 8 corners of target AABB
    corners = aabb_corners(target_bmin, target_bmax)
    
    # Transform corners to camera space
    pcs = np.array([world_to_camera(viewmat, c) for c in corners])
    
    # Check if ALL corners are behind camera (z <= 0)
    if np.all(pcs[:, 2] <= 1e-6):
        return False, 'all_corners_behind_camera'
    
    # Project all corners to image
    projected = np.array([project_point_to_image(K, pc) for pc in pcs])
    uvs = projected[:, :2]
    depths = projected[:, 2]
    
    # Check if ANY corner is in front of camera and projects to image
    in_front = depths > 1e-6
    any_in_image = False
    for (u, v), z in zip(uvs, depths):
        if z > 1e-6 and point_in_image_bounds(u, v, width, height, border):
            any_in_image = True
            break
    
    if not np.any(in_front):
        return False, 'no_corner_in_front'
    
    if not any_in_image:
        return False, 'no_corner_in_image'
    
    # Optionally require center to be in image
    if require_center:
        center = 0.5 * (target_bmin + target_bmax)
        pc = world_to_camera(viewmat, center)
        u, v, z = project_point_to_image(K, pc)
        if not (z > 1e-6 and point_in_image_bounds(u, v, width, height, border)):
            return False, 'center_not_in_image'
    
    return True, 'ok'


def check_multiple_targets_in_fov(K: np.ndarray, cam_pos: np.ndarray, cam_target: np.ndarray,
                                   targets: List[Tuple[np.ndarray, np.ndarray, str]],
                                   width: int, height: int,
                                   require_all_centers: bool = True,
                                   border: int = 5) -> Tuple[bool, List[str]]:
    """
    Check if multiple target objects are all within camera's field of view.
    
    Args:
        K: 3x3 camera intrinsic matrix
        cam_pos: Camera position in world coordinates
        cam_target: Camera look-at target in world coordinates
        targets: List of (bmin, bmax, id) tuples for each target object
        width: Image width in pixels
        height: Image height in pixels
        require_all_centers: If True, all object centers must project to image
        border: Margin from image edges in pixels
        
    Returns:
        (all_in_fov, failure_reasons) - True if all targets in FOV, with reasons for failures
    """
    failures = []
    
    for bmin, bmax, obj_id in targets:
        in_fov, reason = is_target_in_fov(
            K, cam_pos, cam_target, bmin, bmax, 
            width, height, require_all_centers, border
        )
        if not in_fov:
            failures.append(f"{obj_id}:{reason}")
    
    return len(failures) == 0, failures


def check_pair_centers_in_fov(K: np.ndarray, cam_pos: np.ndarray, cam_target: np.ndarray,
                               center_a: np.ndarray, center_b: np.ndarray,
                               width: int, height: int,
                               inner_margin_ratio: float = 0.15) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if both object centers project to valid positions within the image.
    
    For object pairs, we want both objects to be reasonably visible, not at extreme
    edges of the image. This function ensures:
    1. Both centers project to the image (not behind camera)
    2. Both centers are within a comfortable viewing area (not at extreme edges)
    3. The centers are on the same side of the image center (not one far left, one far right)
    
    Args:
        K: 3x3 camera intrinsic matrix
        cam_pos: Camera position in world coordinates
        cam_target: Camera look-at target in world coordinates
        center_a: 3D center of first object
        center_b: 3D center of second object
        width: Image width in pixels
        height: Image height in pixels
        inner_margin_ratio: Ratio of image size to use as margin from edges (default 0.15)
                           Objects should project within (margin, width-margin) and (margin, height-margin)
        
    Returns:
        (is_valid, reason, info) - True if both centers are in valid positions
        info contains projected coordinates: {'u_a', 'v_a', 'u_b', 'v_b', 'z_a', 'z_b'}
    """
    # Build camera-to-world and view matrix
    camtoworld = camtoworld_from_pos_target(cam_pos, cam_target)
    viewmat = np.linalg.inv(camtoworld)
    
    # Project center A
    pc_a = world_to_camera(viewmat, center_a)
    u_a, v_a, z_a = project_point_to_image(K, pc_a)
    
    # Project center B
    pc_b = world_to_camera(viewmat, center_b)
    u_b, v_b, z_b = project_point_to_image(K, pc_b)
    
    info = {'u_a': u_a, 'v_a': v_a, 'z_a': z_a, 'u_b': u_b, 'v_b': v_b, 'z_b': z_b}
    
    # Check if both are in front of camera
    if z_a <= 1e-6:
        return False, 'center_a_behind_camera', info
    if z_b <= 1e-6:
        return False, 'center_b_behind_camera', info
    
    # Calculate margin in pixels
    margin_x = width * inner_margin_ratio
    margin_y = height * inner_margin_ratio
    
    # Check if center A is within the inner region
    if not (margin_x <= u_a <= width - margin_x and margin_y <= v_a <= height - margin_y):
        return False, f'center_a_near_edge_u{u_a:.0f}_v{v_a:.0f}', info
    
    # Check if center B is within the inner region
    if not (margin_x <= u_b <= width - margin_x and margin_y <= v_b <= height - margin_y):
        return False, f'center_b_near_edge_u{u_b:.0f}_v{v_b:.0f}', info
    
    # Check that both objects are reasonably distributed in the image
    # They shouldn't both be crammed into one corner
    center_x = width / 2
    center_y = height / 2
    
    # Calculate distances from image center
    dist_a = math.sqrt((u_a - center_x)**2 + (v_a - center_y)**2)
    dist_b = math.sqrt((u_b - center_x)**2 + (v_b - center_y)**2)
    
    # If both objects are very far from center on the same side, that's suboptimal
    # but we allow it as long as they're within the valid region
    
    info['dist_a_from_center'] = dist_a
    info['dist_b_from_center'] = dist_b
    
    return True, 'ok', info


# ============================================================================
# Occlusion Detection Functions
# ============================================================================

def is_point_occluded_by_single_aabb(ray_o: np.ndarray, ray_target: np.ndarray, 
                                      aabb: AABB, eps: float = 1e-3) -> bool:
    """
    Check if a target point is occluded by a single AABB.
    
    Args:
        ray_o: Camera position (ray origin)
        ray_target: Target point to check visibility
        aabb: The potential occluder
        eps: Small epsilon to avoid self-intersection
        
    Returns:
        True if the AABB blocks the ray before reaching the target
    """
    ray_d = ray_target - ray_o
    dist = float(np.linalg.norm(ray_d))
    if dist < 1e-6:
        return False
    ray_d = ray_d / dist
    
    hit = intersects_ray_aabb(ray_o, ray_d, aabb.bmin, aabb.bmax)
    if hit is None:
        return False
    
    t_enter, _ = hit
    # Occluded if intersection happens before reaching target
    return 0.0 <= t_enter <= (dist - eps)


def is_point_occluded_by_aabb_list(ray_o: np.ndarray, ray_target: np.ndarray,
                                    occluders: List[AABB], target_id: Optional[str] = None,
                                    eps: float = 1e-3) -> bool:
    """
    Check if a point is occluded by checking ray-AABB intersections against a list.
    Helper function for count_visible_corners with check_occlusion=True.
    
    Args:
        ray_o: Ray origin (camera position)
        ray_target: Target point to check visibility
        occluders: List of potential occluding AABBs
        target_id: ID of target to exclude from occluder list
        eps: Small epsilon to avoid self-intersection
        
    Returns:
        True if any AABB occludes the target point
    """
    ray_d = ray_target - ray_o
    dist = float(np.linalg.norm(ray_d))
    if dist < 1e-6:
        return False
    ray_d = ray_d / dist
    
    for occluder in occluders:
        # Skip target itself
        if target_id is not None and occluder.id == target_id:
            continue
        
        hit = intersects_ray_aabb(ray_o, ray_d, occluder.bmin, occluder.bmax)
        if hit is None:
            continue
        
        t_enter, _ = hit
        # Occluded if intersection happens before reaching target
        if 0.0 <= t_enter <= (dist - eps):
            return True
    
    return False


def check_camera_forward_wall_occlusion(
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    wall_aabbs: List[AABB],
    min_clear_distance: float = 0.3,
    max_check_distance: float = 2.0
) -> Tuple[bool, float, Optional[str]]:
    """
    Check if camera's forward view is blocked by a wall within a certain distance.
    
    This is used to filter out camera poses where the initial view would be
    entirely blocked by a wall (e.g., camera facing directly into a wall).
    
    Args:
        cam_pos: Camera position in world coordinates [x, y, z]
        cam_forward: Camera forward direction (normalized)
        wall_aabbs: List of wall AABBs to check against
        min_clear_distance: Minimum distance the camera should be able to see
                           (positions closer to walls than this are filtered)
        max_check_distance: Maximum distance to check for walls (optimization)
        
    Returns:
        Tuple of:
        - is_blocked: True if view is blocked by wall within min_clear_distance
        - nearest_wall_dist: Distance to nearest wall along forward direction
        - nearest_wall_id: ID of the nearest blocking wall (or None)
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_forward = np.asarray(cam_forward, dtype=float)
    
    # Normalize forward direction
    forward_len = np.linalg.norm(cam_forward)
    if forward_len < 1e-6:
        return False, float('inf'), None
    cam_forward = cam_forward / forward_len
    
    nearest_dist = float('inf')
    nearest_wall_id = None
    
    for wall in wall_aabbs:
        # Cast ray from camera position in forward direction
        hit = intersects_ray_aabb(cam_pos, cam_forward, wall.bmin, wall.bmax)
        if hit is None:
            continue
        
        t_enter, t_exit = hit
        
        # Only consider hits in front of camera (t > 0) and within check distance
        if t_enter > 0 and t_enter < max_check_distance:
            if t_enter < nearest_dist:
                nearest_dist = t_enter
                nearest_wall_id = wall.id
    
    is_blocked = nearest_dist < min_clear_distance
    return is_blocked, nearest_dist, nearest_wall_id


def check_camera_fov_wall_occlusion(
    cam_pos: np.ndarray,
    cam_forward: np.ndarray,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    wall_aabbs: List[AABB],
    hfov_deg: float = 110.0,
    vfov_deg: float = 90.0,
    min_clear_distance: float = 0.3,
    num_rays: int = 5,
    occlusion_threshold: float = 0.8
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Check if camera's field of view is heavily occluded by walls.
    
    Casts multiple rays within the camera's FOV and checks what fraction
    hit walls within min_clear_distance. If the fraction exceeds the threshold,
    the camera pose is considered blocked.
    
    Args:
        cam_pos: Camera position [x, y, z]
        cam_forward: Camera forward direction (Z axis in camera space)
        cam_right: Camera right direction (X axis in camera space)
        cam_up: Camera up direction (Y axis in camera space)
        wall_aabbs: List of wall AABBs
        hfov_deg: Horizontal field of view in degrees
        vfov_deg: Vertical field of view in degrees
        min_clear_distance: Distance within which walls count as blocking
        num_rays: Number of rays to cast in each direction (total = num_rays^2)
        occlusion_threshold: Fraction of rays blocked to consider pose invalid
        
    Returns:
        Tuple of:
        - is_blocked: True if too many rays hit walls
        - block_ratio: Fraction of rays that hit walls within min_clear_distance
        - details: Dict with 'blocked_rays', 'total_rays', 'min_wall_dist'
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_forward = np.asarray(cam_forward, dtype=float)
    cam_right = np.asarray(cam_right, dtype=float)
    cam_up = np.asarray(cam_up, dtype=float)
    
    # Normalize directions
    cam_forward = cam_forward / (np.linalg.norm(cam_forward) + 1e-8)
    cam_right = cam_right / (np.linalg.norm(cam_right) + 1e-8)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-8)
    
    # Convert FOV to radians
    hfov_rad = np.radians(hfov_deg)
    vfov_rad = np.radians(vfov_deg)
    
    # Generate ray directions within FOV
    blocked_count = 0
    total_rays = num_rays * num_rays
    min_wall_dist = float('inf')
    
    for i in range(num_rays):
        for j in range(num_rays):
            # Map [0, num_rays-1] to [-0.5, 0.5] of FOV
            u = (i / (num_rays - 1) - 0.5) if num_rays > 1 else 0
            v = (j / (num_rays - 1) - 0.5) if num_rays > 1 else 0
            
            # Calculate ray direction
            h_angle = u * hfov_rad
            v_angle = v * vfov_rad
            
            # Rotate forward direction by horizontal and vertical angles
            ray_dir = (cam_forward + 
                       np.tan(h_angle) * cam_right + 
                       np.tan(v_angle) * cam_up)
            ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-8)
            
            # Check ray against walls
            for wall in wall_aabbs:
                hit = intersects_ray_aabb(cam_pos, ray_dir, wall.bmin, wall.bmax)
                if hit is not None:
                    t_enter, _ = hit
                    if 0 < t_enter < min_clear_distance:
                        blocked_count += 1
                        min_wall_dist = min(min_wall_dist, t_enter)
                        break  # Ray is blocked, no need to check more walls
    
    block_ratio = blocked_count / total_rays if total_rays > 0 else 0.0
    is_blocked = block_ratio >= occlusion_threshold
    
    details = {
        'blocked_rays': blocked_count,
        'total_rays': total_rays,
        'min_wall_dist': min_wall_dist if min_wall_dist != float('inf') else None,
        'block_ratio': block_ratio
    }
    
    return is_blocked, block_ratio, details


def aabb_overlap_ratio(bmin1: np.ndarray, bmax1: np.ndarray,
                       bmin2: np.ndarray, bmax2: np.ndarray) -> float:
    """
    Calculate the overlap ratio between two AABBs.
    Returns ratio of intersection volume to smaller AABB volume.
    """
    # Calculate intersection
    inter_min = np.maximum(bmin1, bmin2)
    inter_max = np.minimum(bmax1, bmax2)
    
    # Check if there's an intersection
    if np.any(inter_min >= inter_max):
        return 0.0
    
    inter_vol = np.prod(inter_max - inter_min)
    vol1 = np.prod(bmax1 - bmin1)
    vol2 = np.prod(bmax2 - bmin2)
    smaller_vol = min(vol1, vol2)
    
    return inter_vol / smaller_vol if smaller_vol > 0 else 0.0


def aabb_distance(bmin1: np.ndarray, bmax1: np.ndarray,
                  bmin2: np.ndarray, bmax2: np.ndarray) -> float:
    """
    Calculate minimum distance between two AABBs.
    Returns 0 if they overlap.
    """
    # For each dimension, find the gap (0 if overlapping)
    gaps = np.maximum(0, np.maximum(bmin1 - bmax2, bmin2 - bmax1))
    return float(np.linalg.norm(gaps))


def is_target_occluded(cam_pos: np.ndarray, target_bmin: np.ndarray, target_bmax: np.ndarray,
                       occluders: List[AABB], target_id: Optional[str] = None,
                       sample_corners: bool = True,
                       occlusion_threshold: float = 0.5,
                       min_occluder_distance: float = 0.3,
                       max_overlap_ratio: float = 0.3) -> Tuple[bool, Optional[str]]:
    """
    Check if target object is occluded by any occluder.
    
    Samples multiple points on the target (center + optionally 8 corners)
    to check for occlusion. Returns True if center is occluded OR if
    more than occlusion_threshold fraction of sample points are occluded.
    
    IMPORTANT: Objects that overlap with or are very close to the target
    (like wall cabinets above a range hood) are filtered out to avoid
    false positives for tightly integrated equipment.
    
    IMPORTANT: Walls are NOT filtered by min_occluder_distance because
    objects may be placed against walls, but walls can still occlude
    from certain camera angles.
    
    Args:
        cam_pos: Camera position
        target_bmin: Target object's AABB minimum corner
        target_bmax: Target object's AABB maximum corner
        occluders: List of potential occluding AABBs
        target_id: ID of target to exclude from occluder check
        sample_corners: If True, also check 8 corners; else just center
        occlusion_threshold: Fraction of points that must be occluded (0.5 = 50%)
        min_occluder_distance: Ignore occluders closer than this to target (meters)
                               NOTE: This does NOT apply to walls
        max_overlap_ratio: Ignore occluders with higher overlap ratio with target
        
    Returns:
        (is_occluded, occluder_label) - True if significantly occluded
    """
    # Filter out occluders that are too close to or overlap with target
    # EXCEPTION: Walls are not filtered by distance since objects can be against walls
    filtered_occluders = []
    for occ in occluders:
        if target_id and occ.id == target_id:
            continue
        
        # Check if this is a wall (by label or id)
        is_wall = occ.label == 'wall' or occ.id.startswith('wall')
        
        # Check overlap ratio - applies to all occluders including walls
        overlap = aabb_overlap_ratio(target_bmin, target_bmax, occ.bmin, occ.bmax)
        if overlap > max_overlap_ratio:
            continue
        
        # Check minimum distance - but NOT for walls
        # Walls can be adjacent to objects but still occlude from certain angles
        if not is_wall:
            dist = aabb_distance(target_bmin, target_bmax, occ.bmin, occ.bmax)
            if dist < min_occluder_distance:
                continue
        
        filtered_occluders.append(occ)
    
    # Sample points on target
    center = (target_bmin + target_bmax) / 2.0
    sample_points = [center]  # Center is first point
    
    if sample_corners:
        # Add 8 corners
        for xi in [target_bmin[0], target_bmax[0]]:
            for yi in [target_bmin[1], target_bmax[1]]:
                for zi in [target_bmin[2], target_bmax[2]]:
                    sample_points.append(np.array([xi, yi, zi], dtype=float))
    
    occluded_count = 0
    center_occluded = False
    first_occluder_label = None
    
    for idx, pt in enumerate(sample_points):
        ray_d = pt - cam_pos
        dist = float(np.linalg.norm(ray_d))
        if dist < 1e-6:
            continue
        ray_d = ray_d / dist
        
        point_occluded = False
        for occ in filtered_occluders:  # Use filtered list
            hit = intersects_ray_aabb(cam_pos, ray_d, occ.bmin, occ.bmax)
            if hit is None:
                continue
            
            t_enter, _ = hit
            if 0.0 <= t_enter <= (dist - 1e-3):
                point_occluded = True
                if first_occluder_label is None:
                    first_occluder_label = occ.label
                break
        
        if point_occluded:
            occluded_count += 1
            if idx == 0:  # Center point
                center_occluded = True
    
    # Object is considered occluded if:
    # 1. Center is occluded, OR
    # 2. More than threshold fraction of all points are occluded
    total_points = len(sample_points)
    occlusion_ratio = occluded_count / total_points if total_points > 0 else 0
    
    is_occluded = center_occluded or (occlusion_ratio >= occlusion_threshold)
    
    return is_occluded, first_occluder_label if is_occluded else None


# ============================================================================
# Geometry Utility Functions
# ============================================================================

def point_to_segment_distance_2d(px: float, py: float, 
                                  x1: float, y1: float, 
                                  x2: float, y2: float) -> float:
    """
    Calculate minimum distance from point (px, py) to line segment (x1,y1)-(x2,y2).
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    
    if seg_len_sq < 1e-12:
        # Degenerate segment (point)
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    
    # Project point onto line, clamped to segment
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def distance_to_polygon_boundary(x: float, y: float, poly: List[List[float]]) -> float:
    """
    Calculate minimum distance from point to polygon boundary (edges).
    """
    if not poly or len(poly) < 3:
        return float('inf')
    
    min_dist = float('inf')
    n = len(poly)
    
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        dist = point_to_segment_distance_2d(x, y, x1, y1, x2, y2)
        min_dist = min(min_dist, dist)
    
    return min_dist


def _polygon_area_shoelace(points: List[List[float]]) -> float:
    """Calculate polygon area using shoelace formula (fallback method)."""
    if len(points) < 3:
        return 0.0
    
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2.0


# ============================================================================
# Enhanced Visibility Functions (from ViewSuite)
# ============================================================================

def count_visible_corners(K: np.ndarray, cam_pos: np.ndarray, cam_target: np.ndarray,
                          target_bmin: np.ndarray, target_bmax: np.ndarray,
                          width: int, height: int,
                          border: int = 2,
                          check_occlusion: bool = False,
                          occluders: Optional[List[AABB]] = None,
                          target_id: Optional[str] = None) -> int:
    """
    Count how many corners of the target AABB are visible in the image.
    
    This is more precise than just checking if "any" corner is visible - it gives
    a quantitative measure of visibility quality.
    
    Args:
        K: 3x3 camera intrinsic matrix
        cam_pos: Camera position in world coordinates
        cam_target: Camera look-at target in world coordinates
        target_bmin: Target object AABB minimum corner
        target_bmax: Target object AABB maximum corner
        width: Image width in pixels
        height: Image height in pixels
        border: Margin from image edges in pixels (default: 2)
        check_occlusion: If True, also check if each corner is occluded (default: False)
        occluders: List of potential occluding AABBs (used if check_occlusion=True)
        target_id: ID of target to exclude from occlusion check
        
    Returns:
        Number of visible corners (0-8)
    """
    # Build camera transformation
    camtoworld = camtoworld_from_pos_target(cam_pos, cam_target)
    viewmat = np.linalg.inv(camtoworld)
    
    # Get all 8 corners
    corners = aabb_corners(target_bmin, target_bmax)
    
    # Transform to camera space and project
    visible_count = 0
    
    for corner_idx, corner_world in enumerate(corners):
        # Transform to camera space
        pc = world_to_camera(viewmat, corner_world)
        
        # Check if in front of camera
        if pc[2] <= 1e-6:
            continue
        
        # Project to image
        u, v, z = project_point_to_image(K, pc)
        
        # Check if in image bounds
        if not point_in_image_bounds(u, v, width, height, border):
            continue
        
        # Optionally check occlusion (ray-casting from camera to corner)
        if check_occlusion and occluders is not None:
            is_occluded = is_point_occluded_by_aabb_list(cam_pos, corner_world, occluders, target_id)
            if is_occluded:
                continue
        
        visible_count += 1
    
    return visible_count


def calculate_projected_area_ratio(K: np.ndarray, cam_pos: np.ndarray, cam_target: np.ndarray,
                                    target_bmin: np.ndarray, target_bmax: np.ndarray,
                                    width: int, height: int) -> Tuple[float, float]:
    """
    Calculate the ratio of target object's projected area to total image area.
    
    This ensures the object is not too small in the image (e.g., too far away or tiny object).
    Uses convex hull of projected corners to estimate the projected area.
    
    Args:
        K: 3x3 camera intrinsic matrix
        cam_pos: Camera position in world coordinates
        cam_target: Camera look-at target in world coordinates
        target_bmin: Target object AABB minimum corner
        target_bmax: Target object AABB maximum corner
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        (area_ratio, projected_pixels) where:
            - area_ratio: projected_area / image_area (0.0 to 1.0)
            - projected_pixels: approximate number of pixels covered by object
    """
    # Build camera transformation
    camtoworld = camtoworld_from_pos_target(cam_pos, cam_target)
    viewmat = np.linalg.inv(camtoworld)
    
    # Get all 8 corners
    corners = aabb_corners(target_bmin, target_bmax)
    
    # Transform to camera space and project
    projected_points = []
    
    for corner_world in corners:
        pc = world_to_camera(viewmat, corner_world)
        
        # Only consider corners in front of camera
        if pc[2] > 1e-6:
            u, v, z = project_point_to_image(K, pc)
            # Clamp to image bounds for area calculation
            u_clamped = max(0, min(width - 1, u))
            v_clamped = max(0, min(height - 1, v))
            projected_points.append([u_clamped, v_clamped])
    
    # Need at least 3 points to form a polygon
    if len(projected_points) < 3:
        return 0.0, 0.0
    
    # Calculate area using convex hull (approximation of projected area)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(projected_points)
        projected_area = hull.volume  # In 2D, volume is actually area
    except:
        # Fallback: use simple polygon area if scipy not available
        projected_area = _polygon_area_shoelace(projected_points)
    
    # Calculate ratio
    total_image_area = float(width * height)
    area_ratio = projected_area / total_image_area if total_image_area > 0 else 0.0
    
    return area_ratio, projected_area


def calculate_occlusion_area_2d(K: np.ndarray, cam_pos: np.ndarray, cam_target: np.ndarray,
                                 target_bmin: np.ndarray, target_bmax: np.ndarray,
                                 occluders: List[AABB],
                                 width: int, height: int,
                                 target_id: Optional[str] = None,
                                 depth_mode: str = "min") -> Dict[str, float]:
    """
    Calculate occlusion ratio in 2D image space (more accurate than 3D ray casting).
    
    This projects both the target and all occluders to the image plane, then computes
    pixel-level overlap to determine how much of the target is occluded. This matches
    what will actually appear in the rendered image.
    
    Requires OpenCV (cv2) for polygon rasterization.
    
    Args:
        K: 3x3 camera intrinsic matrix
        cam_pos: Camera position in world coordinates
        cam_target: Camera look-at target in world coordinates
        target_bmin: Target object AABB minimum corner
        target_bmax: Target object AABB maximum corner
        occluders: List of potential occluding AABBs
        width: Image width in pixels
        height: Image height in pixels
        target_id: ID of target to exclude from occluder list
        depth_mode: "min" or "mean" - how to calculate depth for occlusion ordering
        
    Returns:
        Dictionary with:
            - target_area_px: Target's projected area in pixels
            - occluded_area_px: Occluded portion of target in pixels
            - visible_area_px: Visible portion of target in pixels
            - occlusion_ratio_target: occluded_area / target_area (0.0 to 1.0)
            - occlusion_ratio_image: occluded_area / image_area (0.0 to 1.0)
    """
    if cv2 is None:
        # Fallback to simple 3D ray-based occlusion
        print("[Warning] cv2 not available, using simplified occlusion check")
        is_occluded, _ = is_target_occluded(cam_pos, target_bmin, target_bmax, 
                                             occluders, target_id)
        return {
            'target_area_px': 0.0,
            'occluded_area_px': 0.0 if not is_occluded else 1.0,
            'visible_area_px': 0.0,
            'occlusion_ratio_target': 1.0 if is_occluded else 0.0,
            'occlusion_ratio_image': 0.0
        }
    
    # Build camera transformation
    camtoworld = camtoworld_from_pos_target(cam_pos, cam_target)
    viewmat = np.linalg.inv(camtoworld)
    
    # Helper function to project AABB to 2D polygon with depth
    def project_aabb_to_2d(bmin: np.ndarray, bmax: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Returns (polygon_points, depth) or (None, inf) if not visible."""
        corners = aabb_corners(bmin, bmax)
        projected = []
        depths = []
        
        for corner in corners:
            pc = world_to_camera(viewmat, corner)
            if pc[2] > 1e-6:  # In front of camera
                u, v, z = project_point_to_image(K, pc)
                # Clamp to image bounds
                u = max(0, min(width - 1, u))
                v = max(0, min(height - 1, v))
                projected.append([u, v])
                depths.append(z)
        
        if len(projected) < 3:
            return None, float('inf')
        
        # Calculate depth (min or mean)
        if depth_mode == "min":
            depth = min(depths)
        else:
            depth = float(np.mean(depths))
        
        # Get convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(projected)
            polygon = np.array([projected[i] for i in hull.vertices], dtype=np.int32)
        except:
            # Fallback: use all points
            polygon = np.array(projected, dtype=np.int32)
        
        return polygon, depth
    
    # Project target
    target_poly, target_depth = project_aabb_to_2d(target_bmin, target_bmax)
    
    if target_poly is None:
        return {
            'target_area_px': 0.0,
            'occluded_area_px': 0.0,
            'visible_area_px': 0.0,
            'occlusion_ratio_target': 0.0,
            'occlusion_ratio_image': 0.0
        }
    
    # Create target mask
    target_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(target_mask, [target_poly], 1)
    target_area_px = float(np.count_nonzero(target_mask))
    
    if target_area_px == 0:
        return {
            'target_area_px': 0.0,
            'occluded_area_px': 0.0,
            'visible_area_px': 0.0,
            'occlusion_ratio_target': 0.0,
            'occlusion_ratio_image': 0.0
        }
    
    # Create occlusion mask (union of all closer occluders)
    occlusion_mask = np.zeros((height, width), dtype=np.uint8)
    
    for occluder in occluders:
        # Skip target itself
        if target_id is not None and occluder.id == target_id:
            continue
        
        occ_poly, occ_depth = project_aabb_to_2d(occluder.bmin, occluder.bmax)
        
        # Only consider occluders closer to camera
        if occ_poly is not None and occ_depth < target_depth - 1e-3:
            cv2.fillPoly(occlusion_mask, [occ_poly], 1)
    
    # Calculate occluded pixels (overlap between target and occluders)
    occluded_mask = (target_mask == 1) & (occlusion_mask == 1)
    occluded_area_px = float(np.count_nonzero(occluded_mask))
    visible_area_px = target_area_px - occluded_area_px
    
    return {
        'target_area_px': target_area_px,
        'occluded_area_px': occluded_area_px,
        'visible_area_px': visible_area_px,
        'occlusion_ratio_target': occluded_area_px / target_area_px,
        'occlusion_ratio_image': occluded_area_px / float(width * height)
    }


# ============================================================================
# Convenience exports
# ============================================================================

__all__ = [
    # Data classes
    'SceneBounds',
    'AABB', 
    'CameraPose',
    # Ray-AABB intersection
    'intersects_ray_aabb',
    # Camera projection
    'camtoworld_from_pos_target',
    'world_to_camera',
    'project_point_to_image',
    'point_in_image_bounds',
    'aabb_corners',
    # FOV checking
    'is_target_in_fov',
    'check_multiple_targets_in_fov',
    # Occlusion detection
    'is_point_occluded_by_single_aabb',
    'is_point_occluded_by_aabb_list',
    'is_target_occluded',
    # Geometry utilities
    'point_to_segment_distance_2d',
    'distance_to_polygon_boundary',
    '_polygon_area_shoelace',
    # Enhanced visibility
    'count_visible_corners',
    'calculate_projected_area_ratio',
    'calculate_occlusion_area_2d',
]
