"""
Path Generator for Active Spatial Navigation
=============================================

This module generates navigation paths from initial camera positions to target positions
using a two-phase navigation strategy.

The path generation uses the target_region information from generated training data
to find the optimal target position and generate a path to reach it.

Action Space (from env.py):
- move_forward: Move 0.1m in camera forward direction
- move_backward: Move 0.1m backward
- turn_left: Rotate -5 degrees yaw
- turn_right: Rotate +5 degrees yaw
- look_up: Rotate +5 degrees pitch
- look_down: Rotate -5 degrees pitch
- done: Terminate episode

Algorithm (Two-Phase Navigation):
Phase 1 - Select Target Position:
    1. Sample multiple candidate positions from target_region
    2. Validate each candidate using CameraSampler
    3. Select the valid position closest to init_camera

Phase 2 - Navigate to Target:
    Stage A (Position): Move towards target position using turn + move_forward
    Stage B (Orientation): Adjust camera orientation to face target object
"""

import json
import math
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from scipy.spatial.transform import Rotation as R
import copy

# Import CameraSampler for position validation
from camera_sampler import CameraSampler, SceneBounds
from camera_utils import AABB
from config import CameraSamplingConfig


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PathConfig:
    """Configuration for path generation."""
    step_translation: float = 0.1      # meters per step
    step_rotation_deg: float = 5.0     # degrees per step
    max_steps: int = 200               # maximum steps per path
    reward_threshold: float = 0.95     # stop when reward > this
    no_improve_limit: int = 20         # stop if no improvement for N steps
    agent_height: float = 1.5          # camera height in meters
    verbose: bool = False
    gs_root: str = None                # Root directory for 3DGS scenes (for validation)
    validate_positions: bool = True    # Whether to validate positions with CameraSampler
    
    # Minimum path complexity requirements
    min_steps: int = 3                 # Minimum steps required (ensures meaningful paths)
    min_distance: float = 0.3          # Minimum distance to target (meters), 0.3m = 3 forward steps
    min_yaw_offset_deg: float = 15.0   # Minimum yaw offset (degrees), 15 = 3 turn steps


@dataclass
class PathStep:
    """A single step in the path."""
    step_idx: int
    action: str
    position: List[float]      # [x, y, z]
    forward: List[float]       # [fx, fy, fz] normalized
    rotation_euler: List[float] # [rx, ry, rz] in degrees
    reward: float
    extrinsics: List[List[float]]  # 4x4 matrix


@dataclass
class GeneratedPath:
    """Complete generated path."""
    data_idx: int              # index in training data
    scene_id: str
    task_type: str
    task_description: str
    object_label: str
    init_camera: Dict[str, Any]
    target_region: Dict[str, Any]
    sample_target: List[float]
    path: List[PathStep]
    final_reward: float
    success: bool
    total_steps: int


# =============================================================================
# Camera Pose Manipulation (copied from utils.py for self-containment)
# =============================================================================

class CameraPose:
    """Camera pose represented as 4x4 camera-to-world matrix."""
    
    def __init__(self, c2w: np.ndarray = None):
        if c2w is None:
            self.c2w = np.eye(4, dtype=np.float64)
        else:
            self.c2w = c2w.astype(np.float64)
    
    @classmethod
    def from_extrinsics(cls, extrinsics: np.ndarray) -> 'CameraPose':
        """Create from 4x4 extrinsics matrix (camera-to-world)."""
        return cls(np.array(extrinsics, dtype=np.float64))
    
    @property
    def position(self) -> np.ndarray:
        """Camera position in world coordinates."""
        return self.c2w[:3, 3].copy()
    
    @property
    def forward(self) -> np.ndarray:
        """Camera forward direction (Z axis in camera frame)."""
        return self.c2w[:3, 2].copy()
    
    @property
    def right(self) -> np.ndarray:
        """Camera right direction (X axis in camera frame)."""
        return self.c2w[:3, 0].copy()
    
    @property
    def up(self) -> np.ndarray:
        """Camera up direction (Y axis in camera frame)."""
        return self.c2w[:3, 1].copy()
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix."""
        return self.c2w[:3, :3].copy()
    
    @property
    def euler_degrees(self) -> np.ndarray:
        """Euler angles in degrees [rx, ry, rz]."""
        return R.from_matrix(self.rotation_matrix).as_euler('xyz', degrees=True)
    
    def copy(self) -> 'CameraPose':
        return CameraPose(self.c2w.copy())
    
    def move_forward(self, distance: float) -> 'CameraPose':
        """Move camera forward along its view direction."""
        new_pose = self.copy()
        direction = new_pose.forward
        new_pose.c2w[:3, 3] += direction * distance
        return new_pose
    
    def yaw(self, angle_rad: float) -> 'CameraPose':
        """
        Rotate camera around the world Z axis (vertical/up axis).
        
        This is the correct yaw rotation for navigation:
        - Positive angle = counterclockwise when viewed from above = turn left
        - Negative angle = clockwise when viewed from above = turn right
        """
        new_pose = self.copy()
        # Rotation around world Z axis
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R_world_z = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        # Apply world rotation: R_new = R_world @ R_old
        new_pose.c2w[:3, :3] = R_world_z @ new_pose.c2w[:3, :3]
        return new_pose
    
    def pitch(self, angle_rad: float) -> 'CameraPose':
        """Rotate camera around its local X axis (pitch)."""
        new_pose = self.copy()
        R_local = R.from_euler("x", angle_rad, degrees=False).as_matrix()
        new_pose.c2w[:3, :3] = new_pose.c2w[:3, :3] @ R_local
        return new_pose
    
    def to_extrinsics(self) -> np.ndarray:
        """Return 4x4 camera-to-world extrinsics."""
        return self.c2w.copy()


def apply_action(pose: CameraPose, action: str, config: PathConfig) -> CameraPose:
    """
    Apply an action to a camera pose and return new pose.
    
    Uses OpenCV/COLMAP convention where:
    - X axis: right
    - Y axis: down  
    - Z axis: forward (camera looks along +Z)
    
    Note: In OpenCV convention, Y points down, so:
    - Yaw (turn left/right) is rotation around the -Y axis (world up)
    - Pitch (look up/down) is rotation around the X axis
    """
    step_t = config.step_translation
    step_r = np.radians(config.step_rotation_deg)
    
    if action == "move_forward":
        return pose.move_forward(step_t)
    elif action == "move_backward":
        return pose.move_forward(-step_t)
    elif action == "turn_left":
        # In OpenCV (Y down), rotating around local Y axis with positive angle
        # causes counterclockwise rotation when viewed from below (Y axis direction)
        # which is clockwise from above, i.e., turning right.
        # So turn_left needs positive angle in OpenCV convention.
        return pose.yaw(step_r)
    elif action == "turn_right":
        return pose.yaw(-step_r)
    elif action == "look_up":
        # Pitch around X axis: in OpenCV (Y down, Z forward),
        # negative X rotation tilts camera up
        return pose.pitch(-step_r)
    elif action == "look_down":
        return pose.pitch(step_r)
    else:
        return pose.copy()


# =============================================================================
# Reward Functions for Each Task Type
# =============================================================================

def normalize_vec(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    """2D distance (XY plane)."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def distance_3d(a: np.ndarray, b: np.ndarray) -> float:
    """3D distance."""
    return np.linalg.norm(np.array(a[:3]) - np.array(b[:3]))


def gaussian_reward(error: float, alpha: float = 1.0) -> float:
    """Gaussian reward: 1.0 at error=0, decays smoothly."""
    return math.exp(-alpha * (error ** 2))


def compute_forward_alignment(current_forward: np.ndarray, 
                               target_forward: np.ndarray) -> float:
    """Compute alignment reward based on forward direction matching."""
    current_norm = normalize_vec(current_forward)
    target_norm = normalize_vec(target_forward)
    cos_sim = np.dot(current_norm, target_norm)
    # Map from [-1, 1] to [0, 1]
    return max(0, (cos_sim + 1) / 2)


def compute_region_reward(pose: CameraPose, target_region: Dict[str, Any]) -> float:
    """
    Compute reward based on target_region constraints.
    
    Different region types have different optimal position requirements:
    - POINT: Must be at exact point
    - CIRCLE: Must be at any point on circle of given radius from center
    - LINE: Must be at any point on line
    - HALF_PLANE: Must be in half plane
    - RAY: Must be on ray
    - ANNULUS: Must be in annulus (between min and max radius)
    - CURVE: Must be on curve
    """
    region_type = target_region.get('type', 'circle').lower()
    params = target_region.get('params', {})
    sample_point = np.array(target_region.get('sample_point', [0, 0, 1.5]))
    height = target_region.get('height', 1.5)
    
    current_pos = pose.position
    current_pos_2d = current_pos[:2]
    
    if region_type == 'point':
        # Must be exactly at the point
        error = distance_3d(current_pos, sample_point)
        return gaussian_reward(error, alpha=2.0)
    
    elif region_type == 'circle':
        # Must be at distance 'radius' from center
        center = np.array(params.get('center', sample_point[:2]))
        target_radius = params.get('radius', 1.0)
        current_dist = distance_2d(current_pos, np.append(center, 0))
        error = abs(current_dist - target_radius)
        return gaussian_reward(error, alpha=1.0)
    
    elif region_type == 'line':
        # Must be on the perpendicular bisector line
        # Compute distance to line
        midpoint = np.array(params.get('midpoint', sample_point[:2]))
        direction = np.array(params.get('direction', [1, 0]))
        
        # Project current position onto line
        to_current = current_pos_2d - midpoint
        proj_length = np.dot(to_current, direction)
        proj_point = midpoint + proj_length * direction
        error = np.linalg.norm(current_pos_2d - proj_point)
        return gaussian_reward(error, alpha=1.0)
    
    elif region_type == 'half_plane':
        # Must be on correct side of dividing line
        normal = np.array(params.get('normal', [1, 0]))
        point_on_line = np.array(params.get('point_on_line', sample_point[:2]))
        
        # Distance from line (positive = correct side)
        signed_dist = np.dot(current_pos_2d - point_on_line, normal)
        # Reward is high if on correct side (positive signed distance)
        return max(0, min(1, 0.5 + 0.5 * np.tanh(signed_dist)))
    
    elif region_type == 'ray':
        # Must be on ray extending from origin in direction
        origin = np.array(params.get('origin', sample_point[:2]))
        direction = normalize_vec(np.array(params.get('direction', [1, 0])))
        
        # Project onto ray
        to_current = current_pos_2d - origin
        proj_length = np.dot(to_current, direction)
        
        if proj_length < 0:
            # Behind origin - penalize
            return gaussian_reward(abs(proj_length), alpha=0.5) * 0.5
        
        # Distance from ray
        proj_point = origin + max(0, proj_length) * direction
        error = np.linalg.norm(current_pos_2d - proj_point)
        return gaussian_reward(error, alpha=1.0)
    
    elif region_type == 'annulus':
        # Must be between min_radius and max_radius from center
        center = np.array(params.get('center', sample_point[:2]))
        min_radius = params.get('min_radius', 1.0)
        max_radius = params.get('max_radius', 3.0)
        
        current_dist = distance_2d(current_pos, np.append(center, 0))
        
        if min_radius <= current_dist <= max_radius:
            # In annulus - reward is 1.0
            return 1.0
        elif current_dist < min_radius:
            error = min_radius - current_dist
            return gaussian_reward(error, alpha=1.0)
        else:
            error = current_dist - max_radius
            return gaussian_reward(error, alpha=0.5)
    
    elif region_type == 'curve':
        # For curve, use distance to sample point as approximation
        error = distance_3d(current_pos, sample_point)
        return gaussian_reward(error, alpha=0.3)
    
    else:
        # Default: use distance to sample point
        error = distance_3d(current_pos, sample_point)
        return gaussian_reward(error, alpha=0.5)


def compute_reward(pose: CameraPose, data_item: Dict[str, Any]) -> float:
    """
    Compute reward for a given camera pose based on distance to sample_target.
    
    Simplified reward: primarily based on distance to the known-good sample_target,
    with a small bonus for correct orientation.
    
    Returns:
        Reward value in [0, 1], where 1.0 = at sample_target with correct orientation
    """
    sample_target = np.array(data_item.get('sample_target', [0, 0, 1.5]))
    target_forward = np.array(data_item.get('camera_params', {}).get('forward', [0, 0, 1]))
    
    current_pos = pose.position
    current_forward = pose.forward
    
    # Primary reward: distance to sample_target (XY plane)
    # Use 2D distance since height is fixed
    position_error_2d = np.linalg.norm(current_pos[:2] - sample_target[:2])
    # Reward = 1.0 when error = 0, decreases with distance
    # Using alpha=2.0 for sharper falloff: reward > 0.95 when within 0.15m
    position_reward = gaussian_reward(position_error_2d, alpha=2.0)
    
    # Secondary reward: orientation alignment
    orientation_reward = compute_forward_alignment(current_forward, target_forward)
    
    # Combine: 70% position, 30% orientation
    # This ensures reaching the position is the primary goal
    reward = 0.7 * position_reward + 0.3 * orientation_reward
    
    return float(reward)


# =============================================================================
# Path Generator
# =============================================================================

ACTIONS = [
    "move_forward",
    "move_backward", 
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
]

# Movement actions only (no rotation)
MOVE_ACTIONS = ["move_forward", "move_backward"]

# Rotation actions only (yaw)
YAW_ACTIONS = ["turn_left", "turn_right"]

# Pitch actions only
PITCH_ACTIONS = ["look_up", "look_down"]


def sample_positions_from_region(
    target_region: Dict[str, Any],
    num_samples: int = 100,
    height: float = 1.5
) -> List[np.ndarray]:
    """
    Sample candidate positions from a target_region.
    
    Args:
        target_region: Target region definition with type and params
        num_samples: Number of candidate positions to sample
        height: Camera height (z coordinate)
        
    Returns:
        List of [x, y, z] position arrays
    """
    region_type = target_region.get('type', 'circle').lower()
    params = target_region.get('params', {})
    sample_point = np.array(target_region.get('sample_point', [0, 0, height]))
    
    positions = []
    
    if region_type == 'point':
        # Single point - just return it
        positions.append(np.array([sample_point[0], sample_point[1], height]))
        
    elif region_type == 'circle':
        # Sample points on circle perimeter
        center = np.array(params.get('center', sample_point[:2]))
        radius = params.get('radius', 1.0)
        for i in range(num_samples):
            theta = 2 * np.pi * i / num_samples
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            positions.append(np.array([x, y, height]))
            
    elif region_type == 'line':
        # Sample points along line
        midpoint = np.array(params.get('midpoint', sample_point[:2]))
        direction = np.array(params.get('direction', [1, 0]))
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        # Sample from -5m to +5m along line
        for i in range(num_samples):
            t = (i / num_samples - 0.5) * 10.0  # -5 to +5
            pos = midpoint + t * direction
            positions.append(np.array([pos[0], pos[1], height]))
            
    elif region_type == 'half_plane':
        # Sample points in half plane (grid sampling)
        normal = np.array(params.get('normal', [1, 0]))
        point_on_line = np.array(params.get('point_on_line', sample_point[:2]))
        # Generate grid in the correct half
        for i in range(int(np.sqrt(num_samples))):
            for j in range(int(np.sqrt(num_samples))):
                # Sample in a 10m x 10m grid centered on point_on_line
                offset = np.array([i - 5, j - 5]) * 1.0
                pos = point_on_line + offset
                # Check if in correct half
                if np.dot(pos - point_on_line, normal) > 0:
                    positions.append(np.array([pos[0], pos[1], height]))
                    
    elif region_type == 'ray':
        # Sample points along ray
        origin = np.array(params.get('origin', sample_point[:2]))
        direction = np.array(params.get('direction', [1, 0]))
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        for i in range(num_samples):
            t = (i + 1) * 0.2  # 0.2m to 20m
            pos = origin + t * direction
            positions.append(np.array([pos[0], pos[1], height]))
            
    elif region_type == 'annulus':
        # Sample points in annulus
        center = np.array(params.get('center', sample_point[:2]))
        min_radius = params.get('min_radius', 1.0)
        max_radius = params.get('max_radius', 3.0)
        # Sample on multiple rings
        num_rings = 5
        for ring in range(num_rings):
            r = min_radius + (max_radius - min_radius) * ring / (num_rings - 1)
            num_on_ring = num_samples // num_rings
            for i in range(num_on_ring):
                theta = 2 * np.pi * i / num_on_ring
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                positions.append(np.array([x, y, height]))
                
    elif region_type == 'curve':
        # For curves, sample around the sample_point with various distances
        center = sample_point[:2]
        for i in range(num_samples):
            r = 0.5 + (i % 10) * 0.3  # Various radii
            theta = 2 * np.pi * i / num_samples
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            positions.append(np.array([x, y, height]))
    else:
        # Default: sample around sample_point
        for i in range(num_samples):
            r = np.random.uniform(0.5, 3.0)
            theta = np.random.uniform(0, 2 * np.pi)
            x = sample_point[0] + r * np.cos(theta)
            y = sample_point[1] + r * np.sin(theta)
            positions.append(np.array([x, y, height]))
    
    return positions


def load_scene_validation_data(gs_root: str, scene_id: str) -> Tuple[
    Optional[SceneBounds], 
    List[AABB], 
    List[AABB], 
    List[List[List[float]]]
]:
    """
    Load scene data needed for CameraSampler validation.
    
    Returns:
        (scene_bounds, object_aabbs, wall_aabbs, room_polys_coords)
    """
    scene_path = os.path.join(gs_root, scene_id)
    
    scene_bounds = None
    object_aabbs = []
    wall_aabbs = []
    room_polys_coords = []
    
    # Load scene bounds from occupancy.json
    occupancy_path = os.path.join(scene_path, "occupancy.json")
    if os.path.exists(occupancy_path):
        try:
            with open(occupancy_path, 'r') as f:
                occ_data = json.load(f)
            scene_bounds = SceneBounds.from_occupancy(occ_data)
        except Exception as e:
            print(f"[Warning] Failed to load occupancy.json: {e}")
    
    # Load structure data (walls, rooms)
    structure_path = os.path.join(scene_path, "structure.json")
    if os.path.exists(structure_path):
        try:
            with open(structure_path, 'r') as f:
                structure = json.load(f)
            
            # Load room polygon coordinates
            for room in structure.get('rooms', []):
                vertices = room.get('vertices', [])
                if len(vertices) >= 3:
                    room_polys_coords.append([[v[0], v[1]] for v in vertices])
            
            # Load wall AABBs
            for wall in structure.get('walls', []):
                if 'bbox_min' in wall and 'bbox_max' in wall:
                    wall_aabbs.append(AABB(
                        bmin=np.array(wall['bbox_min']),
                        bmax=np.array(wall['bbox_max']),
                        label='wall'
                    ))
        except Exception as e:
            print(f"[Warning] Failed to load structure.json: {e}")
    
    # Load object AABBs
    for obj_file in ['furniture.json', 'objects.json']:
        obj_path = os.path.join(scene_path, obj_file)
        if os.path.exists(obj_path):
            try:
                with open(obj_path, 'r') as f:
                    objects = json.load(f)
                
                for obj in objects:
                    if 'bbox_min' in obj and 'bbox_max' in obj:
                        object_aabbs.append(AABB(
                            bmin=np.array(obj['bbox_min']),
                            bmax=np.array(obj['bbox_max']),
                            label=obj.get('label', 'object')
                        ))
                    elif 'aabb' in obj:
                        aabb = obj['aabb']
                        object_aabbs.append(AABB(
                            bmin=np.array(aabb['min']),
                            bmax=np.array(aabb['max']),
                            label=obj.get('label', 'object')
                        ))
            except Exception as e:
                print(f"[Warning] Failed to load {obj_file}: {e}")
    
    return scene_bounds, object_aabbs, wall_aabbs, room_polys_coords


class PathGenerator:
    """Generate navigation paths using two-phase strategy: move to target, then orient."""
    
    def __init__(self, config: PathConfig = None):
        self.config = config or PathConfig()
        # Create CameraSampler with default config if validation is enabled
        if config and config.validate_positions:
            camera_config = CameraSamplingConfig()
            self.camera_sampler = CameraSampler(camera_config)
        else:
            self.camera_sampler = None
        self._scene_cache = {}  # Cache for scene validation data
    
    def _get_scene_data(self, scene_id: str) -> Tuple[
        Optional[SceneBounds], 
        List[AABB], 
        List[AABB], 
        List[List[List[float]]]
    ]:
        """Get cached scene validation data."""
        if scene_id not in self._scene_cache:
            if self.config.gs_root:
                self._scene_cache[scene_id] = load_scene_validation_data(
                    self.config.gs_root, scene_id
                )
            else:
                self._scene_cache[scene_id] = (None, [], [], [])
        return self._scene_cache[scene_id]
    
    def _validate_position(self, pos: np.ndarray, scene_id: str) -> Tuple[bool, str]:
        """
        Validate if a position is valid using CameraSampler.
        
        Args:
            pos: [x, y, z] position array
            scene_id: Scene ID for loading validation data
            
        Returns:
            (is_valid, rejection_reason)
        """
        if self.camera_sampler is None or not self.config.validate_positions:
            return True, 'validation_disabled'
        
        scene_bounds, object_aabbs, wall_aabbs, room_polys_coords = self._get_scene_data(scene_id)
        
        if not room_polys_coords and scene_bounds is None:
            return True, 'no_validation_data'
        
        return self.camera_sampler.validate_camera_position_full(
            cam_pos=pos,
            room_polys=room_polys_coords,
            scene_bounds=scene_bounds,
            object_aabbs=object_aabbs,
            wall_aabbs=wall_aabbs,
            target_room_idx=None,
            require_in_target_room=False,
            check_collision=True,
            check_wall_dist=True,
            min_wall_distance=0.25
        )
    
    def _validate_pose(self, pose: CameraPose, scene_id: str) -> Tuple[bool, str]:
        """Validate a CameraPose."""
        return self._validate_position(pose.position, scene_id)
    
    def _select_closest_valid_target(
        self, 
        init_pos: np.ndarray,
        target_region: Dict[str, Any],
        scene_id: str,
        height: float = 1.5,
        num_samples: int = 100,
        init_yaw: float = None
    ) -> Optional[np.ndarray]:
        """
        Select a valid target position that requires minimum steps to reach.
        
        The target must be at least min_distance away OR require min yaw offset,
        ensuring the path has meaningful navigation steps.
        
        Args:
            init_pos: Initial camera position
            target_region: Target region definition
            scene_id: Scene ID for validation
            height: Camera height
            num_samples: Number of candidates to sample
            init_yaw: Initial camera yaw angle (radians)
            
        Returns:
            Valid target position with sufficient distance/offset, or None if none found
        """
        # Sample candidate positions from target region
        candidates = sample_positions_from_region(target_region, num_samples, height)
        
        min_dist = self.config.min_distance
        min_yaw_rad = np.radians(self.config.min_yaw_offset_deg)
        
        def compute_yaw_to_target(current_pos, target_pos):
            """Compute yaw angle to face target from current position."""
            direction = target_pos[:2] - current_pos[:2]
            return np.arctan2(direction[1], direction[0])
        
        def normalize_angle(angle):
            """Normalize angle to [-pi, pi]."""
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            return angle
        
        # Filter valid positions
        valid_positions = []
        for pos in candidates:
            is_valid, reason = self._validate_position(pos, scene_id)
            if is_valid:
                dist = distance_2d(init_pos, pos)
                
                # Calculate yaw offset to this target
                yaw_offset = 0.0
                if init_yaw is not None:
                    target_yaw = compute_yaw_to_target(init_pos, pos)
                    yaw_offset = abs(normalize_angle(target_yaw - init_yaw))
                
                # Check if this position requires minimum effort
                meets_distance = dist >= min_dist
                meets_yaw = yaw_offset >= min_yaw_rad
                
                # Estimate total steps needed
                move_steps = int(np.ceil(dist / self.config.step_translation))
                turn_steps = int(np.ceil(yaw_offset / np.radians(self.config.step_rotation_deg)))
                total_steps_estimate = move_steps + turn_steps
                
                valid_positions.append({
                    'pos': pos,
                    'dist': dist,
                    'yaw_offset': yaw_offset,
                    'meets_min': meets_distance or meets_yaw,
                    'total_steps': total_steps_estimate
                })
        
        if not valid_positions:
            # Try the sample_point as fallback
            sample_point = target_region.get('sample_point')
            if sample_point:
                pos = np.array([sample_point[0], sample_point[1], height])
                is_valid, _ = self._validate_position(pos, scene_id)
                if is_valid:
                    return pos
            return None
        
        # Prefer positions that meet minimum requirements
        positions_meeting_min = [p for p in valid_positions if p['meets_min']]
        
        if positions_meeting_min:
            # Among those meeting minimum, choose closest (for efficiency)
            positions_meeting_min.sort(key=lambda x: x['dist'])
            selected = positions_meeting_min[0]
            if self.config.verbose:
                print(f"  Selected target: dist={selected['dist']:.2f}m, "
                      f"yaw_offset={np.degrees(selected['yaw_offset']):.1f}°, "
                      f"est_steps={selected['total_steps']}")
            return selected['pos']
        else:
            # No position meets minimum, choose the one requiring most steps
            valid_positions.sort(key=lambda x: x['total_steps'], reverse=True)
            selected = valid_positions[0]
            if self.config.verbose:
                print(f"  [Warning] No target meets min requirements. "
                      f"Best: dist={selected['dist']:.2f}m, "
                      f"yaw_offset={np.degrees(selected['yaw_offset']):.1f}°")
            return selected['pos']

    def _compute_yaw_to_target(self, current_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """Compute the yaw angle needed to face target_pos from current_pos (in radians)."""
        direction = target_pos[:2] - current_pos[:2]
        return np.arctan2(direction[1], direction[0])
    
    def _get_current_yaw(self, pose: CameraPose) -> float:
        """Get current yaw angle from pose (in radians)."""
        forward = pose.forward
        # Project forward onto XY plane
        forward_2d = np.array([forward[0], forward[1]])
        return np.arctan2(forward_2d[1], forward_2d[0])
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def generate_path(self, data_item: Dict[str, Any], data_idx: int = 0) -> GeneratedPath:
        """
        Generate a path using two-phase navigation strategy.
        
        Phase 1: Select the closest valid target position from target_region
        Phase 2: Navigate to target:
            - Stage A: Turn to face target, then move forward
            - Stage B: Adjust orientation to face target object
        
        Args:
            data_item: Training data with init_camera, target_region, etc.
            data_idx: Index of this item in the dataset
            
        Returns:
            GeneratedPath with the action sequence
        """
        scene_id = data_item.get('scene_id', 'unknown')
        target_region = data_item.get('target_region', {})
        
        # Initialize camera pose from data
        init_extrinsics = np.array(data_item['init_camera']['extrinsics'])
        current_pose = CameraPose.from_extrinsics(init_extrinsics)
        init_pos = current_pose.position
        height = init_pos[2]
        
        # Path tracking
        path_steps: List[PathStep] = []
        step_idx = 0
        
        # Add initial step
        init_reward = compute_reward(current_pose, data_item)
        path_steps.append(self._create_path_step(step_idx, "init", current_pose, init_reward))
        step_idx += 1
        
        if self.config.verbose:
            print(f"  Initial position: {init_pos[:2]}, reward: {init_reward:.4f}")
        
        # ===== Phase 1: Use sample_target directly as target position =====
        # The sample_target is already a valid target position computed during data generation
        # No need to re-sample - this guarantees we have a known-good target
        sample_target = data_item.get('sample_target', [0, 0, height])
        target_pos = np.array([sample_target[0], sample_target[1], height])
        
        # Get initial yaw for estimating path complexity
        init_yaw = self._get_current_yaw(current_pose)
        
        if self.config.verbose:
            dist_to_target = distance_2d(init_pos, target_pos)
            print(f"  Target position (sample_target): {target_pos[:2]}, distance: {dist_to_target:.2f}m")
        
        # Get target look-at point (for orientation phase)
        target_forward = np.array(data_item.get('camera_params', {}).get('forward', [0, 0, 1]))
        # Try to get object center to look at
        look_at_point = None
        if 'target_region' in data_item:
            params = target_region.get('params', {})
            if 'center' in params:
                center = params['center']
                look_at_point = np.array([center[0], center[1], height]) if len(center) == 2 else np.array(center)
            elif 'object_center' in params:
                look_at_point = np.array(params['object_center'])
        if look_at_point is None:
            sample_target = data_item.get('sample_target', [0, 0, height])
            # Look at somewhere ahead of target position
            look_at_point = target_pos + target_forward * 2.0
        
        # ===== Pre-navigation: Ensure minimum path complexity =====
        # Calculate estimated steps to reach target
        current_pos = current_pose.position
        dist_to_target = distance_2d(current_pos, target_pos)
        position_threshold = 0.15  # 15cm tolerance
        
        desired_yaw = self._compute_yaw_to_target(current_pos, target_pos)
        current_yaw = self._get_current_yaw(current_pose)
        yaw_error = self._normalize_angle(desired_yaw - current_yaw)
        
        # Estimate actual steps needed
        # If distance is within threshold, no move steps needed
        if dist_to_target <= position_threshold:
            move_steps = 0
        else:
            move_steps = int(np.ceil((dist_to_target - position_threshold) / self.config.step_translation))
        
        # Turn steps are only needed if we actually need to turn to face the target
        # For very close targets, minimal turning is needed to reach the position
        turn_steps = int(np.ceil(abs(yaw_error) / np.radians(self.config.step_rotation_deg)))
        
        # Key insight: when target is very close, turn steps don't count because
        # we can reach the position without facing it first
        if dist_to_target <= position_threshold:
            actual_nav_steps = 0  # No navigation needed, position is already reached
        else:
            actual_nav_steps = move_steps + turn_steps
        
        # Always ensure minimum steps by adding turning steps
        if actual_nav_steps < self.config.min_steps:
            extra_steps = self.config.min_steps - actual_nav_steps
            if self.config.verbose:
                print(f"  Actual nav steps: {actual_nav_steps}, adding {extra_steps} extra turn steps")
            
            # Turn away from target first, then navigation will turn back
            turn_action = "turn_left"  # Turn left first
            for _ in range(extra_steps):
                new_pose = apply_action(current_pose, turn_action, self.config)
                is_valid, _ = self._validate_pose(new_pose, scene_id)
                if is_valid:
                    current_pose = new_pose
                    reward = compute_reward(current_pose, data_item)
                    path_steps.append(self._create_path_step(step_idx, turn_action, current_pose, reward))
                    step_idx += 1
                else:
                    # Try other direction
                    turn_action = "turn_right" if turn_action == "turn_left" else "turn_left"
                    new_pose = apply_action(current_pose, turn_action, self.config)
                    is_valid, _ = self._validate_pose(new_pose, scene_id)
                    if is_valid:
                        current_pose = new_pose
                        reward = compute_reward(current_pose, data_item)
                        path_steps.append(self._create_path_step(step_idx, turn_action, current_pose, reward))
                        step_idx += 1
        
        # ===== Phase 2A: Navigate to target position =====
        
        while step_idx < self.config.max_steps:
            current_pos = current_pose.position
            dist_to_target = distance_2d(current_pos, target_pos)
            
            if dist_to_target < position_threshold:
                if self.config.verbose:
                    print(f"  Step {step_idx}: Reached target position (dist={dist_to_target:.3f}m)")
                break
            
            # Compute desired yaw to face target
            desired_yaw = self._compute_yaw_to_target(current_pos, target_pos)
            current_yaw = self._get_current_yaw(current_pose)
            yaw_error = self._normalize_angle(desired_yaw - current_yaw)
            
            # Decide action: turn or move forward
            yaw_threshold = np.radians(10)  # 10 degrees
            
            if abs(yaw_error) > yaw_threshold:
                # Need to turn first
                if yaw_error > 0:
                    action = "turn_left"
                else:
                    action = "turn_right"
            else:
                # Facing roughly correct direction, move forward
                action = "move_forward"
            
            # Apply action
            new_pose = apply_action(current_pose, action, self.config)
            
            # Validate new position
            is_valid, reason = self._validate_pose(new_pose, scene_id)
            if not is_valid:
                # Try alternative actions
                alternative_found = False
                for alt_action in ACTIONS:
                    if alt_action == action:
                        continue
                    alt_pose = apply_action(current_pose, alt_action, self.config)
                    is_valid, _ = self._validate_pose(alt_pose, scene_id)
                    if is_valid:
                        new_pose = alt_pose
                        action = alt_action
                        alternative_found = True
                        break
                
                if not alternative_found:
                    if self.config.verbose:
                        print(f"  Step {step_idx}: No valid action available, stopping")
                    break
            
            current_pose = new_pose
            reward = compute_reward(current_pose, data_item)
            path_steps.append(self._create_path_step(step_idx, action, current_pose, reward))
            step_idx += 1
        
        # ===== Phase 2B: Adjust orientation to face target object =====
        if self.config.verbose:
            print(f"  Phase 2B: Adjusting orientation...")
        
        # Compute desired orientation towards look_at_point
        orientation_threshold = np.radians(5)  # 5 degrees tolerance
        max_orientation_steps = 50
        orientation_steps = 0
        
        while step_idx < self.config.max_steps and orientation_steps < max_orientation_steps:
            current_pos = current_pose.position
            desired_yaw = self._compute_yaw_to_target(current_pos, look_at_point)
            current_yaw = self._get_current_yaw(current_pose)
            yaw_error = self._normalize_angle(desired_yaw - current_yaw)
            
            if abs(yaw_error) < orientation_threshold:
                if self.config.verbose:
                    print(f"  Step {step_idx}: Orientation aligned (error={np.degrees(yaw_error):.1f}°)")
                break
            
            # Turn towards target
            if yaw_error > 0:
                action = "turn_left"
            else:
                action = "turn_right"
            
            new_pose = apply_action(current_pose, action, self.config)
            current_pose = new_pose
            reward = compute_reward(current_pose, data_item)
            path_steps.append(self._create_path_step(step_idx, action, current_pose, reward))
            step_idx += 1
            orientation_steps += 1
        
        # Add final "done" action
        final_reward = compute_reward(current_pose, data_item)
        path_steps.append(self._create_path_step(step_idx, "done", current_pose, final_reward))
        
        if self.config.verbose:
            print(f"  Final reward: {final_reward:.4f}, total steps: {step_idx}")
        
        return GeneratedPath(
            data_idx=data_idx,
            scene_id=scene_id,
            task_type=data_item.get('task_type', 'unknown'),
            task_description=data_item.get('task_description', ''),
            object_label=data_item.get('object_label', ''),
            init_camera=data_item.get('init_camera', {}),
            target_region=target_region,
            sample_target=data_item.get('sample_target', [0, 0, 1.5]),
            path=[asdict(s) for s in path_steps],
            final_reward=final_reward,
            success=final_reward >= self.config.reward_threshold,
            total_steps=step_idx
        )
    
    def _create_path_step(self, step_idx: int, action: str, 
                          pose: CameraPose, reward: float) -> PathStep:
        """Create a PathStep from current pose."""
        return PathStep(
            step_idx=step_idx,
            action=action,
            position=pose.position.tolist(),
            forward=pose.forward.tolist(),
            rotation_euler=pose.euler_degrees.tolist(),
            reward=reward,
            extrinsics=pose.to_extrinsics().tolist()
        )


# =============================================================================
# Main Functions
# =============================================================================

def filter_by_scene(jsonl_path: Path, scene_id: str) -> List[Tuple[int, Dict[str, Any]]]:
    """Load training data filtered by scene ID."""
    items = []
    with open(jsonl_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            if item.get('scene_id') == scene_id:
                items.append((idx, item))
    return items


def generate_paths_for_scene(
    jsonl_path: Path,
    scene_id: str,
    output_path: Path,
    config: PathConfig = None,
    max_items: int = None
) -> List[GeneratedPath]:
    """
    Generate paths for all items in a specific scene.
    
    Args:
        jsonl_path: Path to training data JSONL
        scene_id: Scene ID to process
        output_path: Path to save generated paths
        config: Path generation configuration
        max_items: Maximum number of items to process (for testing)
    
    Returns:
        List of GeneratedPath objects
    """
    config = config or PathConfig()
    generator = PathGenerator(config)
    
    # Load data for scene
    print(f"Loading data for scene {scene_id}...")
    items = filter_by_scene(jsonl_path, scene_id)
    print(f"Found {len(items)} items for scene {scene_id}")
    
    if max_items:
        items = items[:max_items]
    
    # Generate paths
    generated_paths = []
    for i, (data_idx, item) in enumerate(items):
        if config.verbose or i % 50 == 0:
            print(f"[{i+1}/{len(items)}] Generating path for {item.get('task_type')}...")
        
        path = generator.generate_path(item, data_idx)
        generated_paths.append(path)
        
        if config.verbose:
            print(f"  Final reward: {path.final_reward:.4f}, steps: {path.total_steps}")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for path in generated_paths:
            # Convert to dict for JSON serialization
            path_dict = {
                'data_idx': path.data_idx,
                'scene_id': path.scene_id,
                'task_type': path.task_type,
                'task_description': path.task_description,
                'object_label': path.object_label,
                'init_camera': path.init_camera,
                'target_region': path.target_region,
                'sample_target': path.sample_target,
                'path': path.path,
                'final_reward': path.final_reward,
                'success': path.success,
                'total_steps': path.total_steps,
            }
            f.write(json.dumps(path_dict) + '\n')
    
    print(f"\nSaved {len(generated_paths)} paths to {output_path}")
    
    # Summary statistics
    successful = sum(1 for p in generated_paths if p.success)
    avg_steps = np.mean([p.total_steps for p in generated_paths])
    avg_reward = np.mean([p.final_reward for p in generated_paths])
    
    print(f"\nSummary:")
    print(f"  Success rate: {successful}/{len(generated_paths)} ({100*successful/len(generated_paths):.1f}%)")
    print(f"  Average steps: {avg_steps:.1f}")
    print(f"  Average final reward: {avg_reward:.4f}")
    
    return generated_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate navigation paths for training data")
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Path to training data JSONL file")
    parser.add_argument("--scene_id", type=str, required=True,
                        help="Scene ID to process")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save generated paths")
    parser.add_argument("--gs_root", type=str, default=None,
                        help="Root directory for 3DGS scenes (for position validation)")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per path")
    parser.add_argument("--reward_threshold", type=float, default=0.95,
                        help="Stop when reward exceeds this")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Maximum items to process (for testing)")
    parser.add_argument("--no_validate", action="store_true",
                        help="Disable position validation with CameraSampler")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    config = PathConfig(
        max_steps=args.max_steps,
        reward_threshold=args.reward_threshold,
        verbose=args.verbose,
        gs_root=args.gs_root,
        validate_positions=not args.no_validate and args.gs_root is not None,
    )
    
    generate_paths_for_scene(
        jsonl_path=Path(args.jsonl_path),
        scene_id=args.scene_id,
        output_path=Path(args.output_path),
        config=config,
        max_items=args.max_items,
    )
