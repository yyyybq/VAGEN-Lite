"""
Main Pipeline Module

This module orchestrates the complete active spatial perception dataset generation.
Output format follows DATA_PREPARATION_GUIDE.md specifications.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field

try:
    from .config import PipelineConfig, InitialViewConfig
    from .object_selector import ObjectSelector
    from .camera_sampler import CameraSampler, CameraPose
    from .task_generator import TaskGenerator, TaskResult
except ImportError:
    from config import PipelineConfig, InitialViewConfig
    from object_selector import ObjectSelector
    from camera_sampler import CameraSampler, CameraPose
    from task_generator import TaskGenerator, TaskResult


def validate_target_reachability(task: 'TaskResult', room_polys: List[List[List[float]]]) -> bool:
    """
    Check if the target region is reachable from within any room polygon.
    
    For curve-based tasks (e.g., size_distance_invariance), this checks
    if any of the sampled curve points lie inside a room polygon.
    For circle-based tasks, checks if the circle intersects a room.
    
    Args:
        task: The generated TaskResult
        room_polys: List of room polygons [[x1,y1],[x2,y2],...] for each room
        
    Returns:
        True if the target region is reachable from within a room
    """
    if not room_polys:
        return True  # No room data, can't validate
    
    region = task.target_region
    if region is None:
        return True
    
    params = region.params if hasattr(region, 'params') else {}
    region_type = region.region_type if hasattr(region, 'region_type') else None
    
    def point_in_poly(x, y, poly):
        """Ray-casting point-in-polygon test."""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i][0], poly[i][1]
            xj, yj = poly[j][0], poly[j][1]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside
    
    def point_in_any_room(x, y):
        for poly in room_polys:
            if point_in_poly(x, y, poly):
                return True
        return False
    
    # For CURVE tasks (size_distance_invariance): check sampled curve points
    if hasattr(region_type, 'value') and region_type.value == 'curve' or str(region_type) == 'curve':
        curve_points = params.get("points", [])
        if not curve_points:
            return True
        
        # Check if any curve point is inside a room
        reachable_count = 0
        for pt in curve_points:
            if point_in_any_room(pt[0], pt[1]):
                reachable_count += 1
        
        # Require at least 10% of curve points to be reachable
        return reachable_count >= max(1, len(curve_points) * 0.1)
    
    # For CIRCLE tasks: check if the circle has reachable arc segments
    # Sample points on the circle and check
    center = params.get("object_center", params.get("center", None))
    radius = params.get("radius", params.get("sample_distance", None))
    if center is not None and radius is not None:
        center = np.array(center)
        center_2d = center[:2] if len(center) > 2 else center
        n_samples = 36  # Check every 10 degrees
        reachable_count = 0
        for i in range(n_samples):
            theta = 2 * np.pi * i / n_samples
            x = center_2d[0] + radius * np.cos(theta)
            y = center_2d[1] + radius * np.sin(theta)
            if point_in_any_room(x, y):
                reachable_count += 1
        # Require at least ~10% of the circle to be reachable
        return reachable_count >= max(1, n_samples * 0.1)
    
    return True  # Can't determine, assume reachable


def compute_init_position_score(
    init_pos: np.ndarray,
    init_forward: np.ndarray,
    target_region: Dict[str, Any],
    task_type: str
) -> float:
    """
    Compute a score for the initial camera position relative to the target region.
    
    This is used to filter out initial positions that are already too close to the target,
    ensuring the agent needs to navigate (not start at the optimal position).
    
    Args:
        init_pos: Initial camera position [x, y, z]
        init_forward: Initial camera forward direction [dx, dy, dz]
        target_region: Target region specification
        task_type: Type of task
        
    Returns:
        Score between 0 and 1, where 1 means perfect (at target)
    """
    params = target_region.get("params", {})
    region_type = target_region.get("type", "circle")
    
    # Get relevant points
    sample_point = target_region.get("sample_point")
    if sample_point is not None:
        sample_point = np.array(sample_point)
    
    # Compute position-based score depending on task type
    if task_type in ["absolute_positioning", "delta_control"]:
        # For circle-based tasks: score based on distance to target radius
        object_center = np.array(params.get("object_center", params.get("center", [0, 0, 0])))
        target_radius = params.get("radius", params.get("sample_distance", 2.0))
        
        # Current distance to object center
        current_distance = np.linalg.norm(init_pos[:2] - object_center[:2])
        
        # Score: how close to target radius
        distance_deviation = abs(current_distance - target_radius)
        sigma = max(target_radius * 0.3, 0.5)
        position_score = np.exp(-(distance_deviation ** 2) / (2 * sigma ** 2))
        
    elif task_type == "equidistance":
        # For equidistance tasks: 
        # The sample_point is the target position on the perpendicular bisector.
        # We score based on distance to sample_point, NOT just being on the bisector.
        # This is because the camera sampling already ensures equidistance,
        # but we want the init position to be AWAY from the specific sample_point.
        
        if sample_point is not None:
            # Distance to sample point
            distance_to_target = np.linalg.norm(init_pos[:2] - sample_point[:2])
            # Use a reasonable sigma (1.0m means 1m away gives score ~0.6)
            sigma = 1.0
            position_score = np.exp(-(distance_to_target ** 2) / (2 * sigma ** 2))
        else:
            # Fallback: use distance difference method
            center_a = np.array(params.get("center_a", params.get("object_a_center", [0, 0, 0])))
            center_b = np.array(params.get("center_b", params.get("object_b_center", [1, 0, 0])))
            
            if len(center_a) > 2:
                center_a = center_a[:2]
            if len(center_b) > 2:
                center_b = center_b[:2]
            
            # Distance difference (equidistance means diff = 0)
            dist_a = np.linalg.norm(init_pos[:2] - center_a)
            dist_b = np.linalg.norm(init_pos[:2] - center_b)
            distance_diff = abs(dist_a - dist_b)
            
            obj_distance = np.linalg.norm(center_b - center_a)
            sigma = max(obj_distance * 0.2, 0.5)
            position_score = np.exp(-(distance_diff ** 2) / (2 * sigma ** 2))
        
    elif sample_point is not None:
        # For point-based tasks: score based on distance to sample point
        distance_to_target = np.linalg.norm(init_pos[:2] - sample_point[:2])
        sigma = 1.0
        position_score = np.exp(-(distance_to_target ** 2) / (2 * sigma ** 2))
        
    else:
        # Unknown task type, return neutral score
        position_score = 0.5
    
    return float(np.clip(position_score, 0, 1))


def validate_init_position(
    init_pos: np.ndarray,
    init_forward: np.ndarray,
    target_region: Dict[str, Any],
    task_type: str,
    max_init_score: float = 0.7,
    min_distance_to_target: float = 0.5,
    init_view_config: Optional[InitialViewConfig] = None,
) -> Tuple[bool, str, float]:
    """
    Validate that the initial camera position is not too close to the target.
    
    Enhanced with task-aware difficulty control:
    1. Task-specific minimum distance from init to target sample_point
    2. Task-specific maximum initial score
    3. Minimum angular (yaw) offset between init forward and direction-to-target
    4. Minimum estimated total navigation steps
    
    Args:
        init_pos: Initial camera position
        init_forward: Initial camera forward direction
        target_region: Target region specification
        task_type: Type of task
        max_init_score: Maximum allowed initial score (fallback)
        min_distance_to_target: Minimum distance to sample point (fallback)
        init_view_config: Task-aware initial view configuration
        
    Returns:
        (is_valid, reason, score)
    """
    # Use task-aware config if provided, else use legacy global thresholds
    if init_view_config is not None:
        effective_min_dist = init_view_config.get_min_distance(task_type)
        effective_max_score = init_view_config.get_max_init_score(task_type)
        effective_min_yaw = init_view_config.get_min_yaw_offset(task_type)
        effective_min_steps = init_view_config.get_min_total_steps(task_type)
    else:
        effective_min_dist = min_distance_to_target
        effective_max_score = max_init_score
        effective_min_yaw = 0.0  # Legacy: no yaw check
        effective_min_steps = 0  # Legacy: no step check
    
    # Compute score
    score = compute_init_position_score(init_pos, init_forward, target_region, task_type)
    
    # Check 1: Score should not be too high (already too close to optimal)
    if score > effective_max_score:
        return False, f"init_score_too_high_{score:.2f}_max{effective_max_score:.2f}", score
    
    # Check 2: Distance to sample_point must exceed task-specific minimum
    sample_point = target_region.get("sample_point")
    distance_to_target = float('inf')
    if sample_point is not None:
        sample_point = np.array(sample_point)
        distance_to_target = float(np.linalg.norm(init_pos[:2] - sample_point[:2]))
        if distance_to_target < effective_min_dist:
            return False, f"too_close_{distance_to_target:.2f}m_min{effective_min_dist:.1f}m", score
    
    # Check 3: Angular offset — init forward vs direction to target
    if effective_min_yaw > 0 and sample_point is not None:
        dir_to_target = sample_point[:2] - init_pos[:2]
        dir_to_target_norm = np.linalg.norm(dir_to_target)
        if dir_to_target_norm > 1e-6:
            dir_to_target = dir_to_target / dir_to_target_norm
            init_fwd_2d = init_forward[:2]
            init_fwd_2d_norm = np.linalg.norm(init_fwd_2d)
            if init_fwd_2d_norm > 1e-6:
                init_fwd_2d = init_fwd_2d / init_fwd_2d_norm
                cos_angle = np.clip(np.dot(init_fwd_2d, dir_to_target), -1.0, 1.0)
                yaw_offset_deg = float(np.degrees(np.arccos(cos_angle)))
                if yaw_offset_deg < effective_min_yaw:
                    return False, f"yaw_offset_too_small_{yaw_offset_deg:.1f}deg_min{effective_min_yaw:.0f}deg", score
    
    # Check 4: Estimated total navigation steps
    if effective_min_steps > 0 and sample_point is not None:
        # Estimate steps: distance/0.1 + yaw_offset/5
        distance_steps = distance_to_target / 0.1 if distance_to_target < float('inf') else 0
        
        dir_to_target = sample_point[:2] - init_pos[:2]
        dir_to_target_norm = np.linalg.norm(dir_to_target)
        yaw_steps = 0
        if dir_to_target_norm > 1e-6:
            dir_to_target_unit = dir_to_target / dir_to_target_norm
            init_fwd_2d = init_forward[:2]
            init_fwd_2d_norm = np.linalg.norm(init_fwd_2d)
            if init_fwd_2d_norm > 1e-6:
                init_fwd_2d = init_fwd_2d / init_fwd_2d_norm
                cos_angle = np.clip(np.dot(init_fwd_2d, dir_to_target_unit), -1.0, 1.0)
                yaw_offset_deg = float(np.degrees(np.arccos(cos_angle)))
                yaw_steps = yaw_offset_deg / 5.0  # 5 degrees per turn step
        
        estimated_steps = distance_steps + yaw_steps
        if estimated_steps < effective_min_steps:
            return False, f"too_few_steps_{estimated_steps:.0f}_min{effective_min_steps}", score
    
    return True, "ok", score


def _convert_to_native(value: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.bool_, )):
        return bool(value)
    elif isinstance(value, (np.integer,)):
        return int(value)
    elif isinstance(value, (np.floating,)):
        return float(value)
    elif isinstance(value, dict):
        return {k: _convert_to_native(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_convert_to_native(v) for v in value]
    else:
        return value


def generate_intrinsics(width: int = 512, height: int = 512, fov_deg: float = 60.0) -> List[List[float]]:
    """
    Generate camera intrinsics matrix.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels  
        fov_deg: Field of view in degrees
    
    Returns:
        3x3 intrinsics matrix as nested list
    """
    fov_rad = np.radians(fov_deg)
    fx = fy = width / (2 * np.tan(fov_rad / 2))
    cx = width / 2.0
    cy = height / 2.0
    return [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0]
    ]


def look_at_matrix(camera_position: np.ndarray, look_at_point: np.ndarray, 
                   up_vector: np.ndarray = None) -> List[List[float]]:
    """
    Generate camera extrinsics (camera-to-world) matrix using look-at formulation.
    
    Uses OpenCV/COLMAP convention:
    - X axis: right
    - Y axis: down (opposite of world up)
    - Z axis: forward (camera looks along +Z)
    
    Args:
        camera_position: Camera position [x, y, z]
        look_at_point: Point the camera is looking at [x, y, z]
        up_vector: World up direction [x, y, z], defaults to [0, 0, 1]
    
    Returns:
        4x4 camera-to-world matrix as nested list
    """
    if up_vector is None:
        up_vector = np.array([0, 0, 1], dtype=np.float64)
    
    position = np.array(camera_position, dtype=np.float64)
    target = np.array(look_at_point, dtype=np.float64)
    up = np.array(up_vector, dtype=np.float64)
    
    # Calculate camera coordinate system axes
    # Forward: from camera to target (camera looks along +Z in OpenCV convention)
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-10:
        forward = np.array([1, 0, 0], dtype=np.float64)
    else:
        forward = forward / forward_norm
    
    # Right: perpendicular to forward and world up
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-10:
        # forward is parallel to up, choose arbitrary right
        right = np.array([1, 0, 0], dtype=np.float64)
    else:
        right = right / right_norm
    
    # Down: perpendicular to right and forward (OpenCV Y points down)
    # In OpenCV convention, Y axis points down relative to scene
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    
    # OpenCV/COLMAP convention: X=right, Y=down, Z=forward
    R = np.column_stack([right, down, forward])
    
    # Build 4x4 transformation matrix
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R
    c2w[:3, 3] = position
    
    return c2w.tolist()


@dataclass
class TrainingDataItem:
    """
    A single training data item following DATA_PREPARATION_GUIDE.md format.
    
    Required fields:
    - scene_id: Scene identifier (corresponds to PLY file)
    - object_label: Target object label
    - preset: View direction (front/back/left/right/top/bottom or task-specific)
    - distance: Target distance from object (meters)
    - init_camera: Initial camera parameters (intrinsics + extrinsics)
    - target_region: Target region (where agent should navigate to)
    - sample_target: One sampled target position [x, y, z]
    - camera_params: Target camera parameters (forward direction)
    - target_object: Target object info with bounding box (for occlusion detection)
    """
    scene_id: str
    object_label: str
    preset: str
    distance: float
    init_camera: Dict[str, Any]
    target_region: Dict[str, Any]  # The full target region specification
    sample_target: List[float]     # One sampled point for training
    camera_params: Dict[str, Any]
    # Additional metadata
    task_type: str = ""
    task_description: str = ""
    target_object: Optional[Dict[str, Any]] = None  # Target object with bbox for occlusion detection
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'scene_id': self.scene_id,
            'object_label': self.object_label,
            'preset': self.preset,
            'distance': float(self.distance) if isinstance(self.distance, (np.floating, float)) else self.distance,
            'init_camera': _convert_to_native(self.init_camera),
            'target_region': _convert_to_native(self.target_region),
            'sample_target': _convert_to_native(self.sample_target),
            'camera_params': _convert_to_native(self.camera_params),
            'task_type': self.task_type,
            'task_description': self.task_description,
        }
        if self.target_object is not None:
            result['target_object'] = _convert_to_native(self.target_object)
        return result


class ActiveSpatialPipeline:
    """
    Main pipeline for generating active spatial perception dataset.
    
    Pipeline Flow:
        1. Load scene and filter/select objects
        2. For each selected object/pair, sample camera poses
        3. For each camera pose, generate tasks with target positions
        4. Save the dataset in JSONL format
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        config.validate()
        
        self.object_selector = ObjectSelector(config.object_selection)
        self.camera_sampler = CameraSampler(config.camera_sampling)
        self.task_generator = TaskGenerator(config.task_config)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_scene_list(self) -> List[str]:
        """Get list of scenes to process."""
        if self.config.scene_list:
            return self.config.scene_list
        
        # Auto-discover scenes
        scenes_root = Path(self.config.scenes_root)
        scenes = []
        
        for item in scenes_root.iterdir():
            if item.is_dir():
                labels_path = item / 'labels.json'
                if labels_path.exists():
                    scenes.append(item.name)
        
        return sorted(scenes)
    
    def create_training_item(self, scene_name: str, camera_pose: CameraPose,
                              task: TaskResult, objects: Any) -> TrainingDataItem:
        """
        Create a training data item in the required format.
        
        Args:
            scene_name: Scene identifier
            camera_pose: Initial camera pose
            task: Task result with target position
            objects: Object(s) info (SceneObject or dict)
        
        Returns:
            TrainingDataItem in DATA_PREPARATION_GUIDE format
        """
        cfg = self.config.camera_sampling
        
        # Helper to get object center
        def get_obj_center(obj):
            if hasattr(obj, 'center'):
                return np.array(obj.center, dtype=np.float64)
            elif isinstance(obj, dict):
                return np.array(obj.get('center', [0, 0, 0]), dtype=np.float64)
            return np.array([0, 0, 0], dtype=np.float64)
        
        # Helper to extract object bounding box info
        def get_obj_bbox_info(obj) -> Optional[Dict[str, Any]]:
            """Extract bbox_min, bbox_max, center from object."""
            if obj is None:
                return None
            if hasattr(obj, 'aabb_min') and hasattr(obj, 'aabb_max'):
                return {
                    'label': obj.label if hasattr(obj, 'label') else 'unknown',
                    'id': obj.id if hasattr(obj, 'id') else '',
                    'bbox_min': obj.aabb_min.tolist() if hasattr(obj.aabb_min, 'tolist') else list(obj.aabb_min),
                    'bbox_max': obj.aabb_max.tolist() if hasattr(obj.aabb_max, 'tolist') else list(obj.aabb_max),
                    'center': obj.center.tolist() if hasattr(obj.center, 'tolist') else list(obj.center),
                }
            elif isinstance(obj, dict):
                if 'aabb_min' in obj and 'aabb_max' in obj:
                    return {
                        'label': obj.get('label', 'unknown'),
                        'id': obj.get('id', ''),
                        'bbox_min': obj['aabb_min'],
                        'bbox_max': obj['aabb_max'],
                        'center': obj.get('center', [0, 0, 0]),
                    }
            return None
        
        # Generate camera intrinsics
        intrinsics = generate_intrinsics(
            width=cfg.image_width,
            height=cfg.image_height,
            fov_deg=self.config.task_config.fov_vertical
        )
        
        # Generate camera extrinsics (camera looking at target object)
        # Use the target from CameraPose which is set according to object count:
        # - Single object: looks at object center
        # - Two objects: looks at midpoint between them  
        # - Three+ objects: looks at the SECOND object center
        look_at = camera_pose.target
        
        extrinsics = look_at_matrix(camera_pose.position, look_at)
        
        # Build init_camera
        init_camera = {
            'intrinsics': intrinsics,
            'extrinsics': extrinsics
        }
        
        # Build camera_params (target forward direction from task result)
        forward = task.sample_forward if hasattr(task, 'sample_forward') else task.target_region.sample_forward
        camera_params = {
            'forward': forward.tolist() if hasattr(forward, 'tolist') else forward
        }
        
        # Get target region dict
        target_region_dict = task.target_region.to_dict() if hasattr(task.target_region, 'to_dict') else task.target_region
        
        # Get sample point
        sample_pt = task.sample_point if hasattr(task, 'sample_point') else task.target_region.sample_point
        sample_target = sample_pt.tolist() if hasattr(sample_pt, 'tolist') else sample_pt
        
        # Get distance
        dist = task.distance if hasattr(task, 'distance') else task.target_region.params.get('sample_distance', 0.0)
        
        # Build target_object info for occlusion detection
        # For single object: use that object
        # For multiple objects: use the first object (primary target) or combine all
        target_object = None
        if objects:
            obj_list = objects if isinstance(objects, (list, tuple)) else [objects]
            if len(obj_list) == 1:
                # Single object task
                target_object = get_obj_bbox_info(obj_list[0])
            else:
                # Multiple objects - include all object bboxes for comprehensive occlusion check
                target_object = {
                    'objects': [get_obj_bbox_info(o) for o in obj_list if get_obj_bbox_info(o) is not None],
                    'primary': get_obj_bbox_info(obj_list[0]),  # First object as primary
                }
        
        return TrainingDataItem(
            scene_id=scene_name,
            object_label=task.object_label,
            preset=task.preset,
            distance=dist,
            init_camera=init_camera,
            target_region=target_region_dict,
            sample_target=sample_target,
            camera_params=camera_params,
            task_type=task.task_type,
            task_description=task.description,
            target_object=target_object,
        )
    
    def process_scene(self, scene_name: str) -> List[TrainingDataItem]:
        """
        Process a single scene and generate dataset items.
        
        The pipeline automatically determines how many objects to use based on task type:
        - 1 object: absolute_positioning, delta_control, screen_occupancy
        - 2 objects: equidistance, projective_relations, occlusion_alignment, 
                     fov_inclusion, size_distance_invariance
        - 3 objects: centering
        
        Args:
            scene_name: Name of the scene folder
        
        Returns:
            List of TrainingDataItem objects
        """
        scene_path = Path(self.config.scenes_root) / scene_name
        
        if not scene_path.exists():
            print(f"Scene not found: {scene_path}")
            return []
        
        print(f"Processing scene: {scene_name}")
        
        # Categorize enabled tasks by number of objects required
        enabled = set(self.config.task_config.enabled_tasks)
        
        single_object_tasks = {'absolute_positioning', 'delta_control', 'screen_occupancy'}
        two_object_tasks = {'equidistance', 'projective_relations', 'occlusion_alignment', 
                           'fov_inclusion', 'size_distance_invariance'}
        three_object_tasks = {'centering'}
        
        needs_single = bool(enabled & single_object_tasks)
        needs_pair = bool(enabled & two_object_tasks)
        needs_triple = bool(enabled & three_object_tasks)
        
        data_items = []
        
        # Load room polygons for reachability validation
        room_polys = self.camera_sampler.load_room_polys(scene_path)
        reachability_rejection_count = 0
        
        # Get all valid single objects
        if needs_single or needs_pair or needs_triple:
            single_objects = self.object_selector.select_single_objects(scene_path)
            print(f"  Found {len(single_objects)} valid objects")
        else:
            single_objects = []
        
        # Process single-object tasks
        init_pos_rejection_stats = {}  # Track rejection reasons
        
        if needs_single and single_objects:
            single_tasks = list(enabled & single_object_tasks)
            for obj in single_objects:
                # Sample camera poses for this object
                camera_poses = self.camera_sampler.sample_cameras(
                    scene_path, [obj],
                    num_samples=self.config.camera_sampling.num_cameras_per_item
                )
                
                for camera_pose in camera_poses:
                    tasks = self.task_generator.generate_single_object_tasks(
                        obj, camera_pose, single_tasks
                    )
                    for task in tasks:
                        # Skip invalid tasks (e.g., delta_control that would be too close)
                        if not task.is_valid:
                            continue
                        
                        # Validate target region is reachable from within a room
                        if not validate_target_reachability(task, room_polys):
                            reachability_rejection_count += 1
                            continue
                        
                        # NEW: Validate initial position is not too close to target
                        target_region_dict = task.target_region.to_dict() if hasattr(task.target_region, 'to_dict') else task.target_region
                        init_forward = camera_pose.target - camera_pose.position
                        init_forward = init_forward / (np.linalg.norm(init_forward) + 1e-8)
                        
                        is_valid, reason, init_score = validate_init_position(
                            init_pos=camera_pose.position,
                            init_forward=init_forward,
                            target_region=target_region_dict,
                            task_type=task.task_type,
                            init_view_config=self.config.initial_view,
                        )
                        
                        if not is_valid:
                            init_pos_rejection_stats[reason] = init_pos_rejection_stats.get(reason, 0) + 1
                            continue
                        
                        data_item = self.create_training_item(
                            scene_name, camera_pose, task, [obj]
                        )
                        data_items.append(data_item)
        
        # Process two-object tasks
        if needs_pair and len(single_objects) >= 2:
            pair_tasks = list(enabled & two_object_tasks)
            object_pairs = self.object_selector.select_object_pairs(scene_path, single_objects)
            print(f"  Found {len(object_pairs)} valid object pairs")
            
            for obj_a, obj_b in object_pairs:
                # Sample camera poses for this pair
                camera_poses = self.camera_sampler.sample_cameras(
                    scene_path, [obj_a, obj_b],
                    num_samples=self.config.camera_sampling.num_cameras_per_item
                )
                
                for camera_pose in camera_poses:
                    tasks = self.task_generator.generate_two_object_tasks(
                        obj_a, obj_b, camera_pose, pair_tasks
                    )
                    for task in tasks:
                        # Skip invalid tasks
                        if not task.is_valid:
                            continue
                        
                        # Validate target region is reachable from within a room
                        if not validate_target_reachability(task, room_polys):
                            reachability_rejection_count += 1
                            continue
                        
                        # NEW: Validate initial position is not too close to target
                        target_region_dict = task.target_region.to_dict() if hasattr(task.target_region, 'to_dict') else task.target_region
                        init_forward = camera_pose.target - camera_pose.position
                        init_forward = init_forward / (np.linalg.norm(init_forward) + 1e-8)
                        
                        is_valid, reason, init_score = validate_init_position(
                            init_pos=camera_pose.position,
                            init_forward=init_forward,
                            target_region=target_region_dict,
                            task_type=task.task_type,
                            init_view_config=self.config.initial_view,
                        )
                        
                        if not is_valid:
                            init_pos_rejection_stats[reason] = init_pos_rejection_stats.get(reason, 0) + 1
                            continue
                        
                        data_item = self.create_training_item(
                            scene_name, camera_pose, task, [obj_a, obj_b]
                        )
                        data_items.append(data_item)
        
        # Process three-object tasks
        if needs_triple and len(single_objects) >= 3:
            triple_tasks = list(enabled & three_object_tasks)
            object_triples = self.object_selector.select_object_triples(scene_path, single_objects)
            print(f"  Found {len(object_triples)} valid object triples")
            
            for obj_a, obj_b, obj_c in object_triples:
                # Sample camera poses for this triple
                camera_poses = self.camera_sampler.sample_cameras(
                    scene_path, [obj_a, obj_b, obj_c],
                    num_samples=self.config.camera_sampling.num_cameras_per_item
                )
                
                for camera_pose in camera_poses:
                    tasks = self.task_generator.generate_three_object_tasks(
                        obj_a, obj_b, obj_c, camera_pose, triple_tasks
                    )
                    for task in tasks:
                        # Skip invalid tasks
                        if not task.is_valid:
                            continue
                        
                        # Validate target region is reachable from within a room
                        if not validate_target_reachability(task, room_polys):
                            reachability_rejection_count += 1
                            continue
                        
                        # NEW: Validate initial position is not too close to target
                        target_region_dict = task.target_region.to_dict() if hasattr(task.target_region, 'to_dict') else task.target_region
                        init_forward = camera_pose.target - camera_pose.position
                        init_forward = init_forward / (np.linalg.norm(init_forward) + 1e-8)
                        
                        is_valid, reason, init_score = validate_init_position(
                            init_pos=camera_pose.position,
                            init_forward=init_forward,
                            target_region=target_region_dict,
                            task_type=task.task_type,
                            init_view_config=self.config.initial_view,
                        )
                        
                        if not is_valid:
                            init_pos_rejection_stats[reason] = init_pos_rejection_stats.get(reason, 0) + 1
                            continue
                        
                        data_item = self.create_training_item(
                            scene_name, camera_pose, task, [obj_a, obj_b, obj_c]
                        )
                        data_items.append(data_item)
        
        # Log rejection stats
        if init_pos_rejection_stats:
            print(f"  Init position rejections: {init_pos_rejection_stats}")
        if reachability_rejection_count > 0:
            print(f"  Target reachability rejections: {reachability_rejection_count}")
        
        print(f"  Generated {len(data_items)} data items")
        return data_items
    
    def run(self, verbose: bool = True, scene_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run the complete pipeline.
        
        Args:
            verbose: Whether to print progress
            scene_id: If specified, only process this single scene. Otherwise process all scenes.
        
        Returns:
            List of all generated data items as dictionaries
        """
        # If scene_id is specified, only process that scene
        if scene_id:
            return self.run_single_scene(scene_id, verbose=verbose)
        
        scenes = self.get_scene_list()
        
        if verbose:
            print(f"Found {len(scenes)} scenes to process")
            print(f"Output directory: {self.output_dir}")
            print()
        
        all_data_items = []
        scene_stats = {}
        
        for scene_idx, scene_name in enumerate(scenes):
            if verbose:
                print(f"[{scene_idx+1}/{len(scenes)}] ", end='')
            
            data_items = self.process_scene(scene_name)
            scene_stats[scene_name] = len(data_items)
            
            # Save intermediate results if enabled
            if self.config.save_intermediate and data_items:
                scene_output = self.output_dir / 'scenes' / scene_name
                scene_output.mkdir(parents=True, exist_ok=True)
                
                # Save as JSONL
                with open(scene_output / 'data.jsonl', 'w', encoding='utf-8') as f:
                    for item in data_items:
                        f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')
            
            all_data_items.extend([item.to_dict() for item in data_items])
        
        # Save final dataset as JSONL
        dataset_path = self.output_dir / 'train_data.jsonl'
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for item in all_data_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Also save as JSON for compatibility
        with open(self.output_dir / 'dataset.json', 'w', encoding='utf-8') as f:
            json.dump(all_data_items, f, indent=2, ensure_ascii=False)
        
        # Compute statistics
        stats = self._compute_statistics(all_data_items)
        
        # Save metadata with statistics
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'total_items': len(all_data_items),
            'num_scenes': len(scenes),
            'scene_stats': scene_stats,
            'statistics': stats,
            'config': self.config.to_dict()
        }
        
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        if verbose:
            self._print_statistics(stats, f"Dataset Statistics ({len(scenes)} scenes)")
            print(f"\n✅ Pipeline completed!")
            print(f"   Dataset saved to: {dataset_path}")
        
        return all_data_items
    
    def run_single_scene(self, scene_name: str, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run pipeline for a single scene.
        
        Args:
            scene_name: Name/ID of the scene to process (e.g., '0267_840790')
            verbose: Whether to print progress
        
        Returns:
            List of data items as dictionaries
        """
        if verbose:
            print(f"Processing single scene: {scene_name}")
            print(f"Output directory: {self.output_dir}")
            print()
        
        data_items = self.process_scene(scene_name)
        
        if not data_items:
            if verbose:
                print(f"No data items generated for scene {scene_name}")
            return []
        
        # Save results to scene-specific directory
        scene_output = self.output_dir / 'scenes' / scene_name
        scene_output.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL (scene-specific)
        scene_jsonl_path = scene_output / 'data.jsonl'
        with open(scene_jsonl_path, 'w', encoding='utf-8') as f:
            for item in data_items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + '\n')
        
        # Also save to main output directory with scene_id suffix
        all_data_dicts = [item.to_dict() for item in data_items]
        
        # Save as JSONL in main output dir
        dataset_path = self.output_dir / f'train_data_{scene_name}.jsonl'
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for item in all_data_dicts:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Also save as JSON for compatibility
        with open(self.output_dir / f'dataset_{scene_name}.json', 'w', encoding='utf-8') as f:
            json.dump(all_data_dicts, f, indent=2, ensure_ascii=False)
        
        # Compute statistics
        stats = self._compute_statistics(all_data_dicts)
        
        # Save metadata with statistics
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'total_items': len(all_data_dicts),
            'num_scenes': 1,
            'scene_stats': {scene_name: len(all_data_dicts)},
            'scene_id': scene_name,
            'statistics': stats,
            'config': self.config.to_dict()
        }
        
        with open(self.output_dir / f'metadata_{scene_name}.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        if verbose:
            self._print_statistics(stats, f"Scene: {scene_name}")
            print(f"\n✅ Pipeline completed for scene: {scene_name}")
            print(f"   Dataset saved to: {dataset_path}")
        
        return all_data_dicts

    def _compute_statistics(self, data_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute detailed statistics for generated data items.
        
        Args:
            data_items: List of data item dictionaries
            
        Returns:
            Dictionary with various statistics
        """
        from collections import Counter, defaultdict
        
        if not data_items:
            return {'total_items': 0}
        
        # Basic counts
        total = len(data_items)
        
        # Task type distribution
        task_types = Counter(item.get('task_type', 'unknown') for item in data_items)
        
        # Object label distribution
        object_labels = Counter(item.get('object_label', 'unknown') for item in data_items)
        
        # Preset distribution
        presets = Counter(item.get('preset', 'unknown') for item in data_items)
        
        # Target region type distribution
        region_types = Counter(
            item.get('target_region', {}).get('type', 'unknown') 
            for item in data_items
        )
        
        # Distance statistics
        distances = [item.get('distance', 0) for item in data_items if item.get('distance')]
        if distances:
            dist_stats = {
                'min': round(min(distances), 3),
                'max': round(max(distances), 3),
                'mean': round(sum(distances) / len(distances), 3),
            }
        else:
            dist_stats = {'min': 0, 'max': 0, 'mean': 0}
        
        # Task type counts by preset (e.g., how many absolute_positioning with 'front' preset)
        task_preset_matrix = defaultdict(Counter)
        for item in data_items:
            task_type = item.get('task_type', 'unknown')
            preset = item.get('preset', 'unknown')
            task_preset_matrix[task_type][preset] += 1
        
        return {
            'total_items': total,
            'task_type_distribution': dict(task_types),
            'object_label_distribution': dict(object_labels),
            'preset_distribution': dict(presets),
            'region_type_distribution': dict(region_types),
            'distance_stats': dist_stats,
            'task_preset_matrix': {k: dict(v) for k, v in task_preset_matrix.items()},
        }
    
    def _print_statistics(self, stats: Dict[str, Any], title: str = "Dataset Statistics"):
        """
        Print statistics in a formatted way.
        
        Args:
            stats: Statistics dictionary from _compute_statistics
            title: Title for the statistics section
        """
        print()
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
        
        print(f"\n📊 Total Data Items: {stats.get('total_items', 0)}")
        
        # Task type distribution
        task_dist = stats.get('task_type_distribution', {})
        if task_dist:
            print(f"\n📋 Task Type Distribution ({len(task_dist)} types):")
            print("-" * 50)
            sorted_tasks = sorted(task_dist.items(), key=lambda x: -x[1])
            for task_type, count in sorted_tasks:
                pct = 100 * count / stats['total_items']
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  {task_type:30s} {count:5d} ({pct:5.1f}%) {bar}")
        
        # Region type distribution
        region_dist = stats.get('region_type_distribution', {})
        if region_dist:
            print(f"\n🎯 Target Region Types ({len(region_dist)} types):")
            print("-" * 50)
            for region_type, count in sorted(region_dist.items(), key=lambda x: -x[1]):
                pct = 100 * count / stats['total_items']
                print(f"  {region_type:20s} {count:5d} ({pct:5.1f}%)")
        
        # Preset distribution
        preset_dist = stats.get('preset_distribution', {})
        if preset_dist:
            print(f"\n🧭 Preset Distribution ({len(preset_dist)} types):")
            print("-" * 50)
            for preset, count in sorted(preset_dist.items(), key=lambda x: -x[1]):
                pct = 100 * count / stats['total_items']
                print(f"  {preset:20s} {count:5d} ({pct:5.1f}%)")
        
        # Distance statistics
        dist_stats = stats.get('distance_stats', {})
        if dist_stats and dist_stats.get('max', 0) > 0:
            print(f"\n📏 Distance Statistics:")
            print("-" * 50)
            print(f"  Min:  {dist_stats['min']:.3f} m")
            print(f"  Max:  {dist_stats['max']:.3f} m")
            print(f"  Mean: {dist_stats['mean']:.3f} m")
        
        # Object label distribution (top 10)
        obj_dist = stats.get('object_label_distribution', {})
        if obj_dist:
            print(f"\n🪑 Top Object Labels ({len(obj_dist)} unique):")
            print("-" * 50)
            sorted_objs = sorted(obj_dist.items(), key=lambda x: -x[1])[:10]
            for label, count in sorted_objs:
                pct = 100 * count / stats['total_items']
                print(f"  {label:30s} {count:5d} ({pct:5.1f}%)")
            if len(obj_dist) > 10:
                remaining = sum(c for l, c in sorted(obj_dist.items(), key=lambda x: -x[1])[10:])
                print(f"  {'... and ' + str(len(obj_dist) - 10) + ' more':30s} {remaining:5d}")
        
        print()
        print("=" * 70)


def run_pipeline(scenes_root: str, output_dir: str, scene_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to run the pipeline.
    
    Args:
        scenes_root: Root directory containing scene folders
        output_dir: Output directory for generated data
        scene_id: If specified, only process this single scene (e.g., '0267_840790')
        **kwargs: Additional configuration options
    
    Returns:
        List of all generated data items
    
    Examples:
        # Process all scenes
        run_pipeline('/path/to/scenes', '/path/to/output')
        
        # Process only one specific scene
        run_pipeline('/path/to/scenes', '/path/to/output', scene_id='0267_840790')
    """
    config = PipelineConfig(
        scenes_root=scenes_root,
        output_dir=output_dir,
        **kwargs
    )
    
    pipeline = ActiveSpatialPipeline(config)
    return pipeline.run(scene_id=scene_id)
