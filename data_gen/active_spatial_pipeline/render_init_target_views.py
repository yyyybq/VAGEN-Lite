#!/usr/bin/env python3
"""
Render Initial and Target Views from Training Data
===================================================

This script renders the initial camera view and multiple target views sampled
from the target region defined in training data. It uses the shared SceneRenderer
from render_utils.py for actual rendering.

Key features:
1. Uses CameraSampler validation to ensure target views are within scene bounds
2. Samples multiple valid positions from each target_region
3. Camera always faces the target object(s)

Usage:
    python render_init_target_views.py \
        --jsonl_path ../../data/active_spatial/train_data.jsonl \
        --output_dir ../../data/active_spatial/rendered_views \
        --gs_root /scratch/by2593/project/Active_Spatial/InteriorGS \
        --scene_id 0267_840790 \
        --max_items 20 \
        --num_target_samples 10

Output Structure:
    output_dir/
        scene_id/
            item_000001/
                init_view.png
                target_view_00.png
                target_view_01.png
                ...
                view_info.json
            item_000002/
                ...
        render_summary.json
"""

import json
import os
import sys
import asyncio
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent))

from render_utils import RenderConfig, SceneRenderer, look_at_matrix, compute_intrinsics
from camera_sampler import CameraSampler, SceneBounds
from camera_utils import AABB, CameraPose, is_target_in_fov, is_target_occluded
from config import CameraSamplingConfig

# Try to import shapely for room polygon validation
try:
    from shapely.geometry import Point, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("[Warning] shapely not installed, room polygon validation disabled")


def diverse_sample_items(items: List[Dict[str, Any]], samples_per_category: int = 2) -> List[Dict[str, Any]]:
    """
    Sample items to ensure diversity across task types, distances, presets, and objects.
    
    This function groups items by (task_type, distance, preset) and samples from each group
    to ensure coverage of all categories in the training data.
    
    Args:
        items: List of training data items
        samples_per_category: Number of samples to take from each category
        
    Returns:
        List of diverse items covering all categories
    """
    from collections import defaultdict
    
    # Group items by category
    categories = defaultdict(list)
    
    for idx, item in enumerate(items):
        task_type = item.get('task_type', 'unknown')
        
        # Extract distance from task_description or distance field
        distance = item.get('distance', 0)
        if distance == 0:
            desc = item.get('task_description', '')
            # Try to extract distance like "2.0m" from description
            import re
            match = re.search(r'(\d+\.?\d*)\s*m\b', desc)
            if match:
                distance = float(match.group(1))
        
        # Round distance to avoid float precision issues
        distance = round(distance, 1)
        
        preset = item.get('preset', 'none')
        object_label = item.get('object_label', 'unknown')
        
        # Create category key
        category_key = (task_type, distance, preset)
        categories[category_key].append((idx, item))
    
    # Print category statistics
    print(f"\n  Found {len(categories)} unique categories:")
    category_counts = {}
    for key in sorted(categories.keys()):
        task_type, distance, preset = key
        count = len(categories[key])
        category_counts[key] = count
        if distance > 0:
            print(f"    {task_type} | {distance}m | {preset}: {count} items")
        else:
            print(f"    {task_type} | {preset}: {count} items")
    
    # Sample from each category
    selected_items = []
    selected_indices = set()
    
    for category_key, category_items in categories.items():
        # Shuffle to get random samples from each category
        np.random.shuffle(category_items)
        
        # Take up to samples_per_category items
        for idx, item in category_items[:samples_per_category]:
            if idx not in selected_indices:
                selected_items.append(item)
                selected_indices.add(idx)
    
    # Shuffle the final selection to mix categories
    np.random.shuffle(selected_items)
    
    print(f"  Selected {len(selected_items)} diverse items from {len(categories)} categories\n")
    
    return selected_items


def load_scene_bounds(scene_path: str) -> Optional[SceneBounds]:
    """Load scene bounds from occupancy.json."""
    occupancy_path = os.path.join(scene_path, "occupancy.json")
    if not os.path.exists(occupancy_path):
        print(f"[Warning] occupancy.json not found at {occupancy_path}")
        return None
    
    try:
        with open(occupancy_path, 'r') as f:
            occ_data = json.load(f)
        
        # Use SceneBounds.from_occupancy() which handles all formats correctly
        return SceneBounds.from_occupancy(occ_data)
    except Exception as e:
        print(f"[Warning] Failed to load occupancy.json: {e}")
        return None


def load_room_polygons(scene_path: str) -> List:
    """Load room polygons from structure.json."""
    if not HAS_SHAPELY:
        return []
    
    structure_path = os.path.join(scene_path, "structure.json")
    if not os.path.exists(structure_path):
        return []
    
    room_polys = []
    try:
        with open(structure_path, 'r') as f:
            structure = json.load(f)
        for room in structure.get('rooms', []):
            vertices = room.get('vertices', [])
            if len(vertices) >= 3:
                poly = Polygon([(v[0], v[1]) for v in vertices])
                if poly.is_valid:
                    room_polys.append(poly)
    except Exception as e:
        print(f"[Warning] Failed to load structure.json: {e}")
    
    return room_polys


def load_scene_aabbs(scene_path: str) -> Tuple[List[AABB], List[AABB], List[List[List[float]]]]:
    """
    Load object AABBs and wall AABBs from scene files.
    
    Returns:
        (object_aabbs, wall_aabbs, room_polys_coords)
    """
    object_aabbs = []
    wall_aabbs = []
    room_polys_coords = []
    
    # Load objects from structure.json
    structure_path = os.path.join(scene_path, "structure.json")
    if os.path.exists(structure_path):
        try:
            with open(structure_path, 'r') as f:
                structure = json.load(f)
            
            # Load room polygon coordinates (for CameraSampler)
            for room in structure.get('rooms', []):
                vertices = room.get('vertices', [])
                if len(vertices) >= 3:
                    room_polys_coords.append([[v[0], v[1]] for v in vertices])
            
            # Load wall AABBs
            for wall_idx, wall in enumerate(structure.get('walls', [])):
                if 'bbox_min' in wall and 'bbox_max' in wall:
                    # Direct AABB format
                    wall_aabbs.append(AABB(
                        id=f'wall_{wall_idx}',
                        bmin=np.array(wall['bbox_min']),
                        bmax=np.array(wall['bbox_max']),
                        label='wall'
                    ))
                elif 'location' in wall:
                    # Convert location + thickness + height format to AABB
                    # location is [[x1, y1], [x2, y2]] defining a wall segment
                    loc = wall['location']
                    if len(loc) >= 2:
                        p1 = np.array(loc[0])
                        p2 = np.array(loc[1])
                        thickness = wall.get('thickness', 0.1)
                        height = wall.get('height', 2.8)
                        
                        # Calculate wall direction and perpendicular
                        wall_dir = p2 - p1
                        wall_len = np.linalg.norm(wall_dir)
                        if wall_len > 1e-6:
                            wall_dir = wall_dir / wall_len
                            # Perpendicular direction (for thickness)
                            perp = np.array([-wall_dir[1], wall_dir[0]])
                            half_thick = thickness / 2.0
                            
                            # Compute 4 corners of the wall footprint
                            c1 = p1 + perp * half_thick
                            c2 = p1 - perp * half_thick
                            c3 = p2 + perp * half_thick
                            c4 = p2 - perp * half_thick
                            
                            # Get AABB from corners
                            xs = [c1[0], c2[0], c3[0], c4[0]]
                            ys = [c1[1], c2[1], c3[1], c4[1]]
                            
                            bmin = np.array([min(xs), min(ys), 0.0])
                            bmax = np.array([max(xs), max(ys), height])
                            
                            wall_aabbs.append(AABB(
                                id=f'wall_{wall_idx}',
                                bmin=bmin,
                                bmax=bmax,
                                label='wall'
                            ))
        except Exception as e:
            print(f"[Warning] Failed to load structure.json for AABBs: {e}")
    
    # Load objects from labels.json (primary source for this dataset)
    labels_path = os.path.join(scene_path, "labels.json")
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            
            for obj in labels:
                bbox = obj.get('bounding_box', [])
                if len(bbox) >= 8:
                    # Compute AABB from bounding box corners
                    xs = [p.get('x', 0) for p in bbox]
                    ys = [p.get('y', 0) for p in bbox]
                    zs = [p.get('z', 0) for p in bbox]
                    bmin = np.array([min(xs), min(ys), min(zs)])
                    bmax = np.array([max(xs), max(ys), max(zs)])
                    # Use ins_id as the object ID (labels.json uses ins_id, not id)
                    obj_id = str(obj.get('ins_id', obj.get('id', '')))
                    object_aabbs.append(AABB(
                        bmin=bmin,
                        bmax=bmax,
                        label=obj.get('label', 'object'),
                        id=obj_id
                    ))
        except Exception as e:
            print(f"[Warning] Failed to load labels.json for AABBs: {e}")
    
    # Also try furniture.json or objects.json as fallback
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
    
    return object_aabbs, wall_aabbs, room_polys_coords


def validate_position_simple(pos: np.ndarray, scene_bounds: Optional[SceneBounds], 
                              room_polys: List, margin: float = 0.2) -> bool:
    """
    Simple validation if a position is within scene bounds and room polygons.
    (Used as fallback when CameraSampler is not available)
    
    Args:
        pos: [x, y, z] position
        scene_bounds: Scene boundary constraints
        room_polys: List of room polygons (shapely Polygon objects)
        margin: Safety margin for bounds check
    
    Returns:
        True if position is valid
    """
    x, y, z = pos[0], pos[1], pos[2]
    
    # Check scene bounds
    if scene_bounds is not None:
        # Use contains_point_2d for XY check
        if not scene_bounds.contains_point_2d(x, y, margin):
            return False
        # Check Z using min_point and max_point arrays
        z_min = scene_bounds.min_point[2] if len(scene_bounds.min_point) > 2 else 0
        z_max = scene_bounds.max_point[2] if len(scene_bounds.max_point) > 2 else 3
        if z < z_min or z > z_max + 1.0:  # Allow some margin above
            return False
    
    # Check room polygons
    if room_polys and HAS_SHAPELY:
        pt = Point(x, y)
        in_any_room = any(room.contains(pt) or room.buffer(margin).contains(pt) 
                         for room in room_polys)
        if not in_any_room:
            return False
    
    return True


def validate_camera_with_sampler(
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    camera_sampler: CameraSampler,
    room_polys_coords: List[List[List[float]]],
    scene_bounds: Optional[SceneBounds],
    object_aabbs: List[AABB],
    wall_aabbs: List[AABB],
    target_objects: List[Tuple[np.ndarray, np.ndarray, str]] = None,
    intrinsics: np.ndarray = None,
    image_width: int = 640,
    image_height: int = 480,
) -> Tuple[bool, str]:
    """
    Validate camera position using full CameraSampler validation.
    
    This performs all validation checks:
    1. Scene bounds check
    2. Height check  
    3. Room polygon check
    4. Object collision check
    5. Wall collision check
    6. Wall distance check
    7. FOV check (if target_objects provided)
    8. Occlusion check (if target_objects provided)
    
    Args:
        cam_pos: Camera position [x, y, z]
        cam_target: Camera look-at target
        camera_sampler: CameraSampler instance
        room_polys_coords: Room polygon coordinates for CameraSampler
        scene_bounds: Scene bounds
        object_aabbs: Object AABBs for collision/occlusion check
        wall_aabbs: Wall AABBs for collision check
        target_objects: List of (bmin, bmax, label) for FOV/occlusion checks
        intrinsics: Camera intrinsics for FOV check (unused, camera_sampler has its own)
        image_width: Image width for FOV check (unused, camera_sampler has its own)
        image_height: Image height for FOV check (unused, camera_sampler has its own)
        
    Returns:
        (is_valid, rejection_reason)
    """
    # If target_objects provided, use validate_camera_complete for full validation
    if target_objects:
        return camera_sampler.validate_camera_complete(
            cam_pos=cam_pos,
            cam_target=cam_target,
            room_polys=room_polys_coords,
            scene_bounds=scene_bounds,
            object_aabbs=object_aabbs,
            wall_aabbs=wall_aabbs,
            target_objects=target_objects,
            target_room_idx=None,
            require_in_target_room=False,
            check_collision=True,
            check_wall_dist=True,
            check_fov=True,
            check_occlusion=True,
            min_wall_distance=0.25,
            occlusion_threshold=0.5,
            fov_border=5
        )
    else:
        # Otherwise just validate position without FOV/occlusion
        return camera_sampler.validate_camera_position_full(
            cam_pos=cam_pos,
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


def sample_from_target_region(
    target_region: Dict[str, Any],
    num_samples: int,
    camera_height: float,
    scene_bounds: Optional[SceneBounds],
    room_polys: List,
    camera_sampler: CameraSampler = None,
    room_polys_coords: List[List[List[float]]] = None,
    object_aabbs: List[AABB] = None,
    wall_aabbs: List[AABB] = None,
    target_look_at: np.ndarray = None,
    target_objects: List[Tuple[np.ndarray, np.ndarray, str]] = None,
    intrinsics: np.ndarray = None,
    image_width: int = 640,
    image_height: int = 480,
    max_attempts: int = 1000,
    min_distance_to_object: float = 0.5,  # Minimum distance to target object
) -> List[np.ndarray]:
    """
    Sample valid positions from target region with FULL CameraSampler validation.
    
    This uses CameraSampler.validate_camera_position_full() to ensure:
    - Position is within scene bounds
    - Position is inside a valid room
    - Position doesn't collide with objects or walls
    - Position is at safe distance from walls
    - (Optionally) Target object is in FOV and not occluded
    
    Args:
        target_region: Target region definition from training data
        num_samples: Number of positions to sample
        camera_height: Z coordinate for camera positions
        scene_bounds: Scene boundary constraints
        room_polys: Room polygons for validation
        max_attempts: Maximum sampling attempts
    
    Returns:
        List of valid [x, y, z] positions
    """
    region_type = target_region.get('type', 'circle')
    params = target_region.get('params', target_region)  # params might be nested or flat
    
    valid_positions = []
    attempts = 0
    
    while len(valid_positions) < num_samples and attempts < max_attempts:
        attempts += 1
        pos = None
        
        if region_type == 'circle':
            center = np.array(params.get('center', [0, 0]))
            radius = params.get('radius', 1.0)
            # Sample uniformly in circle
            r = radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            pos = np.array([center[0] + r * np.cos(theta), 
                           center[1] + r * np.sin(theta), 
                           camera_height])
        
        elif region_type == 'annulus':
            center = np.array(params.get('center', [0, 0]))
            inner_r = params.get('min_radius', params.get('inner_radius', 0.5))
            outer_r = params.get('max_radius', params.get('outer_radius', 2.0))
            r = np.sqrt(np.random.uniform(inner_r**2, outer_r**2))
            theta = np.random.random() * 2 * np.pi
            pos = np.array([center[0] + r * np.cos(theta),
                           center[1] + r * np.sin(theta),
                           camera_height])
        
        elif region_type == 'line':
            start = np.array(params.get('start', [0, 0]))
            end = np.array(params.get('end', [1, 0]))
            t = np.random.random()
            xy = start + t * (end - start)
            pos = np.array([xy[0], xy[1], camera_height])
        
        elif region_type == 'ray':
            origin = np.array(params.get('origin', [0, 0]))
            direction = np.array(params.get('direction', [1, 0]))
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            min_dist = params.get('min_distance', 0.5)
            max_dist = min(params.get('max_distance', 5.0), 2.0)  # Cap at 2m
            t = np.random.uniform(min_dist, max_dist)
            xy = origin + t * direction
            pos = np.array([xy[0], xy[1], camera_height])
        
        elif region_type == 'half_plane':
            boundary_pt = np.array(params.get('boundary_point', [0, 0]))
            normal = np.array(params.get('normal', [1, 0]))
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            # Sample in half-plane, limit distance to 2m
            dist = np.random.uniform(0.2, 2.0)
            lateral = np.random.uniform(-2.0, 2.0)
            perp = np.array([-normal[1], normal[0]])
            xy = boundary_pt + dist * normal + lateral * perp
            pos = np.array([xy[0], xy[1], camera_height])
        
        elif region_type == 'curve':
            # Use pre-computed points if available
            points = params.get('points', [])
            if points:
                idx = np.random.randint(len(points))
                pt = points[idx]
                pos = np.array([pt[0], pt[1], camera_height])
            else:
                # Fallback to circle approximation
                center = np.array(params.get('center', [0, 0]))
                radius = params.get('radius', 1.5)
                theta = np.random.random() * 2 * np.pi
                pos = np.array([center[0] + radius * np.cos(theta),
                               center[1] + radius * np.sin(theta),
                               camera_height])
        
        elif region_type == 'point':
            # Single point, just use it
            sample_pt = params.get('sample_point', target_region.get('sample_point'))
            if sample_pt:
                pos = np.array([sample_pt[0], sample_pt[1], camera_height])
        
        else:
            # Unknown type, try to use sample_point from target_region
            sample_pt = target_region.get('sample_point')
            if sample_pt:
                # Add small random offset
                offset = np.random.randn(2) * 0.3
                pos = np.array([sample_pt[0] + offset[0], 
                               sample_pt[1] + offset[1], 
                               camera_height])
        
        # Validate position using full CameraSampler validation if available
        if pos is not None:
            # Check minimum distance to target object(s)
            if target_look_at is not None:
                dist_to_target = np.linalg.norm(pos[:2] - target_look_at[:2])
                if dist_to_target < min_distance_to_object:
                    # Position is too close to target, skip it
                    continue
            
            if camera_sampler is not None and room_polys_coords is not None:
                # Use full CameraSampler validation
                cam_target = target_look_at if target_look_at is not None else pos + np.array([0, 0, -1])
                is_valid, reason = validate_camera_with_sampler(
                    cam_pos=pos,
                    cam_target=cam_target,
                    camera_sampler=camera_sampler,
                    room_polys_coords=room_polys_coords,
                    scene_bounds=scene_bounds,
                    object_aabbs=object_aabbs or [],
                    wall_aabbs=wall_aabbs or [],
                    target_objects=target_objects,
                    intrinsics=intrinsics,
                    image_width=image_width,
                    image_height=image_height,
                )
                if is_valid:
                    valid_positions.append(pos)
            else:
                # Fallback to simple validation
                if validate_position_simple(pos, scene_bounds, room_polys):
                    valid_positions.append(pos)
    
    if len(valid_positions) < num_samples:
        print(f"    [Warning] Only sampled {len(valid_positions)}/{num_samples} valid positions")
    
    return valid_positions


def extract_target_look_at(item: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extract the target point to look at based on object count.
    
    Follows the same logic as pipeline.py:
    - Single object: look at object center
    - Two objects: look at midpoint between them
    - Three+ objects (centering): look at the centering target (object_a center, 
      since the task is to center object_a between B and C)
    
    Args:
        item: Training data item dictionary
        
    Returns:
        3D point to look at, or None if cannot be determined
    """
    if 'target_region' in item:
        region = item['target_region']
        params = region.get('params', region)
        region_type = region.get('type', '')
        
        # Check for centering task (three objects)
        # Centering stores: object_a_center and midpoint_bc
        # For centering, we should look at object_a (the object to be centered)
        # since the forward direction in task also points toward object_a
        if 'midpoint_bc' in params and 'object_a_center' in params:
            # Look at object_a (the centering target)
            return np.array(params['object_a_center'])
        
        # Check for three-object case with explicit centers
        # Has object_a, object_b, object_c centers
        if 'object_a_center' in params and 'object_b_center' in params and 'object_c_center' in params:
            # Look at the SECOND object (object_b) center
            return np.array(params['object_b_center'])
        
        # Check for two-object case (equidistance, projective_relations, etc.)
        # Has object_a_center and object_b_center
        if 'object_a_center' in params and 'object_b_center' in params:
            # Look at the midpoint between two objects
            center_a = np.array(params['object_a_center'])
            center_b = np.array(params['object_b_center'])
            midpoint = (center_a + center_b) / 2
            return midpoint
        
        # Single object case - look at object center
        if 'object_center' in params:
            return np.array(params['object_center'])
        
        # Fallback: use center from params
        if 'center' in params:
            c = params['center']
            return np.array([c[0], c[1], 1.0]) if len(c) == 2 else np.array(c)
    
    # Try target_object field
    if 'target_object' in item:
        obj = item['target_object']
        if 'center' in obj:
            return np.array(obj['center'])
        if 'position' in obj:
            return np.array(obj['position'])
    
    # Fallback: use sample_target
    if 'sample_target' in item:
        return np.array(item['sample_target'])
    
    return None


async def render_item(
    item: Dict[str, Any],
    item_idx: int,
    renderer: SceneRenderer,
    output_dir: Path,
    num_target_samples: int,
    scene_bounds: Optional[SceneBounds],
    room_polys: List,
    camera_sampler: CameraSampler = None,
    room_polys_coords: List[List[List[float]]] = None,
    object_aabbs: List[AABB] = None,
    wall_aabbs: List[AABB] = None,
    image_width: int = 640,
    image_height: int = 480,
) -> Optional[Dict[str, Any]]:
    """
    Render initial view and target views for a single training item.
    
    Uses full CameraSampler validation to ensure all camera positions are valid.
    
    Returns:
        Dictionary with render info, or None on failure
    """
    scene_id = item.get('scene_id', '')
    
    # Extract initial camera matrix
    init_camera = item.get('init_camera', {})
    extrinsics = init_camera.get('extrinsics')
    intrinsics = init_camera.get('intrinsics')
    
    if extrinsics is None or intrinsics is None:
        print(f"    [Skip] Missing camera parameters")
        return None
    
    init_c2w = np.array(extrinsics)
    init_K = np.array(intrinsics)
    
    # Ensure correct shapes
    if init_K.shape == (4, 4):
        init_K = init_K[:3, :3]
    
    # Get camera height from initial position
    init_pos = init_c2w[:3, 3]
    camera_height = init_pos[2]
    
    # Get target to look at
    target_look_at = extract_target_look_at(item)
    if target_look_at is None:
        # Default: look forward from initial camera
        forward = init_c2w[:3, 2]
        target_look_at = init_pos + forward * 2.0
    
    # Extract target object info for FOV/occlusion validation
    # Note: The third element in tuple is object ID (used to exclude target from occlusion check)
    target_objects = None
    if 'target_object' in item:
        obj = item['target_object']
        # Handle single object case (direct bbox_min/bbox_max)
        if 'bbox_min' in obj and 'bbox_max' in obj:
            target_objects = [(
                np.array(obj['bbox_min']),
                np.array(obj['bbox_max']),
                str(obj.get('id', ''))  # Use ID for occlusion exclusion
            )]
        # Handle multi-object case (objects list with primary)
        elif 'objects' in obj:
            target_objects = []
            for sub_obj in obj['objects']:
                if sub_obj and 'bbox_min' in sub_obj and 'bbox_max' in sub_obj:
                    target_objects.append((
                        np.array(sub_obj['bbox_min']),
                        np.array(sub_obj['bbox_max']),
                        str(sub_obj.get('id', ''))  # Use ID for occlusion exclusion
                    ))
        # Fallback: check if primary object is available
        elif 'primary' in obj and obj['primary']:
            primary = obj['primary']
            if 'bbox_min' in primary and 'bbox_max' in primary:
                target_objects = [(
                    np.array(primary['bbox_min']),
                    np.array(primary['bbox_max']),
                    str(primary.get('id', ''))  # Use ID for occlusion exclusion
                )]
    
    # Extract min_distance from target_region if available
    # This is computed by task_generator as max(object_max_dimension, 0.5)
    target_region = item.get('target_region', {})
    params = target_region.get('params', {})
    min_dist_to_obj = params.get('min_distance', 0.5)  # Default to 0.5m if not specified
    
    # Sample target positions with full CameraSampler validation
    target_positions = sample_from_target_region(
        target_region=target_region,
        num_samples=num_target_samples,
        camera_height=camera_height,
        scene_bounds=scene_bounds,
        room_polys=room_polys,
        camera_sampler=camera_sampler,
        room_polys_coords=room_polys_coords,
        object_aabbs=object_aabbs,
        wall_aabbs=wall_aabbs,
        target_look_at=target_look_at,
        target_objects=target_objects,
        intrinsics=init_K,
        image_width=image_width,
        image_height=image_height,
        min_distance_to_object=min_dist_to_obj,
    )
    
    if not target_positions:
        # Fallback: use sample_target with full validation
        sample_target = item.get('sample_target')
        if sample_target:
            pos = np.array([sample_target[0], sample_target[1], camera_height])
            
            # Check minimum distance to target (using the extracted min_dist_to_obj)
            if target_look_at is not None:
                dist_to_target = np.linalg.norm(pos[:2] - target_look_at[:2])
                if dist_to_target < min_dist_to_obj:
                    print(f"    [Skip] Fallback position too close to target ({dist_to_target:.2f}m < {min_dist_to_obj}m)")
                    pos = None
            
            # Use full validation if camera_sampler available
            if pos is not None and camera_sampler is not None and room_polys_coords is not None:
                is_valid, _ = validate_camera_with_sampler(
                    cam_pos=pos,
                    cam_target=target_look_at,
                    camera_sampler=camera_sampler,
                    room_polys_coords=room_polys_coords,
                    scene_bounds=scene_bounds,
                    object_aabbs=object_aabbs or [],
                    wall_aabbs=wall_aabbs or [],
                    target_objects=target_objects,
                    intrinsics=init_K,
                    image_width=image_width,
                    image_height=image_height,
                )
                if is_valid:
                    target_positions = [pos]
            elif pos is not None and validate_position_simple(pos, scene_bounds, room_polys):
                target_positions = [pos]
    
    if not target_positions:
        print(f"    [Skip] Could not sample any valid target positions")
        return None
    
    # Create output directory for this item
    item_dir = output_dir / scene_id / f"item_{item_idx:06d}"
    item_dir.mkdir(parents=True, exist_ok=True)
    
    rendered_views = []
    
    # Render initial view
    try:
        init_img = await renderer.render_image(init_K, init_c2w)
        if init_img is not None:
            init_path = item_dir / "init_view.png"
            init_img.save(init_path)
            rendered_views.append({
                'type': 'init',
                'filename': 'init_view.png',
                'position': init_pos.tolist(),
            })
    except Exception as e:
        print(f"    [Error] Failed to render init view: {e}")
        return None
    
    # Render target views
    for i, target_pos in enumerate(target_positions):
        try:
            # Camera looks at target object
            target_c2w = look_at_matrix(target_pos, target_look_at)
            
            target_img = await renderer.render_image(init_K, target_c2w)
            if target_img is not None:
                target_filename = f"target_view_{i:02d}.png"
                target_img.save(item_dir / target_filename)
                rendered_views.append({
                    'type': 'target',
                    'filename': target_filename,
                    'position': target_pos.tolist(),
                    'look_at': target_look_at.tolist(),
                })
        except Exception as e:
            print(f"    [Error] Failed to render target view {i}: {e}")
    
    # Save view info
    view_info = {
        'scene_id': scene_id,
        'task_type': item.get('task_type', ''),
        'task_description': item.get('task_description', ''),
        'object_label': item.get('object_label', ''),
        'preset': item.get('preset', ''),
        'target_region_type': target_region.get('type', ''),
        'num_target_views': len([v for v in rendered_views if v['type'] == 'target']),
        'views': rendered_views,
    }
    
    with open(item_dir / 'view_info.json', 'w') as f:
        json.dump(view_info, f, indent=2)
    
    return view_info


async def main():
    parser = argparse.ArgumentParser(
        description='Render initial and target views from training data'
    )
    parser.add_argument('--jsonl_path', type=str, required=True,
                        help='Path to train_data.jsonl')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for rendered views')
    parser.add_argument('--gs_root', type=str, required=True,
                        help='Root path to 3DGS scenes (e.g., InteriorGS)')
    parser.add_argument('--scene_id', type=str, default=None,
                        help='Filter to specific scene ID')
    parser.add_argument('--num_target_samples', type=int, default=10,
                        help='Number of target views to sample per item')
    parser.add_argument('--max_items', type=int, default=None,
                        help='Maximum number of items to process')
    parser.add_argument('--render_backend', type=str, default='local',
                        choices=['local', 'client'],
                        help='Rendering backend')
    parser.add_argument('--image_width', type=int, default=640,
                        help='Output image width')
    parser.add_argument('--image_height', type=int, default=480,
                        help='Output image height')
    parser.add_argument('--diverse_sampling', action='store_true',
                        help='Use diverse sampling to cover all task types, distances, and presets')
    parser.add_argument('--samples_per_category', type=int, default=2,
                        help='Number of samples per (task_type, distance, preset) category when using diverse sampling')
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.jsonl_path}...")
    items = []
    with open(args.jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(items)} items")
    
    # Filter by scene_id if specified
    if args.scene_id:
        items = [item for item in items if item.get('scene_id') == args.scene_id]
        print(f"Filtered to {len(items)} items for scene {args.scene_id}")
    
    # Apply diverse sampling if requested
    if args.diverse_sampling:
        items = diverse_sample_items(items, args.samples_per_category)
        print(f"Diverse sampling: selected {len(items)} items covering all categories")
    
    # Limit items if specified (applied after diverse sampling)
    if args.max_items and len(items) > args.max_items:
        # If using diverse sampling, randomly select from the diverse set
        if args.diverse_sampling:
            np.random.shuffle(items)
        items = items[:args.max_items]
        print(f"Limited to {len(items)} items")
    
    if not items:
        print("No items to process!")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create render config
    config = RenderConfig(
        gs_root=args.gs_root,
        render_backend=args.render_backend,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    
    # Cache scene data
    scene_cache = {}
    
    def get_scene_data(scene_id: str):
        if scene_id not in scene_cache:
            scene_path = os.path.join(args.gs_root, scene_id)
            scene_bounds = load_scene_bounds(scene_path)
            room_polys = load_room_polygons(scene_path)
            scene_cache[scene_id] = (scene_bounds, room_polys)
            if scene_bounds:
                print(f"  Loaded bounds for {scene_id}: "
                      f"X[{scene_bounds.min_point[0]:.1f}, {scene_bounds.max_point[0]:.1f}], "
                      f"Y[{scene_bounds.min_point[1]:.1f}, {scene_bounds.max_point[1]:.1f}]")
        return scene_cache[scene_id]
    
    # Extended cache for CameraSampler data
    sampler_cache = {}
    
    def get_sampler_data(scene_id: str):
        """Get CameraSampler instance and AABB data for a scene."""
        if scene_id not in sampler_cache:
            scene_path = os.path.join(args.gs_root, scene_id)
            object_aabbs, wall_aabbs, room_polys_coords = load_scene_aabbs(scene_path)
            
            # Create CameraSamplingConfig with image dimensions from args
            camera_config = CameraSamplingConfig(
                image_width=args.image_width,
                image_height=args.image_height,
                focal_length=300.0  # Default focal length for ~60 deg FoV
            )
            camera_sampler = CameraSampler(config=camera_config)
            
            sampler_cache[scene_id] = (camera_sampler, object_aabbs, wall_aabbs, room_polys_coords)
            print(f"  Loaded AABBs for {scene_id}: {len(object_aabbs)} objects, {len(wall_aabbs)} walls")
        return sampler_cache[scene_id]
    
    # Process items
    results = []
    async with SceneRenderer(config) as renderer:
        for idx, item in enumerate(tqdm(items, desc="Rendering items")):
            scene_id = item.get('scene_id', '')
            if not scene_id:
                continue
            
            # Set scene for renderer
            await renderer.set_scene(scene_id)
            
            # Get scene bounds and room polygons
            scene_bounds, room_polys = get_scene_data(scene_id)
            
            # Get CameraSampler data for full validation
            camera_sampler, object_aabbs, wall_aabbs, room_polys_coords = get_sampler_data(scene_id)
            
            # Render item with full CameraSampler validation
            result = await render_item(
                item=item,
                item_idx=idx,
                renderer=renderer,
                output_dir=output_dir,
                num_target_samples=args.num_target_samples,
                scene_bounds=scene_bounds,
                room_polys=room_polys,
                camera_sampler=camera_sampler,
                room_polys_coords=room_polys_coords,
                object_aabbs=object_aabbs,
                wall_aabbs=wall_aabbs,
                image_width=args.image_width,
                image_height=args.image_height,
            )
            
            if result:
                results.append(result)
    
    # Save summary
    summary = {
        'total_items': len(items),
        'rendered_items': len(results),
        'avg_target_views': np.mean([r['num_target_views'] for r in results]) if results else 0,
        'scenes': list(set(r['scene_id'] for r in results)),
    }
    
    with open(output_dir / 'render_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDone! Rendered {len(results)} items to {output_dir}")
    print(f"  Average target views per item: {summary['avg_target_views']:.1f}")


if __name__ == '__main__':
    asyncio.run(main())
