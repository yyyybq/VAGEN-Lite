"""
Object Selector Module

This module handles filtering and selecting suitable objects and object pairs
from a scene's labels.json file.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from itertools import combinations
from collections import Counter, defaultdict
from dataclasses import dataclass

try:
    from .config import ObjectSelectionConfig
except ImportError:
    from config import ObjectSelectionConfig


@dataclass
class SceneObject:
    """Represents a scene object with its properties."""
    id: str
    label: str
    bbox_points: List[Dict[str, float]]
    dims: np.ndarray  # (width, depth, height)
    center: np.ndarray  # (x, y, z)
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    room_index: Optional[int] = None
    
    @property
    def max_dim(self) -> float:
        return float(np.max(self.dims))
    
    @property
    def min_dim(self) -> float:
        return float(np.min(self.dims))
    
    @property
    def volume(self) -> float:
        return float(np.prod(self.dims))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'label': self.label,
            'center': self.center.tolist(),
            'dims': self.dims.tolist(),
            'aabb_min': self.aabb_min.tolist(),
            'aabb_max': self.aabb_max.tolist(),
            'room_index': self.room_index,
        }


class ObjectSelector:
    """Selects suitable objects and object pairs for spatial QA tasks."""
    
    def __init__(self, config: ObjectSelectionConfig):
        self.config = config
    
    def load_labels(self, labels_path: Path) -> List[Dict[str, Any]]:
        """Load objects from labels.json."""
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json not found: {labels_path}")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_room_polys(self, scene_path: Path) -> List[List[List[float]]]:
        """Load room polygons from structure.json."""
        structure_path = scene_path / 'structure.json'
        if not structure_path.exists():
            return []
        
        try:
            with open(structure_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            room_polys = []
            # Parse structure.json format
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
            print(f"Warning: Failed to parse structure.json: {e}")
            return []
    
    def point_in_poly(self, x: float, y: float, poly: List[List[float]]) -> bool:
        """Check if point (x,y) is inside polygon using ray casting."""
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
    
    def get_room_index_for_point(self, x: float, y: float, room_polys: List[List[List[float]]]) -> Optional[int]:
        """Return index of room containing point, or None."""
        for i, poly in enumerate(room_polys):
            if self.point_in_poly(x, y, poly):
                return i
        return None
    
    def parse_object(self, item: Dict[str, Any], room_polys: List[List[List[float]]]) -> Optional[SceneObject]:
        """Parse a single object from labels data."""
        bbox = item.get('bounding_box', [])
        if not bbox or len(bbox) < 8:
            return None
        
        obj_id = str(item.get('ins_id') or item.get('id') or '')
        label = str(item.get('label', '')).strip().lower()
        
        if not obj_id:
            return None
        
        # Extract coordinates
        xs = [float(p['x']) for p in bbox]
        ys = [float(p['y']) for p in bbox]
        zs = [float(p['z']) for p in bbox]
        
        aabb_min = np.array([min(xs), min(ys), min(zs)], dtype=float)
        aabb_max = np.array([max(xs), max(ys), max(zs)], dtype=float)
        dims = aabb_max - aabb_min
        center = (aabb_min + aabb_max) / 2
        
        # Get room index
        room_idx = self.get_room_index_for_point(float(center[0]), float(center[1]), room_polys)
        
        return SceneObject(
            id=obj_id,
            label=label,
            bbox_points=bbox,
            dims=dims,
            center=center,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            room_index=room_idx
        )
    
    def filter_single_object(self, obj: SceneObject) -> Tuple[bool, str]:
        """
        Filter a single object based on semantic and geometric constraints.
        
        Returns:
            (is_valid, rejection_reason)
        """
        cfg = self.config
        
        # Semantic filter
        if obj.label in cfg.blacklist:
            return False, 'blacklist'
        
        # Dimension filters (optional)
        if cfg.enable_dim_filter:
            if any(d < cfg.min_dim_component for d in obj.dims):
                return False, 'dim_too_small'
            if any(d > cfg.max_dim_component for d in obj.dims):
                return False, 'dim_too_large'
        
        # Volume filter (optional)
        if cfg.enable_volume_filter:
            if obj.volume < cfg.min_volume:
                return False, 'volume_too_small'
        
        # Aspect ratio filter (optional - avoid flat objects)
        if cfg.enable_aspect_ratio_filter:
            if obj.min_dim / obj.max_dim < cfg.min_aspect_ratio:
                return False, 'too_flat'
        
        return True, 'ok'
    
    def aabb_min_distance(self, obj_a: SceneObject, obj_b: SceneObject) -> float:
        """Calculate minimum distance between two AABBs."""
        a_min, a_max = obj_a.aabb_min, obj_a.aabb_max
        b_min, b_max = obj_b.aabb_min, obj_b.aabb_max
        
        dx = max(0, max(a_min[0] - b_max[0], b_min[0] - a_max[0]))
        dy = max(0, max(a_min[1] - b_max[1], b_min[1] - a_max[1]))
        dz = max(0, max(a_min[2] - b_max[2], b_min[2] - a_max[2]))
        
        return float(math.sqrt(dx*dx + dy*dy + dz*dz))
    
    def filter_object_pair(self, obj_a: SceneObject, obj_b: SceneObject) -> Tuple[bool, str]:
        """
        Filter an object pair based on pair constraints.
        
        Returns:
            (is_valid, rejection_reason)
        """
        cfg = self.config
        
        # Same room check - configurable, disabled by default for more pairs
        if cfg.require_same_room:
            if obj_a.room_index is not None and obj_b.room_index is not None:
                if obj_a.room_index != obj_b.room_index:
                    return False, 'different_rooms'
        
        # Distance check using AABB
        dist = self.aabb_min_distance(obj_a, obj_b)
        
        # Dynamic thresholds based on object size
        avg_max_dim = (obj_a.max_dim + obj_b.max_dim) / 2
        dyn_min = max(cfg.min_pair_dist, cfg.dyn_min_mult * avg_max_dim)
        dyn_max = min(cfg.max_pair_dist, cfg.dyn_max_mult * avg_max_dim)
        
        if dist < dyn_min:
            return False, 'too_close'
        if dist > dyn_max:
            return False, 'too_far'
        
        # Size difference check
        dim_ratio = max(obj_a.max_dim, obj_b.max_dim) / (min(obj_a.max_dim, obj_b.max_dim) + 1e-9)
        if dim_ratio > cfg.max_pair_dim_ratio:
            return False, 'size_ratio'
        
        dim_diff = abs(obj_a.max_dim - obj_b.max_dim)
        if dim_diff > cfg.max_pair_dim_diff:
            return False, 'size_diff'
        
        # Height difference check: relaxed to allow 2x the taller object's height
        # This allows pairing objects with different heights (e.g., floor lamp with sofa)
        height_a = obj_a.dims[2]  # dims is (width, depth, height)
        height_b = obj_b.dims[2]
        max_height = max(height_a, height_b)
        height_diff = abs(height_a - height_b)
        if height_diff > 2 * max_height:  # Relaxed from 1x to 2x
            return False, 'height_diff'
        
        # ========== NEW: Angular span check for simultaneous visibility ==========
        # Ensure both objects can be seen together within camera FOV
        # This filters out pairs at opposite corners of a room that can never be
        # viewed simultaneously from any valid camera position
        center_a_2d = obj_a.center[:2]  # XY coordinates only
        center_b_2d = obj_b.center[:2]
        
        # Distance between object centers in XY plane
        dist_ab_2d = float(np.linalg.norm(center_b_2d - center_a_2d))
        
        # Minimum viewing distance based on object sizes
        # Camera needs to be far enough to see both objects
        max_obj_size = max(obj_a.max_dim, obj_b.max_dim)
        min_view_dist = max(max_obj_size * cfg.min_viewing_distance_mult, 0.8)
        
        # Calculate the angular span of the two objects from the minimum viewing distance
        # At distance d from midpoint, looking perpendicular to the line connecting objects:
        # The angle subtended by the two objects is 2 * arctan((dist_ab/2) / d)
        # We check this at the minimum viable viewing distance
        half_angle_rad = math.atan2(dist_ab_2d / 2, min_view_dist)
        full_angular_span_deg = math.degrees(2 * half_angle_rad)
        
        # Effective FOV for pair viewing (FOV minus margins on both sides)
        effective_fov = cfg.pair_fov_deg - 2 * cfg.pair_fov_margin_deg
        
        if full_angular_span_deg > effective_fov:
            return False, f'angular_span_too_large_{full_angular_span_deg:.1f}deg'
        # ========== END: Angular span check ==========
        
        return True, 'ok'
    
    def select_objects(self, scene_path: Path, num_objects: int = 1, 
                       max_results: int = 50) -> List[Any]:
        """
        Select suitable objects or object pairs from a scene.
        
        Args:
            scene_path: Path to scene folder containing labels.json
            num_objects: 1 for single objects, 2-5 for pairs/groups
            max_results: Maximum number of results to return
        
        Returns:
            List of objects (dicts) or object pairs (list of dicts)
        """
        labels_path = scene_path / 'labels.json'
        data = self.load_labels(labels_path)
        room_polys = self.load_room_polys(scene_path)
        
        # Parse all objects
        all_objects = []
        for item in data:
            obj = self.parse_object(item, room_polys)
            if obj is not None:
                all_objects.append(obj)
        
        # Filter individual objects
        candidates = []
        for obj in all_objects:
            is_valid, reason = self.filter_single_object(obj)
            if is_valid:
                candidates.append(obj)
        
        # Apply label uniqueness (within room)
        labels_norm = [c.label for c in candidates]
        global_counts = Counter(labels_norm)
        room_counts = defaultdict(Counter)
        for c in candidates:
            room_counts[c.room_index][c.label] += 1
        
        filtered = []
        for c in candidates:
            gc = global_counts[c.label]
            rc = room_counts[c.room_index][c.label]
            if gc == 1 or rc == 1:
                filtered.append(c)
        
        if num_objects == 1:
            # Return single objects
            result = [{'id': obj.id, 'label': obj.label, 'center': obj.center.tolist()} 
                      for obj in filtered[:max_results]]
            return result
        
        elif 2 <= num_objects <= 5:
            # Generate and filter pairs/groups
            valid_groups = []
            
            for combo in combinations(filtered, num_objects):
                # Check all pairs in the group
                all_valid = True
                for i, obj_a in enumerate(combo):
                    for obj_b in combo[i+1:]:
                        is_valid, reason = self.filter_object_pair(obj_a, obj_b)
                        if not is_valid:
                            all_valid = False
                            break
                    if not all_valid:
                        break
                
                if all_valid:
                    group = [{'id': obj.id, 'label': obj.label, 'center': obj.center.tolist()} 
                             for obj in combo]
                    valid_groups.append(group)
                    
                    if len(valid_groups) >= max_results:
                        break
            
            return valid_groups
        
        else:
            raise ValueError(f"num_objects must be 1-5, got {num_objects}")
    
    def select_single_objects(self, scene_path: Path) -> List[SceneObject]:
        """
        Select all valid single objects from a scene.
        
        Args:
            scene_path: Path to scene folder containing labels.json
        
        Returns:
            List of SceneObject instances
        """
        labels_path = scene_path / 'labels.json'
        data = self.load_labels(labels_path)
        room_polys = self.load_room_polys(scene_path)
        
        # Parse all objects
        all_objects = []
        for item in data:
            obj = self.parse_object(item, room_polys)
            if obj is not None:
                all_objects.append(obj)
        
        # Filter individual objects
        candidates = []
        for obj in all_objects:
            is_valid, reason = self.filter_single_object(obj)
            if is_valid:
                candidates.append(obj)
        
        # Apply label uniqueness (within room)
        labels_norm = [c.label for c in candidates]
        global_counts = Counter(labels_norm)
        room_counts = defaultdict(Counter)
        for c in candidates:
            room_counts[c.room_index][c.label] += 1
        
        filtered = []
        for c in candidates:
            gc = global_counts[c.label]
            rc = room_counts[c.room_index][c.label]
            if gc == 1 or rc == 1:
                filtered.append(c)
        
        return filtered
    
    def select_object_pairs(self, scene_path: Path, 
                            objects: Optional[List[SceneObject]] = None,
                            max_pairs: int = 100) -> List[Tuple[SceneObject, SceneObject]]:
        """
        Select all valid object pairs from a scene.
        
        Args:
            scene_path: Path to scene folder
            objects: Pre-filtered list of objects (if None, will call select_single_objects)
            max_pairs: Maximum number of pairs to return
        
        Returns:
            List of (obj_a, obj_b) tuples
        """
        if objects is None:
            objects = self.select_single_objects(scene_path)
        
        valid_pairs = []
        for obj_a, obj_b in combinations(objects, 2):
            is_valid, reason = self.filter_object_pair(obj_a, obj_b)
            if is_valid:
                valid_pairs.append((obj_a, obj_b))
                if len(valid_pairs) >= max_pairs:
                    break
        
        return valid_pairs
    
    def select_object_triples(self, scene_path: Path,
                              objects: Optional[List[SceneObject]] = None,
                              max_triples: int = 50) -> List[Tuple[SceneObject, SceneObject, SceneObject]]:
        """
        Select all valid object triples from a scene.
        
        For the centering task: object A should be between B and C.
        
        Args:
            scene_path: Path to scene folder
            objects: Pre-filtered list of objects (if None, will call select_single_objects)
            max_triples: Maximum number of triples to return
        
        Returns:
            List of (obj_a, obj_b, obj_c) tuples where A is the center object
        """
        if objects is None:
            objects = self.select_single_objects(scene_path)
        
        valid_triples = []
        for combo in combinations(objects, 3):
            # Check all pairs in the triple
            all_valid = True
            for i, obj_a in enumerate(combo):
                for obj_b in combo[i+1:]:
                    is_valid, reason = self.filter_object_pair(obj_a, obj_b)
                    if not is_valid:
                        all_valid = False
                        break
                if not all_valid:
                    break
            
            if all_valid:
                # For centering, any object can be the "center" object
                # Return as (center_obj, left_obj, right_obj)
                valid_triples.append(combo)
                if len(valid_triples) >= max_triples:
                    break
        
        return valid_triples
