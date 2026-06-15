"""
Active Spatial Perception Dataset Generation Pipeline
=====================================================

This pipeline generates training data for active spatial navigation tasks.
It integrates three main components:

1. Object Selection (choose_object): Filter and select suitable objects/pairs
2. Camera Pose Sampling (find_cam_pos): Find valid camera positions around objects
3. Task-based Target Generation (task_design): Compute optimal target positions

Pipeline Flow:
    labels.json → select_objects → sample_cameras → generate_tasks → dataset.json
"""

from .config import PipelineConfig
from .object_selector import ObjectSelector
from .camera_sampler import CameraSampler
from .task_generator import TaskGenerator
from .pipeline import ActiveSpatialPipeline

# Camera utility functions (decoupled from CameraSampler for reusability)
from .camera_utils import (
    SceneBounds,
    AABB,
    CameraPose,
    intersects_ray_aabb,
    is_target_in_fov,
    is_target_occluded,
    count_visible_corners,
    calculate_projected_area_ratio,
    calculate_occlusion_area_2d,
)

__all__ = [
    # Pipeline components
    'PipelineConfig',
    'ObjectSelector', 
    'CameraSampler',
    'TaskGenerator',
    'ActiveSpatialPipeline',
    # Camera utility classes and functions
    'SceneBounds',
    'AABB',
    'CameraPose',
    'intersects_ray_aabb',
    'is_target_in_fov',
    'is_target_occluded',
    'count_visible_corners',
    'calculate_projected_area_ratio',
    'calculate_occlusion_area_2d',
]
