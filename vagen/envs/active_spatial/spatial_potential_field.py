"""
Spatial Potential Field Module

This module implements task-specific scoring functions that create a continuous
"potential field" for spatial navigation tasks. Given any camera pose (position + 
orientation), the scoring function returns a value in [0, 1] indicating how well
the current state satisfies the task objective.

Key Design Principles:
======================
1. **Dense Rewards**: Every position has a score, enabling gradient-based learning
2. **Task-Specific Geometry**: Each task type has its own geometric scoring logic
3. **Dual Components**: Score = Position Score × α + Orientation Score × β
4. **Smooth Gradients**: Use smooth functions (Gaussian, sigmoid) to avoid discontinuities

Task Types and Their Scoring Logic:
===================================
1. Absolute Positioning (CIRCLE): Score based on distance deviation from target radius
2. Delta Control (POINT): Score based on distance to exact target point
3. Equidistance (LINE): Score based on difference |dist_A - dist_B|
4. Projective Relations (HALF_PLANE): Score based on signed distance to boundary
5. Centering (RAY): Score based on angular centering of A between B and C
6. Occlusion Alignment (RAY): Score based on alignment with occlusion ray
7. FoV Inclusion (ANNULUS): Score based on both objects being in field of view
8. Size-Distance Invariance (CURVE): Score based on equal apparent size ratio
9. Apparent Size Ordering (SAMPLED REGION): Score based on one object appearing larger
10. Screen Occupancy (CIRCLE): Score based on object's screen coverage
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

try:
    from .region_metrics import compute_region_metrics
    from .visual_bbox_metrics import compute_visual_bbox_metrics
except ImportError:
    from region_metrics import compute_region_metrics
    from visual_bbox_metrics import compute_visual_bbox_metrics


class TaskType(Enum):
    """Enumeration of supported task types."""
    ABSOLUTE_POSITIONING = "absolute_positioning"
    DELTA_CONTROL = "delta_control"
    EQUIDISTANCE = "equidistance"
    PROJECTIVE_RELATIONS = "projective_relations"
    CENTERING = "centering"
    OCCLUSION_ALIGNMENT = "occlusion_alignment"
    FOV_INCLUSION = "fov_inclusion"
    SIZE_DISTANCE_INVARIANCE = "size_distance_invariance"
    APPARENT_SIZE_ORDERING = "apparent_size_ordering"
    SCREEN_OCCUPANCY = "screen_occupancy"


@dataclass
class ScoreResult:
    """
    Result of a scoring computation.
    
    Attributes:
        total_score: Combined score in [0, 1]
        position_score: Score for position component in [0, 1]
        orientation_score: Score for orientation component in [0, 1]
        details: Task-specific scoring details for debugging/logging
    """
    total_score: float
    position_score: float
    orientation_score: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": float(self.total_score),
            "position_score": float(self.position_score),
            "orientation_score": float(self.orientation_score),
            "details": self.details,
        }


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute 2D Euclidean distance (ignoring z)."""
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def distance_3d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute 3D Euclidean distance."""
    return float(np.linalg.norm(p1 - p2))


def gaussian_score(value: float, target: float, sigma: float) -> float:
    """
    Compute Gaussian-shaped score centered at target.
    
    Returns 1.0 when value == target, decays smoothly as value deviates.
    sigma controls the width of the Gaussian.
    """
    return float(np.exp(-0.5 * ((value - target) / sigma) ** 2))


def sigmoid_score(value: float, threshold: float, steepness: float = 5.0) -> float:
    """
    Compute sigmoid-shaped score.
    
    Returns ~1.0 when value >> threshold, ~0.0 when value << threshold.
    """
    return float(1.0 / (1.0 + np.exp(-steepness * (value - threshold))))


def linear_decay_score(value: float, max_value: float, min_score: float = 0.0) -> float:
    """
    Compute linearly decaying score.
    
    Returns 1.0 when value == 0, decays to min_score when value >= max_value.
    """
    if max_value <= 0:
        return 1.0 if value <= 0 else min_score
    score = 1.0 - value / max_value
    return float(max(min_score, min(1.0, score)))


def exponential_decay_score(value: float, decay_rate: float) -> float:
    """
    Compute exponentially decaying score.
    
    Returns 1.0 when value == 0, decays exponentially.
    """
    return float(np.exp(-decay_rate * value))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    v1_norm = normalize(v1)
    v2_norm = normalize(v2)
    return float(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))


def point_to_line_distance_2d(point: np.ndarray, 
                               line_point: np.ndarray, 
                               line_direction: np.ndarray) -> float:
    """
    Compute distance from a point to an infinite line in 2D.
    
    Args:
        point: The query point [x, y] or [x, y, z] (z ignored)
        line_point: A point on the line
        line_direction: Direction vector of the line (will be normalized)
    
    Returns:
        Perpendicular distance from point to line
    """
    p = point[:2]
    lp = line_point[:2]
    ld = normalize(line_direction[:2])
    
    # Vector from line point to query point
    v = p - lp
    
    # Project v onto line direction
    proj_length = np.dot(v, ld)
    proj = proj_length * ld
    
    # Perpendicular distance
    perp = v - proj
    return float(np.linalg.norm(perp))


def signed_distance_to_half_plane(point: np.ndarray,
                                   boundary_point: np.ndarray,
                                   normal: np.ndarray) -> float:
    """
    Compute signed distance from point to half-plane boundary.
    
    Positive = inside valid region (same side as normal)
    Negative = outside valid region
    """
    p = point[:2]
    bp = boundary_point[:2]
    n = normalize(normal[:2])
    
    return float(np.dot(p - bp, n))


class SpatialPotentialField:
    """
    Computes task-specific scores for camera poses.
    
    This class implements a "potential field" where each task defines a 
    continuous scoring function over the space of camera poses.
    
    The score combines two components:
    - Position Score: How well the camera position satisfies the task
    - Orientation Score: How well the camera orientation satisfies the task
    
    Both components are in [0, 1], and the final score is a weighted combination.
    
    IMPORTANT: FoV Constraint
    ========================
    All target objects MUST be within the camera's field of view at all times.
    If any target object is outside the FoV, a severe penalty is applied.
    This ensures the agent always keeps targets visible (occlusion is allowed,
    but being outside the viewing frustum is not).
    """
    
    def __init__(self, 
                 position_weight: float = 0.7,
                 orientation_weight: float = 0.3,
                 position_sigma: float = 2.0,
                 orientation_sigma: float = 0.5,
                 max_distance: float = 10.0,
                 fov_horizontal: float = 90.0,
                 fov_vertical: float = 90.0,
                 fov_penalty_weight: float = 0.5,
                 fov_penalty_softness: float = 0.2,
                 use_dynamic_weights: bool = True,
                 dynamic_weight_threshold: float = 0.7,
                 use_visual_bbox_scoring: bool = True):
        """
        Initialize the potential field.
        
        Args:
            position_weight: Weight for position score in final score
            orientation_weight: Weight for orientation score in final score
            position_sigma: Sigma for Gaussian scoring of position
            orientation_sigma: Sigma for Gaussian scoring of orientation
            max_distance: Maximum distance for linear decay scoring
            fov_horizontal: Horizontal field of view in degrees (default 60)
            fov_vertical: Vertical field of view in degrees (default 60)
            fov_penalty_weight: Weight of FoV penalty (0-1, higher = stricter)
            fov_penalty_softness: How soft the FoV boundary is (radians, for smooth transition)
            use_dynamic_weights: If True, dynamically adjust position/orientation weights
                                based on how close to optimal position (default True)
            dynamic_weight_threshold: Position score threshold above which orientation
                                     weight starts increasing (default 0.7)
            use_visual_bbox_scoring: If True, projected bbox metrics override the
                                     reward score for visual-relation tasks.
        """
        self.base_position_weight = position_weight
        self.base_orientation_weight = orientation_weight
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.position_sigma = position_sigma
        self.orientation_sigma = orientation_sigma
        self.max_distance = max_distance
        
        # FoV parameters for visibility constraint
        self.fov_horizontal = np.radians(fov_horizontal)
        self.fov_vertical = np.radians(fov_vertical)
        self.fov_penalty_weight = fov_penalty_weight
        self.fov_penalty_softness = fov_penalty_softness
        
        # Dynamic weight parameters
        self.use_dynamic_weights = use_dynamic_weights
        self.dynamic_weight_threshold = dynamic_weight_threshold
        self.use_visual_bbox_scoring = use_visual_bbox_scoring
        
        # Normalize weights
        total_weight = position_weight + orientation_weight
        self.position_weight /= total_weight
        self.orientation_weight /= total_weight
        self.base_position_weight /= total_weight
        self.base_orientation_weight /= total_weight
    
    def _check_object_in_fov(self,
                              camera_position: np.ndarray,
                              camera_forward: np.ndarray,
                              object_center: np.ndarray,
                              camera_up: np.ndarray = None) -> Tuple[bool, float, float]:
        """
        Check if an object center is within the camera's field of view.
        
        Args:
            camera_position: Camera position [x, y, z]
            camera_forward: Camera forward direction [dx, dy, dz]
            object_center: Object center position [x, y, z]
            camera_up: Camera up vector (default: [0, 0, 1])
        
        Returns:
            (in_fov, horizontal_angle, vertical_angle)
            - in_fov: True if object is within FoV
            - horizontal_angle: Angle from camera center in radians (0 = centered)
            - vertical_angle: Angle from camera center in radians (0 = centered)
        """
        if camera_up is None:
            camera_up = np.array([0, 0, 1])
        
        # Vector from camera to object
        to_object = object_center - camera_position
        distance = np.linalg.norm(to_object)
        
        if distance < 1e-6:
            return True, 0.0, 0.0  # Object at camera position
        
        to_object = to_object / distance
        
        # Calculate camera coordinate system
        forward = normalize(camera_forward)
        right = np.cross(forward, camera_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # Forward is parallel to up, use alternative up
            camera_up = np.array([0, 1, 0])
            right = np.cross(forward, camera_up)
        right = normalize(right)
        up = np.cross(right, forward)
        
        # Project to_object onto camera axes
        forward_dot = np.dot(to_object, forward)
        right_dot = np.dot(to_object, right)
        up_dot = np.dot(to_object, up)
        
        # Check if object is behind camera
        if forward_dot <= 0:
            return False, np.pi, np.pi
        
        # Calculate angles from center of view
        horizontal_angle = np.arctan2(right_dot, forward_dot)
        vertical_angle = np.arctan2(up_dot, forward_dot)
        
        # Check against FoV
        in_fov = (
            abs(horizontal_angle) <= self.fov_horizontal / 2 and
            abs(vertical_angle) <= self.fov_vertical / 2
        )
        
        return in_fov, horizontal_angle, vertical_angle
    
    def _compute_fov_score(self,
                            camera_position: np.ndarray,
                            camera_forward: np.ndarray,
                            target_objects: List[np.ndarray]) -> Tuple[float, Dict[str, Any]]:
        """
        Compute FoV score for keeping target objects in view.
        
        IMPROVED: Steeper penalty at FoV edge to provide stronger gradient signal.
        
        Scoring logic:
        - Object fully in FoV (within half-angle): score = 1.0
        - Object at FoV edge (0-15° beyond): steep decay using sigmoid (1.0 -> 0.3)
        - Object completely invisible (>15° beyond FoV edge): severe penalty (score ~ 0)
        
        Args:
            camera_position: Camera position [x, y, z]
            camera_forward: Camera forward direction [dx, dy, dz]
            target_objects: List of object center positions to check
        
        Returns:
            (fov_score, details)
            - fov_score: 1.0 if all objects visible, decays steeply outside FoV
            - details: Dict with per-object visibility info
        """
        if not target_objects:
            return 1.0, {"all_in_fov": True, "objects_checked": 0}
        
        object_scores = []
        object_details = []
        all_fully_in_fov = True
        any_completely_invisible = False
        
        # IMPROVED: Reduced threshold from 30° to 15° for steeper gradient
        completely_invisible_threshold = np.radians(15.0)
        
        for i, obj_center in enumerate(target_objects):
            in_fov, h_angle, v_angle = self._check_object_in_fov(
                camera_position, camera_forward, np.array(obj_center)
            )
            
            # Calculate how far outside FoV the object is
            h_excess = max(0, abs(h_angle) - self.fov_horizontal / 2)
            v_excess = max(0, abs(v_angle) - self.fov_vertical / 2)
            max_excess = max(h_excess, v_excess)
            
            if in_fov:
                # Object fully in FoV - perfect score
                # IMPROVED: Add small bonus for being well-centered (not at edge)
                # This helps multi-object tasks by encouraging centering
                center_bonus = 1.0 - 0.1 * (max(abs(h_angle), abs(v_angle)) / (self.fov_horizontal / 2))
                obj_score = min(1.0, center_bonus)
                visibility_status = "fully_visible"
            elif max_excess < completely_invisible_threshold:
                # Object at FoV edge (0-15° beyond)
                # IMPROVED: Use sigmoid for steeper, smoother decay
                # At edge (0° excess): score ~ 0.85
                # At 7.5° beyond: score ~ 0.5
                # At 15° beyond: score ~ 0.15
                all_fully_in_fov = False
                # Sigmoid decay: steepness=8 gives good curve
                excess_ratio = max_excess / completely_invisible_threshold
                obj_score = 1.0 / (1.0 + np.exp(8.0 * (excess_ratio - 0.5)))
                obj_score = max(0.1, obj_score)  # Floor at 0.1
                visibility_status = "partially_visible"
            else:
                # Object completely invisible (>15° beyond FoV edge)
                # SEVERE penalty with steeper exponential decay
                all_fully_in_fov = False
                any_completely_invisible = True
                # IMPROVED: Steeper exponential decay (was 0.5, now 0.1 as base)
                extra_excess = max_excess - completely_invisible_threshold
                obj_score = 0.1 * np.exp(-extra_excess / (self.fov_penalty_softness * 0.5))
                obj_score = max(0.0, min(0.1, obj_score))  # Cap at 0.1, floor at 0
                visibility_status = "completely_invisible"
            
            object_scores.append(obj_score)
            object_details.append({
                "object_idx": i,
                "in_fov": in_fov,
                "visibility_status": visibility_status,
                "h_angle_deg": np.degrees(h_angle),
                "v_angle_deg": np.degrees(v_angle),
                "excess_angle_deg": np.degrees(max_excess),
                "score": obj_score,
            })
        
        # IMPROVED: Use soft-min instead of hard min for multi-object tasks
        # This provides smoother gradient when multiple objects are near FoV edge
        # soft_min = -log(mean(exp(-k*scores))) / k, approximates min as k->inf
        if len(object_scores) > 1:
            k = 5.0  # Softness parameter (higher = closer to hard min)
            exp_scores = np.exp(-k * np.array(object_scores))
            fov_score = -np.log(np.mean(exp_scores)) / k
            fov_score = float(np.clip(fov_score, 0.0, 1.0))
        else:
            fov_score = object_scores[0]
        
        details = {
            "all_in_fov": all_fully_in_fov,
            "any_completely_invisible": any_completely_invisible,
            "objects_checked": len(target_objects),
            "min_object_score": min(object_scores),
            "soft_min_score": fov_score,
            "object_details": object_details,
        }
        
        return fov_score, details
    
    def _apply_fov_penalty(self,
                            position_score: float,
                            orientation_score: float,
                            fov_score: float,
                            fov_details: Dict[str, Any]) -> Tuple[float, float]:
        """
        Apply FoV penalty to scores.
        
        IMPORTANT: Only penalize orientation_score, NOT position_score.
        Position quality should be independent of camera orientation.
        This avoids artificially lowering the position score for multi-object
        tasks where the heatmap forward direction may not cover all objects.
        
        Args:
            position_score: Original position score (returned UNCHANGED)
            orientation_score: Original orientation score
            fov_score: FoV visibility score (1.0 = all fully visible)
            fov_details: Details from _compute_fov_score
        
        Returns:
            (position_score_unchanged, adjusted_orientation_score)
        """
        any_completely_invisible = fov_details.get("any_completely_invisible", False)

        # AUDIT FIX (v20-prep): when any target is completely outside FoV, the
        # old code multiplied orientation by fov_score (which can be ~0.001),
        # collapsing the ori channel to zero and destroying its gradient.
        # SFT replay showed env_final_ori_score median = 0.13 across the entire
        # SFT trajectory set as a direct consequence. We add a lower bound of
        # 0.3 to the multiplicative penalty so that:
        #   - "object in FoV" still beats "object out of FoV" (1.0 vs 0.3)
        #   - but the ori channel never gets fully zeroed
        #   - path-finder / RL policy retains a useful gradient toward facing
        FOV_PENALTY_FLOOR = 0.3

        if any_completely_invisible:
            # SEVERE orientation penalty when any object is completely invisible
            # NOTE: floored at FOV_PENALTY_FLOOR so the ori gradient survives.
            adjusted_orientation_score = orientation_score * max(fov_score, FOV_PENALTY_FLOOR)
        elif fov_score < 1.0:
            # Mild orientation penalty for partial visibility (objects at edge)
            adjusted_orientation_score = orientation_score * (0.8 + 0.2 * fov_score)
        else:
            # All objects fully visible - no penalty
            adjusted_orientation_score = orientation_score

        # position_score is NEVER penalized for FoV issues
        return position_score, adjusted_orientation_score
    
    def compute_score(self,
                      camera_position: np.ndarray,
                      camera_forward: np.ndarray,
                      task_type: str,
                      task_params: Dict[str, Any],
                      target_region: Dict[str, Any]) -> ScoreResult:
        """
        Compute the score for a given camera pose and task.
        
        Args:
            camera_position: Camera position [x, y, z]
            camera_forward: Camera forward direction [dx, dy, dz]
            task_type: Type of task (e.g., "absolute_positioning")
            task_params: Task-specific parameters
            target_region: Target region definition from TaskResult
        
        Returns:
            ScoreResult with total, position, and orientation scores
        """
        camera_position = np.array(camera_position)
        camera_forward = normalize(np.array(camera_forward))
        
        # Dispatch to task-specific scoring function
        scorer_map = {
            TaskType.ABSOLUTE_POSITIONING.value: self._score_absolute_positioning,
            TaskType.DELTA_CONTROL.value: self._score_delta_control,
            TaskType.EQUIDISTANCE.value: self._score_equidistance,
            TaskType.PROJECTIVE_RELATIONS.value: self._score_projective_relations,
            TaskType.CENTERING.value: self._score_centering,
            TaskType.OCCLUSION_ALIGNMENT.value: self._score_occlusion_alignment,
            TaskType.FOV_INCLUSION.value: self._score_fov_inclusion,
            TaskType.SIZE_DISTANCE_INVARIANCE.value: self._score_size_distance_invariance,
            TaskType.APPARENT_SIZE_ORDERING.value: self._score_apparent_size_ordering,
            TaskType.SCREEN_OCCUPANCY.value: self._score_screen_occupancy,
        }
        
        scorer = scorer_map.get(task_type, self._score_default)
        result = scorer(camera_position, camera_forward, task_params, target_region)

        # Add standardized diagnostics without changing the reward/success
        # semantics. For region-valued tasks this makes sample_point explicitly
        # auxiliary and exposes distance to the whole valid region.
        details = dict(result.details or {})
        visual_metrics = compute_visual_bbox_metrics(
            camera_position=camera_position,
            camera_forward=camera_forward,
            task_type=task_type,
            task_params=task_params,
            target_region=target_region,
        )
        details["visual_bbox_metrics"] = visual_metrics
        region_metrics = compute_region_metrics(
            camera_position=camera_position,
            camera_forward=camera_forward,
            task_type=task_type,
            task_params=task_params,
            target_region=target_region,
        )
        details["region_metrics"] = region_metrics
        for key in (
            "distance_to_region",
            "region_score",
            "region_target_point",
            "sample_target_distance",
            "sample_target_is_auxiliary",
        ):
            if key in region_metrics:
                details.setdefault(key, region_metrics.get(key))

        visual_tasks = {
            TaskType.PROJECTIVE_RELATIONS.value,
            TaskType.CENTERING.value,
            TaskType.OCCLUSION_ALIGNMENT.value,
            TaskType.FOV_INCLUSION.value,
            TaskType.SIZE_DISTANCE_INVARIANCE.value,
            TaskType.APPARENT_SIZE_ORDERING.value,
            TaskType.SCREEN_OCCUPANCY.value,
        }
        if (
            self.use_visual_bbox_scoring
            and task_type in visual_tasks
            and visual_metrics.get("available")
            and visual_metrics.get("visual_score") is not None
        ):
            visual_position = float(visual_metrics.get("visual_position_score", visual_metrics["visual_score"]))
            visual_orientation = float(visual_metrics.get("visual_orientation_score", visual_metrics["visual_score"]))
            visual_position = float(np.clip(visual_position, 0.0, 1.0))
            visual_orientation = float(np.clip(visual_orientation, 0.0, 1.0))
            pos_weight, ori_weight = self._compute_dynamic_weights(visual_position)
            total_score = pos_weight * visual_position + ori_weight * visual_orientation
            details["visual_bbox_overrode_score"] = True
            details["pre_visual_total_score"] = float(result.total_score)
            details["pre_visual_position_score"] = float(result.position_score)
            details["pre_visual_orientation_score"] = float(result.orientation_score)
            details["dynamic_position_weight"] = pos_weight
            details["dynamic_orientation_weight"] = ori_weight
            details["visual_score"] = float(visual_metrics["visual_score"])
            return ScoreResult(
                total_score=float(np.clip(total_score, 0, 1)),
                position_score=visual_position,
                orientation_score=visual_orientation,
                details=details,
            )

        return ScoreResult(
            total_score=result.total_score,
            position_score=result.position_score,
            orientation_score=result.orientation_score,
            details=details,
        )
    
    def _compute_dynamic_weights(self, position_score: float) -> Tuple[float, float]:
        """
        Compute dynamic position/orientation weights based on position score.
        
        When far from optimal position (position_score low):
            - Position weight is HIGH (focus on moving to better position)
            - Orientation weight is LOW (don't penalize for wrong orientation while moving)
        
        When close to optimal position (position_score high):
            - Position weight decreases
            - Orientation weight increases (now focus on facing correctly)
        
        This solves the problem where agent gets stuck because turning away from
        the target temporarily lowers orientation score, even when moving to a
        better position.
        
        Args:
            position_score: Current position score [0, 1]
            
        Returns:
            (position_weight, orientation_weight) dynamically adjusted
        """
        if not self.use_dynamic_weights:
            return self.position_weight, self.orientation_weight
        
        threshold = self.dynamic_weight_threshold
        
        if position_score < threshold:
            # Far from optimal: position dominates
            # Linear interpolation: at pos_score=0, pos_weight=0.95; at threshold, pos_weight=base
            t = position_score / threshold  # 0 to 1 as we approach threshold
            # Start with 95% position weight, interpolate to base weights at threshold
            pos_weight = 0.95 - (0.95 - self.base_position_weight) * t
            ori_weight = 1.0 - pos_weight
        else:
            # Close to optimal: gradually shift to include orientation
            # At threshold: use base weights
            # At pos_score=1.0: converge to 0.6/0.4 (position still dominant)
            # This ensures total_score can reach close to 1.0 when both are high
            t = (position_score - threshold) / (1.0 - threshold + 1e-6)  # 0 to 1
            min_pos_weight = 0.6  # Converge to 60% position (was 50%)
            pos_weight = self.base_position_weight - (self.base_position_weight - min_pos_weight) * t
            ori_weight = 1.0 - pos_weight
        
        return pos_weight, ori_weight
    
    def _combine_scores(self, 
                        position_score: float, 
                        orientation_score: float,
                        details: Dict[str, Any],
                        camera_position: np.ndarray = None,
                        camera_forward: np.ndarray = None,
                        target_objects: List[np.ndarray] = None) -> ScoreResult:
        """
        Combine position and orientation scores into final result.
        
        IMPROVED: Uses dynamic weights based on position score.
        - When far from optimal position: position_weight dominates (agent focuses on moving)
        - When close to optimal: orientation_weight increases (agent focuses on facing)
        
        If target_objects is provided, applies FoV penalty to ensure all
        target objects remain visible.
        
        Args:
            position_score: Raw position score [0, 1]
            orientation_score: Raw orientation score [0, 1]
            details: Task-specific details dict
            camera_position: Camera position (required for FoV check)
            camera_forward: Camera forward direction (required for FoV check)
            target_objects: List of target object centers to check FoV for
        
        Returns:
            ScoreResult with adjusted scores
        """
        # Apply FoV penalty if target objects are specified
        if target_objects is not None and camera_position is not None and camera_forward is not None:
            fov_score, fov_details = self._compute_fov_score(
                camera_position, camera_forward, target_objects
            )
            
            # Apply FoV penalty
            position_score, orientation_score = self._apply_fov_penalty(
                position_score, orientation_score, fov_score, fov_details
            )
            
            # Add FoV info to details
            details["fov_score"] = fov_score
            details["fov_details"] = fov_details
            details["all_targets_in_fov"] = fov_details.get("all_in_fov", True)
        
        # IMPROVED: Use dynamic weights based on position score
        pos_weight, ori_weight = self._compute_dynamic_weights(position_score)
        
        total_score = pos_weight * position_score + ori_weight * orientation_score
        
        # Add weight info to details for debugging
        details["dynamic_position_weight"] = pos_weight
        details["dynamic_orientation_weight"] = ori_weight
        
        return ScoreResult(
            total_score=float(np.clip(total_score, 0, 1)),
            position_score=float(np.clip(position_score, 0, 1)),
            orientation_score=float(np.clip(orientation_score, 0, 1)),
            details=details
        )
    
    # =========================================================================
    # Task 1.1: Absolute Positioning (CIRCLE)
    # Goal: Be at distance d from object center
    # Position Score: Gaussian around target distance
    # Orientation Score: Looking toward object
    # FoV Constraint: Target object must remain in view
    # =========================================================================
    def _score_absolute_positioning(self,
                                     camera_position: np.ndarray,
                                     camera_forward: np.ndarray,
                                     task_params: Dict[str, Any],
                                     target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for absolute positioning task.
        
        Perfect score when camera is at exactly target_distance from object,
        facing the object, AND the object is in the field of view.
        """
        params = target_region.get("params", {})
        
        # Get object center and target radius
        object_center = np.array(params.get("object_center", params.get("center", [0, 0, 0])))
        target_radius = params.get("radius", params.get("sample_distance", 2.0))
        
        # Current distance to object
        current_distance = distance_2d(camera_position, object_center)
        
        # Position score: Gaussian around target distance
        distance_deviation = abs(current_distance - target_radius)
        sigma = max(target_radius * 0.3, 0.5)  # Adaptive sigma
        position_score = gaussian_score(distance_deviation, 0, sigma)
        
        # Orientation score: Should face the object
        to_object = object_center - camera_position
        to_object[2] = 0  # Ignore vertical component
        to_object = normalize(to_object)
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1], 0]))
        
        cos_angle = cosine_similarity(camera_forward_2d, to_object)
        # Convert from [-1, 1] to [0, 1], where 1 means facing object
        orientation_score = (cos_angle + 1) / 2
        
        details = {
            "current_distance": current_distance,
            "target_distance": target_radius,
            "distance_deviation": distance_deviation,
            "facing_angle_cos": cos_angle,
        }
        
        # Apply FoV constraint - object must be in view
        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[object_center]
        )
    
    # =========================================================================
    # Task 1.2: Delta Control (POINT)
    # Goal: Be at exact target position
    # Position Score: Distance to target point
    # Orientation Score: Looking toward object
    # FoV Constraint: Target object must remain in view
    # =========================================================================
    def _score_delta_control(self,
                              camera_position: np.ndarray,
                              camera_forward: np.ndarray,
                              task_params: Dict[str, Any],
                              target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for delta control task.
        
        Perfect score when camera is at exact target position and object is in view.
        """
        params = target_region.get("params", {})
        
        # Target position (from sample_point)
        target_point = np.array(target_region.get("sample_point", [0, 0, 0]))
        
        # Distance to target
        distance_to_target = distance_2d(camera_position, target_point)
        
        # Position score: Exponential decay from target
        position_score = exponential_decay_score(distance_to_target, decay_rate=1.0)
        
        # Get object center for orientation scoring
        # For delta control, we want to face the original object
        start_pos = np.array(params.get("start_position", camera_position))
        delta = params.get("delta", 0)
        
        # Orientation target direction.
        # sample_forward is the ground-truth facing direction stored by the
        # pipeline.  ~16% of delta_control tasks require the camera to face
        # BACKWARD relative to the movement direction (the object is behind the
        # target position), so inferring direction from delta-sign is wrong for
        # those cases.  Use sample_forward when available; fall back to the
        # original inference only when it is absent.
        sample_fwd_raw = target_region.get("sample_forward", None)
        if sample_fwd_raw is not None:
            sf = np.array(sample_fwd_raw, dtype=float)
            sf_xy = np.array([sf[0], sf[1], 0.0])
            sf_xy_norm = np.linalg.norm(sf_xy)
            if sf_xy_norm > 1e-6:
                object_dir = sf_xy[:2] / sf_xy_norm
            else:
                # sample_forward is nearly vertical — fall through to inference
                object_dir = None
        else:
            object_dir = None
        if object_dir is None:
            # Fallback: infer from movement direction (original heuristic)
            if delta != 0:
                movement_dir = normalize(target_point[:2] - start_pos[:2])
                object_dir = movement_dir if delta > 0 else -movement_dir
            else:
                object_dir = normalize(target_point[:2] - camera_position[:2])
        
        # Orientation score
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1], 0]))
        object_dir_3d = np.array([object_dir[0], object_dir[1], 0])
        cos_angle = cosine_similarity(camera_forward_2d, object_dir_3d)
        orientation_score = (cos_angle + 1) / 2
        
        details = {
            "distance_to_target": distance_to_target,
            "target_point": target_point.tolist(),
            "facing_angle_cos": cos_angle,
        }
        
        # Get object center for FoV check.
        # delta_control does NOT store object_center in params (only start_position
        # and delta are stored).  Falling back to target_point is wrong: as the
        # camera approaches and then crosses the target position, target_point
        # transitions from "straight ahead" to "directly behind", triggering a
        # severe FoV penalty that creates a score barrier at ~0.87 and prevents
        # the path-finder from ever reaching 0.95.
        #
        # Fix: when object_center is absent, synthesise a look-at point 5 m ahead
        # in the sample_forward direction (the ground-truth facing direction stored
        # by the pipeline).  This point is always ahead of the camera once it faces
        # the correct direction, so the FoV check correctly rewards facing
        # sample_forward without penalising the camera for reaching/crossing the
        # target position.
        object_center_raw = params.get("object_center", None)
        if object_center_raw is not None:
            object_center = np.array(object_center_raw)
        else:
            sample_fwd = target_region.get("sample_forward", None)
            if sample_fwd is not None:
                fwd = np.array(sample_fwd, dtype=float)
                fwd_n = np.linalg.norm(fwd)
                if fwd_n > 1e-6:
                    object_center = target_point + (fwd / fwd_n) * 5.0
                else:
                    object_center = target_point
            else:
                # Fallback: use movement direction extended 5 m beyond target
                movement = target_point - np.array(
                    [start_pos[0], start_pos[1], target_point[2]], dtype=float
                )
                m_norm = np.linalg.norm(movement)
                if m_norm > 1e-6:
                    object_center = target_point + (movement / m_norm) * 5.0
                else:
                    object_center = target_point

        # Apply FoV constraint - object must be in view
        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[object_center]
        )
    
    # =========================================================================
    # Task 1.3: Equidistance (LINE)
    # Goal: Be equidistant from objects A and B
    # Position Score: |dist_A - dist_B| should be 0
    # Orientation Score: Looking toward midpoint of A and B
    # FoV Constraint: Both objects must remain in view
    # =========================================================================
    def _score_equidistance(self,
                             camera_position: np.ndarray,
                             camera_forward: np.ndarray,
                             task_params: Dict[str, Any],
                             target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for equidistance task.
        
        Perfect score when camera is exactly equidistant from both objects,
        and both objects are visible in the field of view.
        """
        params = target_region.get("params", {})
        
        # Get object centers
        center_a = np.array(params.get("object_a_center", [0, 0, 0]))
        center_b = np.array(params.get("object_b_center", [1, 0, 0]))
        
        # Distances to both objects
        dist_a = distance_2d(camera_position, center_a)
        dist_b = distance_2d(camera_position, center_b)
        
        # Distance difference (should be 0 for equidistance)
        distance_diff = abs(dist_a - dist_b)
        
        # Position score: Gaussian around 0 difference
        # Use distance between objects as reference for sigma
        obj_distance = distance_2d(center_a, center_b)
        sigma = max(obj_distance * 0.2, 0.5)
        position_score = gaussian_score(distance_diff, 0, sigma)
        
        # Also penalize if too close to objects (min distance constraint)
        min_dist = params.get("min_distance", 0.5)
        min_current_dist = min(dist_a, dist_b)
        if min_current_dist < min_dist:
            proximity_penalty = min_current_dist / min_dist
            position_score *= proximity_penalty
        
        # Orientation score: Should face midpoint
        midpoint = (center_a + center_b) / 2
        to_midpoint = midpoint - camera_position
        to_midpoint[2] = 0
        to_midpoint = normalize(to_midpoint)
        
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1], 0]))
        cos_angle = cosine_similarity(camera_forward_2d, to_midpoint)
        orientation_score = (cos_angle + 1) / 2
        
        details = {
            "distance_to_a": dist_a,
            "distance_to_b": dist_b,
            "distance_difference": distance_diff,
            "facing_angle_cos": cos_angle,
        }
        
        # Apply FoV constraint - both objects must be in view
        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[center_a, center_b]
        )
    
    # =========================================================================
    # Task 2.1: Projective Relations (HALF_PLANE)
    # Goal: Object A appears left/right of Object B
    # Position Score: Signed distance from boundary (positive = correct side)
    # Orientation Score: Looking toward objects
    # FoV Constraint: Both objects must remain in view
    # =========================================================================
    def _score_projective_relations(self,
                                     camera_position: np.ndarray,
                                     camera_forward: np.ndarray,
                                     task_params: Dict[str, Any],
                                     target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for projective relations task (left/right of).
        
        Perfect score when camera is in the correct half-plane, facing objects,
        and both objects are visible in the field of view.
        """
        params = target_region.get("params", {})
        
        # Get boundary parameters
        boundary_point = np.array(params.get("boundary_point", [0, 0]))
        normal = np.array(params.get("normal", [1, 0]))  # Points into valid region
        relation = params.get("relation", "left")
        
        # Object centers
        center_a = np.array(params.get("object_a_center", [0, 0, 0]))
        center_b = np.array(params.get("object_b_center", [1, 0, 0]))
        
        # Signed distance to half-plane boundary
        signed_dist = signed_distance_to_half_plane(camera_position, boundary_point, normal)
        
        # Position score:
        # - Positive signed_dist = in correct half-plane
        # - Use sigmoid for smooth transition
        # - Also reward being further from boundary (more robust view)
        min_dist = params.get("min_distance", 1.0)
        
        if signed_dist > 0:
            # In correct region - higher score for being further from boundary
            position_score = sigmoid_score(signed_dist, threshold=min_dist * 0.5, steepness=2.0)
        else:
            # In wrong region - penalize based on how far into wrong region
            position_score = sigmoid_score(signed_dist, threshold=-min_dist, steepness=2.0)
        
        # Orientation score: Should face midpoint of objects
        midpoint = (center_a + center_b) / 2
        to_midpoint = midpoint - camera_position
        to_midpoint[2] = 0
        to_midpoint = normalize(to_midpoint)
        
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1], 0]))
        cos_angle = cosine_similarity(camera_forward_2d, to_midpoint)
        orientation_score = (cos_angle + 1) / 2
        
        # Verify the projective relation actually holds
        # Project A and B onto camera's image plane (simplified 2D check)
        to_a = center_a[:2] - camera_position[:2]
        to_b = center_b[:2] - camera_position[:2]
        
        # Cross product to determine left/right
        cross = camera_forward[0] * to_a[1] - camera_forward[1] * to_a[0]
        cross_b = camera_forward[0] * to_b[1] - camera_forward[1] * to_b[0]
        
        # A is left of B if cross_a > cross_b (in right-handed coordinate)
        a_left_of_b = cross > cross_b
        relation_satisfied = (relation == "left" and a_left_of_b) or (relation == "right" and not a_left_of_b)
        
        # Bonus for actually satisfying the relation
        if relation_satisfied and signed_dist > 0:
            position_score = min(1.0, position_score * 1.2)
        
        details = {
            "signed_distance": signed_dist,
            "relation": relation,
            "relation_satisfied": relation_satisfied,
            "facing_angle_cos": cos_angle,
        }
        
        # Apply FoV constraint - both objects must be in view
        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[center_a, center_b]
        )
    
    # =========================================================================
    # Task 2.2: Centering (RAY)
    # Goal: Object A appears centered between B and C
    # Position Score: Distance to the centering ray
    # Orientation Score: A should be angularly centered in view
    # FoV Constraint: All three objects must remain in view
    # =========================================================================
    def _score_centering(self,
                          camera_position: np.ndarray,
                          camera_forward: np.ndarray,
                          task_params: Dict[str, Any],
                          target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for centering task.
        
        Perfect score when camera is on the ray, A is centered between B and C,
        and all three objects are visible in the field of view.
        """
        params = target_region.get("params", {})
        
        # Ray parameters
        ray_origin = np.array(params.get("origin", [0, 0]))
        ray_direction = normalize(np.array(params.get("direction", [1, 0])))
        
        # Object centers
        center_a = np.array(params.get("object_a_center", [0, 0, 0]))
        midpoint_bc = np.array(params.get("midpoint_bc", [0, 0, 0]))
        
        # Distance to ray
        dist_to_ray = point_to_line_distance_2d(camera_position, ray_origin, ray_direction)
        
        # Position score: Exponential decay from ray
        position_score = exponential_decay_score(dist_to_ray, decay_rate=0.5)
        
        # Also check if on the correct side of midpoint_bc (looking back at objects)
        # Camera should be beyond midpoint_bc in the ray direction
        cam_to_mid = midpoint_bc[:2] - camera_position[:2]
        proj_dist = np.dot(cam_to_mid, ray_direction[:2])
        if proj_dist > 0:  # Camera is behind the midpoint (wrong side)
            position_score *= 0.5
        
        # Orientation score: A should be centered
        # Compute angles to A relative to forward direction
        to_a = center_a[:2] - camera_position[:2]
        
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1]]))
        to_a_norm = normalize(to_a)
        
        # Angle between forward and to_A (should be ~0 for centering)
        cos_to_a = np.dot(camera_forward_2d, to_a_norm)
        angle_to_a = np.arccos(np.clip(cos_to_a, -1, 1))
        
        # Orientation score: Penalize if A is not centered
        orientation_score = gaussian_score(angle_to_a, 0, np.radians(15))
        
        details = {
            "distance_to_ray": dist_to_ray,
            "angle_to_a_deg": np.degrees(angle_to_a),
            "on_correct_side": proj_dist <= 0,
        }
        
        # Get object B and C centers for FoV check (centering involves A, B, C)
        center_b = np.array(params.get("object_b_center", midpoint_bc))
        center_c = np.array(params.get("object_c_center", midpoint_bc))
        
        # Apply FoV constraint - A, B, C must all be in view
        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[center_a, center_b, center_c]
        )
    
    # =========================================================================
    # Task 2.3: Occlusion Alignment (RAY)
    # Goal: Object A is hidden behind Object B
    # Position Score: Alignment with A-B ray (extended beyond B)
    # Orientation Score: Looking toward B
    # FoV Constraint: Object B (occluder) must be in view
    # =========================================================================
    def _score_occlusion_alignment(self,
                                    camera_position: np.ndarray,
                                    camera_forward: np.ndarray,
                                    task_params: Dict[str, Any],
                                    target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for occlusion alignment task.
        
        Perfect score when camera is on the A-B-camera line, A is occluded by B,
        and B (the occluding object) is visible in the field of view.
        """
        params = target_region.get("params", {})
        
        # Ray parameters
        ray_origin = np.array(params.get("origin", [0, 0]))
        ray_direction = normalize(np.array(params.get("direction", [1, 0])))
        
        # Object centers
        center_a = np.array(params.get("object_a_center", [0, 0, 0]))
        center_b = np.array(params.get("object_b_center", [1, 0, 0]))
        
        # Distance to ray (A-B line extended)
        dist_to_ray = point_to_line_distance_2d(camera_position, ray_origin, ray_direction)
        
        # Position score: Exponential decay from ray
        position_score = exponential_decay_score(dist_to_ray, decay_rate=0.5)

        # Respect finite ray bounds when the data generator constrains the
        # useful occlusion segment (e.g. to keep the occluder's angular width
        # above the discrete yaw resolution).
        rel_to_origin = camera_position[:2] - ray_origin[:2]
        ray_progress = float(np.dot(rel_to_origin, ray_direction[:2]))
        min_ray_dist = float(params.get("min_distance", 0.0) or 0.0)
        max_ray_dist = params.get("max_distance", None)
        if ray_progress < min_ray_dist:
            position_score *= exponential_decay_score(min_ray_dist - ray_progress, decay_rate=0.5)
        if max_ray_dist is not None:
            max_ray_dist = float(max_ray_dist)
            if ray_progress > max_ray_dist:
                position_score *= exponential_decay_score(ray_progress - max_ray_dist, decay_rate=0.5)
        
        # Check if B is between camera and A
        dist_to_a = distance_2d(camera_position, center_a)
        dist_to_b = distance_2d(camera_position, center_b)
        b_is_closer = dist_to_b < dist_to_a
        
        if not b_is_closer:
            # B is not between camera and A - occlusion not possible
            position_score *= 0.3
        
        # Orientation score: Should face B (which occludes A)
        to_b = center_b[:2] - camera_position[:2]
        to_b_norm = normalize(to_b)
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1]]))
        
        cos_angle = np.dot(camera_forward_2d, to_b_norm)
        orientation_score = (cos_angle + 1) / 2
        
        # Check actual occlusion alignment
        # A-B-Camera should be collinear
        to_a = center_a[:2] - camera_position[:2]
        to_a_norm = normalize(to_a)
        alignment = abs(np.dot(to_a_norm, to_b_norm))  # Should be ~1 for alignment

        # Optional angular-width gate for datasets generated with
        # occlusion_min_angular_width_deg. Old datasets do not contain these
        # fields, so their scoring remains unchanged.
        angular_width_deg = None
        min_angular_width_deg = params.get("min_occlusion_angular_width_deg", None)
        occluder_width = params.get("occluder_width_for_angle", None)
        if min_angular_width_deg is not None and occluder_width is not None:
            min_angular_width_deg = float(min_angular_width_deg)
            occluder_width = float(occluder_width)
            angular_width_deg = float(
                np.degrees(2.0 * np.arctan((occluder_width / 2.0) / max(dist_to_b, 1e-6)))
            )
            if min_angular_width_deg > 0.0 and angular_width_deg < min_angular_width_deg:
                position_score *= max(0.1, angular_width_deg / min_angular_width_deg)
        
        details = {
            "distance_to_ray": dist_to_ray,
            "ray_progress": ray_progress,
            "b_is_closer": b_is_closer,
            "collinearity": alignment,
            "facing_angle_cos": cos_angle,
        }
        if angular_width_deg is not None:
            details["occluder_angular_width_deg"] = angular_width_deg
            details["min_occlusion_angular_width_deg"] = float(min_angular_width_deg)
        
        # Apply FoV constraint - B (occluder) must be in view
        # Note: A being hidden is the goal, so we only require B to be visible
        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[center_b]
        )
    
    # =========================================================================
    # Task 3.1: FoV Inclusion (ANNULUS)
    # Goal: Both objects A and B are visible in the camera's field of view
    # Position Score: Far enough to include both in FOV
    # Orientation Score: Both objects within FOV angle
    # =========================================================================
    def _score_fov_inclusion(self,
                              camera_position: np.ndarray,
                              camera_forward: np.ndarray,
                              task_params: Dict[str, Any],
                              target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for FoV inclusion task.
        
        Perfect score when both objects are visible within the camera's field of view.
        """
        params = target_region.get("params", {})
        
        # Annulus parameters
        center = np.array(params.get("center", [0, 0]))
        min_radius = params.get("min_radius", 2.0)
        max_radius = params.get("max_radius", 10.0)
        fov_horizontal = params.get("fov_horizontal", 60.0)
        
        # Object centers
        center_a = np.array(params.get("object_a_center", [0, 0, 0]))
        center_b = np.array(params.get("object_b_center", [1, 0, 0]))
        
        # Distance from camera to midpoint
        midpoint = np.array([center[0], center[1], camera_position[2]])
        dist_to_midpoint = distance_2d(camera_position, midpoint)
        
        # Position score: Should be within annulus
        if dist_to_midpoint < min_radius:
            # Too close - can't see both objects
            position_score = dist_to_midpoint / min_radius
        elif dist_to_midpoint > max_radius:
            # Too far - objects too small
            position_score = max_radius / dist_to_midpoint
        else:
            # In optimal range
            position_score = 1.0
        
        # Orientation score: Both objects should be within FOV
        to_a = center_a[:2] - camera_position[:2]
        to_b = center_b[:2] - camera_position[:2]
        
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1]]))
        to_a_norm = normalize(to_a)
        to_b_norm = normalize(to_b)
        
        # Angles to A and B from forward direction
        cos_to_a = np.dot(camera_forward_2d, to_a_norm)
        cos_to_b = np.dot(camera_forward_2d, to_b_norm)
        
        angle_to_a = np.arccos(np.clip(cos_to_a, -1, 1))
        angle_to_b = np.arccos(np.clip(cos_to_b, -1, 1))
        
        fov_rad = np.radians(fov_horizontal) / 2
        
        # Both should be within half-FOV
        a_visible = angle_to_a < fov_rad
        b_visible = angle_to_b < fov_rad

        if a_visible and b_visible:
            # Both visible — score by how well camera faces the 3-D midpoint of
            # A+B.  Using the full 3-D direction (not just XY) is necessary when
            # the camera has a non-zero pitch: a Z=0 approximation would give
            # cos_to_mid < 1 even at the oracle pose, incorrectly penalising
            # correct camera orientations.
            midpoint_3d = (center_a + center_b) / 2.0
            midpoint_dir = midpoint_3d - camera_position
            midpoint_dir_norm = np.linalg.norm(midpoint_dir)
            if midpoint_dir_norm > 1e-6:
                cos_to_mid = float(np.dot(
                    normalize(camera_forward[:3]),
                    midpoint_dir / midpoint_dir_norm,
                ))
                orientation_score = (cos_to_mid + 1.0) / 2.0
            else:
                orientation_score = 1.0  # camera is at midpoint
        elif a_visible or b_visible:
            # Only one visible
            orientation_score = 0.5
        else:
            # Neither visible — gradient toward the nearer object
            min_angle = min(angle_to_a, angle_to_b)
            orientation_score = max(0.0, 1.0 - min_angle / np.pi)
        
        details = {
            "distance_to_midpoint": dist_to_midpoint,
            "min_radius": min_radius,
            "max_radius": max_radius,
            "angle_to_a_deg": np.degrees(angle_to_a),
            "angle_to_b_deg": np.degrees(angle_to_b),
            "a_visible": a_visible,
            "b_visible": b_visible,
        }
        
        # Apply FoV constraint - both objects must be in view
        # This is especially important for FoV inclusion task
        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[center_a, center_b]
        )
    
    # =========================================================================
    # Task 3.2: Size-Distance Invariance (CURVE)
    # Goal: Objects A and B appear the same size on screen
    # Position Score: size_A/dist_A ≈ size_B/dist_B
    # Orientation Score: Looking toward midpoint of objects
    # FoV Constraint: Both objects must remain in view
    # =========================================================================
    def _score_size_distance_invariance(self,
                                         camera_position: np.ndarray,
                                         camera_forward: np.ndarray,
                                         task_params: Dict[str, Any],
                                         target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for size-distance invariance task.
        
        Perfect score when both objects appear the same angular size,
        and both are visible in the field of view.
        """
        params = target_region.get("params", {})
        
        # Object parameters
        center_a = np.array(params.get("object_a_center", [0, 0, 0]))
        center_b = np.array(params.get("object_b_center", [1, 0, 0]))
        size_a = params.get("object_a_size", 1.0)
        size_b = params.get("object_b_size", 1.0)
        
        # Use 2D (horizontal) distances to match the Apollonius circle constraint
        # used by the pipeline: apparent_size = object_size / horizontal_distance.
        # Using 3D distance would give wrong ratios when objects differ in height.
        dist_a = max(distance_2d(camera_position, center_a), 0.1)
        dist_b = max(distance_2d(camera_position, center_b), 0.1)
        
        # Angular sizes (proportional to size/distance)
        angular_a = size_a / dist_a
        angular_b = size_b / dist_b
        
        # Position score: Ratio should be 1
        if angular_b > 0:
            size_ratio = angular_a / angular_b
        else:
            size_ratio = 1.0
        
        # Score based on how close ratio is to 1
        # Use moderate decay so the agent can achieve high score with discrete steps
        # (agent step_translation=0.3m, Apollonius circles can be small ~0.6m radius)
        ratio_deviation = abs(np.log(size_ratio))  # log scale for symmetry
        position_score = exponential_decay_score(ratio_deviation, decay_rate=1.0)
        
        # Use Apollonius circle distance as secondary gradient signal only
        # The ratio_score is the primary metric (ratio=1 iff exactly on circle)
        if "radius" in params and "center" in params:
            apollonius_center = np.array(params["center"])
            apollonius_radius = params["radius"]
            dist_to_circle = abs(distance_2d(camera_position, apollonius_center) - apollonius_radius)
            circle_score = exponential_decay_score(dist_to_circle, decay_rate=0.3)
            # Ratio score is primary (weight 0.8), circle provides gradient guide (0.2)
            position_score = 0.8 * position_score + 0.2 * circle_score
        
        # Orientation score: Should face midpoint
        midpoint = (center_a + center_b) / 2
        to_midpoint = midpoint - camera_position
        to_midpoint[2] = 0
        to_midpoint = normalize(to_midpoint)
        
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1], 0]))
        cos_angle = cosine_similarity(camera_forward_2d, to_midpoint)
        orientation_score = (cos_angle + 1) / 2
        
        details = {
            "distance_to_a": dist_a,
            "distance_to_b": dist_b,
            "angular_size_a": angular_a,
            "angular_size_b": angular_b,
            "size_ratio": size_ratio,
            "facing_angle_cos": cos_angle,
        }
        
        # FoV constraint.
        # size_distance_invariance can generate oracle poses where the camera is
        # on the Apollonius circle but very close to (and facing AWAY from)
        # center_b.  In those cases forward_dot < 0 → object flagged as
        # completely_invisible → FoV penalty floor 0.3 → total ≈ 0.7.
        # Using sample_forward to build a synthetic far-away look-at point
        # (identical strategy to delta_control) avoids this false penalty.
        sample_fwd_raw = target_region.get("sample_forward", None)
        if sample_fwd_raw is not None:
            fwd = np.array(sample_fwd_raw, dtype=float)
            fwd_n = np.linalg.norm(fwd)
            if fwd_n > 1e-6:
                synthetic_target = camera_position + (fwd / fwd_n) * 5.0
                fov_targets = [synthetic_target]
            else:
                fov_targets = [center_a, center_b]
        else:
            fov_targets = [center_a, center_b]

        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=fov_targets
        )

    # =========================================================================
    # Task 3.3: Apparent Size Ordering (SAMPLED REGION)
    # Goal: The designated larger object appears larger than the other object
    # Position Score: size_large/dist_large >= target_ratio * size_small/dist_small
    # Orientation Score: Looking toward midpoint of objects
    # FoV Constraint: Both objects must remain in view
    # =========================================================================
    def _score_apparent_size_ordering(self,
                                      camera_position: np.ndarray,
                                      camera_forward: np.ndarray,
                                      task_params: Dict[str, Any],
                                      target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for apparent-size ordering.

        Perfect position score once the requested ordering is clearly satisfied.
        Below the threshold, the score decays smoothly on a log-ratio scale.
        """
        params = target_region.get("params", {})

        large_center = np.array(params.get(
            "larger_object_center",
            params.get("object_a_center", [0, 0, 0]),
        ))
        small_center = np.array(params.get(
            "smaller_object_center",
            params.get("object_b_center", [1, 0, 0]),
        ))
        large_size = max(float(params.get("larger_object_size", 1.0)), 0.05)
        small_size = max(float(params.get("smaller_object_size", 1.0)), 0.05)
        target_ratio = max(float(params.get("target_ratio", 1.25)), 1.01)

        dist_large = max(distance_2d(camera_position, large_center), 0.1)
        dist_small = max(distance_2d(camera_position, small_center), 0.1)
        apparent_large = large_size / dist_large
        apparent_small = small_size / dist_small
        apparent_ratio = apparent_large / max(apparent_small, 1e-6)

        log_ratio = np.log(max(apparent_ratio, 1e-6))
        log_target = np.log(target_ratio)
        if log_ratio >= log_target:
            position_score = 1.0
        else:
            # Smoothly penalize violations. sigma=0.35 allows useful gradient
            # when the objects are close to the decision boundary.
            position_score = gaussian_score(log_ratio, log_target, sigma=0.35)

        min_dist = float(params.get("min_distance", 0.5))
        min_current_dist = min(dist_large, dist_small)
        if min_current_dist < min_dist:
            position_score *= max(0.0, min_current_dist / max(min_dist, 1e-6))

        midpoint = (large_center + small_center) / 2
        to_midpoint = midpoint - camera_position
        to_midpoint[2] = 0
        to_midpoint = normalize(to_midpoint)

        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1], 0]))
        cos_angle = cosine_similarity(camera_forward_2d, to_midpoint)
        orientation_score = (cos_angle + 1) / 2

        details = {
            "distance_to_larger": dist_large,
            "distance_to_smaller": dist_small,
            "apparent_larger": apparent_large,
            "apparent_smaller": apparent_small,
            "apparent_ratio": apparent_ratio,
            "target_ratio": target_ratio,
            "ordering_satisfied": apparent_ratio >= target_ratio,
            "facing_angle_cos": cos_angle,
        }

        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[large_center, small_center],
        )

    # =========================================================================
    # Task 3.4: Screen Occupancy (CIRCLE)
    # Goal: Object occupies k% of the vertical field of view
    # Position Score: Distance gives correct occupancy ratio
    # Orientation Score: Looking directly at object
    # FoV Constraint: Target object must remain in view
    # =========================================================================
    def _score_screen_occupancy(self,
                                 camera_position: np.ndarray,
                                 camera_forward: np.ndarray,
                                 task_params: Dict[str, Any],
                                 target_region: Dict[str, Any]) -> ScoreResult:
        """
        Score for screen occupancy task.
        
        Perfect score when object occupies exactly target percentage of vertical FOV,
        and the object is visible in the field of view.
        """
        params = target_region.get("params", {})
        
        # Object and target parameters
        object_center = np.array(params.get("object_center", [0, 0, 0]))
        object_height = params.get("object_height", 1.0)
        target_occupancy = params.get("occupancy_ratio", 0.5)
        fov_vertical = params.get("fov_vertical", 60.0)
        target_radius = params.get("radius", 2.0)
        
        # Current distance
        current_distance = max(distance_2d(camera_position, object_center), 0.1)
        
        # Current occupancy ratio
        fov_rad = np.radians(fov_vertical)
        current_angular_size = 2 * np.arctan(object_height / (2 * current_distance))
        current_occupancy = current_angular_size / fov_rad
        
        # Position score: Gaussian around target occupancy
        occupancy_deviation = abs(current_occupancy - target_occupancy)
        sigma = target_occupancy * 0.3  # 30% tolerance
        position_score = gaussian_score(occupancy_deviation, 0, sigma)
        
        # Alternative: score based on distance to target radius
        distance_deviation = abs(current_distance - target_radius)
        distance_score = gaussian_score(distance_deviation, 0, target_radius * 0.3)
        position_score = 0.5 * position_score + 0.5 * distance_score
        
        # Orientation score: Should face object center
        to_object = object_center - camera_position
        to_object[2] = 0
        to_object = normalize(to_object)
        
        camera_forward_2d = normalize(np.array([camera_forward[0], camera_forward[1], 0]))
        cos_angle = cosine_similarity(camera_forward_2d, to_object)
        orientation_score = (cos_angle + 1) / 2
        
        details = {
            "current_distance": current_distance,
            "target_distance": target_radius,
            "current_occupancy": current_occupancy,
            "target_occupancy": target_occupancy,
            "occupancy_deviation": occupancy_deviation,
            "facing_angle_cos": cos_angle,
        }
        
        # Apply FoV constraint - object must be in view
        return self._combine_scores(
            position_score, orientation_score, details,
            camera_position=camera_position,
            camera_forward=camera_forward,
            target_objects=[object_center]
        )
    
    # =========================================================================
    # Default scorer (fallback)
    # =========================================================================
    def _score_default(self,
                       camera_position: np.ndarray,
                       camera_forward: np.ndarray,
                       task_params: Dict[str, Any],
                       target_region: Dict[str, Any]) -> ScoreResult:
        """
        Default scoring function using sample point distance.
        
        Used as fallback when task type is not recognized.
        """
        # Use sample point as target
        sample_point = np.array(target_region.get("sample_point", camera_position))
        sample_forward = np.array(target_region.get("sample_forward", camera_forward))
        
        # Position score: Distance to sample point
        dist = distance_2d(camera_position, sample_point)
        position_score = exponential_decay_score(dist, decay_rate=0.5)
        
        # Orientation score: Alignment with sample forward
        cos_angle = cosine_similarity(camera_forward, sample_forward)
        orientation_score = (cos_angle + 1) / 2
        
        details = {
            "distance_to_sample": dist,
            "forward_alignment": cos_angle,
        }
        
        return self._combine_scores(position_score, orientation_score, details)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_potential_field(config: Optional[Dict[str, Any]] = None) -> SpatialPotentialField:
    """
    Create a SpatialPotentialField with optional configuration.
    
    Args:
        config: Optional dict with keys:
            - position_weight: Weight for position score (default 0.7)
            - orientation_weight: Weight for orientation score (default 0.3)
            - position_sigma: Sigma for Gaussian scoring (default 2.0)
            - max_distance: Max distance for scoring (default 10.0)
            - fov_horizontal: Horizontal field of view in degrees (default 60.0)
            - fov_vertical: Vertical field of view in degrees (default 60.0)
            - fov_penalty_weight: Weight of FoV penalty (default 0.5)
            - fov_penalty_softness: Softness of FoV boundary in radians (default 0.2)
            - use_dynamic_weights: If True, dynamically adjust weights (default True)
            - dynamic_weight_threshold: Position score threshold for dynamic weights (default 0.7)
            - use_visual_bbox_scoring: If True, visual-relation tasks use projected
              3D bbox metrics as the primary score (default True)
    
    Returns:
        Configured SpatialPotentialField instance
    """
    if config is None:
        config = {}
    
    return SpatialPotentialField(
        position_weight=config.get("position_weight", 0.7),
        orientation_weight=config.get("orientation_weight", 0.3),
        position_sigma=config.get("position_sigma", 2.0),
        orientation_sigma=config.get("orientation_sigma", 0.5),
        max_distance=config.get("max_distance", 10.0),
        fov_horizontal=config.get("fov_horizontal", 90.0),
        fov_vertical=config.get("fov_vertical", 90.0),
        fov_penalty_weight=config.get("fov_penalty_weight", 0.5),
        fov_penalty_softness=config.get("fov_penalty_softness", 0.2),
        use_dynamic_weights=config.get("use_dynamic_weights", True),
        dynamic_weight_threshold=config.get("dynamic_weight_threshold", 0.7),
        use_visual_bbox_scoring=config.get("use_visual_bbox_scoring", True),
    )


def compute_task_score(camera_position: np.ndarray,
                       camera_forward: np.ndarray,
                       task_result: Dict[str, Any],
                       field: Optional[SpatialPotentialField] = None) -> ScoreResult:
    """
    Convenience function to compute score for a task.
    
    Args:
        camera_position: Camera position [x, y, z]
        camera_forward: Camera forward direction [dx, dy, dz]
        task_result: TaskResult as dict (with task_type, task_params, target_region)
        field: Optional pre-created potential field (creates new if None)
    
    Returns:
        ScoreResult with scores and details
    """
    if field is None:
        field = create_potential_field()
    
    return field.compute_score(
        camera_position=camera_position,
        camera_forward=camera_forward,
        task_type=task_result.get("task_type", "default"),
        task_params=task_result.get("task_params", {}),
        target_region=task_result.get("target_region", {}),
    )
