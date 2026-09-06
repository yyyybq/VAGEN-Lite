#!/usr/bin/env python3
"""Generate versioned FOV and projective R1 repairs without changing sources."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from r1_canonical_tasks import (
    CANONICAL_TASK_METRIC_VERSION,
    CANONICAL_CAMERA_H1_RESIZE_V1,
    canonical_fov,
    canonical_projective,
    pose_from_item_target,
    score_observation,
)
from vagen.envs.active_spatial.canonical_camera import camera_pose_from_forward, normalize_vector
from vagen.envs.active_spatial.collision_detector import CollisionDetector
from vagen.envs.active_spatial.utils import ViewManipulator
from data_gen.active_spatial_pipeline.layout_quality import LayoutGeometry, point_in_poly

PROJECTIVE_GENERATOR_VERSION = "projective_canonical_h1_v6_initial_observable_difficulty_v1"
FOV_GENERATOR_VERSION = "fov_canonical_h1_v2_min4_runtime_collision_v2"
MAX_ABS_PITCH_DEG = 30.0
PROJECTIVE_INITIAL_POINT_BUDGET = 32
PROJECTIVE_TARGET_POINT_BUDGET = 64
FOV_REVERSE_TURN_ACTIONS = 9
MIN_FOV_PLANNER_STEPS = 4
PLANNER_ACTIONS = (
    "move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"
)


class SceneConstraints:
    """Load and enforce the existing indoor, wall, and object-collision gates."""

    def __init__(
        self,
        gs_root: Path | None,
        min_wall_clearance: float = 0.5,
        object_margin: float = 0.2,
        structure_y_sign_overrides: dict[str, float] | None = None,
    ):
        self.gs_root = gs_root
        self.min_wall_clearance = float(min_wall_clearance)
        self.object_margin = float(object_margin)
        self.structure_y_sign_overrides = dict(structure_y_sign_overrides or {})
        self._cache: dict[str, tuple[LayoutGeometry, list[tuple[np.ndarray, np.ndarray, str]]]] = {}
        self._collision_cache: dict[str, CollisionDetector] = {}
        self._base_safe_point_cache: dict[tuple[str, int | None, float], list[np.ndarray]] = {}
        self._projective_point_cache: dict[tuple[Any, ...], list[np.ndarray]] = {}
        self.cache_stats: Counter[str] = Counter()

    @property
    def available(self) -> bool:
        return self.gs_root is not None

    def scene(self, scene_id: str) -> tuple[LayoutGeometry, list[tuple[np.ndarray, np.ndarray, str]]]:
        if scene_id in self._cache:
            return self._cache[scene_id]
        if self.gs_root is None:
            result = (LayoutGeometry(room_polys=[], wall_segments=[]), [])
            self._cache[scene_id] = result
            return result
        scene_path = self.gs_root / scene_id
        detector = CollisionDetector(
            camera_radius=0.15,
            floor_height=0.3,
            ceiling_height=2.5,
            safety_margin=max(0.0, self.object_margin - 0.15),
            enable_object_collision=True,
            enable_boundary_collision=True,
            structure_y_sign_overrides=self.structure_y_sign_overrides,
        )
        if not detector.load_scene(scene_path, scene_id=scene_id):
            result = (LayoutGeometry(room_polys=[], wall_segments=[]), [])
            self._cache[scene_id] = result
            return result
        room_polys = [[(float(p[0]), float(p[1])) for p in poly] for poly in detector.room_profiles]
        wall_segments = [
            ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
            for a, b in detector.wall_segments
        ]
        layout = LayoutGeometry(room_polys=room_polys, wall_segments=wall_segments)
        objects = [(box.min_point.copy(), box.max_point.copy(), box.label) for box in detector.object_boxes]
        self._collision_cache[scene_id] = detector
        result = (layout, objects)
        self._cache[scene_id] = result
        return result

    def room_index(self, layout: LayoutGeometry, xy: np.ndarray) -> int | None:
        for index, polygon in enumerate(layout.room_polys):
            if point_in_poly(float(xy[0]), float(xy[1]), polygon):
                return index
        return None

    def validate(
        self,
        item: dict[str, Any],
        point: np.ndarray,
        *,
        initial_room_index: int | None = None,
        check_pair_distance: bool,
    ) -> dict[str, Any]:
        scene_id = str(item.get("scene_id") or "")
        layout, objects = self.scene(scene_id)
        detector = self._collision_cache.get(scene_id)
        convention = detector.convention_record() if detector is not None else None
        scene_available = bool(layout.room_polys)
        xy = np.asarray(point, dtype=float)[:2]
        room_index = self.room_index(layout, xy) if scene_available else None
        wall_distance = float(layout.wall_distance(xy)) if scene_available else None
        collision_label = None
        if scene_available:
            p3 = np.asarray(point, dtype=float)[:3]
            for bbox_min, bbox_max, label in objects:
                # Runtime CollisionDetector boxes already include camera radius
                # and safety margin; do not expand them twice here.
                if np.all(p3 >= bbox_min) and np.all(p3 <= bbox_max):
                    collision_label = label
                    break
        params = item.get("target_region", {}).get("params", {})
        centers = [np.asarray(params[key], dtype=float)[:2] for key in ("object_a_center", "object_b_center") if params.get(key) is not None]
        pair_distance = min((float(np.linalg.norm(xy - center)) for center in centers), default=float("inf"))
        required_pair_distance = float(params.get("min_distance", 0.0) or 0.0)
        gates = {
            "scene_layout_available": scene_available,
            "collision_convention_unambiguous": bool(
                convention and convention.get("status") == "frozen"
            ),
            "inside_room": room_index is not None,
            "same_room_as_initial": initial_room_index is None or room_index == initial_room_index,
            "wall_clearance": wall_distance is not None and wall_distance >= self.min_wall_clearance,
            "object_collision_free": collision_label is None,
            "pair_min_distance": (not check_pair_distance) or pair_distance >= required_pair_distance,
        }
        return {
            "success": all(gates.values()),
            "gates": gates,
            "room_index": room_index,
            "wall_distance": wall_distance,
            "collision_label": collision_label,
            "pair_distance": pair_distance,
            "required_pair_distance": required_pair_distance,
            "collision_convention": convention,
        }

    def layout_candidates(self, item: dict[str, Any], initial_room_index: int | None, limit: int = 512) -> list[np.ndarray]:
        layout, _ = self.scene(str(item.get("scene_id") or ""))
        if not layout.room_polys:
            return []
        params = item["target_region"]["params"]
        object_ids = tuple(
            str(row.get("id")) for row in item.get("target_object", {}).get("objects", [])
        )
        cache_key = (
            str(item.get("scene_id") or ""), object_ids, params.get("relation"),
            initial_room_index, tuple(round(float(value), 4) for value in item["sample_target"]),
        )
        cached = self._projective_point_cache.get(cache_key)
        if cached is not None:
            self.cache_stats["projective_point_hits"] += 1
            return [point.copy() for point in cached[:limit]]
        self.cache_stats["projective_point_misses"] += 1
        boundary = np.asarray(params["boundary_point"], dtype=float)[:2]
        normal = relation_normal(item)
        old = np.asarray(item["sample_target"], dtype=float)
        delta = old[:2] - boundary
        mirrored = boundary + delta - 2.0 * float(np.dot(delta, normal)) * normal
        ranked: list[tuple[float, np.ndarray]] = []
        for xy in layout.candidate_points(grid_spacing=0.20):
            if float(np.dot(xy - boundary, normal)) <= 0.0:
                continue
            point = np.array([xy[0], xy[1], old[2]], dtype=float)
            check = self.validate(item, point, initial_room_index=initial_room_index, check_pair_distance=True)
            if check["success"]:
                ranked.append((float(np.linalg.norm(xy - mirrored)), point))
        ranked.sort(key=lambda row: row[0])
        result = [point for _, point in ranked]
        self._projective_point_cache[cache_key] = [point.copy() for point in result]
        return result[:limit]

    def safe_layout_candidates(
        self,
        item: dict[str, Any],
        room_index: int | None,
        reference: np.ndarray,
        limit: int = 512,
        check_pair_distance: bool = True,
    ) -> list[np.ndarray]:
        layout, _ = self.scene(str(item.get("scene_id") or ""))
        if not layout.room_polys:
            return []
        height = float(np.asarray(item["sample_target"], dtype=float)[2])
        scene_id = str(item.get("scene_id") or "")
        cache_key = (scene_id, room_index, round(height, 6))
        base_points = self._base_safe_point_cache.get(cache_key)
        if base_points is None:
            self.cache_stats["free_space_misses"] += 1
            base_points = []
            for xy in layout.candidate_points(grid_spacing=0.20):
                point = np.array([xy[0], xy[1], height], dtype=float)
                check = self.validate(
                    item,
                    point,
                    initial_room_index=room_index,
                    check_pair_distance=False,
                )
                if check["success"]:
                    base_points.append(point)
            self._base_safe_point_cache[cache_key] = base_points
        else:
            self.cache_stats["free_space_hits"] += 1
        params = item.get("target_region", {}).get("params", {})
        centers = [
            np.asarray(params[key], dtype=float)[:2]
            for key in ("object_a_center", "object_b_center")
            if params.get(key) is not None
        ]
        required_distance = float(params.get("min_distance", 0.0) or 0.0)
        ranked: list[tuple[float, np.ndarray]] = []
        for point in base_points:
            if check_pair_distance and centers and min(
                float(np.linalg.norm(point[:2] - center)) for center in centers
            ) < required_distance:
                continue
            ranked.append(
                (float(np.linalg.norm(point[:2] - np.asarray(reference, dtype=float)[:2])), point)
            )
        ranked.sort(key=lambda row: row[0])
        return [point for _, point in ranked[:limit]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def pair_midpoint(item: dict[str, Any]) -> np.ndarray:
    p = item["target_region"]["params"]
    return (np.asarray(p["object_a_center"], dtype=float) + np.asarray(p["object_b_center"], dtype=float)) / 2.0


def horizontal_forward(direction: np.ndarray) -> np.ndarray:
    result = np.asarray(direction, dtype=float).copy()
    result[2] = 0.0
    if float(np.linalg.norm(result[:2])) <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize_vector(result)


def absolute_pitch_degrees(direction: np.ndarray) -> float:
    value = normalize_vector(np.asarray(direction, dtype=float))
    return abs(math.degrees(math.atan2(float(value[2]), float(np.linalg.norm(value[:2])))))


def reverse_turn_initial(success_pose: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Construct an initial pose with an exact nine-action return certificate."""
    engine = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
    engine.reset(np.asarray(success_pose, dtype=float))
    for _ in range(FOV_REVERSE_TURN_ACTIONS):
        engine.step("turn_left")
    return engine.get_pose(), ["turn_right"] * FOV_REVERSE_TURN_ACTIONS


def canonical_fov_depth_lower_bound(
    item: dict[str, Any],
    initial_pose: np.ndarray,
    detector: CollisionDetector | None,
    max_steps: int,
    metric_cache: dict[tuple[float, ...], bool] | None = None,
) -> dict[str, Any]:
    """Completely enumerate all legal states through ``max_steps``."""
    metric_cache = metric_cache if metric_cache is not None else {}
    frontier = [np.asarray(initial_pose, dtype=float)]
    visited = {
        tuple(np.round(np.concatenate((frontier[0][:3, 3], frontier[0][:3, :3].ravel())), 7))
    }
    states_per_depth = [1]
    collision_rejections: Counter[str] = Counter()
    for depth in range(1, max_steps + 1):
        next_frontier = []
        for pose in frontier:
            for action in PLANNER_ACTIONS:
                engine = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
                engine.reset(pose)
                candidate = engine.step(action)
                if detector is not None and action.startswith("move_"):
                    collision = detector.check_collision(
                        candidate[:3, 3], previous_position=pose[:3, 3]
                    )
                    if collision.has_collision:
                        collision_rejections[collision.collision_type] += 1
                        continue
                key = tuple(
                    np.round(
                        np.concatenate((candidate[:3, 3], candidate[:3, :3].ravel())), 7
                    )
                )
                if key in visited:
                    continue
                visited.add(key)
                success = metric_cache.get(key)
                if success is None:
                    success = bool(canonical_fov(score_observation(item, candidate))["success"])
                    metric_cache[key] = success
                if success:
                    return {
                        "complete": True,
                        "no_success_through_depth": False,
                        "success_depth": depth,
                        "max_depth": max_steps,
                        "states_per_depth": states_per_depth + [len(next_frontier) + 1],
                        "unique_states": len(visited),
                        "collision_rejections": dict(collision_rejections),
                    }
                next_frontier.append(candidate)
        frontier = next_frontier
        states_per_depth.append(len(frontier))
    return {
        "complete": True,
        "no_success_through_depth": True,
        "success_depth": None,
        "max_depth": max_steps,
        "states_per_depth": states_per_depth,
        "unique_states": len(visited),
        "collision_rejections": dict(collision_rejections),
    }


def construct_fov_reachable_initial(
    item: dict[str, Any],
    success_pose: np.ndarray,
    constraints: SceneConstraints | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None, dict[str, Any]] | None:
    """Find a natural partial-FOV initial with an exact inverse-action path.

    Horizontal strafe changes framing without manufacturing a ceiling/floor
    view.  We require both projected objects to remain visible and large enough;
    failure must come from strict full-bbox/center framing.  Every translation
    is checked by the runtime collision detector, including its swept segment.
    """
    sequences: list[list[str]] = []
    for steps in (12, 10, 8, 6, 4, 2):
        sequences.append(["move_forward"] * steps)
        for action in ("move_left", "move_right"):
            sequences.append([action] * steps)
        if steps + 1 <= 12:
            sequences.append(["move_left"] * steps + ["turn_right"])
            sequences.append(["move_right"] * steps + ["turn_left"])
    inverse = {
        "move_left": "move_right",
        "move_right": "move_left",
        "move_forward": "move_backward",
        "turn_left": "turn_right",
        "turn_right": "turn_left",
    }
    detector = None
    if constraints is not None:
        detector = constraints._collision_cache.get(str(item.get("scene_id") or ""))
    metric_cache: dict[tuple[float, ...], bool] = {}
    for reverse_actions in sequences:
        engine = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
        engine.reset(np.asarray(success_pose, dtype=float))
        collision_free = True
        for action in reverse_actions:
            previous = engine.get_pose()
            candidate = engine.step(action)
            if detector is not None and action.startswith("move_"):
                collision = detector.check_collision(
                    candidate[:3, 3], previous_position=previous[:3, 3]
                )
                if collision.has_collision:
                    collision_free = False
                    break
        if not collision_free:
            continue
        initial_pose = engine.get_pose()
        initial_constraints = constraints.validate(
            item,
            initial_pose[:3, 3],
            initial_room_index=constraints.room_index(
                constraints.scene(str(item.get("scene_id") or ""))[0],
                pair_midpoint(item)[:2],
            ),
            check_pair_distance=False,
        ) if constraints else None
        if initial_constraints is not None and not initial_constraints["success"]:
            continue
        metric = canonical_fov(score_observation(item, initial_pose))
        gates = metric["gates"]
        meaningful_failure = (
            not metric["success"]
            and all(bool(gates.get(key)) for key in ("two_objects", "in_front", "visible", "min_area"))
            and (not gates.get("full_bbox_in_frame") or not gates.get("center_margin"))
        )
        if not meaningful_failure:
            continue
        depth_audit = canonical_fov_depth_lower_bound(
            item,
            initial_pose,
            detector,
            MIN_FOV_PLANNER_STEPS - 1,
            metric_cache,
        )
        if not depth_audit["no_success_through_depth"]:
            continue
        certificate_actions = [inverse[action] for action in reversed(reverse_actions)]
        certificate = {
            "construction": "inverse_action_lattice_visible_initial",
            "reverse_construction_actions": reverse_actions,
            "actions": certificate_actions,
            "steps": len(certificate_actions),
            "success_pose_c2w": np.asarray(success_pose, dtype=float).tolist(),
            "success_metric": canonical_fov(score_observation(item, success_pose)),
            "initial_visible_failure": True,
            "minimum_planner_steps_gate": MIN_FOV_PLANNER_STEPS,
            "no_solution_lower_bound": depth_audit,
            "verified_path_upper_bound": len(certificate_actions),
        }
        return initial_pose, metric, initial_constraints, certificate
    return None


def relation_normal(item: dict[str, Any]) -> np.ndarray:
    p = item["target_region"]["params"]
    ab = np.asarray(p["object_b_center"], dtype=float)[:2] - np.asarray(p["object_a_center"], dtype=float)[:2]
    ab /= max(float(np.linalg.norm(ab)), 1e-12)
    return np.array([ab[1], -ab[0]]) if p.get("relation", "left") == "left" else np.array([-ab[1], ab[0]])


DIFFICULTY_BUCKET_EDGES = {
    "translation_m": (1.5, 3.0, 5.0, 7.0),
    "yaw_deg": (30.0, 60.0, 100.0, 140.0),
    "bbox_area_ratio": (0.005, 0.015, 0.04, 0.10),
    "relation_margin_px": (20.0, 40.0, 80.0),
    "planner_step_proxy": (3.0, 6.0, 9.0, 12.0),
}


def _difficulty_bucket(name: str, value: float) -> int:
    return sum(float(value) >= edge for edge in DIFFICULTY_BUCKET_EDGES[name])


def _yaw_delta_degrees(first: np.ndarray, second: np.ndarray) -> float:
    a = horizontal_forward(np.asarray(first, dtype=float))
    b = horizontal_forward(np.asarray(second, dtype=float))
    cosine = float(np.clip(np.dot(a[:2], b[:2]), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def projective_difficulty_profile(
    item: dict[str, Any],
    initial_pose: np.ndarray,
    target_pose: np.ndarray,
    target_result: dict[str, Any],
    target_metric: dict[str, Any],
    planner_steps: int | None = None,
) -> dict[str, Any]:
    translation = float(np.linalg.norm(target_pose[:2, 3] - initial_pose[:2, 3]))
    yaw_degrees = _yaw_delta_degrees(initial_pose[:3, 2], target_pose[:3, 2])
    objects = target_result.get("visual_metrics", {}).get("objects") or []
    bbox_area = min((float(row.get("area_ratio") or 0.0) for row in objects), default=0.0)
    margin = abs(float(target_metric.get("relation_margin_px") or 0.0))
    step_proxy = int(
        planner_steps
        if planner_steps is not None
        else min(12, max(1, math.ceil(translation / 0.3) + math.ceil(yaw_degrees / 20.0)))
    )
    values = {
        "translation_m": translation,
        "yaw_deg": yaw_degrees,
        "bbox_area_ratio": bbox_area,
        "relation_margin_px": margin,
        "planner_step_proxy": step_proxy,
    }
    return {
        **values,
        "buckets": {name: _difficulty_bucket(name, float(value)) for name, value in values.items()},
        "bucket_edges": DIFFICULTY_BUCKET_EDGES,
    }


def _difficulty_rank(source: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, ...]:
    names = (
        "translation_m", "yaw_deg", "bbox_area_ratio",
        "relation_margin_px", "planner_step_proxy",
    )
    bucket_distance = sum(
        abs(int(source["buckets"][name]) - int(candidate["buckets"][name])) for name in names
    )
    # Becoming easier is not forbidden, but receives an explicit extra penalty.
    easier = max(0.0, float(source["planner_step_proxy"]) - float(candidate["planner_step_proxy"]))
    numeric = (
        abs(float(source["translation_m"]) - float(candidate["translation_m"])) / 3.0
        + abs(float(source["yaw_deg"]) - float(candidate["yaw_deg"])) / 60.0
        + abs(float(source["bbox_area_ratio"]) - float(candidate["bbox_area_ratio"])) / 0.03
        + abs(float(source["relation_margin_px"]) - float(candidate["relation_margin_px"])) / 40.0
    )
    return (float(bucket_distance), easier, numeric)


PROJECTIVE_INITIAL_OBSERVABILITY_GATES = (
    "two_objects", "in_front", "visible", "min_area", "inside_frame",
)


def projective_initial_geometry_discernible(metric: dict[str, Any]) -> bool:
    """Cheap pre-render guard for the initial Projective key frame.

    RGB texture remains an official-render decision. This only prevents the
    generator from proposing an initial state where canonical projection says
    an object is absent, too small, or truncated.
    """
    gates = metric.get("gates") or {}
    return all(bool(gates.get(name)) for name in PROJECTIVE_INITIAL_OBSERVABILITY_GATES)


def _rotate_horizontal(direction: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    value = np.asarray(direction, dtype=float)
    return normalize_vector(
        np.array(
            [cosine * value[0] - sine * value[1], sine * value[0] + cosine * value[1], value[2]],
            dtype=float,
        )
    )


def construct_projective_reachable_initial(
    item: dict[str, Any],
    success_pose: np.ndarray,
    constraints: SceneConstraints | None,
    room_index: int | None,
    desired_steps: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Reverse the formal action lattice from a canonical-success state.

    The returned inverse sequence is a constructive upper bound, not a shortest
    path claim.  Every translated intermediate state uses runtime collision plus
    the frozen room/wall convention.
    """
    if constraints is None:
        return None
    detector = constraints._collision_cache.get(str(item.get("scene_id") or ""))
    if detector is None or detector.structure_convention_status != "frozen":
        return None
    desired_steps = max(1, min(12, int(desired_steps)))
    lengths = sorted(range(1, 13), key=lambda value: (abs(value - desired_steps), value))
    inverse = {
        "move_forward": "move_backward", "move_backward": "move_forward",
        "move_left": "move_right", "move_right": "move_left",
        "turn_left": "turn_right", "turn_right": "turn_left",
    }
    candidates: list[tuple[tuple[float, ...], np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for length in lengths:
        sequences = [[action] * length for action in PLANNER_ACTIONS]
        if length >= 2:
            first = length // 2
            second = length - first
            sequences.extend(
                [
                    [move] * first + [turn] * second
                    for move in ("move_left", "move_right", "move_backward")
                    for turn in ("turn_left", "turn_right")
                ]
            )
        for reverse_actions in sequences:
            engine = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
            engine.reset(np.asarray(success_pose, dtype=float))
            legal = True
            for action in reverse_actions:
                previous = engine.get_pose()
                candidate_pose = engine.step(action)
                if action.startswith("move_"):
                    collision = detector.check_collision(
                        candidate_pose[:3, 3], previous_position=previous[:3, 3]
                    )
                    layout_check = constraints.validate(
                        item,
                        candidate_pose[:3, 3],
                        initial_room_index=room_index,
                        check_pair_distance=False,
                    )
                    if collision.has_collision or not layout_check["success"]:
                        legal = False
                        break
            if not legal:
                continue
            initial_pose = engine.get_pose()
            if float(np.linalg.norm(initial_pose[:2, 3] - success_pose[:2, 3])) < 0.5:
                continue
            initial_constraints = constraints.validate(
                item,
                initial_pose[:3, 3],
                initial_room_index=room_index,
                check_pair_distance=False,
            )
            initial_metric = canonical_projective(score_observation(item, initial_pose))
            if (
                not initial_constraints["success"]
                or initial_metric["success"]
                or not projective_initial_geometry_discernible(initial_metric)
                or absolute_pitch_degrees(initial_pose[:3, 2]) > MAX_ABS_PITCH_DEG
            ):
                continue
            certificate_actions = [inverse[action] for action in reversed(reverse_actions)]
            certificate = {
                "construction": "projective_inverse_action_lattice_v1",
                "reverse_construction_actions": reverse_actions,
                "actions": certificate_actions,
                "steps": len(certificate_actions),
                "verified_path_upper_bound": len(certificate_actions),
                "shortest_path_claim": False,
            }
            rank = (abs(length - desired_steps), float(np.linalg.norm(initial_pose[:2, 3] - success_pose[:2, 3])))
            candidates.append((rank, initial_pose, initial_metric, initial_constraints, certificate))
        if candidates and abs(length - desired_steps) >= 1:
            break
    if not candidates:
        return None
    _, pose, metric, layout, certificate = min(candidates, key=lambda row: row[0])
    return pose, metric, layout, certificate


def projective_candidates(
    item: dict[str, Any],
    constraints: SceneConstraints | None = None,
    initial_room_index: int | None = None,
) -> list[np.ndarray]:
    """Start with exact half-plane reflection, then bounded distance/along sweeps."""
    p = item["target_region"]["params"]
    boundary = np.asarray(p["boundary_point"], dtype=float)[:2]
    direction = np.asarray(p["boundary_direction"], dtype=float)[:2]
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    old = np.asarray(item["sample_target"], dtype=float)
    along = float(np.dot(old[:2] - boundary, direction))
    old_normal = np.asarray(p["normal"], dtype=float)[:2]
    old_normal /= max(float(np.linalg.norm(old_normal)), 1e-12)
    distance = abs(float(np.dot(old[:2] - boundary, old_normal)))
    distance = max(distance, float(p.get("min_distance", 2.0) or 2.0), 0.75)
    canonical = relation_normal(item)
    out = []
    # First candidate retains original difficulty exactly except the corrected side.
    # Nearer orthogonal views increase image-space separation.  All candidates
    # are subsequently checked by the full canonical projection gates.
    for scale in (1.0, 0.85, 1.15, 0.70, 1.30, 1.55, 1.85, 2.20, 0.55, 0.40, 0.30, 0.20, 0.15):
        for delta in (0.0, -0.25, 0.25, -0.75, 0.75, -1.5, 1.5, -2.5, 2.5, -4.0, 4.0):
            xy = boundary + (along + delta) * direction + distance * scale * canonical
            out.append(np.array([xy[0], xy[1], old[2]], dtype=float))
    if constraints is not None and constraints.available:
        out.extend(constraints.layout_candidates(item, initial_room_index))
    deduplicated = []
    seen = set()
    for point in out:
        key = tuple(round(float(value), 6) for value in point)
        if key not in seen:
            seen.add(key)
            deduplicated.append(point)
    return deduplicated


def repair_projective(
    source_index: int,
    item: dict[str, Any],
    constraints: SceneConstraints | None = None,
    selection_offset: int = 0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    old_pose = pose_from_item_target(item)
    old_result = score_observation(item, old_pose)
    old_metric = canonical_projective(old_result)
    old_initial_pose = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
    old_initial_position = old_initial_pose[:3, 3]
    old_initial_metric = canonical_projective(score_observation(item, old_initial_pose))
    source_difficulty = projective_difficulty_profile(
        item, old_initial_pose, old_pose, old_result, old_metric
    )
    initial_constraints = (
        constraints.validate(item, old_initial_position, check_pair_distance=False)
        if constraints else None
    )
    desired_room_index = initial_constraints.get("room_index") if initial_constraints else None
    if constraints is not None and constraints.available:
        layout, _ = constraints.scene(str(item.get("scene_id") or ""))
        desired_room_index = constraints.room_index(layout, pair_midpoint(item)[:2])
    fallback_initial = None
    fallback_rank: tuple[float, ...] | None = None
    initial_states_evaluated = 0
    initial_geometry_prefilter_passes = 0
    if constraints is None:
        fallback_initial = (old_initial_pose, old_initial_metric, initial_constraints)
    else:
        old_check = constraints.validate(
            item,
            old_initial_position,
            initial_room_index=desired_room_index,
            check_pair_distance=False,
        )
        if (
            old_check["success"]
            and not old_initial_metric["success"]
            and projective_initial_geometry_discernible(old_initial_metric)
            and absolute_pitch_degrees(old_initial_pose[:3, 2]) <= MAX_ABS_PITCH_DEG
        ):
            fallback_initial = (old_initial_pose, old_initial_metric, old_check)
            fallback_rank = (0.0, 0.0, 0.0)
        initial_points = [] if selection_offset == 0 and fallback_initial is not None else (
            [old_initial_position] if old_check["success"] else []
        )
        initial_points.extend(
            constraints.safe_layout_candidates(
                item,
                desired_room_index,
                old_initial_position,
                limit=256 if selection_offset else 512,
                check_pair_distance=False,
            ) if not (selection_offset == 0 and fallback_initial is not None) else []
        )
        initial_points = initial_points[:PROJECTIVE_INITIAL_POINT_BUDGET]
        for initial_point in initial_points:
            check = constraints.validate(
                item,
                initial_point,
                initial_room_index=desired_room_index,
                check_pair_distance=False,
            )
            to_pair = horizontal_forward(pair_midpoint(item) - initial_point)
            if selection_offset:
                directions = (
                    to_pair, _rotate_horizontal(to_pair, -10.0),
                    _rotate_horizontal(to_pair, 10.0),
                    horizontal_forward(old_initial_pose[:3, 2]),
                    np.array([-to_pair[1], to_pair[0], 0.0], dtype=float),
                    np.array([to_pair[1], -to_pair[0], 0.0], dtype=float), -to_pair,
                )
            else:
                directions = (
                    horizontal_forward(old_initial_pose[:3, 2]), to_pair,
                    np.array([-to_pair[1], to_pair[0], 0.0], dtype=float),
                    np.array([to_pair[1], -to_pair[0], 0.0], dtype=float), -to_pair,
                )
            for direction in directions:
                candidate_pose = camera_pose_from_forward(initial_point, normalize_vector(direction))
                candidate_metric = canonical_projective(score_observation(item, candidate_pose))
                initial_states_evaluated += 1
                if candidate_metric["success"]:
                    continue
                meaningful = projective_initial_geometry_discernible(candidate_metric)
                if meaningful:
                    initial_geometry_prefilter_passes += 1
                if selection_offset == 0:
                    if meaningful:
                        fallback_initial = (candidate_pose, candidate_metric, check)
                        break
                    continue
                gates = candidate_metric["gates"]
                gate_count = sum(
                    bool(gates.get(name)) for name in PROJECTIVE_INITIAL_OBSERVABILITY_GATES
                )
                rank = (
                    0.0 if meaningful else 1.0,
                    -float(gate_count),
                    float(np.linalg.norm(initial_point[:2] - old_initial_position[:2])),
                )
                if fallback_rank is None or rank < fallback_rank:
                    fallback_rank = rank
                    fallback_initial = (candidate_pose, candidate_metric, check)
            if selection_offset == 0 and fallback_initial is not None:
                break
    attempted: list[dict[str, Any]] = []
    failure_gates: Counter[str] = Counter()
    layout_valid_positions = 0
    canonical_success_candidates = 0
    viable: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    all_points = projective_candidates(item, constraints, desired_room_index)
    points = all_points[:PROJECTIVE_TARGET_POINT_BUDGET]
    for point_index, point in enumerate(points, start=1):
        target_constraints = (
            constraints.validate(
                item, point, initial_room_index=desired_room_index, check_pair_distance=True
            )
            if constraints else None
        )
        if target_constraints is not None and target_constraints["success"]:
            layout_valid_positions += 1
        base_forward = normalize_vector(pair_midpoint(item) - point)
        # Position and orientation are searched jointly.  Small yaw offsets can
        # preserve a source-like framing while retaining the frozen 12 px gate.
        for yaw_offset in (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0):
            forward = _rotate_horizontal(base_forward, yaw_offset)
            pose = camera_pose_from_forward(point, forward)
            repaired = copy.deepcopy(item)
            p = repaired["target_region"]["params"]
            p["normal"] = relation_normal(repaired).tolist()
            p["sample_distance"] = float(np.linalg.norm(point[:2] - np.asarray(p["boundary_point"], dtype=float)[:2]))
            repaired["distance"] = p["sample_distance"]
            repaired["target_region"]["sample_point"] = point.tolist()
            repaired["target_region"]["sample_forward"] = forward.tolist()
            repaired["sample_target"] = point.tolist()
            repaired["camera_params"] = dict(repaired.get("camera_params") or {})
            repaired["camera_params"]["forward"] = forward.tolist()
            repaired["task_id"] = f"projective_canonical_h1_v5_{source_index:06d}"
            repaired["camera_model_version"] = CANONICAL_CAMERA_H1_RESIZE_V1
            repaired["canonical_task_metric_version"] = CANONICAL_TASK_METRIC_VERSION
            repaired["generator_version"] = PROJECTIVE_GENERATOR_VERSION
            repaired["repair_lineage"] = {"source_row_index": source_index, "old_task_id": item.get("task_id"), "repair_reason": "legacy_projective_target_pose_wrong_side"}
            result = score_observation(repaired, pose)
            metric = canonical_projective(result)
            target_pitch = absolute_pitch_degrees(forward)
            for name, passed in metric["gates"].items():
                if not passed:
                    failure_gates[name] += 1
            target_success = bool(
                metric["success"]
                and target_pitch <= MAX_ABS_PITCH_DEG
                and (target_constraints is None or bool(target_constraints["success"]))
            )
            if len(attempted) < 512:
                attempted.append(
                    {
                        "point_index": point_index,
                        "point": point.tolist(),
                        "yaw_offset_deg": yaw_offset,
                        "target_success": target_success,
                        "gates": metric["gates"],
                        "margin_px": metric["relation_margin_px"],
                        "target_pitch_degrees": target_pitch,
                        "target_constraints": target_constraints,
                    }
                )
            if not target_success:
                continue
            canonical_success_candidates += 1
            # Reverse construction is the expensive part.  Four distinct
            # canonical-success states provide bounded coverage; remaining
            # states reuse the validated source-like initial and are still
            # checked by the tiered planner downstream.
            construction = (
                construct_projective_reachable_initial(
                    repaired,
                    pose,
                    constraints,
                    desired_room_index,
                    int(source_difficulty["planner_step_proxy"]),
                )
                if canonical_success_candidates <= 4
                else None
            )
            if construction is not None:
                initial_pose, initial_metric, candidate_initial_constraints, certificate = construction
            elif fallback_initial is not None:
                initial_pose, initial_metric, candidate_initial_constraints = fallback_initial
                certificate = None
            else:
                continue
            navigation_distance = float(np.linalg.norm(point[:2] - initial_pose[:2, 3]))
            initial_valid = bool(
                not initial_metric["success"]
                and projective_initial_geometry_discernible(initial_metric)
                and absolute_pitch_degrees(initial_pose[:3, 2]) <= MAX_ABS_PITCH_DEG
                and navigation_distance >= 0.5
                and (
                    candidate_initial_constraints is None
                    or candidate_initial_constraints["success"]
                )
            )
            if not initial_valid:
                continue
            repaired["init_camera"] = dict(repaired["init_camera"])
            repaired["init_camera"]["extrinsics"] = initial_pose.tolist()
            if certificate is not None:
                repaired["reachability_construction"] = certificate
            difficulty = projective_difficulty_profile(
                repaired,
                initial_pose,
                pose,
                result,
                metric,
                planner_steps=certificate.get("steps") if certificate else None,
            )
            rank = _difficulty_rank(source_difficulty, difficulty) + (
                abs(yaw_offset), float(point_index)
            )
            collision_convention = (
                target_constraints.get("collision_convention")
                if target_constraints is not None
                else None
            )
            if collision_convention is not None:
                repaired["collision_convention"] = collision_convention
            details = {
                "source_row_index": source_index, "old_task_id": item.get("task_id"), "new_task_id": repaired["task_id"],
                "scene_id": item.get("scene_id"), "relation": p.get("relation"), "old_sample_target": item.get("sample_target"),
                "new_sample_target": repaired["sample_target"], "old_normal": item["target_region"]["params"].get("normal"),
                "new_normal": p.get("normal"), "camera_model_version": CANONICAL_CAMERA_H1_RESIZE_V1,
                "canonical_task_metric_version": CANONICAL_TASK_METRIC_VERSION, "generator_version": PROJECTIVE_GENERATOR_VERSION,
                "old_metric": old_metric, "new_metric": metric,
                "initial_constraints": candidate_initial_constraints, "target_constraints": target_constraints,
                "old_result": old_result, "new_result": result, "old_initial_metric": old_initial_metric,
                "new_initial_metric": initial_metric, "old_initial_pose_c2w": old_initial_pose.tolist(),
                "new_initial_pose_c2w": initial_pose.tolist(),
                "initial_pose_repaired": not np.allclose(initial_pose, old_initial_pose),
                "initial_position_repaired": not np.allclose(initial_pose[:3, 3], old_initial_position),
                "initial_orientation_repaired": not np.allclose(initial_pose[:3, :3], old_initial_pose[:3, :3]),
                "initial_pitch_degrees": absolute_pitch_degrees(initial_pose[:3, 2]),
                "target_pitch_degrees": target_pitch,
                "navigation_distance": navigation_distance,
                "collision_convention": collision_convention,
                "reachability_construction": certificate,
                "source_difficulty": source_difficulty,
                "repaired_difficulty": difficulty,
                "difficulty_rank": rank,
                "selected_yaw_offset_deg": yaw_offset,
            }
            viable.append((rank, repaired, details))
            if len(viable) >= 96:
                break
        if len(viable) >= 96:
            break
    if viable:
        ordered_viable = sorted(viable, key=lambda row: row[0])
        selected_offset = min(max(0, int(selection_offset)), len(ordered_viable) - 1)
        _, repaired, details = ordered_viable[selected_offset]
        details["viable_selection_offset"] = selected_offset
        details["retry"] = {
            "joint_states_evaluated": sum(failure_gates.values()) + canonical_success_candidates,
            "recorded_attempts": len(attempted),
            "layout_valid_positions": layout_valid_positions,
            "canonical_success_candidates": canonical_success_candidates,
            "viable_candidates": len(viable),
            "failure_gate_counts": dict(failure_gates),
            "attempts": attempted,
            "cache_stats": dict(constraints.cache_stats) if constraints else {},
            "search_budget": {
                "initial_states_evaluated": initial_states_evaluated,
                "initial_geometry_prefilter_passes": initial_geometry_prefilter_passes,
                "initial_points": PROJECTIVE_INITIAL_POINT_BUDGET,
                "target_points": PROJECTIVE_TARGET_POINT_BUDGET,
                "all_target_points_generated": len(all_points),
                "target_points_evaluated": len(points),
                "budget_exhausted": len(all_points) > len(points),
            },
        }
        return repaired, details
    metric_ready = [
        attempt for attempt in attempted
        if all(bool(attempt["gates"].get(key)) for key in ("two_objects", "in_front", "visible", "min_area", "inside_frame", "relation"))
    ]
    if not points or layout_valid_positions == 0:
        failure = "layout_collision_infeasible"
        taxonomy = "layout/collision infeasible"
    elif metric_ready and all(not bool(attempt["gates"].get("margin")) for attempt in metric_ready):
        failure = "projective_margin_infeasible_for_source_pair"
        taxonomy = "source object pair semantically infeasible"
    elif canonical_success_candidates:
        failure = "no_safe_reachable_unsuccessful_initial"
        taxonomy = "12-step unreachable"
    else:
        failure = "joint_position_orientation_search_exhausted"
        taxonomy = "search insufficient"
    return None, {
        "source_row_index": source_index,
        "old_task_id": item.get("task_id"),
        "scene_id": item.get("scene_id"),
        "relation": item["target_region"]["params"].get("relation"),
        "failure": failure,
        "failure_taxonomy": taxonomy,
        "initial_constraints": initial_constraints,
        "source_difficulty": source_difficulty,
        "search_stats": {
            "positions": len(points),
            "layout_valid_positions": layout_valid_positions,
            "canonical_success_candidates": canonical_success_candidates,
            "failure_gate_counts": dict(failure_gates),
            "cache_stats": dict(constraints.cache_stats) if constraints else {},
            "search_budget": {
                "initial_states_evaluated": initial_states_evaluated,
                "initial_geometry_prefilter_passes": initial_geometry_prefilter_passes,
                "initial_points": PROJECTIVE_INITIAL_POINT_BUDGET,
                "target_points": PROJECTIVE_TARGET_POINT_BUDGET,
                "all_target_points_generated": len(all_points),
                "target_points_evaluated": len(points),
                "budget_exhausted": len(all_points) > len(points),
            },
        },
        "attempts": attempted,
    }


def fov_target_candidates(
    item: dict[str, Any],
    constraints: SceneConstraints | None = None,
    room_index: int | None = None,
) -> list[np.ndarray]:
    old = np.asarray(item["sample_target"], dtype=float)
    p = item["target_region"]["params"]
    midpoint = pair_midpoint(item)
    radial = old[:2] - midpoint[:2]
    radial /= max(float(np.linalg.norm(radial)), 1e-12)
    tangent = np.array([-radial[1], radial[0]])
    out = []
    for scale in (1.0, 1.15, 1.30, 1.50, 1.80, 2.10):
        for side in (0.0, -0.25, 0.25, -0.5, 0.5):
            xy = midpoint[:2] + radial * float(np.linalg.norm(old[:2] - midpoint[:2])) * scale + tangent * side
            out.append(np.array([xy[0], xy[1], old[2]], dtype=float))
    if constraints is not None and constraints.available:
        out.extend(constraints.safe_layout_candidates(item, room_index, old))
    deduplicated = []
    seen = set()
    for point in out:
        key = tuple(round(float(value), 6) for value in point)
        if key not in seen:
            seen.add(key)
            deduplicated.append(point)
    return deduplicated


def repair_fov(
    source_index: int,
    item: dict[str, Any],
    constraints: SceneConstraints | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    old_init = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
    old_target = pose_from_item_target(item)
    old_init_metric = canonical_fov(score_observation(item, old_init))
    old_target_metric = canonical_fov(score_observation(item, old_target))
    old_initial_constraints = constraints.validate(item, old_init[:3, 3], check_pair_distance=False) if constraints else None
    desired_room_index = None
    if constraints is not None and constraints.available:
        layout, _ = constraints.scene(str(item.get("scene_id") or ""))
        desired_room_index = constraints.room_index(layout, pair_midpoint(item)[:2])
        old_initial_constraints = constraints.validate(
            item,
            old_init[:3, 3],
            initial_room_index=desired_room_index,
            check_pair_distance=False,
        )
        initial_points = [old_init[:3, 3]] if old_initial_constraints["success"] else []
        initial_points.extend(
            constraints.safe_layout_candidates(
                item,
                desired_room_index,
                old_init[:3, 3],
                limit=128,
                check_pair_distance=False,
            )
        )
    else:
        initial_points = [old_init[:3, 3]]
    if not initial_points:
        return None, {"source_row_index": source_index, "old_task_id": item.get("task_id"), "scene_id": item.get("scene_id"), "failure": "no_safe_initial_pose", "old_initial_constraints": old_initial_constraints, "attempts": []}
    attempts = []
    for attempt, target_point in enumerate(fov_target_candidates(item, constraints, desired_room_index), start=1):
        repaired = copy.deepcopy(item)
        forward = normalize_vector(pair_midpoint(repaired) - target_point)
        target_pose = camera_pose_from_forward(target_point, forward)
        repaired["sample_target"] = target_point.tolist()
        repaired["camera_params"] = dict(repaired.get("camera_params") or {})
        repaired["camera_params"]["forward"] = forward.tolist()
        repaired["target_region"]["sample_point"] = target_point.tolist()
        repaired["target_region"]["sample_forward"] = forward.tolist()
        repaired["target_region"]["params"]["sample_distance"] = float(
            np.linalg.norm(target_point[:2] - pair_midpoint(repaired)[:2])
        )
        repaired["distance"] = repaired["target_region"]["params"]["sample_distance"]
        repaired["task_id"] = f"fov_canonical_h1_v2_{source_index:06d}"
        repaired["camera_model_version"] = CANONICAL_CAMERA_H1_RESIZE_V1
        repaired["canonical_task_metric_version"] = CANONICAL_TASK_METRIC_VERSION
        repaired["generator_version"] = FOV_GENERATOR_VERSION
        repaired["repair_lineage"] = {"source_row_index": source_index, "old_task_id": item.get("task_id"), "repair_reason": "legacy_fov_initial_pose_and_metric_degeneracy"}
        target_result = score_observation(repaired, target_pose)
        target_metric = canonical_fov(target_result)
        target_constraints = constraints.validate(repaired, target_point, initial_room_index=desired_room_index, check_pair_distance=True) if constraints else None
        target_pitch = absolute_pitch_degrees(forward)
        target_ready = (
            bool(target_metric["success"])
            and target_pitch <= MAX_ABS_PITCH_DEG
            and (target_constraints is None or bool(target_constraints["success"]))
        )
        if not target_ready:
            attempts.append({"attempt": attempt, "success": False, "target_metric": target_metric, "initial_metric": None, "target_pitch_degrees": target_pitch, "initial_pitch_degrees": None, "target_constraints": target_constraints, "initial_constraints": None, "reachability_certificate": None})
            continue
        chosen_initial = None
        init_metric = None
        init_pose = None
        initial_constraints = None
        reachability_certificate = None
        construction = construct_fov_reachable_initial(
            repaired, target_pose, constraints
        )
        if construction is not None:
            init_pose, init_metric, initial_constraints, reachability_certificate = construction
            chosen_initial = init_pose[:3, 3]
        initial_pitch = absolute_pitch_degrees(init_pose[:3, 2]) if init_pose is not None else None
        generation_success = bool(target_metric["success"]) and target_pitch <= MAX_ABS_PITCH_DEG and init_metric is not None and initial_pitch <= MAX_ABS_PITCH_DEG and (target_constraints is None or bool(target_constraints["success"]))
        attempts.append({"attempt": attempt, "success": generation_success, "target_metric": target_metric, "initial_metric": init_metric, "target_pitch_degrees": target_pitch, "initial_pitch_degrees": initial_pitch, "target_constraints": target_constraints, "initial_constraints": initial_constraints, "reachability_certificate": reachability_certificate})
        if generation_success:
            repaired["init_camera"] = dict(repaired["init_camera"])
            repaired["init_camera"]["extrinsics"] = init_pose.tolist()
            repaired["reachability_construction"] = reachability_certificate
            collision_convention = (
                target_constraints.get("collision_convention")
                if target_constraints is not None
                else None
            )
            if collision_convention is not None:
                repaired["collision_convention"] = collision_convention
            return repaired, {"source_row_index": source_index, "old_task_id": item.get("task_id"), "new_task_id": repaired["task_id"], "scene_id": item.get("scene_id"), "camera_model_version": CANONICAL_CAMERA_H1_RESIZE_V1, "canonical_task_metric_version": CANONICAL_TASK_METRIC_VERSION, "generator_version": FOV_GENERATOR_VERSION, "old_init_metric": old_init_metric, "old_target_metric": old_target_metric, "new_init_metric": init_metric, "new_target_metric": target_metric, "old_initial_constraints": old_initial_constraints, "initial_constraints": initial_constraints, "target_constraints": target_constraints, "old_initial_pose_c2w": old_init.tolist(), "new_initial_pose_c2w": init_pose.tolist(), "old_target_pose_c2w": old_target.tolist(), "new_target_pose_c2w": target_pose.tolist(), "initial_position_repaired": not np.allclose(chosen_initial, old_init[:3, 3]), "initial_pitch_degrees": initial_pitch, "target_pitch_degrees": target_pitch, "reachability_construction": reachability_certificate, "collision_convention": collision_convention, "retry": {"attempt_count": attempt, "attempts": attempts}}
    return None, {"source_row_index": source_index, "old_task_id": item.get("task_id"), "scene_id": item.get("scene_id"), "failure": "bounded_layout_candidates_exhausted", "old_initial_constraints": old_initial_constraints, "attempts": attempts}


def summarize(values: list[float]) -> dict[str, float | None]:
    return {"min": min(values) if values else None, "max": max(values) if values else None, "mean": statistics.mean(values) if values else None, "median": statistics.median(values) if values else None}


def run_pilot(
    kind: str,
    source: Path,
    out: Path,
    count: int,
    seed: int,
    scenes: set[str] | None = None,
    constraints: SceneConstraints | None = None,
) -> dict[str, Any]:
    rows = read_rows(source)
    all_matching = [(i, row) for i, row in enumerate(rows) if row.get("task_type") == kind]
    selected = [
        (i, row)
        for i, row in all_matching
        if not scenes or str(row.get("scene_id")) in scenes
    ]
    rng = random.Random(seed)
    rng.shuffle(selected)
    valid, mappings, failures = [], [], []
    fn = repair_fov if kind == "fov_inclusion" else repair_projective
    for index, row in selected:
        repaired, details = fn(index, row, constraints)
        if repaired is None:
            failures.append(details)
        else:
            valid.append(repaired)
            mappings.append(details)
        if len(valid) >= count:
            break
    out.mkdir(parents=True, exist_ok=True)
    stem = "fov" if kind == "fov_inclusion" else "projective"
    version_tag = "h1_v2_layout_gated" if kind == "fov_inclusion" else "h1_v4_layout_gated"
    pilot_path = out / f"r1_{stem}_repair_pilot_{version_tag}.jsonl"
    mapping_path = out / f"r1_{stem}_repair_pilot_mapping_{version_tag}.jsonl"
    failures_path = out / f"r1_{stem}_repair_pilot_failures_{version_tag}.jsonl"
    write_jsonl(pilot_path, valid)
    write_jsonl(mapping_path, mappings)
    write_jsonl(failures_path, failures)
    return {
        "kind": kind,
        "requested": count,
        "source_matching": len(all_matching),
        "selected_scene_matching": len(selected),
        "scene_filter": sorted(scenes) if scenes else None,
        "layout_validation": "required" if constraints and constraints.available else "unavailable_projection_only",
        "accepted": len(valid),
        "failures": len(failures),
        "input_sha256": sha256(source),
        "scene_distribution": dict(Counter(x.get("scene_id") for x in valid)),
        "artifacts": {
            "pilot": str(pilot_path),
            "mapping": str(mapping_path),
            "failures": str(failures_path),
        },
    }


def regenerate_projective(
    source: Path,
    output: Path,
    mapping_path: Path,
    failure_path: Path,
    constraints: SceneConstraints,
) -> dict[str, Any]:
    rows = read_rows(source)
    output_rows, mappings, failures = [], [], []
    for index, row in enumerate(rows):
        if row.get("task_type") != "projective_relations":
            output_rows.append(row)
            continue
        repaired, details = repair_projective(index, row, constraints)
        if repaired is None:
            # Preserve cardinality and source data while declaring an unresolved row.
            retained = copy.deepcopy(row)
            retained["r1_projective_repair_status"] = "unresolved_source_retained"
            retained["r1_projective_failure_manifest_ref"] = failure_path.name
            output_rows.append(retained)
            failures.append(details)
        else:
            output_rows.append(repaired)
            mappings.append(details)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, output_rows)
    write_jsonl(mapping_path, mappings)
    write_jsonl(failure_path, failures)
    source_projective = sum(row.get("task_type") == "projective_relations" for row in rows)
    output_projective = sum(row.get("task_type") == "projective_relations" for row in output_rows)
    return {"source": str(source), "source_sha256": sha256(source), "output": str(output), "output_sha256": sha256(output), "source_rows": len(rows), "output_rows": len(output_rows), "source_projective": source_projective, "output_projective": output_projective, "repaired": len(mappings), "failures_retained": len(failures), "generator_version": PROJECTIVE_GENERATOR_VERSION, "layout_validation": "required", "gs_root": str(constraints.gs_root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--kind", choices=("fov_inclusion", "projective_relations"), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260904)
    p.add_argument("--scenes", default=None, help="optional comma-separated scene IDs")
    p.add_argument("--gs-root", type=Path, default=None, help="scene root for layout/collision gates")
    p.add_argument("--min-wall-clearance", type=float, default=0.5)
    p = sub.add_parser("regenerate-projective")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--failures", type=Path, required=True)
    p.add_argument("--gs-root", type=Path, required=True, help="required scene root for layout/collision gates")
    p.add_argument("--min-wall-clearance", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "pilot":
        scenes = {value for value in (args.scenes or "").split(",") if value} or None
        constraints = SceneConstraints(args.gs_root, min_wall_clearance=args.min_wall_clearance) if args.gs_root else None
        result = run_pilot(args.kind, args.source, args.output_dir, args.count, args.seed, scenes, constraints)
    else:
        constraints = SceneConstraints(args.gs_root, min_wall_clearance=args.min_wall_clearance)
        result = regenerate_projective(args.source, args.output, args.mapping, args.failures, constraints)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
