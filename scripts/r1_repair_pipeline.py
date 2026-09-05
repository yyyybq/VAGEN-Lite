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
from data_gen.active_spatial_pipeline.layout_quality import LayoutGeometry, point_in_poly

PROJECTIVE_GENERATOR_VERSION = "projective_canonical_h1_v4_layout_gated"
FOV_GENERATOR_VERSION = "fov_canonical_h1_v2_layout_gated"


class SceneConstraints:
    """Load and enforce the existing indoor, wall, and object-collision gates."""

    def __init__(self, gs_root: Path | None, min_wall_clearance: float = 0.5, object_margin: float = 0.2):
        self.gs_root = gs_root
        self.min_wall_clearance = float(min_wall_clearance)
        self.object_margin = float(object_margin)
        self._cache: dict[str, tuple[LayoutGeometry, list[tuple[np.ndarray, np.ndarray, str]]]] = {}

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
        layout = LayoutGeometry.from_scene_path(scene_path)
        objects: list[tuple[np.ndarray, np.ndarray, str]] = []
        labels_path = scene_path / "labels.json"
        if labels_path.exists():
            for entry in json.loads(labels_path.read_text()):
                points = entry.get("bounding_box") or []
                if not points:
                    continue
                vertices = np.asarray([[point["x"], point["y"], point["z"]] for point in points], dtype=float)
                objects.append((vertices.min(axis=0), vertices.max(axis=0), str(entry.get("label") or entry.get("ins_id") or "object")))
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
        scene_available = bool(layout.room_polys)
        xy = np.asarray(point, dtype=float)[:2]
        room_index = self.room_index(layout, xy) if scene_available else None
        wall_distance = float(layout.wall_distance(xy)) if scene_available else None
        collision_label = None
        if scene_available:
            p3 = np.asarray(point, dtype=float)[:3]
            for bbox_min, bbox_max, label in objects:
                if np.all(p3 >= bbox_min - self.object_margin) and np.all(p3 <= bbox_max + self.object_margin):
                    collision_label = label
                    break
        params = item.get("target_region", {}).get("params", {})
        centers = [np.asarray(params[key], dtype=float)[:2] for key in ("object_a_center", "object_b_center") if params.get(key) is not None]
        pair_distance = min((float(np.linalg.norm(xy - center)) for center in centers), default=float("inf"))
        required_pair_distance = float(params.get("min_distance", 0.0) or 0.0)
        gates = {
            "scene_layout_available": scene_available,
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
        }

    def layout_candidates(self, item: dict[str, Any], initial_room_index: int | None, limit: int = 512) -> list[np.ndarray]:
        layout, _ = self.scene(str(item.get("scene_id") or ""))
        if not layout.room_polys:
            return []
        params = item["target_region"]["params"]
        boundary = np.asarray(params["boundary_point"], dtype=float)[:2]
        normal = relation_normal(item)
        old = np.asarray(item["sample_target"], dtype=float)
        delta = old[:2] - boundary
        mirrored = boundary + delta - 2.0 * float(np.dot(delta, normal)) * normal
        ranked: list[tuple[float, np.ndarray]] = []
        for xy in layout.candidate_points(grid_spacing=0.25):
            if float(np.dot(xy - boundary, normal)) <= 0.0:
                continue
            point = np.array([xy[0], xy[1], old[2]], dtype=float)
            check = self.validate(item, point, initial_room_index=initial_room_index, check_pair_distance=True)
            if check["success"]:
                ranked.append((float(np.linalg.norm(xy - mirrored)), point))
        ranked.sort(key=lambda row: row[0])
        return [point for _, point in ranked[:limit]]

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
        ranked: list[tuple[float, np.ndarray]] = []
        for xy in layout.candidate_points(grid_spacing=0.25):
            point = np.array([xy[0], xy[1], height], dtype=float)
            check = self.validate(
                item,
                point,
                initial_room_index=room_index,
                check_pair_distance=check_pair_distance,
            )
            if check["success"]:
                ranked.append((float(np.linalg.norm(xy - np.asarray(reference, dtype=float)[:2])), point))
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


def relation_normal(item: dict[str, Any]) -> np.ndarray:
    p = item["target_region"]["params"]
    ab = np.asarray(p["object_b_center"], dtype=float)[:2] - np.asarray(p["object_a_center"], dtype=float)[:2]
    ab /= max(float(np.linalg.norm(ab)), 1e-12)
    return np.array([ab[1], -ab[0]]) if p.get("relation", "left") == "left" else np.array([-ab[1], ab[0]])


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
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    old_pose = pose_from_item_target(item)
    old_result = score_observation(item, old_pose)
    old_metric = canonical_projective(old_result)
    old_initial_pose = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
    old_initial_position = old_initial_pose[:3, 3]
    old_initial_metric = canonical_projective(score_observation(item, old_initial_pose))
    initial_pose = old_initial_pose
    initial_metric = old_initial_metric
    initial_constraints = constraints.validate(item, old_initial_position, check_pair_distance=False) if constraints else None
    desired_room_index = initial_constraints.get("room_index") if initial_constraints else None
    initial_pose_repaired = False
    initial_position_repaired = False
    initial_orientation_repaired = False
    if constraints is not None and constraints.available:
        layout, _ = constraints.scene(str(item.get("scene_id") or ""))
        desired_room_index = constraints.room_index(layout, pair_midpoint(item)[:2])
        initial_constraints = constraints.validate(
            item,
            old_initial_position,
            initial_room_index=desired_room_index,
            check_pair_distance=False,
        )
        if not initial_constraints["success"] or initial_metric["success"]:
            replacement = None
            initial_points = []
            if initial_constraints["success"]:
                initial_points.append(old_initial_position)
            initial_points.extend(
                constraints.safe_layout_candidates(
                    item,
                    desired_room_index,
                    old_initial_position,
                    limit=512,
                    check_pair_distance=False,
                )
            )
            for point in initial_points:
                point_constraints = constraints.validate(
                    item,
                    point,
                    initial_room_index=desired_room_index,
                    check_pair_distance=False,
                )
                if not point_constraints["success"]:
                    continue
                to_pair = normalize_vector(pair_midpoint(item) - point)
                directions = [
                    old_initial_pose[:3, 2] if np.allclose(point, old_initial_position) else to_pair,
                    to_pair,
                    np.array([-to_pair[1], to_pair[0], to_pair[2]], dtype=float),
                    np.array([to_pair[1], -to_pair[0], to_pair[2]], dtype=float),
                    -to_pair,
                ]
                for direction in directions:
                    pose = camera_pose_from_forward(point, normalize_vector(direction))
                    metric = canonical_projective(score_observation(item, pose))
                    if not metric["success"]:
                        replacement = (pose, metric, point_constraints)
                        break
                if replacement is not None:
                    break
            if replacement is None:
                return None, {
                    "source_row_index": source_index,
                    "old_task_id": item.get("task_id"),
                    "scene_id": item.get("scene_id"),
                    "relation": item["target_region"]["params"].get("relation"),
                    "failure": "no_safe_unsuccessful_initial_pose",
                    "old_initial_metric": old_initial_metric,
                    "old_initial_constraints": initial_constraints,
                    "attempts": [],
                }
            initial_pose, initial_metric, initial_constraints = replacement
            initial_pose_repaired = not np.allclose(initial_pose, old_initial_pose)
            initial_position_repaired = not np.allclose(initial_pose[:3, 3], old_initial_position)
            initial_orientation_repaired = not np.allclose(initial_pose[:3, :3], old_initial_pose[:3, :3])
    attempted: list[dict[str, Any]] = []
    for attempt, point in enumerate(projective_candidates(item, constraints, desired_room_index), start=1):
        repaired = copy.deepcopy(item)
        forward = normalize_vector(pair_midpoint(repaired) - point)
        pose = camera_pose_from_forward(point, forward)
        p = repaired["target_region"]["params"]
        p["normal"] = relation_normal(repaired).tolist()
        p["sample_distance"] = float(np.linalg.norm(point[:2] - np.asarray(p["boundary_point"], dtype=float)[:2]))
        repaired["target_region"]["sample_point"] = point.tolist()
        repaired["target_region"]["sample_forward"] = forward.tolist()
        repaired["sample_target"] = point.tolist()
        repaired["camera_params"] = dict(repaired.get("camera_params") or {})
        repaired["camera_params"]["forward"] = forward.tolist()
        repaired["task_id"] = f"projective_canonical_h1_v4_{source_index:06d}"
        repaired["camera_model_version"] = CANONICAL_CAMERA_H1_RESIZE_V1
        repaired["canonical_task_metric_version"] = CANONICAL_TASK_METRIC_VERSION
        repaired["generator_version"] = PROJECTIVE_GENERATOR_VERSION
        repaired["repair_lineage"] = {"source_row_index": source_index, "old_task_id": item.get("task_id"), "repair_reason": "legacy_projective_target_pose_wrong_side"}
        if initial_pose_repaired:
            repaired["init_camera"] = dict(repaired["init_camera"])
            repaired["init_camera"]["extrinsics"] = initial_pose.tolist()
        result = score_observation(repaired, pose)
        metric = canonical_projective(result)
        target_constraints = constraints.validate(repaired, point, initial_room_index=desired_room_index, check_pair_distance=True) if constraints else None
        navigation_distance = float(np.linalg.norm(point[:2] - initial_pose[:2, 3]))
        generation_success = bool(metric["success"]) and navigation_distance >= 0.5 and (target_constraints is None or bool(target_constraints["success"]))
        attempted.append({"attempt": attempt, "point": point.tolist(), "success": generation_success, "gates": metric["gates"], "margin_px": metric["relation_margin_px"], "target_constraints": target_constraints})
        if generation_success:
            return repaired, {
                "source_row_index": source_index, "old_task_id": item.get("task_id"), "new_task_id": repaired["task_id"],
                "scene_id": item.get("scene_id"), "relation": p.get("relation"), "old_sample_target": item.get("sample_target"),
                "new_sample_target": repaired["sample_target"], "old_normal": item["target_region"]["params"].get("normal"),
                "new_normal": p.get("normal"), "camera_model_version": CANONICAL_CAMERA_H1_RESIZE_V1,
                "canonical_task_metric_version": CANONICAL_TASK_METRIC_VERSION, "generator_version": PROJECTIVE_GENERATOR_VERSION,
                "retry": {"attempt_count": attempt, "attempts": attempted}, "old_metric": old_metric, "new_metric": metric,
                "initial_constraints": initial_constraints, "target_constraints": target_constraints,
                "old_result": old_result, "new_result": result, "old_initial_metric": old_initial_metric,
                "new_initial_metric": initial_metric, "old_initial_pose_c2w": old_initial_pose.tolist(),
                "new_initial_pose_c2w": initial_pose.tolist(), "initial_pose_repaired": initial_pose_repaired,
                "initial_position_repaired": initial_position_repaired,
                "initial_orientation_repaired": initial_orientation_repaired,
                "navigation_distance": navigation_distance,
            }
    metric_ready = [attempt for attempt in attempted if all(bool(attempt["gates"].get(key)) for key in ("two_objects", "in_front", "visible", "min_area", "inside_frame", "relation"))]
    failure = "projective_margin_infeasible_for_source_pair" if metric_ready and all(not bool(attempt["gates"].get("margin")) for attempt in metric_ready) else "bounded_layout_candidates_exhausted"
    return None, {"source_row_index": source_index, "old_task_id": item.get("task_id"), "scene_id": item.get("scene_id"), "relation": item["target_region"]["params"].get("relation"), "failure": failure, "initial_constraints": initial_constraints, "attempts": attempted}


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
        repaired["task_id"] = f"fov_canonical_h1_v2_{source_index:06d}"
        repaired["camera_model_version"] = CANONICAL_CAMERA_H1_RESIZE_V1
        repaired["canonical_task_metric_version"] = CANONICAL_TASK_METRIC_VERSION
        repaired["generator_version"] = FOV_GENERATOR_VERSION
        repaired["repair_lineage"] = {"source_row_index": source_index, "old_task_id": item.get("task_id"), "repair_reason": "legacy_fov_initial_pose_and_metric_degeneracy"}
        target_result = score_observation(repaired, target_pose)
        target_metric = canonical_fov(target_result)
        target_constraints = constraints.validate(repaired, target_point, initial_room_index=desired_room_index, check_pair_distance=True) if constraints else None
        chosen_initial = None
        init_metric = None
        init_pose = None
        initial_constraints = None
        for initial_point in initial_points:
            if float(np.linalg.norm(np.asarray(initial_point)[:2] - target_point[:2])) < 0.5:
                continue
            to_pair = normalize_vector(pair_midpoint(repaired) - np.asarray(initial_point, dtype=float))
            init_pose_candidate = camera_pose_from_forward(initial_point, -to_pair)
            init_metric_candidate = canonical_fov(score_observation(repaired, init_pose_candidate))
            constraint_candidate = constraints.validate(
                repaired,
                np.asarray(initial_point),
                initial_room_index=desired_room_index,
                check_pair_distance=False,
            ) if constraints else None
            if not init_metric_candidate["success"] and (constraint_candidate is None or constraint_candidate["success"]):
                chosen_initial = np.asarray(initial_point, dtype=float)
                init_pose = init_pose_candidate
                init_metric = init_metric_candidate
                initial_constraints = constraint_candidate
                break
        generation_success = bool(target_metric["success"]) and init_metric is not None and (target_constraints is None or bool(target_constraints["success"]))
        attempts.append({"attempt": attempt, "success": generation_success, "target_metric": target_metric, "initial_metric": init_metric, "target_constraints": target_constraints, "initial_constraints": initial_constraints})
        if generation_success:
            repaired["init_camera"] = dict(repaired["init_camera"])
            repaired["init_camera"]["extrinsics"] = init_pose.tolist()
            return repaired, {"source_row_index": source_index, "old_task_id": item.get("task_id"), "new_task_id": repaired["task_id"], "scene_id": item.get("scene_id"), "camera_model_version": CANONICAL_CAMERA_H1_RESIZE_V1, "canonical_task_metric_version": CANONICAL_TASK_METRIC_VERSION, "generator_version": FOV_GENERATOR_VERSION, "old_init_metric": old_init_metric, "old_target_metric": old_target_metric, "new_init_metric": init_metric, "new_target_metric": target_metric, "old_initial_constraints": old_initial_constraints, "initial_constraints": initial_constraints, "target_constraints": target_constraints, "old_initial_pose_c2w": old_init.tolist(), "new_initial_pose_c2w": init_pose.tolist(), "old_target_pose_c2w": old_target.tolist(), "new_target_pose_c2w": target_pose.tolist(), "initial_position_repaired": not np.allclose(chosen_initial, old_init[:3, 3]), "retry": {"attempt_count": attempt, "attempts": attempts}}
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
