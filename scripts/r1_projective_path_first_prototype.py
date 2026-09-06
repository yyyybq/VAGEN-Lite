#!/usr/bin/env python3
"""Path-first Projective generator prototype.

This is deliberately separate from the frozen R1 generator.  It chooses a
safe canonical-success target first, then reverse-expands the formal action
lattice to find an observable, canonical-fail predecessor.  Every returned
certificate is replayed by the independent runtime planner before it is
reported as a prototype success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from r1_canonical_tasks import canonical_projective, pose_from_item_target, score_observation
from r1_reachability_audit import search_one, state_key
from r1_repair_pipeline import (
    MAX_ABS_PITCH_DEG,
    PLANNER_ACTIONS,
    SceneConstraints,
    absolute_pitch_degrees,
    camera_pose_from_forward,
    horizontal_forward,
    pair_midpoint,
    projective_candidates,
    projective_difficulty_profile,
    projective_initial_geometry_discernible,
    relation_normal,
    construct_projective_reachable_initial,
)
from vagen.envs.active_spatial.utils import ViewManipulator


INVERSE = {
    "move_forward": "move_backward",
    "move_backward": "move_forward",
    "move_left": "move_right",
    "move_right": "move_left",
    "turn_left": "turn_right",
    "turn_right": "turn_left",
}
TRANSLATION = frozenset(PLANNER_ACTIONS[:4])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def pair_signature(item: dict[str, Any]) -> tuple[str, ...]:
    objects = item.get("target_object", {}).get("objects") or []
    if objects:
        return tuple(f"{obj.get('id')}:{obj.get('label')}" for obj in objects)
    target = item.get("target_object", {})
    return (f"{target.get('id')}:{target.get('label')}",)


def target_pose(item: dict[str, Any], point: np.ndarray, yaw_offset: float) -> np.ndarray:
    forward = horizontal_forward(pair_midpoint(item) - point)
    if yaw_offset:
        radians = math.radians(yaw_offset)
        c, s = math.cos(radians), math.sin(radians)
        forward = horizontal_forward(
            np.array([c * forward[0] - s * forward[1], s * forward[0] + c * forward[1], forward[2]])
        )
    return camera_pose_from_forward(point, forward)


def reverse_search(
    item: dict[str, Any],
    target: np.ndarray,
    constraints: SceneConstraints,
    room_index: int | None,
    max_steps: int,
    max_expansions: int,
) -> dict[str, Any]:
    detector = constraints._collision_cache.get(str(item.get("scene_id") or ""))
    if detector is None:
        return {"status": "asset_unavailable", "expansions": 0, "visited_states": 0}
    queue: list[tuple[int, int, np.ndarray, tuple[str, ...]]] = [(0, 0, target, tuple())]
    seen = {state_key(target)}
    expansions = 0
    while queue:
        depth, _, pose, reverse_actions = heapq.heappop(queue)
        if depth >= max_steps:
            continue
        for action in PLANNER_ACTIONS:
            inverse_action = INVERSE[action]
            engine = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
            engine.reset(pose)
            predecessor = engine.step(inverse_action)
            if inverse_action in TRANSLATION:
                collision = detector.check_collision(predecessor[:3, 3], previous_position=pose[:3, 3])
                if collision.has_collision:
                    continue
            layout = constraints.validate(
                item, predecessor[:3, 3], initial_room_index=room_index, check_pair_distance=False
            )
            if not layout.get("success"):
                continue
            key = state_key(predecessor)
            if key in seen:
                continue
            seen.add(key)
            expansions += 1
            new_reverse = reverse_actions + (inverse_action,)
            metric = canonical_projective(score_observation(item, predecessor))
            if (
                not metric.get("success")
                and projective_initial_geometry_discernible(metric)
                and absolute_pitch_degrees(predecessor[:3, 2]) <= MAX_ABS_PITCH_DEG
                and float(np.linalg.norm(predecessor[:2, 3] - target[:2, 3])) >= 0.5
            ):
                actions = [INVERSE[value] for value in reversed(new_reverse)]
                return {
                    "status": "candidate_found",
                    "pose": predecessor,
                    "metric": metric,
                    "layout": layout,
                    "actions": actions,
                    "reverse_actions": list(new_reverse),
                    "steps": len(actions),
                    "expansions": expansions,
                    "visited_states": len(seen),
                }
            if expansions >= max_expansions:
                return {
                    "status": "unverified_expansion_cap",
                    "expansions": expansions,
                    "visited_states": len(seen),
                }
            heapq.heappush(queue, (depth + 1, expansions, predecessor, new_reverse))
    return {
        "status": "exhaustive_no_predecessor_within_12_steps",
        "expansions": expansions,
        "visited_states": len(seen),
    }


def make_candidate(
    source_index: int,
    item: dict[str, Any],
    point: np.ndarray,
    target: np.ndarray,
    initial: dict[str, Any],
) -> dict[str, Any]:
    repaired = copy.deepcopy(item)
    params = repaired["target_region"]["params"]
    forward = target[:3, 2]
    params["normal"] = relation_normal(repaired).tolist()
    repaired["target_region"]["sample_point"] = point.tolist()
    repaired["target_region"]["sample_forward"] = forward.tolist()
    repaired["sample_target"] = point.tolist()
    repaired["camera_params"] = dict(repaired.get("camera_params") or {})
    repaired["camera_params"]["forward"] = forward.tolist()
    repaired["distance"] = float(np.linalg.norm(point[:2] - np.asarray(params["boundary_point"], dtype=float)[:2]))
    repaired["task_id"] = f"projective_path_first_proto_{source_index:06d}"
    repaired["generator_version"] = "projective_path_first_prototype_v1"
    repaired["reachability_construction"] = {
        "construction": "projective_path_first_reverse_action_lattice_v1",
        "actions": initial["actions"],
        "steps": initial["steps"],
        "shortest_path_claim": False,
        "reverse_actions": initial["reverse_actions"],
    }
    repaired["init_camera"] = dict(repaired["init_camera"])
    repaired["init_camera"]["extrinsics"] = initial["pose"].tolist()
    return repaired


def prototype_one(index: int, item: dict[str, Any], constraints: SceneConstraints, budgets: list[int], reverse_cap: int, baseline_item: dict[str, Any] | None = None, baseline_target_pose: np.ndarray | None = None) -> dict[str, Any]:
    started = time.time()
    search_item = baseline_item or item
    initial_pose = np.asarray(search_item["init_camera"]["extrinsics"], dtype=float)
    old_target_pose = pose_from_item_target(search_item)
    old_result = score_observation(search_item, old_target_pose)
    old_metric = canonical_projective(old_result)
    source_difficulty = projective_difficulty_profile(item, initial_pose, old_target_pose, old_result, old_metric)
    initial_check = constraints.validate(search_item, initial_pose[:3, 3], check_pair_distance=False)
    layout, _ = constraints.scene(str(search_item.get("scene_id") or ""))
    try:
        midpoint = pair_midpoint(search_item)
    except (KeyError, TypeError):
        midpoint = pair_midpoint(item)
    room_index = constraints.room_index(layout, midpoint[:2])
    if room_index is None:
        room_index = initial_check.get("room_index")
    points = projective_candidates(search_item, constraints, room_index)[:64]
    target_candidates = 0
    target_success = 0
    reverse_calls = 0
    reverse_resolved = 0
    target_states: list[tuple[tuple[float, ...], np.ndarray, np.ndarray, dict[str, Any]]] = []
    if baseline_target_pose is not None:
        point = np.asarray(search_item.get("sample_target"), dtype=float)
        target_constraints = constraints.validate(search_item, baseline_target_pose[:3, 3], initial_room_index=room_index, check_pair_distance=True)
        metric = canonical_projective(score_observation(search_item, baseline_target_pose))
        target_candidates += 1
        if metric.get("success") and target_constraints.get("success"):
            target_success += 1
            target_states.append(((-1.0, 0.0), point, baseline_target_pose, {"metric": metric, "constraints": target_constraints}))
    for point in points:
        target_constraints = constraints.validate(search_item, point, initial_room_index=room_index, check_pair_distance=True)
        if not target_constraints.get("success"):
            continue
        for yaw_offset in (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0):
            target_candidates += 1
            pose = target_pose(search_item, point, yaw_offset)
            metric = canonical_projective(score_observation(search_item, pose))
            if not metric.get("success"):
                continue
            target_success += 1
            target_states.append((
                (abs(float(np.linalg.norm(point[:2] - np.asarray(search_item["sample_target"], dtype=float)[:2]))), abs(yaw_offset)),
                point, pose, {"metric": metric, "constraints": target_constraints},
            ))
    # Evaluate every target geometrically, but reverse-expand only a bounded
    # shortlist.  A single 25k escalation is allowed after quick 2k probes;
    # this preserves tier semantics without multiplying cost by 448 states.
    target_states.sort(key=lambda row: row[0])
    best: tuple[tuple[float, ...], dict[str, Any], dict[str, Any]] | None = None
    shortlist = target_states[:4]
    for shortlist_index, (_, point, pose, target_info) in enumerate(shortlist):
        result = reverse_search(search_item, pose, constraints, room_index, 12, min(budgets[0], reverse_cap))
        reverse_calls += 1
        if result["status"] != "candidate_found" and shortlist_index == 0 and len(budgets) > 1:
            result = reverse_search(search_item, pose, constraints, room_index, 12, min(budgets[1], reverse_cap))
            reverse_calls += 1
        if result["status"] != "candidate_found":
            # Keep a bounded action-lattice fallback for positive controls. It
            # is still target-first (the target was selected before this
            # call), but avoids declaring a known valid control failed solely
            # because the BFS cap was intentionally conservative.
            fallback = construct_projective_reachable_initial(
                search_item, pose, constraints, room_index,
                int(source_difficulty["planner_step_proxy"]),
            )
            if fallback is not None:
                fallback_pose, fallback_metric, fallback_layout, certificate = fallback
                result = {
                    "status": "candidate_found",
                    "pose": fallback_pose,
                    "metric": fallback_metric,
                    "layout": fallback_layout,
                    "actions": certificate["actions"],
                    "reverse_actions": certificate["reverse_construction_actions"],
                    "steps": certificate["steps"],
                    "expansions": result.get("expansions", 0),
                    "visited_states": result.get("visited_states", 0),
                    "fallback": "bounded_inverse_action_lattice_v1",
                }
        if result["status"] != "candidate_found":
            continue
        reverse_resolved += 1
        target_result = score_observation(search_item, pose)
        difficulty = projective_difficulty_profile(
            item, result["pose"], pose, target_result, target_info["metric"], planner_steps=result["steps"]
        )
        bucket_distance = sum(
            abs(int(source_difficulty["buckets"][name]) - int(difficulty["buckets"][name]))
            for name in source_difficulty["buckets"]
        )
        rank = (float(bucket_distance), abs(result["steps"] - source_difficulty["planner_step_proxy"]), float(result["steps"]))
        candidate = make_candidate(index, search_item, point, pose, result)
        candidate["camera_params"]["forward"] = pose[:3, 2].tolist()
        details = {
            "target_metric": target_info["metric"], "target_constraints": target_info["constraints"],
            "initial_metric": result["metric"], "initial_constraints": result["layout"],
            "source_difficulty": source_difficulty, "repaired_difficulty": difficulty, "reverse": result,
        }
        best = min(best, (rank, candidate, details)) if best else (rank, candidate, details)
        if rank[0] == 0:
            break
    output = {
        "source_row_index": index, "scene_id": item.get("scene_id"), "task_type": item.get("task_type"),
        "input_pair_kind": "baseline_replacement" if baseline_item is not None else "source_pair",
        "status": "candidate_found" if best else "no_candidate_within_budget",
        "target_candidates": target_candidates, "canonical_success_targets": target_success,
        "reverse_calls": reverse_calls, "reverse_resolved": reverse_resolved,
        "elapsed_seconds": time.time() - started, "budgets": budgets,
        "source_difficulty": source_difficulty,
    }
    if best:
        _, candidate, details = best
        output.update({"row": candidate, "details": details})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="2000,25000")
    parser.add_argument("--reverse-expansion-cap", type=int, default=256)
    parser.add_argument("--baseline-jobs-root", type=Path)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    rows_by_split = {split: read_jsonl(Path(path)) for split, path in sources.items()}
    selection = json.loads(args.selection.read_text())["records"]
    overrides_path = args.output_dir / "collision_overrides.json"
    override_payload = {}
    if overrides_path.is_file():
        override_payload = json.loads(overrides_path.read_text()).get("structure_y_sign_overrides", {})
    constraints = SceneConstraints(args.gs_root, structure_y_sign_overrides=override_payload)
    budgets = [int(value) for value in args.budgets.split(",") if value]
    rows = []
    for record in selection:
        split = record["split"]
        index = int(record["source_row_index"])
        item = rows_by_split[split][index]
        if item.get("task_type") != "projective_relations":
            continue
        baseline_item = None
        if args.baseline_jobs_root:
            repaired_path = args.baseline_jobs_root / f"{split}_{index:06d}" / "repair" / split / "trainable.jsonl"
            if repaired_path.is_file():
                repaired_rows = read_jsonl(repaired_path)
                # The repaired trainable schema intentionally normalizes its
                # target object fields, so it is not used as the prototype
                # search item.  Only its saved final pose is borrowed for a
                # same-source positive-control replay below.
                baseline_item = None
        baseline_target_pose = None
        if args.baseline_jobs_root:
            reachability_path = args.baseline_jobs_root / f"{split}_{index:06d}" / "repair" / split / "reachability_manifest.jsonl"
            if reachability_path.is_file():
                reachability_rows = read_jsonl(reachability_path)
                if reachability_rows and reachability_rows[0].get("path"):
                    baseline_target_pose = np.asarray(reachability_rows[0]["path"][-1]["c2w"], dtype=float)
        rows.append(prototype_one(index, item, constraints, budgets, args.reverse_expansion_cap, baseline_item, baseline_target_pose))
        print(json.dumps(rows[-1], default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value), flush=True)
    summary = Counter(row["status"] for row in rows)
    payload = {"version": "projective_path_first_prototype_v1", "selection": str(args.selection), "budgets": budgets, "reverse_expansion_cap": args.reverse_expansion_cap, "rows": rows, "status_counts": dict(summary)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prototype_results.json").write_text(json.dumps(payload, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value) + "\n")
    print(json.dumps({"status_counts": dict(summary), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
