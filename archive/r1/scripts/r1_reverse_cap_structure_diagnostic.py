#!/usr/bin/env python3
"""Trace the frozen reverse search structure without changing its budgets.

This is diagnostic instrumentation around the v2 forward-validated reverse
graph.  It neither changes candidate eligibility nor creates a new selector.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r1_action_graph import INVERSE_ACTION, forward_validated_predecessors
from r1_canonical_tasks import canonical_projective, score_observation
from r1_projective_path_first_prototype import pair_midpoint, success_region_points, target_pose
from r1_reachability_audit import state_key
from r1_repair_pipeline import (
    MAX_ABS_PITCH_DEG,
    SceneConstraints,
    absolute_pitch_degrees,
    projective_initial_geometry_discernible,
)


VERSION = "r1_projective_reverse_cap_structure_diagnostic_v1"
LOW_PRODUCTIVITY_SCENES = ("0011_840866", "0276_840780", "0367_840260", "0229_840306")
# 0240 is the highest-yield scene but has no reverse-cap row to diagnose;
# choose the next frozen high-yield scenes that actually contain this failure.
HIGH_PRODUCTIVITY_SCENES = ("0226_840298", "0014_841007", "0267_840790", "0361_840315")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n")
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def stable_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["split"]), int(row["source_row_index"])


def geometry(item: dict[str, Any]) -> dict[str, Any]:
    objects = item.get("target_object", {}).get("objects", [])
    if len(objects) < 2:
        return {"available": False}
    centers = [np.asarray(obj["center"], dtype=float) for obj in objects[:2]]
    sizes = [np.asarray(obj["bbox_max"], dtype=float) - np.asarray(obj["bbox_min"], dtype=float)
             for obj in objects[:2]]
    delta = centers[0] - centers[1]
    return {
        "available": True,
        "horizontal_center_separation_m": float(np.linalg.norm(delta[:2])),
        "vertical_center_separation_m": float(abs(delta[2])),
        "center_distance_m": float(np.linalg.norm(delta)),
        "object_bbox_sizes_m": [size.tolist() for size in sizes],
        "object_bbox_volumes_m3": [float(np.prod(size)) for size in sizes],
    }


def freeze_selection(selector_path: Path, observability_path: Path, output: Path) -> None:
    selector = json.loads(selector_path.read_text())
    rows = selector["results"]
    cap_rows = []
    for row in rows:
        attempts = row.get("reverse_attempts", [])
        if (row.get("success_region_seeds_considered", 0) > 0
                and row.get("initial_candidates_collected", 0) == 0
                and attempts
                and all(attempt.get("status") == "unverified_expansion_cap" for attempt in attempts)):
            cap_rows.append(row)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cap_rows:
        by_scene[str(row["scene_id"])].append(row)
    selected = []
    strata = [("low", scene) for scene in LOW_PRODUCTIVITY_SCENES] + [("high", scene) for scene in HIGH_PRODUCTIVITY_SCENES]
    for productivity, scene in strata:
        pool = sorted(by_scene[scene], key=lambda row: (float(row.get("elapsed_seconds", 0)), stable_key(row)))
        if len(pool) < 2:
            raise ValueError(f"scene {scene} has fewer than two reverse-cap rows")
        for cost_stratum, row in (("low_cost", pool[0]), ("high_cost", pool[-1])):
            selected.append({
                "split": row["split"], "source_row_index": int(row["source_row_index"]),
                "scene_id": row["scene_id"], "kind": "reverse_cap",
                "productivity_stratum": productivity, "cost_stratum": cost_stratum,
                "baseline_elapsed_seconds": row.get("elapsed_seconds"),
                "baseline_success_seeds": row.get("success_region_seeds_considered"),
            })
    # Four positive controls from four distinct scenes.  These are evidence
    # controls, not part of the 16-row failure denominator.
    passed_tasks = {row["task_id"] for row in read_jsonl(observability_path) if row.get("passed")}
    controls = []
    used_scenes = set()
    for row in rows:
        task_id = (row.get("row") or {}).get("task_id") or (
            f"projective_path_first_proto_{row['split']}_{int(row['source_row_index']):06d}_medium"
        )
        if not task_id.endswith("_medium"):
            task_id += "_medium"
        if row.get("status") != "difficulty_certified_candidate" or task_id not in passed_tasks:
            continue
        scene = str(row["scene_id"])
        if scene in used_scenes:
            continue
        controls.append({
            "split": row["split"], "source_row_index": int(row["source_row_index"]),
            "scene_id": scene, "kind": "known_positive_control",
            "baseline_elapsed_seconds": row.get("elapsed_seconds"),
            "baseline_first_success_step": row.get("certificate_upper_bound"),
            "baseline_task_id": task_id,
        })
        used_scenes.add(scene)
        if len(controls) == 4:
            break
    if len(controls) != 4:
        raise ValueError("could not select four distinct-scene positive controls")
    payload = {
        "version": VERSION,
        "selection_is_frozen": True,
        "selection_rule": {
            "reverse_cap": "min/max baseline elapsed row from four low- and four high-productivity scenes",
            "positive_controls": "first RGB-pass candidate in selector order from four distinct scenes",
            "outcome_independence": "selection uses frozen v2 records only; no diagnostic outcome inspected",
        },
        "selector": {"path": str(selector_path.resolve()), "sha256": file_sha256(selector_path)},
        "observability": {"path": str(observability_path.resolve()), "sha256": file_sha256(observability_path)},
        "budgets": {"seed_cap": 12, "per_seed_expansions": 512,
                    "candidate_cap_per_seed": 48, "max_reverse_steps": 12},
        "classification_rules": {
            "A": "cap reached with pending frontier at depth <=4: budget spent before effective 4-6 coverage completed",
            "B": "some 4-6 states covered but no state satisfies all initial conditions",
            "C": "cross-seed duplicate visitation fraction >=0.25 (descriptive overlap threshold)",
            "D": "concrete forward-state mismatch or quantized alias pose error >1e-6",
            "E": "available evidence does not meet A-D",
        },
        "reverse_cap_count": len(selected), "control_count": len(controls),
        "records": selected + controls,
    }
    write_json(output, payload)
    print(json.dumps({"reverse_cap": len(selected), "controls": len(controls),
                      "scenes": dict(Counter(row["scene_id"] for row in selected))}, sort_keys=True))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_rejection(metric: dict[str, Any], pose: np.ndarray) -> tuple[str | None, list[str]]:
    if metric.get("success"):
        return "canonical_success", []
    if absolute_pitch_degrees(pose[:3, 2]) > MAX_ABS_PITCH_DEG:
        return "pitch_reject", []
    if projective_initial_geometry_discernible(metric):
        return None, []
    gates = metric.get("gates", {})
    failed = sorted(str(key) for key, value in gates.items() if not value)
    return "initial_projection_reject", failed


def trace_seed(item: dict[str, Any], target: np.ndarray, constraints: SceneConstraints,
               room_index: int | None, max_steps: int, max_expansions: int,
               candidate_cap: int) -> tuple[dict[str, Any], set[tuple[float, ...]]]:
    detector = constraints._collision_cache[str(item.get("scene_id") or "")]
    queue: list[tuple[int, int, np.ndarray, tuple[str, ...]]] = [(0, 0, target, tuple())]
    initial_key = state_key(target)
    seen = {initial_key}
    representative = {initial_key: np.asarray(target, dtype=float)}
    queued = Counter({0: 1})
    popped = Counter()
    admitted = Counter()
    condition_eligible = Counter()
    rejection_by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    projection_gate_rejections: dict[int, Counter[str]] = defaultdict(Counter)
    alias_samples = []
    maximum_alias_pose_error = 0.0
    candidates = 0
    expansions = 0
    serial = 0
    started = time.time()
    status = "exhausted"
    while queue:
        depth, _, pose, reverse_actions = heapq.heappop(queue)
        popped[depth] += 1
        if depth >= max_steps:
            continue
        child_depth = depth + 1
        for edge in forward_validated_predecessors(pose, detector):
            reason = edge["rejection_reason"]
            if reason is not None:
                rejection_by_depth[child_depth][str(reason)] += 1
                continue
            predecessor = edge["predecessor"]
            layout = constraints.validate(item, predecessor[:3, 3], initial_room_index=room_index,
                                          check_pair_distance=False)
            if not layout.get("success"):
                rejection_by_depth[child_depth]["layout_reject"] += 1
                continue
            key = state_key(predecessor)
            if key in seen:
                error = float(np.max(np.abs(predecessor - representative[key])))
                maximum_alias_pose_error = max(maximum_alias_pose_error, error)
                rejection_by_depth[child_depth]["dedup_state_key"] += 1
                if len(alias_samples) < 8:
                    alias_samples.append({
                        "depth": child_depth, "pose_max_abs_error": error,
                        "same_state_key": True, "forward_edge_was_validated": True,
                        "action": edge["forward_action"],
                    })
                continue
            seen.add(key)
            representative[key] = predecessor
            expansions += 1
            admitted[child_depth] += 1
            new_reverse = reverse_actions + (edge["inverse_proposal_action"],)
            metric = canonical_projective(score_observation(item, predecessor))
            state_rejection, failed_gates = metric_rejection(metric, predecessor)
            if state_rejection is None:
                condition_eligible[child_depth] += 1
                candidates += 1
            else:
                rejection_by_depth[child_depth][state_rejection] += 1
                if state_rejection == "initial_projection_reject":
                    projection_gate_rejections[child_depth].update(failed_gates)
            if candidates >= candidate_cap:
                status = "candidate_cap"
                break
            if expansions >= max_expansions:
                status = "unverified_expansion_cap"
                break
            serial += 1
            heapq.heappush(queue, (child_depth, serial, predecessor, new_reverse))
            queued[child_depth] += 1
        if status != "exhausted":
            break
    frontier = Counter(row[0] for row in queue)
    forward_mismatches = sum(counts.get("forward_state_mismatch", 0) for counts in rejection_by_depth.values())
    result = {
        "status": status,
        "elapsed_seconds": time.time() - started,
        "expansions_original_semantics": expansions,
        "unique_states_including_seed": len(seen),
        "maximum_admitted_depth": max(admitted, default=0),
        "maximum_popped_depth": max(popped, default=0),
        "states_at_depth_4_to_6": sum(admitted[depth] for depth in (4, 5, 6)),
        "eligible_initials_at_depth_4_to_6": sum(condition_eligible[depth] for depth in (4, 5, 6)),
        "eligible_initials_all_depths": candidates,
        "by_depth": {
            str(depth): {
                "queued": queued[depth], "popped": popped[depth], "unique_admitted": admitted[depth],
                "eligible_initial": condition_eligible[depth],
                "rejections": dict(sorted(rejection_by_depth[depth].items())),
                "initial_projection_failed_gates": dict(sorted(projection_gate_rejections[depth].items())),
            } for depth in sorted(set(queued) | set(popped) | set(admitted) | set(rejection_by_depth))
        },
        "frontier_at_stop": {"count": sum(frontier.values()), "by_depth": dict(sorted(frontier.items())),
                             "minimum_depth": min(frontier, default=None)},
        "state_key_audit": {
            "round_decimals": 7,
            "dedup_events": sum(counts.get("dedup_state_key", 0) for counts in rejection_by_depth.values()),
            "maximum_alias_pose_error": maximum_alias_pose_error,
            "forward_state_mismatches": forward_mismatches,
            "samples": alias_samples,
        },
    }
    return result, seen


def run_diagnostic(selection_path: Path, source_inventory_path: Path, gs_root: Path, output_dir: Path) -> None:
    selection = json.loads(selection_path.read_text())
    inventory = json.loads(source_inventory_path.read_text())
    sources = {split: read_jsonl(Path(value["path"] if isinstance(value, dict) else value))
               for split, value in inventory["sources"].items()}
    constraints = SceneConstraints(gs_root)
    outputs = []
    total_started = time.time()
    for record in selection["records"]:
        started = time.time()
        item = sources[record["split"]][int(record["source_row_index"])]
        layout, _ = constraints.scene(str(item.get("scene_id") or ""))
        room = constraints.room_index(layout, pair_midpoint(item)[:2])
        if room is None:
            room = constraints.validate(item, np.asarray(item["init_camera"]["extrinsics"], dtype=float)[:3, 3],
                                        check_pair_distance=False).get("room_index")
        seeds = []
        for point in success_region_points(item, constraints, room):
            if not constraints.validate(item, point, initial_room_index=room, check_pair_distance=False).get("success"):
                continue
            for yaw in (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0):
                pose = target_pose(item, point, yaw)
                metric = canonical_projective(score_observation(item, pose))
                if metric.get("success"):
                    seeds.append((point, pose, metric))
        seeds = seeds[:selection["budgets"]["seed_cap"]]
        seed_results = []
        seed_state_sets = []
        for index, (_, target, metric) in enumerate(seeds):
            result, states = trace_seed(item, target, constraints, room,
                                        selection["budgets"]["max_reverse_steps"],
                                        selection["budgets"]["per_seed_expansions"],
                                        selection["budgets"]["candidate_cap_per_seed"])
            result["seed_index"] = index
            result["target_metric"] = metric
            seed_results.append(result)
            seed_state_sets.append(states)
        state_visits = Counter(key for states in seed_state_sets for key in states)
        total_seed_states = sum(len(states) for states in seed_state_sets)
        duplicate_across_seeds = sum(count - 1 for count in state_visits.values())
        duplicate_fraction = duplicate_across_seeds / total_seed_states if total_seed_states else 0.0
        pairwise = []
        for first in range(len(seed_state_sets)):
            for second in range(first + 1, len(seed_state_sets)):
                overlap = len(seed_state_sets[first] & seed_state_sets[second])
                union = len(seed_state_sets[first] | seed_state_sets[second])
                pairwise.append({"first_seed": first, "second_seed": second, "intersection": overlap,
                                 "union": union, "jaccard": overlap / union if union else 0.0})
        forward_mismatch = sum(row["state_key_audit"]["forward_state_mismatches"] for row in seed_results)
        alias_error = max((row["state_key_audit"]["maximum_alias_pose_error"] for row in seed_results), default=0.0)
        pending_shallow = any(row["status"] == "unverified_expansion_cap"
                              and row["frontier_at_stop"]["minimum_depth"] is not None
                              and row["frontier_at_stop"]["minimum_depth"] <= 4 for row in seed_results)
        states_4_6 = sum(row["states_at_depth_4_to_6"] for row in seed_results)
        eligible_4_6 = sum(row["eligible_initials_at_depth_4_to_6"] for row in seed_results)
        tags = []
        if pending_shallow:
            tags.append("A")
        if states_4_6 and not eligible_4_6:
            tags.append("B")
        if duplicate_fraction >= 0.25:
            tags.append("C")
        if forward_mismatch or alias_error > 1e-6:
            tags.append("D")
        if not tags:
            tags.append("E")
        primary = "D" if "D" in tags else ("A" if "A" in tags else ("B" if "B" in tags else ("C" if "C" in tags else "E")))
        output = {
            **record,
            "source_geometry": geometry(item),
            "success_seed_count": len(seeds),
            "seed_results": seed_results,
            "cross_seed_overlap": {
                "sum_seed_unique_states": total_seed_states,
                "union_unique_states": len(state_visits),
                "duplicate_visits_across_seeds": duplicate_across_seeds,
                "duplicate_visit_fraction": duplicate_fraction,
                "maximum_pairwise_jaccard": max((row["jaccard"] for row in pairwise), default=0.0),
                "pairwise": pairwise,
            },
            "aggregate": {"states_at_depth_4_to_6": states_4_6,
                          "eligible_initials_at_depth_4_to_6": eligible_4_6},
            "diagnostic_classes": tags,
            "primary_class": primary,
            "elapsed_seconds": time.time() - started,
        }
        outputs.append(output)
        write_json(output_dir / "checkpoint.json", {"version": VERSION, "selection": str(selection_path), "results": outputs})
        print(json.dumps({"split": record["split"], "source_row_index": record["source_row_index"],
                          "kind": record["kind"], "class": primary, "tags": tags,
                          "elapsed_seconds": output["elapsed_seconds"]}), flush=True)
    payload = {
        "version": VERSION,
        "selection": {"path": str(selection_path.resolve()), "sha256": file_sha256(selection_path)},
        "source_inventory": {"path": str(source_inventory_path.resolve()), "sha256": file_sha256(source_inventory_path)},
        "budgets_unchanged": selection["budgets"],
        "elapsed_seconds": time.time() - total_started,
        "class_counts_reverse_cap": dict(Counter(row["primary_class"] for row in outputs if row["kind"] == "reverse_cap")),
        "tag_counts_reverse_cap": dict(Counter(tag for row in outputs if row["kind"] == "reverse_cap" for tag in row["diagnostic_classes"])),
        "results": outputs,
    }
    write_json(output_dir / "reverse_cap_diagnostic.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--selector", type=Path, required=True)
    freeze.add_argument("--observability", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--source-inventory", type=Path, required=True)
    run.add_argument("--gs-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_selection(args.selector, args.observability, args.output)
    else:
        run_diagnostic(args.selection, args.source_inventory, args.gs_root, args.output_dir)


if __name__ == "__main__":
    main()
