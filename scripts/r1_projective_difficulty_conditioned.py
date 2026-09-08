#!/usr/bin/env python3
"""Frozen small-sample difficulty-conditioned Projective initial selector.

This is intentionally separate from the v2 path-first generator.  Reverse
depth proposes candidates only; the requested difficulty label is granted
solely after a complete *forward-runtime* shortcut search against the full
canonical-success region.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from r1_action_graph import ACTIONS, INVERSE_ACTION, TRANSLATION_ACTIONS, forward_transition, forward_validated_predecessors
from r1_canonical_tasks import canonical_projective, score_observation
from r1_projective_path_first_prototype import (
    PATH_FIRST_GENERATOR_VERSION,
    make_candidate,
    pair_midpoint,
    success_region_points,
    target_pose,
)
from r1_reachability_audit import state_key
from r1_repair_pipeline import (
    MAX_ABS_PITCH_DEG,
    SceneConstraints,
    absolute_pitch_degrees,
    projective_initial_geometry_discernible,
)


SELECTOR_VERSION = "projective_difficulty_conditioned_initial_selector_v1"
BUCKETS = {"easy": (1, 3, 0), "medium": (4, 6, 3), "hard": (7, 12, 6)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def stable_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["split"]), int(row["source_row_index"])


def choose_frozen_sources(aggregate: dict[str, Any], count: int, hard_count: int) -> dict[str, Any]:
    """Deterministically retain all scenes, then round-robin path/type strata.

    The selection is from the existing accepted same-pair rows only.  It is a
    constructability experiment, never an estimate of 788-row coverage.
    """
    accepted = [row for row in aggregate["rows"] if row["final_status"] == "same_pair_accepted"]
    if len(accepted) < count:
        raise ValueError(f"only {len(accepted)} accepted rows for requested {count}")
    for row in accepted:
        difficulty = row.get("pose_difficulty") or {}
        row["_baseline_steps"] = int(difficulty.get("certificate_action_length_upper_bound") or 0)
        row["_initial_type"] = str(difficulty.get("initial_relation_type") or "unknown")
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_scene[str(row["scene_id"])].append(row)
    # Use a digest instead of Python hash so the list is stable across hosts.
    def order(row: dict[str, Any]) -> tuple[str, int]:
        digest = hashlib.sha256(f"difficulty-v1:{row['split']}:{row['source_row_index']}".encode()).hexdigest()
        return digest, int(row["source_row_index"])
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    # First pass covers all ten scenes to avoid an accidental large-room bias.
    for scene in sorted(by_scene):
        row = sorted(by_scene[scene], key=lambda value: (-value["_baseline_steps"], value["_initial_type"], order(value)))[0]
        selected.append(row); used.add(stable_key(row))
    # Fill in a deterministic round-robin over scene × relation-type × length.
    remaining = [row for row in accepted if stable_key(row) not in used]
    remaining.sort(key=lambda row: (str(row["scene_id"]), row["_initial_type"], row["_baseline_steps"], order(row)))
    while len(selected) < count and remaining:
        # Prefer under-represented scene/type and alternate short/long values.
        counts = Counter((str(row["scene_id"]), row["_initial_type"]) for row in selected)
        row = min(remaining, key=lambda value: (counts[(str(value["scene_id"]), value["_initial_type"])], order(value)))
        selected.append(row); used.add(stable_key(row)); remaining.remove(row)
    if len(selected) != count:
        raise AssertionError("frozen selection underfilled")
    selected.sort(key=stable_key)
    hard_pool = [row for row in selected if row["_baseline_steps"] >= 7]
    if len(hard_pool) < hard_count:
        hard_pool = sorted(selected, key=lambda row: (-row["_baseline_steps"], order(row)))
    hard = sorted(hard_pool[:hard_count], key=stable_key)
    records = [{
        "split": row["split"], "source_row_index": int(row["source_row_index"]),
        "scene_id": row["scene_id"], "baseline_task_id": row.get("task_id"),
        "baseline_certificate_upper_bound": row["_baseline_steps"],
        "baseline_initial_relation_type": row["_initial_type"],
        # The historical 12 is a proxy/budget field, not a certified minimum.
        "source_action_length": "unknown_historical_proxy_not_used_as_constraint",
    } for row in selected]
    return {
        "version": SELECTOR_VERSION, "selection_seed": "difficulty-v1-sha256", "source_count": count,
        "hard_subset_count": len(hard), "records": records,
        "hard_records": [{"split": row["split"], "source_row_index": int(row["source_row_index"])} for row in hard],
        "source_scope": "existing_225_same_pair_accepted_only_not_788_coverage",
    }


def collect_reverse_candidates(
    item: dict[str, Any], target: np.ndarray, constraints: SceneConstraints, room_index: int | None,
    *, max_steps: int, max_expansions: int, candidate_cap: int,
) -> dict[str, Any]:
    """Collect, rather than first-return, forward-validated initial poses."""
    detector = constraints._collision_cache.get(str(item.get("scene_id") or ""))
    if detector is None:
        return {"status": "asset_unavailable", "candidates": [], "expansions": 0, "visited_states": 0}
    queue: list[tuple[int, int, np.ndarray, tuple[str, ...]]] = [(0, 0, target, tuple())]
    seen = {state_key(target)}
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    expansions = 0
    while queue:
        depth, serial, pose, reverse_actions = heapq.heappop(queue)
        if depth >= max_steps:
            continue
        for edge in forward_validated_predecessors(pose, detector):
            predecessor = edge["predecessor"]
            if edge["rejection_reason"] is not None:
                rejected[str(edge["rejection_reason"])] += 1; continue
            layout = constraints.validate(item, predecessor[:3, 3], initial_room_index=room_index, check_pair_distance=False)
            if not layout.get("success"):
                rejected["layout_reject"] += 1; continue
            key = state_key(predecessor)
            if key in seen:
                rejected["quantization_mismatch"] += 1; continue
            seen.add(key); expansions += 1
            new_reverse = reverse_actions + (edge["inverse_proposal_action"],)
            metric = canonical_projective(score_observation(item, predecessor))
            actions = [INVERSE_ACTION[value] for value in reversed(new_reverse)]
            if (not metric.get("success") and projective_initial_geometry_discernible(metric)
                    and absolute_pitch_degrees(predecessor[:3, 2]) <= MAX_ABS_PITCH_DEG):
                candidates.append({"pose": predecessor, "metric": metric, "layout": layout,
                                   "actions": actions, "reverse_actions": list(new_reverse), "steps": len(actions)})
                if len(candidates) >= candidate_cap:
                    return {"status": "candidate_cap", "candidates": candidates, "expansions": expansions,
                            "visited_states": len(seen), "rejection_reasons": dict(rejected)}
            if expansions >= max_expansions:
                return {"status": "unverified_expansion_cap", "candidates": candidates, "expansions": expansions,
                        "visited_states": len(seen), "rejection_reasons": dict(rejected)}
            heapq.heappush(queue, (depth + 1, serial + expansions, predecessor, new_reverse))
    return {"status": "exhausted", "candidates": candidates, "expansions": expansions,
            "visited_states": len(seen), "rejection_reasons": dict(rejected)}


def runtime_shortcut_lower_bound(item: dict[str, Any], initial: np.ndarray, detector: Any, depth_limit: int, max_expansions: int) -> dict[str, Any]:
    """Complete forward enumeration through ``depth_limit`` with no quality filters.

    It uses only formal actions, collision and canonical success.  It does not
    apply observability or intermediate-frame conditions, so it cannot create
    a synthetic difficulty lower bound by hiding runtime-legal shortcuts.
    """
    initial_metric = canonical_projective(score_observation(item, initial))
    if initial_metric.get("success"):
        return {"complete": True, "first_success_step": 0, "certified_lower_bound": 0,
                "expansions": 0, "visited_states": 1, "runtime_shortcut_found": True}
    queue = deque([(initial, 0)])
    seen = {state_key(initial)}
    expansions = 0
    collision_rejections = Counter()
    while queue:
        pose, depth = queue.popleft()
        if depth >= depth_limit:
            continue
        if expansions >= max_expansions:
            return {"complete": False, "first_success_step": None, "certified_lower_bound": None,
                    "expansions": expansions, "visited_states": len(seen), "runtime_shortcut_found": None,
                    "reason": "difficulty_unverified_expansion_cap", "collision_rejections": dict(collision_rejections)}
        expansions += 1
        for action in ACTIONS:
            candidate = forward_transition(pose, action)
            if action in TRANSLATION_ACTIONS:
                collision = detector.check_collision(candidate[:3, 3], previous_position=pose[:3, 3])
                if collision.has_collision:
                    collision_rejections[collision.collision_type] += 1; continue
            key = state_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            metric = canonical_projective(score_observation(item, candidate))
            if metric.get("success"):
                return {"complete": True, "first_success_step": depth + 1,
                        "certified_lower_bound": depth + 1, "expansions": expansions,
                        "visited_states": len(seen), "runtime_shortcut_found": True,
                        "collision_rejections": dict(collision_rejections)}
            queue.append((candidate, depth + 1))
    return {"complete": True, "first_success_step": None, "certified_lower_bound": depth_limit + 1,
            "expansions": expansions, "visited_states": len(seen), "runtime_shortcut_found": False,
            "collision_rejections": dict(collision_rejections)}


def candidate_for_bucket(candidates: list[dict[str, Any]], requested: str) -> dict[str, Any] | None:
    lower, upper, _ = BUCKETS[requested]
    valid = [candidate for candidate in candidates if lower <= candidate["steps"] <= upper]
    if not valid:
        return None
    center = (lower + upper) / 2.0
    return min(valid, key=lambda candidate: (abs(candidate["steps"] - center), tuple(state_key(candidate["pose"]))))


def run_one(item: dict[str, Any], record: dict[str, Any], requested: str, constraints: SceneConstraints,
            *, seed_cap: int, per_seed_expansions: int, candidate_cap: int, lower_expansions: int) -> dict[str, Any]:
    started = time.time()
    layout, _ = constraints.scene(str(item.get("scene_id") or ""))
    midpoint = pair_midpoint(item)
    room_index = constraints.room_index(layout, midpoint[:2])
    if room_index is None:
        room_index = constraints.validate(item, np.asarray(item["init_camera"]["extrinsics"], dtype=float)[:3, 3], check_pair_distance=False).get("room_index")
    target_rows = []
    for point in success_region_points(item, constraints, room_index):
        target_layout = constraints.validate(item, point, initial_room_index=room_index, check_pair_distance=False)
        if not target_layout.get("success"):
            continue
        for yaw in (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0):
            pose = target_pose(item, point, yaw)
            metric = canonical_projective(score_observation(item, pose))
            if metric.get("success"):
                target_rows.append((point, pose, metric, target_layout))
    target_rows = target_rows[:seed_cap]
    all_candidates: list[tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]] = []
    reverse = []
    for seed_index, (point, target, metric, target_layout) in enumerate(target_rows):
        result = collect_reverse_candidates(item, target, constraints, room_index, max_steps=12,
                                            max_expansions=per_seed_expansions, candidate_cap=candidate_cap)
        reverse.append({"seed_index": seed_index, "status": result["status"], "expansions": result["expansions"],
                        "visited_states": result["visited_states"], "candidate_count": len(result["candidates"]),
                        "rejection_reasons": result.get("rejection_reasons", {})})
        all_candidates.extend((point, target, metric, candidate) for candidate in result["candidates"])
    proposed = candidate_for_bucket([row[3] for row in all_candidates], requested)
    base = {"version": SELECTOR_VERSION, "split": record["split"], "source_row_index": record["source_row_index"],
            "scene_id": record["scene_id"], "requested_bucket": requested, "same_pair_only": True,
            "success_region_seeds_considered": len(target_rows), "success_region_states_found": len(target_rows),
            "reverse_seed_cap": seed_cap, "per_seed_expansion_cap": per_seed_expansions, "candidate_cap_per_seed": candidate_cap,
            "reverse_attempts": reverse, "initial_candidates_collected": len(all_candidates),
            "elapsed_seconds": time.time() - started,
            "source_action_length": record["source_action_length"], "baseline_certificate_upper_bound": record["baseline_certificate_upper_bound"]}
    if proposed is None:
        return {**base, "status": "requested_bucket_not_found_within_budget"}
    point, target, target_metric, initial = next(row for row in all_candidates if row[3] is proposed)
    detector = constraints._collision_cache[str(item.get("scene_id") or "")]
    _, _, lower_depth = BUCKETS[requested]
    lower = runtime_shortcut_lower_bound(item, initial["pose"], detector, lower_depth, lower_expansions)
    # A generated reverse certificate may enter the success region before its
    # selected seed.  Runtime difficulty is the *first* success, never the
    # nominal terminal depth or any trailing actions.
    replay_pose = np.asarray(initial["pose"], dtype=float)
    first_success = None
    for step, action in enumerate(initial["actions"], start=1):
        replay_pose = forward_transition(replay_pose, action)
        if canonical_projective(score_observation(item, replay_pose)).get("success"):
            first_success = step
            break
    if first_success is None:
        return {**base, "status": "implementation_error", "error": "stored_certificate_never_reaches_canonical_success"}
    initial = {**initial, "actions": list(initial["actions"][:first_success]),
               "reverse_actions": list(initial["reverse_actions"][-first_success:]), "steps": first_success}
    upper = first_success
    certified = bool(lower["complete"] and not lower["runtime_shortcut_found"] and upper <= BUCKETS[requested][1])
    result = {**base, "proposed_reverse_depth": initial["steps"], "certificate_upper_bound": upper,
              "first_success_step": first_success, "certified_lower_bound": lower["certified_lower_bound"],
              "lower_bound_complete": lower["complete"], "runtime_shortcut_found": lower["runtime_shortcut_found"],
              "lower_bound_expansions": lower["expansions"], "lower_bound_visited_states": lower["visited_states"],
              "lower_bound_reason": lower.get("reason"), "target_metric": target_metric,
              "initial_metric": initial["metric"], "status": "difficulty_certified_candidate" if certified else "difficulty_unverified"}
    if certified:
        candidate = make_candidate(int(record["source_row_index"]), item, point, target, initial, split=str(record["split"]))
        candidate["generator_version"] = SELECTOR_VERSION
        candidate["reachability_construction"].update({"requested_bucket": requested,
            "certificate_upper_bound": upper, "certified_lower_bound": lower["certified_lower_bound"],
            "lower_bound_complete": True, "first_success_step": upper})
        result["row"] = candidate
        result["certificate"] = initial
    result["elapsed_seconds"] = time.time() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True,
                        help="JSON split->path map, or the frozen canary selection containing sources")
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-count", type=int, default=60)
    parser.add_argument("--hard-subset-count", type=int, default=12)
    parser.add_argument("--seed-cap", type=int, default=12)
    parser.add_argument("--per-seed-expansions", type=int, default=512)
    parser.add_argument("--candidate-cap-per-seed", type=int, default=48)
    parser.add_argument("--lower-expansions", type=int, default=100000)
    args = parser.parse_args()
    aggregate = json.loads(args.aggregate.read_text())
    frozen = choose_frozen_sources(aggregate, args.source_count, args.hard_subset_count)
    raw_sources = json.loads(args.sources.read_text())
    # The frozen selection is the authoritative source map and SHA inventory.
    source_map = raw_sources.get("sources", raw_sources)
    rows_by_split = {
        split: read_jsonl(Path(value["path"] if isinstance(value, dict) else value))
        for split, value in source_map.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "frozen_selection.json", frozen)
    hard_keys = {(row["split"], int(row["source_row_index"])) for row in frozen["hard_records"]}
    overrides_path = args.output_dir / "collision_overrides.json"
    overrides = json.loads(overrides_path.read_text()).get("structure_y_sign_overrides", {}) if overrides_path.is_file() else {}
    constraints = SceneConstraints(args.gs_root, structure_y_sign_overrides=overrides)
    results = []
    for record in frozen["records"]:
        item = rows_by_split[record["split"]][int(record["source_row_index"])]
        requested = ["easy", "medium"] + (["hard"] if stable_key(record) in hard_keys else [])
        for bucket in requested:
            try:
                result = run_one(item, record, bucket, constraints, seed_cap=args.seed_cap,
                                 per_seed_expansions=args.per_seed_expansions,
                                 candidate_cap=args.candidate_cap_per_seed, lower_expansions=args.lower_expansions)
            except Exception as error:
                result = {"version": SELECTOR_VERSION, "split": record["split"], "source_row_index": record["source_row_index"],
                          "scene_id": record["scene_id"], "requested_bucket": bucket, "status": "implementation_error", "error": repr(error)}
            results.append(result)
            write_json(args.output_dir / "checkpoint.json", {"version": SELECTOR_VERSION, "results": results})
            print(json.dumps({key: result.get(key) for key in ("split", "source_row_index", "requested_bucket", "status", "certificate_upper_bound", "certified_lower_bound", "elapsed_seconds")}), flush=True)
    status = Counter(row["status"] for row in results)
    payload = {"version": SELECTOR_VERSION, "path_first_base_version": PATH_FIRST_GENERATOR_VERSION,
               "frozen_selection": str(args.output_dir / "frozen_selection.json"), "same_pair_only": True,
               "budgets": {"seed_cap": args.seed_cap, "per_seed_expansions": args.per_seed_expansions,
                           "candidate_cap_per_seed": args.candidate_cap_per_seed, "lower_expansions": args.lower_expansions},
               "results": results, "status_counts": dict(status)}
    write_json(args.output_dir / "selector_results.json", payload)
    print(json.dumps({"source_count": frozen["source_count"], "requested_records": len(results), "status_counts": dict(status)}, indent=2))


if __name__ == "__main__":
    main()
