#!/usr/bin/env python3
"""Projection-aware frontier ordering for one frozen Projective Medium A/B.

This version changes only deterministic expansion and eligible-candidate order.
Every predecessor is still admitted by the frozen forward transition, collision,
layout and state-key rules. Projection quality never filters an intermediate
runtime state and no RGB signal participates in search.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import r1_projective_medium_selector_v2 as v2
from r1_action_graph import INVERSE_ACTION, forward_validated_predecessors
from r1_canonical_tasks import canonical_projective, score_observation
from r1_projective_path_first_prototype import make_candidate, pair_midpoint, success_region_points, target_pose
from r1_reachability_audit import state_key
from r1_repair_pipeline import MAX_ABS_PITCH_DEG, SceneConstraints, absolute_pitch_degrees, projective_initial_geometry_discernible


VERSION = "projective_difficulty_conditioned_medium_selector_v4_projection_frontier"
FRONTIER_POLICY_VERSION = "projective_projection_frontier_order_v1"


def _area(box: Any) -> float:
    if not box or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def projection_quality(item: dict[str, Any], pose: np.ndarray, observation: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen, RGB-free lexicographic frontier quality signals."""
    objects = observation.get("visual_metrics", {}).get("objects") or []
    inside = []
    clipped_area_ratio = []
    for obj in objects[:2]:
        raw = _area(obj.get("bbox_raw"))
        clipped = _area(obj.get("bbox"))
        inside.append(clipped / raw if raw > 1e-8 else 0.0)
        clipped_area_ratio.append(clipped / (256.0 * 256.0))
    while len(inside) < 2:
        inside.append(0.0)
        clipped_area_ratio.append(0.0)
    midpoint = pair_midpoint(item)
    position = np.asarray(pose, dtype=float)[:3, 3]
    forward = np.asarray(pose, dtype=float)[:3, 2]
    direction = midpoint - position
    forward_norm = float(np.linalg.norm(forward))
    direction_norm = float(np.linalg.norm(direction))
    alignment = float(np.dot(forward, direction) / (forward_norm * direction_norm)) if forward_norm > 1e-12 and direction_norm > 1e-12 else -1.0
    return {
        "object_count": min(2, len(objects)),
        "in_front_count": sum(bool(obj.get("center_in_front")) for obj in objects[:2]),
        "visible_count": sum(bool(obj.get("visible")) for obj in objects[:2]),
        "min_inside_frame_fraction": float(min(inside)),
        "min_clipped_bbox_area_ratio": float(min(clipped_area_ratio)),
        "pair_alignment_cosine": max(-1.0, min(1.0, alignment)),
    }


def frontier_priority(depth: int, quality: dict[str, Any], pose: np.ndarray) -> tuple[Any, ...]:
    """Breadth first, then highest projection quality, then stable state key."""
    return (
        int(depth),
        -int(quality["object_count"]),
        -int(quality["in_front_count"]),
        -int(quality["visible_count"]),
        -float(quality["min_inside_frame_fraction"]),
        -float(quality["min_clipped_bbox_area_ratio"]),
        -float(quality["pair_alignment_cosine"]),
        tuple(state_key(pose)),
    )


def collect_projection_aware_reverse_candidates(
    item: dict[str, Any], target: np.ndarray, constraints: SceneConstraints, room_index: int | None,
    *, max_steps: int, max_expansions: int, candidate_cap: int,
) -> dict[str, Any]:
    detector = constraints._collision_cache.get(str(item.get("scene_id") or ""))
    if detector is None:
        return {"status": "asset_unavailable", "candidates": [], "expansions": 0, "visited_states": 0}
    target_observation = score_observation(item, target)
    target_quality = projection_quality(item, target, target_observation)
    serial = 0
    queue: list[tuple[Any, ...]] = [frontier_priority(0, target_quality, target) + (serial, target, tuple())]
    seen = {state_key(target)}
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    popped_quality: dict[int, list[float]] = defaultdict(list)
    expansions = 0
    status = "exhausted"
    while queue:
        entry = heapq.heappop(queue)
        depth, pose, reverse_actions = int(entry[0]), entry[-2], entry[-1]
        by_depth[depth]["popped"] += 1
        popped_quality[depth].append(float(-entry[4]))
        if depth >= max_steps:
            continue
        child_depth = depth + 1
        for edge in forward_validated_predecessors(pose, detector):
            predecessor = edge["predecessor"]
            if edge["rejection_reason"] is not None:
                reason = str(edge["rejection_reason"])
                rejected[reason] += 1
                by_depth[child_depth][reason] += 1
                continue
            layout = constraints.validate(item, predecessor[:3, 3], initial_room_index=room_index, check_pair_distance=False)
            if not layout.get("success"):
                rejected["layout_reject"] += 1
                by_depth[child_depth]["layout_reject"] += 1
                continue
            key = state_key(predecessor)
            if key in seen:
                rejected["quantization_mismatch"] += 1
                by_depth[child_depth]["dedup_state_key"] += 1
                continue
            seen.add(key)
            expansions += 1
            by_depth[child_depth]["unique_admitted"] += 1
            new_reverse = reverse_actions + (edge["inverse_proposal_action"],)
            observation = score_observation(item, predecessor)
            metric = canonical_projective(observation)
            quality = projection_quality(item, predecessor, observation)
            actions = [INVERSE_ACTION[value] for value in reversed(new_reverse)]
            if (not metric.get("success") and projective_initial_geometry_discernible(metric)
                    and absolute_pitch_degrees(predecessor[:3, 2]) <= MAX_ABS_PITCH_DEG):
                candidates.append({"pose": predecessor, "metric": metric, "layout": layout,
                                   "actions": actions, "reverse_actions": list(new_reverse),
                                   "steps": len(actions), "projection_quality": quality})
                by_depth[child_depth]["eligible_initial"] += 1
                if len(candidates) >= candidate_cap:
                    status = "candidate_cap"
                    break
            elif metric.get("success"):
                by_depth[child_depth]["canonical_success"] += 1
            else:
                by_depth[child_depth]["initial_projection_or_pitch_reject"] += 1
                for gate, passed in (metric.get("gates") or {}).items():
                    if not passed:
                        by_depth[child_depth][f"gate_{gate}"] += 1
            if expansions >= max_expansions:
                status = "unverified_expansion_cap"
                break
            serial += 1
            heapq.heappush(queue, frontier_priority(child_depth, quality, predecessor) + (serial, predecessor, new_reverse))
            by_depth[child_depth]["queued"] += 1
        if status != "exhausted":
            break
    frontier_depth = Counter(int(entry[0]) for entry in queue)
    telemetry = {}
    for depth in sorted(set(by_depth) | set(popped_quality)):
        values = popped_quality.get(depth, [])
        telemetry[str(depth)] = {
            **dict(sorted(by_depth[depth].items())),
            "popped_min_inside_frame_fraction": min(values) if values else None,
            "popped_mean_inside_frame_fraction": sum(values) / len(values) if values else None,
            "popped_max_inside_frame_fraction": max(values) if values else None,
        }
    return {
        "status": status, "candidates": candidates, "expansions": expansions,
        "visited_states": len(seen), "rejection_reasons": dict(rejected),
        "frontier_policy_version": FRONTIER_POLICY_VERSION,
        "frontier_at_stop": {"count": sum(frontier_depth.values()),
                             "by_depth": dict(sorted(frontier_depth.items())),
                             "minimum_depth": min(frontier_depth, default=None)},
        "by_depth": telemetry,
    }


def _candidate_rank(candidate: dict[str, Any], upper: int) -> tuple[Any, ...]:
    quality = candidate["projection_quality"]
    return (abs(upper - 5),) + frontier_priority(0, quality, candidate["pose"])[1:]


def one(item: dict[str, Any], rec: dict[str, Any], constraints: SceneConstraints, cfg: Any) -> dict[str, Any]:
    started = time.time()
    layout, _ = constraints.scene(str(item.get("scene_id") or ""))
    room = constraints.room_index(layout, pair_midpoint(item)[:2])
    if room is None:
        room = constraints.validate(item, np.asarray(item["init_camera"]["extrinsics"])[:3, 3], check_pair_distance=False).get("room_index")
    targets = []
    for point in success_region_points(item, constraints, room):
        if not constraints.validate(item, point, initial_room_index=room, check_pair_distance=False).get("success"):
            continue
        for yaw in (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0):
            pose = target_pose(item, point, yaw)
            metric = canonical_projective(score_observation(item, pose))
            if metric["success"]:
                targets.append((point, pose, metric))
    targets = targets[:cfg.seed_cap]
    pool = []
    reverse = []
    for seed_index, (point, target, metric) in enumerate(targets):
        result = collect_projection_aware_reverse_candidates(
            item, target, constraints, room, max_steps=12,
            max_expansions=cfg.per_seed_expansions, candidate_cap=cfg.candidate_cap,
        )
        reverse.append({
            "seed_index": seed_index, "status": result["status"], "expansions": result["expansions"],
            "visited_states": result["visited_states"], "candidate_count": len(result["candidates"]),
            "rejection_reasons": result.get("rejection_reasons", {}),
            "frontier_at_stop": result.get("frontier_at_stop"), "by_depth": result.get("by_depth", {}),
        })
        pool.extend((point, target, metric, candidate) for candidate in result["candidates"])
    detector = constraints._collision_cache[str(item.get("scene_id") or "")]
    eligible = []
    for point, target, metric, candidate in pool:
        upper = v2.first_success(item, candidate)
        if upper is not None and 4 <= upper <= 6:
            chosen = {**candidate, "actions": candidate["actions"][:upper], "steps": upper}
            eligible.append((_candidate_rank(chosen, upper), point, target, metric, chosen))
    eligible.sort(key=lambda value: value[0])
    events = []
    accepted = None
    unverified = False
    for rank, point, target, metric, candidate in eligible:
        probe = v2.shortcut_probe(item, candidate["pose"], detector, cfg.lower_expansions)
        events.append({"candidate_rank": list(rank[:-1]), "projection_quality": candidate["projection_quality"],
                       "certificate_upper_bound": candidate["steps"], "initial_metric": candidate["metric"],
                       "shortcut_probe": probe})
        if not probe["complete"]:
            unverified = True
            continue
        if probe["shortcut_found"]:
            continue
        accepted = (point, target, metric, candidate, probe)
        break
    base = {
        "version": VERSION, "frontier_policy_version": FRONTIER_POLICY_VERSION,
        "split": rec["split"], "source_row_index": rec["source_row_index"], "scene_id": rec["scene_id"],
        "requested_bucket": "medium", "same_pair_only": True,
        "success_region_seeds_considered": len(targets), "reverse_seed_cap": cfg.seed_cap,
        "per_seed_expansion_cap": cfg.per_seed_expansions, "candidate_cap_per_seed": cfg.candidate_cap,
        "reverse_attempts": reverse, "initial_candidates_collected": len(pool),
        "eligible_medium_candidates": len(eligible), "pre_screen_events": events,
        "source_action_length": rec.get("source_action_length", "unknown"),
        "baseline_certificate_upper_bound": rec.get("baseline_certificate_upper_bound", "unknown"),
    }
    if accepted:
        point, target, target_metric, candidate, probe = accepted
        row = make_candidate(int(rec["source_row_index"]), item, point, target, candidate, split=str(rec["split"]))
        row["generator_version"] = VERSION
        row["reachability_construction"].update({"requested_bucket": "medium",
            "certificate_upper_bound": candidate["steps"], "certified_lower_bound": 4,
            "lower_bound_complete": True, "first_success_step": candidate["steps"]})
        output = {**base, "status": "difficulty_certified_candidate",
                  "certificate_upper_bound": candidate["steps"], "certified_lower_bound": 4,
                  "lower_bound_complete": True, "runtime_shortcut_found": False,
                  "initial_metric": candidate["metric"], "target_metric": target_metric,
                  "row": row, "certificate": candidate}
    elif eligible and not unverified:
        output = {**base, "status": "shortcut_rejected", "runtime_shortcut_found": True}
    elif unverified:
        output = {**base, "status": "difficulty_unverified"}
    else:
        output = {**base, "status": "requested_bucket_not_found_within_budget"}
    output["elapsed_seconds"] = time.time() - started
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-cap", type=int, default=12)
    parser.add_argument("--per-seed-expansions", type=int, default=512)
    parser.add_argument("--candidate-cap-per-seed", type=int, default=48)
    parser.add_argument("--lower-expansions", type=int, default=100000)
    parser.add_argument("--source-keys")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    raw_sources = json.loads(args.sources.read_text())
    source_map = raw_sources.get("sources", raw_sources)
    sources = {key: v2.read_jsonl(Path(value["path"] if isinstance(value, dict) else value)) for key, value in source_map.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = type("Cfg", (), {"seed_cap": args.seed_cap, "per_seed_expansions": args.per_seed_expansions,
                            "candidate_cap": args.candidate_cap_per_seed, "lower_expansions": args.lower_expansions})()
    constraints = SceneConstraints(args.gs_root)
    requested = {(value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1])) for value in (args.source_keys or "").split(",") if value}
    rows = []
    for rec in selection["records"]:
        if requested and (rec["split"], int(rec["source_row_index"])) not in requested:
            continue
        try:
            result = one(sources[rec["split"]][int(rec["source_row_index"])], rec, constraints, cfg)
        except Exception as error:
            result = {"version": VERSION, "split": rec["split"], "source_row_index": rec["source_row_index"],
                      "scene_id": rec["scene_id"], "requested_bucket": "medium",
                      "status": "implementation_error", "error": repr(error)}
        rows.append(result)
        v2.write_json(args.output_dir / "checkpoint.json", {"version": VERSION, "results": rows})
        print(json.dumps({key: result.get(key) for key in ("split", "source_row_index", "status", "eligible_medium_candidates", "elapsed_seconds")}), flush=True)
    v2.write_json(args.output_dir / "selector_results.json", {
        "version": VERSION, "frontier_policy_version": FRONTIER_POLICY_VERSION,
        "selection": str(args.selection), "same_pair_only": True,
        "budgets": {"seed_cap": args.seed_cap, "per_seed_expansions": args.per_seed_expansions,
                    "candidate_cap_per_seed": args.candidate_cap_per_seed, "lower_expansions": args.lower_expansions},
        "frontier_order": ["depth_ascending", "object_count_descending", "in_front_count_descending",
                           "visible_count_descending", "min_inside_frame_fraction_descending",
                           "min_clipped_bbox_area_ratio_descending", "pair_alignment_cosine_descending",
                           "state_key_ascending"],
        "results": rows, "status_counts": dict(Counter(row["status"] for row in rows)),
    })


if __name__ == "__main__":
    main()
