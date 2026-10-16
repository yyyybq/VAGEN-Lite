#!/usr/bin/env python3
"""Stepwise known-certificate test for the R1 Projective action graph."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from r1_action_graph import ACTIONS, forward_transition, forward_validated_predecessors, pose_error
from r1_reachability_audit import state_key
from r1_repair_pipeline import SceneConstraints


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def serialise_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in candidate.items() if key not in {"predecessor", "forward_replay"}},
        "predecessor": candidate["predecessor"].tolist(),
        "forward_replay": candidate["forward_replay"].tolist(),
        "predecessor_state_key": list(state_key(candidate["predecessor"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {row["task_id"]: row for row in read_jsonl(args.rows)}
    certificates = read_jsonl(args.reachability)
    constraints = SceneConstraints(args.gs_root)
    results: list[dict[str, Any]] = []
    all_action_roundtrips: Counter[str] = Counter()
    started = time.time()

    for certificate in certificates:
        item = rows.get(certificate["task_id"])
        result: dict[str, Any] = {
            "task_id": certificate["task_id"],
            "source_row_index": certificate.get("source_row_index"),
            "status": "error",
            "steps": [],
        }
        if item is None:
            result["reason"] = "task_not_found"
            results.append(result)
            continue
        layout, _ = constraints.scene(str(item["scene_id"]))
        detector = constraints._collision_cache.get(str(item["scene_id"]))
        path = certificate.get("path") or []
        if detector is None or not path:
            result["reason"] = "asset_or_certificate_unavailable"
            results.append(result)
            continue
        room_index = constraints.room_index(layout, np.asarray(path[0]["position"], dtype=float)[:2])
        # Explicitly test every formal action on a known c2w state, including
        # move_left which is absent from the saved five certificates.
        probe = np.asarray(path[0]["c2w"], dtype=float)
        synthetic = []
        for action in ACTIONS:
            next_pose = forward_transition(probe, action)
            reverse = next(
                row for row in forward_validated_predecessors(next_pose, detector)
                if row["forward_action"] == action
            )
            matches_probe = state_key(reverse["predecessor"]) == state_key(probe)
            synthetic.append({
                "action": action,
                "roundtrip_matches_probe": matches_probe,
                "pose_error": pose_error(reverse["predecessor"], probe),
                "forward_state_key_match": reverse["forward_state_key_match"],
                "rejection_reason": reverse["rejection_reason"],
            })
            if matches_probe and reverse["forward_state_key_match"]:
                all_action_roundtrips[action] += 1
        result["all_action_roundtrips"] = synthetic

        for step_index, expected in enumerate(path[1:], start=1):
            previous = np.asarray(path[step_index - 1]["c2w"], dtype=float)
            current = np.asarray(expected["c2w"], dtype=float)
            action = str(expected["action"])
            forward = forward_transition(previous, action)
            candidates = forward_validated_predecessors(current, detector)
            expected_candidate = next(row for row in candidates if row["forward_action"] == action)
            predecessor_error = pose_error(expected_candidate["predecessor"], previous)
            predecessor_contains_expected = (
                state_key(expected_candidate["predecessor"]) == state_key(previous)
                and expected_candidate["forward_state_key_match"]
                and not expected_candidate["collision"]["has_collision"]
            )
            previous_layout = constraints.validate(
                item, previous[:3, 3], initial_room_index=room_index, check_pair_distance=False
            )
            current_layout = constraints.validate(
                item, current[:3, 3], initial_room_index=room_index, check_pair_distance=False
            )
            forward_error = pose_error(forward, current)
            rejection = expected_candidate["rejection_reason"]
            if not predecessor_contains_expected and rejection is None:
                rejection = "quantization_mismatch"
            result["steps"].append({
                "step": step_index,
                "action": action,
                "expected_predecessor": previous.tolist(),
                "generated_predecessor_candidates": [serialise_candidate(row) for row in candidates],
                "forward_replay_result": forward.tolist(),
                "forward_pose_error": forward_error,
                "state_key_error": state_key(forward) != state_key(current),
                "expected_predecessor_pose_error": predecessor_error,
                "predecessors_contains_expected": predecessor_contains_expected,
                "collision_result": expected_candidate["collision"],
                "previous_layout": previous_layout,
                "current_layout": current_layout,
                "rejection_reason": rejection,
            })
        # Reconstruct the entire certificate backwards through the same graph,
        # beginning at the terminal success state.  This is intentionally not
        # a search budget test: it proves that the saved certificate's edges
        # exist in the reverse graph defined by forward runtime dynamics.
        current = np.asarray(path[-1]["c2w"], dtype=float)
        reverse_chain = []
        for step_index in range(len(path) - 1, 0, -1):
            expected_previous = np.asarray(path[step_index - 1]["c2w"], dtype=float)
            action = str(path[step_index]["action"])
            edge = next(
                row for row in forward_validated_predecessors(current, detector)
                if row["forward_action"] == action
            )
            matches = (
                state_key(edge["predecessor"]) == state_key(expected_previous)
                and edge["forward_state_key_match"]
                and not edge["collision"]["has_collision"]
            )
            reverse_chain.append({
                "reverse_step": len(path) - step_index + 1,
                "forward_action": action,
                "matches_expected_predecessor": matches,
                "pose_error": pose_error(edge["predecessor"], expected_previous),
                "rejection_reason": edge["rejection_reason"],
            })
            current = edge["predecessor"]
        result["certificate_graph_reconstruction"] = {
            "terminal_to_initial": reverse_chain,
            "matches_saved_initial": state_key(current) == state_key(np.asarray(path[0]["c2w"], dtype=float)),
            "passed": bool(reverse_chain) and all(row["matches_expected_predecessor"] for row in reverse_chain),
        }
        result["status"] = "pass" if result["steps"] and all(
            row["predecessors_contains_expected"]
            and row["previous_layout"]["success"]
            and row["current_layout"]["success"]
            for row in result["steps"]
        ) and result["certificate_graph_reconstruction"]["passed"] else "fail"
        results.append(result)

    payload = {
        "version": "r1_projective_action_graph_audit_v1",
        "action_graph_truth": "formal_forward_transition_plus_forward_collision_v1",
        "constraint_scope": {
            "all_navigation_states": ["room", "wall_clearance", "object_collision", "frozen_collision_convention"],
            "initial_only": ["canonical_fail", "initial_dual_object_geometry", "initial_observability"],
            "success_target_only": ["canonical_success", "12px_relation_margin", "target_pair_min_distance", "target_pose_selection"],
        },
        "known_certificates": len(results),
        "passed_certificates": sum(row["status"] == "pass" for row in results),
        "all_action_roundtrip_passes": dict(all_action_roundtrips),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "known_certificates", "passed_certificates", "all_action_roundtrip_passes", "elapsed_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
