#!/usr/bin/env python3
"""Independently replay and re-prove frozen FOV min-4 certificates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from r1_action_graph import TRANSLATION_ACTIONS, forward_transition
from r1_canonical_tasks import canonical_fov, score_observation
from r1_repair_pipeline import SceneConstraints, canonical_fov_depth_lower_bound


VERSION = "r1_fov_certificate_replay_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {
        row["task_id"]: row for row in read_jsonl(args.candidates)
        if row.get("task_type") == "fov_inclusion"
    }
    certificates = {
        row["task_id"]: row for row in read_jsonl(args.reachability)
        if row.get("task_id") in candidates and row.get("status") == "reachable"
    }
    if set(candidates) != set(certificates):
        raise ValueError("accepted FOV candidate and reachable certificate IDs do not match")
    constraints = SceneConstraints(args.gs_root)
    results = []
    for task_id in sorted(candidates):
        item, certificate = candidates[task_id], certificates[task_id]
        scene_id = str(item["scene_id"])
        constraints.scene(scene_id)
        detector = constraints._collision_cache[scene_id]
        pose = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
        initial = canonical_fov(score_observation(item, pose))
        lower = canonical_fov_depth_lower_bound(item, pose, detector, 3)
        collisions, mismatches = [], []
        first_success = None
        actions = list(certificate.get("actions") or [])
        for step, action in enumerate(actions, start=1):
            next_pose = forward_transition(pose, action)
            if action in TRANSLATION_ACTIONS:
                collision = detector.check_collision(
                    next_pose[:3, 3], previous_position=pose[:3, 3]
                )
                if collision.has_collision:
                    collisions.append({"step": step, "action": action,
                                       "type": collision.collision_type})
                    break
            expected = certificate.get("path", [])[step] if step < len(certificate.get("path", [])) else None
            expected_pose = expected.get("c2w") if expected else None
            if expected_pose is not None and not np.allclose(
                next_pose, np.asarray(expected_pose, dtype=float), atol=1e-7
            ):
                mismatches.append(
                    {"step": step, "action": action,
                     "max_abs_error": float(np.max(np.abs(next_pose - np.asarray(expected_pose, dtype=float))))}
                )
            pose = next_pose
            if canonical_fov(score_observation(item, pose))["success"]:
                first_success = step
                break
        status = "pass" if (
            not initial["success"]
            and bool(lower.get("complete"))
            and bool(lower.get("no_success_through_depth"))
            and not collisions
            and not mismatches
            and first_success is not None
            and 4 <= first_success <= 12
        ) else "fail"
        results.append(
            {
                "task_id": task_id,
                "source_row_index": certificate.get("source_row_index"),
                "status": status,
                "steps": first_success,
                "initial_success": initial["success"],
                "lower_bound_complete": lower.get("complete"),
                "no_success_through_depth_3": lower.get("no_success_through_depth"),
                "lower_bound_audit": lower,
                "collisions": collisions,
                "pose_mismatches": mismatches,
                "collision_convention": detector.convention_record(),
            }
        )
    payload = {
        "version": VERSION,
        "rows": len(results),
        "passed": sum(row["status"] == "pass" for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: payload[key] for key in ("version", "rows", "passed")}))


if __name__ == "__main__":
    main()
