#!/usr/bin/env python3
"""Independently replay materialized Projective difficulty certificates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from r1_action_graph import TRANSLATION_ACTIONS, forward_transition
from r1_canonical_tasks import canonical_projective, score_observation
from r1_repair_pipeline import SceneConstraints

VERSION = "r1_projective_certificate_replay_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = {row["task_id"]: row for row in read_jsonl(args.candidates)}
    certificates = {row["task_id"]: row for row in read_jsonl(args.reachability)}
    if set(candidates) != set(certificates):
        raise ValueError("candidate and reachability task IDs do not match")
    constraints = SceneConstraints(args.gs_root)
    results = []
    for task_id in sorted(candidates):
        item, certificate = candidates[task_id], certificates[task_id]
        scene_id = str(item["scene_id"])
        constraints.scene(scene_id)
        detector = constraints._collision_cache[scene_id]
        pose = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
        initial = canonical_projective(score_observation(item, pose))
        collisions, mismatches = [], []
        first_success = None
        for step, action in enumerate(certificate["actions"], start=1):
            next_pose = forward_transition(pose, action)
            if action in TRANSLATION_ACTIONS:
                collision = detector.check_collision(next_pose[:3, 3], previous_position=pose[:3, 3])
                if collision.has_collision:
                    collisions.append({"step": step, "action": action, "type": collision.collision_type})
                    break
            expected = certificate.get("path", [])[step] if step < len(certificate.get("path", [])) else None
            expected_pose = expected.get("c2w") if expected else None
            if expected_pose is not None and not np.allclose(next_pose, np.asarray(expected_pose, dtype=float), atol=1e-7):
                mismatches.append({"step": step, "action": action, "max_abs_error": float(np.max(np.abs(next_pose - np.asarray(expected_pose, dtype=float))) )})
            pose = next_pose
            if canonical_projective(score_observation(item, pose))["success"]:
                first_success = step
                break
        status = "pass" if (not initial["success"] and not collisions and not mismatches
                              and first_success is not None
                              and first_success == int(certificate["first_success_step"])) else "fail"
        results.append({
            "task_id": task_id, "source_row_index": certificate["source_row_index"], "status": status,
            "steps": first_success, "initial_success": initial["success"], "final_success": first_success is not None,
            "collisions": collisions, "pose_mismatches": mismatches,
            "collision_convention": detector.convention_record(),
        })
    payload = {"version": VERSION, "rows": len(results),
               "passed": sum(row["status"] == "pass" for row in results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("version", "rows", "passed")}))


if __name__ == "__main__":
    main()
