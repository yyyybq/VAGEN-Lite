#!/usr/bin/env python3
"""Replay saved action certificates with the current canonical/runtime rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from r1_canonical_tasks import canonical_projective, score_observation
from vagen.envs.active_spatial.collision_detector import create_collision_detector
from vagen.envs.active_spatial.utils import ViewManipulator


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {row["task_id"]: row for row in read_jsonl(args.rows)}
    detector = create_collision_detector({
        "camera_radius": 0.15, "floor_height": 0.3, "ceiling_height": 2.5,
        "safety_margin": 0.05, "enable_object_collision": True,
        "enable_boundary_collision": True,
    })
    results = []
    for certificate in read_jsonl(args.reachability):
        task_id = certificate["task_id"]
        item = rows.get(task_id)
        result = {"task_id": task_id, "source_row_index": certificate.get("source_row_index"), "status": "error"}
        if item is None:
            result["reason"] = "task_not_found"
            results.append(result)
            continue
        if not detector.load_scene_from_gs_root(str(args.gs_root), str(item["scene_id"])):
            result["reason"] = "asset_load_failed"
            results.append(result)
            continue
        path = certificate.get("path") or []
        if not path:
            result["reason"] = "empty_certificate"
            results.append(result)
            continue
        engine = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
        pose = np.asarray(path[0]["c2w"], dtype=float)
        initial_metric = canonical_projective(score_observation(item, pose))
        collisions = []
        pose_mismatches = []
        for expected in path[1:]:
            action = expected.get("action")
            previous = pose
            engine.reset(previous)
            pose = engine.step(action)
            expected_pose = np.asarray(expected["c2w"], dtype=float)
            if not np.allclose(pose, expected_pose, atol=1e-6):
                pose_mismatches.append(action)
            if action.startswith("move_"):
                collision = detector.check_collision(pose[:3, 3], previous_position=previous[:3, 3])
                if collision.has_collision:
                    collisions.append({"action": action, "type": collision.collision_type})
        final_metric = canonical_projective(score_observation(item, pose))
        result.update({
            "status": "pass" if (
                not initial_metric["success"] and final_metric["success"]
                and not collisions and not pose_mismatches
            ) else "fail",
            "steps": len(path) - 1,
            "initial_success": bool(initial_metric["success"]),
            "final_success": bool(final_metric["success"]),
            "initial_relation_margin_px": initial_metric.get("relation_margin_px"),
            "final_relation_margin_px": final_metric.get("relation_margin_px"),
            "collisions": collisions,
            "pose_mismatches": pose_mismatches,
            "collision_convention": detector.convention_record(),
        })
        results.append(result)
    payload = {
        "version": "r1_projective_certificate_replay_v1",
        "rows": len(results),
        "passed": sum(row["status"] == "pass" for row in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
