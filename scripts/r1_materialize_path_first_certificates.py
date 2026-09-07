#!/usr/bin/env python3
"""Materialize path-first prototype candidates for independent validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from r1_action_graph import forward_transition
from r1_canonical_tasks import canonical_projective, score_observation
from r1_reachability_audit import pose_record


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repaired: list[dict] = []
    reachability: list[dict] = []
    seen: set[int] = set()
    for artifact in args.prototype:
        payload = json.loads(artifact.read_text())
        for result in payload.get("rows", []):
            if result.get("status") != "candidate_found":
                continue
            source_index = int(result["source_row_index"])
            if source_index in seen:
                raise ValueError(f"duplicate source row {source_index}")
            seen.add(source_index)
            item = result["row"]
            reverse = result["details"]["reverse"]
            pose = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
            metric = canonical_projective(score_observation(item, pose))
            path = [pose_record(pose, None, metric)]
            for action in reverse["actions"]:
                pose = forward_transition(pose, action)
                metric = canonical_projective(score_observation(item, pose))
                path.append(pose_record(pose, action, metric))
            repaired.append(item)
            reachability.append({
                "version": "r1_path_first_certificate_materialization_v1",
                "source_prototype": str(artifact),
                "source_row_index": source_index,
                "scene_id": item["scene_id"],
                "task_id": item["task_id"],
                "task_type": item["task_type"],
                "status": "reachable",
                "reachability_verified": False,
                "validation_pending": "independent_runtime_replay_and_official_rgb",
                "actions": list(reverse["actions"]),
                "steps": len(reverse["actions"]),
                "found_path_length_upper_bound": len(reverse["actions"]),
                "path": path,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "candidate_rows.jsonl", repaired)
    write_jsonl(args.output_dir / "reachability_manifest.jsonl", reachability)
    summary = {
        "version": "r1_path_first_certificate_materialization_v1",
        "candidate_count": len(repaired),
        "source_row_indices": sorted(seen),
        "prototypes": [str(value) for value in args.prototype],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
