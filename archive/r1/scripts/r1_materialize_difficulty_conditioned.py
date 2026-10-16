#!/usr/bin/env python3
"""Materialize certified difficulty-selector rows for independent validation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from r1_action_graph import forward_transition
from r1_canonical_tasks import canonical_projective, score_observation
from r1_reachability_audit import pose_record


VERSION = "r1_difficulty_conditioned_materialization_v1"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.selector.read_text())
    repaired, reachability, records = [], [], []
    seen: set[tuple[str, int, str]] = set()
    task_ids: set[str] = set()
    for result in source["results"]:
        key = (str(result["split"]), int(result["source_row_index"]), str(result["requested_bucket"]))
        if key in seen:
            raise ValueError(f"duplicate selector record {key}")
        seen.add(key)
        if result.get("status") != "difficulty_certified_candidate":
            continue
        item = copy.deepcopy(result["row"])
        item["task_id"] = f"{item['task_id']}_{key[2]}"
        if item["task_id"] in task_ids:
            raise ValueError(f"duplicate materialized task_id {item['task_id']}")
        task_ids.add(item["task_id"])
        certificate = result["certificate"]
        pose = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
        path = [pose_record(pose, None, canonical_projective(score_observation(item, pose)))]
        first_success_step = None
        for step, action in enumerate(certificate["actions"], start=1):
            pose = forward_transition(pose, action)
            metric = canonical_projective(score_observation(item, pose))
            path.append(pose_record(pose, action, metric))
            if first_success_step is None and metric["success"]:
                first_success_step = step
                # Extra actions cannot establish difficulty; discard them.
                break
        if first_success_step is None:
            raise ValueError(f"certificate never reaches success {key}")
        selected_upper = int(result["certificate_upper_bound"])
        upper = first_success_step
        repaired.append(item)
        reachability.append({
            "version": VERSION, "split": key[0], "source_row_index": key[1], "requested_bucket": key[2],
            "scene_id": item["scene_id"], "task_id": item["task_id"], "task_type": item["task_type"],
            "status": "reachable", "reachability_verified": False,
            "validation_pending": "independent_runtime_replay_and_official_rgb",
            "actions": list(certificate["actions"]), "steps": upper,
            "found_path_length_upper_bound": upper, "first_success_step": first_success_step,
            "generator_certificate_length_before_first_success_truncation": selected_upper,
            "certified_lower_bound": result["certified_lower_bound"], "lower_bound_complete": result["lower_bound_complete"],
            "path": path,
        })
        records.append({"key": list(key), "task_id": item["task_id"], "certificate_upper_bound": upper,
                        "certified_lower_bound": result["certified_lower_bound"],
                        "generator_certificate_length_before_first_success_truncation": selected_upper})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "candidate_rows.jsonl", repaired)
    write_jsonl(args.output_dir / "reachability_manifest.jsonl", reachability)
    payload = {"version": VERSION, "selector": str(args.selector), "requested_records": len(seen),
               "certified_candidates": len(repaired), "records": records}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
