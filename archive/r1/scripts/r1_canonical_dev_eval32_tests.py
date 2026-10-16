#!/usr/bin/env python3
"""Static contract tests for the frozen 32-source canonical policy eval."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.frozen_dir / "eval_protocol.json").read_text())
    policy = rows(args.frozen_dir / "policy_input_rows.jsonl")
    audit = rows(args.frozen_dir / "development_regression_manifest.jsonl")
    assert len(policy) == len(audit) == 32
    assert len({(x["split"], x["source_row_index"]) for x in audit}) == 32
    assert all(x["split"] != "train" for x in audit)
    assert Counter(x["split"] for x in audit) == {
        "id_test": 16, "ood_scene": 6, "ood_instance": 5,
        "validation_proxy": 3, "ood_geometry": 1, "ood_template": 1,
    }
    forbidden = {"actions", "certificate", "reachability_construction", "terminal_pose_c2w", "planner", "evidence"}
    assert all(not (set(row) & forbidden) for row in policy)
    assert all(row["canonical_task_metric_version"] == "canonical_spatial_task_h1_v1" for row in policy)
    assert all(
        row["camera_model_version"]
        == "canonical_h1_from_frozen_candidate_intrinsics_and_pose"
        for row in policy
    )
    assert all(row.get("collision_convention", {}).get("version") for row in policy)
    assert all(row["init_camera"]["extrinsics"] == audit[i]["initial_pose_c2w"] for i, row in enumerate(policy))
    assert protocol["environment"]["max_primitive_actions"] == 12
    assert protocol["policy_contract"]["history"].startswith("no_concat")
    assert protocol["environment"]["distance_to_sample_target_in_observation"] is False
    assert sha256(args.frozen_dir / "policy_input_rows.jsonl") == protocol["policy_input"]["sha256"]
    assert sha256(args.frozen_dir / "development_regression_manifest.jsonl") == protocol["audit_manifest"]["sha256"]
    print(json.dumps({"passed": True, "tests": 13, "rows": 32}, sort_keys=True))


if __name__ == "__main__":
    main()
