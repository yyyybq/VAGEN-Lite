#!/usr/bin/env python3
"""Audit local FOV score behavior around versioned repaired targets."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from r1_canonical_tasks import canonical_fov, pose_from_item_target, score_observation
from vagen.envs.active_spatial.canonical_camera import camera_pose_from_forward, normalize_vector


def rotate_xy(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    c, s = float(np.cos(radians)), float(np.sin(radians))
    out = vector.copy()
    out[:2] = (c * vector[0] - s * vector[1], s * vector[0] + c * vector[1])
    return normalize_vector(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.pilot.open() if line.strip()][: args.limit]
    records = []
    for index, item in enumerate(rows):
        pose = pose_from_item_target(item)
        position = np.asarray(pose[:3, 3], dtype=float)
        forward = normalize_vector(np.asarray(pose[:3, 2], dtype=float))
        right = normalize_vector(np.asarray(pose[:3, 0], dtype=float))
        variants = {"base": pose}
        for angle in (10.0, 20.0, 30.0):
            variants[f"yaw_plus_{int(angle)}"] = camera_pose_from_forward(position, rotate_xy(forward, angle))
            variants[f"yaw_minus_{int(angle)}"] = camera_pose_from_forward(position, rotate_xy(forward, -angle))
        for distance in (0.25, 0.50):
            variants[f"strafe_right_{distance:.2f}"] = camera_pose_from_forward(position + right * distance, forward)
            variants[f"forward_{distance:.2f}"] = camera_pose_from_forward(position + forward * distance, forward)

        metrics = {name: canonical_fov(score_observation(item, c2w)) for name, c2w in variants.items()}
        records.append({"pilot_index": index, "task_id": item.get("task_id"), "scene_id": item.get("scene_id"), "metrics": metrics})

    def yaw_nonincreasing(record: dict[str, object], sign: str) -> bool:
        labels = ["base", f"yaw_{sign}_10", f"yaw_{sign}_20", f"yaw_{sign}_30"]
        values = [record["metrics"][label]["score"] for label in labels]
        return all(a + 1e-9 >= b for a, b in zip(values, values[1:]))

    summary = {
        "metric_version": "canonical_spatial_task_h1_v1",
        "rows": len(records),
        "base_success": sum(bool(r["metrics"]["base"]["success"]) for r in records),
        "yaw_plus_nonincreasing": sum(yaw_nonincreasing(r, "plus") for r in records),
        "yaw_minus_nonincreasing": sum(yaw_nonincreasing(r, "minus") for r in records),
        "mean_scores": {
            name: statistics.mean(r["metrics"][name]["score"] for r in records)
            for name in records[0]["metrics"]
        } if records else {},
        "note": "Yaw is expected to reduce a centered target score. Strafe/forward values are diagnostic because FOV framing is not globally monotonic in translation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
