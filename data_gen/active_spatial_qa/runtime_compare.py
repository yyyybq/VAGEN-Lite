#!/usr/bin/env python3
"""Compare QA labels with the formal ActiveSpatialEnv scoring path.

This is an environment-side check, not a second scorer: it instantiates the
existing env and calls its own `_calculate_current_score` / canonical gate for
the exact pose serialized in each QA row.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig


def compare(bank: str, output: str, limit: int = 0):
    rows = [json.loads(x) for x in Path(bank).read_text().splitlines() if x.strip()]
    report = {"bank": bank, "rows": 0, "matches": 0, "mismatches": 0, "errors": 0, "skipped": 0, "details": []}
    for row in rows[:limit or None]:
        if row.get("private_answer") not in {"Yes", "No"}:
            report["skipped"] += 1
            continue
        try:
            audit = row.get("_audit", {}).get("predicate", {})
            raw_pose = row.get("state_pose_c2w") or audit.get("pose") or row.get("pose")
            pose = np.asarray(raw_pose, dtype=float) if raw_pose is not None else None
            if pose is None:
                # Older QA rows did not persist the pose. They remain useful for
                # label audits but cannot satisfy the environment-side check.
                raise ValueError("pose_not_serialized")
            config = dict(row.get("evaluation_config") or {})
            config.update(jsonl_path="", render_backend=None, enable_collision_detection=False,
                          enable_potential_field=True)
            env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**config))
            env.current_item = {
                **row.get("source_camera_metadata", {}),
                "task_type": row["task_type"],
                "task_params": row.get("complete_goal", {}).get("task_params", {}),
                "target_region": row.get("complete_goal", {}).get("target_region", {}),
                "target_object": row.get("complete_goal", {}).get("target_object"),
                "init_camera": row.get("source_init_camera") or {"intrinsics": row.get("camera", {}).get("intrinsics")},
                "scene_id": row.get("scene_id"),
                "canonical_task_metric_version": row.get("source_task_metric_version"),
                "camera_model_version": row.get("source_camera_model_version"),
            }
            env.current_task = {"task_type": env.current_item["task_type"], "task_params": env.current_item["task_params"], "target_region": env.current_item["target_region"]}
            env.camera_intrinsics = np.asarray(env.current_item["init_camera"]["intrinsics"], dtype=float)
            env.view_engine.reset(pose)
            canonical = env._calculate_canonical_metric()
            score = float(canonical["score"]) if canonical is not None else float(env._calculate_current_score())
            success = bool(canonical["success"]) if canonical is not None else score >= float(env.config.success_score_threshold)
            gold = row["private_answer"] == "Yes"
            report["rows"] += 1
            match = success == gold
            report["matches"] += int(match); report["mismatches"] += int(not match)
            report["details"].append({"sample_id": row.get("sample_id"), "qa_success": gold, "runtime_success": success, "runtime_backend": row.get("runtime_backend"), "runtime_score": score, "match": match})
        except Exception as exc:
            report["errors"] += 1
            report["details"].append({"sample_id": row.get("sample_id"), "error": str(exc)})
    report["status"] = "PASS" if report["rows"] > 0 and not report["errors"] and not report["mismatches"] else "FAIL"
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bank", required=True); ap.add_argument("--output", required=True); ap.add_argument("--limit", type=int, default=0)
    report = compare(**vars(ap.parse_args()))
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__": main()
