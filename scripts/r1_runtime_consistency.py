#!/usr/bin/env python3
"""Replay planner certificates through the formal ActiveSpatialEnv runtime."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task
from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def config(args: argparse.Namespace) -> ActiveSpatialEnvConfig:
    return ActiveSpatialEnvConfig(
        jsonl_path=str(args.repaired),
        total_lines=-1,
        render_backend="http",
        client_url=args.renderer_url,
        gs_root=str(args.gs_root),
        image_width=256,
        image_height=256,
        render_width=256,
        render_height=256,
        step_translation=0.3,
        step_rotation_deg=20.0,
        max_actions_per_step=1,
        action_space="strafe",
        enable_explicit_done=False,
        enable_potential_field=True,
        use_visual_bbox_scoring=True,
        potential_field_progress_mode="potential",
        potential_field_gamma=0.99,
        success_score_threshold=0.65,
        enable_auto_termination=True,
        max_episode_steps=args.max_steps,
        enable_collision_detection=True,
        collision_camera_radius=0.15,
        collision_floor_height=0.3,
        collision_ceiling_height=2.5,
        collision_safety_margin=0.05,
        collision_invalidate_action=True,
        max_consecutive_collisions=3,
        enable_low_info_frame_check=True,
        low_info_image_std_threshold=8.0,
        max_consecutive_low_info_frames=3,
        enable_visibility_check=True,
        prompt_format="free_think",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    repaired = read_jsonl(args.repaired)
    planner_rows = read_jsonl(args.reachability)
    if args.limit > 0:
        planner_rows = planner_rows[: args.limit]
    item_index = {row["task_id"]: index for index, row in enumerate(repaired)}
    planner_by_id = {row["task_id"]: row for row in planner_rows}
    env = ActiveSpatialEnv(config(args))
    output = []

    for task_id, planner in planner_by_id.items():
        index = item_index[task_id]
        item = repaired[index]
        observation, reset_info = env.reset(seed=index)
        del observation
        initial_pose = env.view_engine.get_pose()
        initial_metric = score_canonical_task(item, initial_pose)
        checks = {
            "initial_pose_matches_planner": bool(
                np.allclose(initial_pose, np.asarray(planner["path"][0]["c2w"]), atol=1e-7)
            ),
            "initial_not_success": not bool(initial_metric["success"]),
            "historical_backend_not_selected_without_version": False,
        }
        historical = dict(item)
        historical.pop("canonical_task_metric_version", None)
        current = env.current_item
        env.current_item = historical
        checks["historical_backend_not_selected_without_version"] = env._calculate_canonical_metric() is None
        env.current_item = current

        steps = []
        premature_done = False
        for step_index, action in enumerate(planner.get("actions", []), start=1):
            expected = planner["path"][step_index]
            obs, reward, done, info = env.step(f"<action>{action}</action>")
            del obs
            actual_pose = env.view_engine.get_pose()
            runtime_metric = score_canonical_task(item, actual_pose)
            info_metric = info.get("canonical_task_metric")
            pose_matches = np.allclose(
                actual_pose, np.asarray(expected["c2w"], dtype=float), atol=1e-7
            )
            metric_matches = (
                info_metric is not None
                and bool(info_metric["success"]) == bool(runtime_metric["success"])
                and abs(float(info_metric["score"]) - float(runtime_metric["score"])) <= 1e-9
            )
            if done and not runtime_metric["success"]:
                premature_done = True
            steps.append(
                {
                    "step": step_index,
                    "action": action,
                    "pose_matches_planner": bool(pose_matches),
                    "collision_count": int(info.get("collision_count", 0)),
                    "runtime_score": float(runtime_metric["score"]),
                    "runtime_success": bool(runtime_metric["success"]),
                    "info_metric_matches": bool(metric_matches),
                    "done": bool(done),
                    "auto_terminated": bool(info.get("auto_terminated", False)),
                    "low_info_frame": bool(info.get("low_info_frame", False)),
                    "reward": float(reward),
                }
            )
            if done:
                break
        checks.update(
            {
                "all_poses_match_planner": all(step["pose_matches_planner"] for step in steps),
                "all_collision_free": all(step["collision_count"] == 0 for step in steps),
                "all_metrics_match": all(step["info_metric_matches"] for step in steps),
                "no_premature_done": not premature_done,
                "terminal_success": bool(steps and steps[-1]["runtime_success"]),
                "terminal_done": bool(steps and steps[-1]["done"]),
                "within_budget": len(steps) <= args.max_steps,
            }
        )
        status = "passed" if all(checks.values()) else "failed"
        output.append(
            {
                "task_id": task_id,
                "scene_id": item["scene_id"],
                "status": status,
                "checks": checks,
                "planner_steps": planner.get("steps"),
                "runtime_steps": len(steps),
                "reset_info_keys": sorted(reset_info),
                "steps": steps,
            }
        )
        print(json.dumps({"task_id": task_id, "status": status, "steps": len(steps)}))

    counts = Counter(row["status"] for row in output)
    summary = {
        "rows": len(output),
        "status_counts": dict(counts),
        "all_passed": counts.get("passed", 0) == len(output),
        "runtime_backend": "version-selected canonical_spatial_task_h1_v1",
        "historical_compatibility": "rows without version keep SpatialPotentialField backend",
        "max_steps": args.max_steps,
        "renderer_url": args.renderer_url,
    }
    write_jsonl(args.output_dir / "runtime_consistency_manifest.jsonl", output)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
