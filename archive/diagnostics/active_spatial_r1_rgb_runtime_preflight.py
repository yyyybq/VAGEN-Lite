#!/usr/bin/env python3
"""Live H1/RGB gate for the frozen diagnostic policy rows.

Run only inside the authorized capture worker.  It calls the same
``ActiveSpatialEnv`` runtime used by PPO, before policy sampling begins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    policy = [json.loads(line) for line in args.policy_jsonl.read_text().splitlines() if line.strip()]
    audit = [json.loads(line) for line in args.audit_jsonl.read_text().splitlines() if line.strip()]
    if len(policy) != 8 or len(audit) != 8:
        raise RuntimeError("diagnostic RGB gate requires exactly eight frozen rows")
    config = ActiveSpatialEnvConfig(
        jsonl_path=str(args.policy_jsonl), render_backend="http", client_url=args.renderer_url,
        gs_root=str(args.gs_root), image_width=256, image_height=256,
        step_translation=0.3, step_rotation_deg=20.0, action_space="strafe",
        enable_explicit_done=False, enable_auto_termination=True, max_actions_per_step=5,
        max_episode_steps=12, turn_budget=12, enable_potential_field=True,
        potential_field_progress_mode="potential", potential_field_gamma=0.95,
        enable_potential_shaping_reward=True, enable_near_success_reward=False,
        enable_visibility_shaping_reward=False, success_reward=5.0,
        success_score_threshold=0.65, render_fail_fast=True,
    )
    env = ActiveSpatialEnv(config)
    rows, failures = [], []
    try:
        for index, audit_row in enumerate(audit):
            obs, _ = env.reset(seed=index)
            actual = list((obs.get("multi_modal_data") or {}).values())
            images = [image for group in actual for image in (group or [])]
            if len(images) != 1:
                raise RuntimeError(f"row {index}: expected one current image, got {len(images)}")
            metric = env._calculate_canonical_metric()
            evidence = audit_row["evidence_audit_only"]["observability"]
            if sha256(Path(evidence["path"])) != evidence["sha256"]:
                raise RuntimeError(f"row {index}: observability manifest SHA mismatch")
            frozen_row = [json.loads(line) for line in Path(evidence["path"]).read_text().splitlines() if line.strip()][int(evidence["record_index"])]
            frozen = np.asarray(Image.open(frozen_row["frame_images"][0]).convert("RGB"), dtype=np.int16)
            current = np.asarray(images[0].convert("RGB"), dtype=np.int16)
            if frozen.shape != current.shape:
                raise RuntimeError(f"row {index}: RGB shape mismatch {current.shape} != {frozen.shape}")
            error = np.abs(current - frozen)
            mae, p99 = float(error.mean()), float(np.quantile(error, 0.99))
            passed = mae <= 0.5 and p99 <= 2.0 and bool(metric) and not bool(metric["success"])
            item = env.current_item or {}
            row = {"index": index, "source_key": audit_row["source_key"], "passed": passed,
                   "mae": mae, "p99_abs": p99, "max_abs": int(error.max()),
                   "initial_canonical_success": bool(metric and metric["success"]),
                   "camera_model_version": item.get("camera_model_version"),
                   "canonical_task_metric_version": item.get("canonical_task_metric_version"),
                   "collision_convention": item.get("collision_convention")}
            rows.append(row)
            if not passed:
                failures.append(row)
    finally:
        env.close()
    report = {"status": "PASS" if not failures else "BLOCKED", "rows": rows, "failures": failures,
              "frozen_tolerance": {"mae": 0.5, "p99_abs": 2.0},
              "runtime_contract": "PPO ActiveSpatialEnv / H1 / canonical / frozen collision / 12 primitive cap"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
