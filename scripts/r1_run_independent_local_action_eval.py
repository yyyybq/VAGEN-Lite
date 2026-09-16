#!/usr/bin/env python3
"""Run one oracle-free policy decision on frozen independent d*=1/2 states."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r1_canonical_tasks import inside_fraction, score_observation
from r1_run_canonical_dev_eval32 import (
    VllmPolicy, atomic_json, env_config, jsonable, read_jsonl,
    save_observation_image, sha256, stable_hash, verify_model_sha256s,
)
from r1_run_canonical_dev_eval32_floor_diagnostics import (
    compare_prior_rgb, effective_camera, execute_first_only, parse,
    policy_input_record,
)


VERSION = "r1_independent_local_action_runner_v1"
SMOKE = (0, 1, 2, 3)


def projection_details(item: dict[str, Any], pose: np.ndarray) -> dict[str, Any]:
    scored = score_observation(item, np.asarray(pose, dtype=float))
    vm = scored["visual_metrics"]
    objects = vm.get("objects") or []
    fractions = [inside_fraction(obj) for obj in objects]
    return {
        "relation_satisfied": bool(vm.get("visual_relation_satisfied")),
        "relation_margin_px": vm.get("visual_relation_margin_px"),
        "inside_frame_fraction_min": min(fractions) if fractions else 0.0,
        "inside_frame_fractions": fractions,
        "object_area_ratios": [float(obj.get("area_ratio", 0.0) or 0.0) for obj in objects],
        "object_visible": [bool(obj.get("visible")) for obj in objects],
        "object_in_front": [bool(obj.get("center_in_front")) for obj in objects],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    by_distance: dict[str, Counter] = defaultdict(Counter)
    by_scene: dict[str, Counter] = defaultdict(Counter)
    actions = Counter()
    parents = defaultdict(lambda: {"states": 0, "hits": 0})
    for row in complete:
        distance = f"d{row['exact_distance_audit_only']}"
        by_distance[distance]["states"] += 1
        by_distance[distance]["optimal_first_action_hits"] += int(row["optimal_first_action_hit"])
        by_distance[distance]["invalid_or_empty"] += int(row["first_parsed_action"] is None)
        by_distance[distance]["collisions"] += int(row["collision_attempts"])
        scene = row["scene_id"]
        by_scene[scene]["states"] += 1
        by_scene[scene]["hits"] += int(row["optimal_first_action_hit"])
        parents[row["source_key"]]["states"] += 1
        parents[row["source_key"]]["hits"] += int(row["optimal_first_action_hit"])
        actions[str(row.get("first_parsed_action"))] += 1
    return {
        "requested": len(rows), "complete": len(complete),
        "infrastructure_errors": len(rows) - len(complete),
        "by_distance": {key: dict(value) for key, value in sorted(by_distance.items())},
        "by_scene": {key: dict(value) for key, value in sorted(by_scene.items())},
        "parent_sources": len(parents), "source_grouped": dict(sorted(parents.items())),
        "actions": dict(sorted(actions.items())),
    }


def run_one(
    env: Any, policy: VllmPolicy, index: int, audit: dict[str, Any],
    seed_base: int, attempt_dir: Path,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=index)
    identity = (env.current_item or {}).get("source_identity", {})
    if int(identity.get("local_state_index", -1)) != index:
        raise RuntimeError(f"wrong independent state loaded at {index}: {identity}")
    pose_before = env.view_engine.get_pose().copy()
    expected = np.asarray(audit["pose_c2w"], dtype=float)
    pose_error = float(np.max(np.abs(pose_before - expected)))
    if pose_error > 1e-8:
        raise RuntimeError(f"independent pose mismatch at {index}: {pose_error}")
    before = env._calculate_canonical_metric()
    if before is None or before.get("success"):
        raise RuntimeError(f"independent state already successful at {index}")
    initial_camera = effective_camera(env)
    projection_before = projection_details(env.current_item, pose_before)
    frame = save_observation_image(obs, attempt_dir / "initial.png")
    rgb = compare_prior_rgb(frame["image"], audit["prior_rgb_audit_only"])
    seed = seed_base + index
    input_record = policy_input_record(env, obs, frame, seed)
    raw, inference_seconds = policy.generate(
        input_record["system_text"], obs["obs_str"], frame["image"], seed,
    )
    original_parse = parse(env, raw)
    result, executed_raw, executed, discarded = execute_first_only(env, raw)
    pose_after = env.view_engine.get_pose().copy()
    after = env._calculate_canonical_metric()
    projection_after = projection_details(env.current_item, pose_after)
    post_frame = save_observation_image(result["obs"], attempt_dir / "after_first_action.png")
    turn_metrics = (result["info"].get("metrics") or {}).get("turn_metrics") or {}
    first = executed[0] if executed else None
    return {
        "version": VERSION, "status": "complete", "local_state_index": index,
        "parent_eval_index": audit["parent_eval_index"], "source_key": audit["source_key"],
        "split": audit["split"], "scene_id": audit["scene_id"],
        "r1_development_seen": audit["r1_development_seen"],
        "v46_training_scene_seen": audit["v46_training_scene_seen"],
        "v46_training_exact_source_seen": audit["v46_training_exact_source_seen"],
        "exact_distance_audit_only": audit["exact_distance"],
        "optimal_first_actions_audit_only": audit["optimal_first_actions_audit_only"],
        "model_input_oracle_free": True, "pose_max_abs_error": pose_error,
        "initial_rgb_evidence": rgb, "effective_camera_initial": initial_camera,
        "input": input_record, "raw_completion": raw,
        "original_parse": jsonable(original_parse), "executed_raw_completion": executed_raw,
        "first_parsed_action": first, "discarded_actions": discarded,
        "optimal_first_action_hit": first in set(audit["optimal_first_actions_audit_only"]),
        "pose_before_c2w": pose_before.tolist(), "pose_after_c2w": pose_after.tolist(),
        "canonical_before": jsonable(before), "canonical_after": jsonable(after),
        "projection_before": jsonable(projection_before),
        "projection_after": jsonable(projection_after),
        "post_action_rgb": {
            "path": post_frame["path"], "sha256": post_frame["sha256"],
            "rgb_mean": post_frame["rgb_mean"], "rgb_std": post_frame["rgb_std"],
        },
        "collision_attempts": int(turn_metrics.get("collision_count", 0) or 0),
        "action_valid": bool(turn_metrics.get("action_is_valid", bool(executed))),
        "done": bool(result["done"]), "reward": float(result["reward"]),
        "inference_seconds": inference_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-sha256s", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=4480)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=2)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    policy_path = Path(protocol["policy_input"]["path"])
    audit_path = Path(protocol["audit_manifest"]["path"])
    for path, expected in (
        (policy_path, protocol["policy_input"]["sha256"]),
        (audit_path, protocol["audit_manifest"]["sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen independent input hash mismatch: {path}")
    audits = read_jsonl(audit_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / "model_hash_verification.json",
        verify_model_sha256s(args.model_path, args.model_sha256s),
    )
    fingerprint = stable_hash({
        "runner": VERSION, "protocol_sha256": sha256(args.protocol),
        "model_key": args.model_key, "model_sha256_manifest": sha256(args.model_sha256s),
        "renderer_url": args.renderer_url,
    })
    atomic_json(args.output_dir / "run_environment.json", {
        "version": VERSION, "run_fingerprint": fingerprint,
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.executable, "python_version": sys.version,
        "git_commit_injected": os.environ.get("R1_INDEPENDENT_LOCAL_COMMIT"),
        "renderer_url": args.renderer_url, "gs_root": str(args.gs_root),
        "model_key": args.model_key, "model_path": str(args.model_path),
        "model_sha256s": {"path": str(args.model_sha256s), "sha256": sha256(args.model_sha256s)},
        "requested_attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "requested_dtype": "auto_no_downcast_override", "quantization": None,
    })
    policy = VllmPolicy(args, args.output_dir)
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    env = ActiveSpatialEnv(env_config(args, policy_path))
    ledger_path = args.output_dir / "diagnostic_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "version": VERSION, "run_fingerprint": fingerprint, "states": {},
    }
    if ledger.get("run_fingerprint") != fingerprint:
        raise RuntimeError("existing ledger belongs to another independent run")

    def execute(index: int) -> None:
        key = f"{index:03d}"
        if ledger["states"].get(key, {}).get("status") == "complete":
            return
        attempts = list(ledger["states"].get(key, {}).get("attempts", []))
        final = None
        for attempt in range(len(attempts), args.max_infrastructure_attempts):
            directory = args.output_dir / "states" / key / f"attempt_{attempt:02d}"
            try:
                final = run_one(env, policy, index, audits[index], int(protocol["local_decision_seed_base"]), directory)
                atomic_json(directory / "result.json", final)
                attempts.append({"attempt": attempt, "status": "complete", "path": str(directory / "result.json")})
                break
            except Exception as error:
                failure = {
                    "attempt": attempt, "status": "infrastructure_error",
                    "error_type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                atomic_json(directory / "error.json", failure)
                attempts.append(failure)
        if final is None:
            final = {
                "version": VERSION, "status": "infrastructure_error",
                "source_key": audits[index]["source_key"], "local_state_index": index,
            }
        final["attempts"] = attempts
        ledger["states"][key] = final
        atomic_json(ledger_path, ledger)
        atomic_json(args.output_dir / "summary.json", summarize(list(ledger["states"].values())))

    try:
        for index in SMOKE:
            if index < len(audits):
                execute(index)
        smoke = [ledger["states"].get(f"{index:03d}", {}) for index in SMOKE if index < len(audits)]
        passed = all(
            row.get("status") == "complete"
            and (row.get("initial_rgb_evidence") or {}).get("passed") for row in smoke
        )
        atomic_json(args.output_dir / "smoke_summary.json", {
            "indices": list(SMOKE), "infrastructure_and_rgb_consistency_pass": passed,
        })
        if not passed:
            raise RuntimeError("independent local-action smoke gate failed")
        for index in range(len(audits)):
            execute(index)
    finally:
        env.close()
        atomic_json(args.output_dir / "summary.json", summarize(list(ledger["states"].values())))
    print(json.dumps(json.loads((args.output_dir / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
