#!/usr/bin/env python3
"""Run frozen local-action and first-action-only R1 floor diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r1_run_canonical_dev_eval32 import (
    VllmPolicy, atomic_json, compare_frozen_initial_rgb, env_config, jsonable,
    read_jsonl, save_observation_image, sha256, stable_hash, verify_model_sha256s,
)
from vagen.envs.active_spatial.env import runtime_render_camera_parameters


VERSION = "r1_canonical_dev_eval32_floor_diagnostic_runner_v1"
LOCAL_SMOKE = (0, 1, 2, 3)
REPLAN_SMOKE = (0, 10, 21, 31)


def first_action_only_response(raw: str, first_action: str) -> str:
    """Preserve the model response but replace its first action body with one action."""
    pattern = re.compile(r"(<action>)(.*?)(</action>)", flags=re.IGNORECASE | re.DOTALL)
    match = pattern.search(raw or "")
    if match is None:
        raise ValueError("parsed completion has no replaceable <action> body")
    return raw[:match.start(2)] + first_action + "|" + raw[match.end(2):]


def compare_prior_rgb(image: Image.Image, prior: dict[str, Any]) -> dict[str, Any]:
    path = Path(prior["path"])
    if sha256(path) != prior["sha256"]:
        raise RuntimeError(f"prior RGB hash mismatch: {path}")
    actual = np.asarray(image.convert("RGB"), dtype=np.int16)
    expected = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    if actual.shape != expected.shape:
        raise RuntimeError(f"derived RGB shape mismatch: {actual.shape} != {expected.shape}")
    delta = np.abs(actual - expected)
    result = {
        "prior_path": str(path), "prior_sha256": prior["sha256"],
        "pixel_equal": bool(np.array_equal(actual, expected)),
        "mean_absolute_error": float(delta.mean()),
        "p99_absolute_error": float(np.quantile(delta, 0.99)),
        "maximum_absolute_error": int(delta.max()),
    }
    result["passed"] = result["mean_absolute_error"] <= 0.5 and result["p99_absolute_error"] <= 2.0
    if not result["passed"]:
        raise RuntimeError(f"derived initial RGB mismatch: {result}")
    return result


def effective_camera(env: Any) -> dict[str, Any]:
    pose = env.view_engine.get_pose().copy()
    K, w2c = runtime_render_camera_parameters(
        env.current_item, pose, env.camera_intrinsics,
        (int(env.config.image_width), int(env.config.image_height)),
    )
    return {"K_effective": K.tolist(), "w2c": w2c.tolist(), "c2w": pose.tolist()}


def parse(env: Any, raw: str) -> dict[str, Any]:
    return env._default_parse_func(
        raw, action_sep=env.config.action_sep, max_actions=env.config.max_actions_per_step,
    )


def policy_input_record(env: Any, obs: dict[str, Any], frame: dict[str, Any], seed: int) -> dict[str, Any]:
    system = env.system_prompt()
    return {
        "history_contract": "no_concat_system_plus_current_observation_only",
        "system_text": system, "user_text": obs["obs_str"],
        "image_path": frame["path"], "image_sha256": frame["sha256"],
        "image_rgb_mean": frame["rgb_mean"], "image_rgb_std": frame["rgb_std"],
        "seed": seed,
        "messages_fingerprint": stable_hash({"system": system, "user": obs["obs_str"], "image_sha256": frame["sha256"]}),
    }


def execute_first_only(env: Any, raw: str) -> tuple[dict[str, Any], str, list[str], list[str]]:
    parsed = parse(env, raw)
    full = [str(value) for value in parsed.get("actions") or []]
    if full:
        executed_raw = first_action_only_response(raw, full[0])
        reparsed = parse(env, executed_raw)
        if [str(value) for value in reparsed.get("actions") or []] != [full[0]]:
            raise RuntimeError("first-action response does not reparse to exactly one original first action")
    else:
        # Preserve baseline invalid-output behavior; do not rescue a later token.
        executed_raw = raw
    obs, reward, done, info = env.step(executed_raw)
    return {"obs": obs, "reward": reward, "done": done, "info": info}, executed_raw, full[:1], full[1:]


def run_local_decision(
    env: Any, policy: VllmPolicy, index: int, audit: dict[str, Any], seed_base: int,
    attempt_dir: Path,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=index)
    identity = (env.current_item or {}).get("source_identity", {})
    if int(identity.get("local_state_index", -1)) != index:
        raise RuntimeError(f"wrong derived state loaded at {index}: {identity}")
    pose = env.view_engine.get_pose().copy()
    expected = np.asarray(audit["pose_c2w"], dtype=float)
    pose_error = float(np.max(np.abs(pose - expected)))
    if pose_error > 1e-8:
        raise RuntimeError(f"derived pose mismatch at {index}: {pose_error}")
    before = env._calculate_canonical_metric()
    if before is None or before.get("success"):
        raise RuntimeError(f"derived state is already success at {index}")
    frame = save_observation_image(obs, attempt_dir / "initial.png")
    rgb = compare_prior_rgb(frame["image"], audit["prior_rgb_audit_only"])
    initial_camera = effective_camera(env)
    seed = seed_base + index
    input_record = policy_input_record(env, obs, frame, seed)
    raw, inference_seconds = policy.generate(input_record["system_text"], obs["obs_str"], frame["image"], seed)
    original_parse = parse(env, raw)
    pre_step = int(env._current_step)
    result, executed_raw, executed, discarded = execute_first_only(env, raw)
    post_step = int(env._current_step)
    after = env._calculate_canonical_metric()
    turn_metrics = (result["info"].get("metrics") or {}).get("turn_metrics") or {}
    first = executed[0] if executed else None
    return {
        "version": VERSION, "status": "complete", "local_state_index": index,
        "parent_eval_index": audit["parent_eval_index"], "source_key": audit["source_key"],
        "split": audit["split"], "scene_id": audit["scene_id"],
        "exact_distance_audit_only": audit["exact_distance"],
        "optimal_first_actions_audit_only": audit["optimal_first_actions_audit_only"],
        "model_input_oracle_free": True, "pose_max_abs_error": pose_error,
        "initial_rgb_evidence": rgb, "effective_camera_initial": initial_camera,
        "input": input_record, "raw_completion": raw,
        "original_parse": jsonable(original_parse), "executed_raw_completion": executed_raw,
        "first_parsed_action": first, "discarded_actions": discarded,
        "optimal_first_action_hit": first in set(audit["optimal_first_actions_audit_only"]),
        "primitive_step_before": pre_step, "primitive_step_after": post_step,
        "canonical_before": jsonable(before), "canonical_after": jsonable(after),
        "collision_attempts": int(turn_metrics.get("collision_count", 0) or 0),
        "action_valid": bool(turn_metrics.get("action_is_valid", bool(executed))),
        "done": bool(result["done"]), "reward": float(result["reward"]),
        "inference_seconds": inference_seconds,
    }


def run_replan_episode(
    env: Any, policy: VllmPolicy, row: dict[str, Any], audit: dict[str, Any], index: int,
    seed_base: int, attempt_dir: Path, repository_root: Path,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=index)
    identity = (env.current_item or {}).get("source_identity", {})
    if identity.get("episode_fingerprint") != audit["episode_fingerprint"]:
        raise RuntimeError(f"wrong original episode loaded at {index}")
    pose_error = float(np.max(np.abs(env.view_engine.get_pose() - np.asarray(audit["initial_pose_c2w"], dtype=float))))
    if pose_error > 1e-8:
        raise RuntimeError(f"original initial pose mismatch at {index}: {pose_error}")
    initial_metric = env._calculate_canonical_metric()
    if initial_metric is None or initial_metric.get("success"):
        raise RuntimeError(f"invalid original initial state at {index}")
    initial_camera = effective_camera(env)
    turns, actions = [], []
    collisions = invalid_turns = inference_calls = 0
    first_success = None
    done = False
    initial_rgb = None
    started = time.time()
    for turn_index in range(12):
        if done or int(env._current_step) >= 12:
            break
        frame = save_observation_image(obs, attempt_dir / "frames" / f"turn_{turn_index:02d}_step_{env._current_step:02d}.png")
        if turn_index == 0:
            initial_rgb = compare_frozen_initial_rgb(frame["image"], audit, repository_root)
        seed = seed_base + index * 100 + turn_index
        input_record = policy_input_record(env, obs, frame, seed)
        pre_pose = env.view_engine.get_pose().copy()
        raw, inference_seconds = policy.generate(input_record["system_text"], obs["obs_str"], frame["image"], seed)
        inference_calls += 1
        original_parse = parse(env, raw)
        before_step = int(env._current_step)
        result, executed_raw, executed, discarded = execute_first_only(env, raw)
        obs, done, info = result["obs"], bool(result["done"]), result["info"]
        after_step = int(env._current_step)
        metric_after = env._calculate_canonical_metric()
        turn_metrics = (info.get("metrics") or {}).get("turn_metrics") or {}
        collision = int(turn_metrics.get("collision_count", 0) or 0)
        collisions += collision
        invalid_turns += int(not bool(turn_metrics.get("action_is_valid", bool(executed))))
        if executed and after_step > before_step:
            actions.append(executed[0])
        if metric_after and metric_after.get("success") and first_success is None:
            first_success = after_step
        turns.append({
            "turn_index": turn_index, "primitive_step_before": before_step, "primitive_step_after": after_step,
            "input": input_record, "raw_completion": raw, "original_parse": jsonable(original_parse),
            "executed_raw_completion": executed_raw, "executed_first_action": executed[0] if executed else None,
            "discarded_actions": discarded, "pre_pose_c2w": pre_pose.tolist(),
            "post_pose_c2w": env.view_engine.get_pose().tolist(), "canonical_metric": jsonable(metric_after),
            "collision_attempts": collision, "action_valid": bool(turn_metrics.get("action_is_valid", bool(executed))),
            "strict_format_correct": bool(turn_metrics.get("strict_format_correct", False)),
            "reward": float(result["reward"]), "done": done,
            "termination_reason": info.get("termination_reason"), "inference_seconds": inference_seconds,
        })
    final = env._calculate_canonical_metric()
    success = bool(final and final.get("success") and done)
    if success:
        done_reason = "canonical_success"
    elif int(env._current_step) >= 12:
        done_reason = "max_primitive_actions"
    elif len(turns) >= 12:
        done_reason = "max_model_turns"
    else:
        done_reason = "environment_done_without_success" if done else "incomplete"
    return {
        "version": VERSION, "status": "complete", "eval_index": index,
        "source_key": audit["source_key"], "split": audit["split"], "scene_id": audit["scene_id"],
        "episode_fingerprint": audit["episode_fingerprint"], "success": success,
        "first_success_step": first_success, "done_reason": done_reason,
        "primitive_steps": int(env._current_step), "model_turns": len(turns),
        "model_inference_calls": inference_calls, "collision_attempts": collisions,
        "invalid_turns": invalid_turns, "actions": actions,
        "initial_pose_max_abs_error": pose_error, "initial_rgb_evidence": initial_rgb,
        "effective_camera_initial": initial_camera,
        "initial_canonical_metric": jsonable(initial_metric), "final_canonical_metric": jsonable(final),
        "elapsed_seconds": time.time() - started, "turns": turns,
    }


def summarize_local(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    result: dict[str, Any] = {"requested": len(rows), "complete": len(complete), "infrastructure_errors": len(rows) - len(complete)}
    for distance in (1, 2):
        subset = [row for row in complete if row.get("exact_distance_audit_only") == distance]
        result[f"d{distance}"] = {
            "states": len(subset), "optimal_first_action_hits": sum(bool(row.get("optimal_first_action_hit")) for row in subset),
            "invalid_or_empty": sum(row.get("first_parsed_action") is None for row in subset),
            "collisions": sum(int(row.get("collision_attempts", 0)) for row in subset),
        }
    return result


def summarize_replan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    return {
        "requested": 32, "complete": len(complete), "infrastructure_errors": len(rows) - len(complete),
        "success": sum(bool(row.get("success")) for row in complete),
        "collision_attempts": sum(int(row.get("collision_attempts", 0)) for row in complete),
        "invalid_turns": sum(int(row.get("invalid_turns", 0)) for row in complete),
        "primitive_actions": sum(int(row.get("primitive_steps", 0)) for row in complete),
        "model_inference_calls": sum(int(row.get("model_inference_calls", 0)) for row in complete),
        "inference_seconds": sum(sum(float(turn.get("inference_seconds", 0)) for turn in row.get("turns", [])) for row in complete),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
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
    parent = json.loads(args.parent_protocol.read_text())
    diagnostic = json.loads(args.diagnostic_protocol.read_text())
    local_policy_path, local_audit_path = Path(diagnostic["policy_input"]["path"]), Path(diagnostic["audit_manifest"]["path"])
    original_policy_path, original_audit_path = Path(parent["policy_input"]["path"]), Path(parent["audit_manifest"]["path"])
    for path, expected in ((local_policy_path, diagnostic["policy_input"]["sha256"]), (local_audit_path, diagnostic["audit_manifest"]["sha256"]), (original_policy_path, parent["policy_input"]["sha256"]), (original_audit_path, parent["audit_manifest"]["sha256"])):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input hash mismatch: {path}")
    local_audits, original_audits = read_jsonl(local_audit_path), read_jsonl(original_audit_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "model_hash_verification.json", verify_model_sha256s(args.model_path, args.model_sha256s))
    fingerprint = stable_hash({
        "runner": VERSION, "diagnostic_protocol_sha256": sha256(args.diagnostic_protocol),
        "parent_protocol_sha256": sha256(args.parent_protocol), "model_key": args.model_key,
        "model_sha256_manifest": sha256(args.model_sha256s), "renderer_url": args.renderer_url,
    })
    atomic_json(args.output_dir / "run_environment.json", {
        "version": VERSION, "run_fingerprint": fingerprint, "hostname": socket.gethostname(),
        "platform": platform.platform(), "python": sys.executable, "python_version": sys.version,
        "git_commit_injected": os.environ.get("R1_FLOOR_COMMIT"), "renderer_url": args.renderer_url,
        "gs_root": str(args.gs_root), "model_key": args.model_key, "model_path": str(args.model_path),
        "model_sha256s": {"path": str(args.model_sha256s), "sha256": sha256(args.model_sha256s)},
        "requested_attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "requested_dtype": "auto_no_downcast_override", "quantization": None,
    })
    policy = VllmPolicy(args, args.output_dir)
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    repository_root = next(parent for parent in args.parent_protocol.resolve().parents if (parent / "vagen").is_dir())
    local_env = ActiveSpatialEnv(env_config(args, local_policy_path))
    original_env = ActiveSpatialEnv(env_config(args, original_policy_path))
    ledger_path = args.output_dir / "diagnostic_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"version": VERSION, "run_fingerprint": fingerprint, "local": {}, "replan": {}}
    if ledger.get("run_fingerprint") != fingerprint:
        raise RuntimeError("existing ledger belongs to another frozen diagnostic")

    def execute(kind: str, index: int) -> None:
        key = f"{index:03d}"
        if ledger[kind].get(key, {}).get("status") == "complete":
            return
        attempts = list(ledger[kind].get(key, {}).get("attempts", []))
        final = None
        for attempt in range(len(attempts), args.max_infrastructure_attempts):
            directory = args.output_dir / kind / key / f"attempt_{attempt:02d}"
            try:
                if kind == "local":
                    final = run_local_decision(local_env, policy, index, local_audits[index], int(diagnostic["local_decision_seed_base"]), directory)
                else:
                    final = run_replan_episode(original_env, policy, read_jsonl(original_policy_path)[index], original_audits[index], index, int(diagnostic["replan_seed_base"]), directory, repository_root)
                atomic_json(directory / "result.json", final)
                attempts.append({"attempt": attempt, "status": "complete", "path": str(directory / "result.json")})
                break
            except Exception as exc:
                error = {"attempt": attempt, "status": "infrastructure_error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
                atomic_json(directory / "error.json", error); attempts.append(error)
        if final is None:
            audit = local_audits[index] if kind == "local" else original_audits[index]
            final = {"version": VERSION, "status": "infrastructure_error", "source_key": audit["source_key"], "local_state_index" if kind == "local" else "eval_index": index}
        final["attempts"] = attempts; ledger[kind][key] = final; atomic_json(ledger_path, ledger)
        atomic_json(args.output_dir / "summary.json", {"local": summarize_local(list(ledger["local"].values())), "replan": summarize_replan(list(ledger["replan"].values()))})

    try:
        for index in LOCAL_SMOKE:
            if index < len(local_audits): execute("local", index)
        for index in REPLAN_SMOKE: execute("replan", index)
        smoke_local = [ledger["local"].get(f"{i:03d}", {}) for i in LOCAL_SMOKE if i < len(local_audits)]
        smoke_replan = [ledger["replan"].get(f"{i:03d}", {}) for i in REPLAN_SMOKE]
        smoke_pass = all(row.get("status") == "complete" and (row.get("initial_rgb_evidence") or {}).get("passed") for row in smoke_local + smoke_replan)
        atomic_json(args.output_dir / "smoke_summary.json", {"local_indices": list(LOCAL_SMOKE), "replan_indices": list(REPLAN_SMOKE), "infrastructure_and_rgb_consistency_pass": smoke_pass})
        if not smoke_pass:
            raise RuntimeError("fixed smoke gate failed")
        for index in range(len(local_audits)): execute("local", index)
        for index in range(32): execute("replan", index)
    finally:
        local_env.close(); original_env.close()
        atomic_json(args.output_dir / "summary.json", {"local": summarize_local(list(ledger["local"].values())), "replan": summarize_replan(list(ledger["replan"].values()))})
    print(json.dumps(json.loads((args.output_dir / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
