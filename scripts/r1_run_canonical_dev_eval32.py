#!/usr/bin/env python3
"""Run paired, checkpointed policy-only evaluation on the frozen R1 dev set."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
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
from PIL import Image


VERSION = "r1_canonical_dev_eval32_runner_v6_h1_render_camera"
SMOKE_INDICES = (0, 10, 21, 31)
FROZEN_RGB_MAX_MAE = 0.5
FROZEN_RGB_MAX_P99_ABS_ERROR = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def image_data_url(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def save_observation_image(obs: dict[str, Any], path: Path) -> dict[str, Any]:
    images = []
    for values in (obs.get("multi_modal_data") or {}).values():
        images.extend(values or [])
    if len(images) != 1:
        raise RuntimeError(f"expected exactly one current observation image, got {len(images)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].convert("RGB").save(path)
    arr = np.asarray(images[0].convert("RGB"), dtype=np.float32)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "width": int(images[0].width),
        "height": int(images[0].height),
        "rgb_mean": float(arr.mean()),
        "rgb_std": float(arr.std()),
        "image": images[0],
    }


def resolve_frozen_path(path: str, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def compare_frozen_initial_rgb(
    current: Image.Image,
    audit: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    """Compare a policy first frame with its already-frozen official RGB evidence."""
    evidence = audit["evidence_audit_only"]["observability"]
    manifest_path = resolve_frozen_path(evidence["path"], repository_root)
    actual_manifest_sha = sha256(manifest_path)
    if actual_manifest_sha != evidence["sha256"]:
        raise RuntimeError(
            f"frozen observability manifest SHA mismatch: {actual_manifest_sha} "
            f"!= {evidence['sha256']}"
        )
    records = read_jsonl(manifest_path)
    record = records[int(evidence["record_index"])]
    if int(record["source_row_index"]) != int(audit["source_row_index"]):
        raise RuntimeError("frozen observability evidence points to a different source row")
    prior_path = resolve_frozen_path(record["frame_images"][0], repository_root)
    prior = np.asarray(Image.open(prior_path).convert("RGB"), dtype=np.int16)
    actual = np.asarray(current.convert("RGB"), dtype=np.int16)
    if prior.shape != actual.shape:
        raise RuntimeError(
            f"frozen initial RGB shape mismatch: current={actual.shape}, prior={prior.shape}"
        )
    absolute_error = np.abs(actual - prior)
    mae = float(absolute_error.mean())
    p99 = float(np.quantile(absolute_error, 0.99))
    passed = mae <= FROZEN_RGB_MAX_MAE and p99 <= FROZEN_RGB_MAX_P99_ABS_ERROR
    comparison = {
        "passed": passed,
        "prior_path": str(prior_path),
        "prior_sha256": sha256(prior_path),
        "pixel_equal": bool(np.array_equal(actual, prior)),
        "mean_absolute_error": mae,
        "p99_absolute_error": p99,
        "maximum_absolute_error": int(absolute_error.max()),
        "maximum_allowed_mae": FROZEN_RGB_MAX_MAE,
        "maximum_allowed_p99_absolute_error": FROZEN_RGB_MAX_P99_ABS_ERROR,
        "observability_manifest": str(manifest_path),
        "observability_manifest_sha256": actual_manifest_sha,
        "observability_record_index": int(evidence["record_index"]),
    }
    if not passed:
        raise RuntimeError(
            "current initial RGB does not match frozen official evidence: "
            f"mae={mae:.6f}, p99={p99:.6f}, prior={prior_path}"
        )
    return comparison


def model_file_inventory(model_path: Path) -> dict[str, Any]:
    required = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "preprocessor_config.json", "model.safetensors.index.json",
    ]
    missing = [name for name in required if not (model_path / name).is_file()]
    weights = sorted(model_path.glob("*.safetensors"))
    if missing or not weights:
        raise RuntimeError(f"incomplete model export at {model_path}: missing={missing}, weights={len(weights)}")
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    indexed_weights = sorted(set(index.get("weight_map", {}).values()))
    actual_weights = sorted(path.name for path in weights)
    if indexed_weights != actual_weights:
        raise RuntimeError(
            f"weight index mismatch at {model_path}: indexed={indexed_weights}, actual={actual_weights}"
        )
    from safetensors import safe_open
    tensor_dtypes: Counter[str] = Counter()
    tensor_count = 0
    for path in weights:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor_dtypes[str(handle.get_slice(key).get_dtype())] += 1
                tensor_count += 1
    files = []
    for path in sorted(p for p in model_path.iterdir() if p.is_file()):
        files.append({"name": path.name, "size": path.stat().st_size})
    return {
        "path": str(model_path.resolve()),
        "files": files,
        "total_bytes": sum(x["size"] for x in files),
        "indexed_weight_files": indexed_weights,
        "tensor_count": tensor_count,
        "tensor_dtypes": dict(sorted(tensor_dtypes.items())),
    }


def verify_model_sha256s(model_path: Path, manifest_path: Path) -> dict[str, Any]:
    expected: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"malformed SHA256 line in {manifest_path}: {line!r}")
        name = Path(parts[1].strip().lstrip("*")).name
        if name in expected:
            raise RuntimeError(f"duplicate filename in SHA256 manifest: {name}")
        expected[name] = parts[0].lower()
    actual_files = sorted(path for path in model_path.iterdir() if path.is_file())
    actual_names = {path.name for path in actual_files}
    if set(expected) != actual_names:
        raise RuntimeError(
            f"SHA256 manifest inventory mismatch: missing={sorted(actual_names - set(expected))}, "
            f"unexpected={sorted(set(expected) - actual_names)}"
        )
    verified = []
    started = time.time()
    for path in actual_files:
        actual = sha256(path)
        if actual != expected[path.name]:
            raise RuntimeError(f"SHA256 mismatch for {path.name}: {actual} != {expected[path.name]}")
        verified.append({"name": path.name, "size": path.stat().st_size, "sha256": actual})
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "verified_file_count": len(verified),
        "verified_total_bytes": sum(row["size"] for row in verified),
        "verification_seconds": time.time() - started,
        "files": verified,
    }


class VllmPolicy:
    def __init__(self, args: argparse.Namespace, output_dir: Path):
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM

        self.args = args
        started = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
        self.tokenizer_class = type(tokenizer).__name__
        self.processor_class = type(processor).__name__
        self.llm = LLM(
            model=str(args.model_path),
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=False,
            dtype="auto",
            max_model_len=args.max_model_len,
            # Match the frozen v46 evaluation recipe. This also avoids a
            # model-dependent CUDA-graph warmup path in the paired smoke.
            enforce_eager=True,
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 1},
        )
        self.load_seconds = time.time() - started
        config = json.loads((args.model_path / "config.json").read_text())
        actual_dtype = str(self.llm.llm_engine.model_config.dtype)
        atomic_json(output_dir / "model_load.json", {
            "version": VERSION,
            "model_key": args.model_key,
            "model_path": str(args.model_path.resolve()),
            "model_files": model_file_inventory(args.model_path),
            "config_architectures": config.get("architectures"),
            "config_model_type": config.get("model_type"),
            "config_dtype": config.get("dtype", config.get("torch_dtype")),
            "requested_dtype": "auto_no_downcast_override",
            "actual_vllm_dtype": actual_dtype,
            "enforce_eager": True,
            "enable_prefix_caching": True,
            "trust_remote_code": False,
            "tokenizer_class": self.tokenizer_class,
            "processor_class": self.processor_class,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "load_seconds": self.load_seconds,
        })

    def generate(self, system_text: str, user_text: str, image: Image.Image, seed: int) -> tuple[str, float]:
        from vllm import SamplingParams

        # Match GymAgentLoop.convert_obs_to_content: replace the single
        # placeholder in-place instead of prepending a second image token.
        segments = user_text.split("<image>")
        if len(segments) != 2:
            raise RuntimeError(f"expected one <image> placeholder, got {len(segments) - 1}")
        user_content = []
        if segments[0]:
            user_content.append({"type": "text", "text": segments[0]})
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
        if segments[1]:
            user_content.append({"type": "text", "text": segments[1]})
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]
        params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=384,
            stop=["</action>"],
            include_stop_str_in_output=True,
            seed=seed,
        )
        started = time.time()
        outputs = self.llm.chat(messages, sampling_params=params, use_tqdm=False)
        elapsed = time.time() - started
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no completion")
        return outputs[0].outputs[0].text, elapsed


def env_config(args: argparse.Namespace, policy_path: Path) -> Any:
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    return ActiveSpatialEnvConfig(
        jsonl_path=str(policy_path),
        render_backend="http",
        client_url=args.renderer_url,
        gs_root=str(args.gs_root),
        image_width=256,
        image_height=256,
        step_translation=0.3,
        step_rotation_deg=20.0,
        action_space="strafe",
        enable_explicit_done=False,
        enable_auto_termination=True,
        max_actions_per_step=5,
        max_episode_steps=12,
        turn_budget=12,
        prompt_format="free_think",
        enable_distance_in_obs=False,
        enable_potential_field=True,
        potential_field_position_weight=0.7,
        potential_field_orientation_weight=0.3,
        potential_field_progress_mode="potential",
        potential_field_gamma=0.99,
        potential_field_reward_scale=1.0,
        near_success_threshold=0.55,
        near_success_bonus=0.5,
        success_reward=5.0,
        success_score_threshold=0.65,
        enable_collision_detection=True,
        collision_camera_radius=0.15,
        collision_floor_height=0.3,
        collision_ceiling_height=2.5,
        collision_safety_margin=0.05,
        collision_invalidate_action=True,
        render_fail_fast=True,
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    success = [row for row in complete if row.get("success")]
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_scene: dict[str, Counter] = defaultdict(Counter)
    actions = Counter()
    for row in complete:
        for grouping, key in ((by_split, row["split"]), (by_scene, row["scene_id"])):
            grouping[key]["episodes"] += 1
            grouping[key]["success"] += int(bool(row.get("success")))
        actions.update(row.get("actions", []))
    def groups(value: dict[str, Counter]) -> dict[str, Any]:
        return {key: dict(counts) for key, counts in sorted(value.items())}
    return {
        "requested": 32,
        "complete": len(complete),
        "infrastructure_errors": sum(row.get("status") == "infrastructure_error" for row in rows),
        "success_count": len(success),
        "success_rate": len(success) / len(complete) if complete else None,
        "by_split": groups(by_split),
        "by_scene": groups(by_scene),
        "actions": dict(sorted(actions.items())),
        "collision_attempts": sum(int(row.get("collision_attempts", 0)) for row in complete),
        "invalid_turns": sum(int(row.get("invalid_turns", 0)) for row in complete),
    }


def run_episode(
    env: Any,
    policy: VllmPolicy,
    row: dict[str, Any],
    audit: dict[str, Any],
    eval_index: int,
    paired_seed_base: int,
    attempt_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    obs, reset_info = env.reset(seed=eval_index)
    actual_identity = (env.current_item or {}).get("source_identity", {})
    if actual_identity.get("episode_fingerprint") != audit["episode_fingerprint"]:
        raise RuntimeError(f"environment selected wrong episode at index {eval_index}")
    actual_initial_pose = env.view_engine.get_pose().copy()
    expected_initial_pose = np.asarray(audit["initial_pose_c2w"], dtype=np.float64)
    initial_pose_max_abs_error = float(np.max(np.abs(actual_initial_pose - expected_initial_pose)))
    if initial_pose_max_abs_error > 1e-8:
        raise RuntimeError(
            f"initial pose mismatch at index {eval_index}: max_abs_error={initial_pose_max_abs_error}"
        )
    system_text = env.system_prompt()
    initial_metric = env._calculate_canonical_metric()
    if initial_metric is None or initial_metric.get("success"):
        raise RuntimeError(f"invalid initial canonical state at index {eval_index}: {initial_metric}")
    turns = []
    all_actions: list[str] = []
    collision_attempts = 0
    invalid_turns = 0
    first_success_step = None
    done = False
    final_info: dict[str, Any] = reset_info or {}
    initial_rgb_evidence: dict[str, Any] | None = None
    started = time.time()
    for turn_index in range(12):
        if done or env._current_step >= 12:
            break
        frame_path = attempt_dir / "frames" / f"turn_{turn_index:02d}_step_{env._current_step:02d}.png"
        frame = save_observation_image(obs, frame_path)
        if turn_index == 0:
            initial_rgb_evidence = compare_frozen_initial_rgb(
                frame["image"], audit, repository_root
            )
        pre_pose = env.view_engine.get_pose().copy()
        model_seed = paired_seed_base + eval_index * 100 + turn_index
        input_record = {
            "history_contract": "no_concat_system_plus_current_observation_only",
            "system_text": system_text,
            "user_text": obs["obs_str"],
            "image_path": frame["path"],
            "image_sha256": frame["sha256"],
            "image_rgb_mean": frame["rgb_mean"],
            "image_rgb_std": frame["rgb_std"],
            "seed": model_seed,
            "messages_fingerprint": stable_hash({
                "system": system_text, "user": obs["obs_str"], "image_sha256": frame["sha256"]
            }),
        }
        raw_completion, inference_seconds = policy.generate(system_text, obs["obs_str"], frame["image"], model_seed)
        primitive_before = int(env._current_step)
        obs, reward, done, step_info = env.step(raw_completion)
        primitive_after = int(env._current_step)
        post_pose = env.view_engine.get_pose().copy()
        metric = env._calculate_canonical_metric()
        parsed = [str(action) for action in (step_info.get("actions") or [])]
        all_actions.extend(parsed[: max(0, primitive_after - primitive_before)])
        turn_metrics = (step_info.get("metrics") or {}).get("turn_metrics") or {}
        collisions = int(turn_metrics.get("collision_count", 0) or 0)
        collision_attempts += collisions
        invalid_turns += int(not bool(turn_metrics.get("action_is_valid", bool(parsed))))
        if metric and metric.get("success") and first_success_step is None:
            first_success_step = primitive_after
        turns.append({
            "turn_index": turn_index,
            "primitive_step_before": primitive_before,
            "primitive_step_after": primitive_after,
            "input": input_record,
            "raw_completion": raw_completion,
            "parsed_actions": parsed,
            "executed_action_count": primitive_after - primitive_before,
            "pre_pose_c2w": pre_pose.tolist(),
            "post_pose_c2w": post_pose.tolist(),
            "reward": float(reward),
            "canonical_metric": jsonable(metric),
            "collision_attempts": collisions,
            "action_valid": bool(turn_metrics.get("action_is_valid", bool(parsed))),
            "strict_format_correct": bool(turn_metrics.get("strict_format_correct", False)),
            "done": bool(done),
            "termination_reason": step_info.get("termination_reason"),
            "inference_seconds": inference_seconds,
            "environment_info": jsonable({
                key: step_info.get(key) for key in (
                    "auto_terminated", "truncated_max_primitive_steps", "early_terminated_collision",
                    "early_terminated_invalid", "early_terminated_low_info", "termination_reason",
                    "terminated", "truncated", "current_potential_score", "canonical_task_metric",
                ) if key in step_info
            }),
        })
        final_info = step_info
    final_metric = env._calculate_canonical_metric()
    success = bool(final_metric and final_metric.get("success") and done)
    if success and first_success_step is None:
        first_success_step = int(env._current_step)
    if success:
        done_reason = final_info.get("termination_reason") or "canonical_success"
    elif final_info.get("early_terminated_collision"):
        done_reason = "collision"
    elif final_info.get("early_terminated_invalid"):
        done_reason = "invalid_action"
    elif final_info.get("early_terminated_low_info"):
        done_reason = "low_information"
    elif int(env._current_step) >= 12:
        done_reason = "max_primitive_actions"
    elif len(turns) >= 12:
        done_reason = "max_model_turns"
    else:
        done_reason = "environment_done_without_success" if done else "incomplete"
    return {
        "version": VERSION,
        "status": "complete",
        "eval_index": eval_index,
        "source_key": audit["source_key"],
        "split": audit["split"],
        "source_row_index": audit["source_row_index"],
        "scene_id": audit["scene_id"],
        "object_pair": audit["object_pair"],
        "relation": audit["relation"],
        "episode_fingerprint": audit["episode_fingerprint"],
        "paired_seed": paired_seed_base + eval_index * 100,
        "success": success,
        "first_success_step": first_success_step,
        "done_reason": done_reason,
        "primitive_steps": int(env._current_step),
        "model_turns": len(turns),
        "collision_attempts": collision_attempts,
        "invalid_turns": invalid_turns,
        "actions": all_actions,
        "initial_canonical_metric": jsonable(initial_metric),
        "initial_rgb_evidence": initial_rgb_evidence,
        "initial_pose_max_abs_error": initial_pose_max_abs_error,
        "final_canonical_metric": jsonable(final_metric),
        "elapsed_seconds": time.time() - started,
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-sha256s", type=Path)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-model-len", type=int, default=4480)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=2)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    repository_root = next(
        (
            parent
            for parent in args.protocol.resolve().parents
            if (parent / "vagen").is_dir() and (parent / "exps").is_dir()
        ),
        None,
    )
    if repository_root is None:
        raise RuntimeError(f"cannot resolve repository root from protocol path: {args.protocol}")
    policy_path = Path(protocol["policy_input"]["path"])
    audit_path = Path(protocol["audit_manifest"]["path"])
    if sha256(policy_path) != protocol["policy_input"]["sha256"]:
        raise RuntimeError("policy input SHA mismatch")
    if sha256(audit_path) != protocol["audit_manifest"]["sha256"]:
        raise RuntimeError("audit manifest SHA mismatch")
    policy_rows, audit_rows = read_jsonl(policy_path), read_jsonl(audit_path)
    if len(policy_rows) != 32 or len(audit_rows) != 32:
        raise RuntimeError("frozen evaluation must contain exactly 32 rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model_sha256s is None:
        raise RuntimeError("--model-sha256s is required for the frozen evaluation")
    hash_verification = verify_model_sha256s(args.model_path, args.model_sha256s)
    atomic_json(args.output_dir / "model_hash_verification.json", hash_verification)
    run_fingerprint = stable_hash({
        "runner": VERSION,
        "protocol_sha256": sha256(args.protocol),
        "model_key": args.model_key,
        "model_path": str(args.model_path.resolve()),
        "model_sha256s": sha256(args.model_sha256s) if args.model_sha256s else None,
        "renderer_url": args.renderer_url,
    })
    environment = {
        "version": VERSION,
        "run_fingerprint": run_fingerprint,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "python_version": sys.version,
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "renderer_url": args.renderer_url,
        "gs_root": str(args.gs_root.resolve()),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol),
        "model_key": args.model_key,
        "model_path": str(args.model_path.resolve()),
        "model_sha256s": ({"path": str(args.model_sha256s.resolve()), "sha256": sha256(args.model_sha256s)} if args.model_sha256s else None),
    }
    atomic_json(args.output_dir / "run_environment.json", environment)
    policy = VllmPolicy(args, args.output_dir)
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    env = ActiveSpatialEnv(env_config(args, policy_path))
    ledger_path = args.output_dir / "episode_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "version": VERSION, "run_fingerprint": run_fingerprint, "episodes": {}
    }
    if ledger.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError("existing ledger belongs to a different frozen run")

    def execute(index: int) -> None:
        key = f"{index:03d}"
        prior = ledger["episodes"].get(key)
        if prior and prior.get("status") == "complete":
            return
        attempts = list((prior or {}).get("attempts", []))
        final: dict[str, Any] | None = None
        for attempt in range(len(attempts), args.max_infrastructure_attempts):
            attempt_dir = args.output_dir / "episodes" / key / f"attempt_{attempt:02d}"
            try:
                result = run_episode(
                    env, policy, policy_rows[index], audit_rows[index], index,
                    int(protocol["paired_seed_base"]), attempt_dir, repository_root,
                )
                atomic_json(attempt_dir / "episode.json", result)
                attempts.append({"attempt": attempt, "status": "complete", "path": str(attempt_dir / "episode.json")})
                final = result
                break
            except Exception as exc:
                error = {
                    "attempt": attempt, "status": "infrastructure_error",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                atomic_json(attempt_dir / "error.json", error)
                attempts.append(error)
                # Recreate the environment after a renderer or state error.
                try:
                    env.close()
                except Exception:
                    pass
                env.__init__(env_config(args, policy_path))
        if final is None:
            final = {
                "version": VERSION, "status": "infrastructure_error", "eval_index": index,
                "source_key": audit_rows[index]["source_key"], "split": audit_rows[index]["split"],
                "source_row_index": audit_rows[index]["source_row_index"],
                "scene_id": audit_rows[index]["scene_id"], "attempts": attempts,
            }
        final["attempts"] = attempts
        ledger["episodes"][key] = final
        atomic_json(ledger_path, ledger)
        rows = [ledger["episodes"][k] for k in sorted(ledger["episodes"])]
        atomic_json(args.output_dir / "summary.json", summarize(rows))

    smoke = list(SMOKE_INDICES)
    for index in smoke:
        execute(index)
    smoke_rows = [ledger["episodes"].get(f"{i:03d}", {}) for i in smoke]
    smoke_infra_ok = all(row.get("status") == "complete" for row in smoke_rows)
    smoke_initial_rgb_ok = all(
        row.get("turns") and float(row["turns"][0]["input"].get("image_rgb_std", 0.0)) > 1e-6
        for row in smoke_rows
    )
    smoke_difficulty_ok = all(
        not row.get("success")
        or int(row.get("first_success_step")) >= int(audit_rows[index]["difficulty"]["certified_lower_bound"])
        for index, row in zip(smoke, smoke_rows)
    )
    smoke_frozen_rgb_ok = all(
        bool((row.get("initial_rgb_evidence") or {}).get("passed")) for row in smoke_rows
    )
    smoke_pass = (
        smoke_infra_ok and smoke_initial_rgb_ok and smoke_frozen_rgb_ok and smoke_difficulty_ok
    )
    atomic_json(args.output_dir / "smoke_summary.json", {
        "indices": smoke,
        "infrastructure_pass": smoke_infra_ok,
        "initial_rgb_structural_pass": smoke_initial_rgb_ok,
        "frozen_official_initial_rgb_consistency_pass": smoke_frozen_rgb_ok,
        "frozen_rgb_thresholds": {
            "maximum_mae": FROZEN_RGB_MAX_MAE,
            "maximum_p99_absolute_error": FROZEN_RGB_MAX_P99_ABS_ERROR,
        },
        "certified_lower_bound_consistency_pass": smoke_difficulty_ok,
        "full_evaluation_allowed": smoke_pass,
        "rows": [{k: row.get(k) for k in (
            "eval_index", "source_key", "status", "success", "first_success_step",
            "primitive_steps", "done_reason", "initial_pose_max_abs_error",
        )} for row in smoke_rows],
    })
    if not smoke_pass:
        raise RuntimeError("fixed four-episode smoke gate failed; full evaluation not started")
    for index in range(32):
        execute(index)
    try:
        env.close()
    finally:
        rows = [ledger["episodes"][k] for k in sorted(ledger["episodes"])]
        atomic_json(args.output_dir / "summary.json", summarize(rows))
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
