#!/usr/bin/env python3
"""Fixed-manifest one-turn protocol evaluation for Phase 4B.

The normal mode evaluates one deterministic shard. ``--aggregate-only`` merges
all shard outputs and verifies exact manifest coverage. No optimizer, backward,
or training service is created by this script.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vagen.models.sensenova_u1_processor import SenseNovaU1ProcessorWrapper
import vagen.models.u1_neo_compat  # noqa: F401

from tools.u1_protocol_prepost_manifest import model_identity, sha256_bytes, sha256_file


DEFAULT_MANIFEST = ROOT / "exps/vagen_active_spatial/protocol_only_prepost/protocol_eval_manifest.jsonl"
DEFAULT_MODEL = Path("/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT")
DEFAULT_ENV = ROOT / "examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml"


def dump_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def nested_u1_config(path: Path):
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    vc = cfg.vision_config
    for name in ("downsample_ratio", "llm_hidden_size"):
        value = getattr(vc, name)
        if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            setattr(vc, name, tuple(value[0]))
    return cfg


def load_env_config(path: Path, renderer_url: str):
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    doc = yaml.safe_load(path.read_text())
    values = dict(doc["env1"]["env_config"])
    values["render_backend"] = "http"
    values["client_url"] = renderer_url
    values["gpu_device"] = None
    # The fixed evaluation manifest is indexed against the complete 2373-line
    # clean-layout file and intentionally covers all seven task types. The
    # training config keeps its original task filter; only evaluation disables
    # the dataclass default ``exclude_task_types=['delta_control']``.
    values["exclude_task_types"] = []
    fields = {f.name for f in dataclasses.fields(ActiveSpatialEnvConfig)}
    return ActiveSpatialEnvConfig(**{k: v for k, v in values.items() if k in fields})


def observation_images(obs: dict[str, Any]) -> list[Image.Image]:
    mm = obs.get("multi_modal_input") or obs.get("multi_modal_data") or {}
    raw = mm.get("<image>") or mm.get("image") or []
    return [x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x).convert("RGB") for x in raw]


def classify(text: str) -> str:
    tool = "<tool_call>" in text
    action = "<action>" in text
    if tool and action:
        return "tool_call_and_action"
    if tool:
        return "tool_call_only"
    if action:
        return "action_only"
    return "missing_or_invalid"


def decode_generated(tokenizer, generated: torch.Tensor, prompt_len: int) -> str:
    ids = generated[0]
    if ids.shape[0] > prompt_len:
        ids = ids[prompt_len:]
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def binary_metrics(row: dict[str, Any]) -> dict[str, float]:
    parser = row.get("parser") or {}
    flags = row.get("flags") or {}
    return {
        "strict_action_tag_rate": float(row.get("format_class") == "action_only"),
        "tool_call_rate": float("tool_call" in str(row.get("format_class"))),
        "fallback_parse_rate": float(bool(parser.get("fallback_parse"))),
        "missing_action_rate": float(bool(parser.get("missing_action_tag")) or row.get("format_class") == "missing_or_invalid"),
        "unknown_action_rate": float(bool(parser.get("unknown_action_name"))),
        "multiple_action_rate": float(bool(parser.get("multiple_action_tag"))),
        "invalid_or_empty_rate": float(bool(flags.get("invalid_or_empty"))),
        "strict_parse_success_rate": float(bool(parser.get("strict_parse_success"))),
        "action_executable_rate": float(bool(parser.get("action_executable"))),
        "env_exception_rate": float(bool(flags.get("env_exception"))),
        "success_rate": float(bool(flags.get("success"))),
        "contradictory_action_rate": float(bool(flags.get("contradictory_action"))),
        "format_penalty_rate": float(bool((row.get("rewards") or {}).get("format_penalty_applied"))),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    names = list(binary_metrics(rows[0])) if rows else []
    rates = {name: sum(binary_metrics(row)[name] for row in rows) / max(n, 1) for name in names}
    actions = Counter()
    formats = Counter()
    for row in rows:
        formats[str(row.get("format_class"))] += 1
        actions.update((row.get("parser") or {}).get("actions") or [])
    rewards = [float((row.get("rewards") or {}).get("total_reward", 0.0)) for row in rows]
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tasks[str(row.get("task_type"))].append(row)
    per_task = {}
    for task, task_rows in sorted(tasks.items()):
        task_rates = {
            name: sum(binary_metrics(row)[name] for row in task_rows) / len(task_rows)
            for name in names
        }
        per_task[task] = {"n": len(task_rows), **task_rates}
    return {
        "n": n,
        **rates,
        "format_counts": dict(sorted(formats.items())),
        "action_distribution": dict(sorted(actions.items())),
        "total_reward_mean": float(sum(rewards) / max(n, 1)),
        "total_reward_min": float(min(rewards)) if rewards else None,
        "total_reward_max": float(max(rewards)) if rewards else None,
        "per_task": per_task,
    }


def aggregate(args, manifest: list[dict[str, Any]]) -> int:
    shard_files = sorted((args.out / "shards").glob("shard_*/raw_rollouts_shard_*.jsonl"))
    if len(shard_files) != args.num_shards:
        raise RuntimeError(f"expected {args.num_shards} shard files, found {len(shard_files)}: {shard_files}")
    rows = []
    for path in shard_files:
        rows.extend(load_jsonl(path))
    rows.sort(key=lambda row: int(row["order"]))
    expected = [row["sample_id"] for row in manifest]
    actual = [row["sample_id"] for row in rows]
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"manifest coverage/order mismatch missing={missing} extra={extra}")
    identities = {row.get("evaluated_model_identity") for row in rows}
    if len(identities) != 1:
        raise RuntimeError(f"multiple model identities across shards: {identities}")
    write_jsonl(rows, args.out / "raw_rollouts.jsonl")
    report = {
        "status": "PASS" if not any(binary_metrics(row)["env_exception_rate"] for row in rows) else "FAIL",
        "checkpoint_label": args.checkpoint_label,
        "model": rows[0]["evaluated_model_path"],
        "model_identity": next(iter(identities)),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "raw_rollouts": str(args.out / "raw_rollouts.jsonl"),
        "independent_model_load_processes": args.num_shards,
        "metrics": summarize(rows),
        "metric_semantics": {
            "task_reward": "raw env.step reward minus the explicit strict/invalid format component",
            "env_reward": "raw scalar returned by env.step",
            "total_reward": "same one-turn raw env.step scalar; retained as explicit comparison field",
        },
        "evaluation_task_filter": {"include": "ALL", "exclude": []},
    }
    dump_json(report, args.out / "aggregate_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


def run_shard(args, manifest: list[dict[str, Any]]) -> int:
    selected = [row for row in manifest if int(row["order"]) % args.num_shards == args.shard_index]
    if not selected:
        raise RuntimeError(f"empty shard {args.shard_index}/{args.num_shards}")
    renderer_urls = {row["renderer_url"] for row in selected}
    if len(renderer_urls) != 1:
        raise RuntimeError(f"multiple renderer URLs in shard: {renderer_urls}")
    renderer_url = next(iter(renderer_urls))
    identity = model_identity(args.model)
    identity_hash = identity["identity_sha256"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = SenseNovaU1ProcessorWrapper(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=nested_u1_config(args.model),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    model.config.use_cache = True
    model.img_context_token_id = processor.img_context_token_id
    model.img_start_token_id = processor.img_start_token_id

    from vagen.envs.active_spatial.env import ActiveSpatialEnv

    env = ActiveSpatialEnv(load_env_config(args.env_yaml, renderer_url))
    out_dir = args.out / "shards" / f"shard_{args.shard_index:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f"raw_rollouts_shard_{args.shard_index:02d}.jsonl"
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_file}; pass --overwrite")

    results = []
    started = time.time()
    try:
        for spec in selected:
            row_started = time.time()
            base = {
                "manifest_version": spec["manifest_version"],
                "sample_id": spec["sample_id"],
                "order": spec["order"],
                "seed": spec["seed"],
                "scene_id": spec["scene_id"],
                "task_id": spec["task_id"],
                "task_type": spec["task_type"],
                "object_label": spec["object_label"],
                "preset": spec["preset"],
                "checkpoint_label": args.checkpoint_label,
                "evaluated_model_path": str(args.model),
                "evaluated_model_identity": identity_hash,
                "renderer_url": renderer_url,
                "decoding": {
                    "generation_seed": spec["generation_seed"],
                    "max_output_length": spec["max_output_length"],
                    "temperature": spec["temperature"],
                    "top_p": spec["top_p"],
                },
            }
            try:
                random.seed(int(spec["generation_seed"]))
                np.random.seed(int(spec["generation_seed"]) % (2**32 - 1))
                torch.manual_seed(int(spec["generation_seed"]))
                torch.cuda.manual_seed_all(int(spec["generation_seed"]))
                obs, reset_info = env.reset(seed=int(spec["seed"]))
                if reset_info.get("scene_id") != spec["scene_id"] or reset_info.get("task_type") != spec["task_type"]:
                    raise RuntimeError(
                        f"manifest drift: expected scene/task={spec['scene_id']}/{spec['task_type']} "
                        f"got {reset_info.get('scene_id')}/{reset_info.get('task_type')}"
                    )
                images = observation_images(obs)
                if not images:
                    raise RuntimeError("reset returned no image")
                image_path = image_dir / f"{int(spec['order']):03d}_{spec['sample_id']}.png"
                images[0].save(image_path)
                system_text = env.system_prompt()
                user_text = obs["obs_str"]
                messages = [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ]
                rendered_prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
                )
                inputs = processor(rendered_prompt, images=images, return_tensors="pt")
                prompt_len = int(inputs["input_ids"].shape[1])
                batch = {
                    key: (value.cuda().to(torch.bfloat16) if key == "pixel_values" else value.cuda())
                    for key, value in inputs.items()
                }
                generation_started = time.time()
                with torch.inference_mode():
                    generated = model.generate(
                        **batch,
                        do_sample=True,
                        temperature=float(spec["temperature"]),
                        top_p=float(spec["top_p"]),
                        max_new_tokens=int(spec["max_output_length"]),
                    )
                output = decode_generated(tokenizer, generated, prompt_len)
                parsed = env._default_parse_func(
                    output, action_sep=env.config.action_sep, max_actions=env.config.max_actions_per_step
                )
                try:
                    _, env_reward, done, info = env.step(output)
                    env_exception = False
                    env_exception_text = None
                except Exception as exc:
                    env_reward, done, info = 0.0, True, {}
                    env_exception = True
                    env_exception_text = f"{type(exc).__name__}: {exc}"
                metrics = (info.get("metrics") or {}) if info else {}
                turn = metrics.get("turn_metrics") or {}
                traj = metrics.get("traj_metrics") or {}
                strict = bool(parsed.get("format_correct"))
                format_component = float(env.config.format_reward if strict else env.config.invalid_format_penalty)
                action_executable = bool(parsed.get("actions")) and not env_exception
                contradictory = bool(turn.get("contradictory_action", False))
                result = {
                    **base,
                    "raw_prompt": {"system": system_text, "user": user_text, "messages": messages},
                    "rendered_prompt": rendered_prompt,
                    "rendered_prompt_token_count": prompt_len,
                    "rendered_prompt_token_sha256": sha256_bytes(inputs["input_ids"].cpu().numpy().tobytes()),
                    "condition_image": str(image_path),
                    "condition_image_sha256": sha256_file(image_path),
                    "raw_model_output": output,
                    "output": output,
                    "format_class": classify(output),
                    "parser_branch": "strict" if parsed.get("strict_parse_success") else (
                        "fallback" if parsed.get("fallback_parse") else "invalid"
                    ),
                    "parsed_action": parsed.get("actions") or [],
                    "parser": {key: parsed.get(key) for key in (
                        "format_correct", "actions", "parse_error", "has_tool_call", "has_strict_action_tag",
                        "fallback_parse", "strict_parse_success", "missing_action_tag", "empty_action_body",
                        "unknown_action_name", "truncated_before_action", "multiple_action_tag",
                    )} | {"action_executable": action_executable},
                    "rewards": {
                        "format_penalty": format_component if not strict else 0.0,
                        "format_reward_component": format_component,
                        "format_penalty_applied": bool(not strict),
                        "task_reward": float(env_reward) - format_component,
                        "env_reward": float(env_reward),
                        "total_reward": float(env_reward),
                    },
                    "flags": {
                        "success": bool(info.get("success", traj.get("success", False))) if info else False,
                        "invalid_or_empty": bool(not action_executable),
                        "empty_action": bool(not parsed.get("actions")),
                        "unknown_action": bool(parsed.get("unknown_action_name")),
                        "multiple_action": bool(parsed.get("multiple_action_tag")),
                        "contradictory_action": contradictory,
                        "env_exception": env_exception,
                        "env_exception_text": env_exception_text,
                        "collision_termination": bool(info.get("early_terminated_collision", False)) if info else False,
                        "low_info_termination": bool(info.get("early_terminated_low_info", False)) if info else False,
                        "done": bool(done),
                    },
                    "env_info": {
                        "is_format_rewarded": info.get("is_format_rewarded") if info else None,
                        "is_format_penalized": info.get("is_format_penalized") if info else None,
                        "current_potential_score": info.get("current_potential_score") if info else None,
                        "env_step": info.get("env_step") if info else None,
                        "image_std": info.get("image_std") if info else None,
                        "collision_count": info.get("collision_count") if info else None,
                    },
                    "timing": {
                        "generation_seconds": time.time() - generation_started,
                        "sample_seconds": time.time() - row_started,
                    },
                }
            except Exception as exc:
                result = {
                    **base,
                    "raw_prompt": None,
                    "rendered_prompt": None,
                    "raw_model_output": None,
                    "output": "",
                    "format_class": "missing_or_invalid",
                    "parser_branch": "exception",
                    "parsed_action": [],
                    "parser": {"action_executable": False},
                    "rewards": {
                        "format_penalty": 0.0,
                        "format_reward_component": 0.0,
                        "format_penalty_applied": False,
                        "task_reward": 0.0,
                        "env_reward": 0.0,
                        "total_reward": 0.0,
                    },
                    "flags": {
                        "success": False,
                        "invalid_or_empty": True,
                        "empty_action": True,
                        "unknown_action": False,
                        "multiple_action": False,
                        "contradictory_action": False,
                        "env_exception": True,
                        "env_exception_text": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                    "timing": {"sample_seconds": time.time() - row_started},
                }
            results.append(result)
            write_jsonl(results, output_file)
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "sample_id": result["sample_id"],
                        "format_class": result["format_class"],
                        "parser_branch": result["parser_branch"],
                        "env_exception": result["flags"]["env_exception"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        env.close()

    report = {
        "status": "PASS" if not any(row["flags"]["env_exception"] for row in results) else "FAIL",
        "checkpoint_label": args.checkpoint_label,
        "model": str(args.model),
        "model_identity": identity,
        "manifest_sha256": sha256_file(args.manifest),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n": len(results),
        "metrics": summarize(results),
        "elapsed_seconds": time.time() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "training_started": False,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "evaluation_task_filter": {"include": "ALL", "exclude": []},
    }
    dump_json(report, out_dir / "shard_report.json")
    return 0 if report["status"] == "PASS" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--env-yaml", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-label", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    manifest = load_jsonl(args.manifest)
    if len(manifest) != 64 or len({row["sample_id"] for row in manifest}) != 64:
        raise RuntimeError("manifest must contain exactly 64 unique samples")
    if args.aggregate_only:
        return aggregate(args, manifest)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    return run_shard(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
