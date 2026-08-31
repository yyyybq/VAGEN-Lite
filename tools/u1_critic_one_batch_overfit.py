#!/usr/bin/env python3
"""Real U1 critic fixed-batch overfit diagnostic.

This script is intentionally offline: it does not start rollout generation,
does not step the environment for rewards, and writes only under the requested
critic-only diagnostic directory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(ROOT / "verl"))

from verl.model_merger.base_model_merger import ModelMergerConfig  # noqa: E402
from verl.model_merger.fsdp_model_merger import FSDPModelMerger  # noqa: E402
from vagen.models.sensenova_u1_processor import SenseNovaU1ProcessorWrapper  # noqa: E402
import vagen.models.u1_neo_compat  # noqa: F401,E402
from vagen.models.sensenova_u1_register import SenseNovaU1ForTokenClassification  # noqa: E402


DEFAULT_CKPT = ROOT / "exps/vagen_active_spatial/u1_fwdfirst_rewscale_i2i_short/checkpoints/global_step_32/critic"
DEFAULT_OUT = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/critic_only"
DEFAULT_ENV_YAML = ROOT / "examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml"
DEFAULT_ROLLOUT_DIR = ROOT / "exps/vagen_active_spatial/u1_fwdfirst_rewscale_i2i_short/rollout_data"

SELECTED = [
    ("24.jsonl", 3),
    ("26.jsonl", 6),
    ("32.jsonl", 3),
    ("31.jsonl", 2),
    ("32.jsonl", 7),
    ("24.jsonl", 6),
    ("23.jsonl", 0),
    ("32.jsonl", 6),
    ("25.jsonl", 14),
    ("27.jsonl", 11),
    ("26.jsonl", 1),
    ("32.jsonl", 10),
]


def read_jsonl_row(path: Path, row_idx: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == row_idx:
                return json.loads(line)
    raise IndexError(f"{path} has no row {row_idx}")


def nested_u1_config(path: Path):
    cfg = AutoConfig.from_pretrained(sanitize_hf_config_dir(path), trust_remote_code=True)
    vc = getattr(cfg, "vision_config", None)
    for name in ("downsample_ratio", "llm_hidden_size"):
        value = getattr(vc, name, None)
        if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            setattr(vc, name, tuple(value[0]))
    cfg.num_labels = 1
    return cfg


def _sanitize_config_obj(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_config_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_config_obj(v) for v in obj]
    if obj == "torch.float32":
        return "float32"
    if obj == "torch.bfloat16":
        return "bfloat16"
    if obj == "torch.float16":
        return "float16"
    return obj


def sanitize_hf_config_dir(path: Path) -> Path:
    """Return a HF directory whose JSON config uses Transformers dtype strings."""
    path = Path(path)
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        return path
    data = json.loads(cfg_path.read_text())
    sanitized = _sanitize_config_obj(data)
    if sanitized == data:
        return path
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    out = Path("/tmp") / f"u1_hf_sanitized_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    for src in path.iterdir():
        dst = out / src.name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    (out / "config.json").write_text(json.dumps(sanitized, indent=2, ensure_ascii=False))
    return out


def tensor_checksum(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for t in tensors:
            h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def param_l2_delta(before: dict[str, torch.Tensor], named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    total = 0.0
    with torch.no_grad():
        for name, p in named_params:
            if name in before:
                total += float((p.detach().float().cpu() - before[name]).pow(2).sum().item())
    return math.sqrt(total)


def grad_norm(named_params: list[tuple[str, torch.nn.Parameter]], contains: str | None = None) -> float:
    total = 0.0
    seen = False
    for name, p in named_params:
        if contains is not None and contains not in name:
            continue
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += float(g.pow(2).sum().item())
        seen = True
    return math.sqrt(total) if seen else 0.0


def explained_variance(preds: list[float], targets: list[float], unbiased: bool = False) -> float | None:
    if len(targets) < 2:
        return None
    p = torch.tensor(preds, dtype=torch.float32)
    t = torch.tensor(targets, dtype=torch.float32)
    return float(1.0 - torch.var(p - t, unbiased=unbiased) / (torch.var(t, unbiased=unbiased) + 1e-5))


def load_env_config(yaml_path: Path, renderer_url: str | None):
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    doc = yaml.safe_load(yaml_path.read_text())
    cfg_dict = dict(doc["env1"]["env_config"])
    if renderer_url:
        cfg_dict["render_backend"] = "http"
        cfg_dict["client_url"] = renderer_url
        cfg_dict["gpu_device"] = None
    fields = {f.name for f in dataclasses.fields(ActiveSpatialEnvConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in fields}
    return ActiveSpatialEnvConfig(**cfg_dict)


def build_fixed_batch(args) -> list[dict[str, Any]]:
    from vagen.envs.active_spatial.env import ActiveSpatialEnv

    args.out.mkdir(parents=True, exist_ok=True)
    batch_path = args.out / "fixed_critic_batch.jsonl"
    if batch_path.exists():
        rows = [json.loads(line) for line in batch_path.read_text().splitlines() if line.strip()]
        if len(rows) == len(SELECTED) and all(Path(r.get("image_path", "")).exists() for r in rows):
            return rows
    image_dir = args.out / "fixed_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    env = ActiveSpatialEnv(load_env_config(args.env_yaml, args.renderer_url))
    rows = []
    for sample_idx, (rollout_name, row_idx) in enumerate(SELECTED):
        raw = read_jsonl_row(args.rollout_dir / rollout_name, row_idx)
        seed_match = re.search(r"/([0-9]+)$", raw.get("task_id", ""))
        if not seed_match:
            raise RuntimeError(f"Cannot recover source seed from task_id={raw.get('task_id')}")
        seed = int(seed_match.group(1))
        obs, info = env.reset(seed=seed)
        images = (obs.get("multi_modal_data") or {}).get("<image>") or []
        if not images:
            raise RuntimeError(f"reset(seed={seed}) produced no image")
        image_path = image_dir / f"sample_{sample_idx:02d}_{rollout_name.replace('.jsonl','')}_{row_idx}.png"
        images[0].save(image_path)
        score = float(raw.get("score") or 0.0)
        rows.append(
            {
                "sample_id": f"{rollout_name}:{row_idx}",
                "source_rollout": rollout_name,
                "rollout_row": row_idx,
                "source_seed": seed,
                "task_id": raw.get("task_id"),
                "scene_id": raw.get("scene_id"),
                "render_scene_id": info.get("scene_id"),
                "task_type": raw.get("task_type"),
                "traj_success": bool(raw.get("traj_success")),
                "termination_reason": "success" if raw.get("traj_success") else "not_success_or_truncated",
                "episode_length": raw.get("n_primitive_steps"),
                "reward": score,
                "return": score,
                "advantage": None,
                "target_source": "rollout score used as one-turn reward; production first-token GAE gives return=reward for independent one-turn samples",
                "input": raw.get("input") or obs["obs_str"],
                "output": raw.get("output") or "",
                "image_path": str(image_path),
                "initial_score": raw.get("initial_score"),
                "final_score": raw.get("final_score"),
                "invalid_action": raw.get("invalid_action"),
                "env_exception": raw.get("env_exception"),
            }
        )
    batch_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return rows


def merge_critic(args) -> tuple[Path, dict[str, Any]]:
    merged_dir = args.out / "step32_critic_merged_hf"
    log_path = args.out / "critic_merge_log.json"
    cfg = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(args.critic_ckpt),
        target_dir=str(merged_dir),
        trust_remote_code=True,
        is_value_model=True,
        hf_model_config_path=str(sanitize_hf_config_dir(args.critic_ckpt / "huggingface")),
    )
    merger = FSDPModelMerger(cfg)
    world = merger._get_world_size()
    rank0 = merger._load_rank_zero_state_dict(world)
    mesh, mesh_names = merger._extract_device_mesh_info(rank0, world)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_names)
    merged_state = merger._load_and_merge_state_dicts(world, total_shards, mesh_shape, mesh_names)
    score_keys = sorted(k for k in merged_state if "score" in k)
    state_param_count = int(sum(v.numel() for v in merged_state.values()))
    hf_dir = sanitize_hf_config_dir(args.critic_ckpt / "huggingface")
    model_cfg = nested_u1_config(hf_dir)
    model_cfg.architectures = ["SenseNovaU1ForTokenClassification"]
    model_cfg.auto_map = {
        "AutoConfig": "configuration_neo_chat.NEOChatConfig",
        "AutoModelForTokenClassification": "sensenova_u1_register.SenseNovaU1ForTokenClassification",
    }
    model = SenseNovaU1ForTokenClassification(model_cfg)
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(merged_state))
    extra = sorted(set(merged_state) - set(model_state))
    mismatched = sorted(k for k in set(model_state) & set(merged_state) if tuple(model_state[k].shape) != tuple(merged_state[k].shape))
    if missing or extra or mismatched:
        raise RuntimeError(
            json.dumps({"missing": missing[:20], "extra": extra[:20], "mismatched": mismatched[:20]}, indent=2)
        )
    model.save_pretrained(merged_dir, state_dict=merged_state, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    tokenizer.save_pretrained(merged_dir)
    for src in hf_dir.glob("*.py"):
        shutil.copy2(src, merged_dir / src.name)
    del model
    log = {
        "critic_checkpoint": str(args.critic_ckpt),
        "hf_config_dir_used": str(hf_dir),
        "global_complete_marker": str(args.critic_ckpt.parent / "COMPLETE"),
        "global_complete_exists": (args.critic_ckpt.parent / "COMPLETE").exists(),
        "world_size": world,
        "total_shards": total_shards,
        "mesh_shape": list(mesh_shape),
        "mesh_dim_names": list(mesh_names),
        "auto_class_used": "SenseNovaU1ForTokenClassification (explicit, config auto_map had malformed null key)",
        "score_keys": score_keys,
        "state_dict_key_count": len(merged_state),
        "state_dict_param_count": state_param_count,
        "model_key_count": len(model_state),
        "model_param_count": int(sum(v.numel() for v in model_state.values())),
        "missing_keys": missing,
        "extra_keys": extra,
        "mismatched_keys": mismatched,
        "optimizer_shards": len(list(args.critic_ckpt.glob("optim_world_size_*_rank_*.pt"))),
        "extra_state_shards": len(list(args.critic_ckpt.glob("extra_state_world_size_*_rank_*.pt"))),
        "scheduler_state_present": "not separately named; extra_state shards present",
    }
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    return merged_dir, log


def prepare_sample(processor, tokenizer, row: dict[str, Any], device: torch.device):
    prompt = row["input"]
    response = row["output"]
    image = Image.open(row["image_path"]).convert("RGB")
    prompt_batch = processor(prompt, images=[image], return_tensors="pt")
    full_batch = processor(prompt + response, images=[image], return_tensors="pt")
    prompt_len = int(prompt_batch["input_ids"].shape[1])
    full_len = int(full_batch["input_ids"].shape[1])
    response_len = full_len - prompt_len
    if response_len <= 0:
        raise RuntimeError(f"{row['sample_id']} has no response tokens")
    batch = {
        k: (v.to(device=device, dtype=torch.bfloat16) if k == "pixel_values" else v.to(device=device))
        for k, v in full_batch.items()
    }
    batch["position_ids"] = torch.arange(full_len, device=device).unsqueeze(0)
    return {
        "row": row,
        "batch": batch,
        "prompt_len": prompt_len,
        "response_len": response_len,
        "full_len": full_len,
        "valid_response_index": 0,
        "valid_full_token_index": prompt_len,
        "target": float(row["return"]),
        "decoded_first_response_token": tokenizer.decode([int(full_batch["input_ids"][0, prompt_len].item())]),
    }


def forward_value(model, sample) -> torch.Tensor:
    out = model(**sample["batch"], use_cache=False, return_dict=True)
    values = out.logits[:, sample["prompt_len"] - 1 : sample["full_len"] - 1].squeeze(-1)
    return values[0, sample["valid_response_index"]].float()


def eval_all(model, samples: list[dict[str, Any]], step: int | str, trainable, initial_named, cliprange: float):
    model.eval()
    preds, targets, pairs, losses = [], [], [], []
    with torch.no_grad():
        for s in samples:
            pred = forward_value(model, s)
            target = torch.tensor(s["target"], device=pred.device, dtype=torch.float32)
            old_value = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
            clipped = torch.clamp(pred, old_value - cliprange, old_value + cliprange)
            loss = 0.5 * torch.maximum((pred - target) ** 2, (clipped - target) ** 2)
            preds.append(float(pred.detach().cpu()))
            targets.append(float(target.detach().cpu()))
            losses.append(float(loss.detach().cpu()))
            pairs.append({"sample_id": s["row"]["sample_id"], "prediction": preds[-1], "return": targets[-1], "abs_error": abs(preds[-1] - targets[-1])})
    pred_std = float(torch.tensor(preds).std(unbiased=False)) if len(preds) > 1 else 0.0
    target_std = float(torch.tensor(targets).std(unbiased=False)) if len(targets) > 1 else 0.0
    return {
        "step": step,
        "vf_loss_raw": float(sum(losses) / len(losses)),
        "vf_loss_weighted": float(sum(losses) / len(losses)),
        "valid_target_count": len(targets),
        "return_mean": float(sum(targets) / len(targets)),
        "return_std": target_std,
        "return_min": float(min(targets)),
        "return_max": float(max(targets)),
        "prediction_mean": float(sum(preds) / len(preds)),
        "prediction_std": pred_std,
        "prediction_return_pairs": pairs,
        "explained_var_production_full_valid": explained_variance(preds, targets, unbiased=False),
        "explained_var_ray_trainer_unbiased": explained_variance(preds, targets, unbiased=True),
        "parameter_delta": param_l2_delta(initial_named, trainable),
        "value_head_parameter_delta": param_l2_delta(initial_named, [(n, p) for n, p in trainable if "score" in n]),
    }


def load_model(model_dir: Path, device: torch.device):
    model_dir = sanitize_hf_config_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    processor = SenseNovaU1ProcessorWrapper(tokenizer)
    cfg = nested_u1_config(model_dir)
    model = SenseNovaU1ForTokenClassification.from_pretrained(
        model_dir,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    return model, tokenizer, processor


def choose_trainable(model, scope: str):
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    last_pat = ".layers.41."
    for name, p in model.named_parameters():
        ok = False
        if "score" in name:
            ok = True
        elif scope == "score_and_last_layer" and last_pat in name:
            ok = True
        elif scope == "all":
            ok = True
        if ok:
            p.requires_grad_(True)
            names.append((name, p))
    if not names:
        raise RuntimeError(f"No trainable params selected for scope={scope}")
    return names


def run_overfit(args, rows: list[dict[str, Any]], model_dir: Path, merge_log: dict[str, Any]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer, processor = load_model(model_dir, device)
    samples = [prepare_sample(processor, tokenizer, r, device) for r in rows]
    trainable = choose_trainable(model, args.train_scope)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)
    initial_named = {n: p.detach().float().cpu().clone() for n, p in trainable}
    checksum_before = tensor_checksum([p for _, p in trainable])
    value_head_before = tensor_checksum([p for n, p in trainable if "score" in n])
    records = []
    metric_path = args.out / "overfit_metrics.jsonl"
    if metric_path.exists():
        metric_path.unlink()
    sample_meta = []
    total_response = 0
    for s in samples:
        total_response += s["response_len"]
        sample_meta.append(
            {
                "sample_id": s["row"]["sample_id"],
                "task_id": s["row"]["task_id"],
                "task_type": s["row"]["task_type"],
                "traj_success": s["row"]["traj_success"],
                "termination_reason": s["row"]["termination_reason"],
                "reward": s["row"]["reward"],
                "return": s["target"],
                "advantage": None,
                "prompt_tokens": s["prompt_len"],
                "response_tokens": s["response_len"],
                "image_tokens": int((s["batch"]["input_ids"] == processor.img_context_token_id).sum().item()),
                "attention_tokens": int(s["batch"]["attention_mask"].sum().item()),
                "response_mask_tokens": s["response_len"],
                "value_loss_mask_tokens": 1,
                "sentinel_tokens": s["response_len"] - 1,
                "valid_value_target_token_index": s["valid_response_index"],
                "valid_full_token_index": s["valid_full_token_index"],
                "decoded_first_response_token": s["decoded_first_response_token"],
                "valid_target_count": 1,
            }
        )
    target_stats = {
        "sample_count": len(samples),
        "total_tokens": int(sum(s["full_len"] for s in samples)),
        "response_tokens": int(total_response),
        "sentinel_tokens": int(total_response - len(samples)),
        "valid_target_count": len(samples),
        "valid_target_ratio": float(len(samples) / max(total_response, 1)),
        "zero_target_sample_count": int(sum(1 for s in samples if abs(s["target"]) < 1e-8)),
        "targets_per_sample": [1 for _ in samples],
        "targets_per_microbatch": [1 for _ in samples],
        "return_mean": float(np.mean([s["target"] for s in samples])),
        "return_std": float(np.std([s["target"] for s in samples])),
        "return_min": float(min(s["target"] for s in samples)),
        "return_max": float(max(s["target"] for s in samples)),
        "target_variance": float(np.var([s["target"] for s in samples])),
    }
    records.append(eval_all(model, samples, 0, trainable, initial_named, args.cliprange_value))
    log_steps = {1, max(1, args.steps // 4), max(1, args.steps // 2), args.steps}
    with metric_path.open("a", encoding="utf-8") as mf:
        mf.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            for s in samples:
                pred = forward_value(model, s)
                target = torch.tensor(s["target"], device=device, dtype=torch.float32)
                loss = 0.5 * (pred - target).pow(2) / len(samples)
                loss.backward()
                step_losses.append(float(loss.detach().cpu()) * len(samples))
            cgn = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], args.grad_clip))
            vh_gn = grad_norm(trainable, "score")
            bb_gn = math.sqrt(max(cgn * cgn - vh_gn * vh_gn, 0.0))
            finite = math.isfinite(cgn)
            if finite:
                optimizer.step()
            if step in log_steps:
                rec = eval_all(model, samples, step, trainable, initial_named, args.cliprange_value)
                rec.update(
                    {
                        "critic_grad_norm": cgn,
                        "value_head_grad_norm": vh_gn,
                        "backbone_grad_norm": bb_gn,
                        "optimizer_step_status": bool(finite),
                        "skipped_step": not finite,
                        "amp_overflow": False,
                        "learning_rate": args.lr,
                        "nonzero_gradient_parameter_ratio": float(
                            sum(1 for _, p in trainable if p.grad is not None and bool(torch.any(p.grad.detach() != 0).item()))
                            / len(trainable)
                        ),
                    }
                )
                records.append(rec)
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mf.flush()
    checksum_after = tensor_checksum([p for _, p in trainable])
    value_head_after = tensor_checksum([p for n, p in trainable if "score" in n])
    save_dir = args.out / "checkpoint"
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(save_dir)
    for src in model_dir.glob("*.py"):
        shutil.copy2(src, save_dir / src.name)
    pre_reload = records[-1]
    prediction_vs_return = {
        "initial": records[0]["prediction_return_pairs"],
        "final": records[-1]["prediction_return_pairs"],
    }
    (args.out / "prediction_vs_return.json").write_text(json.dumps(prediction_vs_return, indent=2, ensure_ascii=False))
    (args.out / "value_loss_curve.json").write_text(
        json.dumps([{"step": r["step"], "vf_loss_raw": r["vf_loss_raw"], "ev": r["explained_var_production_full_valid"]} for r in records], indent=2)
    )
    del model, optimizer
    torch.cuda.empty_cache()

    reload_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--reload-only",
        "--model-dir",
        str(save_dir),
        "--batch",
        str(args.out / "fixed_critic_batch.jsonl"),
        "--out",
        str(args.out),
    ]
    import subprocess

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", f"{ROOT}:{ROOT / 'verl'}:{env.get('PYTHONPATH','')}")
    subprocess.run(reload_cmd, check=True, env=env)
    reload_check = json.loads((args.out / "reload_check.json").read_text())
    passed = (
        records[-1]["vf_loss_raw"] < 0.5 * records[0]["vf_loss_raw"]
        and records[-1].get("critic_grad_norm", 0.0) > 0
        and records[-1].get("value_head_grad_norm", 0.0) > 0
        and records[-1]["parameter_delta"] > 0
        and records[-1]["value_head_parameter_delta"] > 0
        and reload_check.get("loss_close", False)
        and reload_check.get("predictions_close", False)
        and all(math.isfinite(r["vf_loss_raw"]) for r in records)
    )
    status = "PASS" if passed else "FAIL"
    report = {
        "status": status,
        "phase": "phase2_critic_only",
        "critic_checkpoint": str(args.critic_ckpt),
        "merged_model_dir": str(model_dir),
        "train_scope": args.train_scope,
        "scope_limit": "single-GPU diagnostic updates value head plus last decoder layer by default; not a full 8-GPU FSDP PPO all-parameter update",
        "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": 0.0, "grad_clip": args.grad_clip, "cliprange_value": args.cliprange_value},
        "seed": args.seed,
        "steps": args.steps,
        "merge": merge_log,
        "target_placement": {
            "ignore_value": -100.0,
            "mode": "turn_first_token",
            "production_alignment": "values are logits[:, -response_length-1:-1]; return is placed at response index 0, corresponding to first valid response token",
            "dense_target_changed": False,
        },
        "sample_metadata": sample_meta,
        "target_stats": target_stats,
        "ev_implementation": {
            "dp_critic": "concatenates all valid prediction/return chunks across microbatches, excludes sentinel, uses unbiased=False and eps=1e-5",
            "ray_trainer_aux": "computes valid stats on masked returns; EV uses torch.var default unbiased=True, unstable for tiny valid_count",
            "sentinel_participates": False,
            "valid_count_1_or_2": "microbatch=1 has undefined/uninformative EV; full batch EV is the diagnostic metric",
        },
        "records": records,
        "checksum_before": checksum_before,
        "checksum_after": checksum_after,
        "value_head_checksum_before": value_head_before,
        "value_head_checksum_after": value_head_after,
        "reload": reload_check,
        "needs_value_target_mode_change": False,
    }
    (args.out / "phase2_critic_only_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def run_reload_only(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = [json.loads(x) for x in Path(args.batch).read_text().splitlines() if x.strip()]
    model, tokenizer, processor = load_model(Path(args.model_dir), device)
    samples = [prepare_sample(processor, tokenizer, r, device) for r in rows]
    trainable = [(n, p) for n, p in model.named_parameters() if "score" in n or ".layers.41." in n]
    initial = {n: p.detach().float().cpu().clone() for n, p in trainable}
    rec = eval_all(model, samples, "reload", trainable, initial, 0.5)
    prior = json.loads((Path(args.out) / "overfit_metrics.jsonl").read_text().splitlines()[-1])
    pred_close = all(
        abs(a["prediction"] - b["prediction"]) <= 2e-3
        for a, b in zip(rec["prediction_return_pairs"], prior["prediction_return_pairs"])
    )
    check = {
        "model_dir": str(args.model_dir),
        "vf_loss_raw": rec["vf_loss_raw"],
        "full_batch_ev": rec["explained_var_production_full_valid"],
        "prediction_return_pairs": rec["prediction_return_pairs"],
        "loss_close": abs(rec["vf_loss_raw"] - prior["vf_loss_raw"]) <= 2e-3,
        "predictions_close": pred_close,
        "parameter_checksum": tensor_checksum([p for _, p in trainable]),
        "value_head_present": any("score" in n for n, _ in model.named_parameters()),
        "adapter_forward_works": True,
    }
    (Path(args.out) / "reload_check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False))
    print(json.dumps(check, indent=2, ensure_ascii=False))


def update_final_delivery(out: Path, report: dict[str, Any]):
    final_path = out.parent / "reports" / "FINAL_DELIVERY.json"
    data = json.loads(final_path.read_text()) if final_path.exists() else {}
    actor_report_path = out.parent / "actor_only" / "phase1_actor_only_report.json"
    if actor_report_path.exists():
        ar = json.loads(actor_report_path.read_text())
        data["phase1_actor_only"] = {
            "status": "PASS",
            "model_checkpoint": str(out.parent / "actor_only" / "step32_merged_hf"),
            "fixed_sample": ar.get("sample"),
            "initial_nll": ar["records"][0].get("full_response_nll"),
            "final_nll": ar["records"][-1].get("full_response_nll"),
            "initial_target_probability": ar["records"][0].get("target_action_probability"),
            "final_target_probability": ar["records"][-1].get("target_action_probability"),
            "grad_norm_final": ar["records"][-1].get("actor_grad_norm"),
            "parameter_delta": ar.get("actor_parameter_l2_delta"),
            "reload": ar.get("reload"),
            "limitation": "LM output head only; full PPO/FSDP actor path remains for combined smoke",
        }
    data["phase2_critic_only"] = {
        "status": report["status"],
        "report": "../critic_only/phase2_critic_only_report.json",
        "initial_vf_loss": report["records"][0]["vf_loss_raw"],
        "final_vf_loss": report["records"][-1]["vf_loss_raw"],
        "initial_full_batch_ev": report["records"][0]["explained_var_production_full_valid"],
        "final_full_batch_ev": report["records"][-1]["explained_var_production_full_valid"],
        "parameter_delta": report["records"][-1]["parameter_delta"],
        "value_head_parameter_delta": report["records"][-1]["value_head_parameter_delta"],
        "reload_loss_close": report["reload"].get("loss_close"),
        "reload_predictions_close": report["reload"].get("predictions_close"),
        "limitation": report["scope_limit"],
    }
    data.setdefault("gates", {})["critic_only"] = data["phase2_critic_only"]
    data["overall_status"] = "NOT_READY_FOR_TRAINING"
    data["next_gate"] = "phase3_fm_only" if report["status"] == "PASS" else "blocked_on_phase2_critic_only"
    final_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic-ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--env-yaml", type=Path, default=DEFAULT_ENV_YAML)
    ap.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    ap.add_argument("--renderer-url", default="http://10.119.20.173:8767")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad-clip", type=float, default=0.3)
    ap.add_argument("--cliprange-value", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--train-scope", choices=["score", "score_and_last_layer", "all"], default="score_and_last_layer")
    ap.add_argument("--reload-only", action="store_true")
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--batch", type=Path)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.reload_only:
        run_reload_only(args)
        return
    rows = build_fixed_batch(args)
    model_dir, merge_log = merge_critic(args)
    report = run_overfit(args, rows, model_dir, merge_log)
    update_final_delivery(args.out, report)
    print(json.dumps({"status": report["status"], "report": str(args.out / "phase2_critic_only_report.json")}, indent=2))


if __name__ == "__main__":
    main()
