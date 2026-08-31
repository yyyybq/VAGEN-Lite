#!/usr/bin/env python3
"""Phase 3 FM forward-only gate for fixed ActiveSpatial transitions.

No renderer, no backward, no optimizer, no FM_BACKPROP=1.  This script loads
the formal U1 adapter and computes the real FM aux loss with
``U1_FM_BACKPROP=0`` for each fixed transition independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(ROOT / "verl"))

os.environ.setdefault("SENSENOVA_U1_SRC", "/mnt/umm/users/yinbaiqiao/SenseNova-U1/src")

from vagen.models.sensenova_u1_processor import SenseNovaU1ProcessorWrapper  # noqa: E402
import vagen.models.u1_neo_compat  # noqa: F401,E402
from vagen.models.sensenova_u1_register import SenseNovaU1ForCausalLMAdapter  # noqa: E402

DEFAULT_OUT = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/fm_only"
DEFAULT_MODEL = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/actor_only/step32_merged_hf"
DEFAULT_FINAL = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/reports/FINAL_DELIVERY.json"


def jdump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_hf_dir(path: Path) -> Path:
    def fix(obj):
        if isinstance(obj, dict):
            return {k: fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix(v) for v in obj]
        return {"torch.float32": "float32", "torch.bfloat16": "bfloat16", "torch.float16": "float16"}.get(obj, obj)

    cfg_path = path / "config.json"
    if not cfg_path.exists():
        return path
    data = json.loads(cfg_path.read_text())
    fixed = fix(data)
    if fixed == data:
        return path
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    out = Path("/tmp") / f"u1_forward_hf_sanitized_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    for src in path.iterdir():
        dst = out / src.name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    (out / "config.json").write_text(json.dumps(fixed, ensure_ascii=False, indent=2))
    return out


def image_audit(path: str) -> dict[str, Any]:
    p = Path(path)
    img = Image.open(p).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    return {
        "path": str(p),
        "exists": p.exists(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "normalization_range": [float(arr.min()), float(arr.max())],
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "bytes": int(p.stat().st_size),
    }


def diff_audit(a: str, b: str) -> dict[str, float]:
    arr = np.asarray(ImageChops.difference(Image.open(a).convert("RGB"), Image.open(b).convert("RGB")), dtype=np.float32)
    return {
        "pixel_diff_mean": float(arr.mean()),
        "pixel_diff_max": float(arr.max()),
        "pixel_diff_nonzero_fraction": float((arr > 0).mean()),
    }


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach()
    xf = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(xf.min().item()) if x.numel() else None,
        "max": float(xf.max().item()) if x.numel() else None,
        "mean": float(xf.mean().item()) if x.numel() else None,
        "std": float(xf.std(unbiased=False).item()) if x.numel() else None,
        "norm": float(xf.norm().item()) if x.numel() else None,
        "finite": bool(torch.isfinite(xf).all().item()) if x.numel() else True,
    }


def make_prompt(action: str) -> str:
    return (
        "<|im_start|>system\nYou are an embodied navigation agent. Respond with the fixed action only.<|im_end|>\n"
        "<|im_start|>user\n<image>\nUse the fixed Phase-3 transition action.<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<think>Use the audited fixed transition.</think>\n<action>{action}|</action><|im_end|>"
    )


def load_rows(out: Path) -> list[dict[str, Any]]:
    p = out / "fixed_transitions.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        raise RuntimeError(f"need at least 2 fixed transitions; got {len(rows)}")
    for r in rows:
        if not r.get("valid_transition"):
            raise RuntimeError(f"invalid fixed transition in input: idx={r.get('idx')} reject={r.get('reject_reason')}")
        for k in ("condition_image", "target_image"):
            if not Path(r[k]).exists():
                raise FileNotFoundError(r[k])
    return rows


def load_model_processor(model_dir: Path, device: str):
    def scalar(v, default):
        if v is None:
            return default
        while isinstance(v, (tuple, list)) and len(v) == 1:
            v = v[0]
        return v

    model_dir = sanitize_hf_dir(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vc = getattr(cfg, "vision_config", None)
    for name in ("downsample_ratio", "llm_hidden_size"):
        value = getattr(vc, name, None)
        if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            setattr(vc, name, tuple(value[0]))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = SenseNovaU1ForCausalLMAdapter.from_pretrained(
        model_dir,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = SenseNovaU1ProcessorWrapper(
        tok,
        patch_size=int(scalar(getattr(cfg.vision_config, "patch_size", 16), 16)),
        downsample_ratio=float(scalar(getattr(cfg.vision_config, "downsample_ratio", 0.5), 0.5)),
    )
    return model, processor, tok


def checksum_params(model) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for _, p in model.named_parameters():
            h.update(p.detach().view(-1)[:16].float().cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--final-delivery", type=Path, default=DEFAULT_FINAL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args()

    os.environ["U1_FM_BACKPROP"] = "0"
    os.environ["U1_FM_USE_UND_KV"] = "0"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    rows = load_rows(args.out)
    model, processor, tokenizer = load_model_processor(args.model, device)
    model.train()  # formal FM aux only participates in training mode
    model_dtype = next(model.parameters()).dtype
    before_checksum = checksum_params(model)

    metrics_path = args.out / "fm_forward_sample_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    sample_metrics: list[dict[str, Any]] = []
    target_latents: list[torch.Tensor] = []
    condition_gen_latents: list[torch.Tensor] = []

    mem0 = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    for row in rows:
        idx = int(row["idx"])
        torch.manual_seed(args.seed + idx)
        cond_img = Image.open(row["condition_image"]).convert("RGB")
        tgt_img = Image.open(row["target_image"]).convert("RGB")
        enc = processor(make_prompt(row["action"]), images=[cond_img], return_tensors="pt")
        gen = processor.preprocess_images([tgt_img])
        cond_as_gen = processor.preprocess_images([cond_img])
        sample = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
            "pixel_values": enc["pixel_values"].to(device=device, dtype=model_dtype),
            "grid_hw": enc["grid_hw"].to(device),
            "u1_gen_pixel_values": gen["pixel_values"].to(device=device, dtype=model_dtype),
            "u1_gen_grid_hw": gen["grid_hw"].to(device),
            "u1_gen_valid": torch.tensor([1], dtype=torch.bool, device=device),
        }
        image_token_id = processor.img_context_token_id
        image_token_count = int((sample["input_ids"] == image_token_id).sum().item())
        target_count = int(sample["u1_gen_valid"].sum().item())
        mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        pending_before = bool(getattr(model, "_u1_pending_fm", None))
        out = model(**sample, labels=None, return_dict=True, use_cache=False)
        pending_after = bool(getattr(model, "_u1_pending_fm", None))
        loss = out.loss
        mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        if loss is None:
            raise RuntimeError(f"sample {idx}: FM forward returned loss=None")
        # Target encoding audit using the same formal helper; fixed seed above.
        with torch.no_grad():
            noisy, target_latent, target_velocity, timestep, noise_scale = model._prepare_single_gen_target(
                sample["u1_gen_pixel_values"].to(device=device, dtype=next(model.parameters()).dtype),
                sample["u1_gen_grid_hw"].to(device=device),
            )
            _, cond_latent, _, cond_timestep, cond_noise_scale = model._prepare_single_gen_target(
                cond_as_gen["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype),
                cond_as_gen["grid_hw"].to(device=device),
            )
        target_latents.append(target_latent.detach().float().cpu())
        condition_gen_latents.append(cond_latent.detach().float().cpu())
        metric = {
            "idx": idx,
            "scene_id": row["scene_id"],
            "action": row["action"],
            "condition_image": row["condition_image"],
            "target_image": row["target_image"],
            "condition_image_audit": image_audit(row["condition_image"]),
            "target_image_audit": image_audit(row["target_image"]),
            "condition_target_pixel_diff": diff_audit(row["condition_image"], row["target_image"]),
            "input_ids_shape": list(sample["input_ids"].shape),
            "attention_mask_shape": list(sample["attention_mask"].shape),
            "condition_pixel_values": tensor_stats(sample["pixel_values"]),
            "condition_grid_hw": sample["grid_hw"].detach().cpu().tolist(),
            "image_context_token_count": image_token_count,
            "u1_gen_pixel_values": tensor_stats(sample["u1_gen_pixel_values"]),
            "u1_gen_grid_hw": sample["u1_gen_grid_hw"].detach().cpu().tolist(),
            "u1_gen_valid": bool(sample["u1_gen_valid"].item()),
            "generation_target_count": target_count,
            "target_representation": "formal U1 flow-matching target: next-frame pixels -> normalized patches -> merged latent x/z/velocity target via _prepare_single_gen_target",
            "target_latent": tensor_stats(target_latent),
            "target_velocity": tensor_stats(target_velocity),
            "target_noisy_z": tensor_stats(noisy),
            "timestep": tensor_stats(timestep),
            "noise_scale": tensor_stats(noise_scale),
            "condition_as_gen_latent": tensor_stats(cond_latent),
            "condition_target_latent_l2": float((target_latent.detach().float() - cond_latent.detach().float()).norm().item()),
            "fm_loss_raw": float(loss.detach().float().item()),
            "loss_dtype": str(loss.dtype),
            "loss_device": str(loss.device),
            "loss_finite": bool(torch.isfinite(loss.detach()).all().item()),
            "loss_requires_grad": bool(loss.requires_grad),
            "valid_fm_target_count": target_count,
            "fm_skipped": False,
            "skip_reason": None,
            "fm_module_class": model.__class__.__name__,
            "fm_backprop_env": os.environ.get("U1_FM_BACKPROP"),
            "fm_use_und_kv_env": os.environ.get("U1_FM_USE_UND_KV"),
            "pending_before_forward": pending_before,
            "pending_after_forward": pending_after,
            "compute_pending_fm_aux_loss_called": False,
            "optimizer_step_executed": False,
            "backward_executed": False,
            "gpu_memory_before": int(mem_before),
            "gpu_memory_after": int(mem_after),
        }
        sample_metrics.append(metric)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")

    after_checksum = checksum_params(model)
    target_latent_pair_l2 = float((target_latents[0] - target_latents[1]).norm().item()) if len(target_latents) >= 2 else None
    table = [
        {
            "sample": m["idx"],
            "action": m["action"],
            "valid": m["u1_gen_valid"],
            "FM loss": m["fm_loss_raw"],
            "target latent norm": m["target_latent"]["norm"],
            "condition-target latent diff": m["condition_target_latent_l2"],
            "skipped": m["fm_skipped"],
        }
        for m in sample_metrics
    ]
    pass_gate = (
        len(sample_metrics) >= 2
        and all(m["u1_gen_valid"] for m in sample_metrics)
        and all(m["loss_finite"] and m["fm_loss_raw"] != 0.0 for m in sample_metrics)
        and all(not m["fm_skipped"] for m in sample_metrics)
        and target_latent_pair_l2 is not None
        and target_latent_pair_l2 > 0
        and before_checksum == after_checksum
        and not any(m["pending_after_forward"] for m in sample_metrics)
    )
    report = {
        "phase": "phase3_fm_forward_only",
        "status": "PASS" if pass_gate else "FAIL",
        "model": str(args.model),
        "device": device,
        "seed": args.seed,
        "fixed_transitions": str(args.out / "fixed_transitions.jsonl"),
        "sample_metrics": str(metrics_path),
        "table": table,
        "target_latent_pair_l2": target_latent_pair_l2,
        "backprop_static_confirmation": {
            "U1_FM_BACKPROP": os.environ.get("U1_FM_BACKPROP"),
            "model_path": "metrics-only/no-grad FM branch in SenseNovaU1ForCausalLMAdapter.forward",
            "compute_pending_fm_aux_loss_called": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "parameter_checksum_unchanged": before_checksum == after_checksum,
        },
        "gpu_memory_initial": int(mem0),
        "gpu_memory_final": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        "allow_next_gate": "phase3_fm_backprop_ab" if pass_gate else None,
        "overall_status": "NOT_READY_FOR_TRAINING",
        "next_gate": "phase3_fm_backprop_ab" if pass_gate else "blocked_on_phase3_fm_forward",
    }
    jdump(report, args.out / "phase3_fm_forward_only_report.json")

    if args.final_delivery.exists():
        fd = json.loads(args.final_delivery.read_text())
        fd["phase3_fm_forward_only"] = {
            "status": report["status"],
            "report": "../fm_only/phase3_fm_forward_only_report.json",
            "sample_metrics": "../fm_only/fm_forward_sample_metrics.jsonl",
            "target_latent_pair_l2": target_latent_pair_l2,
            "losses": [m["fm_loss_raw"] for m in sample_metrics],
        }
        fd["phase3_fm_only"] = {
            "status": "NOT_EXECUTED",
            "reason": "Only FM forward-only gate was executed; no backward/optimizer/FSDP/overfit was run.",
        }
        fd["overall_status"] = "NOT_READY_FOR_TRAINING"
        fd["next_gate"] = report["next_gate"]
        jdump(fd, args.final_delivery)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if pass_gate else 5


if __name__ == "__main__":
    raise SystemExit(main())
