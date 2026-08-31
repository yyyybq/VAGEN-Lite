#!/usr/bin/env python3
"""Bounded Phase-3 gate for the real U1 FM_BACKPROP=0/1 behavior.

This entry point deliberately runs only one fixed transition through two
branches.  It does not start a renderer, rollout, FSDP, overfit loop, smoke
test, checkpoint save/reload, or long training.
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
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
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
DEFAULT_FINAL = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/reports/FINAL_DELIVERY.json"
DEFAULT_MODEL = Path("/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT")


def jdump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_hf_dir(path: Path) -> Path:
    cfg_path = path / "config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))

    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if obj in ("torch.float32", "torch.bfloat16", "torch.float16"):
            return obj.removeprefix("torch.")
        return obj

    sanitized = clean(data)
    if sanitized == data:
        return path
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    out = Path("/tmp") / f"u1_phase3_hf_sanitized_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    for src in path.iterdir():
        dst = out / src.name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    (out / "config.json").write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"no fixed transitions in {path}")
    return rows


def make_prompt(action: str) -> str:
    return (
        "<|im_start|>system\nYou are an embodied navigation agent. Respond with a brief rationale and exactly one action tag.<|im_end|>\n"
        "<|im_start|>user\n<image>\nMove according to the fixed Phase-3 transition probe.<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<think>Use the fixed transition action.</think>\n<action>{action}|</action><|im_end|>"
    )


def load_model_processor(model_dir: Path, device: str):
    def scalar(value: Any, default: Any) -> Any:
        if value is None:
            return default
        while isinstance(value, (tuple, list)) and len(value) == 1:
            value = value[0]
        return value

    model_dir = sanitize_hf_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    vc = getattr(cfg, "vision_config", None)
    for name in ("downsample_ratio", "llm_hidden_size"):
        value = getattr(vc, name, None)
        if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            setattr(vc, name, tuple(value[0]))
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = SenseNovaU1ForCausalLMAdapter.from_pretrained(
        model_dir,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = SenseNovaU1ProcessorWrapper(
        tokenizer,
        patch_size=int(scalar(getattr(cfg.vision_config, "patch_size", 16), 16)),
        downsample_ratio=float(scalar(getattr(cfg.vision_config, "downsample_ratio", 0.5), 0.5)),
    )
    return model, processor, dtype


def prepare_sample(processor, row: dict[str, Any], device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    condition = Image.open(row["condition_image"]).convert("RGB")
    target = Image.open(row["target_image"]).convert("RGB")
    encoded = processor(make_prompt(str(row["action"])), images=[condition], return_tensors="pt")
    generated = processor.preprocess_images([target])
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "pixel_values": encoded["pixel_values"].to(device=device, dtype=dtype),
        "grid_hw": encoded["grid_hw"].to(device),
        "u1_gen_pixel_values": generated["pixel_values"].to(device=device, dtype=dtype),
        "u1_gen_grid_hw": generated["grid_hw"].to(device),
        "u1_gen_valid": torch.tensor([True], device=device),
    }


def fm_group(name: str) -> str | None:
    if name.startswith("fm_modules.vision_model_mot_gen"):
        return "vision_model_mot_gen"
    if name.startswith("fm_modules.timestep_embedder"):
        return "timestep_embedder"
    if name.startswith("fm_modules.noise_scale_embedder"):
        return "noise_scale_embedder"
    if name.startswith("fm_modules.fm_head"):
        return "fm_head"
    if "_mot_gen" in name:
        return "language_model_mot_gen"
    return None


def select_fm_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    selected = [(name, param) for name, param in model.named_parameters() if fm_group(name) is not None]
    if not selected:
        raise RuntimeError("formal U1 FM parameter selection is empty")
    return selected


def snapshot_cpu(named: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: param.detach().to(device="cpu", dtype=torch.float32).clone() for name, param in named}


def parameter_signature(named: list[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    sq = 0.0
    total = 0
    finite = True
    for _, param in named:
        value = param.detach().float()
        sq += float(value.square().sum().item())
        total += param.numel()
        finite = finite and bool(torch.isfinite(value).all().item())
    return {"parameter_count": total, "l2_norm": math.sqrt(sq), "finite": finite}


def module_metrics(
    named: list[tuple[str, torch.nn.Parameter]],
    before: dict[str, torch.Tensor],
    optimizer_ids: set[int],
) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "tensor_count": 0,
            "parameter_count": 0,
            "optimizer_tensor_count": 0,
            "grad_sq": 0.0,
            "delta_sq": 0.0,
            "grad_present_tensors": 0,
            "grad_finite": True,
            "parameter_finite": True,
        }
    )
    for name, param in named:
        group = fm_group(name)
        assert group is not None
        item = buckets[group]
        item["tensor_count"] += 1
        item["parameter_count"] += param.numel()
        item["optimizer_tensor_count"] += int(id(param) in optimizer_ids)
        current = param.detach().float()
        item["parameter_finite"] = item["parameter_finite"] and bool(torch.isfinite(current).all().item())
        item["delta_sq"] += float((current.cpu() - before[name]).square().sum().item())
        if param.grad is not None:
            grad = param.grad.detach().float()
            item["grad_present_tensors"] += 1
            item["grad_finite"] = item["grad_finite"] and bool(torch.isfinite(grad).all().item())
            item["grad_sq"] += float(grad.square().sum().item())
    result = {}
    for name, item in sorted(buckets.items()):
        result[name] = {
            "tensor_count": item["tensor_count"],
            "parameter_count": item["parameter_count"],
            "optimizer_group_membership": item["optimizer_tensor_count"] == item["tensor_count"],
            "optimizer_tensor_count": item["optimizer_tensor_count"],
            "grad_present_tensors": item["grad_present_tensors"],
            "grad_norm": math.sqrt(item["grad_sq"]),
            "grad_finite": item["grad_finite"],
            "parameter_delta_l2": math.sqrt(item["delta_sq"]),
            "parameter_finite": item["parameter_finite"],
        }
    return result


def totals(modules: dict[str, Any]) -> dict[str, Any]:
    return {
        "grad_norm": math.sqrt(sum(float(v["grad_norm"]) ** 2 for v in modules.values())),
        "parameter_delta_l2": math.sqrt(sum(float(v["parameter_delta_l2"]) ** 2 for v in modules.values())),
        "all_grad_finite": all(bool(v["grad_finite"]) for v in modules.values()),
        "all_parameters_finite": all(bool(v["parameter_finite"]) for v in modules.values()),
        "all_in_optimizer": all(bool(v["optimizer_group_membership"]) for v in modules.values()),
    }


def gpu_memory() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }


def update_final(path: Path, report: dict[str, Any]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    status = report["status"]
    data["phase3_fm_backprop_ab"] = {
        "status": status,
        "report": "../fm_only/phase3_fm_backprop_ab_report.json",
        "model": report.get("model"),
        "sample_idx": report.get("sample", {}).get("idx"),
        "backprop_0_grad_norm": report.get("backprop_0", {}).get("totals", {}).get("grad_norm"),
        "backprop_0_parameter_delta": report.get("backprop_0", {}).get("totals", {}).get("parameter_delta_l2"),
        "backprop_1_grad_norm": report.get("backprop_1", {}).get("totals", {}).get("grad_norm"),
        "backprop_1_parameter_delta": report.get("backprop_1", {}).get("totals", {}).get("parameter_delta_l2"),
    }
    data["phase3_fm_only"] = {
        "status": "NOT_EXECUTED",
        "reason": "Data, forward-only, and BACKPROP A/B gates were executed; FSDP, overfit, visualization, and save/reload gates remain pending.",
    }
    data["overall_status"] = "NOT_READY_FOR_TRAINING"
    data["next_gate"] = "phase3_fsdp_empty_storage" if status == "PASS" else "blocked_on_phase3_fm_backprop_ab"
    jdump(data, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixed-transitions", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--final-delivery", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "phase3_fm_backprop_ab_report.json"
    started = time.time()

    report: dict[str, Any] = {
        "phase": "phase3_fm_backprop_ab",
        "status": "RUNNING",
        "scope": "one real fixed transition; BACKPROP=0/1 only; no renderer/FSDP/overfit/smoke/save-reload",
        "model": str(args.model),
        "model_substitution": (
            "The prior step32_merged_hf path is absent. The original SenseNova-U1 FM initialization is used. "
            "Prior U1 training had U1_FM_BACKPROP=0, and this gate uses U1_FM_USE_UND_KV=0, so actor-only "
            "weights are outside the pending FM loss graph."
        ),
        "seed": args.seed,
        "optimizer": {"class": "torch.optim.SGD", "lr": args.lr, "momentum": 0.0, "weight_decay": 0.0},
    }
    jdump(report, report_path)

    try:
        if not args.model.is_dir():
            raise FileNotFoundError(f"model directory missing: {args.model}")
        transitions_path = args.fixed_transitions or (args.out / "fixed_transitions.jsonl")
        rows = load_rows(transitions_path)
        row = rows[args.sample_index]
        for key in ("condition_image", "target_image"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"fixed-transition image missing: {row[key]}")

        seed_everything(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, processor, dtype = load_model_processor(args.model, args.device)
        model.train()
        sample = prepare_sample(processor, row, args.device, dtype)

        for param in model.parameters():
            param.requires_grad_(False)
        named = select_fm_parameters(model)
        for _, param in named:
            param.requires_grad_(True)
        frozen_trainable = [name for name, param in model.named_parameters() if param.requires_grad and fm_group(name) is None]
        if frozen_trainable:
            raise RuntimeError(f"unexpected trainable non-FM parameters: {frozen_trainable[:10]}")

        before = snapshot_cpu(named)
        initial_signature = parameter_signature(named)
        optimizer = torch.optim.SGD([param for _, param in named], lr=args.lr, momentum=0.0, weight_decay=0.0)
        optimizer_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}

        # BACKPROP=0: formal metrics-only forward. No pending pass, backward, or optimizer step.
        os.environ["U1_FM_USE_UND_KV"] = "0"
        os.environ["U1_FM_BACKPROP"] = "0"
        model.zero_grad(set_to_none=True)
        seed_everything(args.seed)
        pending0_before = bool(getattr(model, "_u1_pending_fm", None))
        out0 = model(**sample, labels=None, return_dict=True, use_cache=False)
        pending0_after = bool(getattr(model, "_u1_pending_fm", None))
        if out0.loss is None:
            raise RuntimeError("BACKPROP=0 formal forward returned no FM metric loss")
        modules0 = module_metrics(named, before, optimizer_ids)
        total0 = totals(modules0)
        backprop0 = {
            "forward_executed": True,
            "fm_loss_raw": float(out0.loss.detach().float().item()),
            "loss_finite": bool(torch.isfinite(out0.loss).all().item()),
            "loss_requires_grad": bool(out0.loss.requires_grad),
            "pending_before_forward": pending0_before,
            "pending_after_forward": pending0_after,
            "compute_pending_fm_aux_loss_called": False,
            "backward_executed": False,
            "optimizer_step_executed": False,
            "modules": modules0,
            "totals": total0,
        }

        # BACKPROP=1: identical unchanged initialization, sample, RNG seed and optimizer.
        os.environ["U1_FM_BACKPROP"] = "1"
        model.zero_grad(set_to_none=True)
        seed_everything(args.seed)
        pending1_before = bool(getattr(model, "_u1_pending_fm", None))
        out1 = model(**sample, labels=None, return_dict=True, use_cache=False)
        pending1_after_forward = bool(getattr(model, "_u1_pending_fm", None))
        pending_loss = model.compute_pending_fm_aux_loss()
        pending1_after_compute = bool(getattr(model, "_u1_pending_fm", None))
        if pending_loss is None:
            raise RuntimeError("BACKPROP=1 did not return a pending FM loss")
        pending_loss.backward()
        modules1_before_step = module_metrics(named, before, optimizer_ids)
        totals1_before_step = totals(modules1_before_step)
        optimizer.step()
        modules1_after_step = module_metrics(named, before, optimizer_ids)
        totals1_after_step = totals(modules1_after_step)
        backprop1 = {
            "forward_executed": True,
            "forward_loss_is_none": out1.loss is None,
            "pending_before_forward": pending1_before,
            "pending_after_forward": pending1_after_forward,
            "compute_pending_fm_aux_loss_called": True,
            "pending_after_compute": pending1_after_compute,
            "fm_loss_raw": float(pending_loss.detach().float().item()),
            "loss_finite": bool(torch.isfinite(pending_loss).all().item()),
            "loss_requires_grad": bool(pending_loss.requires_grad),
            "backward_executed": True,
            "optimizer_step_executed": True,
            "modules_before_optimizer_step": modules1_before_step,
            "modules": modules1_after_step,
            "totals_before_optimizer_step": totals1_before_step,
            "totals": totals1_after_step,
        }

        pass0 = (
            backprop0["loss_finite"]
            and not pending0_before
            and not pending0_after
            and total0["grad_norm"] == 0.0
            and total0["parameter_delta_l2"] == 0.0
        )
        pass1 = (
            backprop1["forward_loss_is_none"]
            and not pending1_before
            and pending1_after_forward
            and not pending1_after_compute
            and backprop1["loss_finite"]
            and backprop1["loss_requires_grad"]
            and totals1_before_step["grad_norm"] > 0.0
            and totals1_before_step["all_grad_finite"]
            and totals1_after_step["parameter_delta_l2"] > 0.0
            and totals1_after_step["all_parameters_finite"]
            and totals1_after_step["all_in_optimizer"]
        )
        report.update(
            {
                "status": "PASS" if pass0 and pass1 else "FAIL",
                "sample": {
                    "idx": row.get("idx", args.sample_index),
                    "scene_id": row.get("scene_id"),
                    "task_id": row.get("task_id"),
                    "action": row.get("action"),
                    "condition_image": row.get("condition_image"),
                    "target_image": row.get("target_image"),
                },
                "same_initialization_proof": {
                    "backprop_0_parameter_delta_before_backprop_1": total0["parameter_delta_l2"],
                    "unchanged": total0["parameter_delta_l2"] == 0.0,
                    "initial_fm_signature": initial_signature,
                },
                "trainable_scope": {
                    "groups": sorted(set(fm_group(name) for name, _ in named)),
                    "tensor_count": len(named),
                    "parameter_count": sum(param.numel() for _, param in named),
                    "non_fm_trainable_count": len(frozen_trainable),
                },
                "backprop_0": backprop0,
                "backprop_1": backprop1,
                "gpu_memory": gpu_memory(),
                "amp": {"enabled": False, "overflow": False},
                "elapsed_seconds": time.time() - started,
                "overall_status": "NOT_READY_FOR_TRAINING",
                "next_gate": "phase3_fsdp_empty_storage" if pass0 and pass1 else "blocked_on_phase3_fm_backprop_ab",
            }
        )
    except Exception as exc:
        report.update(
            {
                "status": "FAIL",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
                "overall_status": "NOT_READY_FOR_TRAINING",
                "next_gate": "blocked_on_phase3_fm_backprop_ab",
            }
        )

    jdump(report, report_path)
    update_final(args.final_delivery, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
