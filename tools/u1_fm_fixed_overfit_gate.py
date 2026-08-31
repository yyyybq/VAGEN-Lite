#!/usr/bin/env python3
"""Bounded real-U1 FM-only fixed-transition overfit and visualization gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(ROOT / "verl"))

from tools.u1_fm_backprop_ab_gate import (  # noqa: E402
    DEFAULT_FINAL,
    DEFAULT_MODEL,
    DEFAULT_OUT,
    fm_group,
    jdump,
    load_model_processor,
    load_rows,
    prepare_sample,
    seed_everything,
    select_fm_parameters,
    snapshot_cpu,
)
from vagen.models.sensenova_u1_register import IMG_CONTEXT_TOKEN_ID, IMG_START_TOKEN_ID  # noqa: E402


def all_param_delta(before: dict[str, torch.Tensor], named: list[tuple[str, torch.nn.Parameter]]) -> float:
    total = 0.0
    with torch.no_grad():
        for name, param in named:
            total += float((param.detach().float().cpu() - before[name]).square().sum().item())
    return math.sqrt(total)


def grad_norm(named: list[tuple[str, torch.nn.Parameter]]) -> float:
    total = 0.0
    for _, param in named:
        if param.grad is not None:
            total += float(param.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def append_jsonl(obj: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def fixed_prediction(model, sample: dict[str, torch.Tensor], seed: int) -> dict[str, Any]:
    """Mirror the formal _compute_fm_aux_split and expose x prediction/target."""
    seed_everything(seed)
    model.img_context_token_id = model.img_context_token_id or IMG_CONTEXT_TOKEN_ID
    model.img_start_token_id = model.img_start_token_id or IMG_START_TOKEN_ID
    indexes = model._build_indexes(sample["input_ids"], grid_hw=sample["grid_hw"], position_ids=None)
    gen_pv = sample["u1_gen_pixel_values"]
    gen_hw = sample["u1_gen_grid_hw"]
    merge = int(round(1 / model.downsample_ratio))
    token_h = int(gen_hw[0, 0].item()) // merge
    token_w = int(gen_hw[0, 1].item()) // merge
    n_gen = token_h * token_w
    t_eps = float(getattr(model.config, "t_eps", 0.05))

    noisy, noisy_x, target_v, timestep, noise_scale = model._prepare_single_gen_target(gen_pv, gen_hw)
    gen_vit = model.extract_feature(noisy, gen_model=True, grid_hw=gen_hw)
    t_emb = model.fm_modules["timestep_embedder"](timestep)
    if model.add_noise_scale_embedding:
        t_emb = t_emb + model.fm_modules["noise_scale_embedder"](noise_scale)
    gen_embeds = (gen_vit + t_emb.to(dtype=gen_vit.dtype)).unsqueeze(0)
    indexes_gen = model._build_t2i_image_indexes(
        token_h,
        token_w,
        int(indexes[0].max().item()) + 1,
        device=gen_embeds.device,
    )
    indicators = torch.ones((1, n_gen), dtype=torch.bool, device=gen_embeds.device)
    gen_out = model.language_model.model(
        inputs_embeds=gen_embeds,
        indexes=indexes_gen,
        attention_mask={"full_attention": None},
        past_key_values=None,
        use_cache=False,
        image_gen_indicators=indicators,
        update_cache=False,
    )
    hidden = gen_out.last_hidden_state[0]
    pred_x = model._predict_x_from_hidden(hidden, timestep, token_h, token_w)
    denom = (1 - timestep.view(-1, 1)).clamp_min(t_eps)
    target_x = noisy_x + denom * target_v
    pred_v = (pred_x - noisy_x) / denom
    fm_loss = F.mse_loss(pred_v.float(), target_v.float())
    latent_rmse = torch.sqrt(F.mse_loss(pred_x.float(), target_x.float()))
    return {
        "fm_loss": fm_loss,
        "latent_rmse": latent_rmse,
        "pred_x": pred_x,
        "target_x": target_x,
        "timestep": float(timestep[0].detach().float().item()),
        "noise_scale": float(noise_scale[0].detach().float().item()),
        "token_hw": [token_h, token_w],
    }


def decode_merged_pixels(tokens: torch.Tensor, token_hw: list[int], path: Path) -> dict[str, Any]:
    token_h, token_w = token_hw
    patch_after_merge = int(round(math.sqrt(tokens.shape[-1] / 3)))
    image = (
        tokens.detach()
        .float()
        .view(token_h, token_w, patch_after_merge, patch_after_merge, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(token_h * patch_after_merge, token_w * patch_after_merge, 3)
    )
    clipped = image.clamp(-1, 1)
    array = ((clipped + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    return {
        "path": str(path),
        "shape": list(array.shape),
        "raw_min": float(image.min().item()),
        "raw_max": float(image.max().item()),
        "raw_mean": float(image.mean().item()),
        "raw_std": float(image.std().item()),
        "clipped_fraction": float(((image < -1) | (image > 1)).float().mean().item()),
    }


def sampled_signature(named: list[tuple[str, torch.nn.Parameter]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    norm_sq = 0.0
    total = 0
    finite = True
    for name, param in named:
        value = param.detach().float().view(-1)
        total += value.numel()
        norm_sq += float(value.square().sum().item())
        finite = finite and bool(torch.isfinite(value).all().item())
        sample = torch.cat((value[:16], value[-16:])).cpu().numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(sample)
    return {
        "parameter_count": total,
        "l2_norm": math.sqrt(norm_sq),
        "sampled_sha256": digest.hexdigest(),
        "finite": finite,
    }


def update_final(path: Path, report: dict[str, Any]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["phase3_fm_fixed_transition_overfit"] = {
        "status": report["status"],
        "report": "../fm_only/phase3_fm_fixed_overfit_report.json",
        "initial_fm_loss": report.get("initial", {}).get("fm_loss_raw"),
        "final_fm_loss": report.get("final", {}).get("fm_loss_raw"),
        "loss_ratio": report.get("loss_ratio"),
        "initial_latent_rmse": report.get("initial", {}).get("latent_rmse"),
        "final_latent_rmse": report.get("final", {}).get("latent_rmse"),
        "parameter_delta_l2": report.get("parameter_delta_l2"),
        "checkpoint": report.get("checkpoint"),
    }
    data["phase3_fm_only"] = {
        "status": "NOT_EXECUTED",
        "reason": "All Phase-3 gates through fixed-transition overfit/visualization were executed; independent save/reload remains pending.",
    }
    data["overall_status"] = "NOT_READY_FOR_TRAINING"
    data["next_gate"] = "phase3_fm_save_reload" if report["status"] == "PASS" else "blocked_on_phase3_fm_overfit"
    jdump(data, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixed-transitions", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--final-delivery", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--fm-coef", type=float, default=0.1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "phase3_fm_fixed_overfit_report.json"
    metrics_path = args.out / "phase3_fm_overfit_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    vis_dir = args.out / "visualizations"
    started = time.time()
    report: dict[str, Any] = {
        "phase": "phase3_fm_fixed_transition_overfit",
        "status": "RUNNING",
        "scope": "one fixed real transition, fixed noise/timestep, 20 updates; no rollout/FSDP/Phase4/long training",
        "model": str(args.model),
        "model_substitution": "Original U1 FM initialization; prior step32_merged_hf path is absent and prior FM_BACKPROP was 0.",
        "steps": args.steps,
        "seed": args.seed,
        "optimizer": {"class": "torch.optim.SGD", "lr": args.lr},
        "fm_coef": args.fm_coef,
    }
    jdump(report, report_path)

    try:
        rows = load_rows(args.fixed_transitions or (args.out / "fixed_transitions.jsonl"))
        row = rows[args.sample_index]
        seed_everything(args.seed)
        torch.cuda.reset_peak_memory_stats()
        model, processor, dtype = load_model_processor(args.model, args.device)
        model.train()
        sample = prepare_sample(processor, row, args.device, dtype)

        for param in model.parameters():
            param.requires_grad_(False)
        named = select_fm_parameters(model)
        for _, param in named:
            param.requires_grad_(True)
        before = snapshot_cpu(named)
        optimizer = torch.optim.SGD([param for _, param in named], lr=args.lr, momentum=0.0, weight_decay=0.0)
        os.environ["U1_FM_USE_UND_KV"] = "0"
        os.environ["U1_FM_BACKPROP"] = "1"

        captures: dict[str, Any] = {}

        def capture(label: str, step: int) -> dict[str, Any]:
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                diag = fixed_prediction(model, sample, args.seed)
            pred_meta = decode_merged_pixels(diag["pred_x"], diag["token_hw"], vis_dir / f"{label}_prediction.png")
            target_meta = decode_merged_pixels(diag["target_x"], diag["token_hw"], vis_dir / "target_latent_decoded.png")
            rec = {
                "step": step,
                "fm_loss_raw": float(diag["fm_loss"].item()),
                "fm_loss_weighted": float(diag["fm_loss"].item()) * args.fm_coef,
                "latent_rmse": float(diag["latent_rmse"].item()),
                "timestep": diag["timestep"],
                "noise_scale": diag["noise_scale"],
                "prediction": pred_meta,
                "target_latent_decoded": target_meta,
                "parameter_delta_l2": all_param_delta(before, named),
            }
            captures[label] = rec
            return rec

        initial = capture("step0", 0)
        append_jsonl({**initial, "kind": "evaluation"}, metrics_path)
        middle_step = max(1, args.steps // 2)
        train_records = []
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            seed_everything(args.seed)
            output = model(**sample, labels=None, return_dict=True, use_cache=False)
            pending_created = bool(getattr(model, "_u1_pending_fm", None))
            raw_loss = model.compute_pending_fm_aux_loss()
            if raw_loss is None:
                raise RuntimeError(f"step {step}: pending FM loss missing")
            weighted = raw_loss * args.fm_coef
            weighted.backward()
            current_grad_norm = grad_norm(named)
            optimizer.step()
            rec = {
                "kind": "train",
                "step": step,
                "fm_loss_raw": float(raw_loss.detach().float().item()),
                "fm_loss_weighted": float(weighted.detach().float().item()),
                "latent_rmse_from_objective": math.sqrt(float(raw_loss.detach().float().item()))
                * (1.0 - initial["timestep"]),
                "grad_norm": current_grad_norm,
                "pending_created": pending_created,
                "pending_cleared": not bool(getattr(model, "_u1_pending_fm", None)),
                "skipped": False,
                "amp_overflow": False,
                "gpu_allocated": torch.cuda.memory_allocated(),
                "gpu_reserved": torch.cuda.memory_reserved(),
            }
            train_records.append(rec)
            append_jsonl(rec, metrics_path)
            if step == middle_step:
                middle = capture("middle", step)
                append_jsonl({**middle, "kind": "evaluation"}, metrics_path)

        final = capture("final", args.steps)
        append_jsonl({**final, "kind": "evaluation"}, metrics_path)
        delta = all_param_delta(before, named)
        signature = sampled_signature(named)

        checkpoint = args.out / "fm_only_overfit_trainable_state.pt"
        cpu_state = {name: param.detach().cpu().clone() for name, param in named}
        torch.save(
            {
                "state_dict": cpu_state,
                "metadata": {
                    "base_model": str(args.model),
                    "seed": args.seed,
                    "sample_index": args.sample_index,
                    "steps": args.steps,
                    "lr": args.lr,
                    "fm_coef": args.fm_coef,
                    "signature": signature,
                    "final_fm_loss": final["fm_loss_raw"],
                    "final_latent_rmse": final["latent_rmse"],
                },
            },
            checkpoint,
        )

        loss_ratio = final["fm_loss_raw"] / initial["fm_loss_raw"]
        latent_ratio = final["latent_rmse"] / initial["latent_rmse"]
        passed = (
            math.isfinite(final["fm_loss_raw"])
            and loss_ratio <= 0.7
            and latent_ratio <= 0.9
            and delta > 0.0
            and all(record["grad_norm"] > 0 and math.isfinite(record["grad_norm"]) for record in train_records)
            and signature["finite"]
        )
        report.update(
            {
                "status": "PASS" if passed else "FAIL",
                "sample": {
                    "idx": row.get("idx", args.sample_index),
                    "scene_id": row.get("scene_id"),
                    "task_id": row.get("task_id"),
                    "action": row.get("action"),
                    "condition_image": row.get("condition_image"),
                    "target_image": row.get("target_image"),
                },
                "fixed_diagnostics": {"noise": True, "timestep": True, "augmentation": False, "input": True},
                "initial": initial,
                "middle": captures.get("middle"),
                "final": final,
                "loss_ratio": loss_ratio,
                "latent_ratio": latent_ratio,
                "pass_criteria": {
                    "fm_loss_ratio_max": 0.7,
                    "latent_rmse_ratio_max": 0.9,
                    "rationale": "FM loss must drop substantially; target/prediction latent distance must also clearly decrease.",
                },
                "parameter_delta_l2": delta,
                "trainable_signature": signature,
                "skipped_steps": sum(int(record["skipped"]) for record in train_records),
                "amp": {"enabled": False, "overflow": False},
                "gpu_memory": {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                    "peak_reserved": torch.cuda.max_memory_reserved(),
                },
                "metrics": str(metrics_path),
                "visualizations": str(vis_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "elapsed_seconds": time.time() - started,
                "overall_status": "NOT_READY_FOR_TRAINING",
                "next_gate": "phase3_fm_save_reload" if passed else "blocked_on_phase3_fm_overfit",
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
                "next_gate": "blocked_on_phase3_fm_overfit",
            }
        )

    jdump(report, report_path)
    update_final(args.final_delivery, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
