#!/usr/bin/env python3
"""Phase 3 gate: real U1 FM-only fixed-transition overfit.

This diagnostic is intentionally offline and bounded:
  * no rollout generation service is started;
  * no combined PPO/rollout smoke is launched;
  * writes only under the requested fm_only directory.

It exercises the VAGEN formal U1 FM path in
``vagen.models.sensenova_u1_register.SenseNovaU1ForCausalLMAdapter``:
rollout next-frame image -> processor.preprocess_images ->
u1_gen_pixel_values/u1_gen_grid_hw/u1_gen_valid ->
forward stash when U1_FM_BACKPROP=1 ->
compute_pending_fm_aux_loss() two-pass gen-only backward.
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
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageChops, ImageDraw
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
DEFAULT_MODEL = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/actor_only/step32_merged_hf"
DEFAULT_ENV_YAML = ROOT / "examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml"

TRANSITION_SPECS = [
    {"seed": 1520, "action": "move_forward"},
    {"seed": 440, "action": "turn_left"},
    {"seed": 830, "action": "turn_right"},
]


def jdump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_stats(path: Path) -> dict[str, Any]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "sha256": sha256_file(path),
    }


def image_diff_stats(a: Path, b: Path) -> dict[str, Any]:
    ia = Image.open(a).convert("RGB")
    ib = Image.open(b).convert("RGB")
    diff = ImageChops.difference(ia, ib)
    arr = np.asarray(diff, dtype=np.float32)
    return {
        "mean_abs_pixel_diff": float(arr.mean()),
        "max_abs_pixel_diff": float(arr.max()),
        "nonzero_fraction": float((arr > 0).mean()),
    }


def pose_vec(env) -> list[float]:
    E = np.asarray(env.view_engine.get_pose(), dtype=np.float64)
    return [float(x) for x in E[:3, 3].tolist()] + [float(x) for x in E[:3, 2].tolist()]


def first_image(obs: dict[str, Any]) -> Image.Image:
    mm = obs.get("multi_modal_input") or obs.get("multi_modal_data") or {}
    imgs = mm.get("<image>") or mm.get("image") or []
    if not imgs:
        raise RuntimeError(f"observation has no image keys: {list(obs.keys())}")
    img = imgs[0]
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(img).convert("RGB")


def load_env_config(yaml_path: Path, gpu_device: int | None):
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    doc = yaml.safe_load(yaml_path.read_text())
    cfg_dict = dict(doc["env1"]["env_config"])
    cfg_dict["render_backend"] = "local"
    if gpu_device is not None:
        cfg_dict["gpu_device"] = int(gpu_device)
    fields = {f.name for f in dataclasses.fields(ActiveSpatialEnvConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in fields}
    return ActiveSpatialEnvConfig(**cfg_dict)


def build_transitions(args) -> list[dict[str, Any]]:
    from vagen.envs.active_spatial.env import ActiveSpatialEnv

    out = args.out
    img_dir = out / "fixed_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    env = ActiveSpatialEnv(load_env_config(args.env_yaml, args.gpu_device))
    for idx, spec in enumerate(TRANSITION_SPECS):
        seed = int(spec["seed"])
        action = str(spec["action"])
        obs, info0 = env.reset(seed=seed)
        pose0 = pose_vec(env)
        cond = first_image(obs)
        cond_path = img_dir / f"condition_{idx}.png"
        cond.save(cond_path)
        action_text = f"<think>Phase-3 fixed transition probe.</think>\n<action>{action}|</action>"
        obs1, reward, done, info1 = env.step(action_text)
        pose1 = pose_vec(env)
        target = first_image(obs1)
        target_path = img_dir / f"target_{idx}.png"
        target.save(target_path)
        rec = {
            "idx": idx,
            "seed": seed,
            "action": action,
            "action_text": action_text,
            "scene_id": env.current_item.get("scene_id"),
            "task_id": "/".join([
                str(env.current_item.get("scene_id")),
                str(env.current_item.get("object_label")),
                str(env.current_item.get("preset")),
                str(seed),
            ]),
            "reward": float(reward),
            "done": bool(done),
            "action_executable": bool(info1.get("action_executable")),
            "strict_format_correct": bool(info1.get("strict_format_correct")),
            "env_feedback": info1.get("env_feedback"),
            "pose_before": pose0,
            "pose_after": pose1,
            "pose_delta_l2": float(np.linalg.norm(np.asarray(pose1) - np.asarray(pose0))),
            "condition_image": str(cond_path),
            "target_image": str(target_path),
            "condition_stats": image_stats(cond_path),
            "target_stats": image_stats(target_path),
            "image_diff": image_diff_stats(cond_path, target_path),
        }
        records.append(rec)
    path = out / "fixed_transitions.jsonl"
    path.write_text("", encoding="utf-8")
    for rec in records:
        append_jsonl(rec, path)
    return records


def write_dataflow(out: Path, transitions: list[dict[str, Any]] | None) -> dict[str, Any]:
    data = {
        "phase": "phase3_fm_only",
        "status": "DATAFLOW_TRACED",
        "formal_path": [
            "ActiveSpatialEnv.step(action_text)",
            "post-action obs['multi_modal_input']['<image>']",
            "AgentLoopOutput.extra_fields.nfp_target_images/nfp_valid",
            "SenseNovaU1ProcessorWrapper.preprocess_images(target[:1])",
            "multi_modal_inputs.u1_gen_pixel_values",
            "multi_modal_inputs.u1_gen_grid_hw",
            "multi_modal_inputs.u1_gen_valid",
            "SenseNovaU1ForCausalLMAdapter.forward(training=True)",
            "U1_FM_BACKPROP=0: _compute_fm_aux_split under no_grad, detached metrics-only loss",
            "U1_FM_BACKPROP=1: forward stashes CPU cloned gen tensors",
            "dp_actor.compute_pending_fm_aux_loss() calls gen-only FM backward",
        ],
        "code_anchors": {
            "agent_loop_next_frame": "vagen/agent_loop/gym_agent_loop_no_concat.py:383",
            "u1_mm_wiring": "vagen/agent_loop/agent_loop_no_concat.py:833",
            "u1_forward_fm_switch": "vagen/models/sensenova_u1_register.py:303",
            "u1_twopass_stash": "vagen/models/sensenova_u1_register.py:361",
            "u1_pending_backward_entry": "vagen/models/sensenova_u1_register.py:414",
            "actor_twopass_call": "verl/verl/workers/actor/dp_actor.py:566",
        },
        "expected_alignment": {
            "condition": "current observation image before action",
            "action": "strict <action> action applied by ActiveSpatialEnv.step",
            "target": "first image in observation rendered after that action",
            "valid": "u1_gen_valid=True only for non-terminal, non-empty next frame",
        },
        "transition_count": len(transitions or []),
        "actions": sorted({r["action"] for r in transitions or []}),
        "transitions": transitions or [],
    }
    jdump(data, out / "phase3_fm_dataflow.json")
    return data


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


def sanitize_hf_dir(path: Path) -> Path:
    cfg_path = path / "config.json"
    if not cfg_path.exists():
        return path
    data = json.loads(cfg_path.read_text())
    sanitized = _sanitize_config_obj(data)
    if sanitized == data:
        return path
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    out = Path("/tmp") / f"u1_phase3_hf_sanitized_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    for src in path.iterdir():
        dst = out / src.name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    (out / "config.json").write_text(json.dumps(sanitized, indent=2, ensure_ascii=False))
    return out


def load_transitions(out: Path) -> list[dict[str, Any]]:
    path = out / "fixed_transitions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_prompt(action: str) -> str:
    return (
        "<|im_start|>system\nYou are an embodied navigation agent. Respond with a brief rationale and exactly one action tag.<|im_end|>\n"
        "<|im_start|>user\n<image>\nMove according to the fixed Phase-3 transition probe.<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"<think>Use the fixed transition action.</think>\n<action>{action}|</action><|im_end|>"
    )


def load_model_and_processor(model_dir: Path, device: str):
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
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = SenseNovaU1ForCausalLMAdapter.from_pretrained(
        model_dir,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    processor = SenseNovaU1ProcessorWrapper(
        tokenizer,
        patch_size=int(getattr(cfg.vision_config, "patch_size", 16)),
        downsample_ratio=float(getattr(cfg.vision_config, "downsample_ratio", 0.5)),
    )
    return model, processor, tokenizer


def fm_named_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    keep = []
    pats = ("fm_modules", "fm_head", "timestep_embedder", "noise_scale_embedder", "mot_gen", "gen")
    for name, p in model.named_parameters():
        if any(pat in name for pat in pats):
            keep.append((name, p))
    # Avoid accidentally taking the whole LM because of generic "gen" in unrelated names.
    filtered = [
        (n, p)
        for n, p in keep
        if ("fm_modules" in n or "fm_head" in n or "timestep_embedder" in n or "noise_scale_embedder" in n or "mot_gen" in n)
    ]
    return filtered or keep


def all_param_delta(before: dict[str, torch.Tensor], named: list[tuple[str, torch.nn.Parameter]]) -> float:
    total = 0.0
    with torch.no_grad():
        for name, p in named:
            b = before.get(name)
            if b is None:
                continue
            total += float((p.detach().float().cpu() - b).pow(2).sum().item())
    return math.sqrt(total)


def grad_norm(named: list[tuple[str, torch.nn.Parameter]]) -> float:
    total = 0.0
    seen = False
    for _, p in named:
        if p.grad is None:
            continue
        seen = True
        total += float(p.grad.detach().float().pow(2).sum().item())
    return math.sqrt(total) if seen else 0.0


def prepare_sample(processor, rec: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    cond = Image.open(rec["condition_image"]).convert("RGB")
    target = Image.open(rec["target_image"]).convert("RGB")
    enc = processor(make_prompt(rec["action"]), images=[cond], return_tensors="pt")
    gen = processor.preprocess_images([target])
    sample = {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "pixel_values": enc["pixel_values"].to(device),
        "grid_hw": enc["grid_hw"].to(device),
        "u1_gen_pixel_values": gen["pixel_values"].to(device),
        "u1_gen_grid_hw": gen["grid_hw"].to(device),
        "u1_gen_valid": torch.tensor([1], dtype=torch.bool, device=device),
    }
    return sample


def fm_loss_once(model, sample: dict[str, torch.Tensor], backprop: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    os.environ["U1_FM_USE_UND_KV"] = "0"
    os.environ["U1_FM_BACKPROP"] = "1" if backprop else "0"
    model.train()
    out = model(**sample, labels=None, return_dict=True, use_cache=False)
    pending = bool(getattr(model, "_u1_pending_fm", None))
    if backprop:
        loss = model.compute_pending_fm_aux_loss()
        if loss is None:
            raise RuntimeError("BACKPROP=1 did not create pending FM loss")
    else:
        loss = out.loss
        if loss is None:
            raise RuntimeError("BACKPROP=0 did not return metrics-only FM loss")
    return loss, {"pending_created": pending, "forward_loss_is_none": out.loss is None}


def save_montage(records: list[dict[str, Any]], path: Path, title: str) -> None:
    tiles = []
    for r in records:
        for key in ("condition_image", "target_image"):
            img = Image.open(r[key]).convert("RGB").resize((192, 192))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, 191, 20], fill=(255, 255, 255))
            d.text((4, 4), f"{r['idx']} {key.split('_')[0]} {r['action']}", fill=(0, 0, 0))
            tiles.append(img)
    w = 2 * 192
    h = math.ceil(len(tiles) / 2) * 192
    out = Image.new("RGB", (w, h), (230, 230, 230))
    for i, img in enumerate(tiles):
        out.paste(img, ((i % 2) * 192, (i // 2) * 192))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def train_phase(args) -> dict[str, Any]:
    out = args.out
    transitions = load_transitions(out)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, processor, _ = load_model_and_processor(args.model, device)
    samples = [prepare_sample(processor, r, device) for r in transitions]

    for _, p in model.named_parameters():
        p.requires_grad_(False)
    fm_params = fm_named_params(model)
    for _, p in fm_params:
        p.requires_grad_(True)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    before_fm = {n: p.detach().float().cpu().clone() for n, p in trainable}
    before_frozen = {n: p.detach().float().cpu().clone() for n, p in frozen[:200]}

    ab = []
    for flag in (0, 1):
        model.zero_grad(set_to_none=True)
        loss, meta = fm_loss_once(model, samples[0], bool(flag))
        total_for_backward = loss if flag else loss * 0.0
        total_for_backward.backward()
        ab.append({
            "U1_FM_BACKPROP": flag,
            "loss": float(loss.detach().float().item()),
            "loss_finite": bool(torch.isfinite(loss).all().item()),
            "requires_grad": bool(loss.requires_grad),
            "pending_created": meta["pending_created"],
            "forward_loss_is_none": meta["forward_loss_is_none"],
            "fm_grad_norm_after_backward": grad_norm(trainable),
        })
    jdump({
        "status": "PASS" if ab[0]["fm_grad_norm_after_backward"] == 0.0 and ab[1]["fm_grad_norm_after_backward"] > 0.0 else "FAIL",
        "records": ab,
        "trainable_param_count": sum(p.numel() for _, p in trainable),
        "trainable_param_names_sample": [n for n, _ in trainable[:30]],
    }, out / "fm_backprop_ab_report.json")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    metrics_path = out / "fm_overfit_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    losses = []
    for step in range(args.steps):
        model.zero_grad(set_to_none=True)
        sample = samples[step % len(samples)]
        loss, _ = fm_loss_once(model, sample, True)
        loss.backward()
        gn = grad_norm(trainable)
        opt.step()
        val = float(loss.detach().float().item())
        losses.append(val)
        append_jsonl({
            "step": step,
            "loss": val,
            "loss_finite": bool(math.isfinite(val)),
            "fm_grad_norm": gn,
            "sample_idx": step % len(samples),
        }, metrics_path)

    fm_delta = all_param_delta(before_fm, trainable)
    frozen_delta_probe = all_param_delta(before_frozen, frozen[:200])
    jdump({
        "fm_l2_delta": fm_delta,
        "frozen_l2_delta_probe_first_200": frozen_delta_probe,
        "nonzero_fm_delta": fm_delta > 0,
        "frozen_probe_unchanged": frozen_delta_probe == 0.0,
    }, out / "fm_parameter_delta.json")

    state_path = out / "fm_only_trainable_state.pt"
    torch.save({n: p.detach().cpu() for n, p in trainable}, state_path)
    reload_model, reload_processor, _ = load_model_and_processor(args.model, device)
    for _, p in reload_model.named_parameters():
        p.requires_grad_(False)
    reload_trainable = fm_named_params(reload_model)
    state = torch.load(state_path, map_location="cpu")
    with torch.no_grad():
        for n, p in reload_trainable:
            if n in state:
                p.copy_(state[n].to(device=p.device, dtype=p.dtype))
    reload_sample = prepare_sample(reload_processor, transitions[0], device)
    reload_loss, _ = fm_loss_once(reload_model, reload_sample, True)
    jdump({
        "status": "PASS" if torch.isfinite(reload_loss).all().item() else "FAIL",
        "state_path": str(state_path),
        "reload_loss": float(reload_loss.detach().float().item()),
        "reload_loss_finite": bool(torch.isfinite(reload_loss).all().item()),
    }, out / "fm_reload_check.json")

    fsdp_report = {
        "status": "PASS",
        "scope": "single-process formal-module empty-storage proxy; no distributed rollout/smoke launched",
        "backprop_1_forward_backward_optimizer_save_reload": True,
        "empty_storage_tensors": [],
        "nonfinite_parameters": [],
    }
    for n, p in trainable:
        if p.detach().numel() == 0 or p.detach().storage().size() == 0:
            fsdp_report["empty_storage_tensors"].append(n)
        if not torch.isfinite(p.detach()).all().item():
            fsdp_report["nonfinite_parameters"].append(n)
    if fsdp_report["empty_storage_tensors"] or fsdp_report["nonfinite_parameters"]:
        fsdp_report["status"] = "FAIL"
    jdump(fsdp_report, out / "fsdp_fm_empty_storage_report.json")

    vis_dir = out / "visualizations"
    save_montage(transitions, vis_dir / "fixed_transition_condition_target_montage.png", "fixed")

    drop = (losses[0] - losses[-1]) if losses else 0.0
    report = {
        "phase": "phase3_fm_only",
        "status": "PASS" if (
            len({r["action"] for r in transitions}) >= 2
            and all(r["pose_delta_l2"] > 0 for r in transitions)
            and all(r["image_diff"]["mean_abs_pixel_diff"] > 0 for r in transitions)
            and ab[0]["fm_grad_norm_after_backward"] == 0.0
            and ab[1]["fm_grad_norm_after_backward"] > 0.0
            and fm_delta > 0
            and losses
            and losses[-1] < losses[0]
            and fsdp_report["status"] == "PASS"
            and bool(torch.isfinite(reload_loss).all().item())
        ) else "FAIL",
        "overall_status": "NOT_READY_FOR_TRAINING",
        "next_gate_if_pass": "phase4_protocol_live_test",
        "model": str(args.model),
        "device": device,
        "steps": args.steps,
        "lr": args.lr,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_drop": drop,
        "transition_count": len(transitions),
        "actions": sorted({r["action"] for r in transitions}),
        "artifacts": {
            "dataflow": str(out / "phase3_fm_dataflow.json"),
            "fixed_transitions": str(out / "fixed_transitions.jsonl"),
            "ab": str(out / "fm_backprop_ab_report.json"),
            "metrics": str(metrics_path),
            "delta": str(out / "fm_parameter_delta.json"),
            "reload": str(out / "fm_reload_check.json"),
            "empty_storage": str(out / "fsdp_fm_empty_storage_report.json"),
            "visualizations": str(vis_dir),
        },
    }
    jdump(report, out / "phase3_fm_only_report.json")
    return report


def maybe_update_final_delivery(final_path: Path, phase3_report: dict[str, Any] | None) -> None:
    if not final_path.exists() or phase3_report is None:
        return
    data = json.loads(final_path.read_text())
    data["overall_status"] = "NOT_READY_FOR_TRAINING"
    data["phase3_fm_only"] = {
        "status": phase3_report.get("status"),
        "report": "../fm_only/phase3_fm_only_report.json",
        "initial_loss": phase3_report.get("initial_loss"),
        "final_loss": phase3_report.get("final_loss"),
        "loss_drop": phase3_report.get("loss_drop"),
        "transition_count": phase3_report.get("transition_count"),
    }
    data["next_gate"] = "phase4_protocol_live_test" if phase3_report.get("status") == "PASS" else "phase3_fm_only_blocked"
    jdump(data, final_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["data", "train", "all"], default="all")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--env-yaml", type=Path, default=DEFAULT_ENV_YAML)
    ap.add_argument("--final-delivery", type=Path, default=DEFAULT_FINAL)
    ap.add_argument("--gpu-device", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    transitions = None
    if args.mode in ("data", "all"):
        try:
            transitions = build_transitions(args)
            write_dataflow(args.out, transitions)
        except Exception as exc:
            fail = {
                "phase": "phase3_fm_only",
                "status": "FAIL",
                "failed_stage": "fixed_transition_construction",
                "error": repr(exc),
                "overall_status": "NOT_READY_FOR_TRAINING",
                "next_gate": "phase3_fm_only_blocked",
            }
            jdump(fail, args.out / "phase3_fm_only_report.json")
            write_dataflow(args.out, transitions)
            print(json.dumps(fail, ensure_ascii=False, indent=2))
            return 2

    report = None
    if args.mode in ("train", "all"):
        try:
            report = train_phase(args)
            maybe_update_final_delivery(args.final_delivery, report)
        except Exception as exc:
            fail = {
                "phase": "phase3_fm_only",
                "status": "FAIL",
                "failed_stage": "fm_training",
                "error": repr(exc),
                "overall_status": "NOT_READY_FOR_TRAINING",
                "next_gate": "phase3_fm_only_blocked",
            }
            jdump(fail, args.out / "phase3_fm_only_report.json")
            maybe_update_final_delivery(args.final_delivery, fail)
            print(json.dumps(fail, ensure_ascii=False, indent=2))
            return 3
    print(json.dumps(report or {"status": "DATA_READY", "out": str(args.out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
