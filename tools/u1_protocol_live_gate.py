#!/usr/bin/env python3
"""Bounded real-U1 Phase-4 protocol live test (generation + env.step only).

This diagnostic intentionally performs no backward pass, optimizer step, rollout
service launch, or training.  It loads one U1 checkpoint, generates exactly one
response for each requested real Active Spatial seed, and passes the response
through the production environment parser/reward path.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import sys
import time
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


DEFAULT_MODEL = Path("/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT")
DEFAULT_ENV = ROOT / "examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml"
DEFAULT_OUT = ROOT / "exps/vagen_active_spatial/u1_step32_diagnostic_snapshot/protocol_live"
DEFAULT_HISTORY = ROOT / "exps/vagen_active_spatial/u1_fwdfirst_rewscale_i2i_short_diag_20260803/rollout_data_full"


def dump_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    fields = {f.name for f in dataclasses.fields(ActiveSpatialEnvConfig)}
    return ActiveSpatialEnvConfig(**{k: v for k, v in values.items() if k in fields})


def first_images(obs: dict[str, Any]) -> list[Image.Image]:
    mm = obs.get("multi_modal_input") or obs.get("multi_modal_data") or {}
    raw = mm.get("<image>") or mm.get("image") or []
    return [x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x).convert("RGB") for x in raw]


def classify(text: str) -> str:
    has_tool = "<tool_call>" in text
    has_action = "<action>" in text
    if has_tool and has_action:
        return "tool_call_and_action"
    if has_tool:
        return "tool_call_only"
    if has_action:
        return "action_only"
    return "missing_or_invalid"


def summarize_formats(rows: list[dict[str, Any]], field: str = "output") -> dict[str, Any]:
    n = len(rows)
    classes = [classify(str(row.get(field) or "")) for row in rows]
    counts = {name: classes.count(name) for name in sorted(set(classes))}
    denom = max(n, 1)
    return {
        "n": n,
        "format_counts": counts,
        "strict_action_tag_rate": classes.count("action_only") / denom,
        "tool_call_rate": sum("tool_call" in name for name in classes) / denom,
        "missing_or_invalid_rate": classes.count("missing_or_invalid") / denom,
    }


def historical_summary(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    files = []
    for step in range(23, 28):
        current = path / f"{step}.jsonl"
        if not current.exists():
            continue
        files.append(str(current))
        rows.extend(json.loads(line) for line in current.read_text().splitlines() if line.strip())
    return {"files": files, **summarize_formats(rows)}


def decode_generated(tokenizer, generated: torch.Tensor, prompt_len: int) -> str:
    ids = generated[0]
    # U1 custom generate returns new ids only; standard HF returns prompt + new ids.
    if ids.shape[0] > prompt_len:
        ids = ids[prompt_len:]
    return tokenizer.decode(ids.tolist(), skip_special_tokens=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--env-yaml", type=Path, default=DEFAULT_ENV)
    ap.add_argument("--renderer-url", default="http://10.119.27.237:8768/render")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    ap.add_argument("--seeds", default="1520,440,830,1021,276,361,780,1200")
    ap.add_argument("--generation-seed", type=int, default=20260828)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--checkpoint-role", choices=("base", "trained"), default="base")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    random.seed(args.generation_seed)
    np.random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)

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
    # Direct HF generation does not run U1's ``chat()`` initializer.  Match
    # the production vLLM adapter by explicitly installing the processor's
    # registered multimodal token ids on the root model.
    model.img_context_token_id = processor.img_context_token_id
    model.img_start_token_id = processor.img_start_token_id

    from vagen.envs.active_spatial.env import ActiveSpatialEnv

    env = ActiveSpatialEnv(load_env_config(args.env_yaml, args.renderer_url))
    rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for index, seed in enumerate(seeds):
            torch.manual_seed(args.generation_seed + index)
            torch.cuda.manual_seed_all(args.generation_seed + index)
            obs, reset_info = env.reset(seed=seed)
            images = first_images(obs)
            if not images:
                raise RuntimeError(f"seed {seed}: reset returned no image")
            messages = [
                {"role": "system", "content": env.system_prompt()},
                {"role": "user", "content": obs["obs_str"]},
            ]
            prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
            inputs = processor(prompt, images=images, return_tensors="pt")
            prompt_len = int(inputs["input_ids"].shape[1])
            batch = {
                key: (value.cuda().to(torch.bfloat16) if key == "pixel_values" else value.cuda())
                for key, value in inputs.items()
            }
            t0 = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
            text = decode_generated(tokenizer, generated, prompt_len)
            parsed = env._default_parse_func(text)
            _, reward, done, step_info = env.step(text)
            row = {
                "index": index,
                "seed": seed,
                "scene_id": reset_info.get("scene_id"),
                "task_type": reset_info.get("task_type"),
                "object_label": reset_info.get("object_label"),
                "preset": reset_info.get("preset"),
                "prompt_tokens": prompt_len,
                "response_chars": len(text),
                "generation_seconds": time.time() - t0,
                "format_class": classify(text),
                "parser": {key: parsed.get(key) for key in (
                    "format_correct", "actions", "parse_error", "has_tool_call",
                    "has_strict_action_tag", "fallback_parse", "strict_parse_success",
                )},
                "reward": float(reward),
                "done": bool(done),
                "reward_protocol": {key: step_info.get(key) for key in (
                    "is_format_rewarded", "is_format_penalized", "format_correct",
                    "fallback_parse", "strict_parse_success", "has_tool_call",
                )},
                "output": text,
            }
            rows.append(row)
            dump_json(row, args.out / f"sample_{index:02d}_seed_{seed}.json")
            print(json.dumps({k: row[k] for k in ("index", "seed", "format_class", "parser", "reward_protocol")}, ensure_ascii=False), flush=True)
    finally:
        env.close()

    live = summarize_formats(rows)
    live["fallback_parse_rate"] = sum(bool(x["parser"].get("fallback_parse")) for x in rows) / max(len(rows), 1)
    live["strict_parse_success_rate"] = sum(bool(x["parser"].get("strict_parse_success")) for x in rows) / max(len(rows), 1)
    live["format_penalized_rate"] = sum(bool(x["reward_protocol"].get("is_format_penalized")) for x in rows) / max(len(rows), 1)

    protocol_errors = []
    for row in rows:
        parser = row["parser"]
        reward_protocol = row["reward_protocol"]
        if "tool_call" in row["format_class"] and parser.get("format_correct"):
            protocol_errors.append(f"seed {row['seed']}: tool_call incorrectly marked format_correct")
        if parser.get("format_correct") and not parser.get("strict_parse_success"):
            protocol_errors.append(f"seed {row['seed']}: format_correct without strict_parse_success")
        if not parser.get("format_correct") and not reward_protocol.get("is_format_penalized"):
            protocol_errors.append(f"seed {row['seed']}: non-strict response was not format-penalized")

    execution_status = "PASS" if len(rows) == len(seeds) and not protocol_errors else "FAIL"
    trained_checkpoint_available = args.checkpoint_role == "trained"
    gate_status = execution_status if trained_checkpoint_available else ("INCONCLUSIVE" if execution_status == "PASS" else "FAIL")
    report = {
        "phase": "phase4_protocol_live_test",
        "scope": "real U1 generation, real Active Spatial reset/step, one turn per seed; no backward/optimizer/training",
        "execution_status": execution_status,
        "gate_status": gate_status,
        "protocol_errors": protocol_errors,
        "model": str(args.model),
        "checkpoint_role": args.checkpoint_role,
        "trained_checkpoint_available": trained_checkpoint_available,
        "model_provenance_caveat": (
            "The requested step-32/actor-only trained checkpoint is absent from shared storage. "
            "This run uses the original U1 base model, so it validates the patched live parser/reward/telemetry "
            "chain but cannot establish trained-policy tool_call_rate decrease or strict_action_tag_rate increase."
            if not trained_checkpoint_available else None
        ),
        "renderer_url": args.renderer_url,
        "generation": {
            "seed": args.generation_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "historical_steps_23_27": historical_summary(args.history),
        "live": live,
        "directional_rate_comparison_valid": trained_checkpoint_available,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "next_gate": "phase5_combined_smoke" if gate_status == "PASS" else "blocked_on_phase4_protocol_trained_checkpoint",
    }
    dump_json(report, args.out / "phase4_protocol_live_report.json")
    print(json.dumps({k: report[k] for k in ("execution_status", "gate_status", "live", "next_gate")}, ensure_ascii=False, indent=2))
    return 0 if execution_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
