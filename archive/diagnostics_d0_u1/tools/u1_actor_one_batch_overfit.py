#!/usr/bin/env python3
"""Deterministic, single-sample U1 actor-head overfit diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from vagen.models.sensenova_u1_processor import SenseNovaU1ProcessorWrapper
import vagen.models.u1_neo_compat  # noqa: F401  # promote the PPO/FSDP adapter onto HF dynamic U1 classes


ACTIONS = ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right")


def nested_u1_config(path: Path):
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    vc = cfg.vision_config
    for name in ("downsample_ratio", "llm_hidden_size"):
        value = getattr(vc, name)
        if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            setattr(vc, name, tuple(value[0]))
    return cfg


def find_subsequence(haystack, needle):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    raise RuntimeError(f"target token sequence not found (needle length={len(needle)})")


def trainable_checksum(params):
    h = hashlib.sha256()
    with torch.no_grad():
        for p in params:
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def classify(text):
    strict = bool(re.search(r"<action>\s*(?:move_forward|move_backward|move_left|move_right|turn_left|turn_right)(?:\|(?:move_forward|move_backward|move_left|move_right|turn_left|turn_right))*\|?\s*</action>", text))
    tool = "<tool_call>" in text
    unknown = False
    bodies = re.findall(r"<action>(.*?)</action>", text, flags=re.S)
    for body in bodies:
        unknown |= any(x.strip() and x.strip() not in ACTIONS for x in body.split("|"))
    return strict, tool, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    rows = [json.loads(x) for x in args.batch.read_text().splitlines() if x.strip()]
    sample = rows[0]
    assert "<tool_call>" not in sample["output"]
    prompt, response = sample["input"], sample["output"]
    action_match = re.search(r"<action>.*?</action>", response, flags=re.S)
    if not action_match:
        raise RuntimeError("legal action span missing")
    action_text = action_match.group(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = SenseNovaU1ProcessorWrapper(tokenizer)
    cfg = nested_u1_config(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).cuda().eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [(n, p) for n, p in model.named_parameters() if "lm_head" in n]
    if not trainable:
        raise RuntimeError("no lm_head parameters found")
    for _, p in trainable:
        p.requires_grad_(True)
    params = [p for _, p in trainable]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    image = Image.open(args.image).convert("RGB")
    prompt_batch = processor(prompt, images=[image], return_tensors="pt")
    full_batch = processor(prompt + response, images=[image], return_tensors="pt")
    full_ids = full_batch["input_ids"][0].tolist()
    prompt_len = prompt_batch["input_ids"].shape[1]
    action_ids = tokenizer.encode(action_text, add_special_tokens=False)
    action_start = find_subsequence(full_ids[prompt_len:], action_ids) + prompt_len
    action_positions = list(range(action_start, action_start + len(action_ids)))
    labels = full_batch["input_ids"].clone()
    labels[:, :prompt_len] = -100
    batch = {
        k: (v.cuda().to(torch.bfloat16) if k == "pixel_values" else v.cuda())
        for k, v in full_batch.items()
    }
    labels = labels.cuda()
    initial = [p.detach().float().cpu().clone() for p in params]
    checksum_before = trainable_checksum(params)

    def evaluate(step, do_generate=False):
        model.eval()
        with torch.no_grad():
            out = model(**batch, labels=labels, use_cache=False, return_dict=True)
            shift_logits = out.logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            valid = shift_labels.ne(-100)
            full_nll = F.cross_entropy(shift_logits[valid], shift_labels[valid]).item()
            pos = torch.tensor([p - 1 for p in action_positions], device="cuda")
            tgt = labels[0, action_positions]
            lp = F.log_softmax(shift_logits[0, pos], dim=-1)
            target_lp = lp.gather(-1, tgt[:, None]).squeeze(-1)
            alt_ids = sorted({i for a in ACTIONS for i in tokenizer.encode(a, add_special_tokens=False)})
            alt = torch.tensor(alt_ids, device="cuda")
            non_target_lp = lp[:, alt].mean().item()
            teacher_text = tokenizer.decode(shift_logits[0, prompt_len - 1:].argmax(-1).tolist())
            decoded = teacher_text
            if do_generate:
                pb = {
                    k: (v.cuda().to(torch.bfloat16) if k == "pixel_values" else v.cuda())
                    for k, v in prompt_batch.items()
                }
                generated = model.generate(**pb, do_sample=False, max_new_tokens=384)
                # U1's custom generate returns only newly generated ids, while
                # standard HF generate returns prompt+new ids.
                new_ids = generated[0] if generated.shape[1] <= prompt_len else generated[0, prompt_len:]
                decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
        strict, tool, unknown = classify(decoded)
        exact = action_text in decoded
        return {
            "step": step, "pg_loss": full_nll, "full_response_nll": full_nll,
            "action_span_nll": float(-target_lp.mean()), "target_action_logprob": float(target_lp.mean()),
            "target_action_probability": float(target_lp.exp().mean()), "non_target_action_logprob": non_target_lp,
            "strict_action_decode_rate": float(strict), "exact_target_action_rate": float(exact),
            "tool_call_rate": float(tool), "unknown_action_rate": float(unknown), "decode": decoded,
        }

    records = [evaluate(0, True)]
    log_steps = {1, max(1, args.steps // 4), max(1, args.steps // 2), args.steps}
    for step in range(1, args.steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        out = model(**batch, labels=labels, use_cache=False, return_dict=True)
        logits = out.logits[:, :-1].float(); shifted = labels[:, 1:]; valid = shifted.ne(-100)
        full_loss = F.cross_entropy(logits[valid], shifted[valid])
        pos = torch.tensor([p - 1 for p in action_positions], device="cuda")
        action_loss = F.cross_entropy(logits[0, pos], labels[0, action_positions])
        loss = 0.5 * full_loss + 0.5 * action_loss
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
        if step in log_steps:
            rec = evaluate(step, step == args.steps); rec["actor_grad_norm"] = grad_norm; records.append(rec)

    with torch.no_grad():
        delta = math.sqrt(sum(float((p.detach().float().cpu() - b).pow(2).sum()) for p, b in zip(params, initial)))
    checksum_after = trainable_checksum(params)
    pre_reload = {
        "records": records, "actor_parameter_l2_delta": delta,
        "checksum_before": checksum_before, "checksum_after": checksum_after,
    }
    (args.out / "phase1_actor_only_pre_reload.json").write_text(
        json.dumps(pre_reload, indent=2, ensure_ascii=False)
    )
    save_dir = args.out / "checkpoint"
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(save_dir)
    # Promoting the training adapter onto a trust-remote-code class prevents
    # Transformers from discovering all of the original source dependencies.
    # Keep the diagnostic checkpoint self-contained for a genuine reload.
    for source in args.model.glob("*.py"):
        shutil.copy2(source, save_dir / source.name)
    del model, optimizer
    torch.cuda.empty_cache()

    reload_cfg = nested_u1_config(save_dir)
    reloaded = AutoModelForCausalLM.from_pretrained(
        save_dir, config=reload_cfg, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).cuda().eval()
    from vagen.models.u1_neo_compat import _promote_adapter_onto
    _promote_adapter_onto(reloaded.__class__, "actor-only reloaded checkpoint")
    model = reloaded
    reload_record = evaluate("reload", True)
    report = {
        "status": "PASS" if (
            records[-1]["action_span_nll"] < 0.5 * records[0]["action_span_nll"]
            and delta > 0
            and records[-1]["strict_action_decode_rate"] == 1.0
            and records[-1]["exact_target_action_rate"] == 1.0
            and reload_record["strict_action_decode_rate"] == 1.0
            and reload_record["exact_target_action_rate"] == 1.0
        ) else "FAIL",
        "seed": args.seed, "steps": args.steps, "lr": args.lr, "sample": {k: sample.get(k) for k in ("source_rollout","task_id","scene_id","task_type","format_class")},
        "target_action": action_text, "prompt_tokens": prompt_len, "response_tokens": len(full_ids)-prompt_len,
        "action_token_positions": action_positions, "trainable_names": [n for n, _ in trainable],
        "records": records, "actor_parameter_l2_delta": delta,
        "checksum_before": checksum_before, "checksum_after": checksum_after,
        "reload": reload_record,
        "fm_loss_enabled": False, "critic_loss_enabled": False, "rollout_resampling": False,
    }
    (args.out / "phase1_actor_only_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
