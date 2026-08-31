#!/usr/bin/env python3
"""D0.6 fixed-sequence HF full/incremental probe for Cambrian-S.

This script consumes the production fixed-sequence manifest captured by the
bounded D0.6 rollout smoke. It does not sample, train, or step an optimizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


IMAGE_TOKEN_INDEX = -200


def install_cache_compat_shim() -> None:
    """Patch the standalone HF probe for Cambrian's older cache API usage."""
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if hasattr(DynamicCache, "get_usable_length"):
        return

    def get_usable_length(self, new_seq_length: int | None = None, layer_idx: int = 0) -> int:
        del new_seq_length
        return int(self.get_seq_length(layer_idx))

    DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]


def install_decoder_cache_tuple_shim(model: Any) -> None:
    """Make current Qwen2DecoderLayer outputs fit Cambrian's old use_cache loop."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return
    for layer in layers:
        if getattr(layer, "_d0_6_cache_tuple_shim", False):
            continue
        old_forward = layer.forward

        def wrapped_forward(*args, _old_forward=old_forward, **kwargs):
            use_cache = bool(kwargs.get("use_cache", False))
            cache_obj = kwargs.get("past_key_value", kwargs.get("past_key_values", None))
            out = _old_forward(*args, **kwargs)
            if use_cache and torch.is_tensor(out):
                return (out, cache_obj)
            return out

        layer.forward = wrapped_forward
        layer._d0_6_cache_tuple_shim = True


def tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().float().cpu().reshape(-1)
    sample = flat[: min(16, flat.numel())]
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(flat).item()) if flat.numel() else 0.0,
        "first_values": [float(x) for x in sample.tolist()],
        "sha1_first_4096": hashlib.sha1(flat[: min(4096, flat.numel())].numpy().tobytes()).hexdigest(),
    }


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float().cpu().reshape(-1)
    bf = b.detach().float().cpu().reshape(-1)
    out: dict[str, Any] = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "checksum_a": hashlib.sha256(af.numpy().tobytes()).hexdigest(),
        "checksum_b": hashlib.sha256(bf.numpy().tobytes()).hexdigest(),
    }
    if af.shape != bf.shape:
        out["shape_mismatch"] = True
        return out
    diff = af - bf
    denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
    out.update(
        {
            "shape_mismatch": False,
            "max_abs_diff": float(diff.abs().max().item()) if diff.numel() else 0.0,
            "mean_abs_diff": float(diff.abs().mean().item()) if diff.numel() else 0.0,
            "l2": float(torch.linalg.vector_norm(diff).item()) if diff.numel() else 0.0,
            "cosine": float(torch.dot(af, bf).item() / denom.item()) if denom.item() != 0 else float("nan"),
        }
    )
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return float(xs[idx])


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = [v - mx for v in x]
    vy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in vx) * sum(v * v for v in vy))
    return float(sum(a * b for a, b in zip(vx, vy)) / denom) if denom else float("nan")


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def pair_stats(left: list[float], right: list[float], mask: list[bool]) -> dict[str, Any]:
    vals = [(l, r) for l, r, m in zip(left, right, mask) if m and math.isfinite(l) and math.isfinite(r)]
    if not vals:
        return {"token_count": 0}
    deltas = [l - r for l, r in vals]
    abs_d = [abs(d) for d in deltas]
    ratios = [math.exp(max(-80.0, min(80.0, d))) for d in deltas]
    lvals = [v[0] for v in vals]
    rvals = [v[1] for v in vals]
    return {
        "token_count": len(vals),
        "delta_logp_mean_first_minus_second": float(sum(deltas) / len(deltas)),
        "mean_abs_delta_logp": float(sum(abs_d) / len(abs_d)),
        "median_abs_delta_logp": quantile(abs_d, 0.5),
        "p95_abs_delta_logp": quantile(abs_d, 0.95),
        "max_abs_delta_logp": max(abs_d),
        "ratio_mean_exp_first_minus_second": float(sum(ratios) / len(ratios)),
        "ratio_median": quantile(ratios, 0.5),
        "ratio_p5": quantile(ratios, 0.05),
        "ratio_p95": quantile(ratios, 0.95),
        "frac_abs_ratio_minus_1_gt_0p01": float(sum(abs(r - 1.0) > 0.01 for r in ratios) / len(ratios)),
        "frac_abs_ratio_minus_1_gt_0p05": float(sum(abs(r - 1.0) > 0.05 for r in ratios) / len(ratios)),
        "pearson": pearson(lvals, rvals),
        "spearman": spearman(lvals, rvals),
    }


def token_type(token_id: int, start: int, end: int, text_l: str, eos_id: int | None, pad_id: int | None) -> str:
    if pad_id is not None and token_id == pad_id:
        return "pad"
    if eos_id is not None and token_id == eos_id:
        return "eos_special"

    def intersects(a: int, b: int, c: int, d: int) -> bool:
        return max(a, c) < min(b, d)

    for tag in ("<action>", "</action>", "<think>", "</think>"):
        pos = text_l.find(tag)
        while pos >= 0:
            if intersects(start, end, pos, pos + len(tag)):
                return "action_tag" if "action" in tag else "think_tag"
            pos = text_l.find(tag, pos + 1)

    action_start = text_l.find("<action>")
    action_end = text_l.find("</action>", action_start + 1) if action_start >= 0 else -1
    if action_start >= 0 and action_end >= 0 and start >= action_start + len("<action>") and end <= action_end:
        return "action_name"
    think_start = text_l.find("<think>")
    think_end = text_l.find("</think>", think_start + 1) if think_start >= 0 else -1
    if think_start >= 0 and (think_end < 0 or (start >= think_start + len("<think>") and end <= think_end)):
        return "think_text"
    return "other"


def classify_response(tokenizer: Any, sample: dict[str, Any]) -> list[str]:
    active_ids = sample["active_response_token_ids"]
    response_ids = sample["response_token_ids"]
    response_mask = [bool(x) for x in sample["response_mask"]]
    text = tokenizer.decode(active_ids, skip_special_tokens=False)
    text_l = text.lower()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    out: list[str] = []
    active_prefix: list[int] = []
    for tid, is_active in zip(response_ids, response_mask):
        if not is_active:
            out.append("pad")
            continue
        start = len(tokenizer.decode(active_prefix, skip_special_tokens=False))
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        end = start + len(piece)
        out.append(token_type(tid, start, end, text_l, eos_id, pad_id))
        active_prefix.append(tid)
    return out


def topk_case(logits: torch.Tensor, token_id: int, tokenizer: Any, k: int = 10) -> dict[str, Any]:
    logits_f = logits.detach().float().cpu()
    logp = torch.log_softmax(logits_f, dim=-1)
    sampled_logit = float(logits_f[token_id].item())
    sampled_logp = float(logp[token_id].item())
    rank = int((logits_f > logits_f[token_id]).sum().item() + 1)
    vals, ids = torch.topk(logp, k=min(k, logp.numel()))
    return {
        "sampled_token_logit": sampled_logit,
        "sampled_token_logprob": sampled_logp,
        "sampled_token_rank": rank,
        "logsumexp": float(torch.logsumexp(logits_f, dim=-1).item()),
        "top_token_ids": [int(x) for x in ids.tolist()],
        "top_decoded": [tokenizer.decode([int(x)], skip_special_tokens=False) for x in ids.tolist()],
        "top_logprobs": [float(x) for x in vals.tolist()],
        "top_logits": [float(logits_f[int(x)].item()) for x in ids.tolist()],
    }


def resolve_images(manifest: dict[str, Any], vision_manifest: Path, processor: Any) -> dict[int, dict[str, Any]]:
    records = []
    with vision_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    resolved: dict[int, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for sample in manifest["samples"]:
        target = sample.get("image_artifact", {}).get("pixel_values", {})
        best = None
        for rec in records:
            path = rec.get("pre_processor_path")
            if not path or path in seen_paths or not Path(path).exists():
                continue
            img = Image.open(path).convert("RGB")
            pv = processor.preprocess_images([img])["pixel_values"]
            dg = tensor_digest(pv)
            score = (
                int(dg.get("sha1_first_4096") == target.get("sha1_first_4096")),
                -abs(float(dg.get("mean", 0.0)) - float(target.get("mean", 0.0))),
                -abs(float(dg.get("std", 0.0)) - float(target.get("std", 0.0))),
                -abs(float(dg.get("l2", 0.0)) - float(target.get("l2", 0.0))),
            )
            cand = (score, rec, dg, pv)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None or best[0][0] != 1:
            raise RuntimeError(f"Could not resolve exact image for sample {sample['sample']}")
        _, rec, dg, pv = best
        seen_paths.add(rec["pre_processor_path"])
        resolved[int(sample["sample"])] = {
            "pre_processor_path": rec["pre_processor_path"],
            "pixel_values_path": rec.get("pixel_values_path"),
            "vision_sanity_record": rec,
            "recomputed_digest": dg,
            "pixel_values": pv,
        }
    return resolved


def score_sample(
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    pixel_values: torch.Tensor,
    device: torch.device,
    *,
    top_indices: set[int],
) -> dict[str, Any]:
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.tensor(sample["position_ids"], dtype=torch.long, device=device).unsqueeze(0)
    response_ids = sample["response_token_ids"]
    response_len = len(response_ids)
    prompt_len = input_ids.shape[1] - response_len
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

    full_logps: list[float] = []
    full_top: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[0]
        for j, tid in enumerate(response_ids):
            prev_abs = prompt_len + j - 1
            step_logits = logits[prev_abs]
            full_logps.append(float(torch.log_softmax(step_logits.float(), dim=-1)[tid].item()))
            if j in top_indices:
                full_top[j] = topk_case(step_logits, tid, tokenizer)
        del out, logits
        torch.cuda.empty_cache()

    incr_logps: list[float] = []
    incr_top: dict[int, dict[str, Any]] = {}
    with torch.inference_mode():
        prompt_ids = input_ids[:, :prompt_len]
        prompt_attn = attention_mask[:, :prompt_len]
        prompt_pos = position_ids[:, :prompt_len]
        out = model(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            position_ids=prompt_pos,
            pixel_values=pixel_values,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        first_logits = out.logits[0, -1]
        first_tid = response_ids[0]
        incr_logps.append(float(torch.log_softmax(first_logits.float(), dim=-1)[first_tid].item()))
        if 0 in top_indices:
            incr_top[0] = topk_case(first_logits, first_tid, tokenizer)
        del out

        for j in range(1, response_len):
            prev_tid = response_ids[j - 1]
            target_tid = response_ids[j]
            abs_prev = prompt_len + j - 1
            cur_len = prompt_len + j
            step_ids = torch.tensor([[prev_tid]], dtype=torch.long, device=device)
            step_attn = attention_mask[:, :cur_len]
            step_pos = position_ids[:, abs_prev : abs_prev + 1]
            out = model(
                input_ids=step_ids,
                attention_mask=step_attn,
                position_ids=step_pos,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            step_logits = out.logits[0, -1]
            incr_logps.append(float(torch.log_softmax(step_logits.float(), dim=-1)[target_tid].item()))
            if j in top_indices:
                incr_top[j] = topk_case(step_logits, target_tid, tokenizer)
            del out

    return {
        "hf_full_logps": full_logps,
        "hf_incremental_logps": incr_logps,
        "hf_full_top": full_top,
        "hf_incremental_top": incr_top,
    }


def vision_feature_probe(model: Any, sample: dict[str, Any], pixel_values: torch.Tensor, device: torch.device) -> dict[str, Any]:
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    pixel_values_dev = pixel_values.to(device=device, dtype=torch.bfloat16)
    cfg = model.config
    inner = model.get_model()
    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    start = int(image_positions[0].item())
    length = int(image_positions.numel())

    with torch.inference_mode():
        fsdp_embeds = model._embed_multimodal_batch(input_ids, pixel_values_dev)
        fsdp_visual = fsdp_embeds[:, start : start + length, :]

        encoded = model.encode_images([pixel_values_dev])[0]
        vt = inner.vision_tower_aux_list[0]
        vt_model = getattr(vt, "vision_tower", vt)
        vt_dtype = next(vt_model.parameters()).dtype
        direct_outputs = vt_model(pixel_values=pixel_values_dev.to(vt_dtype), output_hidden_states=True)
        config_select_layer = int(getattr(cfg, "mm_vision_select_layer", -1))
        layer_count = len(vt_model.vision_model.encoder.layers)
        effective_select_layer = config_select_layer
        if config_select_layer == -2 and layer_count == 26:
            effective_select_layer = -1
        config_selected = direct_outputs.hidden_states[config_select_layer]
        selected = direct_outputs.hidden_states[effective_select_layer]

        proj_dtype = inner.mm_projector[0].weight.dtype
        projected = inner.mm_projector(selected.to(proj_dtype)).to(pixel_values_dev.dtype)
        from vagen.models.cambrian_miv import append_image_newline, build_cambrian_visual_features, build_miv_features

        si_token_len = int(getattr(cfg, "si_token_len", 729))
        si_side_len = int(si_token_len**0.5)
        miv_token_len = int(getattr(cfg, "miv_token_len", 0) or 0)
        newline_expanded = append_image_newline(projected, inner.image_newline, si_side_len=si_side_len)
        miv = build_miv_features(projected, si_side_len=si_side_len, miv_token_len=miv_token_len)
        vllm_style_final, miv2 = build_cambrian_visual_features(
            projected,
            inner.image_newline,
            si_token_len=si_token_len,
            mm_use_newline=bool(getattr(cfg, "mm_use_im_newline_token", True)),
            nfp_head=bool(getattr(cfg, "nfp_head", False)),
            miv_token_len=miv_token_len,
        )

    return {
        "sample": int(sample["sample"]),
        "image_block_start": start,
        "image_block_length": length,
        "processed_pixels": tensor_digest(pixel_values),
        "config_select_layer": config_select_layer,
        "effective_select_layer": effective_select_layer,
        "vision_layer_count": layer_count,
        "config_selected_hidden_vs_encode_images": compare_tensors(encoded, config_selected),
        "selected_hidden_vs_encode_images": compare_tensors(encoded, selected),
        "selected_hidden": tensor_digest(selected),
        "projected_729": tensor_digest(projected),
        "newline_expanded_756": tensor_digest(newline_expanded),
        "miv_64": tensor_digest(miv),
        "miv_helper_consistency": compare_tensors(miv, miv2),
        "final_visual_embeddings": tensor_digest(vllm_style_final),
        "fsdp_scattered_visual_embeddings": tensor_digest(fsdp_visual),
        "fsdp_scatter_vs_vllm_style_final": compare_tensors(fsdp_visual, vllm_style_final),
    }


def write_outputs(
    manifest: dict[str, Any],
    tokenizer: Any,
    results: dict[int, dict[str, Any]],
    resolved_images: dict[int, dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = manifest["samples"]
    token_rows: list[dict[str, Any]] = []
    sample_metrics: dict[str, Any] = {}

    for sample in samples:
        sid = int(sample["sample"])
        res = results[sid]
        types = classify_response(tokenizer, sample)
        masks_by_type: dict[str, list[bool]] = {}
        active_mask = [bool(x) for x in sample["response_mask"]]
        masks_by_type["all_response"] = active_mask
        for t in ["think_text", "think_tag", "action_tag", "action_name", "eos_special", "pad", "other"]:
            masks_by_type[t] = [m and typ == t for m, typ in zip(active_mask, types)]

        paths = {
            "A_prod_fsdp_full": sample["fsdp_log_probs_T1"],
            "A_prime_standalone_hf_full": res["hf_full_logps"],
            "B_hf_incremental": res["hf_incremental_logps"],
            "C_vllm_production_incremental": sample["vllm_rollout_log_probs"],
        }
        pairs = [
            ("A_prod_vs_A_prime", "A_prod_fsdp_full", "A_prime_standalone_hf_full"),
            ("A_vs_B", "A_prod_fsdp_full", "B_hf_incremental"),
            ("A_vs_C", "A_prod_fsdp_full", "C_vllm_production_incremental"),
            ("B_vs_C", "B_hf_incremental", "C_vllm_production_incremental"),
            ("A_prime_vs_B", "A_prime_standalone_hf_full", "B_hf_incremental"),
        ]
        sample_metrics[str(sid)] = {
            "selection": sample.get("selection"),
            "request_id": sample.get("request_id"),
            "image_path": resolved_images[sid]["pre_processor_path"],
            "pair_stats": {
                pair_name: {typ: pair_stats(paths[left], paths[right], mask) for typ, mask in masks_by_type.items()}
                for pair_name, left, right in pairs
            },
            "position_corr_abs_delta": {
                pair_name: pearson(
                    [float(i) for i, m in enumerate(active_mask) if m],
                    [abs(paths[left][i] - paths[right][i]) for i, m in enumerate(active_mask) if m],
                )
                for pair_name, left, right in pairs
            },
        }

        active_prefix: list[int] = []
        for idx, tid in enumerate(sample["response_token_ids"]):
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
            row = {
                "sample": sid,
                "selection": sample.get("selection"),
                "request_id": sample.get("request_id"),
                "response_idx": idx,
                "response_mask": int(active_mask[idx]),
                "token_id": int(tid),
                "decoded_token": decoded,
                "token_type": types[idx],
                "position_id": sample["position_ids"][len(sample["prompt_token_ids"]) + idx],
            }
            for name, vals in paths.items():
                row[name] = vals[idx]
            row["delta_A_minus_B"] = paths["A_prod_fsdp_full"][idx] - paths["B_hf_incremental"][idx]
            row["delta_A_minus_C"] = paths["A_prod_fsdp_full"][idx] - paths["C_vllm_production_incremental"][idx]
            row["delta_B_minus_C"] = paths["B_hf_incremental"][idx] - paths["C_vllm_production_incremental"][idx]
            token_rows.append(row)
            if active_mask[idx]:
                active_prefix.append(tid)

    aggregate_masks: dict[str, list[bool]] = {}
    aggregate_paths: dict[str, list[float]] = {}
    for sample in samples:
        sid = int(sample["sample"])
        res = results[sid]
        types = classify_response(tokenizer, sample)
        active_mask = [bool(x) for x in sample["response_mask"]]
        paths = {
            "A_prod_fsdp_full": sample["fsdp_log_probs_T1"],
            "A_prime_standalone_hf_full": res["hf_full_logps"],
            "B_hf_incremental": res["hf_incremental_logps"],
            "C_vllm_production_incremental": sample["vllm_rollout_log_probs"],
        }
        for name, vals in paths.items():
            aggregate_paths.setdefault(name, []).extend(vals)
        aggregate_masks.setdefault("all_response", []).extend(active_mask)
        for t in ["think_text", "think_tag", "action_tag", "action_name", "eos_special", "pad", "other"]:
            aggregate_masks.setdefault(t, []).extend([m and typ == t for m, typ in zip(active_mask, types)])

    pair_defs = [
        ("A_prod_vs_A_prime", "A_prod_fsdp_full", "A_prime_standalone_hf_full"),
        ("A_vs_B", "A_prod_fsdp_full", "B_hf_incremental"),
        ("A_vs_C", "A_prod_fsdp_full", "C_vllm_production_incremental"),
        ("B_vs_C", "B_hf_incremental", "C_vllm_production_incremental"),
        ("A_prime_vs_B", "A_prime_standalone_hf_full", "B_hf_incremental"),
    ]
    scoring_matrix = {
        "tag": "d0_6_forced_scoring_matrix",
        "model_checkpoint": manifest.get("model_checkpoint"),
        "sampling_temperature": manifest.get("sampling_temperature"),
        "logprob_temperature": manifest.get("logprob_temperature"),
        "paths": {
            "A_prod_fsdp_full": "production FSDP/HF full-sequence old_log_prob captured before optimizer step",
            "A_prime_standalone_hf_full": "standalone HF adapter full-sequence reconstruction from manifest",
            "B_hf_incremental": "standalone HF adapter teacher-forced incremental KV-cache scoring",
            "C_vllm_production_incremental": "production vLLM sampled-token raw logprob from rollout",
            "D_vllm_forced_scoring": "NOT AVAILABLE in this run",
        },
        "aggregate_pair_stats": {
            pair_name: {
                typ: pair_stats(aggregate_paths[left], aggregate_paths[right], mask)
                for typ, mask in aggregate_masks.items()
            }
            for pair_name, left, right in pair_defs
        },
        "sample_metrics": sample_metrics,
    }

    with (out_dir / "d0_6_scoring_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(scoring_matrix, f, indent=2, ensure_ascii=False, sort_keys=True)

    with (out_dir / "d0_6_token_level_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(token_rows[0].keys()))
        writer.writeheader()
        writer.writerows(token_rows)

    top_cases: list[dict[str, Any]] = []
    for sample in samples:
        sid = int(sample["sample"])
        res = results[sid]
        by_idx = {int(x["idx"]): x for x in sample.get("top_residual_tokens", [])[:10]}
        for idx, src in by_idx.items():
            case = {
                "sample": sid,
                "selection": sample.get("selection"),
                "request_id": sample.get("request_id"),
                "response_idx": idx,
                "token_id": int(sample["response_token_ids"][idx]),
                "decoded_token": tokenizer.decode([int(sample["response_token_ids"][idx])], skip_special_tokens=False),
                "source_residual": src,
                "A_prod_logprob": sample["fsdp_log_probs_T1"][idx],
                "A_prime_full_logprob": res["hf_full_logps"][idx],
                "B_incremental_logprob": res["hf_incremental_logps"][idx],
                "C_vllm_logprob": sample["vllm_rollout_log_probs"][idx],
                "A_prime_full_top": res["hf_full_top"].get(idx),
                "B_incremental_top": res["hf_incremental_top"].get(idx),
            }
            top_cases.append(case)
    with (out_dir / "d0_6_top_logits_cases.json").open("w", encoding="utf-8") as f:
        json.dump({"tag": "d0_6_top_logits_cases", "cases": top_cases}, f, indent=2, ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    parser.add_argument("--vision-manifest", default="exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/manifest.jsonl")
    parser.add_argument("--out-dir", default="docs/diagnosis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()

    os.environ.setdefault("CAMBRIAN_SRC", "/mnt/umm/users/yinbaiqiao/cambrian-s")
    os.environ.setdefault("VAGEN_ALLOW_HF_DOWNLOAD", "0")

    install_cache_compat_shim()

    import vagen.models.cambrian_register  # noqa: F401
    from vagen.models.cambrian_processor import CambrianProcessorWrapper
    from vagen.models.cambrian_vllm import _preprocess_images as vllm_preprocess_images

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    checkpoint = manifest["model_checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    processor = CambrianProcessorWrapper(tokenizer)
    resolved_images = resolve_images(manifest, Path(args.vision_manifest), processor)

    device = torch.device(args.device)
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    install_decoder_cache_tuple_shim(model)

    results: dict[int, dict[str, Any]] = {}
    vision_records: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        sid = int(sample["sample"])
        pixel_values = resolved_images[sid]["pixel_values"]
        vllm_pixels = vllm_preprocess_images([Image.open(resolved_images[sid]["pre_processor_path"]).convert("RGB")])
        vision = vision_feature_probe(model, sample, pixel_values, device)
        vision["vllm_preprocess_vs_fsdp_processor"] = compare_tensors(vllm_pixels, pixel_values)
        vision["resolved_image"] = {
            key: value for key, value in resolved_images[sid].items() if key != "pixel_values"
        }
        vision_records.append(vision)

        top_indices = {int(x["idx"]) for x in sample.get("top_residual_tokens", [])[:10]}
        results[sid] = score_sample(model, tokenizer, sample, pixel_values, device, top_indices=top_indices)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "d0_6_vision_feature_equality.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": "d0_6_vision_feature_equality",
                "note": "Standalone paired probe using actual CambrianForCausalLMAdapter modules plus the vLLM preprocessing/model construction sequence. Production vLLM weight equality is covered by d0_6_weight_equality*.json.",
                "records": vision_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    write_outputs(manifest, tokenizer, results, resolved_images, out_dir)

    mode_backend = {
        "tag": "d0_6_mode_backend_audit",
        "standalone_hf_model_training": bool(model.training),
        "standalone_hf_dtype": str(torch_dtype),
        "standalone_hf_attention_implementation": getattr(model.config, "_attn_implementation", None),
        "config_attention_dropout": getattr(model.config, "attention_dropout", None),
        "config_hidden_dropout": getattr(model.config, "hidden_dropout", None),
        "config_use_cache": getattr(model.config, "use_cache", None),
        "production_vllm_from_smoke_log": {
            "dtype": "torch.bfloat16",
            "attention_backend": "TORCH_SDPA / enforce_eager observed in D0.6b smoke",
            "kv_cache_dtype": "auto",
            "prefix_caching": True,
            "chunked_prefill": True,
            "temperature": manifest.get("sampling_temperature"),
            "logprob_temperature": manifest.get("logprob_temperature"),
        },
        "dropout_hypothesis": "attention_dropout is 0.0 in Cambrian config; standalone model is eval(). No stochastic dropout evidence in policy scoring path.",
    }
    with (out_dir / "d0_6_mode_backend_audit.json").open("w", encoding="utf-8") as f:
        json.dump(mode_backend, f, indent=2, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
