#!/usr/bin/env python3
"""D0.2 Cambrian-S-LFP train/rollout parity probes.

This script is intentionally diagnostic-only. It does not start RL training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


ACTION_STRINGS = [
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
]

IMAGE_TOKEN_TEXT = "<image>"


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().float().cpu()
    flat = x.flatten()
    return {
        "shape": list(x.shape),
        "dtype": str(t.dtype),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
    }


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    aa = a.detach().float().cpu()
    bb = b.detach().float().cpu()
    if aa.shape != bb.shape:
        return {
            "shape_a": list(aa.shape),
            "shape_b": list(bb.shape),
            "shape_mismatch": True,
        }
    diff = (aa - bb).abs()
    return {
        "shape_a": list(aa.shape),
        "shape_b": list(bb.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cosine": float(F.cosine_similarity(aa.flatten(), bb.flatten(), dim=0).item()),
    }


def summarize(values: list[float]) -> dict[str, float]:
    values = sorted(float(v) for v in values)
    if not values:
        return {}

    def pct(p: float) -> float:
        if len(values) == 1:
            return values[0]
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
        return values[idx]

    return {
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": values[0],
        "max": values[-1],
        "p5": pct(0.05),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }


def load_images(paths: list[str], limit: int) -> list[Image.Image]:
    images = []
    for p in paths[:limit]:
        images.append(Image.open(p).convert("RGB"))
    return images


def old_vllm_preprocess(images: list[Image.Image]) -> torch.Tensor:
    from transformers import SiglipImageProcessor
    from vagen.models.cambrian_vllm import SIGLIP_CACHE

    def expand2square(img: Image.Image, bg=(122, 116, 104)) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        out = Image.new("RGB", (side, side), bg)
        out.paste(img, ((side - w) // 2, (side - h) // 2))
        return out

    proc = SiglipImageProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384",
        cache_dir=SIGLIP_CACHE,
        local_files_only=True,
    )
    prepared = [expand2square(img.convert("RGB")) for img in images]
    return proc(images=prepared, return_tensors="pt").pixel_values


def processor_parity(args: argparse.Namespace) -> dict[str, Any]:
    from vagen.models.cambrian_processor import CambrianProcessorWrapper, preprocess_siglip_images
    from vagen.models.cambrian_vllm import _get_siglip_image_processor, _preprocess_images
    from transformers import AutoTokenizer

    images = load_images(args.images, args.limit)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    wrapper = CambrianProcessorWrapper(tok)
    training_pixels = wrapper.preprocess_images(images)["pixel_values"]
    vllm_pixels = _preprocess_images(images)
    old_pixels = old_vllm_preprocess(images)
    proc = _get_siglip_image_processor()

    return {
        "image_count": len(images),
        "training_stats": tensor_stats(training_pixels),
        "vllm_after_stats": tensor_stats(vllm_pixels),
        "vllm_before_stats": tensor_stats(old_pixels),
        "processor_class_after": proc.__class__.__name__,
        "processor_model_after": getattr(proc, "name_or_path", None),
        "before_vs_training": compare_tensors(training_pixels, old_pixels),
        "after_vs_training": compare_tensors(training_pixels, vllm_pixels),
        "direct_helper_reuse_check": compare_tensors(
            training_pixels,
            preprocess_siglip_images(images, wrapper._get_siglip_processor()),
        ),
    }


def miv_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    from vagen.models.cambrian_miv import build_cambrian_visual_features

    torch.manual_seed(0)
    projected = torch.randn(args.limit, 729, 3584, dtype=torch.float32)
    newline = torch.randn(3584, dtype=torch.float32)
    after, miv = build_cambrian_visual_features(
        projected,
        newline,
        si_token_len=729,
        mm_use_newline=True,
        nfp_head=True,
        miv_token_len=64,
    )
    before, _ = build_cambrian_visual_features(
        projected,
        newline,
        si_token_len=729,
        mm_use_newline=True,
        nfp_head=False,
        miv_token_len=64,
    )
    same_helper, _ = build_cambrian_visual_features(
        projected,
        newline,
        si_token_len=729,
        mm_use_newline=True,
        nfp_head=True,
        miv_token_len=64,
    )
    return {
        "sample_count": args.limit,
        "miv_shape": list(miv.shape),
        "visual_shape": list(after.shape),
        "before_vs_after_first64": compare_tensors(before[:, :64], after[:, :64]),
        "before_vs_after_full": compare_tensors(before, after),
        "shared_helper_self_check_first64": compare_tensors(after[:, :64], same_helper[:, :64]),
        "shared_helper_self_check_full": compare_tensors(after, same_helper),
    }


def load_vision_projector(model_path: str, device: str):
    from safetensors.torch import safe_open
    from transformers import SiglipVisionConfig, SiglipVisionModel

    from vagen.models.cambrian_vllm import _build_siglip_vision_config

    cfg = _build_siglip_vision_config()
    vision = SiglipVisionModel(cfg).to(device=device, dtype=torch.bfloat16).eval()
    projector = torch.nn.Sequential(
        torch.nn.Linear(1152, 3584),
        torch.nn.GELU(),
        torch.nn.Linear(3584, 3584),
    ).to(device=device, dtype=torch.bfloat16).eval()
    image_newline = torch.zeros(3584, device=device, dtype=torch.bfloat16)

    tensor_path = Path(model_path) / "model.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as f:
        v_sd = {}
        p_sd = {}
        for key in f.keys():
            if key.startswith("model.vision_tower_aux_list.0.vision_tower."):
                v_sd[key[len("model.vision_tower_aux_list.0.vision_tower.") :]] = f.get_tensor(key)
            elif key.startswith("model.mm_projector."):
                p_sd[key[len("model.mm_projector.") :]] = f.get_tensor(key)
            elif key == "model.image_newline":
                image_newline = f.get_tensor(key).to(device=device, dtype=torch.bfloat16)
    vision.load_state_dict(v_sd, strict=True)
    projector.load_state_dict(p_sd, strict=True)
    return vision, projector, image_newline


def miv_real(args: argparse.Namespace) -> dict[str, Any]:
    from vagen.models.cambrian_miv import build_cambrian_visual_features
    from vagen.models.cambrian_vllm import _preprocess_images

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    images = load_images(args.images, args.limit)
    pixels = _preprocess_images(images).to(device)
    vision, projector, image_newline = load_vision_projector(args.model, device)

    with torch.no_grad():
        vt_dtype = next(vision.parameters()).dtype
        raw = vision(pixel_values=pixels.to(vt_dtype)).last_hidden_state
        projected = projector(raw.to(next(projector.parameters()).dtype))
        before, _ = build_cambrian_visual_features(
            projected,
            image_newline,
            si_token_len=729,
            mm_use_newline=True,
            nfp_head=False,
            miv_token_len=64,
        )
        after, miv = build_cambrian_visual_features(
            projected,
            image_newline,
            si_token_len=729,
            mm_use_newline=True,
            nfp_head=True,
            miv_token_len=64,
        )
        same, _ = build_cambrian_visual_features(
            projected,
            image_newline,
            si_token_len=729,
            mm_use_newline=True,
            nfp_head=True,
            miv_token_len=64,
        )

    return {
        "device": device,
        "image_count": len(images),
        "pixel_stats": tensor_stats(pixels),
        "raw_vision_stats": tensor_stats(raw),
        "projected_stats": tensor_stats(projected),
        "miv_shape": list(miv.shape),
        "before_vs_after_first64": compare_tensors(before[:, :64], after[:, :64]),
        "before_vs_after_full": compare_tensors(before, after),
        "shared_helper_self_check_first64": compare_tensors(after[:, :64], same[:, :64]),
        "shared_helper_self_check_full": compare_tensors(after, same),
    }


def find_image_positions(ids: list[int], image_token_id: int) -> list[int]:
    return [i for i, tid in enumerate(ids) if tid == image_token_id]


def load_prompt_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = []
    with open(args.rollout_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(obj)
            if len(samples) >= args.limit:
                break
    return samples


def restore_logged_image_prompt(text: str) -> tuple[str, bool]:
    """Restore the current-view image token removed by rollout logging."""
    if IMAGE_TOKEN_TEXT in text:
        return text, True
    markers = (
        "[Observation]:\n\n",
        "[Initial Observation]:\n\n",
    )
    for marker in markers:
        if marker in text:
            return text.replace(marker, marker[:-1] + IMAGE_TOKEN_TEXT + "\n", 1), False
    return IMAGE_TOKEN_TEXT + "\n" + text, False


def first_action_prefix(output: str) -> tuple[str | None, str | None]:
    tag = "<action>"
    pos = output.find(tag)
    if pos < 0:
        return None, None
    start = pos + len(tag)
    tail = output[start:]
    for action in ACTION_STRINGS:
        if tail.startswith(action):
            return output[:start], action
    return output[:start], None


def expand_for_hf(input_ids: list[int], image_token_id: int) -> list[int]:
    out = []
    for tid in input_ids:
        if tid == image_token_id:
            out.extend([-200] * 756)
        else:
            out.append(tid)
    return out


def sequence_logprobs_from_logits(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = token_ids[:, 1:]
    logp = torch.log_softmax(shift_logits, dim=-1)
    return torch.gather(logp, -1, shift_labels.unsqueeze(-1)).squeeze(-1)


def logprob_entry_value(entry: Any, token_id: int) -> float | None:
    if entry is None:
        return None
    if token_id in entry:
        val = entry[token_id]
    elif str(token_id) in entry:
        val = entry[str(token_id)]
    else:
        return None
    if hasattr(val, "logprob"):
        return float(val.logprob)
    if isinstance(val, dict) and "logprob" in val:
        return float(val["logprob"])
    return float(val)


def hf_logprob_probe(args: argparse.Namespace) -> dict[str, Any]:
    import vagen.models.cambrian_register  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vagen.models.cambrian_processor import CambrianProcessorWrapper

    device = "cuda:0"
    dtype = torch.bfloat16
    samples = load_prompt_samples(args)
    images = load_images(args.images, len(samples))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    processor = CambrianProcessorWrapper(tok)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": device},
    ).eval()

    records = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            prompt_text, had_image_token = restore_logged_image_prompt(sample["input"])
            text = prompt_text + sample["output"]
            compact_ids = tok.encode(text, add_special_tokens=False)
            image_positions = find_image_positions(compact_ids, processor.image_token_id)
            hf_ids = expand_for_hf(compact_ids, processor.image_token_id)
            safe_ids = [0 if tid < 0 else tid for tid in hf_ids]
            input_ids = torch.tensor([hf_ids], device=device, dtype=torch.long)
            labels_for_gather = torch.tensor([safe_ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            pixels = processor.preprocess_images([images[i]])["pixel_values"].to(device=device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixels,
                use_cache=False,
            )
            token_logp = sequence_logprobs_from_logits(out.logits, labels_for_gather)[0].detach().cpu()

            prompt_len = len(tok.encode(prompt_text, add_special_tokens=False))
            response_token_count = len(tok.encode(sample["output"], add_special_tokens=False))
            expanded_prompt_len = len(expand_for_hf(tok.encode(prompt_text, add_special_tokens=False), processor.image_token_id))
            response_logp = token_logp[expanded_prompt_len - 1 : expanded_prompt_len - 1 + response_token_count]
            action_ids = {a: tok.encode(a, add_special_tokens=False) for a in ACTION_STRINGS}
            action_prefix, first_action = first_action_prefix(sample["output"])
            legal = {}
            first_action_logp = None
            first_action_token_count = 0
            if action_prefix is not None:
                action_prefix_ids = tok.encode(prompt_text + action_prefix, add_special_tokens=False)
                expanded_action_prefix_len = len(expand_for_hf(action_prefix_ids, processor.image_token_id))
                next_logits = out.logits[0, expanded_action_prefix_len - 1].float()
                next_logp = torch.log_softmax(next_logits, dim=-1).detach().cpu()
                legal = {a: float(next_logp[ids[0]].item()) for a, ids in action_ids.items() if ids}
                if first_action is not None:
                    action_prefix_response_token_count = len(tok.encode(action_prefix, add_special_tokens=False))
                    first_action_token_count = len(tok.encode(first_action, add_special_tokens=False))
                    start = expanded_prompt_len - 1 + action_prefix_response_token_count
                    stop = start + first_action_token_count
                    first_action_logp = float(token_logp[start:stop].sum().item())
            records.append(
                {
                    "idx": i,
                    "task_id": sample.get("task_id"),
                    "had_image_token_in_log": had_image_token,
                    "image_positions_compact": image_positions,
                    "prompt_compact_len": prompt_len,
                    "compact_len": len(compact_ids),
                    "expanded_len": len(hf_ids),
                    "response_logprob_sum": float(response_logp.sum().item()),
                    "response_logprob_mean": float(response_logp.mean().item()),
                    "legal_first_token_logprobs": legal,
                    "legal_top1": max(legal, key=legal.get) if legal else None,
                    "first_action": first_action,
                    "first_action_token_count": first_action_token_count,
                    "first_action_logprob_sum": first_action_logp,
                }
            )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"samples": records}


def vllm_logprob_probe(args: argparse.Namespace) -> dict[str, Any]:
    import vagen.models.cambrian_vllm  # noqa: F401
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vagen.models.cambrian_processor import CambrianProcessorWrapper

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
    os.environ.setdefault("VLLM_TORCH_COMPILE_LEVEL", "0")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_SKIP_DEEP_GEMM_WARMUP", "1")

    samples = load_prompt_samples(args)
    images = load_images(args.images, len(samples))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    processor = CambrianProcessorWrapper(tok)
    action_ids = {a: tok.encode(a, add_special_tokens=False) for a in ACTION_STRINGS}

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.limit_images},
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    full_prompts = []
    action_prompts = []
    metadata = []
    for i, sample in enumerate(samples):
        prompt_text, had_image_token = restore_logged_image_prompt(sample["input"])
        output_ids = tok.encode(sample["output"], add_special_tokens=False)
        compact_ids = tok.encode(prompt_text + sample["output"], add_special_tokens=False)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        expanded_prompt_len = len(expand_for_hf(prompt_ids, processor.image_token_id))
        full_prompts.append(
            {
                "prompt_token_ids": compact_ids,
                "multi_modal_data": {"image": images[i]},
            }
        )
        action_prefix, first_action = first_action_prefix(sample["output"])
        action_prefix_response_token_count = (
            len(tok.encode(action_prefix, add_special_tokens=False))
            if action_prefix is not None
            else 0
        )
        first_action_token_count = (
            len(tok.encode(first_action, add_special_tokens=False))
            if first_action is not None
            else 0
        )
        if action_prefix is not None:
            action_prompts.append(
                {
                    "prompt_token_ids": tok.encode(prompt_text + action_prefix, add_special_tokens=False),
                    "multi_modal_data": {"image": images[i]},
                }
            )
        else:
            action_prompts.append(None)
        metadata.append(
            {
                "idx": i,
                "task_id": sample.get("task_id"),
                "had_image_token_in_log": had_image_token,
                "image_positions_compact": find_image_positions(compact_ids, processor.image_token_id),
                "prompt_compact_len": len(prompt_ids),
                "response_token_count": len(output_ids),
                "expanded_prompt_len": expanded_prompt_len,
                "first_action": first_action,
                "action_prefix_response_token_count": action_prefix_response_token_count,
                "first_action_token_count": first_action_token_count,
            }
        )

    full_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=0,
        logprobs=None,
        detokenize=False,
    )
    full_outputs = llm.generate(full_prompts, full_params)

    action_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        # This vLLM build caps sampled logprobs at 20. This is enough to verify
        # top-1 agreement and usually includes the action-family alternatives.
        logprobs=20,
        detokenize=False,
    )
    action_outputs = []
    for p in action_prompts:
        if p is None:
            action_outputs.append(None)
        else:
            action_outputs.append(llm.generate([p], action_params)[0])

    records = []
    for meta, out, action_out in zip(metadata, full_outputs, action_outputs):
        prompt_logprobs = out.prompt_logprobs or []
        prompt_token_ids = out.prompt_token_ids or []
        start = meta["expanded_prompt_len"]
        stop = start + meta["response_token_count"]
        vals: list[float] = []
        missing = 0
        for j in range(start, min(stop, len(prompt_logprobs))):
            lp = logprob_entry_value(prompt_logprobs[j], prompt_token_ids[j])
            if lp is None:
                missing += 1
            else:
                vals.append(lp)

        first_action_vals: list[float] = []
        first_action_missing = 0
        if meta["first_action"] is not None:
            a_start = (
                meta["expanded_prompt_len"]
                + meta["action_prefix_response_token_count"]
            )
            a_stop = a_start + meta["first_action_token_count"]
            for j in range(a_start, min(a_stop, len(prompt_logprobs))):
                lp = logprob_entry_value(prompt_logprobs[j], prompt_token_ids[j])
                if lp is None:
                    first_action_missing += 1
                else:
                    first_action_vals.append(lp)

        legal = {}
        if action_out is not None and action_out.outputs and action_out.outputs[0].logprobs:
            first_step = action_out.outputs[0].logprobs[0]
            for action, ids in action_ids.items():
                if ids:
                    lp = logprob_entry_value(first_step, ids[0])
                    if lp is not None:
                        legal[action] = lp

        records.append(
            {
                **meta,
                "vllm_returned_prompt_len": len(prompt_token_ids),
                "response_logprob_sum": float(sum(vals)) if vals else float("nan"),
                "response_logprob_mean": float(mean(vals)) if vals else float("nan"),
                "response_logprob_token_count": len(vals),
                "response_logprob_missing_count": missing,
                "legal_first_token_logprobs": legal,
                "legal_top1": max(legal, key=legal.get) if legal else None,
                "first_action_logprob_sum": (
                    float(sum(first_action_vals)) if first_action_vals else None
                ),
                "first_action_logprob_token_count": len(first_action_vals),
                "first_action_logprob_missing_count": first_action_missing,
            }
        )

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "samples": records,
        "vllm_config": {
            "max_model_len": args.max_model_len,
            "tensor_parallel_size": args.tp,
            "limit_images": args.limit_images,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for rank, idx in enumerate(order):
        out[idx] = float(rank)
    return out


def compare_logprob_probe(args: argparse.Namespace) -> dict[str, Any]:
    hf = json.loads(Path(args.hf_json).read_text(encoding="utf-8"))
    vv = json.loads(Path(args.vllm_json).read_text(encoding="utf-8"))
    by_idx = {s["idx"]: s for s in vv["samples"]}
    rows = []
    ratios = []
    response_deltas = []
    legal_top1_matches = []
    rank_corrs = []
    first_action_deltas = []
    first_action_ratios = []

    for hs in hf["samples"]:
        vs = by_idx.get(hs["idx"])
        if vs is None:
            continue
        delta = hs["response_logprob_sum"] - vs["response_logprob_sum"]
        ratio = math.exp(max(-80.0, min(80.0, delta)))
        ratios.append(ratio)
        response_deltas.append(delta)
        hlegal = hs.get("legal_first_token_logprobs", {})
        vlegal = vs.get("legal_first_token_logprobs", {})
        common = [a for a in ACTION_STRINGS if a in hlegal and a in vlegal]
        rc = None
        if len(common) >= 2:
            rc = pearson(ranks([hlegal[a] for a in common]), ranks([vlegal[a] for a in common]))
            if rc is not None:
                rank_corrs.append(rc)
        top1_match = hs.get("legal_top1") == vs.get("legal_top1") if hlegal and vlegal else None
        if top1_match is not None:
            legal_top1_matches.append(float(top1_match))
        first_action_delta = None
        first_action_ratio = None
        if (
            hs.get("first_action_logprob_sum") is not None
            and vs.get("first_action_logprob_sum") is not None
        ):
            first_action_delta = hs["first_action_logprob_sum"] - vs["first_action_logprob_sum"]
            first_action_ratio = math.exp(max(-80.0, min(80.0, first_action_delta)))
            first_action_deltas.append(first_action_delta)
            first_action_ratios.append(first_action_ratio)
        rows.append(
            {
                "idx": hs["idx"],
                "task_id": hs.get("task_id"),
                "response_logprob_train": hs["response_logprob_sum"],
                "response_logprob_rollout": vs["response_logprob_sum"],
                "response_logprob_delta_train_minus_rollout": delta,
                "synced_policy_ratio": ratio,
                "first_action": hs.get("first_action"),
                "legal_top1_train": hs.get("legal_top1"),
                "legal_top1_rollout": vs.get("legal_top1"),
                "legal_top1_match": top1_match,
                "legal_rank_correlation": rc,
                "first_action_logprob_train": hs.get("first_action_logprob_sum"),
                "first_action_logprob_rollout": vs.get("first_action_logprob_sum"),
                "first_action_logprob_delta_train_minus_rollout": first_action_delta,
                "first_action_synced_policy_ratio": first_action_ratio,
            }
        )

    return {
        "sample_count": len(rows),
        "synced_policy_ratio": summarize(ratios),
        "response_logprob_delta_train_minus_rollout": summarize(response_deltas),
        "legal_top1_agreement": mean(legal_top1_matches) if legal_top1_matches else None,
        "legal_ranking_correlation": summarize(rank_corrs),
        "first_action_synced_policy_ratio": summarize(first_action_ratios),
        "first_action_logprob_delta_train_minus_rollout": summarize(first_action_deltas),
        "samples": rows,
    }


def module_group(name: str) -> str:
    if "vision_tower" in name:
        return "vision_tower"
    if "mm_projector" in name:
        return "mm_projector"
    if "nfp_head" in name:
        return "nfp_head"
    if name.endswith("image_newline") or "image_newline" in name:
        return "miv_related"
    if name.startswith("lm_head") or ".lm_head" in name:
        return "lm_head"
    marker = "model.layers."
    if marker in name:
        rest = name.split(marker, 1)[1]
        try:
            layer_idx = int(rest.split(".", 1)[0])
        except Exception:
            return "llm_other"
        if layer_idx < 9:
            return "llm_early"
        if layer_idx < 19:
            return "llm_middle"
        return "llm_late"
    if "embed_tokens" in name or "model.norm" in name:
        return "llm_other"
    return "other"


def empty_group_stats() -> dict[str, Any]:
    return {
        "requires_grad_params": 0,
        "optimizer_member_params": 0,
        "params_with_grad": 0,
        "grad_norm": 0.0,
        "manual_update_norm": 0.0,
    }


def collect_group_stats(model: torch.nn.Module, lr: float) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for name, p in model.named_parameters():
        group = module_group(name)
        rec = stats.setdefault(group, empty_group_stats())
        n = p.numel()
        if p.requires_grad:
            rec["requires_grad_params"] += n
            rec["optimizer_member_params"] += n
        if p.grad is not None:
            g = p.grad.detach().float()
            rec["params_with_grad"] += n
            rec["grad_norm"] += float(torch.sum(g * g).item())
            rec["manual_update_norm"] += float(torch.sum((g * lr) * (g * lr)).item())
    for rec in stats.values():
        rec["grad_norm"] = math.sqrt(rec["grad_norm"])
        rec["manual_update_norm"] = math.sqrt(rec["manual_update_norm"])
    return stats


def apply_manual_update(model: torch.nn.Module, lr: float) -> None:
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(p.grad.to(p.dtype), alpha=-lr)


def grad_audit_probe(args: argparse.Namespace) -> dict[str, Any]:
    import vagen.models.cambrian_register  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vagen.models.cambrian_processor import CambrianProcessorWrapper

    device = "cuda:0"
    dtype = torch.bfloat16
    lr = args.audit_lr
    samples = load_prompt_samples(args)
    sample = next((s for s in samples if first_action_prefix(s.get("output", ""))[1] is not None), samples[0])
    image_idx = samples.index(sample)
    images = load_images(args.images, len(samples))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    processor = CambrianProcessorWrapper(tok)
    prompt_text, had_image_token = restore_logged_image_prompt(sample["input"])
    compact_prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    compact_response_ids = tok.encode(sample["output"], add_special_tokens=False)
    compact_all_ids = compact_prompt_ids + compact_response_ids
    expanded_ids = expand_for_hf(compact_all_ids, processor.image_token_id)
    expanded_prompt_len = len(expand_for_hf(compact_prompt_ids, processor.image_token_id))
    safe_ids = [0 if tid < 0 else tid for tid in expanded_ids]
    input_ids = torch.tensor([expanded_ids], device=device, dtype=torch.long)
    labels_for_gather = torch.tensor([safe_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    response_mask = torch.zeros(input_ids.shape[1] - 1, device=device, dtype=torch.float32)
    response_start = expanded_prompt_len - 1
    response_mask[response_start : response_start + len(compact_response_ids)] = 1.0
    pixels = processor.preprocess_images([images[image_idx]])["pixel_values"].to(device=device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": device},
    )
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.zero_grad(set_to_none=True)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        pixel_values=pixels,
        use_cache=False,
    )
    token_logp = sequence_logprobs_from_logits(out.logits, labels_for_gather)[0].to(device)
    policy_loss = -((token_logp * response_mask).sum() / response_mask.sum().clamp_min(1.0))
    policy_loss.backward()

    nfp_aux_loss = None
    nfp_loss_mask = torch.zeros_like(input_ids, dtype=torch.float32, device=device)
    image_positions = (input_ids[0] == -200).nonzero(as_tuple=True)[0]
    if image_positions.numel() >= 64:
        nfp_loss_mask[0, image_positions[:64]] = 1.0
        nfp_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixels,
            nfp_pixel_values=pixels,
            nfp_loss_mask=nfp_loss_mask,
            use_cache=False,
        )
        nfp_aux_loss = nfp_out.loss
        if nfp_aux_loss is not None:
            nfp_aux_loss.backward()

    stats_before_update = collect_group_stats(model, lr)
    apply_manual_update(model, lr)

    flat_ratio = torch.ones_like(token_logp)
    valid_logp = token_logp.detach()[response_mask.bool()]
    result = {
        "mode_note": "single-GPU real CambrianForCausalLMAdapter backward audit; manual SGD-style update, not full Ray/FSDP AdamW shard audit",
        "sample": {
            "task_id": sample.get("task_id"),
            "had_image_token_in_log": had_image_token,
            "compact_prompt_len": len(compact_prompt_ids),
            "compact_response_len": len(compact_response_ids),
            "expanded_seq_len": len(expanded_ids),
            "expanded_prompt_len": expanded_prompt_len,
            "image_token_count": int(image_positions.numel()),
            "response_mask_tokens": float(response_mask.sum().item()),
        },
        "losses": {
            "policy_pg_like_loss": float(policy_loss.detach().item()),
            "nfp_aux_loss": float(nfp_aux_loss.detach().item()) if nfp_aux_loss is not None else None,
            "total_audited_loss": float(policy_loss.detach().item() + (nfp_aux_loss.detach().item() if nfp_aux_loss is not None else 0.0)),
            "kl": 0.0,
            "advantage_mean": 1.0,
            "advantage_std": 0.0,
            "policy_ratio_mean": float(flat_ratio[response_mask.bool()].mean().item()) if response_mask.sum() else 1.0,
            "policy_ratio_min": 1.0,
            "policy_ratio_max": 1.0,
            "clip_fraction": 0.0,
            "response_logprob_mean": float(valid_logp.mean().item()) if valid_logp.numel() else None,
        },
        "manual_update": {
            "lr": lr,
            "applied": True,
        },
        "groups": stats_before_update,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_result(args: argparse.Namespace, result: dict[str, Any]) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["processor", "miv-synthetic", "miv-real", "hf-logprob", "vllm-logprob", "compare-logprob", "grad-audit"])
    parser.add_argument("--model", default="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP")
    parser.add_argument("--images", nargs="+", default=[])
    parser.add_argument("--rollout-jsonl", default="exps/vagen_active_spatial/b5_c8_wrapper_img25_actionvalid/rollout_data/100.jsonl")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--limit-images", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--hf-json", default="")
    parser.add_argument("--vllm-json", default="")
    parser.add_argument("--audit-lr", type=float, default=1e-7)
    args = parser.parse_args()

    if args.mode == "processor":
        result = processor_parity(args)
    elif args.mode == "miv-synthetic":
        result = miv_synthetic(args)
    elif args.mode == "miv-real":
        result = miv_real(args)
    elif args.mode == "hf-logprob":
        result = hf_logprob_probe(args)
    elif args.mode == "vllm-logprob":
        result = vllm_logprob_probe(args)
    elif args.mode == "compare-logprob":
        result = compare_logprob_probe(args)
    elif args.mode == "grad-audit":
        result = grad_audit_probe(args)
    else:
        raise ValueError(args.mode)
    result = {"mode": args.mode, **result}
    write_result(args, result)


if __name__ == "__main__":
    main()
