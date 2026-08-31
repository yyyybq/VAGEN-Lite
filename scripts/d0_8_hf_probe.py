#!/usr/bin/env python3
"""D0.8 HF-side fixed sample5 probe.

Loads CambrianForCausalLMAdapter, reconstructs the same sample5 fixed sequence,
and records valid-token layout, position semantics, layer-0 input embeddings,
and teacher-forced incremental raw logits for P0/P1/P2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

IMAGE_TOKEN_INDEX = -200
SELECTED_IDXS = [0, 73, 87]
SELECTED_POSITIONS = [0, 469, 470, 501, 533, 534, 600, 1000, 1225, 1226, 1413, 1414, 1487, 1501]


def install_cache_compat_shim() -> None:
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
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return
    for layer in layers:
        if getattr(layer, "_d0_8_cache_tuple_shim", False):
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
        layer._d0_8_cache_tuple_shim = True


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    td = t.detach()
    tf = td.float()
    flat = tf.reshape(-1).cpu()
    first = flat[: min(8, flat.numel())].tolist()
    arr = flat[: min(4096, flat.numel())].numpy().tobytes()
    return {
        "shape": list(td.shape),
        "dtype": str(td.dtype),
        "device": str(td.device),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else 0.0,
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "sha1_first_4096": hashlib.sha1(arr).hexdigest(),
        "checksum_first_4096_sum": float(flat[: min(4096, flat.numel())].sum().item()) if flat.numel() else 0.0,
        "first_values": [float(x) for x in first],
    }


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def topk_case(logits: torch.Tensor, token_id: int, tokenizer: Any, k: int = 20) -> dict[str, Any]:
    logits_f = logits.detach().float().cpu()
    logp = torch.log_softmax(logits_f, dim=-1)
    rank = int((logits_f > logits_f[token_id]).sum().item() + 1)
    vals, ids = torch.topk(logp, k=min(k, logp.numel()))
    return {
        "target_token_id": int(token_id),
        "target_decoded_token": decode_token(tokenizer, token_id),
        "target_raw_logit": float(logits_f[token_id].item()),
        "target_logprob": float(logp[token_id].item()),
        "target_rank": rank,
        "logsumexp": float(torch.logsumexp(logits_f, dim=-1).item()),
        "top20": [
            {
                "rank": int(i + 1),
                "token_id": int(tid),
                "decoded_token": decode_token(tokenizer, int(tid)),
                "raw_logit": float(logits_f[int(tid)].item()),
                "logprob": float(vals[i].item()),
            }
            for i, tid in enumerate(ids.tolist())
        ],
    }


def position_slices(position_ids: list[int]) -> dict[str, Any]:
    return {
        "first_16_prompt_positions": position_ids[:16],
        "before_visual_block_16": position_ids[454:470],
        "visual_block_beginning_16": position_ids[470:486],
        "visual_block_end_16": position_ids[1210:1226],
        "after_visual_block_16": position_ids[1226:1242],
        "last_32_prompt_positions": position_ids[1382:1414],
        "first_response_prediction_position": position_ids[1414],
        "required_prediction_position": position_ids[1487],
        "move_prediction_position": position_ids[1501],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    ap.add_argument("--out", default="docs/diagnosis/d0_8_hf_probe.json")
    ap.add_argument("--model", default="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.environ.setdefault("CAMBRIAN_SRC", "/mnt/umm/users/yinbaiqiao/cambrian-s")
    import vagen.models.cambrian_register  # noqa: F401
    from vagen.models.cambrian_processor import CambrianProcessorWrapper

    repo = Path.cwd()
    manifest = json.loads((repo / args.manifest).read_text())
    sample = next(s for s in manifest["samples"] if int(s["sample"]) == 5)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    processor = CambrianProcessorWrapper(tokenizer)
    image_path = repo / "exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/pid216636_sample_000_pre_processor.png"
    pixel_values = processor.preprocess_images([Image.open(image_path).convert("RGB")])["pixel_values"]

    install_cache_compat_shim()
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    install_decoder_cache_tuple_shim(model)

    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.tensor(sample["position_ids"], dtype=torch.long, device=device).unsqueeze(0)
    response_ids = [int(x) for x in sample["response_token_ids"]]
    active_response_ids = [int(x) for x in sample["active_response_token_ids"]]
    prompt_len_with_pad = input_ids.shape[1] - len(response_ids)
    valid_mask = attention_mask[0].bool()
    valid_input_ids = input_ids[0, valid_mask]
    valid_positions = position_ids[0, valid_mask]
    valid_position_list = [int(x) for x in valid_positions.detach().cpu().tolist()]
    image_positions = (valid_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        embeds = model._embed_multimodal_batch(input_ids, pixel_values)[0, valid_mask]

    selected_vectors = []
    for pos in SELECTED_POSITIONS:
        matches = (valid_positions == pos).nonzero(as_tuple=True)[0]
        if matches.numel() == 0:
            continue
        local_idx = int(matches[0].item())
        vec = embeds[local_idx]
        selected_vectors.append({
            "absolute_position": pos,
            "valid_local_index": local_idx,
            "token_id": int(valid_input_ids[local_idx].item()),
            "summary": tensor_stats(vec),
            "values": [float(x) for x in vec.detach().float().cpu().tolist()],
        })

    visual_component_vectors = []
    with torch.inference_mode():
        inner = model.get_model()
        encoded = model.encode_images([pixel_values])[0]
        proj_dtype = inner.mm_projector[0].weight.dtype
        projected = inner.mm_projector(encoded.to(proj_dtype)).to(pixel_values.dtype)
        from vagen.models.cambrian_miv import append_image_newline, build_cambrian_visual_features

        si_token_len = int(getattr(model.config, "si_token_len", 729))
        si_side_len = int(si_token_len**0.5)
        miv_token_len = int(getattr(model.config, "miv_token_len", 0) or 0)
        newline_expanded = append_image_newline(projected, inner.image_newline, si_side_len=si_side_len)
        final_visual, _ = build_cambrian_visual_features(
            projected,
            inner.image_newline,
            si_token_len=si_token_len,
            mm_use_newline=bool(getattr(model.config, "mm_use_im_newline_token", True)),
            nfp_head=bool(getattr(model.config, "nfp_head", False)),
            miv_token_len=miv_token_len,
        )
        for offset in [0, 31, 63, 64, 130, 530, 755]:
            visual_component_vectors.append({
                "visual_offset": offset,
                "absolute_position": 470 + offset,
                "newline_expanded_summary": tensor_stats(newline_expanded[0, offset]),
                "newline_expanded_values": [float(x) for x in newline_expanded[0, offset].detach().float().cpu().tolist()],
                "final_post_miv_summary": tensor_stats(final_visual[0, offset]),
                "final_post_miv_values": [float(x) for x in final_visual[0, offset].detach().float().cpu().tolist()],
            })

    raw_logits = {}
    with torch.inference_mode():
        prompt_ids = input_ids[:, :prompt_len_with_pad]
        prompt_attn = attention_mask[:, :prompt_len_with_pad]
        prompt_pos = position_ids[:, :prompt_len_with_pad]
        out = model(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            position_ids=prompt_pos,
            pixel_values=pixel_values,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        raw_logits["0"] = topk_case(out.logits[0, -1], active_response_ids[0], tokenizer)
        del out

        for j in range(1, max(SELECTED_IDXS) + 1):
            prev_tid = active_response_ids[j - 1]
            target_tid = active_response_ids[j]
            abs_prev = prompt_len_with_pad + j - 1
            cur_len = prompt_len_with_pad + j
            out = model(
                input_ids=torch.tensor([[prev_tid]], dtype=torch.long, device=device),
                attention_mask=attention_mask[:, :cur_len],
                position_ids=position_ids[:, abs_prev:abs_prev + 1],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            if j in SELECTED_IDXS:
                raw_logits[str(j)] = topk_case(out.logits[0, -1], target_tid, tokenizer)
            del out

    payload = {
        "tag": "d0_8_hf_probe",
        "model": args.model,
        "dtype": "torch.bfloat16",
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "model_training": bool(model.training),
        "sequence_layout": {
            "full_input_len_with_left_pad": int(input_ids.shape[1]),
            "prompt_len_with_left_pad": int(prompt_len_with_pad),
            "valid_total_len": int(valid_input_ids.numel()),
            "expanded_prompt_len_no_pad": 1414,
            "active_response_len": len(active_response_ids),
            "visual_span": [int(image_positions[0].item()), int(image_positions[-1].item()) + 1],
            "visual_token_count": int(image_positions.numel()),
            "first_response_position": 1414,
            "selected_response_indices": SELECTED_IDXS,
            "selected_absolute_positions": [1414 + i for i in SELECTED_IDXS],
        },
        "position_semantics": position_slices(valid_position_list),
        "input_embedding_selected_vectors": selected_vectors,
        "visual_component_vectors": visual_component_vectors,
        "raw_logits_incremental": raw_logits,
    }
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
