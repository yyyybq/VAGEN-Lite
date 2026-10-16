#!/usr/bin/env python3
"""D0.10 HF incremental probe for attention root-cause isolation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from d0_9_hf_layer_probe import (
    install_cache_compat_shim,
    install_decoder_cache_tuple_shim,
    topk_case,
)

SELECTED_IDXS = [0, 73, 87]
PREDICTION_POSITIONS = [1413, 1486, 1500]
PREFIX_QUERIES = [1413, 1486]


def _flat_heads(t: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4:
        seq_len = position_ids.reshape(-1).numel() // max(int(position_ids.shape[0]), 1)
        if t.shape[1] == seq_len:
            return t.reshape(t.shape[0] * t.shape[1], -1)
        return t.permute(0, 2, 1, 3).reshape(t.shape[0] * t.shape[2], -1)
    if t.dim() == 3:
        return t.reshape(-1, t.shape[-1])
    return t.reshape(t.shape[0], -1)


def install_layer0_prefix_hook(model: Any, capture_state: dict[str, Any]) -> None:
    layer = model.model.layers[0]
    rotary_emb = model.model.rotary_emb

    def wrapped_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        past_key_value: Optional[Any] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        del use_cache
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        q = layer.self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
        k = layer.self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        v = layer.self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)

        if position_embeddings is None:
            position_embeddings = rotary_emb(normed, position_ids)
        cos, sin = position_embeddings
        q_rope, k_rope_current = apply_rotary_pos_emb(q, k, cos, sin)

        if position_ids is not None:
            pos = [int(x) for x in position_ids.reshape(-1).detach().cpu().tolist()]
            mask_1d = getattr(model, "_d0_10_current_attention_mask_1d", None)
            if mask_1d is None:
                keep = [True] * len(pos)
            else:
                keep = [bool(x) for x in mask_1d.reshape(-1).detach().cpu().tolist()]
            q_flat = _flat_heads(q_rope, position_ids).detach().float().cpu()
            k_flat = _flat_heads(k_rope_current, position_ids).detach().float().cpu()
            v_flat = _flat_heads(v, position_ids).detach().float().cpu()
            for local_idx, abs_pos in enumerate(pos):
                if local_idx >= len(keep) or not keep[local_idx]:
                    continue
                if 0 <= abs_pos <= max(PREFIX_QUERIES):
                    capture_state["k_rope_by_pos"][abs_pos] = [float(x) for x in k_flat[local_idx].tolist()]
                    capture_state["v_by_pos"][abs_pos] = [float(x) for x in v_flat[local_idx].tolist()]
                if abs_pos in PREFIX_QUERIES:
                    capture_state["q_rope_by_pos"][abs_pos] = [float(x) for x in q_flat[local_idx].tolist()]

        cache_obj = past_key_values if past_key_values is not None else past_key_value
        k_rope = k_rope_current
        if cache_obj is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_rope, v = cache_obj.update(k_rope, v, layer.self_attn.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if layer.self_attn.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[layer.self_attn.config._attn_implementation]
        attn_output, _ = attention_interface(
            layer.self_attn,
            q_rope,
            k_rope,
            v,
            attention_mask,
            dropout=0.0 if not layer.training else layer.self_attn.attention_dropout,
            scaling=layer.self_attn.scaling,
            sliding_window=layer.self_attn.sliding_window,
            **kwargs,
        )
        if position_ids is not None:
            pos = [int(x) for x in position_ids.reshape(-1).detach().cpu().tolist()]
            attn_flat = _flat_heads(attn_output, position_ids).detach().float().cpu()
            for local_idx, abs_pos in enumerate(pos):
                if abs_pos in PREFIX_QUERIES:
                    capture_state["attention_pre_o_by_pos"][abs_pos] = [
                        float(x) for x in attn_flat[local_idx].tolist()
                    ]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = layer.self_attn.o_proj(attn_output)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
        return hidden_states

    layer.forward = wrapped_forward


def build_prefix_payload(capture_state: dict[str, Any]) -> dict[str, Any]:
    records = []
    for query_pos in PREFIX_QUERIES:
        positions = [p for p in sorted(capture_state["k_rope_by_pos"]) if 0 <= p <= query_pos]
        records.append({
            "query_position": query_pos,
            "positions": positions,
            "q_rope_values": capture_state["q_rope_by_pos"].get(query_pos),
            "k_rope_values": [capture_state["k_rope_by_pos"][p] for p in positions],
            "v_values": [capture_state["v_by_pos"][p] for p in positions],
            "attention_pre_o_values": capture_state["attention_pre_o_by_pos"].get(query_pos),
        })
    return {"tag": "d0_10_hf_prefix_kv", "layer_idx": 0, "records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix-out", default="")
    ap.add_argument("--model", default="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--instrument", action="store_true")
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

    capture_state = {
        "q_rope_by_pos": {},
        "k_rope_by_pos": {},
        "v_by_pos": {},
        "attention_pre_o_by_pos": {},
    }
    if args.instrument:
        install_layer0_prefix_hook(model, capture_state)
    install_decoder_cache_tuple_shim(model)

    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.tensor(sample["position_ids"], dtype=torch.long, device=device).unsqueeze(0)
    response_ids = [int(x) for x in sample["response_token_ids"]]
    active_response_ids = [int(x) for x in sample["active_response_token_ids"]]
    prompt_len_with_pad = input_ids.shape[1] - len(response_ids)
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)

    raw_logits = {}
    with torch.inference_mode():
        prompt_ids = input_ids[:, :prompt_len_with_pad]
        prompt_attn = attention_mask[:, :prompt_len_with_pad]
        prompt_pos = position_ids[:, :prompt_len_with_pad]
        model._d0_10_current_attention_mask_1d = prompt_attn
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
            model._d0_10_current_attention_mask_1d = torch.ones((1, 1), dtype=torch.long, device=device)
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
        "tag": "d0_10_hf_probe",
        "instrument": bool(args.instrument),
        "model": args.model,
        "dtype": "torch.bfloat16",
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "selected_response_indices": SELECTED_IDXS,
        "selected_prediction_positions": PREDICTION_POSITIONS,
        "raw_logits_incremental": raw_logits,
    }
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if args.instrument and args.prefix_out:
        prefix_path = repo / args.prefix_out
        prefix_path.parent.mkdir(parents=True, exist_ok=True)
        prefix_path.write_text(json.dumps(build_prefix_payload(capture_state), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
