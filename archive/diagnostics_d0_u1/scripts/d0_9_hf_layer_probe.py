#!/usr/bin/env python3
"""D0.9 HF incremental layer capture for fixed sample 5."""

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

IMAGE_TOKEN_INDEX = -200
SELECTED_IDXS = [0, 73, 87]
PREDICTION_POSITIONS = [1413, 1486, 1500]


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
        if getattr(layer, "_d0_9_cache_tuple_shim", False):
            continue
        old_forward = layer.forward

        def wrapped_forward(*args, _old_forward=old_forward, **kwargs):
            use_cache = bool(kwargs.get("use_cache", False))
            cache_obj = kwargs.get("past_key_values", kwargs.get("past_key_value", None))
            out = _old_forward(*args, **kwargs)
            if use_cache and torch.is_tensor(out):
                return (out, cache_obj)
            return out

        layer.forward = wrapped_forward
        layer._d0_9_cache_tuple_shim = True


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


def record_vectors(
    *,
    out_f,
    kind: str,
    layer_idx: Optional[int],
    name: str,
    position_ids: Optional[torch.Tensor],
    tensor: torch.Tensor,
    target_positions: set[int],
) -> None:
    if position_ids is None:
        return
    pos = [int(x) for x in position_ids.reshape(-1).detach().cpu().tolist()]
    vec = tensor.detach()
    if vec.dim() == 4:
        # Q/K/V are (B, heads, S, D); eager attention output is (B, S, heads, D).
        seq_len = position_ids.reshape(-1).numel() // max(int(position_ids.shape[0]), 1)
        if tensor.shape[1] == seq_len:
            vec = vec.reshape(vec.shape[0] * vec.shape[1], -1)
        else:
            vec = vec.permute(0, 2, 1, 3).reshape(vec.shape[0] * vec.shape[2], -1)
    elif vec.dim() == 3:
        vec = vec.reshape(-1, vec.shape[-1])
    elif vec.dim() > 4:
        vec = vec.reshape(vec.shape[0], -1)

    records = []
    for local_idx, abs_pos in enumerate(pos):
        if abs_pos not in target_positions or local_idx >= vec.shape[0]:
            continue
        v = vec[local_idx].float().cpu()
        records.append({
            "absolute_position": abs_pos,
            "local_index": int(local_idx),
            "summary": {
                "shape": list(tensor.shape),
                "vector_shape": list(v.shape),
                "dtype": str(tensor.dtype),
                "mean": float(v.mean().item()) if v.numel() else 0.0,
                "std": float(v.std(unbiased=False).item()) if v.numel() else 0.0,
                "min": float(v.min().item()) if v.numel() else 0.0,
                "max": float(v.max().item()) if v.numel() else 0.0,
                "l2": float(torch.linalg.vector_norm(v).item()) if v.numel() else 0.0,
                "checksum": float(v[: min(4096, v.numel())].sum().item()) if v.numel() else 0.0,
            },
            "values": [float(x) for x in v.tolist()],
        })
    if not records:
        return
    out_f.write(json.dumps({
        "kind": kind,
        "layer_idx": layer_idx,
        "name": name,
        "records": records,
    }, ensure_ascii=False) + "\n")
    out_f.flush()


def install_layer_hooks(
    *,
    model: Any,
    capture_path: Path,
    breakdown_layer: int,
    target_positions: set[int],
) -> None:
    layers = list(model.model.layers)
    rotary_emb = model.model.rotary_emb
    capture_layers = set(range(len(layers)))
    out_f = capture_path.open("w")
    model._d0_9_capture_file = out_f

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in capture_layers and layer_idx != breakdown_layer:
            continue

        def wrapped_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Any] = None,
            past_key_value: Optional[Any] = None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            *,
            _layer=layer,
            _layer_idx=layer_idx,
            **kwargs,
        ) -> torch.Tensor:
            del use_cache
            do_breakdown = breakdown_layer == _layer_idx
            if _layer_idx == 0 or do_breakdown:
                record_vectors(
                    out_f=out_f,
                    kind="stream",
                    layer_idx=_layer_idx,
                    name="layer_input",
                    position_ids=position_ids,
                    tensor=hidden_states,
                    target_positions=target_positions,
                )

            residual = hidden_states
            normed = _layer.input_layernorm(hidden_states)
            if do_breakdown:
                record_vectors(
                    out_f=out_f,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="input_rmsnorm",
                    position_ids=position_ids,
                    tensor=normed,
                    target_positions=target_positions,
                )

            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, _layer.self_attn.head_dim)
            q = _layer.self_attn.q_proj(normed).view(hidden_shape).transpose(1, 2)
            k = _layer.self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
            v = _layer.self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
            if do_breakdown:
                for name, tensor in (("q", q), ("k", k), ("v", v)):
                    record_vectors(
                        out_f=out_f,
                        kind="breakdown",
                        layer_idx=_layer_idx,
                        name=name,
                        position_ids=position_ids,
                        tensor=tensor,
                        target_positions=target_positions,
                    )

            if position_embeddings is None:
                position_embeddings = rotary_emb(normed, position_ids)
            cos, sin = position_embeddings
            q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
            if do_breakdown:
                for name, tensor in (("q_rope", q_rope), ("k_rope", k_rope)):
                    record_vectors(
                        out_f=out_f,
                        kind="breakdown",
                        layer_idx=_layer_idx,
                        name=name,
                        position_ids=position_ids,
                        tensor=tensor,
                        target_positions=target_positions,
                    )

            cache_obj = past_key_values if past_key_values is not None else past_key_value
            if cache_obj is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k_rope, v = cache_obj.update(k_rope, v, _layer.self_attn.layer_idx, cache_kwargs)

            attention_interface = eager_attention_forward
            if _layer.self_attn.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[_layer.self_attn.config._attn_implementation]
            attn_output, _ = attention_interface(
                _layer.self_attn,
                q_rope,
                k_rope,
                v,
                attention_mask,
                dropout=0.0 if not _layer.training else _layer.self_attn.attention_dropout,
                scaling=_layer.self_attn.scaling,
                sliding_window=_layer.self_attn.sliding_window,
                **kwargs,
            )
            if do_breakdown:
                record_vectors(
                    out_f=out_f,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="attention_pre_o",
                    position_ids=position_ids,
                    tensor=attn_output,
                    target_positions=target_positions,
                )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = _layer.self_attn.o_proj(attn_output)
            if do_breakdown:
                record_vectors(
                    out_f=out_f,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="attention_output",
                    position_ids=position_ids,
                    tensor=attn_output,
                    target_positions=target_positions,
                )
            hidden_states = residual + attn_output
            if do_breakdown:
                record_vectors(
                    out_f=out_f,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="post_attention_residual",
                    position_ids=position_ids,
                    tensor=hidden_states,
                    target_positions=target_positions,
                )
            residual = hidden_states
            mlp_input = _layer.post_attention_layernorm(hidden_states)
            if do_breakdown:
                record_vectors(
                    out_f=out_f,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="post_attention_rmsnorm",
                    position_ids=position_ids,
                    tensor=mlp_input,
                    target_positions=target_positions,
                )
            mlp_output = _layer.mlp(mlp_input)
            if do_breakdown:
                record_vectors(
                    out_f=out_f,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="mlp_output",
                    position_ids=position_ids,
                    tensor=mlp_output,
                    target_positions=target_positions,
                )
            hidden_states = residual + mlp_output
            if do_breakdown:
                record_vectors(
                    out_f=out_f,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="post_mlp_residual",
                    position_ids=position_ids,
                    tensor=hidden_states,
                    target_positions=target_positions,
                )
            if _layer_idx in capture_layers:
                record_vectors(
                    out_f=out_f,
                    kind="stream",
                    layer_idx=_layer_idx,
                    name="layer_output",
                    position_ids=position_ids,
                    tensor=hidden_states,
                    target_positions=target_positions,
                )
            return hidden_states

        layer.forward = wrapped_forward

    old_norm_forward = model.model.norm.forward

    def norm_forward(hidden_states: torch.Tensor, _old_norm_forward=old_norm_forward):
        out = _old_norm_forward(hidden_states)
        record_vectors(
            out_f=out_f,
            kind="stream",
            layer_idx=None,
            name="final_norm",
            position_ids=getattr(model, "_d0_9_current_position_ids", None),
            tensor=out,
            target_positions=target_positions,
        )
        return out

    model.model.norm.forward = norm_forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    ap.add_argument("--out", default="docs/diagnosis/d0_9_hf_layer_probe.json")
    ap.add_argument("--capture", default="docs/diagnosis/d0_9_runs/hf_layers.jsonl")
    ap.add_argument("--model", default="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--breakdown-layer", type=int, default=-1)
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
    capture_path = repo / args.capture
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    install_layer_hooks(
        model=model,
        capture_path=capture_path,
        breakdown_layer=args.breakdown_layer,
        target_positions=set(PREDICTION_POSITIONS),
    )
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
        model._d0_9_current_position_ids = prompt_pos
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
            model._d0_9_current_position_ids = position_ids[:, abs_prev:abs_prev + 1]
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

    model._d0_9_capture_file.close()

    payload = {
        "tag": "d0_9_hf_layer_probe",
        "model": args.model,
        "dtype": "torch.bfloat16",
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "model_training": bool(model.training),
        "breakdown_layer": args.breakdown_layer,
        "capture_path": str(capture_path),
        "selected_response_indices": SELECTED_IDXS,
        "selected_prediction_positions": PREDICTION_POSITIONS,
        "selected_target_positions": [1414 + i for i in SELECTED_IDXS],
        "raw_logits_incremental": raw_logits,
    }
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
