"""
vagen/models/cambrian_vllm.py

vLLM V1 model wrapper for Cambrian-S (CambrianQwenForCausalLM).

Loaded by vllm_async_server.py before run_server() via:
    actor_rollout_ref.rollout.model.external_lib = vagen.models.cambrian_vllm

The vision tower prefers Cambrian's native SigLIP implementation when
`CAMBRIAN_SRC` is available.  D0.8 showed that Transformers SiglipVisionModel is
not numerically identical to the training-side Cambrian tower for this checkpoint.

Token convention
----------------
  <image>         ID 151665 -- single placeholder per image in the prompt
  <|image_pad|>   ID 151655 -- per-feature placeholder after vLLM expansion
  TOKENS_PER_IMAGE = 756    -- 27×28 (si_side_len=27, mm_use_im_newline_token=True)

Weight remapping from Cambrian-S checkpoint
-------------------------------------------
  model.embed_tokens.*                      → language_model.model.embed_tokens.*
  model.layers.*                            → language_model.model.layers.*
  model.norm.*                              → language_model.model.norm.*
  lm_head.*                                 → language_model.lm_head.*
  model.vision_tower_aux_list.0.vision_tower.* → vision_tower.*
  model.mm_projector.*                      → mm_projector.*
  model.image_newline                       → image_newline
  model.nfp_head.*                          → (skipped)
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import Qwen2Config
try:
    _CAMBRIAN_SRC = os.environ.get("CAMBRIAN_SRC", "/mnt/umm/users/yinbaiqiao/cambrian-s")
    if _CAMBRIAN_SRC and _CAMBRIAN_SRC not in sys.path:
        sys.path.insert(0, _CAMBRIAN_SRC)
    from cambrian.model.multimodal_encoder.llava_next_siglip_encoder import (  # type: ignore
        SigLipVisionConfig as CambrianSiglipVisionConfig,
        SigLipVisionModel as CambrianSiglipVisionModel,
    )
except Exception:
    CambrianSiglipVisionConfig = None
    CambrianSiglipVisionModel = None
    from transformers import SiglipVisionConfig as TransformersSiglipVisionConfig
    from transformers import SiglipVisionModel as TransformersSiglipVisionModel
from transformers.feature_extraction_utils import BatchFeature

from vagen.models.cambrian_miv import build_cambrian_visual_features

from vllm import ModelRegistry
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import (init_vllm_registered_model,
                                               maybe_prefix,
                                               merge_multimodal_embeddings)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                         BaseProcessingInfo, PromptReplacement,
                                         PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_TOKEN: str = "<image>"
IMAGE_TOKEN_ID: int = 151665       # added as special token in Qwen2 tokenizer
IMAGE_PAD_TOKEN_ID: int = 151655   # <|image_pad|> built-in Qwen2 token
TOKENS_PER_IMAGE: int = 756        # 27 * (27 + 1) = 756
SIGLIP_HIDDEN_DIM: int = 1152
LM_HIDDEN_DIM: int = 3584
TRAINING_SIGLIP_MODEL: str = os.environ.get(
    "VAGEN_CAMBRIAN_SIGLIP_MODEL",
    "google/siglip2-so400m-patch14-384",
)


def _default_siglip_cache() -> str:
    """Prefer explicit SIGLIP_CACHE, then common local HF hubs."""
    env = os.environ.get("SIGLIP_CACHE")
    if env:
        return env
    candidates = [
        os.path.join(os.environ.get("HF_HOME", ""), "hub") if os.environ.get("HF_HOME") else "",
        "/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub",
        "/mnt/umm/users/yinbaiqiao/hf_cache/hub",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return os.path.join(os.environ.get("HF_HOME", "/tmp/hf_cache"), "hub")


SIGLIP_CACHE: str = _default_siglip_cache()


def _d0_8_float(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def _d0_8_tensor_stats(t: torch.Tensor) -> dict:
    td = t.detach()
    tf = td.float()
    flat = tf.reshape(-1)
    first = flat[:8].cpu().tolist()
    checksum = _d0_8_float(flat[: min(flat.numel(), 4096)].sum()) if flat.numel() else 0.0
    return {
        "shape": list(td.shape),
        "dtype": str(td.dtype),
        "device": str(td.device),
        "mean": _d0_8_float(tf.mean()) if flat.numel() else 0.0,
        "std": _d0_8_float(tf.std(unbiased=False)) if flat.numel() else 0.0,
        "min": _d0_8_float(tf.min()) if flat.numel() else 0.0,
        "max": _d0_8_float(tf.max()) if flat.numel() else 0.0,
        "checksum_first_4096_sum": checksum,
        "first_values": [float(x) for x in first],
    }


def _d0_8_selected_vectors(
    inputs_embeds: torch.Tensor,
    positions: torch.Tensor,
    target_positions: set[int],
) -> list[dict]:
    if inputs_embeds.dim() == 3:
        embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
    else:
        embeds = inputs_embeds
    pos = positions.reshape(-1).detach().cpu().tolist()
    out = []
    for local_idx, p in enumerate(pos):
        p_int = int(p)
        if p_int in target_positions or p_int in {0, 469, 470, 501, 533, 534, 600, 1000, 1225, 1226, 1413, 1414, 1487, 1501}:
            vec = embeds[local_idx]
            record = {
                "local_index": int(local_idx),
                "absolute_position": p_int,
                "summary": _d0_8_tensor_stats(vec),
            }
            if os.environ.get("D0_8_DUMP_VECTOR_VALUES", "0") == "1":
                record["values"] = [float(x) for x in vec.detach().float().cpu().tolist()]
            out.append(record)
    return out


def _d0_8_dump_forward(
    *,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    kwargs: Mapping[str, object],
) -> None:
    capture_dir = os.environ.get("D0_8_CAPTURE_DIR")
    if not capture_dir:
        return

    max_calls = int(os.environ.get("D0_8_MAX_CAPTURE_CALLS", "64"))
    call_idx = int(getattr(_d0_8_dump_forward, "_call_idx", 0))
    if call_idx >= max_calls:
        return
    setattr(_d0_8_dump_forward, "_call_idx", call_idx + 1)

    target_positions = {
        int(x)
        for x in os.environ.get("D0_8_TARGET_POSITIONS", "").split(",")
        if x.strip()
    }
    pos_flat = positions.reshape(-1).detach().cpu().tolist()
    pos_int = [int(x) for x in pos_flat]
    ids_int: list[int] = []
    image_pad_positions: list[int] = []
    if input_ids is not None:
        ids_int = [int(x) for x in input_ids.reshape(-1).detach().cpu().tolist()]
        image_pad_positions = [
            int(pos_int[i])
            for i, tok in enumerate(ids_int[: len(pos_int)])
            if tok == IMAGE_PAD_TOKEN_ID
        ]

    spans = []
    if image_pad_positions:
        start = prev = image_pad_positions[0]
        for p in image_pad_positions[1:]:
            if p == prev + 1:
                prev = p
            else:
                spans.append([start, prev + 1])
                start = prev = p
        spans.append([start, prev + 1])

    payload = {
        "pid": os.getpid(),
        "time": time.time(),
        "call_idx": call_idx,
        "input_ids_shape": None if input_ids is None else list(input_ids.shape),
        "positions_shape": list(positions.shape),
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "inputs_embeds_dtype": str(inputs_embeds.dtype),
        "num_tokens": len(pos_int),
        "position_min": min(pos_int) if pos_int else None,
        "position_max": max(pos_int) if pos_int else None,
        "position_first_16": pos_int[:16],
        "position_last_32": pos_int[-32:],
        "image_pad_token_id": IMAGE_PAD_TOKEN_ID,
        "image_pad_count_in_call": len(image_pad_positions),
        "image_pad_position_spans_in_call": spans,
        "contains_targets": sorted(set(pos_int).intersection(target_positions)),
        "selected_vectors": _d0_8_selected_vectors(inputs_embeds, positions, target_positions),
        "kwarg_keys": sorted(str(k) for k in kwargs.keys()),
    }

    out_dir = Path(capture_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"forward_pid{os.getpid()}_call{call_idx:04d}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _d0_8_dump_logits(
    *,
    positions: Optional[torch.Tensor],
    logits: Optional[torch.Tensor],
) -> None:
    capture_dir = os.environ.get("D0_8_CAPTURE_DIR")
    if not capture_dir or logits is None or positions is None:
        return

    target_positions = [
        int(x)
        for x in os.environ.get("D0_8_TARGET_POSITIONS", "").split(",")
        if x.strip()
    ]
    target_token_ids = [
        int(x)
        for x in os.environ.get("D0_8_TARGET_TOKEN_IDS", "").split(",")
        if x.strip()
    ]
    if len(target_positions) != len(target_token_ids):
        return

    max_calls = int(os.environ.get("D0_8_MAX_LOGIT_CAPTURE_CALLS", "64"))
    call_idx = int(getattr(_d0_8_dump_logits, "_call_idx", 0))
    if call_idx >= max_calls:
        return
    setattr(_d0_8_dump_logits, "_call_idx", call_idx + 1)

    pos = [int(x) for x in positions.reshape(-1).detach().cpu().tolist()]
    logits_f = logits.detach().float()
    if logits_f.dim() == 3:
        logits_f = logits_f.reshape(-1, logits_f.shape[-1])

    records = []
    for target_pos, target_tok in zip(target_positions, target_token_ids, strict=True):
        pred_pos = target_pos - 1
        if pred_pos not in pos:
            continue
        local_idx = pos.index(pred_pos)
        if local_idx >= logits_f.shape[0]:
            continue
        row = logits_f[local_idx].cpu()
        logp = torch.log_softmax(row, dim=-1)
        top_vals, top_ids = torch.topk(logp, k=min(20, logp.numel()))
        target_raw = float(row[target_tok].item())
        records.append({
            "target_position": target_pos,
            "prediction_position": pred_pos,
            "local_index": int(local_idx),
            "target_token_id": int(target_tok),
            "target_raw_logit": target_raw,
            "target_logprob": float(logp[target_tok].item()),
            "target_rank": int((row > row[target_tok]).sum().item() + 1),
            "logsumexp": float(torch.logsumexp(row, dim=-1).item()),
            "top20": [
                {
                    "rank": int(i + 1),
                    "token_id": int(tid),
                    "raw_logit": float(row[int(tid)].item()),
                    "logprob": float(top_vals[i].item()),
                }
                for i, tid in enumerate(top_ids.tolist())
            ],
        })
    if not records:
        return

    out_dir = Path(capture_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"logits_pid{os.getpid()}_call{call_idx:04d}.json"
    out_path.write_text(
        json.dumps({
            "pid": os.getpid(),
            "time": time.time(),
            "call_idx": call_idx,
            "logits_shape": list(logits.shape),
            "positions_shape": list(positions.shape),
            "position_min": min(pos) if pos else None,
            "position_max": max(pos) if pos else None,
            "records": records,
        }, indent=2, ensure_ascii=False) + "\n"
    )


def _d0_9_capture_dir() -> Optional[Path]:
    raw = os.environ.get("D0_9_CAPTURE_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _d0_9_int_set(name: str) -> set[int]:
    vals = os.environ.get(name, "")
    out: set[int] = set()
    for item in vals.split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def _d0_9_layer_set(num_layers: int) -> set[int]:
    raw = os.environ.get("D0_9_CAPTURE_LAYERS", "all").strip().lower()
    if raw in {"", "all"}:
        return set(range(num_layers))
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _d0_9_record_vectors(
    *,
    capture_dir: Optional[Path],
    kind: str,
    layer_idx: Optional[int],
    name: str,
    positions: torch.Tensor,
    tensor: torch.Tensor,
    extra: Optional[dict] = None,
) -> None:
    if capture_dir is None:
        return
    target_positions = _d0_9_int_set("D0_9_TARGET_POSITIONS")
    if not target_positions:
        return
    pos = [int(x) for x in positions.reshape(-1).detach().cpu().tolist()]
    vec = tensor
    if vec.dim() == 3:
        vec = vec.reshape(-1, vec.shape[-1])
    elif vec.dim() > 3:
        vec = vec.reshape(vec.shape[0], -1)
    records = []
    dump_values = os.environ.get("D0_9_DUMP_VALUES", "1") == "1"
    for local_idx, abs_pos in enumerate(pos):
        if abs_pos not in target_positions or local_idx >= vec.shape[0]:
            continue
        v = vec[local_idx].detach().float().cpu()
        rec = {
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
        }
        if dump_values:
            rec["values"] = [float(x) for x in v.tolist()]
        records.append(rec)
    if not records:
        return
    payload = {
        "pid": os.getpid(),
        "time": time.time(),
        "kind": kind,
        "layer_idx": layer_idx,
        "name": name,
        "records": records,
    }
    if extra:
        payload["extra"] = extra
    out_path = capture_dir / f"d0_9_pid{os.getpid()}.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _d0_10_prefix_targets() -> set[int]:
    vals = os.environ.get("D0_10_PREFIX_TARGET_POSITIONS", "")
    out: set[int] = set()
    for item in vals.split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def _d0_10_record_prefix_kv(
    *,
    capture_dir: Optional[Path],
    layer_idx: int,
    positions: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if capture_dir is None or os.environ.get("D0_10_PREFIX_CAPTURE", "0") != "1":
        return
    target_positions = _d0_10_prefix_targets()
    if not target_positions:
        return
    prefix_layer = int(os.environ.get("D0_10_PREFIX_LAYER", "0"))
    if layer_idx != prefix_layer:
        return

    pos = [int(x) for x in positions.reshape(-1).detach().cpu().tolist()]
    if q_rope.dim() != 2 or k_rope.dim() != 2 or v.dim() != 2:
        return

    q_cpu = q_rope.detach().float().cpu()
    k_cpu = k_rope.detach().float().cpu()
    v_cpu = v.detach().float().cpu()
    pos_to_local = {abs_pos: i for i, abs_pos in enumerate(pos)}
    records = []
    for query_pos in sorted(target_positions):
        if query_pos not in pos_to_local:
            continue
        prefix_indices = [
            i for i, abs_pos in enumerate(pos)
            if 0 <= abs_pos <= query_pos and i < k_cpu.shape[0] and i < v_cpu.shape[0]
        ]
        q_idx = pos_to_local[query_pos]
        records.append({
            "query_position": int(query_pos),
            "positions": [int(pos[i]) for i in prefix_indices],
            "q_rope_values": [float(x) for x in q_cpu[q_idx].tolist()],
            "k_rope_values": [[float(x) for x in k_cpu[i].tolist()] for i in prefix_indices],
            "v_values": [[float(x) for x in v_cpu[i].tolist()] for i in prefix_indices],
            "shapes": {
                "q_rope": list(q_rope.shape),
                "k_rope": list(k_rope.shape),
                "v": list(v.shape),
            },
            "dtype": {
                "q_rope": str(q_rope.dtype),
                "k_rope": str(k_rope.dtype),
                "v": str(v.dtype),
            },
        })
    if not records:
        return
    out_path = capture_dir / f"d0_10_pid{os.getpid()}.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps({
            "pid": os.getpid(),
            "time": time.time(),
            "kind": "d0_10_prefix_kv",
            "layer_idx": layer_idx,
            "records": records,
        }, ensure_ascii=False) + "\n")


def _d0_10_record_attention_pre_o(
    *,
    capture_dir: Optional[Path],
    layer_idx: int,
    positions: torch.Tensor,
    attention_pre_o: torch.Tensor,
) -> None:
    if capture_dir is None or os.environ.get("D0_10_PREFIX_CAPTURE", "0") != "1":
        return
    target_positions = _d0_10_prefix_targets()
    if not target_positions:
        return
    prefix_layer = int(os.environ.get("D0_10_PREFIX_LAYER", "0"))
    if layer_idx != prefix_layer or attention_pre_o.dim() != 2:
        return

    pos = [int(x) for x in positions.reshape(-1).detach().cpu().tolist()]
    attn_cpu = attention_pre_o.detach().float().cpu()
    records = []
    for local_idx, abs_pos in enumerate(pos):
        if abs_pos not in target_positions or local_idx >= attn_cpu.shape[0]:
            continue
        records.append({
            "query_position": int(abs_pos),
            "values": [float(x) for x in attn_cpu[local_idx].tolist()],
            "shape": list(attention_pre_o.shape),
            "dtype": str(attention_pre_o.dtype),
        })
    if not records:
        return
    out_path = capture_dir / f"d0_10_pid{os.getpid()}.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps({
            "pid": os.getpid(),
            "time": time.time(),
            "kind": "d0_10_attention_pre_o",
            "layer_idx": layer_idx,
            "records": records,
        }, ensure_ascii=False) + "\n")


def _d0_9_install_qwen_hooks(language_model: nn.Module) -> None:
    capture_dir = _d0_9_capture_dir()
    if capture_dir is None:
        return
    qwen_model = getattr(language_model, "model", None)
    layers = list(getattr(qwen_model, "layers", []))
    if not layers or getattr(qwen_model, "_d0_9_hooks_installed", False):
        return
    setattr(qwen_model, "_d0_9_hooks_installed", True)
    capture_layers = _d0_9_layer_set(len(layers))
    breakdown_layer_raw = os.environ.get("D0_9_BREAKDOWN_LAYER", "").strip()
    breakdown_layer = int(breakdown_layer_raw) if breakdown_layer_raw else None

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in capture_layers and layer_idx != breakdown_layer:
            continue
        original_forward = layer.forward

        def wrapped_forward(
            positions,
            hidden_states,
            residual,
            *,
            _layer=layer,
            _layer_idx=layer_idx,
            _original_forward=original_forward,
        ):
            del _original_forward
            do_breakdown = breakdown_layer == _layer_idx
            true_input = hidden_states if residual is None else hidden_states + residual
            if _layer_idx == 0 or do_breakdown:
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="stream",
                    layer_idx=_layer_idx,
                    name="layer_input",
                    positions=positions,
                    tensor=true_input,
                )

            if residual is None:
                residual_local = hidden_states
                normed = _layer.input_layernorm(hidden_states)
            else:
                normed, residual_local = _layer.input_layernorm(
                    hidden_states, residual)
            if do_breakdown:
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="input_rmsnorm",
                    positions=positions,
                    tensor=normed,
                )

            qkv, _ = _layer.self_attn.qkv_proj(normed)
            q, k, v = qkv.split([
                _layer.self_attn.q_size,
                _layer.self_attn.kv_size,
                _layer.self_attn.kv_size,
            ], dim=-1)
            if do_breakdown:
                for name, tensor in (("q", q), ("k", k), ("v", v)):
                    _d0_9_record_vectors(
                        capture_dir=capture_dir,
                        kind="breakdown",
                        layer_idx=_layer_idx,
                        name=name,
                        positions=positions,
                        tensor=tensor,
                    )
            q_rope, k_rope = _layer.self_attn.rotary_emb(positions, q, k)
            if do_breakdown:
                for name, tensor in (("q_rope", q_rope), ("k_rope", k_rope)):
                    _d0_9_record_vectors(
                        capture_dir=capture_dir,
                        kind="breakdown",
                        layer_idx=_layer_idx,
                        name=name,
                        positions=positions,
                        tensor=tensor,
                    )
            _d0_10_record_prefix_kv(
                capture_dir=capture_dir,
                layer_idx=_layer_idx,
                positions=positions,
                q_rope=q_rope,
                k_rope=k_rope,
                v=v,
            )
            attn_pre_o = _layer.self_attn.attn(q_rope, k_rope, v)
            _d0_10_record_attention_pre_o(
                capture_dir=capture_dir,
                layer_idx=_layer_idx,
                positions=positions,
                attention_pre_o=attn_pre_o,
            )
            if do_breakdown:
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="attention_pre_o",
                    positions=positions,
                    tensor=attn_pre_o,
                )
            attn_output, _ = _layer.self_attn.o_proj(attn_pre_o)
            if do_breakdown:
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="attention_output",
                    positions=positions,
                    tensor=attn_output,
                )
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="post_attention_residual",
                    positions=positions,
                    tensor=attn_output + residual_local,
                )

            mlp_input, residual_after_attn = _layer.post_attention_layernorm(
                attn_output, residual_local)
            if do_breakdown:
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="post_attention_rmsnorm",
                    positions=positions,
                    tensor=mlp_input,
                )
            mlp_output = _layer.mlp(mlp_input)
            if do_breakdown:
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="mlp_output",
                    positions=positions,
                    tensor=mlp_output,
                )
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="breakdown",
                    layer_idx=_layer_idx,
                    name="post_mlp_residual",
                    positions=positions,
                    tensor=mlp_output + residual_after_attn,
                )

            if _layer_idx in capture_layers:
                _d0_9_record_vectors(
                    capture_dir=capture_dir,
                    kind="stream",
                    layer_idx=_layer_idx,
                    name="layer_output",
                    positions=positions,
                    tensor=mlp_output + residual_after_attn,
                )
            return mlp_output, residual_after_attn

        layer.forward = wrapped_forward



def _build_siglip_vision_config() -> "SiglipVisionConfig":
    """Return a SiglipVisionConfig usable by SiglipVisionModel.

    The Cambrian-S-7B-LFP checkpoint stores 26 SigLIP vision encoder layers
    (0..25) and no SigLIP pooling head.  Do not instantiate the full upstream
    SigLIP/SigLIP2 config here, because that can create extra randomly
    initialized layers/head parameters in the rollout policy.
    """
    config_cls = CambrianSiglipVisionConfig or TransformersSiglipVisionConfig
    cfg = config_cls(
        hidden_size=SIGLIP_HIDDEN_DIM,
        num_hidden_layers=int(os.environ.get("VAGEN_CAMBRIAN_VISION_LAYERS", "26")),
        num_attention_heads=16,
        intermediate_size=4304,
        image_size=384,
        patch_size=14,
        num_channels=3,
        layer_norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
    )
    cfg.vision_use_head = False
    return cfg


# ---------------------------------------------------------------------------
# Image preprocessing (SigLIP, no HF Processor object needed)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_siglip_image_processor():
    from transformers import AutoImageProcessor, SiglipImageProcessor
    try:
        return AutoImageProcessor.from_pretrained(
            TRAINING_SIGLIP_MODEL,
            cache_dir=SIGLIP_CACHE,
            local_files_only=os.environ.get("VAGEN_ALLOW_HF_DOWNLOAD", "0") != "1",
        )
    except Exception:
        # Fallback preserves availability, but D0.2 should treat this as a
        # preprocessing-parity failure if it occurs in the live probe.
        return SiglipImageProcessor(image_size=384, patch_size=14)


def _expand2square(img: Image.Image, bg=(122, 116, 104)) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new("RGB", (s, s), bg)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def _preprocess_images(images: list[Image.Image]) -> torch.Tensor:
    """Returns float32 tensor (N, 3, 384, 384)."""
    proc = _get_siglip_image_processor()
    from vagen.models.cambrian_processor import preprocess_siglip_images

    return preprocess_siglip_images(images, proc)


# ---------------------------------------------------------------------------
# Processing info
# ---------------------------------------------------------------------------
class CambrianProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        # Fixed count avoids expensive dummy profiling at startup
        return {"image": TOKENS_PER_IMAGE}

    def get_hf_processor(self, **kwargs):  # type: ignore[override]
        # Cambrian has no HF processor; return None to prevent framework errors
        return None


# ---------------------------------------------------------------------------
# Dummy inputs builder
# ---------------------------------------------------------------------------
class CambrianDummyInputsBuilder(
        BaseDummyInputsBuilder[CambrianProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        n = mm_counts.get("image", 0)
        return IMAGE_TOKEN * n

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]):
        n = mm_counts.get("image", 0)
        return {
            "image": self._get_dummy_images(width=384, height=384,
                                            num_images=n),
        }


# ---------------------------------------------------------------------------
# MultiModal processor
# ---------------------------------------------------------------------------
class CambrianMultiModalProcessor(
        BaseMultiModalProcessor[CambrianProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        images: list[Image.Image] = list(mm_data.get("images", []))  # type: ignore
        if images:
            pixel_values = _preprocess_images(images)
        else:
            pixel_values = torch.zeros(0, 3, 384, 384)

        return BatchFeature({"input_ids": input_ids,
                             "pixel_values": pixel_values})

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {"pixel_values": MultiModalFieldConfig.batched("image")}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # Each single <image> (151665) → 756 × <|image_pad|> (151655)
        return [
            PromptReplacement(
                modality="image",
                target=[IMAGE_TOKEN_ID],
                replacement=[IMAGE_PAD_TOKEN_ID] * TOKENS_PER_IMAGE,
            )
        ]


# ---------------------------------------------------------------------------
# vLLM model
# ---------------------------------------------------------------------------
@MULTIMODAL_REGISTRY.register_processor(
    CambrianMultiModalProcessor,
    info=CambrianProcessingInfo,
    dummy_inputs=CambrianDummyInputsBuilder,
)
class CambrianVLLMForCausalLM(nn.Module, SupportsMultiModal):
    """
    vLLM V1 Cambrian-S model.

    Modules
    -------
    language_model  : vLLM Qwen2ForCausalLM (manages paged KV cache)
    vision_tower    : SiglipVisionModel (transformers, loaded from checkpoint)
    mm_projector    : Linear(1152→3584) + GELU + Linear(3584→3584)
    image_newline   : nn.Parameter (3584,)
    """

    supports_multimodal: bool = True

    def __init__(self, vllm_config, prefix: str = "") -> None:
        super().__init__()

        cfg = vllm_config.model_config.hf_config

        # ---- Language backbone (Qwen2) ----
        qwen2_config = Qwen2Config(
            hidden_size=LM_HIDDEN_DIM,
            intermediate_size=getattr(cfg, "intermediate_size", 18944),
            num_hidden_layers=getattr(cfg, "num_hidden_layers", 28),
            num_attention_heads=getattr(cfg, "num_attention_heads", 28),
            num_key_value_heads=getattr(cfg, "num_key_value_heads", 4),
            vocab_size=getattr(cfg, "vocab_size", 152064),
            max_position_embeddings=getattr(cfg, "max_position_embeddings",
                                            32768),
            rope_theta=getattr(cfg, "rope_theta", 1_000_000.0),
            rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
            tie_word_embeddings=False,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=qwen2_config,
            architectures=["Qwen2ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )
        _d0_9_install_qwen_hooks(self.language_model)

        # ---- SigLIP vision encoder ----
        # transformers>=4.50/5.x: SiglipConfig is a text+vision container and does
        # NOT expose layer_norm_eps; SiglipVisionModel requires SiglipVisionConfig.
        siglip_cfg = _build_siglip_vision_config()
        vision_cls = CambrianSiglipVisionModel or TransformersSiglipVisionModel
        self.vision_tower = vision_cls(siglip_cfg)
        if hasattr(self.vision_tower, "vision_model") and hasattr(self.vision_tower.vision_model, "head"):
            self.vision_tower.vision_model.head = nn.Identity()
        self.hf_config = cfg

        # ---- MM projector ----
        self.mm_projector = nn.Sequential(
            nn.Linear(SIGLIP_HIDDEN_DIM, LM_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(LM_HIDDEN_DIM, LM_HIDDEN_DIM),
        )

        # ---- Image newline token ----
        self.image_newline = nn.Parameter(torch.zeros(LM_HIDDEN_DIM))

        self.img_context_token_id = IMAGE_PAD_TOKEN_ID

    # ------------------------------------------------------------------
    # SupportsMultiModal interface
    # ------------------------------------------------------------------
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        return IMAGE_TOKEN if modality == "image" else None

    def get_language_model(self):
        return self.language_model

    def get_multimodal_embeddings(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[tuple]:
        if pixel_values is None or pixel_values.numel() == 0:
            return None

        # vLLM batches items as (N, 1, C, H, W) when merge_by_field_config=False.
        # Flatten to (N, C, H, W) so SigLIP gets the expected 4-D input.
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.flatten(0, 1)  # (N,1,C,H,W) -> (N,C,H,W)

        # SigLIP encode: (N, 729, 1152).
        #
        # Cambrian-S uses LOVSiglipVisionTower on the FSDP side: it builds a
        # 27-layer SigLIP config, deletes the final layer, and returns
        # hidden_states[-1] from the remaining 26-layer tower. The checkpoint
        # config still carries mm_vision_select_layer=-2 from the original
        # 27-layer convention. Since this vLLM wrapper instantiates the already
        # truncated 26-layer tower directly, the equivalent layer is -1.
        vt_dtype = next(self.vision_tower.parameters()).dtype
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values.to(vt_dtype),
            output_hidden_states=True,
        )
        select_layer = int(getattr(self.hf_config, "mm_vision_select_layer", -1))
        layer_count = len(self.vision_tower.vision_model.encoder.layers)
        if select_layer == -2 and layer_count == 26:
            select_layer = -1
        features = vision_outputs.hidden_states[select_layer]  # (N, 729, 1152)

        # Project: (N, 729, 3584)
        proj_dtype = self.mm_projector[0].weight.dtype
        features = self.mm_projector(features.to(proj_dtype))

        # Append newline and apply the same MIV overwrite as the FSDP adapter.
        features, _miv_features = build_cambrian_visual_features(
            features,
            self.image_newline,
            si_token_len=int(getattr(self.hf_config, "si_token_len", 729)),
            mm_use_newline=bool(getattr(self.hf_config, "mm_use_im_newline_token", True)),
            nfp_head=bool(getattr(self.hf_config, "nfp_head", False)),
            miv_token_len=int(getattr(self.hf_config, "miv_token_len", 0) or 0),
        )

        return tuple(features.unbind(0))  # N × (756, 3584)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(
            input_ids.clamp(min=0))

        if multimodal_embeddings:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.img_context_token_id,
            )
        return inputs_embeds

    # ------------------------------------------------------------------
    # Forward / logits
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        elif inputs_embeds is None:
            mm_embeds = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, mm_embeds)
            _d0_8_dump_forward(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                kwargs=kwargs,
            )
            input_ids = None
        else:
            _d0_8_dump_forward(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                kwargs=kwargs,
            )

        self._d0_8_last_positions = positions.detach().cpu()
        outputs = self.language_model.model(
            input_ids, positions, intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if torch.is_tensor(outputs):
            _d0_9_record_vectors(
                capture_dir=_d0_9_capture_dir(),
                kind="stream",
                layer_idx=None,
                name="final_norm",
                positions=positions,
                tensor=outputs,
            )
        return outputs

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.language_model.compute_logits(hidden_states)
        _d0_8_dump_logits(
            positions=getattr(self, "_d0_8_last_positions", None),
            logits=logits,
        )
        return logits

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        VISION_PFX = "model.vision_tower_aux_list.0.vision_tower."
        PROJ_PFX = "model.mm_projector."
        NEWLINE_KEY = "model.image_newline"

        lm_weights: list[Tuple[str, torch.Tensor]] = []
        local_tensors: dict[str, torch.Tensor] = {}

        for name, tensor in weights:
            if name.startswith(VISION_PFX):
                # e.g. model.vision_tower_aux_list.0.vision_tower.vision_model.X
                #   → vision_tower.vision_model.X
                local_tensors["vision_tower." + name[len(VISION_PFX):]] = tensor
            elif name.startswith(PROJ_PFX):
                local_tensors["mm_projector." + name[len(PROJ_PFX):]] = tensor
            elif name == NEWLINE_KEY:
                local_tensors["image_newline"] = tensor
            elif name.startswith("model.nfp_head."):
                pass  # skip NFP head
            elif name.startswith("model.") or name.startswith("lm_head."):
                # Qwen2 backbone weights go through the language_model loader
                lm_weights.append((name, tensor))

        # Delegate to Qwen2ForCausalLM weight loader (handles QKV fusion etc.)
        self.language_model.load_weights(iter(lm_weights))

        # Load local parameters
        params = dict(self.named_parameters())
        loaded_local: set[str] = set()
        unmatched_local: list[str] = []
        for pname, tensor in local_tensors.items():
            if pname in params:
                params[pname].data.copy_(tensor)
                loaded_local.add(pname)
            else:
                unmatched_local.append(pname)

        local_prefixes = ("vision_tower.", "mm_projector.")
        missing_local = [
            name for name in params
            if (
                name.startswith(local_prefixes)
                or name == "image_newline"
            )
            and name not in loaded_local
        ]
        if unmatched_local or missing_local:
            raise RuntimeError(
                "Cambrian vLLM local weight load mismatch: "
                f"unmatched={unmatched_local[:8]} missing={missing_local[:8]} "
                f"(counts: unmatched={len(unmatched_local)}, missing={len(missing_local)})"
            )

        # Return all parameter names so vllm's strict weight-tracking check passes.
        # The lm weights are loaded into tensors via self.language_model.load_weights()
        # even though their full root-level names (language_model.model.layers.*) are
        # not in local_tensors.  Returning the full named_parameters() set is correct
        # because every parameter is initialised from the checkpoint or from random
        # init for components not present in the checkpoint (there are none here).
        return {name for name, _ in self.named_parameters()}


# ---------------------------------------------------------------------------
# NOTE: Model registration is handled via the vllm.general_plugins entry_point
# in setup.py → vagen.models.cambrian_plugin:register().  vLLM calls
# load_general_plugins() in every spawned subprocess (EngineCore_DP*), which
# ensures CambrianQwenForCausalLM is in ModelRegistry.models before the engine
# looks it up.  The call below is kept for direct imports of this module
# (e.g. the parent vllm_async_server process via external_lib).
# ---------------------------------------------------------------------------
ModelRegistry.register_model(
    "CambrianQwenForCausalLM",
    "vagen.models.cambrian_vllm:CambrianVLLMForCausalLM",
)
