"""Tiny random Qwen3.5 models and real processor inputs for the q38s4 CPU tests.

The config is the Qwen3.5-4B hub snapshot's (same qwen3_5 classes, token ids and vision patching as Qwen3.8-27B)
shrunk to 2 decoder layers (one gated delta net, one full attention) and a 2-block vision tower (so blocks[-2] is
not the last block); the inputs come from Qwen3.8-27B's own processor (64x64 frames -> 4x4 patches -> 4 video tokens
per unit).  The gated delta net runs on its torch fallbacks (FLA's kernels need a GPU).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

SMOKE_4B = Path("/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/"
                "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
MODEL_27B = Path("/mnt/umm/users/yinbaiqiao/q38/models/Qwen3.8-27B")
needs_models = pytest.mark.skipif(not ((SMOKE_4B / "config.json").is_file() and
                                       (MODEL_27B / "tokenizer.json").is_file()),
                                  reason="model folders not readable")


@pytest.fixture(scope="module")
def cpu_gdn():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    mp = pytest.MonkeyPatch()
    for name in ("FusedRMSNormGated", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        if hasattr(M, name):
            mp.setattr(M, name, None)
    yield M
    mp.undo()


def tiny_config():
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SMOKE_4B)
    assert cfg.model_type == "qwen3_5"
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size = 64, 128
    t.num_hidden_layers, t.layer_types, t.full_attention_interval = 2, ["linear_attention", "full_attention"], 2
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 2, 1, 64
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 2, 16, 16
    rope = dict(t.rope_parameters)
    rope["mrope_section"] = [3, 3, 2]
    t.rope_parameters = rope
    t.mtp_num_hidden_layers = 0
    v.depth, v.hidden_size, v.num_heads, v.intermediate_size, v.out_hidden_size = 2, 32, 2, 64, 64
    return cfg


def build(seed=0, untied=False):
    """untied: a separate lm_head as in the 27B (the 4B config ties it to the input embeddings)."""
    from transformers import Qwen3_5ForConditionalGeneration

    cfg = tiny_config()
    if untied:
        cfg.tie_word_embeddings = cfg.text_config.tie_word_embeddings = False
    torch.manual_seed(seed)
    m = Qwen3_5ForConditionalGeneration._from_config(cfg, attn_implementation="eager", torch_dtype=torch.float32)
    if untied:
        assert m.lm_head.weight.data_ptr() != m.get_input_embeddings().weight.data_ptr()
    return m.train()


def freeze_encoder(model):
    """What LLaMA-Factory's freeze_vision_tower does for qwen3_5 (visual.pos_embed / patch_embed / blocks)."""
    for name, p in model.named_parameters():
        if any(k in name for k in ("visual.pos_embed", "visual.patch_embed", "visual.blocks")):
            p.requires_grad_(False)


def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_27B)


def frames(n, size=64, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(n)]


def video_batch(proc, n_frames=(6,), answer="B", size=64, seed=0):
    """One sample: a user turn with len(n_frames) videos and a question, the answer; labels on the answer only."""
    vids = [frames(n, size, seed + i) for i, n in enumerate(n_frames)]
    meta = [{"fps": 2.0, "total_num_frames": n, "frames_indices": list(range(n))} for n in n_frames]
    user = "".join("<|vision_start|><|video_pad|><|vision_end|>" for _ in n_frames) + "Which one?"
    prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    full = prompt + f"{answer}<|im_end|>\n"
    out = proc(text=[full], videos=vids, video_metadata=meta, do_sample_frames=False, return_tensors="pt")
    n_prompt = len(proc(text=[prompt], videos=vids, video_metadata=meta, do_sample_frames=False,
                        return_tensors="pt")["input_ids"][0])
    labels = out["input_ids"].clone()
    labels[:, :n_prompt] = -100
    batch = {k: out[k] for k in ("input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw")}
    if "mm_token_type_ids" in out:
        batch["mm_token_type_ids"] = out["mm_token_type_ids"]
    batch["labels"] = labels
    return batch


def unit_groups(input_ids, video_token_id):
    """Positions of the video tokens grouped by unit, read from the token layout (each unit is one run of video
    tokens between <|vision_start|> and <|vision_end|>), independent of nfp.unit_pairs."""
    ids = input_ids[0].tolist()
    groups, cur = [], []
    for i, tok in enumerate(ids):
        if tok == video_token_id:
            cur.append(i)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups
