#!/usr/bin/env python3
"""D0.11 HF full-sequence layer0 attention_pre_o capture for sample 5 P1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from d0_9_hf_layer_probe import install_cache_compat_shim, topk_case
from d0_10_hf_probe import install_layer0_prefix_hook

P1_POSITION = 1486
P1_RESPONSE_IDX = 73


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    ap.add_argument("--out", default="docs/diagnosis/d0_11_hf_full_attention_p1.json")
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

    capture_state = {
        "q_rope_by_pos": {},
        "k_rope_by_pos": {},
        "v_by_pos": {},
        "attention_pre_o_by_pos": {},
    }
    install_layer0_prefix_hook(model, capture_state)

    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
    position_ids = torch.tensor(sample["position_ids"], dtype=torch.long, device=device).unsqueeze(0)
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    active_response_ids = [int(x) for x in sample["active_response_token_ids"]]
    target_tid = active_response_ids[P1_RESPONSE_IDX]

    with torch.inference_mode():
        model._d0_10_current_attention_mask_1d = attention_mask
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            use_cache=False,
            return_dict=True,
        )
        local_indices = [
            i for i, pos in enumerate(position_ids.reshape(-1).detach().cpu().tolist())
            if int(pos) == P1_POSITION and int(attention_mask.reshape(-1)[i].item()) == 1
        ]
        if len(local_indices) != 1:
            raise RuntimeError(f"Expected one active P1 local index, got {local_indices}")
        p1_logits = topk_case(out.logits.reshape(-1, out.logits.shape[-1])[local_indices[0]], target_tid, tokenizer)

    payload = {
        "tag": "d0_11_hf_full_attention_probe",
        "model": args.model,
        "dtype": "torch.bfloat16",
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "query_position": P1_POSITION,
        "response_idx": P1_RESPONSE_IDX,
        "target_token_id": int(target_tid),
        "target_decoded_token": tokenizer.decode([int(target_tid)], skip_special_tokens=False),
        "attention_pre_o_values": capture_state["attention_pre_o_by_pos"].get(P1_POSITION),
        "raw_logits_full_sequence": p1_logits,
    }
    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
