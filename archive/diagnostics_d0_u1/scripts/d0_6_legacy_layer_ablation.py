#!/usr/bin/env python3
"""D0.6 bounded ablation: score fixed sequences with the legacy SigLIP layer.

This is diagnostic only. It monkeypatches the standalone HF adapter's
``encode_images`` method to return the checkpoint config layer directly
(``mm_vision_select_layer=-2``) instead of the Cambrian-S effective layer used by
the repaired path. No rollout, training, or optimizer step is run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from d0_6_hf_incremental_probe import (
    classify_response,
    install_cache_compat_shim,
    install_decoder_cache_tuple_shim,
    pair_stats,
    pearson,
    resolve_images,
    score_sample,
)


def patch_encode_images_to_config_layer(model: Any) -> None:
    inner = model.get_model()
    vt = inner.vision_tower_aux_list[0]
    vt_model = getattr(vt, "vision_tower", vt)
    select_layer = int(getattr(model.config, "mm_vision_select_layer", -1))

    def legacy_encode_images(images: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        for pixel_values in images:
            vt_dtype = next(vt_model.parameters()).dtype
            vision_outputs = vt_model(
                pixel_values=pixel_values.to(vt_dtype),
                output_hidden_states=True,
            )
            out.append(vision_outputs.hidden_states[select_layer])
        return out

    model.encode_images = legacy_encode_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    parser.add_argument(
        "--vision-manifest",
        default="exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/manifest.jsonl",
    )
    parser.add_argument("--out", default="docs/diagnosis/d0_6_legacy_layer_ablation.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()

    os.environ.setdefault("CAMBRIAN_SRC", "/mnt/umm/users/yinbaiqiao/cambrian-s")
    os.environ.setdefault("VAGEN_ALLOW_HF_DOWNLOAD", "0")
    install_cache_compat_shim()

    import vagen.models.cambrian_register  # noqa: F401
    from vagen.models.cambrian_processor import CambrianProcessorWrapper

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
    patch_encode_images_to_config_layer(model)

    samples_out: dict[str, Any] = {}
    aggregate_masks: dict[str, list[bool]] = {}
    aggregate_paths: dict[str, list[float]] = {
        "A_prod_fsdp_full": [],
        "legacy_layer_full": [],
        "legacy_layer_incremental": [],
        "C_vllm_production_incremental": [],
    }
    for sample in manifest["samples"]:
        sid = int(sample["sample"])
        pixel_values = resolved_images[sid]["pixel_values"]
        top_indices = {int(x["idx"]) for x in sample.get("top_residual_tokens", [])[:10]}
        res = score_sample(model, tokenizer, sample, pixel_values, device, top_indices=top_indices)
        types = classify_response(tokenizer, sample)
        active_mask = [bool(x) for x in sample["response_mask"]]
        masks_by_type = {"all_response": active_mask}
        for typ in ["think_text", "think_tag", "action_tag", "action_name", "eos_special", "pad", "other"]:
            masks_by_type[typ] = [m and t == typ for m, t in zip(active_mask, types)]

        paths = {
            "A_prod_fsdp_full": sample["fsdp_log_probs_T1"],
            "legacy_layer_full": res["hf_full_logps"],
            "legacy_layer_incremental": res["hf_incremental_logps"],
            "C_vllm_production_incremental": sample["vllm_rollout_log_probs"],
        }
        samples_out[str(sid)] = {
            "selection": sample.get("selection"),
            "request_id": sample.get("request_id"),
            "pair_stats": {
                "legacy_full_vs_C": {
                    typ: pair_stats(paths["legacy_layer_full"], paths["C_vllm_production_incremental"], mask)
                    for typ, mask in masks_by_type.items()
                },
                "A_vs_legacy_full": {
                    typ: pair_stats(paths["A_prod_fsdp_full"], paths["legacy_layer_full"], mask)
                    for typ, mask in masks_by_type.items()
                },
                "legacy_full_vs_legacy_incremental": {
                    typ: pair_stats(paths["legacy_layer_full"], paths["legacy_layer_incremental"], mask)
                    for typ, mask in masks_by_type.items()
                },
            },
            "position_corr_abs_delta_legacy_full_vs_C": pearson(
                [float(i) for i, m in enumerate(active_mask) if m],
                [
                    abs(paths["legacy_layer_full"][i] - paths["C_vllm_production_incremental"][i])
                    for i, m in enumerate(active_mask)
                    if m
                ],
            ),
            "top_cases": [
                {
                    "idx": int(x["idx"]),
                    "decoded_token": x.get("decoded_token"),
                    "A_prod_fsdp_full": float(paths["A_prod_fsdp_full"][int(x["idx"])]),
                    "legacy_layer_full": float(paths["legacy_layer_full"][int(x["idx"])]),
                    "legacy_layer_incremental": float(paths["legacy_layer_incremental"][int(x["idx"])]),
                    "C_vllm_production_incremental": float(paths["C_vllm_production_incremental"][int(x["idx"])]),
                }
                for x in sample.get("top_residual_tokens", [])[:10]
            ],
        }
        for name, vals in paths.items():
            aggregate_paths[name].extend(vals)
        aggregate_masks.setdefault("all_response", []).extend(active_mask)
        for typ in ["think_text", "think_tag", "action_tag", "action_name", "eos_special", "pad", "other"]:
            aggregate_masks.setdefault(typ, []).extend([m and t == typ for m, t in zip(active_mask, types)])

    out = {
        "tag": "d0_6_legacy_layer_ablation",
        "note": "Standalone HF adapter scored with encode_images forced to config mm_vision_select_layer=-2.",
        "model_checkpoint": checkpoint,
        "sampling_temperature": manifest.get("sampling_temperature"),
        "logprob_temperature": manifest.get("logprob_temperature"),
        "aggregate_pair_stats": {
            "legacy_full_vs_C": {
                typ: pair_stats(
                    aggregate_paths["legacy_layer_full"],
                    aggregate_paths["C_vllm_production_incremental"],
                    mask,
                )
                for typ, mask in aggregate_masks.items()
            },
            "A_vs_legacy_full": {
                typ: pair_stats(aggregate_paths["A_prod_fsdp_full"], aggregate_paths["legacy_layer_full"], mask)
                for typ, mask in aggregate_masks.items()
            },
            "legacy_full_vs_legacy_incremental": {
                typ: pair_stats(
                    aggregate_paths["legacy_layer_full"],
                    aggregate_paths["legacy_layer_incremental"],
                    mask,
                )
                for typ, mask in aggregate_masks.items()
            },
        },
        "samples": samples_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
