#!/usr/bin/env python3
"""D0.6 vLLM native prompt-logprob forced scoring probe.

This script reconstructs the compact vLLM prompt from the fixed-sequence
manifest, appends the known response tokens, and asks vLLM to score that full
prompt via ``prompt_logprobs``. It does not sample a rollout, train, or step an
optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoTokenizer

from d0_6_hf_incremental_probe import classify_response, pair_stats, resolve_images


IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_ID = 151665
IMAGE_PAD_TOKEN_ID = 151655


def compact_prompt_from_manifest(sample: dict[str, Any]) -> tuple[list[int], int]:
    prompt_ids = sample["prompt_token_ids"]
    prompt_attn = sample["attention_mask"][: len(prompt_ids)]
    nonpad = [int(t) for t, m in zip(prompt_ids, prompt_attn) if int(m) != 0]
    compact: list[int] = []
    i = 0
    while i < len(nonpad):
        if nonpad[i] == IMAGE_TOKEN_INDEX:
            compact.append(IMAGE_TOKEN_ID)
            while i < len(nonpad) and nonpad[i] == IMAGE_TOKEN_INDEX:
                i += 1
        else:
            compact.append(nonpad[i])
            i += 1
    return compact, len(nonpad)


def logprob_item_to_float(item: Any) -> float | None:
    if item is None:
        return None
    val = getattr(item, "logprob", None)
    return None if val is None else float(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    parser.add_argument(
        "--vision-manifest",
        default="exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/manifest.jsonl",
    )
    parser.add_argument("--out", default="docs/diagnosis/d0_6_vllm_forced_scoring.json")
    parser.add_argument("--token-csv", default="docs/diagnosis/d0_6_vllm_forced_token_level.csv")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    os.environ.setdefault("CAMBRIAN_SRC", "/mnt/umm/users/yinbaiqiao/cambrian-s")
    os.environ.setdefault("VAGEN_ALLOW_HF_DOWNLOAD", "0")

    import vagen.models.cambrian_vllm  # noqa: F401
    from vagen.models.cambrian_processor import CambrianProcessorWrapper
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    checkpoint = manifest["model_checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    processor = CambrianProcessorWrapper(tokenizer)
    resolved_images = resolve_images(manifest, Path(args.vision_manifest), processor)

    llm = LLM(
        model=checkpoint,
        tokenizer=checkpoint,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        max_model_len=2176,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        prompt_logprobs=0,
    )

    prompts = []
    sample_meta = []
    for sample in manifest["samples"]:
        sid = int(sample["sample"])
        compact_prompt, expanded_prompt_len = compact_prompt_from_manifest(sample)
        response_ids = [int(x) for x in sample["active_response_token_ids"]]
        full_prompt_ids = compact_prompt + response_ids
        image = Image.open(resolved_images[sid]["pre_processor_path"]).convert("RGB")
        prompts.append(TokensPrompt(prompt_token_ids=full_prompt_ids, multi_modal_data={"image": [image]}))
        sample_meta.append(
            {
                "sample": sid,
                "selection": sample.get("selection"),
                "request_id": sample.get("request_id"),
                "compact_prompt_len": len(compact_prompt),
                "expanded_prompt_len": expanded_prompt_len,
                "response_len": len(response_ids),
            }
        )

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    token_rows: list[dict[str, Any]] = []
    aggregate_masks: dict[str, list[bool]] = {}
    aggregate_paths: dict[str, list[float]] = {
        "A_prod_fsdp_full": [],
        "B_hf_incremental": [],
        "C_vllm_production_incremental": [],
        "D_vllm_forced_prompt_logprobs": [],
    }
    samples_out: dict[str, Any] = {}

    for sample, meta, out in zip(manifest["samples"], sample_meta, outputs):
        sid = int(sample["sample"])
        active_len = int(meta["response_len"])
        expanded_prompt_len = int(meta["expanded_prompt_len"])
        prompt_token_ids = list(getattr(out, "prompt_token_ids", []) or [])
        prompt_logprobs = list(getattr(out, "prompt_logprobs", []) or [])
        d_logps: list[float] = []
        available = True
        missing_positions: list[int] = []
        for j, tid in enumerate(sample["response_token_ids"]):
            if j >= active_len:
                d_logps.append(0.0)
                continue
            pos = expanded_prompt_len + j
            if pos >= len(prompt_logprobs):
                available = False
                missing_positions.append(pos)
                d_logps.append(float("nan"))
                continue
            lpdict = prompt_logprobs[pos]
            item = None if lpdict is None else lpdict.get(int(tid))
            val = logprob_item_to_float(item)
            if val is None:
                available = False
                missing_positions.append(pos)
                d_logps.append(float("nan"))
            else:
                d_logps.append(val)

        types = classify_response(tokenizer, sample)
        active_mask = [bool(x) for x in sample["response_mask"]]
        masks_by_type = {"all_response": active_mask}
        for typ in ["think_text", "think_tag", "action_tag", "action_name", "eos_special", "pad", "other"]:
            masks_by_type[typ] = [m and t == typ for m, t in zip(active_mask, types)]

        paths = {
            "A_prod_fsdp_full": sample["fsdp_log_probs_T1"],
            "C_vllm_production_incremental": sample["vllm_rollout_log_probs"],
            "D_vllm_forced_prompt_logprobs": d_logps,
        }
        # B is read from the main matrix token CSV when available.
        b_by_idx: dict[int, float] = {}
        token_csv = Path("docs/diagnosis/d0_6_token_level_matrix.csv")
        if token_csv.exists():
            with token_csv.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if int(row["sample"]) == sid:
                        b_by_idx[int(row["response_idx"])] = float(row["B_hf_incremental"])
        paths["B_hf_incremental"] = [
            b_by_idx.get(i, float("nan")) for i in range(len(sample["response_token_ids"]))
        ]

        samples_out[str(sid)] = {
            **meta,
            "available": available,
            "missing_positions": missing_positions[:20],
            "vllm_processed_prompt_len": len(prompt_token_ids),
            "expected_processed_prompt_len": expanded_prompt_len + active_len,
            "generated_token_ids": [int(x) for x in out.outputs[0].token_ids],
            "pair_stats": {
                "A_vs_D": {
                    typ: pair_stats(paths["A_prod_fsdp_full"], paths["D_vllm_forced_prompt_logprobs"], mask)
                    for typ, mask in masks_by_type.items()
                },
                "B_vs_D": {
                    typ: pair_stats(paths["B_hf_incremental"], paths["D_vllm_forced_prompt_logprobs"], mask)
                    for typ, mask in masks_by_type.items()
                },
                "C_vs_D": {
                    typ: pair_stats(
                        paths["C_vllm_production_incremental"],
                        paths["D_vllm_forced_prompt_logprobs"],
                        mask,
                    )
                    for typ, mask in masks_by_type.items()
                },
            },
        }
        for i, tid in enumerate(sample["response_token_ids"]):
            row = {
                "sample": sid,
                "selection": sample.get("selection"),
                "request_id": sample.get("request_id"),
                "response_idx": i,
                "response_mask": int(active_mask[i]),
                "token_id": int(tid),
                "decoded_token": tokenizer.decode([int(tid)], skip_special_tokens=False),
                "token_type": types[i],
                "A_prod_fsdp_full": paths["A_prod_fsdp_full"][i],
                "B_hf_incremental": paths["B_hf_incremental"][i],
                "C_vllm_production_incremental": paths["C_vllm_production_incremental"][i],
                "D_vllm_forced_prompt_logprobs": paths["D_vllm_forced_prompt_logprobs"][i],
            }
            row["delta_A_minus_D"] = row["A_prod_fsdp_full"] - row["D_vllm_forced_prompt_logprobs"]
            row["delta_B_minus_D"] = row["B_hf_incremental"] - row["D_vllm_forced_prompt_logprobs"]
            row["delta_C_minus_D"] = row["C_vllm_production_incremental"] - row["D_vllm_forced_prompt_logprobs"]
            token_rows.append(row)

        for name, vals in paths.items():
            aggregate_paths[name].extend(vals)
        aggregate_masks.setdefault("all_response", []).extend(active_mask)
        for typ in ["think_text", "think_tag", "action_tag", "action_name", "eos_special", "pad", "other"]:
            aggregate_masks.setdefault(typ, []).extend([m and t == typ for m, t in zip(active_mask, types)])

    aggregate_pair_stats = {
        "A_vs_D": {
            typ: pair_stats(aggregate_paths["A_prod_fsdp_full"], aggregate_paths["D_vllm_forced_prompt_logprobs"], mask)
            for typ, mask in aggregate_masks.items()
        },
        "B_vs_D": {
            typ: pair_stats(aggregate_paths["B_hf_incremental"], aggregate_paths["D_vllm_forced_prompt_logprobs"], mask)
            for typ, mask in aggregate_masks.items()
        },
        "C_vs_D": {
            typ: pair_stats(
                aggregate_paths["C_vllm_production_incremental"],
                aggregate_paths["D_vllm_forced_prompt_logprobs"],
                mask,
            )
            for typ, mask in aggregate_masks.items()
        },
    }
    payload = {
        "tag": "d0_6_vllm_forced_scoring",
        "status": "AVAILABLE" if all(s["available"] for s in samples_out.values()) else "PARTIAL",
        "model_checkpoint": checkpoint,
        "paths": {
            "D_vllm_forced_prompt_logprobs": "vLLM native prompt_logprobs for compact prompt + fixed response",
        },
        "aggregate_pair_stats": aggregate_pair_stats,
        "samples": samples_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    if token_rows:
        with Path(args.token_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(token_rows[0].keys()))
            writer.writeheader()
            writer.writerows(token_rows)


if __name__ == "__main__":
    main()
