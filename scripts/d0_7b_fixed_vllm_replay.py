#!/usr/bin/env python3
"""D0.7b renderer-free fixed-request Cambrian vLLM replay.

This script intentionally avoids ActiveSpatial, renderer, GymAgentLoop, PPO, and
Ray trainer. It reconstructs compact vLLM prompts from D0.6 expanded FSDP
artifacts, attaches the exact recovered PNG observations from vision_sanity, and
uses the production CambrianVLLMForCausalLM registration through vLLM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoTokenizer

import vagen.models.cambrian_vllm  # noqa: F401 - registers CambrianVLLMForCausalLM
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


PAD_ID = 151643
IMAGE_SENTINEL = -200
IMAGE_TOKEN_ID = 151665
TOKENS_PER_IMAGE = 756


def collapse_expanded_prompt(expanded_ids: list[int], pad_id: int = PAD_ID) -> tuple[list[int], dict[str, Any]]:
    """Remove left padding and collapse each -200 x 756 image block to 151665."""
    first = 0
    while first < len(expanded_ids) and expanded_ids[first] == pad_id:
        first += 1
    ids = expanded_ids[first:]
    compact: list[int] = []
    blocks = []
    i = 0
    while i < len(ids):
        if ids[i] == IMAGE_SENTINEL:
            start = i
            while i < len(ids) and ids[i] == IMAGE_SENTINEL:
                i += 1
            length = i - start
            blocks.append({"expanded_start_no_pad": start, "length": length, "compact_index": len(compact)})
            compact.append(IMAGE_TOKEN_ID)
        else:
            compact.append(int(ids[i]))
            i += 1
    return compact, {"left_pad": first, "blocks": blocks}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_token_rows(path: Path) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out.setdefault(int(row["sample"]), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["response_idx"]))
    return out


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def logprob_entry_to_dict(tokenizer, token_id: int, obj: Any) -> dict[str, Any]:
    return {
        "token_id": int(token_id),
        "decoded_token": getattr(obj, "decoded_token", None) or decode_token(tokenizer, int(token_id)),
        "logprob": None if obj is None else float(obj.logprob),
        "rank": None if obj is None else getattr(obj, "rank", None),
    }


def top_logprobs_to_list(tokenizer, logprob_dict: Any) -> list[dict[str, Any]]:
    if not logprob_dict:
        return []
    items = []
    for token_id, obj in logprob_dict.items():
        items.append(logprob_entry_to_dict(tokenizer, int(token_id), obj))
    items.sort(key=lambda x: (x["rank"] is None, x["rank"] if x["rank"] is not None else 10**9))
    return items


def selected_indices(sample: dict[str, Any], token_rows: list[dict[str, str]]) -> list[int]:
    idxs = {0}
    for item in sample.get("top_residual_tokens", [])[:4]:
        idxs.add(int(item["idx"]))
    for row in token_rows:
        if row.get("token_type") == "action_name":
            idxs.add(int(row["response_idx"]))
            break
    for row in token_rows:
        if row.get("decoded_token") == " required":
            idxs.add(int(row["response_idx"]))
    return sorted(i for i in idxs if i < len(sample["active_response_token_ids"]))


def summarize_generation(tokenizer, sample: dict[str, Any], request_output: Any, token_rows: list[dict[str, str]]) -> dict[str, Any]:
    out0 = request_output.outputs[0]
    gen_ids = [int(x) for x in out0.token_ids]
    hist_ids = [int(x) for x in sample["active_response_token_ids"]]
    logprob_steps = out0.logprobs or []
    selected = selected_indices(sample, token_rows)
    records = []
    for idx in selected:
        hist_token = hist_ids[idx] if idx < len(hist_ids) else None
        gen_token = gen_ids[idx] if idx < len(gen_ids) else None
        lp_dict = logprob_steps[idx] if idx < len(logprob_steps) else None
        hist_entry = lp_dict.get(hist_token) if lp_dict and hist_token is not None else None
        gen_entry = lp_dict.get(gen_token) if lp_dict and gen_token is not None else None
        hist_c = None
        if idx < len(sample["vllm_rollout_log_probs"]):
            hist_c = float(sample["vllm_rollout_log_probs"][idx])
        records.append(
            {
                "idx": idx,
                "historical_token_id": hist_token,
                "historical_token_decoded": None if hist_token is None else decode_token(tokenizer, hist_token),
                "generated_token_id": gen_token,
                "generated_token_decoded": None if gen_token is None else decode_token(tokenizer, gen_token),
                "same_token_as_historical": hist_token == gen_token,
                "historical_C_logprob": hist_c,
                "direct_generated_token_logprob": None if gen_entry is None else float(gen_entry.logprob),
                "direct_historical_token_logprob_if_returned": None if hist_entry is None else float(hist_entry.logprob),
                "direct_historical_token_rank_if_returned": None if hist_entry is None else getattr(hist_entry, "rank", None),
                "top20": top_logprobs_to_list(tokenizer, lp_dict)[:20],
            }
        )
    return {
        "generated_token_ids": gen_ids,
        "generated_text": out0.text,
        "historical_active_token_ids": hist_ids,
        "prefix_token_match_count": sum(1 for a, b in zip(gen_ids, hist_ids) if a == b),
        "full_prefix_matches_historical": gen_ids[: len(hist_ids)] == hist_ids[: len(gen_ids)],
        "records": records,
    }


def summarize_forced_prompt_logprobs(
    tokenizer,
    sample: dict[str, Any],
    request_output: Any,
    compact_prompt_len: int,
    expanded_prompt_len: int,
    token_rows: list[dict[str, str]],
) -> dict[str, Any]:
    prompt_lps = request_output.prompt_logprobs or []
    selected = selected_indices(sample, token_rows)
    records = []
    for idx in selected:
        tok = int(sample["active_response_token_ids"][idx])
        compact_abs_idx = compact_prompt_len + idx
        abs_prompt_idx = expanded_prompt_len + idx
        lp_dict = prompt_lps[abs_prompt_idx] if abs_prompt_idx < len(prompt_lps) else None
        entry = lp_dict.get(tok) if lp_dict else None
        hist_c = float(sample["vllm_rollout_log_probs"][idx]) if idx < len(sample["vllm_rollout_log_probs"]) else None
        hist_b = None
        if idx < len(token_rows):
            hist_b = float(token_rows[idx]["B_hf_incremental"])
        records.append(
            {
                "idx": idx,
                "absolute_compact_prompt_index": compact_abs_idx,
                "absolute_expanded_prompt_index": abs_prompt_idx,
                "target_token_id": tok,
                "target_decoded_token": decode_token(tokenizer, tok),
                "historical_B_hf_incremental": hist_b,
                "historical_C_vllm_incremental": hist_c,
                "forced_vllm_target_logprob": None if entry is None else float(entry.logprob),
                "forced_vllm_target_rank": None if entry is None else getattr(entry, "rank", None),
                "top20": top_logprobs_to_list(tokenizer, lp_dict)[:20],
            }
        )
    return {"records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    ap.add_argument("--token-csv", default="docs/diagnosis/d0_6_token_level_matrix.csv")
    ap.add_argument("--out-dir", default="docs/diagnosis")
    ap.add_argument("--model", default="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP")
    ap.add_argument("--chunked-prefill", choices=["on", "off"], default="on")
    ap.add_argument("--prefix-cache", choices=["on", "off"], default="on")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--top-logprobs", type=int, default=20)
    args = ap.parse_args()

    repo = Path.cwd()
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (repo / args.manifest).open() as f:
        manifest = json.load(f)
    token_rows_by_sample = load_token_rows(repo / args.token_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)

    image_paths = {
        3: repo / "exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/pid216624_sample_000_pre_processor.png",
        5: repo / "exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/pid216636_sample_000_pre_processor.png",
    }

    replay_samples = []
    generation_prompts = []
    forced_prompts = []
    for sample in manifest["samples"]:
        sample_id = int(sample["sample"])
        if sample_id not in {3, 5}:
            continue
        compact_ids, collapse_info = collapse_expanded_prompt(sample["prompt_token_ids"], PAD_ID)
        image_path = image_paths[sample_id]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        active_response_ids = [int(x) for x in sample["active_response_token_ids"]]
        expanded_prompt_len = len(compact_ids)
        for block in collapse_info["blocks"]:
            expanded_prompt_len += int(block["length"]) - 1
        generation_prompts.append(
            TokensPrompt(
                prompt_token_ids=compact_ids,
                multi_modal_data={"image": [image]},
            )
        )
        forced_prompts.append(
            TokensPrompt(
                prompt_token_ids=compact_ids + active_response_ids,
                multi_modal_data={"image": [image]},
            )
        )
        replay_samples.append(
            {
                "sample": sample_id,
                "selection": sample["selection"],
                "request_id": sample["request_id"],
                "compact_prompt_len": len(compact_ids),
                "expanded_prompt_len": expanded_prompt_len,
                "active_response_len": len(active_response_ids),
                "compact_image_token_positions": [i for i, tid in enumerate(compact_ids) if tid == IMAGE_TOKEN_ID],
                "collapse_info": collapse_info,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "image_artifact": sample["image_artifact"],
                "sampling_temperature": manifest["sampling_temperature"],
                "logprob_temperature": manifest["logprob_temperature"],
                "top_p": 0.9,
                "top_k": -1,
                "max_tokens": args.max_tokens,
                "exact_decoded_response": sample["exact_decoded_response"],
            }
        )

    fixed_manifest = {
        "tag": "d0_7b_fixed_replay_manifest",
        "source_manifest": args.manifest,
        "model": args.model,
        "runtime": {
            "tensor_parallel_size": 2,
            "dtype": "bfloat16",
            "attention_backend_env": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "enforce_eager": True,
            "chunked_prefill": args.chunked_prefill,
            "prefix_cache": args.prefix_cache,
            "renderer": "not used",
            "active_spatial_env": "not used",
        },
        "samples": replay_samples,
    }
    (out_dir / "d0_7b_fixed_replay_manifest.json").write_text(
        json.dumps(fixed_manifest, indent=2, ensure_ascii=False) + "\n"
    )

    llm_kwargs = dict(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=2,
        dtype="bfloat16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.35,
        max_model_len=2176,
        seed=0,
        limit_mm_per_prompt={"image": 25},
    )
    if args.chunked_prefill == "on":
        llm_kwargs["enable_chunked_prefill"] = True
        llm_kwargs["max_num_batched_tokens"] = 4096
    else:
        llm_kwargs["enable_chunked_prefill"] = False
    if args.prefix_cache == "on":
        llm_kwargs["enable_prefix_caching"] = True
    else:
        llm_kwargs["enable_prefix_caching"] = False

    llm = LLM(**llm_kwargs)

    gen_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=float(manifest["sampling_temperature"]),
        top_p=0.9,
        top_k=-1,
        repetition_penalty=1.0,
        seed=0,
        logprobs=args.top_logprobs,
        skip_special_tokens=False,
    )
    gen_outputs = llm.generate(generation_prompts, gen_params)

    forced_params = SamplingParams(
        max_tokens=1,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        seed=0,
        prompt_logprobs=args.top_logprobs,
        skip_special_tokens=False,
    )
    forced_outputs = llm.generate(forced_prompts, forced_params)

    direct = {
        "tag": "d0_7b_direct_replay_parity",
        "chunked_prefill": args.chunked_prefill,
        "prefix_cache": args.prefix_cache,
        "samples": [],
    }
    raw_topk = {
        "tag": "d0_7b_vllm_raw_topk",
        "semantics": "vLLM raw logprobs/prompt_logprobs from renderer-free production CambrianVLLMForCausalLM engine",
        "samples": [],
    }
    hf_compare = {
        "tag": "d0_7b_hf_vs_vllm_topk",
        "note": "HF incremental B values/ranks are historical from D0.6 token CSV/top logits artifacts; vLLM top20 is captured in this direct replay.",
        "samples": [],
    }

    samples_by_id = {int(s["sample"]): s for s in manifest["samples"]}
    for replay, gen_out, forced_out in zip(replay_samples, gen_outputs, forced_outputs, strict=True):
        sid = int(replay["sample"])
        sample = samples_by_id[sid]
        rows = token_rows_by_sample[sid]
        gen_summary = summarize_generation(tokenizer, sample, gen_out, rows)
        forced_summary = summarize_forced_prompt_logprobs(
            tokenizer,
            sample,
            forced_out,
            replay["compact_prompt_len"],
            replay["expanded_prompt_len"],
            rows,
        )
        direct["samples"].append({**replay, "generation": gen_summary, "forced_scoring": forced_summary})
        raw_topk["samples"].append(
            {
                "sample": sid,
                "selection": replay["selection"],
                "request_id": replay["request_id"],
                "generation_records": gen_summary["records"],
                "forced_scoring_records": forced_summary["records"],
            }
        )
        hf_compare["samples"].append(
            {
                "sample": sid,
                "selection": replay["selection"],
                "request_id": replay["request_id"],
                "records": forced_summary["records"],
            }
        )

    suffix = "" if args.chunked_prefill == "on" and args.prefix_cache == "on" else f"_{args.chunked_prefill}_{args.prefix_cache}"
    (out_dir / f"d0_7b_direct_replay_parity{suffix}.json").write_text(
        json.dumps(direct, indent=2, ensure_ascii=False) + "\n"
    )
    (out_dir / f"d0_7b_vllm_raw_topk{suffix}.json").write_text(
        json.dumps(raw_topk, indent=2, ensure_ascii=False) + "\n"
    )
    (out_dir / f"d0_7b_hf_vs_vllm_topk{suffix}.json").write_text(
        json.dumps(hf_compare, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
