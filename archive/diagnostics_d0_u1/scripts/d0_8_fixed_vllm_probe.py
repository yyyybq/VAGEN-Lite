#!/usr/bin/env python3
"""D0.8 fixed-request vLLM prefill/position probe.

Renderer-free, env-free, trainer-free.  Runs only sample 5 from the D0.6 fixed
manifest and asks vLLM to forced-score the historical response token sequence.
When D0_8_CAPTURE_DIR is set, CambrianVLLMForCausalLM dumps actual worker
forward inputs (positions, image-pad spans, selected layer-0 embeddings).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoTokenizer

import vagen.models.cambrian_vllm  # noqa: F401
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

PAD_ID = 151643
IMAGE_SENTINEL = -200
IMAGE_TOKEN_ID = 151665
TOKENS_PER_IMAGE = 756
SELECTED_IDXS = [0, 73, 87]


def collapse_expanded_prompt(expanded_ids: list[int]) -> tuple[list[int], dict[str, Any]]:
    first = 0
    while first < len(expanded_ids) and expanded_ids[first] == PAD_ID:
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
            blocks.append({"expanded_start_no_pad": start, "length": i - start, "compact_index": len(compact)})
            compact.append(IMAGE_TOKEN_ID)
        else:
            compact.append(int(ids[i]))
            i += 1
    return compact, {"left_pad": first, "blocks": blocks}


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def top_logprobs_to_list(tokenizer, logprob_dict: Any) -> list[dict[str, Any]]:
    if not logprob_dict:
        return []
    out = []
    for token_id, obj in logprob_dict.items():
        out.append({
            "token_id": int(token_id),
            "decoded_token": getattr(obj, "decoded_token", None) or decode_token(tokenizer, int(token_id)),
            "logprob": float(obj.logprob),
            "rank": getattr(obj, "rank", None),
        })
    out.sort(key=lambda x: (x["rank"] is None, x["rank"] if x["rank"] is not None else 10**9))
    return out


def extract_records(tokenizer, sample: dict[str, Any], output: Any, expanded_prompt_len: int) -> list[dict[str, Any]]:
    prompt_lps = output.prompt_logprobs or []
    records = []
    for idx in SELECTED_IDXS:
        tok = int(sample["active_response_token_ids"][idx])
        abs_idx = expanded_prompt_len + idx
        lp_dict = prompt_lps[abs_idx] if abs_idx < len(prompt_lps) else None
        entry = lp_dict.get(tok) if lp_dict else None
        records.append({
            "idx": idx,
            "absolute_expanded_prompt_index": abs_idx,
            "target_token_id": tok,
            "target_decoded_token": decode_token(tokenizer, tok),
            "historical_C_vllm_logprob": float(sample["vllm_rollout_log_probs"][idx]),
            "direct_vllm_forced_logprob": None if entry is None else float(entry.logprob),
            "direct_vllm_forced_rank": None if entry is None else getattr(entry, "rank", None),
            "top20": top_logprobs_to_list(tokenizer, lp_dict)[:20],
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    ap.add_argument("--out-dir", default="docs/diagnosis")
    ap.add_argument("--model", default="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP")
    ap.add_argument("--max-num-batched-tokens", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--prefix-cache", choices=["on", "off"], default="on")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--top-logprobs", type=int, default=20)
    args = ap.parse_args()

    repo = Path.cwd()
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = out_dir / "d0_8_runs" / args.run_tag / "worker_forward"
    capture_dir.mkdir(parents=True, exist_ok=True)
    os.environ["D0_8_CAPTURE_DIR"] = str(capture_dir)
    os.environ["D0_8_TARGET_POSITIONS"] = "1414,1487,1501"
    os.environ["D0_8_TARGET_TOKEN_IDS"] = "13708,2567,3397"
    os.environ["D0_8_MAX_CAPTURE_CALLS"] = "64"
    os.environ["D0_8_MAX_LOGIT_CAPTURE_CALLS"] = "64"
    os.environ["D0_8_DUMP_VECTOR_VALUES"] = "1"

    manifest = json.loads((repo / args.manifest).read_text())
    sample = next(s for s in manifest["samples"] if int(s["sample"]) == 5)
    compact_ids, collapse_info = collapse_expanded_prompt(sample["prompt_token_ids"])
    expanded_prompt_len = len(compact_ids) + sum(int(b["length"]) - 1 for b in collapse_info["blocks"])
    response_ids = [int(x) for x in sample["active_response_token_ids"]]
    image_path = repo / "exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/pid216636_sample_000_pre_processor.png"
    image = Image.open(image_path).convert("RGB")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)

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
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.prefix_cache == "on",
    )
    llm = LLM(**llm_kwargs)

    forced_prompt = TokensPrompt(
        prompt_token_ids=compact_ids + response_ids,
        multi_modal_data={"image": [image]},
    )
    params = SamplingParams(
        max_tokens=1,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        seed=0,
        prompt_logprobs=args.top_logprobs,
        skip_special_tokens=False,
    )

    repeats = []
    for rep in range(args.repeat):
        output = llm.generate([forced_prompt], params)[0]
        repeats.append({"repeat": rep, "records": extract_records(tokenizer, sample, output, expanded_prompt_len)})

    payload = {
        "tag": "d0_8_fixed_vllm_probe",
        "run_tag": args.run_tag,
        "runtime_request": {
            "attention_backend_env": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "prefix_cache": args.prefix_cache,
            "dtype": "bfloat16",
            "tensor_parallel_size": 2,
            "enforce_eager": True,
        },
        "sequence_layout": {
            "compact_prompt_len": len(compact_ids),
            "expanded_prompt_len": expanded_prompt_len,
            "compact_image_token_positions": [i for i, x in enumerate(compact_ids) if x == IMAGE_TOKEN_ID],
            "expanded_image_span": [
                int(collapse_info["blocks"][0]["expanded_start_no_pad"]),
                int(collapse_info["blocks"][0]["expanded_start_no_pad"] + collapse_info["blocks"][0]["length"]),
            ],
            "response_len": len(response_ids),
            "selected_response_indices": SELECTED_IDXS,
            "selected_absolute_positions": [expanded_prompt_len + i for i in SELECTED_IDXS],
        },
        "capture_dir": str(capture_dir),
        "repeats": repeats,
    }
    (out_dir / f"d0_8_vllm_forced_{args.run_tag}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
