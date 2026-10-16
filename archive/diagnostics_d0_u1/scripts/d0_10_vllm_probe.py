#!/usr/bin/env python3
"""D0.10 renderer-free vLLM probe for attention root-cause isolation."""

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
SELECTED_IDXS = [0, 73, 87]
PREDICTION_POSITIONS = [1413, 1486, 1500]
TARGET_POSITIONS = [1414, 1487, 1501]
TARGET_TOKEN_IDS = [13708, 2567, 3397]
PREFIX_QUERIES = [1413, 1486]


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


def extract_prompt_records(tokenizer, sample: dict[str, Any], output: Any, expanded_prompt_len: int) -> list[dict[str, Any]]:
    prompt_lps = output.prompt_logprobs or []
    records = []
    for idx, pred_pos in zip(SELECTED_IDXS, PREDICTION_POSITIONS, strict=True):
        tok = int(sample["active_response_token_ids"][idx])
        target_pos = expanded_prompt_len + idx
        lp_dict = prompt_lps[target_pos] if target_pos < len(prompt_lps) else None
        entry = lp_dict.get(tok) if lp_dict else None
        records.append({
            "idx": idx,
            "prediction_position": pred_pos,
            "target_position": target_pos,
            "target_token_id": tok,
            "target_decoded_token": decode_token(tokenizer, tok),
            "direct_vllm_forced_logprob": None if entry is None else float(entry.logprob),
            "direct_vllm_forced_rank": None if entry is None else getattr(entry, "rank", None),
            "top20": top_logprobs_to_list(tokenizer, lp_dict)[:20],
        })
    return records


def load_raw_logit_records(capture_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(capture_dir.glob("logits_pid*_call*.json")):
        obj = json.loads(path.read_text())
        for rec in obj.get("records", []):
            item = dict(rec)
            item["source_file"] = str(path)
            records.append(item)
    records.sort(key=lambda x: (int(x["prediction_position"]), int(x["target_token_id"])))
    return records


def mutate_suffix(response_ids: list[int], suffix_after_idx: int | None, fill_token_id: int) -> list[int]:
    out = list(response_ids)
    if suffix_after_idx is None:
        return out
    for i in range(suffix_after_idx + 1, len(out)):
        out[i] = fill_token_id
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--manifest", default="docs/diagnosis/d0_6_fixed_sequence_manifest.json")
    ap.add_argument("--out-dir", default="docs/diagnosis")
    ap.add_argument("--model", default="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP")
    ap.add_argument("--instrument", action="store_true")
    ap.add_argument("--prefix-capture", action="store_true")
    ap.add_argument("--suffix-after-idx", type=int, default=None)
    ap.add_argument("--fill-token-id", type=int, default=279)
    ap.add_argument("--top-logprobs", type=int, default=20)
    args = ap.parse_args()

    repo = Path.cwd()
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = out_dir / "d0_10_runs" / args.run_tag
    worker_dir = capture_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)

    os.environ["D0_8_CAPTURE_DIR"] = str(worker_dir)
    os.environ["D0_8_TARGET_POSITIONS"] = ",".join(str(x) for x in TARGET_POSITIONS)
    os.environ["D0_8_TARGET_TOKEN_IDS"] = ",".join(str(x) for x in TARGET_TOKEN_IDS)
    os.environ["D0_8_MAX_CAPTURE_CALLS"] = "16"
    os.environ["D0_8_MAX_LOGIT_CAPTURE_CALLS"] = "16"
    os.environ["D0_8_DUMP_VECTOR_VALUES"] = "0"
    if args.instrument:
        os.environ["D0_9_CAPTURE_DIR"] = str(worker_dir)
        os.environ["D0_9_TARGET_POSITIONS"] = ",".join(str(x) for x in PREDICTION_POSITIONS)
        os.environ["D0_9_CAPTURE_LAYERS"] = "0"
        os.environ["D0_9_BREAKDOWN_LAYER"] = "0"
        os.environ["D0_9_DUMP_VALUES"] = "1"
    if args.prefix_capture:
        os.environ["D0_10_PREFIX_CAPTURE"] = "1"
        os.environ["D0_10_PREFIX_LAYER"] = "0"
        os.environ["D0_10_PREFIX_TARGET_POSITIONS"] = ",".join(str(x) for x in PREFIX_QUERIES)

    manifest = json.loads((repo / args.manifest).read_text())
    sample = next(s for s in manifest["samples"] if int(s["sample"]) == 5)
    compact_ids, collapse_info = collapse_expanded_prompt(sample["prompt_token_ids"])
    expanded_prompt_len = len(compact_ids) + sum(int(b["length"]) - 1 for b in collapse_info["blocks"])
    response_ids = [int(x) for x in sample["active_response_token_ids"]]
    response_ids = mutate_suffix(response_ids, args.suffix_after_idx, args.fill_token_id)

    image_path = repo / "exps/vagen_active_spatial/d0_6_b5_forced_scoring_matrix/vision_sanity/pid216636_sample_000_pre_processor.png"
    image = Image.open(image_path).convert("RGB")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)

    llm = LLM(
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
        max_num_batched_tokens=2176,
        max_num_seqs=1,
        enable_prefix_caching=True,
    )

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
    output = llm.generate([forced_prompt], params)[0]
    payload = {
        "tag": "d0_10_vllm_probe",
        "run_tag": args.run_tag,
        "instrument": bool(args.instrument),
        "prefix_capture": bool(args.prefix_capture),
        "suffix_after_idx": args.suffix_after_idx,
        "fill_token_id": args.fill_token_id,
        "capture_dir": str(worker_dir),
        "sequence_layout": {
            "compact_prompt_len": len(compact_ids),
            "expanded_prompt_len": expanded_prompt_len,
            "selected_response_indices": SELECTED_IDXS,
            "selected_prediction_positions": PREDICTION_POSITIONS,
            "selected_target_positions": [expanded_prompt_len + i for i in SELECTED_IDXS],
        },
        "prompt_logprob_records": extract_prompt_records(tokenizer, sample, output, expanded_prompt_len),
        "raw_logit_records": load_raw_logit_records(worker_dir),
    }
    (out_dir / f"d0_10_vllm_probe_{args.run_tag}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
