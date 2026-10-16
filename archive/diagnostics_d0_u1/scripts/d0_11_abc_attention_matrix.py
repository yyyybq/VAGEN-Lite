#!/usr/bin/env python3
"""D0.11 P1-only A/B/C attention matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from d0_11_precision_reference import (
    PREFIX_QUERIES,
    concat_vllm,
    load_hf_prefix,
    load_vllm_attn,
    load_vllm_prefix,
    metrics,
    reference_r32,
    reference_rbf16,
    tensor,
)

P1 = 1486


def close(row: dict, *, mean: float = 1e-4, max_abs: float = 5e-3) -> bool:
    return bool(row.get("shape_match")) and row["mean_abs_diff"] <= mean and row["max_abs_diff"] <= max_abs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-prefix", default="docs/diagnosis/d0_10_hf_prefix_kv.json")
    ap.add_argument("--vllm-worker-dir", default="docs/diagnosis/d0_10_runs/instrumented_prefix/worker")
    ap.add_argument("--hf-full", default="docs/diagnosis/d0_11_hf_full_attention_p1.json")
    ap.add_argument("--out", default="docs/diagnosis/d0_11_abc_attention_matrix.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if P1 not in PREFIX_QUERIES:
        raise RuntimeError("P1 must be part of D0.11 reference query set")
    device = torch.device(args.device)
    hf = load_hf_prefix(Path(args.hf_prefix))
    vp = load_vllm_prefix(Path(args.vllm_worker_dir))
    va = load_vllm_attn(Path(args.vllm_worker_dir))
    hf_full = json.loads(Path(args.hf_full).read_text())

    hf_rec = hf[P1]
    vllm_rec = concat_vllm(hf_rec, vp[P1], va[P1])
    q = tensor(hf_rec["q_rope_values"])
    k = tensor(hf_rec["k_rope_values"])
    v = tensor(hf_rec["v_values"])

    A = tensor(hf_full["attention_pre_o_values"])
    B = tensor(hf_rec["attention_pre_o_values"])
    C = vllm_rec["attention_pre_o_values"]
    R32 = reference_r32(q, k, v, device)
    Rbf16 = reference_rbf16(q, k, v, device)

    pairwise = {
        "A_HF_full_vs_B_HF_incremental": metrics(A, B),
        "A_HF_full_vs_Rbf16": metrics(A, Rbf16),
        "B_HF_incremental_vs_Rbf16": metrics(B, Rbf16),
        "C_vLLM_vs_R32": metrics(C, R32),
        "C_vLLM_vs_Rbf16": metrics(C, Rbf16),
        "A_HF_full_vs_C_vLLM": metrics(A, C),
        "B_HF_incremental_vs_C_vLLM": metrics(B, C),
        "R32_vs_Rbf16": metrics(R32, Rbf16),
    }
    payload = {
        "tag": "d0_11_abc_attention_matrix",
        "query_position": P1,
        "token": " required",
        "device": str(device),
        "definitions": {
            "A": "HF/FSDP-style full-sequence layer0 attention_pre_o",
            "B": "HF incremental layer0 attention_pre_o from D0.10",
            "C": "vLLM layer0 attention_pre_o from D0.10",
            "R32": "FP32 reference attention from saved D0.10 Q/K/V",
            "Rbf16": "BF16 eager-flow reference attention from saved D0.10 Q/K/V",
        },
        "pairwise": pairwise,
        "forms_two_groups": {
            "A_B_Rbf16_group": close(pairwise["A_HF_full_vs_B_HF_incremental"]) and close(pairwise["A_HF_full_vs_Rbf16"]) and close(pairwise["B_HF_incremental_vs_Rbf16"]),
            "C_R32_group": close(pairwise["C_vLLM_vs_R32"]),
            "between_group_material_gap": pairwise["B_HF_incremental_vs_C_vLLM"]["mean_abs_diff"] > 1e-3,
        },
        "hf_full_raw_logits": hf_full.get("raw_logits_full_sequence"),
    }
    payload["precision_closure_pass"] = all(payload["forms_two_groups"].values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "precision_closure_pass": payload["precision_closure_pass"],
        "forms_two_groups": payload["forms_two_groups"],
        "key_pairwise": {
            k: pairwise[k]
            for k in [
                "A_HF_full_vs_B_HF_incremental",
                "B_HF_incremental_vs_Rbf16",
                "C_vLLM_vs_R32",
                "B_HF_incremental_vs_C_vLLM",
            ]
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
