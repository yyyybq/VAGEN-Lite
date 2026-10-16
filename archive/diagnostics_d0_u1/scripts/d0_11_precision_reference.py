#!/usr/bin/env python3
"""D0.11 attention precision reference from saved D0.10 Q/K/V artifacts."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch

PREFIX_QUERIES = [1413, 1486]
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
GROUP = NUM_HEADS // NUM_KV_HEADS


def tensor(x: Any, *, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device)


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float().reshape(-1).cpu()
    b = b.detach().float().reshape(-1).cpu()
    if a.numel() != b.numel():
        return {
            "shape_match": False,
            "a_numel": int(a.numel()),
            "b_numel": int(b.numel()),
            "max_abs_diff": math.inf,
            "mean_abs_diff": math.inf,
            "cosine": None,
            "l2": math.inf,
        }
    d = a - b
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return {
        "shape_match": True,
        "a_numel": int(a.numel()),
        "b_numel": int(b.numel()),
        "max_abs_diff": float(d.abs().max().item()) if d.numel() else 0.0,
        "mean_abs_diff": float(d.abs().mean().item()) if d.numel() else 0.0,
        "cosine": float(torch.dot(a, b).item() / denom.item()) if denom.item() else None,
        "l2": float(torch.linalg.vector_norm(d).item()) if d.numel() else 0.0,
    }


def load_hf_prefix(path: Path) -> dict[int, dict[str, Any]]:
    obj = json.loads(path.read_text())
    return {int(rec["query_position"]): rec for rec in obj["records"]}


def load_vllm_prefix(worker_dir: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(worker_dir.glob("d0_10_pid*.jsonl")):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("kind") != "d0_10_prefix_kv":
                    continue
                for rec in obj.get("records", []):
                    item = dict(rec)
                    item["pid"] = obj.get("pid")
                    out.setdefault(int(rec["query_position"]), []).append(item)
    return out


def load_vllm_attn(worker_dir: Path) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(worker_dir.glob("d0_10_pid*.jsonl")):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("kind") != "d0_10_attention_pre_o":
                    continue
                for rec in obj.get("records", []):
                    item = dict(rec)
                    item["pid"] = obj.get("pid")
                    out.setdefault(int(rec["query_position"]), []).append(item)
    return out


def concat_vllm(hf_rec: dict[str, Any], shards: list[dict[str, Any]], attn_shards: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    hf_q = tensor(hf_rec["q_rope_values"])
    for perm in itertools.permutations(shards):
        if any(x["positions"] != perm[0]["positions"] for x in perm):
            continue
        q = torch.cat([tensor(x["q_rope_values"]) for x in perm], dim=0)
        score = metrics(hf_q, q)["mean_abs_diff"]
        if best is None or score < best[0]:
            best = (score, perm)
    if best is None:
        raise RuntimeError("Could not merge vLLM TP prefix shards")
    perm = best[1]
    pid_order = [x.get("pid") for x in perm]
    by_pid = {x.get("pid"): x for x in attn_shards}
    return {
        "positions": perm[0]["positions"],
        "q_rope_values": torch.cat([tensor(x["q_rope_values"]) for x in perm], dim=0),
        "k_rope_values": torch.cat([tensor(x["k_rope_values"]) for x in perm], dim=1),
        "v_values": torch.cat([tensor(x["v_values"]) for x in perm], dim=1),
        "attention_pre_o_values": torch.cat([tensor(by_pid[pid]["values"]) for pid in pid_order], dim=0),
        "pid_order": pid_order,
    }


def reference_r32(q_vec: torch.Tensor, k_mat: torch.Tensor, v_mat: torch.Tensor, device: torch.device) -> torch.Tensor:
    q = q_vec.to(device=device, dtype=torch.float32).reshape(NUM_HEADS, HEAD_DIM)
    k = k_mat.to(device=device, dtype=torch.float32).reshape(k_mat.shape[0], NUM_KV_HEADS, HEAD_DIM)
    v = v_mat.to(device=device, dtype=torch.float32).reshape(v_mat.shape[0], NUM_KV_HEADS, HEAD_DIM)
    outs = []
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for h in range(NUM_HEADS):
        kv_h = h // GROUP
        scores = torch.matmul(k[:, kv_h, :], q[h]) * scale
        probs = torch.softmax(scores, dim=0)
        outs.append(torch.matmul(probs, v[:, kv_h, :]))
    return torch.stack(outs, dim=0).reshape(-1)


def reference_rbf16(q_vec: torch.Tensor, k_mat: torch.Tensor, v_mat: torch.Tensor, device: torch.device) -> torch.Tensor:
    q = q_vec.to(device=device, dtype=torch.bfloat16).reshape(NUM_HEADS, HEAD_DIM)
    k = k_mat.to(device=device, dtype=torch.bfloat16).reshape(k_mat.shape[0], NUM_KV_HEADS, HEAD_DIM)
    v = v_mat.to(device=device, dtype=torch.bfloat16).reshape(v_mat.shape[0], NUM_KV_HEADS, HEAD_DIM)
    outs = []
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for h in range(NUM_HEADS):
        kv_h = h // GROUP
        scores = torch.matmul(k[:, kv_h, :], q[h])
        scores = (scores * scale).to(torch.bfloat16)
        probs = torch.softmax(scores, dim=0, dtype=torch.float32).to(torch.bfloat16)
        outs.append(torch.matmul(probs.unsqueeze(0), v[:, kv_h, :]).squeeze(0))
    return torch.stack(outs, dim=0).reshape(-1).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-prefix", default="docs/diagnosis/d0_10_hf_prefix_kv.json")
    ap.add_argument("--vllm-worker-dir", default="docs/diagnosis/d0_10_runs/instrumented_prefix/worker")
    ap.add_argument("--out", default="docs/diagnosis/d0_11_precision_reference.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    hf = load_hf_prefix(Path(args.hf_prefix))
    vp = load_vllm_prefix(Path(args.vllm_worker_dir))
    va = load_vllm_attn(Path(args.vllm_worker_dir))
    rows = []

    for query_pos in PREFIX_QUERIES:
        hf_rec = hf[query_pos]
        vllm_rec = concat_vllm(hf_rec, vp[query_pos], va[query_pos])
        q = tensor(hf_rec["q_rope_values"])
        k = tensor(hf_rec["k_rope_values"])
        v = tensor(hf_rec["v_values"])
        hf_attn = tensor(hf_rec["attention_pre_o_values"])
        vllm_attn = vllm_rec["attention_pre_o_values"]
        r32 = reference_r32(q, k, v, device)
        rbf16 = reference_rbf16(q, k, v, device)
        rows.append({
            "query_position": query_pos,
            "token": "<th" if query_pos == 1413 else " required",
            "reference_inputs": "HF D0.10 saved layer0 Q/K/V; vLLM K/V effectively matched in D0.10",
            "r32_vs_hf": metrics(r32, hf_attn),
            "r32_vs_vllm": metrics(r32, vllm_attn),
            "rbf16_vs_hf": metrics(rbf16, hf_attn),
            "rbf16_vs_vllm": metrics(rbf16, vllm_attn),
            "r32_vs_rbf16": metrics(r32, rbf16),
        })

    closer = []
    for row in rows:
        closer.append({
            "query_position": row["query_position"],
            "r32_closer_to": "vllm" if row["r32_vs_vllm"]["mean_abs_diff"] < row["r32_vs_hf"]["mean_abs_diff"] else "hf",
            "rbf16_closer_to": "vllm" if row["rbf16_vs_vllm"]["mean_abs_diff"] < row["rbf16_vs_hf"]["mean_abs_diff"] else "hf",
            "r32_hf_mean_abs": row["r32_vs_hf"]["mean_abs_diff"],
            "r32_vllm_mean_abs": row["r32_vs_vllm"]["mean_abs_diff"],
            "rbf16_hf_mean_abs": row["rbf16_vs_hf"]["mean_abs_diff"],
            "rbf16_vllm_mean_abs": row["rbf16_vs_vllm"]["mean_abs_diff"],
        })
    precision_hypothesis = all(
        x["r32_closer_to"] == "vllm" and x["rbf16_closer_to"] == "hf"
        for x in closer
    )
    payload = {
        "tag": "d0_11_precision_reference",
        "device": str(device),
        "queries": PREFIX_QUERIES,
        "rbf16_definition": [
            "Q/K/V cast to bfloat16",
            "BF16 QK matmul",
            "scale, then cast scores to bfloat16",
            "softmax(dtype=float32)",
            "cast softmax probabilities to bfloat16",
            "BF16 probability-V matmul",
        ],
        "rows": rows,
        "closer": closer,
        "precision_hypothesis_pass": precision_hypothesis,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "precision_hypothesis_pass": precision_hypothesis,
        "closer": closer,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
