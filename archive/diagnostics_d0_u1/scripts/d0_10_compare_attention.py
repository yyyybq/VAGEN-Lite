#!/usr/bin/env python3
"""Build D0.10 attention-isolation artifacts from HF/vLLM probes."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch

SELECTED = [
    {"label": "P0", "idx": 0, "token": "<th", "prediction_position": 1413, "target_position": 1414},
    {"label": "P1", "idx": 73, "token": " required", "prediction_position": 1486, "target_position": 1487},
    {"label": "P2", "idx": 87, "token": "move", "prediction_position": 1500, "target_position": 1501},
]
PREFIX_QUERIES = [1413, 1486]
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128


def tensor(x: Any) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
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


def close(m: dict[str, Any]) -> bool:
    cosine = m.get("cosine")
    return (
        bool(m.get("shape_match"))
        and m["mean_abs_diff"] <= 1e-3
        and m["max_abs_diff"] <= 1e-2
        and (cosine is None or cosine >= 0.99999)
    )


def raw_by_idx_hf(path: Path) -> dict[int, dict[str, Any]]:
    obj = json.loads(path.read_text())
    return {int(k): v for k, v in obj["raw_logits_incremental"].items()}


def raw_by_idx_vllm(path: Path) -> dict[int, dict[str, Any]]:
    obj = json.loads(path.read_text())
    out = {}
    for rec in obj.get("raw_logit_records", []):
        target_pos = int(rec["target_position"])
        idx = target_pos - int(obj["sequence_layout"]["expanded_prompt_len"])
        out[idx] = rec
    return out


def compare_raw(a: dict[int, dict[str, Any]], b: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sel in SELECTED:
        idx = int(sel["idx"])
        ra = a.get(idx)
        rb = b.get(idx)
        if not ra or not rb:
            rows.append({"idx": idx, "label": sel["label"], "missing": True})
            continue
        rows.append({
            "idx": idx,
            "label": sel["label"],
            "token": sel["token"],
            "prediction_position": sel["prediction_position"],
            "raw_logit_a": float(ra["target_raw_logit"]),
            "raw_logit_b": float(rb["target_raw_logit"]),
            "delta_raw_logit": float(ra["target_raw_logit"]) - float(rb["target_raw_logit"]),
            "logprob_a": float(ra["target_logprob"]),
            "logprob_b": float(rb["target_logprob"]),
            "delta_logprob": float(ra["target_logprob"]) - float(rb["target_logprob"]),
            "rank_a": int(ra["target_rank"]),
            "rank_b": int(rb["target_rank"]),
            "rank_equal": int(ra["target_rank"]) == int(rb["target_rank"]),
        })
    return rows


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


def concat_vllm_records(hf_rec: dict[str, Any], shards: list[dict[str, Any]], attn_shards: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not shards:
        return None, {"source": "missing"}
    best = None
    for perm in itertools.permutations(shards):
        positions = perm[0]["positions"]
        if any(x["positions"] != positions for x in perm):
            continue
        q = torch.cat([tensor(x["q_rope_values"]) for x in perm], dim=0)
        score = metrics(tensor(hf_rec["q_rope_values"]), q)["mean_abs_diff"]
        if best is None or score < best[0]:
            best = (score, perm)
    if best is None:
        return None, {"source": "position_mismatch"}
    perm = best[1]
    pid_order = [x.get("pid") for x in perm]
    positions = perm[0]["positions"]
    k = torch.cat([tensor(x["k_rope_values"]) for x in perm], dim=1)
    v = torch.cat([tensor(x["v_values"]) for x in perm], dim=1)
    q = torch.cat([tensor(x["q_rope_values"]) for x in perm], dim=0)

    attn = None
    if attn_shards:
        by_pid = {x.get("pid"): x for x in attn_shards}
        if all(pid in by_pid for pid in pid_order):
            attn = torch.cat([tensor(by_pid[pid]["values"]) for pid in pid_order], dim=0)
    return {
        "positions": positions,
        "q_rope_values": q,
        "k_rope_values": k,
        "v_values": v,
        "attention_pre_o_values": attn,
    }, {"source": "tp_concat", "pid_order": pid_order}


def prefix_parity(hf: dict[int, dict[str, Any]], vllm: dict[int, list[dict[str, Any]]], vllm_attn: dict[int, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    query_payloads = {}
    rows = []
    first_mismatch = None
    total_mismatch_count = 0
    effective_match = True
    for query_pos in PREFIX_QUERIES:
        hf_rec = hf[query_pos]
        merged, choice = concat_vllm_records(hf_rec, vllm.get(query_pos, []), vllm_attn.get(query_pos, []))
        if merged is None:
            rows.append({"query_position": query_pos, "missing": "vllm", "choice": choice})
            first_mismatch = first_mismatch or {"query_position": query_pos, "position": None, "reason": "missing_vllm"}
            continue
        hf_pos = [int(x) for x in hf_rec["positions"]]
        v_pos = [int(x) for x in merged["positions"]]
        pos_match = hf_pos == v_pos
        hf_k = tensor(hf_rec["k_rope_values"])
        hf_v = tensor(hf_rec["v_values"])
        v_k = merged["k_rope_values"]
        v_v = merged["v_values"]
        q_metrics = metrics(tensor(hf_rec["q_rope_values"]), merged["q_rope_values"])
        k_metrics = metrics(hf_k, v_k)
        v_metrics = metrics(hf_v, v_v)
        row = {
            "query_position": query_pos,
            "choice": choice,
            "position_list_match": pos_match,
            "prefix_len_hf": len(hf_pos),
            "prefix_len_vllm": len(v_pos),
            "q_rope": q_metrics,
            "k_rope_full_prefix": k_metrics,
            "v_full_prefix": v_metrics,
        }
        per_pos = []
        mismatch_count = 0
        if pos_match:
            for i, pos in enumerate(hf_pos):
                km = metrics(hf_k[i], v_k[i])
                vm = metrics(hf_v[i], v_v[i])
                bad = not close(km) or not close(vm)
                if bad:
                    mismatch_count += 1
                per_pos.append({
                    "position": pos,
                    "k_rope": km,
                    "v": vm,
                    "mismatch": bad,
                })
                if bad and first_mismatch is None:
                    first_mismatch = {"query_position": query_pos, "position": pos, "k_rope": km, "v": vm}
        row["first_mismatching_prefix_position"] = next((p for p in per_pos if p["mismatch"]), None)
        row["strict_mismatching_prefix_position_count"] = mismatch_count
        total_mismatch_count += mismatch_count
        row["effectively_match"] = (
            pos_match
            and q_metrics["mean_abs_diff"] <= 1e-6
            and k_metrics["mean_abs_diff"] <= 1e-6
            and v_metrics["mean_abs_diff"] <= 1e-6
            and mismatch_count <= 1
        )
        effective_match = effective_match and row["effectively_match"]
        rows.append(row)
        query_payloads[query_pos] = {"hf": hf_rec, "vllm": merged}
    payload = {
        "tag": "d0_10_prefix_kv_parity",
        "layer_idx": 0,
        "full_prefix_kv_strict_match": first_mismatch is None,
        "full_prefix_kv_effectively_match": effective_match,
        "strict_mismatching_prefix_position_count_total": total_mismatch_count,
        "first_mismatching_prefix_position": first_mismatch,
        "strict_threshold": "per-position mean_abs<=1e-3 and max_abs<=1e-2 and cosine>=0.99999",
        "effective_threshold": "full-prefix q/k/v mean_abs<=1e-6 and <=1 strict outlier per query",
        "rows": rows,
    }
    return payload, query_payloads


def reference_attention(q_vec: torch.Tensor, k_mat: torch.Tensor, v_mat: torch.Tensor) -> torch.Tensor:
    q = q_vec.reshape(NUM_HEADS, HEAD_DIM).float()
    k = k_mat.reshape(k_mat.shape[0], NUM_KV_HEADS, HEAD_DIM).float()
    v = v_mat.reshape(v_mat.shape[0], NUM_KV_HEADS, HEAD_DIM).float()
    group = NUM_HEADS // NUM_KV_HEADS
    outs = []
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for h in range(NUM_HEADS):
        kv_h = h // group
        scores = torch.matmul(k[:, kv_h, :], q[h]) * scale
        probs = torch.softmax(scores, dim=0)
        outs.append(torch.matmul(probs, v[:, kv_h, :]))
    return torch.stack(outs, dim=0).reshape(-1)


def build_reference_payload(query_payloads: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for query_pos, payload in query_payloads.items():
        hf = payload["hf"]
        vllm = payload["vllm"]
        ref = reference_attention(
            tensor(hf["q_rope_values"]),
            tensor(hf["k_rope_values"]),
            tensor(hf["v_values"]),
        )
        hf_attn = tensor(hf["attention_pre_o_values"])
        vllm_attn = vllm.get("attention_pre_o_values")
        row = {
            "query_position": query_pos,
            "reference_vs_hf_attention_pre_o": metrics(ref, hf_attn),
            "reference_vs_vllm_attention_pre_o": None if vllm_attn is None else metrics(ref, vllm_attn),
            "hf_vs_vllm_attention_pre_o": None if vllm_attn is None else metrics(hf_attn, vllm_attn),
        }
        rows.append(row)
    closer = []
    for row in rows:
        hf_m = row["reference_vs_hf_attention_pre_o"]["mean_abs_diff"]
        v_m = None if row["reference_vs_vllm_attention_pre_o"] is None else row["reference_vs_vllm_attention_pre_o"]["mean_abs_diff"]
        closer.append({
            "query_position": row["query_position"],
            "reference_closer_to": "vllm" if v_m is not None and v_m < hf_m else "hf",
            "ref_hf_mean_abs": hf_m,
            "ref_vllm_mean_abs": v_m,
        })
    return {"tag": "d0_10_reference_attention", "layer_idx": 0, "rows": rows, "closer": closer}


def suffix_payload(base_vllm: Path, suffix_paths: list[Path], out_dir: Path) -> dict[str, Any]:
    base = raw_by_idx_vllm(base_vllm)
    rows = []
    for path in suffix_paths:
        obj = json.loads(path.read_text())
        variant = raw_by_idx_vllm(path)
        rows.append({
            "variant": path.name,
            "suffix_after_idx": obj.get("suffix_after_idx"),
            "comparisons": compare_raw(base, variant),
        })
    return {"tag": "d0_10_suffix_invariance", "baseline": str(base_vllm), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-original", required=True)
    ap.add_argument("--hf-instrumented", required=True)
    ap.add_argument("--hf-prefix", required=True)
    ap.add_argument("--vllm-original", required=True)
    ap.add_argument("--vllm-instrumented", required=True)
    ap.add_argument("--vllm-prefix-worker-dir", required=True)
    ap.add_argument("--vllm-suffix", action="append", default=[])
    ap.add_argument("--out-dir", default="docs/diagnosis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hook_payload = {
        "tag": "d0_10_hook_neutrality",
        "hf_original_vs_instrumented": compare_raw(raw_by_idx_hf(Path(args.hf_original)), raw_by_idx_hf(Path(args.hf_instrumented))),
        "vllm_original_vs_instrumented": compare_raw(raw_by_idx_vllm(Path(args.vllm_original)), raw_by_idx_vllm(Path(args.vllm_instrumented))),
        "neutral_threshold": "raw/logprob abs delta <= 1e-4 and rank unchanged for P0/P1/P2",
    }
    for group in ("hf_original_vs_instrumented", "vllm_original_vs_instrumented"):
        hook_payload[group + "_neutral"] = all(
            abs(r.get("delta_raw_logit", 999.0)) <= 1e-4
            and abs(r.get("delta_logprob", 999.0)) <= 1e-4
            and r.get("rank_equal") is True
            for r in hook_payload[group]
        )
    (out_dir / "d0_10_hook_neutrality.json").write_text(json.dumps(hook_payload, indent=2, ensure_ascii=False) + "\n")

    hf_prefix = load_hf_prefix(Path(args.hf_prefix))
    vllm_prefix = load_vllm_prefix(Path(args.vllm_prefix_worker_dir))
    vllm_attn = load_vllm_attn(Path(args.vllm_prefix_worker_dir))
    prefix_payload, query_payloads = prefix_parity(hf_prefix, vllm_prefix, vllm_attn)
    (out_dir / "d0_10_prefix_kv_parity.json").write_text(json.dumps(prefix_payload, indent=2, ensure_ascii=False) + "\n")

    suffix = suffix_payload(Path(args.vllm_original), [Path(x) for x in args.vllm_suffix], out_dir)
    (out_dir / "d0_10_suffix_invariance.json").write_text(json.dumps(suffix, indent=2, ensure_ascii=False) + "\n")

    if prefix_payload["full_prefix_kv_effectively_match"]:
        reference = build_reference_payload(query_payloads)
    else:
        reference = {
            "tag": "d0_10_reference_attention",
            "skipped": True,
            "reason": "full prefix K/V parity failed",
        }
    (out_dir / "d0_10_reference_attention.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")

    before_after = {
        "tag": "d0_10_before_after",
        "repair_applied": False,
        "before": {
            "P1_required": {
                "hf": raw_by_idx_hf(Path(args.hf_original)).get(73),
                "vllm": raw_by_idx_vllm(Path(args.vllm_original)).get(73),
            }
        },
        "after": None,
    }
    (out_dir / "d0_10_before_after.json").write_text(json.dumps(before_after, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "hook_hf_neutral": hook_payload["hf_original_vs_instrumented_neutral"],
        "hook_vllm_neutral": hook_payload["vllm_original_vs_instrumented_neutral"],
        "full_prefix_kv_strict_match": prefix_payload["full_prefix_kv_strict_match"],
        "full_prefix_kv_effectively_match": prefix_payload["full_prefix_kv_effectively_match"],
        "reference_skipped": reference.get("skipped", False),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
