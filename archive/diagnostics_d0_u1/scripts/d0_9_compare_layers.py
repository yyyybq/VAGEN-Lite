#!/usr/bin/env python3
"""Compare D0.9 HF and vLLM layer captures."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

SELECTED = [
    {"label": "P0", "idx": 0, "token": "<th", "prediction_position": 1413, "target_position": 1414},
    {"label": "P1", "idx": 73, "token": " required", "prediction_position": 1486, "target_position": 1487},
    {"label": "P2", "idx": 87, "token": "move", "prediction_position": 1500, "target_position": 1501},
]
TP_LOCAL_NAMES = {"q", "k", "v", "q_rope", "k_rope", "attention_pre_o"}


def load_jsonl(path: Path) -> dict[tuple[str, int | None, str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int | None, str, int], list[dict[str, Any]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for rec in obj.get("records", []):
                key = (
                    obj.get("kind"),
                    obj.get("layer_idx"),
                    obj.get("name"),
                    int(rec["absolute_position"]),
                )
                item = dict(rec)
                if "pid" in obj:
                    item["pid"] = obj["pid"]
                out[key].append(item)
    return out


def load_vllm_dir(path: Path) -> dict[tuple[str, int | None, str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int | None, str, int], list[dict[str, Any]]] = defaultdict(list)
    for p in sorted(path.glob("d0_9_pid*.jsonl")):
        data = load_jsonl(p)
        for key, vals in data.items():
            out[key].extend(vals)
    return out


def tensor_from_record(rec: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(rec["values"], dtype=torch.float32)


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    if a.numel() != b.numel():
        return {
            "shape_match": 0.0,
            "hf_numel": float(a.numel()),
            "vllm_numel": float(b.numel()),
            "max_abs_diff": math.inf,
            "mean_abs_diff": math.inf,
            "cosine": float("nan"),
            "l2": math.inf,
        }
    d = (a - b).float()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    cosine = float(torch.dot(a, b).item() / denom.item()) if denom.item() != 0 else float("nan")
    return {
        "shape_match": 1.0,
        "hf_numel": float(a.numel()),
        "vllm_numel": float(b.numel()),
        "max_abs_diff": float(d.abs().max().item()) if d.numel() else 0.0,
        "mean_abs_diff": float(d.abs().mean().item()) if d.numel() else 0.0,
        "cosine": cosine,
        "l2": float(torch.linalg.vector_norm(d).item()) if d.numel() else 0.0,
    }


def choose_vllm_vector(
    *,
    hf_vec: torch.Tensor,
    vllm_records: list[dict[str, Any]],
    name: str,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not vllm_records:
        return None, {"source": "missing"}
    if name in TP_LOCAL_NAMES and len(vllm_records) > 1:
        shards = [(rec.get("pid"), tensor_from_record(rec)) for rec in vllm_records]
        best: tuple[float, torch.Tensor, tuple[Any, ...]] | None = None
        for perm in itertools.permutations(shards):
            vec = torch.cat([x[1] for x in perm], dim=0)
            m = metrics(hf_vec, vec)
            score = m["mean_abs_diff"]
            if best is None or score < best[0]:
                best = (score, vec, tuple(x[0] for x in perm))
        assert best is not None
        return best[1], {"source": "tp_concat", "pid_order": list(best[2])}
    candidates = [(rec.get("pid"), tensor_from_record(rec)) for rec in vllm_records]
    best_pid, best_vec, best_m = None, None, None
    for pid, vec in candidates:
        m = metrics(hf_vec, vec)
        if best_m is None or m["mean_abs_diff"] < best_m["mean_abs_diff"]:
            best_pid, best_vec, best_m = pid, vec, m
    return best_vec, {"source": "single_or_duplicate", "pid": best_pid}


def compare_key(
    hf: dict[tuple[str, int | None, str, int], list[dict[str, Any]]],
    vllm: dict[tuple[str, int | None, str, int], list[dict[str, Any]]],
    key: tuple[str, int | None, str, int],
) -> dict[str, Any] | None:
    hf_records = hf.get(key, [])
    if not hf_records:
        return None
    hf_vec = tensor_from_record(hf_records[0])
    vllm_vec, choice = choose_vllm_vector(hf_vec=hf_vec, vllm_records=vllm.get(key, []), name=key[2])
    if vllm_vec is None:
        return {
            "kind": key[0],
            "layer_idx": key[1],
            "name": key[2],
            "absolute_position": key[3],
            "missing": "vllm",
        }
    row = {
        "kind": key[0],
        "layer_idx": key[1],
        "name": key[2],
        "absolute_position": key[3],
        "choice": choice,
    }
    row.update(metrics(hf_vec, vllm_vec))
    return row


def is_divergent(row: dict[str, Any]) -> bool:
    if row.get("missing"):
        return True
    return (
        row["mean_abs_diff"] > 1e-3
        or row["max_abs_diff"] > 1e-2
        or (not math.isnan(row["cosine"]) and row["cosine"] < 0.99999)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-jsonl", default="docs/diagnosis/d0_9_runs/hf_layers.jsonl")
    ap.add_argument("--vllm-dir", required=True)
    ap.add_argument("--out-layer", default="docs/diagnosis/d0_9_layer_divergence.json")
    ap.add_argument("--out-first", default="docs/diagnosis/d0_9_first_divergent_layer.json")
    ap.add_argument("--out-breakdown", default="docs/diagnosis/d0_9_first_layer_breakdown.json")
    args = ap.parse_args()

    hf = load_jsonl(Path(args.hf_jsonl))
    vllm = load_vllm_dir(Path(args.vllm_dir))
    positions = [x["prediction_position"] for x in SELECTED]

    rows = []
    for pos in positions:
        rows.append(compare_key(hf, vllm, ("stream", 0, "layer_input", pos)))
        for layer in range(28):
            rows.append(compare_key(hf, vllm, ("stream", layer, "layer_output", pos)))
        rows.append(compare_key(hf, vllm, ("stream", None, "final_norm", pos)))
    rows = [r for r in rows if r is not None]

    by_layer_summary = []
    for name, layer in [("layer_input", 0)] + [("layer_output", i) for i in range(28)] + [("final_norm", None)]:
        layer_rows = [r for r in rows if r["name"] == name and r["layer_idx"] == layer]
        if not layer_rows:
            continue
        by_layer_summary.append({
            "name": name,
            "layer_idx": layer,
            "max_abs_diff_max": max(r.get("max_abs_diff", math.inf) for r in layer_rows),
            "mean_abs_diff_max": max(r.get("mean_abs_diff", math.inf) for r in layer_rows),
            "cosine_min": min(r.get("cosine", float("nan")) for r in layer_rows),
            "divergent": any(is_divergent(r) for r in layer_rows),
            "rows": layer_rows,
        })

    last_matching = None
    first_divergent = None
    for item in by_layer_summary:
        label = "layer0_input" if item["name"] == "layer_input" else (
            "final_norm" if item["name"] == "final_norm" else f"layer_{item['layer_idx']}_output"
        )
        if item["divergent"]:
            first_divergent = label
            break
        last_matching = label

    first_layer_idx = None
    if first_divergent and first_divergent.startswith("layer_") and first_divergent.endswith("_output"):
        first_layer_idx = int(first_divergent.split("_")[1])

    breakdown_rows = []
    if first_layer_idx is not None:
        for pos in positions:
            for name in [
                "layer_input",
                "input_rmsnorm",
                "q",
                "k",
                "v",
                "q_rope",
                "k_rope",
                "attention_pre_o",
                "attention_output",
                "post_attention_residual",
                "post_attention_rmsnorm",
                "mlp_output",
                "post_mlp_residual",
            ]:
                row = compare_key(hf, vllm, ("breakdown", first_layer_idx, name, pos))
                if row is not None:
                    row["divergent"] = is_divergent(row)
                    breakdown_rows.append(row)
        first_tensor = None
        for name in [
            "layer_input",
            "input_rmsnorm",
            "q",
            "k",
            "v",
            "q_rope",
            "k_rope",
            "attention_pre_o",
            "attention_output",
            "post_attention_residual",
            "post_attention_rmsnorm",
            "mlp_output",
            "post_mlp_residual",
        ]:
            candidates = [r for r in breakdown_rows if r["name"] == name]
            if candidates and any(r["divergent"] for r in candidates):
                first_tensor = name
                break
    else:
        first_tensor = None

    layer_payload = {
        "tag": "d0_9_layer_divergence",
        "selected": SELECTED,
        "threshold": "mean_abs>1e-3 or max_abs>1e-2 or cosine<0.99999",
        "by_layer_summary": by_layer_summary,
    }
    first_payload = {
        "tag": "d0_9_first_divergent_layer",
        "last_matching_layer": last_matching,
        "first_divergent_layer": first_divergent,
        "first_divergent_layer_idx": first_layer_idx,
        "first_divergent_tensor": first_tensor,
    }
    breakdown_payload = {
        "tag": "d0_9_first_layer_breakdown",
        "first_divergent_layer_idx": first_layer_idx,
        "first_divergent_tensor": first_tensor,
        "rows": breakdown_rows,
    }
    Path(args.out_layer).write_text(json.dumps(layer_payload, indent=2, ensure_ascii=False) + "\n")
    Path(args.out_first).write_text(json.dumps(first_payload, indent=2, ensure_ascii=False) + "\n")
    Path(args.out_breakdown).write_text(json.dumps(breakdown_payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(first_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
