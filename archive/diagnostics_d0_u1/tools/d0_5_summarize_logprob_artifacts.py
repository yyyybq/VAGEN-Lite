#!/usr/bin/env python3
"""Summarize D0.4/D0.5 logprob-alignment artifacts.

This is offline-only: it reads existing token-alignment CSV files and writes small
JSON summaries for residual drift localization. It never starts training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Iterable


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {"token_count": 0}
    deltas = [float(r["delta_raw_T1"]) for r in rows]
    abs_deltas = sorted(abs(x) for x in deltas)
    ratios = [float(r["ratio_raw_T1"]) for r in rows]
    return {
        "token_count": len(rows),
        "mean_abs_delta_logp": statistics.mean(abs_deltas),
        "median_abs_delta_logp": statistics.median(abs_deltas),
        "p95_abs_delta_logp": abs_deltas[int(0.95 * (len(abs_deltas) - 1))],
        "max_abs_delta_logp": max(abs_deltas),
        "ratio_median": statistics.median(ratios),
        "frac_abs_ratio_minus_1_gt_0p01": sum(abs(x - 1) > 0.01 for x in ratios) / len(ratios),
        "frac_abs_ratio_minus_1_gt_0p05": sum(abs(x - 1) > 0.05 for x in ratios) / len(ratios),
        "pearson": _pearson(
            [float(r["vllm_logp"]) for r in rows],
            [float(r["fsdp_logp_raw_T1"]) for r in rows],
        ),
    }


def _read_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("response_mask") != "1":
                continue
            row["sample"] = int(row["sample"])
            row["idx"] = int(row["idx"])
            rows.append(row)
    return rows


def _position_summary(rows: list[dict], source: str) -> dict:
    buckets = {}
    for lo in range(0, 128, 16):
        hi = lo + 15
        buckets[f"{lo}-{hi}"] = _stats([r for r in rows if lo <= int(r["idx"]) <= hi])

    by_sample = {}
    for sample in sorted({int(r["sample"]) for r in rows}):
        sample_rows = [r for r in rows if int(r["sample"]) == sample]
        seq_len = max(int(r["idx"]) for r in sample_rows) + 1
        mid = seq_len // 2
        by_sample[str(sample)] = {
            "token_count": len(sample_rows),
            "early": _stats([r for r in sample_rows if int(r["idx"]) < mid]),
            "late": _stats([r for r in sample_rows if int(r["idx"]) >= mid]),
        }

    return {
        "source": source,
        "token_count": len(rows),
        "bucket_size": 16,
        "buckets": buckets,
        "corr_response_position_abs_delta_logp": _pearson(
            [int(r["idx"]) for r in rows],
            [abs(float(r["delta_raw_T1"])) for r in rows],
        ),
        "by_sample_early_late": by_sample,
    }


def _token_type_summary(rows: list[dict], source: str) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["token_type"]].append(row)
    return {
        "source": source,
        "overall": _stats(rows),
        "by_token_type": {name: _stats(group) for name, group in sorted(by_type.items())},
    }


def _loss_impact(rows: list[dict], source: str) -> dict:
    # This approximates the PPO old-logprob mismatch impact in probability-ratio
    # space. The true signed PG contribution also depends on advantages, which are
    # not stored in the D0.4 CSV artifacts.
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["token_type"]].append(row)

    groups = {}
    total = len(rows)
    for name, group in sorted(by_type.items()):
        ratios = [float(r["ratio_raw_T1"]) for r in group]
        abs_ratio_error = [abs(x - 1.0) for x in ratios]
        groups[name] = {
            "token_count": len(group),
            "token_fraction": len(group) / total if total else 0.0,
            "mean_ratio": statistics.mean(ratios),
            "median_ratio": statistics.median(ratios),
            "mean_abs_ratio_error": statistics.mean(abs_ratio_error),
            "frac_abs_ratio_minus_1_gt_0p01": sum(x > 0.01 for x in abs_ratio_error) / len(group),
            "frac_abs_ratio_minus_1_gt_0p05": sum(x > 0.05 for x in abs_ratio_error) / len(group),
        }
    action_rows = [r for r in rows if r["token_type"] in ("action_tag", "action_name")]
    return {
        "source": source,
        "note": "Offline proxy only: D0.4 CSV lacks advantages/current-policy logprobs, so signed PG/KL contribution is not available here.",
        "all_response": _stats(rows),
        "action_only_counterfactual_proxy": _stats(action_rows),
        "by_token_type_ratio_error_proxy": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--prefix", default="d0_5")
    parser.add_argument("--out-dir", default="docs/diagnosis")
    args = parser.parse_args()

    csv_path = os.path.join(args.artifact_dir, "d0_4_token_alignment.csv")
    rows = _read_rows(csv_path)
    os.makedirs(args.out_dir, exist_ok=True)

    outputs = {
        f"{args.prefix}_position_buckets.json": _position_summary(rows, args.artifact_dir),
        f"{args.prefix}_token_type_summary.json": _token_type_summary(rows, args.artifact_dir),
        f"{args.prefix}_loss_contribution.json": _loss_impact(rows, args.artifact_dir),
    }
    for name, payload in outputs.items():
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(path)


if __name__ == "__main__":
    main()
