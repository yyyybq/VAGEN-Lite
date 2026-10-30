#!/usr/bin/env python3
"""C1's short run check (the user's decision of 2026-10-30: report the short run before the formal one), the five
items of the design (reports/C1_DESIGN_20261028.md):
    1. both losses fall: the mean of the last 50 logged steps is below that of the first 50 (action loss; text loss
       overall, and per source: trajectories, QA)
    2. lm_head receives a gradient: the text loss's gradient norm on lm_head is above zero at every diagnosed step
       (and the action loss's is zero, as X4 found)
    3. time per step (StarVLA's timing/model, steps 21 on) against the plain run of the same recipe (REFERENCE's
       metrics.jsonl, DS-B-030-s1)
    4. the GPU memory peak (EXP/gpu.csv, sampled every 10 s)
    5. on the language model's last layer, the gradient norm of lambda * text loss over that of the action loss
Writes EXP/cotrain_check.json and prints it; exit 0 when 1 and 2 hold.

    python downstream/train/cotrain_check.py --exp EXP [--reference DS-B-030-s1 experiment dir]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REFERENCE = Path("/mnt/umm/users/yinbaiqiao/downstream_runs/experiments/DS-B-030-s1")


def rows(path: Path) -> list:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "step" in r:
                out.append(r)
    return out


def mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def falls(rs: list, key: str, n: int = 50) -> dict:
    vals = [r[key] for r in rs if key in r]
    first, last = mean(vals[:n]), mean(vals[-n:])
    return {"first": first, "last": last, "falls": bool(last < first), "n": len(vals)}


def step_time(rs: list, skip: int = 20) -> float:
    return mean([r.get("timing/model") for r in rs if r["step"] > skip])


def memory_peak_mib(gpu_csv: Path) -> float:
    peak = 0.0
    if not Path(gpu_csv).is_file():
        return float("nan")
    for row in csv.reader(Path(gpu_csv).read_text().splitlines()):
        if len(row) >= 3:
            try:
                peak = max(peak, float(row[2].strip().split()[0]))
            except ValueError:
                continue
    return peak


def check(exp: Path, reference: Path = REFERENCE) -> dict:
    rs = rows(exp / "run/metrics.jsonl")
    diag = [r for r in rs if "grad_lm_head_text" in r]
    ref_rows = rows(reference / "run/metrics.jsonl") if (reference / "run/metrics.jsonl").is_file() else []
    res = {
        "steps_logged": len(rs),
        "1_losses_fall": {k: falls(rs, k) for k in ("action_dit_loss", "text_loss", "text_loss_traj", "text_loss_qa")},
        "2_lm_head_gradient": {"diagnosed_steps": len(diag),
                               "text_min": min((r["grad_lm_head_text"] for r in diag), default=float("nan")),
                               "text_mean": mean([r["grad_lm_head_text"] for r in diag]),
                               "action_max": max((r["grad_lm_head_action"] for r in diag), default=float("nan"))},
        "3_seconds_per_step": {"cotrain": step_time(rs), "reference": step_time(ref_rows[:2000]) if ref_rows else None,
                               "reference_exp": str(reference)},
        "4_memory_peak_mib": memory_peak_mib(exp / "gpu.csv"),
        "5_last_layer_grad_ratio": {"per_step": [r["grad_last_layer_text"] / r["grad_last_layer_action"]
                                                 for r in diag if r.get("grad_last_layer_action")],
                                    "mean": None},
    }
    ratios = res["5_last_layer_grad_ratio"]["per_step"]
    res["5_last_layer_grad_ratio"]["mean"] = mean(ratios)
    t = res["3_seconds_per_step"]
    if t["reference"]:
        t["slower_by"] = t["cotrain"] / t["reference"] - 1
    ok1 = res["1_losses_fall"]["action_dit_loss"]["falls"] and res["1_losses_fall"]["text_loss"]["falls"]
    ok2 = bool(diag) and res["2_lm_head_gradient"]["text_min"] > 0
    res["passed"] = {"1_losses_fall": bool(ok1), "2_lm_head_gradient": bool(ok2)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    args = ap.parse_args(argv)
    res = check(args.exp, args.reference)
    (args.exp / "cotrain_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if all(res["passed"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
