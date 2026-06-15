#!/usr/bin/env python3
"""
Parse validation/{step}.jsonl files for an experiment and split metrics by:
  - in-domain (first test_size * VAL_N records)
  - OOD       (next ood_n_envs * VAL_N records, if OOD val is enabled)

Usage:
    python parse_val_ood.py <exp_dir> [--test_size 19] [--ood_n 19] [--val_n 4]

Example:
    python parse_val_ood.py exps/vagen_active_spatial/v22_cosine_no_farm
"""
import argparse
import json
import sys
from pathlib import Path


def compute_metrics(records, n_per_prompt):
    """records: list of dicts with 'traj_success' and 'score'.
       Group every n_per_prompt records into one prompt.
       Returns: (tsucc_b4, tsucc_mean, score_best, score_mean, n_prompts)
    """
    if not records:
        return None
    n = len(records) // n_per_prompt
    succ_best, succ_mean, sc_best, sc_mean = [], [], [], []
    for i in range(n):
        group = records[i * n_per_prompt : (i + 1) * n_per_prompt]
        ts = [float(r.get("traj_success", 0)) for r in group]
        sc = [float(r.get("score", 0)) for r in group]
        succ_best.append(max(ts))
        succ_mean.append(sum(ts) / len(ts))
        sc_best.append(max(sc))
        sc_mean.append(sum(sc) / len(sc))
    return {
        "tsucc_b4": sum(succ_best) / n,
        "tsucc_m4": sum(succ_mean) / n,
        "score_b4": sum(sc_best) / n,
        "score_m4": sum(sc_mean) / n,
        "n_prompts": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=str)
    ap.add_argument("--test_size", type=int, default=19, help="in-domain val n_envs")
    ap.add_argument("--ood_n", type=int, default=19, help="OOD val n_envs (0 = disabled)")
    ap.add_argument("--val_n", type=int, default=4, help="rollouts per prompt")
    args = ap.parse_args()

    val_dir = Path(args.exp_dir) / "validation"
    if not val_dir.exists():
        print(f"ERROR: {val_dir} not found", file=sys.stderr)
        sys.exit(1)

    files = sorted(val_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not files:
        print(f"No val files in {val_dir}")
        return

    n_id = args.test_size * args.val_n
    n_ood = args.ood_n * args.val_n

    header = f"{'step':>6}  {'ID tsucc_b4':>11}  {'ID m@4':>7}  {'ID sc_b4':>9}"
    if args.ood_n > 0:
        header += f"   |  {'OOD tsucc_b4':>12}  {'OOD m@4':>8}  {'OOD sc_b4':>10}"
    print(header)
    print("-" * len(header))

    for f in files:
        step = int(f.stem)
        records = [json.loads(l) for l in f.open()]
        id_recs = records[:n_id]
        ood_recs = records[n_id : n_id + n_ood] if args.ood_n > 0 else []

        id_m = compute_metrics(id_recs, args.val_n)
        line = f"{step:>6}  {id_m['tsucc_b4']:>11.4f}  {id_m['tsucc_m4']:>7.4f}  {id_m['score_b4']:>9.3f}"
        if args.ood_n > 0:
            ood_m = compute_metrics(ood_recs, args.val_n)
            if ood_m is None or ood_m["n_prompts"] == 0:
                line += f"   |  {'N/A':>12}"
            else:
                line += (
                    f"   |  {ood_m['tsucc_b4']:>12.4f}  {ood_m['tsucc_m4']:>8.4f}  "
                    f"{ood_m['score_b4']:>10.3f}"
                )
        print(line)


if __name__ == "__main__":
    main()
