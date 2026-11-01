"""Reward line (item 2 of 2026-11-01): which way the update pushes each turn position, from the training records.

The drift analysis (reward/drift_s0.py) finds the unreadable answers almost only from the 4th turn of an episode on,
and mostly in failed episodes.  This counts, per run, for the episodes the update trained on (kept_for_update), the
answers by turn position (1st, 2nd-3rd, 4th+) whose episode-level GRPO advantage is positive / zero / negative (the
group's logged returns, std n-1, as verl's grpo; keep's floor scaling ignored: it changes sizes, not signs), over
updates LO..HI.

    python -m reward.late_turn_push --run NAME=RUN_DIR ... --lo 1 --hi 150 --out FILE.json [--workers 4]
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from reward.drift_s0 import answers_of, position


def run_counts(args):
    name, run_dir, lo, hi = args
    out = defaultdict(lambda: {"pos": 0, "zero": 0, "neg": 0})
    groups_kept = 0
    for f in sorted(Path(run_dir, "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem)):
        u = int(f.stem)
        if u < lo or u > hi:
            continue
        by = defaultdict(list)
        for line in f.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by[r.get("task_id")].append(r)
        for eps in by.values():
            if not any(e.get("kept_for_update") for e in eps) or len(eps) < 2:
                continue
            groups_kept += 1
            s = np.array([float(e.get("score") or 0) for e in eps])
            adv = (s - s.mean()) / (s.std(ddof=1) + 1e-6)
            for e, a in zip(eps, adv):
                n = len(answers_of(e.get("output")))
                sign = "pos" if a > 1e-6 else ("neg" if a < -1e-6 else "zero")
                for i in range(n):
                    out[position(i)][sign] += 1
    res = {k: dict(v, total=sum(v.values()), pos_share=round(v["pos"] / max(1, sum(v.values())), 4)) for k, v in sorted(out.items())}
    return name, {"groups_kept": groups_kept, "by_position": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=150)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    runs = [tuple(r.split("=", 1)) + (args.lo, args.hi) for r in args.run]
    with Pool(min(args.workers, len(runs))) as pool:
        res = dict(pool.map(run_counts, runs))
    Path(args.out).write_text(json.dumps({"lo": args.lo, "hi": args.hi, "runs": res}, indent=1))
    for n, r in res.items():
        b = r["by_position"]
        print(n, "groups", r["groups_kept"], {k: (v["pos_share"], v["total"]) for k, v in b.items()})


if __name__ == "__main__":
    main()
