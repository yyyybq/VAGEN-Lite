#!/usr/bin/env python3
"""Summarise the step-2 closed-loop runs (supplementary requirements 2 and 3, success criterion 3).

    python downstream/step2/analyze_eval.py EXP/step2/eval2 --out eval_summary.json

Per parallelism P (eval_P<P>/): wall time, episodes per hour, mean episode / reset / per-step time, policy latency,
GPU utilisation of the policy server (gpu_util_P<P>.csv, sampled every 2 s); whether every (task, seed) gives the
same initial-state fingerprint and outcome in all runs; and the success rate on the 15 episodes of the step-2 plan
(seeds 1000-1004 of each task).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def gpu_util(path: Path):
    values = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[2].endswith("%"):
            values.append(float(parts[2].rstrip("%").strip()))
    busy = [v for v in values if v > 0]
    return {"samples": len(values), "mean_percent": round(statistics.mean(values), 1) if values else None,
            "share_of_samples_busy": round(len(busy) / len(values), 3) if values else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    runs, by_key = {}, {}
    for d in sorted((p for p in args.eval_dir.glob("eval_P*") if p.is_dir()), key=lambda p: int(p.name[6:])):
        if not d.is_dir():
            continue
        p = int(d.name[6:])
        rows = [json.loads(x) for x in (d / "episodes.jsonl").read_text().splitlines() if x.strip()]
        summary = json.loads((d / "summary.json").read_text())
        done = [r for r in rows if r.get("status") == "complete"]
        runs[p] = {
            "episodes": len(rows), "complete": len(done), "successes": summary["successes"],
            "wall_seconds": summary["wall_seconds"], "episodes_per_hour": summary["episodes_per_hour"],
            "episode_wall_mean": summary["episode_wall_mean_seconds"], "reset_mean": summary["reset_mean_seconds"],
            "steps_mean": summary["steps_mean"], "seconds_per_step_mean": summary["seconds_per_step_mean"],
            "policy_latency_mean": round(statistics.mean(r["inference_mean_seconds"] for r in done), 4) if done else None,
            "gpu": gpu_util(args.eval_dir / f"gpu_util_P{p}.csv")}
        for r in done:
            by_key.setdefault((r["task"], r["seed"]), {})[p] = (r["init"], r["success"], r["steps"])
    consistent = {f"{t}/{s}": len({json.dumps(v[0], sort_keys=True) for v in per.values()}) == 1
                  and len({v[1] for v in per.values()}) == 1 for (t, s), per in by_key.items()}
    steps_identical = all(len({v[2] for v in per.values()}) == 1 for per in by_key.values())
    first = min(runs) if runs else None
    plan15 = {}
    if first is not None:
        for (t, s), per in by_key.items():
            if 1000 <= s <= 1004:
                st = plan15.setdefault(t, [0, 0])
                st[0] += int(per[first][1])
                st[1] += 1
    out = {"runs": runs, "same_init_and_outcome_in_every_run": all(consistent.values()) if consistent else None,
           "same_step_count_in_every_run": steps_identical,
           "pairs_checked": len(consistent),
           "plan15": {t: f"{a}/{n}" for t, (a, n) in plan15.items()},
           "plan15_total": f"{sum(a for a, _ in plan15.values())}/{sum(n for _, n in plan15.values())}"}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
