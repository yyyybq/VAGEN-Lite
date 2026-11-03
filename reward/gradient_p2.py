"""Reward line, phase 2 report (the brief's section 3: every report gives the unique-signal share and the tasks and
turns per update in the gradient): per run, from verl's rollout dumps (kept_for_update, traj_success, n_turns per
episode) and, for the per-turn-credit runs, the trainer's log (turn_credit/tied_groups, turn_credit/turn_only_share).

Per update, then averaged over blocks of 25 and over the whole run:

* tasks in the gradient: groups with an episode kept for the update; episodes and turns (n_turns) of those;
* unique-signal tasks: kept tasks the S0 comparison gives nothing, i.e. whose 8 episodes all succeed or all fail
  (rw_g: the all-fail tasks keep refills; the per-turn runs: the groups whose S0 returns are all equal, from the log;
  rw_s0: none by construction), and their share of the tasks in the gradient;
* per-turn runs only: the share of gradient turns whose only signal is the turn term (episode term 0).

    python -m reward.gradient_p2 --run NAME=RL_RUN_DIR [--run ...] --out FILE.json [--last 150]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from reward.tc_watch import log_metrics


def per_update(path):
    groups = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            groups.setdefault(r["task_id"], []).append(r)
    kept = {t: eps for t, eps in groups.items() if any(bool(e.get("kept_for_update", True)) for e in eps)}
    ke = [e for eps in kept.values() for e in eps if bool(e.get("kept_for_update", True))]
    same = sum(len({float(e.get("traj_success") or 0) > 0.5 for e in eps}) == 1 for eps in kept.values())
    return {"tasks": len(kept), "episodes": len(ke), "turns": sum(int(e.get("n_turns") or 0) for e in ke),
            "same_outcome_tasks": same}


def summarize(rows):
    keys = [k for k in rows[0] if k != "update"]
    return {k: round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 4)
            for k in keys if any(r.get(k) is not None for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=RL_RUN_DIR (rl_run_<name>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--last", type=int, default=150)
    args = ap.parse_args()
    out = {}
    for spec in args.run:
        name, d = spec.split("=", 1)
        d = Path(d)
        per = log_metrics(list((d / "logs").glob("rl-*.log")))
        rows = []
        for u in range(1, args.last + 1):
            p = d / "rollout_data" / f"{u}.jsonl"
            if not p.is_file():
                continue
            r = {"update": u, **per_update(p)}
            m = per.get(u, {})
            tied = m.get("turn_credit/tied_groups")
            if tied is not None:  # per-turn credit: every task is in the gradient, S0-equal returns only via T
                r["unique_tasks"] = tied
                r["turn_only_turn_share"] = m.get("turn_credit/turn_only_share")
            else:  # GRPO: a kept task with one outcome is a keep fill (rw_g); none for S0's mixed filter
                r["unique_tasks"] = r["same_outcome_tasks"]
            r["unique_task_share"] = r["unique_tasks"] / r["tasks"] if r["tasks"] else None
            rows.append(r)
        blocks = {f"{k * 25 + 1}-{(k + 1) * 25}": summarize([r for r in rows if k * 25 < r["update"] <= (k + 1) * 25])
                  for k in range(args.last // 25) if any(k * 25 < r["update"] <= (k + 1) * 25 for r in rows)}
        out[name] = {"updates": len(rows), "all": summarize(rows) if rows else None, "blocks": blocks}
    Path(args.out).write_text(json.dumps(out, indent=1))
    L = ["| 运行 | 每次更新进梯度的题 | 局 | 回合 | 只有新信号的题 | 占进梯度的题 | 只有逐回合项的回合占比 |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in out.items():
        a = v["all"] or {}
        L.append(f"| {name} | {a.get('tasks')} | {a.get('episodes')} | {a.get('turns')} | {a.get('unique_tasks')} | "
                 f"{a.get('unique_task_share')} | {a.get('turn_only_turn_share', '—')} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
