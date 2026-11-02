"""Reward line, phase 2 direction 2 (docs/reward/reports/DESIGN_JSRL_20261102.md, section 7): one look at a guided
chain's runs, from the trainer's console log and verl's rollout dumps (the agent loop's guide record per episode:
guide_eligible, guide_k, guide_actions, guide_turns, guide_stop).

* alert: the unreadable-answer share over the last 10 updates above 1% (reward/tc_watch.py's rule);
* per block of 25 updates:
  - guide share: guided episodes and tasks, guide actions over all primitive actions (n_primitive_steps counts the
    guide's too), the guide's stop reasons;
  - hard tasks (d* >= 6 or equidistance, the guide's own rule): drawn, with at least one success here, and in rw_s0
    and rw_g at the same update (every run of seed 20261019 draws the same tasks);
  - training success here and on the same tasks in rw_s0 / rw_g; unreadable share, entropy, KL (max).

    python -m reward.jsrl_watch --chain RR/runs/<chain> [--out FILE.json]

Prints "STATE runs=<n> blocks=<b> alerts=<a> ended=<e>" first, then the summary JSON; --out also writes FILE.md.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from reward.tc_watch import BLOCK, KEYS, REFERENCE, WINDOW, INVALID, alerts, log_metrics

HARD_D_STAR, HARD_FAMILIES = 6, ("equidistance",)


def episodes(path):
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.append({"task_id": r["task_id"], "success": float(r.get("traj_success") or 0) > 0.5,
                        "steps": int(r.get("n_primitive_steps") or 0), "guide": int(r.get("guide_actions") or 0),
                        "stop": r.get("guide_stop") or ""})
    return out


def by_task(eps):
    out = {}
    for e in eps:
        out[e["task_id"]] = out.get(e["task_id"], 0) + int(e["success"])
    return out


def block(us, per, data, hard):
    eps = [e for u in us for e in data[u]["here"]]
    guided = [e for e in eps if e["guide"] > 0]
    tasks = {(u, e["task_id"]) for u in us for e in data[u]["here"]}
    guided_tasks = {(u, e["task_id"]) for u in us for e in data[u]["here"] if e["guide"] > 0}
    hard_drawn = {(u, t) for u in us for t in data[u]["succ_here"] if t in hard}
    row = {"updates": f"{us[0]}-{us[-1]}",
           "guided_episode_share": round(len(guided) / max(1, len(eps)), 4),
           "guided_task_share": round(len(guided_tasks) / max(1, len(tasks)), 4),
           "guide_action_share": round(sum(e["guide"] for e in eps) / max(1, sum(e["steps"] for e in eps)), 4),
           "guide_stops": dict(Counter(e["stop"] for e in guided if e["stop"])),
           "hard_tasks": len(hard_drawn)}
    for name in ("here", "rw_s0", "rw_g"):
        row[f"hard_with_success_{name}"] = sum(data[u][f"succ_{name}"].get(t, 0) > 0 for u, t in hard_drawn)
    n = 8 * len(tasks)
    for name in ("here", "rw_s0", "rw_g"):
        row[f"success_{name}"] = round(sum(sum(data[u][f"succ_{name}"].values()) for u in us) / max(1, n), 4)
    for key in ("invalid", "entropy"):
        v = [per[u][KEYS[key]] for u in us if KEYS[key] in per.get(u, {})]
        row[key] = round(float(np.mean(v)), 4) if v else None
    kl = [per[u][KEYS["kl"]] for u in us if KEYS["kl"] in per.get(u, {})]
    row["kl_max"] = round(max(kl), 4) if kl else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--rows", default=None, help="the runs' task rows (default: the plan's rl.rows)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    chain = Path(args.chain)
    plan = json.loads((chain / "plan.json").read_text())
    rows_path = Path(args.rows or plan["rl"]["rows"])
    hard = set()
    for line in rows_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if int(r["difficulty"]["d_star"]) >= HARD_D_STAR or r["task_type"] in HARD_FAMILIES:
                hard.add(r["task_id"])
    out, n_blocks, n_alerts, ended = {}, 0, 0, 0
    for run in plan["rl_runs"]:
        base = chain / f"rl_run_{run}"
        per = log_metrics(list((base / "logs").glob("rl-*.log")))
        ended += int((base / "logs" / "rl_exit.json").is_file())
        done = sorted(int(p.stem) for p in (base / "rollout_data").glob("*.jsonl")) if (base / "rollout_data").is_dir() else []
        done = [u for u in done if u in per][:-1]  # the newest dump may still be being written
        data = {}
        for u in done:
            here = episodes(base / "rollout_data" / f"{u}.jsonl")
            d = {"here": here, "succ_here": by_task(here)}
            for name in ("rw_s0", "rw_g"):
                p = REFERENCE[name] / f"{u}.jsonl"
                d[f"succ_{name}"] = by_task(episodes(p)) if p.is_file() else {}
            data[u] = d
        blocks = [block(done[k * BLOCK:(k + 1) * BLOCK], per, data, hard) for k in range(len(done) // BLOCK)]
        rest = done[len(blocks) * BLOCK:]
        n_blocks += len(blocks)
        al = alerts(per)
        n_alerts += len(al)
        inv10 = [per[s][INVALID] for s in sorted(per)[-WINDOW:] if INVALID in per[s]]
        out[run] = {"updates_logged": max(per) if per else 0,
                    "invalid_mean_last10": round(float(np.mean(inv10)), 4) if inv10 else None, "alerts": al,
                    "blocks": blocks, "open_block": block(rest, per, data, hard) if rest else None}
    print(f"STATE runs={len(out)} blocks={n_blocks} alerts={n_alerts} ended={ended}")
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        cols = ["updates", "guided_episode_share", "guided_task_share", "guide_action_share", "hard_tasks",
                "hard_with_success_here", "hard_with_success_rw_s0", "hard_with_success_rw_g", "success_here",
                "success_rw_s0", "success_rw_g", "invalid", "entropy", "kl_max", "guide_stops"]
        L = []
        for run, v in out.items():
            L += [f"## {run}", "", "| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
            rows = v["blocks"] + ([dict(v["open_block"], updates=v["open_block"]["updates"] + " (open)")]
                                  if v["open_block"] else [])
            L += ["| " + " | ".join(str(b.get(c)) for c in cols) + " |" for b in rows]
            L.append("")
        Path(args.out).with_suffix(".md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
