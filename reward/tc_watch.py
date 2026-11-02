"""Reward line, phase 2 first round (the user's guards of 2026-11-02): one look at a direction-1 chain's runs, from the
trainer's console log ("step:N - key:value - ...") and verl's rollout dumps (rollout_data/<update>.jsonl).

* alert: the unreadable-answer share (transition/invalid_action_rate/mean) averaged over the last 10 updates above
  1%, each time it crosses from below (the user: report at once, do not stop; the chain's own stop stays at 10%);
* per block of 25 updates: unreadable share, entropy, KL (max), training success; the turn term's share of the
  advantage magnitude, the positive share of turns 4+ (turn_credit/late_pos_share), the sign flips, groups with equal
  returns; over the first 50 updates also both terms' 10th / 90th percentiles (the brief's two distributions);
* "原全失败题出成功": every update draws the same 24 tasks in every run of seed 20261019 (rw_s0, rw_g and these runs; the
  task order is the data seed's), so per update the tasks whose 8 episodes all failed in rw_s0 at that update, and
  of those how many have at least one success here, and in rw_g (the V4 run) as the reference.

    python -m reward.tc_watch --chain RR/runs/rw_p2_r1 [--out FILE.json]

Prints one line "STATE runs=<n> blocks=<b> alerts=<a> ended=<e>" first (for the local watcher), then the summary
JSON; --out also writes it (and FILE.md, the blocks as a table).  Caches each update's per-task successes in
CHAIN/tc_watch_cache.json.  CPU, one process.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

RR = Path("/mnt/umm/users/yinbaiqiao/reward_runs")
REFERENCE = {"rw_s0": RR / "runs/rw_b_w1/rl_run_rw_s0/rollout_data", "rw_g": RR / "runs/rw_b_w2/rl_run_rw_g/rollout_data"}
METRIC = re.compile(r"([\w./@-]+):(?:np\.float64\()?(-?[0-9.]+(?:e-?[0-9]+)?)\)?")
INVALID = "transition/invalid_action_rate/mean"
ALERT, WINDOW, BLOCK = 0.01, 10, 25
KEYS = {"invalid": INVALID, "entropy": "actor/entropy", "kl": "actor/kl_loss", "success": "episode/success_rate",
        "turn_share": "turn_credit/turn_magnitude_share", "late_pos": "turn_credit/late_pos_share",
        "flip": "turn_credit/sign_flip_share", "tied": "turn_credit/tied_groups",
        "turn_p10": "turn_credit/turn_term_p10", "turn_p90": "turn_credit/turn_term_p90",
        "episode_p10": "turn_credit/episode_term_p10", "episode_p90": "turn_credit/episode_term_p90",
        "miss": "turn_credit/table_miss_share", "seconds": "timing_s/step", "grad_norm": "actor/grad_norm"}


def log_metrics(logs):
    per = {}
    for p in sorted(logs):  # attempts in time order; a later attempt's line for an update wins
        for line in p.read_text(errors="replace").splitlines():
            m = re.search(r"step:(\d+) - (.*)", line)
            if m and int(m.group(1)) > 0:
                per[int(m.group(1))] = {k: float(v) for k, v in METRIC.findall(m.group(2))}
    return per


def successes(path):
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["task_id"]] = out.get(r["task_id"], 0) + int(float(r.get("traj_success") or 0) > 0.5)
    return out


def cached(cache, key, path):
    if key not in cache:
        cache[key] = successes(path)
    return cache[key]


def alerts(per):
    steps = sorted(s for s in per if INVALID in per[s])
    out, above = [], False
    for i, s in enumerate(steps):
        win = [per[x][INVALID] for x in steps[max(0, i - WINDOW + 1):i + 1]]
        mean = sum(win) / len(win)
        now = len(win) == WINDOW and mean > ALERT
        if now and not above:
            out.append({"update": s, "mean_10": round(mean, 4)})
        above = now
    return out


def block_row(per, us, fails):
    def mean(key):
        v = [per[u][KEYS[key]] for u in us if KEYS[key] in per.get(u, {})]
        return round(float(np.mean(v)), 4) if v else None
    kl = [per[u][KEYS["kl"]] for u in us if KEYS["kl"] in per.get(u, {})]
    row = {"updates": f"{us[0]}-{us[-1]}", **{k: mean(k) for k in KEYS if k != "kl"},
           "kl_max": round(max(kl), 4) if kl else None}
    row["s0_all_fail_tasks"] = sum(f["s0_all_fail"] for f in fails)
    row["of_them_success_here"] = sum(f["here"] for f in fails)
    row["of_them_success_rw_g"] = sum(f["rw_g"] for f in fails)
    return row


def first50(per):
    """The brief's two distributions over updates 1-50: each update's 10th / 90th percentile of both terms (min of
    the 10th, max of the 90th, and their means), the magnitude share and the sign flips."""
    us = [u for u in sorted(per) if u <= 50]
    out = {"updates": len(us)}
    for key in ("turn_p10", "episode_p10", "turn_p90", "episode_p90", "turn_share", "flip"):
        v = [per[u][KEYS[key]] for u in us if KEYS[key] in per[u]]
        if v:
            out[key] = {"mean": round(float(np.mean(v)), 4), "min": round(float(min(v)), 4),
                        "max": round(float(max(v)), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="default CHAIN/tc_watch_cache.json")
    args = ap.parse_args()
    chain = Path(args.chain)
    plan = json.loads((chain / "plan.json").read_text())
    cache_path = Path(args.cache) if args.cache else chain / "tc_watch_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    out, n_blocks, n_alerts, ended = {}, 0, 0, 0
    for run in plan["rl_runs"]:
        base = chain / f"rl_run_{run}"
        per = log_metrics(list((base / "logs").glob("rl-*.log")))
        ended += int((base / "logs" / "rl_exit.json").is_file())
        done = sorted(int(p.stem) for p in (base / "rollout_data").glob("*.jsonl")) if (base / "rollout_data").is_dir() else []
        done = [u for u in done if u in per]  # both the dump and the log line written
        fails = {}
        for u in done[:-1] if len(done) > 1 else []:  # the newest dump may still be being written
            s0p, gp = REFERENCE["rw_s0"] / f"{u}.jsonl", REFERENCE["rw_g"] / f"{u}.jsonl"
            if not (s0p.is_file() and gp.is_file()):
                continue
            s0 = cached(cache, f"rw_s0/{u}", s0p)
            g = cached(cache, f"rw_g/{u}", gp)
            here = cached(cache, f"{run}/{u}", base / "rollout_data" / f"{u}.jsonl")
            af = [t for t, n in s0.items() if n == 0]
            fails[u] = {"s0_all_fail": len(af), "here": sum(here.get(t, 0) > 0 for t in af),
                        "rw_g": sum(g.get(t, 0) > 0 for t in af), "same_tasks": set(s0) == set(here)}
        blocks = []
        full = [u for u in sorted(fails)]
        for k in range(0, len(full) // BLOCK):
            us = full[k * BLOCK:(k + 1) * BLOCK]
            blocks.append(block_row(per, us, [fails[u] for u in us]))
        n_blocks += len(blocks)
        al = alerts(per)
        n_alerts += len(al)
        last = max(per) if per else 0
        inv10 = [per[s][INVALID] for s in sorted(per)[-WINDOW:] if INVALID in per[s]]
        out[run] = {"updates_logged": last, "invalid_mean_last10": round(float(np.mean(inv10)), 4) if inv10 else None,
                    "alerts": al, "blocks": blocks,
                    "tasks_differ_from_rw_s0": [u for u, f in fails.items() if not f["same_tasks"]],
                    "first50_distributions": first50(per)}
    cache_path.write_text(json.dumps(cache))
    print(f"STATE runs={len(out)} blocks={n_blocks} alerts={n_alerts} ended={ended}")
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        cols = ["updates", "invalid", "entropy", "kl_max", "success", "turn_share", "late_pos", "flip", "tied",
                "s0_all_fail_tasks", "of_them_success_here", "of_them_success_rw_g", "seconds"]
        L = []
        for run, v in out.items():
            L += [f"## {run}", "", "| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
            L += ["| " + " | ".join(str(b.get(c)) for c in cols) + " |" for b in v["blocks"]]
            L.append("")
        Path(args.out).with_suffix(".md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
