"""Reward line, phase 2 direction 3 round 1 (PREREG_DIR3.md sections 4 and 5): one look at a statement chain's runs
(rw_st0, rw_st1), from the trainer's console log and verl's rollout dumps (rollout_data/<update>.jsonl), as
reward/tc_watch.py does for direction 1.

* alert: the unreadable-answer share averaged over the last 10 updates above 1%, each time it crosses from below
  (the user: report at once, do not stop; the chain's own stop stays at 10%);
* per block of 25 updates: unreadable share, entropy, KL (max), training success, and on the same tasks of the same
  update the success of rw_s0 / rw_g (seed 20261019 draws the same tasks per update; their start was C0 e8, not e5);
  the share of tasks with both successes and failures;
* the statement fields' episode means (statement_* of the env's statement_record: readable, accuracy, where / face /
  dist / missing; the accuracy also over successful and over failed episodes; per turn and per turn index only the
  evaluations have them, reward/statement_eval.py);
* actions per turn (n_primitive_steps / n_turns over the block's episodes; rw_tc1 drifted to more actions per turn);
  rw_st1 also the estimator's per-turn statement accuracy over the update (turn_credit/value_mean) and the statement
  share of the answer tokens (turn_credit/span_token_share).

    python -m reward.stmt_watch --chain RR/runs/rw_p2_st1 [--out FILE.json]

Prints one line "STATE runs=<n> blocks=<b> alerts=<a> ended=<e>" first (for the local watcher), then the summary
JSON; --out also writes it (and FILE.md, the blocks as a table).  Caches each update's sums in
CHAIN/stmt_watch_cache.json (redone when the dump's size or time changes).  CPU, one process.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from reward.tc_watch import BLOCK, INVALID, REFERENCE, WINDOW, alerts, log_metrics

FIELDS = ("readable", "accuracy", "where", "face", "dist", "missing")
KEYS = {"invalid": INVALID, "entropy": "actor/entropy", "turn_accuracy": "turn_credit/value_mean",
        "statement_token_share": "turn_credit/span_token_share", "seconds": "timing_s/step",
        "grad_norm": "actor/grad_norm"}


def update_sums(path):
    """One update's dump: per task [successes, episodes]; sums and counts per statement field and of the accuracy
    over successful / failed episodes; actions and turns."""
    s = {"tasks": {}, "actions": 0, "turns": 0, **{f: [0.0, 0] for f in FIELDS + ("acc_success", "acc_fail")}}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ok = float(r.get("traj_success") or 0) > 0.5
        t = s["tasks"].setdefault(r["task_id"], [0, 0])
        t[0] += int(ok)
        t[1] += 1
        s["actions"] += int(r.get("n_primitive_steps") or 0)
        s["turns"] += int(r.get("n_turns") or 0)
        for f in FIELDS:
            if r.get("statement_" + f) is not None:
                s[f][0] += float(r["statement_" + f])
                s[f][1] += 1
        if r.get("statement_accuracy") is not None:
            k = "acc_success" if ok else "acc_fail"
            s[k][0] += float(r["statement_accuracy"])
            s[k][1] += 1
    return s


def cached(cache, key, path):
    st = path.stat()
    stamp = [st.st_size, int(st.st_mtime)]
    if (cache.get(key) or {}).get("stamp") != stamp:
        cache[key] = {"stamp": stamp, "sums": update_sums(path)}
    return cache[key]["sums"]


def block_row(per, us, sums, refs):
    def mean(key):
        v = [per[u][KEYS[key]] for u in us if KEYS[key] in per.get(u, {})]
        return round(float(np.mean(v)), 4) if v else None

    kl = [per[u]["actor/kl_loss"] for u in us if "actor/kl_loss" in per.get(u, {})]
    S = [sums[u] for u in us]
    tasks = [t for s in S for t in s["tasks"].values()]
    n = sum(t[1] for t in tasks)
    row = {"updates": f"{us[0]}-{us[-1]}", **{k: mean(k) for k in KEYS}, "kl_max": round(max(kl), 4) if kl else None,
           "success": round(sum(t[0] for t in tasks) / n, 4) if n else None,
           "mixed_share": round(float(np.mean([0 < t[0] < t[1] for t in tasks])), 4) if tasks else None}
    for f in FIELDS + ("acc_success", "acc_fail"):
        tot, cnt = sum(s[f][0] for s in S), sum(s[f][1] for s in S)
        row[("statement_" + f) if f in FIELDS else f] = round(tot / cnt, 4) if cnt else None
    turns = sum(s["turns"] for s in S)
    row["actions_per_turn"] = round(sum(s["actions"] for s in S) / turns, 3) if turns else None
    # the same tasks of the same updates: successes here, in rw_s0 and in rw_g, over their episodes
    same = np.zeros(4)
    for u in us:
        r = refs.get(u)
        if not r:
            continue
        common = set(sums[u]["tasks"]) & set(r["rw_s0"]["tasks"]) & set(r["rw_g"]["tasks"])
        for i, d in enumerate((sums[u]["tasks"], r["rw_s0"]["tasks"], r["rw_g"]["tasks"])):
            same[i] += sum(d[t][0] for t in common)
        same[3] += sum(sums[u]["tasks"][t][1] for t in common)
    if same[3]:
        row.update({"succ_same_tasks": round(same[0] / same[3], 4), "succ_rw_s0": round(same[1] / same[3], 4),
                    "succ_rw_g": round(same[2] / same[3], 4)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="default CHAIN/stmt_watch_cache.json")
    args = ap.parse_args()
    chain = Path(args.chain)
    plan = json.loads((chain / "plan.json").read_text())
    cache_path = Path(args.cache) if args.cache else chain / "stmt_watch_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    out, n_blocks, n_alerts, ended = {}, 0, 0, 0
    for run in plan["rl_runs"]:
        base = chain / f"rl_run_{run}"
        per = log_metrics(list((base / "logs").glob("rl-*.log")))
        finished = (base / "logs" / "rl_exit.json").is_file()
        ended += int(finished)
        rd = base / "rollout_data"
        done = sorted(int(p.stem) for p in rd.glob("*.jsonl")) if rd.is_dir() else []
        done = [u for u in done if u in per]  # both the dump and the log line written
        # the newest dump may still be being written, unless the trainer has exited
        done = done if finished else (done[:-1] if len(done) > 1 else [])
        sums, refs, differ = {}, {}, []
        for u in done:
            sums[u] = cached(cache, f"{run}/{u}", rd / f"{u}.jsonl")
            ref = {k: REFERENCE[k] / f"{u}.jsonl" for k in ("rw_s0", "rw_g")}
            if all(p.is_file() for p in ref.values()):
                refs[u] = {k: cached(cache, f"{k}/{u}", p) for k, p in ref.items()}
                if set(refs[u]["rw_s0"]["tasks"]) != set(sums[u]["tasks"]):
                    differ.append(u)
        blocks = [block_row(per, done[k * BLOCK:(k + 1) * BLOCK], sums, refs) for k in range(len(done) // BLOCK)]
        n_blocks += len(blocks)
        al = alerts(per)
        n_alerts += len(al)
        rest = done[len(blocks) * BLOCK:]
        inv10 = [per[s][INVALID] for s in sorted(per)[-WINDOW:] if INVALID in per[s]]
        out[run] = {"updates_logged": max(per) if per else 0,
                    "invalid_mean_last10": round(float(np.mean(inv10)), 4) if inv10 else None,
                    "alerts": al, "blocks": blocks, "open_block": block_row(per, rest, sums, refs) if rest else None,
                    "tasks_differ_from_rw_s0": differ}
    cache_path.write_text(json.dumps(cache))
    print(f"STATE runs={len(out)} blocks={n_blocks} alerts={n_alerts} ended={ended}")
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        cols = ["updates", "invalid", "entropy", "kl_max", "success", "succ_same_tasks", "succ_rw_s0", "succ_rw_g",
                "mixed_share", "statement_readable", "statement_accuracy", "acc_success", "acc_fail",
                "statement_where", "statement_face", "statement_dist", "statement_missing", "actions_per_turn",
                "turn_accuracy", "statement_token_share", "seconds"]
        L = []
        for run, v in out.items():
            L += [f"## {run}", "", "| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
            L += ["| " + " | ".join(str(b.get(c)) for c in cols) + " |"
                  for b in v["blocks"] + ([dict(v["open_block"], updates=v["open_block"]["updates"] + " (open)")]
                                          if v["open_block"] else [])]
            L.append("")
        Path(args.out).with_suffix(".md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
