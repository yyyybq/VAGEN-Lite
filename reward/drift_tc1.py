"""Reward line, phase 2 direction 1 (the offline check of docs/reward/reports/ROUND1_P2_20261103.md section 6, the
user's decision 1甲 of 2026-11-03): why rw_tc1 (E + 0.5 T) came to pack more actions into a turn and rw_tc2 (0.5 T
alone) did not.  CPU only.

Every EVERY-th update of each run is replayed without rendering (reward.turn_credit_study.worker: the S0 env with
progress_version optimal and the depth-12 tables, per turn the progress score before and after and the S0 reward), and
both estimators of vagen/turn_credit.py are recomputed on the whole update: "tc1" (episode_weight 1, turn_weight 0.5)
and "tc2" (0, 0.5), each on every run's records, so each run is also seen under the other's estimator.  Per turn: E
(the episode term, from the logged return), T (the turn term), the answer's actions (its <action> list) and tokens
(the run's tokenizer, the answer plus <|im_end|> as the turn's span; the token-mean loss weighs every token alike).

Check: each run's own estimator (tc2 for a run whose name has "tc2", else tc1) against the trainer's log, per update (turn_credit/tied_groups, turn_magnitude_share,
late_pos_share, pos_share), and the replayed S0 return against the logged one.

Per run, estimator and block of 25 updates, by the actions in the turn (0 = unreadable, 1..5, 6+):
* turns and their share of the answer tokens;
* the mean advantage per token of the episode part, of the turn part and of their sum: what the update pushes on a
  turn of that size (a mean that rises with the size pushes towards more actions a turn); and the sum's mean per
  turn (each turn has one closing decision);
* the sum's mean per token in successful and in failed episodes;
and per run and block, over the groups with both outcomes: actions per turn and turns per episode of the successful
minus the failed episodes (what the episode term rewards).

    python -m reward.drift_tc1 --run NAME=RL_RUN_DIR [--run ...] --s0-yaml TRAIN_ENV.yaml --tables DIR --rows ROWS
        --tokenizer DIR --out OUT [--every 3] [--max-update N] [--workers 4]

OUT/summary.json, OUT/summary.md, OUT/turns.jsonl.gz (the replay, one line per run, update and task).
"""

import argparse
import gzip
import json
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from reward.replay_q3 import answers_of
from reward.tc_watch import BLOCK, log_metrics
from reward.turn_credit_study import worker

ESTIMATORS = {"tc1": {"episode_weight": 1.0, "turn_weight": 0.5}, "tc2": {"episode_weight": 0.0, "turn_weight": 0.5}}
BUCKETS = ("0", "1", "2", "3", "4", "5", "6+")
CHECK = ("turn_credit/tied_groups", "turn_credit/turn_magnitude_share", "turn_credit/late_pos_share",
         "turn_credit/pos_share")
ACTION = re.compile(r"<action>(.*?)</action>", re.S)


def n_actions(answer):
    m = ACTION.search(answer)
    return len([a for a in m.group(1).split("|") if a.strip()]) if m else 0


def bucket(k):
    return str(k) if k <= 5 else "6+"


def work(job):
    run, args = job
    import torch

    torch.set_num_threads(1)
    return run, worker(args)


def estimate(groups, cfg):
    """Per episode (in the groups' order) E and per turn T of one update under one estimator; and the update's
    metrics as the trainer logs them."""
    from vagen.turn_credit import combine_metrics, episode_terms, turn_advantages

    records, uids, returns = [], [], []
    for g in groups:
        for e in g["episodes"]:
            records.append({"phi0": e["turns"][0]["phi"] if e["turns"] else 0.0,
                            "turns": [{"phi_after": t["phi_next"], "success": t["kind"] == "success",
                                       "d_after": t["d_next"]} for t in e["turns"]]})
            uids.append(g["task_id"])
            returns.append(e["score"])
    z, m = turn_advantages(records, uids, cfg)
    ep, m2 = episode_terms(returns, uids, cfg)
    m.update(m2)
    m.update(combine_metrics(z, ep, cfg))
    return ep, z, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=RL_RUN_DIR (rl_run_<name>)")
    ap.add_argument("--s0-yaml", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--max-update", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer

    from vagen.turn_credit import settings

    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n_tokens = {}

    def tokens(a):
        if a not in n_tokens:
            n_tokens[a] = len(tok.encode(a, add_special_tokens=False)) + 1  # + <|im_end|>
        return n_tokens[a]

    rows = {}
    for line in Path(args.rows).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["task_id"]] = r
    runs = dict(spec.split("=", 1) for spec in args.run)
    out = Path(args.out)
    work_dir = out / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs, answers = [], {}
    for run, d in runs.items():
        files = sorted(Path(d, "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem))
        files = [f for f in files if args.max_update is None or int(f.stem) <= args.max_update][::args.every]
        groups = []
        for f in files:
            by = defaultdict(list)
            for line in f.read_text(errors="ignore").splitlines():
                if line.strip():
                    r = json.loads(line)
                    by[r["task_id"]].append(r)
            for tid, eps in sorted(by.items()):
                u = int(f.stem)
                answers[(run, u, tid)] = [{"answers": answers_of(e["output"]), "score": float(e["score"])} for e in eps]
                groups.append({"update": u, "row": rows[tid],
                               "episodes": [{"answers": a["answers"], "kept": bool(e.get("kept_for_update")),
                                             "success": bool(float(e.get("traj_success") or 0))}
                                            for a, e in zip(answers[(run, u, tid)], eps)]})
        for k in range(0, len(groups), 20):
            p = work_dir / f"{run}_{k:06d}.jsonl"
            p.write_text("".join(json.dumps(g) + "\n" for g in groups[k:k + 20]))
            jobs.append((run, (str(p), args.s0_yaml, args.tables)))
    replay = defaultdict(lambda: defaultdict(list))  # run -> update -> groups
    errors = defaultdict(int)
    with Pool(args.workers) as pool:
        for run, part in pool.imap_unordered(work, jobs):
            for g in part:
                if "error" in g:
                    errors[run] += 1
                    continue
                for e, a in zip(g["episodes"], answers[(run, g["update"], g["task_id"])]):
                    e["score"] = a["score"]
                    for t, ans in zip(e["turns"], a["answers"]):
                        t["actions"], t["tokens"] = n_actions(ans), tokens(ans)
                replay[run][g["update"]].append(g)
    with gzip.open(out / "turns.jsonl.gz", "wt") as f:
        for run in replay:
            for u in sorted(replay[run]):
                for g in sorted(replay[run][u], key=lambda g: g["task_id"]):
                    f.write(json.dumps({"run": run, **g}) + "\n")

    cfgs = {name: settings({"turn_credit": w}) for name, w in ESTIMATORS.items()}
    summary = {"runs": runs, "every": args.every, "errors": dict(errors), "estimators": ESTIMATORS}
    md = []
    for run, d in runs.items():
        logged = log_metrics(list(Path(d, "logs").glob("rl-*.log")))
        own = "tc2" if "tc2" in run else "tc1"
        check = defaultdict(list)
        return_mismatch = episodes = 0
        acc = defaultdict(lambda: np.zeros(6))  # (est, block, bucket) -> turns, tokens, token-weighted e, t, sum; sum
        outc = defaultdict(lambda: np.zeros(4))  # (est, block, bucket, outcome) -> tokens, e, t, sum
        link = defaultdict(list)  # block -> (d actions per turn, d turns per episode) per mixed group
        for u in sorted(replay[run]):
            groups = sorted(replay[run][u], key=lambda g: g["task_id"])
            blk = f"{(u - 1) // BLOCK * BLOCK + 1}-{((u - 1) // BLOCK + 1) * BLOCK}"
            for g in groups:
                for e in g["episodes"]:
                    episodes += 1
                    return_mismatch += int(abs(sum(t["r"] for t in e["turns"]) - e["score"]) > 1e-3)
                succ = [e for e in g["episodes"] if e["success"]]
                fail = [e for e in g["episodes"] if not e["success"]]
                if succ and fail:
                    def apt(es):
                        return float(np.mean([np.mean([t["actions"] for t in e["turns"]]) for e in es if e["turns"]]))
                    link[blk].append((apt(succ) - apt(fail),
                                      float(np.mean([len(e["turns"]) for e in succ]) - np.mean([len(e["turns"]) for e in fail]))))
            for est, cfg in cfgs.items():
                ep, z, m = estimate(groups, cfg)
                if est == own and u in logged:
                    for k in CHECK:
                        if k in logged[u] and k in m:
                            check[k].append(abs(logged[u][k] - m[k]))
                i = 0
                for g in groups:
                    for e in g["episodes"]:
                        we, wt = cfg["episode_weight"] * ep[i], [cfg["turn_weight"] * x for x in z[i]]
                        for t, tt in zip(e["turns"], wt):
                            b = bucket(t["actions"])
                            n = t["tokens"]
                            acc[(est, blk, b)] += [1, n, n * we, n * tt, n * (we + tt), we + tt]
                            outc[(est, blk, b, "success" if e["success"] else "fail")] += [n, n * we, n * tt, n * (we + tt)]
                        i += 1
        res = {"own_estimator": own, "episodes": episodes, "return_mismatch": return_mismatch,
               "check_vs_log": {k: {"updates": len(v), "max_abs_diff": round(max(v), 6), "mean_abs_diff": round(float(np.mean(v)), 6)}
                                for k, v in check.items()},
               "success_minus_fail": {b: {"groups": len(v), "actions_per_turn": round(float(np.mean([x[0] for x in v])), 3),
                                          "turns_per_episode": round(float(np.mean([x[1] for x in v])), 3)}
                                      for b, v in sorted(link.items(), key=lambda kv: int(kv[0].split("-")[0]))},
               "by_size": {}}
        md += [f"## {run}（自己的估计器 {own}）", "",
               f"重放 {episodes} 局，S0 回报和记录不符 {return_mismatch} 局；和训练日志的差（最大）："
               + "，".join(f"{k.split('/')[-1]} {v['max_abs_diff']}" for k, v in res["check_vs_log"].items()), "",
               "| 估计器 | 更新 | 回合动作数 | 回合数 | token 占比 | 整局项 / token | 逐回合项 / token | 合计 / token | 合计 / 回合 | 合计（成功局） | 合计（失败局） |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for est in cfgs:
            blocks = sorted({k[1] for k in acc if k[0] == est}, key=lambda s: int(s.split("-")[0]))
            res["by_size"][est] = {}
            for blk in blocks:
                total_tokens = sum(acc[(est, blk, b)][1] for b in BUCKETS)
                rows_b = {}
                for b in BUCKETS:
                    a = acc.get((est, blk, b))
                    if a is None or not a[0]:
                        continue
                    s, f_ = outc.get((est, blk, b, "success")), outc.get((est, blk, b, "fail"))
                    rows_b[b] = {"turns": int(a[0]), "token_share": round(a[1] / total_tokens, 4),
                                 "episode_part": round(a[2] / a[1], 4), "turn_part": round(a[3] / a[1], 4),
                                 "sum": round(a[4] / a[1], 4), "sum_per_turn": round(a[5] / a[0], 4),
                                 "sum_success": round(s[3] / s[0], 4) if s is not None and s[0] else None,
                                 "sum_fail": round(f_[3] / f_[0], 4) if f_ is not None and f_[0] else None}
                    r = rows_b[b]
                    md.append(f"| {est} | {blk} | {b} | {r['turns']} | {r['token_share']} | {r['episode_part']} | "
                              f"{r['turn_part']} | {r['sum']} | {r['sum_per_turn']} | {r['sum_success']} | {r['sum_fail']} |")
                res["by_size"][est][blk] = rows_b
        md += ["", "成功局减失败局（有成有败的组）：", "", "| 更新 | 组数 | 每回合动作数 | 每局回合数 |", "| --- | --- | --- | --- |"]
        md += [f"| {b} | {v['groups']} | {v['actions_per_turn']} | {v['turns_per_episode']} |"
               for b, v in res["success_minus_fail"].items()]
        md.append("")
        summary[run] = res
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({run: {k: summary[run][k] for k in ("episodes", "return_mismatch", "check_vs_log")}
                      for run in runs}, indent=1))


if __name__ == "__main__":
    main()
