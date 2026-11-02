"""Reward line, phase 2: the screening comparison (the user's brief of 2026-11-01, section 3).

Each design run (e.g. rw_tc1, rw_tc2) against rw_g (V4) and rw_s0 (S0) of the same seed, start and tasks, at the
same updates 50 / 100 / 150, on validation v2 + test v2 (705 tasks, "both"):

* per checkpoint: the success rates, the difference in points, the tasks only one side solved and the exact
  two-sided sign test (compare_b.compare: evaluation.paired one set at a time, the counts summed);
* the screening rule: the run's success on both sets, averaged over the three checkpoints, 3 points or more above
  the reference's average, and at least 2 of the 3 checkpoints in the same direction ("过", next round: two seeds,
  300 updates); otherwise "分不出" (stop).  Read against each reference; the brief asks for both;
* per checkpoint, both sets together (evaluation.metrics): collision rate (per action), unreadable turns,
  action efficiency (actions to success over d*, successful episodes), success by d* band and by family.

    python -m reward.compare_p2 --set val=DIR --set test=DIR --run rw_tc1 --run rw_tc2 --out FILE.json

FILE.json and FILE.md (the tables in Chinese).
"""

import argparse
import json
from pathlib import Path

from data_gen.interiorgs_v10.evaluation import metrics
from reward.compare_b import compare, load_arm, rates

REFS = ("rw_g", "rw_s0")
UPDATES = (50, 100, 150)
PASS_POINTS, PASS_SAME = 3.0, 2


def screen(run_res, ref_res):
    per = {u: compare(run_res[u], ref_res[u]) for u in UPDATES if u in run_res and u in ref_res}
    if len(per) < len(UPDATES):
        return {"complete": False, "updates": sorted(per), "per_update": per}
    avg_a = sum(rates(run_res[u])["both"]["success"] for u in UPDATES) / len(UPDATES)
    avg_b = sum(rates(ref_res[u])["both"]["success"] for u in UPDATES) / len(UPDATES)
    diff = round(100 * (avg_a - avg_b), 2)
    ahead = sum(per[u]["both"]["diff_points"] > 0 for u in UPDATES)
    behind = sum(per[u]["both"]["diff_points"] < 0 for u in UPDATES)
    same = ahead if diff > 0 else behind
    return {"complete": True, "avg_run": round(avg_a, 4), "avg_ref": round(avg_b, 4), "avg_diff_points": diff,
            "same_direction": same, "passes": bool(diff >= PASS_POINTS and same >= PASS_SAME), "per_update": per}


def extras(by_set):
    rows = [r for rs in by_set.values() for r in rs]
    m = metrics(rows)
    return {k: m[k] for k in ("success", "actions_over_dstar", "collision_rate", "invalid_rate", "success_by_band",
                              "success_by_family", "used_limit")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="NAME=DIR (val, test)")
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = dict(s.split("=", 1) for s in args.set)
    res = {name: load_arm(sets, name) for name in [*args.run, *REFS]}
    out = {"sets": sets, "updates": UPDATES, "rule": {"points": PASS_POINTS, "same_direction": PASS_SAME},
           "screen": {}, "extras": {}}
    for run in args.run:
        out["screen"][run] = {ref: screen(res[run], res[ref]) for ref in REFS}
    for name, r in res.items():
        out["extras"][name] = {u: extras(r[u]) for u in UPDATES if u in r}
    Path(args.out).write_text(json.dumps(out, indent=1))
    L = ["| 运行 | 对照 | 第 50 次 | 第 100 次 | 第 150 次 | 三次平均（本 / 对照） | 差 | 同向 | 判定 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run, refs in out["screen"].items():
        for ref, s in refs.items():
            cells = []
            for u in UPDATES:
                c = s["per_update"].get(u)
                cells.append(f"{c['both']['diff_points']:+.1f}（{c['both']['only_a']} 对 {c['both']['only_b']}，"
                             f"p {c['both']['p']:.3f}）" if c else "—")
            if s["complete"]:
                tail = (f"{100 * s['avg_run']:.1f}% / {100 * s['avg_ref']:.1f}% | {s['avg_diff_points']:+.1f} | "
                        f"{s['same_direction']} / 3 | {'过' if s['passes'] else '分不出'}")
            else:
                tail = "— | — | — | 未评完"
            L.append(f"| {run} | {ref} | " + " | ".join(cells) + f" | {tail} |")
    L += ["", "| 运行 | 更新 | 合并成功率 | 动作效率（做成的局：动作数 / d\\*） | 碰撞率（每个动作） | 读不出（每回合） |",
          "| --- | --- | --- | --- | --- | --- |"]
    for name, by_u in out["extras"].items():
        for u, e in by_u.items():
            L.append(f"| {name} | {u} | {100 * e['success']:.1f}% | {e['actions_over_dstar']} | {e['collision_rate']} | "
                     f"{e['invalid_rate']} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
