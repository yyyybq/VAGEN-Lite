"""Reward line, phase 2 direction 3 round 1: the main comparison of PREREG_DIR3.md (sections 2 and 3).

The statement run (rw_st1) against its control (rw_st0, same start e5, seed and tasks) at the given updates, on
validation v2 + test v2 (705 tasks, "both"):

* per checkpoint: the success rates, the difference in points, the tasks only one side solved and the exact
  two-sided sign test (compare_b.compare: evaluation.paired one set at a time, the counts summed);
* the rule: the run's success on both sets, averaged over the given checkpoints, 3 points or more above the
  control's average, and at least 2 checkpoints in the same direction ("过"); otherwise "分不出" (or "更差" when the
  run is 3 points or more below with 2 checkpoints behind).  The updates are an argument, not a default: the
  pre-registration names 50 / 100 / 150, and when a run stops early the user decides which ones are compared;
* references (section 3, descriptive only, a different start): both runs against rw_g (V4) and rw_s0 (S0) of the
  same seed and tasks from C0 e8, at the same updates; the start e5 on both sets;
* per checkpoint, both sets together (compare_p2.extras): collision rate, unreadable turns, action efficiency,
  success by d* band and by family.

    python -m reward.compare_dir3 --set val=DIR --set test=DIR --run rw_st1 --control rw_st0 --updates 50 100 \
        --start sft_q3stmt_e5 --out FILE.json

FILE.json and FILE.md (the tables in Chinese).
"""

import argparse
import json
from pathlib import Path

from reward.compare_b import _jsonl, compare, load_arm, rates
from reward.compare_p2 import extras

REFS = ("rw_g", "rw_s0")
PASS_POINTS, PASS_SAME = 3.0, 2


def screen(run_res, ref_res, updates):
    per = {u: compare(run_res[u], ref_res[u]) for u in updates if u in run_res and u in ref_res}
    if len(per) < len(updates):
        return {"complete": False, "updates": sorted(per), "per_update": per}
    avg_a = sum(rates(run_res[u])["both"]["success"] for u in updates) / len(updates)
    avg_b = sum(rates(ref_res[u])["both"]["success"] for u in updates) / len(updates)
    diff = round(100 * (avg_a - avg_b), 2)
    ahead = sum(per[u]["both"]["diff_points"] > 0 for u in updates)
    behind = sum(per[u]["both"]["diff_points"] < 0 for u in updates)
    verdict = ("过" if diff >= PASS_POINTS and ahead >= PASS_SAME else
               "更差" if diff <= -PASS_POINTS and behind >= PASS_SAME else "分不出")
    return {"complete": True, "avg_run": round(avg_a, 4), "avg_ref": round(avg_b, 4), "avg_diff_points": diff,
            "ahead": ahead, "behind": behind, "verdict": verdict, "per_update": per}


def load_start(sets, name):
    """{set: rows} of one result name (the start has no update suffix), or None when a set lacks it."""
    out = {}
    for s, d in sets.items():
        p = Path(d) / name
        if not ((p / "done.json").is_file() and (p / "results.jsonl").is_file()):
            return None
        out[s] = _jsonl(p / "results.jsonl")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="NAME=DIR (val, test)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--updates", type=int, nargs="+", required=True)
    ap.add_argument("--start", default=None, help="the start's result name on every set")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = dict(s.split("=", 1) for s in args.set)
    updates = tuple(args.updates)
    res = {name: load_arm(sets, name) for name in [args.run, args.control, *REFS]}
    out = {"sets": sets, "updates": updates, "rule": {"points": PASS_POINTS, "same_direction": PASS_SAME},
           "main": screen(res[args.run], res[args.control], updates),
           "references": {a: {ref: screen(res[a], res[ref], updates) for ref in REFS}
                          for a in (args.run, args.control)},
           "start": None, "extras": {}}
    start = load_start(sets, args.start) if args.start else None
    if start:
        out["start"] = {"name": args.start, **{k: v["success"] for k, v in rates(start).items()}}
    for name, r in res.items():
        out["extras"][name] = {u: extras(r[u]) for u in updates if u in r}
    Path(args.out).write_text(json.dumps(out, indent=1))

    def row(a, b, s):
        cells = []
        for u in updates:
            c = s["per_update"].get(u)
            cells.append(f"{c['both']['diff_points']:+.1f}（{c['both']['only_a']} 对 {c['both']['only_b']}，"
                         f"p {c['both']['p']:.3f}）" if c else "—")
        tail = (f"{100 * s['avg_run']:.1f}% / {100 * s['avg_ref']:.1f}% | {s['avg_diff_points']:+.1f} | "
                f"{s['ahead']} 高 {s['behind']} 低 | {s['verdict']}" if s["complete"] else "— | — | — | 未评完")
        return f"| {a} | {b} | " + " | ".join(cells) + f" | {tail} |"

    head = ("| 运行 | 对照 | " + " | ".join(f"第 {u} 次：差（只有本 / 只有对照，p）" for u in updates) +
            " | 平均（本 / 对照） | 差 | 逐次 | 判定 |")
    sep = "|" + " --- |" * (len(updates) + 6)
    L = ["## 主比较（合并 705 道）", "", head, sep, row(args.run, args.control, out["main"]), "",
         "## 参照（起点不同，只作描述）", "", head, sep]
    L += [row(a, ref, s) for a, refs in out["references"].items() for ref, s in refs.items()]
    if out["start"]:
        st = out["start"]
        L += ["", f"起点 {st['name']}：" + "，".join(f"{k} {100 * v:.1f}%" for k, v in st.items() if k != "name"
                                                  and v is not None)]
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
