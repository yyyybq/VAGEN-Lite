"""Reward line, step 4: PREREG_B section 3's confirmation over two seeds ("涨" only if all three hold: both seeds the
same direction, the task-by-task sign test over both seeds together p < 0.05, the mean difference at least +2 points).

Each --pair is one seed's arm A and arm B as run prefixes (rw_ok,rw_s0 for seed 1; rw_s2ok,rw_s2s0 for seed 2).  Per
seed, view 2 as PREREG section 4 reads it: each run's best checkpoint on the selection set (ties: the earlier), a run
stopped by a guard (the chains' guard_stops) only among its checkpoints before the stop; and view 1 at the same
updates (50, 100, 150).  Both sets together, the only-A / only-B counts of every seed summed for the pooled sign test
(evaluation.sign_test), the mean of the seeds' differences for the +2 rule.

    python -m reward.two_seed_b --set val=DIR --set test=DIR --pair rw_ok,rw_s0 --pair rw_s2ok,rw_s2s0
        --progress CHAIN/progress.json [...] --out FILE.json

FILE.json and FILE.md.
"""

import argparse
import json
from pathlib import Path

from data_gen.interiorgs_v10.evaluation import sign_test
from reward.compare_b import best, compare, load_arm, rates

UPDATES = (50, 100, 150)


def verdict(per_seed):
    """PREREG section 3 over the seeds' comparisons (each with both.diff_points, both.only_a, both.only_b)."""
    diffs = [c["both"]["diff_points"] for c in per_seed]
    wa = sum(c["both"]["only_a"] for c in per_seed)
    wb = sum(c["both"]["only_b"] for c in per_seed)
    same = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    p = sign_test(wa, wb)
    mean = sum(diffs) / len(diffs)
    return {"diffs": [round(d, 2) for d in diffs], "same_direction": same, "only_a": wa, "only_b": wb,
            "p_pooled": round(p, 4), "mean_diff": round(mean, 2),
            "rises": bool(same and all(d > 0 for d in diffs) and p < 0.05 and mean >= 2.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="NAME=DIR, the selection set first")
    ap.add_argument("--pair", action="append", required=True, help="RUN_A,RUN_B for one seed")
    ap.add_argument("--progress", action="append", default=[], help="a chain's progress.json (guard stops)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = dict(s.split("=", 1) for s in args.set)
    sel = next(iter(sets))
    stops = {}
    for p in args.progress:
        st = json.loads(Path(p).read_text())
        for run in (st.get("guard_stops") or {}):
            stops[run] = (st.get("rl_runs") or {}).get(run, {}).get("last_update")
    seeds = []
    for spec in args.pair:
        a, b = spec.split(",")
        ra, rb = load_arm(sets, a), load_arm(sets, b)
        ua, ub = best(ra, sel, stops.get(a)), best(rb, sel, stops.get(b))
        view2 = {"a": a, "b": b, "update_a": ua, "update_b": ub, **compare(ra[ua], rb[ub])}
        view1 = {u: compare(ra[u], rb[u]) for u in UPDATES if u in ra and u in rb}
        seeds.append({"a": a, "b": b, "stops": {a: stops.get(a), b: stops.get(b)}, "view2": view2, "view1": view1,
                      "rates": {"a": {u: rates(r)["both"]["success"] for u, r in ra.items()},
                                "b": {u: rates(r)["both"]["success"] for u, r in rb.items()}}})
    out = {"sets": sets, "select_by": sel, "seeds": seeds, "view2": verdict([s["view2"] for s in seeds]),
           "view1": {u: verdict([s["view1"][u] for s in seeds]) for u in UPDATES if all(u in s["view1"] for s in seeds)}}
    Path(args.out).write_text(json.dumps(out, indent=1))

    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    L = ["## 每个种子（口径 2：各自验证集最好）", "",
         "| 种子 | 甲 | 乙 | 甲 合并 | 乙 合并 | 差（只有甲 / 只有乙，p） |", "| --- | --- | --- | --- | --- | --- |"]
    for i, s in enumerate(seeds, 1):
        v = s["view2"]
        L.append(f"| {i} | {v['a']} 第 {v['update_a']} 次 | {v['b']} 第 {v['update_b']} 次 | {pct(v['both']['a'])} | "
                 f"{pct(v['both']['b'])} | {v['both']['diff_points']:+.1f}（{v['both']['only_a']} / {v['both']['only_b']}，"
                 f"p {v['both']['p']:.3f}） |")
    L += ["", "## PREREG 第 3 节的判定", "", "| 口径 | 两个种子的差 | 同向 | 合并逐题（只有甲 / 只有乙） | 合并 p | 平均差 | 涨 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    rows = [("口径 2", out["view2"])] + [(f"口径 1：第 {u} 次", v) for u, v in out["view1"].items()]
    for name, v in rows:
        L.append(f"| {name} | {' / '.join(f'{d:+.1f}' for d in v['diffs'])} | {'是' if v['same_direction'] else '否'} | "
                 f"{v['only_a']} / {v['only_b']} | {v['p_pooled']:.3f} | {v['mean_diff']:+.1f} | {'是' if v['rises'] else '否'} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
