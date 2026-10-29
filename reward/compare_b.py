"""Reward line, step 4: task B's comparisons the way docs/reward/PREREG_B.md fixed them (2026-10-27).

For every arm found (S0 rw_s0, G rw_g, O rw_o, O-keep rw_ok) every checkpoint evaluated on both sets (a result
folder rw_<arm>_<NNN> with done.json in each set): success on validation v2, test v2 and both together.

* view 1, the same update: 50, 100 and 150;
* view 2, each arm's best checkpoint on validation (ties: the earlier one); when S0 was stopped by a guard (the
  chain's guard_stops), both versions: (a) every arm picks only among the checkpoints before S0's stop, (b) the other
  arms among all theirs, S0 among the ones before its stop;
* each comparison (O vs G first, then O vs S0, O-keep vs S0, O-keep vs G, as far as both arms exist): the success
  rates, the difference in points, the tasks only one side solved (per set and together) and the exact two-sided
  sign test (the main line's evaluation.paired / sign_test, one set at a time, the counts summed for both sets);
* the second-seed rule (PREREG section 4, the user's rule 6): O or O-keep against S0 in view 2 with p < 0.1 on both
  sets together or test success 2.4 points or more higher.  Read with the arm ahead of S0; a p < 0.1 with the arm
  behind is reported apart, not acted on.

    python -m reward.compare_b --set val=DIR --set test=DIR --progress CHAIN/progress.json [...] --out FILE.json

FILE.json and FILE.md (the tables in Chinese).
"""

import argparse
import json
from pathlib import Path

from data_gen.interiorgs_v10.evaluation import _jsonl, metrics, paired, sign_test

ARMS = {"S0": "rw_s0", "G": "rw_g", "O": "rw_o", "OK": "rw_ok"}
NAMES = {"S0": "S0", "G": "G", "O": "O", "OK": "O-keep"}
PAIRS = [("O", "G"), ("O", "S0"), ("OK", "S0"), ("OK", "G")]  # the main comparison first
SAME_UPDATES = (50, 100, 150)
SEED2_P, SEED2_TEST_POINTS = 0.1, 2.4


def load_arm(sets, run):
    """{update: {set: rows}} for the checkpoints evaluated on every set."""
    found = {}
    for set_name, d in sets.items():
        for p in Path(d).glob(f"{run}_[0-9][0-9][0-9]"):
            if (p / "done.json").is_file() and (p / "results.jsonl").is_file():
                found.setdefault(int(p.name.rsplit("_", 1)[1]), {})[set_name] = _jsonl(p / "results.jsonl")
    return {u: v for u, v in sorted(found.items()) if set(v) == set(sets)}


def rates(by_set):
    out, solved, n = {}, 0, 0
    for s, rows in by_set.items():
        m = metrics(rows)
        k = sum(bool(r["success"]) for r in rows if "error" not in r)
        out[s] = {"success": m["success"], "episodes": m["episodes"], "errors": m["errors"],
                  "collision_rate": m["collision_rate"], "invalid_rate": m["invalid_rate"]}
        solved += k
        n += m["episodes"]
    out["both"] = {"success": round(solved / n, 4) if n else None, "episodes": n}
    return out


def compare(a, b):
    """a against b, checkpoint result dicts {set: rows}: per set and both together."""
    out, wa, wb = {}, 0, 0
    for s in a:
        pr = paired(b[s], a[s])  # base = b: only_other = tasks only a solved
        ra, rb = metrics(a[s])["success"], metrics(b[s])["success"]
        out[s] = {"a": ra, "b": rb, "diff_points": round(100 * (ra - rb), 2), "only_a": pr["only_other"],
                  "only_b": pr["only_base"], "tasks": pr["tasks"], "p": pr["p"]}
        wa += pr["only_other"]
        wb += pr["only_base"]
    ra, rb = rates(a)["both"]["success"], rates(b)["both"]["success"]
    out["both"] = {"a": ra, "b": rb, "diff_points": round(100 * (ra - rb), 2), "only_a": wa, "only_b": wb,
                   "p": round(sign_test(wa, wb), 4)}
    return out


def best(arm_results, sel, limit=None):
    cand = {u: r for u, r in arm_results.items() if limit is None or u <= limit}
    if not cand:
        return None
    return max(cand, key=lambda u: (metrics(cand[u][sel])["success"], -u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="NAME=DIR, the selection set first")
    ap.add_argument("--progress", action="append", default=[], help="a chain's progress.json (guard stops)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = dict(s.split("=", 1) for s in args.set)
    sel = next(iter(sets))
    stops = {}
    for p in args.progress:
        st = json.loads(Path(p).read_text())
        for run, why in (st.get("guard_stops") or {}).items():
            stops[run] = {"reason": why.get("reason"), "utc": why.get("utc"),
                          "last_update": (st.get("rl_runs") or {}).get(run, {}).get("last_update")}
    res = {arm: load_arm(sets, run) for arm, run in ARMS.items()}
    res = {arm: r for arm, r in res.items() if r}
    table = {arm: {u: rates(r[u]) for u in r} for arm, r in res.items()}
    pairs = [(a, b) for a, b in PAIRS if a in res and b in res]

    view1 = {}
    for u in SAME_UPDATES:
        for a, b in pairs:
            if u in res[a] and u in res[b]:
                view1[f"{a}-{b}@{u}"] = {"a": a, "b": b, "update": u, **compare(res[a][u], res[b][u])}

    s0_stop = (stops.get(ARMS["S0"]) or {}).get("last_update")
    versions = {"b": None} if s0_stop is None else {"a": s0_stop, "b": None}
    view2 = {}
    for v, limit in versions.items():
        picks = {arm: best(r, sel, limit if (v == "a" or arm == "S0") else None) for arm, r in res.items()}
        comps = {f"{a}-{b}": {"a": a, "b": b, "update_a": picks[a], "update_b": picks[b],
                              **compare(res[a][picks[a]], res[b][picks[b]])}
                 for a, b in pairs if picks.get(a) and picks.get(b)}
        view2[v] = {"limit": limit, "picks": picks, "comparisons": comps}

    seed2 = {}
    for v, d in view2.items():
        for arm in ("O", "OK"):
            c = d["comparisons"].get(f"{arm}-S0")
            if not c:
                continue
            ahead = c["both"]["diff_points"] > 0
            seed2[f"{arm}@{v}"] = {"p_both": c["both"]["p"], "diff_both": c["both"]["diff_points"],
                                   "diff_test": c["test"]["diff_points"] if "test" in c else None,
                                   "fires": bool((c["both"]["p"] < SEED2_P and ahead) or
                                                 ("test" in c and c["test"]["diff_points"] >= SEED2_TEST_POINTS)),
                                   "p_below_but_behind": bool(c["both"]["p"] < SEED2_P and not ahead)}
    out = {"sets": sets, "select_by": sel, "stops": stops, "checkpoints": table, "view1": view1, "view2": view2,
           "seed2": seed2}
    Path(args.out).write_text(json.dumps(out, indent=1))

    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    L = ["## 各检查点", "", "| 臂 | 检查点 | 验证集 | 评测集 | 合并 | 碰撞率（验证 / 评测） | 读不出（验证 / 评测） |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, by_u in table.items():
        for u, r in by_u.items():
            sv, st = r.get("val") or {}, r.get("test") or {}
            L.append(f"| {NAMES[arm]} | {u} | {pct(sv.get('success'))} | {pct(st.get('success'))} | "
                     f"{pct(r['both']['success'])} | {sv.get('collision_rate')} / {st.get('collision_rate')} | "
                     f"{sv.get('invalid_rate')} / {st.get('invalid_rate')} |")

    def comp_rows(items, label):
        rows = []
        for k, c in items:
            parts = []
            for s in list(sets) + ["both"]:
                x = c[s]
                parts.append(f"{x['diff_points']:+.1f}（{x['only_a']} / {x['only_b']}，p {x['p']:.3f}）")
            rows.append(f"| {label(c)} | {NAMES[c['a']]} 对 {NAMES[c['b']]} | {pct(c['both']['a'])} 对 {pct(c['both']['b'])} | "
                        + " | ".join(parts) + " |")
        return rows

    head = "| 口径 | 比较 | 合并成功率 | " + " | ".join(f"{s}：差（只有甲 / 只有乙，p）" for s in sets) + \
           " | 合并：差（只有甲 / 只有乙，p） |"
    sep = "| --- | --- | --- | " + " | ".join("---" for _ in sets) + " | --- |"
    L += ["", "## 口径 1：同一更新次数", "", head, sep]
    L += comp_rows(view1.items(), lambda c: f"第 {c['update']} 次")
    for v, d in view2.items():
        title = {"a": f"口径 2（a）走样前最好（都只在第 {d['limit']} 次以前挑）", "b": "口径 2（b）各臂到 300 次（S0 在它停下前挑）"}[v] \
            if s0_stop is not None else "口径 2：各自验证集最好"
        L += ["", f"## {title}", "", "挑中的检查点：" + "，".join(f"{NAMES[a]} 第 {u} 次" for a, u in d["picks"].items() if u), "",
              head, sep]
        L += comp_rows(d["comparisons"].items(), lambda c: f"第 {c['update_a']} 次 对 第 {c['update_b']} 次")
    if seed2:
        L += ["", "## 第二个种子的条件（PREREG 第 4 节）", "",
              "第 4 节用口径 2 的各自最好（S0 停下时就是上面的（b））；（a）只作参考。合并 p < 0.1 只在该臂领先 S0 时算数。", "",
              "| 臂（口径） | 合并 p | 合并差 | 评测集差 | 满足 |", "| --- | --- | --- | --- | --- |"]
        for k, s in seed2.items():
            arm, v = k.split("@")
            L.append(f"| {NAMES[arm]}（{v}） | {s['p_both']:.3f} | {s['diff_both']:+.1f} | "
                     f"{'—' if s['diff_test'] is None else format(s['diff_test'], '+.1f')} | "
                     f"{'是' if s['fires'] else '否'}{'（p < 0.1 但落后）' if s['p_below_but_behind'] else ''} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
