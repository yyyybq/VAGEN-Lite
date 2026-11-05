#!/usr/bin/env python3
"""N2 split by the first action of each instance's shortest path (the user's task book of 2026-11-05, item 8: the
move-backward-first bias; analysis only, nothing retrained).

    python downstream/n/n2_backward.py --n2 N2 --instances I.jsonl [--exclude IDS ...] --out DIR

The instances whose recorded shortest path (n_gen.py's: the first success pose of at least 2 actions its search found,
and the path to it; the search tries the actions in n_env.ACTIONS order, so among equally short paths the one that
starts with move_backward is kept whenever moving forward is blocked) starts with move_backward against the rest.  In each part and tier (0 = the model as it is, N1 and its second set;
100, 300, 1,000 demonstrations, n2_report.py's): each model's success; each trained start against its generation's
base and each RL start against its SFT start (n_report.compare, paired on the part's instances); whether the two
parts give the same difference (n_report.agree: z test, the parts taken as independent sets).  Per model, tier and
part: the share of episodes whose first action (the first action of the first readable reply, a blocked one included)
is the recorded shortest path's first action ("first step optimal": roughly a lower bound, as another path just as
short may start with another action; not where a single other action already succeeds, which the generator's 2-action
minimum leaves out) and the share whose first action is move_backward.  --exclude: n_report.excluded's files, left out everywhere.  DIR/report.json, report.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n2_report as N2R  # noqa: E402
import n_report as R  # noqa: E402

PARTS = ("backward", "other", "all")
PART_ZH = {"backward": "第一步后退", "other": "第一步不是后退", "all": "全部"}


def first_action(row: dict):
    """The episode's first executed action (the first turn with actions), None if it never acted."""
    for t in row.get("turn_log") or []:
        if t.get("executed"):
            return t["executed"][0]
    return None


def rows(paths, exclude=()) -> dict:
    out = {}
    for p in paths:
        for r in R.read_jsonl(p):
            if not r.get("error") and r["instance_id"] not in exclude:
                out[r["instance_id"]] = r
    return out


def load(n2: Path, runs: Path = N2R.RUNS, exclude=(), models=R.MODELS) -> dict:
    """{model: {tier: {instance: episode row}}}, the tiers as n2_report.load reads them."""
    out = {}
    for m in models:
        tiers = {0: rows([runs / "n/n1_eval_20261031" / m / "episodes.jsonl",
                          runs / "n/n1b_eval_20261102" / m / "episodes.jsonl"], exclude)}
        for n in N2R.TIERS[1:]:
            s = rows([Path(n2) / m / "eval" / str(n) / "episodes.jsonl"], exclude)
            if s:
                tiers[n] = s
        out[m] = tiers
    return out


def first_steps(insts: dict) -> dict:
    return {i: ((r.get("shortest") or {}).get("path") or [None])[0] for i, r in insts.items()}


def build(data: dict, insts: dict, exclude=()) -> dict:
    first = {i: a for i, a in first_steps(insts).items() if i not in exclude}
    ids = {"backward": {i for i, a in first.items() if a == "move_backward"},
           "other": {i for i, a in first.items() if a != "move_backward"}}
    ids["all"] = ids["backward"] | ids["other"]
    succ = {m: {n: {i: bool(r.get("success")) for i, r in s.items()} for n, s in tiers.items()}
            for m, tiers in data.items()}
    rep = {"parts": {p: len(v) for p, v in ids.items()},
           "shortest_first_action": dict(collections.Counter(first.values()).most_common()),
           "excluded": sorted(exclude), "rates": {}, "vs_base": {}, "vs_sft": {}, "gap": {}, "first_step": {}}
    for part in PARTS:
        rep["rates"][part], rep["vs_base"][part], rep["vs_sft"][part] = {}, {}, {}
        for m, tiers in succ.items():
            rep["rates"][part][m] = {str(n): R.rate(s, ids[part]) for n, s in tiers.items()}
            b, sft = R.base_of(m), R.sft_of(m)
            if m != b and b in succ:
                rep["vs_base"][part][m] = {str(n): R.compare(s, succ[b][n], ids[part]) for n, s in tiers.items()
                                           if n in succ[b]}
            if sft and sft in succ:
                rep["vs_sft"][part][m] = {str(n): R.compare(s, succ[sft][n], ids[part]) for n, s in tiers.items()
                                          if n in succ[sft]}
    for key in ("vs_base", "vs_sft"):
        rep["gap"][key] = {m: {n: R.agree(c, rep[key]["other"][m][n]) for n, c in by.items()
                               if n in rep[key]["other"].get(m, {})} for m, by in rep[key]["backward"].items()}
    for m, tiers in data.items():
        rep["first_step"][m] = {}
        for n, rs in tiers.items():
            cells = {}
            for part in PARTS:
                sel = [r for i, r in rs.items() if i in ids[part]]
                fa = [first_action(r) for r in sel]
                k_opt = sum(a is not None and a == first[r["instance_id"]] for a, r in zip(fa, sel))
                k_back = sum(a == "move_backward" for a in fa)
                cells[part] = {"n": len(sel), "optimal": k_opt, "backward": k_back,
                               "optimal_rate": k_opt / len(sel) if sel else math.nan,
                               "backward_rate": k_back / len(sel) if sel else math.nan}
            rep["first_step"][m][str(n)] = cells
    return rep


def tier_name(n) -> str:
    return "0 条" if str(n) == "0" else f"{int(n):,} 条"


def markdown(rep: dict) -> str:
    p = rep["parts"]
    lines = ["# N2 按最短路径第一步分开（自动生成）", "",
             f"第一步后退 {p['backward']} 个实例，第一步不是后退 {p['other']} 个，共 {p['all']} 个"
             + (f"（剔除 {len(rep['excluded'])} 个）" if rep["excluded"] else "") + "。",
             "最短路径第一步：" + "，".join(f"{a} {k}" for a, k in rep["shortest_first_action"].items()) + "。", "",
             "## 成功率", "", "| 模型 | 档 | 第一步后退 | 第一步不是后退 | 全部 |", "| --- | --- | --- | --- | --- |"]
    for m, by in rep["rates"]["all"].items():
        for n in by:
            lines.append(f"| {m} | {tier_name(n)} | " + " | ".join(R.cell(rep["rates"][part][m][n]) for part in PARTS)
                         + " |")
    for key, title in (("vs_base", "训练过的减底座"), ("vs_sft", "RL 减 SFT")):
        lines += ["", f"## {title}", "", "| 模型 | 档 | 第一步后退 | 第一步不是后退 | 两部分之差 | 全部 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for m, by in rep[key]["all"].items():
            for n in by:
                lines.append(f"| {m} | {tier_name(n)} | {R.diff_cell(rep[key]['backward'][m].get(n, {}))} | "
                             f"{R.diff_cell(rep[key]['other'][m].get(n, {}))} | "
                             f"{R.gap_cell(rep['gap'][key].get(m, {}).get(n))} | {R.diff_cell(by[n])} |")
    lines += ["", "## 第一步", "", "第一步最优 = 回合里第一个读得出的回复的第一个动作（被挡住的也算），就是记录的最短走法的第一步。"
              "大体是下限：别的一样短的走法可能换个动作开头；例外是一个别的动作一步就能做成、而记录的第一步一步做不成的实例"
              "（生成器只认至少 2 个动作的走法）。", "",
              "| 模型 | 档 | 第一步最优：后退部分 | 第一步最优：其余 | 第一步后退：后退部分 | 第一步后退：其余 |",
              "| --- | --- | --- | --- | --- | --- |"]

    def share(c, k):
        return f"{c[k]}/{c['n']}（{R.pct(c[k + '_rate'])}%）" if c["n"] else "—"
    for m, by in rep["first_step"].items():
        for n, c in by.items():
            lines.append(f"| {m} | {tier_name(n)} | {share(c['backward'], 'optimal')} | {share(c['other'], 'optimal')} | "
                         f"{share(c['backward'], 'backward')} | {share(c['other'], 'backward')} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n2", type=Path, required=True)
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--runs", type=Path, default=N2R.RUNS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    exclude = R.excluded(args.exclude)
    insts = {r["instance_id"]: r for r in R.read_jsonl(args.instances)}
    rep = build(load(args.n2, args.runs, exclude), insts, exclude)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (args.out / "report.md").write_text(markdown(rep), encoding="utf-8")
    print(json.dumps(rep["parts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
