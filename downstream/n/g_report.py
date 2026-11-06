#!/usr/bin/env python3
"""Tier g's report (the user's task book of 2026-11-06; design reports/G_DESIGN_20261106.md section 5): the N series
with dialogue history (g) set against the same instances without it (the recorded runs).

    python downstream/n/g_report.py --g N3/gd --d N1 N1b --flag FLAG.jsonl [--exclude SCAN.json ...] --out DIR

The user's decisions of 2026-11-06: option 乙, (g-d) first; (g-c) is submitted only if RS − S comes back to +3 points
or more, or a trained model's difference to its base moves by more than 3 points (gc_condition below); the instances
of the scenes whose rebuild looked inconsistent (--flag, a jsonl of instances) are marked apart in every table: every
table is given on all instances and again without them.
Per model (main table): both rates, g − d in points with a 95% interval from resampling whole scenes (DRAWS, as N1's
main table), the discordant instances and McNemar's exact p (uncorrected).  Per pairing (training − base: S, RS, RD
− B and Q3S, Q3RS, Q3RD − Q3B; RL − SFT: RS, RD − S and Q3RS, Q3RD − Q3S; RD − RS and Q3RD − Q3RS): the difference
with history, without, and "with minus without" (DiD), each with its whole-scene 95% interval; the DiD also with a
two-sided bootstrap p, Holm-adjusted over the 12 pairings.  A pairing "changes N1's conclusion" when its Holm p is
below 0.05 or its interval's position (above 0 / below 0 / across 0) differs between the two versions.  Reading
rules, per family: "RS − S comes back" when RS − S with history and its DiD both have intervals above 0; "RD − S
changes" when the DiD's interval leaves out 0.  Per task type the rates of both versions (descriptive only).
Error rows (a server failure still there after the replay) and the instances of --exclude are left out of both
sides.  Prompt lengths: the server's count of each call's prompt (turn_log prompt_tokens), mean and max per turn.
Writes DIR/report.json and DIR/report.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n_report as R  # noqa: E402

stats = R.stats
MODELS = R.MODELS
PAIRS = (("S", "B"), ("RS", "B"), ("RD", "B"), ("Q3S", "Q3B"), ("Q3RS", "Q3B"), ("Q3RD", "Q3B"),
         ("RS", "S"), ("RD", "S"), ("Q3RS", "Q3S"), ("Q3RD", "Q3S"), ("RD", "RS"), ("Q3RD", "Q3RS"))
TRAINED_VS_BASE = PAIRS[:6]
FAMILIES = (("RS", "S"), ("Q3RS", "Q3S"))  # "comes back"
RD_FAMILIES = (("RD", "S"), ("Q3RD", "Q3S"))  # "changes"
GC_RS_POINTS, GC_MOVE_POINTS = 0.03, 0.03  # the user's condition for (g-c)
DRAWS, SEED = 2000, 20261106
SUBSETS = ("all", "without flagged")
SUBSET_ZH = {"all": "全部实例", "without flagged": "去掉可疑场景的实例"}


def load(dirs, models=MODELS, exclude=()) -> tuple:
    """({model: {instance: success}}, {instance: scene}, {model: [error instances]}, {model: [rows]}); a later
    folder's row for an instance replaces an earlier one."""
    succ, scenes, errors, rows = {}, {}, {}, {}
    for m in models:
        got, bad, keep = {}, [], {}
        for d in dirs:
            for r in R.read_jsonl(Path(d) / m / "episodes.jsonl"):
                i = r["instance_id"]
                scenes.setdefault(i, r.get("scene_id", i))
                if i in exclude:
                    continue
                if r.get("error"):
                    bad.append(i)
                    got.pop(i, None)
                    continue
                got[i] = bool(r.get("success"))
                keep[i] = r
        if got or bad:
            succ[m], errors[m], rows[m] = got, sorted(set(bad) - set(got)), list(keep.values())
    return succ, scenes, errors, rows


def scene_boot(per: dict, scenes: dict, draws: int = DRAWS, seed: int = SEED) -> dict:
    """per: {instance: value}.  The mean with a 95% interval from resampling whole scenes, and a two-sided bootstrap
    p (twice the smaller share of resampled means on either side of 0, at most 1).  A p of 0 (no resampled mean on one
    side; the two-sided p moves in steps of 2 / draws) is set to 2 / draws with "p_floor" true, so the Holm
    adjustment stays an upper bound and the table prints "< 0.001" (2,000 draws)."""
    if not per:
        return {"n": 0, "mean": math.nan, "low": math.nan, "high": math.nan, "p": math.nan}
    groups = collections.defaultdict(list)
    for i, v in per.items():
        groups[scenes.get(i, i)].append(v)
    g, rng, means = list(groups.values()), random.Random(seed), []
    for _ in range(draws):
        pick = [g[rng.randrange(len(g))] for _ in g]
        means.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
    means.sort()
    le, ge = sum(x <= 0 for x in means) / draws, sum(x >= 0 for x in means) / draws
    p = min(1.0, 2 * min(le, ge))
    return {"n": len(per), "mean": sum(per.values()) / len(per), "low": means[int(0.025 * draws)],
            "high": means[int(0.975 * draws) - 1], "p": max(p, 2 / draws), "p_floor": p < 2 / draws}


def position(c: dict) -> str:
    if math.isnan(c["low"]):
        return "none"
    return "above" if c["low"] > 0 else "below" if c["high"] < 0 else "across"


def level(g: dict, d: dict, scenes: dict, ids) -> dict:
    ids = [i for i in ids if i in g and i in d]
    out = stats.paired({i: g[i] for i in ids}, {i: d[i] for i in ids})
    out["scene"] = scene_boot({i: int(g[i]) - int(d[i]) for i in ids}, scenes)
    return out


def pair_row(g: dict, d: dict, a: str, b: str, scenes: dict, ids) -> dict:
    ids = [i for i in ids if all(i in x.get(m, {}) for x in (g, d) for m in (a, b))]
    def diff(x):
        return {i: int(x[a][i]) - int(x[b][i]) for i in ids}
    dg, dd = diff(g), diff(d)
    return {"n": len(ids), "with": scene_boot(dg, scenes), "without": scene_boot(dd, scenes),
            "with_mcnemar": stats.paired({i: g[a][i] for i in ids}, {i: g[b][i] for i in ids})["p"] if ids else math.nan,
            "without_mcnemar": stats.paired({i: d[a][i] for i in ids}, {i: d[b][i] for i in ids})["p"] if ids else math.nan,
            "did": scene_boot({i: dg[i] - dd[i] for i in ids}, scenes)}


def by_type(succ: dict, rows: dict) -> dict:
    types = {r["instance_id"]: r.get("task_type") for rs in rows.values() for r in rs}
    out = {}
    for m, s in succ.items():
        c = collections.defaultdict(lambda: [0, 0])
        for i, v in s.items():
            c[types.get(i)][0] += int(v)
            c[types.get(i)][1] += 1
        out[m] = {t: {"k": k, "n": n} for t, (k, n) in c.items()}
    return out


def prompt_lengths(rows: dict) -> dict:
    out = {}
    for m, rs in rows.items():
        per = collections.defaultdict(list)
        for r in rs:
            for t in r.get("turn_log") or []:
                if t.get("prompt_tokens") is not None:
                    per[t["turn"]].append(t["prompt_tokens"])
        out[m] = {str(k): {"calls": len(v), "mean": sum(v) / len(v), "max": max(v)} for k, v in sorted(per.items())}
    return out


def build(g_dirs, d_dirs, flagged=(), exclude=(), models=MODELS) -> dict:
    g, sg, g_err, g_rows = load(g_dirs, models, exclude)
    d, sd, d_err, d_rows = load(d_dirs, models, exclude)
    scenes = {**sd, **sg}
    models = [m for m in models if m in g and m in d]
    gone = {i for m in models for i in g_err.get(m, []) + d_err.get(m, [])}  # out of both sides, every model
    universe = sorted({i for m in models for i in g[m]} - gone)
    rep = {"models": models, "errors": {"g": g_err, "d": d_err}, "left_out_for_errors": sorted(gone),
           "excluded_present": sorted(set(exclude) & (set(sg) | set(sd))),
           "flagged": sorted(set(flagged) & set(universe)), "subsets": {}}
    for sub in SUBSETS:
        ids = [i for i in universe if sub == "all" or i not in set(flagged)]
        lv = {m: level(g[m], d[m], scenes, ids) for m in models}
        pairs = {f"{a}-{b}": pair_row(g, d, a, b, scenes, ids) for a, b in PAIRS if a in models and b in models}
        adj = stats.holm([p["did"]["p"] for p in pairs.values()])
        for (k, p), h in zip(pairs.items(), adj):
            p["did"]["holm"] = h
            p["changed"] = bool((not math.isnan(h) and h < 0.05) or position(p["with"]) != position(p["without"]))
        rules = {}
        for a, b in FAMILIES:
            p = pairs.get(f"{a}-{b}")
            if p:
                rules[f"{a}-{b} comes back"] = bool(p["with"]["low"] > 0 and p["did"]["low"] > 0)
        for a, b in RD_FAMILIES:
            p = pairs.get(f"{a}-{b}")
            if p:
                rules[f"{a}-{b} changes"] = (("up" if p["did"]["low"] > 0 else "down")
                                            if p["did"]["low"] > 0 or p["did"]["high"] < 0 else False)
        gc = {f"{a}-{b} with history >= +3": pairs[f"{a}-{b}"]["with"]["mean"] >= GC_RS_POINTS
              for a, b in FAMILIES if f"{a}-{b}" in pairs}
        gc.update({f"{a}-{b} moved more than 3": abs(pairs[f"{a}-{b}"]["did"]["mean"]) > GC_MOVE_POINTS
                   for a, b in TRAINED_VS_BASE if f"{a}-{b}" in pairs})
        rep["subsets"][sub] = {"n": len(ids), "levels": lv, "pairs": pairs, "rules": rules,
                               "gc_condition": {"parts": gc, "met": any(gc.values())},
                               "level_changed": {m: position(v["scene"]) != "across" for m, v in lv.items()}}
    keep = set(universe)
    rep["by_type"] = {"g": by_type({m: {i: v for i, v in g[m].items() if i in keep} for m in models}, g_rows),
                      "d": by_type({m: {i: v for i, v in d[m].items() if i in keep} for m in models}, d_rows)}
    rep["flagged_episodes"] = {m: {i: {"g": g[m].get(i), "d": d[m].get(i)} for i in rep["flagged"]} for m in models}
    rep["prompt_tokens"] = prompt_lengths({m: g_rows[m] for m in models})
    return rep


def pts(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    v = round(100 * x, 1)
    return "0.0" if v == 0 else (f"+{v:.1f}" if v > 0 else f"−{-v:.1f}")


def ci(c: dict) -> str:
    return f"{pts(c['mean'])}（{pts(c['low'])} 到 {pts(c['high'])}）"


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def pv(p) -> str:
    """p as the reports write it: 3 decimals from 0.001 up, else 2 significant digits as m×10⁻ᵏ."""
    if p is None or math.isnan(p):
        return "—"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}×10{str(int(e)).translate(SUP)}"


def pair_zh(k: str) -> str:
    return k.replace("-", " − ")


def markdown(rep: dict) -> str:
    ex = rep.get("excluded_present") or []
    L = ["# (g) 档：带对话历史 对 不带（自动生成）", "",
         f"- 因出错，从全部模型的带、不带两边一起去掉的实例 {len(rep['left_out_for_errors'])} 个："
         + ("、".join(rep["left_out_for_errors"]) or "无") + "。",
         f"- 按剔除清单从两边一起去掉的实例 {len(ex)} 个：" + ("、".join(ex) or "无") + "。",
         f"- 可疑场景的实例 {len(rep['flagged'])} 个：每个模型的表和配对表都另给去掉它们的版本。", ""]
    for sub, s in rep["subsets"].items():
        L += [f"## {SUBSET_ZH[sub]}（{s['n']} 个）", "", "### 每个模型：带历史 − 不带",
              "", "| 模型 | 带历史 | 不带 | 差（百分点，整场景 95% 区间） | 只带历史成功 | 只不带成功 | McNemar p（未校正） | 改变 N1 的结论（水平：区间不含 0） |",
              "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
        for m, v in s["levels"].items():
            L.append(f"| {m} | {100 * v['rate_a']:.1f}% | {100 * v['rate_b']:.1f}% | {ci(v['scene'])} | "
                     f"{v['only_a']} | {v['only_b']} | {pv(v['p'])} | {'是' if s['level_changed'][m] else '否'} |")
        L += ["", "### 配对：带历史的差、不带的差、带减不带", "",
              "| 配对 | 实例 | 带历史 | 不带 | 带减不带 | 带减不带 p（重抽） | Holm | 改变 N1 的结论 |",
              "| --- | ---: | --- | --- | --- | ---: | ---: | --- |"]
        for k, p in s["pairs"].items():
            fl = p["did"].get("p_floor")
            L.append(f"| {pair_zh(k)} | {p['n']} | {ci(p['with'])} | {ci(p['without'])} | {ci(p['did'])} | "
                     f"{'< ' + pv(p['did']['p']) if fl else pv(p['did']['p'])} | "
                     f"{'≤ ' if fl else ''}{pv(p['did']['holm'])} | {'是' if p['changed'] else '否'} |")
        zh = {"up": "变了（变大）", "down": "变了（变小）", True: "是", False: "否"}
        rules = []
        for k, v in s["rules"].items():
            a, b = k.split(" ")[0].split("-")
            verb = "回来了" if k.endswith("comes back") else "变了"
            rules.append(f"{a} − {b} {verb}：{zh[v] if verb == '回来了' else ('否' if v is False else zh[v])}")
        first, second = [], []
        for k, v in s["gc_condition"]["parts"].items():
            a, b = k.split(" ")[0].split("-")
            p = s["pairs"][f"{a}-{b}"]
            if "with history" in k:
                first.append((f"{a} − {b} {pts(p['with']['mean'])}", v))
            else:
                second.append((f"{a} − {b} {pts(p['did']['mean'])}", v))
        part = lambda xs: ("是" if any(v for _, v in xs) else "否") + "（" + "，".join(t for t, _ in xs) + "）"
        L += ["", "判定（设计第 5 节事先写定；\"回来了\"= 带历史的差和\"带减不带\"的区间都在 0 以上，"
              "\"变了\"= \"带减不带\"的区间不含 0）：" + "；".join(rules) + "。", ""]
        if second:  # the (g-c) condition needs the bases (tier gd only)
            L += ["(g-c) 的条件（11-06 第 1 条，两条满足一条即交）：第一条，RS − S 带历史回到 +3 个百分点以上："
                  + part(first) + "；第二条，训练过的对底座的差变动超过 3 个百分点（\"带减不带\"的点估计）："
                  + part(second) + f"。满足：{'是' if s['gc_condition']['met'] else '否'}。", ""]
    L += ["## 分类型（只作描述）", "", "| 模型 | 类型 | 带历史 | 不带 |", "| --- | --- | ---: | ---: |"]
    for m, t in rep["by_type"]["g"].items():
        for ty, v in sorted(t.items(), key=lambda x: str(x[0])):
            dv = rep["by_type"]["d"].get(m, {}).get(ty, {"k": 0, "n": 0})
            L.append(f"| {m} | {R.TYPE_ZH.get(ty, ty)} | {v['k']}/{v['n']} | {dv['k']}/{dv['n']} |")
    L += ["", "## 可疑场景的实例：每个模型两版的成败", "", "| 模型 | 两版都成 | 只带历史成 | 只不带成 | 都不成 |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for m, e in rep["flagged_episodes"].items():
        c = collections.Counter((v["g"], v["d"]) for v in e.values())
        L.append(f"| {m} | {c[(True, True)]} | {c[(True, False)]} | {c[(False, True)]} | {c[(False, False)]} |")
    L += ["", "## 每轮输入长度（词元，服务端计数；带历史的全部回合，出错行除外；每轮只算走到这一轮的回合）", "",
          "| 模型 | 第 1 轮平均 | 第 6 轮平均 | 第 12 轮平均（最多） | 第 12 轮调用次数 |", "| --- | ---: | ---: | ---: | ---: |"]
    for m, t in rep["prompt_tokens"].items():
        def cell(k, mx=False):
            v = t.get(k)
            return "—" if not v else (f"{v['mean']:.0f}（{v['max']}）" if mx else f"{v['mean']:.0f}")
        L.append(f"| {m} | {cell('1')} | {cell('6')} | {cell('12', True)} | {(t.get('12') or {}).get('calls', 0)} |")
    return "\n".join(L) + "\n"


def relabel(md: str, g_name: str = "带历史", d_name: str = "不带", ref: str = "N1", title=None) -> str:
    """The same tables for another pair of runs (e.g. tier gc against tier gd, both with history): the two sides'
    names, whose conclusions the 'changed' columns refer to, and the title.  "带减不带" (the difference of the
    differences) keeps its meaning: the --g side's difference minus the --d side's."""
    if (g_name, d_name, ref, title) == ("带历史", "不带", "N1", None):
        return md

    def name(x):  # marked, so a space goes in only where the name meets a Chinese character with an ASCII edge
        return f"\x01{x}\x02"
    md = md.replace("带减不带", "\0DID\0").replace("N1 的结论", f"{name(ref)}的结论").replace("N1 主表", f"{name(ref)}主表")
    md = md.replace("带历史", name(g_name)).replace("不带", name(d_name))
    md = md.replace("\0DID\0", f"{name(g_name)}减{name(d_name)}")
    md = re.sub(r"(?<=[一-鿿])\x01(?=[ -~])", " ", md)  # "只 (c) 档成功", "减 (d) 档"
    md = re.sub(r"(?<=[ -~])\x02(?=[一-鿿])", " ", md)  # "(c) − (d) 的结论"
    md = md.replace("\x01", "").replace("\x02", "")
    if title:
        md = md.replace(md.splitlines()[0], f"# {title}", 1)
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g", nargs="+", required=True, help="the with-history folders (<MODEL>/episodes.jsonl)")
    ap.add_argument("--d", nargs="+", required=True, help="the recorded folders without history")
    ap.add_argument("--flag", default=None, help="jsonl of the instances marked apart (instance_id)")
    ap.add_argument("--exclude", nargs="*", default=[], help="n_frames.py scan reports or id files, left out")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--g-name", default="带历史", help="what the --g side is called in the tables")
    ap.add_argument("--d-name", default="不带", help="what the --d side is called in the tables")
    ap.add_argument("--ref-name", default="N1", help="whose conclusions the 'changed' columns refer to")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    flagged = {r["instance_id"] for r in R.read_jsonl(args.flag)} if args.flag else set()
    rep = build(args.g, args.d, flagged, R.excluded(args.exclude), args.models)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(relabel(markdown(rep), args.g_name, args.d_name, args.ref_name, args.title),
                                   encoding="utf-8")
    print(json.dumps({sub: s["gc_condition"] for sub, s in rep["subsets"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
