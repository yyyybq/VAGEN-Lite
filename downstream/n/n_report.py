#!/usr/bin/env python3
"""N1's report (the user's decision 3 of 2026-10-31): the eight models' episodes on the N1 instances.

    python downstream/n/n_report.py --gen GEN --eval EVAL [--models B S RS RD Q3B Q3S Q3RS Q3RD] [--out DIR]
    python downstream/n/n_report.py --gen GEN1 GEN2 --eval EVAL1 EVAL2 [--names N1 N1b] --out DIR

Reads GEN/instances_eval.jsonl (n_gen.py collect) and EVAL/<model>/episodes.jsonl (run_n_episodes.sh).  Per model:
  * successes / episodes per task type and overall, with the 95% Wilson interval;
  * paired against the base model of the same generation (S, RS, RD against B; Q3S, Q3RS, Q3RD against Q3B) and each
    RL start against the SFT start it grew from (RS, RD against S; Q3RS, Q3RD against Q3S), on the instances both
    have: the episodes only one of the two solved, the difference with its 95% interval and McNemar's exact p (the
    overall comparison is the main one, also with a 95% interval from resampling whole scenes; the six task types are
    exploratory, with Holm's adjustment over the six);
  * the same overall comparisons on the near-threshold instances (n_gen: even the most comfortable success pose passes
    some gate only narrowly) and on the rest, apart (the user's decision: no margin, counted apart);
  * behaviour: unreadable turns per turn, blocked actions per executed action, turns per episode, the mean of the
    highest upstream shaping score an episode reached, how episodes ended;
  * instances without an episode, and episodes that failed with an error (left out of every rate).
Several sets (the user's decision 3 of 2026-11-02: N1's 300 and the second set's 300, reported merged as 600): each
GEN with the EVAL in the same place; each set apart, all of them merged (an instance id in two sets is an error), and
with two sets whether they agree on each overall pairing (the two paired differences, independent sets, z test).
Writes DIR/report.json and DIR/report.md (DIR defaults to EVAL for one set; required for several).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n_env as N  # noqa: E402

_spec = importlib.util.spec_from_file_location("ds_report_stats", HERE.parent / "report" / "stats.py")
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)

QWEN25 = ("B", "S", "RS", "RD")
QWEN3 = ("Q3B", "Q3S", "Q3RS", "Q3RD")
MODELS = QWEN25 + QWEN3
TYPE_ZH = {"projective_relations": "左右关系", "equidistance": "等距", "absolute_positioning": "绝对距离",
           "occlusion_alignment": "遮挡", "fov_inclusion": "同框", "screen_occupancy": "画面占比", "all": "全部"}
PAIRINGS = (("versus", "和同一代底座配对"), ("versus_sft", "RL 和 SFT 配对（RS、RD 对 S，Q3RS、Q3RD 对 Q3S）"))


def base_of(model: str) -> str:
    return "B" if model in QWEN25 else "Q3B"


def sft_of(model: str):
    """The SFT start an RL start grew from; None for the others."""
    return {"RS": "S", "RD": "S", "Q3RS": "Q3S", "Q3RD": "Q3S"}.get(model)


def read_jsonl(path: Path) -> list:
    if not Path(path).is_file():
        return []
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def rate(succ: dict, ids) -> dict:
    ids = [i for i in ids if i in succ]
    k = sum(1 for i in ids if succ[i])
    r, lo, hi = stats.wilson(k, len(ids))
    return {"k": k, "n": len(ids), "rate": r, "low": lo, "high": hi}


def compare(succ: dict, base: dict, ids) -> dict:
    keep = set(ids)
    return stats.paired({i: v for i, v in succ.items() if i in keep}, {i: v for i, v in base.items() if i in keep})


def scene_interval(succ: dict, ref: dict, insts: dict, ids, draws: int = 2000, seed: int = 20261102) -> tuple:
    """95% percentile interval of the paired difference with whole scenes resampled (one scene's instances share its
    kitchen, so they are not independent); (nan, nan) without pairs."""
    groups = collections.defaultdict(lambda: [0, 0])
    for i in ids:
        if i in succ and i in ref:
            g = groups[insts[i].get("scene_id", i)]
            g[0] += int(succ[i]) - int(ref[i])
            g[1] += 1
    groups = list(groups.values())
    if not groups:
        return (math.nan, math.nan)
    rng, k, diffs = random.Random(seed), len(groups), []
    for _ in range(draws):
        pick = [groups[rng.randrange(k)] for _ in range(k)]
        diffs.append(sum(p[0] for p in pick) / sum(p[1] for p in pick))
    diffs.sort()
    return (diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1])


def agree(a: dict, b: dict) -> dict:
    """Do two independent sets give the same paired difference?  gap = a's − b's, z test with each set's standard
    error (the one stats.paired's interval uses)."""
    if not a.get("n") or not b.get("n"):
        return {"gap": math.nan, "z": math.nan, "p": math.nan}
    se = [math.sqrt(max(0.0, c["only_a"] + c["only_b"] - (c["only_a"] - c["only_b"]) ** 2 / c["n"])) / c["n"]
          for c in (a, b)]
    gap, s = a["diff"] - b["diff"], math.hypot(*se)
    if s == 0:
        return {"gap": gap, "z": math.nan, "p": 1.0 if gap == 0 else math.nan}
    z = gap / s
    return {"gap": gap, "z": z, "p": math.erfc(abs(z) / math.sqrt(2))}


def behaviour(recs: list) -> dict:
    turns = sum(int(r.get("turns", 0)) for r in recs)
    steps = sum(int(r.get("steps", 0)) for r in recs)
    return {"episodes": len(recs),
            "unreadable_per_turn": sum(int(r.get("invalid_turns", 0)) for r in recs) / turns if turns else math.nan,
            "blocked_per_action": sum(int(r.get("blocked", 0)) for r in recs) / steps if steps else math.nan,
            "turns_per_episode": turns / len(recs) if recs else math.nan,
            "max_shaping_mean": (sum(float(r.get("max_shaping_score") or 0.0) for r in recs) / len(recs)
                                 if recs else math.nan),
            "ends": dict(collections.Counter(r.get("end") for r in recs))}


def load(gen: Path, evals: Path, models=MODELS, exclude=()) -> tuple:
    """(instances by id, model -> its episodes on them) of one set, without the instance ids in exclude."""
    insts = {x["instance_id"]: x for x in read_jsonl(Path(gen) / "instances_eval.jsonl")
             if x["instance_id"] not in exclude}
    recs = {m: [r for r in read_jsonl(Path(evals) / m / "episodes.jsonl") if r.get("instance_id") in insts]
            for m in models}
    return insts, recs


def merge(parts) -> tuple:
    """One (instances, episodes) of several sets; an instance id in two sets is an error."""
    insts, recs = {}, collections.defaultdict(list)
    for gi, ri in parts:
        dup = sorted(set(insts) & set(gi))
        if dup:
            raise ValueError(f"{len(dup)} instance ids in two sets, e.g. {dup[:3]}")
        insts.update(gi)
        for m, rs in ri.items():
            recs[m] += rs
    return insts, dict(recs)


def summarize(insts: dict, recs: dict, models=MODELS) -> dict:
    by_type = collections.defaultdict(list)
    for i, x in insts.items():
        by_type[x["task_type"]].append(i)
    types = [t for t in N.FAMILIES if t in by_type]
    near = [i for i, x in insts.items() if x.get("near_threshold")]
    clear = [i for i, x in insts.items() if not x.get("near_threshold")]
    out = {"instances": len(insts), "per_type": {t: len(by_type[t]) for t in types}, "near_threshold": len(near),
           "models": {}}
    succ = {}
    for m in models:
        ok = [r for r in recs[m] if not r.get("error")]
        succ[m] = {r["instance_id"]: bool(r.get("success")) for r in ok}
        out["models"][m] = {"episodes": len(recs[m]), "errors": [r["instance_id"] for r in recs[m] if r.get("error")],
                            "missing": sorted(set(insts) - {r["instance_id"] for r in recs[m]}),
                            "rates": {t: rate(succ[m], by_type[t]) for t in types},
                            "behaviour": behaviour(ok)}
        out["models"][m]["rates"]["all"] = rate(succ[m], list(insts))
        out["models"][m]["rates_near"] = rate(succ[m], near)
        out["models"][m]["rates_clear"] = rate(succ[m], clear)

    def pair(m: str, ref: str) -> dict:
        per_type = {t: compare(succ[m], succ[ref], by_type[t]) for t in types}
        adj = stats.holm([per_type[t]["p"] for t in types])
        for t, p in zip(types, adj):
            per_type[t]["p_holm"] = p
        whole = compare(succ[m], succ[ref], list(insts))
        whole["scene_low"], whole["scene_high"] = scene_interval(succ[m], succ[ref], insts, list(insts))
        return {"base": ref, "all": whole, "near": compare(succ[m], succ[ref], near),
                "clear": compare(succ[m], succ[ref], clear), "per_type": per_type}

    for m in models:
        b, s = base_of(m), sft_of(m)
        if m != b and succ.get(b):
            out["models"][m]["versus"] = pair(m, b)
        if s and succ.get(s):
            out["models"][m]["versus_sft"] = pair(m, s)
    return out


def build(gen: Path, evals: Path, models=MODELS, exclude=()) -> dict:
    return summarize(*load(gen, evals, models, exclude), models)


def build_sets(sets, models=MODELS, exclude=()) -> dict:
    """sets: [(name, gen, eval)].  Each set apart, all of them merged, and with two sets their agreement."""
    parts = {name: load(g, e, models, exclude) for name, g, e in sets}
    out = {"sets": {n: summarize(*p, models) for n, p in parts.items()},
           "combined": summarize(*merge(parts.values()), models)}
    if len(parts) == 2:
        a, b = (out["sets"][n]["models"] for n in parts)
        out["agreement"] = {m: {k: agree(a[m][k]["all"], b[m][k]["all"]) for k, _ in PAIRINGS
                                if k in a[m] and k in b[m]} for m in models}
    return out


def pct(x, nd=1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}"


def cell(r: dict) -> str:
    return f"{r['k']}/{r['n']}（{pct(r['rate'])}%，{pct(r['low'])}–{pct(r['high'])}）" if r["n"] else "—"


def diff_cell(c: dict) -> str:
    if not c.get("n"):
        return "—"
    return (f"{'+' if c['diff'] >= 0 else '−'}{pct(abs(c['diff']))}（{pct(c['low'])} 到 {pct(c['high'])}；"
            f"{c['only_a']} 对 {c['only_b']}，p = {c['p']:.3g}）")


def gap_cell(a) -> str:
    if not a or math.isnan(a["gap"]):
        return "—"
    p = "—" if math.isnan(a["p"]) else f"{a['p']:.2g}"
    return f"{'+' if a['gap'] >= 0 else '−'}{pct(abs(a['gap']))}（p = {p}）"


def pair_tables(rep: dict, key: str, label: str, h: str) -> list:
    rows = [(m, r[key]) for m, r in rep["models"].items() if key in r]
    if not rows:
        return []
    types = list(rep["per_type"])
    lines = ["", f"{h} {label}（百分点；括号里是 95% 区间、只有一方成功的回合数、精确 McNemar p）", "",
             "| 模型 | 对照 | 全部 | 全部，按场景重抽的 95% 区间 | 门槛附近 | 其余 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for m, v in rows:
        lo, hi = v["all"].get("scene_low", math.nan), v["all"].get("scene_high", math.nan)
        lines.append(f"| {m} | {v['base']} | {diff_cell(v['all'])} | {pct(lo)} 到 {pct(hi)} | {diff_cell(v['near'])} | "
                     f"{diff_cell(v['clear'])} |")
    lines += ["", "按任务类型（探索性，p 是六类内 Holm 校正后的）：", "",
              "| 模型 | 对照 | " + " | ".join(TYPE_ZH[t] for t in types) + " |", "| --- | --- |" + " --- |" * len(types)]
    for m, v in rows:
        row = []
        for t in types:
            c = v["per_type"][t]
            row.append("—" if not c.get("n") else
                       f"{'+' if c['diff'] >= 0 else '−'}{pct(abs(c['diff']))}（{c['only_a']} 对 {c['only_b']}，"
                       f"p = {c['p_holm']:.3g}）")
        lines.append(f"| {m} | {v['base']} | " + " | ".join(row) + " |")
    return lines


def markdown(rep: dict, title: str = None, head: str = "#") -> str:
    types = list(rep["per_type"])
    h = head + "#"
    title = title or f"N1 结果（{rep['instances']} 个实例，门槛附近 {rep['near_threshold']} 个）"
    lines = [f"{head} {title}", "",
             f"{h} 成功率（成功 / 回合，括号里是百分比和 95% 区间）", "",
             "| 模型 | " + " | ".join(TYPE_ZH[t] for t in types + ["all"]) + " |",
             "| --- |" + " ---: |" * (len(types) + 1)]
    for m, r in rep["models"].items():
        lines.append(f"| {m} | " + " | ".join(cell(r["rates"][t]) for t in types + ["all"]) + " |")
    for key, label in PAIRINGS:
        lines += pair_tables(rep, key, label, h)
    lines += ["", f"{h} 行为", "",
              "| 模型 | 回合 | 读不出的轮次占比 | 被挡的动作占比 | 平均轮数 | 最高塑形分（平均） | 结束方式 | 出错 / 缺 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for m, r in rep["models"].items():
        b = r["behaviour"]
        ends = "、".join(f"{k} {v}" for k, v in sorted(b["ends"].items(), key=lambda kv: str(kv[0])))
        lines.append(f"| {m} | {b['episodes']} | {pct(b['unreadable_per_turn'])}% | {pct(b['blocked_per_action'])}% | "
                     f"{b['turns_per_episode']:.1f} | {b['max_shaping_mean']:.3f} | {ends} | "
                     f"{len(r['errors'])} / {len(r['missing'])} |")
    lines += ["", "- 配对：同一个实例上两个模型各跑一回合（贪心解码，所以每个模型每个实例只有一个结果）。"
                  "S、RS、RD 对 B，Q3S、Q3RS、Q3RD 对 Q3B；RS、RD 对 S，Q3RS、Q3RD 对 Q3S。"
                  "差值 = 模型成功率 − 对照成功率；“a 对 b”= 只有模型成功的回合数对只有对照成功的回合数。",
              "- 按场景重抽：同一个场景（同一个任务、同一个重置种子）的几个实例共用一个厨房，不独立；把场景整个有放回地"
              "重抽 2,000 次，取配对差的第 2.5 和第 97.5 百分位。它比按实例算的区间宽多少，反映场景之间的差别有多大。",
              "- 门槛附近：生成实例时，连余量最大的成功位姿也只是勉强过某道门槛（框在画面内 0.5–0.6、可见 0.3%–0.6%、"
              "左右间隔 12–30 像素、夹角 40°–45°、容差的最外 1/5 等，见 `n_env.NEAR_BANDS`）。",
              "- 塑形分：主线判定给的 0–1 分（过所有门槛时 ≥ 0.5），回合里每执行一个动作算一次，取最高。",
              "- 结束方式：success 成功；turns 用满 12 轮；steps 用满 12 个原子动作；invalid 连续 3 轮读不出。"]
    return "\n".join(lines) + "\n"


def markdown_sets(rep: dict) -> str:
    names, comb, agr = list(rep["sets"]), rep["combined"], rep.get("agreement")
    sizes = "、".join(f"{n} {rep['sets'][n]['instances']} 个" for n in names)
    lines = [f"# N 系列合并结果（{comb['instances']} 个实例：{sizes}）", "",
             "## 各套分开看：成功率（全部实例；成功 / 回合，括号里是百分比和 95% 区间）", "",
             "| 模型 | " + " | ".join(names) + " | 合并 |", "| --- |" + " ---: |" * (len(names) + 1)]
    for m, r in comb["models"].items():
        lines.append(f"| {m} | " + " | ".join(cell(rep["sets"][n]["models"][m]["rates"]["all"]) for n in names)
                     + f" | {cell(r['rates']['all'])} |")
    for key, label in PAIRINGS:
        rows = [m for m, r in comb["models"].items() if key in r]
        if not rows:
            continue
        lines += ["", f"## 各套分开看：{label}（全部实例，百分点；括号同下）", "",
                  "| 模型 | 对照 | " + " | ".join(names) + " | 合并 |" + (" 两套之差 |" if agr else ""),
                  "| --- | --- |" + " --- |" * (len(names) + 1 + bool(agr))]
        for m in rows:
            sets_m = [rep["sets"][n]["models"][m] for n in names]
            row = (f"| {m} | {comb['models'][m][key]['base']} | "
                   + " | ".join(diff_cell(s[key]["all"]) if key in s else "—" for s in sets_m)
                   + f" | {diff_cell(comb['models'][m][key]['all'])} |")
            if agr:
                row += f" {gap_cell(agr.get(m, {}).get(key))} |"
            lines.append(row)
    if agr:
        lines += ["", f"- 两套之差：{names[0]} 的配对差减 {names[1]} 的（百分点），两套实例不同，按独立样本做 z 检验；"
                      "p 大只说明看不出两套不一致，不说明一致。"]
    return ("\n".join(lines) + "\n\n"
            + markdown(comb, f"合并的 {comb['instances']} 个实例（门槛附近 {comb['near_threshold']} 个）", "##"))


def excluded(paths) -> set:
    """Instance ids from n_frames.py scan reports (.json: every instance with a corrupted frame in any evaluation) or
    plain files (one id per line)."""
    out = set()
    for p in paths:
        text = Path(p).read_text()
        if str(p).endswith(".json"):
            out |= {i for r in json.loads(text)["evals"].values() for i in r["instances"]}
        else:
            out |= {x.strip() for x in text.splitlines() if x.strip()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, nargs="+", required=True)
    ap.add_argument("--eval", type=Path, nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None, help="one name per set (default: the EVAL folders' names)")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="instance ids to leave out: n_frames.py scan reports (every instance with a corrupted frame) "
                         "or files of one id per line")
    args = ap.parse_args(argv)
    exclude = excluded(args.exclude)
    if len(args.gen) != len(args.eval):
        ap.error("one --eval per --gen")
    if len(args.gen) == 1:
        out = args.out or args.eval[0]
        rep = build(args.gen[0], args.eval[0], args.models, exclude)
        md, summary = markdown(rep), rep
    else:
        if args.out is None:
            ap.error("--out is required with more than one set")
        names = args.names or [e.name for e in args.eval]
        if len(names) != len(args.gen) or len(set(names)) != len(names):
            ap.error("one distinct --names entry per set")
        out = args.out
        rep = build_sets(list(zip(names, args.gen, args.eval)), args.models, exclude)
        md, summary = markdown_sets(rep), rep["combined"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(md)
    print(json.dumps({m: r["rates"]["all"] for m, r in summary["models"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
