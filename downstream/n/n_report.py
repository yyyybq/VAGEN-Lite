#!/usr/bin/env python3
"""N1's report (the user's decision 3 of 2026-10-31): the eight models' episodes on the N1 instances.

    python downstream/n/n_report.py --gen GEN --eval EVAL [--models B S RS RD Q3B Q3S Q3RS Q3RD] [--out DIR]

Reads GEN/instances_eval.jsonl (n_gen.py collect) and EVAL/<model>/episodes.jsonl (run_n_episodes.sh).  Per model:
  * successes / episodes per task type and overall, with the 95% Wilson interval;
  * paired against the base model of the same generation (S, RS, RD against B; Q3S, Q3RS, Q3RD against Q3B) on the
    instances both have: the episodes only one of the two solved, the difference with its 95% interval and McNemar's
    exact p (the overall comparison is the main one; the six task types are exploratory, with Holm's adjustment over
    the six);
  * the same overall comparison on the near-threshold instances (n_gen: even the most comfortable success pose passes
    some gate only narrowly) and on the rest, apart (the user's decision: no margin, counted apart);
  * behaviour: unreadable turns per turn, blocked actions per executed action, turns per episode, the mean of the
    highest upstream shaping score an episode reached, how episodes ended;
  * instances without an episode, and episodes that failed with an error (left out of every rate).
Writes DIR/report.json and DIR/report.md (DIR defaults to EVAL).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
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


def base_of(model: str) -> str:
    return "B" if model in QWEN25 else "Q3B"


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


def build(gen: Path, evals: Path, models=MODELS) -> dict:
    insts = {x["instance_id"]: x for x in read_jsonl(Path(gen) / "instances_eval.jsonl")}
    by_type = collections.defaultdict(list)
    for i, x in insts.items():
        by_type[x["task_type"]].append(i)
    types = [t for t in N.FAMILIES if t in by_type]
    near = [i for i, x in insts.items() if x.get("near_threshold")]
    clear = [i for i, x in insts.items() if not x.get("near_threshold")]
    out = {"instances": len(insts), "per_type": {t: len(by_type[t]) for t in types}, "near_threshold": len(near),
           "models": {}}
    succ, recs_of = {}, {}
    for m in models:
        recs = [r for r in read_jsonl(Path(evals) / m / "episodes.jsonl") if r.get("instance_id") in insts]
        ok = [r for r in recs if not r.get("error")]
        succ[m] = {r["instance_id"]: bool(r.get("success")) for r in ok}
        recs_of[m] = ok
        out["models"][m] = {"episodes": len(recs), "errors": [r["instance_id"] for r in recs if r.get("error")],
                            "missing": sorted(set(insts) - {r["instance_id"] for r in recs}),
                            "rates": {t: rate(succ[m], by_type[t]) for t in types},
                            "behaviour": behaviour(ok)}
        out["models"][m]["rates"]["all"] = rate(succ[m], list(insts))
        out["models"][m]["rates_near"] = rate(succ[m], near)
        out["models"][m]["rates_clear"] = rate(succ[m], clear)
    for m in models:
        b = base_of(m)
        if m == b or b not in succ or not succ[b]:
            continue
        per_type = {t: compare(succ[m], succ[b], by_type[t]) for t in types}
        adj = stats.holm([per_type[t]["p"] for t in types])
        for t, p in zip(types, adj):
            per_type[t]["p_holm"] = p
        out["models"][m]["versus"] = {"base": b, "all": compare(succ[m], succ[b], list(insts)),
                                      "near": compare(succ[m], succ[b], near),
                                      "clear": compare(succ[m], succ[b], clear), "per_type": per_type}
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


def markdown(rep: dict) -> str:
    types = list(rep["per_type"])
    lines = [f"# N1 结果（{rep['instances']} 个实例，门槛附近 {rep['near_threshold']} 个）", "",
             "## 成功率（成功 / 回合，括号里是百分比和 95% 区间）", "",
             "| 模型 | " + " | ".join(TYPE_ZH[t] for t in types + ["all"]) + " |",
             "| --- |" + " ---: |" * (len(types) + 1)]
    for m, r in rep["models"].items():
        lines.append(f"| {m} | " + " | ".join(cell(r["rates"][t]) for t in types + ["all"]) + " |")
    lines += ["", "## 和同一代底座配对（百分点；括号里是 95% 区间、只有一方成功的回合数、精确 McNemar p）", "",
              "| 模型 | 对照 | 全部 | 门槛附近 | 其余 |", "| --- | --- | --- | --- | --- |"]
    for m, r in rep["models"].items():
        v = r.get("versus")
        if v:
            lines.append(f"| {m} | {v['base']} | {diff_cell(v['all'])} | {diff_cell(v['near'])} | {diff_cell(v['clear'])} |")
    lines += ["", "按任务类型（探索性，p 是六类内 Holm 校正后的）：", "",
              "| 模型 | " + " | ".join(TYPE_ZH[t] for t in types) + " |", "| --- |" + " --- |" * len(types)]
    for m, r in rep["models"].items():
        v = r.get("versus")
        if v:
            row = []
            for t in types:
                c = v["per_type"][t]
                row.append("—" if not c.get("n") else
                           f"{'+' if c['diff'] >= 0 else '−'}{pct(abs(c['diff']))}（{c['only_a']} 对 {c['only_b']}，"
                           f"p = {c['p_holm']:.3g}）")
            lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines += ["", "## 行为", "", "| 模型 | 回合 | 读不出的轮次占比 | 被挡的动作占比 | 平均轮数 | 最高塑形分（平均） | 结束方式 | 出错 / 缺 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for m, r in rep["models"].items():
        b = r["behaviour"]
        ends = "、".join(f"{k} {v}" for k, v in sorted(b["ends"].items(), key=lambda kv: str(kv[0])))
        lines.append(f"| {m} | {b['episodes']} | {pct(b['unreadable_per_turn'])}% | {pct(b['blocked_per_action'])}% | "
                     f"{b['turns_per_episode']:.1f} | {b['max_shaping_mean']:.3f} | {ends} | "
                     f"{len(r['errors'])} / {len(r['missing'])} |")
    lines += ["", "- 配对：同一个实例上两个模型各跑一回合（贪心解码，所以每个模型每个实例只有一个结果）。"
                  "S、RS、RD 对 B，Q3S、Q3RS、Q3RD 对 Q3B。差值 = 模型成功率 − 底座成功率。",
              "- 门槛附近：生成实例时，连余量最大的成功位姿也只是勉强过某道门槛（框在画面内 0.5–0.6、可见 0.3%–0.6%、"
              "左右间隔 12–30 像素、夹角 40°–45°、容差的最外 1/5 等，见 `n_env.NEAR_BANDS`）。",
              "- 塑形分：主线判定给的 0–1 分（过所有门槛时 ≥ 0.5），回合里每执行一个动作算一次，取最高。",
              "- 结束方式：success 成功；turns 用满 12 轮；steps 用满 12 个原子动作；invalid 连续 3 轮读不出。"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or args.eval
    out.mkdir(parents=True, exist_ok=True)
    rep = build(args.gen, args.eval, args.models)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(markdown(rep))
    print(json.dumps({m: r["rates"]["all"] for m, r in rep["models"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
