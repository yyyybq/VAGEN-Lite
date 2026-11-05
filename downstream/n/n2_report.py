#!/usr/bin/env python3
"""N2's report (few-shot; the user's decisions 2-3 of 2026-11-02 and 1 of 2026-11-03): each model's success on the
600 instances against the number of demonstrations it was tuned on.

    python downstream/n/n2_report.py --n2 N2 [--exclude SCAN.json ...] [--out DIR] [--figure]

Tiers: 0 = the model as it is (N1 and its second set: n1_eval_20261031, n1b_eval_20261102), then 100, 300, 1,000
demonstrations (N2/<model>/eval/<n>/episodes.jsonl, n2_train.py's LoRA).  Per model and tier: successes / episodes with
the 95% Wilson interval; paired on the instances (n_report.compare: difference, 95% interval, McNemar's exact p):
  * each tier against the model's own tier 0 (what the demonstrations add);
  * within a tier, each trained start against its generation's base (S, RS, RD against B; Q3S, Q3RS, Q3RD against
    Q3B) and each RL start against its SFT start (RS, RD against S; Q3RS, Q3RD against Q3S);
  * the design's question, fewer demonstrations against more: each trained start at 100 and 300 against its base at
    1,000.
--exclude: n_frames.py scan reports, every instance with a corrupted frame in any of them left out of every model and
tier alike.  --figure: DIR/n2_curve.png and .svg (one panel per generation, Wilson intervals).
Writes DIR/report.json and report.md (DIR defaults to N2).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n_report as R  # noqa: E402

RUNS = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")
TIERS = (0, 100, 300, 1000)
GENERATIONS = (("Qwen2.5-VL-7B", ("B", "S", "RS", "RD")), ("Qwen3-VL-8B", ("Q3B", "Q3S", "Q3RS", "Q3RD")))
ROLE_COLOR = ("#7a8291", "#3569b5", "#d9822b", "#2f9a6a")


def episodes(paths, exclude=()) -> dict:
    succ = {}
    for p in paths:
        for r in R.read_jsonl(p):
            if not r.get("error") and r["instance_id"] not in exclude:
                succ[r["instance_id"]] = bool(r.get("success"))
    return succ


def load(n2: Path, runs: Path = RUNS, exclude=()) -> dict:
    """{model: {tier: {instance: success}}}; tier 0 from N1 and its second set."""
    out = {}
    for m in R.MODELS:
        tiers = {0: episodes([runs / "n/n1_eval_20261031" / m / "episodes.jsonl",
                              runs / "n/n1b_eval_20261102" / m / "episodes.jsonl"], exclude)}
        for n in TIERS[1:]:
            s = episodes([Path(n2) / m / "eval" / str(n) / "episodes.jsonl"], exclude)
            if s:
                tiers[n] = s
        out[m] = tiers
    return out


def pair(a: dict, b: dict) -> dict:
    return R.compare(a, b, sorted(set(a) & set(b)))


def build(data: dict, exclude=()) -> dict:
    rep = {"rates": {}, "vs_tier0": {}, "vs_base": {}, "vs_sft": {}, "fewer_vs_base_1000": {},
           "excluded": sorted(exclude)}
    for m, tiers in data.items():
        rep["rates"][m] = {str(n): R.rate(s, list(s)) for n, s in tiers.items()}
        rep["vs_tier0"][m] = {str(n): pair(s, tiers[0]) for n, s in tiers.items() if n and tiers.get(0)}
        b, sft = R.base_of(m), R.sft_of(m)
        if m != b:
            rep["vs_base"][m] = {str(n): pair(s, data[b][n]) for n, s in tiers.items() if n in data[b]}
            if 1000 in data[b]:
                rep["fewer_vs_base_1000"][m] = {str(n): pair(tiers[n], data[b][1000]) for n in (100, 300)
                                                if n in tiers}
        if sft:
            rep["vs_sft"][m] = {str(n): pair(s, data[sft][n]) for n, s in tiers.items() if n in data[sft]}
    return rep


def cell(c: dict) -> str:
    if not c or not c.get("n"):
        return "—"
    return (f"{100 * c['diff']:+.1f}（{100 * c['low']:+.1f} 到 {100 * c['high']:+.1f}；{c['only_a']} 对 {c['only_b']}；"
            f"p = {c['p']:.2g}）")


def markdown(rep: dict) -> str:
    lines = ["# N2 少样本（自动生成）", "", "成功 / 回合（95% Wilson 区间）。0 档是模型原样（N1 合并的 600 个实例）。", "",
             "| 模型 | " + " | ".join(f"{n} 条" for n in TIERS) + " |", "| --- |" + " ---: |" * len(TIERS)]
    for m, rates in rep["rates"].items():
        cells = []
        for n in TIERS:
            r = rates.get(str(n))
            cells.append(f"{r['k']}/{r['n']}（{100 * r['rate']:.1f}%，{100 * r['low']:.1f}–{100 * r['high']:.1f}）"
                         if r and r["n"] else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    for key, title in (("vs_tier0", "和自己的 0 档配对（示教加了多少）"), ("vs_base", "同一档和同一代底座配对"),
                       ("vs_sft", "同一档 RL 减 SFT"), ("fewer_vs_base_1000", "少示教的训练过的起点对 1,000 条的底座")):
        lines += ["", f"## {title}", "", "差值是百分点（区间；只有前者成功 对 只有后者成功；精确 McNemar p）。", ""]
        cols = sorted({n for v in rep[key].values() for n in v}, key=int)
        lines += ["| 模型 | " + " | ".join(f"{n} 条" for n in cols) + " |", "| --- |" + " --- |" * len(cols)]
        for m, v in rep[key].items():
            lines.append(f"| {m} | " + " | ".join(cell(v.get(n)) for n in cols) + " |")
    if rep["excluded"]:
        lines += ["", f"- 画面坏了、所有模型所有档都剔除的实例：{len(rep['excluded'])} 个。"]
    return "\n".join(lines) + "\n"


def ylabel(rep: dict) -> str:
    """The figure's axis label with the instances it is drawn on (600, or fewer in an excluded version: the most any
    model and tier has)."""
    n = max((r["n"] for by in rep["rates"].values() for r in by.values()), default=0)
    return f"Success on {n} instances (%)"


def figure(rep: dict, out: Path) -> list:
    sys.path.insert(0, str(HERE.parent / "report"))
    import n_figures as NF

    plt = NF.plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    xs = list(range(len(TIERS)))
    for ax, (gen, models) in zip(axes, GENERATIONS):
        for i, m in enumerate(models):
            pts = [(x, rep["rates"][m].get(str(n))) for x, n in zip(xs, TIERS)]
            pts = [(x + (i - 1.5) * 0.06, r) for x, r in pts if r and r["n"]]
            if not pts:
                continue
            p = [100 * r["rate"] for _, r in pts]
            ax.errorbar([x for x, _ in pts], p, yerr=[[a - 100 * r["low"] for a, (_, r) in zip(p, pts)],
                                                      [100 * r["high"] - a for a, (_, r) in zip(p, pts)]],
                        color=ROLE_COLOR[i], marker="o", ms=4, lw=1.4, elinewidth=0.8, capsize=2, label=m)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in TIERS])
        ax.set_xlabel("Demonstrations (LoRA)")
        ax.set_title(gen, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e4e7eb", lw=0.7)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel(ylabel(rep))
    fig.tight_layout()
    files = NF.save(fig, out, "n2_curve")
    plt.close(fig)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n2", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args(argv)
    exclude = R.excluded(args.exclude)
    rep = build(load(args.n2, exclude=exclude), exclude)
    out = args.out or args.n2
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (out / "report.md").write_text(markdown(rep), encoding="utf-8")
    if args.figure:
        figure(rep, out)
    print(json.dumps({m: {n: r["k"] for n, r in v.items()} for m, v in rep["rates"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
