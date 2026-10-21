#!/usr/bin/env python3
"""Build the result report of the downstream line (login node, CPU; the vagen-lite Python has matplotlib / scipy).

    python -m downstream.report.build --out RUNS/report                      # real runs (downstream_runs)
    python -m downstream.report.build --runs-root /tmp/sim --out /tmp/sim/report   # simulated (report.simulate)

Writes OUT/index.html (self-contained: figures and example strips embedded; publishable as an Artifact),
OUT/figures/*.png, OUT/results.json (every number on the page) and OUT/report.md (summary and figure links).

Comparison points:
  * D5: the latest checkpoint at which every start of the 100% / seed-1 runs has an evaluation (all present
    starts if none is common yet);
  * step 3: step 200,000 for each data share, over the training seeds whose evaluation exists for every start.
The main comparison (tables, per-task figures, examples) is step 3 at 100% when it exists, else D5.  All
comparisons are paired against the reference start B; see the glossary on the page.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path

from . import collect, examples, page, stats

FINAL_STEP = 200000
RUNS_ROOT = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")


def jsonable(x):
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 6)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def code_version() -> str:
    try:
        res = subprocess.run(["git", "-C", str(collect.DOWNSTREAM), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return res.stdout.strip() or None
    except Exception:
        return None


def d5_point(coll, ratio="100", seed=1):
    steps = stats.common_steps(coll, ratio, [seed])
    if not steps:
        steps = sorted({e.step for e in coll.first() if e.ratio == ratio and e.seed == seed})
    return steps[-1] if steps else None


def final_seeds(coll, ratio):
    have = {}
    for e in coll.first():
        if e.ratio == ratio and e.step == FINAL_STEP:
            have.setdefault(e.seed, set()).add(e.start)
    starts = set(coll.starts())
    full = sorted(k for k, got in have.items() if got >= starts)
    return full or sorted(have)


def noise_rows(coll):
    first = {(e.exp, e.step): e for e in coll.first()}
    out = []
    for r in coll.evals:
        if r.kind != "repeat" or (r.exp, r.step) not in first:
            continue
        a = first[(r.exp, r.step)]
        oa = {(x["task"], x["eval_seed"]): x["success"] for x in a.episodes}
        ob = {(x["task"], x["eval_seed"]): x["success"] for x in r.episodes}
        common = set(oa) & set(ob)
        sa, sb = a.summary.get("episode_wall_mean_seconds") or 0.0, r.summary.get("episode_wall_mean_seconds") or 0.0
        out.append({"exp": a.exp, "step": a.step, "workers": [a.workers, r.workers],
                    "successes": [a.successes, r.successes], "episodes": [len(a.episodes), len(r.episodes)],
                    "rate_diff": r.successes / max(1, len(r.episodes)) - a.successes / max(1, len(a.episodes)),
                    "changed": sum(oa[k] != ob[k] for k in common), "compared": len(common),
                    "seconds": [sa, sb], "slowdown": sb / sa if sa else math.nan})
    return out


def describe(cmp, what: str) -> list:
    """Plain statements of the numbers of one comparison (no interpretation)."""
    if not cmp or not cmp["starts"]:
        return []
    lines = []
    for g in ("all", "spatial"):
        parts = []
        for s in cmp["starts"]:
            k, n = cmp["groups"][s][g]
            r, lo, hi = stats.wilson(k, n)
            parts.append(f"{s} {k}/{n}（{page.pct(r)}，区间 {page.pct(lo)}–{page.pct(hi)}）")
        lines.append(f"{what}，{stats.GROUP_ZH[g]}的成功回合 / 回合：" + "；".join(parts) + "。")
        diffs = []
        for s, v in cmp["vs_reference"].items():
            d = v["groups"][g]
            diffs.append(f"{s} {page.pts(d['diff'])} 个百分点（区间 {page.pts(d['low'])} 到 {page.pts(d['high'])}，"
                         f"McNemar p = {page.pval(d['p'])}；{d['tasks_up']} 个任务高于 {cmp['reference']}、"
                         f"{d['tasks_down']} 个低于，Wilcoxon p = {page.pval(d['wilcoxon_p'])}）")
        if diffs:
            lines.append(f"{what}，{stats.GROUP_ZH[g]}与 {cmp['reference']} 的配对差：" + "；".join(diffs) + "。")
    return lines


def build(runs_root: Path, out: Path, reference: str = "B") -> dict:
    from . import figures as F

    coll = collect.collect(runs_root)
    simulated = (Path(runs_root) / "SIMULATED").exists()
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    d5_step = d5_point(coll)
    d5 = stats.compare(coll, "100", d5_step, [1], reference) if d5_step else None
    finals = {r: stats.compare(coll, r, FINAL_STEP, final_seeds(coll, r), reference)
              for r in ("010", "030", "100") if final_seeds(coll, r)}
    seed_tables = {r: stats.seed_table(coll, r, FINAL_STEP) for r in ("010", "030", "100")}
    main = finals.get("100") or d5
    if main is not None and main is finals.get("100"):
        main_label = f"第三步，100% 数据，第 {FINAL_STEP // 1000}k 步，训练种子 {','.join(map(str, main['seeds']))}"
    elif main is not None:
        main_label = f"D5，100% 数据，训练种子 1，第 {d5_step // 1000}k 步检查点"
    else:
        main_label = "还没有评测结果"
    lines = []
    if not coll.first():
        lines.append("还没有完成的闭环评测。训练损失见下文；评测完成后重新生成本页即可。")
    if d5:
        lines += describe(d5, f"D5 第 {d5_step // 1000}k 步（100% 数据、训练种子 1）")
        lines.append("这是中间检查点：学习率按 20 万步余弦衰减，此时仍高，结果只用来看成功率何时离开地板，"
                     "不能当作更短训练的结果。")
    for r, cmp in finals.items():
        lines += describe(cmp, f"第三步 {collect.RATIO_PERCENT[r]}% 数据第 {FINAL_STEP // 1000}k 步"
                               f"（训练种子 {','.join(map(str, cmp['seeds']))}）")
    noise = noise_rows(coll)
    for x in noise:
        lines.append(f"评测噪声：{x['exp']} 第 {x['step'] // 1000}k 步评两遍（并行进程 {x['workers'][0]} 与 "
                     f"{x['workers'][1]}），成功率差 {page.pts(x['rate_diff'])} 个百分点，{x['changed']}/{x['compared']} "
                     f"个回合结果不同；平均每回合 {x['seconds'][0]:.0f} 与 {x['seconds'][1]:.0f} 秒（{x['slowdown']:.2f} 倍）。")

    figs = {}

    def save(name, fig):
        if fig is not None:
            path = out / "figures" / f"{name}.png"
            fig.savefig(path)
            F.plt.close(fig)
            figs[name] = path
    save("loss", F.loss_curves(coll, reference=reference))
    save("checkpoints", F.checkpoint_curves(coll))
    save("learning", F.learning_curves(coll, FINAL_STEP))
    if main:
        save("heatmap", F.task_heatmap(coll, main))
        save("forest", F.forest(coll, main))
        save("agreement", F.agreement(coll, main))
        save("efficiency", F.efficiency(coll, main["ratio"], main["step"], main["seeds"]))
    save("noise", F.noise(coll))
    picked = examples.pick(coll, main) if main else []

    results = {
        "generated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "simulated": simulated, "code": code_version(), "runs_root": str(runs_root),
        "counts": {"experiments": len(coll.experiments), "evaluations": len(coll.first()),
                   "episodes": sum(len(e.episodes) for e in coll.first())},
        "reference": reference, "d5_step": d5_step, "d5": d5, "finals": finals, "final_step": FINAL_STEP,
        "seed_tables": seed_tables, "main_cmp": main, "main_label": main_label,
        "task_order": F.task_order(coll, main) if main else [], "noise": noise, "summary_lines": lines,
        "examples_rule": ("范例按固定规则挑选，不凭观感：同一场景（同任务、同评测种子、同训练种子）里起点结果不一致的回合，"
                          "每个起点各取同样多个“它成功、B 失败”和“B 成功、它失败”，空间任务优先，再按任务名和种子排序；"
                          "另加一个全部成功、一个全部失败的场景。每条画面是该回合等间隔的 8 帧（左侧相机），"
                          "下方色条绿为成功、红为失败。"),
        "examples": [{k: v for k, v in x.items() if k != "per_start"} | {"starts": list(x["per_start"])} for x in picked],
        "sources": [f"实验目录：{runs_root}/experiments/DS-<起点>-<数据比例>-s<训练种子>",
                    "评测协议：docs/downstream/EVAL_PROTOCOL.md（950 回合、评测种子、台面区域顺序固定）",
                    "每次评测：evals/steps_<步数>/out/episodes/{summary.json, episodes.jsonl, frames/}",
                    "统计与图：downstream/report/（stats.py、figures.py、examples.py）",
                    f"本页数字的原始值：{out}/results.json"],
    }
    (out / "results.json").write_text(json.dumps(jsonable(results), ensure_ascii=False, indent=1) + "\n")
    html_text = page.render(results, coll, figs, picked)
    (out / "index.html").write_text(html_text, encoding="utf-8")
    md = [f"# 下游起点对照（{results['generated']}）", ""]
    if simulated:
        md += ["> 模拟数据：所有数字都是编出来的，只用来预览。", ""]
    md += [f"- {x}" for x in lines] + [""] + [f"![{k}](figures/{p.name})" for k, p in figs.items()]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"out": str(out), "figures": sorted(figs), "examples": len(picked), "html_bytes": len(html_text.encode()),
            "simulated": simulated, "evaluations": results["counts"]["evaluations"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", default="B")
    args = ap.parse_args(argv)
    print(json.dumps(build(args.runs_root, args.out, args.reference)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
