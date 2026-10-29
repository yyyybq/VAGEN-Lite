#!/usr/bin/env python3
"""X2 (the user's decision of 2026-10-26): very little demonstration data, 1% and 3% (5 and 15 demonstrations per
task, nested in 10%), 2 epochs, each final checkpoint evaluated with the 944-episode protocol.

Reads EXPERIMENTS/X2-<start>-{001,003}-s1/evals/steps_<N>/out/episodes/episodes.jsonl and, for the same comparison
at more data, the final checkpoints of E2 (DS-<start>-{010,030}-s1) and E1 (DS-<start>-100-s1, the final 200k only:
the E1 report's main numbers are three-point means), and writes OUT/results.json and results.md:
  - per start and ratio the success rate overall, single-step, multi-step and spatial group (Wilson 95% intervals);
  - per ratio each start against B task by task (the task as the pairing unit, as in E1: mean difference with a
    bootstrap 95% interval over tasks, Wilcoxon signed-rank; downstream/report/stats.py), all 34 tasks and the 9
    spatial tasks, on the episodes both evaluations completed;
  - the design's question: are the starts' differences larger with little data than with 10%, 30%, 100%?  (Those
    three ran a fixed 200,000 steps, not 2 epochs: the curve mixes data and steps.)

    python downstream/x/x2_analyze.py --experiments downstream_runs/experiments --out downstream_runs/x/X2/results
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKTREE))
from x0_qa import wilson  # noqa: E402
from downstream.report import stats  # noqa: E402

STARTS = ("B", "S", "RS", "RD")
RATIOS = ("001", "003", "010", "030", "100")
GROUPS = ("all", "atomic_seen", "composite_seen", "spatial")
GROUP_ZH = {"all": "总体", "atomic_seen": "单步", "composite_seen": "多步", "spatial": "空间组"}


def experiment(ratio: str, start: str) -> str:
    return f"X2-{start}-{ratio}-s1" if ratio in ("001", "003") else f"DS-{start}-{ratio}-s1"


def episodes_file(experiments: Path, ratio: str, start: str):
    """The final checkpoint's episodes: X2's only evaluation; E1 / E2 at 200,000 steps."""
    root = Path(experiments) / experiment(ratio, start) / "evals"
    found = sorted(root.glob("steps_*/out/episodes/episodes.jsonl"), key=lambda p: int(p.parts[-4].split("_")[1]))
    if ratio not in ("001", "003"):
        found = [p for p in found if p.parts[-4] == "steps_200000"]
    return found[-1] if found else None


def read_rows(path: Path, excluded=frozenset()) -> dict:
    """{(task, seed): success} of the complete episodes, the fixed exclusion list left out."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        key = (r.get("task"), int(r.get("seed", -1)))
        if r.get("status") == "complete" and key not in excluded:
            out[key] = bool(r["success"])
    return out


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [lo, hi]}


def _clean(x):
    return None if isinstance(x, float) and math.isnan(x) else (round(x, 4) if isinstance(x, float) else x)


def in_group(task: str, group: str, group_of: dict, spatial: set) -> bool:
    return group == "all" or (group == "spatial" and task in spatial) or group_of.get(task) == group


def analyze(experiments: Path, group_of: dict, spatial: set, excluded=frozenset()) -> dict:
    runs = {}
    for ratio in RATIOS:
        for s in STARTS:
            p = episodes_file(experiments, ratio, s)
            if p is not None:
                runs[(ratio, s)] = read_rows(p, excluded)
    res = {"rates": {}, "vs_B": {}}
    for (ratio, s), rows in runs.items():
        res["rates"].setdefault(ratio, {})[s] = {
            g: rate(sum(v for k, v in rows.items() if in_group(k[0], g, group_of, spatial)),
                    sum(1 for k in rows if in_group(k[0], g, group_of, spatial))) for g in GROUPS}
    for ratio in RATIOS:
        ref = runs.get((ratio, "B"))
        if ref is None:
            continue
        for s in STARTS[1:]:
            mine = runs.get((ratio, s))
            if mine is None:
                continue
            for g in ("all", "spatial"):
                a = {k: v for k, v in ref.items() if in_group(k[0], g, group_of, spatial)}
                b = {k: v for k, v in mine.items() if in_group(k[0], g, group_of, spatial)}
                tl = stats.task_level(stats.common_task_diffs(a, b))
                res["vs_B"].setdefault(ratio, {}).setdefault(s, {})[g] = {k: _clean(v) for k, v in tl.items()}
    return res


def _pct(r) -> str:
    """From k / n itself, not the rate rounded to 4 places: 42 / 944 = 4.449% is 4.4%, the rounded 0.0445 printed 4.5%
    (the first version of the X2 report, corrected 10-29)."""
    if not r or not r.get("n"):
        return "—"
    lo, hi = r["ci95"]
    return f"{100 * r['k'] / r['n']:.1f}%（{r['k']}/{r['n']}，{100 * lo:.1f}–{100 * hi:.1f}）"


def _diff(r) -> str:
    if not r or r.get("task_mean_diff") is None:
        return "—"
    ci = f"{100 * r['task_low']:+.1f} 到 {100 * r['task_high']:+.1f}" if r.get("task_low") is not None else "—"
    p = r.get("wilcoxon_p")
    return f"{100 * r['task_mean_diff']:+.1f}（{ci}；p = {p if p is not None else '—'}）"


RATIO_ZH = {"001": "1%", "003": "3%", "010": "10%", "030": "30%", "100": "100%"}


def markdown(res: dict) -> str:
    lines = ["# X2 极低数据（自动生成）", "",
             "1%、3% 是 X2（2 个 epoch：10,368 步、32,080 步）；10%、30%、100% 是 E2、E1 的最终检查点（固定 20 万步，"
             "100% 只取第 20 万步，不是 E1 报告里的三点平均）。成功率后面括号里是成功回合 / 回合和 95% 区间。", ""]
    for ratio in ("001", "003"):
        if ratio not in res["rates"]:
            continue
        lines += [f"## {RATIO_ZH[ratio]}：成功率", "", "| 起点 | " + " | ".join(GROUP_ZH[g] for g in GROUPS) + " |",
                  "| --- |" + " ---: |" * len(GROUPS)]
        for s in STARTS:
            r = res["rates"][ratio].get(s)
            if r:
                lines.append(f"| {s} | " + " | ".join(_pct(r[g]) for g in GROUPS) + " |")
        lines.append("")
    lines += ["## 和 B 逐任务比（起点 − B，按任务平均，百分点；括号里是 95% 区间和 Wilcoxon p）", "",
              "| 数据量 | 起点 | 34 个任务 | 9 个空间任务 |", "| --- | --- | --- | --- |"]
    for ratio in RATIOS:
        for s, v in res["vs_B"].get(ratio, {}).items():
            lines.append(f"| {RATIO_ZH[ratio]} | {s} | {_diff(v.get('all'))} | {_diff(v.get('spatial'))} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from downstream.eval.watch import groups
    group_of, spatial, _ = groups()
    excluded = {tuple(x) for x in json.loads((WORKTREE / "downstream/eval/excluded_episodes.json").read_text())["episodes"]}
    res = analyze(args.experiments, group_of, spatial, excluded)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "results.md").write_text(markdown(res), encoding="utf-8")
    print(json.dumps({RATIO_ZH[r]: {s: v["all"]["rate"] for s, v in d.items()} for r, d in res["rates"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
