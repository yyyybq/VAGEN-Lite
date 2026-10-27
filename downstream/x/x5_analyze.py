#!/usr/bin/env python3
"""X5 (the user's decision 7 of 2026-10-26): the E1 final checkpoints on tasks our SFT never saw.

Reads X5/<start>/out/episodes/episodes.jsonl (the evaluation of downstream/x/x5_plan.json: 32 tasks x 20 episodes)
and writes X5/results/results.json and results.md: per start and group (unseen_atomic, the main result;
unseen_composite, the supplement) the success rate (Wilson 95% interval) and per task; each start against B task by
task (the task as the pairing unit, as in E1: mean difference with a bootstrap 95% interval over tasks, Wilcoxon
signed-rank and sign tests; downstream/report/stats.py).  A group where no start succeeds at all is reported as
"none of the starts can do these tasks" and not compared (the design's rule).

    python downstream/x/x5_analyze.py --x5 downstream_runs/x/X5
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
TASKS = HERE / "x5_tasks.json"
GROUPS = ("unseen_atomic", "unseen_composite")


def read_rows(path: Path) -> dict:
    """{(task, seed): success} of the complete episodes (a later row of a key wins: reruns)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("status") == "complete":
            out[(r["task"], int(r["seed"]))] = bool(r["success"])
    return out


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [lo, hi]}


def _clean(x):
    return None if isinstance(x, float) and math.isnan(x) else (round(x, 4) if isinstance(x, float) else x)


def analyze(x5: Path, tasks_file: Path = TASKS) -> dict:
    group_of = {t["task"]: t["group"] for t in json.loads(Path(tasks_file).read_text())["tasks"]}
    runs = {s: read_rows(x5 / s / "out/episodes/episodes.jsonl") for s in STARTS
            if (x5 / s / "out/episodes/episodes.jsonl").is_file()}
    res = {"starts": {}, "vs_B": {}, "nobody": {}}
    for s, rows in runs.items():
        res["starts"][s] = {}
        for g in GROUPS:
            mine = {k: v for k, v in rows.items() if group_of.get(k[0]) == g}
            per = {}
            for (t, _), ok in mine.items():
                per.setdefault(t, [0, 0])
                per[t][0] += int(ok)
                per[t][1] += 1
            res["starts"][s][g] = dict(rate(sum(mine.values()), len(mine)),
                                       per_task={t: {"k": k, "n": n} for t, (k, n) in sorted(per.items())})
    for g in GROUPS:
        res["nobody"][g] = bool(runs) and all(res["starts"][s][g]["k"] == 0 for s in runs)
        if res["nobody"][g] or "B" not in runs:
            continue
        ref = {k: v for k, v in runs["B"].items() if group_of.get(k[0]) == g}
        for s in runs:
            if s == "B":
                continue
            mine = {k: v for k, v in runs[s].items() if group_of.get(k[0]) == g}
            tl = stats.task_level(stats.common_task_diffs(ref, mine))
            res["vs_B"].setdefault(s, {})[g] = {k: _clean(v) for k, v in tl.items()}
    out = x5 / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "results.md").write_text(markdown(res), encoding="utf-8")
    return res


def _pct(r) -> str:
    if not r or not r.get("n"):
        return "—"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}%（{r['k']}/{r['n']}，{100 * lo:.1f}–{100 * hi:.1f}）"


def markdown(res: dict) -> str:
    zh = {"unseen_atomic": "未见原子任务（16 个，主评）", "unseen_composite": "未见组合任务（16 个，补评）"}
    lines = ["# X5 未见任务（自动生成）", "", "成功率后面括号里是成功回合 / 回合和 95% 区间。", "",
             "| 起点 | " + " | ".join(zh[g] for g in GROUPS) + " |", "| --- |" + " ---: |" * len(GROUPS)]
    for s, st in res["starts"].items():
        lines.append(f"| {s} | " + " | ".join(_pct(st[g]) for g in GROUPS) + " |")
    for g in GROUPS:
        lines += ["", f"## {zh[g]}：和 B 逐任务比（起点 − B，按任务平均，百分点）", ""]
        if res["nobody"].get(g):
            lines.append("四个起点在这组任务上一次都没成功：未见任务上都不会，不做比较。")
            continue
        lines += ["| 起点 | 任务数 | 平均差（95% 区间） | 升 / 降的任务 | Wilcoxon p |", "| --- | ---: | ---: | --- | ---: |"]
        for s, v in res["vs_B"].items():
            r = v.get(g)
            if not r:
                continue
            ci = (f"{100 * r['task_low']:+.1f}–{100 * r['task_high']:+.1f}" if r.get("task_low") is not None else "—")
            lines.append(f"| {s} | {r['tasks']} | {100 * r['task_mean_diff']:+.1f}（{ci}） | "
                         f"{r['tasks_up']} / {r['tasks_down']} | {r['wilcoxon_p'] if r['wilcoxon_p'] is not None else '—'} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x5", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, default=TASKS)
    args = ap.parse_args(argv)
    res = analyze(args.x5, args.tasks)
    print(json.dumps({s: {g: v[g]["rate"] for g in GROUPS} for s, v in res["starts"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
