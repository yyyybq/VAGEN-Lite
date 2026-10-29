#!/usr/bin/env python3
"""E3 (step 4, the user's decisions of 2026-10-23 evening / 2026-10-26): PPO in RLinf from each start's E2 30% final
checkpoint, 100 updates on 3 spatial tasks, then the same 944-episode evaluation.

Reads, per start, the evaluation before RL (EXPERIMENTS/DS-<start>-030-s1/evals/steps_200000) and after RL
(EXPERIMENTS/E3-<start>/evals/final), the fixed exclusion list left out, and the RLinf training log of the run that
finished (RUN/rlinf_train_<TAG>.log: per update "Global Step: N" and the rollouts' "success_once"), and writes
OUT/results.json and results.md:
  - per start, before and after on the episodes both evaluations completed: all 34 tasks, the 9 spatial tasks, the
    RL tasks and the other tasks; episodes gained / lost and the exact McNemar test; per RL task;
  - each start's change against B's change, task by task (per task: (after - before) of the start minus that of
    B; mean over tasks with a bootstrap 95% interval and Wilcoxon, downstream/report/stats.py), all tasks and the
    RL tasks;
  - the rollouts' success during training: mean "success_once" over updates 1-10, 41-60 and 91-100.

    python downstream/x/e3_analyze.py --experiments downstream_runs/experiments --out downstream_runs/report_e3_20261029
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKTREE))
from x0_qa import mcnemar_exact, wilson  # noqa: E402
from x2_analyze import read_rows  # noqa: E402
from downstream.report import stats  # noqa: E402

STARTS = ("B", "S", "RS", "RD")
RL_TASKS = ("CoffeeSetupMug", "StackBowlsCabinet", "TurnOffStove")
WINDOWS = ((1, 10), (41, 60), (91, 100))


def before_file(experiments: Path, start: str) -> Path:
    return Path(experiments) / f"DS-{start}-030-s1/evals/steps_200000/out/episodes/episodes.jsonl"


def after_file(experiments: Path, start: str) -> Path:
    return Path(experiments) / f"E3-{start}/evals/final/out/episodes/episodes.jsonl"


def training_success(run: Path) -> dict:
    """{update: success_once} from the RLinf log of the training that reached the most updates (a hung first run's
    log is in logs_hung_*/ or an older rlinf_train_*.log with fewer updates)."""
    best = {}
    for log in sorted(Path(run).glob("rlinf_train_*.log")):
        text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        step, seen = None, {}
        for line in text.splitlines():
            m = re.search(r"Global Step:\s*(\d+)", line)
            if m:
                step = int(m.group(1))
            m = re.search(r"success_once=([0-9.eE+-]+)", line)
            if m and step is not None:
                seen[step] = float(m.group(1))
        if len(seen) > len(best):
            best = seen
    return best


def window_means(series: dict) -> dict:
    out = {}
    for lo, hi in WINDOWS:
        vals = [v for k, v in series.items() if lo <= k <= hi]
        out[f"{lo}-{hi}"] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def compare(before: dict, after: dict, keep) -> dict:
    keys = sorted(k for k in set(before) & set(after) if keep(k[0]))
    n = len(keys)
    b, a = sum(before[k] for k in keys), sum(after[k] for k in keys)
    gained = sum(1 for k in keys if after[k] and not before[k])
    lost = sum(1 for k in keys if before[k] and not after[k])
    return {"n": n, "before": b, "after": a, "before_rate": round(b / n, 4) if n else None,
            "after_rate": round(a / n, 4) if n else None, "diff": round((a - b) / n, 4) if n else None,
            "gained": gained, "lost": lost, "mcnemar_p": round(mcnemar_exact(gained, lost), 4)}


def task_changes(before: dict, after: dict) -> dict:
    """{task: (after rate - before rate)} on the episodes both have."""
    per = {}
    for k in set(before) & set(after):
        s = per.setdefault(k[0], [0, 0, 0])
        s[0] += int(before[k])
        s[1] += int(after[k])
        s[2] += 1
    return {t: (a - b) / n for t, (b, a, n) in per.items()}


def _clean(x):
    return None if isinstance(x, float) and math.isnan(x) else (round(x, 4) if isinstance(x, float) else x)


def analyze(experiments: Path, spatial: set, excluded=frozenset(), runs_root: Path = None) -> dict:
    runs_root = Path(runs_root) if runs_root else Path(experiments)
    res = {"starts": {}, "vs_B": {}, "training": {}}
    changes = {}
    for s in STARTS:
        bf, af = before_file(experiments, s), after_file(experiments, s)
        if not (bf.is_file() and af.is_file()):
            continue
        before, after = read_rows(bf, excluded), read_rows(af, excluded)
        res["starts"][s] = {
            "all": compare(before, after, lambda t: True),
            "spatial": compare(before, after, lambda t: t in spatial),
            "rl_tasks": compare(before, after, lambda t: t in RL_TASKS),
            "other_tasks": compare(before, after, lambda t: t not in RL_TASKS),
            "per_rl_task": {t: compare(before, after, lambda x, t=t: x == t) for t in RL_TASKS}}
        changes[s] = task_changes(before, after)
        series = training_success(runs_root / f"E3-{s}")
        res["training"][s] = {"updates": len(series), "success_once": window_means(series)}
    if "B" in changes:
        for s in STARTS[1:]:
            if s not in changes:
                continue
            for name, keep in (("all", lambda t: True), ("rl_tasks", lambda t: t in RL_TASKS)):
                tasks = sorted(t for t in set(changes[s]) & set(changes["B"]) if keep(t))
                diffs = [changes[s][t] - changes["B"][t] for t in tasks]
                tl = stats.task_level(diffs)
                res["vs_B"].setdefault(s, {})[name] = {k: _clean(v) for k, v in tl.items()}
    return res


def _pct(k, n) -> str:
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}%（{k}/{n}，{100 * lo:.1f}–{100 * hi:.1f}）"


def markdown(res: dict) -> str:
    zh = {"all": "34 个任务", "spatial": "9 个空间任务", "rl_tasks": "RL 的 3 个任务", "other_tasks": "其余 31 个任务"}
    lines = ["# E3 RL 前后（自动生成）", "",
             "RL 前 = E2 30% 第 20 万步；RL 后 = PPO 100 次更新后导出的策略；同一套 944 回合评测，只比两边都完成的回合。"
             "p 是逐回合 McNemar 精确检验。", ""]
    for g in ("all", "spatial", "rl_tasks", "other_tasks"):
        lines += [f"## {zh[g]}", "", "| 起点 | RL 前 | RL 后 | 差（百分点） | 变好 / 变坏的回合 | p |",
                  "| --- | ---: | ---: | ---: | --- | ---: |"]
        for s, v in res["starts"].items():
            c = v[g]
            lines.append(f"| {s} | {_pct(c['before'], c['n'])} | {_pct(c['after'], c['n'])} | {100 * c['diff']:+.1f} | "
                         f"{c['gained']} / {c['lost']} | {c['mcnemar_p']} |")
        lines.append("")
    lines += ["## RL 的 3 个任务，逐个", "", "| 起点 | " + " | ".join(RL_TASKS) + " |", "| --- |" + " --- |" * len(RL_TASKS)]
    for s, v in res["starts"].items():
        cells = [f"{c['before']}/{c['n']} → {c['after']}/{c['n']}（p {c['mcnemar_p']}）"
                 for c in (v["per_rl_task"][t] for t in RL_TASKS)]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "## 各起点的变化减去 B 的变化（按任务平均，百分点；括号里是 95% 区间和 Wilcoxon p）", "",
              "| 起点 | 34 个任务 | RL 的 3 个任务 |", "| --- | --- | --- |"]
    for s, v in res["vs_B"].items():
        cells = []
        for g in ("all", "rl_tasks"):
            r = v.get(g) or {}
            if r.get("task_mean_diff") is None:
                cells.append("—")
                continue
            lo, hi, p = r.get("task_low"), r.get("task_high"), r.get("wilcoxon_p")
            ci = f"{100 * lo:+.1f} 到 {100 * hi:+.1f}" if lo is not None else "—"
            cells.append(f"{100 * r['task_mean_diff']:+.1f}（{ci}；p = {p if p is not None else '—'}）")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines += ["", "## 训练时采样回合的成功率（RLinf 的 success_once，按更新区间平均）", "",
              "| 起点 | 记到的更新数 | " + " | ".join(f"第 {w[0]}–{w[1]} 次" for w in WINDOWS) + " |",
              "| --- | ---: |" + " ---: |" * len(WINDOWS)]
    for s, v in res["training"].items():
        m = v["success_once"]
        lines.append(f"| {s} | {v['updates']} | " + " | ".join(
            f"{100 * m[f'{w[0]}-{w[1]}']:.1f}%" if m[f"{w[0]}-{w[1]}"] is not None else "—" for w in WINDOWS) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    from downstream.eval.watch import groups
    _, spatial, _ = groups()
    excluded = {tuple(x) for x in json.loads((WORKTREE / "downstream/eval/excluded_episodes.json").read_text())["episodes"]}
    res = analyze(args.experiments, spatial, excluded)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "results.md").write_text(markdown(res), encoding="utf-8")
    print(json.dumps({s: v["all"]["diff"] for s, v in res["starts"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
