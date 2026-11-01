#!/usr/bin/env python3
"""V4 recheck (the user's decision of 2026-10-30, "V4复核先做两个"; asked by the main window on 2026-10-29): two
more Qwen3-VL-8B upstream starts through the same E1-protocol downstream run as the model line's DS-Q3RD-100-s1
(100% data, 200,000 steps, full state every 25,000, the final checkpoint evaluated inside the job, 150k and 175k by
the model line's watch_points), to tell whether DS-Q3RD's drop against DS-Q3B (-3.2 points, p = 0.032) belongs to the
V4 recipe or to that one checkpoint:
    Q3V4S3   V4, seed 20261022, update 250 (705 questions 49.9%)
    Q3DCKS2  DCK, seed 20261021, update 300 (705 questions 51.2%)
The configs come from the model line's builder (model_matrix/downstream/make_config.build, read only, by path) with
its start registry pointed at this recheck's own (RUNS/v4_recheck/starts.json), so they differ from DS-Q3RD-100-s1
only in the start; the job's payload is the model line's run_full.sh (its Qwen3-VL entry; our train_entry.py and
run_eval_ckpt.sh underneath).  Nothing of the model line is changed.

    python downstream/x/v4_recheck.py make            # experiments, prepared 8-card jobs, queue entries q11_v4_*
    python downstream/x/v4_recheck.py report          # RUNS/v4_recheck/report.{json,md}

The report (the user's decision 6 of 2026-10-31: "评三点，和模型线的 DS-Q3B、DS-Q3RD 逐任务配对") reads the model
line's DS-Q3B-100-s1 and DS-Q3RD-100-s1 (read only) and this recheck's two runs as one collection and compares every
start with each of the two on E1's three-point mean (report.stats.compare_mean: per task the mean over 150k, 175k and
200k of the paired difference on the episodes both have; bootstrap interval and Wilcoxon over the tasks; the pooled
difference at each step with McNemar's p).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
RUNS = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")
MWT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models")
EXPS = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial"
DATA = RUNS / "data" / "target50_100"
STARTS = {
    "Q3V4S3": {"role": "main", "model_type": "qwen3_vl",
               "desc": "Qwen3-VL-8B, dense RL V4, seed 20261022, update 250 picked by validation (705 questions 49.9%); "
                       "the V4 recheck the user asked for on 2026-10-30",
               "path": f"{EXPS}/interiorgs_v10_train_q3s3_20261027/rl_run_qw3v4s3/checkpoints/global_step_250/actor/"
                       "huggingface"},
    "Q3DCKS2": {"role": "main", "model_type": "qwen3_vl",
                "desc": "Qwen3-VL-8B, dense RL DCK, seed 20261021, update 300 picked by validation (705 questions "
                        "51.2%); the V4 recheck the user asked for on 2026-10-30",
                "path": f"{EXPS}/interiorgs_v10_train_q3s2dck_20261027/rl_run_qw3dcks2/checkpoints/global_step_300/"
                        "actor/huggingface"},
}
PROTOCOL = {"steps": 200000, "save_interval": 25000, "log_every": 100, "seed": 1, "full_state_interval": 25000,
            "eval_points": [200000]}  # DS-Q3RD-100-s1's record
MODELS_RUNS = Path("/mnt/umm/users/yinbaiqiao/models_runs/downstream")  # the model line's runs root, read only
CONTROLS = ("Q3B", "Q3RD")
CONTROL_DESC = {"Q3B": "原版 Qwen3-VL-8B（模型线）", "Q3RD": "V4 稠密奖励 RL 第 300 次（模型线的 RD 起点）"}
START_DESC = {"Q3V4S3": "V4 换种子 20261022，第 250 次", "Q3DCKS2": "DCK 种子 20261021，第 300 次"}
POINTS = (150000, 175000, 200000)
GROUP_ROWS = ("all", "atomic_seen", "composite_seen", "spatial", "other")


def exp_id(start: str) -> str:
    return f"DS-{start}-100-s1"


def write_registry(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict({"_doc": "starts of the V4 recheck (downstream/x/v4_recheck.py)"}, **STARTS),
                               indent=1) + "\n")
    return path


def queue_entry(qdir: Path, i: int, start: str, run: Path) -> Path:
    q = qdir / f"q11_v4_{i}_{start}.json"
    q.write_text(json.dumps({"run": str(run), "cards": 8, "what": f"V4 复核 {start}（Qwen3，E1 协议）",
                             "pools": ["zoetrope", "h800"]}, ensure_ascii=False, indent=1) + "\n")
    return q


def make(starts, runs: Path = RUNS, make_config=None, sco_job=None) -> list:
    """Builds each start's experiment (skipped when its experiment.json exists), prepares its job, queues it."""
    if make_config is None:
        sys.path.insert(0, str(MWT))
        from model_matrix.downstream import make_config
    if sco_job is None:
        sys.path.insert(0, str(DOWNSTREAM))
        import sco_job
    make_config.STARTS = write_registry(runs / "v4_recheck" / "starts.json")
    out = []
    for i, start in enumerate(starts, start=1):
        exp = runs / "experiments" / exp_id(start)
        if not (exp / "experiment.json").is_file():
            make_config.build(start, DATA, exp, PROTOCOL["steps"], PROTOCOL["save_interval"], PROTOCOL["log_every"],
                              PROTOCOL["seed"], full_state_interval=PROTOCOL["full_state_interval"],
                              eval_points=PROTOCOL["eval_points"])
        if not (exp / "job" / "prepared.json").is_file():
            sco_job.prepare(exp, 8, "zoetrope", f"v4-{start.lower()}",
                            ["bash", str(MWT / "model_matrix/downstream/run_full.sh"), str(exp)])
        out.append(str(queue_entry(runs / "queue", i, start, exp)))
    return out


def _report_modules():
    sys.path.insert(0, str(DOWNSTREAM.parent))
    from downstream.report import collect, stats
    return collect, stats


def merged(roots, starts):
    """One report.collect.Collection of the experiments of `starts` (ratio 100, seed 1) over several runs roots."""
    collect, _ = _report_modules()
    exps, loss, evs, tasks = {}, {}, [], None
    for root in roots:
        c = collect.collect(Path(root))
        tasks = c.tasks
        for name, parsed in c.experiments.items():
            if parsed[0] not in starts or parsed[1:] != ("100", 1):
                continue
            if name in exps:
                raise SystemExit(f"{name} is in more than one runs root")
            exps[name], loss[name] = parsed, c.losses[name]
            evs += [e for e in c.evals if e.exp == name]
    return collect.Collection(Path(roots[0]), tasks, exps, loss, evs)


def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def _pts(x) -> str:
    return f"{100 * x:+.1f}"


def _p(x) -> str:
    return f"{x:.2g}" if x >= 0.001 else f"{x:.1e}"


def report(out: Path, roots=(MODELS_RUNS, RUNS), starts=CONTROLS + tuple(STARTS)) -> dict:
    """Writes out/report.json and out/report.md; stops when a start lacks a finished evaluation at one of POINTS."""
    _, stats = _report_modules()
    coll = merged(roots, starts)
    first = coll.first()
    lacking = [(s, p) for s in starts for p in POINTS
               if not any(e.start == s and e.step == p and e.ratio == "100" for e in first)]
    if lacking:
        raise SystemExit(f"no finished evaluation for {lacking}")
    counts = {s: [[sum(e.successes for e in first if (e.start, e.step) == (s, p)),
                   sum(len(e.episodes) for e in first if (e.start, e.step) == (s, p))] for p in POINTS] for s in starts}
    vs = {ref: stats.compare_mean(coll, "100", POINTS, [1], reference=ref) for ref in CONTROLS}
    m = vs[CONTROLS[0]]
    res = {"points": list(POINTS), "starts": list(starts), "counts": counts,
           "groups": {s: m["groups"][s] for s in starts}, "tasks": {s: m["tasks"][s] for s in starts},
           "vs": {ref: {s: vs[ref]["vs_reference"][s]["groups"] for s in starts if s != ref} for ref in CONTROLS},
           "group_zh": stats.GROUP_ZH}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
    (out / "report.md").write_text("\n".join(report_lines(res, {**CONTROL_DESC, **START_DESC})) + "\n")
    return res


def report_lines(res, desc) -> list:
    starts, pts = res["starts"], res["points"]
    head = " / ".join(f"第 {p // 1000}k 步" for p in pts)
    L = ["# V4 复核：逐任务配对（数字表）", "",
         f"E1 协议，100% 示教 20 万步，训练种子 1；每个点评 34 个任务约 944 回合（{head}），主数字是三点平均。", "",
         "## 成功率（全部 34 个任务，成功回合 / 回合）", "",
         "| 起点 | 是什么 | " + " | ".join(f"{p // 1000}k" for p in pts) + " | 三点平均 |", "| --- | --- |" + " --- |" * (len(pts) + 1)]
    for s in starts:
        cells = [f"{k}/{n}（{_pct(k / n)}）" for k, n in res["counts"][s]]
        L.append(f"| {s} | {desc.get(s, '')} | " + " | ".join(cells) + f" | {_pct(res['groups'][s]['all']['mean'])} |")
    for ref, rows in res["vs"].items():
        L += ["", f"## 和 {ref} 配对（差 = 起点 − {ref}，百分点）", "",
              "按任务配对：每个任务先算三个点各自在两边都有的回合上的成功率差，再取三点平均；任务间平均差、"
              "按任务重抽 2,000 次的 95% 区间、升/降任务数、Wilcoxon 符号秩检验。合并：所有回合合在一起的差，"
              "三个点各一个 McNemar 精确检验 p。", "",
              "| 起点 | 任务组 | 任务平均差 [95% 区间] | 升 / 降 / 任务数 | Wilcoxon p | 合并差（三点平均） | 合并 McNemar p（三个点） |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for s, groups in rows.items():
            for g in GROUP_ROWS:
                d = groups[g]
                L.append(f"| {s} | {res['group_zh'][g]} | {_pts(d['task_mean_diff'])} [{_pts(d['task_low'])}, "
                         f"{_pts(d['task_high'])}] | {d['tasks_up']} / {d['tasks_down']} / {d['tasks']} | "
                         f"{_p(d['wilcoxon_p'])} | {_pts(d['pooled_mean_diff'])} | "
                         + " / ".join(_p(x["p"]) for x in d["pooled_by_step"]) + " |")
    L += ["", "## 每个任务的三点平均成功率", "", "| 任务 | " + " | ".join(starts) + " |", "| --- |" + " --- |" * len(starts)]
    for t in sorted(res["tasks"][starts[0]]):
        L.append(f"| {t} | " + " | ".join(_pct(res["tasks"][s][t]) if t in res["tasks"][s] else "—" for s in starts) + " |")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("make", "report"))
    ap.add_argument("--starts", nargs="+", default=list(STARTS))
    ap.add_argument("--out", type=Path, default=RUNS / "v4_recheck")
    a = ap.parse_args(argv)
    if a.action == "report":
        res = report(a.out)
        print(json.dumps({s: res["groups"][s]["all"] for s in res["starts"]}, indent=1))
        return 0
    print(json.dumps(make(a.starts), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
