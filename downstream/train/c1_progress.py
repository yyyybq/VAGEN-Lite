#!/usr/bin/env python3
"""C1's progress every 25,000 steps (the user's decision 1 of 2026-10-31: once the formal run is going, report every
25k steps the time per step, the GPU memory peak and the two losses).

    python downstream/train/c1_progress.py --exp EXP [--every 25000] [--reference DS-B-030-s1 dir]
                                           [--eval-card-hours 65] [--approved 910]

From EXP/run/metrics.jsonl (a row every 100 steps, with its wall-clock time), EXP/gpu.csv (nvidia-smi every 10 s) and
EXP/job/worker.log (the job's start and end lines), one row per finished window of `every` steps:
  s_per_step         wall clock per step over the window: the time between consecutive logged rows over their steps,
                     summed, leaving out the intervals across a relaunch (their time holds the restart); saves are in;
  s_per_step_median  the median of those intervals (training alone, without the saves);
  mem_alloc_peak_gib torch's allocated-memory peak of a step on the worst card (mem_peak_gib_max), the max over the
                     window's logged steps (one step in 100);
  mem_used_peak_gib  nvidia-smi's memory.used, the max over the 8 cards and the window's time (every step, PyTorch's
                     cached memory included);
  losses             means over the window's logged steps: action_dit_loss, text_loss (8 cards), text_loss_traj (the
                     4 even cards), text_loss_qa (the 4 odd cards); the reference run's action_dit_loss over the same
                     steps beside them;
  skipped_share      over-long trajectory samples passed over in the window (increments of text_too_long_total, a
                     counter that starts at 0 with each launch) over the trajectory samples drawn (4 per step plus the
                     skipped ones), as in the short run's 57 / 857.
Rows logged twice (a relaunch redoes the steps after its full state) count once, the later launch's.  Card-hours: spent
so far (8 x the job's wall time from worker.log) and projected (spent + the remaining steps x the last window's
s_per_step x 8 + the in-job evaluation's estimate) against the approved number.
Writes EXP/progress/progress.json and progress.md and prints the json.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import re
import statistics
import sys
from pathlib import Path

REFERENCE = Path("/mnt/umm/users/yinbaiqiao/downstream_runs/experiments/DS-B-030-s1")
CARDS = 8
TRAJ_PER_STEP = 4  # the even ranks draw trajectories (cotrain_text: even ranks traj, odd ranks qa)
LOSSES = ("action_dit_loss", "text_loss", "text_loss_traj", "text_loss_qa")
GAP_FACTOR = 20  # an interval over 20x the median is a restart, also without a step going back
LAUNCH = re.compile(r"\[launch\] host=\S+ date=(\S+)")
END = re.compile(r"\[launch\] rc=\S+ date=(\S+)")


def utc(text: str) -> float:
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()


def read_rows(path: Path) -> list:
    """Logged rows in file order, each with its launch number (a new launch: the step goes back or stays, the skip
    counter goes back, or the gap to the previous row is over GAP_FACTOR x the median gap)."""
    rows = []
    if not Path(path).is_file():
        return rows
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "step" in r and "time" in r:
                rows.append(r)
    gaps = [b["time"] - a["time"] for a, b in zip(rows, rows[1:]) if b["step"] > a["step"]]
    limit = GAP_FACTOR * statistics.median(gaps) if gaps else math.inf
    launch = 0
    for i, r in enumerate(rows):
        if i:
            p = rows[i - 1]
            if r["step"] <= p["step"] or r["time"] - p["time"] > limit \
                    or float(r.get("text_too_long_total", 0) or 0) < float(p.get("text_too_long_total", 0) or 0):
                launch += 1
        r["_launch"] = launch
    return rows


def kept(rows: list) -> list:
    """The rows that stand, in step order: a launch's first row drops every earlier launch's row from its step on (a
    relaunch redoes the steps after its full state; the rows of a crashed launch past that point are stale)."""
    out = {}
    for i, r in enumerate(rows):
        if i and r["_launch"] != rows[i - 1]["_launch"]:
            for s in [s for s in out if s >= r["step"]]:
                del out[s]
        out[r["step"]] = r
    return [out[s] for s in sorted(out)]


def mean(xs) -> float:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def gpu_samples(path: Path) -> list:
    """(unix time, memory.used GiB) of every card's sample; nvidia-smi writes the job's local time, UTC in the jobs."""
    out = []
    if not Path(path).is_file():
        return out
    for row in csv.reader(Path(path).read_text().splitlines()):
        if len(row) < 3:
            continue
        try:
            t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            mib = float(row[2].strip().split()[0])
        except ValueError:
            continue
        out.append((t.replace(tzinfo=datetime.timezone.utc).timestamp(), mib / 1024.0))
    return out


def job_runs(worker_log: Path) -> list:
    """[start, end or None] of every run of the job (a resubmitted or relaunched job appends to worker.log)."""
    runs = []
    if not Path(worker_log).is_file():
        return runs
    for line in Path(worker_log).read_text(errors="replace").splitlines():
        m = LAUNCH.search(line)
        if m:
            runs.append([utc(m.group(1)), None])
            continue
        m = END.search(line)
        if m and runs and runs[-1][1] is None:
            runs[-1][1] = utc(m.group(1))
    return runs


def windows(rows: list, every: int, gpu: list, ref_rows: list) -> list:
    out = []
    if not rows:
        return out
    last_step = rows[-1]["step"]
    ref = {r["step"]: r for r in ref_rows}
    for k in range(last_step // every):
        lo, hi = k * every, (k + 1) * every
        pairs = [(a, b) for a, b in zip([None] + rows[:-1], rows) if lo < b["step"] <= hi]
        cur = [b for _, b in pairs]
        timed = [(b["time"] - a["time"], b["step"] - a["step"]) for a, b in pairs
                 if a is not None and a["_launch"] == b["_launch"] and b["step"] > a["step"]]
        skipped = 0
        for a, b in pairs:
            c = float(b.get("text_too_long_total", 0) or 0)
            skipped += c - float(a.get("text_too_long_total", 0) or 0) \
                if a is not None and a["_launch"] == b["_launch"] else c
        steps = hi - lo
        t0 = min(a["time"] for a, _ in pairs if a is not None) if any(a is not None for a, _ in pairs) \
            else cur[0]["time"]
        t1 = max(b["time"] for b in cur)
        used = [g for t, g in gpu if t0 <= t <= t1]
        ref_cur = [ref[s] for s in range(lo + 100, hi + 1, 100) if s in ref]
        out.append({
            "window": [lo, hi], "logged_steps": len(cur),
            "s_per_step": sum(t for t, _ in timed) / sum(n for _, n in timed) if timed else float("nan"),
            "s_per_step_median": statistics.median(t / n for t, n in timed) if timed else float("nan"),
            "mem_alloc_peak_gib": max((float(b["mem_peak_gib_max"]) for b in cur if "mem_peak_gib_max" in b),
                                      default=float("nan")),
            "mem_used_peak_gib": max(used, default=float("nan")),
            **{k2: mean(b.get(k2) for b in cur) for k2 in LOSSES},
            "reference_action_dit_loss": mean(r.get("action_dit_loss") for r in ref_cur) if ref_cur else None,
            "skipped": int(skipped),
            "skipped_share": skipped / (TRAJ_PER_STEP * steps + skipped) if steps else float("nan"),
            "launches": sorted({b["_launch"] for b in cur}),
            "end_utc": datetime.datetime.fromtimestamp(t1, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        })
    return out


def card_hours(runs: list, rows: list, now: float) -> float:
    """8 x the job's wall time so far (a run still going counts up to now)."""
    total = 0.0
    for start, end in runs:
        total += (end if end is not None else now) - start
    if not runs and rows:  # no worker.log (tests, or a run made by hand): the logged span
        total = rows[-1]["time"] - rows[0]["time"]
    return CARDS * total / 3600.0


def progress(exp: Path, every: int = 25000, reference: Path = REFERENCE, eval_card_hours: float = 65.0,
             approved: float = 910.0, now: float = None) -> dict:
    exp = Path(exp)
    rec = json.loads((exp / "experiment.json").read_text())
    steps = int(rec["steps"])
    rows = kept(read_rows(exp / "run/metrics.jsonl"))
    ref_path = Path(reference) / "run/metrics.jsonl"
    ref_rows = kept(read_rows(ref_path)) if ref_path.is_file() else []
    wins = windows(rows, every, gpu_samples(exp / "gpu.csv"), ref_rows)
    now = now if now is not None else datetime.datetime.now(datetime.timezone.utc).timestamp()
    runs = job_runs(exp / "job/worker.log")
    spent = card_hours(runs, rows, now)
    last_step = rows[-1]["step"] if rows else 0
    pace = wins[-1]["s_per_step"] if wins and not math.isnan(wins[-1]["s_per_step"]) else None
    projected = spent + CARDS * (steps - last_step) * pace / 3600.0 + eval_card_hours \
        if pace and last_step < steps else None  # training done: the evaluation is in spent already
    return {"exp": str(exp), "steps": steps, "every": every, "last_logged_step": last_step,
            "last_logged_utc": datetime.datetime.fromtimestamp(rows[-1]["time"], datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%MZ") if rows else None,
            "launches": (rows[-1]["_launch"] + 1) if rows else 0, "job_runs": len(runs),
            "card_hours_spent": spent, "card_hours_projected": projected, "eval_card_hours_estimate": eval_card_hours,
            "card_hours_approved": approved, "reference": str(reference), "windows": wins}


def fmt(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def markdown(p: dict) -> str:
    lines = [f"# C1 进度（{Path(p['exp']).name}）", "",
             f"最后记录：第 {p['last_logged_step']:,} 步（{p['last_logged_utc'] or '—'} UTC），共 {p['steps']:,} 步。"
             f"卡时：已用 {fmt(p['card_hours_spent'], 0)}，按最近一段的每步时间推算全程约 "
             f"{fmt(p['card_hours_projected'], 0)}（含任务内评测估计 {fmt(p['eval_card_hours_estimate'], 0)}），"
             f"批的是 {fmt(p['card_hours_approved'], 0)}。", "",
             "| 步数段 | 结束（UTC） | 每步秒数（含保存） | 中位数 | 显存峰值 torch（GiB） | 显存峰值 nvidia-smi（GiB） "
             "| 动作损失 | 基线动作损失 | 文字损失 | 轨迹 | QA | 跳过的长轨迹 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for w in p["windows"]:
        lo, hi = w["window"]
        lines.append(f"| {lo // 1000}k–{hi // 1000}k | {w['end_utc']} | {fmt(w['s_per_step'])} | "
                     f"{fmt(w['s_per_step_median'])} | {fmt(w['mem_alloc_peak_gib'], 1)} | {fmt(w['mem_used_peak_gib'], 1)} | "
                     f"{fmt(w['action_dit_loss'], 3)} | {fmt(w['reference_action_dit_loss'], 3)} | "
                     f"{fmt(w['text_loss'], 3)} | {fmt(w['text_loss_traj'], 3)} | {fmt(w['text_loss_qa'], 3)} | "
                     f"{w['skipped']}（{fmt(100 * w['skipped_share'], 1)}%） |")
    lines += ["", "- 每步秒数：相邻两条记录（每 100 步一条）的时间差除以步数，跨重启的那一段不算；含存续训状态的时间。"
                  "中位数是不含保存的纯训练时间。",
              "- 显存峰值 torch：每条记录那一步 8 张卡里最高的已分配显存（只看记录到的步，每 100 步一步）；"
              "nvidia-smi：这段时间里 8 张卡每 10 秒一次的已用显存最大值，含 PyTorch 缓存，每一步都算在内。上限 80 GiB。",
              "- 动作损失、文字损失：这段里记录到的步的平均。文字损失是 8 张卡的平均；轨迹是偶数卡（4 张），QA 是奇数卡（4 张）。"
              "基线动作损失是 DS-B-030-s1 同一段步数的平均（同起点、同数据、同种子，只少文字损失）。",
              "- 跳过的长轨迹：超过 1,800 个 token 换下一条的轨迹样本数，占这段抽到的轨迹样本（每步 4 条加上被跳过的）的比例。"
              "短任务第四次是 6.7%（57 ÷ 857）。被跳过的是长回合，文字数据因此偏向短回合。"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, required=True)
    ap.add_argument("--every", type=int, default=25000)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--eval-card-hours", type=float, default=65.0)
    ap.add_argument("--approved", type=float, default=910.0)
    args = ap.parse_args(argv)
    p = progress(args.exp, args.every, args.reference, args.eval_card_hours, args.approved)
    out = args.exp / "progress"
    out.mkdir(exist_ok=True)
    (out / "progress.json").write_text(json.dumps(p, indent=1, ensure_ascii=False) + "\n")
    (out / "progress.md").write_text(markdown(p))
    print(json.dumps(p, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
