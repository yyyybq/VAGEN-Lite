#!/usr/bin/env python3
"""Evaluate every checkpoint of running experiments as it appears, and keep a board (login node, CPU).

    python downstream/eval/watch.py --exps DS-B-100-s1 DS-S-100-s1 DS-RS-100-s1 DS-RD-100-s1 \
        --cards 8 --servers 1 --workers 24 --every 900 --board RUNS/BOARD_D5.md

For each experiment EXP (<runs-root>/<id>): checkpoints are EXP/run/checkpoints/steps_<N>_pytorch_model.pt and
EXP/run/final_model/pytorch_model.pt (tagged steps_<max_train_steps>).  A checkpoint is evaluated once it has not
changed for --settle seconds: EXP/evals/<tag>/ gets an sco job running downstream/eval/run_eval_ckpt.sh with the
fixed plan eval/eval_d6.json, on the pool with fewer waiting jobs.  Nothing is ever deleted.  The board lists per
experiment the training job state, the last logged step and loss, and per checkpoint the eval state and success
(overall, atomic_seen, composite_seen, spatial group).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
sys.path.insert(0, str(DOWNSTREAM))
import sco_job  # noqa: E402

RUNS_ROOT = Path("/mnt/umm/users/yinbaiqiao/downstream_runs/experiments")


def checkpoints(exp: Path) -> list:
    """[(tag, path)] in step order."""
    out = []
    for p in sorted((exp / "run/checkpoints").glob("steps_*_pytorch_model.pt")):
        out.append((p.name.split("_pytorch")[0], p))
    final = exp / "run/final_model/pytorch_model.pt"
    if final.is_file():
        steps = json.loads((exp / "experiment.json").read_text())["steps"]
        out.append((f"steps_{steps}", final))
    return sorted(out, key=lambda x: int(x[0].split("_")[1]))


def settled(path: Path, settle: float) -> bool:
    return path.is_file() and time.time() - path.stat().st_mtime > settle


def groups() -> dict:
    tasks = json.loads((DOWNSTREAM / "demos/target50_tasks.json").read_text())["tasks"]
    spatial = json.loads((DOWNSTREAM / "demos/spatial_tasks.json").read_text())["spatial"]
    out = {t["task"]: t["group"] for t in tasks}
    return out, set(spatial)


def score(summary_path: Path) -> dict | None:
    if not summary_path.is_file():
        return None
    per_task = json.loads(summary_path.read_text())["per_task"]
    group_of, spatial = groups()
    agg = {}
    for task, s in per_task.items():
        for key in ("all", group_of.get(task, "?"), "spatial" if task in spatial else None):
            if key:
                a = agg.setdefault(key, [0, 0])
                a[0] += s["successes"]
                a[1] += s["n"]
    return {k: f"{v[0]}/{v[1]}" for k, v in agg.items()}


def last_metric(exp: Path):
    path = exp / "run/metrics.jsonl"
    if not path.is_file():
        return None
    lines = path.read_text().splitlines()
    return json.loads(lines[-1]) if lines else None


def tick(exp_ids, runs_root: Path, cards: int, servers: int, workers: int, settle: float, dry: bool = False) -> list:
    rows = []
    for exp_id in exp_ids:
        exp = runs_root / exp_id
        train_state = None
        if (exp / "job/submission.json").exists():
            try:
                train_state = sco_job.status(exp)["state"]
            except Exception as exc:
                train_state = f"status error {type(exc).__name__}"
        metric = last_metric(exp)
        row = {"exp": exp_id, "train": train_state,
               "step": metric.get("step") if metric else None,
               "loss": metric.get("action_dit_loss") if metric else None, "evals": []}
        for tag, path in checkpoints(exp):
            run = exp / "evals" / tag
            entry = {"tag": tag, "state": "waiting for the file to settle"}
            if (run / "job/submission.json").exists():
                st = sco_job.status(run)
                entry["state"] = st["state"] if not st["finished"] else f"{st['state']} exit {st['exit_code']}"
                entry["score"] = score(run / "out/episodes/summary.json")
            elif settled(path, settle):
                if dry:
                    entry["state"] = "would submit"
                else:
                    pool = sco_job.pick_pool()
                    sco_job.prepare(run, cards, pool, f"ev-{exp_id.lower()}-{tag.replace('_', '')}"[:40],
                                    ["bash", str(HERE / "run_eval_ckpt.sh"), str(path), str(run / "out"),
                                     str(servers), str(workers)])
                    entry["state"] = f"submitted {sco_job.submit(run)} ({pool})"
            row["evals"].append(entry)
        rows.append(row)
    return rows


def board(rows) -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# 第三步 D5 四条线（自动生成，{now}）", "",
             "| 实验 | 训练任务状态 | 最近记录的步数 | 动作损失 | 检查点：评测状态（成功数/回合数：全部 / 单步 / 多步 / 空间组） |",
             "| --- | --- | ---: | ---: | --- |"]
    for r in rows:
        evals = "<br>".join(
            f"{e['tag']}: {e['state']}" + (f" — {e['score'].get('all', '-')} / {e['score'].get('atomic_seen', '-')}"
                                            f" / {e['score'].get('composite_seen', '-')} / {e['score'].get('spatial', '-')}"
                                            if e.get("score") else "")
            for e in r["evals"]) or "—"
        loss = f"{r['loss']:.4f}" if isinstance(r["loss"], float) else "—"
        lines.append(f"| {r['exp']} | {r['train'] or '—'} | {r['step'] if r['step'] is not None else '—'} | {loss} | {evals} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--cards", type=int, default=8, help="cards per eval job (they buy CPUs: 14 vCPU each)")
    ap.add_argument("--servers", type=int, default=1, help="policy servers per eval job (one GPU each)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--settle", type=float, default=600, help="seconds a checkpoint file must be unchanged")
    ap.add_argument("--every", type=float, default=0, help="loop period in seconds (0: run once)")
    ap.add_argument("--board", type=Path, default=None)
    ap.add_argument("--dry", action="store_true", help="never submit")
    args = ap.parse_args(argv)
    while True:
        rows = tick(args.exps, args.runs_root, args.cards, args.servers, args.workers, args.settle, args.dry)
        text = board(rows)
        if args.board:
            args.board.write_text(text)
        print(text, flush=True)
        if args.every <= 0:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    sys.exit(main())
