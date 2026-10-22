#!/usr/bin/env python3
"""Full training state and the segmented stage-1 job (the user's decisions of 2026-10-22).

Full state (train_entry.py writes it every `trainer.full_state_interval` steps, keeping only the newest):
    RUN/full_state/steps_<N>/            accelerate save_state: the DeepSpeed checkpoint (bf16 module weights, fp32
                                         master weights and both AdamW moments, ZeRO-2 partitions), the learning-rate
                                         scheduler, every process's random states
    RUN/full_state/steps_<N>/downstream_state.json   step, batches the data loader handed out (the data position),
                                         epoch, schedule length, size and save time; written last, so a directory
                                         without it is incomplete
    RUN/full_state/steps_<N>.partial/    a save in progress (a kill leaves it; the next launch removes it)

Segmented job (run_segmented.sh): train to an evaluation point, leave the training processes, evaluate the
checkpoint in the same job (8 policy servers x 12 simulator processes), resume from the full state, and so on to the
last step.  Evaluation points: `eval_points` of experiment.json (50k / 100k / 150k / 200k, up to the run's steps).
After the first in-job evaluation, `speed` compares its time per episode with the 24-process figures of the step-2
report; slower by more than 1.5 times -> only the final checkpoint is evaluated in the job, the intermediate ones
by the watcher's one-card shards.

    python downstream/train/full_state.py points EXP          # evaluation points still to reach, in order
    python downstream/train/full_state.py ckpt EXP STEP       # the weights file of that point
    python downstream/train/full_state.py eval-mode EXP STEP  # injob | shards | done
    python downstream/train/full_state.py speed EXP STEP      # writes EXP/injob_eval.json after the first evaluation
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

STATE_FILE = "downstream_state.json"
STEP_RE = re.compile(r"^steps_(\d+)$")
# 24 simulator processes, one policy server, 112 vCPUs (step-2 report, section 3.3; 3 tasks x 8 seeds)
P24_SECONDS_PER_STEP = 0.196
P24_RESET_SECONDS = 21.6
SLOWDOWN_LIMIT = 1.5


# ------------------------------------------------------------------------------------------- full-state folders
def complete_states(root: Path) -> list:
    """[(step, path)] of the complete full states, oldest first."""
    out = []
    for d in Path(root).glob("steps_*"):
        m = STEP_RE.match(d.name)
        if m and d.is_dir() and (d / STATE_FILE).is_file():
            out.append((int(m[1]), d))
    return sorted(out)


def latest(root: Path):
    """(path, meta) of the newest complete full state, or None."""
    states = complete_states(root)
    if not states:
        return None
    path = states[-1][1]
    return path, json.loads((path / STATE_FILE).read_text())


def partials(root: Path) -> list:
    return sorted(d for d in Path(root).glob("steps_*.partial") if d.is_dir())


def prune(root: Path, keep: Path) -> list:
    """Remove every full state but `keep`, and unfinished saves; returns what was removed."""
    gone = []
    for _, d in complete_states(root):
        if d != keep:
            shutil.rmtree(d)
            gone.append(d.name)
    for d in partials(root):
        if d != keep:
            shutil.rmtree(d)
            gone.append(d.name)
    return gone


def dir_bytes(path: Path) -> int:
    total = 0
    for base, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total


def data_position(batches: int, per_epoch: int):
    """(epoch, batches to skip in it) after `batches` batches of a loader with `per_epoch` batches per epoch."""
    if per_epoch <= 0:
        return 0, 0
    return batches // per_epoch, batches % per_epoch


# ------------------------------------------------------------------------------------------- segmented job
def experiment(exp: Path) -> dict:
    return json.loads((Path(exp) / "experiment.json").read_text())


def eval_points(exp: Path) -> list:
    """The run's evaluation points, ascending, always ending with its last step."""
    rec = experiment(exp)
    steps = int(rec["steps"])
    pts = sorted({int(p) for p in rec.get("eval_points", []) if 0 < int(p) <= steps} | {steps})
    return pts


def ckpt(exp: Path, step: int, run: str = "run") -> Path:
    exp = Path(exp)
    if step == int(experiment(exp)["steps"]):
        return exp / run / "final_model/pytorch_model.pt"
    return exp / run / "checkpoints" / f"steps_{step}_pytorch_model.pt"


def summary(exp: Path, step: int) -> Path:
    return Path(exp) / "evals" / f"steps_{step}" / "out/episodes/summary.json"


def finished(path: Path) -> bool:
    try:
        return "per_task" in json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False


def eval_mode(exp: Path, step: int) -> str:
    """done: evaluated already; injob: evaluate in this job; shards: leave it to the watcher's one-card shards."""
    if finished(summary(exp, step)):
        return "done"
    if step == eval_points(exp)[-1]:
        return "injob"
    rule = Path(exp) / "injob_eval.json"
    if rule.is_file() and json.loads(rule.read_text()).get("mode") == "final_only":
        return "shards"
    return "injob"


def speed(exp: Path, step: int) -> dict:
    """Time per episode of an in-job evaluation (8 servers x 12 processes = 96) against what the same episodes would
    take at 24 processes by the step-2 figures (reset 21.6 s + 0.196 s per control step).  The first call decides the
    rule for the rest of the run and writes EXP/injob_eval.json; later calls only read it."""
    rule = Path(exp) / "injob_eval.json"
    if rule.is_file():
        return json.loads(rule.read_text())
    out = dict(step=step, **speed_of(Path(exp) / "evals" / f"steps_{step}" / "out/episodes"))
    rule.write_text(json.dumps(out, indent=1) + "\n")
    return out


def speed_of(eps: Path) -> dict:
    """The speed figures of one evaluation's episodes (EPS/episodes.jsonl) against the 24-process figures."""
    rows = [json.loads(x) for x in (Path(eps) / "episodes.jsonl").read_text().splitlines() if x.strip()]
    done = [r for r in rows if r.get("status") == "complete"]
    if not done:
        raise SystemExit(f"no complete episodes in {eps}")
    wall = sum(float(r["wall_seconds"]) for r in done)
    predicted = sum(P24_RESET_SECONDS + P24_SECONDS_PER_STEP * int(r["steps"]) for r in done)
    steps = sum(int(r["steps"]) for r in done)
    per_step = sum(float(r["wall_seconds"]) - float(r.get("reset_seconds") or 0.0) for r in done) / max(1, steps)
    ratio = wall / predicted
    return {"episodes": len(done), "wall_mean_seconds": round(wall / len(done), 1),
           "p24_predicted_mean_seconds": round(predicted / len(done), 1), "ratio": round(ratio, 3),
           "seconds_per_step_96": round(per_step, 4), "seconds_per_step_24": P24_SECONDS_PER_STEP,
           "reset_mean_seconds_96": round(sum(float(r.get("reset_seconds") or 0.0) for r in done) / len(done), 1),
           "limit": SLOWDOWN_LIMIT, "mode": "final_only" if ratio > SLOWDOWN_LIMIT else "all",
           "rule": "ratio = sum of episode wall times at 96 processes / sum over the same episodes of "
                   "(21.6 s + 0.196 s x control steps), the 24-process figures of the step-2 report; above 1.5: "
                   "only the final checkpoint is evaluated in the job, the intermediate ones by one-card shards"}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    cmd, exp = argv[0], Path(argv[1])
    if cmd == "points":
        print(" ".join(str(p) for p in eval_points(exp)))
    elif cmd == "ckpt":
        print(ckpt(exp, int(argv[2])))
    elif cmd == "eval-mode":
        print(eval_mode(exp, int(argv[2])))
    elif cmd == "speed":
        print(json.dumps(speed(exp, int(argv[2]))))
    elif cmd == "speed-of":  # EXP is an episodes folder here
        print(json.dumps(speed_of(exp)))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
