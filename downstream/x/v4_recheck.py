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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("make",))
    ap.add_argument("--starts", nargs="+", default=list(STARTS))
    a = ap.parse_args(argv)
    print(json.dumps(make(a.starts), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
