#!/usr/bin/env python3
"""Step-3 runs DS-<start>-<ratio>-s<seed>: make their experiment directories and submit their training jobs.

    python downstream/train/runs.py make   --starts B S RS RD --ratios 100 --seeds 2 3
    python downstream/train/runs.py submit --starts B S RS RD --ratios 100 --seeds 2 3
    python downstream/train/runs.py list   --ratios 030 010 --seeds 1 2 3

Fixed for every step-3 run: 200,000 steps; data downstream_runs/data/target50_<ratio> (downstream/demos/prepare_data.py
from downstream/demos/manifests/target50_<ratio>.json); one 8-card job (downstream/train/run_train.sh); only the final
checkpoint is kept (save interval = steps, decision 7 of 2026-10-21).  The D5 runs DS-*-100-s1 were made earlier with
a save interval of 25,000 and serve as the seed-1 runs of the 100% column.  `make` skips existing directories (it
runs make_experiment.py, which verifies the start, the StarVLA patch and the data first); `submit` skips runs that
already have a job and sends each to the pool with fewer waiting jobs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
sys.path.insert(0, str(DOWNSTREAM))
import sco_job  # noqa: E402

RUNS_ROOT = Path("/mnt/umm/users/yinbaiqiao/downstream_runs")
STEPS = 200_000
STARTS = ("B", "S", "RS", "RD")
RATIOS = ("010", "030", "100")
PYTHON = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python"


def exp_id(start: str, ratio: str, seed: int) -> str:
    if ratio not in RATIOS:
        raise ValueError(f"ratio {ratio!r} not one of {RATIOS}")
    return f"DS-{start}-{ratio}-s{int(seed)}"


def grid(starts, ratios, seeds):
    return [(s, r, int(k)) for r in ratios for s in starts for k in seeds]


def make_command(start: str, ratio: str, seed: int, runs_root: Path = RUNS_ROOT, python: str = PYTHON) -> list:
    return [python, str(HERE / "make_experiment.py"), "--exp-id", exp_id(start, ratio, seed), "--start", start,
            "--data", str(runs_root / "data" / f"target50_{ratio}"), "--seed", str(seed), "--steps", str(STEPS),
            "--save-interval", str(STEPS), "--runs-root", str(runs_root / "experiments")]


def make(start: str, ratio: str, seed: int, runs_root: Path = RUNS_ROOT, run=subprocess.run) -> dict:
    exp = runs_root / "experiments" / exp_id(start, ratio, seed)
    if (exp / "experiment.json").exists():
        return {"exp": exp.name, "made": False, "note": "exists"}
    res = run(make_command(start, ratio, seed, runs_root), capture_output=True, text=True)
    if res.returncode != 0:
        return {"exp": exp.name, "made": False, "error": (res.stderr or res.stdout)[-800:]}
    return {"exp": exp.name, "made": True}


def submit(start: str, ratio: str, seed: int, runs_root: Path = RUNS_ROOT, jobs=sco_job, pool: str = None) -> dict:
    """One 8-card job: run_segmented.sh for a staged-plan experiment (experiment.json "segmented": full state,
    evaluation inside the job), run_train.sh otherwise; to `pool`, or the pool with fewer waiting jobs."""
    exp = runs_root / "experiments" / exp_id(start, ratio, seed)
    if not (exp / "experiment.json").exists():
        return {"exp": exp.name, "job": None, "error": "no experiment directory (run make first)"}
    if (exp / "job/submission.json").exists():
        return {"exp": exp.name, "job": json.loads((exp / "job/submission.json").read_text())["job_id"],
                "note": "already submitted"}
    script = "run_segmented.sh" if json.loads((exp / "experiment.json").read_text()).get("segmented") else "run_train.sh"
    pool = pool or jobs.pick_pool()
    jobs.prepare(exp, 8, pool, exp.name.lower(), ["bash", str(HERE / script), str(exp)])
    return {"exp": exp.name, "job": jobs.submit(exp), "pool": pool, "script": script}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["make", "submit", "list"])
    ap.add_argument("--starts", nargs="+", default=list(STARTS))
    ap.add_argument("--ratios", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    args = ap.parse_args(argv)
    failed = 0
    for start, ratio, seed in grid(args.starts, args.ratios, args.seeds):
        if args.action == "list":
            out = {"exp": exp_id(start, ratio, seed)}
        elif args.action == "make":
            out = make(start, ratio, seed, args.runs_root)
        else:
            out = submit(start, ratio, seed, args.runs_root)
        failed += "error" in out
        print(json.dumps(out), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
