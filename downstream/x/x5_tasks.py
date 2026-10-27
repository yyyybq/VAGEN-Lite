#!/usr/bin/env python3
"""X5 (the user's decision 7 of 2026-10-26): tasks our SFT never saw, i.e. RoboCasa 1.0.1 registry tasks outside
target50 (the 50 tasks of our training data; RoboCasa's own "composite_unseen" 16 are among them).  For each kind
(atomic, composite): the registry's tasks minus target50, sorted by name, shuffled with random.Random(0), then reset
one by one as the closed-loop evaluation does (target split, seed 1000, stable counter regions) until N reset without
an error; those N, in that order, are the list.  Failures are kept with their error.

    python downstream/x/x5_tasks.py --kind atomic --n 16 --out OUT/atomic.json
    python downstream/x/x5_tasks.py --rows-from OUT/atomic.json OUT/composite.json --out downstream/x/x5_tasks.json

Run with the RoboCasa simulator Python (downstream/eval/sim_env.sh).  OUT is rewritten after every try.  --rows-from
(any Python) turns list files into the task file of the X5 evaluation plan (downstream/x/x5_plan.json): one row per
chosen task, {"task", "group": "unseen_<kind>", "horizon"} (RoboCasa's task horizon, as target50_tasks.json has it).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
TARGET50 = WORKTREE / "downstream/demos/target50_tasks.json"
SEED = 1000


def order(registry, target, seed: int = 0) -> list:
    """Candidates in trial order: registry tasks not in `target`, sorted by name, then shuffled with seed."""
    out = sorted(set(registry) - set(target))
    random.Random(seed).shuffle(out)
    return out


def try_reset(task: str) -> dict:
    import gymnasium as gym
    import robocasa.wrappers.gym_wrapper  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    t = time.time()
    env = None
    try:
        env = gym.make(f"robocasa/{task}", enable_render=True, split="target", seed=SEED)
        env.reset(seed=SEED)
        lang = env.unwrapped.env.get_ep_meta().get("lang")
        return {"task": task, "ok": True, "lang": lang, "horizon": int(get_task_horizon(task)),
                "seconds": round(time.time() - t, 1)}
    except Exception as exc:  # a task that cannot reset in our simulator is left out, with its error
        return {"task": task, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300], "seconds": round(time.time() - t, 1)}
    finally:
        if env is not None:
            env.close()


def task_rows(lists) -> list:
    """Evaluation rows of the chosen tasks of X5 list files, in list order: {task, group "unseen_<kind>", horizon}."""
    rows = []
    for rec in lists:
        horizon = {t["task"]: int(t["horizon"]) for t in rec["tried"] if t.get("ok")}
        rows += [{"task": t, "group": f"unseen_{rec['kind']}", "horizon": horizon[t]} for t in rec["list"]]
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("atomic", "composite"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-from", nargs="+", help="X5 list files: write the evaluation task file to --out instead")
    args = ap.parse_args(argv)
    if args.rows_from:
        lists = [json.loads(Path(p).read_text()) for p in args.rows_from]
        Path(args.out).write_text(json.dumps({
            "_doc": "X5 (the user's decision 7 of 2026-10-26, lists confirmed 10-27): RoboCasa 1.0.1 tasks outside "
                    "target50, chosen by downstream/x/x5_tasks.py (registry minus target50, sorted by name, "
                    "random.Random(0) shuffle, first 16 per kind that reset); horizon = RoboCasa's task horizon.",
            "sources": [str(p) for p in args.rows_from], "tasks": task_rows(lists)}, indent=1) + "\n")
        return 0
    if not args.kind:
        ap.error("give --kind (or --rows-from)")
    sys.path.insert(0, str(WORKTREE))
    from downstream.eval import closed_loop
    from robocasa.utils import dataset_registry as R

    closed_loop.stable_counter_regions()
    registry = R.ATOMIC_TASK_DATASETS if args.kind == "atomic" else R.COMPOSITE_TASK_DATASETS
    target = [t["task"] for t in json.loads(TARGET50.read_text())["tasks"]]
    candidates = order(registry, target)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"kind": args.kind, "rule": "registry minus target50, sorted by name, random.Random(0) shuffle, first "
                                      f"{args.n} that reset (target split, seed {SEED})",
           "candidates": len(candidates), "tried": [], "list": []}
    for task in candidates:
        if len(rec["list"]) >= args.n:
            break
        r = try_reset(task)
        rec["tried"].append(r)
        if r["ok"]:
            rec["list"].append(task)
        out.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"{task}: {'ok' if r['ok'] else r['error'][:80]} ({r['seconds']} s)", flush=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0 if len(rec["list"]) >= args.n else 1


if __name__ == "__main__":
    sys.exit(main())
