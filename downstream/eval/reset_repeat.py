#!/usr/bin/env python3
"""Is a seeded RoboCasa365 reset identical across fresh processes?  (RoboCasa simulator Python, CPU, OSMesa.)

    MUJOCO_GL=osmesa python downstream/eval/reset_repeat.py --tasks OpenStandMixerHead NavigateKitchen \
        --seeds 1002,1006 --repeats 4 [--workers 12] [--stable-regions] [--seed-globals] [--images] --out x.json
    ... --tasks plan  (the tasks of downstream/eval/eval_d6.json)

Every (task, seed, repeat) runs in its own new spawned process (pool with one task per child): gym.make(seed) +
reset(seed), then closed_loop.state_fingerprint plus every joint's qpos.  With --stable-regions the process first
applies closed_loop.stable_counter_regions (as the evaluation does).  With --seed-globals it first seeds numpy's
and Python's global RNGs with the same seed.  With --images the first camera images are saved next to --out
(<out stem>_<task>_<seed>_r<i>.npz).  Per (task, seed) the output gives the number of distinct fingerprints (layout,
style, instruction, qpos rounded to 1e-6, image hash), whether layout / style / instruction ever differ, the largest
joint difference between repeats, split into free joints (movable objects, the robot base if free) and the rest, and
the joints that differ by more than 1e-4.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
PLAN = HERE / "eval_d6.json"
TASKS = WORKTREE / "downstream/demos/target50_tasks.json"
SCENE_KEYS = ("layout_id", "style_id", "instruction", "nq")


def seed_globals(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def plan_tasks(plan: Path = PLAN, tasks: Path = TASKS):
    groups = json.loads(plan.read_text())["groups"]
    return [t["task"] for t in json.loads(tasks.read_text())["tasks"] if t["group"] in groups]


def joints(env) -> dict:
    """name -> (is_free, qpos values rounded to 1e-6)."""
    import mujoco

    inner = getattr(env, "unwrapped", env)
    sim = getattr(inner, "env", inner).sim
    model, data = sim.model._model, sim.data._data
    width = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}
    out = {}
    for j in range(model.njnt):
        adr, kind = int(model.jnt_qposadr[j]), int(model.jnt_type[j])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        out[name] = [kind == int(mujoco.mjtJoint.mjJNT_FREE),
                     [round(float(x), 6) for x in data.qpos[adr:adr + width.get(kind, 1)]]]
    return out


def one(item):
    task, seed, rep, stable_regions, seed_globals_too, images_to = item
    sys.path.insert(0, str(HERE))
    from closed_loop import stable_counter_regions, state_fingerprint

    if seed_globals_too:
        seed_globals(seed)
    import gymnasium as gym
    import numpy as np
    import robocasa.wrappers.gym_wrapper  # noqa: F401

    if stable_regions:
        stable_counter_regions()

    t0 = time.monotonic()
    env = None
    try:
        env = gym.make(f"robocasa/{task}", enable_render=True, split="target", seed=seed)
        obs = env.reset(seed=seed)[0]
        state = dict(state_fingerprint(env, obs), joints=joints(env))
        if images_to:
            np.savez_compressed(images_to, **{k: np.asarray(v) for k, v in obs.items() if k.startswith("video.")})
            state["images"] = images_to
    except Exception as exc:
        state = {"error": f"{type(exc).__name__}: {exc}"[:500]}
    finally:
        if env is not None:
            env.close()
    return dict(state, task=task, seed=seed, rep=rep, seconds=round(time.monotonic() - t0, 1))


def compare(states):
    """Summary of the repeats of one (task, seed)."""
    ok = [s for s in states if "error" not in s]
    fingerprints = {json.dumps({k: v for k, v in s.items() if k not in ("joints", "images", "rep", "seconds",
                                                                         "task", "seed")}, sort_keys=True)
                    for s in ok}
    out = {"repeats": len(states), "errors": len(states) - len(ok), "distinct_fingerprints": len(fingerprints),
           "scene_differs": any(len({json.dumps(s.get(k)) for s in ok}) > 1 for k in SCENE_KEYS),
           "max_diff_free": 0.0, "max_diff_other": 0.0, "joints_over_1e-4": []}
    if len(ok) < 2 or out["scene_differs"]:
        return out
    ref = ok[0]["joints"]
    for s in ok[1:]:
        if list(s["joints"]) != list(ref):
            out["scene_differs"] = True
            return out
        for name, (free, values) in s["joints"].items():
            diff = max(abs(a - b) for a, b in zip(values, ref[name][1])) if values else 0.0
            key = "max_diff_free" if free else "max_diff_other"
            out[key] = max(out[key], round(diff, 6))
            if diff > 1e-4 and name not in out["joints_over_1e-4"]:
                out["joints_over_1e-4"].append(name)
    return out


def parse_seeds(text: str):
    sys.path.insert(0, str(HERE))
    from closed_loop import parse_seeds as parse

    return parse(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True, help="task names, or 'plan' for the eval_d6.json tasks")
    ap.add_argument("--seeds", required=True, help="e.g. 1000-1001 or 1002,1006")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--stable-regions", action="store_true")
    ap.add_argument("--seed-globals", action="store_true")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    tasks = plan_tasks() if args.tasks == ["plan"] else args.tasks
    items = [(t, s, r, args.stable_regions, args.seed_globals,
              str(args.out.with_name(f"{args.out.stem}_{t}_{s}_r{r}.npz")) if args.images else None)
             for t in tasks for s in parse_seeds(args.seeds) for r in range(args.repeats)]
    t0 = time.monotonic()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        rows = list(pool.imap_unordered(one, items))
    pairs = {}
    for row in rows:
        pairs.setdefault(f"{row['task']}/{row['seed']}", []).append(row)
    summary = {key: compare(sorted(states, key=lambda s: s["rep"])) for key, states in sorted(pairs.items())}
    out = {"tasks": tasks, "seeds": args.seeds, "repeats": args.repeats, "workers": args.workers,
           "stable_regions": args.stable_regions, "seed_globals": args.seed_globals,
           "wall_seconds": round(time.monotonic() - t0, 1), "pairs": len(summary),
           "pairs_bit_identical": sum(v["distinct_fingerprints"] == 1 and not v["errors"] for v in summary.values()),
           "pairs_scene_differs": sorted(k for k, v in summary.items() if v["scene_differs"]),
           "pairs_joint_over_1e-4": sorted(k for k, v in summary.items() if v["joints_over_1e-4"]),
           "pairs_with_errors": sorted(k for k, v in summary.items() if v["errors"]),
           "max_diff_free": max((v["max_diff_free"] for v in summary.values()), default=0.0),
           "max_diff_other": max((v["max_diff_other"] for v in summary.values()), default=0.0),
           "summary": summary, "states": sorted(rows, key=lambda s: (s["task"], s["seed"], s["rep"]))}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("pairs", "pairs_bit_identical", "pairs_scene_differs",
                                          "pairs_joint_over_1e-4", "pairs_with_errors", "max_diff_free",
                                          "max_diff_other", "wall_seconds")}))
    return 0 if not out["pairs_with_errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
