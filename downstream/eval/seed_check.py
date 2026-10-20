#!/usr/bin/env python3
"""Does the evaluation seed really reach the environment?  (Run with the RoboCasa simulator Python, CPU.)

    MUJOCO_GL=osmesa python downstream/eval/seed_check.py --tasks NavigateKitchen OpenStandMixerHead --seed 1000 --out x.json

Per task, the initial state (closed_loop.state_fingerprint: layout / style, all joint positions, instruction,
camera images) of
  a: a new env, reset(seed)            b: the same env, reset(seed) again
  c: another new env, reset(seed)      d: that env, reset(seed + 1)
Pass when a == b == c and d != a.  The StarVLA README notes that its own evaluator never passes the seed on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop import state_fingerprint  # noqa: E402


def check_task(task: str, seed: int, split: str = "target") -> dict:
    import gymnasium as gym
    import robocasa.wrappers.gym_wrapper  # noqa: F401

    out = {}
    env = gym.make(f"robocasa/{task}", enable_render=True, split=split, seed=seed)
    try:
        out["a"] = state_fingerprint(env, env.reset(seed=seed)[0])
        out["b"] = state_fingerprint(env, env.reset(seed=seed)[0])
    finally:
        env.close()
    env = gym.make(f"robocasa/{task}", enable_render=True, split=split, seed=seed)
    try:
        out["c"] = state_fingerprint(env, env.reset(seed=seed)[0])
        out["d"] = state_fingerprint(env, env.reset(seed=seed + 1)[0])
    finally:
        env.close()
    out["same_seed_same_state"] = out["a"] == out["b"] == out["c"]
    out["other_seed_other_state"] = out["d"] != out["a"]
    out["pass"] = out["same_seed_same_state"] and out["other_seed_other_state"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {"seed": args.seed, "tasks": {}}
    for task in args.tasks:
        try:
            report["tasks"][task] = check_task(task, args.seed)
        except Exception as exc:
            report["tasks"][task] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    report["pass"] = all(t.get("pass") for t in report["tasks"].values())
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({t: v.get("pass") for t, v in report["tasks"].items()}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
