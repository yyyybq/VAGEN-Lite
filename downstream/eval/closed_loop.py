#!/usr/bin/env python3
"""Seeded, parallel RoboCasa365 closed-loop evaluation of a StarVLA policy served over websocket.

Run with the RoboCasa simulator Python (probe_spatial/robocasa365/policies/diffusion_policy/.venv) and the
PYTHONPATH of examples/evaluate/robocasa/run_starvla_mobile.sh; policy servers run separately
(run_starvla_mobile.sh server, one per GPU).

    python downstream/eval/closed_loop.py --ckpt CKPT --servers 127.0.0.1:5678 --tasks NavigateKitchen \
        --seeds 1000-1007 --workers 12 --out OUT
    python downstream/eval/closed_loop.py --ckpt CKPT --servers ... --eval-plan downstream/eval/eval_d6.json ...

Per episode: gym.make('robocasa/<task>', split=target, seed=s) and env.reset(seed=s) (the seed reaches the env's
RNG, which draws the kitchen layout / style and object placement), the task-specific horizon of RoboCasa 1.0.1
(downstream/demos/target50_tasks.json) as the step limit, one policy call per `execute_steps` actions (= the
model's 16-step chunk), stop on success.  Episodes run longest horizon first.  Each worker process keeps one
websocket client and first makes RoboCasa's counter-region order process-independent (stable_counter_regions; off
with --no-stable-regions).  Writes OUT/episodes.jsonl (one line per episode, including a fingerprint of the initial
state and its joint positions) and OUT/summary.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
TASKS = WORKTREE / "downstream/demos/target50_tasks.json"
_STATE = {}


def parse_seeds(text: str):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def horizons(path: Path = TASKS) -> dict:
    return {t["task"]: int(t["horizon"]) for t in json.loads(path.read_text())["tasks"]}


def plan(tasks, seeds, horizon_of=None):
    horizon_of = horizon_of or horizons()
    return [{"task": t, "seed": int(s), "max_steps": horizon_of[t]} for t in tasks for s in seeds]


def eval_plan_episodes(path: Path, tasks_path: Path = TASKS):
    """(episodes, plan dict) of a fixed evaluation plan (downstream/eval/eval_d6.json): every task of the plan's
    groups with the plan's `seeds`; the spatial tasks (plan `spatial_tasks` file, key "spatial") with `seeds_spatial`
    when the plan has it."""
    p = json.loads(Path(path).read_text())
    task_rows = json.loads(Path(tasks_path).read_text())["tasks"]
    horizon_of = {t["task"]: int(t["horizon"]) for t in task_rows}
    spatial = set()
    if p.get("seeds_spatial"):
        spatial = set(json.loads((WORKTREE / p["spatial_tasks"]).read_text())["spatial"])
    episodes = []
    for t in (row["task"] for row in task_rows if row["group"] in p["groups"]):
        seeds = p["seeds_spatial"] if t in spatial else p["seeds"]
        episodes += plan([t], parse_seeds(seeds), horizon_of)
    return episodes, p


def state_fingerprint(env, obs) -> dict:
    """What the seed must fix: kitchen layout / style, all joint positions (robot and free objects), the instruction
    and the first camera images."""
    import numpy as np

    inner = getattr(env, "unwrapped", env)
    kitchen = getattr(inner, "env", inner)
    qpos = np.asarray(kitchen.sim.data.qpos, dtype=np.float64)
    images = hashlib.sha256()
    for key in sorted(k for k in obs if k.startswith("video.")):
        images.update(np.ascontiguousarray(obs[key]).tobytes())
    return {"layout_id": _as_int(getattr(kitchen, "layout_id", None)), "style_id": _as_int(getattr(kitchen, "style_id", None)),
            "qpos_sha": hashlib.sha256(np.round(qpos, 6).tobytes()).hexdigest()[:16], "nq": int(qpos.size),
            "images_sha": images.hexdigest()[:16], "instruction": obs.get("annotation.human.task_description")}


def _ordered_set(items=()):
    return dict.fromkeys(items)  # list(...) of it: the distinct items in first-seen order


def with_ordered_set(func):
    """A copy of func in which the name `set` is an insertion-ordered de-duplication, so list(set(xs)) keeps the
    first-seen order of xs.  Nothing else changes (same code, defaults, closure; globals copied)."""
    import types

    new = types.FunctionType(func.__code__, dict(func.__globals__, set=_ordered_set), func.__name__,
                             func.__defaults__, func.__closure__)
    new.__kwdefaults__ = func.__kwdefaults__
    new.__qualname__, new.__doc__, new.__module__ = func.__qualname__, func.__doc__, func.__module__
    new.downstream_ordered_set = True
    return new


def stable_counter_regions() -> bool:
    """Make RoboCasa 1.0.1's counter reset regions come in the same order in every process (this process only;
    RoboCasa's files are not changed).  Counter.get_reset_regions (robocasa/models/fixtures/counter.py:813) de-duplicates the
    candidate counter-top geoms with list(set(geoms)); the geoms are XML elements hashed by memory address, so the
    region order differs between processes, and Fixture.sample_reset_region (fixture.py:383) then picks a region with
    self.rng.choice(regions): the same seed puts a fixture (and the robot in front of it) on a different counter.
    Returns True if the patch was applied now, False if it already was."""
    from robocasa.models.fixtures.counter import Counter

    if getattr(Counter.get_reset_regions, "downstream_ordered_set", False):
        return False
    Counter.get_reset_regions = with_ordered_set(Counter.get_reset_regions)
    return True


def initial_qpos(env) -> list:
    """All joint positions after reset, rounded to 1e-6, so runs can be compared by size of difference, not only by
    hash (fresh processes can differ by ~1e-6 in passive joints: downstream/eval/reset_repeat.py)."""
    inner = getattr(env, "unwrapped", env)
    return [round(float(x), 6) for x in getattr(inner, "env", inner).sim.data.qpos]


def longest_first(episodes):
    """Run order: longest task horizon first (stable), so the long episodes do not form the tail of the run."""
    return sorted(episodes, key=lambda e: -e["max_steps"])


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None if value is None else str(value)


def connect(ckpt: str, server: str, starvla_root: str):
    """A websocket client to one policy server, checked against the checkpoint's contract."""
    sys.path.insert(0, str(WORKTREE))
    sys.path.insert(0, starvla_root)
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from vagen.envs.robocasa.starvla_contract import PolicyContract, checkpoint_config, verify_server

    config, _ = checkpoint_config(ckpt)
    contract = PolicyContract.from_config(config)
    host, port = server.split(":")
    policy = WebsocketClientPolicy(host, int(port))
    verify_server(policy.get_server_metadata(), ckpt, contract)
    return contract, policy


def _init_worker(ckpt: str, servers, starvla_root: str, stable_regions: bool = True):
    """Never raises: a failing initializer makes multiprocessing respawn workers forever, so the error is kept and
    every episode of this worker reports it instead."""
    worker = mp.current_process()._identity[0] if mp.current_process()._identity else 1
    server = servers[(worker - 1) % len(servers)]
    _STATE.update(server=server, worker=worker)
    try:
        if stable_regions:
            import robocasa.wrappers.gym_wrapper  # noqa: F401  (imports RoboCasa before patching it)
            stable_counter_regions()
        contract, policy = connect(ckpt, server, starvla_root)
        _STATE.update(contract=contract, policy=policy)
    except Exception as exc:
        _STATE["init_error"] = f"{type(exc).__name__}: {exc}"[:500]


def run_episode(spec: dict) -> dict:
    out = dict(spec, worker=_STATE.get("worker"), server=_STATE.get("server"), pid=os.getpid())
    if "init_error" in _STATE:
        return dict(out, status="error", error="worker init: " + _STATE["init_error"], wall_seconds=0.0)
    import gymnasium as gym
    import numpy as np
    import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers robocasa/<task>)
    from vagen.envs.robocasa.starvla_contract import gym_action

    contract, policy = _STATE["contract"], _STATE["policy"]
    env = None
    started = time.monotonic()
    try:
        env = gym.make(f"robocasa/{spec['task']}", enable_render=True, split=spec["split"], seed=spec["seed"])
        obs, _ = env.reset(seed=spec["seed"])
        out["reset_seconds"] = round(time.monotonic() - started, 2)
        out["init"] = state_fingerprint(env, obs)
        out["init_qpos"] = initial_qpos(env)
        steps, calls, success, latencies = 0, 0, False, []
        terminated = truncated = False
        while steps < spec["max_steps"] and not (success or terminated or truncated):
            t = time.monotonic()
            response = policy.predict_action({"examples": [contract.example(obs)], "do_sample": False})
            latencies.append(time.monotonic() - t)
            actions = contract.actions(response["data"]["actions"])
            calls += 1
            for vector in actions[:min(spec["execute_steps"], spec["max_steps"] - steps)]:
                obs, reward, terminated, truncated, info = env.step(gym_action(vector, env.action_space))
                steps += 1
                success = success or bool(info.get("success", False))
                if success or terminated or truncated:
                    break
        out.update(success=bool(success), steps=steps, policy_calls=calls, terminated=bool(terminated),
                   truncated=bool(truncated), step_limit=steps >= spec["max_steps"],
                   inference_mean_seconds=round(float(np.mean(latencies)), 4) if latencies else None,
                   status="complete")
    except Exception as exc:
        out.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500], traceback=traceback.format_exc()[-2000:])
    finally:
        if env is not None:
            env.close()
    out["wall_seconds"] = round(time.monotonic() - started, 2)
    return out


def summarize(rows, workers: int, wall: float) -> dict:
    done = [r for r in rows if r.get("status") == "complete"]
    per_task = {}
    for r in done:
        t = per_task.setdefault(r["task"], {"n": 0, "successes": 0})
        t["n"] += 1
        t["successes"] += int(r["success"])
    for t in per_task.values():
        t["success_rate"] = t["successes"] / t["n"]
    mean = lambda key: round(sum(r[key] for r in done) / len(done), 3) if done else None  # noqa: E731
    return {"episodes": len(rows), "complete": len(done), "errors": len(rows) - len(done),
            "successes": sum(int(r["success"]) for r in done), "per_task": per_task, "workers": workers,
            "wall_seconds": round(wall, 1), "episode_wall_mean_seconds": mean("wall_seconds"),
            "reset_mean_seconds": mean("reset_seconds"), "steps_mean": mean("steps"),
            "seconds_per_step_mean": round(sum(r["wall_seconds"] - r["reset_seconds"] for r in done)
                                           / max(1, sum(r["steps"] for r in done)), 4) if done else None,
            "episodes_per_hour": round(len(done) / wall * 3600, 1) if wall > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--servers", required=True, help="host:port[,host:port...]; workers are spread over them")
    ap.add_argument("--eval-plan", type=Path, help="a fixed plan file (downstream/eval/eval_d6.json): tasks, seeds, "
                                                   "split, execute steps")
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--seeds", help="e.g. 1000-1019 or 1000,1003")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="target")
    ap.add_argument("--execute-steps", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--starvla-root", default=str(WORKTREE / "third_party/starVLA"))
    ap.add_argument("--no-stable-regions", dest="stable_regions", action="store_false",
                    help="leave RoboCasa's counter-region order as it is (differs between processes)")
    args = ap.parse_args()
    eval_plan = None
    if args.eval_plan:
        if args.tasks or args.seeds:
            ap.error("--eval-plan replaces --tasks / --seeds")
        episodes, eval_plan = eval_plan_episodes(args.eval_plan)
        args.split, args.execute_steps = eval_plan["split"], int(eval_plan["execute_steps"])
        if not eval_plan.get("stable_counter_regions", True):
            args.stable_regions = False
    elif args.tasks and args.seeds:
        episodes = plan(args.tasks, parse_seeds(args.seeds))
    else:
        ap.error("give --eval-plan, or --tasks and --seeds")
    args.out.mkdir(parents=True, exist_ok=False)
    episodes = longest_first(episodes)
    for e in episodes:
        e.update(split=args.split, execute_steps=args.execute_steps)
    (args.out / "plan.json").write_text(json.dumps({"ckpt": args.ckpt, "servers": args.servers, "workers": args.workers,
                                                    "stable_counter_regions": args.stable_regions,
                                                    "eval_plan": eval_plan and {"path": str(args.eval_plan),
                                                                                "name": eval_plan["name"]},
                                                    "episodes": episodes}, indent=1) + "\n")
    for server in args.servers.split(","):  # fail fast, before any simulator starts
        try:
            connect(args.ckpt, server, args.starvla_root)
        except Exception as exc:
            (args.out / "summary.json").write_text(json.dumps({"error": f"server {server}: {type(exc).__name__}: {exc}"},
                                                              indent=1) + "\n")
            print(f"[closed_loop] server {server} failed the contract check: {type(exc).__name__}: {exc}")
            return 2
    rows = []
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.ckpt, args.servers.split(","),
                                                                    args.starvla_root, args.stable_regions)) as pool:
        with open(args.out / "episodes.jsonl", "a") as f:
            for row in pool.imap_unordered(run_episode, episodes):
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
    summary = dict(summarize(rows, args.workers, time.monotonic() - t0), stable_counter_regions=args.stable_regions,
                   eval_plan=eval_plan["name"] if eval_plan else None)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("episodes", "complete", "successes", "wall_seconds",
                                              "episode_wall_mean_seconds", "episodes_per_hour")}))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
