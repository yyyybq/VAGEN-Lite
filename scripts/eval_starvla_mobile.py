#!/usr/bin/env python3
"""Seeded RoboCasa closed-loop evaluation using a StarVLA websocket server."""
from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/starVLA"))


def render_preflight():
    """Exercise the real render context before loading a multi-GB policy."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size="0.1" rgba="1 0 0 1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        image = renderer.render()
        if image.shape != (64, 64, 3) or not np.isfinite(image).all() or image.max() == image.min():
            raise RuntimeError("Renderer did not produce a nonconstant RGB image")
        return {"shape": list(image.shape), "range": [int(image.min()), int(image.max())]}
    finally:
        renderer.close()


def rollout(env, policy, contract, seed, max_steps, execute_steps, writer=None):
    """Single-env loop: no vector autoreset ambiguity or cross-episode chunks."""
    import numpy as np
    from vagen.envs.robocasa.starvla_contract import gym_action

    obs, _ = env.reset(seed=seed)
    language = obs["annotation.human.task_description"]
    steps, calls, success = 0, 0, False
    latencies = []
    started = time.monotonic()
    while steps < max_steps:
        t = time.monotonic()
        response = policy.predict_action({"examples": [contract.example(obs)], "do_sample": False})
        latencies.append(time.monotonic() - t)
        actions = contract.actions(response["data"]["actions"])
        calls += 1
        for vector in actions[:min(execute_steps, max_steps - steps)]:
            obs, reward, terminated, truncated, info = env.step(gym_action(vector, env.action_space))
            steps += 1
            if writer:
                writer.append_data(obs[contract.cameras[0]])
            success = success or bool(info.get("success", False))
            if success or terminated or truncated or steps >= max_steps:
                return {"seed": seed, "instruction": language, "success": success,
                        "steps": steps, "policy_calls": calls, "terminated": bool(terminated),
                        "truncated": bool(truncated), "step_limit": steps >= max_steps,
                        "wall_seconds": time.monotonic() - started,
                        "inference_mean_seconds": float(np.mean(latencies))}
    raise RuntimeError("No environment steps executed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--task", action="append", default=[])
    p.add_argument("--split", choices=["pretrain", "target"], default="target")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5678)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video", action="store_true")
    p.add_argument("--render-only", action="store_true")
    args = p.parse_args()
    if min(args.episodes, args.max_steps, args.execute_steps) <= 0:
        p.error("episodes, max-steps and execute-steps must be positive")
    if not args.render_only and args.ckpt is None:
        p.error("--ckpt is required for closed-loop evaluation")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "runtime": {"python": sys.executable, "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
                          "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
                          "EGL_library": ctypes.util.find_library("EGL"),
                          "NUMBA_DISABLE_JIT": os.environ.get("NUMBA_DISABLE_JIT")}, "episodes": []}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    policy = None
    try:
        report["renderer"] = render_preflight()
        if not args.render_only:
            import gymnasium as gym
            import robocasa.wrappers.gym_wrapper  # registers environments
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
            from vagen.envs.robocasa.starvla_contract import PolicyContract, checkpoint_config, verify_server

            config, _ = checkpoint_config(args.ckpt)
            contract = PolicyContract.from_config(config)
            if args.execute_steps > contract.horizon:
                raise ValueError("execute-steps exceeds the model action horizon")
            report["contract"] = vars(contract)
            policy = WebsocketClientPolicy(args.host, args.port)
            report["server"] = policy.get_server_metadata()
            verify_server(report["server"], args.ckpt, contract)
            for task in args.task or ["NavigateKitchen"]:
                if not task.replace("_", "").isalnum():
                    raise ValueError(f"Invalid task name {task!r}")
                for episode in range(args.episodes):
                    seed = args.seed + episode
                    env, writer = None, None
                    try:
                        env = gym.make(f"robocasa/{task}", enable_render=True, split=args.split, seed=seed)
                        if args.video:
                            import imageio.v2 as imageio
                            writer = imageio.get_writer(str(args.out / f"{task}_{seed}.mp4"), fps=20)
                        result = rollout(env, policy, contract, seed, args.max_steps, args.execute_steps, writer)
                        result["task"] = task
                        report["episodes"].append(result)
                        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                    finally:
                        if writer is not None:
                            writer.close()
                        if env is not None:
                            env.close()
            report["per_task"] = {}
            for task in sorted({e["task"] for e in report["episodes"]}):
                episodes = [e for e in report["episodes"] if e["task"] == task]
                report["per_task"][task] = {"n": len(episodes), "success_rate": sum(e["success"] for e in episodes) / len(episodes)}
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if policy is not None:
            policy.close()
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
