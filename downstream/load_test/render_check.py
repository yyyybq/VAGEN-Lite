#!/usr/bin/env python3
"""Can a RoboCasa365 kitchen render off-screen here?  One backend per process (MUJOCO_GL is read at import).

Run with the RoboCasa simulator Python that the StarVLA closed-loop client uses
(probe_spatial/robocasa365/policies/diffusion_policy/.venv) and the same PYTHONPATH as
examples/evaluate/robocasa/run_starvla_mobile.sh:

    MUJOCO_GL=egl python downstream/load_test/render_check.py --out egl.json [--task NavigateKitchen]

Two stages, each recorded separately: (1) a one-sphere MuJoCo scene rendered at 64x64 (the existing preflight);
(2) a real kitchen: gym.make('robocasa/<task>', split='target'), reset, a few zero actions, three cameras.
A stage-1 pass alone does not prove a kitchen renders.
"""
from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sphere() -> dict:
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size="0.1" rgba="1 0 0 1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        image = renderer.render()
        return {"ok": bool(image.shape == (64, 64, 3) and image.max() > image.min()), "shape": list(image.shape),
                "range": [int(image.min()), int(image.max())]}
    finally:
        renderer.close()


def kitchen(task: str, steps: int, seed: int) -> dict:
    import gymnasium as gym
    import numpy as np
    import robocasa  # noqa: F401
    import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers robocasa/<task>)
    from vagen.envs.robocasa.starvla_contract import gym_action

    t0 = time.time()
    env = gym.make(f"robocasa/{task}", enable_render=True, split="target", seed=seed)
    try:
        obs, _ = env.reset(seed=seed)
        reset_s = time.time() - t0
        cams = sorted(k for k in obs if k.startswith("video."))
        t1 = time.time()
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(gym_action(np.zeros(12, dtype=np.float32), env.action_space))
        step_s = (time.time() - t1) / max(steps, 1)
        images = {k: {"shape": list(np.asarray(obs[k]).shape), "nonconstant": bool(np.ptp(np.asarray(obs[k])) > 0)}
                  for k in cams}
        return {"ok": bool(cams) and all(v["nonconstant"] for v in images.values()), "task": task, "seed": seed,
                "instruction": obs.get("annotation.human.task_description"), "cameras": images,
                "reset_seconds": round(reset_s, 1), "seconds_per_step": round(step_s, 3), "steps": steps,
                "robocasa": getattr(robocasa, "__version__", None), "robocasa_file": robocasa.__file__}
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default="NavigateKitchen")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report = {"MUJOCO_GL": os.environ.get("MUJOCO_GL"), "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
              "EGL_library": ctypes.util.find_library("EGL"), "OSMesa_library": ctypes.util.find_library("OSMesa"),
              "NUMBA_DISABLE_JIT": os.environ.get("NUMBA_DISABLE_JIT"), "python": sys.executable,
              "host": os.uname().nodename}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for stage, fn in (("sphere", sphere), ("kitchen", lambda: kitchen(args.task, args.steps, args.seed))):
        try:
            report[stage] = fn()
        except Exception as exc:
            report[stage] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500],
                             "traceback": traceback.format_exc()[-3000:]}
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
        if not report[stage]["ok"]:
            break
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in report.items()}))
    return 0 if report.get("kitchen", {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
