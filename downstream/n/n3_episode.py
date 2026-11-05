#!/usr/bin/env python3
"""N3 (the transfer ladder; the user's decision 2 of 2026-11-03): n_episode.py with n3_env's scene, for tier (c)
(the upstream camera-sphere collision), tier (e) (the robot's own camera) and, since 11-05, tier (f) (N1's robot body
with turns never blocked).

    python downstream/n/n3_episode.py --collision sphere|robot|robot_free_turn --instances FILE [n_episode.py's options]

The camera is the instances' own: tier (e)'s carry the one they were made with (n3_gen.py scene: "camera"), N1's
carry none (camera A).  All instances of one run must share it (stops otherwise); it is set in this process before
any episode is forked (n3_env.use_camera), so every episode, its judge and its prompt's pose line use it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import n3_env as N3  # noqa: E402
import n_env as N  # noqa: E402
import n_episode as E  # noqa: E402


def camera_of(insts: list):
    """(mode, values or None) shared by all instances, the height to 0.1 mm and the field of view to 0.01 degree (each
    scene measures its own robot camera: 1.75 m to within 1e-8, 11-03's first tier-e run stopped on that);
    SystemExit when they differ beyond that."""
    def key(i):
        c = i.get("camera") or {"mode": "A"}
        return json.dumps({"mode": c.get("mode"),
                           "height_m": None if c.get("height_m") is None else round(float(c["height_m"]), 4),
                           "fovy_deg": None if c.get("fovy_deg") is None else round(float(c["fovy_deg"]), 2)},
                          sort_keys=True)

    keys = {key(i) for i in insts}
    if len(keys) > 1:
        raise SystemExit(f"the instances use {len(keys)} different cameras: {sorted(keys)[:3]}")
    c = json.loads(keys.pop()) if keys else {"mode": "A"}
    values = {"height_m": c["height_m"], "fovy_deg": c["fovy_deg"]} if c["mode"] == "robot" else None
    return c["mode"], values


def install(collision: str, insts: list) -> tuple:
    """n_env.Scene replaced by n3_env's with this collision rule and the instances' camera (set now)."""
    mode, values = camera_of(insts)
    if values:
        N3.use_camera(values["height_m"], values["fovy_deg"])
    N.Scene = lambda task, seed, hide_robot=False: N3.Scene(task, seed, hide_robot=hide_robot, collision=collision,
                                                          camera=mode, camera_values=values)
    return mode, values


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--collision", choices=N3.COLLISIONS, default="robot")
    ap.add_argument("--instances", nargs="+", required=True)
    args, rest = ap.parse_known_args(argv)
    mode, values = install(args.collision, E.load_instances(args.instances, 0))
    print(f"[n3_episode] collision {args.collision}, camera {mode} {values or ''}", flush=True)
    return E.main(["--instances", *args.instances, *rest])


if __name__ == "__main__":
    sys.exit(main())
