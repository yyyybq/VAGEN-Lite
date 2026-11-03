#!/usr/bin/env python3
"""N3 (the transfer ladder; the user's decision 2 of 2026-11-03, design reports/N3_DESIGN_20261102.md): the two
changes to N1's RoboCasa side, as a subclass of n_env.Scene (n_env itself is left as it is: N2's jobs read it).

* collision "sphere" (tier c, the upstream body): only the camera's centre counts (camera A's, 1.5 m above the floor
  over the rotation axis).  A move is blocked as the upstream CollisionDetector.check_collision blocks it
  (vagen/envs/active_spatial/collision_detector.py, read here, not imported): the new centre inside, or the straight
  path from the old centre crossing before its end (slab test, t < 1), any collidable geom's world box (MuJoCo
  geom_aabb of the geoms with contype or conaffinity set, neither the robot's nor the floor's) grown by the upstream
  camera_radius 0.15 m plus safety_margin 0.05 m.  A turn keeps the centre where it is, so it is blocked only when the
  centre is already inside a box (as upstream).  RoboCasa's walls are geoms, so they count as boxes.  The robot is
  still posed (renders hide it) but its own contacts no longer block anything.
* camera "robot" (tier e, the robot's own camera): the height and the vertical field of view that the robot's own
  robot0_agentview_center camera has right after the reset (measured, not assumed), kept level, looking along the
  heading, over the rotation axis; rendered at 320 x 240 and resized to 256 x 256 as camera A.  The judge follows:
  every item gets the native intrinsics of that camera (640 x 480, fy = 240 / tan(fovy / 2), fx = fy, centre
  (320, 240)), and the prompt's pose line shows the new height.  The settings are n_env's module values
  (CAM_HEIGHT_M, FOVY_DEG, NATIVE_K, make_item) set by use_camera() once per process (each generation scene and each
  evaluation episode is a process of its own).

    python downstream/n/n3_env.py measure --task T --seed S   # the robot camera's height and field of view
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import n_env as N  # noqa: E402

UPSTREAM_RADIUS_M, UPSTREAM_MARGIN_M = 0.15, 0.05  # CollisionDetector(camera_radius, safety_margin) defaults
COLLISIONS = ("robot", "sphere")
CAMERAS = ("A", "robot")
_MAKE_ITEM = N.make_item


# ----------------------------------------------------------------------------------------------- camera
def native_k(fovy_deg: float) -> list:
    """Native (640 x 480) intrinsics of a camera with this vertical field of view and square pixels."""
    f = 240.0 / math.tan(math.radians(fovy_deg) / 2)
    return [[f, 0.0, 320.0], [0.0, f, 240.0], [0.0, 0.0, 1.0]]


def make_item_with_k(task_type, scene, ents, params, start_pose):
    """n_env.make_item with the current camera's intrinsics on the item (the judges read them from it)."""
    item = _MAKE_ITEM(task_type, scene, ents, params, start_pose)
    item["init_camera"]["intrinsics"] = [list(r) for r in N.NATIVE_K]
    return item


def use_camera(height_m: float, fovy_deg: float) -> None:
    """Camera settings for this process: n_env's height, field of view, native K and item maker."""
    N.CAM_HEIGHT_M, N.FOVY_DEG, N.NATIVE_K = float(height_m), float(fovy_deg), native_k(fovy_deg)
    N.make_item = make_item_with_k


# -------------------------------------------------------------------------------------------- collision
def segment_hits(lo, hi, p0, p1):
    """Per box (rows of lo / hi): does the path p0 -> p1 enter it before p1 (upstream AABB.intersects_segment with
    t < 1), or is p1 inside it (AABB.contains)?  numpy arrays; a zero-length path counts as p1 inside."""
    import numpy as np

    inside = np.all((p1 >= lo) & (p1 <= hi), axis=1)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 1e-10:
        return inside
    u = d / length
    t_min, t_max = np.zeros(len(lo)), np.full(len(lo), length)
    miss = np.zeros(len(lo), bool)
    for i in range(3):
        if abs(u[i]) < 1e-10:
            miss |= (p0[i] < lo[:, i]) | (p0[i] > hi[:, i])
        else:
            t1, t2 = (lo[:, i] - p0[i]) / u[i], (hi[:, i] - p0[i]) / u[i]
            t_min = np.maximum(t_min, np.minimum(t1, t2))
            t_max = np.minimum(t_max, np.maximum(t1, t2))
    crosses = ~miss & (t_min <= t_max) & (t_min / length < 1.0)
    return inside | crosses


class Scene(N.Scene):
    """n_env.Scene with N3's collision rule and camera (see the module text)."""

    def __init__(self, task: str, seed: int, split: str = "target", hide_robot: bool = True,
                 collision: str = "robot", camera: str = "A", camera_values=None):
        if collision not in COLLISIONS or camera not in CAMERAS:
            raise ValueError(f"collision {collision!r} / camera {camera!r}")
        self.collision_mode, self.camera_mode, self.camera_values = collision, camera, camera_values
        self.robot_camera = None
        super().__init__(task, seed, split=split, hide_robot=hide_robot)
        if collision == "sphere":
            self._boxes()

    def _place_camera(self) -> None:
        m, d = self.m, self.d
        self.robot_camera = {"height_m": float(d.cam_xpos[self.cam][2]), "fovy_deg": float(m.cam_fovy[self.cam])}
        if self.camera_mode == "robot":
            v = self.camera_values or self.robot_camera
            use_camera(v["height_m"], v["fovy_deg"])
        super()._place_camera()

    def camera_record(self) -> dict:
        return {"mode": self.camera_mode, "height_m": N.CAM_HEIGHT_M, "fovy_deg": N.FOVY_DEG,
                "native_k": N.NATIVE_K, "robot_camera_at_reset": self.robot_camera}

    # sphere collision
    def _boxes(self) -> None:
        np, m, d = self.np, self.m, self.d
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        grow = UPSTREAM_RADIUS_M + UPSTREAM_MARGIN_M
        ids, lo, hi = [], [], []
        for g in range(m.ngeom):
            if self.robot[g] or self.floor[g] or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            pts = np.asarray(d.geom_xpos[g]) + (c + h * signs) @ np.asarray(d.geom_xmat[g]).reshape(3, 3).T
            ids.append(g)
            lo.append(pts.min(0) - grow)
            hi.append(pts.max(0) + grow)
        self.box_geoms, self.box_lo, self.box_hi = ids, np.array(lo), np.array(hi)

    def centre(self, pose):
        return self.np.array([pose[0], pose[1], N.CAM_HEIGHT_M])

    def sphere_hit(self, old, new):
        """None, or the name of the first geom whose grown box blocks the camera centre's move old -> new."""
        hits = segment_hits(self.box_lo, self.box_hi, self.centre(old), self.centre(new))
        if not hits.any():
            return None
        g = self.box_geoms[int(self.np.flatnonzero(hits)[0])]
        return self.geom_entity.get(g) or self.geom_names[g] or "object"

    def try_action(self, pose, action: str):
        if self.collision_mode != "sphere":
            return super().try_action(pose, action)
        new = N.apply_action(pose, action)
        hit = self.sphere_hit(pose, new)
        self.set_pose(pose if hit is not None else new)
        return (pose, hit) if hit is not None else (new, None)

    def blocked(self, pose, new) -> bool:
        """Is the move pose -> new blocked under this scene's rule?  (The robot rule leaves the simulator at new; the
        sphere rule needs no simulator.)"""
        if self.collision_mode == "sphere":
            return self.sphere_hit(pose, new) is not None
        self.set_pose(new)
        return self.collision() is not None


def reachable(scene, max_depth: int = 12, max_states: int = 15000) -> dict:
    """n_smoke.reachable with the scene's own rule for each move (the sphere rule needs the move, not just the
    pose): {key: (pose, depth, parent key, action)} breadth first; a pose blocked from one parent may still be reached
    from another (the sphere rule depends on the path), so blocked moves are not remembered by pose."""
    start = scene.start
    k0 = N.pose_key(start)
    seen = {k0: (start, 0, None, None)}
    queue, capped, blocked = deque([k0]), False, 0
    while queue:
        k = queue.popleft()
        pose, depth, _, _ = seen[k]
        if depth >= max_depth:
            continue
        for action in N.ACTIONS:
            new = N.apply_action(pose, action)
            nk = N.pose_key(new)
            if nk in seen:
                continue
            if scene.blocked(pose, new):
                blocked += 1
                continue
            seen[nk] = (new, depth + 1, k, action)
            queue.append(nk)
            if len(seen) >= max_states:
                capped, queue = True, deque()
                break
    scene.set_pose(start)
    return {"states": seen, "blocked": blocked, "capped": capped}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    me = sub.add_parser("measure")
    me.add_argument("--task", required=True)
    me.add_argument("--seed", type=int, required=True)
    args = ap.parse_args(argv)
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    sc = Scene(args.task, args.seed, hide_robot=True)
    print(json.dumps({"task": args.task, "seed": args.seed, "robot_camera": sc.robot_camera}), flush=True)
    import os

    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
