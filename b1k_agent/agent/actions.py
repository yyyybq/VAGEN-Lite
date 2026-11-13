"""Action helpers that use proprioception only (legal inputs).

Shared by the loop (hold action) and by modules that build scripted motions (recovery,
approach). Pure functions; no module state.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agent.fields import ACTION_SLICES, PROPRIO_SLICES
from agent.interfaces import ACTION_DIM_R1PRO

# Gripper finger joint range on R1Pro used to map a measured width back to a smooth-mode command.
# Fingers start open at 0.05 per finger (eval/r1pro.yaml reset_joint_pos); fully closed is 0.0.
GRIPPER_FINGER_OPEN = 0.05


def gripper_command_from_qpos(finger_qpos: np.ndarray) -> float:
    """Approximate the smooth-mode command (-1 closed .. +1 open) that holds the current width."""
    width = float(np.clip(np.mean(finger_qpos) / GRIPPER_FINGER_OPEN, 0.0, 1.0))
    return 2.0 * width - 1.0


def hold_action(proprio: np.ndarray, last_action: Optional[np.ndarray] = None) -> np.ndarray:
    """Zero base velocity, hold torso and arm joints where they are, keep gripper commands."""
    p = np.asarray(proprio, dtype=np.float32)
    a = np.zeros(ACTION_DIM_R1PRO, dtype=np.float32)
    a[ACTION_SLICES["torso"]] = p[PROPRIO_SLICES["trunk_qpos"]]
    a[ACTION_SLICES["left_arm"]] = p[PROPRIO_SLICES["arm_left_qpos"]]
    a[ACTION_SLICES["right_arm"]] = p[PROPRIO_SLICES["arm_right_qpos"]]
    if last_action is not None:
        la = np.asarray(last_action, dtype=np.float32).reshape(-1)
        a[ACTION_SLICES["left_gripper"]] = la[ACTION_SLICES["left_gripper"]]
        a[ACTION_SLICES["right_gripper"]] = la[ACTION_SLICES["right_gripper"]]
    else:
        a[ACTION_SLICES["left_gripper"]] = gripper_command_from_qpos(p[PROPRIO_SLICES["gripper_left_qpos"]])
        a[ACTION_SLICES["right_gripper"]] = gripper_command_from_qpos(p[PROPRIO_SLICES["gripper_right_qpos"]])
    return a


def interpolate_joints(start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
    """(steps, n) linear joint-space interpolation that ends exactly at goal."""
    steps = max(int(steps), 1)
    alphas = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)[:, None]
    return (1.0 - alphas) * np.asarray(start, np.float32)[None] + alphas * np.asarray(goal, np.float32)[None]


def validate_actions(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    if a.ndim != 2 or a.shape[1] != ACTION_DIM_R1PRO:
        raise ValueError(f"actions must have shape (K, {ACTION_DIM_R1PRO}), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("actions contain non-finite values")
    return a
