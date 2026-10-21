"""M-check, rule implementation (proprioception only, no model calls).

Detects only what proprioception can show:
  * stall: base, arm and gripper joints essentially still for ``stall_window_steps`` while a
    subtask is running -> FAILED("no_progress");
  * holding estimate: a gripper commanded closed (last action < 0) whose fingers stopped short of
    fully closed is probably holding something -> holding "object"; fully closed -> "".
It never claims DONE (proprioception cannot see goal conditions); otherwise IN_PROGRESS.
"""

from __future__ import annotations

import numpy as np

from agent.fields import ACTION_SLICES, PROPRIO_SLICES
from agent.interfaces import Judgment, JudgmentStatus


class RuleChecker:
    def __init__(self, deps):
        p = dict(deps.params)
        self.window = int(p.get("stall_window_steps", 240))
        self.joint_eps = float(p.get("joint_motion_eps", 0.02))  # rad, max abs change over the window
        self.base_eps = float(p.get("base_speed_eps", 0.02))  # m/s or rad/s
        self.closed_width = float(p.get("closed_width", 0.004))  # m, sum of the two fingers

    def reset(self, ctx):
        pass

    def judge(self, obs, state) -> Judgment:
        hist = state.proprio_history
        holding = {}
        last = state.last_action
        for arm, gs, asl in (("left", "gripper_left_qpos", "left_gripper"), ("right", "gripper_right_qpos", "right_gripper")):
            width = float(np.sum(obs.proprio[PROPRIO_SLICES[gs]]))
            if last is not None and float(np.asarray(last)[ACTION_SLICES[asl]][0]) < 0:
                holding[arm] = "object" if width > self.closed_width else ""
        if state.current_subtask is None or len(hist) < self.window or state.steps_in_subtask() < self.window:
            return Judgment(status=JudgmentStatus.IN_PROGRESS, confidence=0.5, evidence="not enough history",
                            holding_updates=holding, source="check.rule")
        window = [p for (s, p) in hist if s > state.step - self.window]
        arr = np.stack(window)
        joints = np.concatenate([
            arr[:, PROPRIO_SLICES["arm_left_qpos"]], arr[:, PROPRIO_SLICES["arm_right_qpos"]],
            arr[:, PROPRIO_SLICES["trunk_qpos"]], arr[:, PROPRIO_SLICES["gripper_left_qpos"]],
            arr[:, PROPRIO_SLICES["gripper_right_qpos"]]], axis=1)
        joint_motion = float(np.max(np.abs(joints - joints[0])))
        base_speed = float(np.max(np.abs(arr[:, PROPRIO_SLICES["base_qvel"]])))
        if joint_motion < self.joint_eps and base_speed < self.base_eps:
            return Judgment(status=JudgmentStatus.FAILED, confidence=0.8, failure_type="no_progress",
                            evidence=f"no motion for {self.window} steps (joint {joint_motion:.3f} rad, base {base_speed:.3f})",
                            holding_updates=holding, source="check.rule")
        return Judgment(status=JudgmentStatus.IN_PROGRESS, confidence=0.5,
                        evidence=f"moving (joint {joint_motion:.3f} rad, base {base_speed:.3f})",
                        holding_updates=holding, source="check.rule")
