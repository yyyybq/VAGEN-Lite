"""M-recover, minimal bounded recovery.

Scripted primitives use proprioception only and the official 23-D action layout:
  retract_arms     interpolate both arms to the safe pose (default: the R1Pro reset pose, all arm
                   joints 0 -- eval/r1pro.yaml reset_joint_pos), torso and grippers held, base still
  back_up          base vx = -back_up_speed (normalised command) for N steps, joints held
  rotate_in_place  base wz = +-rotate_speed for N steps (direction alternates per attempt), joints held
Instruction primitives (text for the executor) are proposed only for skills that the catalog marks
available AND verified (the Step-5 diagnosis); otherwise the module never emits them.

The loop bounds retries (loop.max_retries_per_subtask) and replans; this module bounds per-failure
attempts and returns None when it has nothing sensible left (the loop then keeps executing).
"""

from __future__ import annotations

import numpy as np

from agent.actions import hold_action, interpolate_joints
from agent.fields import ACTION_SLICES, PROPRIO_SLICES
from agent.interfaces import RecoveryKind, RecoveryPlan

DEFAULT_PLAYBOOK = {
    "grasp_failed": ["retract_arms", "back_up"],
    "object_dropped": ["retract_arms"],
    "no_progress": ["back_up", "rotate_in_place", "retract_arms"],
    "out_of_reach": ["back_up", "rotate_in_place"],
    "collision": ["back_up", "retract_arms"],
    "timeout": ["retract_arms", "rotate_in_place"],
    "wrong_object": ["retract_arms"],
    "subgoal_regressed": ["retract_arms"],
    "unknown": ["retract_arms"],
}


class ScriptedRecovery:
    def __init__(self, deps):
        p = dict(deps.params)
        self.playbook = {**DEFAULT_PLAYBOOK, **dict(p.get("playbook") or {})}
        self.retract_steps = int(p.get("retract_steps", 60))
        self.back_up_steps = int(p.get("back_up_steps", 45))
        self.back_up_speed = float(p.get("back_up_speed", 0.3))
        self.rotate_steps = int(p.get("rotate_steps", 60))
        self.rotate_speed = float(p.get("rotate_speed", 0.4))
        self.safe_left = np.asarray(p.get("safe_arm_left", [0.0] * 7), np.float32)
        self.safe_right = np.asarray(p.get("safe_arm_right", [0.0] * 7), np.float32)
        self.instruction_recovery = bool(p.get("instruction_recovery", False))
        self.instruction_skills = dict(p.get("instruction_skills") or {})  # failure_type -> skill name
        self.max_attempts_per_type = int(p.get("max_attempts_per_type", 3))

    def reset(self, ctx):
        self.ctx = ctx
        self.attempts: dict[str, int] = {}

    def _held(self, obs, state, n: int) -> np.ndarray:
        a = hold_action(obs.proprio, state.last_action)
        return np.repeat(a[None], n, axis=0)

    def primitive(self, name: str, obs, state) -> np.ndarray:
        if name == "retract_arms":
            acts = self._held(obs, state, self.retract_steps)
            acts[:, ACTION_SLICES["left_arm"]] = interpolate_joints(obs.proprio[PROPRIO_SLICES["arm_left_qpos"]], self.safe_left, self.retract_steps)
            acts[:, ACTION_SLICES["right_arm"]] = interpolate_joints(obs.proprio[PROPRIO_SLICES["arm_right_qpos"]], self.safe_right, self.retract_steps)
            return acts
        if name == "back_up":
            acts = self._held(obs, state, self.back_up_steps)
            acts[:, 0] = -abs(self.back_up_speed)
            return acts
        if name == "rotate_in_place":
            acts = self._held(obs, state, self.rotate_steps)
            sign = 1.0 if (sum(self.attempts.values()) % 2 == 0) else -1.0
            acts[:, 2] = sign * abs(self.rotate_speed)
            return acts
        raise KeyError(name)

    def propose(self, obs, state, failure_type: str):
        ft = failure_type if failure_type in self.playbook else "unknown"
        n = self.attempts.get(ft, 0)
        if n >= self.max_attempts_per_type:
            return None
        self.attempts[ft] = n + 1
        if self.instruction_recovery and ft in self.instruction_skills:
            skill = next((s for s in self.ctx.skills if s.name == self.instruction_skills[ft]), None)
            if skill is not None and skill.available and skill.verified and not skill.inputs:
                return RecoveryPlan(kind=RecoveryKind.INSTRUCTION, primitive=skill.name,
                                    instruction=skill.instruction_template, duration_steps=self.retract_steps * 3,
                                    reason=f"{ft}: verified recovery skill", source="recover.scripted")
        seq = self.playbook[ft]
        name = seq[n % len(seq)]
        return RecoveryPlan(kind=RecoveryKind.SCRIPTED, primitive=name, actions=self.primitive(name, obs, state),
                            duration_steps=0, reason=f"{ft} attempt {n + 1}", source="recover.scripted")
