"""Test helpers: build an AgentPolicy with a recording fake executor and scripted providers.

Everything here is a MOCK: no simulator, no model. It checks wiring and the red lines, not skill.
"""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np

from agent.actions import hold_action
from agent.config import load_config
from agent.interfaces import ActionChunk
from agent.recorder import RunRecorder
from agent.registry import build_modules
from agent.server import AgentPolicy
from agent.task_meta import load_task_metadata
from executor import load_skill_catalog
from providers import build_providers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")
TASK_PROMPT = "Turn on the radio receiver that's on the table in the living room."

PLAN_REPLY = {"task_complete": False, "skill": "press_power", "args": [], "target_subgoals": ["g0"],
              "max_steps": 300, "success_check": "radio display lit", "failure_cues": ["radio on the floor"],
              "evidence": "radio visible on the table"}
CHECK_REPLIES = [
    {"status": "in_progress", "confidence": 0.7, "evidence": "arm moving toward radio", "failure_type": "none",
     "subgoals": [{"id": "g0", "estimate": "unsatisfied"}], "holding_left": "", "holding_right": ""},
    {"status": "failed", "confidence": 0.9, "evidence": "gripper closed on nothing", "failure_type": "grasp_failed",
     "subgoals": [], "holding_left": "", "holding_right": ""},
    {"status": "done", "confidence": 0.9, "evidence": "radio display lit", "failure_type": "none",
     "subgoals": [{"id": "g0", "estimate": "satisfied"}], "holding_left": "", "holding_right": ""},
]


class FakeExecutor:
    """Deterministic stand-in for the pi0.5 executor: action depends only on (proprio, instruction)."""

    name = "fake"

    def __init__(self, env_idx=0):
        self.calls = []
        self.interrupts = 0
        self.resets = 0

    def default_prompt(self, task_name):
        return TASK_PROMPT

    def reset(self, ctx):
        self.resets += 1

    def interrupt(self):
        self.interrupts += 1

    def act(self, obs, instruction, horizon):
        self.calls.append((obs.step, instruction))
        a = hold_action(obs.proprio)
        a[0] = 0.01 * (len(instruction) % 7)  # instruction-dependent, deterministic
        return ActionChunk(actions=a[None], instruction=instruction, source="exec.fake")

    def close(self):
        pass


def make_cfg(name: str, overrides=()) -> dict:
    cfg = load_config(os.path.join(ROOT, "configs", f"{name}.yaml"), list(overrides))
    cfg = copy.deepcopy(cfg)
    cfg["task_metadata"] = {"id_to_name": {0: "turning_on_radio"}, "bddl_root": FIX, "task_misc_csv": None}
    cfg["executor"]["skill_catalog"] = os.path.join(FIX, "skills_test.yaml")
    scripted = {"kind": "scripted", "replies": {"plan.next_subtask": [PLAN_REPLY], "check.judge": CHECK_REPLIES}}
    cfg["providers"] = {"astra": dict(scripted), "claude": dict(scripted)}
    return cfg


def make_policy(cfg: dict, tmp: str | None = None, max_steps: int = 2000):
    tmp = tmp or tempfile.mkdtemp()
    rec = RunRecorder("test", tmp, save_frames=False)
    providers = build_providers(cfg["providers"], recorder=rec)
    executors = {}

    def make_executor(i):
        executors[i] = FakeExecutor(i)
        return executors[i]

    policy = AgentPolicy(
        cfg,
        make_executor=make_executor,
        make_modules=lambda i: build_modules(cfg["modules"], providers, recorder=rec),
        task_meta=load_task_metadata(cfg["task_metadata"]),
        recorder=rec,
        run_id="test",
        max_steps=max_steps,
        skills_for_task=load_skill_catalog(cfg["executor"].get("skill_catalog")),
        clock=_FakeClock(),
    )
    return policy, executors, tmp


class _FakeClock:
    """Monotonic fake clock: 1/30 s per call, so runs are deterministic and budget logic is exercised."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1.0 / 30.0
        return self.t


def run_episode(policy, obs_fn, n_steps: int) -> list[np.ndarray]:
    policy.reset()
    actions = []
    for t in range(n_steps):
        actions.append(policy.act(obs_fn(t)).copy())
    policy.reset()  # closes the episode (writes summaries)
    return actions
