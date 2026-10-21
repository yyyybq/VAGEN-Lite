"""Interface contracts between the episode loop (agent/loop.py) and the modules.

These types and protocols are the only things the loop and the modules share. They are kept
stable: a change here is a protocol change and must be noted in docs/ARCHITECTURE.md.

Rules that this file encodes (see CLAUDE.md, "modularity red lines"):
  * every value a module returns is one of the frozen dataclasses below;
  * the loop branches only on these return values (``Judgment.status``, ``RecoveryPlan.kind``,
    ``Subtask is None``, ...), never on module internals;
  * modules receive the episode state only as a read-only ``StateView`` (agent/state.py).

Nothing in here may carry simulator ground truth. ``Observation`` holds only fields of the
"policy_observation" category (agent/fields.py).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from agent.state import StateView

# R1Pro action layout used by the official evaluator (OmniGibson/omnigibson/eval/utils/eval_utils.py,
# ACTION_QPOS_INDICES["R1Pro"]); 23 dims, see docs/PROTOCOL.md section "Action space".
ACTION_DIM_R1PRO = 23

FULL_TASK_SKILL = "__full_task__"


# ----------------------------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Observation:
    """One logical environment's legal observation at one control step.

    ``policy_dict`` is the whitelisted subset of the dict the official evaluator sent for this
    environment, with the leading batch dimension of size 1 kept, so an executor can forward it
    unchanged (L0 must be byte-identical to a direct evaluator -> executor connection).
    The parsed views (``rgb``, ``depth``, ``proprio``) are conveniences for modules.
    """

    step: int  # control steps since the episode reset (0 = first observation)
    received_at: float  # time.monotonic() when the server received it
    task_id: int
    rgb: Mapping[str, np.ndarray]  # camera role ("head", "left_wrist", "right_wrist") -> (H, W, 3) uint8
    depth: Mapping[str, np.ndarray]  # camera role -> (H, W) float32, metres; empty without an RGB-D wrapper
    proprio: np.ndarray  # (61,) float32, R1Pro layout (agent/fields.py PROPRIO_SLICES)
    cam_rel_poses: Optional[np.ndarray]  # camera poses relative to the robot base, as sent by the evaluator
    policy_dict: Mapping[str, np.ndarray]
    dropped_keys: tuple[str, ...] = ()  # keys received but not whitelisted (logged, never used)


# ----------------------------------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ActionChunk:
    """Actions an executor proposes from one observation.

    ``actions`` has shape (H, 23). In passthrough mode H == 1 and the executor's own server applies
    its official chunking / temporal-ensembling rules. ``open_loop_steps`` is how many of the H
    actions the loop may execute without asking the executor again.
    """

    actions: np.ndarray
    instruction: str
    source: str  # e.g. "exec.openpi_ws@passthrough"
    infer_ms: float = 0.0
    open_loop_steps: int = 1
    meta: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class SkillSpec:
    """A skill the executor can perform, as described to the planner.

    Only skills with ``available=True`` may appear in planner prompts. ``verified`` records
    whether a Step-5 diagnosis showed that the executor actually responds to this instruction.
    """

    name: str
    instruction_template: str  # e.g. "pick up the {object}"; FULL_TASK_SKILL uses the task prompt
    description: str
    inputs: tuple[str, ...] = ()
    ends_when: str = ""
    known_failures: tuple[str, ...] = ()
    available: bool = True
    verified: bool = False
    evidence: str = ""  # where the verification / diagnosis result lives


@runtime_checkable
class Executor(Protocol):
    name: str

    def reset(self, ctx: "EpisodeContext") -> None: ...

    def act(self, obs: Observation, instruction: str, horizon: int) -> ActionChunk: ...

    def interrupt(self) -> None:
        """Drop any buffered actions so the next act() re-infers from the current observation."""
        ...

    def close(self) -> None: ...


# ----------------------------------------------------------------------------------------------
# Task / goal description (from the BDDL task definition, a legal policy input)
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Subgoal:
    subgoal_id: str  # "g0", "g1", ...
    literal: str  # BDDL text of the literal, e.g. "(toggled_on radio.n.01_1)"
    predicate: str  # "toggled_on"
    args: tuple[str, ...]
    negated: bool = False
    option: int = 0  # index of the disjunctive goal option it belongs to (0 when no disjunction)
    quantified: bool = False  # True when it came from an unexpanded forall/exists/forn


@dataclasses.dataclass(frozen=True)
class GoalSpec:
    task_name: str
    bddl_path: str
    objects: Mapping[str, str]  # object instance -> category, from :objects
    init_literals: tuple[str, ...]
    subgoals: tuple[Subgoal, ...]
    goal_text: str  # raw :goal s-expression
    parse_notes: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class EpisodeContext:
    """Static facts about one episode, fixed at reset."""

    episode_id: str
    env_index: int
    task_name: str
    task_id: int
    task_instruction: str  # natural-language task prompt (what the executor was trained with)
    goal_spec: Optional[GoalSpec]
    skills: tuple[SkillSpec, ...]
    max_steps: Optional[int]  # official step cap; the 2026 wall-clock budget is max_steps * 1 s
    wallclock_budget_s: Optional[float]
    seed: int = 0


# ----------------------------------------------------------------------------------------------
# Planner / checker / recovery / memory / scorer / goal manager
# ----------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Subtask:
    subtask_id: str
    skill: str
    instruction: str
    args: Mapping[str, str] = dataclasses.field(default_factory=dict)
    target_subgoals: tuple[str, ...] = ()
    max_steps: Optional[int] = None  # executor steps before the loop forces a check
    source: str = ""
    evidence: str = ""


class JudgmentStatus(str, enum.Enum):
    DONE = "done"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    UNKNOWN = "unknown"


# Failure types a checker may report; recovery modules key off these strings.
FAILURE_TYPES = (
    "no_progress",  # stall: the robot is not changing anything relevant
    "grasp_failed",
    "object_dropped",
    "out_of_reach",  # arrived but cannot reach / wrong standing position
    "wrong_object",
    "collision",
    "subgoal_regressed",  # a subgoal estimated as satisfied is now estimated unsatisfied
    "timeout",  # subtask exceeded its step budget
    "unknown",
)


@dataclasses.dataclass(frozen=True)
class Judgment:
    status: JudgmentStatus
    confidence: float = 0.0
    evidence: str = ""  # visible evidence (what in the observation supports the judgment)
    failure_type: Optional[str] = None
    # subgoal id -> "satisfied" | "unsatisfied" | "unknown"; system estimates, never ground truth
    subgoal_updates: Mapping[str, str] = dataclasses.field(default_factory=dict)
    # arm ("left" | "right") -> held object name, "" for empty, "unknown" when unsure
    holding_updates: Mapping[str, str] = dataclasses.field(default_factory=dict)
    source: str = ""


class RecoveryKind(str, enum.Enum):
    SCRIPTED = "scripted"  # a fixed action sequence computed from proprioception only
    INSTRUCTION = "instruction"  # a text instruction for the executor (needs a verified skill)
    REPLAN = "replan"  # ask the planner for a new subtask
    GIVE_UP = "give_up"  # stop retrying this subtask


@dataclasses.dataclass(frozen=True)
class RecoveryPlan:
    kind: RecoveryKind
    primitive: str = ""
    actions: Optional[np.ndarray] = None  # (K, 23) for SCRIPTED
    instruction: Optional[str] = None  # for INSTRUCTION
    duration_steps: int = 0
    reason: str = ""
    source: str = ""


@dataclasses.dataclass(frozen=True)
class Experience:
    exp_id: str
    provenance: str  # dataset / version / episode / frame range, or dev-run id
    skill: str
    objects: tuple[str, ...]
    summary: str
    outcome: str  # "success" | "failure" | "partial"
    payload: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class AdjustPlan:
    """A short adjustment before handing over to the next subtask (memory bridge, approach)."""

    actions: Optional[np.ndarray] = None  # (K, 23)
    instruction: Optional[str] = None
    reason: str = ""
    source: str = ""
    verify_after: bool = True  # loop runs a check once the adjustment finishes


@runtime_checkable
class Planner(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...

    def next_subtask(self, obs: Observation, state: "StateView") -> Optional[Subtask]: ...


@runtime_checkable
class Checker(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...

    def judge(self, obs: Observation, state: "StateView") -> Judgment: ...


@runtime_checkable
class Recovery(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...

    def propose(self, obs: Observation, state: "StateView", failure_type: str) -> Optional[RecoveryPlan]: ...


@runtime_checkable
class Memory(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...

    def retrieve(self, obs: Observation, state: "StateView", k: int) -> list[Experience]: ...

    def propose_adjustment(
        self, obs: Observation, state: "StateView", next_subtask: Subtask
    ) -> Optional[AdjustPlan]: ...


@runtime_checkable
class Scorer(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...

    def score(self, obs: Observation, state: "StateView", candidates: Sequence[ActionChunk]) -> list[float]: ...


@runtime_checkable
class GoalManager(Protocol):
    def reset(self, ctx: EpisodeContext) -> None: ...

    def order(self, goal_spec: Optional[GoalSpec], state: "StateView") -> list[str]: ...

    def regression_risk(self, plan: Subtask, state: "StateView") -> float: ...

    def low_risk_mode(self, state: "StateView") -> bool:
        """True when the remaining budget is so small that the loop should avoid risky motion."""
        ...


@runtime_checkable
class Approach(Protocol):
    """M-approach: adjust the base to a workable distance before a manipulation subtask."""

    def reset(self, ctx: EpisodeContext) -> None: ...

    def propose(self, obs: Observation, state: "StateView", subtask: Subtask) -> Optional[AdjustPlan]: ...


@runtime_checkable
class SpatialMemory(Protocol):
    """M-spatial: a map built only from legal observations (no prior maps, no instance labels)."""

    def reset(self, ctx: EpisodeContext) -> None: ...

    def update(self, obs: Observation, state: "StateView") -> Optional[str]:
        """Return a short text summary for the planner, or None."""
        ...


@runtime_checkable
class Explorer(Protocol):
    """M-explore: propose an exploration subtask when the target is not in view. Off by default."""

    def reset(self, ctx: EpisodeContext) -> None: ...

    def propose(self, obs: Observation, state: "StateView") -> Optional[Subtask]: ...


# Slot name -> protocol. The registry (agent/registry.py) builds one object per slot.
SLOT_PROTOCOLS: dict[str, type] = {
    "plan": Planner,
    "check": Checker,
    "goal": GoalManager,
    "recover": Recovery,
    "approach": Approach,
    "memory": Memory,
    "score": Scorer,
    "spatial": SpatialMemory,
    "explore": Explorer,
}
