"""EpisodeState: the system's own estimate of where an episode stands.

One EpisodeState per (episode, logical environment); nothing is shared across environments or
episodes. It holds ESTIMATES only -- simulator ground truth never enters it (ground truth for
offline analysis lives in diagnostics/ records keyed by episode_id and step). Unknown is "unknown".

Only the loop writes the state, and only from interface return values. Modules get a StateView,
which is read-only except for the module's own private scratch dict. Every field below is part of
the documented contract; a module that needs another module's output reads a field here, and a new
field must be added here (with a comment) before any module relies on it.
"""

from __future__ import annotations

import collections
import dataclasses
import time
import types
from typing import Any, Deque, Mapping, Optional

import numpy as np

from agent.interfaces import (
    EpisodeContext,
    Experience,
    Judgment,
    JudgmentStatus,
    Observation,
    Subtask,
)

HISTORY_LEN = 200
PROPRIO_HISTORY_LEN = 600


@dataclasses.dataclass(frozen=True)
class HistoryEvent:
    step: int
    kind: str  # "plan" | "judge" | "recover" | "adjust" | "subtask_start" | "subtask_end" | "fallback" | "budget"
    summary: str
    source: str = ""


@dataclasses.dataclass(frozen=True)
class SubtaskRecord:
    subtask: Subtask
    start_step: int
    end_step: int
    outcome: str  # "done" | "abandoned" | "failed" | "exhausted"
    retries: int


@dataclasses.dataclass(frozen=True)
class Estimate:
    value: str  # "satisfied" / "unsatisfied" / "unknown"; for holding: object name / "" / "unknown"
    confidence: float
    step: int
    source: str


@dataclasses.dataclass
class EpisodeState:
    ctx: EpisodeContext

    # time
    step: int = -1  # index of the latest observation
    t_first_obs: Optional[float] = None  # time.monotonic() of the first observation
    # Budget clock start: the end of the first query. The 2026 evaluator (branch 2026/eval) does not
    # charge the first query (<= 600 s) and starts accounting before the first action executes.
    t_budget_start: Optional[float] = None
    t_latest_obs: Optional[float] = None
    provider_time_s: float = 0.0  # wall-clock spent inside strong-model calls
    provider_calls: int = 0

    # plan
    current_subtask: Optional[Subtask] = None
    subtask_start_step: int = 0
    completed: list = dataclasses.field(default_factory=list)  # list[SubtaskRecord]
    rejected_candidates: list = dataclasses.field(default_factory=list)  # list[Subtask] vetoed this round
    plan_exhausted: bool = False  # the planner returned None: it believes the task is finished
    instruction_override: Optional[str] = None  # set by an INSTRUCTION recovery
    override_until_step: int = -1
    subgoal_order: tuple = ()  # ordered subgoal ids from the goal manager

    # estimates (system's beliefs, NOT ground truth)
    subgoal_estimates: dict = dataclasses.field(default_factory=dict)  # subgoal_id -> Estimate
    holding: dict = dataclasses.field(default_factory=dict)  # "left"/"right" -> Estimate
    last_judgment: Optional[Judgment] = None
    low_risk: bool = False

    # retries / recovery
    retries: dict = dataclasses.field(default_factory=dict)  # subtask_id -> int
    replans: int = 0
    recoveries: int = 0
    fallbacks: int = 0

    # memory
    retrieved: tuple = ()  # tuple[Experience]
    spatial_summary: str = ""

    # history
    history: Deque[HistoryEvent] = dataclasses.field(default_factory=lambda: collections.deque(maxlen=HISTORY_LEN))
    proprio_history: Deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=PROPRIO_HISTORY_LEN))
    executed_actions: int = 0
    last_action: Optional[np.ndarray] = None

    # private per-module scratch: slot name -> dict (only that module touches its own dict)
    scratch: dict = dataclasses.field(default_factory=dict)

    _subtask_counter: int = 0
    # the loop's clock (time.monotonic in production, a fake clock in tests) so that every
    # time-dependent estimate in one episode uses the same time source
    clock: Any = dataclasses.field(default=time.monotonic, repr=False, compare=False)

    # ---- writes (loop only) -----------------------------------------------------------------

    def observe(self, obs: Observation) -> None:
        self.step = obs.step
        if self.t_first_obs is None:
            self.t_first_obs = obs.received_at
        self.t_latest_obs = obs.received_at
        self.proprio_history.append((obs.step, np.array(obs.proprio, copy=True)))
        if self.instruction_override is not None and obs.step >= self.override_until_step:
            self.instruction_override = None

    def log(self, kind: str, summary: str, source: str = "") -> None:
        self.history.append(HistoryEvent(step=self.step, kind=kind, summary=summary, source=source))

    def new_subtask_id(self) -> str:
        self._subtask_counter += 1
        return f"st{self._subtask_counter}"

    def start_subtask(self, subtask: Subtask) -> None:
        self.current_subtask = subtask
        self.subtask_start_step = max(self.step, 0)
        self.retries.setdefault(subtask.subtask_id, 0)
        self.rejected_candidates.clear()
        self.instruction_override = None
        self.log("subtask_start", f"{subtask.skill}: {subtask.instruction}", subtask.source)

    def end_subtask(self, outcome: str) -> None:
        if self.current_subtask is None:
            return
        st = self.current_subtask
        self.completed.append(
            SubtaskRecord(
                subtask=st,
                start_step=self.subtask_start_step,
                end_step=max(self.step, 0),
                outcome=outcome,
                retries=self.retries.get(st.subtask_id, 0),
            )
        )
        self.log("subtask_end", f"{st.subtask_id} {outcome}", st.source)
        self.current_subtask = None

    def apply_judgment(self, j: Judgment) -> None:
        self.last_judgment = j
        for sg, value in j.subgoal_updates.items():
            self.subgoal_estimates[sg] = Estimate(value=value, confidence=j.confidence, step=self.step, source=j.source)
        for arm, value in j.holding_updates.items():
            self.holding[arm] = Estimate(value=value, confidence=j.confidence, step=self.step, source=j.source)
        self.log("judge", f"{j.status.value} ({j.confidence:.2f}) {j.failure_type or ''} {j.evidence[:160]}", j.source)

    def add_provider_time(self, seconds: float, calls: int = 1) -> None:
        self.provider_time_s += seconds
        self.provider_calls += calls

    # ---- derived ----------------------------------------------------------------------------

    def elapsed_s(self, now: Optional[float] = None) -> float:
        """Accounted time since the budget clock started (0 during the free first query)."""
        if self.t_budget_start is None:
            return 0.0
        return (now if now is not None else self.clock()) - self.t_budget_start

    def remaining_budget_s(self, now: Optional[float] = None) -> Optional[float]:
        if self.ctx.wallclock_budget_s is None:
            return None
        return self.ctx.wallclock_budget_s - self.elapsed_s(now)

    def steps_in_subtask(self) -> int:
        return max(self.step, 0) - self.subtask_start_step

    def current_instruction(self) -> str:
        if self.instruction_override is not None:
            return self.instruction_override
        if self.current_subtask is not None:
            return self.current_subtask.instruction
        return self.ctx.task_instruction

    def summary_text(self, max_events: int = 12) -> str:
        """Compact history for prompts: recent events plus finished subtasks."""
        lines = []
        for rec in self.completed[-8:]:
            lines.append(f"[{rec.start_step}-{rec.end_step}] {rec.subtask.skill} '{rec.subtask.instruction}' -> {rec.outcome}")
        for ev in list(self.history)[-max_events:]:
            lines.append(f"t={ev.step} {ev.kind}: {ev.summary}")
        return "\n".join(lines)


class StateView:
    """Read-only view of an EpisodeState handed to one module.

    Attribute reads are forwarded, containers come back as immutable copies, and any attribute
    write raises. ``scratch()`` returns the calling module's own private dict.
    """

    __slots__ = ("_state", "_slot")

    def __init__(self, state: EpisodeState, slot: str):
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_slot", slot)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in _WRITE_METHODS or name == "scratch":
            raise AttributeError(f"StateView does not expose '{name}'")
        value = getattr(object.__getattribute__(self, "_state"), name)
        if isinstance(value, dict):
            return types.MappingProxyType(dict(value))
        if isinstance(value, (list, collections.deque)):
            return tuple(value)
        if isinstance(value, np.ndarray):
            v = value.copy()
            v.setflags(write=False)
            return v
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("modules may not write EpisodeState; return values to the loop instead")

    def module_scratch(self) -> dict:
        state = object.__getattribute__(self, "_state")
        slot = object.__getattribute__(self, "_slot")
        return state.scratch.setdefault(slot, {})

    # read-only helpers forwarded explicitly (methods are not wrapped by __getattr__ copies)
    def elapsed_s(self, now: Optional[float] = None) -> float:
        return object.__getattribute__(self, "_state").elapsed_s(now)

    def remaining_budget_s(self, now: Optional[float] = None) -> Optional[float]:
        return object.__getattribute__(self, "_state").remaining_budget_s(now)

    def steps_in_subtask(self) -> int:
        return object.__getattribute__(self, "_state").steps_in_subtask()

    def current_instruction(self) -> str:
        return object.__getattribute__(self, "_state").current_instruction()

    def summary_text(self, max_events: int = 12) -> str:
        return object.__getattribute__(self, "_state").summary_text(max_events)


_WRITE_METHODS = frozenset(
    {
        "observe",
        "log",
        "new_subtask_id",
        "start_subtask",
        "end_subtask",
        "apply_judgment",
        "add_provider_time",
    }
)


def judgment_unknown(source: str) -> Judgment:
    return Judgment(status=JudgmentStatus.UNKNOWN, confidence=0.0, evidence="", source=source)


def experiences(xs) -> tuple[Experience, ...]:
    return tuple(xs or ())


def frozen_mapping(d: Mapping) -> Mapping:
    return types.MappingProxyType(dict(d))
