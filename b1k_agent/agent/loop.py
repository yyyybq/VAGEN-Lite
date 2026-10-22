"""Episode loop: one object per (episode, logical environment).

The official evaluator drives time: it calls the policy once per control step (30 Hz) and waits
for the answer (the simulator does not advance while we think; OmniGibson/omnigibson/eval/
evaluator.py _step_fn). So the loop is step-driven: ``step(obs) -> action``.

Control flow (all branches look only at interface return values, config, and the step budget):

    queued actions (scripted recovery / adjustment / open-loop chunk) -> emit the next one
    decision point? (no subtask yet | check interval reached | a scripted sequence just finished |
                     the subtask's own step budget ran out)
        judge   = check.judge()                     (null checker -> UNKNOWN, i.e. keep going)
        DONE    -> close subtask, plan next          (plan.next_subtask; None -> plan exhausted)
        FAILED  -> recover.propose(failure_type)     (bounded retries -> replan -> full-task fallback)
    otherwise ask the executor for the current instruction and emit its first action

With every slot disabled the planner is the null planner (the full task prompt), so the loop is a
pure passthrough to the executor: that is configuration L0.

Not yet wired: the ``score`` slot (M-score) needs several executor candidates per decision, which no
executor provides yet, so the loop never calls it; enabling it changes nothing until then.
Batched runs (one evaluator port, num_envs > 1): the evaluator keeps sending a finished env's frozen
observation and the loop keeps running for it (its actions are ignored, but its strong-model calls
cost time and money). Agent runs therefore use --num-envs 1, one rollout per evaluator process.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Optional

import numpy as np

from agent import context as ctxvars
from agent.actions import hold_action, validate_actions
from agent.interfaces import (
    FULL_TASK_SKILL,
    ActionChunk,
    EpisodeContext,
    Executor,
    Judgment,
    JudgmentStatus,
    Observation,
    RecoveryKind,
    Subtask,
)
from agent.registry import ModuleSet
from agent.state import EpisodeState, StateView, judgment_unknown


@dataclasses.dataclass(frozen=True)
class LoopConfig:
    exec_horizon: int = 1  # actions requested per executor call (1 = passthrough)
    check_interval_steps: int = 150  # control steps between checks (5 s at 30 Hz)
    max_retries_per_subtask: int = 2
    max_replans: int = 6
    max_rejections_per_round: int = 2
    regression_risk_threshold: float = 0.5
    on_plan_exhausted: str = "full_task"  # "full_task" (keep running the task prompt) | "hold"
    # Budget gate for strong-model calls (2026 rule: over budget = 0). A call is made only if, after it,
    # the remaining budget still covers the remaining steps at the observed per-step cost:
    #   remaining >= provider_min_remaining_s + provider_call_reserve_s + budget_safety * steps_left * s_per_step
    provider_min_remaining_s: float = 60.0
    provider_call_reserve_s: float = 120.0  # >= a module's worst-case call (its timeout_s is a total deadline)
    budget_safety: float = 1.2
    max_provider_s_per_step: float = 300.0  # keep every query well under the 600 s per-query limit
    retrieve_k: int = 3
    check_after_scripted: bool = True

    @staticmethod
    def from_dict(d: Optional[dict]) -> "LoopConfig":
        d = dict(d or {})
        fields = {f.name for f in dataclasses.fields(LoopConfig)}
        unknown = set(d) - fields
        if unknown:
            raise ValueError(f"unknown loop config keys: {sorted(unknown)}")
        return LoopConfig(**d)


@dataclasses.dataclass
class _Queued:
    action: np.ndarray
    source: str  # "exec" | "scripted" | "adjust" | "hold"
    verify: bool = True  # check once the last action of this source has run (AdjustPlan.verify_after)


class EpisodeLoop:
    def __init__(
        self,
        modules: ModuleSet,
        executor: Executor,
        cfg: LoopConfig,
        recorder=None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.m = modules
        self.executor = executor
        self.cfg = cfg
        self.recorder = recorder
        self.clock = clock
        self.state: Optional[EpisodeState] = None
        self._queue: list[_Queued] = []
        self._last_check_step = 0
        self._pending_verify = False
        self._views: dict[str, StateView] = {}
        self._step_provider_s = 0.0

    # ------------------------------------------------------------------ lifecycle

    def reset(self, ctx: EpisodeContext) -> None:
        self.state = EpisodeState(ctx=ctx, clock=self.clock)
        self._views = {slot: StateView(self.state, slot) for slot in self.m.slots()}
        self._queue = []
        self._last_check_step = 0
        self._pending_verify = False
        for slot in self.m.slots():
            self.m.get(slot).reset(ctx)
        self.executor.reset(ctx)
        if self.recorder is not None:
            self.recorder.episode_start(ctx, self.m.describe())

    def close(self) -> None:
        if self.recorder is not None and self.state is not None:
            self.recorder.episode_end(self.state)

    # ------------------------------------------------------------------ main step

    def step(self, obs: Observation) -> np.ndarray:
        st = self.state
        assert st is not None, "reset() must be called before step()"
        st.observe(obs)
        self._step_provider_s = 0.0
        if st.t_budget_start is None and obs.step > 0:
            # the first query raised before its action was emitted; start the clock anyway
            st.t_budget_start = obs.received_at

        if self.m.get("goal").low_risk_mode(self._views["goal"]) and not st.low_risk:
            st.low_risk = True
            st.log("budget", "goal manager switched to low-risk mode", "goal")

        if not self._queue and self._decision_due(obs):
            self._decide(obs)

        if not self._queue:
            self._fill_from_executor(obs)

        q = self._queue.pop(0)
        if st.t_budget_start is None:
            st.t_budget_start = self.clock()
        if q.source in ("scripted", "adjust") and not any(x.source == q.source for x in self._queue):
            self._pending_verify = self.cfg.check_after_scripted and q.verify
        st.last_action = q.action
        st.executed_actions += 1
        if self.recorder is not None:
            self.recorder.step(st, obs, q.action, q.source)
        return q.action

    # ------------------------------------------------------------------ decisions

    def _decision_due(self, obs: Observation) -> bool:
        st = self.state
        if st.current_subtask is None and not st.plan_exhausted:
            return True
        if self._pending_verify:
            return True
        if obs.step - self._last_check_step >= self.cfg.check_interval_steps:
            return True
        sub = st.current_subtask
        if sub is not None and sub.max_steps is not None and st.steps_in_subtask() >= sub.max_steps:
            return True
        return False

    def _provider_allowed(self, slot: str) -> bool:
        if not self.m.uses_provider(slot):
            return True
        st = self.state
        if self._step_provider_s >= self.cfg.max_provider_s_per_step:
            st.log("budget", f"skip {slot}: {self._step_provider_s:.0f}s of model calls in this query already", "loop")
            return False
        now = self.clock()
        remaining = st.remaining_budget_s(now)
        if remaining is None:
            return True
        steps_left = max(0, (st.ctx.max_steps or 0) - max(st.step, 0) - 1)
        s_per_step = max(0.0, st.elapsed_s(now) - st.provider_time_s) / max(1, st.executed_actions)
        need = (self.cfg.provider_min_remaining_s + self.cfg.provider_call_reserve_s
                + self.cfg.budget_safety * steps_left * s_per_step)
        if remaining < need:
            st.log("budget", f"skip {slot}: {remaining:.0f}s left < {need:.0f}s needed", "loop")
            return False
        return True

    def _timed(self, slot: str, fn, *args):
        t0 = self.clock()
        tok = ctxvars.current_slot.set(slot)
        try:
            out = fn(*args)
        finally:
            ctxvars.current_slot.reset(tok)
        if self.m.uses_provider(slot):
            dt = self.clock() - t0
            self.state.add_provider_time(dt)
            self._step_provider_s += dt
        if self.recorder is not None:
            self.recorder.decision(self.state, slot, out, self.clock() - t0)
        return out

    def _decide(self, obs: Observation) -> None:
        st = self.state
        self._pending_verify = False
        if st.current_subtask is None and not st.plan_exhausted:
            self._last_check_step = obs.step
            self._safe_plan(obs)
            return

        self._last_check_step = obs.step
        if self._provider_allowed("check"):
            j: Judgment = self._timed("check", self.m.get("check").judge, obs, self._views["check"])
        else:
            j = judgment_unknown("loop.budget")
        st.apply_judgment(j)

        sub = st.current_subtask
        over_budget = sub is not None and sub.max_steps is not None and st.steps_in_subtask() >= sub.max_steps
        if j.status == JudgmentStatus.DONE:
            st.end_subtask("done")
            self._clear_exec()
            # the evaluator is still asking, so the task is not finished: plan again (bounded)
            if st.plan_exhausted and st.replans < self.cfg.max_replans:
                st.replans += 1
                st.plan_exhausted = False
            if not st.plan_exhausted:
                self._safe_plan(obs)
            elif self.cfg.on_plan_exhausted == "full_task":
                st.start_subtask(self._fallback_subtask("done_but_running"))
        elif j.status == JudgmentStatus.FAILED:
            self._recover(obs, j.failure_type or "unknown")
        elif over_budget:
            self._recover(obs, "timeout")
        if over_budget and st.current_subtask is sub:
            # new budget window for the same subtask (retries still bound the total), otherwise the
            # budget check would trigger a decision -- and a model call -- on every following step
            st.subtask_start_step = max(st.step, 0)

    def _safe_plan(self, obs: Observation) -> None:
        """_plan with a deterministic fallback: a planning error must not re-trigger planning every step."""
        st = self.state
        try:
            self._plan(obs)
        except Exception as e:
            st.log("plan", f"planning error: {e!r}", "loop")
            if st.current_subtask is None:
                st.start_subtask(self._fallback_subtask("plan_error"))

    def _plan(self, obs: Observation) -> None:
        st = self.state
        if self._provider_allowed("goal"):
            order = self._timed("goal", self.m.get("goal").order, st.ctx.goal_spec, self._views["goal"])
            st.subgoal_order = tuple(order or ())
        if self._provider_allowed("memory"):
            exps = self._timed("memory", self.m.get("memory").retrieve, obs, self._views["memory"], self.cfg.retrieve_k)
            st.retrieved = tuple(exps or ())
        if self._provider_allowed("spatial"):
            summary = self._timed("spatial", self.m.get("spatial").update, obs, self._views["spatial"])
            if summary:
                st.spatial_summary = summary

        nxt: Optional[Subtask] = None
        vetoed = False
        for _ in range(self.cfg.max_rejections_per_round + 1):
            vetoed = False
            if not self._provider_allowed("plan"):
                nxt = self._fallback_subtask("budget")
                break
            nxt = self._timed("plan", self.m.get("plan").next_subtask, obs, self._views["plan"])
            if nxt is None and self._provider_allowed("explore"):
                nxt = self._timed("explore", self.m.get("explore").propose, obs, self._views["explore"])
            if nxt is None:
                break
            risk = self._timed("goal", self.m.get("goal").regression_risk, nxt, self._views["goal"])
            if risk <= self.cfg.regression_risk_threshold:
                break
            st.rejected_candidates.append(nxt)
            st.log("plan", f"rejected {nxt.instruction!r}: regression risk {risk:.2f}", "goal")
            nxt = None
            vetoed = True

        if nxt is None and vetoed:
            # every candidate was vetoed: that is not "nothing left to do"
            nxt = self._fallback_subtask("vetoed")
        if nxt is None:
            st.plan_exhausted = True
            st.log("plan", "planner returned no subtask", "plan")
            if self.cfg.on_plan_exhausted == "full_task":
                nxt = self._fallback_subtask("exhausted")
            else:
                return

        if not nxt.subtask_id:
            nxt = dataclasses.replace(nxt, subtask_id=st.new_subtask_id())
        st.start_subtask(nxt)
        self._clear_exec()

        adj = None
        if self._provider_allowed("approach"):
            adj = self._timed("approach", self.m.get("approach").propose, obs, self._views["approach"], nxt)
        if adj is None and self._provider_allowed("memory"):
            adj = self._timed("memory", self.m.get("memory").propose_adjustment, obs, self._views["memory"], nxt)
        if adj is not None:
            if adj.actions is not None:
                for a in validate_actions(adj.actions):
                    self._queue.append(_Queued(a, "adjust", verify=adj.verify_after))
            if adj.instruction:
                st.instruction_override = adj.instruction
                st.override_until_step = st.step + max(1, self.cfg.check_interval_steps)
            st.log("adjust", adj.reason, adj.source)

    def _fallback_subtask(self, why: str) -> Subtask:
        st = self.state
        st.fallbacks += 1
        st.log("fallback", f"full task prompt ({why})", "loop")
        return Subtask(
            subtask_id=st.new_subtask_id(),
            skill=FULL_TASK_SKILL,
            instruction=st.ctx.full_task_prompt,
            source=f"loop.fallback.{why}",
        )

    def _recover(self, obs: Observation, failure_type: str) -> None:
        st = self.state
        sub = st.current_subtask
        if sub is None:
            st.log("recover", f"dropped {failure_type}: no current subtask", "loop")
            return
        st.retries[sub.subtask_id] = st.retries.get(sub.subtask_id, 0) + 1
        if st.retries[sub.subtask_id] > self.cfg.max_retries_per_subtask:
            st.end_subtask("failed")
            self._clear_exec()
            if st.replans < self.cfg.max_replans:
                st.replans += 1
                st.plan_exhausted = False
                self._safe_plan(obs)
            else:
                st.start_subtask(self._fallback_subtask("replans"))
            return

        plan = None
        if self._provider_allowed("recover"):
            plan = self._timed("recover", self.m.get("recover").propose, obs, self._views["recover"], failure_type)
        if plan is None:
            return
        st.recoveries += 1
        st.log("recover", f"{plan.kind.value}:{plan.primitive} {plan.reason}", plan.source)
        if plan.kind == RecoveryKind.SCRIPTED and plan.actions is not None:
            self._clear_exec()
            for a in validate_actions(plan.actions):
                self._queue.append(_Queued(a, "scripted"))
        elif plan.kind == RecoveryKind.INSTRUCTION and plan.instruction:
            self._clear_exec()
            st.instruction_override = plan.instruction
            st.override_until_step = st.step + max(1, plan.duration_steps or self.cfg.check_interval_steps)
        elif plan.kind == RecoveryKind.REPLAN:
            st.end_subtask("abandoned")
            self._clear_exec()
            if st.replans < self.cfg.max_replans:
                st.replans += 1
                st.plan_exhausted = False
                self._safe_plan(obs)
            else:
                st.start_subtask(self._fallback_subtask("replans"))
        elif plan.kind == RecoveryKind.GIVE_UP:
            st.retries[sub.subtask_id] = self.cfg.max_retries_per_subtask

    # ------------------------------------------------------------------ execution

    def _hold(self, obs: Observation) -> None:
        self._queue.append(_Queued(hold_action(obs.proprio, self.state.last_action), "hold"))
        # the executor's buffered chunk was planned before the hold; it must not resume afterwards
        self.executor.interrupt()

    def _clear_exec(self) -> None:
        self._queue = [q for q in self._queue if q.source != "exec"]
        self.executor.interrupt()

    def _fill_from_executor(self, obs: Observation) -> None:
        st = self.state
        if st.current_subtask is None and st.plan_exhausted and self.cfg.on_plan_exhausted == "hold":
            self._hold(obs)
            return
        if st.low_risk and st.current_subtask is not None:
            risk = self.m.get("goal").regression_risk(st.current_subtask, self._views["goal"])
            if risk > self.cfg.regression_risk_threshold:
                self._hold(obs)
                return
        chunk: ActionChunk = self.executor.act(obs, st.current_instruction(), self.cfg.exec_horizon)
        actions = validate_actions(chunk.actions)
        n = max(1, min(int(chunk.open_loop_steps), actions.shape[0]))
        for a in actions[:n]:
            self._queue.append(_Queued(a, "exec"))
        if self.recorder is not None:
            self.recorder.executor_call(st, chunk)
