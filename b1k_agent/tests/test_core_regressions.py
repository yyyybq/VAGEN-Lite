"""Regression tests for confirmed review findings in the core loop, modules and providers.
All MOCK (fake executor, scripted providers, fake clocks); no simulator, no model."""

import json

import numpy as np
import pytest

from agent.bddl import parse_problem
from agent.interfaces import FULL_TASK_SKILL, EpisodeContext, Judgment, JudgmentStatus, Subtask
from agent.state import Estimate, EpisodeState, StateView
from helpers import CHECK_REPLIES, PLAN_REPLY, FakeExecutor, make_cfg, make_policy
from synth import synth_obs
from test_units import ctx, deps, obs, providers_with


def _drive(policy, n):
    policy.reset()
    for t in range(n):
        policy.act(synth_obs(t))
    return policy._loops[0]


# ------------------------------------------------------------------ executor prompt per task

class _SnakeCaseExecutor(FakeExecutor):
    def default_prompt(self, task_name):
        return "turning on radio"  # the prompt the single-task checkpoint was trained with


class _UnservedExecutor(FakeExecutor):
    def default_prompt(self, task_name):
        return None  # e.g. a single-task server asked about another task


@pytest.mark.parametrize("cls,expected", [(_SnakeCaseExecutor, "turning on radio"), (_UnservedExecutor, None)])
def test_full_task_prompt_comes_from_the_executor(cls, expected):
    cfg = make_cfg("L0_exec_only")
    policy, _, _ = make_policy(cfg)
    execs = {}
    policy.make_executor = lambda i: execs.setdefault(i, cls(i))
    loop = _drive(policy, 3)
    c = loop.state.ctx
    assert c.executor_prompt == expected
    assert c.full_task_prompt == (expected or c.task_instruction)
    assert {instr for _, instr in execs[0].calls} == {c.full_task_prompt}


# ------------------------------------------------------------------ plan exhausted -> bounded re-plan

def test_done_after_exhausted_plan_replans_then_falls_back():
    cfg = make_cfg("L0_exec_only")
    cfg["modules"]["plan"] = {"enabled": True, "impl": "llm", "provider": "astra", "prompt_version": "v1"}
    cfg["modules"]["check"] = {"enabled": True, "impl": "llm", "provider": "astra", "prompt_version": "v1"}
    cfg["loop"]["check_interval_steps"] = 5
    cfg["loop"]["max_replans"] = 3
    policy, _, _ = make_policy(cfg)
    policy.reset()
    loop = policy._loop(0)
    calls = []

    def plan_none(o, view):
        calls.append(o.step)
        return None

    loop.m.get("plan").next_subtask = plan_none
    loop.m.get("check").judge = lambda o, view: Judgment(status=JudgmentStatus.DONE, confidence=1.0, source="test")
    for t in range(200):
        policy.act(synth_obs(t))
    st = loop.state
    assert len(calls) == 1 + 3, calls  # first plan + max_replans re-plans, then no more model calls
    assert st.replans == 3
    assert st.current_subtask is not None and st.current_subtask.source == "loop.fallback.done_but_running"


# ------------------------------------------------------------------ budget gate

@pytest.mark.parametrize("max_steps,gated", [(150, True), (3000, False)])
def test_budget_gate_skips_model_calls_when_time_is_short(max_steps, gated):
    cfg = make_cfg("L0_exec_only")
    cfg["modules"]["check"] = {"enabled": True, "impl": "llm", "provider": "astra", "prompt_version": "v1"}
    cfg["loop"]["check_interval_steps"] = 10
    policy, _, _ = make_policy(cfg, max_steps=max_steps)
    loop = _drive(policy, 60)
    judge_calls = sum(1 for r in loop.m.get("check").provider.inner.requests if r.purpose == "check.judge")
    # max_steps x 1 s = 150 s < provider_min_remaining_s + provider_call_reserve_s (60 + 120)
    assert (judge_calls == 0) == gated, judge_calls
    assert any("skip check" in e.summary for e in loop.state.history) == gated


# ------------------------------------------------------------------ goal manager: shared receptacle

def test_regression_risk_ignores_a_shared_receptacle():
    from modules.goal.bddl import BDDLGoalManager

    text = """(define (problem p) (:domain d) (:objects a.n.01_1 b.n.01_1 - a.n.01 bin.n.01_1 - bin.n.01)
      (:init) (:goal (and (inside ?a.n.01_1 ?bin.n.01_1) (inside ?b.n.01_1 ?bin.n.01_1))))"""
    gs = parse_problem(text, "t")
    c = EpisodeContext(episode_id="e", env_index=0, task_name="t", task_id=0, task_instruction="t", goal_spec=gs,
                       skills=(), max_steps=100, wallclock_budget_s=100.0)
    gm = BDDLGoalManager(deps())
    gm.reset(c)
    st = EpisodeState(ctx=c)
    st.subgoal_estimates["g0"] = Estimate("satisfied", 0.9, 0, "test")  # a.n.01_1 already in the bin
    put_b = Subtask(subtask_id="s", skill="put_inside", instruction="i", target_subgoals=("g1",),
                    args={"object": "b.n.01_1", "container": "bin.n.01_1"})
    move_a = Subtask(subtask_id="s", skill="pick", instruction="i", args={"object": "a.n.01_1"})
    assert gm.regression_risk(put_b, StateView(st, "goal")) == 0.0
    assert gm.regression_risk(move_a, StateView(st, "goal")) == 1.0


# ------------------------------------------------------------------ checker post-processing

def _checker(reply):
    from modules.check.llm import LLMChecker

    chk = LLMChecker(deps({"min_confidence": 0.6}, provider=providers_with(check_replies=[reply]), version="v1", slot="check"))
    c = ctx()
    chk.reset(c)
    return chk, c


def test_checker_unknown_does_not_overwrite_estimates():
    reply = {**CHECK_REPLIES[0], "subgoals": [{"id": "g0", "estimate": "unknown"}],
             "holding_left": "unknown", "holding_right": "cup"}
    chk, c = _checker(reply)
    j = chk.judge(obs(), StateView(EpisodeState(ctx=c), "check"))
    assert "g0" not in j.subgoal_updates
    assert j.holding_updates == {"right": "cup"}


def test_checker_never_reports_the_whole_task_done():
    chk, c = _checker(CHECK_REPLIES[2])  # done, confidence 0.9
    st = EpisodeState(ctx=c)
    st.start_subtask(Subtask(subtask_id="s0", skill=FULL_TASK_SKILL, instruction=c.full_task_prompt))
    j = chk.judge(obs(), StateView(st, "check"))
    assert j.status == JudgmentStatus.IN_PROGRESS
    assert j.subgoal_updates == {"g0": "satisfied"}  # estimates are kept


# ------------------------------------------------------------------ provider deadline and redaction

class _FakeTime:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


class _Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text


def _openai(monkeypatch, responses, **cfg):
    import providers.openai_compat as oc

    clock, timeouts = _FakeTime(), []

    def post(url, headers=None, content=None, timeout=None):
        timeouts.append(timeout)
        clock.t += 0.2
        return responses.pop(0) if len(responses) > 1 else responses[0]

    monkeypatch.setattr(oc, "time", clock)
    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setenv("T_URL", "https://x.example/v1")
    monkeypatch.setenv("T_KEY", "sk-abcdefghijklmnop")
    p = oc.OpenAICompatProvider("t", {"kind": "openai_compat", "base_url_env": "T_URL", "api_key_env": "T_KEY",
                                      "model": "m", **cfg})
    return p, clock, timeouts


def test_openai_timeout_is_a_total_deadline(monkeypatch):
    from providers.base import ProviderRequest

    p, clock, timeouts = _openai(monkeypatch, [_Resp(503)], max_retries=5)
    with pytest.raises(Exception):
        p.complete(ProviderRequest(system="s", user_text="u", timeout_s=4.0))
    assert clock.t <= 4.0, clock.t
    assert timeouts[0] == pytest.approx(4.0) and all(b < a for a, b in zip(timeouts, timeouts[1:]))


def test_openai_error_bodies_are_redacted(monkeypatch):
    from providers.base import ProviderRequest

    p, _, _ = _openai(monkeypatch, [_Resp(401, "bad key sk-abcdefghijklmnop; Authorization: Bearer sk-abcdefghijklmnop")])
    with pytest.raises(Exception) as e:
        p.complete(ProviderRequest(system="s", user_text="u", timeout_s=4.0))
    assert "sk-abcdefghijklmnop" not in str(e.value) and "***" in str(e.value)
