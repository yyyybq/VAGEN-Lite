"""Unit tests: BDDL reader, config loading/hashing, goal manager, scripted recovery, rule checker,
LLM planner/checker reply validation (scripted provider, no model)."""

import copy
import os

import numpy as np
import pytest

from agent.bddl import load_goal_spec, parse_problem
from agent.config import config_hash, load_config
from agent.fields import ACTION_SLICES, PROPRIO_SLICES, build_observation
from agent.interfaces import FULL_TASK_SKILL, EpisodeContext, JudgmentStatus, RecoveryKind, SkillSpec, Subtask
from agent.loop import LoopConfig
from agent.registry import ModuleDeps
from agent.state import Estimate, EpisodeState, StateView
from conftest import FIXTURES, ROOT
from helpers import CHECK_REPLIES, PLAN_REPLY, TASK_PROMPT
from providers import build_providers
from synth import synth_obs


def obs(t=0):
    return build_observation({k: v[0:1] for k, v in synth_obs(t).items()}, step=t, received_at=0.0)


def ctx(task="turning_on_radio", skills=None):
    return EpisodeContext(
        episode_id="e", env_index=0, task_name=task, task_id=0, task_instruction=TASK_PROMPT,
        goal_spec=load_goal_spec(FIXTURES, task), max_steps=1000, wallclock_budget_s=1000.0,
        skills=skills or (SkillSpec(name=FULL_TASK_SKILL, instruction_template="{task_prompt}", description="all"),),
    )


# ---------------------------------------------------------------- BDDL

def test_bddl_simple_goal():
    g = load_goal_spec(FIXTURES, "turning_on_radio")
    assert [(s.predicate, s.args) for s in g.subgoals] == [("toggled_on", ("radio_receiver.n.01_1",))]
    assert g.objects["radio_receiver.n.01_1"] == "radio_receiver.n.01"
    assert "(not (toggled_on radio_receiver.n.01_1))" in g.init_literals


def test_bddl_forall_is_grounded_over_declared_instances():
    g = load_goal_spec(FIXTURES, "picking_up_trash")
    assert [s.args for s in g.subgoals] == [(f"can__of__soda.n.01_{i}", "ashcan.n.01_1") for i in (1, 2, 3)]


def test_bddl_or_and_not():
    text = """(define (problem p) (:domain d) (:objects a.n.01_1 - a.n.01 b.n.01_1 - b.n.01)
      (:init (open b.n.01_1)) (:goal (and (not (open ?b.n.01_1)) (or (ontop ?a.n.01_1 ?b.n.01_1) (inside ?a.n.01_1 ?b.n.01_1)))))"""
    g = parse_problem(text)
    assert g.subgoals[0].negated and g.subgoals[0].predicate == "open"
    assert [s.option for s in g.subgoals[1:]] == [1, 2]


def test_bddl_option_ids_unique_across_or_nodes():
    text = """(define (problem p) (:domain d) (:objects a.n.01_1 - a.n.01 b.n.01_1 - b.n.01)
      (:init) (:goal (and (or (open ?a.n.01_1) (open ?b.n.01_1))
                          (or (ontop ?a.n.01_1 ?b.n.01_1) (or (inside ?a.n.01_1 ?b.n.01_1) (under ?a.n.01_1 ?b.n.01_1))))))"""
    g = parse_problem(text)
    assert [(s.predicate, s.option) for s in g.subgoals] == [("open", 1), ("open", 2), ("ontop", 3), ("inside", 5), ("under", 6)]
    assert "option 5 is a disjunct inside option 4" in g.parse_notes


def test_bddl_forall_with_disjunctive_body_stays_quantified():
    text = """(define (problem p) (:domain d) (:objects c.n.01_1 c.n.01_2 - c.n.01 t.n.01_1 - t.n.01 s.n.01_1 - s.n.01)
      (:init) (:goal (and (forall (?c.n.01 - c.n.01) (or (inside ?c.n.01 ?t.n.01_1) (ontop ?c.n.01 ?s.n.01_1)))
                          (forall (?c.n.01 - c.n.01) (not (open ?c.n.01))))))"""
    g = parse_problem(text)
    assert g.subgoals[0].quantified and g.subgoals[0].predicate == "forall" and g.subgoals[0].option == 0
    assert [(s.predicate, s.args, s.negated) for s in g.subgoals[1:]] == [
        ("open", ("c.n.01_1",), True), ("open", ("c.n.01_2",), True)]  # a plain forall body is still grounded


# ---------------------------------------------------------------- config

def test_config_inheritance_and_hash():
    l0 = load_config(os.path.join(ROOT, "configs", "L0_exec_only.yaml"))
    l4 = load_config(os.path.join(ROOT, "configs", "L4_recover.yaml"))
    assert not l0["modules"]["plan"]["enabled"]
    assert l4["modules"]["plan"]["enabled"] and l4["modules"]["recover"]["impl"] == "scripted"
    assert config_hash(l0) == config_hash(copy.deepcopy(l0))
    o = load_config(os.path.join(ROOT, "configs", "L0_exec_only.yaml"), ["loop.check_interval_steps=90"])
    assert o["loop"]["check_interval_steps"] == 90 and config_hash(o) != config_hash(l0)
    with pytest.raises(ValueError):
        LoopConfig.from_dict({"nonexistent": 1})


# ---------------------------------------------------------------- goal manager

def deps(params=None, provider=None, version=None, slot="goal"):
    from agent.prompts import PromptLoader

    return ModuleDeps(slot=slot, impl="x", params=params or {}, provider=provider,
                      prompts=PromptLoader(slot) if version else None, prompt_version=version)


def test_goal_order_and_regression_risk():
    from modules.goal.bddl import BDDLGoalManager

    text = """(define (problem p) (:domain d) (:objects c.n.01_1 - c.n.01 x.n.01_1 - x.n.01 r.n.01_1 - r.n.01)
      (:init) (:goal (and (inside ?x.n.01_1 ?c.n.01_1) (toggled_on ?r.n.01_1) (open ?c.n.01_1))))"""
    gs = parse_problem(text, "t")
    c = EpisodeContext(episode_id="e", env_index=0, task_name="t", task_id=0, task_instruction="t", goal_spec=gs,
                       skills=(), max_steps=100, wallclock_budget_s=100.0)
    gm = BDDLGoalManager(deps())
    gm.reset(c)
    st = EpisodeState(ctx=c)
    order = gm.order(gs, StateView(st, "goal"))
    ids = {s.predicate: s.subgoal_id for s in gs.subgoals}
    assert order.index(ids["open"]) < order.index(ids["inside"])
    st.subgoal_estimates[ids["toggled_on"]] = Estimate("satisfied", 0.9, 0, "test")
    order2 = gm.order(gs, StateView(st, "goal"))
    assert order2[-1] == ids["toggled_on"]
    risky = Subtask(subtask_id="s", skill="pick", instruction="pick r", args={"object": "r.n.01_1"})
    assert gm.regression_risk(risky, StateView(st, "goal")) == 1.0
    full = Subtask(subtask_id="s", skill=FULL_TASK_SKILL, instruction="t")
    assert gm.regression_risk(full, StateView(st, "goal")) == 0.0


# ---------------------------------------------------------------- recovery

def test_scripted_recovery_primitives():
    from modules.recover.scripted import ScriptedRecovery

    rec = ScriptedRecovery(deps({"retract_steps": 10, "back_up_steps": 5}, slot="recover"))
    rec.reset(ctx())
    st = EpisodeState(ctx=ctx())
    view = StateView(st, "recover")
    o = obs(5)
    p = rec.propose(o, view, "grasp_failed")
    assert p.kind == RecoveryKind.SCRIPTED and p.primitive == "retract_arms" and p.actions.shape == (10, 23)
    np.testing.assert_allclose(p.actions[-1, ACTION_SLICES["left_arm"]], 0.0, atol=1e-6)
    np.testing.assert_allclose(p.actions[:, ACTION_SLICES["torso"]], np.repeat(o.proprio[PROPRIO_SLICES["trunk_qpos"]][None], 10, 0))
    assert np.all(p.actions[:, 0:3] == 0)
    p2 = rec.propose(o, view, "grasp_failed")
    assert p2.primitive == "back_up" and np.all(p2.actions[:, 0] < 0)
    rec.propose(o, view, "grasp_failed")
    assert rec.propose(o, view, "grasp_failed") is None  # bounded


def test_instruction_recovery_requires_verified_skill():
    from modules.recover.scripted import ScriptedRecovery

    unverified = SkillSpec(name="retry_grasp", instruction_template="Grasp it again.", description="", available=True, verified=False)
    verified = SkillSpec(name="retry_grasp", instruction_template="Grasp it again.", description="", available=True, verified=True)
    params = {"instruction_recovery": True, "instruction_skills": {"grasp_failed": "retry_grasp"}}
    for skill, kind in ((unverified, RecoveryKind.SCRIPTED), (verified, RecoveryKind.INSTRUCTION)):
        rec = ScriptedRecovery(deps(params, slot="recover"))
        c = ctx(skills=(SkillSpec(name=FULL_TASK_SKILL, instruction_template="", description=""), skill))
        rec.reset(c)
        assert rec.propose(obs(), StateView(EpisodeState(ctx=c), "recover"), "grasp_failed").kind == kind


# ---------------------------------------------------------------- rule checker

def test_rule_checker_detects_stall():
    from modules.check.rule import RuleChecker

    chk = RuleChecker(deps({"stall_window_steps": 30}, slot="check"))
    c = ctx()
    chk.reset(c)
    st = EpisodeState(ctx=c)
    st.start_subtask(Subtask(subtask_id="s1", skill=FULL_TASK_SKILL, instruction=TASK_PROMPT))
    frozen = obs(0)
    for t in range(40):
        st.observe(build_observation({k: v[0:1] for k, v in synth_obs(0).items()}, step=t, received_at=float(t)))
    j = chk.judge(frozen, StateView(st, "check"))
    assert j.status == JudgmentStatus.FAILED and j.failure_type == "no_progress"


# ---------------------------------------------------------------- LLM planner / checker validation

def providers_with(plan_replies=None, check_replies=None):
    return build_providers({"p": {"kind": "scripted", "replies": {
        "plan.next_subtask": plan_replies or [PLAN_REPLY], "check.judge": check_replies or CHECK_REPLIES}}})["p"]


def test_planner_only_uses_available_skills_and_renders_templates():
    from modules.plan.llm import LLMPlanner

    skills = (
        SkillSpec(name=FULL_TASK_SKILL, instruction_template="{task_prompt}", description="all"),
        SkillSpec(name="go_to", instruction_template="Go to the {object}.", description="nav", inputs=("object",), available=True),
        SkillSpec(name="hidden", instruction_template="x", description="", available=False),
    )
    replies = [
        {**PLAN_REPLY, "skill": "go_to", "args": [{"name": "object", "value": "radio"}]},
        {**PLAN_REPLY, "skill": "hidden"},
        {**PLAN_REPLY, "skill": "go_to", "args": []},
        {**PLAN_REPLY, "task_complete": True},
    ]
    pl = LLMPlanner(deps(provider=providers_with(plan_replies=replies), version="v1", slot="plan"))
    c = ctx(skills=skills)
    pl.reset(c)
    view = StateView(EpisodeState(ctx=c), "plan")
    s1 = pl.next_subtask(obs(), view)
    assert s1.skill == "go_to" and s1.instruction == "Go to the radio." and s1.target_subgoals == ("g0",)
    s2 = pl.next_subtask(obs(), view)
    assert s2.skill == FULL_TASK_SKILL and s2.source == "plan.llm.fallback" and s2.instruction == TASK_PROMPT
    s3 = pl.next_subtask(obs(), view)
    assert s3.source == "plan.llm.fallback"  # missing args
    # "task complete" is never true while the evaluator still asks: keep going with the full task
    s4 = pl.next_subtask(obs(), view)
    assert s4.skill == FULL_TASK_SKILL and s4.source == "plan.llm.fallback"
    schema = pl.provider.inner.requests[0].json_schema
    assert schema["properties"]["skill"]["enum"] == [FULL_TASK_SKILL, "go_to"]


def test_checker_downgrades_low_confidence_and_handles_errors():
    from modules.check.llm import LLMChecker

    replies = [{**CHECK_REPLIES[2], "confidence": 0.3}, {**CHECK_REPLIES[1]}]
    chk = LLMChecker(deps({"min_confidence": 0.6}, provider=providers_with(check_replies=replies), version="v1", slot="check"))
    c = ctx()
    chk.reset(c)
    view = StateView(EpisodeState(ctx=c), "check")
    assert chk.judge(obs(), view).status == JudgmentStatus.UNKNOWN
    j = chk.judge(obs(), view)
    assert j.status == JudgmentStatus.FAILED and j.failure_type == "grasp_failed"
    broken = build_providers({"p": {"kind": "scripted", "replies": {}}})["p"]
    chk2 = LLMChecker(deps({}, provider=broken, version="v1", slot="check"))
    chk2.reset(c)
    assert chk2.judge(obs(), view).status == JudgmentStatus.UNKNOWN


# ---------------------------------------------------------------- loop budget regression

def test_subtask_step_budget_does_not_trigger_a_check_every_step():
    """Regression: an over-budget subtask whose checker keeps saying in_progress (and no recovery
    module) must get a fresh budget window, not a model call on every following step."""
    from agent.interfaces import SkillSpec
    from helpers import make_cfg, make_policy, run_episode

    cfg = make_cfg("L0_exec_only")
    cfg["modules"]["plan"] = {"enabled": True, "impl": "llm", "provider": "astra", "prompt_version": "v1"}
    cfg["modules"]["check"] = {"enabled": True, "impl": "llm", "provider": "astra", "prompt_version": "v1"}
    cfg["providers"]["astra"]["replies"] = {
        "plan.next_subtask": [{**PLAN_REPLY, "skill": "go_to", "args": [{"name": "object", "value": "radio"}], "max_steps": 60}],
        "check.judge": [CHECK_REPLIES[0]],  # always in_progress
    }
    cfg["loop"]["check_interval_steps"] = 1000
    skills = (SkillSpec(name=FULL_TASK_SKILL, instruction_template="{task_prompt}", description="all"),
              SkillSpec(name="go_to", instruction_template="Go to the {object}.", description="nav",
                        inputs=("object",), available=True))
    policy, _, _ = make_policy(cfg)
    policy.skills_for_task = lambda task: skills
    run_episode(policy, lambda t: synth_obs(t), 600)
    requests = policy._loops[0].m.get("check").provider.inner.requests
    judge_calls = sum(1 for r in requests if r.purpose == "check.judge")
    assert judge_calls <= 600 // 60 + 2, judge_calls


def test_call_ids_are_unique_across_providers():
    from agent.recorder import RunRecorder
    import tempfile
    from providers import ProviderRequest

    rec = RunRecorder("t", tempfile.mkdtemp(), save_frames=False)
    pool = build_providers({"a": {"kind": "scripted", "replies": {"x": [{"k": 1}]}},
                            "b": {"kind": "scripted", "replies": {"x": [{"k": 2}]}}}, recorder=rec)
    seen = []
    rec.provider_call = lambda eid, r: seen.append(r["call_id"])
    for name in ("a", "b", "a", "b"):
        pool[name].complete(ProviderRequest(system="s", user_text="u", purpose="x"))
    assert len(set(seen)) == 4


def test_steps_log_keeps_exact_float32_actions():
    import glob
    import json as _json

    from helpers import make_cfg, make_policy, run_episode

    policy, _, tmp = make_policy(make_cfg("L0_exec_only"))
    acts = run_episode(policy, lambda t: synth_obs(t), 20)
    rows = [_json.loads(l) for l in open(glob.glob(f"{tmp}/test/episodes/*/steps.jsonl")[0])]
    assert [r["step"] for r in rows] == list(range(20))
    np.testing.assert_array_equal(np.asarray(rows[7]["a"], np.float32), acts[7][0])
