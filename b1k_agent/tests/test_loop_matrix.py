"""L0 passthrough, the on/off switch matrix (red line 7) and truth isolation (red line 8).

MOCK episodes: synthetic observations (tests/synth.py), a deterministic fake executor and scripted
providers. They prove construction, control flow and isolation, not task performance.
"""

import copy
import glob
import json

import numpy as np
import pytest

from agent.interfaces import SLOT_PROTOCOLS
from helpers import TASK_PROMPT, make_cfg, make_policy, run_episode
from synth import synth_obs

N_STEPS = 700

# Real implementations to switch on per slot (slots without one use "null" through the enabled path).
ON = {
    "plan": {"enabled": True, "impl": "llm", "provider": "astra", "prompt_version": "v1"},
    "check": {"enabled": True, "impl": "llm", "provider": "astra", "prompt_version": "v1"},
    "goal": {"enabled": True, "impl": "bddl"},
    "recover": {"enabled": True, "impl": "scripted"},
}


def cfg_with(enabled: dict) -> dict:
    cfg = make_cfg("L0_exec_only")
    for slot in SLOT_PROTOCOLS:
        cfg["modules"][slot] = copy.deepcopy(enabled.get(slot, {"enabled": False}))
    return cfg


def test_l0_is_a_pure_passthrough():
    policy, execs, tmp = make_policy(make_cfg("L0_exec_only"))
    actions = run_episode(policy, lambda t: synth_obs(t), 300)
    ex = execs[0]
    assert len(ex.calls) == 300
    assert {instr for _, instr in ex.calls} == {TASK_PROMPT}
    assert policy.errors == 0
    summary = json.load(open(glob.glob(f"{tmp}/test/episodes/*/summary.json")[0]))
    assert summary["provider_calls"] == 0 and summary["fallbacks"] == 0 and summary["recoveries"] == 0
    # identical to calling the executor directly
    from helpers import FakeExecutor
    from agent.fields import build_observation

    direct = FakeExecutor()
    for t in (0, 150, 299):
        o = build_observation({k: v[0:1] for k, v in synth_obs(t).items()}, step=t, received_at=0.0)
        np.testing.assert_array_equal(actions[t][0], direct.act(o, TASK_PROMPT, 1).actions[0])


@pytest.mark.parametrize("slot", sorted(SLOT_PROTOCOLS))
def test_each_slot_on_alone_runs(slot):
    on = {slot: ON.get(slot, {"enabled": True, "impl": "null"})}
    policy, execs, _ = make_policy(cfg_with(on))
    actions = run_episode(policy, lambda t: synth_obs(t), N_STEPS)
    assert policy.errors == 0
    assert all(a.shape == (1, 23) and np.all(np.isfinite(a)) for a in actions)


@pytest.mark.parametrize("slot", sorted(ON))
def test_leave_one_out_from_full_stack_runs(slot):
    on = {k: v for k, v in ON.items() if k != slot}
    policy, execs, _ = make_policy(cfg_with(on))
    run_episode(policy, lambda t: synth_obs(t), N_STEPS)
    assert policy.errors == 0


def test_check_rule_impl_runs():
    policy, _, _ = make_policy(cfg_with({"check": {"enabled": True, "impl": "rule"}, "recover": ON["recover"]}))
    run_episode(policy, lambda t: synth_obs(t), N_STEPS)
    assert policy.errors == 0


def test_full_stack_uses_modules():
    policy, execs, tmp = make_policy(cfg_with(ON))
    run_episode(policy, lambda t: synth_obs(t), N_STEPS)
    summary = json.load(open(glob.glob(f"{tmp}/test/episodes/*/summary.json")[0]))
    assert summary["provider_calls"] > 0
    assert summary["recoveries"] >= 1  # the scripted checker reports grasp_failed once
    assert any(s.get("outcome") == "done" for s in summary["subtasks"])


PRIVILEGED = {
    "robot_r1::robot_pos": lambda rng: rng.normal(size=(1, 3)).astype(np.float32),
    "task::object_poses": lambda rng: rng.normal(size=(1, 7, 7)).astype(np.float32),
    "robot_r1::robot_r1:zed_link:Camera:0::seg_instance": lambda rng: rng.integers(0, 50, size=(1, 224, 224)).astype(np.int32),
    "privileged::goal_predicates": lambda rng: rng.integers(0, 2, size=(1, 5)).astype(np.int8),
}


def decisions(tmp):
    out = []
    for path in sorted(glob.glob(f"{tmp}/test/episodes/*/events.jsonl")):
        for line in open(path):
            rec = json.loads(line)
            if rec["kind"] in ("decision", "provider_call"):
                rec.pop("seconds", None)
                rec.pop("latency_s", None)
                rec.pop("call_id", None)
                rec.pop("images", None)
                out.append(json.dumps(rec, sort_keys=True))
    return out


@pytest.mark.parametrize("variant", ["added", "values_changed"])
def test_truth_isolation(variant):
    """Adding (or changing) simulator-truth fields in the observation must not change any decision."""
    base_policy, _, tmp_a = make_policy(cfg_with(ON))
    a = run_episode(base_policy, lambda t: synth_obs(t), N_STEPS)

    def with_truth(t, seed):
        rng = np.random.default_rng(seed * 7919 + t)
        return synth_obs(t, extra={k: f(rng) for k, f in PRIVILEGED.items()})

    p1, _, tmp_b = make_policy(cfg_with(ON))
    b = run_episode(p1, lambda t: with_truth(t, 1), N_STEPS)
    if variant == "values_changed":
        p2, _, tmp_c = make_policy(cfg_with(ON))
        c = run_episode(p2, lambda t: with_truth(t, 2), N_STEPS)
        assert all(np.array_equal(x, y) for x, y in zip(b, c))
        assert decisions(tmp_b) == decisions(tmp_c)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert decisions(tmp_a) == decisions(tmp_b)


def test_batched_envs_have_separate_state():
    policy, execs, _ = make_policy(cfg_with(ON))
    policy.reset()
    for t in range(200):
        assert policy.act(synth_obs(t, num_envs=2)).shape == (2, 23)
    s0, s1 = policy._loops[0].state, policy._loops[1].state
    assert s0 is not s1 and s0.ctx.episode_id != s1.ctx.episode_id
    assert policy._loops[0].m.get("plan") is not policy._loops[1].m.get("plan")
    assert execs[0] is not execs[1]
