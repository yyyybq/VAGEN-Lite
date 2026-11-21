"""Phase 2 (docs/q38/DESIGN_ASYNC.md 2.3, the row of vagen/q38/gym_agent_loop_v1.py): the per-turn records of the q38
agent loop, on test_agent_loop.py's fake server and env (no weights, no render service).

- turn_credit: extra_fields.turn_records = the reward line's episode_record of one turn_record per env turn (RW 26afbc6f
  gym_agent_loop.py:299-302, 376-379, 513-517), spans = the generated turns' spans;
- turn_gae: extra_fields.turn_rewards = turn_gae.turn_record(spans, env rewards) (GL:362-365), accepted by the
  estimator's own span check (turn_anchors: the spans cover exactly the mask-1 positions);
- with either: extra_fields.turn_meta (per turn: weight versions, stop, generation and env seconds);
- GRPO: none of the three keys (a GRPO episode's extra_fields are as before).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from vagen.q38.gym_agent_loop_v1 import GymAgentLoopV1
from vagen.q38.tests.test_agent_loop import (
    FakeEnv,
    FakeServer,
    assert_json_style,
    default_model,
    make_config,
    run_episode,
)

TURN_KEYS = ("turn_records", "turn_rewards", "turn_meta")


@pytest.fixture(autouse=True)
def _fake_env_registered(monkeypatch):
    monkeypatch.setitem(GymAgentLoopV1._env_classes, "FakeEnv", FakeEnv)
    FakeEnv.closed = []
    yield


def cfg_for(estimator):
    cfg = make_config()
    if estimator is not None:
        OmegaConf.update(cfg, "algorithm.adv_estimator", estimator, force_add=True)
    return cfg


def _mask_positions(out):
    return [i for i, m in enumerate(out.response_mask) if m]


def _span_positions(spans):
    return [p for s, e in spans for p in range(s, e)]


@pytest.mark.parametrize("estimator", [None, "grpo"])
def test_grpo_episode_has_no_turn_fields(estimator):
    model = default_model()
    _, out = run_episode(model, FakeServer(model.tokenizer), cfg=cfg_for(estimator), done_at=3)
    assert not set(TURN_KEYS) & set(out.extra_fields)
    assert set(out.extra_fields) == {"min_global_steps", "max_global_steps", "reward_extra_info", "vagen_stats"}


def test_turn_credit_records():
    from vagen.turn_credit import selected

    model = default_model()
    server = FakeServer(model.tokenizer, versions=[(3, 3), (3, 4), (4, 4)], ends=["im_end", "endoftext", "im_end"])
    loop, out = run_episode(model, server, cfg=cfg_for("turn_credit"), done_at=3)
    assert selected(loop.config) and loop.record_turn_records and not loop.record_turn_rewards
    ef = out.extra_fields
    assert "turn_records" in ef and "turn_meta" in ef and "turn_rewards" not in ef
    assert_json_style(ef)
    rec, stats = ef["turn_records"], ef["vagen_stats"]
    assert (rec["phi0"], rec["d0"]) == (0.25, None)  # initial_potential_score; the fake env has no table distance
    turns = rec["turns"]
    assert len(turns) == stats["env_turns"] == 3
    assert [t["span"] for t in turns] == stats["turn_spans"]
    assert _span_positions(stats["turn_spans"]) == _mask_positions(out)
    # FakeEnv: -0.1 a step, success (5.0) at done_at; current_potential_score 0.25 + 0.1 t; env_step 2 t
    assert [t["reward"] for t in turns] == [-0.1, -0.1, 5.0]
    assert [t["phi_after"] for t in turns] == pytest.approx([0.35, 0.45, 0.55])
    assert [t["success"] for t in turns] == [False, False, True]
    assert [t["valid"] for t in turns] == [True, True, True]
    assert [t["steps"] for t in turns] == [2, 4, 6]
    assert all(t["d_after"] is None for t in turns)
    meta = ef["turn_meta"]
    assert meta["ver"] == [[3, 3], [3, 4], [4, 4]] and meta["stop"] == ["eos", "endoftext", "eos"]
    assert len(meta["gen_s"]) == len(meta["render_s"]) == 3 and all(x >= 0 for x in meta["gen_s"] + meta["render_s"])


def test_turn_credit_unreadable_and_env_error():
    model = default_model()
    _, out = run_episode(model, FakeServer(model.tokenizer), cfg=cfg_for("turn_credit"), done_at=2, invalid=True)
    assert [t["valid"] for t in out.extra_fields["turn_records"]["turns"]] == [False, False]
    # an env.step exception: the turn is recorded as the reward line's loop records it (reward 0, done, readable)
    _, out = run_episode(model, FakeServer(model.tokenizer), cfg=cfg_for("turn_credit"), raise_at=2)
    turns = out.extra_fields["turn_records"]["turns"]
    assert len(turns) == 2 and turns[1]["reward"] == 0.0 and turns[1]["phi_after"] == 0.0
    assert turns[1]["valid"] is True and turns[1]["success"] is False and "steps" not in turns[1]


@pytest.mark.parametrize("done_at", [1, 3, None])
def test_turn_gae_records(done_at):
    from vagen import turn_gae as TG

    model = default_model()
    kw = {"done_at": done_at} if done_at else {"max_turns": 12}
    loop, out = run_episode(model, FakeServer(model.tokenizer), cfg=cfg_for("turn_gae"), **kw)
    assert loop.record_turn_rewards and not loop.record_turn_records
    ef = out.extra_fields
    assert "turn_rewards" in ef and "turn_meta" in ef and "turn_records" not in ef
    assert_json_style(ef)
    rec, stats = ef["turn_rewards"], ef["vagen_stats"]
    assert rec["spans"] == stats["turn_spans"] and len(rec["rewards"]) == stats["env_turns"]
    assert sum(rec["rewards"]) == pytest.approx(out.reward_score)
    mask = torch.tensor(out.response_mask)
    anchors, folded = TG.turn_anchors(rec, mask)  # the estimator's own check of the spans against the mask
    assert folded == 0 and [a for a, _, _ in anchors] == [s for s, _ in rec["spans"]]
    assert len(ef["turn_meta"]["ver"]) == len(rec["spans"])
