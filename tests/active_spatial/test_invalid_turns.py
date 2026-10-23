"""Round 12 freeze (the user's 3.2 of 2026-10-23): unreadable turns get no positive update (vagen/invalid_turns.py,
PPO config algorithm.invalid_turn_nonpositive_adv, default off)."""

import numpy as np
import pytest
import torch

from vagen.invalid_turns import clamp_invalid_turn_advantages, enabled, spans_mask


def test_spans_mask_lists_dicts_and_clipping():
    m = spans_mask([[[1, 3]], {"spans": [[0, 1], [4, 9]]}, None, {"spans": []}], 6)
    assert m.tolist() == [[False, True, True, False, False, False],
                          [True, False, False, False, True, True],
                          [False] * 6, [False] * 6]


def test_successful_episode_with_an_unreadable_turn():
    # three episodes of one group, 8 response positions each; positions 6-7 of episode 0 are padding
    adv = torch.tensor([[1.2] * 8, [-0.8] * 8, [0.5] * 8])
    response_mask = torch.ones(3, 8)
    response_mask[0, 6:] = 0
    spans = [{"spans": [[2, 4]]},   # a successful episode (advantage > 0) with one unreadable turn at positions 2-3
             {"spans": [[0, 2]]},   # a failed episode with an unreadable turn: still pushed down
             {"spans": []}]         # no unreadable turn
    out, metrics = clamp_invalid_turn_advantages(adv, response_mask, spans)
    assert out[0].tolist() == pytest.approx([1.2, 1.2, 0.0, 0.0, 1.2, 1.2, 1.2, 1.2])
    assert out[1].tolist() == pytest.approx([-0.8] * 8)
    assert out[2].tolist() == pytest.approx([0.5] * 8)
    assert metrics["actor/invalid_turn_token_share"] == pytest.approx(4 / 22)
    assert metrics["actor/invalid_turn_clamped_share"] == pytest.approx(0.5)
    assert metrics["actor/invalid_turn_episodes"] == 2.0
    unchanged, _ = clamp_invalid_turn_advantages(adv, response_mask, [None, None, None])
    assert torch.equal(unchanged, adv)


def test_switch_reading_and_batch_packing():
    from omegaconf import OmegaConf

    assert not enabled(None)
    assert not enabled(OmegaConf.create({"adv_estimator": "grpo"}))
    assert enabled(OmegaConf.create({"invalid_turn_nonpositive_adv": True}))
    assert enabled({"invalid_turn_nonpositive_adv": True})
    # the trainer packs extra fields into a 1-D object array: one dict per sample stays one element
    arr = np.empty(2, dtype=object)
    arr[:] = [{"spans": [[0, 3]]}, {"spans": [[1, 2]]}]
    assert arr.shape == (2,) and arr[1]["spans"] == [[1, 2]]


def test_compute_advantage_applies_the_switch():
    from omegaconf import OmegaConf
    from verl import DataProto

    from vagen.ray_trainer import AdvantageEstimator, compute_advantage

    rewards = torch.zeros(4, 5)
    rewards[0, 4] = 5.0  # one success in a group of four: positive advantage for episode 0
    batch = {"token_level_rewards": rewards, "response_mask": torch.ones(4, 5), "responses": torch.zeros(4, 5, dtype=torch.long)}
    non = {"uid": np.array(["g"] * 4, dtype=object),
           "invalid_turn_spans": np.array([{"spans": [[1, 3]]}, {"spans": []}, {"spans": [[0, 1]]}, {"spans": []}], dtype=object)}
    off = compute_advantage(DataProto.from_dict(tensors=dict(batch), non_tensors=dict(non)), AdvantageEstimator.GRPO,
                            config=OmegaConf.create({}))
    on = compute_advantage(DataProto.from_dict(tensors=dict(batch), non_tensors=dict(non)), AdvantageEstimator.GRPO,
                           config=OmegaConf.create({"invalid_turn_nonpositive_adv": True}))
    a_off, a_on = off.batch["advantages"], on.batch["advantages"]
    assert a_off[0, 0] > 0
    assert a_on[0, 1:3].tolist() == [0.0, 0.0] and a_on[0, 0] == a_off[0, 0]  # the success's unreadable turn
    assert torch.equal(a_on[1:], a_off[1:])  # failed episodes keep their (negative) advantages
    assert "invalid_turn_metrics" in on.meta_info and "invalid_turn_metrics" not in off.meta_info


def test_agent_loop_records_the_unreadable_turns_spans():
    """The episode record (AgentData) as the agent loop drives it: generated turns take response positions (mask 1),
    the env's observation tokens follow (mask 0); a turn the env judged unreadable keeps its span, which is where its
    tokens sit in the response."""
    from vagen.agent_loop.gym_agent_loop import AgentData, turn_invalid

    d = AgentData(messages=[], image_data=[], metrics={}, request_id="r", env=None, response_limit=100, env_name="e")
    d.add_generated_turn(5)                      # turn 1: readable
    d.end_turn(turn_invalid({"metrics": {"turn_metrics": {"action_is_valid": True}}}))
    d.response_mask += [0] * 7                   # the observation after turn 1
    d.add_generated_turn(4)                      # turn 2: unreadable
    d.end_turn(turn_invalid({"metrics": {"turn_metrics": {"action_is_valid": False}}}))
    d.response_mask += [0] * 3
    d.add_generated_turn(2)                      # turn 3: readable
    d.end_turn(False)
    assert d.invalid_spans == [[12, 16]]
    assert d.response_mask[12:16] == [1, 1, 1, 1] and d.response_mask[11] == 0 and d.response_mask[16] == 0
    assert spans_mask([{"spans": d.invalid_spans}], len(d.response_mask))[0].nonzero().flatten().tolist() == [12, 13, 14, 15]
