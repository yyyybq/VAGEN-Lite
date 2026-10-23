"""Round 12 freeze, dense-reward design variant V8 (the user's 10th supplement of 2026-10-23): the group comparison
normalises the success part (return - progress) and the progress part apart and adds them, the progress part weighted
(vagen/split_reward_adv.py, PPO config algorithm.progress_adv_weight, default off)."""

import numpy as np
import pytest
import torch

from vagen.split_reward_adv import progress_of_turn, progress_tensor, split_grpo_advantages, weight


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std(ddof=1) + 1e-6)


def test_split_advantages_by_hand():
    total = torch.tensor([5.2, 0.3, 0.1, 0.4, 1.0, 1.0])
    progress = torch.tensor([0.2, 0.3, 0.1, 0.4, 0.5, 0.5])
    uids = ["a", "a", "a", "a", "b", "b"]
    mask = torch.ones(6, 3)
    mask[1, 2] = 0
    adv = split_grpo_advantages(total, progress, uids, mask, 0.5)
    want_a = z([5.0, 0.0, 0.0, 0.0]) + 0.5 * z([0.2, 0.3, 0.1, 0.4])
    assert adv[:4, 0].tolist() == pytest.approx(want_a.tolist(), abs=1e-5)
    assert adv[1, 2] == 0.0  # padding stays 0
    assert adv[4:].abs().max() == 0.0  # a group whose parts do not differ has no signal
    # the failed episodes (success part 0) are now ordered by their progress alone
    assert adv[3, 0] > adv[1, 0] > adv[2, 0]
    # plain GRPO on the total would rank the failed ones by the same totals but scale them by the success's spread
    plain = z([5.2, 0.3, 0.1, 0.4])
    assert abs(adv[3, 0] - adv[2, 0]) > abs(plain[3] - plain[2])


def test_switch_and_parts():
    from omegaconf import OmegaConf

    assert weight(None) is None
    assert weight(OmegaConf.create({"adv_estimator": "grpo"})) is None
    assert weight(OmegaConf.create({"progress_adv_weight": 0.5})) == 0.5
    assert weight({"progress_adv_weight": 0}) == 0.0
    info = {"reward_trace": {"components": {"potential": {"applied": 0.3}, "milestone": {"applied": 0.5},
                                            "success": {"applied": 5.0}, "step": {"applied": -0.01}}}}
    assert progress_of_turn(info) == pytest.approx(0.8)
    assert progress_of_turn({}) == 0.0 and progress_of_turn(None) == 0.0
    arr = np.empty(3, dtype=object)
    arr[:] = [{"progress": 0.25}, {"progress": -0.1}, None]
    assert progress_tensor(list(arr)).tolist() == pytest.approx([0.25, -0.1, 0.0])


def test_compute_advantage_applies_the_switch():
    from omegaconf import OmegaConf
    from verl import DataProto

    from vagen.ray_trainer import AdvantageEstimator, compute_advantage

    rewards = torch.zeros(4, 5)
    rewards[:, 4] = torch.tensor([5.2, 0.3, 0.1, 0.4])
    batch = {"token_level_rewards": rewards, "response_mask": torch.ones(4, 5), "responses": torch.zeros(4, 5, dtype=torch.long)}
    parts = np.empty(4, dtype=object)
    parts[:] = [{"progress": 0.2}, {"progress": 0.3}, {"progress": 0.1}, {"progress": 0.4}]
    non = {"uid": np.array(["g"] * 4, dtype=object), "reward_parts": parts}
    off = compute_advantage(DataProto.from_dict(tensors=dict(batch), non_tensors=dict(non)), AdvantageEstimator.GRPO,
                            config=OmegaConf.create({}))
    on = compute_advantage(DataProto.from_dict(tensors=dict(batch), non_tensors=dict(non)), AdvantageEstimator.GRPO,
                           config=OmegaConf.create({"progress_adv_weight": 0.5}))
    want = z([5.0, 0.0, 0.0, 0.0]) + 0.5 * z([0.2, 0.3, 0.1, 0.4])
    assert on.batch["advantages"][:, 0].tolist() == pytest.approx(want.tolist(), abs=1e-5)
    assert off.batch["advantages"][:, 0].tolist() == pytest.approx(z([5.2, 0.3, 0.1, 0.4]).tolist(), abs=1e-5)
    assert "split_adv_metrics" in on.meta_info and "split_adv_metrics" not in off.meta_info


def test_agent_data_starts_with_no_progress():
    from vagen.agent_loop.gym_agent_loop import AgentData

    d = AgentData(messages=[], image_data=[], metrics={}, request_id="r", env=None, response_limit=100, env_name="e")
    assert d.progress_return == 0.0
