"""The main line's per-turn PPO (the user's decision 1A of 2026-11-09): GAE over the turns of a concat episode
(vagen/turn_gae.py, algorithm.adv_estimator "turn_gae", package p1tgae)."""

import numpy as np
import pytest
import torch

from vagen import turn_gae as TG


def _batch():
    """Two concat episodes, 12 response positions each (1 = generated, 0 = observation tokens).

    Episode 0: three turns at 0-2, 5-7, 10-11, rewards -0.01 / 0.5 / 5.0 (sum 5.49 on the last token).
    Episode 1: turns at 0-3 and 6-9, a third turn cut off by the response length (span 12-15), rewards
    0.1 / 0.2 / 0.3 (sum 0.6 on its last generated token): the cut-off turn's reward goes to turn 2."""
    mask = torch.zeros(2, 12)
    mask[0, 0:3] = 1; mask[0, 5:8] = 1; mask[0, 10:12] = 1
    mask[1, 0:4] = 1; mask[1, 6:10] = 1
    rewards = torch.zeros(2, 12)
    rewards[0, 11] = 5.49
    rewards[1, 9] = 0.6
    values = torch.full((2, 12), 9.0)  # only the anchors' values may matter
    values[0, 0], values[0, 5], values[0, 10] = 1.0, 2.0, 3.0
    values[1, 0], values[1, 6] = 0.5, 0.25
    values = values * mask
    records = [{"spans": [[0, 3], [5, 8], [10, 12]], "rewards": [-0.01, 0.5, 5.0]},
               {"spans": [[0, 4], [6, 10], [12, 15]], "rewards": [0.1, 0.2, 0.3]}]
    return rewards, values, mask, records


def test_turn_recursion():
    # gamma = lam = 1, values 0: each turn's advantage is the reward still to come (reward-to-go)
    adv, ret = TG.turn_gae([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 1.0, 1.0)
    assert adv == pytest.approx([6.0, 5.0, 3.0]) and ret == pytest.approx([6.0, 5.0, 3.0])
    # one turn: A = r - V, return = r
    adv, ret = TG.turn_gae([5.0], [2.0], 1.0, 0.95)
    assert adv == pytest.approx([3.0]) and ret == pytest.approx([5.0])
    # gamma 0.9, lam 0.5 by hand: delta_2 = 1 - 0.5 = 0.5; delta_1 = 0 + 0.9 * 0.5 - 1 = -0.55;
    # A_1 = -0.55 + 0.45 * 0.5 = -0.325
    adv, ret = TG.turn_gae([0.0, 1.0], [1.0, 0.5], 0.9, 0.5)
    assert adv == pytest.approx([-0.325, 0.5]) and ret == pytest.approx([0.675, 1.0])


def test_anchors_clip_and_fold():
    mask = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])
    # a first turn with an empty span gives its reward to the next kept turn; a turn past the end to the one before
    anchors, folded = TG.turn_anchors({"spans": [[0, 0], [0, 2], [4, 6], [8, 11]], "rewards": [0.5, 1.0, 2.0, 4.0]}, mask)
    assert anchors == [(0, 2, 1.5), (4, 6, 6.0)] and folded == 2
    # a turn cut in the middle (by the response length) keeps its anchor and its clipped end
    cut = torch.tensor([1, 1, 0, 0, 1, 1, 1, 1])
    anchors, folded = TG.turn_anchors({"spans": [[0, 2], [4, 9]], "rewards": [1.0, 2.0]}, cut)
    assert anchors == [(0, 2, 1.0), (4, 8, 2.0)] and folded == 0
    with pytest.raises(KeyError):
        TG.turn_anchors(None, mask)
    with pytest.raises(ValueError):  # rewards but no position at all
        TG.turn_anchors({"spans": [[8, 9]], "rewards": [1.0]}, torch.zeros(8))
    # the spans must cover exactly the generated positions: another sample's record, or a mask changed afterwards
    with pytest.raises(ValueError, match="cover exactly"):
        TG.turn_anchors({"spans": [[0, 2], [4, 7]], "rewards": [1.0, 2.0]}, mask)  # covers observation token 6
    with pytest.raises(ValueError, match="cover exactly"):
        TG.turn_anchors({"spans": [[0, 2]], "rewards": [1.0]}, mask)  # leaves generated tokens 4-5 out


def test_compute_places_advantages_and_returns():
    rewards, values, mask, records = _batch()
    adv, ret, metrics = TG.compute(rewards, values, mask, records, gamma=1.0, lam=0.5)
    # episode 0 by hand: delta_3 = 5 - 3 = 2; delta_2 = 0.5 + 3 - 2 = 1.5, A_2 = 1.5 + 0.5 * 2 = 2.5;
    # delta_1 = -0.01 + 2 - 1 = 0.99, A_1 = 0.99 + 0.5 * 2.5 = 2.24
    expect0 = [2.24] * 3 + [0, 0] + [2.5] * 3 + [0, 0] + [2.0] * 2
    assert adv[0].tolist() == pytest.approx(expect0)
    # episode 1: turn 2 carries 0.2 + 0.3; delta_2 = 0.5 - 0.25 = 0.25; delta_1 = 0.1 + 0.25 - 0.5 = -0.15,
    # A_1 = -0.15 + 0.5 * 0.25 = -0.025
    expect1 = [-0.025] * 4 + [0, 0] + [0.25] * 4 + [0, 0]
    assert adv[1].tolist() == pytest.approx(expect1)
    # returns only at each turn's first token, the sentinel everywhere else
    sentinel = TG.IGNORE_VALUE
    want = torch.full((2, 12), sentinel)
    want[0, 0], want[0, 5], want[0, 10] = 3.24, 4.5, 5.0
    want[1, 0], want[1, 6] = 0.475, 0.5
    assert torch.allclose(ret, want, atol=1e-5)
    assert metrics["critic/turn_gae/turns_per_sample"] == pytest.approx(2.5)
    assert metrics["critic/turn_gae/folded_turns"] == 1.0
    assert metrics["critic/turn_gae/first_turn_value_mean"] == pytest.approx(0.75)
    assert metrics["critic/turn_gae/reward_sum_max_diff"] < 1e-5


def test_episode_total_must_match_the_turn_rewards():
    rewards, values, mask, records = _batch()
    rewards[0, 11] = 5.0  # not the sum of the turns' rewards
    with pytest.raises(ValueError, match="sum"):
        TG.compute(rewards, values, mask, records, gamma=1.0, lam=0.95)
    with pytest.raises(ValueError, match="records"):
        TG.compute(rewards, values, mask, records[:1], gamma=1.0, lam=0.95)


def _proto(records=True):
    from verl import DataProto

    rewards, values, mask, recs = _batch()
    tensors = {"token_level_rewards": rewards, "values": values, "response_mask": mask,
               "responses": torch.zeros(2, 12, dtype=torch.long)}
    non = {"uid": np.array(["a", "b"], dtype=object)}
    if records:
        arr = np.empty(2, dtype=object)
        arr[:] = recs
        non[TG.RECORD_KEY] = arr
    return DataProto.from_dict(tensors=tensors, non_tensors=non)


def test_registered_estimator_through_compute_advantage():
    from omegaconf import OmegaConf

    from vagen.ray_trainer import compute_advantage, compute_value_mask

    raw, raw_ret, _ = TG.compute(*_batch()[:3], _batch()[3], gamma=1.0, lam=0.5)
    out = compute_advantage(_proto(), "turn_gae", gamma=1.0, lam=0.5, config=OmegaConf.create({"use_kl_in_reward": False}))
    # the estimator hands back the raw advantages (the trainer whitens them after the filter) and the returns
    assert torch.allclose(out.batch["advantages"].float(), raw, atol=1e-6)
    assert torch.allclose(out.batch["returns"].float(), raw_ret, atol=1e-6)
    assert "turn_gae_metrics" in out.meta_info
    # the value network is trained at the turns' first tokens only
    vm = compute_value_mask(out)
    assert vm.nonzero().tolist() == [[0, 0], [0, 5], [0, 10], [1, 0], [1, 6]]
    # a per-token KL reward would be dropped, missing records cannot be guessed: both refused
    with pytest.raises(ValueError, match="KL"):
        compute_advantage(_proto(), "turn_gae", gamma=1.0, lam=0.5, config=OmegaConf.create({"use_kl_in_reward": True}))
    with pytest.raises(KeyError):
        compute_advantage(_proto(records=False), "turn_gae", gamma=1.0, lam=0.5, config=OmegaConf.create({}))


def test_settings_the_estimator_refuses():
    from omegaconf import OmegaConf

    TG.check_config(OmegaConf.create({"use_kl_in_reward": False, "rollout_correction": {"rollout_is": None,
                                      "rollout_rs": None, "rollout_token_veto_threshold": None}}))
    TG.check_config(None)
    for bad, word in (({"use_kl_in_reward": True}, "KL"), ({"rollout_correction": {"rollout_rs": "token"}}, "rollout_rs"),
                      ({"rollout_correction": {"rollout_token_veto_threshold": 1e-4}}, "veto"),
                      ({"invalid_turn_nonpositive_adv": True}, "invalid_turn")):
        with pytest.raises(ValueError, match=word):
            TG.check_config(OmegaConf.create(bad))


def test_whiten_matches_verl_on_the_kept_rows():
    import verl.utils.torch_functional as verl_F

    torch.manual_seed(0)
    adv = torch.randn(4, 10)
    mask = (torch.rand(4, 10) > 0.3).float()
    ours = TG.whiten(adv, mask)
    theirs = verl_F.masked_whiten(adv, mask) * mask
    assert torch.allclose(ours, theirs, atol=1e-5)
    assert ours[mask == 0].abs().max().item() == 0.0
    # the reviewers' case: whitened over the kept groups only, the dropped all-fail group no longer shifts them
    raw = torch.tensor([[0.5], [-0.5], [-0.8], [-0.8]])
    ones = torch.ones(4, 1)
    before = TG.whiten(raw, ones)[:2, 0]                 # over all four (as before the filter)
    after = TG.whiten(raw[:2], ones[:2])[:, 0]           # over the kept group
    assert float(after.sum()) == pytest.approx(0.0, abs=1e-6) and float(before.sum()) > 1.0
    assert float(after[1]) < 0 < float(after[0])


def test_keep_filter_skips_its_grpo_scale_under_turn_gae():
    from verl import DataProto

    from vagen.custom_filter.filter import keep_by_return_std_filter

    # two all-fail tasks of 2 episodes, returns spread 0.5 (under the floor 1.0): filled, GRPO would scale by ~0.5
    rewards = torch.zeros(4, 3)
    rewards[:, 2] = torch.tensor([0.0, 0.7071068, 1.0, 1.7071068])
    tensors = {"token_level_rewards": rewards, "advantages": torch.ones(4, 3), "response_mask": torch.ones(4, 3)}
    non = {"uid": np.array(["a", "a", "b", "b"], dtype=object), "traj_success": np.zeros(4)}
    kw = dict(target_groups=2, std_threshold=0.1, std_floor=1.0)
    grpo, m_g = keep_by_return_std_filter(DataProto.from_dict(tensors=dict(tensors), non_tensors=dict(non)), {}, **kw)
    proto = DataProto.from_dict(tensors=dict(tensors), non_tensors=dict(non))
    proto.meta_info["adv_estimator"] = "turn_gae"
    tg, m_t = keep_by_return_std_filter(proto, {}, **kw)
    assert m_g["filter/keep/filled"] == 2 and m_t["filter/keep/filled"] == 2  # the same selection
    assert float(grpo.batch["advantages"].max()) == pytest.approx(0.5, abs=1e-4)
    assert torch.equal(tg.batch["advantages"], torch.ones(4, 3))
    assert m_t["filter/keep/adv_scale_skipped"] == 1.0 and "filter/keep/adv_scale_skipped" not in m_g


def test_agent_loop_records_every_turn():
    from omegaconf import OmegaConf

    from vagen.agent_loop.gym_agent_loop import AgentData

    d = AgentData(messages=[], image_data=[], metrics={}, request_id="r", env=None, response_limit=100, env_name="e")
    d.add_generated_turn(3); d.env_rewards.append(-0.01); d.response_mask += [0] * 2
    d.add_generated_turn(3); d.env_rewards.append(0.5); d.response_mask += [0] * 2
    d.add_generated_turn(2); d.env_rewards.append(5.0)
    assert d.turn_spans == [[0, 3], [5, 8], [10, 12]]
    rec = TG.turn_record(d.turn_spans, d.env_rewards)
    assert rec == {"spans": [[0, 3], [5, 8], [10, 12]], "rewards": [-0.01, 0.5, 5.0]}
    with pytest.raises(ValueError):
        TG.turn_record(d.turn_spans, d.env_rewards[:2])
    # the switch: only this estimator records
    assert TG.enabled(OmegaConf.create({"adv_estimator": "turn_gae"})) and TG.enabled({"adv_estimator": "turn_gae"})
    assert not TG.enabled(OmegaConf.create({"adv_estimator": "gae"})) and not TG.enabled(None)


def test_runner_accepts_the_concat_mode_only():
    from omegaconf import OmegaConf

    from vagen.interiorgs_v10_ppo import history_mode

    cfg = lambda est, concat: OmegaConf.create({"algorithm": {"adv_estimator": est}, "trainer": {"concat_multi_turn": concat}})  # noqa: E731
    assert history_mode(cfg("turn_gae", True)) == "concat"
    assert history_mode(cfg("gae", True)) == "concat" and history_mode(cfg("grpo", True)) == "concat"
    with pytest.raises(ValueError):
        history_mode(cfg("turn_gae", False))
