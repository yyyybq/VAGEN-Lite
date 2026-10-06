from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vagen.custom_advantage.no_concat_gae import compute_gae_no_concat_advantage_return_firsttok
from vagen.utils.active_spatial_ppo_snapshot import (
    SCHEMA_VERSION,
    SnapshotValidationError,
    _config_to_plain,
    load_snapshot,
    maybe_write_pre_update_snapshot,
    write_snapshot,
)
from vagen.utils.active_spatial_ppo_replay import (
    gradient_diagnosis,
    no_concat_gae,
    recompute_branches,
    reward_from_trace,
)


def _trace(*, potential, near=0.0, visibility=0.0, success=0.0, phi_prev=0.2, phi=0.4, terminated=False, truncated=False):
    components = {}
    for name, applied in {
        "potential": potential, "near": near, "visibility": visibility, "success": success,
        "format": 0.01, "invalid": 0.0, "collision": 0.0, "step": -0.01,
    }.items():
        components[name] = {"raw": applied, "scale": 1.0, "scaled": applied, "applied": applied, "enabled": True}
    total = sum(item["applied"] for item in components.values())
    return {
        "schema_version": "active_spatial_reward_trace_v1",
        "identity": {"task_id": "task", "episode_id": "episode", "turn_id": 1, "task_family": "delta_control"},
        "score": {"phi_prev": phi_prev, "phi": phi},
        "components": components,
        "outcome": {"terminated": terminated, "truncated": truncated, "termination_reason": "timeout" if truncated else "success" if terminated else "continuing"},
        "accounting": {"actual_total_reward": total, "reconstructed_total_reward": total, "reconstruction_error": 0.0},
    }


def _fixture_data():
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.long)
    traces = [
        _trace(potential=0.18, phi_prev=0.2, phi=0.4),
        _trace(potential=0.275, near=0.5, phi_prev=0.2, phi=0.5, truncated=True),
        _trace(potential=0.085, visibility=0.3, success=5.0, phi_prev=0.3, phi=0.4, terminated=True),
    ]
    scores = torch.zeros_like(response_mask, dtype=torch.float32)
    for row, trace in enumerate(traces):
        scores[row, int(torch.nonzero(response_mask[row]).view(-1)[-1])] = trace["accounting"]["actual_total_reward"]
    values = torch.tensor([[0.1, 0.2, 0.0], [0.3, 0.4, 0.5], [0.6, 0.0, 0.0]])
    raw = no_concat_gae(scores, values, response_mask, [0, 0, 1], [0, 0, 0], [1, 2, 1], gamma=0.95, lam=0.95)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 5, 6], [1, 2, 6, 0, 0]]),
        "responses": torch.tensor([[3, 4, 0], [3, 5, 6], [6, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]),
        "response_mask": response_mask,
        "position_ids": torch.arange(5).repeat(3, 1),
        "old_log_probs": torch.full((3, 3), -0.7),
        "ref_log_prob": torch.full((3, 3), -0.8),
        "values": values,
        "token_level_scores": scores,
        "token_level_rewards": scores.clone(),
        "advantages": raw["advantages"],
        "returns": raw["returns"],
        "value_mask": (raw["returns"] != -100).long(),
    }
    non_tensor = {
        "group_idx": [0, 0, 1], "traj_idx": [0, 0, 0], "turn_idx": [1, 2, 1], "__last_turn__": [False, True, True],
        "reward_trace": traces, "task_id": ["a", "a", "b"], "task_type": ["delta_control", "delta_control", "absolute_positioning"],
        "terminated": [False, False, True], "truncated": [False, True, False], "termination_reason": ["continuing", "max_llm_turns", "success"],
        "action_text": ["<action>move_forward</action>", "<action>move_left</action>", "<action>done</action>"],
        "parsed_primitive_actions": [["move_forward", "turn_left"], ["move_left"], ["done"]],
        "bootstrap_context": [
            {"capture_enabled": True, "next_state_available": True, "next_observation_text": "next", "next_images": []},
            {"capture_enabled": True, "next_state_available": True, "next_observation_text": "limit", "next_images": []},
            {"capture_enabled": True, "next_state_available": False, "next_observation_text": "", "next_images": []},
        ],
        "multi_modal_inputs": [{"pixel_values": torch.ones(1, 2)}, None, {"pixel_values": torch.zeros(1, 2)}],
    }
    config = {
        "algorithm": {"gamma": 0.95, "lam": 0.95},
        "actor_rollout_ref": {"rollout": {"temperature": 0.8, "logprob_temperature": 1.0}, "actor": {"clip_ratio": 0.2, "loss_agg_mode": "token-mean", "use_kl_loss": True, "kl_loss_type": "low_var_kl", "kl_loss_coef": 0.3}},
        "critic": {"cliprange_value": 0.5, "loss_agg_mode": "token-mean"},
    }
    return SimpleNamespace(batch=batch, non_tensor_batch=non_tensor), config


def _identity():
    return {"global_step": 7, "actor": {"path": "actor", "content_sha256": "a" * 64}, "critic": {"path": "critic", "content_sha256": "b" * 64}, "reference": {"path": "ref", "content_sha256": "c" * 64}}


def test_snapshot_roundtrip_schema_hash_and_required_fields(tmp_path):
    data, config = _fixture_data()
    path = write_snapshot(tmp_path, data, config, _identity())
    payload, manifest, loaded_config = load_snapshot(path)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["payload_sha256"]
    assert loaded_config == config
    assert payload["turn_token_map"][1]["valid_response_token_positions"] == [0, 1, 2]
    assert torch.equal(payload["optimization_mask"], payload["tensor_batch"]["response_mask"])


def test_snapshot_accepts_numpy_scalar_metadata_but_rejects_unknown_objects(tmp_path):
    """Real Ray metadata uses np.int64; the boundary stays strict otherwise."""
    data, config = _fixture_data()
    data.non_tensor_batch["group_idx"][0] = np.int64(7)
    data.non_tensor_batch["bootstrap_context"][0]["primitive_count"] = np.int32(2)
    path = write_snapshot(tmp_path, data, config, _identity())
    payload, _, _ = load_snapshot(path)
    assert payload["metadata"]["group_idx"][0] == 7
    assert payload["metadata"]["bootstrap_context"][0]["primitive_count"] == 2

    data, config = _fixture_data()
    data.non_tensor_batch["group_idx"][0] = object()
    with pytest.raises(SnapshotValidationError, match="unsupported snapshot payload type: object"):
        write_snapshot(tmp_path, data, config, _identity())


def test_resolved_omegaconf_config_becomes_plain_json_data():
    omegaconf = pytest.importorskip("omegaconf")
    config = omegaconf.OmegaConf.create({"algorithm": {"gamma": "${base}"}, "base": 0.95})
    plain = _config_to_plain(config)
    assert plain == {"algorithm": {"gamma": 0.95}, "base": 0.95}


def test_snapshot_fails_closed_on_hash_or_missing_metadata(tmp_path):
    data, config = _fixture_data()
    path = write_snapshot(tmp_path, data, config, _identity())
    (path / "payload.pt").write_bytes(b"not a snapshot")
    with pytest.raises(SnapshotValidationError, match="SHA256"):
        load_snapshot(path)
    data, config = _fixture_data()
    del data.non_tensor_batch["reward_trace"]
    with pytest.raises(SnapshotValidationError, match="reward_trace"):
        write_snapshot(tmp_path, data, config, _identity(), snapshot_name="missing")


def test_snapshot_rejects_unknown_schema_and_hook_is_disabled_by_default(tmp_path, monkeypatch):
    data, config = _fixture_data()
    path = write_snapshot(tmp_path, data, config, _identity())
    manifest_path = path / "manifest.json"
    import json
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "unknown_v999"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SnapshotValidationError, match="unsupported snapshot schema"):
        load_snapshot(path)
    monkeypatch.delenv("VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR", raising=False)
    assert maybe_write_pre_update_snapshot(data, config, global_step=7) is None


def test_historical_reward_and_no_concat_gae_reconstruct_with_whitening(tmp_path):
    data, config = _fixture_data()
    path = write_snapshot(tmp_path, data, config, _identity())
    payload, _, loaded = load_snapshot(path)
    replay = recompute_branches(payload, loaded)
    for row, trace in enumerate(payload["metadata"]["reward_trace"]):
        assert reward_from_trace(trace, "historical", ppo_gamma=0.95) == pytest.approx(
            payload["tensor_batch"]["token_level_scores"][row].sum().item(), abs=1e-6
        )
    assert torch.equal(replay["historical"]["advantages"], payload["tensor_batch"]["advantages"])
    assert torch.equal(replay["historical"]["returns"], payload["tensor_batch"]["returns"])
    assert torch.equal(replay["branches"]["S1"]["raw_advantages"], replay["branches"]["S1"]["raw_advantages"])
    assert not torch.equal(replay["branches"]["S1"]["independent_advantages"], replay["branches"]["S1"]["fixed_reference_advantages"])


def test_s1_minus_s0_is_only_potential_and_terminal_truncation_metadata_is_preserved(tmp_path):
    data, config = _fixture_data()
    path = write_snapshot(tmp_path, data, config, _identity())
    payload, _, loaded = load_snapshot(path)
    replay = recompute_branches(payload, loaded)
    traces = payload["metadata"]["reward_trace"]
    for row, trace in enumerate(traces):
        expected = 0.95 * trace["score"]["phi"] - trace["score"]["phi_prev"]
        assert replay["branches"]["S1"]["turn_totals"][row] - replay["branches"]["S0"]["turn_totals"][row] == pytest.approx(expected)
    assert payload["metadata"]["truncated"] == [False, True, False]
    assert payload["metadata"]["parsed_primitive_actions"][0] == ["move_forward", "turn_left"]


def test_explicitly_skipped_potential_event_stays_zero_in_reward_ablations():
    trace = _trace(potential=0.0, phi_prev=0.1, phi=0.9)
    trace["components"]["potential"].update({"events": 0, "scale": None, "raw": 0.0, "scaled": 0.0, "applied": 0.0})
    assert reward_from_trace(trace, "S1", ppo_gamma=0.95) == pytest.approx(reward_from_trace(trace, "S0", ppo_gamma=0.95))
    assert reward_from_trace(trace, "S5", ppo_gamma=0.95) == pytest.approx(reward_from_trace(trace, "S0", ppo_gamma=0.95))


def test_no_concat_turn_order_is_numeric_with_string_group_ids():
    scores = torch.ones((3, 1), dtype=torch.float32)
    values = torch.zeros_like(scores)
    mask = torch.ones_like(scores, dtype=torch.long)
    # Row order is arbitrary; trajectory order must be turn 1 -> 2 -> 10.
    result = no_concat_gae(
        scores,
        values,
        mask,
        ["uuid-group"] * 3,
        [0, 0, 0],
        [10, 2, 1],
        gamma=1.0,
        lam=1.0,
    )
    assert result["returns"].squeeze(1).tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_no_concat_prefix_padding_and_reorder_preserve_unique_whitening_population():
    # Five variable-length turns cannot be split over four DP ranks. Mimic
    # production prefix padding (5 -> 8) followed by an arbitrary balance
    # reorder; the three copies must not affect GAE or whitening statistics.
    response_mask = torch.tensor(
        [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [1, 1, 0]],
        dtype=torch.long,
    )
    scores = torch.zeros((5, 3), dtype=torch.float32)
    scores[0, 0], scores[1, 1], scores[2, 2] = 0.1, -0.2, 0.3
    scores[3, 0], scores[4, 1] = -0.4, 0.5
    values = torch.tensor(
        [[0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]],
        dtype=torch.float32,
    )
    groups = ["uuid-a", "uuid-a", "uuid-a", "uuid-b", "uuid-b"]
    trajectories = [0, 0, 0, 1, 1]
    turns = [1, 2, 10, 1, 2]

    baseline = no_concat_gae(
        scores, values, response_mask, groups, trajectories, turns, gamma=0.95, lam=0.95
    )
    padded_source = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    balance_order = torch.tensor([6, 2, 0, 7, 4, 1, 5, 3])
    selected = padded_source.index_select(0, balance_order)
    padded_scores = scores.index_select(0, selected)
    padded_values = values.index_select(0, selected)
    padded_mask = response_mask.index_select(0, selected)
    padded_groups = [groups[index] for index in selected.tolist()]
    padded_trajectories = [trajectories[index] for index in selected.tolist()]
    padded_turns = [turns[index] for index in selected.tolist()]
    replay = no_concat_gae(
        padded_scores,
        padded_values,
        padded_mask,
        padded_groups,
        padded_trajectories,
        padded_turns,
        gamma=0.95,
        lam=0.95,
    )
    production_advantages, production_returns = compute_gae_no_concat_advantage_return_firsttok(
        SimpleNamespace(
            batch={"token_level_scores": padded_scores, "values": padded_values, "response_mask": padded_mask},
            non_tensor_batch={
                "group_idx": padded_groups,
                "traj_idx": padded_trajectories,
                "turn_idx": padded_turns,
            },
        ),
        gamma=0.95,
        lam=0.95,
    )

    assert replay["unique_keys"].shape[0] == 5
    assert replay["inverse_indices"].shape[0] == 8
    assert torch.equal(replay["advantages"], production_advantages)
    assert torch.equal(replay["returns"], production_returns)
    assert torch.equal(replay["whitening_stats"][0], baseline["whitening_stats"][0])
    assert torch.equal(replay["whitening_stats"][1], baseline["whitening_stats"][1])


def test_bootstrap_pixel_source_roundtrips_without_becoming_a_value_estimate(tmp_path):
    data, config = _fixture_data()
    data.non_tensor_batch["bootstrap_context"][1]["next_images"] = [{"mode": "L", "size": [1, 1], "pixel_bytes": b"\x7f"}]
    path = write_snapshot(tmp_path, data, config, _identity())
    payload, _, _ = load_snapshot(path)
    source = payload["metadata"]["bootstrap_context"][1]["next_images"][0]
    assert source["pixel_bytes"] == {"__snapshot_bytes_hex__": "7f"}


class _Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.frozen = torch.nn.Linear(1, 1, bias=False)
        self.frozen.weight.requires_grad_(False)


class _Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.2))


def test_gradient_branches_are_independent_and_never_step(tmp_path):
    data, config = _fixture_data()
    path = write_snapshot(tmp_path, data, config, _identity())
    payload, _, loaded = load_snapshot(path)
    actor, critic = _Actor(), _Critic()
    before_actor = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    before_critic = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    report = gradient_diagnosis(
        payload, loaded, actor=actor, critic=critic,
        actor_logprob=lambda model, item: model.scale * item["tensor_batch"]["responses"].float(),
        critic_value=lambda model, item: model.scale * (item["tensor_batch"]["responses"].float() + 1.0),
    )
    assert report["optimizer_step_called"] is False
    assert report["s1_vs_s0"]["actor_gradient_cosine"] is not None
    assert report["branches"]["S0"]["actor_gradients"]["frozen"]["status"] == "FROZEN"
    assert all(torch.equal(actor.state_dict()[key], value) for key, value in before_actor.items())
    assert all(torch.equal(critic.state_dict()[key], value) for key, value in before_critic.items())
