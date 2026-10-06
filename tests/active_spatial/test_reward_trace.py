from __future__ import annotations

import math

import pytest

from vagen.envs.active_spatial.collision_detector import CollisionResult
from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
from vagen.envs.active_spatial.reward_trace import (
    TurnRewardTrace,
    near_reward_values,
    potential_reward_values,
    visibility_reward_values,
)
from vagen.envs.active_spatial.spatial_potential_field import ScoreResult
from vagen.envs.active_spatial.visibility_checker import (
    VisibilityResult,
    compute_visibility_reward,
)


class SequencePotential:
    def __init__(self, scores):
        self.scores = list(scores)

    def compute_score(self, **_kwargs):
        phi, pos, ori = self.scores.pop(0)
        return ScoreResult(
            total_score=phi,
            position_score=pos,
            orientation_score=ori,
            details={
                "dynamic_position_weight": 0.7,
                "dynamic_orientation_weight": 0.3,
            },
        )


class AlwaysCollision:
    scene_loaded = False

    def check_collision(self, **_kwargs):
        return CollisionResult(True, "object", collision_object="fixture")


def make_env(scores, **overrides):
    kwargs = dict(
        jsonl_path="",
        render_backend=None,
        action_space="strafe",
        enable_collision_detection=False,
        enable_visibility_check=False,
        enable_low_info_frame_check=False,
        format_reward=0.01,
        invalid_format_penalty=-0.2,
        potential_field_progress_mode="potential",
        potential_field_gamma=0.95,
        potential_field_reward_scale=1.0,
        success_score_threshold=0.9,
        success_reward=5.0,
        near_success_bonus=0.0,
        step_penalty=-0.01,
    )
    kwargs.update(overrides)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**kwargs))
    env._renderer_initialized = True

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "fixture", "multi_modal_input": {}}

    env._render = fake_render
    env.potential_field = SequencePotential(scores)
    _, reset_info = env.reset(0)
    return env, reset_info


def action(*names):
    return f"<think>fixture</think><action>{'|'.join(names)}|</action>"


def test_default_formula_matches_pre_componentization_behavior():
    env, _ = make_env([(0.1, 0.08, 0.02), (0.3, 0.24, 0.06)])
    _, reward, done, info = env.step(action("move_forward"))
    old_total = (0.95 * 0.3 - 0.1) + 0.01 - 0.01
    assert reward == pytest.approx(old_total, abs=1e-12)
    assert done is False
    assert info["success"] is False
    assert info["reward_trace"]["accounting"]["reconstruction_error"] == pytest.approx(0.0, abs=1e-12)


def test_component_sum_reconstructs_actual_total():
    trace = TurnRewardTrace("task", "episode", 1, "absolute_positioning")
    trace.record("collision", raw=2, scale=-0.15)
    trace.record("format", raw=1, scale=0.01)
    trace.record("potential", raw=0.2, scale=1.0)
    trace.record("step", raw=1, scale=-0.01)
    trace.finalize(
        actual_total_reward=-0.1,
        primitive_step_end=2,
        success=False,
        terminated=False,
        truncated=False,
        termination_reason="continuing",
    )
    assert trace.reconstructed_total() == pytest.approx(-0.1, abs=1e-12)
    assert abs(trace.reconstruction_error) <= 1e-12


def test_shaping_off_keeps_score_success_and_termination():
    scores = [(0.4, 0.3, 0.1), (0.7, 0.5, 0.2)]
    common = dict(success_score_threshold=0.65, near_success_threshold=0.55, near_success_bonus=0.5)
    legacy, _ = make_env(scores, **common)
    sparse, _ = make_env(
        scores,
        enable_potential_shaping_reward=False,
        enable_near_success_reward=False,
        enable_visibility_shaping_reward=False,
        **common,
    )
    _, reward_legacy, done_legacy, info_legacy = legacy.step(action("move_forward"))
    _, reward_sparse, done_sparse, info_sparse = sparse.step(action("move_forward"))
    lt = info_legacy["reward_trace"]
    st = info_sparse["reward_trace"]
    assert lt["score"] == st["score"]
    assert info_legacy["success"] == info_sparse["success"] is True
    assert done_legacy == done_sparse is True
    assert lt["outcome"] == st["outcome"]
    assert reward_legacy != reward_sparse
    assert st["components"]["potential"]["scaled"] != 0.0
    assert st["components"]["potential"]["applied"] == 0.0
    assert st["components"]["near"]["scaled"] == 0.5
    assert st["components"]["near"]["applied"] == 0.0


def test_near_interval_records_raw_scaled_and_applied():
    env, _ = make_env(
        [(0.4, 0.3, 0.1), (0.6, 0.45, 0.15)],
        enable_potential_shaping_reward=False,
        near_success_threshold=0.55,
        near_success_bonus=0.5,
    )
    _, reward, done, info = env.step(action("move_forward"))
    near = info["reward_trace"]["components"]["near"]
    assert near["raw"] == 1.0
    assert near["scale"] == 0.5
    assert near["scaled"] == near["applied"] == 0.5
    assert reward == pytest.approx(0.5, abs=1e-12)
    assert done is False


def test_success_termination_and_initial_success_are_explicit():
    env, reset_info = make_env(
        [(0.7, 0.6, 0.1), (0.7, 0.6, 0.1)],
        success_score_threshold=0.65,
        enable_potential_shaping_reward=False,
    )
    assert reset_info["initial_potential_score"] == 0.7
    assert env.episode_done is False
    _, reward, done, info = env.step(action("turn_left"))
    assert done is True
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert info["termination_reason"] == "success_auto"
    assert info["reward_trace"]["components"]["success"]["applied"] == 5.0
    assert reward == pytest.approx(5.0, abs=1e-12)


def test_primitive_timeout_logs_terminal_phi_without_terminal_shaping():
    env, _ = make_env([(0.1, 0.08, 0.02), (0.3, 0.24, 0.06)], max_episode_steps=1)
    _, reward, done, info = env.step(action("move_forward"))
    trace = info["reward_trace"]
    assert done is True
    assert info["terminated"] is False
    assert info["truncated"] is True
    assert info["termination_reason"] == "max_primitive_steps"
    assert trace["score"]["phi_prev"] == 0.1
    assert trace["score"]["phi"] == 0.3
    assert trace["components"]["potential"]["events"] == 0
    assert trace["components"]["step"]["events"] == 0
    assert reward == pytest.approx(0.01, abs=1e-12)


def test_collision_and_consecutive_collision_are_separate_terms():
    env, _ = make_env(
        [(0.1, 0.08, 0.02)],
        enable_collision_detection=True,
        max_consecutive_collisions=1,
        collision_penalty=-0.15,
        consecutive_collision_penalty=-0.5,
    )
    env.collision_detector = AlwaysCollision()
    _, reward, done, info = env.step(action("move_forward"))
    components = info["reward_trace"]["components"]
    assert done is True
    assert info["termination_reason"] == "consecutive_collision"
    assert components["collision"]["applied"] == -0.15
    assert components["consecutive_collision"]["applied"] == -0.5
    assert components["potential"]["events"] == 0
    assert reward == pytest.approx(-0.64, abs=1e-12)


def test_invalid_action_override_and_termination_are_audited():
    env, _ = make_env([(0.1, 0.08, 0.02)], max_consecutive_invalid_actions=1)
    _, reward, done, info = env.step("not an action")
    trace = info["reward_trace"]
    assert reward == -0.2
    assert done is True
    assert info["termination_reason"] == "consecutive_invalid"
    assert trace["components"]["invalid"]["applied"] == -0.2
    assert trace["processing"]["overrides"][0]["kind"] == "hard_invalid_reward_assignment"


def test_post_render_low_info_adjustment_is_not_hidden_in_a_residual():
    env, _ = make_env(
        [(0.1, 0.08, 0.02), (0.1, 0.08, 0.02)],
        enable_low_info_frame_check=True,
        max_consecutive_low_info_frames=1,
        low_info_frame_penalty=-0.2,
    )

    def low_info_render(**_kwargs):
        env.last_image_std = 0.0
        env.last_low_info_frame = True
        env.consecutive_low_info_frame_count = 1
        env.low_info_frame_count = 1
        return {"obs_str": "fixture", "multi_modal_input": {}}

    env._render = low_info_render
    _, _, done, info = env.step(action("move_forward"))
    trace = info["reward_trace"]
    assert done is True
    assert info["termination_reason"] == "low_info"
    assert trace["components"]["low_info"]["applied"] == -0.2
    assert trace["accounting"]["reconstruction_error"] == pytest.approx(0.0, abs=1e-12)


def test_multi_primitive_bundle_is_one_turn_and_one_step_penalty():
    env, _ = make_env([(0.1, 0.08, 0.02), (0.2, 0.16, 0.04)])
    _, _, done, info = env.step(action("move_forward", "turn_left", "move_right"))
    trace = info["reward_trace"]
    assert done is False
    assert trace["identity"]["turn_id"] == 1
    assert trace["clock"]["primitive_action_count"] == 3
    assert trace["components"]["step"]["events"] == 1


def test_potential_and_visibility_helpers_match_legacy_formulas():
    raw, scaled, _ = potential_reward_values(
        mode="potential", phi_prev=0.2, phi=0.4, scale=1.2, gamma=0.95
    )
    assert raw == pytest.approx(0.95 * 0.4 - 0.2)
    assert scaled == pytest.approx(1.2 * (0.95 * 0.4 - 0.2))

    previous = VisibilityResult(False, False, 1.0, 0.0, 2.0)
    current = VisibilityResult(True, True, 0.0, 0.1, 1.0)
    visibility_raw, _ = visibility_reward_values(current, previous)
    assert visibility_raw * 0.3 == pytest.approx(
        compute_visibility_reward(current, previous, reward_scale=0.3), abs=1e-12
    )

    near_raw, near_scaled, hit = near_reward_values(
        phi=0.55, threshold=0.55, bonus=0.5, mode="constant", steepness=10.0
    )
    assert (near_raw, near_scaled, hit) == (1.0, 0.5, True)


def test_all_flat_component_fields_are_present_even_when_zero():
    trace = TurnRewardTrace("task", "episode", 1, "fov_inclusion")
    trace.finalize(
        actual_total_reward=0.0,
        primitive_step_end=0,
        success=False,
        terminated=False,
        truncated=False,
        termination_reason="continuing",
    )
    flat = trace.flat()
    for name in ("potential", "near", "visibility", "success", "format", "invalid", "collision", "step"):
        assert f"reward_trace/{name}/raw" in flat
        assert f"reward_trace/{name}/scaled" in flat
        assert f"reward_trace/{name}/applied" in flat
    assert math.isclose(flat["reward_trace/reconstruction_error"], 0.0, abs_tol=1e-12)
