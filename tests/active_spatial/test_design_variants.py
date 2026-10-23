"""Round 12 freeze, the dense-reward design variants (the user's 10th supplement of 2026-10-23), each behind its own
setting, default off: the grouped progress's weights (V1 / V2), the soft gate (V3), the progress reward's scale
(V4 / V5, an existing env setting), the one-off group milestones instead of the potential difference (V6)."""

from __future__ import annotations

import argparse

import pytest

from data_gen.interiorgs_v10 import training_prep as TP
from vagen.envs.active_spatial import target_visibility as TV
from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig


def groups(frame=1.0, face=1.0, relation=1.0, frame_met=True, face_met=True, relation_met=False):
    return {"frame": frame, "face": face, "relation": relation, "frame_met": frame_met, "face_met": face_met,
            "relation_met": relation_met}


def test_grouped_progress_weights_and_gates():
    g = groups(frame=0.6, face=0.8, relation=0.5, frame_met=False)
    assert TV.grouped_progress(g) == pytest.approx(0.25 * 0.6 + 0.25 * 0.8)  # hard gate: no relation
    assert TV.grouped_progress(g, "grouped_soft") == pytest.approx(0.25 * 0.6 + 0.25 * 0.8 + 0.5 * 0.5 * 0.6)
    assert TV.grouped_progress(g, "grouped_gated", [0.15, 0.15, 0.7]) == pytest.approx(0.15 * 0.6 + 0.15 * 0.8)
    met = groups(frame=1.0, face=0.8, relation=0.5)
    third = 1 / 3
    assert TV.grouped_progress(met, "grouped_gated", [third] * 3) == pytest.approx(third * (1.0 + 0.8 + 0.5))
    # with every frame condition met the soft and the hard gate agree (frame group 1)
    assert TV.grouped_progress(met, "grouped_soft") == pytest.approx(TV.grouped_progress(met))


def test_apply_visibility_with_weights_and_soft_gate():
    frame = {"objects_present": 1.0, "in_front": 1.0, "visible": 1.0, "min_area": 0.4, "inside_frame": 1.0}
    metric = {"success": False, "continuous_gate_progress": 0.0,
              "gates": {**{k: True for k in frame}, "min_area": False, "occupancy": False},
              "shaping_components": {**frame, "occupancy": 0.5}}
    vis = {"targets": {"a": {"visible": True}}, "references": {}, "found": True, "visible": True, "progress": 1.0}
    default = TV.apply_visibility(metric, vis, progress_version="grouped_gated")
    assert TV.apply_visibility(metric, vis, progress_version="grouped_gated",
                               progress_weights=[0.25, 0.25, 0.5]) == default
    assert default["continuous_gate_progress"] == pytest.approx(0.25 * 0.4 + 0.25)  # min_area not met: gate off
    soft = TV.apply_visibility(metric, vis, progress_version="grouped_soft")
    assert soft["continuous_gate_progress"] == pytest.approx(0.25 * 0.4 + 0.25 + 0.5 * 0.5 * 0.4)
    assert soft["success"] == default["success"] and soft["gates"] == default["gates"]
    g = soft["progress_groups"]
    assert (g["frame_met"], g["face_met"], g["relation_met"]) == (False, True, False)
    with pytest.raises(ValueError):
        TV.apply_visibility(metric, vis, progress_version="grouped_gated", progress_weights=[0.5, 0.5])


def test_new_milestones_pay_once_per_group():
    reached = set()
    assert TV.new_milestones(groups(face_met=False), reached, [0.5, 0.5, 1.0]) == (0.5, ["frame"])
    assert TV.new_milestones(groups(face_met=False), reached, [0.5, 0.5, 1.0]) == (0.0, [])
    assert TV.new_milestones(groups(relation_met=True), reached, [0.5, 0.5, 1.0]) == (1.5, ["face", "relation"])
    assert TV.new_milestones(groups(relation_met=True), reached, [0.5, 0.5, 1.0]) == (0.0, [])
    assert reached == {"frame", "face", "relation"}


def metric_of(g, success=False):
    progress = TV.grouped_progress(g)
    return {"success": success, "metric_version": "fixture", "shaping_score": 0.5 * progress + 0.5 * success,
            "score": 0.5 * progress, "gates": {}, "progress_groups": g}


def make_env(state, **overrides):
    kwargs = dict(jsonl_path="", render_backend=None, action_space="strafe", enable_collision_detection=False,
                  enable_visibility_check=False, enable_low_info_frame_check=False, format_reward=0.0,
                  potential_field_progress_mode="potential", potential_field_gamma=0.99,
                  potential_field_reward_scale=10.0, success_reward=5.0, near_success_bonus=0.0, step_penalty=0.0,
                  progress_version="grouped_gated", enable_auto_termination=False)
    kwargs.update(overrides)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**kwargs))
    env._renderer_initialized = True

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "fixture", "multi_modal_input": {}}

    env._render = fake_render
    env._calculate_canonical_metric = lambda: metric_of(state["g"])
    env.potential_field = object()  # the canonical branch only needs one to exist
    env.reset(0)
    return env


def act():
    return "<think>fixture</think><action>move_forward|</action>"


def test_env_milestones_replace_the_potential_difference():
    values = [0.5, 0.5, 1.0]
    # the frame group is met at the start: it counts as reached and never pays
    state = {"g": groups(frame=1.0, face=0.3, face_met=False, relation=0.0)}
    env = make_env(state, milestone_rewards=values)
    if not env.current_task.get("target_region"):
        pytest.skip("the fixture item has no target region (the canonical branch is not reached)")
    paid = []
    for g in (groups(face=1.0, relation=0.2),                       # face met: +0.5
              groups(face=1.0, relation=0.6),                       # progress only: 0 (no potential reward)
              groups(face=0.4, face_met=False, relation=0.6),       # face lost
              groups(face=1.0, relation=0.7),                       # face again: already reached, 0
              groups(face=1.0, relation=1.0, relation_met=True)):   # relation met: +1.0
        state["g"] = g
        _, reward, _, info = env.step(act())
        comps = info["reward_trace"]["components"]
        assert comps["potential"]["applied"] == 0.0
        assert info["reward_trace"]["accounting"]["reconstruction_error"] == pytest.approx(0.0, abs=1e-12)
        paid.append(comps["milestone"]["applied"])
    assert paid == pytest.approx([0.5, 0.0, 0.0, 0.0, 1.0])
    # without the setting the same poses pay the potential difference and no milestone
    state["g"] = groups(frame=1.0, face=0.3, face_met=False, relation=0.0)
    plain = make_env(state)
    state["g"] = groups(face=1.0, relation=0.2)
    _, _, _, info = plain.step(act())
    comps = info["reward_trace"]["components"]
    assert comps["potential"]["applied"] != 0.0
    assert comps.get("milestone", {}).get("applied", 0.0) == 0.0


def test_rl_config_design_variant_overrides():
    dense = TP.reward_overrides("D_cond")
    ns = argparse.Namespace(progress_version="grouped_soft", progress_weights="0.15,0.15,0.70",
                            progress_reward_scale=30.0, milestone_rewards=None)
    assert TP.design_variant_overrides(ns, dense) == {"progress_version": "grouped_soft",
                                                      "progress_weights": [0.15, 0.15, 0.7],
                                                      "potential_field_reward_scale": 30.0}
    assert TP.design_variant_overrides(argparse.Namespace(), dense) == {}
    with pytest.raises(SystemExit):  # weights need a grouped version
        TP.design_variant_overrides(argparse.Namespace(progress_weights="1,1,1"), TP.reward_overrides("dense"))
    with pytest.raises(SystemExit):  # the progress options need a dense preset
        TP.design_variant_overrides(argparse.Namespace(progress_reward_scale=3.0), TP.reward_overrides("S0"))
    with pytest.raises(SystemExit):
        TP.design_variant_overrides(argparse.Namespace(milestone_rewards="0.5,0.5"), dense)
    assert ActiveSpatialEnvConfig().progress_weights is None and ActiveSpatialEnvConfig().milestone_rewards is None
