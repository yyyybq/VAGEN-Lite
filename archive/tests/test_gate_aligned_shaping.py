from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from vagen.envs.active_spatial.canonical_task_metrics import canonical_projective
from vagen.envs.active_spatial.prompt import system_prompt
from vagen.envs.active_spatial.env import _canonical_shaping_score


def _object(*, inside_fraction: float = 1.0, area_ratio: float = 0.01):
    side = 100.0 * inside_fraction**0.5
    return {
        "center_in_front": True,
        "visible": True,
        "area_ratio": area_ratio,
        "bbox_raw": [0.0, 0.0, 100.0, 100.0],
        "bbox": [0.0, 0.0, side, side],
    }


def _result(*, margin: float, inside_fraction: float = 1.0, legacy_score: float = 0.99):
    return {
        "visual_metrics": {
            "objects": [_object(inside_fraction=inside_fraction) for _ in range(2)],
            "visual_relation_margin_px": margin,
            "visual_relation_satisfied": margin > 0.0,
            "visual_score": legacy_score,
        }
    }


def test_canonical_gate_is_preserved_and_margin_is_centered_at_12px():
    below = canonical_projective(_result(margin=11.999))
    boundary = canonical_projective(_result(margin=12.0))
    above = canonical_projective(_result(margin=12.001))

    assert below["success"] is False
    assert boundary["success"] is True
    assert above["success"] is True
    assert boundary["shaping_components"]["margin"] == pytest.approx(0.5)
    assert below["shaping_score"] < boundary["shaping_score"] < above["shaping_score"]
    assert below["shaping_score"] < 0.5 <= boundary["shaping_score"]
    assert boundary["gates"]["margin"] is True


def test_large_margin_cannot_mask_an_out_of_frame_bbox():
    metric = canonical_projective(
        _result(margin=300.0, inside_fraction=0.25, legacy_score=0.999)
    )

    assert metric["success"] is False
    assert metric["gates"]["inside_frame"] is False
    assert metric["score"] == pytest.approx(0.999)
    assert metric["inside_frame_fraction_min"] == pytest.approx(0.25)
    assert metric["shaping_score"] <= 0.125
    assert _canonical_shaping_score(metric) == metric["shaping_score"]


def test_projective_prompt_disambiguates_image_order_from_turn_direction():
    prompt = system_prompt(
        task_type="projective_relations",
        action_space="strafe",
        format_reward=0.0,
        invalid_format_penalty=-0.1,
    )

    assert "rendered image" in prompt
    assert "do not map directly to turn_left/turn_right" in prompt
    assert "first try lateral translation" in prompt
    assert "If collision feedback" in prompt
    assert "gives no positive reward" in prompt
    assert "invalid format: -0.1" in prompt
    assert "bbox inside-frame fraction" in prompt
    assert "centered at 12 px" in prompt
    assert "Canonical projective success" in prompt
    assert "Progress toward target pose" not in prompt


def test_fixed32_evaluation_uses_the_training_reward_contract():
    script = Path(__file__).resolve().parents[2] / "scripts/r1_run_canonical_dev_eval32.py"
    spec = importlib.util.spec_from_file_location("gate_aligned_eval_runner_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    config = module.env_config(
        SimpleNamespace(renderer_url="http://renderer/render", gs_root=Path("/assets")),
        Path("/frozen/policy.jsonl"),
    )

    assert config.potential_field_progress_mode == "potential"
    assert config.potential_field_gamma == pytest.approx(0.95)
    assert config.enable_potential_shaping_reward is True
    assert config.enable_near_success_reward is False
    assert config.enable_visibility_shaping_reward is False
    assert config.near_success_bonus == 0.0
    assert config.format_reward == 0.0
    assert config.invalid_format_penalty == pytest.approx(-0.1)
