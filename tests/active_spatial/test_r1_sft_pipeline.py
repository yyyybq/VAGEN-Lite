from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_gen.active_spatial_sft.convert_to_qwen25vl_sft import convert_jsonl
from data_gen.active_spatial_sft.config import SFTGenerationConfig
from data_gen.active_spatial_sft.path_finder import (
    Trajectory,
    TrajStep,
    find_trajectory,
    score_c2w,
    simulate_action,
    task_success,
)
from data_gen.active_spatial_sft.sft_formatter import format_trajectory
from data_gen.active_spatial_sft.sft_generator import SFTDataGenerator
from data_gen.active_spatial_sft.visualize_score_guidance import visualize


def _canonical_item():
    return {
        "canonical_task_metric_version": "canonical_spatial_task_h1_v1",
        "camera_model_version": "canonical_h1_resize_v1",
        "scene_id": "fixture",
        "task_id": "fixture-task",
        "task_type": "projective_relations",
        "task_description": "Position where chair appears to the left of table",
        "init_camera": {
            "extrinsics": np.eye(4).tolist(),
            "intrinsics": [[128.0, 0.0, 128.0], [0.0, 128.0, 128.0], [0.0, 0.0, 1.0]],
        },
    }


def _metric_for_pose(_item, pose):
    score = float(np.clip(0.2 + pose[0, 3], 0.0, 1.0))
    success = score >= 0.49
    return {
        "metric_version": "canonical_spatial_task_h1_v1",
        "shaping_score_version": "canonical_gate_aligned_shaping_v1",
        "score": 0.99,
        "shaping_score": score,
        "success": success,
        "gates": {"fixture": success},
        "shaping_components": {"fixture": score},
        "inside_frame_fraction_min": score,
        "relation_margin_px": score * 20.0,
    }


def test_canonical_search_uses_gate_aligned_shaping_not_historical_score(monkeypatch):
    import vagen.envs.active_spatial.canonical_task_metrics as metrics

    monkeypatch.setattr(metrics, "score_canonical_task", _metric_for_pose)
    item = _canonical_item()
    params = {"_canonical_item": item}
    score, position, orientation = score_c2w(
        np.eye(4), object(), item["task_type"], params, {}
    )
    assert (score, position, orientation) == (0.2, 0.2, 0.2)
    assert task_success(np.eye(4), score, params, 0.1) is False


def test_beam_search_stops_at_first_success_depth(monkeypatch):
    import vagen.envs.active_spatial.canonical_task_metrics as metrics

    monkeypatch.setattr(metrics, "score_canonical_task", _metric_for_pose)
    item = _canonical_item()
    trajectory = find_trajectory(
        np.eye(4),
        item["task_type"],
        {"_canonical_item": item, "_action_space": "strafe"},
        {},
        step_translation=0.3,
        step_rotation_deg=20.0,
        success_threshold=0.49,
        max_total_actions=5,
        max_actions_per_turn=5,
        min_improvement=0.001,
        beam_width=8,
    )
    assert trajectory.success is True
    assert trajectory.total_actions == 1
    assert trajectory.steps[0].actions == ["move_right"]


def test_formatter_emits_atomic_score_and_runtime_reward_trace(monkeypatch):
    import vagen.envs.active_spatial.canonical_task_metrics as metrics

    monkeypatch.setattr(metrics, "score_canonical_task", _metric_for_pose)
    item = _canonical_item()
    before = np.eye(4)
    middle = simulate_action(before, "move_right", step_translation=0.3, step_rotation_deg=20.0)
    after = simulate_action(middle, "turn_left", step_translation=0.3, step_rotation_deg=20.0)
    step = TrajStep(0, before, ["move_right", "turn_left"], after, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5)
    runtime = {"accounting": {"actual_total_reward": 0.27}}
    trajectory = Trajectory(
        [step], after, 0.2, 0.5, True, 2,
        scene_id="fixture", item_idx=0, runtime_reward_traces=[runtime],
    )
    record = format_trajectory(
        item,
        trajectory,
        ["images/start.png", "images/end.png"],
        "fixture-sft",
        step_translation=0.3,
        step_rotation_deg=20.0,
        action_space="strafe",
        enable_explicit_done=False,
    )
    assert record["score_contract"]["search_score_versions"] == [
        "canonical_gate_aligned_shaping_v1"
    ]
    assert len(record["trajectory_trace"]) == 2
    assert record["trajectory_trace"][1]["runtime_reward"] == 0.27
    assert len(record["primitive_trajectory_trace"]) == 3
    assert record["primitive_trajectory_trace"][1]["action_from_previous"] == "move_right"
    assert record["primitive_trajectory_trace"][-1]["score"] == 0.5
    assert record["conversations"][-1]["role"] == "assistant"
    assert record["conversation_image_paths"] == ["images/start.png"]


def test_primitive_frame_render_reuses_replay_boundaries_and_restores_pose():
    before = np.eye(4)
    middle = simulate_action(before, "move_right", 0.3, 20.0)
    after = simulate_action(middle, "turn_left", 0.3, 20.0)
    step = TrajStep(0, before, ["move_right", "turn_left"], after, 0.2, 0.5,
                    0.2, 0.5, 0.2, 0.5)
    trajectory = Trajectory([step], after, 0.2, 0.5, True, 2)
    start_image = Image.new("RGB", (8, 8), (1, 2, 3))
    final_image = Image.new("RGB", (8, 8), (4, 5, 6))

    class ViewEngine:
        def __init__(self):
            self.pose = after.copy()

        def get_pose(self):
            return self.pose

        def reset(self, pose):
            self.pose = np.asarray(pose).copy()

    class Runtime:
        def __init__(self):
            self.view_engine = ViewEngine()
            self.extra_renders = 0

        def _render_image(self):
            self.extra_renders += 1
            return Image.new("RGB", (8, 8), (7, 8, 9))

    generator = SFTDataGenerator(SFTGenerationConfig(
        enable_collision_detection=False,
        step_translation=0.3,
        step_rotation_deg=20.0,
    ))
    generator._runtime = Runtime()
    frames = generator._render_primitive_frames(
        trajectory, [start_image, final_image]
    )
    assert len(frames) == 3
    assert frames[0] is start_image
    assert frames[-1] is final_image
    assert generator._runtime.extra_renders == 1
    assert np.allclose(generator._runtime.view_engine.get_pose(), after)


def test_qwen_conversion_and_visualization_validate_image_alignment(tmp_path, monkeypatch):
    import vagen.envs.active_spatial.canonical_task_metrics as metrics

    monkeypatch.setattr(metrics, "score_canonical_task", _metric_for_pose)
    item = _canonical_item()
    before = np.eye(4)
    after = simulate_action(before, "move_right", step_translation=0.3, step_rotation_deg=20.0)
    step = TrajStep(0, before, ["move_right"], after, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5)
    trajectory = Trajectory([step], after, 0.2, 0.5, True, 1, scene_id="fixture", item_idx=0)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name, color in (("start.png", (20, 40, 60)), ("end.png", (80, 100, 120))):
        Image.new("RGB", (64, 64), color).save(image_dir / name)
    record = format_trajectory(
        item,
        trajectory,
        ["images/start.png", "images/end.png"],
        "fixture-sft",
        step_translation=0.3,
        step_rotation_deg=20.0,
        action_space="strafe",
        enable_explicit_done=False,
    )
    record["primitive_image_paths"] = ["images/start.png", "images/end.png"]
    source = tmp_path / "sft_data.jsonl"
    source.write_text(json.dumps(record) + "\n")
    qwen = tmp_path / "qwen.jsonl"
    assert convert_jsonl(source, qwen, tmp_path, strict=True) == (1, 1)
    converted = json.loads(qwen.read_text())
    assert len(converted["images"]) == 1
    assert all(Path(path).is_file() for path in converted["images"])

    qwen_no_think = tmp_path / "qwen_no_think.jsonl"
    assert convert_jsonl(
        source, qwen_no_think, tmp_path, strict=True, strip_think=True
    ) == (1, 1)
    no_think = json.loads(qwen_no_think.read_text())
    assert all("<think>" not in turn["content"] for turn in no_think["messages"])
    assert "provide only your action" in no_think["messages"][0]["content"]

    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    qwen_parquet = tmp_path / "qwen.parquet"
    assert convert_jsonl(
        source, qwen_parquet, tmp_path, strict=True, to_parquet=True
    ) == (1, 1)
    parquet_row = pandas.read_parquet(qwen_parquet).iloc[0]
    assert len(parquet_row["images"]) == 1

    summary = visualize(source, tmp_path / "visualization", max_dashboards=1)
    assert summary["records"] == 1
    assert summary["success_rate"] == 1.0
    assert (tmp_path / "visualization/index.html").is_file()
    assert (tmp_path / "visualization/dashboards/fixture-sft.png").is_file()


def test_strict_qwen_conversion_does_not_publish_partial_output(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({
        "conversations": [
            {"role": "system", "content": "navigate"},
            {
                "role": "user",
                "content": "<image>\nTask: navigate",
                "image_path": "images/missing.png",
            },
            {"role": "assistant", "content": "<action>move_forward|</action>"},
        ],
        "image_paths": ["images/missing.png"],
    }) + "\n")
    output = tmp_path / "qwen.jsonl"
    with pytest.raises(ValueError, match="failed validation"):
        convert_jsonl(source, output, tmp_path, strict=True)
    assert not output.exists()
    assert not (tmp_path / ".qwen.jsonl.tmp").exists()
