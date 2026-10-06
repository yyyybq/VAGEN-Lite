"""Regression coverage for the ten end-to-end Active Spatial audit findings."""
import copy
import json
from dataclasses import asdict, fields
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
import importlib.util
import sys
from pathlib import Path


def script_module(name):
    key = "audit_test_" + name
    spec = importlib.util.spec_from_file_location(key, Path(__file__).resolve().parents[2] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module

from data_gen.active_spatial_pipeline.splits import (
    assert_disjoint, categories, content_key, id_candidates, instances,
    split_training_rows, write_rows,
)
from vagen.utils.observation_history import ObservationHistory


def row(index=0, scene="seen", labels=("chair",), task="absolute_positioning"):
    return {"task_id": str(index), "scene_id": scene, "task_type": task,
            "object_label": "+".join(labels), "task_description": "Move to any position 1.0m from chair",
            "distance": float(index + 1), "init_camera": {"extrinsics": np.eye(4).tolist()},
            "target_region": {"sample_point": [index, 0, 0]},
            "target_object": {"objects": [{"id": f"{index}-{label}", "label": label,
                                            "center": [index, 0, 1]} for label in labels]}}


def test_holdout_reserved_before_filtering():
    rows = [row(i, task="delta_control" if i % 3 == 0 else "absolute_positioning") for i in range(12)]
    train, val = split_training_rows(rows, train_size=9, test_size=3, train_exclude=["delta_control"])
    assert {r["task_id"] for r in train} == {"1", "2", "4", "5", "7", "8"}
    assert {r["task_id"] for r in val} == {"9", "10", "11"}
    assert_disjoint(train, val)
    with pytest.raises(ValueError, match="quota"):
        split_training_rows(rows, train_size=9, test_size=3, delta_min=2)


def test_leakage_detects_paraphrases_and_source_aliases():
    a = row()
    b = {**a, "task_id": "alias", "task_description": "paraphrased"}
    assert content_key(a) == content_key(b)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint([a], [b])
    with pytest.raises(ValueError, match="sources"):
        assert_disjoint([{**a, "source_key": "x"}], [{**row(2), "source_key": "x"}])


def test_frozen_manifest_is_not_overwritten(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_rows(path, [row()])
    write_rows(path, [row()])
    with pytest.raises(FileExistsError):
        write_rows(path, [row(2)])


def test_eval_uses_env_success_not_high_score(tmp_path):
    from evaluation.eval_config import EvalConfig
    from evaluation.eval_runner import EvalRunner
    runner = EvalRunner(EvalConfig(output_dir=str(tmp_path), max_steps_per_episode=1))
    runner.env = SimpleNamespace(
        reset=lambda seed: ({}, {}), _current_step=1,
        step=lambda action: ({}, 0.0, True, {"current_potential_score": .99, "auto_terminated": True,
                                            "metrics": {"traj_metrics": {"success": False}}}),
    )
    runner.agent = SimpleNamespace(reset=lambda: None, act=lambda *args: "<action>move_forward</action>")
    result = runner._run_single_episode(0, row(), 0)
    assert result.final_score == .99
    assert result.success is False


def test_eval_runtime_schema_is_shared():
    from evaluation.eval_config import EvalEnvConfig
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
    assert asdict(EvalEnvConfig()) == asdict(ActiveSpatialEnvConfig())
    assert {f.name for f in fields(EvalEnvConfig)} == {f.name for f in fields(ActiveSpatialEnvConfig)}


def test_checkpoint_protocol_wins_over_suite_defaults(tmp_path):
    make_eval_config = script_module("active_spatial_eval_sweep").make_eval_config
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump({"envs": [{"max_turns": 12, "config": {
        "jsonl_path": "train.jsonl", "action_space": "strafe", "enable_explicit_done": False,
        "image_width": 256, "image_height": 256, "success_score_threshold": .65,
        "step_rotation_deg": 20, "history_window_size": 3}}]}))
    exp = SimpleNamespace(name="test", train_yaml=path)
    ckpt = SimpleNamespace(step=1, model_dir=tmp_path)
    suite = {"name": "id_test", "jsonl_path": "eval.jsonl"}
    defaults = {"defaults": {"env": {"image_width": 512, "success_score_threshold": .85}}}
    result = make_eval_config(exp, ckpt, suite, defaults, tmp_path, "model")
    assert result["env"]["image_width"] == 256
    assert result["env"]["success_score_threshold"] == .65
    assert result["env"]["action_space"] == "strafe"
    assert result["env"]["enable_explicit_done"] is False
    assert result["env"]["history_window_size"] == 3
    assert result["max_steps_per_episode"] == 12
    with pytest.raises(ValueError, match="protocol_overrides"):
        make_eval_config(exp, ckpt, {**suite, "env": {"action_space": "legacy"}}, defaults, tmp_path, "model")


@pytest.mark.parametrize("size", [1, 3])
def test_history_keeps_images_and_complete_turns_together(size):
    history = ObservationHistory(size)
    for i in range(5):
        history.append({"role": "user", "content": str(i)}, [f"image{i}"])
        history.append({"role": "assistant", "content": f"action{i}"})
    assert history.images == [f"image{i}" for i in range(5-size, 5)]
    assert len(history.messages) == 2 * size
    assert history.messages[0]["role"] == "user"
    if size > 1:
        assert history.drop_oldest_turn()
        assert len(history.images) == size - 1
    else:
        assert not history.drop_oldest_turn()


def test_active_spatial_seeds_are_exhaustive_half_open():
    from vagen.gym_agent_dataset import EnvSpec, _generate_seeds_for_spec
    spec = EnvSpec(name="ActiveSpatial", n_envs=25, seed=[0, 25])
    assert _generate_seeds_for_spec(spec, 42, 0) == list(range(25))
    spec.n_envs = 26
    with pytest.raises(ValueError, match="unique"):
        _generate_seeds_for_spec(spec, 42, 0)
    spec.seed, spec.n_envs = [4], 3
    assert _generate_seeds_for_spec(spec, 42, 0) == [4, 5, 6]


def test_id_scene_task_and_atomic_category_constraints():
    train = [row(0), row(1, labels=("table",))]
    good = row(2, labels=("chair", "table"))
    candidates = [good, row(3, scene="new"), row(4, task="new_task"), row(5, labels=("new_category",))]
    assert id_candidates(train, candidates) == [good]
    assert categories(good) == {"chair", "table"}


def test_ood_instance_and_category_are_distinct_from_scene_and_composition():
    module = script_module("gen_ood_splits")
    make_ood_instance, make_ood_category, geometry_thresholds = module.make_ood_instance, module.make_ood_category, module.geometry_thresholds
    train = [row(0), row(1, labels=("table",))]
    new_instance = row(2, labels=("chair", "table"))
    unseen_scene = row(3, scene="new")
    novel_category = row(4, labels=("vase",))
    labels = {"chair", "table"}
    known_instances = set().union(*(instances(r) for r in train))
    candidates = [*train, new_instance, unseen_scene, novel_category]
    assert make_ood_instance(candidates, {"seen"}, labels, 100, known_instances) == [new_instance]
    assert make_ood_category(candidates, labels, 100) == [novel_category]
    assert geometry_thresholds([row(0), row(99)])["absolute_positioning"] != (1.50, 2.58)


def test_formal_contract_binds_rows_and_protocol(tmp_path):
    from vagen.envs.active_spatial.dataset_contract import (
        CONTRACT_VERSION, protocol, row_digest, validate_contract,
        CANONICAL_CAMERA_H1_RESIZE_V1, CANONICAL_TASK_METRIC_VERSION,
    )
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
    cfg = ActiveSpatialEnvConfig()
    canonical = {**row(task="projective_relations"), "camera_model_version": CANONICAL_CAMERA_H1_RESIZE_V1,
                 "canonical_task_metric_version": CANONICAL_TASK_METRIC_VERSION}
    path = tmp_path / "contract.json"
    doc = {"version": CONTRACT_VERSION, "status": "PASS", "protocol": protocol(cfg),
           "rows": {row_digest(canonical): {"success": True, "rgb_match": True, "steps": 2}}}
    path.write_text(json.dumps(doc))
    validate_contract([canonical], cfg, path)
    with pytest.raises(ValueError, match="canonical"):
        validate_contract([row()], cfg, path)
    with pytest.raises(ValueError, match="uncertified"):
        validate_contract([{**canonical, "task_description": "changed"}], cfg, path)
    cfg.step_rotation_deg = 20
    with pytest.raises(ValueError, match="protocol"):
        validate_contract([canonical], cfg, path)
    with pytest.raises(ValueError, match="missing"):
        validate_contract([canonical], cfg, tmp_path / "missing.json")


def test_difficulty_estimate_uses_action_protocol():
    from data_gen.active_spatial_pipeline.config import InitialViewConfig
    from data_gen.active_spatial_pipeline.pipeline import validate_init_position
    # All gates except the step estimate disabled; 1 metre is 3.33, not 10 steps.
    cfg = InitialViewConfig()
    cfg.task_min_distances = {"absolute_positioning": 0}
    cfg.task_max_init_scores = {"absolute_positioning": 2}
    cfg.task_min_yaw_offsets = {"absolute_positioning": 0}
    cfg.task_min_total_steps = {"absolute_positioning": 5}
    region = {"type": "circle", "sample_point": [1, 0, 0], "params": {"center": [1, 0], "radius": 1}}
    good, reason, _ = validate_init_position(np.zeros(3), np.array([1, 0, 0]), region,
                                                      "absolute_positioning", init_view_config=cfg)
    assert not good and "too_few_steps" in reason


def test_sft_metric_has_no_search_bonus():
    from data_gen.active_spatial_sft.path_finder import score_c2w, task_success
    field = SimpleNamespace(compute_score=lambda **kwargs: SimpleNamespace(total_score=.93, position_score=.93, orientation_score=1.0))
    score = score_c2w(np.eye(4), field, "absolute_positioning", {}, {"params": {"object_center": [0, 0, 1]}})[0]
    assert score == .93
    assert not task_success(np.eye(4), score, {}, .95)


@pytest.mark.parametrize("action", ["move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right", "look_up", "look_down"])
def test_sft_actions_match_runtime(action):
    from data_gen.active_spatial_sft.path_finder import simulate_action
    from vagen.envs.active_spatial.utils import ViewManipulator
    engine = ViewManipulator(step_translation=.3, step_rotation_deg=20)
    engine.reset(np.eye(4))
    engine.step(action)
    assert np.allclose(simulate_action(np.eye(4), action, .3, 20), engine.get_pose())


def test_sft_prompt_uses_configured_motion_and_actions():
    from data_gen.active_spatial_sft.path_finder import Trajectory
    from data_gen.active_spatial_sft.sft_formatter import format_trajectory
    trajectory = Trajectory([], np.eye(4), .93, .93, False, 0)
    result = format_trajectory(row(), trajectory, ["frame.png"], "test", step_translation=.4,
                               step_rotation_deg=20, action_space="strafe", enable_explicit_done=False)
    system = result["conversations"][0]["content"]
    assert "20" in system and "0.4" in system and "move_left" in system
    assert "<action>done" not in json.dumps(result["conversations"])


def test_qa_invalid_predictions_remain_in_denominators():
    from data_gen.active_spatial_qa.qa_eval import answer_metrics, parse_answer
    result = answer_metrics(["Yes"]*10 + ["No"]*10, ["Yes"] + [None]*9 + ["No"] + [None]*9)
    assert result["accuracy"] == pytest.approx(.1)
    assert result["balanced_accuracy"] == pytest.approx(.1)
    assert result["format_failure_rate"] == pytest.approx(.9)
    assert parse_answer("Yes, no") is None
    assert parse_answer("Yes. No.") is None
    assert parse_answer("No.") == "No"


def test_sapave_unwraps_nested_objects_and_marks_unknown():
    from data_gen.sapave.taxonomy import _objects, sapave_visibility
    sample = row(labels=("chair", "table"))
    assert len(_objects(sample)) == 2
    assert sapave_visibility({}) == "unknown"
    assert sapave_visibility({"task_type": "occlusion_alignment"}) == "unknown"


@pytest.mark.parametrize("nested", [False, True])
def test_actual_launcher_materializes_disjoint_enumerated_manifests(tmp_path, nested):
    import os
    import subprocess
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source.jsonl"
    write_rows(source, [row(i) for i in range(8)])
    config = {"env1": {"env_config": {"jsonl_path": str(source), "exclude_task_types": []}}}
    target = config["env1"]["env_config"] if nested else config["env1"]
    target.update(train_size=6, test_size=2)
    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    launcher = (root / "examples/train/active_spatial/run_experiment.sh").read_text()
    # Execute only the real YAML generation heredoc, never Ray/GPU/cache setup.
    body = launcher.split("$PYTHON - <<PYEOF\n", 1)[1].split("\nPYEOF", 1)[0]
    env = dict(os.environ)
    env.update(ENV_CONFIG_PATH=str(yaml_path), EXPERIMENT_DIR=str(tmp_path), WINDOW_SIZE="3",
               REMOTE_ENV_URLS="", REMOTE_ENV_TIMEOUT="30", REMOTE_ENV_RETRIES="1",
               REMOTE_ENV_TOKEN="", REMOTE_ENV_RENDERING_GPU="", RENDERING_GPU="0",
               RENDER_BACKEND="http", CLIENT_URL="http://unused/render", MAX_TURNS="12",
               MAX_RESPONSE_LENGTH="160", TRAIN_YAML=str(tmp_path / "train.yaml"),
               VAL_YAML=str(tmp_path / "val.yaml"), ALLOW_LEGACY_ACTIVE_SPATIAL="1",
               PYTHONPATH=str(root))
    result = subprocess.run(["bash", "-c", f'"{sys.executable}" - <<PYEOF\n{body}\nPYEOF'],
                            env=env, cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    train = yaml.safe_load((tmp_path / "train.yaml").read_text())["envs"][0]
    val = yaml.safe_load((tmp_path / "val.yaml").read_text())["envs"][0]
    assert train["seed_list"] == list(range(6))
    assert val["seed_list"] == [0, 1]
    assert train["config"]["jsonl_path"] != val["config"]["jsonl_path"]
    assert train["config"]["history_window_size"] == 3
    assert "train_size" not in train["config"]


def test_sft_replay_does_not_keep_optimistic_success():
    from data_gen.active_spatial_sft.config import SFTGenerationConfig
    from data_gen.active_spatial_sft.path_finder import Trajectory, TrajStep
    from data_gen.active_spatial_sft.sft_generator import SFTDataGenerator
    item = row()
    pose = np.eye(4)
    step = TrajStep(0, pose, ["turn_left"], pose, .1, .99, .1, .99, .1, .99)
    trajectory = Trajectory([step], pose, .1, .99, True, 1)
    generator = SFTDataGenerator(SFTGenerationConfig(enable_collision_detection=False, enable_explicit_done=False))
    image = object()
    obs = {"multi_modal_data": {"image": [image]}}
    generator._runtime = SimpleNamespace(
        reset=lambda seed: (obs, {}), current_item=item, collision_count=0, invalid_action_count=0,
        view_engine=SimpleNamespace(get_pose=lambda: pose), final_score=.93,
        step=lambda action: (obs, 0., True, {"metrics": {"traj_metrics": {"success": False, "final_score": .93}}}),
    )
    assert len(generator._replay(item, 0, trajectory)) == 2
    assert not trajectory.success and trajectory.final_score == .93


def test_canonical_unknown_version_never_falls_back():
    from vagen.envs.active_spatial.canonical_task_metrics import uses_canonical_backend
    with pytest.raises(ValueError, match="unknown"):
        uses_canonical_backend({"canonical_task_metric_version": "future_version", "task_type": "fov_inclusion"})


def test_result_fingerprint_tracks_protocol_and_data(tmp_path):
    from evaluation.eval_config import EvalConfig, evaluation_fingerprint
    source = tmp_path / "data.jsonl"
    write_rows(source, [row()])
    config = EvalConfig()
    config.env.jsonl_path = str(source)
    first = evaluation_fingerprint(config.to_dict())
    config.output_dir = "different/output"
    assert evaluation_fingerprint(config.to_dict()) == first
    config.env.action_space = "strafe"
    assert evaluation_fingerprint(config.to_dict()) != first
    config.env.action_space = "legacy"
    source.write_text(json.dumps(row(1)) + "\n")
    assert evaluation_fingerprint(config.to_dict()) != first
