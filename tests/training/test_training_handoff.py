"""Data isolation and configuration contracts across the training stages."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from export_rlinf_starvla import extract_policy
from prepare_active_spatial_rl import prepare as prepare_rl
from prepare_active_spatial_sft import prepare, qa_record
from prepare_rlinf_robocasa import build_config, check_rlinf_patch

from vagen.envs.active_spatial.dataset_contract import (
    CONTRACT_VERSION,
    protocol,
    row_digest,
)
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
from vagen.envs.robocasa.starvla_contract import CAMERAS, PolicyContract


def qa(scene, image):
    return {
        "sample_id": scene,
        "scene_id": scene,
        "split": "train",
        "task_type": "projective_relations",
        "private_answer": "Yes",
        "qa_sft_allowed": True,
        "label_validity": "valid",
        "observability_validity": "valid",
        "public_observation": {
            "image_path": str(image),
            "observation_config": "single_image",
            "history": [],
        },
        "question": "Is the chair left of the table? Answer Yes or No.",
        "_audit": {"secret": "ORACLE_ONLY"},
        "state_pose_c2w": "PRIVATE_POSE",
    }


def inputs(tmp):
    image = tmp / "image.png"
    Image.new("RGB", (8, 8), "red").save(image)
    model = tmp / "base"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2_5_vl"}')
    trajectories, bank = [], []
    for scene in ("a", "b", "c", "d"):
        trajectories.append(
            {
                "id": scene,
                "scene_id": scene,
                "success": True,
                "score_contract": {"success_source": "canonical_gates"},
                "image_paths": [str(image)],
                "conversations": [
                    {
                        "role": "user",
                        "content": "<image> Move left",
                        "image_path": str(image),
                    },
                    {
                        "role": "assistant",
                        "content": "<think>oracle score 0.9</think><action>move_left</action>",
                    },
                ],
            }
        )
        bank.append(qa(scene, image))
    traj, qa_path = tmp / "traj.jsonl", tmp / "qa.jsonl"
    for path, rows in ((traj, trajectories), (qa_path, bank)):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return SimpleNamespace(
        out=str(tmp / "prepared"),
        model=str(model),
        trajectory=[str(traj)],
        qa_bank=[str(qa_path)],
        qa_probability=0.5,
        val_fraction=0.25,
        seed=42,
        split_manifest=None,
        cutoff_len=16384,
    )


def test_mixed_sft_shares_scene_split_and_excludes_oracle_metadata(tmp_path):
    args = inputs(tmp_path)
    result = prepare(args)
    out = Path(args.out)
    assert not set(result["train_scenes"]) & set(result["val_scenes"])
    for scene in "abcd":
        records = [r for r in result["records"] if r["scene"] == scene]
        assert len({r["split"] for r in records}) == 1
    text = (out / "qa_train.jsonl").read_text() + (
        out / "trajectory_train.jsonl"
    ).read_text()
    for private in ("ORACLE_ONLY", "PRIVATE_POSE", "oracle score", "<think>"):
        assert private not in text
    cfg = yaml.safe_load((out / "train.yaml").read_text())
    assert cfg["val_size"] == 0 and cfg["eval_dataset"] == "trajectory_val,qa_val"
    args.out = str(tmp_path / "ablation")
    args.qa_probability = 0
    args.split_manifest = str(out / "split_manifest.json")
    assert prepare(args)["val_scenes"] == result["val_scenes"]
    with pytest.raises(FileExistsError):
        prepare(args)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"qa_sft_allowed": False}, "qa_sft_disallowed"),
        ({"private_answer": None}, "invalid_label"),
        ({"observability_validity": "invalid_missing_render"}, "unobservable"),
        ({"task_type": "delta_control"}, "history_not_supported"),
    ],
)
def test_qa_rejects_unsuitable_supervision(tmp_path, change, reason):
    assert qa_record(qa("a", tmp_path / "missing.png") | change, tmp_path) == (
        None,
        reason,
    )


def test_qa_never_imports_test_labels(tmp_path):
    with pytest.raises(ValueError, match="training bank"):
        qa_record(qa("a", "missing.png") | {"split": "test"}, tmp_path)


def test_missing_or_corrupt_image_prevents_publication(tmp_path):
    args = inputs(tmp_path)
    (tmp_path / "image.png").write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        prepare(args)
    assert not Path(args.out).exists()


def test_ppo_warmstart_preserves_sft_holdout_and_action_protocol(tmp_path):
    args = inputs(tmp_path)
    split = prepare(args)
    (Path(args.model) / "model.safetensors").write_bytes(b"fixture")
    template = tmp_path / "ppo.yaml"
    template.write_text(
        yaml.safe_dump(
            {
                "algorithm": {"adv_estimator": "no_concat_gae"},
                "trainer": {"concat_multi_turn": False},
                "actor_rollout_ref": {
                    "model": {"path": "old_base"},
                    "actor": {"optim": {}},
                },
                "critic": {"model": {"path": "old_base"}, "optim": {}},
                "data": {},
            }
        )
    )
    env = tmp_path / "env.yaml"
    env.write_text(
        yaml.safe_dump(
            {"envs": [{"name": "ActiveSpatial", "config": {"step_rotation_deg": 20}}]}
        )
    )
    tasks = tmp_path / "tasks.jsonl"
    rows = [
        {
            "scene_id": s,
            "task_type": "projective_relations",
            "canonical_task_metric_version": "canonical_spatial_task_h1_v1",
            "camera_model_version": "canonical_camera_h1_resize_v1",
        }
        for s in "abcd"
    ]
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rl_args = SimpleNamespace(
        out=str(tmp_path / "ppo_run"),
        model=args.model,
        steps=2,
        tasks=str(tasks),
        template=str(template),
        env_yaml=str(env),
        split_manifest=str(Path(args.out) / "split_manifest.json"),
    )
    with pytest.raises(ValueError, match="missing dataset contract"):
        prepare_rl(rl_args)
    assert not Path(rl_args.out).exists()
    certificate = Path(str(tasks) + ".contract.json")
    certificate.write_text(
        json.dumps(
            {
                "version": CONTRACT_VERSION,
                "status": "PASS",
                "protocol": protocol(ActiveSpatialEnvConfig(step_rotation_deg=20)),
                "rows": {
                    row_digest(row): {"success": True, "rgb_match": True, "steps": 1}
                    for row in rows
                },
            }
        )
    )
    config = prepare_rl(rl_args)
    cfg = yaml.safe_load(config.read_text())
    assert cfg["actor_rollout_ref"]["model"]["path"] == args.model
    assert cfg["critic"]["model"]["path"] == args.model
    assert cfg["trainer"]["resume_mode"] == "disable"
    for name in ("train", "val"):
        spec = yaml.safe_load(Path(cfg["data"][name + "_files"]).read_text())["envs"][
            0
        ]["config"]
        scenes = {
            json.loads(line)["scene_id"]
            for line in Path(spec["jsonl_path"]).read_text().splitlines()
        }
        assert scenes == set(split[name + "_scenes"])
        assert spec["prompt_format"] == "no_think" and spec["step_rotation_deg"] == 20
        assert spec["require_verified_dataset"] is True
        assert spec["dataset_contract_path"] == str(certificate)
    rl_args.out = str(tmp_path / "changed_protocol")
    env.write_text(
        env.read_text().replace("step_rotation_deg: 20", "step_rotation_deg: 15")
    )
    with pytest.raises(ValueError, match="protocol differs"):
        prepare_rl(rl_args)
    assert not Path(rl_args.out).exists()


def test_canonical_qa_camera_matches_runtime_resize():
    from data_gen.active_spatial_qa.render_paired_bank import render_camera

    row = {
        "task_type": "projective_relations",
        "source_task_metric_version": "canonical_spatial_task_h1_v1",
        "source_camera_model_version": "canonical_camera_h1_resize_v1",
        "state_pose_c2w": np.eye(4).tolist(),
        "camera": {"intrinsics": [[640, 0, 320], [0, 640, 240], [0, 0, 1]]},
    }
    K, w2c = render_camera(row, 256, 256)
    np.testing.assert_allclose(K, [[256, 0, 128], [0, 640 * 256 / 480, 128], [0, 0, 1]])
    np.testing.assert_allclose(w2c, np.eye(4))
    row.pop("source_camera_model_version")
    with pytest.raises(ValueError, match="camera_model_version"):
        render_camera(row, 256, 256)


def test_runtime_compare_does_not_silently_downgrade_canonical_version(tmp_path):
    from data_gen.active_spatial_qa.runtime_compare import compare

    row = {
        "sample_id": "a",
        "task_type": "projective_relations",
        "private_answer": "No",
        "source_task_metric_version": "future_unsupported",
        "state_pose_c2w": np.eye(4).tolist(),
        "camera": {"intrinsics": np.eye(3).tolist()},
    }
    bank = tmp_path / "bank.jsonl"
    bank.write_text(
        json.dumps(row) + "\n" + json.dumps({"private_answer": None}) + "\n"
    )
    report = compare(str(bank), str(tmp_path / "report.json"))
    assert (
        report["status"] == "FAIL" and report["errors"] == 1 and report["skipped"] == 1
    )
    assert "canonical_task_metric_version" in report["details"][0]["error"]


@pytest.mark.skipif(
    not (ROOT / "third_party/RLinf").is_dir(),
    reason="Optional RLinf checkout is absent",
)
def test_rlinf_recipe_overrides_libero_contract():
    template = yaml.safe_load(
        (
            ROOT
            / "third_party/RLinf/examples/embodiment/config/libero_spatial_grpo_starvla.yaml"
        ).read_text()
    )
    cfg = build_config(
        template,
        Path("/policy.pt"),
        PolicyContract(CAMERAS, (224, 224), False, 16),
        Path("/out"),
        ["NavigateKitchen"],
        "pretrain",
        "target",
        2,
        8,
        512,
        "new_embodiment",
    )
    assert cfg["actor"]["model"]["unnorm_key"] == "new_embodiment"
    assert cfg["actor"]["model"]["policy_setup"] == "robocasa365"
    assert cfg["actor"]["model"]["action_dim"] == 12
    assert cfg["actor"]["model"]["num_action_chunks"] == 16
    assert cfg["actor"]["model"]["add_value_head"] is True
    assert cfg["algorithm"]["adv_type"] == "gae"
    assert cfg["env"]["train"]["seed"] != cfg["env"]["eval"]["seed"]
    assert cfg["env"]["train"]["observation"]["extra_camera_keys"] == [
        "robot0_agentview_right_image"
    ]
    assert cfg["env"]["train"]["action_space"]["disable_base_control"] is False


@pytest.mark.skipif(
    not (ROOT / "third_party/RLinf").is_dir(),
    reason="Optional RLinf checkout is absent",
)
def test_rlinf_robocasa_action_contract():
    check_rlinf_patch(ROOT / "third_party/RLinf")
    path = (
        ROOT / "third_party/RLinf/rlinf/models/embodiment/starvla/utils/action_space.py"
    )
    spec = importlib.util.spec_from_file_location("rlinf_actions_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = {
        "q01": np.full(12, -2.0),
        "q99": np.full(12, 2.0),
        "mask": np.ones(12, dtype=bool),
    }
    x = np.linspace(-1.2, 1.2, 24, dtype=np.float32).reshape(1, 2, 12)
    np.testing.assert_allclose(
        module.unnormalize_actions_for_env(x, stats, "robocasa365"), x * 2, atol=1e-6
    )
    stats["mask"][6] = False
    with pytest.raises(ValueError, match="continuous"):
        module.unnormalize_actions_for_env(x, stats, "robocasa365")


def test_rlinf_export_preserves_learned_action_head_and_rejects_wrong_backbones():
    reference = {
        "qwen_vl_interface.model.embed_tokens.weight": np.zeros((8, 4)),
        "action_model.net.weight": np.zeros((12, 4)),
    }
    trained = {"starvla_model." + k: v + 1 for k, v in reference.items()}
    trained.update(actor_logstd=np.zeros(12), **{"value_head.weight": np.zeros((1, 4))})
    exported = extract_policy(trained, reference)
    assert np.all(exported["action_model.net.weight"] == 1)
    assert set(exported) == set(reference)
    with pytest.raises(ValueError, match="shape"):
        extract_policy(
            trained, reference | {"action_model.net.weight": np.zeros((7, 4))}
        )
    with pytest.raises(ValueError, match="does not match"):
        extract_policy(trained | {"unexpected_adapter.weight": np.zeros(1)}, reference)
