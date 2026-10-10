#!/usr/bin/env python3
"""Bind RLinf PPO to a PandaOmron QwenOFT checkpoint and its training contract."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_active_spatial_sft import file_hash

from vagen.envs.robocasa.starvla_contract import PolicyContract


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def inspect_policy(ckpt, unnorm_key=None):
    ckpt = Path(ckpt).resolve(strict=True)
    if ckpt.suffix != ".pt" or not ckpt.is_file() or ckpt.stat().st_size == 0:
        raise ValueError(
            "Pass an actual StarVLA .pt file, not a HF backbone or RLinf resume directory"
        )
    # Match starVLA read_mode_config exactly, including final_model/*.pt.
    run = ckpt.parent.parent
    config_path, stats_path = run / "config.yaml", run / "dataset_statistics.json"
    cfg = yaml.safe_load(config_path.read_text())
    contract = PolicyContract.from_config(cfg)
    if contract.include_state:
        raise ValueError(
            "RLinf OFT is state-free; a state-conditioned SFT checkpoint needs another adapter"
        )
    action = cfg["framework"]["action_model"]
    if int(action.get("past_action_window_size", 0)) != 0:
        raise ValueError("This handoff requires future-only action chunks")
    if (
        "future_action_window_size" in action
        and int(action["future_action_window_size"]) + 1 != contract.horizon
    ):
        raise ValueError("Conflicting StarVLA action horizon aliases")
    base = Path(cfg["framework"]["qwenvl"]["base_vlm"])
    if not base.is_absolute() or not (base / "config.json").is_file():
        raise ValueError("Saved base_vlm must be an existing absolute HF path")
    stats = json.loads(stats_path.read_text())
    key = unnorm_key or (next(iter(stats)) if len(stats) == 1 else None)
    if key not in stats:
        raise ValueError(
            f"Select --unnorm-key from checkpoint statistics: {list(stats)}"
        )
    action_stats = stats[key]["action"]
    low, high = (np.asarray(action_stats[k], dtype=np.float32) for k in ("min", "max"))
    mask = np.asarray(action_stats.get("mask", [True] * 12))
    if (
        low.shape != (12,)
        or high.shape != (12,)
        or mask.shape != (12,)
        or not mask.all()
    ):
        raise ValueError(
            "Expected 12 continuous min/max statistics in gym action order"
        )
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
        raise ValueError("Invalid normalization ranges")
    return (
        cfg,
        contract,
        {
            "checkpoint": file_identity(ckpt),
            "unnorm_key": key,
            "files": {
                str(p): file_hash(p)
                for p in (config_path, stats_path, base / "config.json")
            },
            "weight_identity": "file size and mtime; not a full tensor hash",
        },
    )


def check_rlinf_patch(root):
    path = root / "rlinf/models/embodiment/starvla/utils/action_space.py"
    spec = importlib.util.spec_from_file_location("rlinf_action_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        getattr(module, "ROBOCASA365_ACTION_CONTRACT", None)
        != "panda_omron_continuous_minmax_v1"
    ):
        raise ValueError(
            "Apply examples/train/robocasa/patches/rlinf_robocasa_starvla.patch to this RLinf checkout"
        )
    sample = np.full((1, 16, 12), -0.75, dtype=np.float32)
    actual = module.unnormalize_actions_for_env(
        sample,
        {"q01": -np.ones(12), "q99": np.ones(12), "mask": np.ones(12, dtype=bool)},
        "robocasa365",
    )
    np.testing.assert_allclose(actual, sample)


def build_config(
    template,
    ckpt,
    contract,
    out,
    tasks,
    train_split,
    eval_split,
    steps,
    num_envs,
    episode_steps,
    unnorm_key,
):
    if (
        steps <= 0
        or num_envs <= 0
        or episode_steps <= 0
        or episode_steps % contract.horizon
    ):
        raise ValueError(
            "Positive budgets required; episode steps must be divisible by the action horizon"
        )
    cfg = copy.deepcopy(template)
    cfg["defaults"][0:2] = ["env/robocasa365@env.train", "env/robocasa365@env.eval"]
    if "_self_" not in cfg["defaults"]:
        cfg["defaults"].insert(-1, "_self_")
    cfg["runner"].update(
        max_steps=steps, max_epochs=steps, val_check_interval=1, save_interval=1
    )
    cfg["runner"]["logger"].update(
        log_path=str(out / "logs"), experiment_name="robocasa365_starvla_ppo"
    )
    cfg["algorithm"].update(
        adv_type="gae",
        loss_type="actor_critic",
        group_size=1,
        update_epoch=1,
        kl_beta=0.0,
        clip_ratio_high=0.2,
        clip_ratio_low=0.2,
    )
    for mode, split, seed in (("train", train_split, 42), ("eval", eval_split, 1042)):
        env = cfg["env"][mode]
        env.update(
            total_num_envs=num_envs,
            rollout_epoch=1,
            split=split,
            task_soup=None,
            task_names=tasks,
            task_mode=None,
            task_filter=[],
            group_size=1,
            seed=seed,
            auto_reset=True,
            ignore_terminations=False,
            episode_horizon_source="max_episode_steps",
            max_episode_steps=episode_steps,
            max_steps_per_rollout_epoch=episode_steps,
            task_sampling_strategy="ordered" if mode == "eval" else "random",
        )
        # LeRobot cameras are rendered at 256 then resized by the training PIL transform.
        env["init_params"] = {"camera_heights": 256, "camera_widths": 256}
        env["observation"] = {
            "main_camera_key": "robot0_agentview_left_image",
            "extra_camera_keys": ["robot0_agentview_right_image"],
            "wrist_camera_key": "robot0_eye_in_hand_image",
            "flip_images_vertical": True,
        }
        env["action_space"] = {
            "env_action_dim": 12,
            "disable_base_control": False,
            "binarize_gripper_control": False,
        }
    cfg["actor"].update(micro_batch_size=1, global_batch_size=64)
    cfg["actor"]["fsdp_config"]["save_full_model_weights"] = True
    cfg["actor"]["optim"].update(lr=1e-6)
    model = cfg["actor"]["model"]
    model.update(
        model_path=str(ckpt),
        action_dim=12,
        num_action_chunks=contract.horizon,
        unnorm_key=unnorm_key,
        policy_setup="robocasa365",
        action_stats_source="minmax",
        use_proprio=False,
        add_value_head=True,
    )
    model["starvla"].update(
        enable_state_input=False, expected_image_size=list(contract.image_size)
    )
    cfg["rollout"]["model"]["model_path"] = str(ckpt)
    # OFT eval must use its mean; the native train path samples a Gaussian.
    cfg["rollout"]["sampling_params"]["do_sample"] = False
    return cfg


def prepare(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    rlinf = Path(args.rlinf_root).resolve(strict=True)
    check_rlinf_patch(rlinf)
    _, contract, identity = inspect_policy(args.ckpt, args.unnorm_key)
    ckpt = Path(identity["checkpoint"]["path"])
    template_path = (
        rlinf / "examples/embodiment/config/libero_spatial_grpo_starvla.yaml"
    )
    cfg = build_config(
        yaml.safe_load(template_path.read_text()),
        ckpt,
        contract,
        out,
        args.task,
        args.train_split,
        args.eval_split,
        args.steps,
        args.num_envs,
        args.episode_steps,
        identity["unnorm_key"],
    )
    cfg["hydra"]["searchpath"] = ["file://" + str(rlinf / "examples/embodiment/config")]
    identity.update(
        rlinf_root=str(rlinf),
        rlinf_revision=subprocess.check_output(
            ["git", "-C", str(rlinf), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_template_sha256=file_hash(template_path),
        note="Metadata preflight only. Run GPU forward/replay and an optimizer-step smoke before scaling.",
    )
    out.mkdir(parents=True)
    config_path = out / "train.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    identity["config_sha256"] = file_hash(config_path)
    (out / "handoff.json").write_text(json.dumps(identity, indent=2) + "\n")
    return config_path


def check(config):
    path = Path(config).resolve(strict=True)
    saved = json.loads((path.parent / "handoff.json").read_text())
    if file_hash(path) != saved["config_sha256"]:
        raise ValueError(
            "Prepared config changed; regenerate this handoff for a new experiment"
        )
    _, _, actual = inspect_policy(saved["checkpoint"]["path"], saved["unnorm_key"])
    for key in ("checkpoint", "files"):
        if actual[key] != saved[key]:
            raise ValueError(f"StarVLA source changed: {key}")
    rlinf = Path(saved["rlinf_root"])
    revision = subprocess.check_output(
        ["git", "-C", str(rlinf), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != saved["rlinf_revision"]:
        raise ValueError("RLinf revision changed after preparation")
    check_rlinf_patch(rlinf)
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("prepare")
    gen.add_argument("--ckpt", required=True)
    gen.add_argument("--out", required=True)
    gen.add_argument(
        "--unnorm-key",
        help="Required only if the checkpoint contains multiple statistics keys",
    )
    gen.add_argument("--rlinf-root", default=str(ROOT / "third_party/RLinf"))
    gen.add_argument("--task", action="append", required=True)
    gen.add_argument(
        "--train-split", choices=("pretrain", "target"), default="pretrain"
    )
    gen.add_argument("--eval-split", choices=("pretrain", "target"), default="target")
    gen.add_argument("--steps", type=int, default=2)
    gen.add_argument("--num-envs", type=int, default=8)
    gen.add_argument("--episode-steps", type=int, default=512)
    verify = sub.add_parser("check")
    verify.add_argument("--config", required=True)
    args = p.parse_args()
    print(
        prepare(args)
        if args.mode == "prepare"
        else json.dumps(check(args.config), indent=2)
    )


if __name__ == "__main__":
    main()
