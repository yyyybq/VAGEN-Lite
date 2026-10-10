#!/usr/bin/env python3
"""Warm-start Active Spatial PPO while preserving the mixed-SFT scene split."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml
from prepare_active_spatial_sft import file_hash, read_rows

from vagen.envs.active_spatial.dataset_contract import validate_contract
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig


def prepare(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    model = Path(args.model).resolve(strict=True)
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError(
            "Pass a complete HF SFT model, not a metadata-only actor directory"
        )
    if (model / "adapter_config.json").exists():
        raise ValueError(
            "Merge the LoRA adapter before using it as a PPO initialization"
        )
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    split = json.loads(Path(args.split_manifest).read_text())
    if split.get("trajectory_prompt_format") != "no_think":
        raise ValueError(
            "Expected a mixed-SFT manifest with the no_think trajectory protocol"
        )
    from active_spatial_release_contract import scene_partitions, load_registry, bind_raw
    lookup = scene_partitions(split)
    strict = split.get("version") == 2
    registry = load_registry(getattr(args, "task_registry", None), split) if strict else {}
    train_scenes, val_scenes = set(split["train_scenes"]), set(split["val_scenes"])
    test_scenes = set(split.get("test_scenes", []))
    if train_scenes & val_scenes:
        raise ValueError("Overlapping train/validation scenes")
    rows = read_rows(args.tasks)
    if {str(r["scene_id"]) for r in rows} - (train_scenes | val_scenes | test_scenes):
        raise ValueError("Unknown task scenes; use the same frozen corpus as SFT")
    if strict:
        for row in rows:
            bind_raw(row, registry, lookup)
    source_env = yaml.safe_load(Path(args.env_yaml).read_text())
    if len(source_env["envs"]) != 1 or source_env["envs"][0]["name"] != "ActiveSpatial":
        raise ValueError(
            "This recipe requires one local ActiveSpatial env entry (HTTP rendering is supported)"
        )
    env_config = source_env["envs"][0]["config"]
    contract_path = Path(
        getattr(args, "dataset_contract", None)
        or env_config.get("dataset_contract_path")
        or str(Path(args.tasks).resolve()) + ".contract.json"
    ).resolve()
    validate_contract(rows, ActiveSpatialEnvConfig(**env_config), contract_path)
    env_config.update(
        require_verified_dataset=True, dataset_contract_path=str(contract_path)
    )
    cfg = yaml.safe_load(Path(args.template).read_text())
    if "defaults" in cfg or "actor_rollout_ref" not in cfg:
        raise ValueError(
            "Pass a complete saved/resolved PPO config, not a Hydra defaults fragment"
        )
    if (
        cfg["trainer"]["concat_multi_turn"]
        or cfg["algorithm"]["adv_estimator"] != "no_concat_gae"
    ):
        raise ValueError(
            "R1 warm-start recipe requires no_concat_gae and concat_multi_turn=false"
        )
    partitions = {
        "train": [r for r in rows if str(r["scene_id"]) in train_scenes],
        "val": [r for r in rows if str(r["scene_id"]) in val_scenes],
    }
    if test_scenes:
        partitions["test"] = [r for r in rows if str(r["scene_id"]) in test_scenes]
    if not all(partitions.values()):
        raise ValueError("Empty task partition")
    cfg["actor_rollout_ref"]["model"].update(
        path=str(model), tokenizer_path=None, hf_config_path=None
    )
    cfg["critic"]["model"].update(path=str(model), tokenizer_path=str(model))
    cfg["trainer"].update(
        total_training_steps=args.steps,
        experiment_name=out.name,
        default_local_dir=str(out / "checkpoints"),
        default_hdfs_dir=None,
        resume_mode="disable",
        resume_from_path=None,
    )
    cfg["actor_rollout_ref"]["actor"]["optim"]["total_training_steps"] = args.steps
    cfg["critic"]["optim"]["total_training_steps"] = args.steps
    out.mkdir(parents=True)
    for name, selected in partitions.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in selected))
        env = copy.deepcopy(source_env)
        entry = env["envs"][0]
        entry.update(
            n_envs=len(selected),
            seed=[0, len(selected)],
            seed_list=list(range(len(selected))),
        )
        entry["config"].update(
            jsonl_path=str(path),
            train_size=len(selected),
            test_size=0,
            prompt_format=split["trajectory_prompt_format"],
        )
        env_path = out / f"{name}_env.yaml"
        env_path.write_text(yaml.safe_dump(env, sort_keys=False))
        if name in ("train", "val"):
            cfg["data"][f"{name}_files"] = str(env_path)
    (out / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = {
        "model": str(model),
        "counts": {k: len(v) for k, v in partitions.items()},
        "steps": args.steps,
        "inputs": {
            str(Path(p).resolve()): file_hash(p)
            for p in (
                args.template,
                args.tasks,
                args.env_yaml,
                args.split_manifest,
                contract_path,
            )
        },
        "note": "New PPO run and optimizer schedule; do not launch with the historical fixed-endpoint pilot wrapper.",
    }
    (out / "handoff.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out / "train.yaml"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", required=True, help="Complete saved R1 PPO config")
    p.add_argument(
        "--env-yaml",
        required=True,
        help="Same authoritative protocol as SFT generation",
    )
    p.add_argument(
        "--tasks",
        required=True,
        help="Original pipeline task JSONL, not SFT conversations",
    )
    p.add_argument(
        "--split-manifest", required=True, help="Mixed-SFT scene split manifest"
    )
    p.add_argument("--task-registry", help="Raw task registry for release-v2 splits")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--dataset-contract",
        help="Passing runtime replay certificate; defaults to env config or TASKS.contract.json",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=150)
    print(prepare(p.parse_args()))
