#!/usr/bin/env python3
"""Evaluate raw action errors on held-out episodes using TRAINING normalization.

Requires a checkpoint trained with prepare_starvla_transfer.py's split manifest.
The old all-episode NavigateKitchen checkpoint cannot be scored as held-out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/starVLA"))

from vagen.envs.robocasa.starvla_contract import PolicyContract, checkpoint_config
from vagen.envs.robocasa.utils.actions import GYM_SLICES, ROBOCASA_LEROBOT_SLICES


def frame_at(path, frame):
    import cv2

    reader = cv2.VideoCapture(str(path))
    try:
        reader.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = reader.read()
        if not ok:
            raise RuntimeError(f"Failed to decode {path} frame {frame}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        reader.release()


def samples(dataset, contract, max_episodes, frames_per_episode, seed):
    import pyarrow.parquet as pq

    source = Path(dataset["source"])
    info = json.loads((source / "meta/info.json").read_text())
    modality = json.loads((source / "meta/modality.json").read_text())
    tasks = {r["task_index"]: r["task"] for r in map(json.loads, (source / "meta/tasks.jsonl").read_text().splitlines())}
    ids = list(dataset["val_ids"])
    random.Random(seed).shuffle(ids)
    for episode in ids[:max_episodes]:
        args = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"]}
        table = pq.read_table(source / info["data_path"].format(**args))
        if len(table) < contract.horizon:
            continue
        frames = np.unique(np.linspace(0, len(table) - contract.horizon, frames_per_episode, dtype=int))
        for frame in frames:
            row = table.slice(int(frame), 1).to_pylist()[0]
            obs = {"annotation.human.task_description": tasks[int(row["annotation.human.task_description"])]}
            for camera in contract.cameras:
                key = modality["video"][camera.split(".", 1)[1]]["original_key"]
                obs[camera] = frame_at(source / info["video_path"].format(**args, video_key=key), int(frame))
            if contract.include_state:
                for key, spec in modality["state"].items():
                    obs["state." + key] = np.asarray(row[spec["original_key"]])[spec["start"]:spec["end"]]
            packed = np.asarray(table.slice(int(frame), contract.horizon)["action"].to_pylist(), dtype=np.float32)
            gold = np.concatenate([packed[:, slice(*ROBOCASA_LEROBOT_SLICES[key.split(".", 1)[1]])] for key in GYM_SLICES], axis=-1)
            if gold.shape != (contract.horizon, 12) or not np.isfinite(gold).all():
                raise ValueError(f"Invalid action labels: {source}, episode {episode}, frame {frame}")
            yield episode, int(frame), contract.example(obs), gold


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes-per-task", type=int, default=20)
    p.add_argument("--frames-per-episode", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.ckpt = args.ckpt.resolve()
    args.out = args.out.resolve()
    if min(args.episodes_per_task, args.frames_per_episode) <= 0:
        p.error("Sample counts must be positive")
    cfg, _ = checkpoint_config(args.ckpt)
    contract = PolicyContract.from_config(cfg)
    manifest_path = Path(cfg["transfer"]["split_manifest"])
    manifest = json.loads(manifest_path.read_text())
    data = cfg["datasets"]["vla_data"]
    expected = [[d["name"] + "/lerobot", 1.0, "panda_omron_robocasa365"] for d in manifest["datasets"]]
    if data.get("mixture_spec") != expected:
        raise ValueError("Checkpoint training mixture does not match the held-out manifest")
    for d in manifest["datasets"]:
        if set(d["train_ids"]) & set(d["val_ids"]):
            raise ValueError("Train/validation episode overlap")
        train_root = Path(data["data_root_dir"]) / d["name"] / "lerobot"
        ids = {int(p.stem.split("_")[-1]) for p in train_root.glob("data/*/*.parquet")}
        if ids != set(d["train_ids"]):
            raise ValueError("Training data contains files outside the frozen training split")
    args.out.mkdir(parents=True, exist_ok=False)
    # Some StarVLA modules use paths relative to the repository.
    os.chdir(ROOT / "third_party/starVLA")
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    policy = PolicyServerWrapper(str(args.ckpt.resolve()), device="cuda", use_bf16=True)
    report = {"ckpt": str(args.ckpt), "manifest": str(manifest_path), "seed": args.seed,
              "metric_space": "unnormalized controller actions, complete 16-step chunks",
              "per_task": {}, "samples": []}
    for dataset in manifest["datasets"]:
        errors = []
        for episode, frame, example, gold in samples(dataset, contract, args.episodes_per_task, args.frames_per_episode, args.seed):
            prediction = contract.actions(policy.predict_action([example])["actions"])
            error = prediction - gold
            errors.append(error)
            report["samples"].append({"task": dataset["name"], "episode": episode, "frame": frame,
                                      "mae": float(np.abs(error).mean())})
        if not errors:
            raise ValueError(f"No complete validation chunks for {dataset['name']}")
        error = np.stack(errors)
        report["per_task"][dataset["name"]] = {
            "n_chunks": len(errors), "mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean()),
            "mae_per_dimension": np.abs(error).mean(axis=(0, 1)).tolist(),
            "mae_per_action_group": {k: float(np.abs(error[..., start:end]).mean()) for k, (start, end) in GYM_SLICES.items()},
        }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["per_task"], indent=2))


if __name__ == "__main__":
    main()
