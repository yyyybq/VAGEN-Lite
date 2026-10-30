"""Open-loop action error of downstream checkpoints, with the images as they are and without them (the user's
decision 34 of 2026-10-30): DS-CSS and DS-CSRS at 150k fit their training data better than DS-CSB (lower
action_dit_loss and mse_score) yet succeed 5-10 times less often in closed loop.  If a checkpoint's error hardly
changes when its images are taken away, it barely looks at them.

    python [model_matrix/downstream/cambrian_entry.py] model_matrix/downstream/openloop.py --data-config RUN/config.yaml \
        --ckpt LABEL=CKPT [--ckpt LABEL=CKPT ...] [--samples 256] [--batch 8] [--seed 1] --out OUT.json

Run from StarVLA's root in its env, as the policy server (model_matrix/downstream/openloop.sh).  Every checkpoint is
loaded as the policy server loads it (baseframework.from_pretrained: its run's config.yaml and statistics, the state
dict strictly; bf16, eval mode).  The samples are drawn once from the training data of --data-config (the downstream
line's get_vla_dataset with that run's seed, read only, by path): --samples dataset positions chosen with --seed, the
same positions -- hence the same samples -- in every invocation.  Each batch is predicted three ways with the same
diffusion noise (torch seeded per batch): the images as they are ("images"), every image black at its own size
("black"), and the batch's images moved one sample on ("shuffled": real images of another sample).  Per checkpoint
and way: the mean squared error of the normalized actions (mse) and the trainer's mse_score form (the batch's
Euclidean distance over its number of values, averaged over batches).  Writes OUT.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

DS_TRAIN = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream/downstream/train")
WAYS = ("images", "black", "shuffled")


def blank_like(image):
    """A black image of the same size and kind (PIL image or array)."""
    if isinstance(image, np.ndarray):
        return np.zeros_like(image)
    from PIL import Image

    return Image.new(image.mode, image.size)


def with_images(examples: List[dict], way: str) -> List[dict]:
    """The batch for one way; the originals are left as they are."""
    if way == "images":
        return examples
    out = [copy.copy(e) for e in examples]
    if way == "black":
        for e in out:
            e["image"] = [blank_like(im) for im in e["image"]] if isinstance(e["image"], list) else blank_like(e["image"])
    elif way == "shuffled":
        if len(out) < 2:
            raise ValueError("shuffled needs at least two samples in a batch")
        for i, e in enumerate(out):
            e["image"] = examples[(i + 1) % len(examples)]["image"]
    else:
        raise ValueError(f"unknown way {way}; known {WAYS}")
    return out


def positions(length: int, samples: int, seed: int) -> List[int]:
    """--samples distinct dataset positions, sorted, from --seed."""
    if samples > length:
        raise ValueError(f"{samples} samples from a dataset of {length}")
    return sorted(int(i) for i in np.random.default_rng(seed).choice(length, samples, replace=False))


class Errors:
    """Running sums for one checkpoint and way."""

    def __init__(self):
        self.squared, self.values, self.scores = 0.0, 0, []

    def add(self, predicted: np.ndarray, target: np.ndarray) -> None:
        predicted, target = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if predicted.shape != target.shape:
            raise ValueError(f"predicted {predicted.shape} against target {target.shape}")
        self.squared += float(((predicted - target) ** 2).sum())
        self.values += int(target.size)
        self.scores.append(float(np.linalg.norm(predicted - target)) / target.size)  # the trainer's mse_score

    def summary(self) -> Dict[str, float]:
        return {"mse": self.squared / self.values, "mse_score": float(np.mean(self.scores)),
                "values": self.values, "batches": len(self.scores)}


def summarize(errors: Dict[str, Errors]) -> dict:
    out = {way: e.summary() for way, e in errors.items()}
    base = out["images"]["mse"]
    for way in WAYS[1:]:
        if way in out:
            out[way]["ratio_to_images"] = out[way]["mse"] / base if base else None
    return out


def seed_all(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_samples(config_yaml: str, samples: int, seed: int):
    """The examples at the chosen positions of the run's training data, and a digest of where they come from."""
    sys.path.insert(0, str(DS_TRAIN))
    import train_entry as TE  # the downstream line's (read only)

    cfg = TE.load_config(config_yaml, None)
    data_cfg = cfg.datasets.vla_data
    module = TE.seed_dataset(int(cfg.seed))
    dataset = module.get_vla_dataset(data_cfg=data_cfg,
                                     balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
                                     balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False))
    chosen = positions(len(dataset), samples, seed)
    where, examples = [], []
    index = {id(d): i for i, d in enumerate(dataset.datasets)}
    for p in chosen:
        ds, traj, step = dataset.sample_step(p)
        where.append([p, index[id(ds)], int(traj), int(step)])
        seed_all(seed * 1_000_003 + p)  # the transforms' randomness, the same in every invocation
        examples.append(dataset[p])
    digest = hashlib.sha256(json.dumps(where).encode()).hexdigest()[:16]
    return examples, {"length": len(dataset), "positions_sha": digest, "first": where[:4], "dataset_seed": int(cfg.seed)}


def evaluate(ckpt: str, examples: List[dict], batch: int, seed: int) -> dict:
    import torch
    from starVLA.model.framework.base_framework import baseframework

    t0 = time.time()
    model = baseframework.from_pretrained(ckpt).to(torch.bfloat16).to("cuda").eval()
    load_s = time.time() - t0
    errors = {way: Errors() for way in WAYS}
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(examples), batch):
            chunk = examples[b:b + batch]
            target = np.array([e["action"] for e in chunk])
            for way in WAYS:
                seed_all(seed * 7919 + b)  # the same diffusion noise for every way and checkpoint
                out = model.predict_action(examples=with_images(chunk, way), use_ddim=True, num_ddim_steps=20)
                errors[way].add(out["normalized_actions"], target)
    result = {"ckpt": ckpt, "load_seconds": round(load_s, 1), "predict_seconds": round(time.time() - t0, 1),
              **summarize(errors)}
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", required=True, help="the run config.yaml whose training data are sampled")
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CHECKPOINT.pt (repeatable)")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ckpts = dict(c.split("=", 1) for c in a.ckpt)
    examples, data = load_samples(a.data_config, a.samples, a.seed)
    record = {"data_config": a.data_config, "samples": a.samples, "batch": a.batch, "seed": a.seed, "data": data,
              "ways": list(WAYS), "checkpoints": {}}
    for label, ckpt in ckpts.items():
        record["checkpoints"][label] = evaluate(ckpt, examples, a.batch, a.seed)
        a.out.write_text(json.dumps(record, indent=1) + "\n")  # after every checkpoint
        r = record["checkpoints"][label]
        print(label, {w: round(r[w]["mse"], 5) for w in WAYS}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
