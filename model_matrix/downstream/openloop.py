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

``--ways`` picks the ways (default the three above); the user's decision 36 of 2026-10-30 adds mild perturbations of
every image (PERTURB): re-encoded as JPEG at quality 30 ("jpeg"), a Gaussian blur of radius 1 ("blur"), brightness
x1.15 and contrast x0.85 ("color") -- how much a checkpoint's error grows under small image changes such as those
between the recorded demonstration videos and the simulator's live frames.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

DS_TRAIN = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream/downstream/train")
WAYS = ("images", "black", "shuffled")
PERTURB = ("jpeg", "blur", "color")
ALL_WAYS = WAYS + PERTURB


def perturbed(image, way: str):
    """``image`` (PIL image or array) changed by one of PERTURB, of the same size and kind."""
    from PIL import Image, ImageEnhance, ImageFilter

    array = isinstance(image, np.ndarray)
    im = Image.fromarray(image) if array else image
    if way == "jpeg":
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=30)
        buf.seek(0)
        out = Image.open(buf).convert(im.mode)
        out.load()
    elif way == "blur":
        out = im.filter(ImageFilter.GaussianBlur(radius=1.0))
    elif way == "color":
        out = ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(1.15)).enhance(0.85)
    else:
        raise ValueError(f"unknown perturbation {way}; known {PERTURB}")
    return np.asarray(out, dtype=image.dtype) if array else out


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
    elif way in PERTURB:
        for e in out:
            e["image"] = ([perturbed(im, way) for im in e["image"]] if isinstance(e["image"], list)
                          else perturbed(e["image"], way))
    else:
        raise ValueError(f"unknown way {way}; known {ALL_WAYS}")
    return out


def parse_ways(text: str) -> tuple:
    """--ways: "images" first (the ratios are to it), then any of the others, each once."""
    ways = tuple(w.strip() for w in text.split(",") if w.strip())
    if not ways or ways[0] != "images" or len(set(ways)) != len(ways) or any(w not in ALL_WAYS for w in ways):
        raise ValueError(f"--ways {text!r}: images first, then any of {ALL_WAYS[1:]}, each once")
    return ways


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
    for way in out:
        if way != "images":
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


def eval_data(folder) -> tuple:
    """--eval-data: a folder in the downstream line's prepared layout (downstream/demos/prepare_data.py): the
    data_root_dir (its train/) and the mixture_spec the run's data section is pointed at."""
    folder = Path(folder)
    spec = json.loads((folder / "mixture_spec.json").read_text())
    if not spec or any(len(row) != 3 for row in spec):
        raise ValueError(f"{folder}/mixture_spec.json: expected [[<Task>/lerobot, weight, robot tag], ...]")
    if not (folder / "train").is_dir():
        raise ValueError(f"{folder} has no train/ folder")
    return str(folder / "train"), spec


def build(module, data_cfg):
    return module.get_vla_dataset(data_cfg=data_cfg,
                                  balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
                                  balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False))


def load_samples(config_yaml: str, samples: int, seed: int, other=None):
    """The examples at the chosen positions of the run's training data -- or, with ``other`` (--eval-data), of that
    folder's data normalized with the training data's merged statistics, as the model was trained -- and a digest
    of where they come from."""
    import tempfile

    sys.path.insert(0, str(DS_TRAIN))
    import train_entry as TE  # the downstream line's (read only)

    cfg = TE.load_config(config_yaml, None)
    data_cfg = cfg.datasets.vla_data
    module = TE.seed_dataset(int(cfg.seed))
    dataset = build(module, data_cfg)
    stats = {}
    saved = Path(config_yaml).parent / "dataset_statistics.json"  # what the run saved and the policy server reads
    if saved.is_file():
        tmp = Path(tempfile.mkdtemp()) / "dataset_statistics.json"
        dataset.save_dataset_statistics(tmp)
        stats["training_stats_equal_saved"] = json.loads(tmp.read_text()) == json.loads(saved.read_text())
    if other is not None:
        root, spec = eval_data(other)
        data_cfg.data_root_dir, data_cfg.mixture_spec = root, spec
        merged = dataset.merged_metadata
        dataset = build(module, data_cfg)
        for d in dataset.datasets:  # normalize as in training, not with the other folder's own statistics
            d.set_transforms_metadata(merged[d.tag])
        stats.update(eval_data=str(other), eval_tasks=len(spec), normalized_with="the training data's merged statistics")
    chosen = positions(len(dataset), samples, seed)
    where, examples = [], []
    index = {id(d): i for i, d in enumerate(dataset.datasets)}
    for p in chosen:
        ds, traj, step = dataset.sample_step(p)
        where.append([p, index[id(ds)], int(traj), int(step)])
        seed_all(seed * 1_000_003 + p)  # the transforms' randomness, the same in every invocation
        examples.append(dataset[p])
    digest = hashlib.sha256(json.dumps(where).encode()).hexdigest()[:16]
    return examples, {"length": len(dataset), "positions_sha": digest, "first": where[:4], "dataset_seed": int(cfg.seed),
                      **stats}


def evaluate(ckpt: str, examples: List[dict], batch: int, seed: int, ways=WAYS) -> dict:
    import torch
    from starVLA.model.framework.base_framework import baseframework

    t0 = time.time()
    model = baseframework.from_pretrained(ckpt).to(torch.bfloat16).to("cuda").eval()
    load_s = time.time() - t0
    errors = {way: Errors() for way in ways}
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(examples), batch):
            chunk = examples[b:b + batch]
            target = np.array([e["action"] for e in chunk])
            for way in ways:
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
    ap.add_argument("--eval-data", default=None,
                    help="sample this prepared folder (prepare_data.py layout) instead, normalized with the training "
                         "data's statistics (decision 35: demonstrations from scenes the training never saw)")
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CHECKPOINT.pt (repeatable)")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ways", default=",".join(WAYS),
                    help="comma-separated, images first; known: " + ",".join(ALL_WAYS) +
                         " (decision 36: the perturbations)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ways = parse_ways(a.ways)
    ckpts = dict(c.split("=", 1) for c in a.ckpt)
    examples, data = load_samples(a.data_config, a.samples, a.seed, a.eval_data)
    record = {"data_config": a.data_config, "eval_data": a.eval_data, "samples": a.samples, "batch": a.batch,
              "seed": a.seed, "data": data,
              "ways": list(ways), "checkpoints": {}}
    for label, ckpt in ckpts.items():
        record["checkpoints"][label] = evaluate(ckpt, examples, a.batch, a.seed, ways)
        a.out.write_text(json.dumps(record, indent=1) + "\n")  # after every checkpoint
        r = record["checkpoints"][label]
        print(label, {w: round(r[w]["mse"], 5) for w in ways}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
