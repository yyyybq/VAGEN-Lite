#!/usr/bin/env python3
"""Training steps per epoch of a StarVLA config, without building the model (CPU, the StarVLA environment).

    python downstream/train/epoch_steps.py --config EXP/config.yaml [--gpus 8]

Builds only the VLA dataset as the training does (train_entry.seed_dataset, get_vla_dataset) and prints its length,
the global batch (per-device batch x gpus) and batches per epoch = ceil(length / global batch), the
len(vla_train_dataloader) the trainer reports as "batches_per_epoch" (556,601 / 170,084 / 55,436 for target50 100% /
30% / 10%).  X2 (the user's decision of 2026-10-26) sets its steps to 2 x this.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_entry as te  # noqa: E402


def batches_per_epoch(length: int, per_device: int, gpus: int) -> int:
    return math.ceil(length / (per_device * gpus))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    args = ap.parse_args(argv)
    config = str(Path(args.config).resolve())
    os.chdir(te.STARVLA)
    sys.path.insert(0, str(te.STARVLA))
    from starVLA.training.trainer_utils.config_tracker import wrap_config

    cfg = te.load_config(config, [])
    module = te.seed_dataset(int(cfg.seed))
    cfg = wrap_config(cfg)
    data_cfg = cfg.datasets.vla_data
    dataset = module.get_vla_dataset(data_cfg=data_cfg,
                                     balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
                                     balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False))
    per_device = int(data_cfg.per_device_batch_size)
    out = {"config": config, "length": len(dataset), "per_device_batch": per_device, "gpus": args.gpus,
           "batches_per_epoch": batches_per_epoch(len(dataset), per_device, args.gpus)}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
