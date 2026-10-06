#!/usr/bin/env python3
"""Load one NavigateKitchen batch and assert action_dim=12."""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--data_root_dir", default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_yaml)
    if args.data_root_dir:
        cfg.datasets.vla_data.data_root_dir = Path(args.data_root_dir)
    ds = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
    print("[dataloader] len", len(ds))
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    ex = batch[0]
    act = ex["action"]
    print("[dataloader] keys", sorted(ex.keys()))
    print("[dataloader] action shape", getattr(act, "shape", None))
    print("[dataloader] lang", ex.get("lang"))
    print("[dataloader] n_images", len(ex.get("image") or []))
    assert act.shape[-1] == 12, act.shape
    print("[dataloader] ok action_dim=12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
