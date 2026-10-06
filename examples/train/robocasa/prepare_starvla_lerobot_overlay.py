#!/usr/bin/env python3
"""Build a writable StarVLA overlay on top of an existing RoboCasa LeRobot dump.

StarVLA writes ``meta/stats_gr00t.json`` and ``meta/steps_data_index.pkl``.
Keep the original dump intact by symlink-ing ``data/`` and ``videos/``, copying
``meta/`` (LeRobot v2.1: info.json, tasks.jsonl, modality.json, ...), and
leaving cache files writable in the overlay.

Default source layout (unchanged):

    <src-root>/v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot

Default overlay:

    <dst-root>/v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_RELPATH = "v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot"
META_FILES = (
    "info.json",
    "modality.json",
    "stats.json",
    "tasks.jsonl",
    "tasks.parquet",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "embodiment.json",
)


def is_lerobot_v2(root: Path) -> bool:
    return (root / "meta" / "info.json").is_file() and (root / "data").is_dir()


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def build_overlay(src_root: Path, dst_root: Path, relpath: str) -> Path:
    src = (src_root / relpath).resolve()
    dst = (dst_root / relpath).resolve()
    if not is_lerobot_v2(src):
        raise FileNotFoundError(
            f"LeRobot v2 root not found at {src} "
            "(need meta/info.json and data/)"
        )
    if src == dst:
        print(f"[overlay] src==dst, keep in-repo tree: {dst}")
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    _link_or_copy(src / "data", dst / "data", copy=False)
    if (src / "videos").is_dir():
        _link_or_copy(src / "videos", dst / "videos", copy=False)

    meta_src = src / "meta"
    meta_dst = dst / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    for name in META_FILES:
        src_file = meta_src / name
        if not src_file.exists():
            continue
        dst_file = meta_dst / name
        if dst_file.exists() or dst_file.is_symlink():
            dst_file.unlink()
        # Copy meta so the loader can write sibling cache files.
        shutil.copy2(src_file, dst_file)
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src-root",
        type=Path,
        default=Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/playground/Datasets/robocasa365"),
        help="Existing RoboCasa365 datasets root (contains v1.0/)",
    )
    p.add_argument(
        "--dst-root",
        type=Path,
        default=None,
        help="Writable overlay root (StarVLA data_root_dir)",
    )
    p.add_argument(
        "--relpath",
        default=DEFAULT_RELPATH,
        help="Relative LeRobot path under src/dst roots",
    )
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[3]
    dst_root = args.dst_root or (repo / "playground" / "Datasets" / "robocasa365")
    dst = build_overlay(args.src_root, dst_root, args.relpath)
    print(f"[overlay] src={args.src_root / args.relpath}")
    print(f"[overlay] dst={dst}")
    print(f"[overlay] data_root_dir={dst_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
