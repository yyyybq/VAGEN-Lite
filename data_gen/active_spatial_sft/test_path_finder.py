#!/usr/bin/env python3
"""
Quick sanity-check for the SFT data generation pipeline.

Runs path-finding only (no rendering) on a handful of items from a pipeline
JSONL to verify that trajectories are found and have the expected structure.

Usage
-----
    python test_path_finder.py \\
        --jsonl_path /path/to/pipeline_output.jsonl \\
        --num_items 5 \\
        --verbose
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VAGEN_ROOT = _HERE.parent.parent
for p in [str(_VAGEN_ROOT), str(_HERE.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from active_spatial_sft.path_finder import find_trajectory
from active_spatial_sft.sft_formatter import format_trajectory
from active_spatial_sft.config import SFTGenerationConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl_path", required=True)
    p.add_argument("--num_items", type=int, default=5)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--success_threshold", type=float, default=0.95)
    args = p.parse_args()

    items = []
    with open(args.jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
                if len(items) >= args.num_items:
                    break

    print(f"Loaded {len(items)} items from {args.jsonl_path}")

    import numpy as np

    successes = 0
    for idx, item in enumerate(items):
        E = np.array(item["init_camera"]["extrinsics"], dtype=np.float64)
        task_type = item.get("task_type", "absolute_positioning")
        task_params = item.get("task_params", {})
        target_region = item.get("target_region", {})

        if not target_region:
            print(f"  Item {idx}: no target_region – skip")
            continue

        traj = find_trajectory(
            init_c2w=E,
            task_type=task_type,
            task_params=task_params,
            target_region=target_region,
            success_threshold=args.success_threshold,
            verbose=args.verbose,
            item_idx=idx,
            scene_id=item.get("scene_id", ""),
        )

        status = "SUCCESS" if traj.success else "PARTIAL"
        print(
            f"  Item {idx} [{status}]: "
            f"steps={len(traj.steps)}, actions={traj.total_actions}, "
            f"score {traj.initial_score:.4f} → {traj.final_score:.4f}"
        )
        if traj.success:
            successes += 1

        # Test formatter (no real images)
        record = format_trajectory(
            item=item,
            trajectory=traj,
            image_paths=[f"images/sft_{idx:06d}_step{s:02d}.jpg"
                         for s in range(len(traj.steps) + 1)],
            sft_id=f"sft_{idx:06d}",
        )
        n_turns = len([t for t in record["conversations"] if t["role"] == "assistant"])
        print(f"    → {n_turns} assistant turns in conversation")

    print(f"\nSummary: {successes}/{len(items)} trajectories succeeded.")


if __name__ == "__main__":
    main()
