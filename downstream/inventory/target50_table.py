#!/usr/bin/env python3
"""The RoboCasa365 target-50 task table from the installed RoboCasa registry (CPU, read-only).

Run with the RoboCasa365 simulator Python:  python downstream/inventory/target50_table.py --out target50.tsv
Columns: group (atomic_seen / composite_seen / composite_unseen), task, task-specific horizon (control steps),
target/human dataset path in the registry, and the demo counts the registry's 10% / 30% fractions keep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import robocasa
    from robocasa.utils.dataset_registry import TARGET_TASKS
    from robocasa.utils.dataset_registry_utils import get_ds_meta, get_task_horizon

    rows = []
    for group, tasks in TARGET_TASKS.items():
        for task in tasks:
            full = get_ds_meta(task=task, split="target", source="human") or {}
            frac = {}
            for f in (0.1, 0.3):
                meta = get_ds_meta(task=task, split="target", source="human", demo_fraction=f) or {}
                frac[f] = {k: meta.get(k) for k in ("demo_fraction", "filter_key", "num_demos") if k in meta}
            rows.append({"group": group, "task": task, "horizon": get_task_horizon(task),
                         "path": full.get("path"), "meta_keys": sorted(full), "frac10": frac[0.1], "frac30": frac[0.3]})
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r["horizon"])
    print(json.dumps({"robocasa": robocasa.__version__, "file": robocasa.__file__,
                      "groups": {g: {"tasks": len(h), "horizon_sum": sum(h), "horizon_min": min(h), "horizon_max": max(h)}
                                 for g, h in by.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
