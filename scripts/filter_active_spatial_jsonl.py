#!/usr/bin/env python3
"""
Filter Active Spatial JSONL files by task_type.

Example:
    python scripts/filter_active_spatial_jsonl.py \
      --input data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl \
      --output data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_no_delta.jsonl \
      --exclude delta_control
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def task_type_of(item: Dict[str, Any]) -> str:
    task_type = item.get("task_type")
    if task_type:
        return str(task_type)
    desc = str(item.get("task_description", "")).lower()
    if "closer" in desc or "farther" in desc:
        return "delta_control"
    if "appears larger" in desc or "looks bigger" in desc:
        return "apparent_size_ordering"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter Active Spatial JSONL by task_type.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--include", nargs="*", default=None, help="Optional task_type allow-list.")
    parser.add_argument("--exclude", nargs="*", default=["delta_control"], help="Task types to drop.")
    args = parser.parse_args()

    include = set(args.include or [])
    exclude = set(args.exclude or [])
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    kept_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()

    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            item = json.loads(line)
            task_type = task_type_of(item)
            allowed = True
            if include and task_type not in include:
                allowed = False
            if exclude and task_type in exclude:
                allowed = False
            if allowed:
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
                kept_counts[task_type] += 1
            else:
                dropped += 1
                dropped_counts[task_type] += 1

    print(f"Input:   {in_path}")
    print(f"Output:  {out_path}")
    print(f"Kept:    {kept}  {dict(sorted(kept_counts.items()))}")
    print(f"Dropped: {dropped}  {dict(sorted(dropped_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
