#!/usr/bin/env python3
"""Freeze the v3 A/B rows after the independent v2 screen closes."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


VERSION = "r1_independent_holdout_v3_ab_selection_v1"


def digest(row: dict, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{row['split']}:{int(row['source_row_index'])}".encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-selection", type=Path, required=True)
    parser.add_argument("--v2-selector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortcut-min", type=int, default=24)
    parser.add_argument("--shortcut-max", type=int, default=32)
    parser.add_argument("--positive-controls", type=int, default=5)
    args = parser.parse_args()
    holdout = json.loads(args.holdout_selection.read_text())
    v2 = json.loads(args.v2_selector.read_text())
    source = {(row["split"], int(row["source_row_index"])): row for row in holdout["records"]}
    shortcuts = [row for row in v2["results"] if row.get("status") == "shortcut_rejected"]
    shortcuts.sort(key=lambda row: (digest(row, "v3-holdout-shortcut"), row["split"], int(row["source_row_index"])))
    shortcuts = shortcuts[:args.shortcut_max]
    controls_pool = [row for row in v2["results"] if row.get("status") == "difficulty_certified_candidate"]
    controls_pool.sort(key=lambda row: (digest(row, "v3-holdout-positive"), row["split"], int(row["source_row_index"])))
    controls = []
    used_scenes = set()
    for row in controls_pool:
        if row["scene_id"] in used_scenes:
            continue
        controls.append(row); used_scenes.add(row["scene_id"])
        if len(controls) == args.positive_controls:
            break
    if len(controls) < args.positive_controls:
        for row in controls_pool:
            if row in controls:
                continue
            controls.append(row)
            if len(controls) == args.positive_controls:
                break
    records = []
    for kind, rows in (("v2_shortcut_rejected", shortcuts), ("v2_positive_control", controls)):
        for row in rows:
            record = dict(source[(row["split"], int(row["source_row_index"]))])
            record.update({"ab_kind": kind, "v2_status": row["status"],
                           "v2_elapsed_seconds": row.get("elapsed_seconds")})
            records.append(record)
    records.sort(key=lambda row: (row["ab_kind"], row["selection_digest"], row["source_row_index"]))
    payload = {
        "version": VERSION, "selection_is_frozen": True,
        "holdout_selection": {"path": str(args.holdout_selection.resolve()), "sha256": sha256(args.holdout_selection)},
        "v2_selector": {"path": str(args.v2_selector.resolve()), "sha256": sha256(args.v2_selector)},
        "rule": {
            "shortcut": f"stable digest, target {args.shortcut_min}-{args.shortcut_max}; use actual if fewer than minimum",
            "positive_controls": f"stable digest, distinct scenes first, target {args.positive_controls}",
            "no_screen_expansion": True,
        },
        "sources": holdout["sources"], "scenes": holdout["scenes"],
        "shortcut_available_in_128_screen": sum(row.get("status") == "shortcut_rejected" for row in v2["results"]),
        "shortcut_selected": len(shortcuts), "shortcut_min_target_met": len(shortcuts) >= args.shortcut_min,
        "positive_control_selected": len(controls), "record_count": len(records),
        "counts": {"kind": dict(Counter(row["ab_kind"] for row in records)),
                   "scene": dict(Counter(row["scene_id"] for row in records))},
        "frozen_config": holdout["frozen_config"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: payload[key] for key in ("shortcut_available_in_128_screen", "shortcut_selected",
                                                     "positive_control_selected", "record_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
