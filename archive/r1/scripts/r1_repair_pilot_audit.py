#!/usr/bin/env python3
"""Summarize canonical and scene-constraint gates for an R1 repair pilot."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def describe(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("fov", "projective"), required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mappings = rows(args.mapping)
    failures = rows(args.failures)
    if args.kind == "fov":
        target_success = sum(bool(row["new_target_metric"]["success"]) for row in mappings)
        initial_success = sum(bool(row["new_init_metric"]["success"]) for row in mappings)
        initial_repaired = sum(bool(row.get("initial_position_repaired")) for row in mappings)
        target_margins = [float(row["new_target_metric"]["center_margin_px"]) for row in mappings]
        inside = [float(row["new_target_metric"]["inside_frame_fraction_min"]) for row in mappings]
    else:
        target_success = sum(bool(row["new_metric"]["success"]) for row in mappings)
        initial_success = sum(bool(row["new_initial_metric"]["success"]) for row in mappings)
        initial_repaired = sum(bool(row.get("initial_position_repaired")) for row in mappings)
        target_margins = [float(row["new_metric"]["relation_margin_px"]) for row in mappings]
        inside = []
    translation = []
    for row in mappings:
        old = row.get("old_sample_target") or row.get("old_target_pose_c2w", [[], [], [], []])[0:3]
        new = row.get("new_sample_target") or row.get("new_target_pose_c2w", [[], [], [], []])[0:3]
        if args.kind == "fov":
            old_pose = np.asarray(row["old_target_pose_c2w"], dtype=float)
            new_pose = np.asarray(row["new_target_pose_c2w"], dtype=float)
            translation.append(float(np.linalg.norm(new_pose[:3, 3] - old_pose[:3, 3])))
        elif len(old) >= 3 and len(new) >= 3:
            translation.append(float(np.linalg.norm(np.asarray(new, dtype=float)[:3] - np.asarray(old, dtype=float)[:3])))
    summary = {
        "kind": args.kind,
        "accepted": len(mappings),
        "failures": len(failures),
        "new_target_success": target_success,
        "new_initial_success": initial_success,
        "initial_position_repaired": initial_repaired,
        "initial_constraints_success": sum(bool(row.get("initial_constraints", {}).get("success")) for row in mappings),
        "target_constraints_success": sum(bool(row.get("target_constraints", {}).get("success")) for row in mappings),
        "failure_counts": dict(Counter(row.get("failure", "unknown") for row in failures)),
        "target_margin": describe(target_margins),
        "inside_frame_fraction_min": describe(inside),
        "old_to_new_target_translation": describe(translation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
