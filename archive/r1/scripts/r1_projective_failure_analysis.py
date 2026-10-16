#!/usr/bin/env python3
"""Classify unresolved projective rows from a versioned regeneration run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def describe(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "min": min(finite) if finite else None,
        "mean": statistics.mean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def classify_attempts(attempts: list[dict[str, Any]]) -> tuple[str, float | None]:
    usable = []
    for attempt in attempts:
        gates = attempt.get("gates") or {}
        base = all(bool(gates.get(key)) for key in ("two_objects", "in_front", "visible", "min_area", "relation"))
        if base:
            usable.append(attempt)
    if not usable:
        return "projection_or_relation_infeasible", None
    inside = [row for row in usable if bool(row.get("gates", {}).get("inside_frame"))]
    best_inside_margin = max((float(row.get("margin_px", -math.inf)) for row in inside), default=None)
    if inside and best_inside_margin is not None and best_inside_margin < 12.0:
        return "margin_only", best_inside_margin
    margin = [row for row in usable if bool(row.get("gates", {}).get("margin"))]
    if margin and not any(bool(row.get("gates", {}).get("inside_frame")) for row in margin):
        return "inside_frame_vs_margin_conflict", best_inside_margin
    return "mixed_gate_conflict", best_inside_margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = json.loads(args.sources_json.read_text())
    details = []
    for split, source_path in sources.items():
        source_rows = read_jsonl(Path(source_path))
        failure_path = args.input_dir / f"{split}_failures.jsonl"
        for failure in read_jsonl(failure_path):
            index = int(failure["source_row_index"])
            item = source_rows[index]
            params = item["target_region"]["params"]
            a = params["object_a_center"]
            b = params["object_b_center"]
            category, best_margin = classify_attempts(failure.get("attempts") or [])
            objects = (item.get("target_object") or {}).get("objects") or []
            details.append(
                {
                    "split": split,
                    "source_row_index": index,
                    "scene_id": item.get("scene_id"),
                    "relation": params.get("relation"),
                    "object_ids": [obj.get("id") for obj in objects],
                    "object_labels": [obj.get("label") for obj in objects],
                    "pair_xy_separation": math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])),
                    "pair_z_separation": abs(float(b[2]) - float(a[2])),
                    "min_world_distance": float(params.get("min_distance", 0.0) or 0.0),
                    "classification": category,
                    "best_inside_frame_margin_px": best_margin,
                    "attempt_count": len(failure.get("attempts") or []),
                }
            )

    summary = {
        "rows": len(details),
        "split_counts": dict(Counter(row["split"] for row in details)),
        "classification_counts": dict(Counter(row["classification"] for row in details)),
        "relation_counts": dict(Counter(row["relation"] for row in details)),
        "pair_xy_separation": describe([row["pair_xy_separation"] for row in details]),
        "pair_z_separation": describe([row["pair_z_separation"] for row in details]),
        "best_inside_frame_margin_px": describe(
            [row["best_inside_frame_margin_px"] for row in details if row["best_inside_frame_margin_px"] is not None]
        ),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
