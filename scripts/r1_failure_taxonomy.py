#!/usr/bin/env python3
"""Normalize legacy/new R1 failures into the frozen Projective taxonomy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from r1_full_regeneration import atomic_json, atomic_jsonl, read_jsonl


def classify(row: dict[str, Any]) -> str:
    explicit = row.get("failure_taxonomy") or row.get("repair", {}).get("failure_taxonomy")
    if explicit:
        return str(explicit)
    reason = str(row.get("failure") or row.get("repair", {}).get("failure") or "")
    if reason == "bounded_layout_candidates_exhausted":
        return "search insufficient"
    if "margin_infeasible" in reason or "geometry_constraint" in reason:
        return "source object pair semantically infeasible"
    if "collision" in reason or "safe_unsuccessful_initial" in reason:
        return "layout/collision infeasible"
    if reason == "reachability_unverified_expansion_cap":
        return "planner budget unverified"
    if reason.startswith("unreachable"):
        return "12-step unreachable"
    if "observability" in reason:
        return "observability failure"
    return "search insufficient"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for split_dir in sorted(path for path in args.aggregate_dir.iterdir() if path.is_dir()):
        accounting = split_dir / "accounting.jsonl"
        if not accounting.is_file():
            continue
        for row in read_jsonl(accounting):
            if row.get("task_type") != "projective_relations":
                continue
            if row.get("status") not in {"hard_failure", "unverified", "observability_failure"}:
                continue
            rows.append({
                "split": split_dir.name,
                "source_row_index": row.get("source_row_index"),
                "scene_id": row.get("scene_id"),
                "status": row.get("status"),
                "raw_failure": row.get("failure") or row.get("repair", {}).get("failure"),
                "taxonomy": classify(row),
            })
    summary = {
        "rows": len(rows),
        "taxonomy_counts": dict(Counter(row["taxonomy"] for row in rows)),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "scene_counts": dict(Counter(row["scene_id"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "legacy_caveat": "Legacy bounded search exhaustion is search-insufficient, not proof of layout or semantic infeasibility.",
    }
    atomic_jsonl(args.output_dir / "failure_taxonomy_manifest.jsonl", rows)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
