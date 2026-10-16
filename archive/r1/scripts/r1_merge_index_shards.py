#!/usr/bin/env python3
"""Merge deterministic source-index shards for one scene/split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from r1_full_regeneration import (
    TARGET_TASKS, atomic_json, atomic_jsonl, read_jsonl, sha256, tier_summary,
)


def index_of(row):
    value = row.get("source_row_index")
    if value is None:
        value = row.get("repair_lineage", {}).get("source_row_index")
    return int(value)


def reject_duplicate_replacement_donors(merged):
    """Make the per-split one-use donor invariant explicit across index shards.

    Index shards cannot share the in-memory ``used_replacement_pairs`` set.  Keep
    the first source-index use deterministically and turn every later use into an
    auditable hard failure instead of silently emitting duplicated replacements.
    """
    seen = {}
    conflicts = []
    rejected_indices = set()
    for row in sorted(merged["accounting.jsonl"], key=index_of):
        if row.get("status") != "count_matched_replacement":
            continue
        lineage = row.get("replacement_lineage") or {}
        new_pair = lineage.get("new_pair")
        if not new_pair:
            raise RuntimeError(
                f"replacement missing new_pair lineage at source index {index_of(row)}"
            )
        donor_key = tuple(new_pair)
        if donor_key not in seen:
            seen[donor_key] = index_of(row)
            continue
        source_index = index_of(row)
        kept_index = seen[donor_key]
        conflicts.append({
            "source_row_index": source_index,
            "kept_source_row_index": kept_index,
            "new_pair": list(new_pair),
            "replacement_lineage": lineage,
        })
        rejected_indices.add(source_index)
        row.update({
            "status": "hard_failure",
            "failure": "duplicate_replacement_donor_across_index_shards",
            "failure_taxonomy": "search insufficient",
            "repair_status_before_merge": "count_matched_replacement",
            "duplicate_replacement_donor": {
                "kept_source_row_index": kept_index,
                "new_pair": list(new_pair),
            },
        })
    if rejected_indices:
        for name in ("trainable.jsonl", "mapping.jsonl"):
            merged[name] = [
                row for row in merged[name] if index_of(row) not in rejected_indices
            ]
    return conflicts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--part-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.sources.read_text())
    source_path = Path(paths[args.split])
    source_rows = read_jsonl(source_path)
    expected = {
        index for index, row in enumerate(source_rows)
        if row.get("task_type") in TARGET_TASKS and str(row.get("scene_id")) == args.scene
    }
    names = (
        "accounting.jsonl", "trainable.jsonl", "mapping.jsonl",
        "reachability_manifest.jsonl", "candidate_manifest.jsonl",
    )
    merged = {name: [] for name in names}
    for part in range(args.part_count):
        split_dir = args.parts_dir / f"part_{part:02d}" / args.split
        if not (split_dir / "summary.json").is_file():
            raise RuntimeError(f"part incomplete: {split_dir}")
        for name in names:
            merged[name].extend(read_jsonl(split_dir / name))
    accounting_indices = [index_of(row) for row in merged["accounting.jsonl"]]
    if set(accounting_indices) != expected or len(accounting_indices) != len(set(accounting_indices)):
        raise RuntimeError("index shard accounting does not equal expected source set")
    replacement_conflicts = reject_duplicate_replacement_donors(merged)
    accepted = {
        index_of(row) for row in merged["accounting.jsonl"]
        if row.get("status") in {"strict_same_pair_repair", "count_matched_replacement"}
    }
    for name in ("trainable.jsonl", "mapping.jsonl"):
        indices = [index_of(row) for row in merged[name]]
        if set(indices) != accepted or len(indices) != len(set(indices)):
            raise RuntimeError(f"{name} does not equal accepted source set")
    for name, rows in merged.items():
        rows.sort(key=index_of)
        atomic_jsonl(args.output_dir / name, rows)
    accounting = merged["accounting.jsonl"]
    failures = [row for row in accounting if row.get("status") == "hard_failure"]
    unverified = [row for row in accounting if row.get("status") == "unverified"]
    atomic_jsonl(args.output_dir / "failure_manifest.jsonl", failures)
    atomic_jsonl(args.output_dir / "unverified_manifest.jsonl", unverified)
    atomic_jsonl(
        args.output_dir / "duplicate_replacement_donor_conflicts.jsonl",
        replacement_conflicts,
    )
    counts = Counter(row["status"] for row in accounting)
    paths_found = [
        int(row["found_path_length_upper_bound"]) for row in accounting
        if row.get("found_path_length_upper_bound") is not None
    ]
    reachability = merged["reachability_manifest.jsonl"]
    summary = {
        "split": args.split, "source": str(source_path), "source_sha256": sha256(source_path),
        "source_target_rows_in_scope": len(expected), "accounted_target_rows": len(accounting),
        "accounting_closed": len(accounting) == len(expected), "status_counts": dict(counts),
        "found_path_length_upper_bound": {
            "min": min(paths_found) if paths_found else None,
            "max": max(paths_found) if paths_found else None,
            "distribution": dict(Counter(paths_found)),
        },
        "reachability_tier_results": tier_summary(reachability, [2000, 25000, 250000]),
        "output_rows": len(merged["trainable.jsonl"]),
        "output": str(args.output_dir / "trainable.jsonl"),
        "output_sha256": sha256(args.output_dir / "trainable.jsonl"),
        "index_shards_merged": args.part_count,
        "duplicate_replacement_donor_conflicts_rejected": len(replacement_conflicts),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
