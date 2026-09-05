#!/usr/bin/env python3
"""Merge scene-sharded R1 canary outputs and prove source-row accounting closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from r1_full_regeneration import TARGET_TASKS, atomic_json, atomic_jsonl, read_jsonl, sha256


def source_index(row: dict[str, Any]) -> int:
    value = row.get("source_row_index")
    if value is None:
        value = row.get("repair_lineage", {}).get("source_row_index")
    if value is None:
        raise RuntimeError(f"row has no source index: {row.get('task_id')}")
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", required=True, help="comma-separated scene IDs")
    args = parser.parse_args()

    scenes = [value for value in args.scenes.split(",") if value]
    if len(scenes) != len(set(scenes)):
        raise RuntimeError("duplicate scene IDs")
    raw_sources = json.loads(args.sources.read_text())
    sources = {
        split: (path if (path := Path(value)).is_absolute() else Path.cwd() / path)
        for split, value in raw_sources.items()
    }
    missing = [scene for scene in scenes if not (args.shards_dir / scene / "summary.json").is_file()]
    if missing:
        raise RuntimeError(f"incomplete shards: {missing}")

    total_counts: Counter[str] = Counter()
    task_status: dict[str, Counter[str]] = defaultdict(Counter)
    scene_status: dict[str, Counter[str]] = defaultdict(Counter)
    split_summaries: dict[str, Any] = {}
    expected_total = 0

    for split, source_path in sources.items():
        source_rows = read_jsonl(source_path)
        expected = {
            index
            for index, row in enumerate(source_rows)
            if row.get("task_type") in TARGET_TASKS and str(row.get("scene_id")) in scenes
        }
        expected_total += len(expected)
        accounting: list[dict[str, Any]] = []
        repaired: list[dict[str, Any]] = []
        mappings: list[dict[str, Any]] = []
        reachability: list[dict[str, Any]] = []
        for scene in scenes:
            split_dir = args.shards_dir / scene / split
            accounting.extend(read_jsonl(split_dir / "accounting.jsonl"))
            repaired.extend(read_jsonl(split_dir / "trainable.jsonl"))
            mappings.extend(read_jsonl(split_dir / "mapping.jsonl"))
            reachability.extend(read_jsonl(split_dir / "reachability_manifest.jsonl"))

        actual_indices = [source_index(row) for row in accounting]
        duplicates = sorted(index for index, count in Counter(actual_indices).items() if count != 1)
        if duplicates:
            raise RuntimeError(f"{split}: duplicate accounting indices: {duplicates[:20]}")
        if set(actual_indices) != expected:
            missing_indices = sorted(expected - set(actual_indices))
            extra_indices = sorted(set(actual_indices) - expected)
            raise RuntimeError(
                f"{split}: accounting mismatch missing={missing_indices[:20]} extra={extra_indices[:20]}"
            )
        repaired_indices = [source_index(row) for row in repaired]
        mapping_indices = [source_index(row) for row in mappings]
        accepted = {
            source_index(row)
            for row in accounting
            if row.get("status") in {"strict_same_pair_repair", "count_matched_replacement"}
        }
        if set(repaired_indices) != accepted or set(mapping_indices) != accepted:
            raise RuntimeError(f"{split}: accepted/mapping/repaired mismatch")
        if len(repaired_indices) != len(set(repaired_indices)) or len(mapping_indices) != len(set(mapping_indices)):
            raise RuntimeError(f"{split}: duplicate accepted output")

        accounting.sort(key=source_index)
        repaired.sort(key=source_index)
        mappings.sort(key=source_index)
        reachability.sort(key=source_index)
        failures = [row for row in accounting if row.get("status") == "hard_failure"]
        unverified = [row for row in accounting if row.get("status") == "unverified"]
        output_dir = args.output_dir / split
        atomic_jsonl(output_dir / "accounting.jsonl", accounting)
        atomic_jsonl(output_dir / "trainable.jsonl", repaired)
        atomic_jsonl(output_dir / "mapping.jsonl", mappings)
        atomic_jsonl(output_dir / "reachability_manifest.jsonl", reachability)
        atomic_jsonl(output_dir / "failure_manifest.jsonl", failures)
        atomic_jsonl(output_dir / "unverified_manifest.jsonl", unverified)

        counts = Counter(row["status"] for row in accounting)
        paths = [
            int(row["found_path_length_upper_bound"])
            for row in accounting
            if row.get("found_path_length_upper_bound") is not None
        ]
        for row in accounting:
            task_status[str(row.get("task_type"))][str(row["status"])] += 1
            scene_status[str(row.get("scene_id"))][str(row["status"])] += 1
        total_counts.update(counts)
        summary = {
            "split": split,
            "source": str(source_path),
            "source_sha256": sha256(source_path),
            "source_target_rows_in_scope": len(expected),
            "accounted_target_rows": len(accounting),
            "accounting_closed": len(accounting) == len(expected),
            "status_counts": dict(counts),
            "found_path_length_upper_bound": {
                "min": min(paths) if paths else None,
                "max": max(paths) if paths else None,
                "distribution": dict(sorted(Counter(paths).items())),
            },
            "output_rows": len(repaired),
            "output": str(output_dir / "trainable.jsonl"),
            "output_sha256": sha256(output_dir / "trainable.jsonl"),
        }
        atomic_json(output_dir / "summary.json", summary)
        split_summaries[split] = summary

    aggregate = {
        "scenes": scenes,
        "shards_dir": str(args.shards_dir),
        "expected_target_rows": expected_total,
        "accounted_target_rows": sum(total_counts.values()),
        "all_accounting_closed": expected_total == sum(total_counts.values()),
        "status_counts": dict(total_counts),
        "task_status_counts": {task: dict(counts) for task, counts in sorted(task_status.items())},
        "scene_status_counts": {scene: dict(counts) for scene, counts in sorted(scene_status.items())},
        "splits": split_summaries,
        "reachability_search": {
            "algorithm": "bounded_best_first_not_shortest",
            "max_steps": 12,
            "max_expansions": 2000,
            "budget_exhaustion_semantics": "unverified_not_unreachable",
        },
        "gate_pass": total_counts.get("hard_failure", 0) == 0 and total_counts.get("unverified", 0) == 0,
    }
    canonical = json.dumps(aggregate, sort_keys=True, separators=(",", ":")).encode()
    aggregate["summary_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(args.output_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
