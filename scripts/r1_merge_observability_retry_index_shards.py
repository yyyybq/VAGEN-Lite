#!/usr/bin/env python3
"""Merge deterministic failure-index retry parts for one scene/split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from r1_full_regeneration import atomic_json, atomic_jsonl, read_jsonl
from r1_projective_observability_retry import VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observability", type=Path, required=True)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--part-count", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        int(row["source_row_index"])
        for row in read_jsonl(args.observability)
        if not row.get("passed") and row.get("reasons") != ["renderer_error"]
    }
    rows = []
    for part in range(args.part_count):
        part_dir = args.parts_dir / f"part_{part:02d}"
        if not (part_dir / "summary.json").is_file():
            raise RuntimeError(f"retry part incomplete: {part_dir}")
        rows.extend(read_jsonl(part_dir / "retry_manifest.jsonl"))
    indices = [int(row["source_row_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise RuntimeError("duplicate source index across retry parts")
    if set(indices) != expected:
        raise RuntimeError(
            f"retry part mismatch missing={sorted(expected - set(indices))[:20]} "
            f"extra={sorted(set(indices) - expected)[:20]}"
        )
    rows.sort(key=lambda row: int(row["source_row_index"]))
    attempts = Counter(len(row.get("attempts") or []) for row in rows)
    summary = {
        "version": VERSION,
        "split": args.split,
        "input_failures": len(rows),
        "recovered": sum(bool(row.get("recovered")) for row in rows),
        "still_failed": sum(not bool(row.get("recovered")) for row in rows),
        "attempt_count_distribution": dict(sorted(attempts.items())),
        "failure_index_shards_merged": args.part_count,
        "failed_observability_accounting_exact": True,
    }
    atomic_jsonl(args.output_dir / "retry_manifest.jsonl", rows)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
