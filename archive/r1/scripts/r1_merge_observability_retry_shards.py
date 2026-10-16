#!/usr/bin/env python3
"""Merge per-scene observability retries with exact failed-audit accounting."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from r1_full_regeneration import atomic_json, atomic_jsonl, read_jsonl
from r1_projective_observability_retry import VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--observability-dir", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    splits = json.loads(args.sources.read_text())
    scenes = [value for value in args.scenes.split(",") if value]
    for split in splits:
        expected = {
            int(row["source_row_index"])
            for row in read_jsonl(
                args.observability_dir / split / "observability_manifest.jsonl"
            )
            if not row.get("passed") and row.get("reasons") != ["renderer_error"]
        }
        rows = []
        for scene in scenes:
            manifest = args.shards_dir / scene / split / "retry_manifest.jsonl"
            if not manifest.is_file():
                raise RuntimeError(f"missing retry shard: {manifest}")
            rows.extend(read_jsonl(manifest))
        indices = [int(row["source_row_index"]) for row in rows]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"{split}: duplicate retry source indices")
        if set(indices) != expected:
            raise RuntimeError(
                f"{split}: retry mismatch missing={sorted(expected - set(indices))[:20]} "
                f"extra={sorted(set(indices) - expected)[:20]}"
            )
        rows.sort(key=lambda row: int(row["source_row_index"]))
        attempt_counts = Counter(len(row.get("attempts") or []) for row in rows)
        summary = {
            "version": VERSION,
            "split": split,
            "input_failures": len(rows),
            "recovered": sum(bool(row.get("recovered")) for row in rows),
            "still_failed": sum(not bool(row.get("recovered")) for row in rows),
            "attempt_count_distribution": dict(sorted(attempt_counts.items())),
            "failed_observability_accounting_exact": True,
            "scenes": scenes,
        }
        output = args.output_dir / split
        atomic_jsonl(output / "retry_manifest.jsonl", rows)
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
