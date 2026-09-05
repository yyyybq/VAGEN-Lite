#!/usr/bin/env python3
"""Merge per-scene Projective observability audits with exact accepted-row checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from r1_full_regeneration import atomic_json, atomic_jsonl, read_jsonl
from r1_projective_observability import OBSERVABILITY_THRESHOLDS, PROJECTIVE_OBSERVABILITY_VERSION


ACCEPTED = {"strict_same_pair_repair", "count_matched_replacement"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    splits = json.loads(args.sources.read_text())
    scenes = [value for value in args.scenes.split(",") if value]
    for split in splits:
        expected = {
            int(row["source_row_index"])
            for row in read_jsonl(args.aggregate_dir / split / "accounting.jsonl")
            if row.get("task_type") == "projective_relations"
            and row.get("status") in ACCEPTED
        }
        rows = []
        for scene in scenes:
            manifest = args.shards_dir / scene / split / "observability_manifest.jsonl"
            if not manifest.is_file():
                raise RuntimeError(f"missing observability shard: {manifest}")
            rows.extend(read_jsonl(manifest))
        indices = [int(row["source_row_index"]) for row in rows]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"{split}: duplicate observability source indices")
        if set(indices) != expected:
            raise RuntimeError(
                f"{split}: observability mismatch missing={sorted(expected - set(indices))[:20]} "
                f"extra={sorted(set(indices) - expected)[:20]}"
            )
        rows.sort(key=lambda row: int(row["source_row_index"]))
        reasons = Counter(reason for row in rows for reason in row.get("reasons", []))
        summary = {
            "version": PROJECTIVE_OBSERVABILITY_VERSION,
            "rows": len(rows),
            "passed": sum(bool(row.get("passed")) for row in rows),
            "failed": sum(not bool(row.get("passed")) for row in rows),
            "failure_reasons": dict(reasons),
            "thresholds": OBSERVABILITY_THRESHOLDS,
            "oracle_claim": "heuristic_not_true_occlusion_oracle",
            "accepted_projective_accounting_exact": True,
            "scenes": scenes,
        }
        output = args.output_dir / split
        atomic_jsonl(output / "observability_manifest.jsonl", rows)
        atomic_json(output / "summary.json", summary)
        print(json.dumps({"split": split, **summary}, indent=2))


if __name__ == "__main__":
    main()
