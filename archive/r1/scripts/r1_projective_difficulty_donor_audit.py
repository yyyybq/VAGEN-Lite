#!/usr/bin/env python3
"""Audit versioned difficulty signatures for existing replacement lineages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "translation_m",
    "yaw_deg",
    "bbox_area_ratio",
    "relation_margin_px",
    "planner_step_proxy",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def distance(source: dict[str, Any], repaired: dict[str, Any]) -> dict[str, Any]:
    source_buckets = source.get("buckets") or {}
    repaired_buckets = repaired.get("buckets") or {}
    bucket_distance = sum(
        abs(int(source_buckets.get(field, 0)) - int(repaired_buckets.get(field, 0)))
        for field in FIELDS
    )
    numeric_distance = sum(
        abs(float(source.get(field, 0.0)) - float(repaired.get(field, 0.0)))
        for field in FIELDS
    )
    return {
        "bucket_distance": bucket_distance,
        "numeric_l1": numeric_distance,
        "planner_easier": float(repaired.get("planner_step_proxy", 0.0))
        < float(source.get("planner_step_proxy", 0.0)),
        "bucket_match": bucket_distance == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    replacements = []
    for row in rows:
        if row.get("generation_status") != "count_matched_replacement":
            continue
        details = row.get("repair_details") or {}
        lineage = details.get("replacement_lineage") or {}
        source = lineage.get("old_source_difficulty") or row.get("source_difficulty") or {}
        donor = lineage.get("donor_source_difficulty") or {}
        repaired = lineage.get("new_repaired_difficulty") or row.get("repaired_difficulty") or {}
        replacements.append({
            "split": row.get("split"),
            "source_row_index": row.get("source_row_index"),
            "old_pair": lineage.get("old_pair"),
            "donor_source_row_index": lineage.get("donor_source_row_index"),
            "new_pair": lineage.get("new_pair"),
            "source_difficulty": source,
            "donor_difficulty": donor,
            "repaired_difficulty": repaired,
            "donor_match_distance": distance(source, donor) if donor else None,
            "repaired_match_distance": distance(source, repaired) if repaired else None,
            "replacement_reason": lineage.get("reason"),
        })
    payload = {
        "version": "r1_projective_difficulty_signature_v1",
        "signature_fields": list(FIELDS),
        "replacement_count": len(replacements),
        "bucket_exact_matches": sum(
            bool(row.get("repaired_match_distance", {}).get("bucket_match"))
            for row in replacements
        ),
        "planner_easier_count": sum(
            bool(row.get("repaired_match_distance", {}).get("planner_easier"))
            for row in replacements
        ),
        "rows": replacements,
        "policy": "same scene/split/relation/OOD-safe and unique donor first; then minimize five-field bucket distance, penalizing easier planner depth",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"replacement_count": len(replacements), "bucket_exact_matches": payload["bucket_exact_matches"], "planner_easier_count": payload["planner_easier_count"]}, indent=2))


if __name__ == "__main__":
    main()
