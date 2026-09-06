#!/usr/bin/env python3
"""Freeze the 16-row v6 diagnostic cohort without rerunning generation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_records = {
        (row["split"], int(row["source_row_index"])): row
        for row in json.loads(args.selection.read_text())["records"]
    }
    rows = [json.loads(line) for line in args.manifest.open() if line.strip()]
    accepted = sorted(
        (row for row in rows if row.get("stage", {}).get("final_accepted")),
        key=lambda row: (row["split"], int(row["source_row_index"])),
    )[:5]
    wall = [
        row for row in rows
        if row["split"] == "id_test" and int(row["source_row_index"]) == 6629
    ]
    if len(wall) != 1:
        raise RuntimeError("frozen wall-clearance negative control id_test:6629 missing")
    failures = [
        row for row in rows
        if row.get("generation_status") == "hard_failure"
        and (row["split"], int(row["source_row_index"])) != ("id_test", 6629)
    ]
    grouped = defaultdict(list)
    for row in failures:
        grouped[row.get("failure_taxonomy") or "unknown"].append(row)
    chosen_failures = []
    for taxonomy in sorted(grouped):
        grouped[taxonomy].sort(key=lambda row: (row["split"], int(row["source_row_index"])))
        chosen_failures.append(grouped[taxonomy][0])
        if len(chosen_failures) == 5:
            break
    if len(chosen_failures) < 5:
        remaining = [row for row in failures if row not in chosen_failures]
        chosen_failures.extend(sorted(remaining, key=lambda row: (row["split"], int(row["source_row_index"])))[: 5 - len(chosen_failures)])
    timeouts = sorted(
        (row for row in rows if row.get("generation_job_status") == "unverified_generation_timeout"),
        key=lambda row: (row["split"], int(row["source_row_index"])),
    )[:5]
    selected = []
    for kind, cohort in (("positive_control", accepted), ("observability_negative", wall), ("generation_failure", chosen_failures), ("generation_timeout", timeouts)):
        for row in cohort:
            key = (row["split"], int(row["source_row_index"]))
            selected.append({**source_records[key], "diagnostic_kind": kind})
    keys = [(row["split"], int(row["source_row_index"])) for row in selected]
    if len(selected) != 16 or len(keys) != len(set(keys)):
        raise RuntimeError(f"diagnostic cohort not 16 unique rows: {len(selected)}")
    payload = {
        "version": "r1_projective_v6_diagnostic16_v1",
        "source_selection": str(args.selection),
        "source_manifest": str(args.manifest),
        "records": selected,
        "cohort_counts": {kind: sum(row["diagnostic_kind"] == kind for row in selected) for kind in ("positive_control", "observability_negative", "generation_failure", "generation_timeout")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
