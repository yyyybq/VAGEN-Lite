#!/usr/bin/env python3
"""Freeze an exact Projective source selection from an existing accounting manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.manifest.read_bytes()
    records = []
    for line in source_bytes.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task_type") != "projective_relations":
            continue
        records.append({
            "source_row_index": int(row["source_row_index"]),
            "split": row["split"],
            "scene_id": row["scene_id"],
            "task_type": row["task_type"],
            "diagnostic_kind": row.get("sample_kind", "unspecified"),
            "old_generation_status": row.get("generation_status"),
            "old_generation_job_status": row.get("generation_job_status"),
            "old_failure": row.get("failure"),
            "old_failure_taxonomy": row.get("failure_taxonomy"),
        })
    payload = {
        "version": "r1_projective_frozen_same_source_selection_v1",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": "deterministic_no_rng",
        "records": records,
        "record_count": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"record_count": len(records), "source_manifest_sha256": payload["source_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
