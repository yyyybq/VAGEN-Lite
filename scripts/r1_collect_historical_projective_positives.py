#!/usr/bin/env python3
"""Collect exactly the baseline accepted Projective rows for regression replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    accepted = [row for row in read_jsonl(args.baseline) if row.get("stage", {}).get("final_accepted")]
    rows, certificates = [], []
    for record in accepted:
        index, split = int(record["source_row_index"]), record["split"]
        job_dir = args.jobs_root / f"{split}_{index:06d}" / "repair" / split
        repaired = read_jsonl(job_dir / "repaired_checkpoint.jsonl")
        matches = [item["row"] for item in repaired if int(item["source_row_index"]) == index]
        reachability = [item for item in read_jsonl(job_dir / "reachability_manifest.jsonl") if int(item.get("source_row_index", -1)) == index]
        if len(matches) != 1 or len(reachability) != 1:
            raise ValueError(f"{index}: rows={len(matches)} reachability={len(reachability)}")
        rows.append(matches[0])
        certificates.append(reachability[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "positive_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.output_dir / "positive_reachability.jsonl").write_text("".join(json.dumps(row) + "\n" for row in certificates))
    payload = {"version": "r1_historical_projective_positive_collection_v1", "rows": len(rows), "source_row_indices": [r["source_row_index"] for r in certificates]}
    (args.output_dir / "collection_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
