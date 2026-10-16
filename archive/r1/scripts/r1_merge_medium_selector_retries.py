#!/usr/bin/env python3
"""Merge narrowly-scoped medium-selector retries without mutating either input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


VERSION = "projective_difficulty_conditioned_medium_selector_v2_merged_retry_v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def key(row: dict) -> tuple[str, int, str]:
    return str(row["split"]), int(row["source_row_index"]), str(row["requested_bucket"])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base, retry = load(args.base), load(args.retry)
    for field in ("version", "selection", "same_pair_only", "budgets"):
        if base.get(field) != retry.get(field):
            raise ValueError(f"incompatible {field}")
    rows = list(base["results"])
    positions = {key(row): index for index, row in enumerate(rows)}
    if len(positions) != len(rows):
        raise ValueError("duplicate base key")
    replaced = []
    for row in retry["results"]:
        row_key = key(row)
        if row_key not in positions:
            raise ValueError(f"retry key absent from base: {row_key}")
        old = rows[positions[row_key]]
        if old.get("status") != "implementation_error":
            raise ValueError(f"retry may replace only implementation_error: {row_key}={old.get('status')}")
        rows[positions[row_key]] = row
        replaced.append({"key": list(row_key), "old_status": old["status"], "new_status": row["status"]})
    if len({tuple(entry["key"]) for entry in replaced}) != len(replaced):
        raise ValueError("duplicate retry key")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    result = {
        "version": VERSION,
        "base_selector": str(args.base),
        "base_sha256": sha256(args.base),
        "retry_selector": str(args.retry),
        "retry_sha256": sha256(args.retry),
        "base_version": base["version"],
        "selection": base["selection"],
        "same_pair_only": base["same_pair_only"],
        "budgets": base["budgets"],
        "results": rows,
        "status_counts": counts,
        "retry_lineage": replaced,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"records": len(rows), "status_counts": counts, "retries": replaced}, indent=2))


if __name__ == "__main__":
    main()
