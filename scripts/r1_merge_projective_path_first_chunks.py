#!/usr/bin/env python3
"""Close split-qualified accounting across frozen path-first canary chunks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    expected = {(str(row["split"]), int(row["source_row_index"])): row for row in selection["records"]}
    generated: dict[tuple[str, int], dict[str, Any]] = {}
    generation_files = sorted(args.generation_root.glob("chunk_*/prototype_results.json"))
    for path in generation_files:
        for row in json.loads(path.read_text()).get("rows", []):
            key = (str(row.get("split")), int(row["source_row_index"]))
            if key in generated:
                raise ValueError(f"duplicate generated key {key}")
            generated[key] = row
    if set(generated) != set(expected):
        raise ValueError(f"generation accounting mismatch missing={sorted(set(expected)-set(generated))[:5]} extra={sorted(set(generated)-set(expected))[:5]}")
    candidate_rows = [row for row in generated.values() if row.get("status") == "candidate_found"]
    task_keys = {row["row"]["task_id"]: key for key, row in generated.items() if row.get("status") == "candidate_found"}
    if len(task_keys) != len(candidate_rows):
        raise ValueError("duplicate candidate task IDs")
    runtime: dict[str, dict[str, Any]] = {}
    runtime_files = sorted(args.runtime_root.glob("chunk_*/runtime_replay.json"))
    for path in runtime_files:
        for row in json.loads(path.read_text()).get("results", []):
            task_id = str(row["task_id"])
            if task_id in runtime:
                raise ValueError(f"duplicate runtime task ID {task_id}")
            if task_id not in task_keys:
                raise ValueError(f"runtime task outside candidates {task_id}")
            runtime[task_id] = {**row, "split": task_keys[task_id][0]}
    missing_runtime = sorted(set(task_keys) - set(runtime))
    if missing_runtime:
        raise ValueError(f"missing runtime results for {len(missing_runtime)} candidates")
    merged_rows = [generated[key] for key in sorted(generated)]
    prototype = {
        "version": "r1_path_first_canary10_merged_v1",
        "selection": str(args.selection),
        "source_rows": len(merged_rows),
        "generation_files": [str(path) for path in generation_files],
        "rows": merged_rows,
        "status_counts": dict(Counter(row["status"] for row in merged_rows)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prototype_results.json").write_text(json.dumps(prototype, indent=2) + "\n")
    runtime_payload = {
        "version": "r1_projective_certificate_replay_merged_v1",
        "rows": len(runtime),
        "passed": sum(row.get("status") == "pass" for row in runtime.values()),
        "results": [runtime[key] for key in sorted(runtime)],
    }
    (args.output_dir / "runtime_replay.json").write_text(json.dumps(runtime_payload, indent=2) + "\n")
    print(json.dumps({
        "source_rows": len(merged_rows), "candidate_rows": len(candidate_rows),
        "generation_status_counts": prototype["status_counts"],
        "runtime_rows": len(runtime), "runtime_passed": runtime_payload["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
