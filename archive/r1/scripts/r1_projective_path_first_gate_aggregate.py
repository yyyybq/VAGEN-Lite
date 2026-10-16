#!/usr/bin/env python3
"""Close fixed-selection accounting across generation, runtime, and RGB gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, action="append", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--observability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = {int(row["source_row_index"]): row for row in json.loads(args.selection.read_text())["records"]}
    generation: dict[int, dict] = {}
    for artifact in args.prototype:
        for row in json.loads(artifact.read_text()).get("rows", []):
            index = int(row["source_row_index"])
            # A retry replaces only an explicit implementation-error record.
            if index not in generation or generation[index].get("status") == "implementation_error":
                generation[index] = {**row, "artifact": str(artifact)}
    runtime = {int(row["source_row_index"]): row for row in json.loads(args.runtime.read_text())["results"]}
    rgb = {int(row["source_row_index"]): row for row in read_jsonl(args.observability)}
    rows = []
    for index, record in selection.items():
        gen = generation.get(index, {})
        replay = runtime.get(index)
        audit = rgb.get(index)
        rows.append({
            "source_row_index": index,
            "split": record["split"],
            "scene_id": record["scene_id"],
            "diagnostic_kind": record["diagnostic_kind"],
            "generation_status": gen.get("status", "missing_accounting"),
            "canonical_success_targets": gen.get("canonical_success_targets", 0),
            "reverse_calls": gen.get("reverse_calls", 0),
            "reverse_resolved": gen.get("reverse_resolved", 0),
            "generation_elapsed_seconds": gen.get("elapsed_seconds", 0.0),
            "runtime_replay": replay.get("status") if replay else None,
            "runtime_steps": replay.get("steps") if replay else None,
            "rgb_observability": audit.get("passed") if audit else None,
            "rgb_reasons": audit.get("reasons") if audit else [],
            "final_accepted": bool(
                gen.get("status") == "candidate_found"
                and replay and replay.get("status") == "pass"
                and audit and audit.get("passed")
            ),
        })
    payload = {
        "version": "r1_projective_path_first_gate_aggregate_v1",
        "source_rows": len(rows),
        "generation_candidates": sum(row["generation_status"] == "candidate_found" for row in rows),
        "runtime_passed": sum(row["runtime_replay"] == "pass" for row in rows),
        "rgb_passed": sum(row["rgb_observability"] is True for row in rows),
        "final_accepted": sum(row["final_accepted"] for row in rows),
        "generation_wall_seconds": sum(float(row["generation_elapsed_seconds"] or 0.0) for row in rows),
        "rows": sorted(rows, key=lambda row: row["source_row_index"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "source_rows", "generation_candidates", "runtime_passed", "rgb_passed",
        "final_accepted", "generation_wall_seconds",
    )}, indent=2))


if __name__ == "__main__":
    main()
