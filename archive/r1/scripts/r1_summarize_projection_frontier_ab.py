#!/usr/bin/env python3
"""Close accounting and the predeclared gate for the frozen 16+4 frontier A/B."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "r1_projection_frontier_ab_summary_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["split"]), int(row["source_row_index"])


def evidence(selector: dict[str, Any], runtime_path: Path, rgb_path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    runtime_payload = json.loads(runtime_path.read_text())
    runtime = {str(row["task_id"]): row for row in runtime_payload["results"]}
    rgb_rows = read_jsonl(rgb_path)
    rgb = {str(row["task_id"]): row for row in rgb_rows}
    output = {}
    for row in selector["results"]:
        task_id = None
        if row.get("status") == "difficulty_certified_candidate":
            task_id = f"{row['row']['task_id']}_medium"
        replay = runtime.get(task_id) if task_id else None
        observation = rgb.get(task_id) if task_id else None
        output[key(row)] = {
            "selector_status": row.get("status"),
            "task_id": task_id,
            "difficulty_certified": row.get("status") == "difficulty_certified_candidate",
            "runtime_status": replay.get("status") if replay else None,
            "rgb_pass": bool(observation and observation.get("passed")),
            "rgb_reasons": observation.get("reasons", []) if observation else [],
            "first_success_step": row.get("certificate_upper_bound"),
            "certified_lower_bound": row.get("certified_lower_bound"),
            "elapsed_seconds": float(row.get("elapsed_seconds", 0.0)),
            "success_seeds": int(row.get("success_region_seeds_considered", 0)),
            "initial_candidates": int(row.get("initial_candidates_collected", 0)),
            "eligible_medium_candidates": int(row.get("eligible_medium_candidates", 0)),
            "reverse_expansions": sum(int(attempt.get("expansions", 0)) for attempt in row.get("reverse_attempts", [])),
        }
    return output, {"runtime": runtime_payload, "rgb_rows": rgb_rows}


def depth_counts_from_baseline(diagnostic: dict[str, Any]) -> dict[str, dict[str, int]]:
    total: dict[str, Counter[str]] = defaultdict(Counter)
    for row in diagnostic["results"]:
        for seed in row.get("seed_results", []):
            for depth, values in seed.get("by_depth", {}).items():
                for name, value in values.items():
                    if isinstance(value, int):
                        total[depth][name] += value
    return {depth: dict(sorted(values.items())) for depth, values in sorted(total.items(), key=lambda item: int(item[0]))}


def depth_counts_from_frontier(selector: dict[str, Any]) -> dict[str, dict[str, int]]:
    total: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selector["results"]:
        for attempt in row.get("reverse_attempts", []):
            for depth, values in attempt.get("by_depth", {}).items():
                for name, value in values.items():
                    if isinstance(value, int):
                        total[depth][name] += value
    return {depth: dict(sorted(values.items())) for depth, values in sorted(total.items(), key=lambda item: int(item[0]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline-selector", type=Path, required=True)
    parser.add_argument("--baseline-runtime", type=Path, required=True)
    parser.add_argument("--baseline-rgb", type=Path, required=True)
    parser.add_argument("--baseline-diagnostic", type=Path, required=True)
    parser.add_argument("--frontier-selector", type=Path, required=True)
    parser.add_argument("--frontier-runtime", type=Path, required=True)
    parser.add_argument("--frontier-rgb", type=Path, required=True)
    parser.add_argument("--phase-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    baseline_selector = json.loads(args.baseline_selector.read_text())
    frontier_selector = json.loads(args.frontier_selector.read_text())
    baseline, baseline_payload = evidence(baseline_selector, args.baseline_runtime, args.baseline_rgb)
    frontier, frontier_payload = evidence(frontier_selector, args.frontier_runtime, args.frontier_rgb)
    records = []
    for selected in selection["records"]:
        source = key(selected)
        old = baseline[source]
        new = frontier[source]
        final = bool(new["difficulty_certified"] and new["runtime_status"] == "pass" and new["rgb_pass"])
        records.append({
            "split": source[0], "source_row_index": source[1], "scene_id": selected["scene_id"],
            "kind": selected["kind"], "baseline": old, "projection_frontier": new,
            "final_accepted": final,
            "change": "recovered" if selected["kind"] == "reverse_cap" and final
                      else "control_preserved" if selected["kind"] == "known_positive_control" and final
                      else "control_regressed" if selected["kind"] == "known_positive_control"
                      else "still_unresolved",
        })
    failures = [row for row in records if row["kind"] == "reverse_cap"]
    controls = [row for row in records if row["kind"] == "known_positive_control"]
    recovered = [row for row in failures if row["final_accepted"]]
    recovered_scenes = sorted({row["scene_id"] for row in recovered})
    baseline_elapsed = sum(row["baseline"]["elapsed_seconds"] for row in records)
    frontier_elapsed = sum(row["projection_frontier"]["elapsed_seconds"] for row in records)
    ratio = frontier_elapsed / baseline_elapsed if baseline_elapsed else None
    controls_passed = sum(row["final_accepted"] for row in controls)
    frontier_certified = [row for row in records if row["projection_frontier"]["difficulty_certified"]]
    frontier_runtime_pass = [row for row in frontier_certified if row["projection_frontier"]["runtime_status"] == "pass"]
    frontier_rgb_pass = [row for row in frontier_runtime_pass if row["projection_frontier"]["rgb_pass"]]
    gate = {
        "recovered_at_least_4_of_16": len(recovered) >= 4,
        "recovery_spans_at_least_3_scenes": len(recovered_scenes) >= 3,
        "positive_controls_4_of_4": controls_passed == 4,
        "all_final_accepted_runtime_rgb_pass": all(row["projection_frontier"]["runtime_status"] == "pass" and row["projection_frontier"]["rgb_pass"] for row in records if row["final_accepted"]),
        "same_source_elapsed_ratio_at_most_1_2": ratio is not None and ratio <= 1.2,
    }
    payload = {
        "version": VERSION,
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {
            "selection": args.selection, "baseline_selector": args.baseline_selector,
            "baseline_runtime": args.baseline_runtime, "baseline_rgb": args.baseline_rgb,
            "baseline_diagnostic": args.baseline_diagnostic, "frontier_selector": args.frontier_selector,
            "frontier_runtime": args.frontier_runtime, "frontier_rgb": args.frontier_rgb,
            "phase_timestamps": args.phase_timestamps,
        }.items()},
        "accounting": {"selected": len(records), "failures": len(failures), "controls": len(controls),
                       "closed": len(records) == 20 and len(failures) == 16 and len(controls) == 4},
        "baseline_status_counts": dict(sorted(Counter(row["baseline"]["selector_status"] for row in records).items())),
        "frontier_status_counts": dict(sorted(Counter(row["projection_frontier"]["selector_status"] for row in records).items())),
        "frontier_funnel": {"difficulty_certified": len(frontier_certified),
                            "runtime_pass": len(frontier_runtime_pass), "rgb_pass": len(frontier_rgb_pass),
                            "failure_sources_recovered": len(recovered), "recovery_scenes": recovered_scenes,
                            "positive_controls_passed": controls_passed},
        "cost": {"baseline_sum_elapsed_seconds": baseline_elapsed,
                 "frontier_sum_elapsed_seconds": frontier_elapsed,
                 "frontier_over_baseline_same_source_ratio": ratio,
                 "phase_timestamps": read_jsonl(args.phase_timestamps)},
        "by_depth": {"baseline_diagnostic": depth_counts_from_baseline(json.loads(args.baseline_diagnostic.read_text())),
                     "projection_frontier": depth_counts_from_frontier(frontier_selector)},
        "gate": {**gate, "passed": all(gate.values()),
                 "decision": "frontier_variant_supported" if all(gate.values()) else "frontier_variant_evidence_insufficient"},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"funnel": payload["frontier_funnel"], "cost": payload["cost"], "gate": payload["gate"]}, indent=2))


if __name__ == "__main__":
    main()
