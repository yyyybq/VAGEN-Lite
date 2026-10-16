#!/usr/bin/env python3
"""Aggregate frozen projective stage events without rerunning generation."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]

def event_index(job: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in job.glob("repair/*/stage_events.jsonl"):
        for event in read_jsonl(path):
            result[event.get("stage", "unknown")] = event
    return result

def funnel_row(row: dict[str, Any], events: dict[str, Any]) -> dict[str, Any]:
    details = row.get("repair_details") or {}
    stats = details.get("retry") or details.get("search_stats") or {}
    budget = stats.get("search_budget") or {}
    search = (events.get("generation_candidate_search_done") or {}).get("search_stats") or {}
    search_budget = search.get("search_budget") or {}
    target_safe = stats.get("layout_valid_positions", search.get("layout_valid_positions"))
    target_success = stats.get("canonical_success_candidates", search.get("canonical_success_candidates"))
    initial_observable = budget.get("initial_geometry_prefilter_passes", search_budget.get("initial_geometry_prefilter_passes"))
    stage = row.get("stage") or {}
    timeout = row.get("generation_job_status") == "unverified_generation_timeout"
    return {
        "split": row.get("split"), "source_row_index": row.get("source_row_index"),
        "scene_id": row.get("scene_id"), "generation_job_status": row.get("generation_job_status"),
        "generation_status": row.get("generation_status"), "failure": row.get("failure"),
        "failure_taxonomy": row.get("failure_taxonomy"), "source": True,
        "object_pair_eligible": row.get("task_type") == "projective_relations",
        "target_candidates_evaluated": stats.get("target_points_evaluated", search_budget.get("target_points_evaluated")),
        "safe_target_candidates": target_safe, "canonical_success_target": target_success,
        "safe_initial_candidates": initial_observable, "dual_target_observable_initial": initial_observable,
        "canonical_fail_initial": 1 if (details.get("new_initial_metric") or {}).get("success") is False else None,
        "layout_collision_valid": bool(target_safe and int(target_safe) > 0) if target_safe is not None else None,
        "planner_invoked": "planner_start" in events,
        "planner_last_stage": "planner_done" if "planner_done" in events else ("planner_start" if "planner_start" in events else None),
        "path_found": bool(stage.get("reachable_path")), "runtime_certificate": bool(stage.get("runtime_consistent")),
        "rgb_observability": bool(stage.get("rgb_observability")), "final_accepted": bool(stage.get("final_accepted")),
        "timeout_stage": "planner_done" if timeout and "planner_done" in events else ("planner_start" if timeout and "planner_start" in events else None),
        "search_budget_exhausted": budget.get("budget_exhausted"), "expansions": (events.get("planner_done") or {}).get("expansions"),
        "visited_states": (events.get("planner_done") or {}).get("visited_states"), "wall_time_seconds": row.get("elapsed_seconds"),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.open() if line.strip()]
    funnel = [funnel_row(row, event_index(Path(row["job"]))) for row in rows]
    fields = ("source", "object_pair_eligible", "safe_target_candidates", "canonical_success_target", "safe_initial_candidates", "dual_target_observable_initial", "canonical_fail_initial", "layout_collision_valid", "planner_invoked", "path_found", "runtime_certificate", "rgb_observability")
    totals: dict[str, Any] = {"input_rows": len(funnel)}
    for field in fields:
        values = [row[field] for row in funnel]; known = [value for value in values if value is not None]
        totals[field] = {"known_rows": len(known), "passed_rows": sum(bool(value) for value in known), "unknown_rows": len(values) - len(known)}
    hard = [row for row in funnel if row["generation_status"] == "hard_failure"]
    timeouts = [row for row in funnel if row["generation_job_status"] == "unverified_generation_timeout"]
    totals["hard_failure_count"] = len(hard)
    totals["hard_failure_by_taxonomy"] = dict(collections.Counter(row.get("failure_taxonomy") for row in hard))
    totals["hard_failure_by_failure"] = dict(collections.Counter(row.get("failure") for row in hard))
    totals["timeout_count"] = len(timeouts)
    totals["timeout_by_stage"] = dict(collections.Counter(row.get("timeout_stage") or "stage_not_persisted" for row in timeouts))
    totals["search_budget_exhausted"] = sum(row.get("search_budget_exhausted") is True for row in funnel)
    payload = {"version": "r1_projective_generation_funnel_v1", "source_manifest": str(args.manifest), "rows": funnel, "totals": totals, "interpretation": {"unknown_stage_values": "Existing v6 artifacts did not persist every initial rejection reason; unknown is retained rather than inferred as failure.", "timeout_semantics": "Wall-clock timeout and planner expansion cap remain separate and do not prove unreachable."}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "totals": totals}, indent=2))

if __name__ == "__main__":
    main()
