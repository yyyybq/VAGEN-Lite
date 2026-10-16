#!/usr/bin/env python3
"""Compare the frozen 71-row Projective baseline against path-first output.

This is an accounting/reporting utility: it never promotes a candidate unless
the independently generated runtime and official-render results both passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_at(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [value_at(row, key) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def old_route(row: dict[str, Any]) -> str:
    status = row.get("generation_status")
    if status == "strict_same_pair_repair":
        return "same_pair"
    if status == "count_matched_replacement":
        return "replacement"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--observability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = {int(row["source_row_index"]): row for row in read_jsonl(args.baseline)}
    selection = {int(row["source_row_index"]): row for row in json.loads(args.selection.read_text())["records"]}
    generated = {int(row["source_row_index"]): row for row in json.loads(args.prototype.read_text())["rows"]}
    runtime = {int(row["source_row_index"]): row for row in json.loads(args.runtime.read_text())["results"]}
    rgb = {int(row["source_row_index"]): row for row in read_jsonl(args.observability)}

    missing = sorted(set(selection) - set(baseline))
    if missing:
        raise ValueError(f"baseline missing selected rows: {missing}")
    rows: list[dict[str, Any]] = []
    for index in sorted(selection):
        old = baseline[index]
        gen = generated.get(index, {})
        replay = runtime.get(index, {})
        audit = rgb.get(index, {})
        candidate = gen.get("status") == "candidate_found"
        runtime_pass = replay.get("status") == "pass"
        rgb_pass = audit.get("passed") is True
        accepted = candidate and runtime_pass and rgb_pass
        old_accepted = bool(old.get("stage", {}).get("final_accepted"))
        if old_accepted and not candidate:
            regression = "generator_search_not_found"
        elif old_accepted and candidate and not runtime_pass:
            regression = "runtime_replay"
        elif old_accepted and runtime_pass and not rgb_pass:
            regression = "observability"
        else:
            regression = None
        difficulty = gen.get("source_difficulty") or old.get("source_difficulty") or {}
        rows.append({
            "source_row_index": index,
            "scene_id": selection[index]["scene_id"],
            "split": selection[index]["split"],
            "diagnostic_kind": selection[index]["diagnostic_kind"],
            "old_status": old.get("generation_status"),
            "old_failure": old.get("failure"),
            "old_failure_taxonomy": old.get("failure_taxonomy"),
            "old_route": old_route(old),
            "old_final_accepted": old_accepted,
            "new_status": gen.get("status", "missing_accounting"),
            "new_route": "same_pair" if candidate else "none",
            "target_candidates": int(gen.get("target_candidates", 0) or 0),
            "canonical_success_targets": int(gen.get("canonical_success_targets", 0) or 0),
            "reverse_calls": int(gen.get("reverse_calls", 0) or 0),
            "graph_connected": int(gen.get("reverse_resolved", 0) or 0),
            "certificate": candidate,
            "certificate_steps_upper_bound": replay.get("steps"),
            "runtime_replay": replay.get("status"),
            "rgb_observability": audit.get("passed"),
            "rgb_reasons": audit.get("reasons", []),
            "final_accepted": accepted,
            "generation_elapsed_seconds": float(gen.get("elapsed_seconds", 0.0) or 0.0),
            "source_difficulty": difficulty,
            "regression_reason": regression,
            "recovery": bool(not old_accepted and accepted),
        })

    old_accepted = [row for row in rows if row["old_final_accepted"]]
    new_accepted = [row for row in rows if row["final_accepted"]]
    old_failure_timeout = [row for row in rows if not row["old_final_accepted"]]
    recoveries = [row for row in old_failure_timeout if row["final_accepted"]]
    old_status = Counter(row["old_status"] for row in rows)
    new_status = Counter(row["new_status"] for row in rows)
    failure_modes = Counter(
        "no_safe_canonical_success_target" if row["canonical_success_targets"] == 0
        else "no_graph_certificate_within_budget"
        for row in rows if not row["certificate"]
    )
    accepted_difficulty = []
    for row in new_accepted:
        difficulty = dict(row["source_difficulty"])
        difficulty["planner_steps"] = row["certificate_steps_upper_bound"]
        accepted_difficulty.append(difficulty)
    old_final_difficulty = []
    for row in old_accepted:
        difficulty = dict(row["source_difficulty"])
        difficulty["planner_steps"] = row.get("certificate_steps_upper_bound")
        old_final_difficulty.append(difficulty)
    payload = {
        "version": "r1_projective_same_source71_path_first_comparison_v1",
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "baseline": args.baseline, "selection": args.selection, "prototype": args.prototype,
            "runtime": args.runtime, "observability": args.observability,
        }.items()},
        "funnel": {
            "source": len(rows),
            "canonical_success_states": sum(row["canonical_success_targets"] for row in rows),
            "rows_with_canonical_success": sum(row["canonical_success_targets"] > 0 for row in rows),
            "valid_observable_initial": sum(row["certificate"] for row in rows),
            "graph_connected": sum(row["graph_connected"] for row in rows),
            "certificate": sum(row["certificate"] for row in rows),
            "runtime_pass": sum(row["runtime_replay"] == "pass" for row in rows),
            "rgb_pass": sum(row["rgb_observability"] is True for row in rows),
            "final_accepted": len(new_accepted),
        },
        "baseline": {"status_counts": dict(old_status), "final_accepted": len(old_accepted)},
        "new_status_counts": dict(new_status),
        "recovery": {
            "old_failure_or_timeout": len(old_failure_timeout),
            "final_accepted": len(recoveries),
            "same_pair": sum(row["new_route"] == "same_pair" for row in recoveries),
            "by_old_status": dict(Counter(row["old_status"] for row in recoveries)),
            "by_old_failure_taxonomy": dict(Counter(row["old_failure_taxonomy"] for row in recoveries)),
            "rows": recoveries,
        },
        "positive_control_regression": {
            "old_final_accepted": len(old_accepted),
            "new_final_accepted": sum(row["final_accepted"] for row in old_accepted),
            "regressions": [row for row in old_accepted if not row["final_accepted"]],
        },
        "route_comparison": {
            "old_same_pair": sum(row["old_route"] == "same_pair" for row in rows),
            "old_replacement": sum(row["old_route"] == "replacement" for row in rows),
            "new_same_pair_candidates": sum(row["new_route"] == "same_pair" for row in rows),
            "new_replacement": 0,
            "new_final_same_pair": sum(row["new_route"] == "same_pair" for row in new_accepted),
        },
        "failure_modes": dict(failure_modes),
        "difficulty": {
            "new_final": {key: distribution(accepted_difficulty, key) for key in (
                "translation_m", "yaw_deg", "bbox_area_ratio", "relation_margin_px", "planner_steps")},
            "old_final_source": {key: distribution(old_final_difficulty, key) for key in (
                "translation_m", "yaw_deg", "bbox_area_ratio", "relation_margin_px", "planner_steps")},
            "new_one_step": sum(row["certificate_steps_upper_bound"] == 1 for row in new_accepted),
        },
        "cost": {
            "generation_wall_seconds": sum(row["generation_elapsed_seconds"] for row in rows),
            "mean_generation_seconds": statistics.mean(row["generation_elapsed_seconds"] for row in rows),
            "max_generation_seconds": max(row["generation_elapsed_seconds"] for row in rows),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"funnel": payload["funnel"], "recovery": payload["recovery"], "positive": payload["positive_control_regression"], "cost": payload["cost"]}, indent=2))


if __name__ == "__main__":
    main()
