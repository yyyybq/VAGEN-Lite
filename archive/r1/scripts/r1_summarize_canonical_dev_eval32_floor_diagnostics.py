#!/usr/bin/env python3
"""Close accounting for R1 dev32 floor diagnostics without rerunning policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "r1_canonical_dev_eval32_floor_diagnostic_summary_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def episodes(path: Path, section: str | None = None) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    return data[section] if section else data["episodes"]


def local_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer: dict[str, Any] = {"states": len(rows), "infrastructure_errors": sum(row.get("status") != "complete" for row in rows)}
    by_distance = {}
    by_source: dict[str, Counter] = defaultdict(Counter)
    for distance in (1, 2):
        subset = [row for row in rows if row.get("status") == "complete" and int(row["exact_distance_audit_only"]) == distance]
        by_distance[f"d{distance}"] = {
            "states": len(subset),
            "optimal_first_action_hits": sum(bool(row["optimal_first_action_hit"]) for row in subset),
            "invalid_or_empty": sum(row.get("first_parsed_action") is None for row in subset),
            "collision_attempts": sum(int(row.get("collision_attempts", 0)) for row in subset),
            "canonical_success_after_first_action": sum(bool((row.get("canonical_after") or {}).get("success")) for row in subset),
        }
        for row in subset:
            by_source[row["source_key"]]["states"] += 1
            by_source[row["source_key"]]["hits"] += int(bool(row["optimal_first_action_hit"]))
    answer["by_distance"] = by_distance
    answer["source_grouped"] = {key: dict(value) for key, value in sorted(by_source.items())}
    return answer


def replan_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    actions = Counter(action for row in complete for action in row.get("actions", []))
    return {
        "complete": len(complete), "infrastructure_errors": len(rows) - len(complete),
        "success": sum(bool(row.get("success")) for row in complete),
        "success_sources": sorted(row["source_key"] for row in complete if row.get("success")),
        "primitive_actions": sum(int(row.get("primitive_steps", 0)) for row in complete),
        "model_turns": sum(int(row.get("model_turns", 0)) for row in complete),
        "model_inference_calls": sum(int(row.get("model_inference_calls", 0)) for row in complete),
        "collision_attempts": sum(int(row.get("collision_attempts", 0)) for row in complete),
        "invalid_turns": sum(int(row.get("invalid_turns", 0)) for row in complete),
        "inference_seconds": sum(sum(float(turn.get("inference_seconds", 0)) for turn in row.get("turns", [])) for row in complete),
        "actions": dict(sorted(actions.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    models = ("pretrained", "v46_step250")
    baseline, current = {}, {}
    for model in models:
        baseline[model] = episodes(args.baseline_root / model / "episode_ledger.json")
        ledger = args.diagnostic_root / model / "diagnostic_ledger.json"
        current[model] = {"local": episodes(ledger, "local"), "replan": episodes(ledger, "replan")}
    local_paired = []
    local_keys = sorted(set(current[models[0]]["local"]) | set(current[models[1]]["local"]))
    for key in local_keys:
        first, second = current[models[0]]["local"].get(key, {}), current[models[1]]["local"].get(key, {})
        local_paired.append({
            "local_state_key": key, "source_key": first.get("source_key") or second.get("source_key"),
            "exact_distance": first.get("exact_distance_audit_only") or second.get("exact_distance_audit_only"),
            "pretrained_action": first.get("first_parsed_action"), "pretrained_hit": first.get("optimal_first_action_hit"),
            "v46_action": second.get("first_parsed_action"), "v46_hit": second.get("optimal_first_action_hit"),
            "both_hit": bool(first.get("optimal_first_action_hit") and second.get("optimal_first_action_hit")),
        })
    replan_paired = []
    pair_counts = Counter()
    for key in sorted(current[models[0]]["replan"]):
        pre = current[models[0]]["replan"][key]; v46 = current[models[1]]["replan"][key]
        outcome = ("both_success" if pre.get("success") and v46.get("success") else
                   "pretrained_only" if pre.get("success") else "v46_only" if v46.get("success") else "both_fail")
        pair_counts[outcome] += 1
        replan_paired.append({
            "eval_index": int(key), "source_key": pre.get("source_key") or v46.get("source_key"),
            "pretrained_batch_baseline_success": bool(baseline[models[0]][key].get("success")),
            "pretrained_first_action_replan_success": bool(pre.get("success")),
            "v46_batch_baseline_success": bool(baseline[models[1]][key].get("success")),
            "v46_first_action_replan_success": bool(v46.get("success")),
            "paired_replan_outcome": outcome,
        })
    model_summary = {}
    for model in models:
        local_rows = [current[model]["local"][key] for key in sorted(current[model]["local"])]
        replan_rows = [current[model]["replan"][key] for key in sorted(current[model]["replan"])]
        model_summary[model] = {
            "local_action_selection": local_summary(local_rows),
            "batch_baseline_success": sum(bool(row.get("success")) for row in baseline[model].values()),
            "first_action_only_replan": replan_summary(replan_rows),
        }
    summary = {
        "version": VERSION, "models": model_summary,
        "replan_paired_outcomes": dict(pair_counts),
        "interpretation_limits": [
            "derived local states are grouped by their 32 parent sources and are not independent test samples",
            "first-action-only replanning changes feedback execution and inference-call cost, not model representation",
            "development_regression scenes are not a final generalization set",
        ],
    }
    write_jsonl(args.output_dir / "local_action_paired.jsonl", local_paired)
    write_jsonl(args.output_dir / "batch_vs_replan_paired.jsonl", replan_paired)
    write_json(args.output_dir / "summary.json", summary)
    hashes = []
    for path in sorted(p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        hashes.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
