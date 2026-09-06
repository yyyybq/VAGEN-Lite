#!/usr/bin/env python3
"""Freeze a deterministic 50--100-row Projective v6 source sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def stable_order(rows):
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{row[0]}:{row[1]}".encode()
        ).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--observability-dir", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--conflict-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    accounting = {}
    observations = {}
    for path in sorted(args.aggregate_dir.glob("*/accounting.jsonl")):
        split = path.parent.name
        for row in read(path):
            if row.get("task_type") == "projective_relations":
                accounting[(split, int(row["source_row_index"]))] = row
    for path in sorted(args.observability_dir.glob("*/observability_manifest.jsonl")):
        split = path.parent.name
        for row in read(path):
            observations[(split, int(row["source_row_index"]))] = row
    taxonomy = {
        (row["split"], int(row["source_row_index"])): row.get("taxonomy")
        for row in read(args.taxonomy)
    }
    conflict_keys = set()
    for path in sorted(args.conflict_root.glob("*/*/duplicate_replacement_donor_conflicts.jsonl")):
        for row in read(path):
            conflict_keys.add((path.parent.name, int(row["source_row_index"])))

    pools = defaultdict(list)
    for key, row in accounting.items():
        obs = observations.get(key, {})
        if row.get("status") in {"strict_same_pair_repair", "count_matched_replacement"} and not obs.get("passed"):
            pools["initial_not_discernible" if "initial_dual_target_not_discernible" in obs.get("reasons", []) else "observability_failure"].append(key)
        if taxonomy.get(key) in {"layout/collision infeasible", "search insufficient", "planner budget unverified"}:
            pools[taxonomy[key].replace("/", "_").replace(" ", "_")].append(key)
        if key in conflict_keys:
            pools["donor_conflict"].append(key)
        if obs.get("passed"):
            pools["positive_control"].append(key)

    quotas = {
        "initial_not_discernible": 20,
        "layout_collision_infeasible": 12,
        "search_insufficient": 10,
        "donor_conflict": 10,
        "planner_budget_unverified": 16,
        "positive_control": 12,
    }
    selected = {}
    records_by_key = {}
    for kind, quota in quotas.items():
        chosen = stable_order(list(dict.fromkeys(pools[kind])))[:quota]
        selected[kind] = [list(key) for key in chosen]
        for key in chosen:
            record = records_by_key.setdefault(
                key,
                {
                    "sample_kind": kind,
                    "sample_kinds": [kind],
                    "split": key[0],
                    "source_row_index": key[1],
                    "scene_id": accounting[key].get("scene_id"),
                    "task_type": accounting[key].get("task_type"),
                },
            )
            if kind not in record["sample_kinds"]:
                record["sample_kinds"].append(kind)
    records = list(records_by_key.values())
    records.sort(key=lambda row: (row["split"], row["source_row_index"], row["sample_kind"]))
    payload = {
        "version": "r1_projective_v6_small_sample_v1",
        "selection_seed": "sha256(split:source_row_index)",
        "selection_is_frozen": True,
        "selection_continues_only_if": [
            "accounting_closed",
            "accepted_rows_pass_semantic_runtime_reachability_and_rgb_gates",
            "same_source_comparison_has_net_recovery",
            "positive_controls_have_no_unexplained_regression",
            "donor_ledger_has_no_duplicate_or_ood_violation",
        ],
        "quotas": quotas,
        "selected_by_kind": selected,
        "records": records,
        "total": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"total": len(records), "pool_sizes": {k: len(v) for k, v in pools.items()}}, indent=2))


if __name__ == "__main__":
    main()
