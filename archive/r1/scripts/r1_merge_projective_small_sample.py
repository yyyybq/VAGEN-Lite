#!/usr/bin/env python3
"""Merge isolated v6 generation and official-render stages into one ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from r1_full_regeneration import atomic_json, atomic_jsonl, read_jsonl, sha256


ACCEPTED = {"strict_same_pair_repair", "count_matched_replacement"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--donor-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    accepted_rows = []
    mappings = []
    for record in json.loads(args.selection.read_text())["records"]:
        split = record["split"]
        index = int(record["source_row_index"])
        job = args.jobs_root / f"{split}_{index:06d}"
        result_path = job / "job_result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else {
            "status": "unverified_generation_missing"
        }
        accounting_path = job / "repair" / split / "accounting.jsonl"
        accounting = read_jsonl(accounting_path) if accounting_path.is_file() else []
        accounting_row = accounting[0] if accounting else None
        stage = {
            "source": True,
            "same_pair_or_replacement": False,
            "canonical_target": False,
            "reachable_path": False,
            "runtime_consistent": False,
            "rgb_observability": False,
            "final_accepted": False,
        }
        row = {
            **record,
            "job": str(job),
            "generation_job_status": result.get("status"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "stage": stage,
        }
        if accounting_row:
            status = accounting_row.get("status")
            row["generation_status"] = status
            row["failure"] = accounting_row.get("failure")
            row["failure_taxonomy"] = accounting_row.get("failure_taxonomy")
            row["path_steps_upper_bound"] = accounting_row.get("found_path_length_upper_bound")
            row["source_difficulty"] = accounting_row.get("source_difficulty")
            row["repaired_difficulty"] = accounting_row.get("repaired_difficulty")
            # Preserve the generator's bounded-search evidence for both
            # accepted and failed rows.  This is intentionally copied from
            # the per-row manifest rather than reconstructed from the final
            # task, so budget exhaustion remains auditable.
            failure_manifest = job / "repair" / split / "failure_manifest.jsonl"
            mapping_manifest = job / "repair" / split / "mapping.jsonl"
            if failure_manifest.is_file():
                failures_for_job = read_jsonl(failure_manifest)
                if failures_for_job:
                    row["repair_details"] = failures_for_job[0].get("repair")
            stage["same_pair_or_replacement"] = status in ACCEPTED
            stage["canonical_target"] = status in ACCEPTED
            stage["reachable_path"] = accounting_row.get("reachability_status") == "reachable"
            stage["runtime_consistent"] = stage["reachable_path"]
            if status in ACCEPTED:
                trainable = read_jsonl(job / "repair" / split / "trainable.jsonl")
                mappings_for_job = read_jsonl(job / "repair" / split / "mapping.jsonl")
                if mappings_for_job:
                    row["repair_details"] = mappings_for_job[0]
                accepted_task_ids = {
                    item.get("new_task_id") for item in mappings_for_job
                    if item.get("new_task_id")
                }
                if trainable:
                    accepted_rows.extend(
                        item for item in trainable
                        if item.get("task_id") in accepted_task_ids
                    )
                mappings.extend(mappings_for_job)
                observation = job / "observability" / "observability_manifest.jsonl"
                if observation.is_file():
                    audits = read_jsonl(observation)
                    audit = audits[0] if audits else {}
                    row["observability_reasons"] = audit.get("reasons")
                    row["observability_version"] = audit.get("version")
                    row["rgb_observability"] = bool(audit.get("passed"))
                    stage["rgb_observability"] = bool(audit.get("passed"))
        row["stage"]["final_accepted"] = all(
            stage[key]
            for key in (
                "source", "same_pair_or_replacement", "canonical_target",
                "reachable_path", "runtime_consistent", "rgb_observability",
            )
        )
        rows.append(row)

    rows.sort(key=lambda row: (row["split"], int(row["source_row_index"])))
    accepted_rows.sort(key=lambda row: row.get("task_id", ""))
    mappings.sort(key=lambda row: int(row.get("source_row_index", 0)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output_dir / "sample_manifest.jsonl", rows)
    atomic_jsonl(args.output_dir / "accepted_trainable.jsonl", accepted_rows)
    atomic_jsonl(args.output_dir / "accepted_mapping.jsonl", mappings)
    statuses = Counter(row["generation_job_status"] for row in rows)
    generation_statuses = Counter(row.get("generation_status", "unverified_generation_timeout") for row in rows)
    stage_counts = {
        key: sum(bool(row["stage"][key]) for row in rows)
        for key in ("same_pair_or_replacement", "canonical_target", "reachable_path", "runtime_consistent", "rgb_observability", "final_accepted")
    }
    ledger = json.loads(args.donor_ledger.read_text()) if args.donor_ledger.is_file() else {"reservations": {}}
    owners = [
        value.get("owner") if isinstance(value, dict) else value
        for value in ledger.get("reservations", {}).values()
    ]
    donors = list(ledger.get("reservations", {}).keys())
    summary = {
        "version": "r1_projective_v6_small_sample_merged_v1",
        "source_rows": len(rows),
        "accounting_closed": len(rows) == len({(row["split"], row["source_row_index"]) for row in rows}),
        "generation_job_status_counts": dict(statuses),
        "generation_status_counts": dict(generation_statuses),
        "stage_counts": stage_counts,
        "accepted_trainable_rows": len(accepted_rows),
        "donor_ledger_reservations": len(donors),
        "donor_ledger_unique": len(donors) == len(set(donors)) and len(owners) == len(set(owners)),
        "donor_ledger_sha256": sha256(args.donor_ledger) if args.donor_ledger.is_file() else None,
        "output_sha256": {
            name: sha256(args.output_dir / name)
            for name in ("sample_manifest.jsonl", "accepted_trainable.jsonl", "accepted_mapping.jsonl")
        },
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
