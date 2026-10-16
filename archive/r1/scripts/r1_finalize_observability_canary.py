#!/usr/bin/env python3
"""Apply Projective observability manifests to a closed geometry canary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from r1_full_regeneration import atomic_json, atomic_jsonl, read_jsonl, sha256


ACCEPTED = {"strict_same_pair_repair", "count_matched_replacement"}


def index_of(row: dict[str, Any]) -> int:
    value = row.get("source_row_index")
    if value is None:
        value = row.get("repair_lineage", {}).get("source_row_index")
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--observability-dir", type=Path, required=True)
    parser.add_argument("--retry-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = json.loads(args.sources.read_text())
    overall = Counter()
    split_summaries = {}
    for split, source in sources.items():
        input_dir = args.aggregate_dir / split
        accounting = read_jsonl(input_dir / "accounting.jsonl")
        repaired = {index_of(row): row for row in read_jsonl(input_dir / "trainable.jsonl")}
        mappings = {index_of(row): row for row in read_jsonl(input_dir / "mapping.jsonl")}
        observations = {
            int(row["source_row_index"]): row
            for row in read_jsonl(args.observability_dir / split / "observability_manifest.jsonl")
        }
        reachability_rows = read_jsonl(input_dir / "reachability_manifest.jsonl")
        reachability = {index_of(row): row for row in reachability_rows}
        recovered_indices = set()
        if args.retry_dir and (args.retry_dir / split / "retry_manifest.jsonl").is_file():
            for retry in read_jsonl(args.retry_dir / split / "retry_manifest.jsonl"):
                if not retry.get("recovered"):
                    continue
                index = int(retry["source_row_index"])
                selected = retry["selected"]
                repaired[index] = selected["row"]
                mappings[index] = selected["mapping"]
                reachability[index] = selected["reachability"]
                observations[index] = selected["observability"]
                recovered_indices.add(index)
        rejected_mappings = []
        for row in accounting:
            if row.get("status") not in ACCEPTED or row.get("task_type") != "projective_relations":
                continue
            index = index_of(row)
            audit = observations.get(index)
            if audit is None:
                row.update({
                    "status": "unverified",
                    "failure": "projective_observability_manifest_missing",
                    "failure_taxonomy": "observability failure",
                })
            elif not audit.get("passed"):
                renderer_only = audit.get("reasons") == ["renderer_error"]
                row.update({
                    "status": "unverified" if renderer_only else "observability_failure",
                    "failure": "projective_observability_unverified" if renderer_only else "projective_observability_v1_failed",
                    "failure_taxonomy": "observability failure",
                    "observability_reasons": audit.get("reasons"),
                })
            else:
                row["projective_observability_version"] = audit.get("version")
                row["projective_observability_passed"] = True
                if index in recovered_indices:
                    row["projective_observability_retry_recovered"] = True
            if row.get("status") not in ACCEPTED:
                repaired.pop(index, None)
                mapping = mappings.pop(index, None)
                if mapping is not None:
                    mapping["rejected_by"] = row.get("failure")
                    rejected_mappings.append(mapping)

        expected = len(accounting)
        indices = [index_of(row) for row in accounting]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"{split}: duplicate accounting rows")
        accepted_indices = {index_of(row) for row in accounting if row.get("status") in ACCEPTED}
        if accepted_indices != set(repaired) or accepted_indices != set(mappings):
            raise RuntimeError(f"{split}: post-observability accepted outputs do not close")
        output_dir = args.output_dir / split
        output_rows = [repaired[index] for index in sorted(repaired)]
        mapping_rows = [mappings[index] for index in sorted(mappings)]
        failures = [row for row in accounting if row.get("status") in {"hard_failure", "observability_failure"}]
        unverified = [row for row in accounting if row.get("status") == "unverified"]
        atomic_jsonl(output_dir / "accounting.jsonl", sorted(accounting, key=index_of))
        atomic_jsonl(output_dir / "trainable.jsonl", output_rows)
        atomic_jsonl(output_dir / "mapping.jsonl", mapping_rows)
        atomic_jsonl(output_dir / "rejected_mapping_lineage.jsonl", rejected_mappings)
        atomic_jsonl(output_dir / "failure_manifest.jsonl", failures)
        atomic_jsonl(output_dir / "unverified_manifest.jsonl", unverified)
        atomic_jsonl(
            output_dir / "reachability_manifest.jsonl",
            [reachability[index] for index in sorted(reachability)],
        )
        atomic_jsonl(
            output_dir / "candidate_manifest.jsonl",
            read_jsonl(input_dir / "candidate_manifest.jsonl"),
        )
        counts = Counter(str(row["status"]) for row in accounting)
        overall.update(counts)
        projective_accepted = [
            row for row in accounting
            if row.get("task_type") == "projective_relations" and row.get("status") in ACCEPTED
        ]
        summary = {
            "split": split,
            "source": source,
            "source_target_rows_in_scope": expected,
            "accounted_target_rows": len(accounting),
            "accounting_closed": expected == len(accounting),
            "status_counts": dict(counts),
            "usable_rows": len(output_rows),
            "usable_coverage": len(output_rows) / expected if expected else None,
            "accepted_projective_rows": len(projective_accepted),
            "accepted_projective_observability_all_pass": all(
                row.get("projective_observability_passed") for row in projective_accepted
            ),
            "observability_retry_recovered": len(recovered_indices),
            "output": str(output_dir / "trainable.jsonl"),
        }
        atomic_json(output_dir / "summary.json", summary)
        summary["output_sha256"] = sha256(output_dir / "trainable.jsonl")
        atomic_json(output_dir / "summary.json", summary)
        split_summaries[split] = summary
    total = sum(overall.values())
    final = {
        "splits": split_summaries,
        "expected_target_rows": total,
        "accounted_target_rows": total,
        "all_accounting_closed": all(row["accounting_closed"] for row in split_summaries.values()),
        "status_counts": dict(overall),
        "usable_rows": overall["strict_same_pair_repair"] + overall["count_matched_replacement"],
        "usable_coverage": (
            (overall["strict_same_pair_repair"] + overall["count_matched_replacement"]) / total
            if total else None
        ),
        "observability_failure": overall["observability_failure"],
        "unverified": overall["unverified"],
    }
    atomic_json(args.output_dir / "summary.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
