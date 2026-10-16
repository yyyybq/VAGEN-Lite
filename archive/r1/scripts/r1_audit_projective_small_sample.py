#!/usr/bin/env python3
"""Official-render audit for accepted isolated v6 sample jobs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

from r1_donor_allocator import AtomicDonorLedger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--renderer-lock", type=Path, required=True)
    parser.add_argument("--collision-convention-overrides", type=Path, required=True)
    parser.add_argument("--donor-ledger", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    donor_ledger = AtomicDonorLedger(args.donor_ledger) if args.donor_ledger else None
    records = json.loads(args.selection.read_text())["records"]
    jobs = []
    for record in records:
        split = record["split"]
        index = int(record["source_row_index"])
        job = args.jobs_root / f"{split}_{index:06d}"
        accounting_path = job / "repair" / split / "accounting.jsonl"
        if not accounting_path.is_file():
            continue
        accounting = [json.loads(line) for line in accounting_path.open() if line.strip()]
        if not accounting or accounting[0].get("status") not in {
            "strict_same_pair_repair", "count_matched_replacement"
        }:
            continue
        if accounting[0].get("task_type") != "projective_relations":
            continue
        jobs.append((record, job, accounting[0]))

    def run_one(job_data):
        record, job, accounting = job_data
        output = job / "observability"
        if (output / "summary.json").is_file():
            summary = json.loads((output / "summary.json").read_text())
            if not any(reason == "renderer_error" for reason in summary.get("failure_reasons", {})):
                result = {"job": job.name, "status": "resume_existing"}
                _finalize_donor(record, accounting, output, result)
                return result
        repaired = job / "repair" / record["split"] / "trainable.jsonl"
        reachability = job / "repair" / record["split"] / "reachability_manifest.jsonl"
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("r1_projective_observability_audit.py")),
            "--repaired", str(repaired),
            "--reachability", str(reachability),
            "--gs-root", str(args.gs_root),
            "--renderer-url", args.renderer_url,
            "--output-dir", str(output),
            "--source-indices", str(record["source_row_index"]),
            "--collision-convention-overrides", str(args.collision_convention_overrides),
            "--renderer-lock", str(args.renderer_lock),
        ]
        with (output / "job.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        result_row = {
            "job": job.name,
            "status": "completed" if result.returncode == 0 else "process_error",
            "returncode": result.returncode,
            "source_row_index": record["source_row_index"],
        }
        _finalize_donor(record, accounting, output, result_row)
        return result_row

    def _finalize_donor(record, accounting, output, result_row):
        if donor_ledger is None:
            return
        lineage = accounting.get("replacement_lineage") or {}
        donor = lineage.get("new_pair")
        if not donor:
            return
        owner = f"{record['split']}:{record['source_row_index']}"
        states = donor_ledger.states()
        donor_key = "\u001f".join(str(part) for part in donor)
        if states.get(donor_key, {}).get("owner") != owner:
            result_row["donor_transition"] = "not_owned"
            return
        # A subprocess/renderer implementation error is not an RGB verdict.
        # Keep the reservation pending so a later resume can retry the same
        # owner/attempt; never release a donor merely because no audit file
        # was produced.
        if result_row.get("status") == "process_error":
            result_row["donor_transition"] = "kept_validation_pending_process_error"
            return
        manifest = output / "observability_manifest.jsonl"
        if not manifest.is_file() or not manifest.read_text().strip():
            result_row["donor_transition"] = "kept_validation_pending_missing_audit"
            return
        audit = json.loads(manifest.read_text().splitlines()[0])
        if audit.get("renderer_error"):
            # An implementation/service error is not an RGB quality verdict;
            # keep validation_pending for a retry and never release a donor
            # merely because the renderer was unavailable.
            result_row["donor_transition"] = "kept_validation_pending_renderer_error"
        elif audit.get("passed"):
            donor_ledger.commit(owner, tuple(donor), attempt=f"{owner}:attempt1")
            result_row["donor_transition"] = "committed"
        else:
            donor_ledger.release(owner, tuple(donor), attempt=f"{owner}:attempt1", reason="rgb_observability_rejected")
            result_row["donor_transition"] = "released_rgb_rejection"

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed": result, "count": len(results), "total": len(jobs)}), flush=True)
    summary = {
        "accepted_generation_jobs": len(jobs),
        "results": results,
        "status_counts": {
            status: sum(row["status"] == status for row in results)
            for status in sorted({row["status"] for row in results})
        },
    }
    (args.jobs_root.parent / "official_render_audit_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
