#!/usr/bin/env python3
"""Run frozen Projective v6 sample rows as isolated, resumable jobs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

from r1_donor_allocator import AtomicDonorLedger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--donor-ledger", type=Path, required=True)
    parser.add_argument("--collision-convention-overrides", type=Path, required=True)
    parser.add_argument("--max-replacement-candidates", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    donor_ledger = AtomicDonorLedger(args.donor_ledger)
    payload = json.loads(args.selection.read_text())
    records = payload["records"]
    jobs_root = args.output_root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)

    def run_one(record):
        split = record["split"]
        index = int(record["source_row_index"])
        job_key = f"{split}_{index:06d}"
        output = jobs_root / job_key
        summary = output / "summary.json"
        if summary.is_file():
            return {"job": job_key, "status": "resume_existing"}
        output.mkdir(parents=True, exist_ok=True)
        selection = output / "selection.json"
        selection.write_text(json.dumps({"records": [record]}, indent=2) + "\n")
        command = [
            sys.executable,
            str(Path(__file__).with_name("r1_full_regeneration.py")),
            "--sources", str(args.sources),
            "--splits", split,
            "--output-dir", str(output / "repair"),
            "--gs-root", str(args.gs_root),
            "--source-index-selection", str(selection),
            "--reachability-tiers", "2000,25000",
            "--disable-final-tier",
            "--donor-ledger", str(args.donor_ledger),
            "--collision-convention-overrides", str(args.collision_convention_overrides),
            "--max-replacement-candidates", str(args.max_replacement_candidates),
        ]
        started = time.monotonic()
        try:
            with (output / "job.log").open("w") as handle:
                completed = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT,
                    timeout=args.timeout_seconds,
                )
            result = {
                "job": job_key,
                "status": "completed" if completed.returncode == 0 else "process_error",
                "returncode": completed.returncode,
            }
        except subprocess.TimeoutExpired:
            result = {
                "job": job_key,
                "status": "unverified_generation_timeout",
                "timeout_seconds": args.timeout_seconds,
            }
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        events_path = output / "repair" / split / "stage_events.jsonl"
        if events_path.is_file():
            events = [json.loads(line) for line in events_path.open() if line.strip()]
            if events:
                result["last_stage"] = events[-1].get("stage")
                result["stage_event_count"] = len(events)
        if result["status"] in {"unverified_generation_timeout", "process_error"}:
            owner = f"{split}:{index}"
            accounting_path = output / "repair" / split / "accounting.jsonl"
            keep_states = set()
            if accounting_path.is_file():
                accounting_rows = [json.loads(line) for line in accounting_path.open() if line.strip()]
                if accounting_rows and accounting_rows[0].get("status") in {"strict_same_pair_repair", "count_matched_replacement"} and accounting_rows[0].get("reachability_status") == "reachable":
                    # A crash after reachability was written is recoverable:
                    # retain validation_pending until official RGB decides.
                    keep_states = {"validation_pending"}
            released = donor_ledger.recover_owner(owner, result["status"], keep_states=keep_states)
            result["released_donors"] = [list(donor) for donor in released]
        (output / "job_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, record) for record in records]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed": result, "count": len(results), "total": len(records)}), flush=True)
    results.sort(key=lambda row: row["job"])
    (args.output_root / "job_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"total": len(results), "status_counts": {
        status: sum(row["status"] == status for row in results)
        for status in sorted({row["status"] for row in results})
    }}, indent=2))


if __name__ == "__main__":
    main()
