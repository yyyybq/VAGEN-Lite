#!/usr/bin/env python3
"""Compare exact-SKU renderer benchmark reports and select the smallest pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GPU_COUNTS = (1, 2, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = {}
    missing = []
    for gpu_count in GPU_COUNTS:
        path = args.run / "results" / f"{gpu_count}gpu_12workers.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        reports[str(gpu_count)] = json.loads(path.read_text(encoding="utf-8"))

    digests = [
        report.get("measured", {}).get("ordered_png_digest_sha256")
        for report in reports.values()
    ]
    passing = [
        gpu_count
        for gpu_count in GPU_COUNTS
        if reports.get(str(gpu_count), {}).get("status") == "PASS"
    ]
    checks = {
        "all_topologies_completed": not missing and len(reports) == len(GPU_COUNTS),
        "all_topologies_produced_full_batch": all(
            report.get("measured", {}).get("requests") == 384 for report in reports.values()
        ) and len(reports) == len(GPU_COUNTS),
        "all_topologies_image_identical": len(digests) == len(GPU_COUNTS)
        and None not in digests
        and len(set(digests)) == 1,
        "at_least_one_topology_meets_target": bool(passing),
        "no_optimizer_step": all(
            report.get("safety", {}).get("optimizer_step_called") is False
            for report in reports.values()
        ) and len(reports) == len(GPU_COUNTS),
        "no_checkpoint_write": all(
            report.get("safety", {}).get("checkpoint_written") is False
            for report in reports.values()
        ) and len(reports) == len(GPU_COUNTS),
    }
    selected = min(passing) if all(checks.values()) else None
    payload = {
        "schema_version": "active_spatial_renderer_microbenchmark_comparison_v1",
        "status": "PASS" if selected is not None else "BLOCKED",
        "checks": checks,
        "missing": missing,
        "selected_topology": None
        if selected is None
        else {"gpu_count": selected, "max_workers": 12, "max_inflight": 12},
        "reports": reports,
        "training_submission_allowed": selected is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
