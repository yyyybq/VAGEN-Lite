#!/usr/bin/env python3
"""Re-evaluate saved official-render observability evidence after a key-frame fix.

No RGB is regenerated: every decision comes from the immutable frame-level
evidence written by the original official-render audit.  This keeps the
versioned recheck cheap while preserving a complete audit trail.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from r1_full_regeneration import atomic_json, atomic_jsonl, read_jsonl
from r1_projective_observability import (
    OBSERVABILITY_THRESHOLDS,
    PROJECTIVE_OBSERVABILITY_VERSION,
)


def recheck(row: dict) -> dict:
    row = dict(row)
    frames = row.get("frames") or []
    row["version"] = PROJECTIVE_OBSERVABILITY_VERSION
    row["thresholds"] = OBSERVABILITY_THRESHOLDS
    if not frames:
        return row
    initial_dual = bool(frames[0].get("dual_target_observation"))
    reasons = list(row.get("reasons") or [])
    if not initial_dual and "initial_dual_target_not_discernible" not in reasons:
        reasons.append("initial_dual_target_not_discernible")
    row["initial_dual_target_observation"] = initial_dual
    row["reasons"] = reasons
    row["passed"] = not reasons
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows = []
    for manifest in sorted(args.input_dir.glob("*/*/observability_manifest.jsonl")):
        relative = manifest.parent.relative_to(args.input_dir)
        rows = [recheck(row) for row in read_jsonl(manifest)]
        all_rows.extend(rows)
        reasons = Counter(reason for row in rows for reason in row.get("reasons") or [])
        summary = {
            "version": PROJECTIVE_OBSERVABILITY_VERSION,
            "source_evidence_version": "projective_observability_v1",
            "scene": relative.parts[0],
            "split": relative.parts[1],
            "rows": len(rows),
            "passed": sum(bool(row.get("passed")) for row in rows),
            "failed": sum(not bool(row.get("passed")) for row in rows),
            "failure_reasons": dict(reasons),
            "recheck_uses_saved_official_render_evidence": True,
        }
        output = args.output_dir / relative
        atomic_jsonl(output / "observability_manifest.jsonl", rows)
        atomic_json(output / "summary.json", summary)

    reasons = Counter(reason for row in all_rows for reason in row.get("reasons") or [])
    atomic_json(
        args.output_dir / "summary.json",
        {
            "version": PROJECTIVE_OBSERVABILITY_VERSION,
            "rows": len(all_rows),
            "passed": sum(bool(row.get("passed")) for row in all_rows),
            "failed": sum(not bool(row.get("passed")) for row in all_rows),
            "failure_reasons": dict(reasons),
            "recheck_uses_saved_official_render_evidence": True,
        },
    )


if __name__ == "__main__":
    main()
