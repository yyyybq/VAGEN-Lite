#!/usr/bin/env python3
"""Calibrate and replay the FOV-specific RGB keyframe audit.

The input pilot is the explicit human-reviewed FOV v3 final pilot.  Unlike
Projective, FOV begins from intentionally incomplete framing, so this tool
never treats raw bbox inside-frame fraction as a quality threshold.  It also
does not silently convert programmatic v1 rejections into human negative
labels: absent human negatives are reported as a calibration limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from r1_fov_observability import (
    FOV_OBSERVABILITY_V2_VERSION,
    FOV_V2_THRESHOLDS,
    evaluate_fov_path_v2,
)
from r1_reachability_audit import atomic_write_json, atomic_write_jsonl, read_jsonl


VERSION = "fov_observability_v2_calibration_20260919"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_images(path_root: Path, task_id: str, path: list[dict[str, Any]]) -> list[Image.Image]:
    directory = path_root / task_id
    image_paths = sorted(directory.glob("step_*.png"))
    if len(image_paths) != len(path):
        raise RuntimeError(
            f"{task_id}: expected {len(path)} rendered steps, found {len(image_paths)} in {directory}"
        )
    return [Image.open(image_path).convert("RGB") for image_path in image_paths]


def audited_rows(
    candidates_path: Path,
    reachability_path: Path,
    path_root: Path,
) -> list[dict[str, Any]]:
    candidates = {str(row["task_id"]): row for row in read_jsonl(candidates_path)}
    reaches = {str(row["task_id"]): row for row in read_jsonl(reachability_path)}
    audits = []
    for task_id in sorted(candidates):
        reach = reaches.get(task_id)
        if not reach or reach.get("status") != "reachable" or not reach.get("path"):
            audits.append({"task_id": task_id, "passed": False, "reasons": ["reachability_path_not_available"]})
            continue
        try:
            audit = evaluate_fov_path_v2(candidates[task_id], reach["path"], load_images(path_root, task_id, reach["path"]), None)
        except Exception as error:  # Preserve renderer/artifact defects distinctly.
            audit = {"version": FOV_OBSERVABILITY_V2_VERSION, "passed": False,
                     "reasons": ["artifact_read_error"], "error": repr(error)}
        audits.append({
            "task_id": task_id,
            "source_row_index": reach.get("source_row_index"),
            "scene_id": candidates[task_id].get("scene_id"),
            "path_steps": reach.get("steps"),
            **audit,
        })
    return audits


def support_bounds(audits: list[dict[str, Any]]) -> dict[str, float | None]:
    values: dict[str, list[float]] = {
        "visible_bbox_area_px": [], "patch_entropy_bits": [], "patch_edge_fraction": [],
        "patch_clipped_luminance_fraction": [], "frame_mean_luminance": [], "frame_luminance_std": [],
    }
    for audit in audits:
        for frame in audit.get("frames", []):
            for obj in frame.get("objects", []):
                values["visible_bbox_area_px"].append(float(obj["bbox_area_px"]))
                patch = obj["rgb_patch"]
                values["patch_entropy_bits"].append(float(patch["entropy_bits_32bin"]))
                values["patch_edge_fraction"].append(float(patch["edge_fraction_gt8"]))
                values["patch_clipped_luminance_fraction"].append(float(patch["clipped_luminance_fraction"]))
            rgb = frame.get("frame_rgb", {})
            if rgb:
                values["frame_mean_luminance"].append(float(rgb["mean_luminance"]))
                values["frame_luminance_std"].append(float(rgb["luminance_std"]))
    return {
        "minimum_visible_bbox_area_px": min(values["visible_bbox_area_px"], default=None),
        "minimum_patch_entropy_bits": min(values["patch_entropy_bits"], default=None),
        "minimum_patch_edge_fraction": min(values["patch_edge_fraction"], default=None),
        "maximum_patch_clipped_luminance_fraction": max(values["patch_clipped_luminance_fraction"], default=None),
        "minimum_frame_mean_luminance": min(values["frame_mean_luminance"], default=None),
        "minimum_frame_luminance_std": min(values["frame_luminance_std"], default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--pilot-candidates", type=Path, required=True)
    parser.add_argument("--pilot-reachability", type=Path, required=True)
    parser.add_argument("--pilot-path-rgb", type=Path, required=True)
    parser.add_argument("--expansion-candidates", type=Path, required=True)
    parser.add_argument("--expansion-reachability", type=Path, required=True)
    parser.add_argument("--expansion-path-rgb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    review = json.loads(args.manual_review.read_text())
    manual_fov = review.get("fov_final_pilot", {})
    reviewed_rows = int(manual_fov.get("rows_reviewed", 0))
    if manual_fov.get("decision") != "pass" or reviewed_rows <= 0:
        raise RuntimeError("the specified review does not contain a passing FOV pilot")
    pilot = audited_rows(args.pilot_candidates, args.pilot_reachability, args.pilot_path_rgb)
    if len(pilot) != reviewed_rows:
        raise RuntimeError(f"manual FOV rows {reviewed_rows} but pilot candidates {len(pilot)}")
    expansion = audited_rows(
        args.expansion_candidates, args.expansion_reachability, args.expansion_path_rgb
    )

    reasons = Counter(reason for row in expansion for reason in row.get("reasons", []))
    pilot_pass = sum(bool(row.get("passed")) for row in pilot)
    report = {
        "version": VERSION,
        "gate_version": FOV_OBSERVABILITY_V2_VERSION,
        "canonical_success_definition": "canonical_spatial_task_h1_v1:fov_full_bbox_h1 (unchanged)",
        "manual_label_evidence": {
            "artifact": str(args.manual_review), "sha256": sha256(args.manual_review),
            "reviewed_rows": reviewed_rows, "human_pass_labels": reviewed_rows,
            "human_fail_labels": 0,
            "missing_negative_label_policy": "v1 programmatic rejections are not recast as human labels",
        },
        "human_agreement": {
            "available": False,
            "reason": "no frame-addressable human FOV rejection labels exist in the archived review",
        },
        "thresholds": FOV_V2_THRESHOLDS,
        "positive_support_bounds_all_frames": support_bounds(pilot),
        "inside_frame_fraction_policy": {
            "threshold": None,
            "reason": "manual FOV PASS contains intentionally clipped target regions; retained only as diagnostic",
        },
        "pilot_replay": {"rows": len(pilot), "v2_passed": pilot_pass, "v2_failed": len(pilot) - pilot_pass},
        "expansion_v2_replay": {
            "rows": len(expansion), "passed": sum(bool(row.get("passed")) for row in expansion),
            "failed": sum(not bool(row.get("passed")) for row in expansion),
            "failure_reasons": dict(sorted(reasons.items())),
        },
        "freeze_status": "FROZEN_RULES_POSITIVE_SUPPORT_ONLY",
        "training_gate_status": "NOT_ELIGIBLE_UNTIL_NEGATIVE_HUMAN_CALIBRATION",
        "interpretation": "The deterministic rules are versioned and may be replayed, but no specificity/human-agreement claim is valid without a separately labelled FOV reject set.",
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "pilot_candidates": args.pilot_candidates, "pilot_reachability": args.pilot_reachability,
            "expansion_candidates": args.expansion_candidates, "expansion_reachability": args.expansion_reachability,
        }.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output_dir / "manual_positive_pilot_v2_replay.jsonl", pilot)
    atomic_write_jsonl(args.output_dir / "expansion_v2_replay.jsonl", expansion)
    atomic_write_json(args.output_dir / "calibration_report.json", report)
    with (args.output_dir / "SHA256SUMS").open("w") as handle:
        for path in sorted(args.output_dir.glob("*.json*")):
            if path.name != "SHA256SUMS":
                handle.write(f"{sha256(path)}  {path.name}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
