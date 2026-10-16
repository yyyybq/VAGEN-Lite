#!/usr/bin/env python3
"""Materialize flat R1 audit fields required by the handoff checklist.

This is a non-destructive postprocessor: it reads the existing R1 H1 FOV and
projective pilot artifacts and writes augmented manifests to a new output
directory. It does not modify original train/eval JSONL files or the existing
renderer outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_AUDIT_ROOT = Path(
    "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/"
    "r1_observation_aligned_reward_audit"
)
DEFAULT_FOV_DIR = DEFAULT_AUDIT_ROOT / "r1_fov_h1_audit_20260902"
DEFAULT_PROJECTIVE_DIR = DEFAULT_AUDIT_ROOT / "r1_projective_repair_pilot_h1_v1"

FOV_REQUIRED = {
    "task_id",
    "source_lineage",
    "scene_id",
    "object_ids",
    "object_labels",
    "initial_pose",
    "sample_target_pose",
    "K_native",
    "K_effective",
    "native_resolution",
    "render_resolution",
    "camera_model_version",
    "bbox_raw",
    "bbox_clipped",
    "bbox_valid",
    "center_in_front",
    "inside_frame_fraction",
    "truncated",
    "out_of_frame",
    "tiny_object",
    "historical_score",
    "h1_canonical_score",
    "initial_success",
    "sample_target_success",
}

PROJECTIVE_REQUIRED = {
    "old_task_id",
    "new_task_id",
    "source_split",
    "scene_id",
    "object_a",
    "object_b",
    "relation",
    "old_normal",
    "new_normal",
    "old_sample_target",
    "new_sample_target",
    "old_canonical_score",
    "new_canonical_score",
    "relation_margin",
    "camera_model_version",
    "generator_version",
    "repair_reason",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def flatten_fov_row(row: dict[str, Any], source_lineage: str) -> dict[str, Any]:
    out = dict(row)
    h0_target = row.get("h0_target") or {}
    h1_initial = row.get("h1_initial") or {}
    h1_target = row.get("h1_target") or {}
    h1_camera = h1_target.get("camera") or {}
    h1_objects = h1_target.get("objects") or []

    out.setdefault("source_lineage", source_lineage)
    out.setdefault("initial_pose", row.get("initial_pose_c2w"))
    out.setdefault("sample_target_pose", row.get("sample_target_pose_c2w"))
    out.setdefault("K_effective", row.get("K_effective_h1") or h1_camera.get("K_effective"))
    out.setdefault("camera_model_version", h1_camera.get("camera_model_version"))
    out.setdefault("bbox_raw", [obj.get("bbox_raw") for obj in h1_objects])
    out.setdefault("bbox_clipped", [obj.get("bbox") for obj in h1_objects])
    out.setdefault("bbox_valid", h1_target.get("bbox_valid"))
    out.setdefault("center_in_front", h1_target.get("center_in_front"))
    out.setdefault("inside_frame_fraction", h1_target.get("inside_frame_fraction_min"))
    out.setdefault("truncated", h1_target.get("truncated"))
    out.setdefault("out_of_frame", h1_target.get("out_of_frame"))
    out.setdefault("tiny_object", h1_target.get("tiny_object"))
    out.setdefault("historical_score", h0_target.get("canonical_score"))
    out.setdefault("h1_canonical_score", h1_target.get("canonical_score"))
    out.setdefault("initial_success", h1_initial.get("success"))
    out.setdefault("sample_target_success", h1_target.get("success"))
    return out


def augment_projective_mapping(
    mapping_row: dict[str, Any],
    validation_row: dict[str, Any],
    source_split: str,
) -> dict[str, Any]:
    out = dict(mapping_row)
    old_vm = (validation_row.get("old_h1_target") or {}).get("visual_metrics") or {}
    new_vm = (validation_row.get("new_h1_target") or {}).get("visual_metrics") or {}

    out.setdefault("source_split", source_split)
    out.setdefault("old_canonical_score", old_vm.get("visual_score"))
    out.setdefault("new_canonical_score", new_vm.get("visual_score"))
    out.setdefault("relation_margin", new_vm.get("visual_relation_margin_px"))
    out.setdefault("old_relation_margin", old_vm.get("visual_relation_margin_px"))
    out.setdefault("new_relation_margin", new_vm.get("visual_relation_margin_px"))
    out.setdefault("old_relation_success", old_vm.get("visual_relation_satisfied"))
    out.setdefault("new_relation_success", new_vm.get("visual_relation_satisfied"))
    out.setdefault("old_sample_target_pose_c2w", validation_row.get("old_pose_c2w"))
    out.setdefault("new_sample_target_pose_c2w", validation_row.get("new_pose_c2w"))
    return out


def missing_required(rows: list[dict[str, Any]], required: set[str]) -> list[dict[str, Any]]:
    examples = []
    for idx, row in enumerate(rows[:20]):
        missing = sorted(key for key in required if key not in row or row.get(key) is None)
        if missing:
            examples.append({"row": idx, "missing": missing})
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fov-dir", type=Path, default=DEFAULT_FOV_DIR)
    parser.add_argument("--projective-dir", type=Path, default=DEFAULT_PROJECTIVE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUDIT_ROOT / "r1_required_manifest_fields_20260904",
    )
    parser.add_argument("--source-lineage", default="v46_baseline_qwen25vl_7b/train_filtered")
    parser.add_argument("--source-split", default="v46_baseline_qwen25vl_7b/train_filtered")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fov_rows = [
        flatten_fov_row(row, args.source_lineage)
        for row in read_jsonl(args.fov_dir / "r1_fov_h1_audit_manifest.jsonl")
    ]
    fov_out = args.output_dir / "r1_fov_h1_audit_manifest_required_fields_20260904.jsonl"
    write_jsonl(fov_out, fov_rows)

    mapping_rows = read_jsonl(args.projective_dir / "r1_projective_repaired_pilot_mapping_h1_v1.jsonl")
    validation_rows = read_jsonl(args.projective_dir / "r1_projective_repaired_pilot_validation_h1_v1.jsonl")
    validation_by_old_index = {row["old_dataset_index"]: row for row in validation_rows}
    augmented_mapping = [
        augment_projective_mapping(row, validation_by_old_index[row["old_dataset_index"]], args.source_split)
        for row in mapping_rows
    ]
    projective_out = args.output_dir / "r1_projective_repaired_pilot_mapping_required_fields_20260904.jsonl"
    write_jsonl(projective_out, augmented_mapping)

    summary = {
        "fov_rows": len(fov_rows),
        "fov_required_missing_examples": missing_required(fov_rows, FOV_REQUIRED),
        "projective_mapping_rows": len(augmented_mapping),
        "projective_required_missing_examples": missing_required(augmented_mapping, PROJECTIVE_REQUIRED),
        "projective_old_relation_success": sum(bool(r.get("old_relation_success")) for r in augmented_mapping),
        "projective_new_relation_success": sum(bool(r.get("new_relation_success")) for r in augmented_mapping),
        "outputs": {
            "fov_required_manifest": str(fov_out),
            "projective_required_mapping": str(projective_out),
        },
    }
    with (args.output_dir / "r1_required_manifest_fields_summary_20260904.json").open("w") as f:
        json.dump(summary, f, indent=2)

    report = [
        "# R1 Required Manifest Fields Materialization",
        "",
        "This is a supplemental, non-destructive artifact generated from the existing R1 H1 audit outputs.",
        "",
        f"- FOV rows: `{summary['fov_rows']}`",
        f"- FOV required-field missing examples: `{len(summary['fov_required_missing_examples'])}`",
        f"- Projective mapping rows: `{summary['projective_mapping_rows']}`",
        f"- Projective required-field missing examples: `{len(summary['projective_required_missing_examples'])}`",
        f"- Projective old relation success: `{summary['projective_old_relation_success']}/{summary['projective_mapping_rows']}`",
        f"- Projective new relation success: `{summary['projective_new_relation_success']}/{summary['projective_mapping_rows']}`",
    ]
    (args.output_dir / "r1_required_manifest_fields_report_20260904.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
