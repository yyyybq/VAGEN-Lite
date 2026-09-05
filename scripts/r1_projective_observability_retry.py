#!/usr/bin/env python3
"""Bounded alternate-pose repair for Projective observability failures."""

from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import json
from pathlib import Path
from typing import Any

import numpy as np

from r1_full_regeneration import (
    atomic_json, atomic_jsonl, audit_reachability_tiered, prepare_replacement,
    read_jsonl, split_constraint,
)
from r1_projective_observability import evaluate_projective_path, save_path_contact_sheet
from r1_reachability_audit import render_task
from r1_repair_pipeline import SceneConstraints, repair_projective
from vagen.envs.active_spatial.collision_detector import create_collision_detector
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS


VERSION = "projective_observability_alternate_pose_retry_v1"


async def main_async(args: argparse.Namespace) -> None:
    source_paths = json.loads(args.sources.read_text())
    all_rows = {split: read_jsonl(Path(path)) for split, path in source_paths.items()}
    source_rows = all_rows[args.split]
    train_scenes = {str(row.get("scene_id")) for row in all_rows["train"]}
    train_labels = {str(row.get("object_label")) for row in all_rows["train"]}
    repaired = {int(row["repair_lineage"]["source_row_index"]): row for row in read_jsonl(args.repaired)}
    mappings = {int(row["source_row_index"]): row for row in read_jsonl(args.mapping)}
    failed = [
        row for row in read_jsonl(args.observability)
        if not row.get("passed") and row.get("reasons") != ["renderer_error"]
    ]
    failed = [
        row for position, row in enumerate(failed)
        if position % args.failure_shard_count == args.failure_shard_id
    ]
    override_payload = json.loads(args.collision_convention_overrides.read_text())
    overrides = override_payload.get("structure_y_sign_overrides", {})
    constraints = SceneConstraints(args.gs_root, structure_y_sign_overrides=overrides)
    planner_detector = create_collision_detector({
        "camera_radius": 0.15, "floor_height": 0.3, "ceiling_height": 2.5,
        "safety_margin": 0.05, "enable_object_collision": True,
        "enable_boundary_collision": True, "structure_y_sign_overrides": overrides,
    })
    frame_detector = create_collision_detector({
        "camera_radius": 0.15, "floor_height": 0.3, "ceiling_height": 2.5,
        "safety_margin": 0.05, "enable_object_collision": True,
        "enable_boundary_collision": True, "structure_y_sign_overrides": overrides,
    })
    offsets = [int(value) for value in args.selection_offsets.split(",") if value]
    results = []
    for failure in failed:
        index = int(failure["source_row_index"])
        old_mapping = mappings[index]
        original_source = source_rows[index]
        lineage = old_mapping.get("replacement_lineage")
        if lineage:
            base = prepare_replacement(
                all_rows[str(lineage["donor_split"])][int(lineage["donor_source_row_index"])],
                args.split,
            )
        else:
            base = original_source
        attempts = []
        selected = None
        for offset in offsets:
            candidate, details = repair_projective(index, base, constraints, selection_offset=offset)
            attempt = {"selection_offset": offset, "generator_failure": details.get("failure")}
            if candidate is None:
                attempts.append(attempt)
                continue
            valid, reason = split_constraint(
                args.split, original_source, candidate, train_scenes, train_labels
            )
            attempt["split_constraint"] = reason
            if not valid:
                attempts.append(attempt)
                continue
            if lineage:
                candidate.setdefault("repair_lineage", {})["replacement"] = lineage
                details["replacement_lineage"] = lineage
            reach = audit_reachability_tiered(
                candidate, planner_detector, args.gs_root, 12, [2000, 25000],
                250000, False, "observability_retry_no_250k",
            )
            reach["source_row_index"] = index
            attempt["reachability_status"] = reach.get("status")
            attempt["reachability_tiers"] = reach.get("reachability_tiers")
            if reach.get("status") != "reachable":
                attempts.append(attempt)
                continue
            scene_id = str(candidate["scene_id"])
            frame_detector.load_scene_from_gs_root(str(args.gs_root), scene_id)
            lock_handle = None
            try:
                if args.renderer_lock:
                    args.renderer_lock.parent.mkdir(parents=True, exist_ok=True)
                    lock_handle = args.renderer_lock.open("a+")
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                # The renderer service owns mutable scene state.  Hold the lock
                # across both scene selection and rendering so parallel CPU
                # candidate searches cannot race at this boundary.
                renderer = UnifiedRenderGS(
                    render_backend="http", client_url=args.renderer_url, scene_id=scene_id
                )
                images = await renderer.render_tasks([
                    render_task(candidate, np.asarray(entry["c2w"], dtype=float))
                    for entry in reach["path"]
                ])
            finally:
                if lock_handle is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    lock_handle.close()
            audit = evaluate_projective_path(candidate, reach["path"], images, frame_detector)
            saved = save_path_contact_sheet(
                images, reach["path"], audit,
                args.output_dir / "path_rgb" / f"{index:06d}_offset_{offset:03d}",
            )
            audit.update(saved)
            audit.update({"source_row_index": index, "task_id": candidate.get("task_id"), "scene_id": scene_id})
            attempt["observability_passed"] = audit.get("passed")
            attempt["observability_reasons"] = audit.get("reasons")
            attempts.append(attempt)
            if audit.get("passed"):
                mapping = {
                    **details,
                    "split": args.split,
                    "r1_status": old_mapping.get("r1_status"),
                    "split_constraint": reason,
                    "reachability_status": "reachable",
                    "found_path_length_upper_bound": reach.get("steps"),
                    "observability_retry_version": VERSION,
                    "observability_retry_from_task_id": repaired[index].get("task_id"),
                }
                selected = {"row": candidate, "mapping": mapping, "reachability": reach, "observability": audit}
                break
        results.append({
            "version": VERSION,
            "split": args.split,
            "source_row_index": index,
            "old_task_id": repaired[index].get("task_id"),
            "recovered": selected is not None,
            "attempts": attempts,
            "selected": selected,
        })
        atomic_jsonl(args.output_dir / "retry_checkpoint.jsonl", results)
        print(json.dumps({"split": args.split, "index": index, "recovered": selected is not None, "attempts": len(attempts)}), flush=True)
    summary = {
        "version": VERSION, "split": args.split, "input_failures": len(failed),
        "recovered": sum(row["recovered"] for row in results),
        "still_failed": sum(not row["recovered"] for row in results),
        "selection_offsets": offsets,
        "failure_shard": {"count": args.failure_shard_count, "id": args.failure_shard_id},
    }
    atomic_jsonl(args.output_dir / "retry_manifest.jsonl", results)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--observability", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--renderer-lock", type=Path)
    parser.add_argument("--collision-convention-overrides", type=Path, required=True)
    parser.add_argument("--selection-offsets", default="32,64,95,16,8,4,1")
    parser.add_argument("--failure-shard-count", type=int, default=1)
    parser.add_argument("--failure-shard-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.failure_shard_count < 1 or not 0 <= args.failure_shard_id < args.failure_shard_count:
        raise ValueError("invalid failure shard")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
