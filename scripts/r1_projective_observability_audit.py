#!/usr/bin/env python3
"""Render and apply projective_observability_v1 to stored action paths."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
from pathlib import Path
from typing import Any

import numpy as np

from r1_projective_observability import (
    PROJECTIVE_OBSERVABILITY_VERSION,
    evaluate_projective_path,
    save_path_contact_sheet,
    summarize_audits,
)
from r1_reachability_audit import atomic_write_json, atomic_write_jsonl, read_jsonl, render_task
from vagen.envs.active_spatial.collision_detector import create_collision_detector
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS


async def run(args: argparse.Namespace) -> dict[str, Any]:
    repaired = {
        row["task_id"]: row
        for row in read_jsonl(args.repaired)
        if row.get("task_id")
    }
    reachability = read_jsonl(args.reachability)
    selected_indices = {
        int(value) for value in (args.source_indices or "").split(",") if value.strip()
    }
    overrides = {}
    if args.collision_convention_overrides:
        payload = json.loads(args.collision_convention_overrides.read_text())
        overrides = payload.get("structure_y_sign_overrides", {})
    detector = create_collision_detector(
        {
            "camera_radius": 0.15,
            "floor_height": 0.3,
            "ceiling_height": 2.5,
            "safety_margin": 0.05,
            "enable_object_collision": True,
            "enable_boundary_collision": True,
            "structure_y_sign_overrides": overrides,
        }
    )
    audits = []
    for position, path_row in enumerate(reachability, start=1):
        source_index = path_row.get("source_row_index")
        if selected_indices and int(source_index) not in selected_indices:
            continue
        task_id = path_row.get("task_id")
        item = repaired.get(task_id)
        base = {
            "source_row_index": source_index,
            "task_id": task_id,
            "scene_id": path_row.get("scene_id"),
            "reachability_status": path_row.get("status"),
            "path_steps": path_row.get("steps"),
        }
        if item is None:
            # Reachability manifests retain diagnostics for candidates that were
            # later hard-failed or left unverified.  They are not accepted rows
            # and therefore are outside this post-acceptance observability gate.
            continue
        if item.get("task_type") != "projective_relations":
            continue
        if path_row.get("status") != "reachable" or not path_row.get("path"):
            audits.append(
                {
                    **base,
                    "version": PROJECTIVE_OBSERVABILITY_VERSION,
                    "passed": False,
                    "reasons": ["reachability_path_not_available"],
                }
            )
            continue
        scene_id = str(item["scene_id"])
        if not detector.load_scene_from_gs_root(str(args.gs_root), scene_id):
            audits.append(
                {
                    **base,
                    "version": PROJECTIVE_OBSERVABILITY_VERSION,
                    "passed": False,
                    "reasons": ["collision_scene_load_failed"],
                }
            )
            continue
        tasks = [
            render_task(item, np.asarray(entry["c2w"], dtype=float))
            for entry in path_row["path"]
        ]
        try:
            lock_handle = None
            try:
                if args.renderer_lock:
                    args.renderer_lock.parent.mkdir(parents=True, exist_ok=True)
                    lock_handle = args.renderer_lock.open("a+")
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                renderer = UnifiedRenderGS(
                    render_backend="http", client_url=args.renderer_url, scene_id=scene_id
                )
                images = await renderer.render_tasks(tasks)
            finally:
                if lock_handle is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    lock_handle.close()
            audit = evaluate_projective_path(
                item, path_row["path"], images, detector
            )
            saved = save_path_contact_sheet(
                images,
                path_row["path"],
                audit,
                args.output_dir / "path_rgb" / str(task_id),
            )
            audit.update(saved)
        except Exception as error:
            audit = {
                "version": PROJECTIVE_OBSERVABILITY_VERSION,
                "passed": False,
                "reasons": ["renderer_error"],
                "renderer_error": repr(error),
            }
        audit = {**base, **audit}
        audits.append(audit)
        atomic_write_jsonl(args.output_dir / "observability_checkpoint.jsonl", audits)
        print(
            json.dumps(
                {
                    "position": position,
                    "task_id": task_id,
                    "passed": audit["passed"],
                    "reasons": audit.get("reasons"),
                }
            ),
            flush=True,
        )

    summary = {
        **summarize_audits(audits),
        "renderer_url": args.renderer_url,
        "repaired": str(args.repaired),
        "reachability": str(args.reachability),
        "gs_root": str(args.gs_root),
        "collision_convention_overrides": (
            str(args.collision_convention_overrides)
            if args.collision_convention_overrides else None
        ),
    }
    atomic_write_jsonl(args.output_dir / "observability_manifest.jsonl", audits)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--reachability", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-indices")
    parser.add_argument("--collision-convention-overrides", type=Path)
    parser.add_argument("--renderer-lock", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
