#!/usr/bin/env python3
"""Render and apply FOV-v2 RGB integrity/observability without touching v1."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
from pathlib import Path
from typing import Any

import numpy as np

from r1_fov_observability import (
    FOV_OBSERVABILITY_V2_VERSION,
    evaluate_fov_path_v2,
    save_path_contact_sheet,
)
from r1_reachability_audit import atomic_write_json, atomic_write_jsonl, read_jsonl, render_task
from vagen.envs.active_spatial.collision_detector import create_collision_detector
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS


async def run(args: argparse.Namespace) -> dict[str, Any]:
    repaired = {row["task_id"]: row for row in read_jsonl(args.repaired) if row.get("task_id")}
    paths = read_jsonl(args.reachability)
    detector = create_collision_detector({
        "camera_radius": 0.15, "floor_height": 0.3, "ceiling_height": 2.5,
        "safety_margin": 0.05, "enable_object_collision": True, "enable_boundary_collision": True,
    })
    audits: list[dict[str, Any]] = []
    for position, path_row in enumerate(paths, start=1):
        task_id = str(path_row.get("task_id") or "")
        item = repaired.get(task_id)
        if item is None or item.get("task_type") != "fov_inclusion":
            continue
        base = {"source_row_index": path_row.get("source_row_index"), "task_id": task_id,
                "scene_id": path_row.get("scene_id"), "reachability_status": path_row.get("status"),
                "path_steps": path_row.get("steps")}
        if path_row.get("status") != "reachable" or not path_row.get("path"):
            audits.append({**base, "version": FOV_OBSERVABILITY_V2_VERSION, "passed": False,
                           "reasons": ["reachability_path_not_available"]})
            continue
        scene_id = str(item["scene_id"])
        if not detector.load_scene_from_gs_root(str(args.gs_root), scene_id):
            audits.append({**base, "version": FOV_OBSERVABILITY_V2_VERSION, "passed": False,
                           "reasons": ["collision_scene_load_failed"]})
            continue
        tasks = [render_task(item, np.asarray(entry["c2w"], dtype=float)) for entry in path_row["path"]]
        try:
            lock_handle = None
            try:
                if args.renderer_lock:
                    args.renderer_lock.parent.mkdir(parents=True, exist_ok=True)
                    lock_handle = args.renderer_lock.open("a+")
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                renderer = UnifiedRenderGS(render_backend="http", client_url=args.renderer_url, scene_id=scene_id)
                images = await renderer.render_tasks(tasks)
            finally:
                if lock_handle is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    lock_handle.close()
            audit = evaluate_fov_path_v2(item, path_row["path"], images, detector)
            audit.update(save_path_contact_sheet(images, path_row["path"], audit,
                                                 args.output_dir / "path_rgb" / task_id))
        except Exception as error:
            audit = {"version": FOV_OBSERVABILITY_V2_VERSION, "passed": False,
                     "reasons": ["renderer_error"], "renderer_error": repr(error)}
        audits.append({**base, **audit})
        atomic_write_jsonl(args.output_dir / "observability_checkpoint.jsonl", audits)
        print(json.dumps({"position": position, "task_id": task_id, "passed": audit["passed"],
                          "reasons": audit.get("reasons")}), flush=True)
    failures: dict[str, int] = {}
    for audit in audits:
        for reason in audit.get("reasons") or []:
            failures[reason] = failures.get(reason, 0) + 1
    summary = {"version": FOV_OBSERVABILITY_V2_VERSION, "rows": len(audits),
               "passed": sum(bool(audit.get("passed")) for audit in audits),
               "failed": sum(not bool(audit.get("passed")) for audit in audits),
               "failure_reasons": dict(sorted(failures.items())), "renderer_url": args.renderer_url,
               "repaired": str(args.repaired), "reachability": str(args.reachability), "gs_root": str(args.gs_root)}
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
    parser.add_argument("--renderer-lock", type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
