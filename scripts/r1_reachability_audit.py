#!/usr/bin/env python3
"""Bounded action-space reachability audit for versioned Active Spatial R1 rows.

The search uses the same six primitive actions, pose updates, collision detector,
and primitive-step accounting as ``ActiveSpatialEnv``.  A row is accepted only
when a collision-free path reaches the versioned canonical task success gate in
at most ``max_steps``.  Exhaustion caps are reported as unverified, never as a
proof of infeasibility.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from r1_canonical_tasks import canonical_fov, canonical_projective, score_observation
from vagen.envs.active_spatial.collision_detector import create_collision_detector
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS
from vagen.envs.active_spatial.utils import ViewManipulator


ACTIONS = (
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "turn_left",
    "turn_right",
)
TRANSLATION_ACTIONS = frozenset(ACTIONS[:4])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def state_key(pose: np.ndarray) -> tuple[float, ...]:
    # ViewManipulator rotations are deterministic 20-degree increments.  Seven
    # decimals merge only floating-point aliases, not distinct action states.
    return tuple(np.round(np.concatenate((pose[:3, 3], pose[:3, :3].ravel())), 7))


def pose_record(pose: np.ndarray, action: str | None, metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": action,
        "c2w": pose.tolist(),
        "position": pose[:3, 3].tolist(),
        "forward": pose[:3, 2].tolist(),
        "canonical_success": bool(metric["success"]),
        "canonical_score": float(metric["score"]),
        "canonical_metric": metric,
    }


def target_distance(item: dict[str, Any], pose: np.ndarray) -> float:
    target = np.asarray(item.get("sample_target", item["target_region"].get("sample_point")), dtype=float)
    return float(np.linalg.norm(pose[:3, 3] - target[:3]))


def search_one(
    item: dict[str, Any],
    detector: Any,
    *,
    kind: str,
    max_steps: int,
    max_expansions: int,
    step_translation: float,
    step_rotation_deg: float,
) -> dict[str, Any]:
    metric_fn = canonical_fov if kind == "fov" else canonical_projective
    start = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
    start_collision = detector.check_collision(start[:3, 3])
    start_metric = metric_fn(score_observation(item, start))
    base = {
        "scene_id": item.get("scene_id"),
        "task_id": item.get("task_id"),
        "task_type": item.get("task_type"),
        "kind": kind,
        "max_steps": max_steps,
        "step_translation": step_translation,
        "step_rotation_deg": step_rotation_deg,
        "action_space": list(ACTIONS),
        "initial_collision": bool(start_collision.has_collision),
        "initial_collision_type": start_collision.collision_type,
        "initial_canonical_success": bool(start_metric["success"]),
        "collision_convention": detector.convention_record(),
        "search_algorithm": "bounded_best_first_not_shortest",
        "no_solution_depth_lower_bound": item.get("reachability_construction", {}).get(
            "no_solution_lower_bound"
        ),
    }
    if detector.structure_convention_status != "frozen":
        return {
            **base,
            "status": "reachability_unverified_collision_convention_ambiguous",
            "reachability_verified": False,
            "path": [],
            "steps": None,
            "expansions": 0,
            "collision_rejections": {},
        }
    if start_collision.has_collision:
        return {
            **base,
            "status": "unreachable_initial_collision",
            "reachability_verified": True,
            "path": [],
            "steps": None,
            "expansions": 0,
            "collision_rejections": {},
        }
    if start_metric["success"]:
        return {
            **base,
            "status": "invalid_initial_already_success",
            "reachability_verified": True,
            "path": [pose_record(start, None, start_metric)],
            "steps": 0,
            "expansions": 0,
            "collision_rejections": {},
        }

    # Bounded best-first search: canonical score and target distance guide the
    # search, while best_depth prevents a longer route from hiding a shorter
    # route to the same state.  A found route is valid but not claimed shortest.
    nodes: list[tuple[np.ndarray, int, str | None]] = [(start, -1, None)]
    queue: list[tuple[float, float, int, int]] = [
        (-float(start_metric["score"]), target_distance(item, start), 0, 0)
    ]
    best_depth = {state_key(start): 0}
    collisions: Counter[str] = Counter()
    expansions = 0
    deepest = 0
    engine = ViewManipulator(
        step_translation=step_translation,
        step_rotation_deg=step_rotation_deg,
        world_up_axis="Z",
    )

    while queue:
        _, _, depth, node_index = heapq.heappop(queue)
        if depth != best_depth.get(state_key(nodes[node_index][0])):
            continue
        deepest = max(deepest, depth)
        if depth >= max_steps:
            continue
        if expansions >= max_expansions:
            return {
                **base,
                "status": "reachability_unverified_expansion_cap",
                "reachability_verified": False,
                "path": [],
                "steps": None,
                "expansions": expansions,
                "visited_states": len(best_depth),
                "deepest_fully_or_partly_searched": deepest,
                "collision_rejections": dict(collisions),
            }
        pose = nodes[node_index][0]
        expansions += 1
        for action in ACTIONS:
            engine.reset(pose)
            candidate = engine.step(action)
            if action in TRANSLATION_ACTIONS:
                collision = detector.check_collision(candidate[:3, 3], previous_position=pose[:3, 3])
                if collision.has_collision:
                    collisions[collision.collision_type] += 1
                    continue
            key = state_key(candidate)
            new_depth = depth + 1
            if best_depth.get(key, max_steps + 1) <= new_depth:
                continue
            best_depth[key] = new_depth
            metric = metric_fn(score_observation(item, candidate))
            new_index = len(nodes)
            nodes.append((candidate, node_index, action))
            if metric["success"]:
                indices: list[int] = []
                cursor = new_index
                while cursor >= 0:
                    indices.append(cursor)
                    cursor = nodes[cursor][1]
                indices.reverse()
                path = []
                for i, index in enumerate(indices):
                    state_pose, _, state_action = nodes[index]
                    state_metric = metric_fn(score_observation(item, state_pose))
                    path.append(pose_record(state_pose, state_action if i else None, state_metric))
                return {
                    **base,
                    "status": "reachable",
                    "reachability_verified": True,
                    "path": path,
                    "actions": [entry["action"] for entry in path[1:]],
                    "steps": new_depth,
                    "found_path_length_upper_bound": new_depth,
                    "expansions": expansions,
                    "visited_states": len(best_depth),
                    "collision_rejections": dict(collisions),
                }
            heapq.heappush(
                queue,
                (-float(metric["score"]), target_distance(item, candidate), new_depth, new_index),
            )

    return {
        **base,
        "status": "unreachable_exhaustive_within_bound",
        "reachability_verified": True,
        "path": [],
        "steps": None,
        "expansions": expansions,
        "visited_states": len(best_depth),
        "deepest_fully_or_partly_searched": deepest,
        "collision_rejections": dict(collisions),
    }


def render_task(item: dict[str, Any], pose: np.ndarray) -> dict[str, Any]:
    result = score_observation(item, pose)
    camera = result["camera"]
    return {
        "mode": "cam_param",
        "intrinsics": camera["K_effective"],
        "extrinsics": camera["w2c"],
        "size": camera["render_resolution"],
    }


async def validate_path_images(
    rows: list[dict[str, Any]],
    repaired_by_id: dict[str, dict[str, Any]],
    renderer_url: str,
    min_rgb_std: float,
    max_consecutive_low_info: int,
    output_dir: Path,
) -> None:
    for row in rows:
        if row["status"] != "reachable":
            row["renderer_path_validation"] = {"status": "not_run_no_path"}
            continue
        item = repaired_by_id[row["task_id"]]
        renderer = UnifiedRenderGS(
            render_backend="http", client_url=renderer_url, scene_id=row["scene_id"]
        )
        tasks = [render_task(item, np.asarray(entry["c2w"], dtype=float)) for entry in row["path"]]
        images = await renderer.render_tasks(tasks)
        consecutive = 0
        maximum = 0
        frame_stats = []
        annotated_frames = []
        task_dir = output_dir / "path_rgb" / str(row["task_id"])
        task_dir.mkdir(parents=True, exist_ok=True)
        for step, image in enumerate(images):
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            std = float(array.std())
            low = std < min_rgb_std
            consecutive = consecutive + 1 if low else 0
            maximum = max(maximum, consecutive)
            action = row["path"][step].get("action") or "initial"
            frame_name = f"step_{step:02d}_{action}.png"
            frame_path = task_dir / frame_name
            image.convert("RGB").save(frame_path)
            annotated = image.convert("RGB").copy()
            draw = ImageDraw.Draw(annotated)
            draw.rectangle((0, 0, annotated.width - 1, 18), fill=(0, 0, 0))
            draw.text((4, 3), f"step={step} action={action}", fill=(255, 255, 0))
            annotated_frames.append(annotated)
            frame_stats.append(
                {
                    "step": step,
                    "action": action,
                    "rgb_std": std,
                    "low_information": low,
                    "image": str(frame_path.relative_to(output_dir)),
                }
            )
        sheet_path = None
        if annotated_frames:
            sheet = Image.new(
                "RGB",
                (
                    sum(frame.width for frame in annotated_frames),
                    max(frame.height for frame in annotated_frames),
                ),
                "white",
            )
            x = 0
            for frame in annotated_frames:
                sheet.paste(frame, (x, 0))
                x += frame.width
            sheet_path = task_dir / "path_contact_sheet.png"
            sheet.save(sheet_path)
        passed = len(images) == len(tasks) and maximum < max_consecutive_low_info
        row["renderer_path_validation"] = {
            "status": "passed" if passed else "failed_low_information_termination",
            "frames_expected": len(tasks),
            "frames_rendered": len(images),
            "min_rgb_std": min_rgb_std,
            "max_allowed_consecutive_low_info": max_consecutive_low_info - 1,
            "observed_max_consecutive_low_info": maximum,
            "frames": frame_stats,
            "contact_sheet": str(sheet_path.relative_to(output_dir)) if sheet_path else None,
        }
        if not passed:
            row["status"] = "reachability_unverified_renderer_termination"
            row["reachability_verified"] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("projective", "fov"), required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-expansions", type=int, default=250000)
    parser.add_argument("--step-translation", type=float, default=0.3)
    parser.add_argument("--step-rotation-deg", type=float, default=20.0)
    parser.add_argument("--renderer-url")
    parser.add_argument("--min-rgb-std", type=float, default=8.0)
    parser.add_argument("--max-consecutive-low-info", type=int, default=3)
    parser.add_argument("--source-indices", help="optional comma-separated source row indices")
    parser.add_argument("--collision-convention-overrides", type=Path)
    args = parser.parse_args()

    repaired = read_jsonl(args.repaired)
    mappings = read_jsonl(args.mapping)
    repaired_by_id = {row["task_id"]: row for row in repaired}
    selected_indices = {
        int(value) for value in (args.source_indices or "").split(",") if value.strip()
    }
    rows = []
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
    for mapping in mappings:
        if selected_indices and int(mapping["source_row_index"]) not in selected_indices:
            continue
        item = repaired_by_id[mapping["new_task_id"]]
        scene_id = item["scene_id"]
        if not detector.load_scene_from_gs_root(str(args.gs_root), scene_id):
            rows.append(
                {
                    "scene_id": scene_id,
                    "task_id": item["task_id"],
                    "kind": args.kind,
                    "status": "reachability_unverified_asset_load",
                    "reachability_verified": False,
                }
            )
            continue
        result = search_one(
            item,
            detector,
            kind=args.kind,
            max_steps=args.max_steps,
            max_expansions=args.max_expansions,
            step_translation=args.step_translation,
            step_rotation_deg=args.step_rotation_deg,
        )
        result["source_row_index"] = int(mapping["source_row_index"])
        rows.append(result)
        atomic_write_jsonl(args.output_dir / "search_checkpoint.jsonl", rows)
        print(json.dumps({k: result.get(k) for k in ("task_id", "status", "steps", "expansions")}))

    if args.renderer_url:
        asyncio.run(
            validate_path_images(
                rows,
                repaired_by_id,
                args.renderer_url,
                args.min_rgb_std,
                args.max_consecutive_low_info,
                args.output_dir,
            )
        )
    else:
        for row in rows:
            row["renderer_path_validation"] = {"status": "not_run"}
            if row.get("status") == "reachable":
                row["status"] = "reachability_unverified_renderer_not_run"
                row["reachability_verified"] = False

    counts = Counter(row["status"] for row in rows)
    accepted = sum(
        row["status"] == "reachable"
        and row.get("renderer_path_validation", {}).get("status") == "passed"
        for row in rows
    )
    summary = {
        "kind": args.kind,
        "rows": len(rows),
        "accepted": accepted,
        "status_counts": dict(counts),
        "all_reachable_and_renderer_validated": accepted == len(rows),
        "max_steps": args.max_steps,
        "max_expansions": args.max_expansions,
        "step_translation": args.step_translation,
        "step_rotation_deg": args.step_rotation_deg,
        "collision_semantics": "ActiveSpatial CollisionDetector radius=0.15 safety_margin=0.05",
        "collision_height_bounds": [0.3, 2.5],
        "search_algorithm": "bounded_best_first_not_shortest",
        "path_length_semantics": "found path is an upper bound; it is not claimed shortest",
        "termination_semantics": "canonical success; max primitive steps; renderer low-info early termination",
        "renderer_url": args.renderer_url,
    }
    atomic_write_jsonl(args.output_dir / "reachability_manifest.jsonl", rows)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
