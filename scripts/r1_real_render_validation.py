#!/usr/bin/env python3
"""Run paired real-render validation for versioned R1 repair artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from r1_canonical_tasks import (
    CANONICAL_CAMERA_H1_RESIZE_V1,
    CANONICAL_TASK_METRIC_VERSION,
    canonical_fov,
    canonical_projective,
    pose_from_item_target,
    score_observation,
)
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def image_stats(image: Image.Image) -> dict[str, float]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def overlay(image: Image.Image, result: dict[str, Any], title: str) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    objects = result["visual_metrics"].get("objects") or []
    colors = ((255, 60, 60), (60, 220, 255))
    for index, obj in enumerate(objects):
        color = colors[index % len(colors)]
        box = obj.get("bbox")
        normalized_box = None
        if box and len(box) == 4 and all(math.isfinite(float(value)) for value in box):
            x0, x1 = sorted((float(box[0]), float(box[2])))
            y0, y1 = sorted((float(box[1]), float(box[3])))
            normalized_box = (x0, y0, x1, y1)
            draw.rectangle(normalized_box, outline=color, width=3)
        center = obj.get("center_uv")
        if center and len(center) >= 2 and all(math.isfinite(float(value)) for value in center[:2]):
            u, v = float(center[0]), float(center[1])
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=color)
        label = str(obj.get("label") or obj.get("id") or f"object_{index}")
        anchor = (normalized_box[0], max(18.0, normalized_box[1])) if normalized_box else (4.0, 22.0 + index * 14.0)
        draw.text(anchor, label, fill=color)
    draw.rectangle((0, 0, output.width - 1, output.height - 1), outline=(255, 255, 0), width=1)
    draw.rectangle((0, 0, output.width - 1, 18), fill=(0, 0, 0))
    draw.text((4, 3), title, fill=(255, 255, 0))
    return output


def render_task(result: dict[str, Any]) -> dict[str, Any]:
    camera = result["camera"]
    return {
        "mode": "cam_param",
        "intrinsics": camera["K_effective"],
        "extrinsics": camera["w2c"],
        "size": camera["render_resolution"],
    }


async def render_chunks(
    renderer: UnifiedRenderGS,
    tasks: list[dict[str, Any]],
    chunk_size: int,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    for offset in range(0, len(tasks), chunk_size):
        images.extend(await renderer.render_tasks(tasks[offset : offset + chunk_size]))
    return images


async def run(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = read_jsonl(args.source)
    repaired_rows = read_jsonl(args.repaired)
    mappings = read_jsonl(args.mapping)
    repaired_by_id = {row.get("task_id"): row for row in repaired_rows}
    selected = [row for row in mappings if row.get("scene_id") == args.scene_id][: args.limit]
    if not selected:
        raise ValueError(f"no mapping rows for scene {args.scene_id}")

    render_results: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    row_specs: list[list[tuple[str, dict[str, Any], dict[str, Any]]]] = []
    metric_fn = canonical_fov if args.kind == "fov" else canonical_projective
    for mapping in selected:
        source_index = int(mapping["source_row_index"])
        old = source_rows[source_index]
        new = repaired_by_id[mapping["new_task_id"]]
        poses = (
            ("old_init", old, np.asarray(old["init_camera"]["extrinsics"], dtype=float)),
            ("new_init", new, np.asarray(new["init_camera"]["extrinsics"], dtype=float)),
            ("old_target", old, pose_from_item_target(old)),
            ("new_target", new, pose_from_item_target(new)),
        )
        spec = []
        for label, item, pose in poses:
            result = score_observation(item, pose)
            metric = metric_fn(result)
            spec.append((label, result, metric))
            tasks.append(render_task(result))
        row_specs.append(spec)

    renderer = UnifiedRenderGS(render_backend="http", client_url=args.renderer_url, scene_id=args.scene_id)
    images = await render_chunks(renderer, tasks, args.chunk_size)
    expected_images = sum(len(spec) for spec in row_specs)
    if len(images) != expected_images:
        raise RuntimeError(f"renderer returned {len(images)} images, expected {expected_images}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_cursor = 0
    contact_rows = []
    for row_index, (mapping, spec) in enumerate(zip(selected, row_specs)):
        panels = []
        manifest_panels = []
        for label, result, metric in spec:
            image = images[image_cursor]
            image_cursor += 1
            stats = image_stats(image)
            if args.kind == "projective":
                detail = f"margin={metric.get('relation_margin_px')}"
            else:
                detail = f"inside={metric.get('inside_frame_fraction_min')}"
            panel = overlay(image, result, f"{label} success={metric['success']} {detail}")
            panels.append(panel)
            manifest_panels.append(
                {
                    "label": label,
                    "metric": metric,
                    "camera": result["camera"],
                    "visual_metrics": result["visual_metrics"],
                    "rgb_stats": stats,
                    "rgb_nonblank": stats["std"] >= args.min_rgb_std,
                }
            )
        paired = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
        x = 0
        for panel in panels:
            paired.paste(panel, (x, 0))
            x += panel.width
        image_name = f"{row_index:03d}_{args.scene_id}_{args.kind}_paired.png"
        paired.save(args.output_dir / image_name)
        contact_rows.append(paired.resize((paired.width // 2, paired.height // 2)))
        render_results.append(
            {
                "row_index": row_index,
                "source_row_index": int(mapping["source_row_index"]),
                "old_task_id": mapping.get("old_task_id"),
                "new_task_id": mapping.get("new_task_id"),
                "scene_id": args.scene_id,
                "kind": args.kind,
                "paired_image": image_name,
                "initial_constraints": mapping.get("initial_constraints"),
                "target_constraints": mapping.get("target_constraints"),
                "navigation_distance": mapping.get("navigation_distance"),
                "panels": manifest_panels,
            }
        )

    contact_width = max(image.width for image in contact_rows)
    contact_height = sum(image.height for image in contact_rows)
    sheet = Image.new("RGB", (contact_width, contact_height), "white")
    y = 0
    for image in contact_rows:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(args.output_dir / "contact_sheet.png")
    with (args.output_dir / "manifest.jsonl").open("w") as handle:
        for row in render_results:
            handle.write(json.dumps(row) + "\n")

    panel_counts: dict[str, dict[str, int]] = {}
    for row in render_results:
        for panel in row["panels"]:
            counts = panel_counts.setdefault(panel["label"], {"rows": 0, "canonical_success": 0, "rgb_nonblank": 0})
            counts["rows"] += 1
            counts["canonical_success"] += int(bool(panel["metric"]["success"]))
            counts["rgb_nonblank"] += int(bool(panel["rgb_nonblank"]))
    summary = {
        "kind": args.kind,
        "scene_id": args.scene_id,
        "rows": len(render_results),
        "images": expected_images,
        "panel_counts": panel_counts,
        "camera_model_version": CANONICAL_CAMERA_H1_RESIZE_V1,
        "canonical_task_metric_version": CANONICAL_TASK_METRIC_VERSION,
        "renderer_url": args.renderer_url,
        "source": str(args.source),
        "repaired": str(args.repaired),
        "mapping": str(args.mapping),
        "render_resolution": [256, 256],
        "min_rgb_std": args.min_rgb_std,
        "acceptance_gates": {
            "new_initial_all_unsuccessful": panel_counts.get("new_init", {}).get("canonical_success", 0) == 0,
            "new_target_all_successful": panel_counts.get("new_target", {}).get("canonical_success", 0) == len(render_results),
            "new_panels_rgb_nonblank": all(
                panel["rgb_nonblank"]
                for row in render_results
                for panel in row["panels"]
                if panel["label"] in {"new_init", "new_target"}
            ),
            "historical_panels_rgb_nonblank_diagnostic_only": all(
                panel["rgb_nonblank"]
                for row in render_results
                for panel in row["panels"]
                if panel["label"] in {"old_init", "old_target"}
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("fov", "projective"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--min-rgb-std", type=float, default=5.0)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
