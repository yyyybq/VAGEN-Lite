#!/usr/bin/env python3
"""Render one immutable source pose to prove a staged scene is renderer-ready."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from r1_reachability_audit import render_task
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def run(args: argparse.Namespace) -> None:
    row = read_jsonl(args.source)[args.source_row_index]
    if str(row.get("scene_id")) != args.scene_id:
        raise ValueError(f"source {args.source_row_index} is not in scene {args.scene_id}")
    pose = np.asarray(row["init_camera"]["extrinsics"], dtype=float)
    task = render_task(row, pose)
    renderer = UnifiedRenderGS(render_backend="http", client_url=args.renderer_url, scene_id=args.scene_id)
    images = await renderer.render_tasks([task])
    if len(images) != 1:
        raise RuntimeError(f"expected one smoke image, got {len(images)}")
    image = images[0].convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    if not np.isfinite(array).all() or float(array.std()) < 1.0:
        raise RuntimeError("renderer smoke produced blank or non-finite RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output.with_suffix(".png"))
    image_sha = hashlib.sha256(args.output.with_suffix(".png").read_bytes()).hexdigest()
    payload = {"scene_id": args.scene_id, "source_row_index": args.source_row_index,
               "image": str(args.output.with_suffix(".png")), "image_sha256": image_sha,
               "shape": list(array.shape), "mean": float(array.mean()), "std": float(array.std())}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-row-index", type=int, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
