#!/usr/bin/env python3
"""Deterministic renderer-only benchmark for one PPO-sized fixed batch.

This script never imports an actor/critic, constructs an optimizer, or mutates
training state.  It warms exactly twelve scenes, then measures 48 logical
trajectories (four per scene) with eight sequential observations each.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import statistics
import time
from pathlib import Path
from typing import Any


SCENES = 12
ROLLOUTS_PER_SCENE = 4
TURNS_PER_TRAJECTORY = 8
EXPECTED_REQUESTS = SCENES * ROLLOUTS_PER_SCENE * TURNS_PER_TRAJECTORY


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def load_fixed_scenes(manifest: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        scene_id = str(row["scene_id"])
        if scene_id in seen:
            continue
        seen.add(scene_id)
        rows.append(row)
        if len(rows) == SCENES:
            break
    if len(rows) != SCENES:
        raise ValueError(f"need {SCENES} distinct scenes, found {len(rows)}")
    return rows


def render_task(row: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from vagen.envs.active_spatial.env import runtime_render_camera_parameters

    c2w = np.asarray(row["init_camera"]["extrinsics"], dtype=np.float64)
    native_k = np.asarray(row["init_camera"]["intrinsics"], dtype=np.float64)
    intrinsics, world_to_camera = runtime_render_camera_parameters(row, c2w, native_k, (256, 256))
    return {
        "mode": "cam_param",
        "intrinsics": intrinsics.tolist(),
        "extrinsics": world_to_camera.tolist(),
        "size": [256, 256],
    }


def image_digest(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return sha256_bytes(buffer.getvalue())


async def one_request(endpoint: str, scene_id: str, task: dict[str, Any]) -> tuple[float, str]:
    from vagen.envs.active_spatial.render.http_render_client import InteriorGSHTTPRenderClient

    client = InteriorGSHTTPRenderClient(endpoint, timeout=900, retries=0)
    started = time.perf_counter()
    images = await client.render(scene_id, [task])
    elapsed = time.perf_counter() - started
    if len(images) != 1 or images[0].size != (256, 256):
        raise RuntimeError(f"unexpected response for {scene_id}: count={len(images)}")
    return elapsed, image_digest(images[0])


async def warm_scenes(endpoint: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    results = await asyncio.gather(
        *(one_request(endpoint, str(row["scene_id"]), render_task(row)) for row in rows)
    )
    return {
        "wall_seconds": time.perf_counter() - started,
        "scene_hashes": {str(row["scene_id"]): digest for row, (_latency, digest) in zip(rows, results)},
        "latencies_seconds": [latency for latency, _digest in results],
    }


async def run_fixed_batch(endpoint: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    async def trajectory(row: dict[str, Any], replica: int) -> dict[str, Any]:
        scene_id = str(row["scene_id"])
        task = render_task(row)
        latencies: list[float] = []
        digests: list[str] = []
        for _turn in range(TURNS_PER_TRAJECTORY):
            latency, digest = await one_request(endpoint, scene_id, task)
            latencies.append(latency)
            digests.append(digest)
        return {"scene_id": scene_id, "replica": replica, "latencies": latencies, "digests": digests}

    started = time.perf_counter()
    trajectories = await asyncio.gather(
        *(trajectory(row, replica) for row in rows for replica in range(ROLLOUTS_PER_SCENE))
    )
    wall = time.perf_counter() - started
    latencies = [value for trajectory_result in trajectories for value in trajectory_result["latencies"]]
    ordered_digests = [value for trajectory_result in trajectories for value in trajectory_result["digests"]]
    return {
        "wall_seconds": wall,
        "requests": len(latencies),
        "requests_per_second": len(latencies) / wall,
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "ordered_png_digest_sha256": sha256_bytes("\n".join(ordered_digests).encode("ascii")),
        "unique_png_hashes": sorted(set(ordered_digests)),
        "trajectories": len(trajectories),
        "turns_per_trajectory": TURNS_PER_TRAJECTORY,
    }


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_fixed_scenes(args.manifest)
    warmup = await warm_scenes(args.endpoint, rows)
    measured = await run_fixed_batch(args.endpoint, rows)
    checks = {
        "twelve_distinct_scenes": len(warmup["scene_hashes"]) == SCENES,
        "ppo_sized_48_trajectories": measured["trajectories"] == SCENES * ROLLOUTS_PER_SCENE,
        "expected_request_count": measured["requests"] == EXPECTED_REQUESTS,
        "wall_within_target": measured["wall_seconds"] <= args.max_wall_seconds,
        "throughput_meets_target": measured["requests_per_second"] >= EXPECTED_REQUESTS / args.max_wall_seconds,
        "no_training_model_loaded": True,
        "optimizer_step_not_called": True,
    }
    return {
        "schema_version": "active_spatial_renderer_microbenchmark_v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "topology": {"gpu_count": args.gpu_count, "max_workers": 12, "max_inflight": 12},
        "fixed_batch": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_bytes(args.manifest.read_bytes()),
            "scene_ids": [str(row["scene_id"]) for row in rows],
            "scenes": SCENES,
            "rollouts_per_scene": ROLLOUTS_PER_SCENE,
            "turns_per_trajectory": TURNS_PER_TRAJECTORY,
            "requests": EXPECTED_REQUESTS,
            "image_size": [256, 256],
        },
        "target": {
            "renderer_wall_seconds": args.max_wall_seconds,
            "minimum_requests_per_second": EXPECTED_REQUESTS / args.max_wall_seconds,
            "relation_to_training_target": "renderer-only budget within the 600-second full-step target",
        },
        "warmup": warmup,
        "measured": measured,
        "checks": checks,
        "safety": {
            "actor_loaded": False,
            "critic_loaded": False,
            "optimizer_constructed": False,
            "optimizer_step_called": False,
            "checkpoint_written": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gpu-count", required=True, type=int, choices=(1, 2, 4))
    parser.add_argument("--max-wall-seconds", type=float, default=300.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(async_main(args))
    except BaseException as exc:
        report = {
            "schema_version": "active_spatial_renderer_microbenchmark_v1",
            "status": "BLOCKED",
            "topology": {"gpu_count": args.gpu_count, "max_workers": 12, "max_inflight": 12},
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "safety": {"optimizer_step_called": False, "checkpoint_written": False},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
