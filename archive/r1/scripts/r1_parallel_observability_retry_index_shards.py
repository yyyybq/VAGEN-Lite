#!/usr/bin/env python3
"""Run deterministic index-sharded retries for selected observability failures.

The retry generator samples candidate poses.  Splitting by the stable ordering of
the failed source indices makes its expensive renderer work resumable without
letting simultaneous jobs overwrite a common checkpoint or output manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def parse_spec(value: str) -> tuple[str, str, int]:
    try:
        scene, split, count = value.split(":")
        count_i = int(count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("spec must be scene:split:part_count") from exc
    if count_i < 1:
        raise argparse.ArgumentTypeError("part_count must be positive")
    return scene, split, count_i


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--observability-shards", type=Path, required=True)
    parser.add_argument("--repair-shards", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--renderer-lock", type=Path, required=True)
    parser.add_argument("--collision-convention-overrides", type=Path, required=True)
    parser.add_argument("--spec", action="append", type=parse_spec, required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    args = parser.parse_args()

    jobs = []
    for scene, split, part_count in args.spec:
        observation = args.observability_shards / scene / split
        repair = args.repair_shards / scene / split
        if not (observation / "observability_manifest.jsonl").is_file():
            raise FileNotFoundError(observation / "observability_manifest.jsonl")
        for part_id in range(part_count):
            output = args.parts_root / scene / split / f"part_{part_id:02d}"
            if (output / "summary.json").is_file():
                continue
            jobs.append((scene, split, part_count, part_id, observation, repair, output))

    def run_one(job):
        scene, split, part_count, part_id, observation, repair, output = job
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("r1_projective_observability_retry.py")),
            "--sources", str(args.sources),
            "--split", split,
            "--repaired", str(repair / "trainable.jsonl"),
            "--mapping", str(repair / "mapping.jsonl"),
            "--observability", str(observation / "observability_manifest.jsonl"),
            "--gs-root", str(args.gs_root),
            "--renderer-url", args.renderer_url,
            "--renderer-lock", str(args.renderer_lock),
            "--collision-convention-overrides", str(args.collision_convention_overrides),
            "--failure-shard-count", str(part_count),
            "--failure-shard-id", str(part_id),
            "--output-dir", str(output),
        ]
        with (output / "job.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"retry part failed: {scene}/{split}/part_{part_id:02d}")
        return {"scene": scene, "split": split, "part": part_id, "parts": part_count}

    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed.append(row)
            print(json.dumps({"completed": row, "count": len(completed), "total": len(jobs)}), flush=True)
    print(json.dumps({"jobs": len(jobs), "completed": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
