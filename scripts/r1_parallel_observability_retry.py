#!/usr/bin/env python3
"""Run independent observability-retry shards in parallel with a renderer lock."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--observability-shards", type=Path, required=True)
    parser.add_argument("--repair-shards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--renderer-lock", type=Path, required=True)
    parser.add_argument("--collision-convention-overrides", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--only", help="optional comma-separated scene:split shard keys")
    args = parser.parse_args()

    selected = {value for value in (args.only or "").split(",") if value}
    jobs = []
    for summary in sorted(args.observability_shards.glob("*/*/summary.json")):
        split = summary.parent.name
        scene = summary.parent.parent.name
        if selected and f"{scene}:{split}" not in selected:
            continue
        output = args.output_dir / scene / split
        if (output / "summary.json").is_file():
            continue
        jobs.append((scene, split, summary.parent, output))

    def run_one(job):
        scene, split, observation, output = job
        output.mkdir(parents=True, exist_ok=True)
        shard = args.repair_shards / scene / split
        command = [
            sys.executable,
            str(Path(__file__).with_name("r1_projective_observability_retry.py")),
            "--sources", str(args.sources),
            "--split", split,
            "--repaired", str(shard / "trainable.jsonl"),
            "--mapping", str(shard / "mapping.jsonl"),
            "--observability", str(observation / "observability_manifest.jsonl"),
            "--gs-root", str(args.gs_root),
            "--renderer-url", args.renderer_url,
            "--renderer-lock", str(args.renderer_lock),
            "--collision-convention-overrides", str(args.collision_convention_overrides),
            "--output-dir", str(output),
        ]
        with (output / "job.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"retry shard failed: {scene}/{split}; see {output / 'job.log'}")
        return {"scene": scene, "split": split, "output": str(output)}

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
