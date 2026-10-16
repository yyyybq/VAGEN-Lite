#!/usr/bin/env python3
"""Run a frozen Medium selector in independent scene processes and merge it."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


VERSION = "r1_medium_selector_scene_shard_runner_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> tuple[str, int, str]:
    return str(row["split"]), int(row["source_row_index"]), str(row.get("requested_bucket", "medium"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector", choices=("v2", "v3", "v4_projection_frontier"), required=True)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--seed-cap", type=int, default=12)
    parser.add_argument("--per-seed-expansions", type=int, default=512)
    parser.add_argument("--candidate-cap-per-seed", type=int, default=48)
    parser.add_argument("--lower-expansions", type=int, default=100000)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for record in selection["records"]:
        by_scene[str(record["scene_id"])].append(record)
    selector_scripts = {
        "v2": "r1_projective_medium_selector_v2.py",
        "v3": "r1_projective_medium_selector_v3_depth_banded.py",
        "v4_projection_frontier": "r1_projective_medium_selector_v4_projection_frontier.py",
    }
    selector_script = Path(__file__).with_name(selector_scripts[args.selector])
    shards_dir = args.output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    def run_scene(scene_records: tuple[str, list[dict]]) -> dict:
        scene, records = scene_records
        shard = shards_dir / scene
        shard.mkdir(parents=True, exist_ok=True)
        expected = {(str(row["split"]), int(row["source_row_index"]), "medium") for row in records}
        result_path = shard / "selector_results.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text())
            if {key(row) for row in existing.get("results", [])} == expected:
                return {"scene_id": scene, "status": "resume_skip", "records": len(expected), "output": str(result_path)}
        keys = ",".join(f"{row['split']}:{int(row['source_row_index'])}" for row in records)
        command = [
            sys.executable, str(selector_script), "--selection", str(args.selection),
            "--sources", str(args.sources), "--gs-root", str(args.gs_root),
            "--output-dir", str(shard), "--source-keys", keys,
            "--seed-cap", str(args.seed_cap), "--per-seed-expansions", str(args.per_seed_expansions),
            "--candidate-cap-per-seed", str(args.candidate_cap_per_seed),
            "--lower-expansions", str(args.lower_expansions),
        ]
        with (shard / "selector.log").open("a") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"selector failed for {scene}: see {shard / 'selector.log'}")
        payload = json.loads(result_path.read_text())
        actual = {key(row) for row in payload["results"]}
        if actual != expected:
            raise ValueError(f"{scene}: expected {len(expected)} keys, got {len(actual)}")
        return {"scene_id": scene, "status": "completed", "records": len(actual), "output": str(result_path)}

    completion = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_scene, item): item[0] for item in sorted(by_scene.items())}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completion.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    results = []
    shard_lineage = []
    for scene in sorted(by_scene):
        path = shards_dir / scene / "selector_results.json"
        payload = json.loads(path.read_text())
        results.extend(payload["results"])
        shard_lineage.append({"scene_id": scene, "path": str(path.resolve()), "sha256": sha256(path),
                              "records": len(payload["results"])})
    results.sort(key=lambda row: (str(row["scene_id"]), str(row["split"]), int(row["source_row_index"])))
    expected_all = {(str(row["split"]), int(row["source_row_index"]), "medium") for row in selection["records"]}
    actual_all = {key(row) for row in results}
    if len(actual_all) != len(results) or actual_all != expected_all:
        raise ValueError("merged selector accounting does not match frozen selection")
    merged = {
        "version": VERSION,
        "selector_version": results[0].get("version") if results else None,
        "selection": str(args.selection.resolve()), "selection_sha256": sha256(args.selection),
        "same_pair_only": True,
        "budgets": {"seed_cap": args.seed_cap, "per_seed_expansions": args.per_seed_expansions,
                    "candidate_cap_per_seed": args.candidate_cap_per_seed,
                    "lower_expansions": args.lower_expansions},
        "runner": {"max_scene_workers": args.max_workers, "scene_switching": "one SceneConstraints process per scene"},
        "shards": shard_lineage,
        "results": results,
        "status_counts": dict(sorted(Counter(row["status"] for row in results).items())),
        "accounting": {"expected": len(expected_all), "actual": len(results), "unique": len(actual_all),
                       "closed": len(expected_all) == len(results) == len(actual_all)},
    }
    atomic_json(args.output_dir / "selector_results.json", merged)
    atomic_json(args.output_dir / "run_summary.json", {
        "version": VERSION, "selector": args.selector, "completion": completion,
        "status_counts": merged["status_counts"], "accounting": merged["accounting"],
    })
    print(json.dumps({"status_counts": merged["status_counts"], "accounting": merged["accounting"]}, sort_keys=True))


if __name__ == "__main__":
    main()
