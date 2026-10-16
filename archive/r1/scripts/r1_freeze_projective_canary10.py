#!/usr/bin/env python3
"""Freeze the Projective-only source inventory for the formal ten-scene canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relation(row: dict[str, Any]) -> str | None:
    return row.get("target_region", {}).get("params", {}).get("relation")


def object_pair(row: dict[str, Any]) -> list[str]:
    objects = row.get("target_object", {}).get("objects") or []
    return [f"{obj.get('id')}:{obj.get('label')}" for obj in objects]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--baseline-accounting-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", required=True, help="comma-separated formal canary scene IDs")
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    if len(scenes) != len(set(scenes)):
        raise ValueError("duplicate scene in canary list")
    scene_set = set(scenes)
    source_map = json.loads(args.sources.read_text())
    old: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(args.baseline_accounting_root.glob("*/*/accounting.jsonl")):
        for row in read_jsonl(path):
            key = (str(row["split"]), int(row["source_row_index"]))
            if key in old:
                raise ValueError(f"duplicate old accounting {key}")
            old[key] = row
    records: list[dict[str, Any]] = []
    source_sha = {}
    for split, raw_path in source_map.items():
        path = Path(raw_path)
        source_sha[split] = {"path": str(path), "sha256": sha256(path)}
        for index, row in enumerate(read_jsonl(path)):
            if row.get("task_type") != "projective_relations" or row.get("scene_id") not in scene_set:
                continue
            key = (split, index)
            prior = old.get(key, {})
            records.append({
                "split": split,
                "source_row_index": index,
                "scene_id": row["scene_id"],
                "task_type": row["task_type"],
                "task_id": row.get("task_id"),
                "object_pair": object_pair(row),
                "relation": relation(row),
                "diagnostic_kind": "full_canary_projective",
                "old_status": prior.get("status"),
                "old_failure": prior.get("failure"),
                "old_failure_taxonomy": prior.get("failure_taxonomy"),
                "old_replacement": bool(prior.get("replacement_lineage")),
            })
    records.sort(key=lambda row: (scenes.index(row["scene_id"]), row["split"], row["source_row_index"]))
    keys = [(row["split"], row["source_row_index"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (split, source_row_index) in inventory")
    by_scene = Counter(row["scene_id"] for row in records)
    by_split = Counter(row["split"] for row in records)
    if set(by_scene) != scene_set:
        raise ValueError(f"scene coverage mismatch: got={sorted(by_scene)} expected={scenes}")
    payload = {
        "version": "r1_projective_path_first_canary10_selection_v1",
        "selection_is_frozen": True,
        "seed": "deterministic_no_rng",
        "scenes": scenes,
        "sources": source_sha,
        "baseline_accounting_root": str(args.baseline_accounting_root),
        "records": records,
        "record_count": len(records),
        "counts": {"scene": dict(by_scene), "split": dict(by_split)},
        "frozen_config": {
            "generator": "projective_path_first_forward_validated_success_region_v2",
            "camera": "canonical_camera_h1_resize_v1",
            "metric": "canonical_spatial_task_h1_v1",
            "relation_margin_px": 12,
            "max_steps": 12,
            "observability": "projective_observability_v1_initial_keyframe",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "source_inventory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    selections = args.output_dir / "scene_selections"
    selections.mkdir(exist_ok=True)
    for scene in scenes:
        scene_records = [row for row in records if row["scene_id"] == scene]
        (selections / f"{scene}.json").write_text(json.dumps({
            "version": payload["version"], "parent_selection": "frozen_selection.json",
            "scene_id": scene, "records": scene_records,
        }, indent=2) + "\n")
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    chunks = args.output_dir / "chunk_selections"
    chunks.mkdir(exist_ok=True)
    for start in range(0, len(records), args.chunk_size):
        chunk_records = records[start:start + args.chunk_size]
        chunk_id = f"chunk_{start // args.chunk_size:03d}"
        (chunks / f"{chunk_id}.json").write_text(json.dumps({
            "version": payload["version"], "parent_selection": "frozen_selection.json",
            "chunk_id": chunk_id, "records": chunk_records,
        }, indent=2) + "\n")
    print(json.dumps({"records": len(records), "by_scene": dict(by_scene), "by_split": dict(by_split)}, indent=2))


if __name__ == "__main__":
    main()
