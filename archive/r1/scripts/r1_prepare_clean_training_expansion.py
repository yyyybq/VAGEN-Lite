#!/usr/bin/env python3
"""Freeze the first clean-training expansion scope without policy-derived data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "r1_clean_training_expansion_scope_v1"
TARGET_SCENES = (
    "0228_840300", "0245_841008", "0251_840828", "0309_840544", "0297_840578",
    "0238_840875", "0305_840555", "0263_840792", "0241_840879", "0303_840566",
)
AMBIGUOUS_COLLISION_SCENES = {
    "0059_839917", "0265_840795", "0270_840784",
    "0314_840535", "0328_840489", "0349_840373",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--verified-train", type=Path, required=True)
    parser.add_argument("--development-eval", type=Path, required=True)
    parser.add_argument("--local-action-parents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train = read_jsonl(args.train)
    verified = read_jsonl(args.verified_train)
    development = read_jsonl(args.development_eval)
    local_action = read_jsonl(args.local_action_parents)
    verified_scenes = {str(row["scene_id"]) for row in verified}
    dev_sources = {(str(row["split"]), int(row["source_row_index"])) for row in development}
    local_sources = {(str(row["split"]), int(row["source_row_index"])) for row in local_action}
    local_scenes = {str(row["scene_id"]) for row in local_action}
    if set(TARGET_SCENES) & verified_scenes:
        raise ValueError("expansion scene overlaps existing verified train inventory")
    if set(TARGET_SCENES) & local_scenes:
        raise ValueError("expansion scene overlaps permanent local-action evaluation scenes")
    if set(TARGET_SCENES) & AMBIGUOUS_COLLISION_SCENES:
        raise ValueError("expansion scene uses ambiguous collision convention")

    records = []
    for index, row in enumerate(train):
        task = str(row.get("task_type"))
        scene = str(row.get("scene_id"))
        key = ("train", index)
        if scene not in TARGET_SCENES or task not in {"projective_relations", "fov_inclusion"}:
            continue
        if key in dev_sources or key in local_sources:
            raise ValueError(f"source leakage in frozen expansion: {key}")
        records.append(
            {
                "split": "train",
                "source_row_index": index,
                "task_id": row.get("task_id"),
                "scene_id": scene,
                "task_type": task,
                "relation": row.get("target_region", {}).get("params", {}).get("relation"),
                "requested_bucket": "medium" if task == "projective_relations" else "fov_min4",
            }
        )
    projective = [row for row in records if row["task_type"] == "projective_relations"]
    fov = [row for row in records if row["task_type"] == "fov_inclusion"]
    by_scene_task = Counter((row["scene_id"], row["task_type"]) for row in records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    projective_payload = {"version": VERSION, "records": projective}
    fov_payload = {"version": VERSION, "records": fov}
    atomic_json(args.output_dir / "projective_medium_selection.json", projective_payload)
    atomic_json(args.output_dir / "fov_min4_selection.json", fov_payload)
    atomic_json(args.output_dir / "fov_source_index_selection.json", {"train": [row["source_row_index"] for row in fov]})
    atomic_json(args.output_dir / "sources_train_only.json", {"train": str(args.train.resolve())})
    summary = {
        "version": VERSION,
        "policy": {
            "scene_selection": "top deterministic train Projective+FOV counts after frozen exclusions",
            "same_pair_projective_only": True,
            "replacement_projective": False,
            "local_action_evaluation": "permanent_evaluation_only_exclusion",
        },
        "inputs": {
            "train": {"path": str(args.train.resolve()), "sha256": sha256(args.train)},
            "verified_train": {"path": str(args.verified_train.resolve()), "sha256": sha256(args.verified_train)},
            "development_eval": {"path": str(args.development_eval.resolve()), "sha256": sha256(args.development_eval)},
            "local_action_parents": {"path": str(args.local_action_parents.resolve()), "sha256": sha256(args.local_action_parents)},
        },
        "scenes": list(TARGET_SCENES),
        "counts": {
            "total": len(records), "projective_relations": len(projective), "fov_inclusion": len(fov),
            "scene_task": {f"{scene}:{task}": count for (scene, task), count in sorted(by_scene_task.items())},
        },
        "exclusions": {
            "existing_verified_train_scenes": sorted(verified_scenes),
            "permanent_local_action_eval_scenes": sorted(local_scenes),
            "ambiguous_collision_scenes": sorted(AMBIGUOUS_COLLISION_SCENES),
        },
    }
    atomic_json(args.output_dir / "scope_summary.json", summary)
    hashes = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
