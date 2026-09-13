#!/usr/bin/env python3
"""Freeze a non-canary development-scene screen for v2/v3 comparison.

Selection never inspects selector outcomes.  It takes the lexicographically
first five asset-valid, collision-unambiguous, previously pending scenes with
at least 20 train Projective rows, then fills a 128-row cap round-robin using a
stable source-key digest within each scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "r1_independent_v3_holdout_selection_v1"
DESIGN_SCENES = {
    "0011_840866", "0014_841007", "0226_840298", "0229_840306", "0240_840881",
    "0267_840790", "0276_840780", "0300_840573", "0361_840315", "0367_840260",
}
AMBIGUOUS_SCENES = {
    "0059_839917", "0265_840795", "0270_840784", "0314_840535", "0328_840489", "0349_840373",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def object_pair(row: dict[str, Any]) -> list[str]:
    objects = row.get("target_object", {}).get("objects", [])
    return [f"{obj.get('id')}:{obj.get('label')}" for obj in objects]


def relation(row: dict[str, Any]) -> str | None:
    return row.get("target_region", {}).get("params", {}).get("relation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True,
                        help="Frozen 10-scene inventory; its source map/SHA is reused, not its records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=5)
    parser.add_argument("--source-cap", type=int, default=128)
    parser.add_argument("--minimum-projective-per-scene", type=int, default=20)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text())
    frozen = json.loads(args.source_inventory.read_text())
    train_info = frozen["sources"]["train"]
    train_path = Path(train_info["path"])
    if sha256(train_path) != train_info["sha256"]:
        raise ValueError("train manifest SHA256 differs from frozen source inventory")
    train = read_jsonl(train_path)
    ledger_by_scene = {row["scene_id"]: row for row in ledger["scenes"]}
    eligible = []
    for row in ledger["scenes"]:
        scene = str(row["scene_id"])
        count = int(row.get("split_task_counts", {}).get("train:projective_relations", 0))
        if (scene not in DESIGN_SCENES and scene not in AMBIGUOUS_SCENES
                and row.get("asset_validation", {}).get("status") == "valid"
                and row.get("download_status") == "ready"
                and row.get("processing_status") == "pending"
                and count >= args.minimum_projective_per_scene):
            eligible.append(scene)
    scenes = sorted(eligible)[:args.scene_count]
    if len(scenes) != args.scene_count:
        raise ValueError(f"only {len(scenes)} eligible independent scenes")
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(train):
        scene = str(row.get("scene_id") or "")
        if scene in scenes and row.get("task_type") == "projective_relations":
            digest = hashlib.sha256(f"v3-independent-holdout:{index}".encode()).hexdigest()
            by_scene[scene].append({
                "split": "train", "source_row_index": index, "scene_id": scene,
                "task_type": row.get("task_type"), "task_id": row.get("task_id"),
                "object_pair": object_pair(row), "relation": relation(row), "selection_digest": digest,
            })
    for scene in scenes:
        by_scene[scene].sort(key=lambda row: (row["selection_digest"], row["source_row_index"]))
    selected = []
    cursor = Counter()
    while len(selected) < args.source_cap:
        progressed = False
        for scene in scenes:
            if cursor[scene] < len(by_scene[scene]) and len(selected) < args.source_cap:
                selected.append(by_scene[scene][cursor[scene]])
                cursor[scene] += 1
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: (row["scene_id"], row["selection_digest"], row["source_row_index"]))
    payload = {
        "version": VERSION,
        "selection_is_frozen": True,
        "selection_rule": {
            "scene_rule": "lexicographically first eligible development scenes before observing v2/v3 outcomes",
            "row_rule": "round-robin scenes; within scene SHA256(v3-independent-holdout:source_row_index)",
            "scene_count": args.scene_count,
            "source_cap": args.source_cap,
            "minimum_projective_per_scene": args.minimum_projective_per_scene,
        },
        "prior_use": {
            "excluded_selector_design_scenes": sorted(DESIGN_SCENES),
            "selected_scene_ledger_status": {
                scene: {
                    "processing_status": ledger_by_scene[scene].get("processing_status"),
                    "renderer_smoke_status": ledger_by_scene[scene].get("renderer_smoke_status"),
                    "interpretation": "not present in the 10-scene selector-design canary; ledger had no R1 processing/render smoke",
                } for scene in scenes
            },
            "scope": "development validation using train-manifest rows only; no final test/OOD manifests",
        },
        "scenes": scenes,
        "sources": {"train": train_info},
        "ledger": {"path": str(args.ledger.resolve()), "sha256": sha256(args.ledger)},
        "collision_convention": {
            "version": "interiorgs_structure_label_alignment_v1",
            "eligible_status": "frozen +1 among the 94 non-ambiguous scenes",
            "excluded_ambiguous_scenes": sorted(AMBIGUOUS_SCENES),
        },
        "record_count": len(selected),
        "counts": {"scene": dict(sorted(Counter(row["scene_id"] for row in selected).items())), "split": {"train": len(selected)}},
        "records": selected,
        "frozen_config": {
            "same_pair_only": True,
            "replacement": False,
            "v2": "projective_difficulty_conditioned_medium_selector_v2",
            "v3": "projective_difficulty_conditioned_medium_selector_v3_depth_banded",
            "budgets": {"seed_cap": 12, "per_seed_expansions": 512,
                        "candidate_cap_per_seed": 48, "candidate_cap_scope": "per success seed",
                        "lower_expansions": 100000},
            "cost_tolerance_predeclared": {
                "v3_generation_elapsed_ratio_vs_v2_max": 1.20,
                "screening_source_cap": args.source_cap,
                "shortcut_ab_source_target": "24-32 if available",
                "positive_controls_target": 5,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"scenes": scenes, "records": len(selected), "counts": payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
