#!/usr/bin/env python3
"""Enumerate the complete statically legal train-only InteriorGS pair universe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_gen.active_spatial_pipeline.config import ObjectSelectionConfig
from data_gen.active_spatial_pipeline.object_selector import ObjectSelector


VERSION = "r1_train_pair_universe_v1_20260928"
TARGETS = {"projective_relations", "fov_inclusion"}
AMBIGUOUS = {"0059_839917", "0265_840795", "0270_840784", "0314_840535", "0328_840489", "0349_840373"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        temp = Path(handle.name)
    temp.replace(path)


def object_pair_key(scene: str, objects: list[dict[str, Any]]) -> str | None:
    ids = [str(obj.get("id") or obj.get("ins_id") or "").strip() for obj in objects]
    if len(ids) != 2 or not all(ids) or ids[0] == ids[1]:
        return None
    return f"{scene}|{'--'.join(sorted(ids))}"


def row_pair_key(row: dict[str, Any]) -> str | None:
    objects = (row.get("target_object") or {}).get("objects") or []
    if len(objects) != 2:
        return None
    return object_pair_key(str(row.get("scene_id") or ""), objects)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--old-train", type=Path, required=True)
    parser.add_argument("--canonical-eval", type=Path, required=True)
    parser.add_argument("--local-action-parents", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--phase-a-sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projective-deficit", type=int, default=1791)
    parser.add_argument("--fov-deficit", type=int, default=1503)
    args = parser.parse_args()

    old = read_jsonl(args.old_train)
    dev = read_jsonl(args.canonical_eval)
    local = read_jsonl(args.local_action_parents)
    eval_scenes = {str(row.get("scene_id")) for row in dev + local if row.get("scene_id")}
    train_scenes = sorted({str(row["scene_id"]) for row in old} - eval_scenes - AMBIGUOUS)
    excluded_pairs: set[str] = set()
    exclusion_origin = Counter()
    inputs = [args.old_train, args.canonical_eval, args.local_action_parents, args.phase_a_sources, *args.exclude_manifest]
    for origin, path in [("old_v46_targets", args.old_train), ("phase_a", args.phase_a_sources), *[(f"existing:{path.name}", path) for path in args.exclude_manifest]]:
        for row in read_jsonl(path):
            if origin == "old_v46_targets" and row.get("task_type") not in TARGETS:
                continue
            key = row_pair_key(row)
            if key:
                excluded_pairs.add(key)
                exclusion_origin[origin] += 1

    selector = ObjectSelector(ObjectSelectionConfig())
    universe: list[dict[str, Any]] = []
    per_scene: dict[str, Any] = {}
    rejection_totals = Counter()
    for scene in train_scenes:
        scene_dir = args.metadata_root / scene
        labels_path, structure_path = scene_dir / "labels.json", scene_dir / "structure.json"
        if not labels_path.is_file() or not structure_path.is_file():
            raise FileNotFoundError(f"missing staged metadata for {scene}")
        labels = json.loads(labels_path.read_text())
        room_polys = selector.load_room_polys(scene_dir)
        parsed = []
        parse_reject = 0
        single_reasons = Counter()
        for item in labels:
            try:
                obj = selector.parse_object(item, room_polys)
            except (KeyError, TypeError, ValueError, OverflowError):
                obj = None
            if obj is None or not all(math.isfinite(float(value)) for value in [*obj.center, *obj.dims]):
                parse_reject += 1
                continue
            passed, reason = selector.filter_single_object(obj)
            if passed:
                parsed.append(obj)
            else:
                single_reasons[reason] += 1
        global_counts = Counter(obj.label for obj in parsed)
        room_counts: dict[int | None, Counter[str]] = defaultdict(Counter)
        for obj in parsed:
            room_counts[obj.room_index][obj.label] += 1
        objects = [obj for obj in parsed if global_counts[obj.label] == 1 or room_counts[obj.room_index][obj.label] == 1]
        ambiguity_reject = len(parsed) - len(objects)
        pair_reasons = Counter()
        scene_pairs = 0
        excluded_existing = 0
        for obj_a, obj_b in combinations(objects, 2):
            passed, reason = selector.filter_object_pair(obj_a, obj_b)
            if not passed:
                pair_reasons[reason.split("_", 5)[0] if reason.startswith("angular_span_too_large") else reason] += 1
                continue
            pair_key = object_pair_key(scene, [{"id": obj_a.id}, {"id": obj_b.id}])
            if pair_key in excluded_pairs:
                excluded_existing += 1
                continue
            ordered = sorted((obj_a, obj_b), key=lambda obj: obj.id)
            category_pair = "--".join(sorted(obj.label for obj in ordered))
            lineage_payload = f"{scene}|{ordered[0].id}|{ordered[1].id}"
            universe.append({
                "version": VERSION, "pair_id": "pair_" + hashlib.sha256(lineage_payload.encode()).hexdigest()[:20],
                "scene_id": scene, "object_ids": [obj.id for obj in ordered], "categories": [obj.label for obj in ordered],
                "category_pair": category_pair, "room_indices": [obj.room_index for obj in ordered],
                "objects": [{"id": obj.id, "label": obj.label, "center": obj.center.tolist(), "dims": obj.dims.tolist(),
                             "bbox_min": obj.aabb_min.tolist(), "bbox_max": obj.aabb_max.tolist(), "room_index": obj.room_index}
                            for obj in ordered],
                "static_geometry": {"center_distance_xy": float(((ordered[0].center[:2] - ordered[1].center[:2]) ** 2).sum() ** 0.5),
                                    "aabb_min_distance": selector.aabb_min_distance(*ordered)},
                "source_construction": {"kind": "aoss_labels_pair_enumeration", "labels_sha256": sha256(labels_path),
                                        "structure_sha256": sha256(structure_path), "pair_key": pair_key,
                                        "object_selector_config": "ObjectSelectionConfig defaults at code version"},
            })
            scene_pairs += 1
        per_scene[scene] = {"raw_labels": len(labels), "parsed_and_single_valid": len(parsed), "unambiguous_objects": len(objects),
                            "parse_reject": parse_reject, "single_reject": dict(single_reasons), "ambiguous_label_reject": ambiguity_reject,
                            "pair_combinations": len(objects) * (len(objects) - 1) // 2, "pair_reject": dict(pair_reasons),
                            "existing_pair_excluded": excluded_existing, "unique_new_pairs": scene_pairs}
        rejection_totals.update({f"single:{key}": value for key, value in single_reasons.items()})
        rejection_totals.update({f"pair:{key}": value for key, value in pair_reasons.items()})
        rejection_totals["parse"] += parse_reject
        rejection_totals["ambiguous_label"] += ambiguity_reject
        rejection_totals["existing_pair"] += excluded_existing

    universe.sort(key=lambda row: (row["scene_id"], row["object_ids"], row["pair_id"]))
    if len({row["pair_id"] for row in universe}) != len(universe):
        raise RuntimeError("duplicate stable pair IDs")
    category_counts = Counter(row["category_pair"] for row in universe)
    summary = {
        "version": VERSION, "train_only_scenes": len(train_scenes), "total_unique_new_pairs": len(universe),
        "task_capacity": {"projective_unique_rows_max": len(universe), "fov_unique_rows_max": len(universe),
                          "tasks_may_share_pair_with_distinct_lineage": True},
        "deficit_relation": {"projective_deficit": args.projective_deficit, "fov_deficit": args.fov_deficit,
                             "enough_projective_capacity": len(universe) >= args.projective_deficit,
                             "enough_fov_capacity": len(universe) >= args.fov_deficit},
        "scene_capacity": {scene: record["unique_new_pairs"] for scene, record in per_scene.items()},
        "category_pairs": dict(sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))),
        "max_category_pair_fraction": max(category_counts.values()) / len(universe) if universe else None,
        "rejection_totals": dict(sorted(rejection_totals.items())), "exclusion_input_rows": dict(exclusion_origin),
        "eval_scenes_excluded": sorted(eval_scenes), "ambiguous_scenes_excluded": sorted(AMBIGUOUS),
        "inputs": {str(path): sha256(path) for path in inputs}, "per_scene_details": per_scene,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output_dir / "train_pair_universe.jsonl", universe)
    atomic_json(args.output_dir / "universe_summary.json", summary)
    paths = [args.output_dir / "train_pair_universe.jsonl", args.output_dir / "universe_summary.json"]
    (args.output_dir / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in paths))
    print(json.dumps({key: summary[key] for key in ("train_only_scenes", "total_unique_new_pairs", "task_capacity", "deficit_relation")}, indent=2))


if __name__ == "__main__":
    main()
