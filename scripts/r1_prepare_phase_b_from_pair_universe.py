#!/usr/bin/env python3
"""Freeze deterministic Phase-B requests from the AOSS pair universe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from r1_projective_request_schema import half_plane_geometry, validate_projective_params


VERSION = "r1_fullscale_canonical_expansion_phase_b_scope_v1_20260928"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        temp = Path(handle.name)
    temp.replace(path)


def rank(pair: dict[str, Any], existing: Counter[str], scene_counts: Counter[str], task: str) -> tuple[Any, ...]:
    category = pair["category_pair"]
    non_bed_wardrobe = category != "bed--wardrobe"
    tie = hashlib.sha256(f"{task}|{pair['pair_id']}".encode()).hexdigest()
    return (0 if task == "fov_inclusion" else 1, 0 if non_bed_wardrobe else 1, existing[category], scene_counts[pair["scene_id"]], tie)


def template(pair: dict[str, Any], camera: dict[str, Any], task: str, index: int, relation: str | None) -> dict[str, Any]:
    objects = [dict(obj) for obj in pair["objects"]]
    a, b = objects[0]["center"], objects[1]["center"]
    midpoint = [(float(a[i]) + float(b[i])) / 2 for i in range(3)]
    params: dict[str, Any] = {"object_a_center": a, "object_b_center": b, "min_distance": 0.5,
                              "sample_distance": max(1.0, math.dist(a, b))}
    if task == "projective_relations":
        params["relation"] = relation
        params.update(half_plane_geometry(a, b, str(relation)))
        validate_projective_params(params)
    else:
        params.update({"fov_horizontal": 110.0, "fov_margin": 0.05, "min_radius": 0.5, "max_radius": 8.0})
    return {
        "task_id": f"r1_phase_b_{task}_{index:06d}", "task_type": task, "scene_id": pair["scene_id"],
        "init_camera": camera, "target_object": {"objects": objects, "primary": objects[0]},
        "target_region": {"type": "half_plane" if task == "projective_relations" else "annulus", "params": params,
                          "sample_point": [midpoint[0], midpoint[1], 1.5], "sample_forward": [0.0, 1.0, 0.0], "height": 1.5},
        "sample_target": [midpoint[0], midpoint[1], 1.5], "camera_params": {"forward": [0.0, 1.0, 0.0]},
        "preset": f"{relation}_of" if relation else "fov_inclusion",
        "object_label": "+".join(obj["label"] for obj in objects),
        "task_description": (f"Position where {objects[0]['label']} appears to the {relation} of {objects[1]['label']}"
                             if relation else f"Position where both {objects[0]['label']} and {objects[1]['label']} are visible"),
        "canonical_task_metric_version": "canonical_spatial_task_h1_v1", "camera_model_version": "canonical_camera_h1_resize_v1",
        "fresh_task_lineage": {"version": VERSION, "pair_id": pair["pair_id"], "scene_id": pair["scene_id"],
                               "object_ids": pair["object_ids"], "categories": pair["categories"], "category_pair": pair["category_pair"],
                               "relation": relation, "task_type": task, "source_construction": pair["source_construction"]},
        "training_eligibility": "candidate_requires_full_validation" if task == "projective_relations" else "FOV_PROVISIONAL_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--old-train", type=Path, required=True)
    parser.add_argument("--projective-inventory", type=Path, required=True)
    parser.add_argument("--fov-provisional-inventory", type=Path, required=True)
    parser.add_argument("--projective-requests", type=int, default=1791)
    parser.add_argument("--fov-requests", type=int, default=1309)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    universe = read_jsonl(args.universe)
    old = read_jsonl(args.old_train)
    projective_inventory = read_jsonl(args.projective_inventory)
    fov_inventory = read_jsonl(args.fov_provisional_inventory)
    camera_by_scene: dict[str, dict[str, Any]] = {}
    for row in old:
        scene = str(row.get("scene_id") or "")
        camera = row.get("init_camera")
        if scene and isinstance(camera, dict) and "extrinsics" in camera and "intrinsics" in camera:
            camera_by_scene.setdefault(scene, camera)
    universe = [row for row in universe if row["scene_id"] in camera_by_scene]
    existing_p = Counter(row.get("category_pair") or "unknown" for row in projective_inventory)
    existing_f = Counter(row.get("category_pair") or "unknown" for row in fov_inventory)

    def select(task: str, count: int, existing: Counter[str]) -> list[dict[str, Any]]:
        scene_counts: Counter[str] = Counter()
        remaining = list(universe)
        selected = []
        # Deterministic coverage-aware greedy ranking is recomputed after each
        # choice so scarce categories and underrepresented scenes remain first.
        while remaining and len(selected) < count:
            remaining.sort(key=lambda pair: rank(pair, existing, scene_counts, task))
            pair = remaining.pop(0)
            selected.append(pair)
            existing[pair["category_pair"]] += 1
            scene_counts[pair["scene_id"]] += 1
        return selected

    fov_pairs = select("fov_inclusion", args.fov_requests, existing_f.copy())
    projective_pairs = select("projective_relations", args.projective_requests, existing_p.copy())
    projective = [template(pair, camera_by_scene[pair["scene_id"]], "projective_relations", index,
                           "left" if index % 2 == 0 else "right") for index, pair in enumerate(projective_pairs)]
    fov = [template(pair, camera_by_scene[pair["scene_id"]], "fov_inclusion", index, None) for index, pair in enumerate(fov_pairs)]
    combined = projective + fov
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "fresh_sources.jsonl", combined)
    write_json(args.output_dir / "sources.json", {"train": str((args.output_dir / "fresh_sources.jsonl").resolve())})
    write_json(args.output_dir / "projective_medium_selection.json", {"version": VERSION, "records": [
        {"split": "train", "source_row_index": index, "scene_id": row["scene_id"], "task_type": row["task_type"],
         "requested_bucket": "medium", "difficulty_priority": "prefer_step_5_6_but_certify_formal_medium_4_6",
         "fresh_task_id": row["task_id"]} for index, row in enumerate(projective)]})
    write_json(args.output_dir / "fov_source_index_selection.json", {"train": list(range(len(projective), len(combined)))})
    summary = {
        "version": VERSION, "counts": {"universe_pairs": len(universe), "projective_requests": len(projective),
                                         "fov_provisional_requests": len(fov), "projective_unfilled": max(0, args.projective_requests - len(projective)),
                                         "fov_unfilled": max(0, args.fov_requests - len(fov))},
        "policy": {"selection": "deterministic coverage-aware greedy; FOV, non-bed--wardrobe, scarce category, scene coverage",
                   "same_pair_cross_task": "allowed with distinct task lineage", "replacement": False,
                   "projective_difficulty": "frozen formal medium 4-6; step-5/6 cannot be known from static labels and is audited after search, not replaced by a geometry proxy",
                   "fov": "FOV_PROVISIONAL_ONLY until blinded human calibration freezes FOV-v3"},
        "scene_counts": dict(sorted(Counter(row["scene_id"] for row in combined).items())),
        "projective_category_pairs": dict(Counter(row["fresh_task_lineage"]["category_pair"] for row in projective).most_common()),
        "fov_category_pairs": dict(Counter(row["fresh_task_lineage"]["category_pair"] for row in fov).most_common()),
        "inputs": {str(path): sha256(path) for path in (args.universe, args.old_train, args.projective_inventory, args.fov_provisional_inventory)},
    }
    write_json(args.output_dir / "request_summary.json", summary)
    paths = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.output_dir / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in paths))
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
