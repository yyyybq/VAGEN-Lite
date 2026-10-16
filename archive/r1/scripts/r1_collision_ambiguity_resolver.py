#!/usr/bin/env python3
"""Conservative second-stage evidence audit for ambiguous collision scenes.

This does not force a coordinate convention.  It compares both structure-Y
interpretations using label centers/corners and historical camera poses.  A
scene remains quarantined unless independent evidence is both sufficiently
large and strongly separated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r1_reachability_audit import atomic_write_json, atomic_write_jsonl, read_jsonl
from vagen.envs.active_spatial.collision_detector import CollisionDetector


VERSION = "collision_ambiguity_resolution_v1"
AMBIGUOUS_SCENES = (
    "0059_839917", "0265_840795", "0270_840784",
    "0314_840535", "0328_840489", "0349_840373",
)


def polygons(structure: dict[str, Any], sign: float) -> list[np.ndarray]:
    return [
        np.asarray([[point[0], sign * point[1]] for point in room.get("profile", [])], dtype=float)
        for room in structure.get("rooms", [])
        if len(room.get("profile", [])) >= 3
    ]


def inside(point: np.ndarray, rooms: list[np.ndarray]) -> bool:
    return any(CollisionDetector._point_in_polygon(point[:2], room) for room in rooms)


def wall_distance(point: np.ndarray, rooms: list[np.ndarray]) -> float | None:
    def segment_distance(value: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        direction = end - start
        length_squared = float(np.dot(direction, direction))
        if length_squared <= 1e-12:
            return float(np.linalg.norm(value - start))
        fraction = float(np.clip(np.dot(value - start, direction) / length_squared, 0.0, 1.0))
        return float(np.linalg.norm(value - (start + fraction * direction)))

    distances = []
    for room in rooms:
        for index, start in enumerate(room):
            end = room[(index + 1) % len(room)]
            distances.append(segment_distance(point[:2], start, end))
    return float(min(distances)) if distances else None


def label_points(labels: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    centers, corners = [], []
    for row in labels:
        box = row.get("bounding_box") or []
        if len(box) != 8:
            continue
        xyz = np.asarray([[p["x"], p["y"], p["z"]] for p in box], dtype=float)
        centers.append((xyz.min(axis=0) + xyz.max(axis=0)) / 2.0)
        corners.extend(xyz)
    return centers, corners


def evidence(points: list[np.ndarray], rooms: list[np.ndarray], require_clearance: bool) -> dict[str, Any]:
    inside_count = sum(inside(point, rooms) for point in points)
    clear_count = sum(
        inside(point, rooms)
        and (wall_distance(point, rooms) or 0.0) >= 0.5
        for point in points
    )
    return {
        "count": len(points),
        "inside_room": inside_count,
        "inside_and_wall_clear_0p5m": clear_count,
        "score": clear_count if require_clearance else inside_count,
    }


def relative_margin(scores: dict[str, int]) -> float:
    return abs(scores["+1"] - scores["-1"]) / max(scores["+1"], scores["-1"], 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", default=",".join(AMBIGUOUS_SCENES))
    args = parser.parse_args()

    source_paths = json.loads(args.sources.read_text())
    rows_by_split = {
        split: read_jsonl(Path(path)) for split, path in source_paths.items()
    }
    requested = [value for value in args.scenes.split(",") if value]
    results = []
    for scene_id in requested:
        scene_path = args.gs_root / scene_id
        labels_path = scene_path / "labels.json"
        structure_path = scene_path / "structure.json"
        if not labels_path.is_file() or not structure_path.is_file():
            results.append({
                "version": VERSION, "scene_id": scene_id, "decision": "quarantine",
                "reason": "assets_missing", "asset_path": str(scene_path),
            })
            continue
        labels = json.loads(labels_path.read_text())
        structure = json.loads(structure_path.read_text())
        centers, corners = label_points(labels)
        initial_points, target_points = [], []
        impact = Counter()
        task_impact = Counter()
        for split, rows in rows_by_split.items():
            for row in rows:
                if str(row.get("scene_id")) != scene_id:
                    continue
                impact[split] += 1
                task_impact[f"{split}:{row.get('task_type')}"] += 1
                pose = row.get("init_camera", {}).get("extrinsics")
                if pose:
                    initial_points.append(np.asarray(pose, dtype=float)[:3, 3])
                target = row.get("sample_target")
                if target:
                    target_points.append(np.asarray(target, dtype=float))
        signs = {}
        for sign in (1.0, -1.0):
            key = f"{sign:+.0f}"
            rooms = polygons(structure, sign)
            signs[key] = {
                "rooms": len(rooms),
                "label_centers": evidence(centers, rooms, False),
                "label_bbox_corners": evidence(corners, rooms, False),
                "historical_initial_poses": evidence(initial_points, rooms, True),
                # Historical targets may encode the known data bug, so they are
                # supporting evidence only and never independently freeze a sign.
                "historical_target_poses": evidence(target_points, rooms, True),
            }
        score_groups = {}
        for group in ("label_centers", "label_bbox_corners", "historical_initial_poses", "historical_target_poses"):
            scores = {sign: int(signs[sign][group]["score"]) for sign in ("+1", "-1")}
            winner = "+1" if scores["+1"] > scores["-1"] else "-1" if scores["-1"] > scores["+1"] else None
            score_groups[group] = {
                "scores": scores, "winner": winner, "relative_margin": relative_margin(scores)
            }
        label = score_groups["label_centers"]
        initial = score_groups["historical_initial_poses"]
        corner = score_groups["label_bbox_corners"]
        freeze = bool(
            len(initial_points) >= 20
            and initial["winner"] is not None
            and initial["relative_margin"] >= 0.25
            and label["winner"] == initial["winner"] == corner["winner"]
            and max(label["scores"].values()) >= 20
        )
        results.append({
            "version": VERSION,
            "scene_id": scene_id,
            "decision": "frozen" if freeze else "quarantine",
            "structure_y_sign": float(initial["winner"]) if freeze else None,
            "reason": "independent_evidence_strong_and_concordant" if freeze else "evidence_insufficient_or_close",
            "evidence": signs,
            "comparisons": score_groups,
            "formal_manifest_impact_rows": dict(impact),
            "formal_manifest_task_impact_rows": dict(task_impact),
            "formal_manifest_total_rows": sum(impact.values()),
            "thresholds": {
                "minimum_historical_initial_poses": 20,
                "minimum_initial_relative_margin": 0.25,
                "minimum_label_evidence": 20,
                "required_concordance": ["label_centers", "label_bbox_corners", "historical_initial_poses"],
            },
        })
    summary = {
        "version": VERSION,
        "scenes": len(results),
        "decision_counts": dict(Counter(row["decision"] for row in results)),
        "quarantined_scene_ids": [row["scene_id"] for row in results if row["decision"] == "quarantine"],
        "impacted_rows": sum(int(row.get("formal_manifest_total_rows") or 0) for row in results),
        "note": "No coordinate convention is forced when evidence is close.",
    }
    atomic_write_jsonl(args.output_dir / "scene_resolution_manifest.jsonl", results)
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_json(
        args.output_dir / "frozen_overrides.json",
        {
            "version": VERSION,
            "structure_y_sign_overrides": {
                row["scene_id"]: row["structure_y_sign"]
                for row in results if row["decision"] == "frozen"
            },
            "quarantined_scene_ids": summary["quarantined_scene_ids"],
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
