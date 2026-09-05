#!/usr/bin/env python3
"""Freeze per-scene InteriorGS structure/label coordinate conventions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from vagen.envs.active_spatial.collision_detector import CollisionDetector


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", help="optional comma-separated scene IDs")
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text())
    requested = {value for value in (args.scenes or "").split(",") if value}
    scene_ids = [
        entry["scene_id"]
        for entry in ledger["scenes"]
        if not requested or entry["scene_id"] in requested
    ]
    rows = []
    for scene_id in scene_ids:
        scene_path = args.gs_root / scene_id
        detector = CollisionDetector(
            camera_radius=0.15,
            floor_height=0.3,
            ceiling_height=2.5,
            safety_margin=0.05,
        )
        loaded = detector.load_scene(scene_path, scene_id=scene_id)
        record = detector.convention_record()
        record.update(
            {
                "scene_id": scene_id,
                "scene_loaded": bool(loaded),
                "labels_sha256": sha256(scene_path / "labels.json") if loaded else None,
                "structure_sha256": sha256(scene_path / "structure.json") if loaded else None,
            }
        )
        rows.append(record)
        print(json.dumps({key: record[key] for key in ("scene_id", "status", "structure_y_sign", "relative_margin")}))
    counts = Counter(row["status"] for row in rows)
    signs = Counter(str(int(row["structure_y_sign"])) for row in rows if row["status"] == "frozen")
    summary = {
        "scenes": len(rows),
        "status_counts": dict(counts),
        "frozen_sign_counts": dict(signs),
        "all_frozen": counts.get("frozen", 0) == len(rows),
        "ledger": str(args.ledger),
        "ledger_sha256_at_audit": sha256(args.ledger),
        "gs_root": str(args.gs_root),
        "scene_filter": sorted(requested) if requested else None,
    }
    write_jsonl(args.output_dir / "collision_conventions.jsonl", rows)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
