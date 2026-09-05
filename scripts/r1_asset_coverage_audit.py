#!/usr/bin/env python3
"""Audit scene-asset coverage for versioned R1 regeneration and rendering."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = json.loads(args.sources_json.read_text())
    report = {"gs_root": str(args.gs_root), "splits": {}}
    for split, source_name in sources.items():
        rows = read_jsonl(Path(source_name))
        projective = [row for row in rows if row.get("task_type") == "projective_relations"]
        fov = [row for row in rows if row.get("task_type") == "fov_inclusion"]
        all_scenes = sorted({str(row.get("scene_id")) for row in rows})
        available_scenes = sorted(
            scene for scene in all_scenes
            if (args.gs_root / scene / "structure.json").exists()
            and (args.gs_root / scene / "labels.json").exists()
            and ((args.gs_root / scene / "3dgs_compressed.ply").exists() or (args.gs_root / scene / "point_cloud.ply").exists())
        )
        available = set(available_scenes)
        report["splits"][split] = {
            "source": source_name,
            "rows": len(rows),
            "scenes": len(all_scenes),
            "available_scenes": available_scenes,
            "projective_rows": len(projective),
            "projective_rows_with_assets": sum(str(row.get("scene_id")) in available for row in projective),
            "fov_rows": len(fov),
            "fov_rows_with_assets": sum(str(row.get("scene_id")) in available for row in fov),
            "projective_scene_distribution_with_assets": dict(Counter(str(row.get("scene_id")) for row in projective if str(row.get("scene_id")) in available)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
