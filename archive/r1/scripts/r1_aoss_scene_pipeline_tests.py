#!/usr/bin/env python3
"""Deterministic tests for the R1 AOSS scene ledger and asset validator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from r1_aoss_scene_pipeline import build_ledger, load_ledger, validate_scene


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_a = root / "train.jsonl"
        source_b = root / "ood.jsonl"
        source_a.write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {"scene_id": "scene_a", "task_type": "projective_relations"},
                    {"scene_id": "scene_b", "task_type": "fov_inclusion"},
                )
            ) + "\n"
        )
        source_b.write_text(json.dumps({"scene_id": "scene_b", "task_type": "projective_relations"}) + "\n")
        sources = root / "sources.json"
        sources.write_text(json.dumps({"train": str(source_a), "ood": str(source_b)}))
        ledger_path = root / "ledger.json"
        tsv_path = root / "ledger.tsv"
        ledger = build_ledger(sources, ledger_path, tsv_path, "bucket", "InteriorGS")
        assert ledger["required_scene_count"] == 2
        assert load_ledger(ledger_path)["scenes"][1]["splits"] == ["ood", "train"]
        assert "s3://bucket/InteriorGS/scene_b/" in tsv_path.read_text()

        scene = root / "scene_a"
        scene.mkdir()
        (scene / "structure.json").write_text(json.dumps({"rooms": [{"profile": [[0, 0], [1, 0], [1, 1]]}]}))
        (scene / "labels.json").write_text(json.dumps([{"label": "chair"}]))
        (scene / "3dgs_compressed.ply").write_bytes(b"ply" + b"\0" * (1024 * 1024))
        validation = validate_scene(scene)
        assert validation["status"] == "valid", validation
        assert set(validation["files"]) == {"structure.json", "labels.json", "3dgs_compressed.ply"}

        (scene / "labels.json").write_text("[]")
        invalid = validate_scene(scene)
        assert invalid["status"] == "invalid"
        assert "labels_empty_or_not_list" in invalid["errors"]

    print(json.dumps({"passed": True, "tests": 7, "schema_version": "r1_aoss_scene_ledger_v1"}, indent=2))


if __name__ == "__main__":
    main()
