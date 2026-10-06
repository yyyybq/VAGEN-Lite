"""Fast contract tests; no renderer or model weights required."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .contract import evaluate_state
from .generate_paired_qa import generate, _pose_from_forward
from .image_contract_v2 import pose_legality
import argparse


def _fixture():
    return {
        "scene_id": "fixture_scene", "task_type": "absolute_positioning",
        "object_label": "chair", "task_description": "be 1m from chair",
        "target_object": {"label": "chair", "center": [0.0, 0.0, 1.0]},
        "init_camera": {"intrinsics": [[320, 0, 320], [0, 0, 240], [0, 0, 1]]},
        "target_region": {"type": "circle", "params": {"center": [0, 0], "radius": 1.0, "object_center": [0, 0, 1.0]},
                          "sample_point": [1.0, 0.0, 1.0], "sample_forward": [-1.0, 0.0, 0.0]},
        "sample_target": [1.0, 0.0, 1.0], "camera_params": {"forward": [-1.0, 0.0, 0.0]},
    }


class ContractTests(unittest.TestCase):
    def test_numeric_geometry_height(self):
        self.assertTrue(pose_legality([0, 0, 1.5], [0, 1, 0])["valid"])
        self.assertIn("height_out_of_range", pose_legality([0, 0, -1], [0, 1, 0])["reasons"])

    def test_shared_predicate_and_public_isolation(self):
        item = _fixture()
        pose = _pose_from_forward([1, 0, 1], [-1, 0, 0])
        result = evaluate_state(item, pose)
        self.assertTrue(result["success"])
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.jsonl"; out = Path(td) / "out"
            src.write_text(json.dumps(item) + "\n")
            summary = generate(argparse.Namespace(input=str(src), output_dir=str(out), split="test", split_scenes="", task_type="", limit=5))
            self.assertEqual(summary["errors"], 0)
            rows = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue(all("score" not in r["public_observation"] for r in rows))
            self.assertTrue(all(r["private_answer"] is None for r in rows))
            self.assertTrue(all(r['geometry_legality']['status']=='coordinate_unconfirmed' for r in rows))
            self.assertTrue((out / "contact_sheet.html").exists())

    def test_existing_state_image_is_preserved(self):
        item = _fixture()
        item["states"] = [{"state_id": "rendered", "c2w": _pose_from_forward([1, 0, 1], [-1, 0, 0]).tolist(), "image_path": "images/rendered.png"}]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.jsonl"; out = Path(td) / "out"
            src.write_text(json.dumps(item) + "\n")
            generate(argparse.Namespace(input=str(src), output_dir=str(out), split="test", split_scenes="", task_type="", limit=5))
            row = json.loads((out / "manifest.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["public_observation"]["image_path"], "images/rendered.png")
            self.assertEqual(row["observability_validity"], "valid")

    def test_missing_scene_context_withholds_label(self):
        item = _fixture()
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "input.jsonl"; out = Path(td) / "out"
            src.write_text(json.dumps(item) + "\n")
            generate(argparse.Namespace(input=str(src), output_dir=str(out), split="test", split_scenes="", task_type="", limit=5, scene_root=""))
            row = json.loads((out / "manifest.jsonl").read_text().splitlines()[0])
            self.assertEqual(row["geometry_legality"]["status"], "coordinate_unconfirmed")
            self.assertEqual(row["label_validity"], "invalid_illegal_camera")


if __name__ == "__main__":
    unittest.main()
