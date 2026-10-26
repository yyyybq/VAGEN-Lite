"""label_outliers.py: label errors found by size (2026-10-14, round 6)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import label_outliers as LO
from data_gen.interiorgs_v10.testing import synthetic_scene as S


class LabelOutliersTest(unittest.TestCase):
    def test_size_rule(self):
        rng = np.random.default_rng(0)
        rows = [("s", f"n{i}", "stool", 0.45 + 0.05 * rng.random(), 0.5 + 0.05 * rng.random(),
                 0.4 + 0.05 * rng.random(), 0.0) for i in range(40)]
        rows += [("s", "high", "stool", 0.45, 0.5, 0.4, 1.96),          # a "stool" on the wall
                 ("s", "thin", "stool", 0.79, 0.9, 0.08, 0.0),          # a panel
                 ("s", "huge", "stool", 1.6, 0.5, 0.4, 0.0)]            # 3.5x the usual height
        rows += [("t", f"p{i}", "pinecone", 0.07, 0.08, 0.07, 0.6) for i in range(25)]
        rows += [("t", "sofa_like", "pinecone", 0.63, 1.33, 0.82, 0.0)]
        rows += [("u", f"x{i}", "rare thing", 0.5, 0.5, 0.5, 0.0) for i in range(5)] + [("u", "big", "rare thing", 5, 5, 5, 0)]
        flags = LO.flag_outliers(rows)
        self.assertEqual(set(flags), {("s", "high"), ("s", "thin"), ("s", "huge"), ("t", "sofa_like")})
        self.assertIn("above the floor", flags[("s", "high")][0])

    def test_build_extends_the_base_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for k in range(25):
                d = root / f"scene_{k:02d}"
                d.mkdir()
                labels = [S._box("1", "stool", 0, 0, 0, 0.4, 0.4, 0.45), S._box("2", "book", 0, 0, 1.0, 0.2, 0.03, 1.3)]
                if k == 3:
                    labels += [S._box("9", "stool", 1, 1, 2.0, 1.4, 1.4, 2.45),       # flagged, may be a target
                               S._box("8", "book", 1, 1, 0, 2.0, 2.0, 2.0)]           # flagged, never a target
                (d / "labels.json").write_text(json.dumps(labels))
            base = {"scenes": {"scene_05": [{"ins_id": "1", "label": "stool", "kind": "uncertain"}]}}
            data = LO.build(root, base)
        self.assertEqual(sorted(data["scenes"]), ["scene_03", "scene_05"])
        self.assertEqual([(o["ins_id"], o["kind"]) for o in data["scenes"]["scene_03"]], [("9", "size_outlier")])
        self.assertEqual(data["counts"]["size_outlier_added"], 1)
        self.assertEqual(data["counts"]["objects"], 2)


if __name__ == "__main__":
    unittest.main()
