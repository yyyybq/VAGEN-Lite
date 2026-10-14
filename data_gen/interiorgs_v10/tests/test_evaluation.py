"""evaluation.py: the fixed test subset and the metrics table (the episodes themselves need a GPU)."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import evaluation as E


def _row(i, fam, d):
    return {"task_id": f"t{i:04d}", "task_type": fam, "scene_id": f"s{i % 7}", "difficulty": {"d_star": d}}


class EvaluationTest(unittest.TestCase):
    def test_select_is_stratified_and_fixed(self):
        rows = [_row(i, "projective_relations", 1 + i % 8) for i in range(200)]
        rows += [_row(1000 + i, "occlusion_alignment", 1 + i % 3) for i in range(20)]   # short of tasks
        rows += [_row(2000 + i, "equidistance", 7 + i % 2) for i in range(80)]          # one band only
        a = E.select_tasks(rows, 30, seed=5)
        b = E.select_tasks(list(reversed(rows)), 30, seed=5)
        self.assertEqual([r["task_id"] for r in a], [r["task_id"] for r in b])  # order of the input is irrelevant
        fam = {f: [r for r in a if r["task_type"] == f] for f in ("projective_relations", "occlusion_alignment",
                                                                   "equidistance")}
        self.assertEqual({f: len(v) for f, v in fam.items()},
                         {"projective_relations": 30, "occlusion_alignment": 20, "equidistance": 30})
        bands = [E.band_name(r["difficulty"]["d_star"]) for r in fam["projective_relations"]]
        self.assertEqual(sorted(bands.count(b) for b in ("1-3", "4-6", "7-8")), [10, 10, 10])
        self.assertNotEqual([r["task_id"] for r in a], [r["task_id"] for r in E.select_tasks(rows, 30, seed=6)])

    def test_metrics_and_table(self):
        rows = [{"task_type": "fov_inclusion", "d_star": 2, "success": True, "actions_to_success": 4, "turns": 2,
                 "actions": 4, "collisions": 1, "invalid_turns": 0},
                {"task_type": "fov_inclusion", "d_star": 8, "success": False, "actions_to_success": None, "turns": 12,
                 "actions": 12, "collisions": 0, "invalid_turns": 3},
                {"index": 2, "error": "renderer"}]
        m = E.metrics(rows)
        self.assertEqual((m["episodes"], m["errors"], m["success"]), (2, 1, 0.5))
        self.assertEqual(m["success_by_band"], {"1-3": 1.0, "7-8": 0.0})
        self.assertEqual(m["actions_over_dstar"], 2.0)
        self.assertEqual((m["collision_rate"], m["invalid_rate"]), (0.0625, round(3 / 14, 4)))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "sft"
            d.mkdir()
            (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            E.cmd_table(type("A", (), {"eval": tmp})())
            table = (Path(tmp) / "table.md").read_text(encoding="utf-8")
        self.assertIn("| sft | 2 | 0.500 |", table)


if __name__ == "__main__":
    unittest.main()
