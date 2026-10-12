"""Tests for make_split.py (scene split with same-source groups and R1 pins)."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import make_split as M
from data_gen.interiorgs_v10 import release as R


class MakeSplitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.run = root / "run"
        (self.run / "scenes").mkdir(parents=True)
        self.ids = [f"s{i:03d}" for i in range(60)]
        for i, sid in enumerate(self.ids):
            res = {"summary": {"scene_id": sid}, "rows": [{}] * (i % 3), "reserve_rows": [{}] * (i % 2)}
            (self.run / "scenes" / f"{sid}.json").write_text(json.dumps(res))
        self.r1 = root / "r1"
        self.r1.mkdir()
        (self.r1 / "train.jsonl").write_text("\n".join(json.dumps({"scene_id": s}) for s in self.ids[:5]))
        (self.r1 / "eval_policy_rows.jsonl").write_text(json.dumps({"scene_id": self.ids[5]}) + "\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_pins_groups_and_counts(self):
        pins = M.r1_pins(self.r1)
        self.assertEqual(sum(v == "train" for v in pins.values()), 5)
        self.assertEqual(pins[self.ids[5]], "test")
        groups = [[self.ids[5], self.ids[40]], [self.ids[10], self.ids[11], self.ids[12]]]
        rep = M.make_split(self.run, groups, pins)
        look = R.split_lookup(rep["manifest"])
        self.assertEqual(look[self.ids[40]], "test")  # pulled along by the pinned R1 evaluation scene
        self.assertEqual(len({look[s] for s in groups[1]}), 1)
        self.assertTrue(all(look[s] == "train" for s in self.ids[:5]))
        self.assertEqual(sum(v["scenes"] for v in rep["splits"].values()), 60)
        self.assertEqual(sum(v["tasks"] for v in rep["splits"].values()), sum(i % 3 for i in range(60)))
        self.assertEqual(rep["checks"]["r1_eval_scenes_in_train"], 0)

    def test_conflicting_pins_fail(self):
        # an R1 evaluation scene grouped with an R1 training scene cannot be placed
        with self.assertRaises(ValueError):
            M.make_split(self.run, [[self.ids[0], self.ids[5]]], M.r1_pins(self.r1))


if __name__ == "__main__":
    unittest.main()
