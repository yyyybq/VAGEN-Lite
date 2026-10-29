"""The experiment matrix registered in main (configs/matrix_r12.json, round 12 freeze, the user's 3.3 of 2026-10-22):
groups A-E planned and C0 done, every group an S0 / D pair of one standard recipe, the P0 / P1 paths left empty."""

import json
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import training_prep as TP

MATRIX = Path(TP.__file__).with_name("configs") / "matrix_r12.json"


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.m = json.loads(MATRIX.read_text(encoding="utf-8"))

    def test_groups_and_status(self):
        g = self.m["groups"]
        self.assertEqual(sorted(g), ["A", "B", "C", "C0", "D", "E"])
        self.assertEqual({k: v["status"] for k, v in g.items()},
                         {"A": "planned", "B": "planned", "C": "planned", "D": "planned", "E": "planned", "C0": "done"})
        self.assertEqual([g[k]["pools"] for k in ("B", "C", "D", "E", "C0")],
                         [["P0"], ["P0", "P2"], ["P0", "P1", "P2"], ["P0", "P1", "P2"], ["P2"]])
        self.assertEqual((g["A"]["start"], g["C0"]["start"]), ("base_7b", "sft_e9"))
        for k in "ABCDE":  # new data shows the future data's format example; C0 trained on the old text
            self.assertEqual(g[k]["format_example"], "action_names", k)
        self.assertEqual(g["C0"]["format_example"], "placeholder")

    def test_pools_and_pair(self):
        self.assertIsNone(self.m["pools"]["P0"]["path"])
        self.assertIsNone(self.m["pools"]["P1"]["path"])
        self.assertTrue(self.m["pools"]["P2"]["path"].endswith("interiorgs_v10_r11_20261016/sft/sft_trajectories.jsonl"))
        standards = TP.load_standards()
        pair = self.m["rl"]["pair"]
        s0, d = standards[pair["S0"]], standards[pair["D"]]
        self.assertEqual((s0["reward_preset"], d["reward_preset"]), ("S0", "dense"))
        self.assertEqual({k: v for k, v in s0.items() if k != "reward_preset"},
                         {k: v for k, v in d.items() if k != "reward_preset"})  # a pair differs only in the reward
        self.assertEqual(self.m["rl"]["seed"], 20261019)


if __name__ == "__main__":
    unittest.main()
