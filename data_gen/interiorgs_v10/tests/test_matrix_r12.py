"""The experiment matrix (configs/matrix_r12.json): registered in round 12 (the user's 3.3 of 2026-10-22), moved to
Qwen3-VL-8B by the user's 14th supplement of 2026-10-27 (decision 78): groups B-E planned on Qwen3 with two seeds,
A closed, C0 the Qwen3 SFT epoch 8 (running); every group an S0 / dense pair; P0 / P1 paths left empty."""

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
                         {"A": "closed", "B": "planned", "C": "planned", "D": "planned", "E": "planned", "C0": "running"})
        self.assertEqual([g[k]["pools"] for k in ("B", "C", "D", "E", "C0")],
                         [["P0", "P2f"], ["P0", "P2q"], ["P0", "P1", "P2q"], ["P0", "P1", "P2q"], ["P2q"]])
        self.assertEqual(g["B"]["name"], "B'")
        for k in ("A", "B", "C", "D", "E"):
            self.assertEqual(g[k]["start"], "base_qwen3", k)
            self.assertEqual(g[k]["format_example"], "action_names", k)
        for k in ("B", "C", "D", "E", "C0"):
            self.assertGreaterEqual(g[k]["seeds"], 2, k)
        self.assertEqual((g["C0"]["start"], g["C0"]["format_example"]), ("sft_q3_e8", "action_names"))
        self.assertTrue(g["A"]["closed"].startswith("2026-10-26"))

    def test_models_pools_and_pair(self):
        self.assertEqual(self.m["model_type"], "qwen3_vl")
        self.assertIn("Qwen3-VL-8B-Instruct", self.m["models"]["base_qwen3"])
        self.assertTrue(self.m["models"]["sft_q3_e8"].endswith("lf_q3formal/model/checkpoint-1232"))
        self.assertIsNone(self.m["pools"]["P0"]["path"])
        self.assertIsNone(self.m["pools"]["P1"]["path"])
        self.assertTrue(self.m["pools"]["P2q"]["path"].endswith("sft_v10_newformat_20261024/sft_trajectories.jsonl"))
        self.assertTrue(self.m["pools"]["P2f"]["path"].endswith("format_sft_v2a/sft_trajectories_format.jsonl"))
        standards = TP.load_standards()
        pair = self.m["rl"]["pair"]
        s0, d = standards[pair["S0"]], standards[pair["D"]]
        self.assertEqual((s0["reward_preset"], d["reward_preset"]), ("S0", "D_cond"))
        self.assertEqual({k: v for k, v in s0.items() if k != "reward_preset"},
                         {k: v for k, v in d.items() if k != "reward_preset"})  # a pair differs only in the reward
        self.assertEqual(self.m["rl"]["dense"]["variant"], "V4")
        self.assertEqual(self.m["rl"]["seeds"], [20261019, 20261021])
        self.assertEqual(self.m["rl"]["guards"]["kl_max"], 1.0)


if __name__ == "__main__":
    unittest.main()
