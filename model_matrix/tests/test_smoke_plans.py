"""The step-2 smoke plans (model_matrix/smoke/plans.py): what each runs, and that a plan is written once, its commit
changed only before its chain starts."""
import json
import tempfile
import unittest
from pathlib import Path


class SmokePlansTest(unittest.TestCase):
    def test_plans(self):
        from model_matrix.smoke import plans as P

        all_plans = P.plans("abc", "20261024")
        self.assertEqual(sorted(all_plans), ["m_cambrian7b_base_20261024", "m_cambrian7b_smoke_20261024",
                                             "m_qwen25vl7b_base_20261024", "m_qwen25vl7b_smoke_20261024",
                                             "m_qwen3vl8b_base_20261024", "m_qwen3vl8b_smoke_20261024"])
        qb = all_plans["m_qwen25vl7b_base_20261024"]
        self.assertNotIn("model_type", qb)  # the default Qwen2.5-VL path
        self.assertEqual(qb["eval"]["models"], {"m_qwen25vl7b_base_t08": {"model": P.QWEN25VL, "options": P.SAMPLED}})
        self.assertEqual(all_plans["m_qwen3vl8b_base_20261024"]["model_type"], "qwen3_vl")
        cb, cs = all_plans["m_cambrian7b_base_20261024"], all_plans["m_cambrian7b_smoke_20261024"]
        self.assertEqual((cb["model_type"], cb["eval_only"]), ("cambrian_qwen", True))
        self.assertEqual(cb["eval"]["models"]["m_cambrian7b_base_t08"], {"model": P.CAMBRIAN, "options": P.SAMPLED})
        self.assertEqual((cs["model_type"], cs["model"], cs["endpoint"], cs["sft_epochs"]),
                         ("cambrian_qwen", P.CAMBRIAN, 10, 1))
        self.assertTrue(cs["rl"]["logprob_audit"])
        self.assertEqual(cs["rl_runs"], {"cv4smoke": {"rl": P.V4}})
        self.assertNotIn("model_type", all_plans["m_qwen25vl7b_smoke_20261024"])  # the default path
        for plan in all_plans.values():
            self.assertEqual((plan["pool"], plan["job_prefix"], plan["commit"]), ("idle", "models", "abc"))

    def test_written_once_recommitted_only_before_the_start(self):
        from model_matrix.smoke import plans as P

        with tempfile.TemporaryDirectory() as tmp:
            mk = lambda c: {"a": {"tr": f"{tmp}/a", "commit": c}, "b": {"tr": f"{tmp}/b", "commit": c}}
            P.write(mk("one"))
            P.write(mk("two"))  # kept
            self.assertEqual(json.loads(Path(f"{tmp}/a/plan.json").read_text())["commit"], "one")
            Path(f"{tmp}/b/progress.json").write_text("{}")  # b's chain has started
            P.write(mk("two"), recommit=True)
            self.assertEqual(json.loads(Path(f"{tmp}/a/plan.json").read_text())["commit"], "two")
            self.assertEqual(json.loads(Path(f"{tmp}/b/plan.json").read_text())["commit"], "one")


if __name__ == "__main__":
    unittest.main()
