"""Round 12 freeze (the user's 3.1.4 of 2026-10-22): the format SFT of the original 7B: short conversations from the
existing SFT data shown with real action names (training_prep format-sft), and the evaluation's token entropy for
the acceptance check (evaluation run --entropy)."""

import copy
import math
import unittest

from data_gen.interiorgs_v10 import evaluation as E
from data_gen.interiorgs_v10 import release as R
from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.admission import load_policy

FAMILIES = ("absolute_positioning", "equidistance", "fov_inclusion", "projective_relations")


def record(i, family, split, assistant_turns, policy):
    conv = [{"role": "system", "content": R.protocol_system_prompt(family, policy)}]
    for k in range(assistant_turns):
        conv.append({"role": "user", "content": f"[Observation]:\n<image>\nturn {k}\n" + R._format_block(policy, k == 0),
                     "image_path": f"/f/{i}_{k}.png"})
        conv.append({"role": "assistant", "content": "<action>move_forward|</action>"})
    return {"id": f"sft_{i}", "source_task_id": f"t{i}", "task_type": family, "split": split, "conversations": conv}


class FormatSftTest(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()
        self.records = [record(i, FAMILIES[i % 4], "val" if i % 10 == 0 else "train", 1 + i % 3, self.policy)
                        for i in range(200)]

    def test_short_conversations_rewritten(self):
        res = TP.format_sft_records(self.records, self.policy, count=40, val_count=6)
        recs, rep = res["records"], res["report"]
        self.assertEqual((rep["train"], rep["val"]), (40, 6))
        self.assertEqual(sum(r["split"] == "train" for r in recs), 40)
        new = copy.deepcopy(self.policy)
        new["sft"]["export"]["format_example"] = "action_names"
        for r in recs:
            turns = [m for m in r["conversations"] if m["role"] == "assistant"]
            self.assertIn(len(turns), (1, 2))  # three-turn conversations are left out
            self.assertEqual(r["conversations"][0]["content"], R.protocol_system_prompt(r["task_type"], new))
            users = [m for m in r["conversations"] if m["role"] == "user"]
            for k, m in enumerate(users):
                self.assertTrue(m["content"].endswith("\n" + R._format_block(new, k == 0)))
                self.assertNotIn("action1", m["content"])
            self.assertNotIn("action1", r["conversations"][0]["content"])
            self.assertEqual(r["format_example"], "action_names")
        self.assertEqual(set(rep["by_family"]["train"]), set(FAMILIES))  # spread over the families
        self.assertEqual(rep["source_format_example"], "placeholder")
        # the same seed draws the same conversations; the source records stay as they were
        again = TP.format_sft_records(self.records, self.policy, count=40, val_count=6)["records"]
        self.assertEqual([r["id"] for r in again], [r["id"] for r in recs])
        self.assertIn("action1", self.records[1]["conversations"][1]["content"])

    def test_texts_that_do_not_match_are_left_out(self):
        bad = copy.deepcopy(self.records)
        for r in bad[:100]:
            r["conversations"][0]["content"] += " changed"
        res = TP.format_sft_records(bad, self.policy, count=20, val_count=2)
        self.assertGreater(res["report"]["skipped"]["system_prompt_differs"], 0)
        self.assertTrue(all(int(r["id"].split("_")[1]) >= 100 for r in res["records"]))

    def test_entropy_estimate(self):
        self.assertAlmostEqual(E.top_logprob_entropy([[math.log(0.5), math.log(0.5)]], 1.0), math.log(2), places=12)
        # a lower temperature sharpens the distribution
        lps = [[math.log(0.7), math.log(0.2), math.log(0.1)]]
        self.assertLess(E.top_logprob_entropy(lps, 0.8), E.top_logprob_entropy(lps, 1.0))
        self.assertAlmostEqual(E.top_logprob_entropy(lps, 0.0), E.top_logprob_entropy(lps, 1.0), places=12)
        self.assertIsNone(E.top_logprob_entropy([], 0.8))
        rows = [{"trajectory": [{"entropy": 0.1, "tokens": 10}, {"entropy": 0.4, "tokens": 30}]},
                {"trajectory": [{"answer": "x"}]}]
        self.assertEqual(E.token_entropy(rows), {"entropy": round((0.1 * 10 + 0.4 * 30) / 40, 5), "entropy_tokens": 40})
        self.assertEqual(E.token_entropy([{"trajectory": [{"answer": "x"}]}]), {})

    def test_stop_line_measure_in_metrics(self):
        base = {"task_type": "equidistance", "d_star": 3, "actions_to_success": None, "actions": 4, "collisions": 0,
                "success": False}
        rows = [dict(base, index=0, turns=4, invalid_turns=2), dict(base, index=1, turns=1, invalid_turns=0)]
        m = E.metrics(rows)
        self.assertEqual(m["invalid_rate"], round(2 / 5, 4))  # pooled over the turns
        self.assertEqual(m["invalid_episode_mean"], round((0.5 + 0.0) / 2, 4))  # the stop line's measure
        self.assertNotIn("entropy", m)


if __name__ == "__main__":
    unittest.main()
