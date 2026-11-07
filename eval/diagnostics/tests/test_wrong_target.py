"""eval/diagnostics/wrong_target.py (the user's item 6b of 2026-11-07)."""
import json
import unittest
from pathlib import Path

from eval.diagnostics import wrong_target as WT

EVAL = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/m3_g_s20261019/eval_test_v2_screened")


def ep(task_type, success, start, gates):
    return {"task_type": task_type, "success": success, "start": {"failed_gates": start},
            "trajectory": [{"failed_gates": g} for g in gates]}


class WrongTargetTest(unittest.TestCase):
    def test_best_index_and_soft_success(self):
        e = ep("absolute_positioning", False, ["distance", "visible"], [["distance", "inside_frame"], ["distance"], ["visible"]])
        self.assertEqual(WT.best_index(e), 1)  # ties go to the earlier turn
        self.assertTrue(WT.soft_success(e))
        self.assertEqual(WT.best_index(ep("x", False, ["a"], [["a", "b"]])), -1)  # the start pose
        self.assertFalse(WT.soft_success(ep("absolute_positioning", False, ["a", "b"], [["visible"]])))
        self.assertFalse(WT.soft_success(ep("absolute_positioning", True, [], [["distance"]])))
        self.assertFalse(WT.soft_success(ep("fov_inclusion", False, [], [["fully_in_frame"]])))  # no tolerance condition
        self.assertTrue(WT.soft_success(ep("projective_relations", False, ["margin", "relation"], [["margin"]])))

    def test_swap_item(self):
        a = {"id": "a", "label": "lamp", "bbox_min": [0, 0, 0], "bbox_max": [1, 1, 1.5]}
        b = {"id": "b", "label": "sofa", "bbox_min": [2, 0, 0], "bbox_max": [3, 1, 1.0]}
        row = {"task_id": "t", "task_type": "occlusion_alignment", "target_object": {"objects": [a, b]},
               "target_region": {"params": {"object_a_id": "a", "object_b_id": "b", "object_a_center": [0, 0, 0],
                                            "object_b_center": [1, 1, 1], "occluded": "a", "occluder": "b"}}}
        s, broken = WT.swap_item(row)
        p = s["target_region"]["params"]
        self.assertEqual((p["occluded"], p["occluder"], p["object_a_id"], p["object_a_center"]), ("b", "a", "b", [1, 1, 1]))
        self.assertEqual([o["id"] for o in s["target_object"]["objects"]], ["b", "a"])
        self.assertEqual(row["target_region"]["params"]["occluded"], "a")  # the original is unchanged
        self.assertIsInstance(broken, list)
        with self.assertRaises(ValueError):
            WT.swap_item(dict(row, task_type="equidistance"))

    def test_mcnemar_and_indicators(self):
        m = WT.mcnemar({"a": True, "b": True, "c": False, "d": True}, {"a": False, "b": False, "c": False, "e": True})
        self.assertEqual((m["tasks"], m["only_first"], m["only_second"]), (3, 2, 0))
        self.assertAlmostEqual(m["p"], 0.5)
        self.assertEqual(WT.mcnemar({}, {})["p"], 1.0)
        res = {"episodes": [{"task_id": "s", "success": True},
                            {"task_id": "w", "success": False, "collisions_match": True, "gates_match": True, "wrong_target": True},
                            {"task_id": "u", "success": False, "collisions_match": False, "gates_match": True, "wrong_target": False}]}
        self.assertEqual(WT.indicators(res), {"s": False, "w": True})

    @unittest.skipUnless((EVAL / "m3gs1d_250" / "results.jsonl").exists() and WT.GS_ROOT.exists(), "cluster data")
    def test_replay_matches_the_recorded_episode(self):
        tasks = {r["task_id"]: r for r in WT.read_jsonl(EVAL / "tasks.jsonl")}
        eps = [e for e in WT.read_jsonl(EVAL / "m3gs1d_250" / "results.jsonl")
               if e["task_type"] == "occlusion_alignment" and not e["success"]][:2]
        res = WT.analyse(tasks, eps)
        for r in res["episodes"]:
            self.assertTrue(r["collisions_match"] and r["gates_match"], json.dumps(r)[:400])
        self.assertEqual(res["families"]["occlusion_alignment"]["judged"], 2)


if __name__ == "__main__":
    unittest.main()
