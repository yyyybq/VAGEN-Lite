"""fail_analysis.py: one primary failure mode per failed episode, the same definition for every round."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import fail_analysis as F


def turn(actions, gates, progress):
    return {"actions": actions, "failed_gates": gates, "progress": progress}


def row(i, fam, success, end=None, collisions=0, traj=(), start=0.3):
    return {"index": i, "task_id": f"t{i}", "task_type": fam, "d_star": 3, "success": success, "end_reason": end,
            "collisions": collisions, "turns": len(traj), "trajectory": list(traj), "start": {"progress": start}}


class FailAnalysisTest(unittest.TestCase):
    def test_modes_in_priority_order(self):
        self.assertEqual(F.mode(row(0, "f", False, end="consecutive_collision")), "collision")     # rounds 6-9
        self.assertEqual(F.mode(row(0, "f", False, collisions=3, end="max_turns")), "collision")    # round 10 rules
        self.assertEqual(F.mode(row(0, "f", False, collisions=2, end="consecutive_invalid")), "invalid_end")
        self.assertEqual(F.mode(row(0, "f", False, end="low_info")), "low_info_end")
        near = [turn(["move_forward"], ["a", "b"], 0.4), turn(["move_forward"], ["a"], 0.6)]
        self.assertEqual(F.mode(row(0, "f", False, traj=near)), "near_miss")
        back = [turn(["move_backward"], ["a", "b"], 0.2), turn(["move_backward"], ["a", "b"], 0.1)]
        self.assertEqual(F.mode(row(0, "f", False, traj=back)), "went_backwards")
        osc = [turn(["turn_left"], ["a", "b"], 0.3), turn(["turn_right"], ["a", "b"], 0.3),
               turn(["turn_left"], ["a", "b"], 0.3), turn(["turn_right"], ["a", "b"], 0.3)]
        self.assertEqual(F.mode(row(0, "f", False, traj=osc)), "oscillating")
        self.assertEqual(F.mode(row(0, "f", False, traj=[turn(["move_left"], ["a", "b"], 0.35)])), "wandered")

    def test_report_groups_and_table(self):
        osc = [turn(["turn_left"], ["a", "b"], 0.3), turn(["turn_right"], ["a", "b"], 0.3),
               turn(["turn_left"], ["a", "b"], 0.3), turn(["turn_right"], ["a", "b"], 0.3)]
        rows = [row(0, "fov_inclusion", True), row(1, "fov_inclusion", False, collisions=4, traj=osc),
                row(2, "fov_inclusion", False, traj=osc), row(3, "equidistance", False, traj=[turn(["move_left"], ["a", "b"], 0.31)]),
                {"index": 4, "error": "renderer"}]
        rep = F.analyse(rows, tasks=[{"task_description": f"task {i}"} for i in range(5)])
        self.assertEqual(rep["all"]["episodes"], 4)
        self.assertEqual(rep["all"]["failed"], 3)
        self.assertEqual(rep["all"]["modes"], {"collision": 1, "oscillating": 1, "wandered": 1})
        self.assertEqual(rep["all"]["groups"], {"撞墙": 1, "来回走": 1, "丢物体": 1, "差一个条件": 0, "格式/画面结束": 0})
        self.assertEqual(rep["all"]["group_share_of_episodes"]["撞墙"], 0.25)
        self.assertEqual(rep["fov_inclusion"]["examples"]["collision"][0]["desc"], "task 1")
        self.assertEqual(rep["all"]["examples"], {})  # examples are kept per family only
        self.assertEqual(rep["all"]["blocked_per_failure"], round(4 / 3, 2))
        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp)
            (ev / "tasks.jsonl").write_text("".join(json.dumps({"task_description": f"task {i}"}) + "\n" for i in range(5)))
            (ev / "m").mkdir()
            (ev / "m" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = ev / "fail.json"
            F.main(["--eval", str(ev), "--models", "m=第十轮", "--out", str(out)])
            self.assertIn("第十轮", json.loads(out.read_text(encoding="utf-8")))
            table = out.with_suffix(".md").read_text(encoding="utf-8")
            self.assertIn("| 第十轮 | 4 | 3 | 1（25.0%） | 1（25.0%） | 1（25.0%） |", table)

    def test_blocked_reaction_and_focus(self):
        """Round 11: the turn after a blocked move (repeat / back off / blocked again) and the three conditions."""
        fb = "Environment Feedback: 1 of 2 actions executed; blocked: move_backward (blocked by wall).\n"
        traj = [dict(turn(["move_forward", "move_backward"], ["inside_frame", "min_area"], 0.2), collisions=1, user="x"),
                dict(turn(["move_forward"], ["inside_frame"], 0.3), collisions=0, user=fb),          # backs off
                dict(turn(["move_left"], ["target_a_clearly_visible"], 0.3), collisions=1, user="ok"),
                dict(turn(["move_right"], ["target_a_clearly_visible", "occluder_clearly_visible"], 0.3),
                     collisions=1, user="blocked: move_left (blocked by sofa)."),                    # backs off, blocked again
                dict(turn(["move_right"], ["min_area", "inside_frame"], 0.3), collisions=0,
                     user="blocked: move_right (blocked by sofa)."),                                 # repeats
                dict(turn(["turn_left"], ["inside_frame"], 0.3), collisions=1, user="")]             # no next turn
        self.assertEqual(F.blocked_in_feedback(fb), ["move_backward"])
        self.assertEqual(F.blocked_in_feedback("Action executed."), [])
        self.assertEqual([F.focus_of(g) for g in ("inside_frame", "reference_b_clearly_visible", "occluder_clearly_visible",
                                                  "min_area", "equidistant")],
                         ["目标不在画面里", "没有一个面看清楚", None, "目标太小", None])
        rep = F.analyse([row(0, "equidistance", False, traj=traj), row(1, "equidistance", True, traj=traj[:2])])
        b = rep["equidistance"]["blocked_reaction"]
        self.assertEqual((b["turns"], b["named"], b["repeat"], b["back_off"], b["blocked_again"]), (4, 4, 1, 3, 1))
        self.assertEqual(b["blocked_again_share"], 0.25)
        self.assertEqual(rep["all"]["focus_at_end"], {"目标不在画面里": 1, "没有一个面看清楚": 0, "目标太小": 0})
        self.assertEqual(rep["all"]["near_misses"], 1)
        self.assertEqual(rep["all"]["focus_near_miss"], {"目标不在画面里": 1, "没有一个面看清楚": 0, "目标太小": 0})
        md = F.markdown({"m": rep})
        self.assertIn("| m | 4 | 1（25.0%） | 3（75.0%） | 1（25.0%） |", md)
        self.assertIn("| m | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |", md)


if __name__ == "__main__":
    unittest.main()
