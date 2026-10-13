"""r1_regression.py: plan / trace / compare on a synthetic R1-style (canonical v1 Projective) row."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import r1_regression as RR
from data_gen.interiorgs_v10 import sampling
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

REPO = Path(__file__).resolve().parents[3]


class R1RegressionTest(unittest.TestCase):
    def test_plan_trace_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = write_synthetic_scenes(tmp / "scenes")
            sdir = paths["two_rooms"]
            scene = load_scene(sdir, sdir.name)
            by_id = {o.id: o for o in scene.objects}
            pose = C.pose_from_xy_yaw(0.6, 3.0, math.radians(0))
            for a in reversed(["move_left", "move_left", "move_left"]):
                pose = C.apply_action(pose, C.INVERSE_ACTION[a])
            row = sampling.projective_item(sdir.name, by_id["2"], by_id["4"], "left", pose)
            row["task_id"] = "r1_like_0"
            rows_path = tmp / "rows.jsonl"
            rows_path.write_text(json.dumps(row) + "\n")
            plan = RR.plan([row], sdir.parent)
            self.assertEqual(len(plan), 1)
            kinds = [s["kind"] for s in plan[0]["sequences"]]
            self.assertEqual(kinds[-RR.RANDOM_WALKS:], [f"random_walk_{k}" for k in range(RR.RANDOM_WALKS)])
            self.assertEqual(plan, RR.plan([row], sdir.parent))  # deterministic
            plan_path = tmp / "plan.json"
            plan_path.write_text(json.dumps(plan))
            out = tmp / "trace.json"
            RR.trace(REPO, rows_path, sdir.parent, plan_path, out)
            trace = json.loads(out.read_text())
            self.assertEqual(len(trace), len(kinds))
            self.assertTrue(all(t["turns"] for t in trace))
            self.assertTrue(RR.compare(trace, trace)["identical"])
            changed = copy.deepcopy(trace)
            changed[0]["turns"][0]["reward"] = "123.0"
            rep = RR.compare(trace, changed)
            self.assertFalse(rep["identical"])
            self.assertEqual(rep["differences"], 1)
            self.assertIn("reward", rep["first_differences"][0]["where"])


if __name__ == "__main__":
    unittest.main()
