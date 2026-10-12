"""make_viewer.py: success regions contain each task's terminal position; the page builds."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import make_viewer as MV
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class ViewerTest(unittest.TestCase):
    def test_paths_end_in_their_success_region_and_page_builds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            paths = write_synthetic_scenes(tmp / "scenes")
            sdir = paths["two_rooms"]
            res = _jsonable(generate_scene(sdir, sdir.name, RunConfig(target_tasks_per_scene=5)))
            rows = res["rows"]
            out = MV.build_scene((str(sdir), rows, []))
            self.assertEqual(len(out["tasks"]), len(rows))
            for t in out["tasks"]:
                region = np.asarray(out["regions"][t["region"]], float).reshape(-1, 2)
                self.assertGreater(len(region), 0, t["id"])
                end = np.asarray(t["path"][-1][:2])
                # the terminal pose is a success pose: its 0.2 m cell, or a neighbour, is in the region
                self.assertLess(np.min(np.abs(region - end).max(axis=1)), MV.REGION_SPACING_M * 1.5, t["id"])
                self.assertEqual(len(t["path"]), t["d"] + 1)
            run = tmp / "run"
            run.mkdir()
            (run / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            html = tmp / "viewer.html"
            MV.main(["--run", str(run), "--scenes-root", str(sdir.parent), "--extra-scenes", sdir.name,
                     "--out", str(html)])
            text = html.read_text(encoding="utf-8")
            self.assertNotIn("/*__DATA__*/null", text)
            self.assertIn(rows[0]["task_id"], text)


if __name__ == "__main__":
    unittest.main()
