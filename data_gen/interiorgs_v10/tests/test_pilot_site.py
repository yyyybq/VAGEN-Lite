"""pilot_site.py: the labelling site is built from a pilot directory and shows no machine judgement."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import pilot as P
from data_gen.interiorgs_v10 import pilot_site as PS
from data_gen.interiorgs_v10.admission import load_policy
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class PilotSiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = write_synthetic_scenes(cls.root / "scenes")
        sdir = cls.paths["two_rooms"]
        cls.rows = _jsonable(generate_scene(sdir, "two_rooms", RunConfig(target_tasks_per_scene=4)))["rows"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_site_is_blind(self):
        pilot = self.root / "pilot"
        (pilot / "frames" / "plain").mkdir(parents=True, exist_ok=True)
        (pilot / "frames" / "boxes").mkdir(parents=True, exist_ok=True)
        tasks = [dict(r, pilot_pool="formal") for r in self.rows[:2]]
        frames = [f for r in tasks for f in P.task_frames(r, load_policy())]
        examples = [{"example_id": f"vis{i:04d}", "group": "general" if i < 6 else "embedded", "scene_id": "two_rooms",
                     "target": {"id": "1", "label": "sofa", "ref": "the sofa"}, "c2w": np.eye(4).tolist(),
                     "metrics": {"visible_fraction": 0.123456, "area_fraction": 0.004321},
                     "bin": {"axis": "fraction", "lo": 0.1, "hi": 0.2}} for i in range(8)]
        (pilot / "pilot_tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tasks))
        (pilot / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))
        (pilot / "threshold_examples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples))
        no_task = {"scenes": {"two_rooms": [{"ins_id": "5", "label": "cabinet", "kind": "uncertain",
                                             "bbox_min": [0, 0, 0], "bbox_max": [1, 2, 1], "reason": "说明"}]}}
        (self.root / "nt.json").write_text(json.dumps(no_task))
        figs = self.root / "figs"
        figs.mkdir(exist_ok=True)
        (figs / "two_rooms_5.png").write_bytes(b"png")
        summary = PS.build_site(pilot, self.root / "site", self.root / "nt.json", figs)
        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(summary["visibility_examples"], 8)
        self.assertEqual(summary["uncertain_boxes"], 1)
        html = (self.root / "site" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("/*__DATA__*/null", html)
        for leak in ("d_star", "certificate", "pilot_pool", "visible_fraction", "0.123456", "area_fraction",
                     "\"bin\"", "reserve"):
            self.assertNotIn(leak, html.split("<script>")[1].split("const STORE_KEY")[0], leak)
        data = PS.build_data(pilot, no_task, figs)
        groups = [v["group"] for v in data["visibility"]]
        self.assertEqual(groups, sorted(groups, key=lambda g: g != groups[0]))  # general first, then embedded
        self.assertTrue((self.root / "site" / "img" / "uncertain" / "two_rooms_5.png").exists())


if __name__ == "__main__":
    unittest.main()
