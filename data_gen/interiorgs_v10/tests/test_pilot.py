"""pilot.py: stratified selection, frames, overlays, top-down views."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import pilot as P
from data_gen.interiorgs_v10.admission import load_policy
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


def _fake_rows():
    rows = []
    for fam in P.FAMILY_QUOTA:
        for i in range(300):
            d = 1 + i % 8
            kind = "formal" if i % 3 else "reserve"
            rows.append((kind, {"task_id": f"{fam}_{i}", "task_type": fam, "scene_id": f"s{i % 25:02d}",
                                "difficulty": {"d_star": d}}))
    return rows


class PilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = write_synthetic_scenes(cls.root / "scenes")
        sdir = cls.paths["two_rooms"]
        cls.rows = _jsonable(generate_scene(sdir, "two_rooms", RunConfig(target_tasks_per_scene=4)))["rows"]
        cls.scene = load_scene(sdir, "two_rooms")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_select_tasks_is_stratified(self):
        picked = P.select_tasks(_fake_rows())
        self.assertEqual(Counter(r["task_type"] for r in picked), Counter(P.FAMILY_QUOTA))
        self.assertEqual(len({r["task_id"] for r in picked}), len(picked))
        for fam, n in P.FAMILY_QUOTA.items():
            bands = Counter(P.band_of(r["difficulty"]["d_star"]) for r in picked if r["task_type"] == fam)
            self.assertLessEqual(max(bands.values()) - min(bands.values()), 1)  # even over d* bands
            per_band_scene = Counter((P.band_of(r["difficulty"]["d_star"]), r["scene_id"])
                                     for r in picked if r["task_type"] == fam)
            self.assertEqual(max(per_band_scene.values()), 1)  # round-robin over scenes inside a band
        self.assertTrue(all(r["pilot_pool"] == "formal" for r in picked))  # formal first
        self.assertEqual(picked, P.select_tasks(_fake_rows()))  # deterministic

    def test_frames_overlay_and_topdown(self):
        row = self.rows[0]
        frames = P.task_frames(row, load_policy())
        d = row["difficulty"]["d_star"]
        sids = [f["state_id"] for f in frames]
        self.assertEqual(sids[0], "s0")
        self.assertEqual(sids[-1], f"s{d}")
        self.assertIn("A", frames[0]["overlay"])
        from PIL import Image

        img = Image.new("RGB", (256, 256), (40, 40, 40))
        drawn = P.draw_overlay(img, {"A": {"bbox": [20, 30, 120, 200], "color": "red", "image_size": [256, 256]}})
        arr = np.asarray(drawn)
        self.assertEqual(drawn.size, (256, 256))
        self.assertTrue(((arr[..., 0] > 200) & (arr[..., 1] < 80)).any())
        self.assertTrue((np.asarray(img) == 40).all())  # the original is untouched
        labels = json.loads((self.paths["two_rooms"] / "labels.json").read_text())
        out = self.root / "td.png"
        P.draw_topdown(self.scene, labels, row, out)
        self.assertGreater(out.stat().st_size, 1000)

    def test_check_page_selection_and_frames(self):
        import copy

        rows = []
        for i, r in enumerate(self.rows):
            r = copy.deepcopy(r)
            objs = r["target_object"]["objects"]
            methods = ("unique_label", "room", "anchor")[i % 3]
            for e in objs:
                e["ref_method"] = methods
                e["ref_anchor"] = None
            if methods == "anchor":
                other = {o.id: o for o in self.scene.objects}["13"]
                objs[0]["ref_anchor"] = {"id": other.id, "label": other.label, "bbox_min": other.aabb_min.tolist(),
                                         "bbox_max": other.aabb_max.tolist()}
            rows.append(("formal", r))
        picked = P.select_check_tasks(rows, groups=(("lateral_near_45", 2), ("name_only", 1), ("room", 1),
                                                     ("reference", 1)))
        self.assertEqual(len({r["task_id"] for r in picked}), len(picked))  # no task twice
        lateral = [r for r in picked if r["check_group"] == "lateral_near_45"]
        self.assertTrue(lateral and all(r["task_type"] == "projective_relations" for r in lateral))
        self.assertTrue(all(r["check_lateral_angle_deg"] is not None for r in lateral))
        for r in picked:
            if r["check_group"] == "reference":
                self.assertIn("anchor", P.ref_methods(r))
                frames = P.check_frames(r)
                self.assertEqual([f["state_id"] for f in frames], ["s0", f"s{r['difficulty']['d_star']}"])
                self.assertIn("RA", frames[0]["overlay"])
                self.assertEqual(frames[0]["overlay"]["RA"]["color"], "lime")


    def test_audit_selection(self):
        """Round 6: 100 formal val / test tasks spread over the families and d* bands, start and end frames."""
        self.assertEqual(P.audit_quota(["b", "a", "c"], 100), {"a": 34, "b": 33, "c": 33})
        run = self.root / "audit_run"
        run.mkdir(exist_ok=True)
        rows = [dict(r, scene_id=sid, task_id=r["task_id"] + "_" + sid) for sid in ("two_rooms", "val_scene", "tr")
                for r in self.rows]
        (run / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        manifest = self.root / "audit_split.json"
        manifest.write_text(json.dumps({"train_scenes": ["tr"], "val_scenes": ["val_scene"],
                                        "test_scenes": ["two_rooms"]}))
        out = self.root / "audit"
        from argparse import Namespace

        P.cmd_audit(Namespace(run=str(run), split_manifest=str(manifest), n=6, seed=1, out=str(out)))
        tasks = [json.loads(l) for l in (out / "pilot_tasks.jsonl").read_text().splitlines()]
        self.assertTrue(0 < len(tasks) <= 6)
        self.assertTrue(all(t["split"] in ("val", "test") and t["check_group"] == P.AUDIT_GROUP for t in tasks))
        frames = [json.loads(l) for l in (out / "frames.jsonl").read_text().splitlines()]
        self.assertEqual(len(frames), 2 * len(tasks))

if __name__ == "__main__":
    unittest.main()
