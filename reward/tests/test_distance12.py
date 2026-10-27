"""reward/distance12.py: the same tables as the main line's distance_table.build, the strict prefilter changes no
table, the limits and the backup replacement.  Synthetic scenes in a temporary directory only (no real paths)."""

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import distance_table as DT
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes
from reward import distance12 as D12

ARRAYS = ("keys", "dist", "depth_of")


class Distance12Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        paths = write_synthetic_scenes(cls.root / "scenes")
        cls.gs_root = cls.root / "scenes"
        cls.scene_dir = paths["two_rooms"]
        cls.scene = load_scene(cls.scene_dir, "two_rooms", wall_model="thick")
        res = _jsonable(generate_scene(cls.scene_dir, cls.scene_dir.name, RunConfig(target_tasks_per_scene=5)))
        cls.rows = res["rows"][:5]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_table_as_main_build(self):
        self.assertGreaterEqual(len(self.rows), 3)
        for row in self.rows:
            depth = 3  # small graphs: the tables must match at any depth
            main = DT.build(self.scene, row, depth=depth)
            ours = D12.build_task(self.scene, row, depth)
            self.assertEqual(ours["status"], "ok")
            for k in ARRAYS:
                self.assertTrue(np.array_equal(main[k], ours[k]), (row["task_id"], k))
            self.assertEqual(ours["d_init"], main["d_init"])
            self.assertEqual(ours["per_level"][0], 1)
            self.assertEqual(sum(ours["per_level"]), ours["states"])

    def test_strict_prefilter_keeps_every_table(self):
        strict_rows = 0
        for row in self.rows:
            depth = 3  # small graphs: the tables must match at any depth
            loose = D12.build_task(self.scene, row, depth, prefilter="loose", check_all=True)
            strict = D12.build_task(self.scene, row, depth, prefilter="strict")
            self.assertEqual(loose["prefilter_missed_success"], 0)
            for k in ARRAYS:
                self.assertTrue(np.array_equal(loose[k], strict[k]), (row["task_id"], k))
            self.assertLessEqual(strict["prefilter_pass"], loose["prefilter_pass"])
            strict_rows += strict["prefilter"] == "strict"
            if D12.strict_objects(row) is None:
                self.assertEqual(strict["prefilter"], "loose")
        print(f"strict prefilter applied to {strict_rows} of {len(self.rows)} synthetic rows")

    def test_strict_mask_is_a_subset(self):
        for row in self.rows:
            keys, P, R, depth_of, _ = DT.reachable_graph(D12.FastGeometry.for_scene(self.scene),
                                                          np.asarray(row["init_camera"]["extrinsics"], float), 3)
            c2w = D12.to_c2w(P, R)
            loose, _ = D12.prefilter_mask(row, c2w, "loose")
            strict, _ = D12.prefilter_mask(row, c2w, "strict")
            self.assertFalse(np.any(strict & ~loose))

    def test_limits(self):
        row = self.rows[0]
        depth = 3  # small graphs: the tables must match at any depth
        over = D12.build_task(self.scene, row, depth, max_states=3)
        self.assertEqual(over["status"], "over_states")
        self.assertNotIn("keys", over)

        def slow(c2w):  # about 2 ms of CPU a state
            t = __import__("time").process_time()
            while __import__("time").process_time() - t < 0.002:
                pass
            return False

        old, D12.CHECK_EVERY = D12.CHECK_EVERY, 1
        try:
            timed = D12.build_task(self.scene, row, depth, judge_cpu_hours=0.01 / 3600, oracle=slow)
        finally:
            D12.CHECK_EVERY = old
        self.assertEqual(timed["status"], "over_time")

    def test_pick_backup_prefers_the_stratum(self):
        def r(t, fam, d):
            return {"task_id": t, "task_type": fam, "difficulty": {"d_star": d}}

        backups = [r("b1", "equidistance", 5), r("b2", "projective_relations", 7), r("b3", "projective_relations", 4),
                   r("b4", "projective_relations", 3)]
        failed = r("f", "projective_relations", 3)
        self.assertEqual(D12.pick_backup(failed, backups, set())["task_id"], "b4")
        self.assertEqual(D12.pick_backup(failed, backups, {"b4"})["task_id"], "b3")
        self.assertEqual(D12.pick_backup(failed, backups, {"b2", "b3", "b4"})["task_id"], "b1")
        self.assertIsNone(D12.pick_backup(failed, backups, {"b1", "b2", "b3", "b4"}))

    def _cli(self, out, ids, backups, extra=()):
        rows_path = self.root / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        (self.root / "ids.txt").write_text("".join(t + "\n" for t in ids))
        (self.root / "backup.txt").write_text("".join(t + "\n" for t in backups))
        D12.main(["build", "--rows", str(rows_path), "--task-ids", str(self.root / "ids.txt"), "--out", str(out),
                  "--gs-root", str(self.gs_root), "--backup-ids", str(self.root / "backup.txt"),
                  "--target", str(len(ids)), "--depth", "3", "--workers", "2", *extra])
        return json.loads((out / "DONE.json").read_text())

    def test_cli_builds_and_resumes(self):
        ids, backups = [r["task_id"] for r in self.rows[:3]], [r["task_id"] for r in self.rows[3:5]]
        out = self.root / "cli_ok"
        done = self._cli(out, ids, backups)
        self.assertEqual(done["final"], 3)
        self.assertEqual(done["replacements"], 0)
        self.assertEqual(sorted((out / "final_ids.txt").read_text().split()), sorted(ids))
        for t in ids:
            self.assertTrue(DT.table_path(out, t).is_file())
            DT.TableCache(out).get(t)  # the env's reader takes our files
        again = self._cli(out, ids, backups)  # resume: nothing rebuilt, nothing logged twice
        self.assertEqual(again["final"], 3)
        lines = [json.loads(l) for l in (out / "summary.jsonl").read_text().splitlines()]
        self.assertEqual(len(lines), 3)

    def test_cli_shards_fill_their_share(self):
        ids, backups = [r["task_id"] for r in self.rows[:4]], [r["task_id"] for r in self.rows[4:5]]
        out = self.root / "cli_shards"
        rows_path = self.root / "rows_s.jsonl"
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        (self.root / "ids_s.txt").write_text("".join(t + "\n" for t in ids))
        (self.root / "backup_s.txt").write_text("".join(t + "\n" for t in backups))
        finals = []
        for shard in ("0/2", "1/2"):
            D12.main(["build", "--rows", str(rows_path), "--task-ids", str(self.root / "ids_s.txt"), "--out", str(out),
                      "--gs-root", str(self.gs_root), "--backup-ids", str(self.root / "backup_s.txt"), "--fill",
                      "--shard", shard, "--depth", "3", "--workers", "2"])
            tag = shard.replace("/", "of")
            done = json.loads((out / f"DONE_{tag}.json").read_text())
            mine = [t for t in ids if DT.in_shard(t, shard)]
            self.assertEqual(done["tasks"], len(mine))
            self.assertEqual(done["final"], len(mine))
            finals += (out / f"final_ids_{tag}.txt").read_text().split()
            self.assertTrue((out / f"summary_{tag}.jsonl").is_file())
        self.assertEqual(sorted(finals), sorted(ids))

    def test_cli_replaces_over_limit_tasks(self):
        ids, backups = [r["task_id"] for r in self.rows[:2]], [r["task_id"] for r in self.rows[2:4]]
        out = self.root / "cli_over"
        done = self._cli(out, ids, backups, ("--max-states", "3"))
        self.assertEqual(done["final"], 0)
        reps = json.loads((out / "replacements.json").read_text())
        self.assertEqual(len(reps), 2)
        self.assertTrue(all(x["reason"] == "over_states" and x["by_status"] == "over_states" for x in reps))


if __name__ == "__main__":
    unittest.main()
