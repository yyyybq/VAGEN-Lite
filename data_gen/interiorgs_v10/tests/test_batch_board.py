"""batch_board.py: the round-12 batch board (states, progress, queue times, expected end, results)."""

import datetime
import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import batch_board as B


def _row(tid, fam, ok):
    return {"task_id": tid, "task_type": fam, "success": ok}


class BatchBoardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        tasks = {"val": [("v1", "fov_inclusion"), ("v2", "equidistance"), ("v3", "occlusion_alignment")],
                 "test": [("t1", "fov_inclusion"), ("t2", "occlusion_alignment")]}
        added = {"val": [("v4", "occlusion_alignment")], "test": [("t3", "occlusion_alignment")]}
        self.sets = {}
        for ver in ("v1", "v2", "v2_oa"):
            self.sets[ver] = {}
            for part in ("val", "test"):
                d = self.root / f"{ver}_{part}"
                d.mkdir()
                rows = tasks[part] + (added[part] if ver != "v1" else [])
                if ver == "v2_oa":
                    rows = [r for r in rows if r[1] == "occlusion_alignment"]
                (d / "tasks.jsonl").write_text("".join(json.dumps({"task_id": t, "task_type": f}) + "\n" for t, f in rows))
                self.sets[ver][part] = str(d)

    def tearDown(self):
        self._tmp.cleanup()

    def result(self, ver, part, name, solved):
        d = Path(self.sets[ver][part]) / name
        d.mkdir(parents=True, exist_ok=True)
        tasks = [json.loads(l) for l in (Path(self.sets[ver][part]) / "tasks.jsonl").read_text().splitlines()]
        (d / "results.jsonl").write_text("".join(json.dumps(_row(t["task_id"], t["task_type"], t["task_id"] in solved)) + "\n"
                                                 for t in tasks))
        (d / "done.json").write_text("{}")

    def test_merged_results_and_comparison(self):
        # a round-11 model: v1 results plus the occlusion v2 re-evaluation
        self.result("v1", "val", "old", {"v1", "v3"})
        self.result("v1", "test", "old", {"t1"})
        self.result("v2_oa", "val", "old", {"v4"})   # the re-run: v3 now fails, the added v4 succeeds
        self.result("v2_oa", "test", "old", set())
        merged = B.results(self.sets, "old", "val")
        self.assertEqual(sorted(merged), ["v1", "v2", "v3", "v4"])
        self.assertEqual((merged["v3"]["success"], merged["v4"]["success"]), (False, True))
        self.assertIsNone(B.results(self.sets, "missing", "val"))
        # a round-12 model evaluated on v2 directly
        self.result("v2", "val", "new_050", {"v1", "v2", "v4"})
        self.result("v2", "test", "new_050", {"t1", "t3"})
        c = B.compare(self.sets, "old", "new_050")
        self.assertEqual((c["base_only"], c["other_only"]), (0, 2))  # v2 and t3
        self.assertAlmostEqual(c["other"], 5 / 7)
        self.result("v2", "val", "new_100", {"v1", "v2", "v4"})   # ties with 050 on val: the earlier is picked
        self.result("v2", "test", "new_100", set())
        self.assertEqual(B.best_checkpoint(self.sets, "new", 100), "new_050")

    def test_rows_and_render(self):
        now = datetime.datetime(2026, 10, 20, 8, 0, tzinfo=datetime.timezone.utc)

        def chain(name, progress):
            d = self.root / name
            d.mkdir()
            (d / "progress.json").write_text(json.dumps(progress))
            return str(d)

        queued = chain("q", {"state": "running", "plan": {"endpoint": 300},
                             "steps": {"rl": {"state": "running", "jobs": {"as0/rl": "STARTING", "as0/renderer": "RUNNING",
                                                                            "ad/rl": "STARTING", "ad/renderer": "RUNNING"}}},
                             "queue": {"pt-1": {"what": "rl/as0/rl", "queued_s": 7200, "started": None},
                                       "pt-2": {"what": "rl/as0/renderer", "queued_s": 60, "started": "x"}}})
        running = chain("r", {"state": "running", "plan": {"endpoint": 300},
                              "steps": {"rl": {"state": "running", "jobs": {"x/rl": "RUNNING", "x/renderer": "RUNNING"}}},
                              "rl_runs": {"x": {"last_update": 100, "seconds_per_update_mean_last10": 120}},
                              "job_retries": {"rl/x": [{"reason": "trainer ended FAILED"}]}})
        failed = chain("f", {"state": "failed", "steps": {"sft": {"state": "failed", "error": "ChainError: boom"}}})
        self.result("v2", "val", "d_050", {"v1"})
        self.result("v2", "test", "d_050", {"t1"})
        self.result("v2", "val", "s_050", {"v1", "v2"})
        self.result("v2", "test", "s_050", {"t1", "t3"})
        self.result("v1", "val", "sftc", {"v1"})
        self.result("v1", "test", "sftc", set())
        self.result("v2_oa", "val", "sftc", set())
        self.result("v2_oa", "test", "sftc", set())
        self.result("v2", "val", "sft_g_e2", {"v1", "v2"})
        self.result("v2", "test", "sft_g_e2", {"t1"})
        done = chain("d", {"state": "done", "plan": {"endpoint": 50},
                           "steps": {"evaluate": {"state": "done"}, "pick_sft": {"decision": {"best": "sft_g_e2"}}}})
        manifest = {"title": "测试批次", **self.sets, "experiments": [
            {"id": "E1", "name": "排队", "chain": queued, "runs": {"as0": "S0", "ad": "D"}},
            {"id": "E2", "name": "在跑", "chain": running, "runs": {"x": "S0"}},
            {"id": "E3", "name": "失败", "chain": failed},
            {"id": "E4", "name": "完成", "chain": done, "runs": {"s": "S0", "d": "D"}, "controls": {"s": "sftc", "d": "sftc"},
             "sft_group": "g", "sft_control": "sftc"},
            {"id": "E5", "name": "没开始", "chain": str(self.root / "none")}]}
        md = B.render(manifest, now)
        self.assertIn("| E1 | 排队 | 排队中 | RL |", md)
        self.assertIn("1 个在排队（最长 2.0 小时）", md)
        self.assertIn("| E2 | 在跑 | 运行中 | RL | x 100/300 |", md)
        self.assertIn("10-20 16:10", md)  # 200 updates x 120 s = 6.7 h, plus 1.5 h of evaluation
        self.assertIn("rl/x 重提 1 次（trainer ended FAILED）", md)
        self.assertIn("| E3 | 失败 |", md)
        self.assertIn("失败原因：ChainError: boom", md)
        self.assertIn("| E5 | 没开始 | 未启动 |", md)
        self.assertIn("SFT 挑出 sft_g_e2", md)
        self.assertIn("S0（s）挑出 s_050", md)
        self.assertIn("D 对 S0", md)
        self.assertIn("对 sftc", md)
        out = self.root / "board.md"
        (self.root / "m.json").write_text(json.dumps(dict(manifest, out=str(out))))
        self.assertEqual(B.main(["--manifest", str(self.root / "m.json")]), 0)
        self.assertIn("测试批次", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
