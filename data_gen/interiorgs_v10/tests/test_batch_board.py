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

    def test_quality(self):
        """Round 12 batch: a model's end progress over its failed episodes, the share of success conditions met at the
        end and the collision rate over validation + test; nothing for rows without a recorded trajectory."""
        for part in ("val", "test"):  # every task of the set, all failed the same way
            d = Path(self.sets["v2"][part]) / "m_100"
            d.mkdir(parents=True)
            tasks = [json.loads(l) for l in (Path(self.sets["v2"][part]) / "tasks.jsonl").read_text().splitlines()]
            rows = [{"task_id": t["task_id"], "task_type": "absolute_positioning", "success": False, "actions": 4,
                     "collisions": 1, "turns": 2, "invalid_turns": 1, "start": {"progress": 0.0, "failed_gates": ["distance"]},
                     "trajectory": [{"progress": 0.2, "failed_gates": ["distance", "visible", "in_front"]}]} for t in tasks]
            (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (d / "done.json").write_text("{}")
        self.assertEqual(B.quality(self.sets, "m_100"),
                         "；终点进度分（失败的题）0.200、终点满足的条件 50.0%、碰撞率 25.0%、读不出动作的回合 50.0%")
        self.result("v2", "val", "plain", {"v1"})  # rows without trajectories
        self.result("v2", "test", "plain", set())
        self.assertEqual(B.quality(self.sets, "plain"), "")

    def test_rows_and_render(self):
        now = datetime.datetime(2026, 10, 20, 8, 0, tzinfo=datetime.timezone.utc)

        def chain(name, progress):
            d = self.root / name
            d.mkdir()
            (d / "progress.json").write_text(json.dumps(progress))
            return str(d)

        queued = chain("q", {"state": "running", "plan": {"endpoint": 300},
                             "steps": {"rl": {"state": "running", "jobs": {"as0/rl": "STARTING", "as0/renderer": "RUNNING",
                                                                            "ad/rl": "QUEUEING", "ad/renderer": "RUNNING"}}},
                             "queue": {"pt-1": {"what": "rl/as0/rl", "queued_s": 7200, "started": None},
                                       "pt-2": {"what": "rl/as0/renderer", "queued_s": 60, "started": "x"}}})
        running = chain("r", {"state": "running", "plan": {"endpoint": 300},
                              "steps": {"rl": {"state": "running", "jobs": {"x/rl": "RUNNING", "x/renderer": "RUNNING"}}},
                              "rl_runs": {"x": {"last_update": 100, "seconds_per_update_mean_last10": 120}},
                              "job_retries": {"rl/x": [{"reason": "trainer ended FAILED"}]}})
        failed = chain("f", {"state": "failed", "steps": {"sft": {"state": "failed", "error": "ChainError: boom"}}})
        sft_wait = chain("w", {"state": "running", "steps": {"sft": {"state": "running", "jobs": {"s050": "RUNNING"}}},
                               "queue": {"pt-9": {"what": "sft/s050", "queued_s": 600, "started": None}}})
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
            {"id": "E5", "name": "没开始", "chain": str(self.root / "none")},
            {"id": "E6", "name": "SFT 等卡", "chain": sft_wait}]}
        md = B.render(manifest, now)
        self.assertIn("| E1 | 排队 | 排队中 | RL |", md)
        self.assertIn("1 个在排队（最长 2.0 小时）", md)
        self.assertIn("| E2 | 在跑 | 运行中 | RL | x 100/300 |", md)
        self.assertIn("10-20 16:10", md)  # 200 updates x 120 s = 6.7 h, plus 1.5 h of evaluation
        self.assertIn("rl/x 重提 1 次（trainer ended FAILED）", md)
        self.assertIn("| E3 | 失败 |", md)
        self.assertIn("失败原因：ChainError: boom", md)
        self.assertIn("| E5 | 没开始 | 未启动 |", md)
        self.assertIn("| E6 | SFT 等卡 | 排队中 | SFT |", md)  # the SFT step lists waiting jobs as running
        self.assertIn("SFT 挑出 sft_g_e2", md)
        self.assertIn("S0（s）挑出 s_050", md)
        self.assertIn("D 对 S0", md)
        self.assertIn("对 sftc", md)
        out = self.root / "board.md"
        (self.root / "m.json").write_text(json.dumps(dict(manifest, out=str(out))))

        # round 12 (the user's section 4): a section listed apart, named comparisons, chain health, guard records
        import os
        import time

        self.result("v2", "val", "k_050", {"v1", "v2", "v3", "v4"})
        self.result("v2", "test", "k_050", {"t1", "t2"})
        keep = chain("k", {"state": "running", "chain_pid": os.getpid(), "plan": {"endpoint": 50},
                           "steps": {"rl": {"state": "done"}, "evaluate": {"state": "done"}}})
        gone = chain("g", {"state": "running", "chain_pid": 2 ** 22 + 12345, "steps": {"rl": {"state": "running"}}})
        events = self.root / "events.jsonl"
        events.write_text(json.dumps({"utc": "2026-10-21T19:20:20+00:00", "kind": "chain_restart",
                                      "text": "plan: renderer_repack", "chains": [keep]}) + "\n"
                          + json.dumps({"utc": "2026-10-21T19:25:00+00:00", "kind": "board_restart",
                                        "text": "board loop gone, restarted"}) + "\n" + '{"utc": "2026-10-21T19:3')
        beat = self.root / "heartbeat"
        beat.write_text(str(now.timestamp() - 20 * 60))
        m2 = dict(manifest, health=True, events=str(events), heartbeat=str(beat), experiments=manifest["experiments"] + [
            {"id": "K1", "name": "第一批", "section": "密集奖励第一批", "chain": keep, "runs": {"k": "D-keep"},
             "controls": {"k": "d_050"}, "comparisons": [["k", "s_050"]]},
            {"id": "G1", "name": "进程没了", "section": "密集奖励第一批", "chain": gone}])
        md2 = B.render(m2, now)
        self.assertIn("## 密集奖励第一批", md2)
        self.assertLess(md2.index("| E6 |"), md2.index("## 密集奖励第一批"))
        self.assertLess(md2.index("## 密集奖励第一批"), md2.index("| K1 |"))
        self.assertIn("D-keep（k）挑出 k_050", md2)
        self.assertIn("k（k_050）对 s_050（s_050）", md2)
        self.assertIn("| K1 | k | 在跑 |", md2)
        self.assertIn("| G1 | g | **进程不在** |", md2)
        self.assertIn("10-21 19:20（plan: renderer_repack）", md2)
        self.assertIn("守护进程最近一次检查在 20 分钟前", md2)
        self.assertIn("可能已停", md2)
        self.assertIn("| 10-21 19:25 | board_restart | board loop gone, restarted |", md2)
        self.assertNotIn("## 训练链健康", md)  # without the keys: as before
        self.assertEqual(B.main(["--manifest", str(self.root / "m.json")]), 0)
        self.assertIn("测试批次", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
