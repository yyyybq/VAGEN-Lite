"""train_chain.py: the unattended v10 training chain (progress readers, stop on failure, resume)."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import train_chain as T

RL_LOG = """(TaskRunner pid=1) step:0 - val-aux/active_spatial/traj_success/mean@4:np.float64(0.03125) - x:1
(TaskRunner pid=1) step:1 - episode/success_rate:0.125 - timing_s/step:153.2
(TaskRunner pid=1) step:2 - episode/success_rate:0.25 - timing_s/step:89.0 - val-aux/active_spatial/traj_success/mean@4:np.float64(0.0625)
"""


class ProgressTest(unittest.TestCase):
    def test_readers(self):
        rl = T.rl_progress(RL_LOG)
        self.assertEqual(rl["last_update"], 2)
        self.assertEqual(rl["train_success_last"], 0.25)
        self.assertAlmostEqual(rl["train_success_mean_last10"], 0.1875)
        self.assertEqual(rl["val_success"], [(0, 0.03125), (2, 0.0625)])
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "trainer_log.jsonl").write_text(
                json.dumps({"current_steps": 5, "total_steps": 90, "loss": 0.8, "epoch": 0.2}) + "\n"
                + json.dumps({"current_steps": 10, "total_steps": 90, "eval_loss": 0.5}) + "\n")
            self.assertEqual(T.sft_progress(d), {"current_steps": 10, "total_steps": 90, "loss": 0.8, "epoch": 0.2,
                                                 "eval_loss": 0.5, "eval_at_step": 10})
        self.assertIsNone(T.still_rising([(0, 0.1), (25, 0.2)]))
        self.assertTrue(T.still_rising([(0, 0.1), (25, 0.2), (50, 0.3)]))
        self.assertFalse(T.still_rising([(0, 0.1), (25, 0.3), (50, 0.31)]))
        self.assertFalse(T.still_rising([(0, 0.4), (25, 0.2), (50, 0.3)]))


class ChainTest(unittest.TestCase):
    def test_stops_on_failure_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"tr": tmp, "sft_dir": tmp, "commit": "x", "endpoint": 150}
            calls = []

            def make(fail_at=None):
                chain = T.Chain(plan, sleep=lambda s: None)
                for step in T.STEPS:
                    def fn(step=step):
                        calls.append(step)
                        if step == fail_at:
                            raise T.ChainError("job sft ended FAILED")
                    setattr(chain, step, fn)
                return chain

            self.assertEqual(make(fail_at="sft").run(), 1)
            self.assertEqual(calls, ["check_code", "render", "sft_data", "sft"])  # nothing after the failure
            state = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["steps"]["sft"]["state"], "failed")
            self.assertIn("ended FAILED", state["steps"]["sft"]["error"])
            self.assertEqual(state["steps"]["rl"]["state"], "pending")
            self.assertIn("失败", (Path(tmp) / "PROGRESS.md").read_text())
            calls.clear()
            self.assertEqual(make().run(), 0)  # a new run continues at the failed step
            self.assertEqual(calls, ["sft", "rl_config", "rl"])
            self.assertEqual(json.loads((Path(tmp) / "progress.json").read_text())["state"], "done")

    def test_wait_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain = T.Chain({"tr": tmp, "sft_dir": tmp}, sleep=lambda s: None)
            seq = iter([{"state": "RUNNING", "finished": False}, {"state": "SUCCEEDED", "finished": True}])
            done = []
            chain.wait_jobs("sft", {"sft": lambda: next(seq)}, "sft", on_finish=lambda: done.append(1))
            self.assertEqual(done, [1])
            with self.assertRaises(T.ChainError):  # the render service dies while PPO runs
                chain.wait_jobs("rl", {"rl": lambda: {"state": "RUNNING", "finished": False},
                                       "renderer": lambda: {"state": "FAILED", "finished": True}}, "rl")


if __name__ == "__main__":
    unittest.main()
