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
        self.assertIsNone(T.keep_training([(0, 0.1)]))
        self.assertTrue(T.keep_training([(0, 0.35), (25, 0.40), (150, 0.31)]))   # 9 points below the best: noise
        self.assertFalse(T.keep_training([(0, 0.35), (25, 0.40), (150, 0.29)]))  # more than 10 below: stop
        self.assertTrue(T.keep_training([(0, 0.1), (25, 0.2), (50, 0.3)]))


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
            self.assertEqual(calls, ["sft", "rl_config", "rl", "rl_extend", "evaluate"])
            self.assertEqual(json.loads((Path(tmp) / "progress.json").read_text())["state"], "done")

    def test_progress_without_the_new_steps(self):
        with tempfile.TemporaryDirectory() as tmp:  # a progress.json written by the six-step chain
            steps = {s: {"state": "done"} for s in T.STEPS[:6]}
            (Path(tmp) / "progress.json").write_text(json.dumps({"plan": {}, "steps": steps}))
            chain = T.Chain({"tr": tmp, "sft_dir": tmp, "endpoint": 150}, sleep=lambda s: None)
            self.assertEqual(chain.run(), 0)  # neither extension nor evaluation planned
            state = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(state["steps"]["evaluate"], dict(state["steps"]["evaluate"], state="done",
                                                              note="not planned"))

    def _chain(self, tmp, **plan):
        chain = T.Chain(dict({"tr": tmp, "sft_dir": tmp, "endpoint": 150}, **plan), sleep=lambda s: None)
        logs = Path(tmp) / "rl_run" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        return chain, logs

    def test_rl_extend(self):
        steady = "".join(f"step:{s} - val-aux/active_spatial/traj_success/mean@4:{v}\n"
                         for s, v in ((0, 0.30), (100, 0.34), (125, 0.38), (150, 0.42)))
        dropped = steady.replace("0.42", "0.27")  # 11 points below the best earlier value (0.38)
        for log, want in ((steady, 250), (dropped, None)):
            with tempfile.TemporaryDirectory() as tmp:
                chain, logs = self._chain(tmp, extend_to=250)
                (logs / "rl-1.log").write_text(log)
                ppo = []
                chain._ppo = lambda *a, **k: ppo.append((a, k))
                chain.rl_extend()
                self.assertEqual(chain.state["steps"]["rl_extend"]["decision"]["extend_to"], want)
                self.assertEqual(ppo, [(("rl_extend", 250), {"suffix": "_ext"})] if want else [])
                if want:  # resumed after the extension started: the decision stands though validation moved on
                    (logs / "rl-2.log").write_text("step:175 - val-aux/active_spatial/traj_success/mean@4:0.1\n")
                    resumed = T.Chain(chain.plan, sleep=lambda s: None)
                    resumed._ppo = lambda *a, **k: ppo.append((a, k))
                    resumed.rl_extend()
                    self.assertEqual(len(ppo), 2)
                    self.assertEqual(resumed.final_update(), 250)
                    self.assertIn("续训到 250 次", (Path(tmp) / "PROGRESS.md").read_text(encoding="utf-8"))
                else:
                    self.assertEqual(chain.final_update(), 150)
        with tempfile.TemporaryDirectory() as tmp:
            chain, _ = self._chain(tmp)
            chain.rl_extend()
            self.assertEqual(chain.state["steps"]["rl_extend"]["note"], "not planned")

    def test_evaluate(self):
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp) / "eval"
            extra = Path(tmp) / "base"
            chain, _ = self._chain(tmp, eval={"dir": str(ev), "every": 50,
                                              "models": {"base": {"model": str(extra)},
                                                         "r1": {"model": str(extra), "prompt_format": "free_think"}}})
            for m in [extra] + [Path(tmp) / "rl_run" / "checkpoints" / f"global_step_{s}" / "actor" / "huggingface"
                                for s in (50, 100, 150)]:
                m.mkdir(parents=True)
                (m / "config.json").write_text("{}")
            (ev / "v10_rl_050").mkdir(parents=True)
            (ev / "v10_rl_050" / "done.json").write_text("{}")  # evaluated before: no new job
            prepared, deleted, fail = {}, [], set()

            def prepare(run, gpus, args, job_dir="job"):
                prepared[run.name] = (gpus, args)
                (run / job_dir).mkdir(parents=True, exist_ok=True)

            def submit(run, job_dir="job"):
                (run / job_dir / "submission.json").write_text(json.dumps({"job_id": "pt-" + run.name}))

            def status(run, job_dir="job"):
                if run.name not in fail:
                    (ev / run.name).mkdir(exist_ok=True)
                    (ev / run.name / "done.json").write_text("{}")
                return {"state": "FAILED" if run.name in fail else "SUCCEEDED", "finished": True}

            def delete(run, job_dir="job"):
                deleted.append(run.name)
                receipt = run / job_dir / "submission.json"
                receipt.replace(receipt.with_name(f"submission_{len(deleted)}_deleted.json"))

            tables = []
            chain.sh = lambda args, **k: tables.append(args)
            with mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete):
                chain.evaluate()
                self.assertEqual(sorted(prepared), ["base", "r1", "v10_rl_100", "v10_rl_150"])
                self.assertEqual(prepared["r1"], (1, ["eval", str(ev), "r1", str(extra), "free_think"]))
                self.assertEqual(prepared["v10_rl_150"][1][3],
                                 str(Path(tmp) / "rl_run" / "checkpoints" / "global_step_150" / "actor" / "huggingface"))
                self.assertEqual(sorted(deleted), sorted(prepared))  # every job deleted after it ended
                self.assertEqual(tables[0][-2:], ["--eval", str(ev)])

                (ev / "base" / "done.json").unlink()  # a failed job: the step fails, its job is still deleted
                fail.add("base")
                prepared.clear()
                deleted.clear()
                with self.assertRaises(T.ChainError):
                    chain.evaluate()
                self.assertEqual((list(prepared), deleted), (["base"], ["base"]))

                chain.plan["eval"]["models"]["missing"] = {"model": str(Path(tmp) / "nowhere")}
                with self.assertRaises(T.ChainError):
                    chain.evaluate()

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
