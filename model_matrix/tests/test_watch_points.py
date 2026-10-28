"""The intermediate downstream points (model_matrix/downstream/watch_points.py, the user's 3.3 of 2026-10-26)."""
import json
import tempfile
import unittest
from pathlib import Path


class WatchPointsTest(unittest.TestCase):
    def test_copy_evaluate_retry_and_missed(self):
        from model_matrix.downstream.watch_points import Watcher

        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "DS-X-100-s1"
            (exp / "run" / "full_state").mkdir(parents=True)
            (exp / "experiment.json").write_text(json.dumps({"start": {"path": "/start"}}))
            calls = {"extract": [], "submit": [], "delete": []}
            states = {}

            def state(n):
                d = exp / "run" / "full_state" / f"steps_{n}"
                d.mkdir()
                (d / "downstream_state.json").write_text("{}")

            def extract(st, run, dest, log):
                calls["extract"].append(st.name)
                (dest / "checkpoints").mkdir(parents=True)
                (dest / "source.json").write_text("{}")
                return 0

            def submit(run, name, cmd):
                calls["submit"].append((run.name, name, cmd))
                (run / "job").mkdir(parents=True, exist_ok=True)
                (run / "job" / "submission.json").write_text("{}")
                return f"pt-{len(calls['submit'])}"

            def delete(run):
                calls["delete"].append(run.name)
                (run / "job" / "submission.json").rename(run / "job" / f"deleted_{len(calls['delete'])}.json")

            w = Watcher([exp], (150000, 175000), extract=extract, submit=submit,
                        status=lambda run: states.get(run.name, {"state": "RUNNING"}), delete=delete, log=lambda m: None)
            self.assertFalse(w.poll())  # nothing saved yet
            self.assertEqual(calls["extract"], [])
            state(150000)
            self.assertFalse(w.poll())
            self.assertEqual(calls["extract"], ["steps_150000"])
            run, name, cmd = calls["submit"][0]
            self.assertEqual((run, name), ("steps_150000", "ds-eval-ds-x-100-s1-150k"))
            self.assertEqual(cmd[2:], ["/start", str(exp / "evals_src" / "steps_150000" / "checkpoints" /
                                                    "steps_150000_pytorch_model.pt"),
                                       str(exp / "evals" / "steps_150000" / "out"), "8", "96"])
            self.assertTrue(cmd[1].endswith("eval_ckpt.sh"))
            states["steps_150000"] = {"state": "FAILED", "finished": True}
            w.poll()  # ended without a result: deleted, then submitted again
            w.poll()
            self.assertEqual(len(calls["submit"]), 2)
            states.pop("steps_150000")
            out = exp / "evals" / "steps_150000" / "out" / "episodes"
            out.mkdir(parents=True)
            (out / "summary.json").write_text(json.dumps({"per_task": {}}))
            # the 175k state was missed: the run is already past it
            import shutil

            shutil.rmtree(exp / "run" / "full_state" / "steps_150000")
            state(200000)
            self.assertTrue(w.poll())
            rec = json.loads((exp / "evals_src" / "watch.json").read_text())
            self.assertTrue(rec["150000"]["evaluated"])
            self.assertIn("steps_200000", rec["175000"]["missed"])
            self.assertEqual(calls["extract"], ["steps_150000"])

    def test_fixed_pool_waits_for_the_pause_file(self):
        """The user's decision 28 (2026-10-28): --pool h800 sends every evaluation there, once the pause file is gone;
        the default "queue" leaves the choice (and the pause wait) to sco_job as before."""
        from unittest import mock

        from model_matrix import sco_job, sco_view
        from model_matrix.downstream.watch_points import Watcher

        with tempfile.TemporaryDirectory() as tmp:
            pause = Path(tmp) / "pause"
            pause.write_text("")
            slept, sent = [], []
            with mock.patch.object(sco_view, "PAUSE", pause), \
                    mock.patch.object(sco_job, "submit", side_effect=lambda run, cards, name, cmd, pool=None:
                                      sent.append((cards, name, pool)) or "pt-1"):
                gone = lambda s: (slept.append(s), pause.unlink())
                self.assertEqual(Watcher._submit(Path(tmp), "n", ["true"], "h800", sleep=gone), "pt-1")
                self.assertEqual((slept, sent), ([sco_job.POLL_S], [(8, "n", "h800")]))
                pause.write_text("")
                Watcher._submit(Path(tmp), "q", ["true"], "queue", sleep=gone)  # no wait here: sco_job's own
                self.assertEqual((len(slept), sent[-1]), (1, (8, "q", "queue")))


if __name__ == "__main__":
    unittest.main()
