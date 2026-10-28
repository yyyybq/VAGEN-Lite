"""The intermediate downstream points (model_matrix/downstream/watch_points.py, the user's 3.3 of 2026-10-26)."""
import json
import os
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

    def test_copy_as_a_job(self):
        """The user's login-node rule (2026-10-28), --extract-on sco: the copy is a 1-card job in the pool (its
        command the downstream line's extract_weights.py, then the copy given back to this user); the point waits
        while the job is queued or runs, its evaluation follows a complete copy; a failed copy is tried once more,
        then the point fails; a point whose state is replaced first is missed and its copy job deleted."""
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco
        from model_matrix import sco_job, sco_view
        from model_matrix.downstream import watch_points as W

        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "DS-X-100-s1"
            (exp / "run" / "full_state" / "steps_150000").mkdir(parents=True)
            (exp / "run" / "full_state" / "steps_150000" / "downstream_state.json").write_text("{}")
            (exp / "experiment.json").write_text(json.dumps({"start": {"path": "/start"}}))
            copies, evals, deleted, state = [], [], [], {"s": "QUEUEING"}

            def submit(run, cards, name, cmd, pool=None):
                copies.append((run, cards, name, cmd, pool))
                (run / "job").mkdir(parents=True, exist_ok=True)
                (run / "job" / "submission.json").write_text(json.dumps({"job_id": f"pt-c{len(copies)}"}))
                return f"pt-c{len(copies)}"

            def delete(run, job_dir="job"):
                deleted.append(run.name)
                (run / job_dir / "submission.json").rename(run / job_dir / f"submission_{len(deleted)}_deleted.json")

            def eval_submit(run, name, cmd):
                evals.append(name)
                (run / "job").mkdir(parents=True, exist_ok=True)
                (run / "job" / "submission.json").write_text("{}")
                return "pt-e1"

            with mock.patch.object(sco_job, "submit", side_effect=submit), \
                    mock.patch.object(sco_view, "PAUSE", Path(tmp) / "no-pause"), \
                    mock.patch.object(job_sco, "status", side_effect=lambda run, job_dir="job": {"state": state["s"]}), \
                    mock.patch.object(job_sco, "delete", side_effect=delete):
                w = W.Watcher([exp], (150000, 175000), submit=eval_submit, status=lambda run: {"state": "RUNNING"},
                              delete=lambda run: None, log=lambda m: None, pool="h800", extract_on="sco")
                self.assertFalse(w.poll())  # the copy job submitted
                run, cards, name, cmd, pool = copies[0]
                self.assertEqual((run.name, cards, name, pool), ("extract_150000", 1, "ds-copy-ds-x-100-s1-150k", "h800"))
                self.assertEqual(cmd[:2], ["bash", "-c"])
                self.assertIn(str(W.EXTRACT), cmd[2])
                self.assertIn(f"chown -R {os.getuid()}:{os.getgid()}", cmd[2])
                self.assertFalse(w.poll())  # still queued: nothing new
                self.assertEqual((len(copies), evals), (1, []))
                self.assertTrue(json.loads((exp / "evals_src" / "watch.json").read_text())["150000"]["copy_pending"])
                state["s"] = "FAILED"  # ended without a copy: tried once more
                w.poll()
                self.assertEqual((deleted, len(copies)), (["extract_150000"], 1))
                w.poll()
                self.assertEqual(len(copies), 2)
                state["s"] = "SUCCEEDED"
                dest = exp / "evals_src" / "steps_150000"
                (dest / "checkpoints").mkdir(parents=True)
                (dest / "source.json").write_text("{}")
                w.poll()  # the copy is there: the job deleted, the evaluation submitted
                self.assertEqual(evals, ["ds-eval-ds-x-100-s1-150k"])
                rec = json.loads((exp / "evals_src" / "watch.json").read_text())["150000"]
                self.assertEqual([e["rc"] for e in rec["extract"]], [1, 0])
                self.assertNotIn("copy_pending", rec)
                # 175k: two failed copies fail the point
                (exp / "run" / "full_state" / "steps_175000").mkdir()
                (exp / "run" / "full_state" / "steps_175000" / "downstream_state.json").write_text("{}")
                state["s"] = "FAILED"
                for _ in range(4):
                    w.poll()
                rec = json.loads((exp / "evals_src" / "watch.json").read_text())["175000"]
                self.assertIn("failed 2 times", rec["failed"])
            # missed: the state replaced while the copy job waits
            with tempfile.TemporaryDirectory() as tmp:
                exp = Path(tmp) / "DS-Y-100-s1"
                (exp / "run" / "full_state" / "steps_150000").mkdir(parents=True)
                (exp / "run" / "full_state" / "steps_150000" / "downstream_state.json").write_text("{}")
                (exp / "experiment.json").write_text(json.dumps({"start": {"path": "/start"}}))
                copies.clear(), deleted.clear()
                state["s"] = "QUEUEING"
                with mock.patch.object(sco_job, "submit", side_effect=submit), \
                        mock.patch.object(sco_view, "PAUSE", Path(tmp) / "no-pause"), \
                        mock.patch.object(job_sco, "status", side_effect=lambda run, job_dir="job": {"state": state["s"]}), \
                        mock.patch.object(job_sco, "delete", side_effect=delete):
                    w = W.Watcher([exp], (150000,), submit=eval_submit, status=lambda run: {"state": "RUNNING"},
                                  delete=lambda run: None, log=lambda m: None, pool="h800", extract_on="sco")
                    w.poll()
                    import shutil

                    shutil.rmtree(exp / "run" / "full_state" / "steps_150000")
                    (exp / "run" / "full_state" / "steps_175000").mkdir()
                    (exp / "run" / "full_state" / "steps_175000" / "downstream_state.json").write_text("{}")
                    self.assertTrue(w.poll())
                    rec = json.loads((exp / "evals_src" / "watch.json").read_text())["150000"]
                    self.assertIn("missed", rec)
                    self.assertEqual(deleted, ["extract_150000"])

    def test_local_copy_keeps_to_the_core_budget(self):
        """The user's login-node rule (2026-10-28, corrected): the copy here runs at nice 19 with 2 threads per
        numeric library, and waits (None, tried at the next poll) while the cores in use plus 2 exceed 12."""
        from unittest import mock

        from model_matrix.downstream import watch_points as W

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "x.log"
            with mock.patch.object(W.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
                self.assertIsNone(W.Watcher._extract(Path("/s"), Path("/r"), Path("/d"), log, in_use=lambda: 10.5))
                run.assert_not_called()
                self.assertEqual(W.Watcher._extract(Path("/s"), Path("/r"), Path("/d"), log, in_use=lambda: 3.0), 0)
                cmd, kw = run.call_args[0][0], run.call_args[1]
                self.assertEqual(cmd[:3], ["nice", "-n", "19"])
                self.assertEqual(cmd[3:], [W.STARVLA_PY, str(W.EXTRACT), "/s", "/r", "/d"])
                for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    self.assertEqual(kw["env"][v], "2")
                self.assertEqual(kw["env"]["TOKENIZERS_PARALLELISM"], "false")
            stat = Path(tmp) / "cpu.stat"
            reads = iter(["usage_usec 1000000\n", "usage_usec 21000000\n"])
            with mock.patch.object(W, "CPU_STAT", stat):
                def advance(s):
                    stat.write_text(next(reads))
                stat.write_text(next(reads))
                self.assertAlmostEqual(W.cores_in_use(5.0, sleep=advance), 4.0)  # 20 s of CPU in 5 s

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
