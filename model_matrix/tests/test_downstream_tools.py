"""The model line's job helper and StarVLA backbone tools (step 2 of 2026-10-24)."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
DS = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream")
SMOKE_DATA = Path("/mnt/umm/users/yinbaiqiao/downstream_runs/data/step2_smoke")


class ScoJobTest(unittest.TestCase):
    def test_submit_waits_then_names_the_job(self):
        from model_matrix import sco_job, sco_view

        busy = {"pool": "h800", "machines": 30, "busy_machines": 30, "idle_estimate": 0, "machines_with_free_cards": 0,
                "waiting": 0, "waiting_main_line": 0, "waiting_main_line_1card": 0, "waiting_downstream": 0,
                "waiting_ours": 0}
        free = dict(busy, machines_with_free_cards=2, waiting_main_line=3)  # main-line trainers wait: 1 card is fine
        polls = iter([{"zoetrope": busy, "h800": busy}, {"zoetrope": busy, "h800": free}])
        current = {}
        slept = []

        def view(pool):
            if pool == "zoetrope":
                current.update(next(polls))
            return dict(current[pool], pool=pool)

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "load"
            done = mock.Mock(returncode=0, stdout="job pt-xyz789 submitted", stderr="")
            with mock.patch.object(sco_view, "view", side_effect=view), \
                    mock.patch.object(sco_view, "PAUSE", Path(tmp) / "none"), \
                    mock.patch.object(sco_job.time, "sleep", side_effect=slept.append), \
                    mock.patch.object(sco_job.job_sco.subprocess, "run", return_value=done) as r:
                self.assertEqual(sco_job.submit(run, 1, "backbone-load-q3b", ["bash", "/s.sh", "a b"]), "pt-xyz789")
                self.assertEqual(slept, [sco_job.POLL_S])
                cmd = r.call_args[0][0]
                self.assertIn("--job-name=models-backbone-load-q3b", cmd)
                self.assertIn("--aec2-name=h800", cmd)
                self.assertIn("--worker-spec=N4lS.Iq.I80.1", cmd)
                self.assertEqual((run / "job" / "launch.sh").read_text(), "#!/usr/bin/env bash\nexec bash /s.sh 'a b'\n")
                self.assertEqual(sco_job.submit(run, 1, "again", ["true"]), "pt-xyz789")  # the receipt stands
                self.assertEqual(r.call_count, 1)
            # --pool queue: no idle wait, the pool with the fewest waiting jobs once the pause file is gone
            run2 = Path(tmp) / "smoke"
            pause = Path(tmp) / "pause"
            pause.write_text("")
            gone = lambda s: pause.unlink()
            done2 = mock.Mock(returncode=0, stdout="job pt-q1 submitted", stderr="")
            with mock.patch.object(sco_view, "PAUSE", pause), \
                    mock.patch.object(sco_job.time, "sleep", side_effect=gone), \
                    mock.patch.object(sco_job.job_sco, "pick_pool", return_value="zoetrope"), \
                    mock.patch.object(sco_view, "view", side_effect=AssertionError("no idle check")), \
                    mock.patch.object(sco_job.job_sco.subprocess, "run", return_value=done2) as r2:
                self.assertEqual(sco_job.submit(run2, 8, "backbone-smoke-q3b", ["true"], pool="queue"), "pt-q1")
                self.assertIn("--aec2-name=zoetrope", r2.call_args[0][0])
                self.assertIn("--worker-spec=N4lS.Iq.I80.8", r2.call_args[0][0])
            self.assertFalse(pause.exists())
            # 8 cards wait for the main line and an idle machine
            self.assertFalse(sco_view.fits(dict(free, idle_estimate=2), 8))
            self.assertTrue(sco_view.fits(dict(free, idle_estimate=2, waiting_main_line=0), 8))
            self.assertFalse(sco_view.fits(dict(free, waiting_main_line_1card=1), 1))


class DownstreamToolsTest(unittest.TestCase):
    def test_scripts_parse(self):
        for f in ("backbone_load.sh", "backbone_smoke.sh"):
            self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / "model_matrix" / "downstream" / f)]).returncode, 0)

    def test_starts(self):
        starts = json.loads((ROOT / "model_matrix" / "downstream" / "starts.json").read_text())
        self.assertEqual({k: v["model_type"] for k, v in starts.items() if not k.startswith("_")},
                         {"Q3B": "qwen3_vl", "CSB": "cambrian_qwen"})

    @unittest.skipUnless((DS / "downstream" / "train" / "make_experiment.py").is_file() and SMOKE_DATA.is_dir(),
                         "needs the downstream worktree and its step-2 data")
    def test_config_is_the_downstream_recipe_with_our_start(self):
        import yaml

        from model_matrix.downstream import make_config as M

        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "DS-Q3B-smoke"
            rec = M.build("Q3B", SMOKE_DATA, exp, steps=100, save_interval=100, log_every=10, seed=1)
            cfg = yaml.safe_load((exp / "config.yaml").read_text())
            recipe = yaml.safe_load(M.RECIPE.read_text())
            self.assertEqual(cfg["framework"]["qwenvl"]["base_vlm"], rec["start"]["path"])
            self.assertEqual(cfg["trainer"]["max_train_steps"], 100)
            self.assertEqual(cfg["framework"]["action_model"], recipe["framework"]["action_model"])  # the recipe's head
            self.assertEqual(cfg["datasets"]["vla_data"]["per_device_batch_size"],
                             recipe["datasets"]["vla_data"]["per_device_batch_size"])
            with self.assertRaises(FileExistsError):
                M.build("Q3B", SMOKE_DATA, exp, steps=100, save_interval=100, log_every=10, seed=1)


if __name__ == "__main__":
    unittest.main()
