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
        for f in ("backbone_load.sh", "backbone_smoke.sh", "run_full.sh", "eval_ckpt.sh"):
            self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / "model_matrix" / "downstream" / f)]).returncode, 0)

    def test_starts(self):
        starts = json.loads((ROOT / "model_matrix" / "downstream" / "starts.json").read_text())
        self.assertEqual({k: v["model_type"] for k, v in starts.items() if not k.startswith("_")},
                         {"Q3B": "qwen3_vl", "CSB": "cambrian_qwen", "Q3S": "qwen3_vl", "Q3RD": "qwen3_vl",
                          "CSS": "cambrian_qwen"})

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
            self.assertNotIn("segmented", rec)  # a smoke folder as before

    @unittest.skipUnless((DS / "downstream" / "train" / "make_experiment.py").is_file() and SMOKE_DATA.is_dir(),
                         "needs the downstream worktree and its step-2 data")
    def test_full_run_has_the_downstream_stage1_fields(self):
        """DS-B-100-s1's form (make_experiment --full-state-interval 25000 --eval-points 200000), the start locked,
        --qwenvl keys in the config; full_state.py (the downstream line's) reads the record."""
        import yaml

        from model_matrix.downstream import make_config as M

        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "DS-CSB-100-s1"
            rec = M.build("CSB", SMOKE_DATA, exp, steps=200000, save_interval=25000, log_every=100, seed=1,
                          full_state_interval=25000, eval_points=[200000], qwenvl={"cambrian_checkpointing": "vision"})
            cfg = yaml.safe_load((exp / "config.yaml").read_text())
            self.assertEqual((cfg["trainer"]["full_state_interval"], cfg["trainer"]["eval_points"],
                              cfg["trainer"]["save_interval"], cfg["trainer"]["logging_frequency"]),
                             (25000, [], 25000, 100))  # as DS-B-100-s1's config
            self.assertEqual(cfg["framework"]["qwenvl"]["cambrian_checkpointing"], "vision")
            for key, value in (("segmented", True), ("full_state_interval", 25000), ("eval_points", [200000]),
                               ("repeat_final", False), ("injob_eval", {"servers": 8, "workers": 96}),
                               ("global_batch", 64), ("data", {"dir": str(SMOKE_DATA)})):
                self.assertEqual(rec[key], value)
            self.assertEqual(M.check_start(exp), [])
            fs = subprocess.run(["python3", str(DS / "downstream" / "train" / "full_state.py"), "points", str(exp)],
                                capture_output=True, text=True)
            self.assertEqual(fs.stdout.split(), ["200000"])
            with self.assertRaises(SystemExit):  # make_experiment's rule: the interval divides the points
                M.build("CSB", SMOKE_DATA, Path(tmp) / "bad", steps=200000, save_interval=30000, log_every=100,
                        seed=1, full_state_interval=25000, eval_points=[200000])

    @unittest.skipUnless((DS / "downstream" / "verify_starts.py").is_file(), "needs the downstream worktree")
    def test_check_start_sees_a_changed_shard(self):
        from model_matrix.downstream import make_config as M

        with tempfile.TemporaryDirectory() as tmp:
            start = Path(tmp) / "start"
            start.mkdir()
            (start / "config.json").write_text("{}")
            (start / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-1.safetensors"}}))
            (start / "model-1.safetensors").write_bytes(b"x" * 100)
            exp = Path(tmp) / "exp"
            exp.mkdir()
            (exp / "experiment.json").write_text(json.dumps({"start": {"path": str(start), "lock": M.lock(str(start))}}))
            self.assertEqual(M.check_start(exp), [])
            (start / "model-1.safetensors").write_bytes(b"y" * 101)
            self.assertTrue(any("size 100 -> 101" in p for p in M.check_start(exp)))
            (exp / "experiment.json").write_text(json.dumps({"start": {"path": str(start)}}))
            self.assertEqual(M.check_start(exp), ["the experiment has no start lock"])
            # an unreadable index (the root-only Qwen3-VL snapshot): what stat sees, the blob names included
            (start / "model-1.safetensors").unlink()
            (start / "blob").write_bytes(b"z" * 7)
            (start / "model-1.safetensors").symlink_to(start / "blob")
            vs = M._verify_starts()
            with mock.patch.object(vs, "fingerprint", side_effect=PermissionError("root only")):
                stat = M.lock(str(start))
            self.assertTrue(stat["stat_only"])
            self.assertEqual((stat["files"]["model-1.safetensors"]["size"],
                              stat["files"]["model-1.safetensors"]["target"]), (7, str(start / "blob")))
            (exp / "experiment.json").write_text(json.dumps({"start": {"path": str(start), "lock": stat}}))
            self.assertEqual(M.check_start(exp), [])
            (start / "blob").write_bytes(b"z" * 8)
            self.assertEqual(len(M.check_start(exp)), 2)  # the link and its blob


if __name__ == "__main__":
    unittest.main()
