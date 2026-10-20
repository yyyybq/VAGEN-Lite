"""job_sco.py: one sco job running a mode of sco_interiorgs_v10_train.sh."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import job_sco as J


class JobScoTest(unittest.TestCase):
    def test_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            launch = J.prepare(run, 1, ["sft-trial", "/lf dir", "30"])
            text = launch.read_text()
            self.assertIn(f"{J.WORKTREE}/{J.SCRIPT} sft-trial {run} '/lf dir' 30", text)
            prepared = json.loads((run / "job" / "prepared.json").read_text())
            self.assertEqual(prepared["spec"], "N4lS.Iq.I80.1")
            self.assertEqual(J.SPECS[8], "N4lS.Iq.I80.8")
            with self.assertRaises(ValueError):
                J.prepare(run, 2, ["sft"])
            with self.assertRaises(ValueError):
                J.prepare(run, 1, [])
            # the render service and the trainer of one PPO run: two job folders on one RUN
            J.prepare(run, 1, ["renderer"], job_dir="job_renderer")
            J.prepare(run, 8, ["rl", "/rl", "150"], job_dir="job_rl")
            self.assertIn(f"rl {run} /rl 150", (run / "job_rl" / "launch.sh").read_text())
            self.assertEqual(json.loads((run / "job_rl" / "prepared.json").read_text())["spec"], "N4lS.Iq.I80.8")
            self.assertTrue((run / "job" / "launch.sh").exists())

    def test_pool_per_job(self):
        """Round 11: zoetrope by default, h800 when asked (the user's decision when zoetrope is full); the pool
        recorded at prepare is the one the submission names."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            J.prepare(run, 1, ["eval", "/e", "n", "/m", "-"])
            J.prepare(run, 8, ["rl", "/rl", "300"], job_dir="job_rl", pool="h800")
            self.assertEqual(json.loads((run / "job" / "prepared.json").read_text())["pool"], "zoetrope")
            self.assertEqual(json.loads((run / "job_rl" / "prepared.json").read_text())["pool"], "h800")
            with self.assertRaises(ValueError):
                J.prepare(run, 1, ["eval"], pool="nowhere")
            calls = []
            done = mock.Mock(returncode=0, stdout="created pt-abc123", stderr="")
            with mock.patch.object(J.subprocess, "run", lambda cmd, **k: calls.append(cmd) or done):
                self.assertEqual(J.submit(run), "pt-abc123")
                self.assertEqual(J.submit(run, "job_rl"), "pt-abc123")
            self.assertIn("--aec2-name=zoetrope", calls[0])
            self.assertIn("--aec2-name=h800", calls[1])
            self.assertIn("--worker-spec=N4lS.Iq.I80.8", calls[1])

    def test_pick_the_pool_with_fewer_waiting_jobs(self):
        """Round 12 batch: the pool with the fewest jobs waiting for cards (every user's jobs in the workspace)."""
        from unittest import mock

        jobs = [{"state": "STARTING", "resource_pool": {"name": "h800"}},
                {"state": "STARTING", "resource_pool": {"name": "h800"}},
                {"state": "RUNNING", "resource_pool": {"name": "zoetrope"}},
                {"state": "STARTING", "resource_pool": {"name": "zoetrope"}},
                {"state": "SUCCEEDED", "resource_pool": {"name": "zoetrope"}}]
        self.assertEqual(J.waiting_by_pool(jobs), {"zoetrope": 1, "h800": 2})

        class Res:
            def __init__(self, out, rc=0):
                self.stdout, self.returncode, self.stderr = out, rc, ""

        with mock.patch.object(J.subprocess, "run", return_value=Res(json.dumps(jobs))):
            self.assertEqual(J.pick_pool(), "zoetrope")
        with mock.patch.object(J.subprocess, "run", return_value=Res(json.dumps(jobs[:1]))):
            self.assertEqual(J.pick_pool(), "zoetrope")  # h800 has one waiting, zoetrope none
        with mock.patch.object(J.subprocess, "run", return_value=Res("", rc=1)):
            self.assertEqual(J.pick_pool(), J.POOLS[0])  # no listing: the first pool
        with mock.patch.object(J.subprocess, "run", return_value=Res("[]")):
            self.assertEqual(J.pick_pool(("h800", "zoetrope")), "h800")  # a tie: the first of the given order

    def test_jobs_run_from_the_cluster_checkout(self):
        """Round 6: the cluster checkout is the only code directory (no more VAGEN-Lite-r4)."""
        self.assertEqual(J.WORKTREE, "/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
        self.assertIn("\nWORKTREE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite\n", script)
        self.assertNotIn("VAGEN-Lite-r4", script)

    def test_gsplat_build_outside_the_checkout(self):
        """Round 11: the validated gsplat build every render service copies lived in an experiment folder of the
        checkout, which the directory clean-up moved into archive/ (every RL and evaluation job would have
        failed at start); it now lives outside the checkout."""
        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
        line = next(l for l in script.splitlines() if l.startswith("VALIDATED_GSPLAT_CACHE="))
        self.assertEqual(line, "VALIDATED_GSPLAT_CACHE=/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007")
        self.assertFalse(line.split("=", 1)[1].startswith(J.WORKTREE + "/"))

    def test_outputs_outside_run_are_handed_back(self):
        """Round 8: the SFT model (LF/model) and evaluation results (EVAL/NAME) were left root-only (mode 600)."""
        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
        self.assertIn('chown -R "${OWNER}" "${RUN}" ${ALSO_OWNED:+"${ALSO_OWNED}"}', script)
        for mode, line in (("sft)", "ALSO_OWNED=${LF}"), ("eval)", "ALSO_OWNED=${EVAL}/${NAME}")):
            block = script.split(f"\n  {mode}\n", 1)[1].split(";;", 1)[0]
            self.assertIn(line, block)

    def test_eval_mode_passes_options(self):
        """Round 9: the switches after MODEL and PROMPT_FORMAT ("-" for the default) reach evaluation.py run."""
        import subprocess
        import tempfile

        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
        block = script.split("\n  eval)\n", 1)[1].split(";;", 1)[0]
        # the eval block as it is, without the GPU and the render service, and with the evaluation echoed
        lines = [l for l in block.splitlines() if "CUDA_VISIBLE_DEVICES" not in l and "start_renderer" not in l]
        body = "\n".join(lines).replace('"${PY}" -m data_gen.interiorgs_v10.evaluation run --eval', "echo run --eval")
        self.assertIn("echo run --eval", body)
        with tempfile.TemporaryDirectory() as tmp:
            sh = Path(tmp) / "eval_args.sh"
            sh.write_text("set -euo pipefail\n" + body + "\n")
            out = lambda *a: subprocess.run(["bash", str(sh), *a], capture_output=True, text=True, check=True).stdout.split()  # noqa: E731
            base = ["run", "--eval", "E", "--name", "N", "--model", "M"]
            self.assertEqual(out("eval", "RUN", "E", "N", "M"), base)
            self.assertEqual(out("eval", "RUN", "E", "N", "M", "free_think"), base + ["--prompt-format", "free_think"])
            self.assertEqual(out("eval", "RUN", "E", "N", "M", "-", "--check-each-action", "--max-actions", "20"),
                             base + ["--check-each-action", "--max-actions", "20"])


if __name__ == "__main__":
    unittest.main()
