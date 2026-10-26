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
            # round 12 batch: the render-service pool is accepted when asked, never picked for other jobs
            J.prepare(run, 1, ["renderer"], job_dir="job_renderer", pool="zoe-cf-render")
            with mock.patch.object(J.subprocess, "run", lambda cmd, **k: calls.append(cmd) or done):
                J.submit(run, "job_renderer")
            self.assertIn("--aec2-name=zoe-cf-render", calls[2])
            self.assertIn("--worker-spec=N4lS.Iq.I80.1", calls[2])
            self.assertNotIn("zoe-cf-render", J.POOLS)
            listing = mock.Mock(returncode=0, stdout=json.dumps([{"state": "STARTING", "resource_pool": {"name": "h800"}}]))
            with mock.patch.object(J.subprocess, "run", return_value=listing):
                self.assertEqual(J.pick_pool(), "zoetrope")

    def test_pick_the_pool_with_fewer_waiting_jobs(self):
        """Round 12 batch: the pool with the fewest jobs waiting for cards (every user's jobs in the workspace)."""
        from unittest import mock

        jobs = [{"state": "STARTING", "resource_pool": {"name": "h800"}},
                {"state": "STARTING", "resource_pool": {"name": "h800"}},
                {"state": "RUNNING", "resource_pool": {"name": "zoetrope"}},
                {"state": "STARTING", "resource_pool": {"name": "zoetrope"}},
                {"state": "SUCCEEDED", "resource_pool": {"name": "zoetrope"}}]
        self.assertEqual(J.waiting_by_pool(jobs), {"zoetrope": 1, "h800": 2})
        # sco's spelling of a queued job (round 12 batch: it was missing, so such jobs counted as not waiting)
        self.assertIn("QUEUEING", J.WAITING)
        self.assertEqual(J.waiting_by_pool(jobs + [{"state": "QUEUEING", "resource_pool": "zoetrope"}] * 2),
                         {"zoetrope": 3, "h800": 2})

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
        """Round 6: the cluster checkout is the code directory (no more VAGEN-Lite-r4).  Round 12 freeze: unless
        VAGEN_LITE_WORKTREE names another copy (the second dense batch, run from a copy the code reorganization does
        not touch): its launch scripts start the job script there and pass the variable on; unset, the launch script
        is as before."""
        import importlib
        import os
        from unittest import mock

        self.assertEqual(J.WORKTREE, "/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
        script = (next(p for p in Path(__file__).resolve().parents if (p / "setup.py").is_file()) / J.SCRIPT).read_text()
        self.assertIn("\nWORKTREE=${VAGEN_LITE_WORKTREE:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}\n", script)
        self.assertNotIn("VAGEN-Lite-r4", script)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            self.assertEqual(J.prepare(run, 1, ["eval", "/e"]).read_text(),
                             f"#!/usr/bin/env bash\nexec bash {J.DEFAULT_WORKTREE}/{J.SCRIPT} eval {run} /e\n")
            try:
                with mock.patch.dict(os.environ, {"VAGEN_LITE_WORKTREE": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-r12dense"}):
                    copy = importlib.reload(J)
                    self.assertEqual(copy.WORKTREE, "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-r12dense")
                    self.assertEqual(copy.prepare(run, 1, ["eval", "/e"]).read_text(),
                                     "#!/usr/bin/env bash\nexec env VAGEN_LITE_WORKTREE=/mnt/umm/users/yinbaiqiao/"
                                     f"VAGEN-Lite-r12dense bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-r12dense/{J.SCRIPT} "
                                     f"eval {run} /e\n")
            finally:
                importlib.reload(J)
        self.assertEqual(J.WORKTREE, J.DEFAULT_WORKTREE)

    def test_gsplat_build_outside_the_checkout(self):
        """Round 11: the validated gsplat build every render service copies lived in an experiment folder of the
        checkout, which the directory clean-up moved into archive/ (every RL and evaluation job would have
        failed at start); it now lives outside the checkout."""
        script = (next(p for p in Path(__file__).resolve().parents if (p / "setup.py").is_file()) / J.SCRIPT).read_text()
        line = next(l for l in script.splitlines() if l.startswith("VALIDATED_GSPLAT_CACHE="))
        self.assertEqual(line, "VALIDATED_GSPLAT_CACHE=/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007")
        self.assertFalse(line.split("=", 1)[1].startswith(J.WORKTREE + "/"))

    def test_outputs_outside_run_are_handed_back(self):
        """Round 8: the SFT model (LF/model) and evaluation results (EVAL/NAME) were left root-only (mode 600)."""
        script = (next(p for p in Path(__file__).resolve().parents if (p / "setup.py").is_file()) / J.SCRIPT).read_text()
        self.assertIn('chown -R "${OWNER}" "${RUN}" ${ALSO_OWNED:+"${ALSO_OWNED}"}', script)
        for mode, line in (("sft)", "ALSO_OWNED=${LF}"), ("eval)", "ALSO_OWNED=${EVAL}/${NAME}")):
            block = script.split(f"\n  {mode}\n", 1)[1].split(";;", 1)[0]
            self.assertIn(line, block)

    def test_eval_mode_passes_options(self):
        """Round 9: the switches after MODEL and PROMPT_FORMAT ("-" for the default) reach evaluation.py run."""
        import subprocess
        import tempfile

        script = (next(p for p in Path(__file__).resolve().parents if (p / "setup.py").is_file()) / J.SCRIPT).read_text()
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

    def test_pool_view(self):
        """Round 12 batch (the chain's renderer_repack): the workspace's running jobs of a pool by machine and the jobs
        waiting there, from every page of the job list (500 per page, newest first: jobs started weeks ago still run
        on later pages) and each running job's workers; nothing when a page or a running job's machine is unreadable."""
        from unittest import mock

        table = ("+----+----+----+----+----+\n| DISPLAY NAME | RESOURCE | HOST IP | POD IP | WORKER STATUS |\n"
                 "| pt-{w}-worker-0 | H800(加速芯片:{c}, vCPU:14核,    | {h} | 10.119.20.27 | Running       |\n"
                 "|                 | 内存:240GiB)                   |              |              |               |\n")
        self.assertEqual(J.worker_hosts(table.format(w="a", c=1, h="10.123.72.36") + table.format(w="b", c=8, h="10.0.0.2")),
                         [(1, "10.123.72.36"), (8, "10.0.0.2")])
        self.assertEqual(J.worker_hosts("| pt-x-worker-0 |          |         |        |               |"), [])
        old = [{"name": f"pt-done{i}", "state": "SUCCEEDED", "resource_pool": {"name": "zoetrope"}} for i in range(498)]
        page1 = old + [{"name": "pt-r1", "display_name": "interiorgs-v10-renderer-rl-run-s", "state": "RUNNING",
                        "resource_pool": {"name": "zoetrope"}},
                       {"name": "pt-t1", "display_name": "interiorgs-v10-rl-rl-run-s", "state": "QUEUEING",
                        "resource_pool": {"name": "zoetrope"}}]
        page2 = [{"name": "pt-old", "display_name": "vln_eval", "state": "RUNNING", "resource_pool": {"name": "zoetrope"}},
                 {"name": "pt-h", "display_name": "elsewhere", "state": "RUNNING", "resource_pool": {"name": "h800"}}]
        where = {"pt-r1": table.format(w="r1", c=1, h="10.0.0.1"), "pt-old": table.format(w="o", c=8, h="10.0.0.9")}
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            if "list" in cmd:
                page = {"1": page1, "2": page2}[cmd[cmd.index("--page-token") + 1]]
                return mock.Mock(returncode=0, stdout=json.dumps(page))
            return mock.Mock(returncode=0, stdout=where.get(cmd[-1], ""))

        with mock.patch.object(J.subprocess, "run", run):
            view = J.pool_view("zoetrope")
        self.assertEqual(view, {"hosts": {"10.0.0.1": [{"job_id": "pt-r1", "name": "interiorgs-v10-renderer-rl-run-s", "cards": 1}],
                                          "10.0.0.9": [{"job_id": "pt-old", "name": "vln_eval", "cards": 8}]},
                                "waiting": ["interiorgs-v10-rl-rl-run-s"]})
        self.assertEqual(sum("list" in c for c in calls), 2)  # page 2 was short: no page 3
        self.assertFalse(any(c[-1] == "pt-h" for c in calls))  # another pool's jobs are not looked up
        del where["pt-old"]  # a running job whose machine cannot be read: no view
        with mock.patch.object(J.subprocess, "run", run):
            self.assertIsNone(J.pool_view("zoetrope"))
        with mock.patch.object(J.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="")):
            self.assertIsNone(J.pool_view("zoetrope"))

    def test_renderer_mode_address_file(self):
        """Round 12 batch (renderer_repack): the renderer mode's optional ENDPOINT_NAME; a second service, started
        before the first is deleted, writes its own address file and its own service log; without it, as before."""
        import subprocess
        import tempfile

        script = (next(p for p in Path(__file__).resolve().parents if (p / "setup.py").is_file()) / J.SCRIPT).read_text()
        block = script.split("\n  renderer)\n", 1)[1].split(";;", 1)[0]
        stubs = ('RUN=$2\nhostname() { echo 10.1.2.3; }\n'
                 'start_renderer() { echo "$2 ${RENDERER_SERVICE_LOG:-default}" > "${RUN}/started"; renderer_pid=$$; }\n')
        with tempfile.TemporaryDirectory() as tmp:
            sh, run = Path(tmp) / "renderer.sh", Path(tmp) / "run"
            sh.write_text("set -euo pipefail\n" + stubs + block + "\n")
            run.mkdir()
            (run / "STOP_RENDERER").touch()  # the service loop ends at once
            subprocess.run(["bash", str(sh), "renderer", str(run), "32"], check=True)
            self.assertEqual((run / "renderer_endpoint.txt").read_text(), "http://10.1.2.3:8915/render\n")
            self.assertEqual((run / "started").read_text(), "32 default\n")
            (run / "renderer_endpoint.txt").write_text("old\n")
            subprocess.run(["bash", str(sh), "renderer", str(run), "32", "renderer_endpoint_next.txt"], check=True)
            self.assertEqual((run / "renderer_endpoint.txt").read_text(), "old\n")  # untouched
            self.assertEqual((run / "renderer_endpoint_next.txt").read_text(), "http://10.1.2.3:8915/render\n")
            self.assertRegex((run / "started").read_text(), r"^32 .*/logs/renderer_service-\d{8}T\d{6}Z\.log\n$")
            self.assertEqual(sorted(p.name for p in run.iterdir() if p.suffix == ".tmp"), [])


if __name__ == "__main__":
    unittest.main()
