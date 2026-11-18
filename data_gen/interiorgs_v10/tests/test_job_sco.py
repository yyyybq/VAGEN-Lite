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
            self.assertEqual(J.SPECS[2], "N4lS.Iq.I80.2")  # q38: the 27B evaluations
            with self.assertRaises(ValueError):
                J.prepare(run, 4, ["sft"])
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
        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
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
            sh.write_text("set -euo pipefail\nuse_train_env() { :; }\n" + body + "\n")  # q38: the env switch, stubbed
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

        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
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

    def test_q38_variables_reach_the_job_script(self):
        """q38: the trainer's environment, its entry module, the override of the CUDA forward-compatibility libraries
        and the eval cards go from the preparing process to the job script; unset, the launch script is as before."""
        import os
        from unittest import mock

        q38 = {"TRAIN_ENV_ROOT": "/q38/envs/verl", "V10_PPO_MODULE": "vagen.q38.ppo_v1", "Q38_CUDA_COMPAT": "0",
               "VAGEN_LITE_EVAL_CARDS": "2"}
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            clean = {k: v for k, v in os.environ.items() if k not in q38}
            with mock.patch.dict(os.environ, clean, clear=True):
                plain = J.prepare(run, 2, ["eval", "/e"]).read_text()
            self.assertNotIn("exec env ", plain)
            self.assertEqual(json.loads((run / "job" / "prepared.json").read_text())["spec"], "N4lS.Iq.I80.2")
            with mock.patch.dict(os.environ, q38):
                text = J.prepare(run, 2, ["eval", "/e"]).read_text()
            for k, v in q38.items():
                self.assertIn(f"{k}={v}", text)
        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
        self.assertIn("\nPPO_MODULE=${V10_PPO_MODULE:-vagen.interiorgs_v10_ppo}\n", script)
        self.assertNotIn('-m vagen.interiorgs_v10_ppo --config', script)  # every trainer call takes PPO_MODULE
        self.assertIn('--conda-env "${RENDER_ENV_ROOT}"', script)  # the render service keeps vagen-lite
        self.assertIn("\numask 022\n", script)
        self.assertNotRegex(script, r"(?m)^\s*umask 077")  # (a comment names pear's)
        # an eval job: the render service on card 0 first, then vLLM on eval_policy_cards in the trainer's env; a QA
        # job stays on card 0
        block = script.split("\n  eval)\n", 1)[1].split(";;", 1)[0]
        order = [block.index(s) for s in ("CUDA_VISIBLE_DEVICES=0\n", "start_renderer 0\n", "use_train_env\n",
                                          "CUDA_VISIBLE_DEVICES=$(eval_policy_cards)", "data_gen.interiorgs_v10.evaluation")]
        self.assertEqual(order, sorted(order))
        qa = script.split("\n  qa)\n", 1)[1].split(";;", 1)[0]
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", qa)
        self.assertNotIn("eval_policy_cards", qa)

    def test_q38_train_env_switch(self):
        """q38: use_train_env (the trainer's CUDA 13 toolkit and library paths, its C compiler, the forward-
        compatibility libraries on a driver older than 580 unless Q38_CUDA_COMPAT overrides) and eval_policy_cards,
        run as the job script runs them."""
        import os
        import subprocess

        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
        funcs = "\neval_policy_cards() {" + script.split("\neval_policy_cards() {", 1)[1].split("\nlf_train() {", 1)[0]
        probe_gcc = "/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/x86_64-conda-linux-gnu-"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, compat, bin_dir = root / "train", root / "compat", root / "bin"
            nvcc = train / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "bin" / "nvcc"
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text("#!/bin/sh\n")
            nvcc.chmod(0o755)
            compat.mkdir()
            (compat / "libcuda.so.580.196.07").write_text("")
            (compat / "libcuda.so.1").symlink_to("libcuda.so.580.196.07")
            (compat / "libcuda.so").symlink_to("libcuda.so.1")
            bin_dir.mkdir()

            def run(train_root, compat_switch=None, driver="550.90.07", compat_dir=compat, cards=None, extra=None):
                # driver None: nvidia-smi answers nothing
                (bin_dir / "nvidia-smi").write_text("#!/bin/sh\n" + (f"echo {driver}\n" if driver else "exit 9\n"))
                (bin_dir / "nvidia-smi").chmod(0o755)
                head = (f"set -euo pipefail\nRENDER_ENV_ROOT=/render\nTRAIN_ENV_ROOT={train_root}\nPY=py\nPPO_MODULE=m\n"
                        "BASE_LD_LIBRARY_PATH=/base/lib BASE_LIBRARY_PATH= BASE_CPATH=\n"
                        "export LD_LIBRARY_PATH=/render/lib:/base/lib LIBRARY_PATH=/render/targets/lib "
                        "CPATH=/render/include CURAND_LIBRARY_DIR=/render/curand CC=/render/gcc CXX=/render/g++\n")
                tail = ('\neval_policy_cards\nuse_train_env\necho "CUDA_HOME=${CUDA_HOME:-}"\n'
                        'echo "LD=${LD_LIBRARY_PATH:-}"\necho "LIB=${LIBRARY_PATH:-}"\necho "CPATH=${CPATH-unset}"\n'
                        'echo "TL=${TRITON_LIBCUDA_PATH:-}"\necho "CURAND=${CURAND_LIBRARY_DIR-unset}"\n'
                        'echo "CC=${CC:-}"\necho "CXX=${CXX:-}"\n')
                env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", Q38_CUDA_COMPAT_DIR=str(compat_dir),
                           TMPDIR=str(root))
                for k in ("Q38_CUDA_COMPAT", "VAGEN_LITE_EVAL_CARDS", "TRAIN_CC", "TRAIN_CXX", "TRAIN_CUDA_HOME"):
                    env.pop(k, None)
                if compat_switch is not None:
                    env["Q38_CUDA_COMPAT"] = compat_switch
                if cards is not None:
                    env["VAGEN_LITE_EVAL_CARDS"] = cards
                env.update(extra or {})
                res = subprocess.run(["bash", "-c", head + funcs + tail], capture_output=True, text=True, env=env)
                lines = res.stdout.splitlines()
                return res.returncode, lines[0] if lines else None, dict(l.split("=", 1) for l in lines if "=" in l
                                                                         and not l.startswith("[")), res.stdout

            # no separate trainer env: nothing changes
            rc, cards, out, _ = run("/render")
            self.assertEqual((rc, cards, out["LD"], out["LIB"], out["CPATH"], out["CC"]),
                             (0, "0", "/render/lib:/base/lib", "/render/targets/lib", "/render/include", "/render/gcc"))
            # an eval job's vLLM: card 0 with one card, card 1 with two (card 0 is the render service's)
            self.assertEqual(run("/render", cards="2")[1], "1")
            self.assertEqual(run("/render", cards="4")[1], "1,2,3")
            self.assertNotEqual(run("/render", cards="two")[0], 0)
            # the q38 env on zoetrope's driver (older than 580): the compatibility libraries by default
            rc, _, out, text = run(train)
            self.assertEqual(rc, 0, text)
            self.assertEqual(out["CUDA_HOME"], str(nvcc.parent.parent))
            self.assertEqual((out["LD"], out["LIB"], out["TL"]), (f"{compat}:/base/lib", str(compat), str(compat)))
            self.assertEqual((out["CPATH"], out["CURAND"]), ("unset", "unset"))
            self.assertEqual((out["CC"], out["CXX"]), (probe_gcc + "gcc", probe_gcc + "g++"))
            self.assertIn("driver 550.90.07 (compatibility libraries below 580): Q38_CUDA_COMPAT=auto -> 1", text)
            self.assertEqual(run(train, "auto")[2]["LD"], f"{compat}:/base/lib")
            # a driver of 580 or newer (h800's 595.71.05 in pear's r6): not them
            for driver in ("595.71.05", "580.65.06"):
                rc, _, out, text = run(train, driver=driver)
                self.assertEqual((rc, out["LD"]), (0, "/base/lib"))
                self.assertTrue(out["LIB"] == out["TL"] and (out["LIB"] == "" or "/q38_cuda_link_" in out["LIB"]))
                self.assertIn("Q38_CUDA_COMPAT=auto -> 0", text)
            # no driver version: not them, with a warning
            rc, _, out, text = run(train, driver=None)
            self.assertEqual((rc, out["LD"]), (0, "/base/lib"))
            self.assertIn("warning: no driver version from nvidia-smi", text)
            # the overrides: 0 never, 1 always
            self.assertEqual(run(train, "0")[2]["LD"], "/base/lib")
            self.assertEqual(run(train, "1", driver="595.71.05")[2]["LD"], f"{compat}:/base/lib")
            # needed but missing: the job stops (torch cu130 would see no GPU)
            self.assertEqual(run(train, compat_dir=root / "nowhere")[0], 1)
            self.assertEqual(run(train, "1", compat_dir=root / "nowhere")[0], 1)
            self.assertEqual(run(train, "0", compat_dir=root / "nowhere")[0], 0)
            self.assertEqual(run(train, driver="595.71.05", compat_dir=root / "nowhere")[0], 0)
            # another compiler
            out = run(train, extra={"TRAIN_CC": "/x/gcc", "TRAIN_CXX": "/x/g++"})[2]
            self.assertEqual((out["CC"], out["CXX"]), ("/x/gcc", "/x/g++"))


if __name__ == "__main__":
    unittest.main()
