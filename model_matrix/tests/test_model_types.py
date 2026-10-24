"""The model type as a field (the model line, step 2 of 2026-10-24): the registry, and how the jobs, the SFT recipe and
the training chain take it.  Qwen2.5-VL is the default and every default path stays byte for byte as before."""
import importlib
import json
import os
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


class ModelTypesTest(unittest.TestCase):
    def test_registry(self):
        from vagen.models import model_types as M

        self.assertEqual(M.get().name, "qwen2_5_vl")
        self.assertEqual(M.get(None), M.TYPES["qwen2_5_vl"])
        q25, q3 = M.get("qwen2_5_vl"), M.get("qwen3_vl")
        self.assertIsNone(q25.python)  # the job script's own default
        self.assertEqual((q25.lf_template, q25.lf_launcher), ("qwen2_vl", "llamafactory.launcher"))
        self.assertEqual(q25.sft_accelerate_config, "examples/train/active_spatial/accelerate_fsdp_qwen25vl.yaml")
        self.assertEqual(q3.python, M.QWEN3VL_RUNTIME_PYTHON)
        self.assertEqual((q3.lf_template, q3.lf_launcher), ("qwen3_vl_nothink", "model_matrix.sft.lf_launcher"))
        for kind in M.TYPES.values():  # every config the job script may get exists in the repository
            self.assertTrue((ROOT / kind.sft_accelerate_config).is_file(), kind.sft_accelerate_config)
            self.assertEqual(kind.stop_token, "<|im_end|>")
        with self.assertRaises(ValueError):
            M.get("llava")

    def test_check_reads_the_models_config(self):
        from vagen.models import model_types as M

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            self.assertIsNone(M.of_checkpoint(tmp))  # unreadable (e.g. root-only): passes
            self.assertEqual(M.check("qwen3_vl", tmp).name, "qwen3_vl")
            (model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
            self.assertEqual(M.of_checkpoint(tmp), "qwen3_vl")
            self.assertEqual(M.check("qwen3_vl", tmp).name, "qwen3_vl")
            with self.assertRaises(ValueError):
                M.check(None, tmp)  # the default type is Qwen2.5-VL

    def test_sft_launch_line(self):
        out = subprocess.run([os.sys.executable, "-m", "vagen.models.model_types", "sft-launch", "qwen3_vl"], cwd=ROOT,
                             capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT)))
        self.assertEqual(out.stdout.split(), ["model_matrix/sft/accelerate_fsdp_qwen3vl.yaml", "model_matrix.sft.lf_launcher"])


class JobsTest(unittest.TestCase):
    def test_launch_script_passes_the_model_type_on(self):
        from data_gen.interiorgs_v10 import job_sco as J

        clean = {k: v for k, v in os.environ.items() if k not in J.PASS_ENV and k != "VAGEN_LITE_WORKTREE"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, clean, clear=True):
            run = Path(tmp) / "run"
            J = importlib.reload(J)
            self.assertEqual(J.prepare(run, 1, ["eval", "/e"]).read_text(),  # unset: as before
                             f"#!/usr/bin/env bash\nexec bash {J.DEFAULT_WORKTREE}/{J.SCRIPT} eval {run} /e\n")
            os.environ.update(VAGEN_LITE_PYTHON="/py dir/python", VAGEN_LITE_MODEL_TYPE="qwen3_vl",
                              VAGEN_D0_3_SMOKE_DIR="/a")
            self.assertEqual(J.prepare(run, 8, ["sft", "/lf"]).read_text(),
                             "#!/usr/bin/env bash\nexec env VAGEN_LITE_PYTHON='/py dir/python' VAGEN_LITE_MODEL_TYPE=qwen3_vl "
                             f"VAGEN_D0_3_SMOKE_DIR=/a bash {J.DEFAULT_WORKTREE}/{J.SCRIPT} sft {run} /lf\n")
            os.environ["VAGEN_LITE_WORKTREE"] = "/copy"
            J = importlib.reload(J)
            self.assertTrue(J.prepare(run, 1, ["eval", "/e"]).read_text().startswith(
                "#!/usr/bin/env bash\nexec env VAGEN_LITE_WORKTREE=/copy VAGEN_LITE_PYTHON="))
        importlib.reload(J)

    def test_job_script_switches(self):
        script = (ROOT / "examples/train/active_spatial/sco_interiorgs_v10_train.sh").read_text()
        self.assertIn("\nPY=${VAGEN_LITE_PYTHON:-${ENV_ROOT}/bin/python}\n", script)
        # the Qwen2.5-VL SFT command is the one of every earlier run
        self.assertIn("  accelerate launch --config_file examples/train/active_spatial/accelerate_fsdp_qwen25vl.yaml \\\n"
                      "    --num_processes \"$2\" -m llamafactory.launcher \"$1\"\n}", script)
        self.assertIn('"${PY}" -m vagen.models.model_types sft-launch "${VAGEN_LITE_MODEL_TYPE}"', script)
        self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / "examples/train/active_spatial/sco_interiorgs_v10_train.sh")]).returncode, 0)

    def test_lf_train_picks_the_types_launcher(self):
        """lf_train run in bash with stand-ins for accelerate and the Python: the default and the Qwen3-VL lines."""
        script = (ROOT / "examples/train/active_spatial/sco_interiorgs_v10_train.sh").read_text()
        body = script[script.index("lf_train() {"):script.index("\n}\n", script.index("lf_train() {")) + 3]
        with tempfile.TemporaryDirectory() as tmp:
            bin_ = Path(tmp)
            (bin_ / "accelerate").write_text("#!/usr/bin/env bash\necho ACC \"$@\"\n")
            (bin_ / "py").write_text("#!/usr/bin/env bash\nif [[ $1 == -m && $2 == vagen.models.model_types ]]; then "
                                     f"cd {ROOT} && exec {os.sys.executable} \"$@\"; fi\necho PY \"$@\"\n")
            for f in ("accelerate", "py"):
                (bin_ / f).chmod(0o755)
            test = f"set -euo pipefail\nPY={bin_ / 'py'}\n{body}\nlf_train /lf/train.yaml 8\n"
            env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", PYTHONPATH=str(ROOT))
            env.pop("VAGEN_LITE_MODEL_TYPE", None)
            out = subprocess.run(["bash", "-c", test], capture_output=True, text=True, env=env).stdout.strip()
            self.assertEqual(out, "ACC launch --config_file examples/train/active_spatial/accelerate_fsdp_qwen25vl.yaml "
                                  "--num_processes 8 -m llamafactory.launcher /lf/train.yaml")
            env["VAGEN_LITE_MODEL_TYPE"] = "qwen3_vl"
            out = subprocess.run(["bash", "-c", test], capture_output=True, text=True, env=env).stdout.strip()
            self.assertEqual(out, "PY -m accelerate.commands.launch --config_file model_matrix/sft/accelerate_fsdp_qwen3vl.yaml "
                                  "--num_processes 8 -m model_matrix.sft.lf_launcher /lf/train.yaml")


class SftRecipeTest(unittest.TestCase):
    def test_template_follows_the_model_type(self):
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        import prepare_active_spatial_sft as P

        self.assertEqual(P._lf_template("qwen2_5_vl"), "qwen2_vl")
        self.assertEqual(P._lf_template("qwen3_vl"), "qwen3_vl_nothink")
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
            for wanted in (None, "qwen2_5_vl"):  # the default refuses a Qwen3-VL model, as before
                with self.assertRaises(ValueError):
                    P.prepare(Namespace(out=str(Path(tmp) / f"o{wanted}"), qa_probability=0, val_fraction=0.1,
                                        model=str(model), model_type=wanted, trajectory=[], qa_bank=[],
                                        split_manifest=None))


class ChainTest(unittest.TestCase):
    def _chain(self, tmp, **plan):
        from data_gen.interiorgs_v10 import train_chain as T

        return T.Chain(dict({"tr": tmp, "sft_dir": tmp, "endpoint": 150}, **plan), sleep=lambda s: None)

    def test_job_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._chain(tmp).job_env(), {})
            self.assertEqual(self._chain(tmp, model_type="qwen2_5_vl").job_env(), {})
            self.assertEqual(self._chain(tmp, model_type="qwen3_vl").job_env(),
                             {"VAGEN_LITE_MODEL_TYPE": "qwen3_vl",
                              "VAGEN_LITE_PYTHON": "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime/bin/python"})
            self.assertEqual(self._chain(tmp, rl={"logprob_audit": True}).job_env(),
                             {"VAGEN_D0_3_SMOKE_DIR": str(Path(tmp) / "logprob_audit")})
            self.assertEqual(self._chain(tmp, job_prefix="models").job_env(), {"VAGEN_LITE_JOB_PREFIX": "models"})

    def test_idle_pool_waits_for_the_main_line_and_an_idle_machine(self):
        from model_matrix import sco_view

        def v(pool, idle, free, main=0, ours=0, main1=0):
            return {"pool": pool, "machines": 10, "busy_machines": 10 - idle, "idle_estimate": idle,
                    "machines_with_free_cards": free, "waiting": main + ours, "waiting_main_line": main,
                    "waiting_main_line_1card": main1,
                    "waiting_downstream": 0, "waiting_ours": ours}

        rounds = [  # each poll: zoetrope, h800
            {"zoetrope": v("zoetrope", 2, 0, main=1, main1=1), "h800": v("h800", 0, 3, main=1)},  # 8 cards: nothing fits
            {"zoetrope": v("zoetrope", 1, 0), "h800": v("h800", 3, 1, ours=1)},  # ours waits on h800
        ]
        with tempfile.TemporaryDirectory() as tmp:
            chain = self._chain(tmp, pool="idle")
            slept = []
            chain.sleep = slept.append
            state = {"i": 0}

            def view(pool):
                return rounds[min(state["i"], len(rounds) - 1)][pool]

            def sleep(s):
                slept.append(s)
                state["i"] += 1

            chain.sleep = sleep
            with mock.patch.object(sco_view, "view", side_effect=view), \
                    mock.patch.object(sco_view, "PAUSE", Path(tmp) / "no_pause"):
                self.assertEqual(chain.job_pool("q3"), "zoetrope")  # an RL run: needs an idle machine
                self.assertEqual(slept, [300.0])
                self.assertEqual(chain.job_pool("q3"), "zoetrope")  # the run keeps its pool
                state["i"] = 0
                self.assertEqual(chain.job_pool(cards=1), "h800")  # one card: free cards, no main-line job waiting
                pause = Path(tmp) / "pause"
                pause.write_text("")
                with mock.patch.object(sco_view, "PAUSE", pause):
                    state["i"] = 1
                    calls = []

                    def sleep_then_resume(s):
                        calls.append(s)
                        pause.unlink()

                    chain.sleep = sleep_then_resume
                    self.assertEqual(chain.job_pool(cards=8), "zoetrope")
                    self.assertEqual(calls, [300.0])  # it waited while the pause file was there
            self.assertIsNone(chain.state["idle_pool_wait"])
            self.assertEqual(self._chain(tmp, pool="h800").job_pool(), "h800")  # other pools as before

    def test_job_names_take_the_prefix(self):
        from data_gen.interiorgs_v10 import job_sco as J

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "rl_run_q3"
            J.prepare(run, 8, ["rl", "/rl", "10"])
            done = mock.Mock(returncode=0, stdout="job pt-abc123 submitted", stderr="")
            for env, want in (({}, "--job-name=interiorgs-v10-rl-rl-run-q3"),
                              ({"VAGEN_LITE_JOB_PREFIX": "models"}, "--job-name=models-rl-rl-run-q3")):
                (run / "job" / "submission.json").unlink(missing_ok=True)
                with mock.patch.dict(os.environ, env), mock.patch.object(J.subprocess, "run", return_value=done) as r:
                    self.assertEqual(J.submit(run), "pt-abc123")
                    self.assertIn(want, r.call_args[0][0])

    def test_check_code_refuses_another_models_type(self):
        from data_gen.interiorgs_v10 import train_chain as T

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
            chain = self._chain(tmp, model_type="qwen3_vl", model=tmp, commit="0")
            with self.assertRaises(T.ChainError) as err:
                chain.check_code()
            self.assertIn("is a qwen2_5_vl model, not qwen3_vl", str(err.exception))

    def test_sft_data_and_rl_config_get_the_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "traj.jsonl").write_text("{}\n")
            chain = self._chain(tmp, model="/m", sft_records=False, sft_groups={"formal": "traj.jsonl"}, gs_root="/ply",
                                model_type="qwen3_vl", rl={"logprob_audit": True, "standard": "v10_p1_S0"})
            calls = []

            def sh(args, **k):
                calls.append(list(args))
                if "prepare_active_spatial_sft.py" in args:
                    (root / "lf_formal").mkdir(exist_ok=True)
                    (root / "lf_formal" / "train.yaml").write_text("x")

            chain.sh = sh
            chain.sft_data()
            prep = next(a for a in calls if "prepare_active_spatial_sft.py" in a)
            self.assertEqual(prep[prep.index("--model-type") + 1], "qwen3_vl")
            self.assertNotIn("--logging-steps", prep)
            chain.plan["sft_logging_steps"] = 1
            calls.clear()
            chain.sft_data()
            self.assertEqual(calls[0][calls[0].index("--logging-steps") + 1], "1")
            chain.sft_model = lambda: Path("/m")
            chain.rl_config()
            rl = next(a for a in calls if "rl-config" in a)
            self.assertIn("--rollout-log-probs", rl)
            calls.clear()
            plain = self._chain(tmp, model="/m", sft_records=False, sft_groups={"formal": "traj.jsonl"})
            plain.sh = sh
            plain.sft_data()
            self.assertNotIn("--model-type", calls[0])  # without the field: the command as before


if __name__ == "__main__":
    unittest.main()
